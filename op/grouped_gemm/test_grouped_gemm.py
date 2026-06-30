"""CUTLASS 与 PyTorch Grouped GEMM 的精度和性能对比。"""

import logging
from collections.abc import Callable

import torch

from cudaop_grouped_gemm import gmm, torch_gmm


LOGGER = logging.getLogger("cudaop_grouped_gemm_test")

BF16_RTOL = 2e-2
BF16_FORWARD_ATOL = 2e-2
BF16_GRAD_ATOL = 4e-2
WARMUP_ITERATIONS = 20
BENCHMARK_ITERATIONS = 100

GroupedGemm = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, bool],
    torch.Tensor,
]


def reference(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for expert, size in enumerate(batch_sizes.tolist()):
        weight = b[expert].transpose(0, 1) if trans_b else b[expert]
        outputs.append(a[offset:offset + size] @ weight)
        offset += size
    return torch.cat(outputs, dim=0)


def run_implementation(
    implementation: GroupedGemm,
    source_a: torch.Tensor,
    source_b: torch.Tensor,
    sizes: torch.Tensor,
    trans_b: bool,
    expected: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = source_a.detach().clone().requires_grad_(True)
    b = source_b.detach().clone().requires_grad_(True)
    output = implementation(a, b, sizes, trans_b)
    torch.testing.assert_close(
        output.float(),
        expected,
        rtol=BF16_RTOL,
        atol=BF16_FORWARD_ATOL,
    )
    output.sum().backward()
    torch.testing.assert_close(
        a.grad.float(),
        reference_a.grad,
        rtol=BF16_RTOL,
        atol=BF16_GRAD_ATOL,
    )
    torch.testing.assert_close(
        b.grad.float(),
        reference_b.grad,
        rtol=BF16_RTOL,
        atol=BF16_GRAD_ATOL,
    )
    return output.detach(), a.grad.detach(), b.grad.detach()


def max_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def run_accuracy_case(trans_b: bool) -> None:
    sizes = torch.tensor([128, 157, 97, 100, 111, 129, 138, 101])
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_in = 256
    hidden_out = 32
    weight_shape = (
        (experts, hidden_out, hidden_in)
        if trans_b
        else (experts, hidden_in, hidden_out)
    )
    source_a = torch.randn(
        tokens,
        hidden_in,
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_b = torch.randn(
        weight_shape,
        device="cuda",
        dtype=torch.bfloat16,
    )
    reference_a = source_a.float().requires_grad_(True)
    reference_b = source_b.float().requires_grad_(True)
    expected = reference(
        reference_a,
        reference_b,
        sizes,
        trans_b,
    )
    expected.sum().backward()

    cutlass_result = run_implementation(
        gmm,
        source_a,
        source_b,
        sizes,
        trans_b,
        expected,
        reference_a,
        reference_b,
    )
    torch_result = run_implementation(
        torch_gmm,
        source_a,
        source_b,
        sizes,
        trans_b,
        expected,
        reference_a,
        reference_b,
    )
    LOGGER.info(
        (
            "trans_b=%-5s output_shape=%-12s "
            "cutlass_error=(%.6f, %.6f, %.6f) "
            "torch_error=(%.6f, %.6f, %.6f) "
            "version_diff=(%.6f, %.6f, %.6f)"
        ),
        trans_b,
        str(tuple(cutlass_result[0].shape)),
        max_error(cutlass_result[0], expected),
        max_error(cutlass_result[1], reference_a.grad),
        max_error(cutlass_result[2], reference_b.grad),
        max_error(torch_result[0], expected),
        max_error(torch_result[1], reference_a.grad),
        max_error(torch_result[2], reference_b.grad),
        max_error(cutlass_result[0], torch_result[0]),
        max_error(cutlass_result[1], torch_result[1]),
        max_error(cutlass_result[2], torch_result[2]),
    )


def benchmark(
    operation: Callable[[], None],
) -> float:
    for _ in range(WARMUP_ITERATIONS):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCHMARK_ITERATIONS):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / BENCHMARK_ITERATIONS


def run_performance_case(trans_b: bool) -> None:
    sizes = torch.tensor(
        [128, 157, 97, 100, 111, 129, 138, 101],
        device="cuda",
    )
    tokens = int(sizes.sum())
    experts = sizes.numel()
    a = torch.randn(
        tokens,
        256,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_shape = (
        (experts, 32, 256)
        if trans_b
        else (experts, 256, 32)
    )
    b = torch.randn(
        weight_shape,
        device="cuda",
        dtype=torch.bfloat16,
    )

    def cutlass_forward() -> None:
        gmm(a, b, sizes, trans_b)

    def torch_forward() -> None:
        torch_gmm(a, b, sizes, trans_b)

    grad_output = torch.ones(
        tokens,
        32,
        device="cuda",
        dtype=torch.bfloat16,
    )

    def cutlass_forward_backward() -> None:
        input_a = a.detach().requires_grad_(True)
        input_b = b.detach().requires_grad_(True)
        gmm(input_a, input_b, sizes, trans_b).backward(grad_output)

    def torch_forward_backward() -> None:
        input_a = a.detach().requires_grad_(True)
        input_b = b.detach().requires_grad_(True)
        torch_gmm(
            input_a,
            input_b,
            sizes,
            trans_b,
        ).backward(grad_output)

    cutlass_forward_us = benchmark(cutlass_forward)
    torch_forward_us = benchmark(torch_forward)
    cutlass_backward_us = benchmark(cutlass_forward_backward)
    torch_backward_us = benchmark(torch_forward_backward)
    LOGGER.info(
        (
            "trans_b=%-5s CUTLASS forward=%9.3f us "
            "forward+backward=%9.3f us"
        ),
        trans_b,
        cutlass_forward_us,
        cutlass_backward_us,
    )
    LOGGER.info(
        (
            "trans_b=%-5s Torch   forward=%9.3f us "
            "forward+backward=%9.3f us"
        ),
        trans_b,
        torch_forward_us,
        torch_backward_us,
    )
    LOGGER.info(
        (
            "trans_b=%-5s Torch/CUTLASS ratio: "
            "forward=%.3fx forward+backward=%.3fx"
        ),
        trans_b,
        torch_forward_us / cutlass_forward_us,
        torch_backward_us / cutlass_backward_us,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 CUDA GPU")

    LOGGER.info(
        "配置：device=%s dtype=bfloat16 arch=sm_%d%d torch=%s",
        torch.cuda.get_device_name(),
        *torch.cuda.get_device_capability(),
        torch.__version__,
    )
    LOGGER.info("")
    LOGGER.info("阶段：CUTLASS 与 Torch 前向、反向精度验证")
    torch.manual_seed(11)
    run_accuracy_case(trans_b=False)
    run_accuracy_case(trans_b=True)
    LOGGER.info("")
    LOGGER.info(
        (
            "阶段：端到端性能对比，包含 sizes/offsets 处理"
            "（warmup=%d iterations=%d）"
        ),
        WARMUP_ITERATIONS,
        BENCHMARK_ITERATIONS,
    )
    run_performance_case(trans_b=False)
    run_performance_case(trans_b=True)
    LOGGER.info("")
    LOGGER.info("[SUCCESS] cudaop_grouped_gemm 对比测试通过")


if __name__ == "__main__":
    main()
