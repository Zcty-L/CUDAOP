"""CUTLASS、PyTorch、Triton 与 cuTile LoRA Grouped GEMM 对比。"""

import logging
import statistics
from collections.abc import Callable

import torch

from cudaop_grouped_gemm import (
    CuTileLoraDownGrouped,
    CuTileLoraFusedDownUpGrouped,
    CuTileLoraUpGrouped,
    LoraDownGrouped,
    LoraFusedDownUpGrouped,
    LoraUpGrouped,
    cutile_fused_lora,
    gmm,
    lora_gmm,
    torch_gmm,
    triton_fused_lora,
)


LOGGER = logging.getLogger("cudaop_grouped_gemm_test")

BF16_RTOL = 2e-2
BF16_ATOL = 2e-2
BF16_GRAD_ATOL = 5e-1
WARMUP_ITERATIONS = 20
BENCHMARK_ITERATIONS = 100
SIZES = [128, 157, 97, 100, 111, 129, 138, 101] * 10


def reference_down(
    a: torch.Tensor,
    weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for expert, size in enumerate(batch_sizes.tolist()):
        outputs.append(
            a[offset:offset + size] @ weight[expert].transpose(0, 1)
        )
        offset += size
    return torch.cat(outputs, dim=0)


def reference_up(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for expert, size in enumerate(batch_sizes.tolist()):
        outputs.append(
            hidden[offset:offset + size] @ weight[expert]
        )
        offset += size
    return torch.cat(outputs, dim=0)


def max_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def run_accuracy() -> None:
    sizes = torch.tensor(SIZES)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = 256
    a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    down_weight = torch.randn(
        experts,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    up_weight = torch.randn_like(down_weight)
    down = LoraDownGrouped(down_weight)
    up = LoraUpGrouped(up_weight)
    fused = LoraFusedDownUpGrouped(down_weight, up_weight)
    cutile_down = CuTileLoraDownGrouped(down_weight)
    cutile_up = CuTileLoraUpGrouped(up_weight)
    cutile_fused = CuTileLoraFusedDownUpGrouped(
        down_weight,
        up_weight,
    )

    expected_hidden = reference_down(
        a.float(),
        down_weight.float(),
        sizes,
    )
    cutlass_hidden = gmm(a, down_weight, sizes, True)
    torch_hidden = torch_gmm(a, down_weight, sizes, True)
    triton_hidden = down(a, sizes)
    cutile_hidden = cutile_down(a, sizes)
    for actual in (
        cutlass_hidden,
        torch_hidden,
        triton_hidden,
        cutile_hidden,
    ):
        torch.testing.assert_close(
            actual.float(),
            expected_hidden,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )

    expected_output = reference_up(
        triton_hidden.float(),
        up_weight.float(),
        sizes,
    )
    cutlass_output = gmm(triton_hidden, up_weight, sizes, False)
    torch_output = torch_gmm(
        triton_hidden,
        up_weight,
        sizes,
        False,
    )
    triton_output = up(triton_hidden, sizes)
    cutile_output = cutile_up(cutile_hidden, sizes)
    fused_hidden, fused_output = fused(a, sizes)
    cutile_fused_hidden, cutile_fused_output = cutile_fused(
        a,
        sizes,
    )
    for actual in (
        cutlass_output,
        torch_output,
        triton_output,
        cutile_output,
    ):
        torch.testing.assert_close(
            actual.float(),
            expected_output,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
        )
    torch.testing.assert_close(
        fused_hidden,
        triton_hidden,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fused_output,
        triton_output,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cutile_fused_hidden,
        cutile_hidden,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cutile_fused_output,
        cutile_output,
        rtol=0.0,
        atol=0.0,
    )

    LOGGER.info(
        (
            "%-8s | %-14s | %14s | %14s | "
            "%14s | %14s | %14s"
        ),
        "stage",
        "output shape",
        "CUTLASS error",
        "Torch error",
        "Triton error",
        "cuTile error",
        "CUTLASS/Triton",
    )
    LOGGER.info("-" * 117)
    accuracy_rows = (
        (
            "down",
            triton_hidden,
            expected_hidden,
            cutlass_hidden,
            torch_hidden,
            cutile_hidden,
        ),
        (
            "up",
            triton_output,
            expected_output,
            cutlass_output,
            torch_output,
            cutile_output,
        ),
    )
    for (
        stage,
        triton_value,
        expected,
        cutlass_value,
        torch_value,
        cutile_value,
    ) in (
        accuracy_rows
    ):
        LOGGER.info(
            (
                "%-8s | %-14s | %14.6f | %14.6f | "
                "%14.6f | %14.6f | %14.6f"
            ),
            stage,
            str(tuple(triton_value.shape)),
            max_error(cutlass_value, expected),
            max_error(torch_value, expected),
            max_error(triton_value, expected),
            max_error(cutile_value, expected),
            max_error(cutlass_value, triton_value),
        )

    LOGGER.info("")
    LOGGER.info(
        "%-14s | %-14s | %18s",
        "fused output",
        "shape",
        "separate diff",
    )
    LOGGER.info("-" * 53)
    LOGGER.info(
        "%-14s | %-14s | %18.6f",
        "saved hidden",
        str(tuple(fused_hidden.shape)),
        max_error(fused_hidden, triton_hidden),
    )
    LOGGER.info(
        "%-14s | %-14s | %18.6f",
        "final output",
        str(tuple(fused_output.shape)),
        max_error(fused_output, triton_output),
    )
    LOGGER.info(
        "%-14s | %-14s | %18.6f",
        "cuTile output",
        str(tuple(cutile_fused_output.shape)),
        max_error(cutile_fused_output, cutile_output),
    )


def run_backward_accuracy() -> None:
    sizes = torch.tensor(SIZES)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = 256
    source_a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_down = torch.randn(
        experts,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_up = torch.randn_like(source_down)
    grad_output = torch.randn_like(source_a)

    def execute(operation):
        a = source_a.detach().clone().requires_grad_(True)
        down_weight = (
            source_down.detach().clone().requires_grad_(True)
        )
        up_weight = (
            source_up.detach().clone().requires_grad_(True)
        )
        output = operation(
            a,
            down_weight,
            up_weight,
            sizes,
        )
        output.backward(grad_output)
        return (
            output.detach(),
            a.grad.detach(),
            down_weight.grad.detach(),
            up_weight.grad.detach(),
        )

    cutlass_results = execute(lora_gmm)
    triton_results = execute(triton_fused_lora)
    cutile_results = execute(cutile_fused_lora)
    names = (
        "output",
        "grad input",
        "grad down",
        "grad up",
    )
    for triton_value, cutlass_value in zip(
        triton_results,
        cutlass_results,
    ):
        torch.testing.assert_close(
            triton_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    for cutile_value, cutlass_value in zip(
        cutile_results,
        cutlass_results,
    ):
        torch.testing.assert_close(
            cutile_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )

    LOGGER.info(
        "%-12s | %-18s | %18s | %18s",
        "tensor",
        "shape",
        "Triton/CUTLASS diff",
        "cuTile/CUTLASS diff",
    )
    LOGGER.info("-" * 77)
    for name, triton_value, cutile_value, cutlass_value in zip(
        names,
        triton_results,
        cutile_results,
        cutlass_results,
    ):
        LOGGER.info(
            "%-12s | %-18s | %18.6f | %18.6f",
            name,
            str(tuple(triton_value.shape)),
            max_error(triton_value, cutlass_value),
            max_error(cutile_value, cutlass_value),
        )

    empty_sizes = torch.tensor([0, 3, 0, 5])
    empty_tokens = int(empty_sizes.sum())
    empty_input = torch.randn(
        empty_tokens,
        32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    empty_down = torch.randn(
        4,
        16,
        32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    empty_up = torch.randn_like(empty_down)
    empty_grad_output = torch.randn_like(empty_input)

    def execute_empty(operation):
        a = empty_input.detach().clone().requires_grad_(True)
        down_weight = (
            empty_down.detach().clone().requires_grad_(True)
        )
        up_weight = (
            empty_up.detach().clone().requires_grad_(True)
        )
        output = operation(
            a,
            down_weight,
            up_weight,
            empty_sizes,
        )
        output.backward(empty_grad_output)
        return a.grad, down_weight.grad, up_weight.grad

    cutlass_empty = execute_empty(lora_gmm)
    triton_empty = execute_empty(triton_fused_lora)
    cutile_empty = execute_empty(cutile_fused_lora)
    for triton_value, cutlass_value in zip(
        triton_empty,
        cutlass_empty,
    ):
        torch.testing.assert_close(
            triton_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    for cutile_value, cutlass_value in zip(
        cutile_empty,
        cutlass_empty,
    ):
        torch.testing.assert_close(
            cutile_value,
            cutlass_value,
            rtol=BF16_RTOL,
            atol=BF16_GRAD_ATOL,
        )
    for implementation in (triton_empty, cutile_empty):
        for weight_grad in implementation[1:]:
            if torch.count_nonzero(weight_grad[[0, 2]]).item() != 0:
                raise AssertionError("空 expert 的权重梯度必须为零")
    LOGGER.info("")
    LOGGER.info(
        "空 expert 与 K 尾块回归：sizes=%s hidden_size=32 [PASS]",
        empty_sizes.tolist(),
    )


def benchmark_once(operation: Callable[[], None]) -> float:
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


def benchmark(operation: Callable[[], None]) -> float:
    samples = [benchmark_once(operation) for _ in range(5)]
    return statistics.median(samples)


def run_performance() -> None:
    sizes = torch.tensor(SIZES, device="cuda")
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = 2048
    a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    down_weight = torch.randn(
        experts,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    up_weight = torch.randn_like(down_weight)
    down = LoraDownGrouped(down_weight)
    up = LoraUpGrouped(up_weight)
    fused = LoraFusedDownUpGrouped(down_weight, up_weight)
    cutile_down = CuTileLoraDownGrouped(down_weight)
    cutile_up = CuTileLoraUpGrouped(up_weight)
    cutile_fused = CuTileLoraFusedDownUpGrouped(
        down_weight,
        up_weight,
    )
    hidden = down(a, sizes)
    cutile_hidden = cutile_down(a, sizes)

    operations = {
        "CUTLASS down": lambda: gmm(
            a,
            down_weight,
            sizes,
            True,
        ),
        "Torch down": lambda: torch_gmm(
            a,
            down_weight,
            sizes,
            True,
        ),
        "Triton down": lambda: down(a, sizes),
        "CUTLASS up": lambda: gmm(
            hidden,
            up_weight,
            sizes,
            False,
        ),
        "Torch up": lambda: torch_gmm(
            hidden,
            up_weight,
            sizes,
            False,
        ),
        "Triton up": lambda: up(hidden, sizes),
        "cuTile down": lambda: cutile_down(a, sizes),
        "cuTile up": lambda: cutile_up(cutile_hidden, sizes),
    }
    timings = {
        name: benchmark(operation)
        for name, operation in operations.items()
    }

    def cutlass_down_up() -> None:
        current = gmm(a, down_weight, sizes, True)
        gmm(current, up_weight, sizes, False)

    def torch_down_up() -> None:
        current = torch_gmm(a, down_weight, sizes, True)
        torch_gmm(current, up_weight, sizes, False)

    def triton_down_up() -> None:
        current = down(a, sizes)
        up(current, sizes)

    timings["CUTLASS total"] = benchmark(cutlass_down_up)
    timings["Torch total"] = benchmark(torch_down_up)
    timings["Triton total"] = benchmark(triton_down_up)
    timings["Triton fused"] = benchmark(lambda: fused(a, sizes))

    def cutile_down_up() -> None:
        current = cutile_down(a, sizes)
        cutile_up(current, sizes)

    timings["cuTile total"] = benchmark(cutile_down_up)
    timings["cuTile fused"] = benchmark(
        lambda: cutile_fused(a, sizes)
    )

    def triton_down_rebuild() -> None:
        down.clear_metadata_cache()
        down(a, sizes)

    def triton_up_rebuild() -> None:
        up.clear_metadata_cache()
        up(hidden, sizes)

    def triton_total_rebuild() -> None:
        down.clear_metadata_cache()
        up.clear_metadata_cache()
        current = down(a, sizes)
        up(current, sizes)

    timings["Triton rebuild down"] = benchmark(
        triton_down_rebuild
    )
    timings["Triton rebuild up"] = benchmark(triton_up_rebuild)
    timings["Triton rebuild total"] = benchmark(
        triton_total_rebuild
    )

    def triton_fused_rebuild() -> None:
        fused.clear_metadata_cache()
        fused(a, sizes)

    timings["Triton rebuild fused"] = benchmark(
        triton_fused_rebuild
    )

    LOGGER.info(
        (
            "%-8s | %12s | %12s | %12s | %12s | "
            "%14s | %14s"
        ),
        "stage",
        "CUTLASS(us)",
        "Torch(us)",
        "Triton(us)",
        "cuTile(us)",
        "CUTLASS/Triton",
        "CUTLASS/cuTile",
    )
    LOGGER.info("-" * 111)
    for stage in ("down", "up", "total"):
        cutlass_us = timings[f"CUTLASS {stage}"]
        torch_us = timings[f"Torch {stage}"]
        triton_us = timings[f"Triton {stage}"]
        cutile_us = timings[f"cuTile {stage}"]
        LOGGER.info(
            (
                "%-8s | %12.3f | %12.3f | %12.3f | "
                "%12.3f | %13.3fx | %13.3fx"
            ),
            stage,
            cutlass_us,
            torch_us,
            triton_us,
            cutile_us,
            cutlass_us / triton_us,
            cutlass_us / cutile_us,
        )
    LOGGER.info(
        (
            "%-8s | %12s | %12s | %12.3f | "
            "%12.3f | %13.3fx | %13.3fx"
        ),
        "fused",
        "-",
        "-",
        timings["Triton fused"],
        timings["cuTile fused"],
        timings["Triton total"] / timings["Triton fused"],
        timings["cuTile total"] / timings["cuTile fused"],
    )
    LOGGER.info(
        (
            "speedup：down/up/total=CUTLASS/实现，"
            "fused=各实现 separate/fused；大于 1 表示后者更快"
        )
    )
    LOGGER.info(
        (
            "Triton metadata rebuild(us)："
            "down=%.3f up=%.3f total=%.3f fused=%.3f"
        ),
        timings["Triton rebuild down"],
        timings["Triton rebuild up"],
        timings["Triton rebuild total"],
        timings["Triton rebuild fused"],
    )

    grad_output = torch.randn_like(a)

    def cutlass_forward_backward() -> None:
        input_value = a.detach().requires_grad_(True)
        down_value = down_weight.detach().requires_grad_(True)
        up_value = up_weight.detach().requires_grad_(True)
        output = lora_gmm(
            input_value,
            down_value,
            up_value,
            sizes,
        )
        output.backward(grad_output)

    def triton_forward_backward() -> None:
        input_value = a.detach().requires_grad_(True)
        down_value = down_weight.detach().requires_grad_(True)
        up_value = up_weight.detach().requires_grad_(True)
        output = triton_fused_lora(
            input_value,
            down_value,
            up_value,
            sizes,
        )
        output.backward(grad_output)

    def cutile_forward_backward() -> None:
        input_value = a.detach().requires_grad_(True)
        down_value = down_weight.detach().requires_grad_(True)
        up_value = up_weight.detach().requires_grad_(True)
        output = cutile_fused_lora(
            input_value,
            down_value,
            up_value,
            sizes,
        )
        output.backward(grad_output)

    cutlass_backward_us = benchmark(cutlass_forward_backward)
    triton_backward_us = benchmark(triton_forward_backward)
    cutile_backward_us = benchmark(cutile_forward_backward)
    LOGGER.info("")
    LOGGER.info(
        (
            "%-18s | %16s | %18s | %12s | %10s"
        ),
        "implementation",
        "backward kernels",
        "forward+backward",
        "speedup",
        "result",
    )
    LOGGER.info("-" * 88)
    LOGGER.info(
        "%-18s | %16d | %15.3f us | %12s | %10s",
        "CUTLASS separate",
        4,
        cutlass_backward_us,
        "-",
        "baseline",
    )
    LOGGER.info(
        "%-18s | %16d | %15.3f us | %11.3fx | %10s",
        "Triton fused",
        3,
        triton_backward_us,
        cutlass_backward_us / triton_backward_us,
        "pass",
    )
    LOGGER.info(
        "%-18s | %16d | %15.3f us | %11.3fx | %10s",
        "cuTile fused",
        3,
        cutile_backward_us,
        cutlass_backward_us / cutile_backward_us,
        "pass",
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
    LOGGER.info("阶段：LoRA down/up 分阶段前向精度验证")
    torch.manual_seed(11)
    run_accuracy()
    LOGGER.info("")
    LOGGER.info("阶段：LoRA fused backward 精度验证")
    run_backward_accuracy()
    LOGGER.info("")
    LOGGER.info(
        (
            "阶段：LoRA down/up 分阶段端到端性能对比，"
            "hidden_size=2048 warmup=%d iterations=%d"
        ),
        WARMUP_ITERATIONS,
        BENCHMARK_ITERATIONS,
    )
    run_performance()
    LOGGER.info("")
    LOGGER.info("[SUCCESS] cudaop_grouped_gemm 对比测试通过")


if __name__ == "__main__":
    main()
