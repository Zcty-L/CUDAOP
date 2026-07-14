"""CUTLASS、Triton 与 cuTile LoRA Grouped GEMM 训练吞吐对比。"""

import logging
import statistics
from collections.abc import Callable
from typing import Any

import torch

from cudaop_grouped_gemm import (
    CuTileLoraBgradGrouped,
    CuTileLoraDownGrouped,
    CuTileLoraFusedAgradGrouped,
    CuTileLoraFusedDownUpGrouped,
    CuTileLoraUpGrouped,
    LoraBgradGrouped,
    LoraDownGrouped,
    LoraFusedAgradGrouped,
    LoraFusedDownUpGrouped,
    LoraUpGrouped,
    gmm,
    lora_gmm,
)


LOGGER = logging.getLogger("cudaop_grouped_gemm_test")

BF16_RTOL = 2e-2
BF16_ATOL = 2e-2
BF16_GRAD_ATOL = 5e-1
WARMUP_ITERATIONS = 20
BENCHMARK_ITERATIONS = 100
HIDDEN_SIZES = (2048, 8192)
SIZES = [128, 157, 97, 100, 111, 129, 138, 101]
SIZES = [i * 30 for i in SIZES]


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


class _TritonSeparateLora(torch.autograd.Function):
    """两 kernel 前向、四 kernel 反向的 Triton 训练路径。"""

    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        sizes: torch.Tensor,
        down: LoraDownGrouped,
        up: LoraUpGrouped,
        backward_down: LoraDownGrouped,
        backward_up: LoraUpGrouped,
        bgrad: LoraBgradGrouped,
    ) -> torch.Tensor:
        hidden = down(a, sizes)
        output = up(hidden, sizes)
        context.save_for_backward(a, hidden, sizes)
        context.backward_down = backward_down
        context.backward_up = backward_up
        context.bgrad = bgrad
        return output

    @staticmethod
    def backward(
        context: Any,
        grad_output: torch.Tensor,
    ) -> tuple:
        a, hidden, sizes = context.saved_tensors
        grad_output = grad_output.contiguous()
        grad_hidden = context.backward_down(grad_output, sizes)
        grad_input = context.backward_up(grad_hidden, sizes)
        grad_down_weight = context.bgrad(grad_hidden, a, sizes)
        grad_up_weight = context.bgrad(hidden, grad_output, sizes)
        return (
            grad_input,
            grad_down_weight,
            grad_up_weight,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _TritonFusedLora(torch.autograd.Function):
    """单 kernel 前向、三 kernel 反向的 Triton 训练路径。"""

    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        sizes: torch.Tensor,
        fused: LoraFusedDownUpGrouped,
        agrad: LoraFusedAgradGrouped,
        bgrad: LoraBgradGrouped,
    ) -> torch.Tensor:
        hidden, output = fused(a, sizes)
        context.save_for_backward(a, hidden, sizes)
        context.agrad = agrad
        context.bgrad = bgrad
        return output

    @staticmethod
    def backward(
        context: Any,
        grad_output: torch.Tensor,
    ) -> tuple:
        a, hidden, sizes = context.saved_tensors
        grad_output = grad_output.contiguous()
        grad_hidden, grad_input = context.agrad(grad_output, sizes)
        grad_down_weight = context.bgrad(grad_hidden, a, sizes)
        grad_up_weight = context.bgrad(hidden, grad_output, sizes)
        return (
            grad_input,
            grad_down_weight,
            grad_up_weight,
            None,
            None,
            None,
            None,
        )


class _CuTileSeparateLora(torch.autograd.Function):
    """两 kernel 前向、四 kernel 反向的 cuTile 训练路径。"""

    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        sizes: torch.Tensor,
        down: CuTileLoraDownGrouped,
        up: CuTileLoraUpGrouped,
        backward_down: CuTileLoraDownGrouped,
        backward_up: CuTileLoraUpGrouped,
        bgrad: CuTileLoraBgradGrouped,
    ) -> torch.Tensor:
        hidden = down(a, sizes)
        output = up(hidden, sizes)
        context.save_for_backward(a, hidden, sizes)
        context.backward_down = backward_down
        context.backward_up = backward_up
        context.bgrad = bgrad
        return output

    @staticmethod
    def backward(
        context: Any,
        grad_output: torch.Tensor,
    ) -> tuple:
        a, hidden, sizes = context.saved_tensors
        grad_output = grad_output.contiguous()
        grad_hidden = context.backward_down(grad_output, sizes)
        grad_input = context.backward_up(grad_hidden, sizes)
        grad_down_weight = context.bgrad(grad_hidden, a, sizes)
        grad_up_weight = context.bgrad(hidden, grad_output, sizes)
        return (
            grad_input,
            grad_down_weight,
            grad_up_weight,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _CuTileFusedLora(torch.autograd.Function):
    """单 kernel 前向、三 kernel 反向的 cuTile 训练路径。"""

    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        sizes: torch.Tensor,
        fused: CuTileLoraFusedDownUpGrouped,
        agrad: CuTileLoraFusedAgradGrouped,
        bgrad: CuTileLoraBgradGrouped,
    ) -> torch.Tensor:
        hidden, output = fused(a, sizes)
        context.save_for_backward(a, hidden, sizes)
        context.agrad = agrad
        context.bgrad = bgrad
        return output

    @staticmethod
    def backward(
        context: Any,
        grad_output: torch.Tensor,
    ) -> tuple:
        a, hidden, sizes = context.saved_tensors
        grad_output = grad_output.contiguous()
        grad_hidden, grad_input = context.agrad(
            grad_output,
            sizes,
        )
        grad_down_weight = context.bgrad(grad_hidden, a, sizes)
        grad_up_weight = context.bgrad(hidden, grad_output, sizes)
        return (
            grad_input,
            grad_down_weight,
            grad_up_weight,
            None,
            None,
            None,
            None,
        )


def create_triton_separate_operations(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple:
    hidden_size = down_weight.shape[2]
    experts = down_weight.shape[0]
    return (
        LoraDownGrouped(down_weight),
        LoraUpGrouped(up_weight),
        LoraDownGrouped(up_weight),
        LoraUpGrouped(down_weight),
        LoraBgradGrouped(experts, hidden_size),
    )


def create_triton_fused_operations(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple:
    hidden_size = down_weight.shape[2]
    experts = down_weight.shape[0]
    return (
        LoraFusedDownUpGrouped(down_weight, up_weight),
        LoraFusedAgradGrouped(up_weight, down_weight),
        LoraBgradGrouped(experts, hidden_size),
    )


def create_cutile_separate_operations(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple:
    hidden_size = down_weight.shape[2]
    experts = down_weight.shape[0]
    return (
        CuTileLoraDownGrouped(down_weight),
        CuTileLoraUpGrouped(up_weight),
        CuTileLoraDownGrouped(up_weight),
        CuTileLoraUpGrouped(down_weight),
        CuTileLoraBgradGrouped(experts, hidden_size),
    )


def create_cutile_fused_operations(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> tuple:
    hidden_size = down_weight.shape[2]
    experts = down_weight.shape[0]
    return (
        CuTileLoraFusedDownUpGrouped(down_weight, up_weight),
        CuTileLoraFusedAgradGrouped(up_weight, down_weight),
        CuTileLoraBgradGrouped(experts, hidden_size),
    )


def triton_separate_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    operations = create_triton_separate_operations(
        down_weight,
        up_weight,
    )
    return _TritonSeparateLora.apply(
        a,
        down_weight,
        up_weight,
        sizes,
        *operations,
    )


def triton_fused_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    operations = create_triton_fused_operations(
        down_weight,
        up_weight,
    )
    return _TritonFusedLora.apply(
        a,
        down_weight,
        up_weight,
        sizes,
        *operations,
    )


def cutile_separate_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    operations = create_cutile_separate_operations(
        down_weight,
        up_weight,
    )
    return _CuTileSeparateLora.apply(
        a,
        down_weight,
        up_weight,
        sizes,
        *operations,
    )


def cutile_fused_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    operations = create_cutile_fused_operations(
        down_weight,
        up_weight,
    )
    return _CuTileFusedLora.apply(
        a,
        down_weight,
        up_weight,
        sizes,
        *operations,
    )


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
    triton_hidden = down(a, sizes)
    cutile_hidden = cutile_down(a, sizes)
    for actual in (
        cutlass_hidden,
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
    triton_output = up(triton_hidden, sizes)
    cutile_output = cutile_up(cutile_hidden, sizes)
    fused_hidden, fused_output = fused(a, sizes)
    cutile_fused_hidden, cutile_fused_output = cutile_fused(
        a,
        sizes,
    )
    for actual in (
        cutlass_output,
        triton_output,
        cutile_output,
        fused_output,
        cutile_fused_output,
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
        cutile_fused_hidden.float(),
        expected_hidden,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )

    LOGGER.info(
        "%-8s | %-14s | %14s | %14s | %14s",
        "stage",
        "output shape",
        "CUTLASS error",
        "Triton error",
        "CUTLASS/Triton",
    )
    LOGGER.info("-" * 83)
    accuracy_rows = (
        (
            "down",
            triton_hidden,
            expected_hidden,
            cutlass_hidden,
        ),
        (
            "up",
            triton_output,
            expected_output,
            cutlass_output,
        ),
    )
    for (
        stage,
        triton_value,
        expected,
        cutlass_value,
    ) in (
        accuracy_rows
    ):
        LOGGER.info(
            "%-8s | %-14s | %14.6f | %14.6f | %14.6f",
            stage,
            str(tuple(triton_value.shape)),
            max_error(cutlass_value, expected),
            max_error(triton_value, expected),
            max_error(cutlass_value, triton_value),
        )

    LOGGER.info("")
    LOGGER.info(
        "%-8s | %14s | %14s",
        "stage",
        "cuTile error",
        "cuTile/Triton",
    )
    LOGGER.info("-" * 44)
    for stage, cutile_value, expected, triton_value in (
        (
            "down",
            cutile_hidden,
            expected_hidden,
            triton_hidden,
        ),
        (
            "up",
            cutile_output,
            expected_output,
            triton_output,
        ),
    ):
        LOGGER.info(
            "%-8s | %14.6f | %14.6f",
            stage,
            max_error(cutile_value, expected),
            max_error(cutile_value, triton_value),
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
        "cuTile hidden",
        str(tuple(cutile_fused_hidden.shape)),
        max_error(cutile_fused_hidden, triton_hidden),
    )
    LOGGER.info(
        "%-14s | %-14s | %18.6f",
        "cuTile output",
        str(tuple(cutile_fused_output.shape)),
        max_error(cutile_fused_output, triton_output),
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
    separate_results = execute(triton_separate_lora)
    fused_results = execute(triton_fused_lora)
    cutile_separate_results = execute(cutile_separate_lora)
    cutile_fused_results = execute(cutile_fused_lora)
    names = (
        "output",
        "grad input",
        "grad down",
        "grad up",
    )
    for implementation in (
        separate_results,
        fused_results,
        cutile_separate_results,
        cutile_fused_results,
    ):
        for actual, expected in zip(
            implementation,
            cutlass_results,
        ):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=BF16_RTOL,
                atol=BF16_GRAD_ATOL,
            )

    LOGGER.info(
        (
            "%-12s | %-18s | %16s | %16s | "
            "%16s | %16s"
        ),
        "tensor",
        "shape",
        "Triton separate",
        "Triton fused",
        "cuTile separate",
        "cuTile fused",
    )
    LOGGER.info("-" * 108)
    for (
        name,
        separate_value,
        fused_value,
        cutile_separate_value,
        cutile_fused_value,
        cutlass_value,
    ) in zip(
        names,
        separate_results,
        fused_results,
        cutile_separate_results,
        cutile_fused_results,
        cutlass_results,
    ):
        LOGGER.info(
            (
                "%-12s | %-18s | %16.6f | %16.6f | "
                "%16.6f | %16.6f"
            ),
            name,
            str(tuple(separate_value.shape)),
            max_error(separate_value, cutlass_value),
            max_error(fused_value, cutlass_value),
            max_error(cutile_separate_value, cutlass_value),
            max_error(cutile_fused_value, cutlass_value),
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


def run_performance(hidden_size: int) -> None:
    sizes = torch.tensor(SIZES)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    down_weight = torch.randn(
        experts,
        16,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ).requires_grad_(True)
    up_weight = torch.randn_like(
        down_weight,
        requires_grad=True,
    )
    grad_output = torch.randn_like(a)
    separate_operations = create_triton_separate_operations(
        down_weight,
        up_weight,
    )
    fused_operations = create_triton_fused_operations(
        down_weight,
        up_weight,
    )
    cutile_separate_operations = create_cutile_separate_operations(
        down_weight,
        up_weight,
    )
    cutile_fused_operations = create_cutile_fused_operations(
        down_weight,
        up_weight,
    )

    def clear_gradients() -> None:
        a.grad = None
        down_weight.grad = None
        up_weight.grad = None

    def cutlass_forward() -> torch.Tensor:
        return lora_gmm(
            a,
            down_weight,
            up_weight,
            sizes,
        )

    def triton_forward() -> torch.Tensor:
        return _TritonSeparateLora.apply(
            a,
            down_weight,
            up_weight,
            sizes,
            *separate_operations,
        )

    def triton_fused_forward() -> torch.Tensor:
        return _TritonFusedLora.apply(
            a,
            down_weight,
            up_weight,
            sizes,
            *fused_operations,
        )

    def cutile_separate_forward() -> torch.Tensor:
        return _CuTileSeparateLora.apply(
            a,
            down_weight,
            up_weight,
            sizes,
            *cutile_separate_operations,
        )

    def cutile_fused_forward() -> torch.Tensor:
        return _CuTileFusedLora.apply(
            a,
            down_weight,
            up_weight,
            sizes,
            *cutile_fused_operations,
        )

    implementations = (
        ("CUTLASS grouped_gemm", cutlass_forward),
        ("Triton separate", triton_forward),
        ("Triton fused", triton_fused_forward),
        ("cuTile separate", cutile_separate_forward),
        ("cuTile fused", cutile_fused_forward),
    )
    timings = []
    for name, forward in implementations:
        saved_output = forward()

        def backward(
            output: torch.Tensor = saved_output,
        ) -> None:
            clear_gradients()
            output.backward(
                grad_output,
                retain_graph=True,
            )

        def forward_backward() -> None:
            clear_gradients()
            output = forward()
            output.backward(grad_output)

        forward_us = benchmark(forward)
        backward_us = benchmark(backward)
        total_us = benchmark(forward_backward)
        timings.append(
            (name, forward_us, backward_us, total_us)
        )
        del saved_output

    baseline_us = timings[0][3]

    LOGGER.info(
        (
            "%-24s | %12s | %12s | %14s | "
            "%20s | %14s"
        ),
        "implementation",
        "forward(us)",
        "backward(us)",
        "forward+backward(us)",
        "throughput(Mtok/s)",
        "speedup",
    )
    LOGGER.info("-" * 111)
    for name, forward_us, backward_us, total_us in timings:
        throughput_mtokens = tokens / total_us
        LOGGER.info(
            (
                "%-24s | %12.3f | %12.3f | %20.3f | "
                "%20.3f | %13.3fx"
            ),
            name,
            forward_us,
            backward_us,
            total_us,
            throughput_mtokens,
            baseline_us / total_us,
        )
    LOGGER.info(
        (
            "speedup=CUTLASS forward+backward latency / "
            "implementation forward+backward latency"
        )
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 CUDA GPU")

    LOGGER.info(
        (
            "配置：device=%s dtype=bfloat16 arch=sm_%d%d torch=%s "
            "experts=%d rank=16 tokens=%d sizes=%s"
        ),
        torch.cuda.get_device_name(),
        *torch.cuda.get_device_capability(),
        torch.__version__,
        len(SIZES),
        sum(SIZES),
        SIZES,
    )
    LOGGER.info("")
    LOGGER.info("阶段：LoRA down/up 分阶段前向精度验证")
    torch.manual_seed(11)
    run_accuracy()
    LOGGER.info("")
    LOGGER.info("阶段：LoRA 前向+反向精度验证")
    run_backward_accuracy()
    for hidden_size in HIDDEN_SIZES:
        LOGGER.info("")
        LOGGER.info(
            (
                "阶段：LoRA down/up 前向+反向吞吐对比，"
                "hidden_size=%d warmup=%d iterations=%d samples=5"
            ),
            hidden_size,
            WARMUP_ITERATIONS,
            BENCHMARK_ITERATIONS,
        )
        run_performance(hidden_size)
    LOGGER.info("")
    LOGGER.info("[SUCCESS] cudaop_grouped_gemm 对比测试通过")


if __name__ == "__main__":
    main()
