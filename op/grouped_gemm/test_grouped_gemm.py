"""CUTLASS、PyTorch 与分阶段 Triton LoRA Grouped GEMM 对比。"""

import logging
import statistics
from collections.abc import Callable

import torch

from cudaop_grouped_gemm import (
    LoraDownGrouped,
    LoraFusedDownUpGrouped,
    LoraUpGrouped,
    gmm,
    torch_gmm,
)


LOGGER = logging.getLogger("cudaop_grouped_gemm_test")

BF16_RTOL = 2e-2
BF16_ATOL = 2e-2
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

    expected_hidden = reference_down(
        a.float(),
        down_weight.float(),
        sizes,
    )
    cutlass_hidden = gmm(a, down_weight, sizes, True)
    torch_hidden = torch_gmm(a, down_weight, sizes, True)
    triton_hidden = down(a, sizes)
    for actual in (cutlass_hidden, torch_hidden, triton_hidden):
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
    fused_hidden, fused_output = fused(a, sizes)
    for actual in (cutlass_output, torch_output, triton_output):
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

    LOGGER.info(
        (
            "%-8s | %-14s | %14s | %14s | "
            "%14s | %14s"
        ),
        "stage",
        "output shape",
        "CUTLASS error",
        "Torch error",
        "Triton error",
        "CUTLASS/Triton",
    )
    LOGGER.info("-" * 100)
    accuracy_rows = (
        (
            "down",
            triton_hidden,
            expected_hidden,
            cutlass_hidden,
            torch_hidden,
        ),
        (
            "up",
            triton_output,
            expected_output,
            cutlass_output,
            torch_output,
        ),
    )
    for stage, triton_value, expected, cutlass_value, torch_value in (
        accuracy_rows
    ):
        LOGGER.info(
            (
                "%-8s | %-14s | %14.6f | %14.6f | "
                "%14.6f | %14.6f"
            ),
            stage,
            str(tuple(triton_value.shape)),
            max_error(cutlass_value, expected),
            max_error(torch_value, expected),
            max_error(triton_value, expected),
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
    hidden = down(a, sizes)

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
            "%-8s | %12s | %12s | %12s | "
            "%16s | %18s"
        ),
        "stage",
        "CUTLASS(us)",
        "Torch(us)",
        "Triton(us)",
        "speedup",
        "metadata rebuild",
    )
    LOGGER.info("-" * 96)
    for stage in ("down", "up", "total"):
        cutlass_us = timings[f"CUTLASS {stage}"]
        torch_us = timings[f"Torch {stage}"]
        triton_us = timings[f"Triton {stage}"]
        LOGGER.info(
            (
                "%-8s | %12.3f | %12.3f | %12.3f | "
                "%15.3fx | %18.3f"
            ),
            stage,
            cutlass_us,
            torch_us,
            triton_us,
            cutlass_us / triton_us,
            timings[f"Triton rebuild {stage}"],
        )
    LOGGER.info(
        (
            "%-8s | %12s | %12s | %12.3f | "
            "%15.3fx | %18.3f"
        ),
        "fused",
        "-",
        "-",
        timings["Triton fused"],
        timings["Triton total"] / timings["Triton fused"],
        timings["Triton rebuild fused"],
    )
    LOGGER.info(
        (
            "speedup：down/up/total=CUTLASS/Triton，"
            "fused=separate/fused；大于 1 表示 Triton 更快"
        )
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
