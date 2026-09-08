"""对比 CUTLASS Grouped GEMM 的 K32/K16 与 K16/K8 配置。"""

import logging
import statistics
from collections.abc import Callable

import torch

from cudaop_grouped_gemm import _C, lora_gmm, lora_gmm_k16


LOGGER = logging.getLogger("cutlass_grouped_gemm_k_tile_test")

BF16_RTOL = 2e-2
BF16_ATOL = 2e-2
BF16_GRAD_ATOL = 5e-2
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
BENCHMARK_SAMPLES = 5
PERFORMANCE_SIZES = [
    value * 30
    for value in (128, 157, 97, 100, 111, 129, 138, 101)
]


def max_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def reference_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    transpose_a: bool,
    transpose_b: bool,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for group, rows in enumerate(batch_sizes.tolist()):
        group_a = a[offset:offset + rows].float()
        if transpose_a:
            group_b = b[offset:offset + rows].float()
            outputs.append(group_a.transpose(0, 1) @ group_b)
        else:
            group_b = b[group].float()
            if transpose_b:
                group_b = group_b.transpose(0, 1)
            outputs.append(group_a @ group_b)
        offset += rows
    if transpose_a:
        return torch.stack(outputs, dim=0)
    return torch.cat(outputs, dim=0)


def run_layout_accuracy() -> None:
    torch.manual_seed(2026)
    sizes = torch.tensor([17, 32, 63], dtype=torch.int64)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    cases = (
        (
            "NN, K=16",
            torch.randn(
                tokens,
                16,
                device="cuda",
                dtype=torch.bfloat16,
            ) * 0.1,
            torch.randn(
                experts,
                16,
                80,
                device="cuda",
                dtype=torch.bfloat16,
            ) * 0.1,
            False,
            False,
        ),
        (
            "NN, K=32",
            torch.randn(
                tokens,
                32,
                device="cuda",
                dtype=torch.bfloat16,
            ) * 0.1,
            torch.randn(
                experts,
                32,
                80,
                device="cuda",
                dtype=torch.bfloat16,
            ) * 0.1,
            False,
            False,
        ),
        (
            "NT, K=40",
            torch.randn(
                tokens,
                40,
                device="cuda",
                dtype=torch.bfloat16,
            ) * 0.1,
            torch.randn(
                experts,
                24,
                40,
                device="cuda",
                dtype=torch.bfloat16,
            ) * 0.1,
            False,
            True,
        ),
        (
            "TN, K=rows",
            torch.randn(
                tokens,
                32,
                device="cuda",
                dtype=torch.bfloat16,
            ) * 0.1,
            torch.randn(
                tokens,
                48,
                device="cuda",
                dtype=torch.bfloat16,
            ) * 0.1,
            True,
            False,
        ),
    )

    LOGGER.info(
        "%-14s | %14s | %14s | %14s",
        "layout",
        "K32 error",
        "K16 error",
        "K32/K16 diff",
    )
    LOGGER.info("-" * 65)
    for name, a, b, transpose_a, transpose_b in cases:
        expected = reference_grouped_gemm(
            a,
            b,
            sizes,
            transpose_a,
            transpose_b,
        )
        baseline = _C.grouped_gemm(
            a,
            b,
            sizes,
            transpose_a,
            transpose_b,
        )
        candidate = _C.grouped_gemm_k16(
            a,
            b,
            sizes,
            transpose_a,
            transpose_b,
        )
        for actual in (baseline, candidate):
            torch.testing.assert_close(
                actual.float(),
                expected,
                rtol=BF16_RTOL,
                atol=BF16_ATOL,
            )
        LOGGER.info(
            "%-14s | %14.6f | %14.6f | %14.6f",
            name,
            max_error(baseline, expected),
            max_error(candidate, expected),
            max_error(baseline, candidate),
        )


def reference_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for group, rows in enumerate(batch_sizes.tolist()):
        group_a = a[offset:offset + rows]
        hidden = group_a @ down_weight[group].transpose(0, 1)
        outputs.append(hidden @ up_weight[group])
        offset += rows
    return torch.cat(outputs, dim=0)


def run_autograd_accuracy() -> None:
    torch.manual_seed(2027)
    sizes = torch.tensor([19, 33, 51], dtype=torch.int64)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = 128
    rank = 16
    source_a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    source_down = torch.randn(
        experts,
        rank,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    source_up = torch.randn_like(source_down) * 0.1
    grad_output = torch.randn_like(source_a) * 0.1

    def execute(
        operation: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        use_float: bool,
    ) -> tuple[torch.Tensor, ...]:
        dtype = torch.float32 if use_float else torch.bfloat16
        a = source_a.to(dtype).detach().requires_grad_(True)
        down_weight = (
            source_down.to(dtype).detach().requires_grad_(True)
        )
        up_weight = source_up.to(dtype).detach().requires_grad_(True)
        output = operation(a, down_weight, up_weight, sizes)
        output.backward(grad_output.to(dtype))
        return (
            output.detach(),
            a.grad.detach(),
            down_weight.grad.detach(),
            up_weight.grad.detach(),
        )

    reference = execute(reference_lora, True)
    implementations = (
        ("K32/K16", execute(lora_gmm, False)),
        ("K16/K8", execute(lora_gmm_k16, False)),
    )
    tensor_names = ("output", "grad input", "grad down", "grad up")
    LOGGER.info(
        "%-10s | %-12s | %14s",
        "config",
        "tensor",
        "FP32 error",
    )
    LOGGER.info("-" * 43)
    for config_name, results in implementations:
        for tensor_name, actual, expected in zip(
            tensor_names,
            results,
            reference,
        ):
            torch.testing.assert_close(
                actual.float(),
                expected,
                rtol=BF16_RTOL,
                atol=BF16_GRAD_ATOL,
            )
            LOGGER.info(
                "%-10s | %-12s | %14.6f",
                config_name,
                tensor_name,
                max_error(actual, expected),
            )


def benchmark_once(operation: Callable[[], torch.Tensor]) -> float:
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
    return (
        start.elapsed_time(end)
        * 1000.0
        / BENCHMARK_ITERATIONS
    )


def benchmark_pair(
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
) -> tuple[float, float]:
    baseline_samples = []
    candidate_samples = []
    for sample in range(BENCHMARK_SAMPLES):
        first, second = (
            (baseline, candidate)
            if sample % 2 == 0
            else (candidate, baseline)
        )
        first_us = benchmark_once(first)
        second_us = benchmark_once(second)
        if sample % 2 == 0:
            baseline_samples.append(first_us)
            candidate_samples.append(second_us)
        else:
            candidate_samples.append(first_us)
            baseline_samples.append(second_us)
    return (
        statistics.median(baseline_samples),
        statistics.median(candidate_samples),
    )


def run_performance_case(
    hidden_size: int,
    rank: int,
) -> None:
    torch.manual_seed(hidden_size + rank)
    sizes = torch.tensor(PERFORMANCE_SIZES, dtype=torch.int64)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    a = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    down_weight = torch.randn(
        experts,
        rank,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    up_weight = torch.randn_like(down_weight) * 0.1
    hidden = torch.randn(
        tokens,
        rank,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    grad_hidden = torch.randn_like(hidden) * 0.1
    grad_output = torch.randn_like(a) * 0.1
    train_a = a.detach().requires_grad_(True)
    train_down_weight = down_weight.detach().requires_grad_(True)
    train_up_weight = up_weight.detach().requires_grad_(True)

    def baseline_down() -> torch.Tensor:
        return _C.grouped_gemm(
            a,
            down_weight,
            sizes,
            False,
            True,
        )

    def candidate_down() -> torch.Tensor:
        return _C.grouped_gemm_k16(
            a,
            down_weight,
            sizes,
            False,
            True,
        )

    def baseline_up() -> torch.Tensor:
        return _C.grouped_gemm(
            hidden,
            up_weight,
            sizes,
            False,
            False,
        )

    def candidate_up() -> torch.Tensor:
        return _C.grouped_gemm_k16(
            hidden,
            up_weight,
            sizes,
            False,
            False,
        )

    def baseline_bgrad() -> torch.Tensor:
        return _C.grouped_gemm(
            grad_hidden,
            a,
            sizes,
            True,
            False,
        )

    def candidate_bgrad() -> torch.Tensor:
        return _C.grouped_gemm_k16(
            grad_hidden,
            a,
            sizes,
            True,
            False,
        )

    def baseline_lora() -> torch.Tensor:
        local_hidden = baseline_down()
        return _C.grouped_gemm(
            local_hidden,
            up_weight,
            sizes,
            False,
            False,
        )

    def candidate_lora() -> torch.Tensor:
        local_hidden = candidate_down()
        return _C.grouped_gemm_k16(
            local_hidden,
            up_weight,
            sizes,
            False,
            False,
        )

    def clear_training_gradients() -> None:
        train_a.grad = None
        train_down_weight.grad = None
        train_up_weight.grad = None

    def baseline_training() -> torch.Tensor:
        clear_training_gradients()
        output = lora_gmm(
            train_a,
            train_down_weight,
            train_up_weight,
            sizes,
        )
        output.backward(grad_output)
        return output

    def candidate_training() -> torch.Tensor:
        clear_training_gradients()
        output = lora_gmm_k16(
            train_a,
            train_down_weight,
            train_up_weight,
            sizes,
        )
        output.backward(grad_output)
        return output

    operations = (
        ("down", str(hidden_size), baseline_down, candidate_down),
        ("up", str(rank), baseline_up, candidate_up),
        ("bgrad", "rows", baseline_bgrad, candidate_bgrad),
        ("lora forward", "mixed", baseline_lora, candidate_lora),
        (
            "lora fwd+bwd",
            "mixed",
            baseline_training,
            candidate_training,
        ),
    )
    for stage, logical_k, baseline, candidate in operations:
        expected = baseline()
        actual = candidate()
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=BF16_RTOL,
            atol=1e-1,
        )
        del expected
        del actual
        baseline_us, candidate_us = benchmark_pair(
            baseline,
            candidate,
        )
        LOGGER.info(
            "%6d/%-4d | %-12s | %8s | %12.3f | %12.3f | %10.3fx",
            hidden_size,
            rank,
            stage,
            logical_k,
            baseline_us,
            candidate_us,
            baseline_us / candidate_us,
        )


def run_performance() -> None:
    LOGGER.info(
        "%11s | %-12s | %8s | %12s | %12s | %11s",
        "H/R",
        "stage",
        "logical K",
        "K32/K16(us)",
        "K16/K8(us)",
        "K16 speedup",
    )
    LOGGER.info("-" * 85)
    for hidden_size, rank in (
        (2048, 16),
        (8192, 16),
        (8192, 32),
    ):
        run_performance_case(hidden_size, rank)
        torch.cuda.empty_cache()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("m16n8k8 BF16 测试要求 SM80 或更高架构")

    LOGGER.info(
        (
            "配置：device=%s arch=sm_%d%d dtype=bfloat16 "
            "baseline=TB128x128x32/W64x64x32/m16n8k16 "
            "candidate=TB128x128x16/W64x64x16/m16n8k8"
        ),
        torch.cuda.get_device_name(),
        major,
        minor,
    )
    LOGGER.info("")
    LOGGER.info("阶段：NN、NT、TN 三种布局正确性验证")
    run_layout_accuracy()
    LOGGER.info("")
    LOGGER.info("阶段：LoRA 前向与自动求导正确性验证")
    run_autograd_accuracy()
    LOGGER.info("")
    LOGGER.info(
        (
            "阶段：LoRA 代表 shape 性能对比，tokens=%d "
            "warmup=%d iterations=%d samples=%d"
        ),
        sum(PERFORMANCE_SIZES),
        WARMUP_ITERATIONS,
        BENCHMARK_ITERATIONS,
        BENCHMARK_SAMPLES,
    )
    run_performance()
    LOGGER.info("")
    LOGGER.info("[SUCCESS] CUTLASS Grouped GEMM K tile 对比测试通过")


if __name__ == "__main__":
    main()
