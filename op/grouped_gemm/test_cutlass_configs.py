"""扫描 CUTLASS Grouped GEMM 的 tile、warp 和 stage 配置。"""

import logging
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from cudaop_grouped_gemm import _C
from cudaop_grouped_gemm import _tuning


LOGGER = logging.getLogger("cutlass_grouped_gemm_config_test")

BF16_RTOL = 2e-2
BF16_ATOL = 2e-2
BF16_GRAD_ATOL = 1e-1
SCAN_WARMUP = 5
SCAN_ITERATIONS = 30
SCAN_SAMPLES = 3
CONFIRM_WARMUP = 10
CONFIRM_ITERATIONS = 100
CONFIRM_SAMPLES = 5
FULL_WARMUP = 10
FULL_ITERATIONS = 50
FULL_SAMPLES = 5
PERFORMANCE_SIZES = [
    value * 30
    for value in (128, 157, 97, 100, 111, 129, 138, 101)
]


Operation = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]
TimedOperation = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class Variant:
    name: str
    operation: Operation


def baseline_up(
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    return _C.grouped_gemm(a, b, sizes, False, False)


def candidate_up(
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    return _C.grouped_gemm_k16(a, b, sizes, False, False)


def baseline_down(
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    return _C.grouped_gemm(a, b, sizes, False, True)


def candidate_down(
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    return _C.grouped_gemm_k16(a, b, sizes, False, True)


def baseline_bgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    return _C.grouped_gemm(a, b, sizes, True, False)


def candidate_bgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    return _C.grouped_gemm_k16(a, b, sizes, True, False)


def collect_variants(
    prefix: str,
    baseline: Operation,
    candidate: Operation,
) -> list[Variant]:
    variants = [
        Variant("baseline_tb128x128_i16_s4", baseline),
        Variant("candidate_tb128x128_i8_s4", candidate),
    ]
    for name in sorted(dir(_tuning)):
        if name.startswith(f"{prefix}_"):
            variants.append(Variant(name, getattr(_tuning, name)))
    return variants


UP_VARIANTS = collect_variants("up", baseline_up, candidate_up)
DOWN_VARIANTS = collect_variants(
    "down",
    baseline_down,
    candidate_down,
)
BGRAD_VARIANTS = collect_variants(
    "bgrad",
    baseline_bgrad,
    candidate_bgrad,
)


def max_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def reference_up(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for group, rows in enumerate(sizes.tolist()):
        outputs.append(
            hidden[offset:offset + rows].float()
            @ weight[group].float()
        )
        offset += rows
    return torch.cat(outputs, dim=0)


def reference_down(
    a: torch.Tensor,
    weight: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for group, rows in enumerate(sizes.tolist()):
        outputs.append(
            a[offset:offset + rows].float()
            @ weight[group].float().transpose(0, 1)
        )
        offset += rows
    return torch.cat(outputs, dim=0)


def reference_bgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for rows in sizes.tolist():
        group_a = a[offset:offset + rows].float()
        group_b = b[offset:offset + rows].float()
        outputs.append(group_a.transpose(0, 1) @ group_b)
        offset += rows
    return torch.stack(outputs, dim=0)


def validate_variants() -> None:
    torch.manual_seed(2030)
    sizes = torch.tensor([17, 32, 63], dtype=torch.int64)
    tokens = int(sizes.sum())
    experts = sizes.numel()
    hidden_size = 128

    LOGGER.info(
        "%-8s | %4s | %8s | %-44s | %14s",
        "path",
        "rank",
        "variants",
        "largest-error config",
        "FP32 error",
    )
    LOGGER.info("-" * 91)
    for rank in (16, 32):
        source = torch.randn(
            tokens,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        ) * 0.1
        weight = torch.randn(
            experts,
            rank,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        ) * 0.1
        hidden = torch.randn(
            tokens,
            rank,
            device="cuda",
            dtype=torch.bfloat16,
        ) * 0.1
        grad_hidden = torch.randn_like(hidden) * 0.1

        cases = (
            (
                "down",
                DOWN_VARIANTS,
                source,
                weight,
                reference_down(source, weight, sizes),
            ),
            (
                "up",
                UP_VARIANTS,
                hidden,
                weight,
                reference_up(hidden, weight, sizes),
            ),
            (
                "bgrad",
                BGRAD_VARIANTS,
                grad_hidden,
                source,
                reference_bgrad(grad_hidden, source, sizes),
            ),
        )
        for path, variants, a, b, expected in cases:
            errors = []
            for variant in variants:
                actual = variant.operation(a, b, sizes)
                torch.testing.assert_close(
                    actual.float(),
                    expected,
                    rtol=BF16_RTOL,
                    atol=BF16_ATOL,
                )
                errors.append((max_error(actual, expected), variant.name))
            error, name = max(errors)
            LOGGER.info(
                "%-8s | %4d | %8d | %-44s | %14.6f",
                path,
                rank,
                len(variants),
                name,
                error,
            )


def benchmark_once(
    operation: TimedOperation,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def benchmark_operations(
    operations: list[tuple[str, TimedOperation]],
    warmup: int,
    iterations: int,
    samples: int,
) -> list[tuple[str, float]]:
    timings = {name: [] for name, _ in operations}
    for sample in range(samples):
        order = list(operations)
        random.Random(4096 + sample).shuffle(order)
        for name, operation in order:
            timings[name].append(
                benchmark_once(operation, warmup, iterations)
            )
    return sorted(
        (
            (name, statistics.median(values))
            for name, values in timings.items()
        ),
        key=lambda item: item[1],
    )


def bind_operations(
    variants: list[Variant],
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
) -> list[tuple[str, TimedOperation]]:
    return [
        (
            variant.name,
            lambda variant=variant: variant.operation(a, b, sizes),
        )
        for variant in variants
    ]


def log_ranking(
    path: str,
    hidden_size: int,
    rank: int,
    ranking: list[tuple[str, float]],
) -> None:
    baseline_us = next(
        latency
        for name, latency in ranking
        if name == "baseline_tb128x128_i16_s4"
    )
    LOGGER.info(
        "%-6s | %6s | %4s | %4s | %-44s | %12s | %9s",
        "path",
        "H",
        "rank",
        "pos",
        "config",
        "latency(us)",
        "speedup",
    )
    LOGGER.info("-" * 103)
    for position, (name, latency_us) in enumerate(ranking, start=1):
        LOGGER.info(
            "%-6s | %6d | %4d | %4d | %-44s | %12.3f | %8.3fx",
            path,
            hidden_size,
            rank,
            position,
            name,
            latency_us,
            baseline_us / latency_us,
        )


def select_confirmed_winner(
    path: str,
    hidden_size: int,
    rank: int,
    variants: list[Variant],
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: torch.Tensor,
) -> Variant:
    operations = bind_operations(variants, a, b, sizes)
    initial = benchmark_operations(
        operations,
        SCAN_WARMUP,
        SCAN_ITERATIONS,
        SCAN_SAMPLES,
    )
    log_ranking(path, hidden_size, rank, initial)

    top_names = {name for name, _ in initial[:3]}
    top_names.add("baseline_tb128x128_i16_s4")
    confirmed_operations = [
        operation
        for operation in operations
        if operation[0] in top_names
    ]
    confirmed = benchmark_operations(
        confirmed_operations,
        CONFIRM_WARMUP,
        CONFIRM_ITERATIONS,
        CONFIRM_SAMPLES,
    )
    LOGGER.info("")
    LOGGER.info("复测前三名：")
    log_ranking(path, hidden_size, rank, confirmed)
    winner_name = confirmed[0][0]
    return next(
        variant
        for variant in variants
        if variant.name == winner_name
    )


class _ConfiguredLora(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        sizes: torch.Tensor,
        down: Operation,
        up: Operation,
        bgrad: Operation,
    ) -> torch.Tensor:
        hidden = down(a, down_weight, sizes)
        output = up(hidden, up_weight, sizes)
        context.save_for_backward(a, hidden, down_weight, up_weight, sizes)
        context.down = down
        context.up = up
        context.bgrad = bgrad
        return output

    @staticmethod
    def backward(
        context: Any,
        grad_output: torch.Tensor,
    ) -> tuple:
        a, hidden, down_weight, up_weight, sizes = (
            context.saved_tensors
        )
        grad_output = grad_output.contiguous()
        grad_hidden = context.down(grad_output, up_weight, sizes)
        grad_input = context.up(grad_hidden, down_weight, sizes)
        grad_down = context.bgrad(grad_hidden, a, sizes)
        grad_up = context.bgrad(hidden, grad_output, sizes)
        return grad_input, grad_down, grad_up, None, None, None, None


def configured_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    sizes: torch.Tensor,
    down: Operation,
    up: Operation,
    bgrad: Operation,
) -> torch.Tensor:
    return _ConfiguredLora.apply(
        a,
        down_weight,
        up_weight,
        sizes,
        down,
        up,
        bgrad,
    )


def execute_lora(
    source_a: torch.Tensor,
    source_down: torch.Tensor,
    source_up: torch.Tensor,
    grad_output: torch.Tensor,
    sizes: torch.Tensor,
    down: Operation,
    up: Operation,
    bgrad: Operation,
) -> tuple[torch.Tensor, ...]:
    a = source_a.detach().clone().requires_grad_(True)
    down_weight = source_down.detach().clone().requires_grad_(True)
    up_weight = source_up.detach().clone().requires_grad_(True)
    output = configured_lora(
        a,
        down_weight,
        up_weight,
        sizes,
        down,
        up,
        bgrad,
    )
    output.backward(grad_output)
    return (
        output.detach(),
        a.grad.detach(),
        down_weight.grad.detach(),
        up_weight.grad.detach(),
    )


def benchmark_full_lora(
    hidden_size: int,
    rank: int,
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    grad_output: torch.Tensor,
    sizes: torch.Tensor,
    winners: tuple[Variant, Variant, Variant],
) -> None:
    LOGGER.info(
        "完整 LoRA：hidden_size=%d rank=%d",
        hidden_size,
        rank,
    )
    winner_down, winner_up, winner_bgrad = winners
    configurations = (
        (
            "baseline K32/K16",
            baseline_down,
            baseline_up,
            baseline_bgrad,
        ),
        (
            "uniform K16/K8",
            candidate_down,
            candidate_up,
            candidate_bgrad,
        ),
        (
            "per-path tuned",
            winner_down.operation,
            winner_up.operation,
            winner_bgrad.operation,
        ),
    )

    expected = execute_lora(
        a,
        down_weight,
        up_weight,
        grad_output,
        sizes,
        baseline_down,
        baseline_up,
        baseline_bgrad,
    )
    for name, down, up, bgrad in configurations[1:]:
        actual = execute_lora(
            a,
            down_weight,
            up_weight,
            grad_output,
            sizes,
            down,
            up,
            bgrad,
        )
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(
                actual_value.float(),
                expected_value.float(),
                rtol=BF16_RTOL,
                atol=BF16_GRAD_ATOL,
            )
        LOGGER.info(
            "完整 LoRA 精度：%-18s 最大差值=%.6f",
            name,
            max(
                max_error(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            ),
        )

    forward_operations = []
    training_operations = []
    for name, down, up, bgrad in configurations:
        forward_operations.append(
            (
                name,
                lambda down=down, up=up: up(
                    down(a, down_weight, sizes),
                    up_weight,
                    sizes,
                ),
            )
        )
        train_a = a.detach().requires_grad_(True)
        train_down = down_weight.detach().requires_grad_(True)
        train_up = up_weight.detach().requires_grad_(True)

        def train(
            down: Operation = down,
            up: Operation = up,
            bgrad: Operation = bgrad,
            train_a: torch.Tensor = train_a,
            train_down: torch.Tensor = train_down,
            train_up: torch.Tensor = train_up,
        ) -> torch.Tensor:
            train_a.grad = None
            train_down.grad = None
            train_up.grad = None
            output = configured_lora(
                train_a,
                train_down,
                train_up,
                sizes,
                down,
                up,
                bgrad,
            )
            output.backward(grad_output)
            return output

        training_operations.append((name, train))

    forward_ranking = benchmark_operations(
        forward_operations,
        FULL_WARMUP,
        FULL_ITERATIONS,
        FULL_SAMPLES,
    )
    training_ranking = benchmark_operations(
        training_operations,
        FULL_WARMUP,
        FULL_ITERATIONS,
        FULL_SAMPLES,
    )
    forward_values = dict(forward_ranking)
    training_values = dict(training_ranking)
    baseline_forward = forward_values["baseline K32/K16"]
    baseline_training = training_values["baseline K32/K16"]

    LOGGER.info(
        "%-18s | %12s | %9s | %14s | %9s",
        "configuration",
        "forward(us)",
        "speedup",
        "fwd+bwd(us)",
        "speedup",
    )
    LOGGER.info("-" * 76)
    for name, _, _, _ in configurations:
        LOGGER.info(
            "%-18s | %12.3f | %8.3fx | %14.3f | %8.3fx",
            name,
            forward_values[name],
            baseline_forward / forward_values[name],
            training_values[name],
            baseline_training / training_values[name],
        )
    LOGGER.info(
        "选择：down=%s up=%s bgrad=%s",
        winner_down.name,
        winner_up.name,
        winner_bgrad.name,
    )


def run_performance_case(hidden_size: int, rank: int) -> None:
    torch.manual_seed(hidden_size + rank + 8192)
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

    LOGGER.info("扫描 down 配置")
    winner_down = select_confirmed_winner(
        "down",
        hidden_size,
        rank,
        DOWN_VARIANTS,
        a,
        down_weight,
        sizes,
    )
    LOGGER.info("")
    LOGGER.info("扫描 up 配置")
    winner_up = select_confirmed_winner(
        "up",
        hidden_size,
        rank,
        UP_VARIANTS,
        hidden,
        up_weight,
        sizes,
    )
    LOGGER.info("")
    LOGGER.info("扫描 bgrad 配置")
    winner_bgrad = select_confirmed_winner(
        "bgrad",
        hidden_size,
        rank,
        BGRAD_VARIANTS,
        grad_hidden,
        a,
        sizes,
    )
    LOGGER.info("")
    LOGGER.info("阶段：组合每条路径赢家并测试完整 LoRA")
    benchmark_full_lora(
        hidden_size,
        rank,
        a,
        down_weight,
        up_weight,
        grad_output,
        sizes,
        (winner_down, winner_up, winner_bgrad),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("测试需要 CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("CUTLASS BF16 Tensor Core 测试要求 SM80+")

    LOGGER.info(
        (
            "配置：device=%s arch=sm_%d%d dtype=bfloat16 "
            "experts=%d tokens=%d up_variants=%d "
            "down_variants=%d bgrad_variants=%d"
        ),
        torch.cuda.get_device_name(),
        major,
        minor,
        len(PERFORMANCE_SIZES),
        sum(PERFORMANCE_SIZES),
        len(UP_VARIANTS),
        len(DOWN_VARIANTS),
        len(BGRAD_VARIANTS),
    )
    LOGGER.info("")
    LOGGER.info("阶段：全部配置 rank=16/32 精度验证")
    validate_variants()
    for hidden_size, rank in (
        (2048, 16),
        (8192, 16),
        (8192, 32),
    ):
        LOGGER.info("")
        LOGGER.info(
            "阶段：全配置性能扫描，hidden_size=%d rank=%d",
            hidden_size,
            rank,
        )
        run_performance_case(hidden_size, rank)
        torch.cuda.empty_cache()
    LOGGER.info("")
    LOGGER.info("[SUCCESS] CUTLASS Grouped GEMM 配置扫描通过")


if __name__ == "__main__":
    main()
