"""LoRAMoENonstandard 的 BF16 精度和反向传播测试。"""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora_moe_nonstandard import LoRAMoENonstandard

from debug_utils import (
    LOGGER,
    log_error,
    log_section,
    log_test_start,
    log_test_success,
)


WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
BACKWARD_WARMUP_ITERATIONS = 5
BACKWARD_BENCHMARK_ITERATIONS = 20


class TestMlp(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = F.silu


def make_module(
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
) -> LoRAMoENonstandard:
    module = LoRAMoENonstandard(
        original_mlp=TestMlp(
            hidden_size,
            intermediate_size,
        ).to(
            device=device,
            dtype=dtype,
        ),
        num_experts=8,
        rank=16,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    with torch.no_grad():
        module.gate_lora_B.normal_()
        module.up_lora_B.normal_()
        module.down_lora_B.normal_()
    return module


def run_method(
    module: LoRAMoENonstandard,
    method: str,
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_section(f"运行 {method} 前向与反向传播测试")
    x_input = x.detach().clone().requires_grad_(True)
    route_weights = (
        weights.detach().clone().requires_grad_(True)
    )
    output = module(
        x_input,
        indices,
        route_weights,
        method=method,
    )
    output.float().square().mean().backward()

    if x_input.grad is None:
        raise AssertionError(f"{method} 未生成输入梯度")
    if route_weights.grad is None:
        raise AssertionError(f"{method} 未生成路由权重梯度")
    for name, parameter in module.named_parameters():
        if "lora_" in name and parameter.grad is None:
            raise AssertionError(
                f"{method} 未生成 LoRA 参数梯度: {name}"
            )
    LOGGER.debug(
        "[DEBUG] %s 完成：output=%s，input_grad=%s，route_grad=%s",
        method,
        tuple(output.shape),
        tuple(x_input.grad.shape),
        tuple(route_weights.grad.shape),
    )
    return output, x_input.grad, route_weights.grad


def assert_results_close(
    actual_results: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    expected_results: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    method: str,
) -> None:
    result_names = (
        "输出",
        "输入梯度",
        "路由权重梯度",
    )
    for name, actual, expected in zip(
        result_names,
        actual_results,
        expected_results,
    ):
        max_absolute_error = (
            actual.float() - expected.float()
        ).abs().max().item()
        log_error(
            method,
            name,
            max_absolute_error,
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=3e-2,
            atol=3e-2,
        )
    LOGGER.info(
        "[DEBUG] %s 与 loop 的精度及梯度对齐通过",
        method,
    )


def benchmark_forward(
    operation: Callable[[], torch.Tensor],
) -> float:
    with torch.inference_mode():
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


def benchmark_backward(
    module: LoRAMoENonstandard,
    operation: Callable[[], torch.Tensor],
    x: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> float:
    def run_once() -> float:
        module.zero_grad(set_to_none=True)
        x.grad = None
        top_k_weights.grad = None
        loss = operation().float().square().mean()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0

    for _ in range(BACKWARD_WARMUP_ITERATIONS):
        run_once()
    return sum(
        run_once()
        for _ in range(BACKWARD_BENCHMARK_ITERATIONS)
    ) / BACKWARD_BENCHMARK_ITERATIONS


def main() -> None:
    torch.manual_seed(11)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.bfloat16
    batch_size = 1
    seq_len = 3507
    top_k = 2
    hidden_size = 2048
    intermediate_size = 8192
    log_test_start(
        "LoRAMoENonstandard",
        f"device={device}，dtype={dtype}，experts=8，rank=16，"
        f"batch={batch_size}，seq_len={seq_len}，top_k={top_k}，"
        f"hidden={hidden_size}，intermediate={intermediate_size}",
    )
    reference_module = make_module(
        device,
        dtype,
        hidden_size,
        intermediate_size,
    )
    pad_module = copy.deepcopy(reference_module)

    x = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    router_logits = torch.randn(
        batch_size,
        seq_len,
        reference_module.num_experts,
        device=device,
        dtype=dtype,
    )
    top_k_logits, indices = torch.topk(
        router_logits,
        top_k,
        dim=-1,
    )
    weights = torch.softmax(
        top_k_logits.float(),
        dim=-1,
    ).to(
        dtype=dtype,
    )

    loop_result = run_method(
        reference_module,
        "loop",
        x,
        indices,
        weights,
    )
    pad_result = run_method(
        pad_module,
        "pad",
        x,
        indices,
        weights,
    )
    assert_results_close(
        pad_result,
        loop_result,
        "pad",
    )

    if torch.cuda.is_available():
        for backend in ("cutlass", "triton", "cutile"):
            group_module = copy.deepcopy(reference_module)
            group_module.gmm_backend = backend
            group_module.zero_grad(set_to_none=True)
            group_result = run_method(
                group_module,
                "group",
                x,
                indices,
                weights,
            )
            assert_results_close(
                group_result,
                loop_result,
                f"group/{backend}",
            )
            reference_parameters = dict(
                reference_module.named_parameters()
            )
            for name, parameter in group_module.named_parameters():
                if "lora_" not in name:
                    continue
                reference_grad = reference_parameters[name].grad
                if parameter.grad is None or reference_grad is None:
                    raise AssertionError(
                        f"group/{backend} 参数梯度缺失: {name}"
                    )
                max_absolute_error = (
                    parameter.grad.float()
                    - reference_grad.float()
                ).abs().max().item()
                log_error(
                    f"group/{backend}",
                    f"{name} 梯度",
                    max_absolute_error,
                )
                torch.testing.assert_close(
                    parameter.grad,
                    reference_grad,
                    rtol=3e-2,
                    atol=3e-2,
                )

        log_section(
            "运行 loop/pad/group 端到端前向与反向性能对比，"
            f"forward={WARMUP_ITERATIONS}+{BENCHMARK_ITERATIONS}，"
            "backward="
            f"{BACKWARD_WARMUP_ITERATIONS}+"
            f"{BACKWARD_BENCHMARK_ITERATIONS}"
        )
        benchmark_x = x.detach()
        benchmark_weights = weights.detach()
        reference_module.eval()

        def loop_forward() -> torch.Tensor:
            return reference_module(
                benchmark_x,
                indices,
                benchmark_weights,
                method="loop",
            )

        def pad_forward() -> torch.Tensor:
            return reference_module(
                benchmark_x,
                indices,
                benchmark_weights,
                method="pad",
            )

        def group_forward() -> torch.Tensor:
            return reference_module(
                benchmark_x,
                indices,
                benchmark_weights,
                method="group",
            )

        benchmark_x_grad = benchmark_x.detach().requires_grad_(True)
        benchmark_weights_grad = (
            benchmark_weights.detach().requires_grad_(True)
        )

        def loop_forward_grad() -> torch.Tensor:
            return reference_module(
                benchmark_x_grad,
                indices,
                benchmark_weights_grad,
                method="loop",
            )

        def pad_forward_grad() -> torch.Tensor:
            return reference_module(
                benchmark_x_grad,
                indices,
                benchmark_weights_grad,
                method="pad",
            )

        def group_forward_grad() -> torch.Tensor:
            return reference_module(
                benchmark_x_grad,
                indices,
                benchmark_weights_grad,
                method="group",
            )

        results = {}
        cases = (
            ("loop", None, loop_forward, loop_forward_grad),
            ("pad", None, pad_forward, pad_forward_grad),
            (
                "group/cutlass",
                "cutlass",
                group_forward,
                group_forward_grad,
            ),
            (
                "group/triton",
                "triton",
                group_forward,
                group_forward_grad,
            ),
            (
                "group/cutile",
                "cutile",
                group_forward,
                group_forward_grad,
            ),
        )
        for name, backend, forward, forward_grad in cases:
            if backend is not None:
                reference_module.gmm_backend = backend
            forward_latency = benchmark_forward(forward)
            backward_latency = benchmark_backward(
                reference_module,
                forward_grad,
                benchmark_x_grad,
                benchmark_weights_grad,
            )
            results[name] = (
                forward_latency,
                backward_latency,
                forward_latency + backward_latency,
            )

        loop_total = results["loop"][2]
        num_tokens = batch_size * seq_len
        LOGGER.info(
            "%-14s | %12s | %12s | %12s | %12s | %9s",
            "method",
            "forward(us)",
            "backward(us)",
            "total(us)",
            "tokens/s",
            "speedup",
        )
        LOGGER.info("-" * 87)
        for name, _, _, _ in cases:
            forward_latency, backward_latency, total_latency = (
                results[name]
            )
            tokens_per_second = (
                num_tokens * 1_000_000.0 / total_latency
            )
            LOGGER.info(
                (
                    "%-14s | %12.3f | %12.3f | %12.3f | "
                    "%12.1f | %8.3fx"
                ),
                name,
                forward_latency,
                backward_latency,
                total_latency,
                tokens_per_second,
                loop_total / total_latency,
            )
    else:
        log_section("CUDA 不可用，跳过 group 路径测试")

    log_test_success("LoRAMoENonstandard")


if __name__ == "__main__":
    main()
