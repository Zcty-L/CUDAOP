"""LoRAMoEStandard 的 Torch 精度与反向传播测试。"""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora_moe_standard import LoRAMoEStandard

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
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu


def slot_reference(
    module: LoRAMoEStandard,
    x: torch.Tensor,
    top_k_indices: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_size = x.shape
    x_flat = x.reshape(-1, hidden_size)
    indices_flat = top_k_indices.reshape(x_flat.shape[0], -1)
    weights_flat = top_k_weights.reshape(x_flat.shape[0], -1)
    output = torch.zeros_like(x_flat)

    for token in range(x_flat.shape[0]):
        for slot in range(indices_flat.shape[1]):
            expert = int(indices_flat[token, slot])
            token_input = x_flat[token:token + 1]
            gate_up_hidden = F.linear(
                token_input,
                module.gate_up_lora_A[expert],
            )
            gate_hidden, up_hidden = gate_up_hidden.chunk(2, dim=-1)
            gate_output = (
                module.original_mlp.gate_proj(token_input)
                + F.linear(
                    gate_hidden,
                    module.gate_lora_B[expert],
                )
                * module.scaling
            )
            up_output = (
                module.original_mlp.up_proj(token_input)
                + F.linear(
                    up_hidden,
                    module.up_lora_B[expert],
                )
                * module.scaling
            )
            intermediate = (
                module.original_mlp.act_fn(gate_output) * up_output
            )
            expert_output = (
                module.original_mlp.down_proj(intermediate)
                + F.linear(
                    F.linear(
                        intermediate,
                        module.down_lora_A[expert],
                    ),
                    module.down_lora_B[expert],
                )
                * module.scaling
            )
            output[token] += expert_output.squeeze(0) * weights_flat[
                token,
                slot,
            ]

    return output.reshape(batch_size, sequence_length, hidden_size)


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
    module: LoRAMoEStandard,
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


def assert_parameter_gradients(
    module: LoRAMoEStandard,
    method: str,
) -> None:
    for name, parameter in module.named_parameters():
        if name.startswith("original_mlp."):
            if parameter.requires_grad:
                raise AssertionError(
                    f"{method} 基础 MLP 参数未冻结: {name}"
                )
            if parameter.grad is not None:
                raise AssertionError(
                    f"{method} 基础 MLP 参数生成了梯度: {name}"
                )
            continue

        if parameter.grad is None:
            raise AssertionError(
                f"{method} LoRA 参数梯度未生成: {name}"
            )


def main() -> None:
    torch.manual_seed(7)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.bfloat16
    batch_size = 2
    seq_len = 1507
    top_k = 2
    hidden_size = 2048
    intermediate_size = 2048
    log_test_start(
        "LoRAMoEStandard",
        f"device={device}，dtype={dtype}，experts=8，rank=16，"
        f"batch={batch_size}，seq_len={seq_len}，top_k={top_k}，"
        f"hidden={hidden_size}，intermediate={intermediate_size}",
    )
    mlp = TestMlp(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(device=device, dtype=dtype)
    module = LoRAMoEStandard(
        original_mlp=mlp,
        num_experts=8,
        rank=16,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )

    with torch.no_grad():
        module.gate_lora_B.normal_()
        module.up_lora_B.normal_()
        module.down_lora_B.normal_()

    x = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    router_logits = torch.randn(
        batch_size,
        seq_len,
        module.num_experts,
        device=device,
        dtype=dtype,
    )
    top_k_logits, top_k_indices = torch.topk(
        router_logits,
        top_k,
        dim=-1,
    )
    top_k_weights = torch.softmax(
        top_k_logits.float(),
        dim=-1,
    ).to(
        dtype=dtype,
    )

    log_section("运行 loop 前向精度及反向传播测试")
    actual = module._forward_loop(
        x,
        top_k_indices,
        top_k_weights,
    )
    expected = slot_reference(
        module,
        x,
        top_k_indices,
        top_k_weights,
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-2,
        atol=2e-2,
    )
    log_error(
        "loop",
        "输出",
        (actual.float() - expected.float()).abs().max().item(),
    )

    actual.sum().backward()
    if x.grad is None:
        raise AssertionError("输入梯度未生成")
    assert_parameter_gradients(module, "loop")
    LOGGER.debug("[DEBUG] loop 前向及反向传播通过")

    log_section("运行 pad 前向精度测试")
    pad_router_logits = torch.randn(
        batch_size,
        seq_len,
        module.num_experts,
        device=device,
        dtype=dtype,
    )
    pad_logits, pad_indices = torch.topk(
        pad_router_logits,
        top_k,
        dim=-1,
    )
    pad_weights = torch.softmax(
        pad_logits.float(),
        dim=-1,
    ).to(
        dtype=dtype,
    )
    pad_actual = module._forward_pad(
        x.detach(),
        pad_indices,
        pad_weights,
    )
    pad_expected = slot_reference(
        module,
        x.detach(),
        pad_indices,
        pad_weights,
    )
    torch.testing.assert_close(
        pad_actual,
        pad_expected,
        rtol=2e-2,
        atol=2e-2,
    )
    log_error(
        "pad",
        "输出",
        (
            pad_actual.float() - pad_expected.float()
        ).abs().max().item(),
    )
    LOGGER.debug("[DEBUG] pad 前向精度通过")

    if torch.cuda.is_available():
        for backend in ("cutlass", "triton", "cutile"):
            log_section(
                f"运行 group/{backend} 前向精度及反向传播测试"
            )
            group_module = copy.deepcopy(module)
            group_module.gmm_backend = backend
            group_module.zero_grad(set_to_none=True)
            group_x = x.detach().clone().requires_grad_(True)
            group_weights = (
                pad_weights.detach().clone().requires_grad_(True)
            )
            group_actual = group_module._forward_group(
                group_x,
                pad_indices,
                group_weights,
            )
            group_expected = slot_reference(
                group_module,
                group_x,
                pad_indices,
                group_weights,
            )
            torch.testing.assert_close(
                group_actual,
                group_expected,
                rtol=2e-2,
                atol=2e-2,
            )
            log_error(
                f"group/{backend}",
                "输出",
                (
                    group_actual.float() - group_expected.float()
                ).abs().max().item(),
            )

            group_actual.sum().backward()
            if group_x.grad is None:
                raise AssertionError(
                    f"group/{backend} 未生成输入梯度"
                )
            if group_weights.grad is None:
                raise AssertionError(
                    f"group/{backend} 未生成路由权重梯度"
                )
            assert_parameter_gradients(
                group_module,
                f"group/{backend}",
            )
            LOGGER.debug(
                "[DEBUG] group/%s 前向及反向传播通过",
                backend,
            )

        log_section(
            "运行 loop/pad/group 端到端前向与反向性能对比，"
            f"forward={WARMUP_ITERATIONS}+{BENCHMARK_ITERATIONS}，"
            "backward="
            f"{BACKWARD_WARMUP_ITERATIONS}+"
            f"{BACKWARD_BENCHMARK_ITERATIONS}"
        )
        benchmark_x = x.detach()
        benchmark_weights = pad_weights.detach()
        module.eval()

        def loop_forward() -> torch.Tensor:
            return module._forward_loop(
                benchmark_x,
                pad_indices,
                benchmark_weights,
            )

        def pad_forward() -> torch.Tensor:
            return module._forward_pad(
                benchmark_x,
                pad_indices,
                benchmark_weights,
            )

        def group_forward() -> torch.Tensor:
            return module._forward_group(
                benchmark_x,
                pad_indices,
                benchmark_weights,
            )

        benchmark_x_grad = benchmark_x.detach().requires_grad_(True)
        benchmark_weights_grad = (
            benchmark_weights.detach().requires_grad_(True)
        )

        def loop_forward_grad() -> torch.Tensor:
            return module._forward_loop(
                benchmark_x_grad,
                pad_indices,
                benchmark_weights_grad,
            )

        def pad_forward_grad() -> torch.Tensor:
            return module._forward_pad(
                benchmark_x_grad,
                pad_indices,
                benchmark_weights_grad,
            )

        def group_forward_grad() -> torch.Tensor:
            return module._forward_group(
                benchmark_x_grad,
                pad_indices,
                benchmark_weights_grad,
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
                module.gmm_backend = backend
            forward_latency = benchmark_forward(forward)
            backward_latency = benchmark_backward(
                module,
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

    log_test_success("LoRAMoEStandard")


if __name__ == "__main__":
    main()
