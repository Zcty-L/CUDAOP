"""LoRAMoENonstandard 的 BF16 精度和反向传播测试。"""

import copy

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
) -> LoRAMoENonstandard:
    module = LoRAMoENonstandard(
        original_mlp=TestMlp(256, 512).to(
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


def main() -> None:
    torch.manual_seed(11)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.bfloat16
    log_test_start(
        "LoRAMoENonstandard",
        f"device={device}，dtype={dtype}，experts=8，rank=16，"
        "hidden=256，intermediate=512",
    )
    reference_module = make_module(device, dtype)
    pad_module = copy.deepcopy(reference_module)

    x = torch.randn(
        2,
        3,
        256,
        device=device,
        dtype=dtype,
    )
    indices = torch.tensor(
        [
            [[0, 1], [2, 2], [3, 4]],
            [[5, 6], [7, 0], [1, 3]],
        ],
        device=device,
        dtype=torch.long,
    )
    weights = torch.rand(
        2,
        3,
        2,
        device=device,
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
        group_module = copy.deepcopy(reference_module)
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
            "group",
        )
    else:
        log_section("CUDA 不可用，跳过 group 路径测试")

    log_test_success("LoRAMoENonstandard")


if __name__ == "__main__":
    main()
