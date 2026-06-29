"""LoRAMoEStandard 的 Torch 精度与反向传播测试。"""

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


def main() -> None:
    torch.manual_seed(7)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.bfloat16
    hidden_size = 256
    intermediate_size = 512
    log_test_start(
        "LoRAMoEStandard",
        f"device={device}，dtype={dtype}，experts=8，rank=16，"
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
        2,
        3,
        hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    top_k_indices = torch.tensor(
        [
            [[0, 1], [2, 2], [3, 4]],
            [[5, 6], [7, 0], [1, 3]],
        ],
        device=device,
        dtype=torch.long,
    )
    top_k_weights = torch.rand(
        2,
        3,
        2,
        device=device,
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
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"参数梯度未生成: {name}")
    LOGGER.debug("[DEBUG] loop 前向及反向传播通过")

    log_section("运行 pad 前向精度测试")
    pad_indices = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [0, 2], [1, 3]],
        ],
        device=device,
        dtype=torch.long,
    )
    pad_weights = torch.rand(
        2,
        3,
        2,
        device=device,
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
        log_section("运行 group 前向精度及反向传播测试")
        module.zero_grad(set_to_none=True)
        group_x = x.detach().clone().requires_grad_(True)
        group_weights = (
            pad_weights.detach().clone().requires_grad_(True)
        )
        group_actual = module._forward_group(
            group_x,
            pad_indices,
            group_weights,
        )
        group_expected = slot_reference(
            module,
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
            "group",
            "输出",
            (
                group_actual.float() - group_expected.float()
            ).abs().max().item(),
        )

        group_actual.sum().backward()
        if group_x.grad is None:
            raise AssertionError("Group 模式输入梯度未生成")
        if group_weights.grad is None:
            raise AssertionError("Group 模式路由权重梯度未生成")
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                raise AssertionError(
                    f"Group 模式参数梯度未生成: {name}"
                )
        LOGGER.debug("[DEBUG] group 前向及反向传播通过")
    else:
        log_section("CUDA 不可用，跳过 group 路径测试")

    log_test_success("LoRAMoEStandard")


if __name__ == "__main__":
    main()
