"""基础 PyTorch 注意力实现测试。"""

import logging

import torch

from flashattn import GQA, LinearAttention, MHA, MLA, MQA


LOGGER = logging.getLogger("flashattn_torch_test")


def _test_module(
    name: str,
    module: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | None,
) -> None:
    output = module(x) if mask is None else module(x, mask)

    expected_shape = x.shape
    if output.shape != expected_shape:
        raise AssertionError(
            f"{name} 输出形状错误：{output.shape} != {expected_shape}"
        )
    if not torch.isfinite(output).all():
        raise AssertionError(f"{name} 输出包含非有限值")

    output.square().mean().backward()
    for parameter in module.parameters():
        if parameter.grad is None:
            raise AssertionError(f"{name} 参数缺少梯度")
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"{name} 参数梯度包含非有限值")

    LOGGER.info(
        "%-6s | output=%-14s | gradient=%s",
        name,
        str(tuple(output.shape)),
        "finite",
    )


def _test_linear_attention_formula(
    module: LinearAttention,
    x: torch.Tensor,
) -> None:
    batch, sequence, _ = x.shape

    q = module.q_proj(x)
    k = module.k_proj(x)
    v = module.v_proj(x)

    q = q.view(
        batch,
        sequence,
        module.num_heads,
        module.head_dim,
    ).transpose(1, 2)
    k = k.view(
        batch,
        sequence,
        module.num_heads,
        module.head_dim,
    ).transpose(1, 2)
    v = v.view(
        batch,
        sequence,
        module.num_heads,
        module.head_dim,
    ).transpose(1, 2)

    q = torch.nn.functional.elu(q) + 1.0
    k = torch.nn.functional.elu(k) + 1.0

    # 显式构造 [S, S] 权重，只用于验证线性形式的等价性。
    attention = q @ k.transpose(-2, -1)
    attention = attention / attention.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(module.eps)
    expected = attention @ v
    expected = expected.transpose(1, 2).contiguous()
    expected = expected.view(batch, sequence, -1)
    expected = module.out_proj(expected)

    actual = module(x)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )
    LOGGER.info(
        "%-6s | 与显式二次形式等价",
        "LINEAR",
    )


def _test_causal_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    module = MHA(
        d_model=x.shape[-1],
        num_heads=8,
    )
    automatic = module(
        x,
        is_causal=True,
    )
    explicit = module(
        x,
        mask=mask,
    )
    torch.testing.assert_close(
        automatic,
        explicit,
        rtol=0.0,
        atol=0.0,
    )
    LOGGER.info(
        "%-6s | 与显式下三角 mask 等价",
        "CAUSAL",
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    torch.manual_seed(20260728)

    batch = 2
    sequence = 8
    d_model = 64
    x = torch.randn(
        batch,
        sequence,
        d_model,
        requires_grad=True,
    )

    # 下三角 mask，用于模拟自回归注意力。
    mask = torch.ones(
        sequence,
        sequence,
        dtype=torch.bool,
    ).tril()

    modules = {
        "MHA": MHA(d_model=d_model, num_heads=8),
        "GQA": GQA(
            d_model=d_model,
            num_query_heads=8,
            num_kv_heads=2,
        ),
        "MQA": MQA(d_model=d_model, num_query_heads=8),
        "MLA": MLA(
            d_model=d_model,
            num_heads=8,
            kv_lora_rank=16,
        ),
        "LINEAR": LinearAttention(
            d_model=d_model,
            num_heads=8,
        ),
    }

    LOGGER.info("配置")
    LOGGER.info(
        "batch=%d, sequence=%d, d_model=%d, dtype=%s",
        batch,
        sequence,
        d_model,
        x.dtype,
    )

    LOGGER.info("")
    LOGGER.info("阶段：前向与反向验证")
    for name, module in modules.items():
        module.zero_grad(set_to_none=True)
        module_mask = None if name == "LINEAR" else mask
        _test_module(name, module, x.detach().clone(), module_mask)

    LOGGER.info("")
    LOGGER.info("阶段：Linear Attention 公式验证")
    _test_linear_attention_formula(
        modules["LINEAR"],
        x.detach().clone(),
    )

    LOGGER.info("")
    LOGGER.info("阶段：Causal mask 验证")
    _test_causal_mask(
        x.detach().clone(),
        mask,
    )

    LOGGER.info("")
    LOGGER.info("[SUCCESS] 所有基础注意力实现验证通过")


if __name__ == "__main__":
    main()
