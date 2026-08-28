"""支持自动求导的 Grouped GEMM 接口。"""

from typing import Any

import torch
import torch.nn.functional as functional

from cudaop_grouped_gemm import _C


class _ContiguousGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return value

    @staticmethod
    def backward(
        context: Any,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        return grad.contiguous()


class _GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_b: bool,
        use_k16: bool,
    ) -> torch.Tensor:
        sizes_cpu = batch_sizes.detach().to(
            device="cpu",
            dtype=torch.int64,
        ).contiguous()
        context.save_for_backward(a, b, sizes_cpu)
        context.trans_b = trans_b
        context.use_k16 = use_k16
        implementation = (
            _C.grouped_gemm_k16 if use_k16 else _C.grouped_gemm
        )
        return implementation(
            a,
            b,
            sizes_cpu,
            False,
            trans_b,
        )

    @staticmethod
    def backward(
        context: Any,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        a, b, batch_sizes = context.saved_tensors
        grad = grad.contiguous()
        trans_b = context.trans_b
        implementation = (
            _C.grouped_gemm_k16
            if context.use_k16
            else _C.grouped_gemm
        )
        grad_a = implementation(
            grad,
            b,
            batch_sizes,
            False,
            not trans_b,
        )
        lhs, rhs = (grad, a) if trans_b else (a, grad)
        grad_b = implementation(
            lhs,
            rhs,
            batch_sizes,
            True,
            False,
        )
        return grad_a, grad_b, None, None, None


def gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool = False,
) -> torch.Tensor:
    """按 ``batch_sizes`` 划分 A，并与各组权重执行矩阵乘。"""
    return _GroupedGemm.apply(a, b, batch_sizes, trans_b, False)


def gmm_k16(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool = False,
) -> torch.Tensor:
    """使用 K16/K8 MMA 配置执行支持自动求导的 Grouped GEMM。"""
    return _GroupedGemm.apply(a, b, batch_sizes, trans_b, True)


def torch_gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool = False,
) -> torch.Tensor:
    """使用 PyTorch ``grouped_mm`` 执行与 :func:`gmm` 相同的计算。"""
    offsets = batch_sizes.to(
        device=a.device,
        dtype=torch.int32,
    ).cumsum(
        dim=0,
        dtype=torch.int32,
    )
    weight = b.transpose(1, 2) if trans_b else b
    output = functional.grouped_mm(
        a,
        weight,
        offs=offsets,
    )
    return _ContiguousGradient.apply(output)


def lora_gmm(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """用两个 CUTLASS Grouped GEMM 组成 LoRA down/up。"""
    hidden = gmm(
        a,
        down_weight,
        batch_sizes,
        trans_b=True,
    )
    return gmm(
        hidden,
        up_weight,
        batch_sizes,
        trans_b=False,
    )


def lora_gmm_k16(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """用两个 K16/K8 CUTLASS Grouped GEMM 组成 LoRA down/up。"""
    hidden = gmm_k16(
        a,
        down_weight,
        batch_sizes,
        trans_b=True,
    )
    return gmm_k16(
        hidden,
        up_weight,
        batch_sizes,
        trans_b=False,
    )
