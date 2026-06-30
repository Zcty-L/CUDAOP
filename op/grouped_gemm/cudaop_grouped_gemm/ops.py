"""支持自动求导的 Grouped GEMM 接口。"""

from typing import Any

import torch

from cudaop_grouped_gemm import _C


class _GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_b: bool,
    ) -> torch.Tensor:
        sizes_cpu = batch_sizes.detach().to(
            device="cpu",
            dtype=torch.int64,
        ).contiguous()
        context.save_for_backward(a, b, sizes_cpu)
        context.trans_b = trans_b
        return _C.grouped_gemm(
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
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        a, b, batch_sizes = context.saved_tensors
        grad = grad.contiguous()
        trans_b = context.trans_b
        grad_a = _C.grouped_gemm(
            grad,
            b,
            batch_sizes,
            False,
            not trans_b,
        )
        lhs, rhs = (grad, a) if trans_b else (a, grad)
        grad_b = _C.grouped_gemm(
            lhs,
            rhs,
            batch_sizes,
            True,
            False,
        )
        return grad_a, grad_b, None, None


def gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool = False,
) -> torch.Tensor:
    """按 ``batch_sizes`` 划分 A，并与各组权重执行矩阵乘。"""
    return _GroupedGemm.apply(a, b, batch_sizes, trans_b)
