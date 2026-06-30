"""标准 LoRA-MoE Group 模式使用的统一算子接口。"""

from typing import Any, Tuple

import torch

import lora_moe_ops

import triton_kernels


def sort(
    values: torch.Tensor,
    end_bit: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sorted_values = torch.empty_like(values)
    sorted_indices = torch.empty_like(values)
    lora_moe_ops.sort(
        values,
        end_bit,
        sorted_values,
        sorted_indices,
    )
    return sorted_values, sorted_indices


def histogram(
    values: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    return lora_moe_ops.histogram(values, num_bins)


def inclusive_cumsum(
    values: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    if values.ndim == 1:
        values_2d = values.view(1, -1)
        output = torch.empty_like(values_2d)
        lora_moe_ops.inclusive_cumsum(values_2d, 1, output)
        return output.squeeze(0)

    output = torch.empty_like(values)
    lora_moe_ops.inclusive_cumsum(values, dim, output)
    return output


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        context.save_for_backward(indices, bin_ids, bins)
        context.top_k = top_k
        return triton_kernels.gather(
            x,
            indices,
            bin_ids,
            None,
            bins,
            top_k,
        )

    @staticmethod
    def backward(
        context: Any,
        grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None, None, None]:
        indices, bin_ids, bins = context.saved_tensors
        x_grad = triton_kernels.scatter(
            grad.contiguous(),
            indices,
            bin_ids,
            None,
            bins,
            context.top_k,
        )
        return x_grad, None, None, None, None


def gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return Gather.apply(
        x,
        indices,
        bin_ids,
        bins,
        top_k,
    )


class Scatter(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        context.save_for_backward(
            x,
            indices,
            bin_ids,
            weights,
            bins,
        )
        context.top_k = top_k
        return triton_kernels.scatter(
            x,
            indices,
            bin_ids,
            weights,
            bins,
            top_k,
        )

    @staticmethod
    def backward(
        context: Any,
        grad: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        None,
        None,
        torch.Tensor,
        None,
        None,
    ]:
        x, indices, bin_ids, weights, bins = context.saved_tensors
        grad = grad.contiguous()
        x_grad = triton_kernels.gather(
            grad,
            indices,
            bin_ids,
            weights,
            bins,
            context.top_k,
        )
        weights_grad = triton_kernels.scatter_weight_grad(
            x,
            grad,
            indices,
            bin_ids,
            bins,
            context.top_k,
        )
        return (
            x_grad,
            None,
            None,
            weights_grad,
            None,
            None,
        )


def scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return Scatter.apply(
        x,
        indices,
        bin_ids,
        weights,
        bins,
        top_k,
    )


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        trans_b: bool,
    ) -> torch.Tensor:
        batch_sizes = batch_sizes.detach().cpu().contiguous()
        context.save_for_backward(a, b, batch_sizes)
        context.trans_b = trans_b
        return lora_moe_ops.grouped_gemm(
            a,
            b,
            batch_sizes,
            False,
            trans_b,
        )

    @staticmethod
    def backward(
        context: Any,
        grad: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        None,
        None,
    ]:
        a, b, batch_sizes = context.saved_tensors
        trans_b = context.trans_b
        grad = grad.contiguous()

        a_grad = lora_moe_ops.grouped_gemm(
            grad,
            b,
            batch_sizes,
            False,
            not trans_b,
        )
        lhs, rhs = (grad, a) if trans_b else (a, grad)
        b_grad = lora_moe_ops.grouped_gemm(
            lhs,
            rhs,
            batch_sizes,
            True,
            False,
        )
        return a_grad, b_grad, None, None


def gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool = False,
) -> torch.Tensor:
    return GroupedGemm.apply(
        a,
        b,
        batch_sizes,
        trans_b,
    )


def triton_lora_down(weight: torch.Tensor):
    """创建采用 ``[E, 16, K]`` 权重的 Triton LoRA down 算子。"""
    from cudaop_grouped_gemm import LoraDownGrouped

    return LoraDownGrouped(weight)


def triton_lora_up(weight: torch.Tensor):
    """创建采用 ``[E, 16, N]`` 权重的 Triton LoRA up 算子。"""
    from cudaop_grouped_gemm import LoraUpGrouped

    return LoraUpGrouped(weight)


def triton_lora_fused(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
):
    """创建融合 down/up 并保存中间矩阵的 Triton 算子。"""
    from cudaop_grouped_gemm import LoraFusedDownUpGrouped

    return LoraFusedDownUpGrouped(down_weight, up_weight)


def triton_lora_autograd(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """调用支持三 kernel 融合反向的 Triton LoRA 算子。"""
    from cudaop_grouped_gemm import triton_fused_lora

    return triton_fused_lora(
        a,
        down_weight,
        up_weight,
        batch_sizes,
    )


def cutile_lora_down(weight: torch.Tensor):
    """创建采用 ``[E, 16, K]`` 权重的 cuTile LoRA down 算子。"""
    from cudaop_grouped_gemm import CuTileLoraDownGrouped

    return CuTileLoraDownGrouped(weight)


def cutile_lora_up(weight: torch.Tensor):
    """创建采用 ``[E, 16, N]`` 权重的 cuTile LoRA up 算子。"""
    from cudaop_grouped_gemm import CuTileLoraUpGrouped

    return CuTileLoraUpGrouped(weight)


def cutile_lora_fused(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
):
    """创建融合 down/up 并保存中间矩阵的 cuTile 算子。"""
    from cudaop_grouped_gemm import CuTileLoraFusedDownUpGrouped

    return CuTileLoraFusedDownUpGrouped(
        down_weight,
        up_weight,
    )


def cutile_lora_autograd(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """调用支持三 kernel 融合反向的 cuTile LoRA 算子。"""
    from cudaop_grouped_gemm import cutile_fused_lora

    return cutile_fused_lora(
        a,
        down_weight,
        up_weight,
        batch_sizes,
    )
