"""标准 LoRA-MoE Group 模式使用的统一算子接口。"""

from typing import Any, Literal, Tuple

import torch

try:
    import cudaop_grouped_gemm
except ModuleNotFoundError as error:
    if error.name != "cudaop_grouped_gemm":
        raise
    raise ModuleNotFoundError(
        "缺少 cudaop_grouped_gemm，请执行："
        "python -m pip install -e ../grouped_gemm"
    ) from error

import lora_moe_ops
import triton_kernels


GmmBackend = Literal["cutlass", "triton", "cutile"]
GMM_BACKENDS = ("cutlass", "triton", "cutile")


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


def gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool = False,
    backend: GmmBackend = "cutlass",
) -> torch.Tensor:
    """按后端调用 ``cudaop_grouped_gemm`` 的统一 GMM 接口。"""
    return cudaop_grouped_gemm.gmm(
        a,
        b,
        batch_sizes,
        trans_b,
        backend,
    )


def lora_gmm(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
    backend: GmmBackend = "cutlass",
) -> torch.Tensor:
    """执行 LoRA down/up Grouped GEMM，并按后端选择实现。"""
    if backend not in GMM_BACKENDS:
        raise ValueError(
            f"不支持的 GMM 后端: {backend}，可选值为 {GMM_BACKENDS}"
        )
    if backend == "cutlass":
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
            trans_b=True,
        )

    if down_weight.shape[1] not in (16, 32):
        raise ValueError(
            f"{backend} LoRA GMM 仅支持 rank=16/32，"
            f"当前 rank={down_weight.shape[1]}"
        )
    up_weight_transposed = (
        up_weight.transpose(1, 2).contiguous()
    )
    return cudaop_grouped_gemm.lora_gmm(
        a,
        down_weight.contiguous(),
        up_weight_transposed,
        batch_sizes,
        backend,
    )


def triton_lora_down(weight: torch.Tensor):
    """创建采用 ``[E, 16, K]`` 权重的 Triton LoRA down 算子。"""
    return cudaop_grouped_gemm.LoraDownGrouped(weight)


def triton_lora_up(weight: torch.Tensor):
    """创建采用 ``[E, 16, N]`` 权重的 Triton LoRA up 算子。"""
    return cudaop_grouped_gemm.LoraUpGrouped(weight)


def triton_lora_fused(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
):
    """创建融合 down/up 并保存中间矩阵的 Triton 算子。"""
    return cudaop_grouped_gemm.LoraFusedDownUpGrouped(
        down_weight,
        up_weight,
    )


def triton_lora_autograd(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """调用支持三 kernel 融合反向的 Triton LoRA 算子。"""
    return cudaop_grouped_gemm.triton_fused_lora(
        a,
        down_weight,
        up_weight,
        batch_sizes,
    )


def cutile_lora_down(weight: torch.Tensor):
    """创建采用 ``[E, 16, K]`` 权重的 cuTile LoRA down 算子。"""
    return cudaop_grouped_gemm.CuTileLoraDownGrouped(weight)


def cutile_lora_up(weight: torch.Tensor):
    """创建采用 ``[E, 16, N]`` 权重的 cuTile LoRA up 算子。"""
    return cudaop_grouped_gemm.CuTileLoraUpGrouped(weight)


def cutile_lora_fused(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
):
    """创建融合 down/up 并保存中间矩阵的 cuTile 算子。"""
    return cudaop_grouped_gemm.CuTileLoraFusedDownUpGrouped(
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
    return cudaop_grouped_gemm.cutile_fused_lora(
        a,
        down_weight,
        up_weight,
        batch_sizes,
    )
