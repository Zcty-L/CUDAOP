"""LoRA-MoE 路由使用的 Triton gather/scatter kernel。

派生自 Megablocks：
https://github.com/databricks/megablocks/blob/main/megablocks/backend/kernels.py
原始代码采用 Apache License 2.0。
"""

from typing import Optional

import torch
import triton
import triton.language as tl


def _check_matrix(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} 必须是二维 Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} 必须位于 CUDA 设备")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} 必须连续")


def _check_vector(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"{name} 必须是一维 Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} 必须位于 CUDA 设备")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} 必须连续")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_X": 64}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=2),
        triton.Config({"BLOCK_X": 256}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=4),
        triton.Config({"BLOCK_X": 256}, num_warps=4),
    ],
    key=["NUM_COLUMNS"],
)
@triton.jit
def _copy(
    a,
    b,
    indices,
    bin_ids,
    weights,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    index_a = tl.load(indices + tl.program_id(0))
    bin_index = tl.load(bin_ids + tl.program_id(0))

    offset_in_bin = tl.program_id(0)
    if bin_index > 0:
        offset_in_bin -= tl.load(bins + bin_index - 1)

    index_b = offset_in_bin
    if bin_index > 0:
        index_b += tl.load(bins + bin_index - 1)

    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    scale = tl.load(weights + index_a) if SCALE else 1.0
    input_pointer = a if A_TO_B else b
    output_pointer = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        value = tl.load(input_pointer + offsets, mask=mask)
        value = value.to(tl.float32) * scale.to(tl.float32)
        tl.store(
            output_pointer + offsets,
            value.to(output_pointer.dtype.element_ty),
            mask=mask,
        )
        offsets += BLOCK_X


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_X": 64}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=2),
        triton.Config({"BLOCK_X": 256}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=4),
        triton.Config({"BLOCK_X": 256}, num_warps=4),
    ],
    key=["NUM_COLUMNS"],
)
@triton.jit
def _scatter_weight_grad(
    x,
    grad,
    weight_grad,
    indices,
    bin_ids,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    output_index = tl.load(indices + tl.program_id(0))
    bin_index = tl.load(bin_ids + tl.program_id(0))

    offset_in_bin = tl.program_id(0)
    if bin_index > 0:
        offset_in_bin -= tl.load(bins + bin_index - 1)

    x_index = offset_in_bin
    if bin_index > 0:
        x_index += tl.load(bins + bin_index - 1)

    weight_grad += output_index
    grad += tl.multiple_of(
        (output_index // TOP_K) * NUM_COLUMNS,
        NUM_COLUMNS,
    )
    x += tl.multiple_of(x_index * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    accumulator = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        value = tl.load(x + offsets, mask=mask).to(tl.float32)
        gradient = tl.load(grad + offsets, mask=mask).to(tl.float32)
        accumulator += value * gradient
        offsets += BLOCK_X

    output = tl.sum(accumulator).to(weight_grad.dtype.element_ty)
    tl.store(weight_grad, output)


def gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    _check_matrix(x, "x")
    _check_vector(indices, "indices")
    _check_vector(bin_ids, "bin_ids")
    _check_vector(bins, "bins")
    if indices.numel() != x.shape[0] * top_k:
        raise ValueError("indices 长度必须等于 token 数量乘 top_k")
    if bin_ids.numel() != indices.numel():
        raise ValueError("bin_ids 和 indices 长度必须相同")
    if weights is not None:
        _check_vector(weights, "weights")
        if weights.numel() != indices.numel():
            raise ValueError("weights 和 indices 长度必须相同")

    output = torch.empty(
        (indices.numel(), x.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    _copy[(indices.numel(),)](
        x,
        output,
        indices,
        bin_ids,
        weights,
        bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k,
        A_TO_B=True,
        SCALE=weights is not None,
    )
    return output


def scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    _check_matrix(x, "x")
    _check_vector(indices, "indices")
    _check_vector(bin_ids, "bin_ids")
    _check_vector(bins, "bins")
    if indices.numel() != x.shape[0]:
        raise ValueError("indices 长度必须等于 x 的行数")
    if bin_ids.numel() != indices.numel():
        raise ValueError("bin_ids 和 indices 长度必须相同")
    if indices.numel() % top_k != 0:
        raise ValueError("indices 长度必须能被 top_k 整除")
    if weights is not None:
        _check_vector(weights, "weights")
        if weights.numel() != indices.numel():
            raise ValueError("weights 和 indices 长度必须相同")

    num_tokens = indices.numel() // top_k
    output = torch.empty(
        (num_tokens, top_k, x.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    _copy[(indices.numel(),)](
        output,
        x,
        indices,
        bin_ids,
        weights,
        bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k,
        A_TO_B=False,
        SCALE=weights is not None,
    )
    if top_k == 1:
        return output.view(num_tokens, x.shape[1])
    return output.sum(dim=1)


def scatter_weight_grad(
    x: torch.Tensor,
    grad: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    _check_matrix(x, "x")
    _check_matrix(grad, "grad")
    _check_vector(indices, "indices")
    _check_vector(bin_ids, "bin_ids")
    _check_vector(bins, "bins")
    if indices.numel() != x.shape[0]:
        raise ValueError("indices 长度必须等于 x 的行数")
    if grad.shape[0] * top_k != indices.numel():
        raise ValueError("grad 行数乘 top_k 必须等于 indices 长度")

    output = torch.empty(
        indices.numel(),
        dtype=x.dtype,
        device=x.device,
    )
    _scatter_weight_grad[(indices.numel(),)](
        x,
        grad,
        output,
        indices,
        bin_ids,
        bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k,
    )
    return output
