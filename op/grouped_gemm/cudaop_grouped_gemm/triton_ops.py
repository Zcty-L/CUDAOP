"""针对 LoRA rank=16 分别优化的 Triton Grouped GEMM。"""

from typing import Any

import torch
import triton
import triton.language as tl


LORA_RANK = 16


@triton.jit
def _find_expert(tile_index, cumulative_tiles, num_experts):
    low = 0
    high = num_experts
    while low < high:
        middle = (low + high) // 2
        boundary = tl.load(cumulative_tiles + middle)
        if boundary <= tile_index:
            low = middle + 1
        else:
            high = middle
    return low


@triton.jit
def grouped_down_kernel(
    a,
    weight_transposed,
    output,
    contraction_size: tl.constexpr,
    cumulative_tiles,
    token_offsets,
    token_counts,
    stride_am,
    stride_ak,
    stride_we,
    stride_wk,
    stride_wr,
    stride_om,
    stride_or,
    num_experts,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    num_stages: tl.constexpr,
):
    """C[M, 16] = A[M, K] @ B[E, 16, K].T。"""
    rank: tl.constexpr = 16
    tile_index = tl.program_id(0)
    expert = _find_expert(
        tile_index,
        cumulative_tiles,
        num_experts,
    )
    previous_tiles = tl.load(
        cumulative_tiles + expert - 1,
        mask=expert > 0,
        other=0,
    )
    local_tile = tile_index - previous_tiles
    token_offset = tl.load(token_offsets + expert)
    token_count = tl.load(token_counts + expert)
    row_start = token_offset + local_tile * block_m

    rows = tl.arange(0, block_m)
    ranks = tl.arange(0, rank)
    reduction = tl.arange(0, block_k)
    row_mask = rows < token_offset + token_count - row_start
    accumulator = tl.zeros(
        (block_m, rank),
        dtype=tl.float32,
    )

    a_base = a + row_start * stride_am
    weight_base = weight_transposed + expert * stride_we
    for reduction_start in tl.range(
        0,
        contraction_size,
        block_k,
        num_stages=num_stages,
    ):
        reduction_offsets = reduction_start + reduction
        reduction_mask = reduction_offsets < contraction_size
        a_pointers = (
            a_base
            + rows[:, None] * stride_am
            + reduction_offsets[None, :] * stride_ak
        )
        weight_pointers = (
            weight_base
            + reduction_offsets[:, None] * stride_wk
            + ranks[None, :] * stride_wr
        )
        a_tile = tl.load(
            a_pointers,
            mask=row_mask[:, None] & reduction_mask[None, :],
            other=0.0,
        )
        weight_tile = tl.load(
            weight_pointers,
            mask=reduction_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(a_tile, weight_tile)

    output_base = output + row_start * stride_om
    output_pointers = (
        output_base
        + rows[:, None] * stride_om
        + ranks[None, :] * stride_or
    )
    tl.store(
        output_pointers,
        accumulator,
        mask=row_mask[:, None],
    )


@triton.jit
def grouped_up_kernel(
    hidden,
    weight,
    output,
    output_size: tl.constexpr,
    cumulative_tiles,
    token_offsets,
    token_counts,
    stride_hm,
    stride_hr,
    stride_we,
    stride_wr,
    stride_wn,
    stride_om,
    stride_on,
    num_experts,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """D[M, N] = C[M, 16] @ W[E, 16, N]。"""
    rank: tl.constexpr = 16
    tile_index = tl.program_id(0)
    output_tile = tl.program_id(1)
    expert = _find_expert(
        tile_index,
        cumulative_tiles,
        num_experts,
    )
    previous_tiles = tl.load(
        cumulative_tiles + expert - 1,
        mask=expert > 0,
        other=0,
    )
    local_tile = tile_index - previous_tiles
    token_offset = tl.load(token_offsets + expert)
    token_count = tl.load(token_counts + expert)
    row_start = token_offset + local_tile * block_m

    rows = tl.arange(0, block_m)
    ranks = tl.arange(0, rank)
    columns = output_tile * block_n + tl.arange(0, block_n)
    row_mask = rows < token_offset + token_count - row_start
    column_mask = columns < output_size

    hidden_base = hidden + row_start * stride_hm
    hidden_pointers = (
        hidden_base
        + rows[:, None] * stride_hm
        + ranks[None, :] * stride_hr
    )
    hidden_tile = tl.load(
        hidden_pointers,
        mask=row_mask[:, None],
        other=0.0,
    )

    weight_base = weight + expert * stride_we
    weight_pointers = (
        weight_base
        + ranks[:, None] * stride_wr
        + columns[None, :] * stride_wn
    )
    weight_tile = tl.load(
        weight_pointers,
        mask=column_mask[None, :],
        other=0.0,
    )
    accumulator = tl.dot(hidden_tile, weight_tile)

    output_base = output + row_start * stride_om
    output_pointers = (
        output_base
        + rows[:, None] * stride_om
        + columns[None, :] * stride_on
    )
    tl.store(
        output_pointers,
        accumulator,
        mask=row_mask[:, None] & column_mask[None, :],
    )


@triton.jit
def grouped_fused_downup_kernel(
    a,
    down_weight_transposed,
    up_weight,
    hidden,
    output,
    input_size: tl.constexpr,
    output_size: tl.constexpr,
    cumulative_tiles,
    token_offsets,
    token_counts,
    stride_am,
    stride_ak,
    stride_de,
    stride_dk,
    stride_dr,
    stride_ue,
    stride_ur,
    stride_un,
    stride_hm,
    stride_hr,
    stride_om,
    stride_on,
    num_experts,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
    num_stages: tl.constexpr,
    num_stages_up: tl.constexpr,
):
    """融合 LoRA down/up，并保存 rank=16 中间矩阵。"""
    rank: tl.constexpr = 16
    tile_index = tl.program_id(0)
    expert = _find_expert(
        tile_index,
        cumulative_tiles,
        num_experts,
    )
    previous_tiles = tl.load(
        cumulative_tiles + expert - 1,
        mask=expert > 0,
        other=0,
    )
    local_tile = tile_index - previous_tiles
    token_offset = tl.load(token_offsets + expert)
    token_count = tl.load(token_counts + expert)
    row_start = token_offset + local_tile * block_m

    rows = tl.arange(0, block_m)
    ranks = tl.arange(0, rank)
    reduction = tl.arange(0, block_k)
    row_mask = rows < token_offset + token_count - row_start

    a_base = a + row_start * stride_am
    down_base = (
        down_weight_transposed + expert * stride_de
    )
    down_accumulator = tl.zeros(
        (block_m, rank),
        dtype=tl.float32,
    )
    for reduction_start in tl.range(
        0,
        input_size,
        block_k,
        num_stages=num_stages,
    ):
        reduction_offsets = reduction_start + reduction
        reduction_mask = reduction_offsets < input_size
        a_pointers = (
            a_base
            + rows[:, None] * stride_am
            + reduction_offsets[None, :] * stride_ak
        )
        down_pointers = (
            down_base
            + reduction_offsets[:, None] * stride_dk
            + ranks[None, :] * stride_dr
        )
        a_tile = tl.load(
            a_pointers,
            mask=row_mask[:, None] & reduction_mask[None, :],
            other=0.0,
        )
        down_tile = tl.load(
            down_pointers,
            mask=reduction_mask[:, None],
            other=0.0,
        )
        down_accumulator += tl.dot(a_tile, down_tile)

    hidden_tile = down_accumulator.to(tl.bfloat16)
    hidden_base = hidden + row_start * stride_hm
    hidden_pointers = (
        hidden_base
        + rows[:, None] * stride_hm
        + ranks[None, :] * stride_hr
    )
    tl.store(
        hidden_pointers,
        hidden_tile,
        mask=row_mask[:, None],
    )

    up_base = up_weight + expert * stride_ue
    output_base = output + row_start * stride_om
    for output_start in tl.range(
        0,
        output_size,
        block_n,
        num_stages=num_stages_up,
    ):
        columns = output_start + tl.arange(0, block_n)
        column_mask = columns < output_size
        up_pointers = (
            up_base
            + ranks[:, None] * stride_ur
            + columns[None, :] * stride_un
        )
        up_tile = tl.load(
            up_pointers,
            mask=column_mask[None, :],
            other=0.0,
        )
        up_accumulator = tl.dot(hidden_tile, up_tile)
        output_pointers = (
            output_base
            + rows[:, None] * stride_om
            + columns[None, :] * stride_on
        )
        tl.store(
            output_pointers,
            up_accumulator,
            mask=row_mask[:, None] & column_mask[None, :],
        )


@triton.jit
def grouped_fused_agrad_kernel(
    grad_output,
    up_weight,
    down_weight,
    grad_hidden,
    grad_input,
    input_size: tl.constexpr,
    output_size: tl.constexpr,
    cumulative_tiles,
    token_offsets,
    token_counts,
    stride_gom,
    stride_gon,
    stride_ue,
    stride_ur,
    stride_un,
    stride_de,
    stride_dr,
    stride_dk,
    stride_ghm,
    stride_ghr,
    stride_gim,
    stride_gik,
    num_experts,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    num_stages: tl.constexpr,
):
    """融合计算 grad_hidden 和 grad_input。"""
    rank: tl.constexpr = 16
    tile_index = tl.program_id(0)
    expert = _find_expert(
        tile_index,
        cumulative_tiles,
        num_experts,
    )
    previous_tiles = tl.load(
        cumulative_tiles + expert - 1,
        mask=expert > 0,
        other=0,
    )
    local_tile = tile_index - previous_tiles
    token_offset = tl.load(token_offsets + expert)
    token_count = tl.load(token_counts + expert)
    row_start = token_offset + local_tile * block_m

    rows = tl.arange(0, block_m)
    ranks = tl.arange(0, rank)
    reduction = tl.arange(0, block_k)
    row_mask = rows < token_offset + token_count - row_start

    grad_output_base = grad_output + row_start * stride_gom
    up_base = up_weight + expert * stride_ue
    hidden_accumulator = tl.zeros(
        (block_m, rank),
        dtype=tl.float32,
    )
    for reduction_start in tl.range(
        0,
        output_size,
        block_k,
        num_stages=num_stages,
    ):
        reduction_offsets = reduction_start + reduction
        reduction_mask = reduction_offsets < output_size
        grad_output_pointers = (
            grad_output_base
            + rows[:, None] * stride_gom
            + reduction_offsets[None, :] * stride_gon
        )
        up_pointers = (
            up_base
            + reduction_offsets[:, None] * stride_un
            + ranks[None, :] * stride_ur
        )
        grad_output_tile = tl.load(
            grad_output_pointers,
            mask=row_mask[:, None] & reduction_mask[None, :],
            other=0.0,
        )
        up_tile = tl.load(
            up_pointers,
            mask=reduction_mask[:, None],
            other=0.0,
        )
        hidden_accumulator += tl.dot(
            grad_output_tile,
            up_tile,
        )

    grad_hidden_tile = hidden_accumulator.to(tl.bfloat16)
    grad_hidden_base = grad_hidden + row_start * stride_ghm
    grad_hidden_pointers = (
        grad_hidden_base
        + rows[:, None] * stride_ghm
        + ranks[None, :] * stride_ghr
    )
    tl.store(
        grad_hidden_pointers,
        grad_hidden_tile,
        mask=row_mask[:, None],
    )

    down_base = down_weight + expert * stride_de
    grad_input_base = grad_input + row_start * stride_gim
    for output_start in tl.range(0, input_size, block_k):
        columns = output_start + reduction
        column_mask = columns < input_size
        down_pointers = (
            down_base
            + ranks[:, None] * stride_dr
            + columns[None, :] * stride_dk
        )
        down_tile = tl.load(
            down_pointers,
            mask=column_mask[None, :],
            other=0.0,
        )
        input_accumulator = tl.dot(
            grad_hidden_tile,
            down_tile,
        )
        grad_input_pointers = (
            grad_input_base
            + rows[:, None] * stride_gim
            + columns[None, :] * stride_gik
        )
        tl.store(
            grad_input_pointers,
            input_accumulator,
            mask=row_mask[:, None] & column_mask[None, :],
        )


@triton.jit
def grouped_bgrad_kernel(
    lhs,
    rhs,
    output,
    hidden_size: tl.constexpr,
    token_offsets,
    token_counts,
    stride_lm,
    stride_lr,
    stride_rm,
    stride_rk,
    stride_oe,
    stride_or,
    stride_ok,
    block_tokens: tl.constexpr,
    block_n: tl.constexpr,
):
    """计算每个 expert 的 ``lhs.T @ rhs`` 权重梯度。"""
    rank: tl.constexpr = 16
    expert = tl.program_id(0)
    output_tile = tl.program_id(1)
    token_offset = tl.load(token_offsets + expert)
    token_count = tl.load(token_counts + expert)
    columns = output_tile * block_n + tl.arange(0, block_n)
    column_mask = columns < hidden_size
    ranks = tl.arange(0, rank)
    accumulator = tl.zeros(
        (rank, block_n),
        dtype=tl.float32,
    )

    for token_start in tl.range(
        0,
        token_count,
        block_tokens,
        num_stages=2,
    ):
        tokens = token_start + tl.arange(0, block_tokens)
        token_mask = tokens < token_count
        lhs_pointers = (
            lhs
            + (token_offset + tokens[None, :]) * stride_lm
            + ranks[:, None] * stride_lr
        )
        rhs_pointers = (
            rhs
            + (token_offset + tokens[:, None]) * stride_rm
            + columns[None, :] * stride_rk
        )
        lhs_tile = tl.load(
            lhs_pointers,
            mask=token_mask[None, :],
            other=0.0,
        )
        rhs_tile = tl.load(
            rhs_pointers,
            mask=token_mask[:, None] & column_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(lhs_tile, rhs_tile)

    output_pointers = (
        output
        + expert * stride_oe
        + ranks[:, None] * stride_or
        + columns[None, :] * stride_ok
    )
    tl.store(
        output_pointers,
        accumulator,
        mask=column_mask[None, :],
    )


def _check_common(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> None:
    if not tensor.is_cuda or not weight.is_cuda:
        raise ValueError("输入和权重必须是 CUDA Tensor")
    if tensor.device != weight.device:
        raise ValueError("输入和权重必须位于同一设备")
    if tensor.dtype != torch.bfloat16:
        raise TypeError("输入必须是 bfloat16")
    if weight.dtype != tensor.dtype:
        raise TypeError("输入和权重的数据类型必须相同")
    if tensor.ndim != 2 or weight.ndim != 3:
        raise ValueError("输入必须为二维，权重必须为三维")
    if not tensor.is_contiguous() or not weight.is_contiguous():
        raise ValueError("输入和权重必须连续")
    if batch_sizes.ndim != 1:
        raise ValueError("batch_sizes 必须为一维")
    if batch_sizes.numel() != weight.shape[0]:
        raise ValueError("batch_sizes 的长度必须等于 expert 数")


def _get_metadata(
    owner,
    batch_sizes: torch.Tensor,
    total_rows: int,
    block_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    version = batch_sizes._version
    if (
        owner._cached_batch_sizes is batch_sizes
        and owner._cached_batch_sizes_version == version
        and owner._cached_total_rows == total_rows
    ):
        return owner._cached_metadata

    sizes_cpu = batch_sizes.detach().to(
        device="cpu",
        dtype=torch.int64,
    ).contiguous()
    if torch.any(sizes_cpu < 0):
        raise ValueError("batch_sizes 不能包含负数")
    if int(sizes_cpu.sum()) != total_rows:
        raise ValueError("batch_sizes 之和必须等于输入行数")
    metadata = _build_metadata(sizes_cpu, block_m, device)
    owner._cached_batch_sizes = batch_sizes
    owner._cached_batch_sizes_version = version
    owner._cached_total_rows = total_rows
    owner._cached_metadata = metadata
    return metadata


def _clear_metadata(owner) -> None:
    owner._cached_batch_sizes = None
    owner._cached_batch_sizes_version = -1
    owner._cached_total_rows = -1
    owner._cached_metadata = None


def _build_metadata(
    sizes_cpu: torch.Tensor,
    block_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    cumulative_tiles = []
    token_offsets = []
    total_tiles = 0
    token_offset = 0
    for size in sizes_cpu.tolist():
        token_offsets.append(token_offset)
        token_offset += size
        total_tiles += triton.cdiv(size, block_m)
        cumulative_tiles.append(total_tiles)

    return (
        torch.tensor(
            cumulative_tiles,
            device=device,
            dtype=torch.int32,
        ),
        torch.tensor(
            token_offsets,
            device=device,
            dtype=torch.int64,
        ),
        sizes_cpu.to(device=device, dtype=torch.int32),
        total_tiles,
    )


class LoraDownGrouped:
    """LoRA down：将权重预打包为连续的 ``[E, K, 16]``。"""

    def __init__(
        self,
        weight: torch.Tensor,
        block_m: int = 128,
        block_k: int = 64,
        num_warps: int = 8,
        num_stages: int = 3,
    ):
        if weight.ndim != 3 or weight.shape[1] != LORA_RANK:
            raise ValueError("down 权重形状必须是 [E, 16, K]")
        if not weight.is_cuda or not weight.is_contiguous():
            raise ValueError("down 权重必须是连续的 CUDA Tensor")
        if weight.dtype != torch.bfloat16:
            raise TypeError("down 权重必须是 bfloat16")

        self.weight_transposed = weight.permute(0, 2, 1).contiguous()
        self.num_experts = weight.shape[0]
        self.contraction_size = weight.shape[2]
        self.block_m = block_m
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        """清除路由元数据；下一次调用会重新构建。"""
        _clear_metadata(self)

    def __call__(
        self,
        a: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        _check_common(
            a,
            self.weight_transposed,
            batch_sizes,
        )
        if a.shape[1] != self.contraction_size:
            raise ValueError("输入和 down 权重的收缩维度不匹配")

        output = torch.empty(
            (a.shape[0], LORA_RANK),
            device=a.device,
            dtype=a.dtype,
        )
        if a.shape[0] == 0:
            return output

        metadata = _get_metadata(
            self,
            batch_sizes,
            a.shape[0],
            self.block_m,
            a.device,
        )
        cumulative_tiles, token_offsets, token_counts, total_tiles = (
            metadata
        )
        grouped_down_kernel[(total_tiles,)](
            a,
            self.weight_transposed,
            output,
            self.contraction_size,
            cumulative_tiles,
            token_offsets,
            token_counts,
            a.stride(0),
            a.stride(1),
            self.weight_transposed.stride(0),
            self.weight_transposed.stride(1),
            self.weight_transposed.stride(2),
            output.stride(0),
            output.stride(1),
            self.num_experts,
            block_m=self.block_m,
            block_k=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return output


class LoraUpGrouped:
    """LoRA up：权重使用连续的 ``[E, 16, N]`` 布局。"""

    def __init__(
        self,
        weight: torch.Tensor,
        block_m: int = 16,
        block_n: int = 256,
        num_warps: int = 8,
    ):
        if weight.ndim != 3 or weight.shape[1] != LORA_RANK:
            raise ValueError("up 权重形状必须是 [E, 16, N]")
        if not weight.is_cuda or not weight.is_contiguous():
            raise ValueError("up 权重必须是连续的 CUDA Tensor")
        if weight.dtype != torch.bfloat16:
            raise TypeError("up 权重必须是 bfloat16")

        self.weight = weight
        self.num_experts = weight.shape[0]
        self.output_size = weight.shape[2]
        self.block_m = block_m
        self.block_n = block_n
        self.num_warps = num_warps
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        """清除路由元数据；下一次调用会重新构建。"""
        _clear_metadata(self)

    def __call__(
        self,
        hidden: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        _check_common(
            hidden,
            self.weight,
            batch_sizes,
        )
        if hidden.shape[1] != LORA_RANK:
            raise ValueError("up 输入的最后一维必须是 16")

        output = torch.empty(
            (hidden.shape[0], self.output_size),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        if hidden.shape[0] == 0:
            return output

        metadata = _get_metadata(
            self,
            batch_sizes,
            hidden.shape[0],
            self.block_m,
            hidden.device,
        )
        cumulative_tiles, token_offsets, token_counts, total_tiles = (
            metadata
        )
        grid = (
            total_tiles,
            triton.cdiv(self.output_size, self.block_n),
        )
        grouped_up_kernel[grid](
            hidden,
            self.weight,
            output,
            self.output_size,
            cumulative_tiles,
            token_offsets,
            token_counts,
            hidden.stride(0),
            hidden.stride(1),
            self.weight.stride(0),
            self.weight.stride(1),
            self.weight.stride(2),
            output.stride(0),
            output.stride(1),
            self.num_experts,
            block_m=self.block_m,
            block_n=self.block_n,
            num_warps=self.num_warps,
        )
        return output


class LoraFusedDownUpGrouped:
    """融合 LoRA down/up，并返回供反向使用的中间矩阵。"""

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        block_m: int = 64,
        block_k: int = 64,
        block_n: int = 256,
        num_warps: int = 4,
        num_stages: int = 3,
        num_stages_up: int = 1,
    ):
        if down_weight.shape != up_weight.shape:
            raise ValueError("down 和 up 权重形状必须相同")
        if (
            down_weight.ndim != 3
            or down_weight.shape[1] != LORA_RANK
        ):
            raise ValueError("权重形状必须是 [E, 16, K]")
        if (
            not down_weight.is_cuda
            or not up_weight.is_cuda
            or not down_weight.is_contiguous()
            or not up_weight.is_contiguous()
        ):
            raise ValueError("权重必须是连续的 CUDA Tensor")
        if (
            down_weight.dtype != torch.bfloat16
            or up_weight.dtype != down_weight.dtype
        ):
            raise TypeError("权重必须是 bfloat16")
        if down_weight.device != up_weight.device:
            raise ValueError("down 和 up 权重必须位于同一设备")

        self.down_weight_transposed = (
            down_weight.permute(0, 2, 1).contiguous()
        )
        self.up_weight = up_weight
        self.num_experts = down_weight.shape[0]
        self.hidden_size = down_weight.shape[2]
        self.block_m = block_m
        self.block_k = block_k
        self.block_n = block_n
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_stages_up = num_stages_up
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        """清除路由元数据；下一次调用会重新构建。"""
        _clear_metadata(self)

    def __call__(
        self,
        a: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_common(
            a,
            self.down_weight_transposed,
            batch_sizes,
        )
        if a.shape[1] != self.hidden_size:
            raise ValueError("输入和 down 权重的收缩维度不匹配")

        hidden = torch.empty(
            (a.shape[0], LORA_RANK),
            device=a.device,
            dtype=a.dtype,
        )
        output = torch.empty(
            (a.shape[0], self.hidden_size),
            device=a.device,
            dtype=a.dtype,
        )
        if a.shape[0] == 0:
            return hidden, output

        metadata = _get_metadata(
            self,
            batch_sizes,
            a.shape[0],
            self.block_m,
            a.device,
        )
        cumulative_tiles, token_offsets, token_counts, total_tiles = (
            metadata
        )
        grouped_fused_downup_kernel[(total_tiles,)](
            a,
            self.down_weight_transposed,
            self.up_weight,
            hidden,
            output,
            self.hidden_size,
            self.hidden_size,
            cumulative_tiles,
            token_offsets,
            token_counts,
            a.stride(0),
            a.stride(1),
            self.down_weight_transposed.stride(0),
            self.down_weight_transposed.stride(1),
            self.down_weight_transposed.stride(2),
            self.up_weight.stride(0),
            self.up_weight.stride(1),
            self.up_weight.stride(2),
            hidden.stride(0),
            hidden.stride(1),
            output.stride(0),
            output.stride(1),
            self.num_experts,
            block_m=self.block_m,
            block_k=self.block_k,
            block_n=self.block_n,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            num_stages_up=self.num_stages_up,
        )
        return hidden, output


class LoraFusedAgradGrouped:
    """融合计算 LoRA up/down 的输入梯度。"""

    def __init__(
        self,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        block_m: int = 64,
        block_k: int = 64,
        num_warps: int = 4,
        num_stages: int = 3,
    ):
        if up_weight.shape != down_weight.shape:
            raise ValueError("down 和 up 权重形状必须相同")
        if (
            up_weight.ndim != 3
            or up_weight.shape[1] != LORA_RANK
        ):
            raise ValueError("权重形状必须是 [E, 16, K]")
        if (
            not up_weight.is_cuda
            or not down_weight.is_cuda
            or not up_weight.is_contiguous()
            or not down_weight.is_contiguous()
        ):
            raise ValueError("权重必须是连续的 CUDA Tensor")
        if (
            up_weight.dtype != torch.bfloat16
            or down_weight.dtype != up_weight.dtype
        ):
            raise TypeError("权重必须是 bfloat16")

        self.up_weight = up_weight
        self.down_weight = down_weight
        self.num_experts = up_weight.shape[0]
        self.hidden_size = up_weight.shape[2]
        self.block_m = block_m
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        """清除路由元数据；下一次调用会重新构建。"""
        _clear_metadata(self)

    def __call__(
        self,
        grad_output: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_common(
            grad_output,
            self.up_weight,
            batch_sizes,
        )
        if grad_output.shape[1] != self.hidden_size:
            raise ValueError("输出梯度维度与权重不匹配")

        grad_hidden = torch.empty(
            (grad_output.shape[0], LORA_RANK),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input = torch.empty_like(grad_output)
        if grad_output.shape[0] == 0:
            return grad_hidden, grad_input

        metadata = _get_metadata(
            self,
            batch_sizes,
            grad_output.shape[0],
            self.block_m,
            grad_output.device,
        )
        cumulative_tiles, token_offsets, token_counts, total_tiles = (
            metadata
        )
        grouped_fused_agrad_kernel[(total_tiles,)](
            grad_output,
            self.up_weight,
            self.down_weight,
            grad_hidden,
            grad_input,
            self.hidden_size,
            self.hidden_size,
            cumulative_tiles,
            token_offsets,
            token_counts,
            grad_output.stride(0),
            grad_output.stride(1),
            self.up_weight.stride(0),
            self.up_weight.stride(1),
            self.up_weight.stride(2),
            self.down_weight.stride(0),
            self.down_weight.stride(1),
            self.down_weight.stride(2),
            grad_hidden.stride(0),
            grad_hidden.stride(1),
            grad_input.stride(0),
            grad_input.stride(1),
            self.num_experts,
            block_m=self.block_m,
            block_k=self.block_k,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        return grad_hidden, grad_input


class LoraBgradGrouped:
    """计算 rank=16 LoRA grouped GEMM 的权重梯度。"""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        block_tokens: int = 128,
        block_n: int = 256,
        num_warps: int = 4,
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.block_tokens = block_tokens
        self.block_n = block_n
        self.num_warps = num_warps
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        """清除路由元数据；下一次调用会重新构建。"""
        _clear_metadata(self)

    def __call__(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        if (
            lhs.ndim != 2
            or lhs.shape[1] != LORA_RANK
            or rhs.ndim != 2
            or rhs.shape[1] != self.hidden_size
        ):
            raise ValueError("lhs/rhs 形状必须是 [N,16] 和 [N,K]")
        if lhs.shape[0] != rhs.shape[0]:
            raise ValueError("lhs 和 rhs 行数必须相同")
        if (
            not lhs.is_cuda
            or not rhs.is_cuda
            or not lhs.is_contiguous()
            or not rhs.is_contiguous()
        ):
            raise ValueError("lhs 和 rhs 必须是连续的 CUDA Tensor")
        if lhs.dtype != torch.bfloat16 or rhs.dtype != lhs.dtype:
            raise TypeError("lhs 和 rhs 必须是 bfloat16")
        if batch_sizes.numel() != self.num_experts:
            raise ValueError("batch_sizes 的长度必须等于 expert 数")

        output = torch.empty(
            (self.num_experts, LORA_RANK, self.hidden_size),
            device=lhs.device,
            dtype=lhs.dtype,
        )
        metadata = _get_metadata(
            self,
            batch_sizes,
            lhs.shape[0],
            self.block_tokens,
            lhs.device,
        )
        _, token_offsets, token_counts, _ = metadata
        grid = (
            self.num_experts,
            triton.cdiv(self.hidden_size, self.block_n),
        )
        grouped_bgrad_kernel[grid](
            lhs,
            rhs,
            output,
            self.hidden_size,
            token_offsets,
            token_counts,
            lhs.stride(0),
            lhs.stride(1),
            rhs.stride(0),
            rhs.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_tokens=self.block_tokens,
            block_n=self.block_n,
            num_warps=self.num_warps,
        )
        return output


class _LoraFusedDownUp(torch.autograd.Function):
    """三 kernel 反向的融合 LoRA down/up Autograd 实现。"""

    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        _check_common(a, down_weight, batch_sizes)
        if down_weight.shape[1] != LORA_RANK:
            raise ValueError("权重形状必须是 [E, 16, K]")
        if a.shape[1] != down_weight.shape[2]:
            raise ValueError("输入和权重的隐藏维度不匹配")
        if (
            up_weight.ndim != 3
            or up_weight.shape[0] != down_weight.shape[0]
            or up_weight.shape[1] != LORA_RANK
        ):
            raise ValueError("up 权重形状必须是 [E, 16, N]")
        if (
            not up_weight.is_cuda
            or not up_weight.is_contiguous()
            or up_weight.dtype != a.dtype
            or up_weight.device != a.device
        ):
            raise ValueError("up 权重的设备、类型或布局不正确")

        block_m = 64
        block_k = 64
        block_n = 256
        num_warps = 4
        num_stages = 3
        num_stages_up = 1
        sizes_cpu = batch_sizes.detach().to(
            device="cpu",
            dtype=torch.int64,
        ).contiguous()
        if torch.any(sizes_cpu < 0):
            raise ValueError("batch_sizes 不能包含负数")
        if int(sizes_cpu.sum()) != a.shape[0]:
            raise ValueError("batch_sizes 之和必须等于输入行数")
        metadata = _build_metadata(
            sizes_cpu,
            block_m,
            a.device,
        )
        cumulative_tiles, token_offsets, token_counts, total_tiles = (
            metadata
        )
        down_weight_transposed = (
            down_weight.permute(0, 2, 1).contiguous()
        )
        hidden = torch.empty(
            (a.shape[0], LORA_RANK),
            device=a.device,
            dtype=a.dtype,
        )
        output = torch.empty(
            (a.shape[0], up_weight.shape[2]),
            device=a.device,
            dtype=a.dtype,
        )
        if total_tiles > 0:
            grouped_fused_downup_kernel[(total_tiles,)](
                a,
                down_weight_transposed,
                up_weight,
                hidden,
                output,
                a.shape[1],
                up_weight.shape[2],
                cumulative_tiles,
                token_offsets,
                token_counts,
                a.stride(0),
                a.stride(1),
                down_weight_transposed.stride(0),
                down_weight_transposed.stride(1),
                down_weight_transposed.stride(2),
                up_weight.stride(0),
                up_weight.stride(1),
                up_weight.stride(2),
                hidden.stride(0),
                hidden.stride(1),
                output.stride(0),
                output.stride(1),
                down_weight.shape[0],
                block_m=block_m,
                block_k=block_k,
                block_n=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
                num_stages_up=num_stages_up,
            )

        context.save_for_backward(
            a,
            down_weight,
            up_weight,
            hidden,
            cumulative_tiles,
            token_offsets,
            token_counts,
        )
        context.total_tiles = total_tiles
        context.block_m = block_m
        context.block_k = block_k
        context.num_warps = num_warps
        context.num_stages = num_stages
        return output

    @staticmethod
    def backward(
        context: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        (
            a,
            down_weight,
            up_weight,
            hidden,
            cumulative_tiles,
            token_offsets,
            token_counts,
        ) = context.saved_tensors
        grad_output = grad_output.contiguous()
        grad_hidden = torch.empty_like(hidden)
        grad_input = torch.empty_like(a)
        num_experts = down_weight.shape[0]
        input_size = a.shape[1]
        output_size = grad_output.shape[1]

        if context.total_tiles > 0:
            grouped_fused_agrad_kernel[(context.total_tiles,)](
                grad_output,
                up_weight,
                down_weight,
                grad_hidden,
                grad_input,
                input_size,
                output_size,
                cumulative_tiles,
                token_offsets,
                token_counts,
                grad_output.stride(0),
                grad_output.stride(1),
                up_weight.stride(0),
                up_weight.stride(1),
                up_weight.stride(2),
                down_weight.stride(0),
                down_weight.stride(1),
                down_weight.stride(2),
                grad_hidden.stride(0),
                grad_hidden.stride(1),
                grad_input.stride(0),
                grad_input.stride(1),
                num_experts,
                block_m=context.block_m,
                block_k=context.block_k,
                num_warps=context.num_warps,
                num_stages=context.num_stages,
            )

        grad_down_weight = torch.empty_like(down_weight)
        grad_up_weight = torch.empty_like(up_weight)
        down_grid = (
            num_experts,
            triton.cdiv(input_size, 256),
        )
        grouped_bgrad_kernel[down_grid](
            grad_hidden,
            a,
            grad_down_weight,
            input_size,
            token_offsets,
            token_counts,
            grad_hidden.stride(0),
            grad_hidden.stride(1),
            a.stride(0),
            a.stride(1),
            grad_down_weight.stride(0),
            grad_down_weight.stride(1),
            grad_down_weight.stride(2),
            block_tokens=128,
            block_n=256,
            num_warps=4,
        )
        up_grid = (
            num_experts,
            triton.cdiv(output_size, 256),
        )
        grouped_bgrad_kernel[up_grid](
            hidden,
            grad_output,
            grad_up_weight,
            output_size,
            token_offsets,
            token_counts,
            hidden.stride(0),
            hidden.stride(1),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_up_weight.stride(0),
            grad_up_weight.stride(1),
            grad_up_weight.stride(2),
            block_tokens=128,
            block_n=256,
            num_warps=4,
        )
        return (
            grad_input,
            grad_down_weight,
            grad_up_weight,
            None,
        )


def triton_fused_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """执行融合前向和三 kernel 反向的 LoRA Grouped GEMM。"""
    return _LoraFusedDownUp.apply(
        a,
        down_weight,
        up_weight,
        batch_sizes,
    )
