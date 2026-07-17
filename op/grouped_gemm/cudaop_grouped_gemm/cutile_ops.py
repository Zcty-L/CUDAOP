"""针对 LoRA rank=16/32 优化的 cuTile Grouped GEMM。"""

from typing import Any

import cuda.tile as ct
import torch


LORA_RANKS = (16, 32)


def _check_lora_rank(rank: int) -> int:
    if rank not in LORA_RANKS:
        raise ValueError("LoRA rank 必须是 16 或 32")
    return rank


@ct.kernel
def grouped_down_kernel(
    a,
    weight_transposed,
    output,
    token_offsets,
    token_counts,
    rank: ct.Constant[int],
    block_m: ct.Constant[int],
    block_k: ct.Constant[int],
):
    """C[M, R] = A[M, K] @ B[E, K, R]。"""
    row_tile = ct.bid(0)
    expert = ct.bid(1)
    token_offset = ct.gather(token_offsets, expert)
    token_count = ct.gather(token_counts, expert)
    rows = (
        token_offset
        + row_tile * block_m
        + ct.arange(block_m, dtype=ct.int32)
    )
    ranks = ct.arange(rank, dtype=ct.int32)
    row_indices = rows.reshape((block_m, 1))
    rank_indices = ranks.reshape((1, rank))
    row_mask = row_indices < token_offset + token_count
    accumulator = ct.zeros((block_m, rank), ct.float32)

    for k_tile in range(ct.cdiv(a.shape[1], block_k)):
        columns = (
            k_tile * block_k
            + ct.arange(block_k, dtype=ct.int32)
        )
        column_indices = columns.reshape((1, block_k))
        a_tile = ct.gather(
            a,
            (row_indices, column_indices),
            mask=row_mask,
            padding_value=0,
        )
        weight_tile = ct.load(
            weight_transposed,
            (expert, k_tile, 0),
            shape=(1, block_k, rank),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((block_k, rank))
        accumulator = ct.mma(a_tile, weight_tile, accumulator)

    ct.scatter(
        output,
        (row_indices, rank_indices),
        accumulator.astype(output.dtype),
        mask=row_mask,
    )


@ct.kernel
def grouped_up_kernel(
    hidden,
    weight,
    output,
    token_offsets,
    token_counts,
    rank: ct.Constant[int],
    block_m: ct.Constant[int],
    block_n: ct.Constant[int],
):
    """D[M, N] = C[M, R] @ W[E, R, N]。"""
    row_tile = ct.bid(0)
    output_tile = ct.bid(1)
    expert = ct.bid(2)
    token_offset = ct.gather(token_offsets, expert)
    token_count = ct.gather(token_counts, expert)
    rows = (
        token_offset
        + row_tile * block_m
        + ct.arange(block_m, dtype=ct.int32)
    )
    columns = (
        output_tile * block_n
        + ct.arange(block_n, dtype=ct.int32)
    )
    ranks = ct.arange(rank, dtype=ct.int32)
    row_indices = rows.reshape((block_m, 1))
    column_indices = columns.reshape((1, block_n))
    rank_indices = ranks.reshape((1, rank))
    row_mask = row_indices < token_offset + token_count

    hidden_tile = ct.gather(
        hidden,
        (row_indices, rank_indices),
        mask=row_mask,
        padding_value=0,
    )
    weight_tile = ct.load(
        weight,
        (expert, 0, output_tile),
        shape=(1, rank, block_n),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((rank, block_n))
    accumulator = ct.matmul(hidden_tile, weight_tile)
    ct.scatter(
        output,
        (row_indices, column_indices),
        accumulator.astype(output.dtype),
        mask=row_mask,
    )


@ct.kernel
def grouped_fused_downup_kernel(
    a,
    down_weight_transposed,
    up_weight,
    hidden,
    output,
    token_offsets,
    token_counts,
    rank: ct.Constant[int],
    block_m: ct.Constant[int],
    block_k: ct.Constant[int],
    block_n: ct.Constant[int],
):
    """融合 LoRA down/up，并保存 rank=16/32 中间矩阵。"""
    row_tile = ct.bid(0)
    expert = ct.bid(1)
    token_offset = ct.gather(token_offsets, expert)
    token_count = ct.gather(token_counts, expert)
    rows = (
        token_offset
        + row_tile * block_m
        + ct.arange(block_m, dtype=ct.int32)
    )
    ranks = ct.arange(rank, dtype=ct.int32)
    row_indices = rows.reshape((block_m, 1))
    rank_indices = ranks.reshape((1, rank))
    row_mask = row_indices < token_offset + token_count
    down_accumulator = ct.zeros(
        (block_m, rank),
        ct.float32,
    )

    for k_tile in range(ct.cdiv(a.shape[1], block_k)):
        columns = (
            k_tile * block_k
            + ct.arange(block_k, dtype=ct.int32)
        )
        column_indices = columns.reshape((1, block_k))
        a_tile = ct.gather(
            a,
            (row_indices, column_indices),
            mask=row_mask,
            padding_value=0,
        )
        down_tile = ct.load(
            down_weight_transposed,
            (expert, k_tile, 0),
            shape=(1, block_k, rank),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((block_k, rank))
        down_accumulator = ct.mma(
            a_tile,
            down_tile,
            down_accumulator,
        )

    hidden_tile = down_accumulator.astype(hidden.dtype)
    ct.scatter(
        hidden,
        (row_indices, rank_indices),
        hidden_tile,
        mask=row_mask,
    )

    for output_tile in range(ct.cdiv(output.shape[1], block_n)):
        columns = (
            output_tile * block_n
            + ct.arange(block_n, dtype=ct.int32)
        )
        column_indices = columns.reshape((1, block_n))
        up_tile = ct.load(
            up_weight,
            (expert, 0, output_tile),
            shape=(1, rank, block_n),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((rank, block_n))
        output_tile_value = ct.matmul(hidden_tile, up_tile)
        ct.scatter(
            output,
            (row_indices, column_indices),
            output_tile_value.astype(output.dtype),
            mask=row_mask,
        )


@ct.kernel
def grouped_fused_agrad_kernel(
    grad_output,
    up_weight,
    down_weight,
    grad_hidden,
    grad_input,
    token_offsets,
    token_counts,
    rank: ct.Constant[int],
    block_m: ct.Constant[int],
    block_k: ct.Constant[int],
):
    """融合计算 grad_hidden 和 grad_input。"""
    row_tile = ct.bid(0)
    expert = ct.bid(1)
    token_offset = ct.gather(token_offsets, expert)
    token_count = ct.gather(token_counts, expert)
    rows = (
        token_offset
        + row_tile * block_m
        + ct.arange(block_m, dtype=ct.int32)
    )
    ranks = ct.arange(rank, dtype=ct.int32)
    row_indices = rows.reshape((block_m, 1))
    rank_indices = ranks.reshape((1, rank))
    row_mask = row_indices < token_offset + token_count
    hidden_accumulator = ct.zeros(
        (block_m, rank),
        ct.float32,
    )

    for k_tile in range(ct.cdiv(grad_output.shape[1], block_k)):
        columns = (
            k_tile * block_k
            + ct.arange(block_k, dtype=ct.int32)
        )
        column_indices = columns.reshape((1, block_k))
        grad_output_tile = ct.gather(
            grad_output,
            (row_indices, column_indices),
            mask=row_mask,
            padding_value=0,
        )
        up_tile = ct.load(
            up_weight,
            (expert, 0, k_tile),
            shape=(1, rank, block_k),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((rank, block_k)).transpose()
        hidden_accumulator = ct.mma(
            grad_output_tile,
            up_tile,
            hidden_accumulator,
        )

    grad_hidden_tile = hidden_accumulator.astype(grad_hidden.dtype)
    ct.scatter(
        grad_hidden,
        (row_indices, rank_indices),
        grad_hidden_tile,
        mask=row_mask,
    )

    for output_tile in range(ct.cdiv(grad_input.shape[1], block_k)):
        columns = (
            output_tile * block_k
            + ct.arange(block_k, dtype=ct.int32)
        )
        column_indices = columns.reshape((1, block_k))
        down_tile = ct.load(
            down_weight,
            (expert, 0, output_tile),
            shape=(1, rank, block_k),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((rank, block_k))
        input_tile = ct.matmul(grad_hidden_tile, down_tile)
        ct.scatter(
            grad_input,
            (row_indices, column_indices),
            input_tile.astype(grad_input.dtype),
            mask=row_mask,
        )


@ct.kernel
def grouped_bgrad_kernel(
    lhs,
    rhs,
    output,
    token_offsets,
    token_counts,
    rank: ct.Constant[int],
    block_tokens: ct.Constant[int],
    block_n: ct.Constant[int],
):
    """计算每个 expert 的 ``lhs.T @ rhs`` 权重梯度。"""
    expert = ct.bid(0)
    output_tile = ct.bid(1)
    token_offset = ct.gather(token_offsets, expert)
    token_count = ct.gather(token_counts, expert)
    ranks = ct.arange(rank, dtype=ct.int32)
    columns = (
        output_tile * block_n
        + ct.arange(block_n, dtype=ct.int32)
    )
    rank_indices = ranks.reshape((rank, 1))
    column_indices = columns.reshape((1, block_n))
    accumulator = ct.zeros((rank, block_n), ct.float32)

    for token_tile in range(ct.cdiv(token_count, block_tokens)):
        tokens = (
            token_offset
            + token_tile * block_tokens
            + ct.arange(block_tokens, dtype=ct.int32)
        )
        token_indices = tokens.reshape((1, block_tokens))
        token_mask = token_indices < token_offset + token_count
        lhs_tile = ct.gather(
            lhs,
            (token_indices, rank_indices),
            mask=token_mask,
            padding_value=0,
        )
        rhs_tile = ct.gather(
            rhs,
            (token_indices.transpose(), column_indices),
            mask=token_mask.transpose(),
            padding_value=0,
        )
        accumulator = ct.mma(lhs_tile, rhs_tile, accumulator)

    ct.store(
        output,
        (expert, 0, output_tile),
        accumulator.astype(output.dtype).reshape(
            (1, rank, block_n)
        ),
    )


def _check_common(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> None:
    if tensor.ndim != 2 or weight.ndim != 3:
        raise ValueError("输入必须是二维 Tensor，权重必须是三维 Tensor")
    if not tensor.is_cuda or not weight.is_cuda:
        raise ValueError("输入和权重必须位于 CUDA")
    if not tensor.is_contiguous() or not weight.is_contiguous():
        raise ValueError("输入和权重必须连续")
    if tensor.dtype != torch.bfloat16 or weight.dtype != tensor.dtype:
        raise TypeError("输入和权重必须是 bfloat16")
    if tensor.device != weight.device:
        raise ValueError("输入和权重必须位于同一设备")
    if batch_sizes.ndim != 1:
        raise ValueError("batch_sizes 必须是一维 Tensor")
    if batch_sizes.numel() != weight.shape[0]:
        raise ValueError("batch_sizes 的长度必须等于 expert 数")


def _clear_metadata(owner) -> None:
    owner._cached_batch_sizes = None
    owner._cached_batch_sizes_version = -1
    owner._cached_total_rows = -1
    owner._cached_metadata = None


def _build_metadata(
    batch_sizes: torch.Tensor,
    total_rows: int,
    block_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    sizes_cpu = batch_sizes.detach().to(
        device="cpu",
        dtype=torch.int64,
    ).contiguous()
    if torch.any(sizes_cpu < 0):
        raise ValueError("batch_sizes 不能包含负数")
    if int(sizes_cpu.sum()) != total_rows:
        raise ValueError("batch_sizes 之和必须等于输入行数")

    offsets = torch.zeros_like(sizes_cpu)
    if sizes_cpu.numel() > 1:
        offsets[1:] = torch.cumsum(sizes_cpu[:-1], dim=0)
    max_tiles = 0
    if sizes_cpu.numel() > 0:
        max_tiles = int(
            torch.div(
                sizes_cpu + block_m - 1,
                block_m,
                rounding_mode="floor",
            ).max()
        )
    return (
        offsets.to(device=device, dtype=torch.int32),
        sizes_cpu.to(device=device, dtype=torch.int32),
        max_tiles,
    )


def _get_metadata(
    owner,
    batch_sizes: torch.Tensor,
    total_rows: int,
    block_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    version = getattr(batch_sizes, "_version", 0)
    if (
        owner._cached_batch_sizes is batch_sizes
        and owner._cached_batch_sizes_version == version
        and owner._cached_total_rows == total_rows
    ):
        return owner._cached_metadata

    metadata = _build_metadata(
        batch_sizes,
        total_rows,
        block_m,
        device,
    )
    owner._cached_batch_sizes = batch_sizes
    owner._cached_batch_sizes_version = version
    owner._cached_total_rows = total_rows
    owner._cached_metadata = metadata
    return metadata


def _launch(
    grid: tuple[int, ...],
    kernel,
    arguments: tuple[Any, ...],
) -> None:
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        arguments,
    )


class CuTileLoraDownGrouped:
    """cuTile LoRA down，SM120 默认使用 128 行 tile。"""

    def __init__(
        self,
        weight: torch.Tensor,
        block_m: int = 128,
        block_k: int = 64,
    ):
        if weight.ndim != 3:
            raise ValueError("down 权重形状必须是 [E, R, K]")
        self.rank = _check_lora_rank(weight.shape[1])
        if not weight.is_cuda or not weight.is_contiguous():
            raise ValueError("down 权重必须是连续的 CUDA Tensor")
        if weight.dtype != torch.bfloat16:
            raise TypeError("down 权重必须是 bfloat16")
        self.weight_transposed = (
            weight.permute(0, 2, 1).contiguous()
        )
        self.num_experts = weight.shape[0]
        self.contraction_size = weight.shape[2]
        self.block_m = block_m
        self.block_k = block_k
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        _clear_metadata(self)

    def __call__(
        self,
        a: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        _check_common(a, self.weight_transposed, batch_sizes)
        if a.shape[1] != self.contraction_size:
            raise ValueError("输入和 down 权重的收缩维度不匹配")
        output = torch.empty(
            (a.shape[0], self.rank),
            device=a.device,
            dtype=a.dtype,
        )
        token_offsets, token_counts, max_tiles = _get_metadata(
            self,
            batch_sizes,
            a.shape[0],
            self.block_m,
            a.device,
        )
        if max_tiles > 0:
            _launch(
                (max_tiles, self.num_experts),
                grouped_down_kernel,
                (
                    a,
                    self.weight_transposed,
                    output,
                    token_offsets,
                    token_counts,
                    self.rank,
                    self.block_m,
                    self.block_k,
                ),
            )
        return output


class CuTileLoraUpGrouped:
    """cuTile LoRA up，SM120 默认使用 128 行 tile。"""

    def __init__(
        self,
        weight: torch.Tensor,
        block_m: int = 128,
        block_n: int = 256,
    ):
        if weight.ndim != 3:
            raise ValueError("up 权重形状必须是 [E, R, N]")
        self.rank = _check_lora_rank(weight.shape[1])
        if not weight.is_cuda or not weight.is_contiguous():
            raise ValueError("up 权重必须是连续的 CUDA Tensor")
        if weight.dtype != torch.bfloat16:
            raise TypeError("up 权重必须是 bfloat16")
        self.weight = weight
        self.num_experts = weight.shape[0]
        self.output_size = weight.shape[2]
        self.block_m = block_m
        self.block_n = block_n
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        _clear_metadata(self)

    def __call__(
        self,
        hidden: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        _check_common(hidden, self.weight, batch_sizes)
        if hidden.shape[1] != self.rank:
            raise ValueError("up 输入的 rank 与权重不匹配")
        output = torch.empty(
            (hidden.shape[0], self.output_size),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        token_offsets, token_counts, max_tiles = _get_metadata(
            self,
            batch_sizes,
            hidden.shape[0],
            self.block_m,
            hidden.device,
        )
        if max_tiles > 0:
            _launch(
                (
                    max_tiles,
                    ct.cdiv(self.output_size, self.block_n),
                    self.num_experts,
                ),
                grouped_up_kernel,
                (
                    hidden,
                    self.weight,
                    output,
                    token_offsets,
                    token_counts,
                    self.rank,
                    self.block_m,
                    self.block_n,
                ),
            )
        return output


class CuTileLoraFusedDownUpGrouped:
    """融合 cuTile LoRA down/up，并返回中间矩阵。"""

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        block_m: int = 64,
        block_k: int = 64,
        block_n: int = 256,
    ):
        if down_weight.shape != up_weight.shape:
            raise ValueError("down 和 up 权重形状必须相同")
        if down_weight.ndim != 3:
            raise ValueError("权重形状必须是 [E, R, K]")
        self.rank = _check_lora_rank(down_weight.shape[1])
        if (
            not up_weight.is_cuda
            or not up_weight.is_contiguous()
            or up_weight.dtype != torch.bfloat16
            or up_weight.device != down_weight.device
        ):
            raise ValueError("up 权重的设备、类型或布局不正确")
        self.down = CuTileLoraDownGrouped(
            down_weight,
            block_m,
            block_k,
        )
        self.down_weight_transposed = self.down.weight_transposed
        self.up_weight = up_weight
        self.num_experts = down_weight.shape[0]
        self.hidden_size = down_weight.shape[2]
        self.block_m = block_m
        self.block_k = block_k
        self.block_n = block_n
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        _clear_metadata(self)

    def __call__(
        self,
        a: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_common(a, self.down_weight_transposed, batch_sizes)
        if a.shape[1] != self.hidden_size:
            raise ValueError("输入和 down 权重的收缩维度不匹配")
        hidden = torch.empty(
            (a.shape[0], self.rank),
            device=a.device,
            dtype=a.dtype,
        )
        output = torch.empty_like(a)
        token_offsets, token_counts, max_tiles = _get_metadata(
            self,
            batch_sizes,
            a.shape[0],
            self.block_m,
            a.device,
        )
        if max_tiles > 0:
            _launch(
                (max_tiles, self.num_experts),
                grouped_fused_downup_kernel,
                (
                    a,
                    self.down_weight_transposed,
                    self.up_weight,
                    hidden,
                    output,
                    token_offsets,
                    token_counts,
                    self.rank,
                    self.block_m,
                    self.block_k,
                    self.block_n,
                ),
            )
        return hidden, output


class CuTileLoraFusedAgradGrouped:
    """融合计算 cuTile LoRA up/down 的输入梯度。"""

    def __init__(
        self,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        block_m: int = 64,
        block_k: int = 64,
    ):
        if up_weight.shape != down_weight.shape:
            raise ValueError("down 和 up 权重形状必须相同")
        if up_weight.ndim != 3:
            raise ValueError("权重形状必须是 [E, R, K]")
        self.rank = _check_lora_rank(up_weight.shape[1])
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
        if up_weight.device != down_weight.device:
            raise ValueError("down 和 up 权重必须位于同一设备")
        self.up_weight = up_weight
        self.down_weight = down_weight
        self.num_experts = up_weight.shape[0]
        self.hidden_size = up_weight.shape[2]
        self.block_m = block_m
        self.block_k = block_k
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        _clear_metadata(self)

    def __call__(
        self,
        grad_output: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_common(grad_output, self.up_weight, batch_sizes)
        if grad_output.shape[1] != self.hidden_size:
            raise ValueError("输出梯度维度与权重不匹配")
        grad_hidden = torch.empty(
            (grad_output.shape[0], self.rank),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input = torch.empty_like(grad_output)
        token_offsets, token_counts, max_tiles = _get_metadata(
            self,
            batch_sizes,
            grad_output.shape[0],
            self.block_m,
            grad_output.device,
        )
        if max_tiles > 0:
            _launch(
                (max_tiles, self.num_experts),
                grouped_fused_agrad_kernel,
                (
                    grad_output,
                    self.up_weight,
                    self.down_weight,
                    grad_hidden,
                    grad_input,
                    token_offsets,
                    token_counts,
                    self.rank,
                    self.block_m,
                    self.block_k,
                ),
            )
        return grad_hidden, grad_input


class CuTileLoraBgradGrouped:
    """计算 rank=16/32 cuTile LoRA grouped GEMM 的权重梯度。"""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        block_tokens: int = 128,
        block_n: int = 256,
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.block_tokens = block_tokens
        self.block_n = block_n
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None

    def clear_metadata_cache(self) -> None:
        _clear_metadata(self)

    def __call__(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        if lhs.ndim != 2 or rhs.ndim != 2:
            raise ValueError("lhs/rhs 必须是二维 Tensor")
        rank = _check_lora_rank(lhs.shape[1])
        if rhs.shape[1] != self.hidden_size:
            raise ValueError("rhs 最后一维与 hidden_size 不匹配")
        if lhs.shape[0] != rhs.shape[0]:
            raise ValueError("lhs 和 rhs 行数必须相同")
        if (
            not lhs.is_cuda
            or not rhs.is_cuda
            or not lhs.is_contiguous()
            or not rhs.is_contiguous()
            or lhs.dtype != rhs.dtype
            or lhs.dtype != torch.bfloat16
            or lhs.device != rhs.device
        ):
            raise ValueError("lhs/rhs 的设备、类型或布局不正确")
        if (
            batch_sizes.ndim != 1
            or batch_sizes.numel() != self.num_experts
        ):
            raise ValueError("batch_sizes 长度必须等于 expert 数")
        output = torch.empty(
            (self.num_experts, rank, self.hidden_size),
            device=lhs.device,
            dtype=lhs.dtype,
        )
        token_offsets, token_counts, _ = _get_metadata(
            self,
            batch_sizes,
            lhs.shape[0],
            self.block_tokens,
            lhs.device,
        )
        _launch(
            (
                self.num_experts,
                ct.cdiv(self.hidden_size, self.block_n),
            ),
            grouped_bgrad_kernel,
            (
                lhs,
                rhs,
                output,
                token_offsets,
                token_counts,
                rank,
                self.block_tokens,
                self.block_n,
            ),
        )
        return output


class _CuTileLoraFusedDownUp(torch.autograd.Function):
    """三 kernel 反向的 cuTile LoRA Autograd 实现。"""

    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        _check_common(a, down_weight, batch_sizes)
        rank = _check_lora_rank(down_weight.shape[1])
        if a.shape[1] != down_weight.shape[2]:
            raise ValueError("输入和权重的隐藏维度不匹配")
        if (
            up_weight.ndim != 3
            or up_weight.shape[0] != down_weight.shape[0]
            or up_weight.shape[1] != rank
        ):
            raise ValueError("up 权重形状必须是 [E, R, N]")
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
        token_offsets, token_counts, max_tiles = _build_metadata(
            batch_sizes,
            a.shape[0],
            block_m,
            a.device,
        )
        down_weight_transposed = (
            down_weight.permute(0, 2, 1).contiguous()
        )
        hidden = torch.empty(
            (a.shape[0], rank),
            device=a.device,
            dtype=a.dtype,
        )
        output = torch.empty(
            (a.shape[0], up_weight.shape[2]),
            device=a.device,
            dtype=a.dtype,
        )
        if max_tiles > 0:
            _launch(
                (max_tiles, down_weight.shape[0]),
                grouped_fused_downup_kernel,
                (
                    a,
                    down_weight_transposed,
                    up_weight,
                    hidden,
                    output,
                    token_offsets,
                    token_counts,
                    rank,
                    block_m,
                    block_k,
                    block_n,
                ),
            )
        context.save_for_backward(
            a,
            down_weight,
            up_weight,
            hidden,
            token_offsets,
            token_counts,
        )
        context.max_tiles = max_tiles
        context.block_m = block_m
        context.block_k = block_k
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
            token_offsets,
            token_counts,
        ) = context.saved_tensors
        grad_output = grad_output.contiguous()
        grad_hidden = torch.empty_like(hidden)
        grad_input = torch.empty_like(a)
        rank = down_weight.shape[1]
        if context.max_tiles > 0:
            _launch(
                (context.max_tiles, down_weight.shape[0]),
                grouped_fused_agrad_kernel,
                (
                    grad_output,
                    up_weight,
                    down_weight,
                    grad_hidden,
                    grad_input,
                    token_offsets,
                    token_counts,
                    rank,
                    context.block_m,
                    context.block_k,
                ),
            )

        grad_down_weight = torch.empty_like(down_weight)
        grad_up_weight = torch.empty_like(up_weight)
        down_grid = (
            down_weight.shape[0],
            ct.cdiv(a.shape[1], 256),
        )
        _launch(
            down_grid,
            grouped_bgrad_kernel,
            (
                grad_hidden,
                a,
                grad_down_weight,
                token_offsets,
                token_counts,
                rank,
                128,
                256,
            ),
        )
        up_grid = (
            down_weight.shape[0],
            ct.cdiv(grad_output.shape[1], 256),
        )
        _launch(
            up_grid,
            grouped_bgrad_kernel,
            (
                hidden,
                grad_output,
                grad_up_weight,
                token_offsets,
                token_counts,
                rank,
                128,
                256,
            ),
        )
        return grad_input, grad_down_weight, grad_up_weight, None


def cutile_fused_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """执行融合前向和三 kernel 反向的 cuTile LoRA Grouped GEMM。"""
    return _CuTileLoraFusedDownUp.apply(
        a,
        down_weight,
        up_weight,
        batch_sizes,
    )
