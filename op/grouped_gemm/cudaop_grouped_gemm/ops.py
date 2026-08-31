"""支持自动求导的 Grouped GEMM 接口。"""

from typing import Any

import torch
import torch.nn.functional as functional

from cudaop_grouped_gemm import _C


CUTLASS_FUSED_BLOCK_M = 32
CUTLASS_FUSED_LORA_RANK = 16


class CutlassLoraBgradGrouped:
    """计算 rank=16 CUTLASS LoRA grouped GEMM 的权重梯度。"""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts 必须大于零")
        if hidden_size <= 0:
            raise ValueError("hidden_size 必须大于零")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_device = None
        self._cached_sizes_cpu = None
        self._cached_token_offsets = None
        self._cached_token_counts = None

    def clear_metadata_cache(self) -> None:
        """清除路由元数据；下一次调用会重新构建。"""
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_device = None
        self._cached_sizes_cpu = None
        self._cached_token_offsets = None
        self._cached_token_counts = None

    def _get_metadata(
        self,
        batch_sizes: torch.Tensor,
        total_rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        version = batch_sizes._version
        if (
            self._cached_batch_sizes is batch_sizes
            and self._cached_batch_sizes_version == version
            and self._cached_total_rows == total_rows
            and self._cached_device == device
        ):
            return (
                self._cached_sizes_cpu,
                self._cached_token_offsets,
                self._cached_token_counts,
            )

        sizes_cpu = batch_sizes.detach().to(
            device="cpu",
            dtype=torch.int64,
        ).contiguous()
        if torch.any(sizes_cpu < 0):
            raise ValueError("batch_sizes 不能包含负数")
        if int(sizes_cpu.sum()) != total_rows:
            raise ValueError("batch_sizes 之和必须等于输入行数")
        if total_rows > torch.iinfo(torch.int32).max:
            raise ValueError("总 token 数超出 int32 范围")

        token_counts = sizes_cpu.to(
            device=device,
            dtype=torch.int32,
        )
        token_offsets = torch.zeros(
            self.num_experts,
            device=device,
            dtype=torch.int32,
        )
        if self.num_experts > 1:
            token_offsets[1:] = torch.cumsum(
                token_counts[:-1],
                dim=0,
                dtype=torch.int32,
            )

        self._cached_batch_sizes = batch_sizes
        self._cached_batch_sizes_version = version
        self._cached_total_rows = total_rows
        self._cached_device = device
        self._cached_sizes_cpu = sizes_cpu
        self._cached_token_offsets = token_offsets
        self._cached_token_counts = token_counts
        return sizes_cpu, token_offsets, token_counts

    def __call__(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        if (
            lhs.ndim != 2
            or lhs.shape[1] != CUTLASS_FUSED_LORA_RANK
            or rhs.ndim != 2
            or rhs.shape[1] != self.hidden_size
        ):
            raise ValueError("lhs/rhs 形状必须是 [N,16] 和 [N,K]")
        if lhs.shape[0] != rhs.shape[0]:
            raise ValueError("lhs 和 rhs 行数必须相同")
        if (
            not lhs.is_cuda
            or not rhs.is_cuda
            or lhs.device != rhs.device
        ):
            raise ValueError("lhs 和 rhs 必须位于同一 CUDA 设备")
        if not lhs.is_contiguous() or not rhs.is_contiguous():
            raise ValueError("lhs 和 rhs 必须连续")
        if (
            lhs.dtype != torch.bfloat16
            or rhs.dtype != torch.bfloat16
        ):
            raise TypeError("lhs 和 rhs 必须是 bfloat16")
        if batch_sizes.ndim != 1:
            raise ValueError("batch_sizes 必须为一维")
        if batch_sizes.numel() != self.num_experts:
            raise ValueError("batch_sizes 的长度必须等于 expert 数")

        sizes_cpu, token_offsets, token_counts = self._get_metadata(
            batch_sizes,
            lhs.shape[0],
            lhs.device,
        )
        if self.hidden_size % 8 == 0 and lhs.shape[0] > 0:
            return _C.lora_bgrad_grouped(lhs, rhs, sizes_cpu)
        return _C.lora_bgrad(
            lhs,
            rhs,
            token_offsets,
            token_counts,
        )


class CutlassLoraFusedDownUpGrouped:
    """CUTLASS/CuTe 融合 LoRA down/up，并保存 BF16 hidden。"""

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        if (
            down_weight.ndim != 3
            or down_weight.shape[1] != CUTLASS_FUSED_LORA_RANK
        ):
            raise ValueError("down_weight 形状必须是 [E, 16, D]")
        if (
            up_weight.ndim != 3
            or up_weight.shape[1] != CUTLASS_FUSED_LORA_RANK
        ):
            raise ValueError("up_weight 形状必须是 [E, 16, I]")
        if down_weight.shape[0] != up_weight.shape[0]:
            raise ValueError("down 和 up 权重的 expert 数必须相同")
        if (
            not down_weight.is_cuda
            or not up_weight.is_cuda
            or not down_weight.is_contiguous()
            or not up_weight.is_contiguous()
        ):
            raise ValueError("权重必须是连续的 CUDA Tensor")
        if (
            down_weight.dtype != torch.bfloat16
            or up_weight.dtype != torch.bfloat16
        ):
            raise TypeError("权重必须是 bfloat16")
        if down_weight.device != up_weight.device:
            raise ValueError("down 和 up 权重必须位于同一设备")

        self.down_weight = down_weight
        self.up_weight = up_weight
        self.down_weight_transposed = (
            down_weight.transpose(1, 2).contiguous()
        )
        self.up_weight_transposed = (
            up_weight.transpose(1, 2).contiguous()
        )
        self._down_weight_version = down_weight._version
        self._up_weight_version = up_weight._version
        self.num_experts = down_weight.shape[0]
        self.input_size = down_weight.shape[2]
        self.output_size = up_weight.shape[2]
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None
        self._cached_sizes_cpu = None

    def _refresh_packed_weights(self) -> None:
        if self.down_weight._version != self._down_weight_version:
            self.down_weight_transposed = (
                self.down_weight.transpose(1, 2).contiguous()
            )
            self._down_weight_version = self.down_weight._version
        if self.up_weight._version != self._up_weight_version:
            self.up_weight_transposed = (
                self.up_weight.transpose(1, 2).contiguous()
            )
            self._up_weight_version = self.up_weight._version

    def clear_metadata_cache(self) -> None:
        """清除路由元数据；下一次调用会重新构建。"""
        self._cached_batch_sizes = None
        self._cached_batch_sizes_version = -1
        self._cached_total_rows = -1
        self._cached_metadata = None
        self._cached_sizes_cpu = None

    def _get_metadata(
        self,
        batch_sizes: torch.Tensor,
        total_rows: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        version = batch_sizes._version
        if (
            self._cached_batch_sizes is batch_sizes
            and self._cached_batch_sizes_version == version
            and self._cached_total_rows == total_rows
        ):
            return self._cached_metadata

        sizes_cpu = batch_sizes.detach().to(
            device="cpu",
            dtype=torch.int64,
        ).contiguous()
        if torch.any(sizes_cpu < 0):
            raise ValueError("batch_sizes 不能包含负数")
        if int(sizes_cpu.sum()) != total_rows:
            raise ValueError("batch_sizes 之和必须等于输入行数")

        cumulative_tiles = []
        token_offsets = []
        token_counts = sizes_cpu.tolist()
        total_tiles = 0
        token_offset = 0
        for token_count in token_counts:
            token_offsets.append(token_offset)
            token_offset += token_count
            total_tiles += (
                token_count + CUTLASS_FUSED_BLOCK_M - 1
            ) // CUTLASS_FUSED_BLOCK_M
            cumulative_tiles.append(total_tiles)
        if token_offset > torch.iinfo(torch.int32).max:
            raise ValueError("总 token 数超出 int32 范围")

        metadata = (
            torch.tensor(
                cumulative_tiles,
                device=self.down_weight.device,
                dtype=torch.int32,
            ),
            torch.tensor(
                token_offsets,
                device=self.down_weight.device,
                dtype=torch.int32,
            ),
            sizes_cpu.to(
                device=self.down_weight.device,
                dtype=torch.int32,
            ),
            total_tiles,
        )
        self._cached_batch_sizes = batch_sizes
        self._cached_batch_sizes_version = version
        self._cached_total_rows = total_rows
        self._cached_metadata = metadata
        self._cached_sizes_cpu = sizes_cpu
        return metadata

    def __call__(
        self,
        a: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not a.is_cuda or a.device != self.down_weight.device:
            raise ValueError("输入必须与权重位于同一 CUDA 设备")
        if a.dtype != torch.bfloat16:
            raise TypeError("输入必须是 bfloat16")
        if a.ndim != 2 or a.shape[1] != self.input_size:
            raise ValueError("输入形状必须是 [N, D]")
        if not a.is_contiguous():
            raise ValueError("输入必须连续")
        if batch_sizes.ndim != 1:
            raise ValueError("batch_sizes 必须为一维")
        if batch_sizes.numel() != self.num_experts:
            raise ValueError("batch_sizes 的长度必须等于 expert 数")

        self._refresh_packed_weights()
        metadata = self._get_metadata(batch_sizes, a.shape[0])
        cumulative_tiles, token_offsets, token_counts, total_tiles = (
            metadata
        )
        hidden, output = _C.fused_lora_forward(
            a,
            self.down_weight,
            self.up_weight_transposed,
            cumulative_tiles,
            token_offsets,
            token_counts,
            total_tiles,
        )
        return hidden, output

    def backward_input(
        self,
        grad_output: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """融合计算并保存 ``grad_hidden``，同时返回 ``grad_input``。"""
        if (
            not grad_output.is_cuda
            or grad_output.device != self.down_weight.device
        ):
            raise ValueError("grad_output 必须与权重位于同一 CUDA 设备")
        if grad_output.dtype != torch.bfloat16:
            raise TypeError("grad_output 必须是 bfloat16")
        if (
            grad_output.ndim != 2
            or grad_output.shape[1] != self.output_size
        ):
            raise ValueError("grad_output 形状必须是 [N, I]")
        if not grad_output.is_contiguous():
            raise ValueError("grad_output 必须连续")
        if batch_sizes.ndim != 1:
            raise ValueError("batch_sizes 必须为一维")
        if batch_sizes.numel() != self.num_experts:
            raise ValueError("batch_sizes 的长度必须等于 expert 数")

        self._refresh_packed_weights()
        metadata = self._get_metadata(
            batch_sizes,
            grad_output.shape[0],
        )
        cumulative_tiles, token_offsets, token_counts, total_tiles = (
            metadata
        )
        grad_hidden, grad_input = _C.fused_lora_backward(
            grad_output,
            self.up_weight,
            self.down_weight_transposed,
            cumulative_tiles,
            token_offsets,
            token_counts,
            total_tiles,
        )
        return grad_hidden, grad_input

    def forward_autograd(
        self,
        a: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """使用缓存的 packed weights/metadata 执行可求导融合路径。"""
        return _CutlassLoraFused.apply(
            a,
            self.down_weight,
            self.up_weight,
            batch_sizes,
            self,
        )


class _CutlassLoraFused(torch.autograd.Function):
    @staticmethod
    def forward(
        context: Any,
        a: torch.Tensor,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        batch_sizes: torch.Tensor,
        operation: CutlassLoraFusedDownUpGrouped | None,
    ) -> torch.Tensor:
        if operation is None:
            operation = CutlassLoraFusedDownUpGrouped(
                down_weight,
                up_weight,
            )
        elif (
            operation.down_weight is not down_weight
            or operation.up_weight is not up_weight
        ):
            raise ValueError("缓存对象与传入权重不匹配")
        hidden, output = operation(a, batch_sizes)
        metadata = operation._cached_metadata
        if metadata is None:
            metadata = operation._get_metadata(
                batch_sizes,
                a.shape[0],
            )
        cumulative_tiles, token_offsets, token_counts, total_tiles = (
            metadata
        )
        sizes_cpu = operation._cached_sizes_cpu
        if sizes_cpu is None:
            raise RuntimeError("CUTLASS 路由元数据未正确初始化")
        context.save_for_backward(
            a,
            down_weight,
            up_weight,
            hidden,
            operation.down_weight_transposed,
            cumulative_tiles,
            token_offsets,
            token_counts,
            sizes_cpu,
        )
        context.total_tiles = total_tiles
        return output

    @staticmethod
    def backward(
        context: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
    ]:
        (
            a,
            down_weight,
            up_weight,
            hidden,
            down_weight_transposed,
            cumulative_tiles,
            token_offsets,
            token_counts,
            sizes_cpu,
        ) = context.saved_tensors
        grad_output = grad_output.contiguous()
        grad_hidden, grad_input = _C.fused_lora_backward(
            grad_output,
            up_weight,
            down_weight_transposed,
            cumulative_tiles,
            token_offsets,
            token_counts,
            context.total_tiles,
        )
        if a.shape[1] % 8 == 0 and a.shape[0] > 0:
            grad_down_weight = _C.lora_bgrad_grouped(
                grad_hidden,
                a,
                sizes_cpu,
            )
        else:
            grad_down_weight = _C.lora_bgrad(
                grad_hidden,
                a,
                token_offsets,
                token_counts,
            )
        if grad_output.shape[1] % 8 == 0 and a.shape[0] > 0:
            grad_up_weight = _C.lora_bgrad_grouped(
                hidden,
                grad_output,
                sizes_cpu,
            )
        else:
            grad_up_weight = _C.lora_bgrad(
                hidden,
                grad_output,
                token_offsets,
                token_counts,
            )
        return (
            grad_input,
            grad_down_weight,
            grad_up_weight,
            None,
            None,
        )


class CutlassLoraFusedDownUp:
    """LoRA down/up 的完整 CUTLASS 1+3 训练调用器。"""

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        self.grouped_operation = CutlassLoraFusedDownUpGrouped(
            down_weight,
            up_weight,
        )

    def clear_metadata_cache(self) -> None:
        """清除路由元数据缓存。"""
        self.grouped_operation.clear_metadata_cache()

    def __call__(
        self,
        a: torch.Tensor,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行一个融合前向，反向由三个 kernel 完成。
        """
        return self.grouped_operation.forward_autograd(
            a,
            batch_sizes,
        )


def cutlass_fused_lora(
    a: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    batch_sizes: torch.Tensor,
) -> torch.Tensor:
    """执行 CUTLASS/CuTe 融合前向和三 kernel 反向。"""
    return _CutlassLoraFused.apply(
        a,
        down_weight,
        up_weight,
        batch_sizes,
        None,
    )


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
