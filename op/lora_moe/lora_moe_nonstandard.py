"""非标准 LoRA-MoE：gate、up、down 三个子层分别执行 MoE 路由。"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAMoENonstandard(nn.Module):
    """将原始 MLP 的三个线性投影分别包装为 LoRA-MoE 子层。

    与标准 MoE 不同，本实现先在每个子层内聚合各 expert 的 LoRA
    增量，再将聚合结果传入下一个子层。路由索引和权重由调用方提供，
    gate、up、down 三个子层共享同一份路由结果。
    """

    _METHODS = {
        "loop": "_forward_loop",
        "torch": "_forward_loop",
        "pad": "_forward_pad",
        "group": "_forward_group",
    }
    _GMM_BACKENDS = ("cutlass", "triton", "cutile")

    def __init__(
        self,
        original_mlp: nn.Module,
        num_experts: int,
        rank: int,
        lora_alpha: float,
        lora_dropout: float = 0.0,
        gmm_backend: str = "cutlass",
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            raise ValueError("num_experts 必须大于 0")
        if rank <= 0:
            raise ValueError("rank 必须大于 0")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError("lora_dropout 必须位于 [0, 1) 区间")
        if gmm_backend not in self._GMM_BACKENDS:
            raise ValueError(
                f"不支持的 GMM 后端: {gmm_backend}，"
                f"可选值为 {self._GMM_BACKENDS}"
            )
        if gmm_backend != "cutlass" and rank not in (16, 32):
            raise ValueError(
                f"{gmm_backend} GMM 后端仅支持 rank=16/32"
            )

        self._validate_original_mlp(original_mlp)
        self.original_mlp = original_mlp
        self.num_experts = num_experts
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.gmm_backend = gmm_backend

        for parameter in self.original_mlp.parameters():
            parameter.requires_grad = False

        gate_proj = original_mlp.gate_proj
        up_proj = original_mlp.up_proj
        down_proj = original_mlp.down_proj
        factory_kwargs = {
            "device": gate_proj.weight.device,
            "dtype": gate_proj.weight.dtype,
        }

        self.gate_lora_A = self._new_weight(
            num_experts,
            rank,
            gate_proj.in_features,
            **factory_kwargs,
        )
        self.gate_lora_B = self._new_weight(
            num_experts,
            gate_proj.out_features,
            rank,
            **factory_kwargs,
        )
        self.up_lora_A = self._new_weight(
            num_experts,
            rank,
            up_proj.in_features,
            **factory_kwargs,
        )
        self.up_lora_B = self._new_weight(
            num_experts,
            up_proj.out_features,
            rank,
            **factory_kwargs,
        )
        self.down_lora_A = self._new_weight(
            num_experts,
            rank,
            down_proj.in_features,
            **factory_kwargs,
        )
        self.down_lora_B = self._new_weight(
            num_experts,
            down_proj.out_features,
            rank,
            **factory_kwargs,
        )

        self.gate_lora_dropout = nn.Dropout(lora_dropout)
        self.up_lora_dropout = nn.Dropout(lora_dropout)
        self.down_lora_dropout = nn.Dropout(lora_dropout)
        self.reset_lora_parameters()

    @staticmethod
    def _new_weight(
        *shape: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Parameter:
        return nn.Parameter(
            torch.empty(*shape, device=device, dtype=dtype)
        )

    @staticmethod
    def _validate_original_mlp(original_mlp: nn.Module) -> None:
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = getattr(original_mlp, name, None)
            if not isinstance(layer, nn.Linear):
                raise TypeError(
                    f"original_mlp.{name} 必须是 torch.nn.Linear"
                )

        if not callable(getattr(original_mlp, "act_fn", None)):
            raise TypeError("original_mlp.act_fn 必须是可调用对象")

        gate_proj = original_mlp.gate_proj
        up_proj = original_mlp.up_proj
        down_proj = original_mlp.down_proj
        if gate_proj.in_features != up_proj.in_features:
            raise ValueError("gate_proj 和 up_proj 的输入维度必须相同")
        if gate_proj.out_features != up_proj.out_features:
            raise ValueError("gate_proj 和 up_proj 的输出维度必须相同")
        if down_proj.in_features != gate_proj.out_features:
            raise ValueError(
                "down_proj 输入维度必须等于 gate_proj 输出维度"
            )
        if down_proj.out_features != gate_proj.in_features:
            raise ValueError(
                "down_proj 输出维度必须等于 gate_proj 输入维度"
            )

    def reset_lora_parameters(self) -> None:
        """LoRA A 使用 Kaiming 初始化，LoRA B 初始化为零。"""

        for weights in (
            self.gate_lora_A,
            self.up_lora_A,
            self.down_lora_A,
        ):
            for expert in range(self.num_experts):
                nn.init.kaiming_uniform_(
                    weights[expert],
                    a=math.sqrt(5),
                )
        nn.init.zeros_(self.gate_lora_B)
        nn.init.zeros_(self.up_lora_B)
        nn.init.zeros_(self.down_lora_B)

    def _validate_inputs(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> None:
        if x.ndim != 3:
            raise ValueError("x 必须是 [B, T, D] 三维张量")
        expected_shape = x.shape[:2]
        if top_k_indices.ndim != 3:
            raise ValueError(
                "top_k_indices 必须是 [B, T, K] 三维张量"
            )
        if top_k_weights.shape != top_k_indices.shape:
            raise ValueError(
                "top_k_weights 的形状必须与 top_k_indices 相同"
            )
        if top_k_indices.shape[:2] != expected_shape:
            raise ValueError("路由张量的 B、T 维度必须与 x 相同")
        if top_k_indices.shape[-1] == 0:
            raise ValueError("top-k 数量必须大于 0")
        if top_k_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("top_k_indices 必须使用 int32 或 int64")

    def _project_loop(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
        base_linear: nn.Linear,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        dropout: nn.Module,
    ) -> torch.Tensor:
        batch_size, sequence_length, in_features = x.shape
        top_k = top_k_indices.shape[-1]
        x_flat = x.reshape(-1, in_features)
        indices_flat = top_k_indices.reshape(-1, top_k)
        weights_flat = top_k_weights.reshape(-1, top_k)
        delta = x_flat.new_zeros(
            x_flat.shape[0],
            base_linear.out_features,
        )
        x_dropped = dropout(x_flat)

        for expert in range(self.num_experts):
            mask_per_slot = indices_flat == expert
            token_mask = mask_per_slot.any(dim=-1)
            if not token_mask.any():
                continue

            token_indices = token_mask.nonzero(as_tuple=True)[0]
            expert_delta = F.linear(
                F.linear(
                    x_dropped[token_indices],
                    lora_A[expert],
                ),
                lora_B[expert],
            )
            combined_weight = (
                mask_per_slot[token_indices].to(x.dtype)
                * weights_flat[token_indices].to(x.dtype)
            ).sum(dim=-1, keepdim=True)
            delta.index_add_(
                0,
                token_indices,
                expert_delta * combined_weight,
            )

        output = base_linear(x_flat) + delta * self.scaling
        return output.reshape(
            batch_size,
            sequence_length,
            base_linear.out_features,
        )

    def _project_pad(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
        base_linear: nn.Linear,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        dropout: nn.Module,
    ) -> torch.Tensor:
        batch_size, sequence_length, in_features = x.shape
        num_tokens = batch_size * sequence_length
        top_k = top_k_indices.shape[-1]
        x_flat = x.reshape(num_tokens, in_features)
        indices_flat = top_k_indices.reshape(num_tokens, top_k)
        weights_flat = top_k_weights.reshape(num_tokens, top_k)

        weight_full = x_flat.new_zeros(
            num_tokens,
            self.num_experts,
        )
        weight_full.scatter_add_(
            1,
            indices_flat.to(torch.long),
            weights_flat.to(x.dtype),
        )
        hidden = torch.einsum(
            "nd,erd->ner",
            dropout(x_flat),
            lora_A,
        )
        hidden = hidden * weight_full.unsqueeze(-1)
        expert_delta = torch.einsum(
            "ner,eor->no",
            hidden,
            lora_B,
        )
        output = (
            base_linear(x_flat)
            + expert_delta * self.scaling
        )
        return output.reshape(
            batch_size,
            sequence_length,
            base_linear.out_features,
        )

    def indices_and_bins(
        self,
        top_experts: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """生成 Grouped GEMM 使用的排序索引和 expert 分组边界。"""

        import gmm_ops

        top_experts = top_experts.int().contiguous()
        sort_end_bit = max(
            int(math.ceil(math.log2(self.num_experts))),
            1,
        )
        bin_ids, indices = gmm_ops.sort(
            top_experts,
            sort_end_bit,
        )
        tokens_per_expert = gmm_ops.histogram(
            top_experts,
            self.num_experts,
        )
        bins = gmm_ops.inclusive_cumsum(
            tokens_per_expert,
            0,
        )
        return indices, bin_ids, bins, tokens_per_expert

    def _project_group(
        self,
        x: torch.Tensor,
        top_k_weights: torch.Tensor,
        route: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        base_linear: nn.Linear,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        dropout: nn.Module,
    ) -> torch.Tensor:
        import gmm_ops

        batch_size, sequence_length, in_features = x.shape
        num_tokens = batch_size * sequence_length
        top_k = top_k_weights.shape[-1]
        indices, bin_ids, bins, tokens_per_expert = route
        x_flat = x.reshape(num_tokens, in_features)
        x_gathered = gmm_ops.gather(
            dropout(x_flat),
            indices,
            bin_ids,
            bins,
            top_k,
        )
        batch_sizes = tokens_per_expert.to(torch.long)
        delta_per_slot = gmm_ops.lora_gmm(
            x_gathered,
            lora_A,
            lora_B,
            batch_sizes,
            self.gmm_backend,
        )
        delta = gmm_ops.scatter(
            delta_per_slot,
            indices,
            bin_ids,
            top_k_weights.reshape(-1).to(x.dtype).contiguous(),
            bins,
            top_k,
        )
        output = base_linear(x_flat) + delta * self.scaling
        return output.reshape(
            batch_size,
            sequence_length,
            base_linear.out_features,
        )

    def _forward_impl(
        self,
        project: Callable[..., torch.Tensor],
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
        route: object = None,
    ) -> torch.Tensor:
        if route is None:
            common_args = (top_k_indices, top_k_weights)
        else:
            common_args = (top_k_weights, route)

        gate_output = project(
            x,
            *common_args,
            self.original_mlp.gate_proj,
            self.gate_lora_A,
            self.gate_lora_B,
            self.gate_lora_dropout,
        )
        up_output = project(
            x,
            *common_args,
            self.original_mlp.up_proj,
            self.up_lora_A,
            self.up_lora_B,
            self.up_lora_dropout,
        )
        intermediate = (
            self.original_mlp.act_fn(gate_output) * up_output
        )
        return project(
            intermediate,
            *common_args,
            self.original_mlp.down_proj,
            self.down_lora_A,
            self.down_lora_B,
            self.down_lora_dropout,
        )

    def _forward_loop(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """使用三个独立的逐 expert Torch 循环执行非标准 MoE。"""

        self._validate_inputs(x, top_k_indices, top_k_weights)
        output = self._forward_impl(
            self._project_loop,
            x,
            top_k_indices,
            top_k_weights,
        )
        ddp_safety = sum(
            parameter.view(-1)[0] * 0
            for name, parameter in self.named_parameters()
            if "lora_" in name
        )
        return output + ddp_safety.to(output.dtype)

    def _forward_pad(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """使用 expert 维填充运算执行三个独立的 MoE 子层。"""

        self._validate_inputs(x, top_k_indices, top_k_weights)
        return self._forward_impl(
            self._project_pad,
            x,
            top_k_indices,
            top_k_weights,
        )

    def _forward_group(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """使用本地路由算子和 CUTLASS Grouped GEMM 执行非标准 MoE。"""

        self._validate_inputs(x, top_k_indices, top_k_weights)
        route = self.indices_and_bins(
            top_k_indices.reshape(-1)
        )
        return self._forward_impl(
            self._project_group,
            x,
            top_k_indices,
            top_k_weights,
            route,
        )

    def forward(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
        method: str = "loop",
    ) -> torch.Tensor:
        method_name = self._METHODS.get(method)
        if method_name is None:
            supported = ", ".join(self._METHODS)
            raise ValueError(
                f"未知 method: {method}；可选值为 {supported}"
            )
        return getattr(self, method_name)(
            x,
            top_k_indices,
            top_k_weights,
        )
