"""标准 LoRA-MoE 的 PyTorch 参考实现。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAMoEStandard(nn.Module):
    """每个 expert 包含完整 gate/up/down LoRA 分支的标准 MoE。"""

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
        if gmm_backend != "cutlass" and rank != 16:
            raise ValueError(
                f"{gmm_backend} GMM 后端仅支持 rank=16"
            )

        self._validate_original_mlp(original_mlp)
        self.original_mlp = original_mlp
        self.num_experts = num_experts
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.gmm_backend = gmm_backend

        for parameter in self.original_mlp.parameters():
            parameter.requires_grad = False

        gate_proj = original_mlp.gate_proj
        up_proj = original_mlp.up_proj
        down_proj = original_mlp.down_proj

        hidden_size = gate_proj.in_features
        intermediate_size = gate_proj.out_features
        factory_kwargs = {
            "device": gate_proj.weight.device,
            "dtype": gate_proj.weight.dtype,
        }

        self.gate_up_lora_A = nn.Parameter(
            torch.empty(
                num_experts,
                2 * rank,
                hidden_size,
                **factory_kwargs,
            )
        )
        self.gate_lora_B = nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                rank,
                **factory_kwargs,
            )
        )
        self.up_lora_B = nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                rank,
                **factory_kwargs,
            )
        )
        self.down_lora_A = nn.Parameter(
            torch.empty(
                num_experts,
                rank,
                down_proj.in_features,
                **factory_kwargs,
            )
        )
        self.down_lora_B = nn.Parameter(
            torch.empty(
                num_experts,
                down_proj.out_features,
                rank,
                **factory_kwargs,
            )
        )

        self.reset_lora_parameters()

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
        """使用常见 LoRA 初始化：A 为 Kaiming，B 为零。"""

        for expert in range(self.num_experts):
            nn.init.kaiming_uniform_(
                self.gate_up_lora_A[expert],
                a=math.sqrt(5),
            )
            nn.init.kaiming_uniform_(
                self.down_lora_A[expert],
                a=math.sqrt(5),
            )

        nn.init.zeros_(self.gate_lora_B)
        nn.init.zeros_(self.up_lora_B)
        nn.init.zeros_(self.down_lora_B)

    def _forward_loop(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """使用逐 expert Torch 循环执行标准 LoRA-MoE。"""

        batch_size, sequence_length, hidden_size = x.shape
        top_k = top_k_indices.shape[-1]

        x_flat = x.reshape(-1, hidden_size)
        indices_flat = top_k_indices.reshape(-1, top_k)
        weights_flat = top_k_weights.reshape(-1, top_k)
        output_flat = torch.zeros_like(x_flat)

        ddp_safety = sum(
            parameter.view(-1)[0] * 0
            for parameter in self.parameters()
        )
        x_dropped = self.lora_dropout(x_flat)

        for expert in range(self.num_experts):
            mask_per_slot = indices_flat == expert
            token_mask = mask_per_slot.any(dim=-1)
            if not token_mask.any():
                continue

            token_indices = token_mask.nonzero(as_tuple=True)[0]
            x_selected = x_dropped[token_indices]

            gate_up_hidden = F.linear(
                x_selected,
                self.gate_up_lora_A[expert],
            )
            gate_hidden, up_hidden = gate_up_hidden.chunk(2, dim=-1)
            gate_output = (
                self.original_mlp.gate_proj(x_flat[token_indices])
                + F.linear(
                    gate_hidden,
                    self.gate_lora_B[expert],
                )
                * self.scaling
            )
            up_output = (
                self.original_mlp.up_proj(x_flat[token_indices])
                + F.linear(
                    up_hidden,
                    self.up_lora_B[expert],
                )
                * self.scaling
            )

            intermediate = (
                self.original_mlp.act_fn(gate_output) * up_output
            )

            down_hidden = F.linear(
                intermediate,
                self.down_lora_A[expert],
            )
            expert_output = (
                self.original_mlp.down_proj(intermediate)
                + F.linear(
                    down_hidden,
                    self.down_lora_B[expert],
                )
                * self.scaling
            )

            combined_weight = (
                mask_per_slot[token_indices].to(x.dtype)
                * weights_flat[token_indices].to(x.dtype)
            ).sum(dim=-1, keepdim=True)
            output_flat.index_add_(
                0,
                token_indices,
                (
                    expert_output * combined_weight
                ).to(output_flat.dtype),
            )

        return output_flat.reshape(
            batch_size,
            sequence_length,
            hidden_size,
        ) + ddp_safety

    def _forward_pad(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """填充 expert 维度并通过批量张量运算执行标准 LoRA-MoE。"""

        batch_size, sequence_length, hidden_size = x.shape
        num_tokens = batch_size * sequence_length
        top_k = top_k_indices.shape[-1]

        x_flat = x.reshape(num_tokens, hidden_size)
        indices_flat = top_k_indices.reshape(num_tokens, top_k)
        weights_flat = top_k_weights.reshape(num_tokens, top_k)

        weight_full = torch.zeros(
            num_tokens,
            self.num_experts,
            device=x.device,
            dtype=x.dtype,
        )
        weight_full.scatter_(
            1,
            indices_flat,
            weights_flat.to(weight_full.dtype),
        )

        x_dropped = self.lora_dropout(x_flat)

        combined_hidden = torch.einsum(
            "nd,erd->ner",
            x_dropped,
            self.gate_up_lora_A,
        )
        gate_hidden, up_hidden = combined_hidden.chunk(2, dim=-1)
        gate_delta = (
            torch.einsum(
                "ner,eor->neo",
                gate_hidden,
                self.gate_lora_B,
            )
            * self.scaling
        )
        up_delta = (
            torch.einsum(
                "ner,eor->neo",
                up_hidden,
                self.up_lora_B,
            )
            * self.scaling
        )

        gate_full = (
            self.original_mlp.gate_proj(x_flat).unsqueeze(1)
            + gate_delta
        )
        up_full = (
            self.original_mlp.up_proj(x_flat).unsqueeze(1)
            + up_delta
        )
        intermediate_per_expert = (
            self.original_mlp.act_fn(gate_full) * up_full
        )
        intermediate = (
            intermediate_per_expert * weight_full.unsqueeze(-1)
        ).sum(dim=1)

        down_base = self.original_mlp.down_proj(intermediate)
        down_hidden = torch.einsum(
            "neo,ero->ner",
            intermediate_per_expert,
            self.down_lora_A,
        )
        down_hidden_weighted = (
            down_hidden * weight_full.unsqueeze(-1)
        )
        down_delta = torch.einsum(
            "ner,eir->ni",
            down_hidden_weighted,
            self.down_lora_B,
        ) * self.scaling

        return (
            down_base + down_delta
        ).reshape(
            batch_size,
            sequence_length,
            hidden_size,
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
        """生成按 expert 分组所需的排序索引和分组边界。"""

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

    def _forward_group(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """使用路由重排和 Grouped GEMM 执行标准 LoRA-MoE。"""

        import gmm_ops

        batch_size, sequence_length, hidden_size = x.shape
        top_k = top_k_indices.shape[-1]
        num_tokens = batch_size * sequence_length

        x_flat = x.reshape(num_tokens, hidden_size)
        ddp_safety = sum(
            parameter.view(-1)[0] * 0
            for parameter in self.parameters()
        )
        top_experts_flat = (
            top_k_indices.reshape(-1).int().contiguous()
        )
        indices, bin_ids, bins, tokens_per_expert = (
            self.indices_and_bins(top_experts_flat)
        )

        x_gathered = gmm_ops.gather(
            x_flat,
            indices,
            bin_ids,
            bins,
            top_k,
        )
        x_dropped = self.lora_dropout(x_gathered)

        gate_base = gmm_ops.gather(
            self.original_mlp.gate_proj(x_flat),
            indices,
            bin_ids,
            bins,
            top_k,
        )
        up_base = gmm_ops.gather(
            self.original_mlp.up_proj(x_flat),
            indices,
            bin_ids,
            bins,
            top_k,
        )

        batch_sizes = tokens_per_expert.to(torch.long)
        if self.gmm_backend == "cutlass":
            gate_up_hidden = gmm_ops.gmm(
                x_dropped,
                self.gate_up_lora_A,
                batch_sizes,
                trans_b=True,
            )
            gate_hidden, up_hidden = gate_up_hidden.chunk(2, dim=-1)
            gate_delta = gmm_ops.gmm(
                gate_hidden.contiguous(),
                self.gate_lora_B,
                batch_sizes,
                trans_b=True,
            )
            up_delta = gmm_ops.gmm(
                up_hidden.contiguous(),
                self.up_lora_B,
                batch_sizes,
                trans_b=True,
            )
        else:
            gate_delta = gmm_ops.lora_gmm(
                x_dropped,
                self.gate_up_lora_A[:, :self.rank],
                self.gate_lora_B,
                batch_sizes,
                self.gmm_backend,
            )
            up_delta = gmm_ops.lora_gmm(
                x_dropped,
                self.gate_up_lora_A[:, self.rank:],
                self.up_lora_B,
                batch_sizes,
                self.gmm_backend,
            )
        gate_delta = gate_delta * self.scaling
        up_delta = up_delta * self.scaling

        gate_full = gate_base + gate_delta
        up_full = up_base + up_delta
        intermediate_per_slot = (
            self.original_mlp.act_fn(gate_full) * up_full
        )

        weights_flat = (
            top_k_weights.reshape(-1)
            .to(dtype=x.dtype)
            .contiguous()
        )
        intermediate = gmm_ops.scatter(
            intermediate_per_slot,
            indices,
            bin_ids,
            weights_flat,
            bins,
            top_k,
        )

        down_base = self.original_mlp.down_proj(intermediate)
        down_delta = gmm_ops.lora_gmm(
            intermediate_per_slot,
            self.down_lora_A,
            self.down_lora_B,
            batch_sizes,
            self.gmm_backend,
        ) * self.scaling
        down_delta_weighted = gmm_ops.scatter(
            down_delta,
            indices,
            bin_ids,
            weights_flat,
            bins,
            top_k,
        )

        return (
            down_base.reshape(num_tokens, hidden_size)
            + down_delta_weighted
        ).reshape(
            batch_size,
            sequence_length,
            hidden_size,
        ) + ddp_safety

    def forward(
        self,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_loop(
            x,
            top_k_indices,
            top_k_weights,
        )
