"""使用基础 PyTorch 算子实现常见注意力机制。"""

import math

import torch
from torch import nn


class MHA(nn.Module):
    """最基础的多头注意力。

    输入和输出形状均为 ``[batch, sequence, d_model]``。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape

        # 1. 线性投影得到 Q、K、V。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分多头：[B, S, H * D] -> [B, H, S, D]。
        q = q.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 3. S = Q @ K^T / sqrt(D)。
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # 4. Causal mask：每个 token 只能看到自己和之前的 token。
        if is_causal:
            causal_mask = torch.ones(
                sequence,
                sequence,
                dtype=torch.bool,
                device=x.device,
            ).tril()
            scores = scores.masked_fill(
                ~causal_mask,
                -torch.inf,
            )

        # 5. mask=True 表示该位置可以参与注意力。
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -torch.inf)
            else:
                scores = scores + mask

        # 6. P = softmax(S)。
        attention = torch.softmax(scores, dim=-1)

        # 7. O = P @ V。
        output = attention @ v

        # 8. 合并多头并进行输出投影。
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, sequence, -1)
        return self.out_proj(output)


class GQA(nn.Module):
    """分组查询注意力：多个 Query 头共享一组 K/V 头。"""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_query_heads != 0:
            raise ValueError("d_model 必须能被 num_query_heads 整除")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "num_query_heads 必须能被 num_kv_heads 整除"
            )

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.heads_per_group = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape

        # 1. Q 使用较多的头，K/V 只生成较少的共享头。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch,
            sequence,
            self.num_query_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch,
            sequence,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch,
            sequence,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 2. 为便于理解，显式复制 K/V，使头数与 Q 一致。
        k = k.repeat_interleave(self.heads_per_group, dim=1)
        v = v.repeat_interleave(self.heads_per_group, dim=1)

        # 3. 后续计算与普通 MHA 相同。
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -torch.inf)
            else:
                scores = scores + mask

        attention = torch.softmax(scores, dim=-1)
        output = attention @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, sequence, -1)
        return self.out_proj(output)


class MQA(nn.Module):
    """多查询注意力：所有 Query 头共享唯一的一组 K/V。"""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_query_heads != 0:
            raise ValueError("d_model 必须能被 num_query_heads 整除")

        self.num_query_heads = num_query_heads
        self.head_dim = d_model // num_query_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape

        # 1. Q 生成多个头，K/V 各自只生成一个头。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch,
            sequence,
            self.num_query_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch,
            sequence,
            1,
            self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch,
            sequence,
            1,
            self.head_dim,
        ).transpose(1, 2)

        # 2. 唯一的 K/V 头被所有 Query 头共享。
        k = k.repeat(1, self.num_query_heads, 1, 1)
        v = v.repeat(1, self.num_query_heads, 1, 1)

        # 3. 后续计算与普通 MHA 相同。
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -torch.inf)
            else:
                scores = scores + mask

        attention = torch.softmax(scores, dim=-1)
        output = attention @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, sequence, -1)
        return self.out_proj(output)


class MLA(nn.Module):
    """便于学习的多头潜在注意力。

    K/V 先压缩为共享的低秩 latent，再分别恢复为多头 K 和 V。
    这个基础版本不包含 RoPE，重点展示 MLA 的 KV 压缩路径。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_lora_rank: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)

        # K/V 共享同一个低秩表示。
        self.kv_down_proj = nn.Linear(
            d_model,
            kv_lora_rank,
            bias=bias,
        )
        self.k_up_proj = nn.Linear(
            kv_lora_rank,
            d_model,
            bias=bias,
        )
        self.v_up_proj = nn.Linear(
            kv_lora_rank,
            d_model,
            bias=bias,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape

        # 1. Q 直接从输入投影得到。
        q = self.q_proj(x)

        # 2. 输入先压缩成低秩的 KV latent。
        kv_latent = self.kv_down_proj(x)

        # 3. 从同一个 latent 分别恢复 K 和 V。
        k = self.k_up_proj(kv_latent)
        v = self.v_up_proj(kv_latent)

        # 4. 拆分多头。
        q = q.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 5. 使用恢复后的 Q、K、V 执行最基础的多头注意力。
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -torch.inf)
            else:
                scores = scores + mask

        attention = torch.softmax(scores, dim=-1)
        output = attention @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, sequence, -1)
        return self.out_proj(output)


class LinearAttention(nn.Module):
    """使用 ``ELU(x) + 1`` 特征映射的线性注意力。

    先计算 ``K^T @ V``，再计算 ``Q @ (K^T @ V)``，因此不会生成
    ``[sequence, sequence]`` 的注意力矩阵。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape

        # 1. 线性投影得到 Q、K、V。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分多头：[B, S, H * D] -> [B, H, S, D]。
        q = q.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = k.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch,
            sequence,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 3. 使用正值特征映射代替 softmax。
        q = torch.nn.functional.elu(q) + 1.0
        k = torch.nn.functional.elu(k) + 1.0

        # 4. 利用矩阵乘法结合律，先计算 K^T @ V。
        # [B, H, D, S] @ [B, H, S, D] -> [B, H, D, D]
        kv = k.transpose(-2, -1) @ v

        # 5. 计算归一化分母：Q @ sum(K)。
        k_sum = k.sum(dim=-2) # [B, H, D]
        denominator = q @ k_sum.unsqueeze(-1) # [B, H, S, D] @ [B, H, D, 1] -> [B, H, S, 1]
        denominator = denominator.clamp_min(self.eps)

        # 6. O = Q @ (K^T @ V) / (Q @ sum(K))。
        output = q @ kv # [B, H, S, D] @ [B, H, D, D] -> [B, H, S, D]
        output = output / denominator

        # 7. 合并多头并进行输出投影。
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, sequence, -1)
        return self.out_proj(output)


__all__ = ["GQA", "LinearAttention", "MHA", "MLA", "MQA"]
