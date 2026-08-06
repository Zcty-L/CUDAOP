"""LLM 自回归推理中，从 decoder logits 采样下一个 token。

该实现只依赖 PyTorch，重点展示单步采样的完整数据流，而不是替代
Transformers、vLLM 等推理框架中的高度优化实现。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """单 token 采样参数。

    默认值使用常见的 temperature + top-k + top-p 组合。min-p 与各类惩罚
    默认关闭，调用方可按模型和任务打开。temperature=0 或 do_sample=False
    表示 greedy decoding。
    """

    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self) -> None:
        """尽早拒绝非法参数，避免在 GPU kernel 中得到难定位的错误。"""
        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample 必须是 bool")
        if not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("temperature 必须是大于等于 0 的有限数")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k 必须是整数")
        if self.top_k < 0:
            raise ValueError("top_k 必须大于等于 0；0 表示关闭")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p 必须位于 (0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p 必须位于 [0, 1]")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty 必须是大于 0 的有限数")
        if not math.isfinite(self.frequency_penalty):
            raise ValueError("frequency_penalty 必须是有限数")
        if not math.isfinite(self.presence_penalty):
            raise ValueError("presence_penalty 必须是有限数")


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """一次单步采样的结果。

    Attributes:
        token_ids: 每个 batch 采出的 token id，形状为 ``[batch_size]``。
        token_probabilities: token 在最终归一化分布中的概率。
        probabilities: 经过惩罚、温度和截断后得到的完整概率分布。
        processed_logits: 用于最终 softmax 的 logits；被过滤项为负无穷。
    """

    token_ids: Tensor
    token_probabilities: Tensor
    probabilities: Tensor
    processed_logits: Tensor


def _normalize_inputs(
    logits: Tensor,
    previous_token_ids: Tensor | None,
    previous_token_mask: Tensor | None,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """统一成 batch 形式，并验证 history 与 logits 是否匹配。"""
    if not logits.is_floating_point():
        raise TypeError("logits 必须是浮点 Tensor")
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError("logits 形状必须是 [vocab_size] 或 [batch, vocab_size]")
    if logits.shape[-1] == 0:
        raise ValueError("vocab_size 不能为 0")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("logits 不能包含 NaN 或正无穷")
    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError("每个 batch 至少需要一个有限 logits")

    if previous_token_ids is None:
        if previous_token_mask is not None:
            raise ValueError("没有 previous_token_ids 时不能传 previous_token_mask")
        return logits, None, None

    if previous_token_ids.ndim == 1:
        # 单 batch 时允许调用方直接传入 [sequence_length]。
        if logits.shape[0] != 1:
            raise ValueError("batch logits 需要 [batch, sequence] 形式的历史 token")
        previous_token_ids = previous_token_ids.unsqueeze(0)
    if previous_token_ids.ndim != 2:
        raise ValueError("previous_token_ids 必须是 [sequence] 或 [batch, sequence]")
    if previous_token_ids.shape[0] != logits.shape[0]:
        raise ValueError("previous_token_ids 与 logits 的 batch_size 不一致")
    if previous_token_ids.device != logits.device:
        raise ValueError("previous_token_ids 与 logits 必须位于同一设备")
    if previous_token_ids.dtype == torch.bool or previous_token_ids.is_floating_point():
        raise TypeError("previous_token_ids 必须是整数 Tensor")

    if previous_token_mask is None:
        previous_token_mask = torch.ones_like(previous_token_ids, dtype=torch.bool)
    else:
        if previous_token_mask.ndim == 1 and previous_token_ids.shape[0] == 1:
            previous_token_mask = previous_token_mask.unsqueeze(0)
        if previous_token_mask.shape != previous_token_ids.shape:
            raise ValueError("previous_token_mask 必须与 previous_token_ids 同形状")
        if previous_token_mask.device != logits.device:
            raise ValueError("previous_token_mask 与 logits 必须位于同一设备")
        previous_token_mask = previous_token_mask.to(dtype=torch.bool)

    valid_ids = previous_token_ids[previous_token_mask]
    if valid_ids.numel() > 0:
        vocab_size = logits.shape[-1]
        if (valid_ids < 0).any() or (valid_ids >= vocab_size).any():
            raise ValueError("有效历史 token id 必须位于 [0, vocab_size)")

    return logits, previous_token_ids.to(dtype=torch.long), previous_token_mask


def _count_previous_tokens(
    previous_token_ids: Tensor,
    previous_token_mask: Tensor,
    vocab_size: int,
    dtype: torch.dtype,
) -> Tensor:
    """统计每个历史 token 的出现次数，padding 位置不参与统计。"""
    batch_size = previous_token_ids.shape[0]
    counts = torch.zeros(
        (batch_size, vocab_size),
        device=previous_token_ids.device,
        dtype=dtype,
    )

    # 被 mask 的 id 先替换成 0，再把对应增量设为 0，避免 padding id 越界。
    safe_ids = previous_token_ids.masked_fill(~previous_token_mask, 0)
    increments = previous_token_mask.to(dtype=dtype)
    counts.scatter_add_(dim=1, index=safe_ids, src=increments)
    return counts


def _apply_history_penalties(
    logits: Tensor,
    token_counts: Tensor | None,
    config: SamplingConfig,
) -> Tensor:
    """应用 repetition、frequency 与 presence 三种常见惩罚。"""
    if token_counts is None:
        return logits

    if config.repetition_penalty != 1.0:
        # Hugging Face 风格的乘法惩罚：对负 logits 做乘法，对正 logits 做除法。
        # 这样无论原始值正负，见过的 token 都会变得更不可能。
        repeated = token_counts > 0
        penalized = torch.where(
            logits < 0,
            logits * config.repetition_penalty,
            logits / config.repetition_penalty,
        )
        logits = torch.where(repeated, penalized, logits)

    if config.frequency_penalty != 0.0:
        # 出现次数越多，线性减去的值越大。
        logits = logits - config.frequency_penalty * token_counts

    if config.presence_penalty != 0.0:
        # 只关心是否出现过，不区分出现一次还是多次。
        logits = logits - config.presence_penalty * (token_counts > 0)

    return logits


def _apply_top_k(logits: Tensor, top_k: int) -> Tensor:
    """只保留 logits 最大的 k 个候选；0 表示不启用。"""
    if top_k == 0 or top_k >= logits.shape[-1]:
        return logits

    threshold = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, -torch.inf)


def _apply_top_p(logits: Tensor, top_p: float) -> Tensor:
    """保留累计概率刚好覆盖 top_p 的最小高概率 token 集合。"""
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    # 右移一位很关键：累计概率首次超过 top_p 的 token 仍需保留。
    sorted_remove_mask = cumulative_probabilities > top_p
    sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
    sorted_remove_mask[..., 0] = False

    remove_mask = torch.zeros_like(sorted_remove_mask)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove_mask)
    return logits.masked_fill(remove_mask, -torch.inf)


def _apply_min_p(logits: Tensor, min_p: float) -> Tensor:
    """过滤概率低于 ``最大 token 概率 * min_p`` 的候选。"""
    if min_p == 0.0:
        return logits

    probabilities = torch.softmax(logits, dim=-1)
    scaled_threshold = probabilities.max(dim=-1, keepdim=True).values * min_p
    return logits.masked_fill(probabilities < scaled_threshold, -torch.inf)


def prepare_sampling_logits(
    logits: Tensor,
    previous_token_ids: Tensor | None = None,
    config: SamplingConfig | None = None,
    previous_token_mask: Tensor | None = None,
) -> Tensor:
    """把 decoder 原始 logits 处理成可用于 softmax/采样的 logits。

    处理顺序为：历史惩罚 -> temperature -> top-k -> top-p -> min-p。
    为避免 FP16/BF16 softmax 的数值问题，所有计算统一提升到 FP32。

    ``previous_token_mask`` 用于 batch 中存在 padding 的情况；True 表示对应
    token 有效。若所有序列都没有 padding，可不传该参数。
    """
    config = config or SamplingConfig()
    logits, previous_token_ids, previous_token_mask = _normalize_inputs(
        logits,
        previous_token_ids,
        previous_token_mask,
    )

    # clone 防止修改模型原始输出；FP32 可提升 softmax 与累计概率的稳定性。
    processed_logits = logits.to(dtype=torch.float32).clone()
    token_counts = None
    if previous_token_ids is not None and previous_token_mask is not None:
        token_counts = _count_previous_tokens(
            previous_token_ids,
            previous_token_mask,
            vocab_size=processed_logits.shape[-1],
            dtype=processed_logits.dtype,
        )

    processed_logits = _apply_history_penalties(
        processed_logits,
        token_counts,
        config,
    )

    greedy = not config.do_sample or config.temperature == 0.0
    if greedy:
        # greedy 仍应用历史惩罚，但不使用 temperature 和候选集截断。
        return processed_logits

    processed_logits = processed_logits / config.temperature
    processed_logits = _apply_top_k(processed_logits, config.top_k)
    processed_logits = _apply_top_p(processed_logits, config.top_p)
    processed_logits = _apply_min_p(processed_logits, config.min_p)

    if not torch.isfinite(processed_logits).any(dim=-1).all():
        raise RuntimeError("采样过滤移除了某个 batch 的全部候选 token")
    return processed_logits


@torch.inference_mode()
def sample_next_token(
    logits: Tensor,
    previous_token_ids: Tensor | None = None,
    config: SamplingConfig | None = None,
    previous_token_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SamplingResult:
    """从 decoder 最后一个位置的 logits 中为每个 batch 输出一个 token。

    Args:
        logits: LM Head 输出，形状 ``[vocab_size]`` 或
            ``[batch_size, vocab_size]``。若模型输出是
            ``[batch, sequence, vocab]``，调用方应先取 ``logits[:, -1, :]``。
        previous_token_ids: 已有上下文的 token id，用于计算重复类惩罚。
        config: 采样配置；不传时使用 :class:`SamplingConfig` 默认值。
        previous_token_mask: 历史 token 的有效位掩码，用于忽略 padding。
        generator: PyTorch 随机数生成器；传入固定 seed 可复现实验。

    Returns:
        包含 token id、被选 token 概率和完整最终分布的 SamplingResult。
    """
    config = config or SamplingConfig()
    processed_logits = prepare_sampling_logits(
        logits=logits,
        previous_token_ids=previous_token_ids,
        config=config,
        previous_token_mask=previous_token_mask,
    )
    probabilities = torch.softmax(processed_logits, dim=-1)

    greedy = not config.do_sample or config.temperature == 0.0
    if greedy:
        token_ids = torch.argmax(probabilities, dim=-1)
    else:
        # multinomial 会按每行的离散概率分布独立抽取一个 token。
        token_ids = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)

    token_probabilities = probabilities.gather(
        dim=-1,
        index=token_ids.unsqueeze(-1),
    ).squeeze(-1)
    return SamplingResult(
        token_ids=token_ids,
        token_probabilities=token_probabilities,
        probabilities=probabilities,
        processed_logits=processed_logits,
    )


def _run_demo() -> None:
    """使用一组模拟 decoder logits 展示单步采样调用。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SamplingConfig(
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        frequency_penalty=0.1,
        presence_penalty=0.1,
    )
    LOGGER.info(
        "配置: device=%s, temperature=%.2f, top_k=%d, top_p=%.2f, min_p=%.2f",
        device,
        config.temperature,
        config.top_k,
        config.top_p,
        config.min_p,
    )

    LOGGER.info("")
    LOGGER.info("阶段 1/3: 准备 decoder 最后位置的 logits 与历史 token")
    decoder_logits = torch.tensor(
        [[1.2, 0.1, 2.4, 1.8, -0.5, 0.7, 1.1, 0.3]],
        device=device,
    )
    previous_token_ids = torch.tensor([[2, 4, 2, 6]], device=device)

    LOGGER.info("阶段 2/3: 应用惩罚、温度缩放与候选集过滤")
    generator = torch.Generator(device=device).manual_seed(2026)
    result = sample_next_token(
        logits=decoder_logits,
        previous_token_ids=previous_token_ids,
        config=config,
        generator=generator,
    )

    LOGGER.info("阶段 3/3: 从最终概率分布抽取一个 token")
    LOGGER.info(
        "关键结果: token_id=%d, probability=%.6f",
        result.token_ids.item(),
        result.token_probabilities.item(),
    )
    LOGGER.info("")
    LOGGER.info("[SUCCESS] decoder 单 token 采样完成")


if __name__ == "__main__":
    _run_demo()
