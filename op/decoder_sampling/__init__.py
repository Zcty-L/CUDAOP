"""LLM decoder 单 token 采样接口。"""

from .decoder_sampling import (
    SamplingConfig,
    SamplingResult,
    prepare_sampling_logits,
    sample_next_token,
)


__all__ = [
    "SamplingConfig",
    "SamplingResult",
    "prepare_sampling_logits",
    "sample_next_token",
]
