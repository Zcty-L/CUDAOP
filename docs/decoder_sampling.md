# LLM Decoder 后如何采样并输出一个 token

本文对应可运行实现
[`op/decoder_sampling/decoder_sampling.py`](../op/decoder_sampling/decoder_sampling.py)。
代码只依赖 PyTorch，演示现代 LLM 推理服务常用的 greedy、temperature、
top-k、top-p（nucleus）、min-p，以及 repetition/frequency/presence penalty。

## 1. Decoder 实际输出了什么

自回归语言模型的一次生成可以分成 prefill 和 decode：

1. tokenizer 把输入文本转换成 token id；
2. prefill 一次处理整个 prompt，并把每层注意力的 K/V 写入 KV Cache；
3. 后续 decode 每轮只输入上一个 token，读取并更新 KV Cache；
4. decoder 最后一层得到最后位置的隐藏状态
   `hidden_state: [batch, hidden_size]`；
5. LM Head 做线性投影，得到
   `logits: [batch, vocab_size]`；
6. sampling 模块把每行 logits 变成概率分布，并选择一个 token id；
7. 新 token 追加到序列。若它是 EOS，生成结束；否则进入下一轮 decode。

因此，严格来说不是直接对 decoder hidden state 采样，而是对 LM Head 输出的
词表 logits 采样。完整模型常返回 `[batch, sequence, vocab_size]`，单步生成只使用
最后一个位置：

```python
next_token_logits = model_output.logits[:, -1, :]
```

## 2. 单步采样的数据流

```text
最后位置 hidden state
        │
        ▼
LM Head: hidden_size → vocab_size
        │
        ▼
原始 logits [batch, vocab_size]
        │
        ├─ repetition / frequency / presence penalty
        ▼
temperature 缩放: logits / T
        │
        ├─ top-k: 只留最高 k 个候选
        ├─ top-p: 只留累计概率达到 p 的最小候选集
        └─ min-p: 过滤低于 max_probability × min_p 的候选
        ▼
FP32 softmax + 重新归一化
        │
        ▼
multinomial 抽样一个 token id
        │
        ├─ EOS → 停止
        └─ 非 EOS → 追加到序列并继续下一轮 decode
```

### 2.1 历史惩罚

假设某个 token 的原始 logit 是 `l_i`：

- `repetition_penalty` 是乘法惩罚。对已出现 token，正 logits 除以惩罚值，
  负 logits 乘以惩罚值。`1.0` 表示关闭；
- `frequency_penalty` 按历史出现次数线性减分；
- `presence_penalty` 只按“是否出现过”减一次分。

三者可以组合，但一般不宜同时设置得很强，否则专有名词、代码标识符等必要重复
也会被破坏。padding 必须通过 `previous_token_mask` 排除。

### 2.2 Temperature

对处理后的 logits 做：

```text
scaled_logits = logits / temperature
```

- `T < 1` 拉大 token 间差距，使输出更确定；
- `T > 1` 拉平分布，使输出更多样；
- 本实现将 `T = 0` 定义成 greedy，直接选择最大 logit。

softmax 采用 FP32。即使模型主体用 FP16/BF16，概率归一化和累计概率也应尽量
避免低精度误差。

### 2.3 Top-k

只保留 logit 最大的 `k` 个 token，其余设为负无穷。它能给候选集设置固定上限，
但不考虑当前分布是尖锐还是平坦。

### 2.4 Top-p（Nucleus Sampling）

先按概率从高到低排序，保留累计概率达到 `p` 所需的最小集合。候选集大小随分布
动态变化。实现时要保留“首次使累计概率超过 p”的边界 token，否则保留的总概率
会小于目标值。

### 2.5 Min-p

设最大 token 概率为 `p_max`，仅保留满足以下条件的 token：

```text
p_i >= min_p * p_max
```

它使用相对阈值：模型很确定时会强力过滤，分布平坦时会保留更多候选。通常把
min-p 作为 top-p 的替代或补充，不建议一开始就把 top-k、top-p、min-p 都设得很严。

### 2.6 Multinomial

过滤后的 logits 再做 softmax，得到总和为 1 的最终分布。`torch.multinomial`
根据这个离散分布抽取一个 token。线上服务通常在 GPU 上完成这一过程，以免把整个
词表的 logits 传回 CPU。

## 3. 可直接使用的 Python 调用

如果已经获得模型输出：

```python
import torch

from op.decoder_sampling.decoder_sampling import SamplingConfig, sample_next_token


config = SamplingConfig(
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    min_p=0.0,                 # 默认关闭，可按模型调成 0.05 左右开始实验
    repetition_penalty=1.05,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

# outputs.logits 通常为 [batch, sequence, vocab_size]。
next_token_logits = outputs.logits[:, -1, :]

# input_ids 可只含生成历史，也可含 prompt + 生成历史；语义由服务统一约定。
generator = torch.Generator(device=next_token_logits.device).manual_seed(2026)
result = sample_next_token(
    logits=next_token_logits,
    previous_token_ids=input_ids,
    config=config,
    generator=generator,
)

next_token_ids = result.token_ids
next_token_probs = result.token_probabilities
input_ids = torch.cat([input_ids, next_token_ids[:, None]], dim=-1)
```

只要一个确定结果时使用 greedy：

```python
greedy_config = SamplingConfig(do_sample=False)
result = sample_next_token(next_token_logits, input_ids, greedy_config)
```

## 4. 参数选择建议

没有适用于所有模型的固定参数，应优先读取模型仓库提供的
`generation_config.json`，然后在自己的验证集上调参。可从以下方向开始：

| 目标 | temperature | top-p | top-k | min-p |
|---|---:|---:|---:|---:|
| 严格可复现、分类或结构化任务 | `0` / greedy | 不生效 | 不生效 | 不生效 |
| 通用对话起点 | `0.7 ~ 0.9` | `0.9 ~ 0.95` | `0` 或 `40 ~ 50` | `0` |
| 更有创造性的文本 | `0.9 ~ 1.1` | `0.95` 左右 | `0` | `0` |
| 尝试 min-p | `0.8 ~ 1.0` | `1.0` | `0` | `0.03 ~ 0.10` |

表格只是实验起点，不是通用最优值。采样器无法修复模型本身不知道的事实，也不能
替代停止条件、坏词过滤、JSON grammar、最大生成长度和安全策略。

## 5. 运行示例与测试

```bash
python op/decoder_sampling/decoder_sampling.py
python -m op.decoder_sampling.test_decoder_sampling
```

测试覆盖 greedy、三类历史惩罚、top-k、top-p 边界、min-p、padding mask、batch、
概率归一化和固定随机种子复现。

## 6. 参考

- [Hugging Face Transformers Generation 文档](https://huggingface.co/docs/transformers/en/main_classes/text_generation)
- [vLLM SamplingParams 文档](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)

两个主流框架均提供 temperature、top-k、top-p、min-p 和重复类惩罚参数；具体默认值
可能随模型配置或框架版本不同，生产环境应以所用版本和模型配置为准。
