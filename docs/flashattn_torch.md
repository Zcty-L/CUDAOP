# 常见注意力机制的基础 PyTorch 实现

代码位于 `op/flashattn/torch_impl.py`。实现只使用 `nn.Linear`、矩阵乘法、
`softmax`、reshape 和 transpose，目的是展示注意力的原始计算过程，不使用
`scaled_dot_product_attention` 或 Flash Attention。

所有模块的输入、输出均为 `[batch, sequence, d_model]`：

- `MHA`：Q、K、V 具有相同数量的头；
- `GQA`：多个 Query 头共享一组 K/V 头；
- `MQA`：所有 Query 头共享唯一的一组 K/V；
- `MLA`：先将输入压缩为共享的低秩 KV latent，再分别恢复 K 和 V。
- `LinearAttention`：通过特征映射与矩阵乘法结合律避免生成 `[S, S]`
  注意力矩阵。

基础计算过程为：

```python
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)

scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)
attention = torch.softmax(scores, dim=-1)
output = attention @ v
```

MHA 可以通过 `is_causal=True` 在内部生成下三角 causal mask，使每个 token
只能关注自己和之前的 token：

```python
output = attention(
    x,
    is_causal=True,
)
```

MLA 的 K/V 投影过程为：

```python
kv_latent = kv_down_proj(x)
k = k_up_proj(kv_latent)
v = v_up_proj(kv_latent)
```

Linear Attention 使用 `ELU(x) + 1` 将 Q/K 映射为正值，然后改变矩阵乘法
顺序：

```python
q = F.elu(q) + 1
k = F.elu(k) + 1

kv = k.transpose(-2, -1) @ v
denominator = q @ k.sum(dim=-2).unsqueeze(-1)
output = (q @ kv) / denominator
```

普通注意力需要生成 `[S, S]` 分数矩阵；Linear Attention 先计算
`[D, D]` 的 `K^T @ V`，序列较长且 `D << S` 时可以降低计算量和中间
张量大小。这个基础版本展示非因果、自注意力形式，不处理任意二维 mask。

使用示例：

```python
from flashattn import MHA

attention = MHA(
    d_model=512,
    num_heads=8,
)
output = attention(x)
```

执行验证：

```bash
PYTHONPATH=op python -m flashattn.test_torch_impl
```
