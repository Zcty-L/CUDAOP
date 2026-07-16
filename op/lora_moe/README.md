# lora_moe_ops

`lora_moe_ops` 是基于 PyTorch C++/CUDA Extension 的自定义算子模块，全部
CUDA/C++ 源码位于本目录的 `csrc`。

## 编译

```bash
conda activate py311
cd /home/if/Codes/CUDAOP/op/lora_moe
python build.py
```

默认查询当前 GPU 并针对其原生 SM 编译。也可通过参数显式指定目标架构：

```bash
python build.py --arch-list 8.0
```

也可从项目根目录通过 CMake 构建：

```bash
cmake -S . -B build
cmake --build build --target lora_moe_ops
cmake --build build --target cudaop_grouped_gemm
```

## Python 调用

```python
import torch
import lora_moe_ops

x = torch.tensor([2, 0, 1, 2, 2], device="cuda", dtype=torch.int32)
counts = lora_moe_ops.histogram(x, 3)
print(counts)
```

完整运行验证：

```bash
python test_ops.py
python test_standard.py
python test_nonstandard.py
```

从源码目录直接运行前，安装 Grouped GEMM Python 包：

```bash
python -m pip install -e ../grouped_gemm
```

`LoRAMoEStandard` 提供三种 Torch 前向路径：

- `_forward_loop`：逐 expert 循环。
- `_forward_pad`：填充 expert 维度后批量计算。
- `_forward_group`：使用本地 Triton gather/scatter 和
  `grouped_gemm.ops.gmm`。

`LoRAMoENonstandard` 将 gate、up、down 拆成三个独立的 MoE 子层，
同样提供 `loop`、`pad` 和 `group` 三种前向路径。每个子层先聚合
LoRA expert 增量，再把结果传给下一个子层。

## 标准与非标准 LoRA-MoE 对比

### 计算语义

记基础 MLP 的 gate、up、down 权重为 `Wg`、`Wu`、`Wd`，expert `e`
对应的 LoRA 增量为 `ΔWg_e`、`ΔWu_e`、`ΔWd_e`，路由权重为 `p_e`。

标准实现先为每个选中 expert 计算完整的 MLP，再对 expert 输出加权：

```text
h_e = act((Wg + ΔWg_e)x) * ((Wu + ΔWu_e)x)
y   = Σ_e p_e * (Wd + ΔWd_e)h_e
```

非标准实现则在每个子层内部先聚合 LoRA 增量：

```text
g = Wg*x + Σ_e p_e*ΔWg_e*x
u = Wu*x + Σ_e p_e*ΔWu_e*x
h = act(g) * u
y = Wd*h + Σ_e p_e*ΔWd_e*h
```

因此二者不是等价变换：标准实现的非线性激活发生在 expert 聚合之前，
非标准实现发生在聚合之后。非标准实现减少了 expert 级 MLP 中间状态，
但改变了模型表达形式。

### 符号与统计口径

以下使用：

- `N = batch_size * sequence_length`：token 数量；
- `D`：hidden size；
- `I`：intermediate size；
- `E`：expert 数量；
- `K`：每个 token 选中的 expert 数量；
- `R`：LoRA rank；
- `b`：每个参数或激活元素的字节数，例如 BF16 为 2；
- 一次乘加计为一个 MAC，换算 FLOPs 时使用 `1 MAC = 2 FLOPs`。

公式只统计矩阵乘的主导计算量，不包含激活、路由排序、gather/scatter、
加权归并和 kernel launch 等低阶开销。假设 top-k 内 expert 不重复。

### 参数量与持久显存

两种实现包含相同数量的基础 MLP 和 LoRA 参数：

| 参数 | 元素数量 |
|------|---------:|
| 基础 MLP | `P_base = 3DI` |
| 全部 LoRA expert | `P_lora = 3ER(D + I)` |
| 合计 | `P_base + P_lora` |

模型参数显存均为 `b(P_base + P_lora)`，所以结构本身不会改变参数显存。
当前代码的训练行为存在一个重要差异：

- `LoRAMoEStandard` 没有冻结 `original_mlp`，默认会为基础 MLP 和 LoRA
  同时计算参数梯度，梯度显存为 `b(P_base + P_lora)`；
- `LoRAMoENonstandard` 在构造函数中冻结了 `original_mlp`，只保存 LoRA
  参数梯度，梯度显存为 `bP_lora`；
- 如果手动冻结标准实现的基础 MLP，两者的参数量、可训练参数量和梯度显存相同。

优化器显存取决于训练配置。若每个可训练参数保存两个、每个宽度为 `q`
字节的 Adam moment，则 moment 显存为 `2qP_train`；若另存 FP32 master
weight，还需增加 `4P_train` 字节。因此，当前标准实现未冻结基础 MLP
会同时增加反向计算、梯度和优化器状态，不能把这部分差异归因于标准
MoE 结构本身。

### 前向计算量

三种执行路径的主导 MAC 如下：

| 实现与路径 | 基础 MLP MAC | LoRA MAC | 前向总 MAC |
|------------|-------------:|---------:|-----------:|
| 标准 `loop` | `3NKDI` | `3NKR(D + I)` | `3NK[DI + R(D + I)]` |
| 标准 `pad` | `3NDI` | `3NER(D + I)` | `3N[DI + ER(D + I)]` |
| 标准 `group` | `3NDI` | `3NKR(D + I)` | `3N[DI + KR(D + I)]` |
| 非标准 `loop` | `3NDI` | `3NKR(D + I)` | `3N[DI + KR(D + I)]` |
| 非标准 `pad` | `3NDI` | `3NER(D + I)` | `3N[DI + ER(D + I)]` |
| 非标准 `group` | `3NDI` | `3NKR(D + I)` | `3N[DI + KR(D + I)]` |

主要区别为：

- 标准 `loop` 对每个选中 expert 重复执行基础 MLP，基础计算随 `K`
  增长；非标准 `loop` 每个子层只执行一次基础 Linear。
- 两种 `pad` 路径都会计算全部 `E` 个 LoRA expert，即使每个 token
  实际只选择 `K` 个。当 `E >> K` 时会产生明显冗余。
- 两种 `group` 路径的主导矩阵乘计算量相同，均只计算 top-k assignments。
  实际延迟差异来自数据流、融合程度、临时张量和路由开销，而不是 MAC 数量。

### 前向与反向总计算量

对一个 Linear，若权重需要梯度，反向的输入梯度与权重梯度合计约为
前向 MAC 的两倍；若权重冻结但仍需向前一层传播梯度，反向约等于一次
前向 MAC。LoRA 权重始终可训练。

在基础 MLP 冻结的公平 LoRA 训练口径下，前向与反向合计为：

| 实现与路径 | 前向 + 反向 MAC |
|------------|----------------:|
| 标准 `loop` | `6NKDI + 9NKR(D + I)` |
| 标准 `pad` | `6NDI + 9NER(D + I)` |
| 标准 `group` | `6NDI + 9NKR(D + I)` |
| 非标准 `loop` | `6NDI + 9NKR(D + I)` |
| 非标准 `pad` | `6NDI + 9NER(D + I)` |
| 非标准 `group` | `6NDI + 9NKR(D + I)` |

当前标准实现的基础 MLP 默认可训练，因此其表中基础项应由 `6` 改为
`9`；非标准实现保持不变。反向单独的 MAC 可用表中总量减去对应的
前向 MAC。

### 激活与训练峰值显存

精确峰值由 PyTorch autograd 保存策略、Grouped GEMM 后端、内存复用和
allocator 状态共同决定。以下列出源码中决定显存规模的主要张量：

| 路径 | 标准实现 | 非标准实现 |
|------|----------|------------|
| `loop` | 只处理被路由 token，但反向需保留各 expert 的 gate/up、中间激活及 LoRA hidden；规模随 `NK` 增长 | 同样是稀疏计算；激活在路由聚合后计算，但三个子层分别建立 autograd 图 |
| `pad` | 产生多个 `[N, E, I]` expert 张量以及 `[N, E, R]` hidden，训练显存通常最高 | B 投影在 einsum 内直接跨 expert 聚合，主要 expert 张量为 `[N, E, R]`，另保留 `[N, I]`/`[N, D]` dense 激活 |
| `group` | 保存 `[NK, D]` gathered input、多个 `[NK, I]` gate/up 和 expert 中间状态 | 仍有 `[NK, D/I]` 路由临时量，但非线性 MLP 状态聚合为 `[N, I]`；gate/up/down 分别执行 gather/scatter |

结论是：

- 标准结构必须在聚合前保留 expert 级非线性状态，`K` 增大时激活显存
  增长更明显；
- 非标准结构尤其适合 `pad` 路径，避免物化多个 `[N, E, I]` 大张量；
- `group` 同时避免 `pad` 的无效 expert 计算和大规模 padding，通常是两种
  实现更合理的吞吐与显存路径；
- 训练峰值还必须加上参数梯度、优化器状态和 autograd workspace，不能只用
  前向张量大小代替实测峰值。

### 当前测试配置示例

`test_standard.py` 和 `test_nonstandard.py` 使用：

```text
N=2*1507=3014, D=2048, I=2048, E=8, K=2, R=16, dtype=BF16
```

该配置下，基础 MLP 参数为 24 MiB，LoRA 参数为 3 MiB，总参数为
27 MiB。当前实现的参数梯度为：标准 27 MiB，非标准 3 MiB；若冻结
标准实现的基础 MLP，则也为 3 MiB。

忽略低阶算子的理论计算量如下：

| 路径 | 标准前向 | 非标准前向 | 标准前向+反向 | 非标准前向+反向 |
|------|---------:|-----------:|----------------:|------------------:|
| `loop` | 154.070 GFLOPs | 78.220 GFLOPs | 462.210 GFLOPs | 158.811 GFLOPs |
| `pad` | 85.331 GFLOPs | 85.331 GFLOPs | 255.993 GFLOPs | 180.143 GFLOPs |
| `group` | 78.220 GFLOPs | 78.220 GFLOPs | 234.660 GFLOPs | 158.811 GFLOPs |

表中的训练计算量按当前代码计算，即标准基础 MLP 可训练、非标准基础 MLP
冻结。若同样冻结标准基础 MLP，标准与非标准 `group` 的主导训练计算量均为
158.811 GFLOPs；两者的实际性能和峰值显存仍会因计算图不同而存在差异。

作为激活规模参考，单个 BF16 `[N, I]` 张量为 11.773 MiB，`[NK, I]`
为 23.547 MiB，`[N, E, I]` 为 94.188 MiB，而 `[N, E, R]` 仅为
0.736 MiB。标准 `pad` 会同时产生或为反向保存多个 `[N, E, I]` 张量，
因此其峰值显存会远高于单张量数值。

Group 路径通过 `gmm_ops.py` 统一调用以下后端：

- 路由排序、直方图和前缀和：`lora_moe_ops`。
- gather/scatter：`triton_kernels.py`。
- 分组矩阵乘：`cudaop_grouped_gemm`，默认使用 CUTLASS BF16 实现。

Grouped GEMM 的构建、Python 接口及后端对比统一维护在
`op/grouped_gemm`。LoRA-MoE 不再编译重复的 CUTLASS kernel。

## Grouped GEMM 后端

标准和非标准 LoRA-MoE 都可通过构造参数选择 Group 模式后端：

```python
module = LoRAMoEStandard(
    original_mlp=mlp,
    num_experts=8,
    rank=16,
    lora_alpha=4.0,
    gmm_backend="triton",
)
```

`gmm_backend` 支持 `cutlass`、`triton` 和 `cutile`，默认值为
`cutlass`。Triton 与 cuTile 使用融合 LoRA down/up kernel，目前仅支持
rank=16；CUTLASS 仍支持通用 rank。
