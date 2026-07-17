# lora_moe_ops

`lora_moe_ops` 是基于 PyTorch C++/CUDA Extension 的自定义算子模块，全部
CUDA/C++ 源码位于本目录的 `csrc`。

## 编译

```bash
conda activate py311
cd /home/if/Codes/CUDAOP/op/lora_moe
python build.py
```

默认针对 `sm_120` 编译。可通过参数指定其他 GPU 架构：

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
`cutlass`。Triton 与 cuTile 支持 rank=16/32，并在 Group 路径中完成
前向和反向分发；CUTLASS 仍支持通用 rank。

标准 LoRA-MoE 的 CUTLASS 路径通过合并 GMM 计算 gate/up；Triton 和
cuTile 路径分别使用各自的两路 fused LoRA kernel，均支持 rank=16/32，
便于在相同模型配置下独立对比三个后端。

## RTX 5070 Ti Laptop 吞吐测试

测试设备与配置：

- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU（SM120，12 GiB）。
- 数据类型：BF16。
- PyTorch：2.12.0+cu132。
- experts=8，rank=16，top_k=2。
- batch_size=1，seq_len=3507。
- hidden_size=2048，intermediate_size=8192。
- 前向 warmup/iterations=10/50。
- 反向 warmup/iterations=5/20。

最初使用 batch_size=4。该配置下 standard 的三种 Grouped GEMM 后端
均通过前向和反向精度验证，但 loop backward 在吞吐测试中超出 12 GiB
显存，因此以下完整端到端对比统一使用 batch_size=1。

### Standard LoRA-MoE

| 方法 | 前向（us） | 反向（us） | 总耗时（us） | tokens/s | 相对 loop | 后端/CUTLASS（tokens/s） |
|---|---:|---:|---:|---:|---:|---:|
| loop | 28675.266 | 58458.053 | 87133.319 | 40248.7 | 1.000x | |
| pad | 38653.164 | 57498.207 | 96151.371 | 36473.7 | 0.906x | |
| group/CUTLASS | 17784.923 | 30433.742 | 48218.665 | 72731.2 | 1.807x | |
| group/Triton | 18372.211 | 30985.094 | 49357.305 | 71053.3 | 1.765x | 0.977x |
| group/cuTile | 18643.066 | 31608.773 | 50251.839 | 69788.5 | 1.734x | 0.960x |

### Nonstandard LoRA-MoE

| 方法 | 前向（us） | 反向（us） | 总耗时（us） | tokens/s | 相对 loop | 后端/CUTLASS（tokens/s） |
|---|---:|---:|---:|---:|---:|---:|
| loop | 33504.404 | 26619.587 | 60123.991 | 58329.5 | 1.000x | |
| pad | 13949.520 | 14744.619 | 28694.139 | 122220.1 | 2.095x | |
| group/CUTLASS | 17382.111 | 19519.107 | 36901.218 | 95037.5 | 1.629x | |
| group/Triton | 17344.379 | 18756.389 | 36100.767 | 97144.7 | 1.665x | 1.022x |
| group/cuTile | 17278.005 | 19042.040 | 36320.045 | 96558.2 | 1.655x | 1.016x |

两个测试的 loop、pad、CUTLASS、Triton 和 cuTile 路径均通过 BF16
前向精度与反向传播验证，并输出 `[SUCCESS]`。
