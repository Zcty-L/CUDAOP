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
