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
cmake --build build --target lora_moe_cutlass_grouped_test
./build/lora_moe_cutlass_grouped_test
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
python test_grouped_gemm.py
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
- 分组矩阵乘：`lora_moe_ops` 内置的 CUTLASS BF16 Grouped GEMM。

本地 Grouped GEMM 为 SM120 编译，包含前向、输入梯度和权重梯度三个
CUTLASS kernel 实例，不依赖外部安装的 `grouped_gemm`。
