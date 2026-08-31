# cudaop_grouped_gemm

独立的 BF16 Grouped GEMM Python 包，包含：

- `gmm`：支持自动求导的 CUTLASS Grouped GEMM。
- `CutlassLoraFusedDownUpGrouped`：CUTLASS/CuTe LoRA down/up 融合前向，
  同时保存供反向使用的 `[N, 16]` hidden。
- `CutlassLoraFusedDownUp`：完整的状态化训练调用器，前向使用
  1 个 fused kernel，反向使用 1 个 fused agrad 和 2 个 bgrad
  kernel。
- `CutlassLoraBgradGrouped`：固定 rank=16 的 CUTLASS LoRA 权重梯度
  Grouped GEMM。
- `cutlass_fused_lora`：支持 CUTLASS/CuTe 融合前向和三 kernel 反向。
- `torch_gmm`：支持自动求导的 `torch.nn.functional.grouped_mm`。
- `LoraDownGrouped`：Triton LoRA down 前向算子。
- `LoraUpGrouped`：Triton LoRA up 前向算子。
- `LoraFusedDownUpGrouped`：融合 down/up，并额外返回供反向使用的
  `[M, 16]` 中间矩阵。
- `triton_fused_lora`：支持自动求导的融合接口。
- `CuTileLoraDownGrouped`、`CuTileLoraUpGrouped`：cuTile 分阶段前向。
- `CuTileLoraFusedDownUpGrouped`：cuTile 融合前向。
- `cutile_fused_lora`：支持三 kernel 反向的 cuTile 融合接口。

Triton LoRA 实现固定 rank=16，并拆分为两个 kernel：

```text
down: [M, K]  @ [E, 16, K].T -> [M, 16]
up:   [M, 16] @ [E, 16, N]   -> [M, N]
```

down 构造时会将权重预打包为连续的 `[E, K, 16]`。up 权重本身必须
采用连续的 `[E, 16, N]` 布局，从而让一次 `tl.dot` 完成完整的
rank=16 收缩。两个算子会按 `batch_sizes` Tensor 对象及版本号缓存
路由元数据；路由变化时自动重新构建，也可以调用
`clear_metadata_cache()` 主动清除。

状态化 Triton 类只提供前向。训练路径可以使用 `gmm`、
`torch_gmm` 或 `triton_fused_lora`；融合 Triton 反向包含：

```text
fused agrad:
    grad_hidden = grad_output @ up_weight.T
    grad_input  = grad_hidden @ down_weight

bgrad down:
    grad_down_weight = grad_hidden.T @ input

bgrad up:
    grad_up_weight = saved_hidden.T @ grad_output
```

完全分开的反向需要四个 grouped GEMM kernel；Triton 实现将两个
输入梯度融合，因此使用三个 kernel。

cuTile 实现采用相同的数学拆分和权重布局。它以
`(row tile, expert)` 为逻辑网格，通过 `token_offsets` 和
`token_counts` 处理不规则分组，使用 bounds-safe gather/scatter
覆盖空 expert 和尾块。矩阵乘输入为 BF16，`ct.mma` 使用 FP32
累加器。

cuTile 需要 CUDA Toolkit 13.1+、支持的 NVIDIA 驱动以及
`cuda-tile` Python 包。当前测试环境为 `cuda-tile 1.4.0` 和
SM120。

CUTLASS fusion 只支持 BF16 和 rank=16。每个 CTA 负责一个 expert
的 32 行 tile：第一个 GEMM 使用 Tensor Core 和 FP32 累加得到 hidden，
将 BF16 hidden 写回一次供反向使用，并保留一份 SMEM 副本直接送入
第二个 GEMM。因此第二个 GEMM 不会从 GMEM 回读 hidden。构造时会将
`up_weight[E, 16, I]` 预打包为连续的 `[E, I, 16]`；前向支持独立的
输入维度 `D`、输出维度 `I`、空 expert 以及 M/N/K 尾块。

MMA operand 使用无 bank conflict 的 swizzled SMEM layout。down 主循环
采用两级 `cp.async` 双缓冲，up 权重使用 16B `cp.async` 向量搬运。
状态化对象会缓存 down/up packed weights 和路由元数据；权重或
`batch_sizes` 的 Tensor 版本变化时自动刷新，也可以调用
`clear_metadata_cache()` 主动清除路由缓存。

CUTLASS 反向将输入梯度的两级 GEMM 同样融合在一个 kernel 中：先计算
`grad_hidden = grad_output @ up_weight.T`，将 BF16 `grad_hidden` 写回
一次供 down 权重梯度使用，同时从 SMEM 直接计算
`grad_input = grad_hidden @ down_weight`。down/up 两个权重梯度各使用一个
Grouped GEMM，因此完整反向共三个 kernel；同样支持 `D != I`。
`CutlassLoraBgradGrouped` 与 Triton 同名类一样计算每个 expert 的
`lhs_e.T @ rhs_e`。输出维度满足 8 元素对齐时使用针对
`M=16` 的 CUTLASS `16x128x64` Grouped GEMM tile；非对齐尾块则
分派到 rank=16 专用的 CuTe bgrad kernel。两条路径都支持空
expert 和非整块 token 数。

## 构建与测试

```bash
conda activate py311
cmake -S . -B build
cmake --build build --target cudaop_grouped_gemm
cmake --build build --target cudaop_grouped_gemm_test
```

也可以在当前目录原地构建：

```bash
python build.py
python test_grouped_gemm.py
```

## Python 调用

```python
import torch
from cudaop_grouped_gemm import LoraDownGrouped, LoraUpGrouped

sizes = torch.tensor([2, 3], device="cuda")
a = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
down_weight = torch.randn(
    2,
    16,
    256,
    device="cuda",
    dtype=torch.bfloat16,
)
up_weight = torch.randn_like(down_weight)

down = LoraDownGrouped(down_weight)
up = LoraUpGrouped(up_weight)
hidden = down(a, sizes)
output = up(hidden, sizes)

from cudaop_grouped_gemm import LoraFusedDownUpGrouped

fused = LoraFusedDownUpGrouped(down_weight, up_weight)
saved_hidden, fused_output = fused(a, sizes)

from cudaop_grouped_gemm import CutlassLoraFusedDownUpGrouped

cutlass_fused = CutlassLoraFusedDownUpGrouped(
    down_weight,
    up_weight,
)
saved_hidden, fused_output = cutlass_fused(a, sizes)

# 训练调用器只返回最终 output，反向自动使用 3 个 kernel。
from cudaop_grouped_gemm import CutlassLoraFusedDownUp

cutlass_train = CutlassLoraFusedDownUp(
    down_weight,
    up_weight,
)
train_output = cutlass_train(a, sizes)
train_output.sum().backward()

from cudaop_grouped_gemm import CutlassLoraBgradGrouped

cutlass_bgrad = CutlassLoraBgradGrouped(
    num_experts=2,
    hidden_size=256,
)
grad_weight = cutlass_bgrad(saved_hidden, a, sizes)

# 重复训练调用使用状态化入口，复用 packed weights 和路由元数据。
a.requires_grad_(True)
down_weight.requires_grad_(True)
up_weight.requires_grad_(True)
cached_output = cutlass_fused.forward_autograd(a, sizes)
cached_output.sum().backward()

from cudaop_grouped_gemm import triton_fused_lora

a.requires_grad_(True)
down_weight.requires_grad_(True)
up_weight.requires_grad_(True)
output = triton_fused_lora(a, down_weight, up_weight, sizes)
output.sum().backward()

from cudaop_grouped_gemm import cutlass_fused_lora

output = cutlass_fused_lora(a, down_weight, up_weight, sizes)
output.sum().backward()

from cudaop_grouped_gemm import (
    CuTileLoraFusedDownUpGrouped,
    cutile_fused_lora,
)

cutile_fused = CuTileLoraFusedDownUpGrouped(
    down_weight,
    up_weight,
)
saved_hidden, fused_output = cutile_fused(a, sizes)

output = cutile_fused_lora(a, down_weight, up_weight, sizes)
output.sum().backward()
```

`op/lora_moe/gmm_ops.py` 提供了惰性构造入口
`triton_lora_down`、`triton_lora_up`、`triton_lora_fused` 和
`triton_lora_autograd`，以及对应的 `cutile_lora_down`、
`cutile_lora_up`、`cutile_lora_fused` 和
`cutile_lora_autograd`。使用前需先安装本包，或将
`op/grouped_gemm` 加入 `PYTHONPATH`。
