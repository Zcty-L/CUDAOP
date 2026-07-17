# cudaop_grouped_gemm

独立的 BF16 Grouped GEMM Python 包，包含：

- `gmm`：支持自动求导的 CUTLASS Grouped GEMM。
- `torch_gmm`：支持自动求导的 `torch.nn.functional.grouped_mm`。
- `LoraDownGrouped`：Triton LoRA down 前向算子。
- `LoraUpGrouped`：Triton LoRA up 前向算子。
- `LoraFusedDownUpGrouped`：融合 down/up，并额外返回供反向使用的
  `[M, R]` 中间矩阵。
- `triton_fused_lora`：支持自动求导的融合接口。
- `CuTileLoraDownGrouped`、`CuTileLoraUpGrouped`：cuTile 分阶段前向。
- `CuTileLoraFusedDownUpGrouped`：cuTile 融合前向。
- `cutile_fused_lora`：支持三 kernel 反向的 cuTile 融合接口。

Triton 与 cuTile LoRA 实现支持 rank=16/32，并拆分为两个 kernel：

```text
down: [M, K] @ [E, R, K].T -> [M, R]
up:   [M, R] @ [E, R, N]   -> [M, N]
```

down 构造时会将权重预打包为连续的 `[E, K, R]`。up 权重本身必须
采用连续的 `[E, R, N]` 布局，从而让一次矩阵乘完成完整的
rank=16/32 收缩。两个算子会按 `batch_sizes` Tensor 对象及版本号缓存
路由元数据；路由变化时自动重新构建，也可以调用
`clear_metadata_cache()` 主动清除。

状态化 Triton/cuTile 类只提供前向。训练路径可以使用 `gmm`、
`torch_gmm`、`triton_fused_lora` 或 `cutile_fused_lora`；融合反向包含：

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

from cudaop_grouped_gemm import triton_fused_lora

a.requires_grad_(True)
down_weight.requires_grad_(True)
up_weight.requires_grad_(True)
output = triton_fused_lora(a, down_weight, up_weight, sizes)
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
