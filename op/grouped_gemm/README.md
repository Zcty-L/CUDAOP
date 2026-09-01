# cudaop_grouped_gemm

独立的 BF16 Grouped GEMM Python 包，包含：

- `gmm`：支持自动求导的 CUTLASS Grouped GEMM。
- `gmm_k16`：采用 CTA/warp K=16 和 `m16n8k8`、支持自动求导的
  CUTLASS Grouped GEMM 对比实现。
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
cmake --build build --target cudaop_grouped_gemm_k_tile_test
cmake --build build --target cudaop_grouped_gemm_config_test
```

也可以在当前目录原地构建：

```bash
python build.py
python test_grouped_gemm.py
```

## CUTLASS shape 配置调优

配置调优使用独立扩展 `cudaop_grouped_gemm._tuning`，用于比较不同的
`ThreadblockShape`、`WarpShape`、instruction K 和 `kStages`。它和生产
扩展 `cudaop_grouped_gemm._C` 分开构建，因此 47 个实验模板实例不会
增加生产扩展的体积或普通构建时间。

当前扫描包含：

- up：17 个实验配置，加原始 K32/K16 和统一 K16/K8，共 19 项。
- down：15 个实验配置，加两个现有配置，共 17 项。
- bgrad：15 个实验配置，加两个现有配置，共 17 项。
- 总计 53 个可调用项，并对 rank=16/32 做正确性验证。

### 使用当前 GPU 构建并测试

从仓库根目录执行：

```bash
conda activate py311
cmake -S . -B build
CUDA_VISIBLE_DEVICES=0 \
  cmake --build build --target cudaop_grouped_gemm_config_test
```

`cudaop_grouped_gemm_config_test` 会依次构建生产扩展、调优扩展，然后运行
全部配置扫描和完整 LoRA forward/forward+backward 测试。构建脚本默认
读取可见 GPU 的 compute capability，例如 RTX 4090 对应 `8.9`。

只构建调优扩展、不运行性能测试：

```bash
CUDA_VISIBLE_DEVICES=0 \
  cmake --build build --target cudaop_grouped_gemm_tuning
```

### 指定 CUDA 架构

在目标 GPU 不可见或需要提前构建时，可以直接指定架构：

```bash
cd op/grouped_gemm
python build_tuning.py --arch-list 8.9
```

也可以通过 `TORCH_CUDA_ARCH_LIST` 同时约束生产和调优扩展：

```bash
TORCH_CUDA_ARCH_LIST="8.9" CUDA_VISIBLE_DEVICES=0 \
  cmake --build build --target cudaop_grouped_gemm_config_test
```

模板实例较多，默认最多使用 4 个编译任务；内存较小的机器可以进一步
限制并发：

```bash
cd op/grouped_gemm
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="8.9" \
  python build_tuning.py
```

### 直接运行调优脚本

也可以绕过 CMake，在算子目录内执行：

```bash
cd op/grouped_gemm
python build.py
python build_tuning.py
CUDA_VISIBLE_DEVICES=0 python test_cutlass_configs.py
```

测试依次输出：

1. GPU、compute capability、数据类型、group/token 数和配置数量。
2. rank=16/32 的 down、up、bgrad FP32 参考误差。
3. H=2048/rank=16、H=8192/rank=16/32 的全配置性能排名。
4. 每条路径前三名复测和完整 LoRA 性能。
5. 最终成功标记：

```text
[SUCCESS] CUTLASS Grouped GEMM 配置扫描通过
```

### 在不同 GPU 上对比

不同 GPU 应分别在设备空闲时运行，并保存完整输出。示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
  cmake --build build --target cudaop_grouped_gemm_config_test \
  2>&1 | tee build/grouped_gemm_config_gpu0.log

CUDA_VISIBLE_DEVICES=1 \
  cmake --build build --target cudaop_grouped_gemm_config_test \
  2>&1 | tee build/grouped_gemm_config_gpu1.log
```

对比时应保持代码版本、输入规模、预热/计时次数和 GPU 工作状态一致。
如果两张卡架构不同，应为每张卡设置对应的 `TORCH_CUDA_ARCH_LIST` 后
重新构建。不要只比较单次最低耗时；测试脚本会打乱执行顺序并使用多个
sample 的中位数。

配置定义、CUTLASS 源码约束、寄存器/shared-memory 数据以及 RTX 4090
结果见
[`docs/grouped_gemm_shape_configuration_tuning.md`](../../docs/grouped_gemm_shape_configuration_tuning.md)。

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
