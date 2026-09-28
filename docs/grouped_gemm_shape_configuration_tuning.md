# CUTLASS Grouped GEMM shape 配置扫描

## 结论

K16/K8 kernel 仍有优化空间，而且只修改 CUTLASS 的
`ThreadblockShape`、`WarpShape` 和 `Stages` 就能获得稳定收益，不需要
手写 CUDA kernel：

- down 的输出 N 是 LoRA rank=16/32，适合把 CTA N 从 128 缩到 32。
- bgrad 的输出 M 是 LoRA rank=16/32，适合把 CTA M 从 128 缩到 32。
- up 的 M、N 都较大，`128x128` 方形 tile 仍最好或接近最好；主要收益
  来自把 `Stages` 从 4 调到 3。
- 原始 up/NN 配置的 `255 registers/thread` 是编译后资源用量，不是一个
  可直接设置的 CUTLASS 参数。根因是 `WarpShape=64x64` 带来的大块 FP32
  accumulator；降低 `kStages` 只能明显降低 shared memory，不能稳定降低
  寄存器。
- 把 up 的 `WarpShape` 缩为 `32x64` 后可降到 139～142 registers/thread。
  其中 `TB64x128/W32x64/K8/S3` 的理论 occupancy 从 8.33% 提至 25%，
  H=8192 仅比吞吐最优配置慢约 0.8%～0.9%，是低寄存器折中方案。
- 2026-09-01 在两张卡均空闲的 RTX 4090 上复测，按路径选择 shape 后，
  H=8192/rank=16 的完整 LoRA forward 和 forward+backward 分别加速
  `1.025x` 和 `1.035x`；rank=32 分别加速 `1.034x` 和 `1.026x`。
- 调整传给当前 Grouped GEMM 的 `ThreadblockSwizzle` 类型不会改变 tile
  次序。CUTLASS grouped kernel 直接按 N-fast 方式计算 tile 坐标，若要
  改 swizzle，需要修改 CUTLASS grouped scheduler，不属于单纯 shape 配置。

本次新增代码只实例化 `DefaultGemmGrouped` 的不同模板参数。没有新增
任何自定义 `__global__` kernel。

## 基线定义

这里的“完整 LoRA 基线”不是 Triton 的 `LoraFusedDownUpGrouped`，也不是
只测加速配置，而是同一次进程、同一组输入上的三组 CUTLASS A/B：

| 名称 | down | up | 两个 bgrad | 含义 |
|---|---|---|---|---|
| `baseline K32/K16` | 原始配置 | 原始配置 | 原始配置 | TB/Warp K=32，instruction K=16，stage=4 |
| `uniform K16/K8` | K8 方形配置 | K8 方形配置 | K8 方形配置 | TB/Warp K=16，instruction K=8，stage=4 |
| `per-path tuned` | down 扫描赢家 | up 扫描赢家 | bgrad 扫描赢家 | 按 NN、NT、TN 路径分别选 shape/stage |

完整 forward 包含两个 CUTLASS grouped GEMM：down、up。完整
forward+backward 包含六个：forward 两个、输入梯度两个、down/up 权重
梯度各一个。`per-path tuned` 只是用 Python autograd 组合这些 CUTLASS
调用，没有做 kernel fusion，因此结果可以单独反映 shape 配置收益。

## 为什么需要按路径配置

三条 LoRA 路径的输出形状不同：

```text
down:  [tokens, H] @ [experts, rank, H].T -> [tokens, rank]
up:    [tokens, rank] @ [experts, rank, H] -> [tokens, H]
bgrad: [rows, rank].T @ [rows, H]          -> [rank, H]
```

原始 `128x128` CTA 对 down/rank=16 只有 `16/128=12.5%` 的 N tile
有效，对 rank=32 也只有 25%。bgrad 在 M 方向存在完全对称的浪费。
因此，统一使用一个方形配置会隐藏 K8 的一部分收益；缩窄 down 的 N 和
bgrad 的 M 后，尾 tile 浪费、每 CTA 线程数、寄存器和共享内存都会下降。

up 的输出 N=H，M 也是大量 tokens。缩小 M 会增加 CTA 数和 grouped
problem visitor 的调度开销，所以它没有复现 down/bgrad 的明显收益。

## 扫描范围

通用配置模板位于
`op/grouped_gemm/csrc/tuning_configs.cuh:8-47`，仍然调用项目已有的
`run<ShapeConfig, TransposeA, TransposeB>()`：

| 路径 | 新增实例数 | ThreadblockShape | WarpShape | instruction/stage |
|---|---:|---|---|---|
| up | 17 | `128x128x16/32`、`64x128x16/32`、`128x64x16`、`64x64x16` | `64x64`、`32x64`、`64x32` 或 `32x32` | K8/K16，stage 2/3/4/5 |
| down | 15 | `128x64x16/32`、`64x64x16`、`128x32x16`、`64x32x16` | 对应 `64x32`、`32x32` | K8/K16，stage 2/3/4/5 |
| bgrad | 15 | down 的 M/N 对称配置 | 对应 `32x64`、`32x32` | K8/K16，stage 2/3/4/5 |

具体实例和 Python 名称分别位于：

- up：`op/grouped_gemm/csrc/tuning_up.cu:18-107`
- down：`op/grouped_gemm/csrc/tuning_down.cu:18-97`
- bgrad：`op/grouped_gemm/csrc/tuning_bgrad.cu:18-97`

每条路径还加入原始 K32/K16 和统一 K16/K8 两个现有配置，所以测试时
实际比较 up 19 个、down 17 个、bgrad 17 个，共 53 个可调用项。独立
调优扩展 `cudaop_grouped_gemm._tuning` 用于避免把 47 个实验实例加入
生产扩展。

## 窄 tile 的 CUTLASS 编译约束

最初测试了 down 的：

```text
ThreadblockShape = 128x32x16
WarpShape        = 64x16x16
```

CTA 含 `(128/64) * (32/16) * 32 = 128` 个线程，但 B tile 只有
`32 * 16 / 8 = 64` 次 128-bit 向量访问。CUTLASS 无法保证每个线程
至少一次访问，最终在以下位置失败：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/transform/
pitch_linear_thread_map.h:292-299
```

`PitchLinearWarpRakedThreadMap::Iterations::kCount` 变为 0，并触发：

```cpp
static_assert(Iterations::kCount,
              "Number of iterations must be non-zero");
```

最终窄 N 配置保持 `ThreadblockShape=128x32x16`，把
`WarpShape` 改为 `64x32x16`，使 CTA 只有 64 线程。bgrad 的窄 M 配置
采用对称的 `ThreadblockShape=32x128x16`、`WarpShape=32x64x16`。

CUDAOP 在 `tuning_configs.cuh:19-33` 显式加入了等价预检。当前
alignment 为 8 个 BF16 元素，因此两个输入 tile 都必须满足：

```text
ThreadblockM * ThreadblockK >= threads * 8
ThreadblockN * ThreadblockK >= threads * 8
```

这也说明 `WarpShape` 不能只根据输出 tile 利用率持续缩小；它还决定 CTA
线程数，而线程数必须与全局到共享内存的向量化 thread map 匹配。

## 255 registers/thread 对应什么配置

Nsight Compute 的 `Registers Per Thread` 是编译后每个线程使用的 32-bit
寄存器数，不是 `ThreadblockShape` 或 `kStages` 直接指定的值。这里的
255 对应原始 up/NN kernel：

```text
op/grouped_gemm/csrc/grouped_gemm.cuh:28-34
ThreadblockShape = 128x128x32
WarpShape        = 64x64x32
InstructionShape = 16x8x16
kStages          = 4
```

这些参数在 `grouped_gemm.cuh:52-72` 传入 `DefaultGemmGrouped`。SM89 每
线程最多 255 个寄存器；Nsight Compute 同时显示该 kernel 的寄存器分配
粒度为 256。`cuobjdump --dump-resource-usage` 的结果为：

```text
REG:255 STACK:0 SHARED:0 LOCAL:0
```

因此这个 kernel 确实到达单线程寄存器上限，但当前编译产物没有 local
memory spill。高寄存器仍会限制并发 CTA，不能因为没有 spill 就忽略。

主要来源是每个 warp 的 FP32 accumulator。CUTLASS 在：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/gemm/warp/
mma_tensor_op.h:256-262
mma_tensor_op_tile_iterator.h:3249-3253,3271-3274
```

把 `WarpShape` 作为 accumulator tile，并按下面的数量为每个线程创建
`FragmentC`：

```text
WarpM * WarpN / 32
```

所以 `WarpShape=64x64` 仅 accumulator 就有 128 个 FP32 元素/线程；
`32x64` 降为 64 个，`32x32` 降为 32 个。A/B fragment、iterator 状态和
Grouped GEMM 调度元数据还会继续占用寄存器，最终分别表现为 220、
139～148 和 100 registers/thread 等实际值。

这也回答了三个配置项的作用：

- 只缩小 `ThreadblockShape`、保持相同 `WarpShape`，通常不会缩小每线程
  accumulator，只会改变 warp 数、shared memory 和 CTA 数。
- 缩小 `WarpShape` 的 M 或 N 才能直接降低每线程 accumulator，是降低
  registers/thread 最有效的 shape 参数。
- 降低 `kStages` 主要减少 mainloop 的 shared-memory pipeline；寄存器
  变化由 CUTLASS 选择的 mainloop 实现和编译器活跃区间决定，不保证下降。

## smem、寄存器和 stage

`Stages` 已从硬编码参数改为 shape policy 的 `kStages`，传递位置是
`op/grouped_gemm/csrc/grouped_gemm.cuh:69-72`。它就是在不手写 kernel
的前提下控制 mainloop shared-memory pipeline 深度的直接手段。

`Stages` 的实测对比进一步说明它不是寄存器旋钮。输入为 BF16，GPU 为
RTX 4090/SM89：

| up 配置 | registers/thread | dynamic smem/CTA |
|---|---:|---:|
| K16 `TB128x128x32/W64x64x32/I16/S4` | 255 | 65,552 B |
| K16 `TB128x128x32/W64x64x32/I16/S3` | 255 | 49,168 B |
| K16 `TB128x128x32/W64x64x32/I16/S2` | 255 | 32,784 B |
| K8 `TB128x128x16/W64x64x16/I8/S4` | 220 | 32,784 B |
| K8 `TB128x128x16/W64x64x16/I8/S3` | 220 | 24,592 B |
| K8 `TB128x128x16/W64x64x16/I8/S2` | 230 | 16,400 B |

K16 从 stage 4 降到 2 时，smem 减半但仍为 255 registers/thread。K8 的
stage 2 甚至从 220 上升为 230，因为 CUTLASS 从 `MmaMultistage` 切换到
`MmaPipelined` 后，寄存器活跃区间不同。

新增低寄存器 up 配置的 `LaunchStats + Occupancy` 如下。`allocated regs`
是考虑硬件分配粒度后的每线程数，occupancy 是理论活跃 warp 占 SM89
最大 48 warp 的比例：

| up 配置 | threads | regs | allocated regs | dynamic smem | CTA/SM 限制（reg/smem） | theoretical occupancy |
|---|---:|---:|---:|---:|---:|---:|
| 原始 `128x128/W64x64/I16/S4` | 128 | 255 | 256 | 65,552 B | 2/1 | 8.33% |
| K8 `128x128/W64x64/S3` | 128 | 220 | 224 | 24,592 B | 2/2 | 16.67% |
| K8 `128x128/W32x64/S4` | 256 | 142 | 144 | 32,784 B | 1/1 | 16.67% |
| K8 `64x128/W32x64/S3` | 128 | 139 | 144 | 18,448 B | 3/3 | 25.00% |
| K8 `64x64/W32x32/S3` | 128 | 100 | 104 | 12,304 B | 4/4 | 33.33% |

原始配置虽然用了 255 个寄存器，但它首先被 65.55 KB smem 限制为每 SM
一个 CTA。`128x128/W32x64` 把每线程寄存器降到 142，却因 CTA 增至
256 线程、总寄存器和 smem 限制仍只有一个 CTA，所以 occupancy 与 220
寄存器的 K8 方形配置同为 16.67%。不能只根据 registers/thread 判断实际
并发度。

三条路径的代表数据如下：

| 路径/配置 | threads/CTA | registers/thread | dynamic smem/CTA |
|---|---:|---:|---:|
| 原始 `TB128x128x32/W64x64x32/I16/S4` | 128 | 255 | 65.55 KB |
| up `TB128x128x16/W64x64x16/I8/S3` | 128 | 220 | 24.59 KB |
| down `TB64x32x16/W32x32x16/I8/S4` | 64 | 96 | 12.30 KB |
| bgrad `TB32x64x16/W32x32x16/I8/S4` | 64 | 96 | 12.30 KB |

因此 smem 确实还有优化空间，但不必更改内部 smem layout：缩小 tile 和
选择 stage 已把代表配置的 dynamic smem 从 65.55 KB 降到
12.30～24.59 KB。实测中 stage 并非越少越好：

- up 的 K8 方形 tile 通常以 stage=3 最优；stage=2 隐藏延迟不足，
  stage=5 增加资源占用。
- down/rank=16 的窄 tile 以 stage=4 最稳。
- bgrad 的窄 tile 在 rank=16/32 均以 stage=4 最稳，rank=32 的 stage=3
  与它只差约 0.1%。

## 低寄存器 up 配置性能

定向测试使用 28,830 tokens、8 个 group，每项 10 次预热、100 次计时、
9 个随机顺序 sample。下表为 2026-09-01 两张 GPU 均空闲时在 GPU 1 上
得到的中位数，单位为微秒：

| 配置 | regs/thread | H=2048/r16 | H=8192/r16 | H=8192/r32 |
|---|---:|---:|---:|---:|
| 原始 `128x128/W64x64/I16/S4` | 255 | 145.439 | 543.969 | 548.260 |
| K8 `128x128/W64x64/S3` | 220 | 146.081 | **541.757** | **544.950** |
| K8 `128x128/W32x64/S4` | 142 | 146.452 | 546.324 | 548.605 |
| K8 `128x128/W64x32/S4` | 144 | 146.555 | 547.144 | 548.228 |
| K8 `64x128/W32x64/S3` | 139 | **144.712** | 546.488 | 549.571 |
| K8 `128x64/W64x32/S3` | 148 | 152.812 | 594.382 | 598.412 |
| K8 `64x64/W32x32/S3` | 100 | 155.269 | 618.260 | 622.408 |

选择结论：

- 只追求 H=8192 吞吐时，保留 `WarpShape=64x64` 的 K8/S3 最快，代价是
  220 registers/thread。
- 需要压低寄存器时，`TB64x128/W32x64/K8/S3` 是最均衡配置：139
  registers/thread、25% 理论 occupancy；H=2048 快 0.5%，H=8192
  相对原始基线仅慢 0.2%～0.5%，相对吞吐最优 K8/S3 慢 0.8%～0.9%。
- `TB64x64/W32x32/K8/S3` 虽降到 100 registers/thread 和 33.33%
  occupancy，却因 tile 数增多及 grouped scheduler 开销慢 6.3%～12.0%，
  不应作为默认值。

## 为什么没有继续测试 swizzle

`DefaultGemmGrouped` 虽然接收 `ThreadblockSwizzle` 类型，但当前 grouped
kernel 只保存该类型别名，没有调用 `get_tile_offset()`。实际 tile 坐标在
CUTLASS 源码中直接计算：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/gemm/kernel/
gemm_grouped.h:318-330
```

```cpp
GemmCoord threadblock_offset(
    int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
    int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN,
    0);
```

所以替换 `GemmBatchedIdentityThreadblockSwizzle` 不会改变这条 device-only
grouped 路径的 N-fast 顺序。真正修改顺序需要改 `ProblemVisitor` 或
`GemmGrouped` scheduler，已经超出“只测试 shape 配置”的范围。

## 正确性结果

全部配置均对 rank=16/32 做了 FP32 参考验证。每一项均通过 BF16 容差；
下表是每条路径所有配置中的最大绝对误差：

| rank | down | up | bgrad |
|---:|---:|---:|---:|
| 16 | 0.000953 | 0.000487 | 0.000751 |
| 32 | 0.000976 | 0.000510 | 0.000954 |

完整 LoRA 的 forward、输入梯度、down 权重梯度和 up 权重梯度也全部通过
原始配置对比；本次输入下，统一 K8 和逐路径赢家组合均与基线的 BF16
结果逐元素一致。

## RTX 4090 性能结果

测试配置：BF16 输入/输出、FP32 累加，8 个 group、28830 tokens；快速
扫描为 5 次预热、30 次计时、3 个 sample，中位数前三名再以 10 次预热、
100 次计时、5 个 sample 复测。完整 LoRA 为 10 次预热、50 次计时、
5 个 sample。每个 sample 都会打乱配置执行顺序。

2026-09-01 的无干扰完整扫描结果如下，单位为微秒：

| H/rank | 配置 | forward | speedup | forward+backward | speedup |
|---:|---|---:|---:|---:|---:|
| 2048/16 | baseline K32/K16 | 312.669 | 1.000x | 1622.774 | 1.000x |
| 2048/16 | uniform K16/K8 | 308.429 | 1.014x | 1502.947 | **1.080x** |
| 2048/16 | per-path tuned | 302.456 | **1.034x** | 1506.488 | 1.077x |
| 8192/16 | baseline K32/K16 | 1143.106 | 1.000x | 3387.699 | 1.000x |
| 8192/16 | uniform K16/K8 | 1144.812 | 0.999x | 3371.356 | 1.005x |
| 8192/16 | per-path tuned | 1115.296 | **1.025x** | 3272.806 | **1.035x** |
| 8192/32 | baseline K32/K16 | 1161.318 | 1.000x | 3385.074 | 1.000x |
| 8192/32 | uniform K16/K8 | 1144.136 | 1.015x | 3370.455 | 1.004x |
| 8192/32 | per-path tuned | 1123.080 | **1.034x** | 3299.845 | **1.026x** |

同轮选出的路径配置为：

| H/rank | down | up | bgrad |
|---:|---|---|---|
| 2048/16 | `TB64x32x16/W32x32x16/I8/S4` | `TB64x128x16/W32x64x16/I8/S4` | `TB64x64x16/W32x32x16/I8/S4` |
| 8192/16 | `TB64x64x16/W32x32x16/I8/S4` | `TB128x128x16/W64x64x16/I8/S3` | `TB32x64x16/W32x32x16/I8/S4` |
| 8192/32 | `TB64x32x16/W32x32x16/I8/S4` | `TB128x128x16/W64x64x16/I8/S3` | `TB32x64x16/W32x32x16/I8/S4` |

H=2048 的训练组合中，逐路径赢家比统一 K8 慢约 0.3%，说明 1% 内的
差异仍容易受到时钟和选择噪声影响。H=8192 的完整组合收益更稳定，但在
修改生产默认分派前仍应在真实模型的 group size 分布上重复测量。本轮
开始和结束时两张 RTX 4090 均为 0% 利用率，没有其他计算进程。

## 构建和复现

CMake 入口位于 `CMakeLists.txt:544-565`：

```bash
cmake -S . -B build
cmake --build build --target cudaop_grouped_gemm_tuning
CUDA_VISIBLE_DEVICES=0 \
  cmake --build build --target cudaop_grouped_gemm_config_test
```

测试脚本是 `op/grouped_gemm/test_cutlass_configs.py`。成功标记为：

```text
[SUCCESS] CUTLASS Grouped GEMM 配置扫描通过
```

调优扩展使用独立的 `op/grouped_gemm/build/tuning` 临时目录，因此 CMake
并行构建生产扩展和调优扩展时不会争用同一个 Ninja 文件。
