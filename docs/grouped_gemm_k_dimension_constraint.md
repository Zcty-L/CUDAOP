# Grouped GEMM 的 K 维 tile 最小值约束

## 调查对象

- CUDAOP 双配置：`op/grouped_gemm/csrc/grouped_gemm.cuh:28-42`
- CUTLASS 源码：`/home/lsbing/CUDA/Codes/cutlass`
- CUTLASS 版本：`v4.6.0-1-ga931725f`
- CUTLASS 提交：`a931725f1d43f44ebbcecaa7450910b36fcb6bc0`

当前实现同时保留基线和 K=16 候选配置：

```cpp
struct K32ShapeConfig
{
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
};

struct K16ShapeConfig
{
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
};
```

## 结论

CUTLASS 同时提供以下两条 SM80 BF16 Tensor Core 指令：

| 指令 shape | CUTLASS 特化 | PTX |
|---|---|---|
| `GemmShape<16, 8, 8>` | `mma_sm80.h:76-137` | `mma_sm80.h:117-124` 的 `mma.sync.aligned.m16n8k8...bf16...` |
| `GemmShape<16, 8, 16>` | `mma_sm80.h:338-401` | `mma_sm80.h:384-389` 的 `mma.sync.aligned.m16n8k16...bf16...` |

因此两个 K tile 能否设为 16，取决于 `InstructionShape::kK`：

| `ThreadblockShape::kK` | `WarpShape::kK` | `InstructionShape::kK` | 结果 |
|---:|---:|---:|---|
| 32 | 32 | 16 | 默认基线，可用 |
| 16 | 16 | 16 | 不可用，warp K 内只有一次 MMA |
| 16 | 16 | 8 | 已实现候选，可用，warp K 内有两次 MMA |

可行的 K=16 配置为：

```cpp
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
```

## Warp K 的直接约束

直接约束位于：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/gemm/threadblock/mma_base.h:116-133
```

`MmaBase` 计算一个 warp tile 在 K 方向包含多少次底层 Tensor Core MMA：

```cpp
static int const kWarpGemmIterations =
    (WarpGemm::kK / Operator::Policy::MmaShape::kK);

static_assert(kWarpGemmIterations > 1,
              "The pipelined structure requires at least two warp-level "
              "GEMM operations.");

static_assert((kWarpGemmIterations % 2) == 0,
              "Inner loop iteration must be an even number.");
```

约束可以写成：

```text
kWarpGemmIterations = WarpShape::kK / InstructionShape::kK
kWarpGemmIterations > 1
kWarpGemmIterations % 2 == 0
```

两个 BF16 指令 shape 分别得到：

| `WarpShape::kK` | `InstructionShape::kK` | 计算 | 结果 |
|---:|---:|---:|---|
| 32 | 16 | `32 / 16 = 2` | 通过 |
| 16 | 16 | `16 / 16 = 1` | 两条静态断言均失败 |
| 16 | 8 | `16 / 8 = 2` | 通过 |

所以 `WarpShape::kK=32` 只是选择 `m16n8k16` 时的最小值，不是 BF16 grouped GEMM 的绝对最小值。选择 CUTLASS 已实现的 `m16n8k8` 后，`WarpShape::kK` 可以降到 16。

## Threadblock K 的传递约束

CUTLASS 用下面的整数除法计算 CTA 在 K 方向的 warp 分区数：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h
```

- A ColumnMajor、B RowMajor：`1267-1281`
- A RowMajor、B ColumnMajor：`1408-1422`
- A ColumnMajor、B ColumnMajor：`1548-1562`
- A RowMajor、B RowMajor：`1687-1701`

四种布局均包含等价逻辑：

```cpp
using WarpCount = GemmShape<Shape::kM / WarpShape::kM,
                            Shape::kN / WarpShape::kN,
                            Shape::kK / WarpShape::kK>;

static int const kThreads = WarpCount::kCount * kWarpSize;
```

如果只把 `ThreadblockShape::kK` 改成 16，却保留 `WarpShape::kK=32`，K 向分区数会变成 `16 / 32 = 0`，进而 `WarpCount::kCount` 和 `kThreads` 都变为 0。模板随后在以下位置发生除零：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/transform/pitch_linear_thread_map.h:276-295
```

此外，epilogue 的 K 分区数也在以下位置由相同的整数除法计算：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/gemm/kernel/default_gemm.h:370-376
```

若同步采用 `ThreadblockShape::kK=16` 和 `WarpShape::kK=16`，K 向分区数为 `16 / 16 = 1`，这部分约束可以满足。一般关系为：

```text
WarpShape::kK >= 2 * InstructionShape::kK
WarpShape::kK / InstructionShape::kK 必须为偶数
ThreadblockShape::kK >= WarpShape::kK
ThreadblockShape::kK % WarpShape::kK = 0
```

当前 `32 / 32 = 1` 和建议配置的 `16 / 16 = 1` 都表示 CTA 在 K 方向使用一个 warp 分区。需要注意，当前这条 grouped GEMM 模板路径没有对 `ThreadblockShape::kK >= WarpShape::kK` 提供一条清晰的专用 `static_assert`；非法值通过整数除法产生 0，最终以线程映射除零等次生错误暴露。

## 为什么有这个限制

CUTLASS 对两种 BF16 指令都提供了完整特化：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/arch/mma_sm80.h:76-137
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/arch/mma_sm80.h:338-401
```

- `mma_sm80.h:79`：BF16 指令 shape 为 `GemmShape<16, 8, 8>`。
- `mma_sm80.h:117-124`：PTX 为 `mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32`。
- `mma_sm80.h:341`：BF16 指令 shape 为 `GemmShape<16, 8, 16>`。
- `mma_sm80.h:384-389`：PTX 为 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`。

真正的限制来自 CUTLASS 软件流水主循环要求一个 warp K tile 至少包含两条底层 MMA 指令：

1. `default_mma.h:514-561` 为本配置选择并构造 `MmaMultistage`。
2. `mma_multistage.h:91-95` 表明 `MmaMultistage` 继承 `MmaBase`，因此必须满足 `mma_base.h:128-133` 的两条断言。
3. `mma_multistage.h:166-183` 为 A、B 各准备两组 warp fragment，用来重叠共享内存读取和计算。
4. `mma_multistage.h:503-523` 使用 `warp_mma_k % 2` 和 `(warp_mma_k + 1) % 2` 在两组 fragment 间交替。

也就是说，该主循环按至少两个、且为偶数个 K 向 warp MMA 迭代组织 ping-pong 流水：

- `Warp K=16 + Instruction K=16` 只有一次迭代，不符合调度不变量。
- `Warp K=16 + Instruction K=8` 有两次迭代，符合调度不变量。

模板参数中的 `Stages=4` 是 threadblock 级流水 stage 数，不能解除 warp 内部至少两次、偶数次 MMA 的要求。

## 模板实例化链

| 层级 | 源码位置 | 作用 |
|---|---|---|
| CUDAOP 配置 | `op/grouped_gemm/csrc/grouped_gemm.cuh:28-42` | 定义 K32/K16 基线和 K16/K8 候选 shape |
| Grouped GEMM 默认 kernel | `cutlass/gemm/kernel/default_gemm_grouped.h:213-248` | 将三个 shape 转交给 `DefaultGemm`，再包装为 `GemmGrouped` |
| SM80 GEMM | `cutlass/gemm/kernel/default_gemm.h:351-388` | 选择 SM80 TensorOp 的 `DefaultMma` 并计算 `kPartitionsK` |
| Multistage MMA | `cutlass/gemm/threadblock/default_mma.h:514-561` | 构造 `DefaultMmaCore` 和 `MmaMultistage` |
| Warp 指令策略 | `cutlass/gemm/warp/mma_tensor_op_policy.h:54-58` | `MmaShape` 取底层 `arch::Mma::Shape`，可选择 K=8 或 K=16 |
| 主循环硬约束 | `cutlass/gemm/threadblock/mma_base.h:103-133` | 计算 K 向迭代数，并要求至少 2 次且为偶数 |

## CUDAOP 实现位置

K16/K8 已作为可独立调用、支持自动求导的完整实现加入项目，同时保留
原 K32/K16 路径用于 A/B 和默认行为：

| 层级 | 源码位置 | 实现内容 |
|---|---|---|
| shape policy | `op/grouped_gemm/csrc/grouped_gemm.cuh:28-42` | `K32ShapeConfig` 与 `K16ShapeConfig` |
| 共用 kernel 模板 | `op/grouped_gemm/csrc/grouped_gemm.cuh:50-74` | 将 shape policy 传给 `DefaultGemmGrouped` |
| 三布局分派 | `op/grouped_gemm/csrc/grouped_gemm.cuh:344-394` | 共用 NN、NT、TN 分派，分别导出 K32/K16 与 K16/K8 实现 |
| C++/Python 绑定 | `op/grouped_gemm/csrc/ops.cu:10-17` | 导出 `_C.grouped_gemm` 与 `_C.grouped_gemm_k16` |
| 自动求导接口 | `op/grouped_gemm/cudaop_grouped_gemm/ops.py:27-103` | 前向和反向都按同一 shape policy 执行，公开 `gmm_k16` |
| LoRA 接口 | `op/grouped_gemm/cudaop_grouped_gemm/ops.py:129-168` | 基线 `lora_gmm` 与候选 `lora_gmm_k16` |
| 对比测试 | `op/grouped_gemm/test_cutlass_k_tile.py:58-552` | 三布局精度、LoRA 自动求导及分阶段性能 A/B |
| CMake 入口 | `CMakeLists.txt:532-542` | `cudaop_grouped_gemm_k_tile_test` |

调用关系为：

```text
gmm / lora_gmm
  -> _C.grouped_gemm
  -> K32ShapeConfig: TB K=32, warp K=32, instruction K=16

gmm_k16 / lora_gmm_k16
  -> _C.grouped_gemm_k16
  -> K16ShapeConfig: TB K=16, warp K=16, instruction K=8
```

## 编译验证

使用与项目一致的 CUTLASS include 路径，强制实例化完整 `cutlass::gemm::device::GemmGrouped` kernel 及其 `run()` 路径。K=16 可行配置对 NN、NT、TN、TT 四种 A/B 布局均进行了验证：

| `ThreadblockShape::kK` | `WarpShape::kK` | `InstructionShape::kK` | 架构 | 结果 |
|---:|---:|---:|---|---|
| 32 | 32 | 16 | SM80 | 编译通过 |
| 16 | 16 | 16 | SM80 | `mma_base.h:128` 和 `mma_base.h:132` 静态断言失败 |
| 16 | 32 | 16 | SM80 | `WarpCount` 的 K 分量为 0，`pitch_linear_thread_map.h:295` 除零 |
| 16 | 16 | 8 | SM80 | 四种布局完整 kernel 编译通过 |
| 16 | 16 | 8 | SM120 | 四种布局完整 kernel 编译通过 |
| 16 | 16 | 8 | SM89 | CUDAOP 扩展的 NN、NT、TN 三条受支持路径构建并运行通过 |

对 SM89 扩展执行 `cuobjdump --dump-sass`，已链接二进制同时包含：

```text
192 HMMA.16816.F32.BF16
192 HMMA.1688.F32.BF16
```

这验证了基线路径实际生成 K16 MMA，新增路径实际生成 K8 MMA，而不是
只有 C++ shape 名称发生变化。

`InstructionShape::kK=16` 时，K=16/K=16 的关键报错为：

```text
mma_base.h(128): error: static assertion failed with
"The pipelined structure requires at least two warp-level GEMM operations."

mma_base.h(132): error: static assertion failed with
"Inner loop iteration must be an even number."
```

## 正确性与性能实测

测试命令：

```bash
cmake --build build --target cudaop_grouped_gemm_k_tile_test
```

测试环境与方法：

- GPU：NVIDIA GeForce RTX 4090，SM89。
- 数据类型：BF16 输入和输出、FP32 累加。
- 性能 shape：8 个 group、28830 tokens，H/R 为 2048/16、8192/16、
  8192/32。
- 每项预热 10 次、计时 50 次、取 5 个 sample 的中位数；每个 sample
  交替基线和候选的执行顺序。
- `K16 speedup = K32/K16 latency / K16/K8 latency`，大于 1 表示
  K16/K8 更快。

独立目标 `cudaop_grouped_gemm_k_tile_test` 已完整通过。项目总目标
`cudaop_grouped_gemm_test` 也已尝试：CUTLASS 扩展重建成功且
rank=16 的 CUTLASS/Triton 精度项通过，随后因当前 cuTile 编译器不支持
RTX 4090 的 `sm_89` 而停止，错误为
`tileiras: Cannot find option named 'sm_89'`；该失败与本次 CUTLASS
K16/K8 实现无关。

两种配置为：

```text
基线：Threadblock K=32, Warp K=32, Instruction K=16
候选：Threadblock K=16, Warp K=16, Instruction K=8
```

### 正确性

NN、NT、TN 三条项目支持的布局均通过 FP32 参考值验证；两种 CUTLASS
配置在这些用例中的 BF16 输出逐元素一致：

| 用例 | K32/K16 最大绝对误差 | K16/K8 最大绝对误差 | 两配置最大差值 |
|---|---:|---:|---:|
| NN，K=16 | 0.000479 | 0.000479 | 0 |
| NN，K=32 | 0.000484 | 0.000484 | 0 |
| NT，K=40 | 0.000605 | 0.000605 | 0 |
| TN，K=每组 rows | 0.000938 | 0.000938 | 0 |

LoRA rank=16 的前向、输入梯度、down 权重梯度和 up 权重梯度也都通过
FP32 参考值验证；两种配置的四项最大绝对误差相同，整体最大值为
0.001005。

### 性能

以下为 2026-08-27 的一次完整 CMake 测试结果，单位为微秒：

| H/R | 阶段 | 逻辑 K | K32/K16 | K16/K8 | K16 speedup |
|---:|---|---:|---:|---:|---:|
| 2048/16 | down | 2048 | 175.862 | 166.400 | 1.057x |
| 2048/16 | up | 16 | 156.918 | 166.013 | 0.945x |
| 2048/16 | bgrad | rows | 190.484 | 182.170 | 1.046x |
| 2048/16 | LoRA forward | mixed | 309.207 | 305.398 | 1.012x |
| 2048/16 | LoRA forward+backward | mixed | 1033.359 | 1049.784 | 0.984x |
| 8192/16 | down | 8192 | 686.510 | 630.497 | 1.089x |
| 8192/16 | up | 16 | 634.696 | 640.041 | 0.992x |
| 8192/16 | bgrad | rows | 638.013 | 604.037 | 1.056x |
| 8192/16 | LoRA forward | mixed | 1314.652 | 1298.248 | 1.013x |
| 8192/16 | LoRA forward+backward | mixed | 3900.948 | 3870.537 | 1.008x |
| 8192/32 | down | 8192 | 659.231 | 659.169 | 1.000x |
| 8192/32 | up | 32 | 617.103 | 638.034 | 0.967x |
| 8192/32 | bgrad | rows | 623.432 | 619.889 | 1.006x |
| 8192/32 | LoRA forward | mixed | 1330.012 | 1329.828 | 1.000x |
| 8192/32 | LoRA forward+backward | mixed | 3912.151 | 3895.542 | 1.004x |

可编译不等于稳定更快。相较于 `m16n8k16`，`m16n8k8` 每条指令只覆盖
一半 K；处理相同问题 K 时需要更多 MMA 指令和主循环迭代。另一方面，
threadblock K 从 32 降到 16 后，每个流水 stage 的共享内存占用和 K
尾块浪费都会降低。因此实际结果取决于布局、逻辑 K、shape 和整条训练
路径，不能仅依据逻辑 K=16 推断候选一定更快。

## 最终判断

CUTLASS 已提供 BF16 `m16n8k8`，所以原实现中的两个 K=32 不是不可降低的绝对限制：

- 保持 `InstructionShape::kK=16` 时，`WarpShape::kK` 的最小值是 32，`ThreadblockShape::kK` 也不能单独降到 16。
- 将 `InstructionShape` 改为 `GemmShape<16, 8, 8>` 后，可以同步把 `WarpShape::kK` 和 `ThreadblockShape::kK` 降到 16。
- K16/K8 已实现为 `gmm_k16` 和 `lora_gmm_k16`，前向和反向均使用
  同一配置；NN、NT、TN 精度及独立 K tile CMake 目标完整通过。
- 本机实测中，K16/K8 的完整 LoRA 前向为持平至快 1.3%，H=8192 的
  forward+backward 快 0.4%～0.8%，但 H=2048/rank=16 的
  forward+backward 慢 1.6%。
- 由于收益较小且不是所有阶段和 shape 都占优，`gmm` 继续使用 K32/K16
  基线；需要对比或明确选择候选时使用 `gmm_k16`。在改变默认值前应在
  目标 GPU 和真实 shape 上重新测量。
