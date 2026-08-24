# Grouped GEMM 的 K 维 tile 最小值约束

## 调查对象

- CUDAOP 配置：`op/grouped_gemm/csrc/grouped_gemm.cuh:26-29`
- CUTLASS 源码：`/home/lsbing/CUDA/Codes/cutlass`
- CUTLASS 版本：`v4.6.0-1-ga931725f`
- CUTLASS 提交：`a931725f1d43f44ebbcecaa7450910b36fcb6bc0`

当前配置为：

```cpp
using Element = cutlass::bfloat16_t;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
```

## 结论

第 28 行的 `WarpShape::kK` 不能从 32 改成 16，直接约束位于：

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

本算子的底层 BF16 指令 K 为 16，因此：

| `WarpShape::kK` | 计算 | 结果 |
|---:|---:|---|
| 32 | `32 / 16 = 2` | 大于 1 且为偶数，通过 |
| 16 | `16 / 16 = 1` | 不大于 1 且为奇数，两条静态断言均失败 |

所以在当前 `DefaultGemmGrouped + Sm80 + MmaMultistage` 实现中，`WarpShape::kK` 必须至少为 32；结合底层指令的整 tile 要求，合法值应为 32 的整数倍。

第 27 行的 `ThreadblockShape::kK` 不能改成 16 是上述限制的传递结果。CUTLASS 用下面的整数除法计算 CTA 在 K 方向的 warp 分区数：

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

由于 `WarpShape::kK` 的最小合法值已经是 32，若只把 `ThreadblockShape::kK` 改为 16，则 K 向分区数是 `16 / 32 = 0`，进而 `WarpCount::kCount` 和 `kThreads` 都变为 0。模板随后在以下位置发生除零：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/transform/pitch_linear_thread_map.h:276-295
```

此外，epilogue 的 K 分区数也在以下位置由相同的整数除法计算：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/gemm/kernel/default_gemm.h:370-376
```

因此当前 kernel 的关系应满足：

```text
InstructionShape::kK = 16
WarpShape::kK >= 2 * InstructionShape::kK = 32
ThreadblockShape::kK >= WarpShape::kK
ThreadblockShape::kK % WarpShape::kK = 0
```

当前 `32 / 32 = 1`，表示 CTA 在 K 方向使用一个 warp 分区，是最小可用配置。需要注意，当前这条 grouped GEMM 模板路径没有对 `ThreadblockShape::kK >= WarpShape::kK` 提供一条清晰的专用 `static_assert`；非法值通过整数除法产生 0，最终以线程映射除零等次生错误暴露。

## 为什么有这个限制

这不是 SM80 BF16 Tensor Core 指令不支持 K=16。硬件指令及 CUTLASS 封装本身就是 K=16：

```text
/home/lsbing/CUDA/Codes/cutlass/include/cutlass/arch/mma_sm80.h:338-386
```

- `mma_sm80.h:341`：指令 shape 为 `GemmShape<16, 8, 16>`。
- `mma_sm80.h:384-386`：实际 PTX 为 `mma.sync.aligned.m16n8k16...bf16...`。

限制来自 CUTLASS 的软件流水主循环：

1. `default_mma.h:514-561` 为本配置选择并构造 `MmaMultistage`。
2. `mma_multistage.h:91-95` 表明 `MmaMultistage` 继承 `MmaBase`，因此必须满足 `mma_base.h:128-133` 的两条断言。
3. `mma_multistage.h:166-183` 为 A、B 各准备两组 warp fragment，用来重叠共享内存读取和计算。
4. `mma_multistage.h:503-523` 使用 `warp_mma_k % 2` 和 `(warp_mma_k + 1) % 2` 在两组 fragment 间交替。

也就是说，该主循环按至少两个、且为偶数个 K 向 warp MMA 迭代组织 ping-pong 流水。把 warp K 降到一条指令的 K=16 后只剩一次迭代，不符合这套主循环的调度不变量。模板参数中的 `Stages=4` 是 threadblock 级流水 stage 数，不能解除 warp 内部至少两次、偶数次 MMA 的要求。

## 模板实例化链

| 层级 | 源码位置 | 作用 |
|---|---|---|
| CUDAOP 配置 | `op/grouped_gemm/csrc/grouped_gemm.cuh:27-29` | 定义 CTA、warp 和指令 shape |
| Grouped GEMM 默认 kernel | `cutlass/gemm/kernel/default_gemm_grouped.h:213-248` | 将三个 shape 转交给 `DefaultGemm`，再包装为 `GemmGrouped` |
| SM80 GEMM | `cutlass/gemm/kernel/default_gemm.h:351-388` | 选择 SM80 TensorOp 的 `DefaultMma` 并计算 `kPartitionsK` |
| Multistage MMA | `cutlass/gemm/threadblock/default_mma.h:514-561` | 构造 `DefaultMmaCore` 和 `MmaMultistage` |
| Warp 指令策略 | `cutlass/gemm/warp/mma_tensor_op_policy.h:54-58` | `MmaShape` 取底层 `arch::Mma::Shape`，本例 K=16 |
| 主循环硬约束 | `cutlass/gemm/threadblock/mma_base.h:103-133` | 计算 K 向迭代数，并要求至少 2 次且为偶数 |

## 编译验证

使用与项目一致的 CUTLASS include 路径，以 `nvcc -std=c++17 -arch=sm_80` 对最小模板实例化进行验证：

| `ThreadblockShape::kK` | `WarpShape::kK` | 结果 |
|---:|---:|---|
| 32 | 32 | 编译通过 |
| 16 | 16 | `mma_base.h:128` 和 `mma_base.h:132` 静态断言失败 |
| 16 | 32 | `WarpCount` 的 K 分量为 0，`pitch_linear_thread_map.h:295` 除零 |

K=16/K=16 的关键报错为：

```text
mma_base.h(128): error: static assertion failed with
"The pipelined structure requires at least two warp-level GEMM operations."

mma_base.h(132): error: static assertion failed with
"Inner loop iteration must be an even number."
```

## 最终判断

当前第 27、28 行的两个 K=32 都应保留：

- `WarpShape::kK=32` 是 CUTLASS `MmaBase` 软件流水对 BF16 `m16n8k16` 指令的直接最小值。
- `ThreadblockShape::kK=32` 是为了容纳最小 warp K tile，并保证 K 向 warp 分区数非零；它是当前组合下的最小传递值。
- 若必须实现 K tile=16，不能只改这两个 `GemmShape` 参数，需要更换或自行实现允许单次 warp MMA K 迭代的 mainloop，而不是绕过现有静态断言。
