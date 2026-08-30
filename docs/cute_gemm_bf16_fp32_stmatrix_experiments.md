# CuTe BF16 GEMM：LDGSTS 与 stmatrix epilogue 实验记录

> 状态说明：当前 `cute_gemm_bf16_fp32` 已统一为 BF16 输入、FP32
> accumulate、FP32 output，并使用直接 `STG.64` epilogue。本文原有的
> stmatrix 章节针对的是“BF16 output”，现保留为历史实验，不能再与当前
> FP32-output kernel 直接比较。

## 当前默认：FP32 output + 直接 STG.64

### 为什么不再使用 stmatrix

`mma.sync...f32.bf16.bf16.f32` 的 accumulator 是 FP32。原 stmatrix
实验在主循环之后先将 accumulator 转为 BF16，再用 `stmatrix.b16` 写 SMEM，
因此输出实际是 BF16。为使它与 `cute_gemm_fp16_fp32` 只在输入类型上不同，
当前 C、cuBLAS reference 和精度检查均已改为 FP32。

`stmatrix` 没有 `b32` 形式；不过单个 m16n8 MMA 的每个 lane 天然持有两组
相邻 FP32。CuTe 分析得到 accumulator 与 C partition 的
`max_common_vector=2`。由于 C 基址、leading dimension 和 CTA offset 均满足
8B 对齐，使用 `copy_aligned(accumulator, thread_global_c)` 可以直接生成
64-bit global store，无需 R2S/LDS/barrier。

### SASS 与单 CTA 指标

配置：`128x128x128`。

| 指标 | 普通 `copy` 基线 | `copy_aligned` |
|---|---:|---:|
| registers/thread | 244 | 244 |
| spill stores / loads | 0 / 0 | 0 / 0 |
| 静态 global store | 128 × `STG.E` | 64 × `STG.E.64` |
| global-store warp requests | 512 | 256 |
| global-store sectors | 4096 | 2048 |
| 理想 global-store sectors | 2048 | 2048 |
| epilogue shared-memory traffic | 无 | 无 |

SourceCounters：

- 普通 `copy`：所有 global 访问共 8192 sectors，理想 6144，excessive 2048。
- `copy_aligned`：所有 global 访问共 6144 sectors，理想 6144，excessive 0。
- 两者 shared wavefront 都是 2176，理想值也是 2176；STG.64 没有引入
  shared-memory epilogue。

### `8192³` A/B

保存同语义的标量基线与 STG.64 二进制，交替顺序运行五轮；每个进程使用
2 次 warmup、5 次 iteration，所有正确性检查均为 0 mismatch。

| 五轮中位数 | Identity ms | Identity TFLOP/s | BlockSwizzle8 ms | BlockSwizzle8 TFLOP/s | cuBLAS ms |
|---|---:|---:|---:|---:|---:|
| 标量 `STG.E` | 30.8263 | 35.6679 | 29.5273 | 37.2372 | 27.5248 |
| 配对 `STG.E.64` | 30.3648 | 36.2101 | 29.8548 | 36.8287 | 27.5874 |

- 绝对中位数下，Identity 快 1.50%，BlockSwizzle8 慢 1.11%。
- 不同进程存在 GPU 时钟漂移；按每轮同进程 cuBLAS 归一化后，Identity
  中位改善 1.94%，BlockSwizzle8 中位改善 1.27%。
- 因此性能收益约为 1%～2%，但比性能数字更确定的结论是：STG.64 将
  FP32 output sectors 从 4096 降到理论最小值 2048，且没有增加 SMEM、
  barrier、寄存器或 spill。

### 回归

- CMake/CTest 通过。
- Compute Sanitizer memcheck：0 errors。
- Compute Sanitizer racecheck：0 errors，0 warnings。
- `8192³` Identity、BlockSwizzle8 相对 cuBLAS 均为 0 mismatch，二者
  bitwise mismatch 为 0。

## 以下为历史 BF16-output / stmatrix 实验

> 工作记录：用于保存此前 BF16-output 语义下的基线、单变量实验和验证结果。

## 环境

- 分支：`feature/linear`
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，compute capability 12.0
- CUDA：13.2，NVCC 13.2.78
- Nsight Compute：2026.1.1
- 目标：`cute_gemm_bf16_fp32`
- GEMM：BF16 × BF16，FP32 accumulate，BF16 output

## 待办

- [x] 区分 `LDGSTS.E...128` 的单指令地址 bank conflict 与多 warp 异步写仲裁，解释为什么已有 `Swizzle<3,3,3>` 时 aggregate counter 仍非零。
- [x] 将当前逐元素 `STG.E.U16` epilogue 改为基于 `stmatrix` 的 R2S，再进行合并的 SMEM → GMEM 写回。
- [x] 验证正确性、SASS、shared bank conflict、global-store sectors/request 和 `8192³` 性能。

## 实验 0：当前实现基线

### 配置

- 问题规模：`M=N=K=8192`
- CTA tile：`128x128x64`
- block：128 threads
- pipeline：3 stages
- G2S：16B `cp.async`
- SMEM：`Swizzle<3,3,3>`
- S2R：`ldmatrix.x4`
- benchmark：2 warmup，5 iterations，rotating order，取程序报告的中位数

### 正确性

三轮均满足：

- Identity vs cuBLAS：0 mismatch
- BlockSwizzle8 vs cuBLAS：0 mismatch
- BlockSwizzle8 vs Identity：0 bitwise mismatch

### 性能

| 轮次 | Identity ms | Identity TFLOP/s | BlockSwizzle8 ms | BlockSwizzle8 TFLOP/s | cuBLAS ms | cuBLAS TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30.9587 | 35.5155 | 29.0782 | 37.8123 | 26.9169 | 40.8484 |
| 2 | 29.3494 | 37.4629 | 28.1777 | 39.0207 | 25.8560 | 42.5245 |
| 3 | 32.6220 | 33.7046 | 31.5174 | 34.8858 | 27.4319 | 40.0815 |
| 三轮中位数 | 30.9587 | 35.5155 | 29.0782 | 37.8123 | 26.9169 | 40.8484 |

三轮中位数下，BlockSwizzle8 为 cuBLAS 的 92.57%，比 Identity 快 1.065 倍。

### 基线指令与访存指标

采用 `128x128x128` 单 CTA 配置隔离单个 CTA 的访存行为：

- S2R：288 条动态 warp `LDSM.16.M88.4`；理想与实际均为 1152 shared wavefront，结构性 bank conflict 为 0。
- G2S：256 条动态 warp `LDGSTS.E...128`；理想为 1024 shared wavefront。
- aggregate 硬件计数连续三次得到 1258、1298、1306 个 LDGSTS wavefront，对应 234、273、282 个 bank-conflict 计数。
- SourceCounters 的逐指令地址分析得到 `L1 Wavefronts Shared = Ideal = 2176`、excessive 为 0，说明单条指令的线程地址映射没有固定的额外 wavefront；非零 aggregate 计数需继续验证是否来自并发异步写仲裁。
- epilogue SASS：128 条静态 `STG.E.U16`，没有 `STSM/stmatrix`。
- 每 CTA 执行 512 条 global-store warp 指令；每请求 8 个 32B sector，共 4096 sectors，理想为 1024 sectors，放大 4 倍。

## 实验 1：直接使用 row-major epilogue SMEM

实现：

1. FP32 accumulator 完成 K 维归约后转换为 BF16 fragment。
2. 使用 `SM90_U32x4_STSM_N` 将 fragment 写入复用的 A pipeline SMEM。
3. 同步后使用每线程 16B 的 `LDS.128 + STG.128` 写回 C。

结果与问题：

- 普通 `sm_120` 不会启用 CuTe 的 `CUTE_ARCH_STSM_SM90_ENABLED`，会生成 `BPT.TRAP`；目标必须使用 family-specific `sm_120f`。
- 改为 `sm_120f` 后正确生成 `STSM.16.M88.4`，小规模精度为 0 mismatch。
- row-major `128x128` SMEM 使 64 条动态 STSM 产生 2048 shared-store wavefront，其中 1792 个为 bank conflict；理想值只有 256。

结论：`stmatrix` 只负责按矩阵 fragment 语义写 SMEM，不会自动改变目标 SMEM 的物理 bank layout；epilogue SMEM 同样需要 swizzle。

## 实验 2：swizzled SMEM + stmatrix + 16B S2G

### 实现

- epilogue C SMEM 使用与主循环一致的 `Swizzle<3,3,3>` atom。
- `cp_async_wait<0>() + __syncthreads()` 后复用 A 的三 stage SMEM，动态共享内存大小不增加。
- FP32 accumulator 转为 BF16 fragment。
- `stmatrix.sync.aligned.x4.m8n8.shared.b16` 完成寄存器到 swizzled SMEM 的矩阵化写入。
- CTA 同步后，每线程通过 16B `LDS.128 + STG.128` 写回连续的 row-major GMEM。
- `cute_gemm_bf16_fp32` CMake 目标使用 `sm_120f`，以启用 Blackwell 上的架构特定 `stmatrix`。
- 运行时要求 compute capability 9.0+；Ampere 不支持本 epilogue 使用的 `stmatrix`。

### SASS 与资源

| 项目 | 基线 | 实验 2 |
|---|---:|---:|
| registers/thread | 244 | 248 |
| spill stores / loads | 0 / 0 | 0 / 0 |
| 静态 global-store 指令 | 128 × `STG.E.U16` | 16 × `STG.E.128` |
| 静态 R2S 指令 | 无 | 16 × `STSM.16.M88.4` |
| 静态 S2G shared-load 指令 | 无 | 16 × `LDS.128` |

### 单 CTA 访存指标

配置：`128x128x128`。

| 指标 | 基线 | 实验 2 |
|---|---:|---:|
| global-store warp requests | 512 | 64 |
| global-store sectors | 4096 | 1024 |
| 理想 global-store sectors | 1024 | 1024 |
| excessive global sectors | 3072 | 0 |
| STSM warp instructions | 0 | 64 |
| STSM wavefronts | 0 | 256，理想 256 |
| STSM bank conflicts | 0 | 0 |
| LDSM + epilogue LDS bank conflicts | 0 | 0 |

实验 2 的 sectors/request 为 16，高于基线的 8，但这是因为每条 warp store 的有效载荷从 64B 增加到 512B；总请求和总 sectors 才是可直接比较的指标。

最终 SourceCounters：

- `L1 Wavefronts Shared = 2688`
- `L1 Wavefronts Shared Ideal = 2688`
- `L1 Wavefronts Shared Excessive = 0`
- `L2 Theoretical Sectors Global = 5120`
- `L2 Theoretical Sectors Global Ideal = 5120`
- `L2 Theoretical Sectors Global Excessive = 0`

### 为什么 G2S 仍有 aggregate bank-conflict 计数

`Swizzle<3,3,3>` 解决的是同一条 warp 访存指令内部的 lane → bank 映射：

- 每个 lane 的 `cp.async` 有效载荷为 16B，一个 warp 共写 512B。
- 32 个 4B banks 每个周期合计服务 128B，因此一条 `LDGSTS.E...128` 理想情况下本来就需要 4 个 wavefront。
- swizzle 让每个 wavefront 内的地址均匀分布到 banks，不能把 512B 压成 1 个 wavefront。
- 多个 warp 和多个异步 stage 同时在途时，每条指令都会覆盖全部 banks，因此它们仍需在 shared-memory pipe 上仲裁；swizzle 无法让不同 warp 使用互不相交的 banks。

证据：

- SourceCounters 对逐指令地址的分析始终得到 `Shared Wavefronts = Ideal`、excessive 为 0。
- aggregate 硬件 counter 随运行调度变化，不是固定值。
- `K=8192`、单 CTA 下，每次均执行 8320 条 LDGSTS，理想 wavefront 为 33280；三次实际值分别为 37515、37845、38260，对应的额外仲裁计数为 4235、4565、4980。
- epilogue 从标量写改为 stmatrix 后，LDGSTS 的 aggregate 波动仍存在，说明它与 C epilogue 和固定 swizzle 地址冲突无关。

因此这里不能通过继续更换 XOR 参数来消除 aggregate 计数。若要减少它，需要改变并发模型，例如 warp-specialized producer 或 TMA；这会改变主循环调度，必须作为独立性能实验评估。

### `8192³` 性能

三轮均为 0 mismatch。每轮使用 2 warmup、5 iterations、rotating order。

| 轮次 | Identity ms | Identity TFLOP/s | BlockSwizzle8 ms | BlockSwizzle8 TFLOP/s | cuBLAS ms | cuBLAS TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.4627 | 37.3187 | 28.3873 | 38.7326 | 26.6461 | 41.2635 |
| 2 | 31.0272 | 35.4370 | 29.9392 | 36.7248 | 28.9140 | 38.0270 |
| 3 | 31.3307 | 35.0938 | 29.0808 | 37.8088 | 29.0798 | 37.8102 |
| 三轮中位数 | 31.0272 | 35.4370 | 29.0808 | 37.8088 | 28.9140 | 38.0270 |

BlockSwizzle8 的绝对时间由基线 29.0782 ms 变为 29.0808 ms，基本性能中性。该规模 K 很大，epilogue 成本被主循环摊薄；测试期间 cuBLAS 中位数由 26.9169 ms 波动到 28.9140 ms，因此不把不同阶段的相对 cuBLAS 百分比解释为 kernel 加速。

### Occupancy 与 cuBLAS 资源对比

在 `8192x8192x8192` 下使用 Nsight Compute 的 `LaunchStats` 和
`Occupancy` section 分别采集当前 CuTe kernel 与同一次
`cublasGemmEx` 实际选择的 kernel：

| 项目 | CuTe BlockSwizzle8 | cuBLAS |
|---|---:|---:|
| 实际 kernel tile | `128x128x64`，3 stages | `64x64x32`，6 stages |
| threads / CTA | 128 | 128 |
| registers / thread | 248 | 88 |
| dynamic SMEM / CTA | 98.30 KB | 49.15 KB |
| 含 driver reserve 的 SMEM / CTA | 99.33 KB | 50.18 KB |
| register block limit | 2 | 5 |
| SMEM block limit | 1 | 2 |
| active CTAs / SM | 1 | 2 |
| active warps / SM | 4.00 | 7.99 |
| theoretical occupancy | 8.33% | 16.67% |
| achieved occupancy | 8.33% | 16.64% |

cuBLAS 选择的实际 kernel 为
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8`。
两边的首要 occupancy 限制均为 shared memory。当前 CuTe kernel 的
248 registers/thread 虽然很高，但寄存器资源仍允许 2 CTA/SM；真正把它
限制为 1 CTA/SM 的是约 99.33 KB/CTA 的 shared memory。

因此只降低寄存器数不会提高当前 occupancy。若要先达到与 cuBLAS 相同的
2 CTA/SM，需要把每 CTA 的总 shared memory 压到不超过 50 KB 左右；候选
单变量实验是将 `BlockK` 从 64 改为 32，使三 stage A/B pipeline 从 96 KiB
降到 48 KiB，再同时观察 pipeline 吞吐和 occupancy，而不是直接用
`maxrregcount` 强制溢出 accumulator。

## 实验 3：BlockK 64 → 32

### 实现与必要布局调整

- CTA tile 从 `128x128x64` 改为 `128x128x32`，保持 128 threads 和
  3-stage pipeline。
- 依据 CUTLASS 的 BF16/F16 K=32 默认配置，mainloop SMEM 从
  `Swizzle<3,3,3>` 改为 `Swizzle<2,3,3>` 的 `8x32` atom，G2S thread
  layout 从 `16x8` 改为 `32x4`；每条 `cp.async` 仍搬运 16B。
- epilogue C 继续使用 `Swizzle<3,3,3>`，避免混入 C layout 变化。
- K=32 时单独的三 stage A buffer 只有 24 KiB，不能容纳 32 KiB C tile；
  因此实验版使用 union，让 epilogue 在主循环结束后复用 A+B 合计 48 KiB
  的 pipeline storage。

### 资源与 occupancy

| 指标 | BlockK=64 | BlockK=32 |
|---|---:|---:|
| registers/thread | 248 | 238 |
| spill stores / loads | 0 / 0 | 0 / 0 |
| dynamic SMEM/CTA | 98.30 KB | 49.15 KB |
| 含 driver reserve 的 SMEM/CTA | 99.33 KB | 50.18 KB |
| active CTA/SM | 1 | 2 |
| theoretical occupancy | 8.33% | 16.67% |
| achieved occupancy | 8.33% | 16.58% |

目标 occupancy 确实翻倍；K=32 同时受到 register block limit=2 和
SMEM block limit=2 的约束。

### G2S bank conflict

`128x128x8192` 单 CTA SourceCounters：

- shared wavefronts：132224，理想值 99200，excessive 33024，占 25%。
- LDGSTS：66048 wavefronts，33024 bank conflicts。
- LDSM：65664 wavefronts，0 bank conflicts。
- STSM：256 wavefronts，0 bank conflicts。

与 K=64 下逐指令地址分析为 `Shared Wavefronts = Ideal` 不同，这里的
33024 excessive wavefronts 是固定的结构性冲突。K=32 的 BF16 行只有 64B；
保持 16B vector base 时，`Swizzle<2,3,3>` 只有两个不重叠的行 XOR bit，
不足以为一个 warp 覆盖的八行提供八种 bank permutation。

### 同环境 K=64/K=32 A/B

两个独立二进制交替执行五轮；每次均为 2 warmup、5 iterations，以下为
五轮中位数：

| 配置 | Identity ms | BlockSwizzle8 ms | cuBLAS ms |
|---|---:|---:|---:|
| BlockK=64 | 31.6926 | 30.7007 | 28.2106 |
| BlockK=32 | 30.7363 | 31.7824 | 28.6140 |

- K=32 Identity 比 K=64 Identity 快约 3.1%。
- K=32 BlockSwizzle8 比 K=64 BlockSwizzle8 慢约 3.4%。
- 两个配置各自最快路径为 30.7363 ms 与 30.7007 ms，差异不足 0.2%，
  occupancy 翻倍没有形成总体性能收益。
- K tile 数从 128 增加到 256，使 fence/wait、pipeline 轮转与循环控制次数
  翻倍；同时 16B LDGSTS 出现结构性 bank conflict，抵消了额外驻留 CTA
  带来的 latency hiding。

### 实验结论

- `8192x8192x8192` 所有 A/B 运行均为 0 mismatch。
- CTest 通过；Compute Sanitizer memcheck 为 0 errors；racecheck 为
  0 errors、0 warnings。
- 该实验未优于现有 K=64 BlockSwizzle8，因此只保留实验记录，默认源码
  恢复 `BlockK=64`。

## 实验 4：64x64x64 CTA tile

### 实现

- CTA tile 从 `128x128x64` 改为 `64x64x64`。
- 保持 128 threads、3-stage、16B `cp.async`、`Swizzle<3,3,3>`、
  `ldmatrix.x4`、BF16 Tensor Core MMA 和 stmatrix epilogue 不变。
- `64x64` C tile 仍可直接复用三 stage A storage，不需要改变 shared
  storage 的结构。

### 资源与 occupancy

| 指标 | 128x128x64 | 64x64x64 |
|---|---:|---:|
| registers/thread | 248 | 96 |
| spill stores / loads | 0 / 0 | 0 / 0 |
| dynamic SMEM/CTA | 98.30 KB | 49.15 KB |
| 含 driver reserve 的 SMEM/CTA | 99.33 KB | 50.18 KB |
| register block limit | 2 | 5 |
| SMEM block limit | 1 | 2 |
| active CTA/SM | 1 | 2 |
| theoretical occupancy | 8.33% | 16.67% |
| achieved occupancy | 8.33% | 16.64% |
| `8192x8192` grid CTA 数 | 4096 | 16384 |

缩小 M/N 后，每线程 FP32 accumulator 数从 128 降到 32，因此寄存器从
248 显著降至 96；最终 occupancy 仍由 48 KiB mainloop SMEM 限制为
2 CTA/SM。

### Bank conflict

`64x64x8192` 单 CTA 的指令分类计数：

| 指令路径 | shared wavefronts | bank conflicts |
|---|---:|---:|
| LDGSTS | 16640 | 0 |
| LDSM | 32832 | 0 |
| STSM | 64 | 0 |

该配置保留 128B 宽的 K=64 BF16 行和 `Swizzle<3,3,3>`，没有 K=32
实验中的结构性 G2S conflict。性能退化不能归因于 bank conflict。

### 同环境 128x128x64/64x64x64 A/B

两个独立二进制交替执行五轮；每次均为 2 warmup、5 iterations，以下为
五轮中位数：

| CTA tile | Identity ms | BlockSwizzle8 ms | cuBLAS ms |
|---|---:|---:|---:|
| 128x128x64 | 32.1931 | 31.4723 | 30.3086 |
| 64x64x64 | 48.0882 | 37.3606 | 32.3822 |

- 64x64x64 的 BlockSwizzle8 比其 Identity 快约 1.287 倍，说明小 tile
  对 CTA 发射顺序和 L2 locality 更敏感。
- 与 128x128x64 BlockSwizzle8 相比，64x64x64 仍慢约 18.7%。
- 输出 tile 面积缩小到四分之一，使 CTA 数增加四倍；每 CTA 的 A+B
  mainloop payload 只减半，因此整个 GEMM 的逻辑 G2S payload 和
  `cp.async` 数量约增加一倍。
- 每 CTA 的 K tile 数不变，CTA 数增加四倍还会放大 fence/wait、pipeline
  轮转、同步和 block 调度开销。忽略 cache reuse 时，CTA 算术强度由约
  64 FLOP/byte 降至约 32 FLOP/byte。

### 实验结论

- `8192x8192x8192` 所有交替 A/B 运行均为 0 mismatch。
- CTest 通过；Compute Sanitizer memcheck 为 0 errors；racecheck 为
  0 errors、0 warnings。
- occupancy 翻倍且 bank conflict 为 0，但数据复用与 CTA 粒度损失更大；
  该配置未保留，默认源码恢复 `128x128x64`。

### 回归

- CMake target 构建通过，`sm_120f`，248 registers/thread，0 spills。
- CTest `cute_gemm_bf16_fp32`：通过。
- Compute Sanitizer memcheck：0 errors。
- Compute Sanitizer racecheck：0 errors，0 warnings。
- `256x256x128` 与 `8192x8192x8192`：Identity/BlockSwizzle8 对 cuBLAS 均为 0 mismatch。
