# CuTe FP16/FP32 GEMM：FP32 向量化 epilogue 实验记录

> 状态说明：当前默认实现已改为直接配对的 `STG.E.64`。下方原有的
> `Swizzle<2,3,2>` FP32 SMEM epilogue 保留为历史对照，不再是默认路径。

## 当前默认：直接 FP32 STG.64

单个 m16n8 MMA 的每个 lane 持有两组相邻 FP32；CuTe 分析得到 accumulator
与 C partition 的 `max_common_vector=2`。C 基址、leading dimension 和 CTA
offset 均满足 8B 对齐，因此：

```cpp
copy_aligned(accumulator, thread_global_c);
```

会把每组两个 FP32 重解释为 64-bit copy，直接生成 `STG.E.64`。这条路径不
需要 `stmatrix`，也不需要 FP32 R2S、CTA barrier 或 LDS。

### SASS 与单 CTA 指标

配置：`128x128x128`。

| 指标 | 标量 `STG.E` | 直接 `STG.E.64` | SMEM + `STG.E.128` |
|---|---:|---:|---:|
| registers/thread | 244 | 244 | 246 |
| spill stores / loads | 0 / 0 | 0 / 0 | 0 / 0 |
| 静态 global store | 128 | 64 | 32 |
| global-store warp requests | 512 | 256 | 128 |
| global-store sectors | 4096 | 2048 | 2048 |
| epilogue `STS.64` | 0 | 0 | 64 |
| epilogue `LDS.128` | 0 | 0 | 32 |
| epilogue CTA barrier | 0 | 0 | 2 |

`STG.64` 与 `STG.128` 都达到 2048 个理论最小 output sectors；区别是
`STG.64` 不增加 shared-memory traffic。SourceCounters 的所有 global 访问为
6144 sectors，理想值也是 6144；shared wavefront 保持主循环本身的 2176，
理想值同样为 2176。

### `8192³` 三方 A/B

三个冻结二进制轮换顺序执行五轮；每个进程使用 2 次 warmup、5 次
iteration，所有运行均为 0 mismatch。

| 五轮中位数 | Identity ms | Identity TFLOP/s | BlockSwizzle8 ms | BlockSwizzle8 TFLOP/s | cuBLAS ms |
|---|---:|---:|---:|---:|---:|
| 标量 `STG.E` | 31.4958 | 34.9098 | 29.7659 | 36.9387 | 27.6015 |
| 直接 `STG.E.64` | 30.9586 | 35.5156 | 29.9260 | 36.7410 | 27.7586 |
| SMEM + `STG.E.128` | 32.0268 | 34.3310 | 29.7500 | 36.9583 | 27.8137 |

- 绝对中位数下，STG.64 的 Identity 比标量快 1.71%，比 SMEM 方案快
  3.34%；BlockSwizzle8 三者差异不超过 0.6%。
- 按每轮同进程 cuBLAS 归一化，STG.64 的 Identity 比标量改善 3.80%，
  BlockSwizzle8 与标量差 0.20%，属于性能持平。
- 因此默认选择 STG.64：写回 sectors 已理想，资源与同步开销最低；继续追求
  `STG.128` 指令数量更少，在此配置下没有转化成端到端收益。

### 回归

- CMake/CTest 通过，244 registers/thread，0 spill。
- Compute Sanitizer memcheck：0 errors。
- Compute Sanitizer racecheck：0 errors，0 warnings。
- `8192³` Identity、BlockSwizzle8 相对 cuBLAS 均为 0 mismatch，二者
  bitwise mismatch 为 0。

## 以下为历史 FP32 SMEM + STG.128 实验

## 目标与环境

- 分支：`feature/linear`
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，compute capability 12.0
- 目标：`cute_gemm_fp16_fp32`
- 计算：FP16 × FP16，FP32 accumulate，FP32 output
- CTA：`128x128x64`，128 threads，3-stage `cp.async` pipeline
- 主循环：16B G2S、`Swizzle<3,3,3>`、`ldmatrix.x4`、
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

## 为什么不能照搬 BF16 的 stmatrix

BF16 版本可先把 FP32 accumulator 转为 BF16，再使用
`stmatrix.sync.aligned.x4.m8n8.shared.b16` 写入共享内存。当前文件的输出语义是
FP32，而 SM90 CuTe `stmatrix` copy atom 只提供 `b16`，没有 `b32` 形式。

因此本实验保持 FP32 输出，采用两步写回：

1. MMA C partition 把 FP32 accumulator 以 `STS.64` 写入 swizzled SMEM。
2. CTA 同步后，以每线程 4 个 FP32 的 `LDS.128 + STG.128` 写回 GMEM。

主循环与 epilogue 不同时使用共享内存，代码用 union 让 64 KiB 的 C tile
复用 96 KiB A/B pipeline storage，所以动态 SMEM 仍为 98.30 KB/CTA。

## 基线：直接从 MMA fragment 写 FP32 GMEM

单 CTA 配置为 `128x128x128`。

| 指标 | 标量 epilogue |
|---|---:|
| registers/thread | 244 |
| spill stores / loads | 0 / 0 |
| 静态 global-store 指令 | 128 × `STG.E` |
| global-store warp requests | 512 |
| global-store sectors | 4096 |
| 理想 global-store sectors | 2048 |
| store sector amplification | 2.0× |

SourceCounters 的所有 global 访问为 8192 sectors，理想值为 6144，额外的
2048 sectors 全部来自该标量 FP32 epilogue。

## Swizzle 单变量实验

FP32 word 为 4B，共享内存 bank 由 word offset 的低 5 bit 决定。实验分别
观察 R2S `STS.64` 和 S2G `LDS.128` 的动态 shared wavefront。

| atom 与 swizzle | R2S wavefront | S2G wavefront | 结论 |
|---|---:|---:|---|
| `8x32 + Swizzle<3,2,3>` | 1024 | 512 | R2S 有 512 excessive |
| `16x16 + Swizzle<3,2,3>` | 1024 | 1024 | R2S、S2G 各有 512 excessive |
| `16x16 + Swizzle<2,3,2>` | 512 | 1024 | R2S 已解决，N=16 处重复 bank |
| `8x32 + Swizzle<2,3,2>` | 512 | 512 | 两段均达到理想值，最终采用 |

最终布局的原理：

- `8x32` row-major atom 让 row bit 0/1 位于 linear word bit 5/6。
- `Swizzle<2,3,2>` 保留最低 3 bit，即连续 8 个 FP32（32B），并将
  source bit 5/6 XOR 到 bank bit 3/4。
- 一条 `STS.64` 的 16-lane phase 中，列方向使用 bank bit 0..2，行方向
  经 swizzle 使用 bit 3/4，两组 bit 正交。
- 一条 `LDS.128` 的 8-lane phase 覆盖同一行连续 32 列；`8x32` atom
  正好覆盖这 32 列，不会像 `16x16` atom 那样在 N=16 处重复映射。

这里不能只看到代码里存在 `Swizzle` 就认为一定无冲突；swizzle 的 source
bit、target bit、atom shape 必须同时匹配真实的 MMA lane layout 和向量 copy
phase。

## 最终 SASS、访存与资源

单 CTA 配置为 `128x128x128`。

| 指标 | 标量基线 | 最终向量 epilogue |
|---|---:|---:|
| registers/thread | 244 | 246 |
| spill stores / loads | 0 / 0 | 0 / 0 |
| 静态 R2S | 无 | 64 × `STS.64` |
| 静态 S2G shared load | 无 | 32 × `LDS.128` |
| 静态 global store | 128 × `STG.E` | 32 × `STG.E.128` |
| global-store warp requests | 512 | 128 |
| global-store sectors | 4096 | 2048 |
| excessive global-store sectors | 2048 | 0 |
| R2S shared wavefront | 无 | 512，理想 512 |
| S2G shared wavefront | 无 | 512，理想 512 |

最终 SourceCounters：

- `L1 Wavefronts Shared = 3200`
- `L1 Wavefronts Shared Ideal = 3200`
- `L1 Wavefronts Shared Excessive = 0`
- `L2 Theoretical Sectors Global = 6144`
- `L2 Theoretical Sectors Global Ideal = 6144`
- `L2 Theoretical Sectors Global Excessive = 0`
- `LDGSTS` bank conflicts = 0
- `LDSM` bank conflicts = 0

aggregate shared-load counter 在一次运行中记录到 8，但逐指令地址分析的
excessive wavefront 为 0，因此它不是固定 lane-to-bank 映射冲突。

在 `8192x8192x8192` 下的资源对比：

| 项目 | 当前 CuTe | 同次 cuBLAS `Kernel2` |
|---|---:|---:|
| threads / CTA | 128 | 128 |
| registers / thread | 246 | 88 |
| dynamic SMEM / CTA | 98.30 KB | 49.15 KB |
| active CTA / SM | 1 | 2 |
| active warps / SM | 4.00 | 7.99 |
| theoretical occupancy | 8.33% | 16.67% |
| achieved occupancy | 8.33% | 16.64% |

当前实现首先受 SMEM 限制为 1 CTA/SM；246 registers/thread 单独看只会把
上限限制为 2 CTA/SM，并不是实际 occupancy 的第一限制。

## `8192³` 性能 A/B

保存修改前后的独立二进制，交替顺序执行 5 轮。每个进程使用 2 次 warmup、
5 次 iteration 和 rotating order；所有轮次均为 0 mismatch。

| 版本（五轮中位数） | Identity ms | Identity TFLOP/s | BlockSwizzle8 ms | BlockSwizzle8 TFLOP/s | cuBLAS ms |
|---|---:|---:|---:|---:|---:|
| 标量 epilogue | 31.1010 | 35.3529 | 30.8549 | 35.6349 | 28.2718 |
| 向量 epilogue | 32.9729 | 33.3460 | 30.7481 | 35.7587 | 27.5987 |

- 推荐的 BlockSwizzle8 路径绝对中位数快 0.35%，属于性能持平。
- Identity 路径中位数回退 6.02%。
- 笔记本 GPU 的不同进程间时钟漂移明显；每轮用同进程 cuBLAS 归一化后，
  BlockSwizzle8 的五轮中位性能从 cuBLAS 的 93.27% 变为 90.29%，约回退
  3.30%。因此不能把 0.35% 的绝对差异解释为稳定加速。
- 本修改确定改善的是写回合并度、global sectors 和结构性 bank conflict；
  对 `K=8192` 这种 compute-bound 问题，端到端性能结论是近似持平，而不是
  已证明的吞吐提升。

## 回归结果

- CMake 目标编译通过，0 spill。
- CTest `cute_gemm_fp16_fp32` 通过。
- `8192³` Identity、BlockSwizzle8 相对 cuBLAS 均为 0 mismatch，二者 bitwise
  mismatch 为 0。
- Compute Sanitizer memcheck：0 errors。
- Compute Sanitizer racecheck：0 errors，0 warnings。
