# conv2d_4x128x256_groups_kernel 优化报告

## 结论

- 状态：`[SUCCESS]`。
- 基线冻结于 2026-07-11，最终验证完成于 2026-07-12。
- K3/K5/K7 编译期特化并使用只读 cache-hinted input load；代表配置分别加速
  `1.264x`、`1.440x`、`1.388x`。
- K7 小 grid 使用单通道 block，代表配置加速 `2.316x`；104 grouped blocks
  起回落到四通道特化。
- K11 shared-input 路径保持 `2.201x-2.592x` 加速。
- 历史数据：[RTX 5070 Ti Laptop 报告](opt_kernel_conv2d_groups_rtx5070ti_laptop_20260710.md)。

## 最终实现与 Dispatch

| 条件 | 实现 | 关键改动 | Fallback |
|------|------|----------|----------|
| `Kh=Kw=11`、`C%4=0`、shared memory 不超过 48 KiB | `conv2d_4x16x16_shared_input_kernel` | 16x16 input tile、compact swizzle、四通道 `sts128`、K11 编译期特化、只读 `.L2::128B` load | 条件不满足时使用通用 kernel |
| `K=7` 且原四通道 grid `<104` blocks | `conv2d_1x128x256_groups_k7_kernel` | 单通道 block，提高小 grid 并行度 | 104 blocks 起使用四通道特化 |
| `K=3/5/7` | `conv2d_4x128x256_groups_kernel<K>` | 编译期消除动态除法、循环边界和尾部工作；input 使用只读 cache hint | 其他 K 使用动态模板实例 |
| 其他配置 | `conv2d_4x128x256_groups_kernel<0>` | 删除冗余 barrier，bias 使用独立 shared 尾部 | — |

K11 路径按运行时 `out_h/out_w` 生成 tile grid。input staging 由每个线程读取
一个 spatial 位置的四个 channel，再以一条 `sts128` 写入交错 shared 布局；
compact swizzle 使计算阶段的 input/weight `LDS.128` 达到 ideal。

## 最终性能

latency 取 3 个 group median 的中位数，`Speedup = Baseline / Final`。
CPU reference 只用于正确性；本轮未建立同语义 GPU 参考实现。

| 配置 | Baseline group medians ms | Final group medians ms | Speedup | Final spread | Reference |
|------|---------------------------|------------------------|--------:|-------------:|----------:|
| K3 N2 C64 H64 S2 | 0.002990 / 0.002989 / 0.002990 | 0.002366 / 0.002366 / 0.002368 | 1.264x | 0.08% | N/A |
| K5 N1 C32 H80 S2 | 0.004212 / 0.004212 / 0.004212 | 0.002923 / 0.002924 / 0.002924 | 1.440x | 0.03% | N/A |
| K7 N32 C128 H80 throughput | 0.208000 / 0.208000 / 0.200576 | 0.149504 / 0.149880 / 0.150016 | 1.388x | 0.34% | N/A |
| K7 N1 C32 H43 small grid | 0.006428 / 0.006428 / 0.006428 | 0.002775 / 0.002774 / 0.002776 | 2.316x | 0.07% | N/A |
| K7 dispatch 96 blocks | 0.006719 / 0.006717 / 0.006717 | 0.004232 / 0.004232 / 0.004232 | 1.587x | 0.00% | N/A |
| K7 dispatch 104 blocks | 0.006684 / 0.006684 / 0.006684 | 0.004288 / 0.004288 / 0.004287 | 1.559x | 0.02% | N/A |
| K11 N32 H80 throughput | 0.429696 / 0.413692 / 0.429568 | 0.195200 / 0.195200 / 0.195200 | 2.201x | 0.00% | N/A |
| K11 N4 H80 main | 0.053680 / 0.052589 / 0.053696 | 0.025504 / 0.025493 / 0.025502 | 2.105x | 0.04% | N/A |
| K11 N4 H64 out32 | 0.032381 / 0.031952 / 0.031472 | 0.014208 / 0.014208 / 0.014222 | 2.249x | 0.10% | N/A |
| K11 N1 H43 tail | 0.013252 / 0.013252 / 0.013252 | 0.005112 / 0.005112 / 0.005112 | 2.592x | 0.00% | N/A |
| K11 N1 65x81 rectangular | 0.013480 / 0.013480 / 0.013480 | 0.005276 / 0.005276 / 0.005277 | 2.555x | 0.02% | N/A |
| K11 N1 H43 stride1 | 0.012748 / 0.012751 / 0.012748 | 0.005260 / 0.005260 / 0.005260 | 2.424x | 0.00% | N/A |

非 K11 行使用同一可执行文件中的动态 fallback 与最终 dispatch 分开预热对比；
K11 保留冻结基线。K11 baseline 存在约 3.7% 漂移，但收益远高于噪声。

## 正确性与安全检查

- 最终实现通过 23 个配置，覆盖 K=3/5/7/11、stride=1/2、C=4/36、
  对齐/非对齐输出、K7 96/104-block dispatch 边界、尾部和矩形输入。
- CPU reference 对比错误元素为 0，最大绝对误差为 `5.245209e-06`，
  最大相对误差为 `2.0`；FP32 容差为 `atol=1e-4`、`rtol=1e-4`。
- `compute-sanitizer 2025.4.0 memcheck` 覆盖全部 23 个配置：0 errors。
- 本轮 `racecheck`/`synccheck` 覆盖 K7 single-channel 和 grouped 边界路径：
  0 errors/hazards；K11 shared-input 沿用同环境已通过结果。
- 初次 racecheck 检出 fallback 尾部 weight 读取与 bias 写入间的 768 个
  hazards；将 bias 移到预留 shared 尾部后，重新完成全部验证。

## 环境与测试

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090，128 SM，24 GB |
| 架构 | Compute Capability 8.9，编译目标 `sm_89` |
| 工具 | CUDA Toolkit 13.1、NVCC 13.1.80、Nsight Compute 2025.4.0.0；Driver 未单独记录 |
| 目标 | `op/dwconv/opt_conv2d_groups_fp32.cu`，CMake target `opt_conv2d_groups_fp32` |
| 数据 | FP32，固定随机种子，CPU depthwise convolution reference |
| 计时 | CUDA Event；20 次 warmup、100 个正式样本、3 个 group |
| 吞吐配置 | K7 N32 C128 H80 S2，7168 blocks，NCU `Waves Per SM=9.33` |

构建架构使用当前 GPU 查询值；项目初始固定的 `sm_120` 不作为本轮性能基线。

## 资源与关键性能证据

| 路径 | Registers | Shared memory | Spill | Baseline / Final NCU | 关键指标 |
|------|----------:|--------------:|------:|----------------------|----------|
| K3 四通道特化 | 42/thread | 2064 B static | 0 | 132.51 / 128.22 us | DRAM 93.77%，接近带宽上限 |
| K5 四通道特化 | 56/thread | 2064 B static | 0 | 157.92 / 131.42 us | DRAM 91.90% |
| K7 四通道特化 | 56/thread | 2064 B static | 0 | 195.55 / 142.75 us | DRAM 84.84%，Compute 77.06% |
| K7 单通道小 grid | 40/thread | 208 B static | 0 | N/A / 6.30 us | 96 grouped blocks 等价为 384 blocks，0.50 waves/SM |
| K11 shared-input S2 | 43/thread | 29616 B dynamic | 0 | 439.14 / 204.10 us | 指令 397.38M→90.57M |
| K11 shared-input S1 | 43/thread | 13296 B dynamic | 0 | 同一实现 | 0 spill |

K3/K5 的 DRAM throughput 已达 93.77%/91.90%，继续增加 shared staging
不具备明确收益。K7 大 grid 是 DRAM 84.84%、SM 77.06% 的混合瓶颈；
小 grid 不用于判断硬件吞吐上限，单通道路径用于增加可调度 block 数。

K11 final 的 L1/TEX throughput 为 84.00%，复合 memory throughput 为
82.32%，DRAM throughput 为 59.69%。shared excessive 为
1.50M/40.88M wavefronts（约 3.7%）；global excessive 为
1.35M/8.18M sectors（约 16.6%）。

剩余主要限制是 L1/MIO 发射和 long/short scoreboard。降低 global excessive
的 warp-row staging 因分支和负载不均衡而回退，因此当前实现可视为该数据流在
RTX 4090 上的合理上限，而非理论绝对上限。

## 实验记录

| 轮次 | 改动 | 关键结果 | 决策 | Commit |
|-----:|------|----------|:----:|:------:|
| 0 | 原始 kernel | K7 NCU 202.02 us | BASELINE | — |
| 1 | 旧 16x16 K7 shared-spatial | NCU 回退 9.4%，occupancy 96.02%→62.49% | REVERT | — |
| 2 | K7 跳过尾部无效 input load | NCU 提升 0.35% | REVERT | — |
| 3 | bias 与初始同步合并 | NCU 提升 0.59% | REVERT | — |
| 4 | 10x20 K7 shared-spatial | throughput 回退 4%-9%，out32 回退约 29% | REVERT | — |
| 5 | 删除循环内冗余 barrier | K7 NCU 提升 2.76%，必测配置改善 | KEEP | `ee7c233` |
| 6 | 128-thread block | out32 回退最高约 3%，throughput 不稳定 | REVERT | — |
| 7 | 在轮次 5 上合并 bias 同步 | 增量 NCU 约 0.15% | REVERT | — |
| 8 | 通用 8x16 shared-input | K5/K7 回退，K11 有局部收益 | 转轮次 9 | — |
| 9 | 16x16 K11 shared-input | K11 NCU 提升 5.04%，多尺寸改善 | KEEP | `ee7c233` |
| 10 | padded shared swizzle | shared 增至 33.55 KiB，Event 回退约 18% | REVERT | — |
| 11 | compact swizzle | `LDS.128` 达到 ideal，NCU 419.65→382.72 us | KEEP | `ee7c233` |
| 12 | warp-row global staging | global excessive 降低，但 Event 回退约 3% | REVERT | — |
| 13 | 四通道读取与 `sts128` input staging | 指令 251.12M→143.82M，约 0.343→0.220 ms | KEEP | `ee7c233` |
| 14 | weight `sts128` staging | 增量约 1.4% | REVERT | — |
| 15 | K11 编译期特化 | 指令 143.82M→94.74M，约 0.220→0.202 ms | KEEP | `ee7c233` |
| 16 | input 只读 `.L2::128B` load | L2 hit 52.56%→77.45%，约 0.202→0.194 ms | KEEP | `ee7c233` |
| 17 | 在最终数据流上重测 8x16 | 约 0.197 ms，慢于 16x16 | REVERT | — |
| 18 | K3/K5/K7 编译期特化 | 代表配置提升约 15%-31% | KEEP | `4184821` |
| 19 | `launch_bounds(256,5)` 压低 K7 registers | 52→48 registers，但约 157→165 us | REVERT | — |
| 20 | 特化 input 使用只读 cache-hinted load | K3/K5/K7 再提升约 5%-13% | KEEP | `5affcbe` |
| 21 | K7 全范围单通道 block | 小 grid 大幅提升，吞吐配置约 150→204 us | 条件保留 | `f786b5a` |
| 22 | K7 single/grouped 阈值搜索 | 96 blocks 选 single，104 blocks 选 grouped | KEEP | `f786b5a` |

轮次 5 最终还包含 shared bias 隔离修复。轮次 9、11、13、15、16 共同构成
最终 K11 路径；现有历史中这些有效改动合并提交于 `ee7c233`。

## 剩余限制

- K3/K5 已接近 DRAM 上限；进一步收益需要减少有效输入字节，而不是微调 occupancy。
- K7 大 grid 仍是 memory/compute 混合瓶颈；旧 shared-input tile 已实测回退，
  当前可信路径已覆盖编译特化、cache hint、资源约束和 channel 映射。
- K7 `<104` blocks 的路径解决小 grid 利用率，不代表达到整卡理论吞吐。
- K11 global staging 仍约有 16.6% excessive sectors。
- `cp.async` 无法直接完成四个非连续 channel 到交错 float4 shared 的转置；
  改回 channel-major 会破坏当前 ideal `LDS.128`。
- 未继续手写 software pipeline；最终 SASS 已将后续 `LDS.128` 与 FMA 交错。
- 没有同语义 GPU 参考库，Reference 为 `N/A`。
- 后续优化应优先降低 staging 指令与分支成本，同时保持 16x16 tile 的 occupancy。

`[SUCCESS]`
