# conv2d_groups_fp16 优化报告

## 结论

- 状态：`[SUCCESS]`。
- K3/K5/K7/K9 使用编译期 channel-pair kernel；K11 使用通用 16×16
  shared-input tile。
- K9 在 pair grid 不少于 140 blocks 且 shared memory 不超过 48 KiB 时
  使用 shared-input，否则回落到 pair kernel。
- 代表配置加速：K3 small `1.623x`、K5 small `2.342x`、K7 small
  `2.605x`、K9 throughput `2.821x`、K11 throughput `3.229x`。

## 最终 Dispatch

| 条件 | 实现 | Fallback |
|------|------|----------|
| `K=3/5/7` | `conv2d_1x128x256_fp16_groups_kernel<K>` | — |
| `K=9`、pair grid `>=140`、shared `<=48 KiB` | `conv2d_8x16x16_fp16_shared_input_kernel<9>` | K9 pair kernel |
| `K=11`、shared `<=48 KiB` | `conv2d_8x16x16_fp16_shared_input_kernel<11>` | K11 四 pair 特化 kernel |
| 其他 `K*K<=128` | `conv2d_4x128x256_fp16_groups_kernel<0>` | — |

shared-input 路径按输出尺寸和 stride 运行时生成 tile grid，不依赖固定 H/W。

## 最终性能

latency 取 3 个 group median 的中位数。每组 20 次 warmup、100 个样本；
极短 kernel 按配置批量 launch。Reference 为包含 bias 的 cuDNN NCHW。

| 配置 | Baseline ms | Final group medians ms | Speedup | Final spread | Reference ms |
|------|------------:|------------------------|--------:|-------------:|-------------:|
| K3 N2 C64 H64 small | 0.003658 | 0.002254 / 0.002254 / 0.002254 | 1.623x | 0.00% | 0.014864 |
| K3 N64 C128 H80 throughput | 0.143440 | 0.142848 / 0.142848 / 0.142848 | 1.004x | 0.00% | 0.227392 |
| K5 N1 C32 H80 small | 0.005732 | 0.002450 / 0.002448 / 0.002448 | 2.342x | 0.08% | 0.015012 |
| K5 N64 C128 H80 throughput | 0.176888 | 0.164856 / 0.164816 / 0.164808 | 1.073x | 0.03% | 0.302000 |
| K7 N64 C128 H80 throughput | 0.315904 | 0.288000 / 0.288000 / 0.288000 | 1.097x | 0.00% | 0.446200 |
| K7 N4 C128 H80 main | 0.022176 | 0.017748 / 0.017808 / 0.017808 | 1.245x | 0.34% | 0.030880 |
| K7 N1 C32 H43 small | 0.007988 | 0.003066 / 0.003066 / 0.003066 | 2.605x | 0.00% | 0.015086 |
| K9 N1 C32 H80 pair | 0.012548 | 0.004108 / 0.004108 / 0.004111 | 3.054x | 0.07% | 0.015400 |
| K9 N64 C128 H80 throughput | 0.480440 | 0.172288 / 0.169806 / 0.170294 | 2.821x | 1.46% | 0.586036 |
| K11 N64 C128 H80 throughput | 0.697088 | 0.215808 / 0.215854 / 0.216888 | 3.229x | 0.50% | 0.832120 |
| K11 N1 C32 H43 tail | 0.015108 | 0.005307 / 0.005307 / 0.005308 | 2.847x | 0.02% | 0.015408 |
| K11 N1 C32 65×81 | 0.014628 | 0.005403 / 0.005404 / 0.005404 | 2.707x | 0.02% | 0.015671 |
| K11 N1 C32 H43 stride1 | 0.014784 | 0.005340 / 0.005343 / 0.005343 | 2.767x | 0.06% | 0.015588 |

K3/K5 throughput baseline 是在增加饱和配置时冻结的 grouped 特化版本；K7 small
和 K11 尾部配置是在对应结构性实验前冻结。其余主配置使用实验 0 baseline。

## 瓶颈与 NCU

| 路径 | Registers | Shared memory | Spill | 最终关键指标 |
|------|----------:|--------------:|------:|--------------|
| K3 pair | 40 | 64 B static | 0 | DRAM 93.25%，已接近带宽上限 |
| K5 pair | 44 | 128 B static | 0 | Compute 88.53%，DRAM 73.70% |
| K7 pair | 48 | 224 B static | 0 | Compute 92.00%，DRAM 40.73% |
| K9 pair | 48 | 352 B static | 0 | Compute 93.91% |
| K9 shared | 43 | 约 26 KiB dynamic | 0 | Memory 74.99%，DRAM 68.89% |
| K11 shared | 43 | 29.6 KiB(S2) / 13.3 KiB(S1) dynamic | 0 | Memory 77.48%，Compute 62.95% |
| K11 grouped fallback | 44 | 2000 B static | 0 | K11 shared 容量超限时使用 |
| 动态 fallback | 40 | 2064 B static | 0 | 其他合法 kernel size 使用 |

原 K7/K11 的 Compute throughput 为 84.63%/86.41%，动态 tap 地址计算和
math-pipe throttle 明显。编译期特化与 pair 映射后，K7 达到 92%。K11 shared
路径将 NCU duration 从 751.65 us 降至 233.09 us；shared excessive 约
`1.06M / 40.43M` wavefronts。

## 实验记录

| 轮次 | 改动 | 结果 | 决策 | Commit |
|-----:|------|------|:----:|:------:|
| 0 | 提取 FP16 baseline | 正确性通过 | BASELINE | `3d6932f` |
| 1 | K3/K5/K7/K11 编译期特化 | K3/K5/K11 明显提升 | KEEP | `db7162e` |
| 2 | compact shared weight staging | K3 +4.3%，K5 +2.8% | KEEP | `95c322d` |
| 3 | 合并 weight/bias barrier | 各配置收益小于 2% | REVERT | — |
| 4 | K7 channel-pair 映射 | small 2.61x，throughput +6.6% | KEEP | `e17e68e` |
| 5 | K11 shared-input tile | 2.7x–3.24x | KEEP | `4327ca4` |
| 6 | K3/K5 泛化 pair kernel | K3 small 1.26x，K5 small 2.09x | KEEP | `c538ef5` |
| 7 | K9 pair kernel | small 3.05x，throughput 1.09x | KEEP | `f9b4b26` |
| 8 | K9 条件 shared-input | throughput 2.58x；阈值 140 blocks | KEEP | `dc1f47c` |

## 正确性与安全检查

- 27 个配置全部通过，覆盖 K3/5/7/9/11、stride 1/2/3、矩形、尾部、
  C=8/40/48、K9 128/140-block dispatch 边界及 K11 shared fallback。
- 与 cuDNN NCHW 对比错误元素为 0；最大绝对误差 `5.37e-3`，容差
  `atol=1e-2`、`rtol=1e-2`。
- `compute-sanitizer 2025.4.0` memcheck 覆盖全部配置：0 errors。
- K7 pair、K9 shared、K11 shared 的 racecheck：0 hazards；synccheck：
  0 errors。
- `/tmp/cudaop_auto_check2` 干净 CMake 构建通过，目标 `sm_89`。

## 环境与剩余限制

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090，128 SM，24 GB |
| 架构 | Compute Capability 8.9，`sm_89` |
| 工具 | CUDA 13.1、Driver 590.44.01、Nsight Compute 2025.4.0 |
| 计时 | CUDA Event；独立 1 秒预热；20 warmup；100 samples；3 groups |

- K3 已接近 DRAM 上限；继续优化需减少有效输入字节。
- K5/K7/K9 pair 已接近 compute pipeline 上限。
- K9/K11 shared 的主要限制是 memory throughput、long/short scoreboard。
- `cp.async` 未实验：输入为跨 channel 的 2 B 离散 load，无法直接形成连续
  global-to-shared copy；TMA/cluster 不支持 `sm_89`。
- 未使用 Tensor Core：depthwise channel 独立，无法形成高复用 dense MMA。

`[SUCCESS]`
