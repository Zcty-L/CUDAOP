# DWConv 与 `/opt_kernel` 优化项目总结

## 项目结论

本项目包含两条相互促进的工作线：

1. 手工完成 FP32、FP16 depthwise convolution 的初版 CUDA kernel；
2. 将实际优化过程沉淀为 `/opt_kernel` SKILL，再用该 SKILL 继续优化初版 kernel。

在 RTX 4090 上，手工初版已经普遍快于 cuDNN NCHW。SKILL 在此基础上仍找到了
编译期特化、线程映射、shared-input staging 和条件 dispatch 等结构性优化：

- FP32 代表配置相对初版提升 `1.264x-2.592x`；
- FP16 代表配置相对初版提升 `1.004x-3.229x`；
- FP32 最终版在有 cuDNN 补测的三个配置上达到 `1.597x-4.423x`，但两者来自
  不同测试批次，且 cuDNN 不含 bias，作为指示性结果使用；
- FP16 最终版在表内代表配置上达到 cuDNN 的 `1.549x-6.594x` 吞吐优势，
  其中 K3/K5 的大吞吐配置增益较小，但没有明显回退。

这里不对不同 shape 的加速比求平均。小 grid、吞吐饱和配置和不同 kernel size
受不同瓶颈控制，平均值会掩盖 dispatch 的实际适用范围。

## 项目过程

### 1. 手工实现 DWConv 初版

2026-07-10 完成两种数据类型：

| 数据类型 | 提交 | 初版核心实现 |
|----------|------|--------------|
| FP32 | `048fd14` | `conv2d_4x128x256_groups_kernel`，每个 block 计算四个 channel |
| FP16 | `7120aed` | `conv2d_4x128x256_fp16_groups_kernel`，使用 `half2` 计算四组 channel pair |

初版已包含 shared weight、融合 bias、NCHW 输出以及按 kernel capacity
拆分的 K32/K64/K128 版本。benchmark 同时测试自定义 kernel、cuDNN NCHW 和
cuDNN NHWC，并用 cuDNN NCHW 校验数值正确性。

#### FP32 初版与 cuDNN

FP32 原优化报告没有保存 cuDNN reference。下表是 2026-07-13 在同一 RTX 4090
上对初版 K128 target 的补测，使用 20 次 warmup、100 次正式迭代；自定义实现为
最初被 SKILL 提取的 `custom` kernel。

| 配置 | 初版 ms | cuDNN NCHW ms | 初版相对 cuDNN |
|------|--------:|---------------:|----------------:|
| K5 N1 C32 H80 S2 | 0.004639 | 0.012933 | `2.788x` |
| K7 N32 C128 H80 S2 | 0.187415 | 0.239296 | `1.277x` |
| K11 N32 C128 H80 S2 | 0.399194 | 0.432814 | `1.084x` |

这说明初版在小配置上明显降低了 cuDNN 的固定开销，在吞吐配置上仍有
约 `8%-28%` 的优势。该补测用于恢复项目结果，不替代 2026-07-11 冻结的
SKILL 性能基线。

#### FP16 初版与 cuDNN

FP16 最终测试程序同时保留了初版 baseline、最终 dispatch 和带 bias 的 cuDNN
NCHW reference，因此三者可以直接比较。

| 配置 | cuDNN ms | 初版 ms | 初版相对 cuDNN |
|------|---------:|--------:|----------------:|
| K3 N2 C64 H64 small | 0.014864 | 0.003658 | `4.063x` |
| K3 N64 C128 H80 throughput | 0.227392 | 0.143440 | `1.585x` |
| K5 N1 C32 H80 small | 0.015012 | 0.005732 | `2.619x` |
| K5 N64 C128 H80 throughput | 0.302000 | 0.176888 | `1.707x` |
| K7 N64 C128 H80 throughput | 0.446200 | 0.315904 | `1.412x` |
| K7 N1 C32 H43 small | 0.015086 | 0.007988 | `1.889x` |
| K9 N64 C128 H80 throughput | 0.586036 | 0.480440 | `1.220x` |
| K11 N64 C128 H80 throughput | 0.832120 | 0.697088 | `1.194x` |

初版 FP16 在这些代表配置上已达到 cuDNN 的 `1.194x-4.063x`。大 kernel 的
初版优势较小，也因此成为后续结构性优化的重点。

### 2. 编写并用 DWConv 打磨 `/opt_kernel` SKILL

初版完成后，将 kernel 优化整理为从预检、基线、NCU 分析、计划、单变量实验、
最终验证到交付的完整流程。DWConv 是 SKILL 的首个系统性实战对象，实际执行中
补齐了以下关键规则：

- 架构使用当前 GPU 查询到的 SM，只有用户显式指定时才覆盖；RTX 4090 使用
  `sm_89`，不再固定或降低目标架构。
- 每个实现和参考实现独立预热，至少测量三组；组间噪声统一使用
  `(max(group_median)-min(group_median))/median(group_median)`。
- 测试矩阵同时覆盖吞吐饱和、业务、小 grid、边界、stride、矩形和 fallback，
  不用单一 shape 代替通用性验证。
- “单变量”限制的是每轮假设数量，不限制改动规模；必须评估 shared tile、
  software pipeline、`cp.async`、向量化、cache hint、TMA 等高级机制，不能只做微调。
- 连续两次实验无提升只触发重新分析，不能直接停止；停止前必须验证过可信的
  结构性方案，或用硬件指标证明已经接近合理上限。
- 某类配置显著受益时保留专用 kernel，通过可解释的运行时条件 dispatch；
  不受益的配置回落到最近的稳定基线，不要求单个实现覆盖所有 shape。
- 每次 `KEEP` 立即创建独立 `perf:` commit，失败实验回退未提交改动，使每个
  有效优化都能单独审查和撤回。
- 最终必须执行干净 CMake 构建、完整正确性矩阵、memcheck，以及 shared/sync
  路径的 racecheck 和 synccheck。
- 报告按算子、GPU 和日期独立存放，正文先给结论和 dispatch，再保留必要的
  性能、NCU、正确性和实验决策，避免把所有 GPU 和流水日志堆在一个文件中。

这使 SKILL 从“优化步骤清单”变成了可形成证据链、支持多配置条件优化并可通过
Git 复现实验结果的工作流。

### 3. 使用 `/opt_kernel` 再次优化 FP32

FP32 SKILL 基线直接来自手工初版。最终根据 kernel size 和工作量形成三类路径：

- K3/K5/K7：编译期特化 tap 循环，并使用只读 cache-hinted input load；
- K7 小 grid：切换为单通道 block，提高可调度 block 数；
- K11：使用运行时尺寸生成的 16x16 shared-input tile，不对固定 40x40 shape
  做特殊化；其他配置保留通用 fallback。

| 配置 | 初版基线 ms | SKILL 最终版 ms | 相对初版提升 |
|------|------------:|-----------------:|---------------:|
| K3 N2 C64 H64 S2 | 0.002990 | 0.002366 | `1.264x` |
| K5 N1 C32 H80 S2 | 0.004212 | 0.002924 | `1.440x` |
| K7 N32 C128 H80 throughput | 0.208000 | 0.149880 | `1.388x` |
| K7 N1 C32 H43 small grid | 0.006428 | 0.002775 | `2.316x` |
| K11 N32 C128 H80 throughput | 0.429696 | 0.195200 | `2.201x` |
| K11 N1 C32 H43 tail | 0.013252 | 0.005112 | `2.592x` |

最终 K3/K5 的 DRAM throughput 达到 `93.77%/91.90%`，K7 吞吐配置成为
memory/compute 混合瓶颈；K11 通过 shared-input staging 将大量重复地址计算和
global load 转化为 tile 内复用，是本轮最大的结构性收益。

#### FP32 SKILL 最终版与 cuDNN

FP32 历史 SKILL benchmark 没有同批次 cuDNN reference。下表将 2026-07-11
冻结的 SKILL 最终结果与 2026-07-13 的 cuDNN 补测对齐到完全相同的 shape：

| 配置 | SKILL 最终版 ms | cuDNN NCHW ms | 最终版相对 cuDNN |
|------|-----------------:|---------------:|-------------------:|
| K5 N1 C32 H80 S2 | 0.002924 | 0.012933 | `4.423x` |
| K7 N32 C128 H80 S2 | 0.149880 | 0.239296 | `1.597x` |
| K11 N32 C128 H80 S2 | 0.195200 | 0.432814 | `2.217x` |

该表说明最终版仍明显快于 cuDNN，但不是严格的同批次对比。动态时钟和运行状态
可能影响绝对时间；此外 cuDNN 数字只包含 convolution，未包含 bias，计时口径
对 cuDNN 更有利。

### 4. 使用 `/opt_kernel` 再次优化 FP16

FP16 先从 `dwconv_fp16_common.cuh` 提取手工初版作为独立 baseline，随后形成：

- K3/K5/K7：编译期 channel-pair kernel；
- K9：pair grid 不少于 140 blocks 且 shared memory 不超过 48 KiB 时使用
  shared-input，否则回落到 pair kernel；
- K11：优先使用通用 16x16 shared-input tile，容量超限时回落到四 pair kernel；
- 其他合法 kernel size：保留动态 fallback。

| 配置 | 初版 ms | 最终版 ms | SKILL 提升 | cuDNN ms | 最终版相对 cuDNN |
|------|--------:|----------:|-----------:|---------:|-------------------:|
| K3 N2 C64 H64 small | 0.003658 | 0.002254 | `1.623x` | 0.014864 | `6.594x` |
| K3 N64 C128 H80 throughput | 0.143440 | 0.142848 | `1.004x` | 0.227392 | `1.592x` |
| K5 N1 C32 H80 small | 0.005732 | 0.002448 | `2.342x` | 0.015012 | `6.132x` |
| K5 N64 C128 H80 throughput | 0.176888 | 0.164808 | `1.073x` | 0.302000 | `1.832x` |
| K7 N64 C128 H80 throughput | 0.315904 | 0.288000 | `1.097x` | 0.446200 | `1.549x` |
| K7 N1 C32 H43 small | 0.007988 | 0.003066 | `2.605x` | 0.015086 | `4.920x` |
| K9 N64 C128 H80 throughput | 0.480440 | 0.170294 | `2.821x` | 0.586036 | `3.441x` |
| K11 N64 C128 H80 throughput | 0.697088 | 0.215854 | `3.229x` | 0.832120 | `3.855x` |
| K11 N1 C32 H43 tail | 0.015108 | 0.005307 | `2.847x` | 0.015408 | `2.903x` |

FP16 结果说明同一优化不能覆盖所有规模：K3/K5/K7 的吞吐配置已经接近
memory 或 compute pipeline 上限，增益仅 `0.4%-9.7%`；小 grid 通过 channel-pair
映射获得 `1.623x-2.605x`。K9/K11 则存在足够的空间复用，shared-input 路径带来
`2.821x-3.229x` 的吞吐提升。

## 验证结果

| 项目 | FP32 | FP16 |
|------|------|------|
| GPU / 架构 | RTX 4090 / `sm_89` | RTX 4090 / `sm_89` |
| 正确性矩阵 | 23 个配置，全部通过 | 27 个配置，全部通过 |
| 参考 | CPU FP32 depthwise convolution | cuDNN NCHW FP16 + bias |
| 最大绝对误差 | `5.245209e-06` | `5.37e-3` |
| memcheck | 0 errors | 0 errors |
| racecheck / synccheck | 接受的 K7/K11 路径通过 | K7 pair、K9/K11 shared 路径通过 |
| spill | 最终路径均为 0 | 最终路径均为 0 |
| 构建 | CMake `sm_89` 通过 | 干净 CMake `sm_89` 通过 |

## 数据口径与限制

- `Speedup = 较慢实现 latency / 较快实现 latency`。
- SKILL 报告 latency 取三组 group median 的中位数；表格展示到足以复核结论的
  精度，完整三组数据见对应优化报告。
- FP32 初版/cuDNN 表是 2026-07-13 补测；FP32 SKILL 表使用 2026-07-11
  冻结基线。两张表的绝对时间不应直接相减或相乘。
- 旧 FP32 初版 benchmark 的自定义 kernel 融合 bias，而 cuDNN 计时只包含
  convolution，bias 在计时后用于正确性校验；因此该对比对 cuDNN 更有利。
- FP16 最终 benchmark 对 cuDNN convolution 与 `cudnnAddTensor` 一起计时，
  与融合 bias 的自定义 kernel 语义一致。
- cuDNN 使用 `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`。本文结论只代表当前
  项目实现、测试算法和 RTX 4090 环境，不代表所有 cuDNN frontend/engine 选择。

## 原始结果与图表

初版完整 sweep 已在 RTX 4090 上重新执行并由 benchmark 程序直接写入 CSV。
每个 `K/C/H` 都包含 `N=1/2/4/8/16/32`，并保存
`custom/biasopt/db/cuDNN NCHW/cuDNN NHWC` 五个系列。

| 数据类型 | CSV | 绘图脚本 |
|----------|-----|----------|
| FP32 | [benchmark_results_fp32.csv](opt_kernel/opt_dwconv/summary/benchmark_results_fp32.csv) | [plot_benchmark_fp32.py](../op/dwconv/plot_benchmark_fp32.py) |
| FP16 | [benchmark_results_fp16.csv](opt_kernel/opt_dwconv/summary/benchmark_results_fp16.csv) | [plot_benchmark_fp16.py](../op/dwconv/plot_benchmark_fp16.py) |

优化报告中的离散配置另存为
[dwconv_opt_kernel_fp32.csv](opt_kernel/opt_dwconv/summary/dwconv_opt_kernel_fp32.csv)
和
[dwconv_opt_kernel_fp16.csv](opt_kernel/opt_dwconv/summary/dwconv_opt_kernel_fp16.csv)。
其中 `time_ms` 是报告采用的 latency，`group_medians_ms` 保留历史报告中存在的
三组 median；历史未保存的 100 个单独 sample 无法从汇总报告反推。

报告汇总 CSV 中 `kernel` 字段含义如下：

- `initial`：手工初版或对应配置冻结的 SKILL baseline；
- `skill_final`：`/opt_kernel` 最终 dispatch；
- `cudnn_nchw`：cuDNN NCHW reference。

绘图脚本已实际运行，按 kernel size 生成吞吐图：

| 数据类型 | 图表 |
|----------|------|
| FP32 | [K3](opt_kernel/opt_dwconv/summary/benchmark_fp32_k3.png) · [K5](opt_kernel/opt_dwconv/summary/benchmark_fp32_k5.png) · [K7](opt_kernel/opt_dwconv/summary/benchmark_fp32_k7.png) · [K9](opt_kernel/opt_dwconv/summary/benchmark_fp32_k9.png) · [K11](opt_kernel/opt_dwconv/summary/benchmark_fp32_k11.png) |
| FP16 | [K3](opt_kernel/opt_dwconv/summary/benchmark_fp16_k3.png) · [K5](opt_kernel/opt_dwconv/summary/benchmark_fp16_k5.png) · [K7](opt_kernel/opt_dwconv/summary/benchmark_fp16_k7.png) · [K9](opt_kernel/opt_dwconv/summary/benchmark_fp16_k9.png) · [K11](opt_kernel/opt_dwconv/summary/benchmark_fp16_k11.png) |

SKILL 最终版使用相同矩阵单独完成 sweep，并将初版 CSV 中的 `custom` 重命名为
`initial`，只保留 `initial/skill_final/cuDNN NCHW` 三个系列：

| 数据类型 | 最终版原始 CSV | 三方合并 CSV |
|----------|----------------|--------------|
| FP32 | [benchmark_skill_final_fp32.csv](opt_kernel/opt_dwconv/summary/benchmark_skill_final_fp32.csv) | [benchmark_skill_comparison_fp32.csv](opt_kernel/opt_dwconv/summary/benchmark_skill_comparison_fp32.csv) |
| FP16 | [benchmark_skill_final_fp16.csv](opt_kernel/opt_dwconv/summary/benchmark_skill_final_fp16.csv) | [benchmark_skill_comparison_fp16.csv](opt_kernel/opt_dwconv/summary/benchmark_skill_comparison_fp16.csv) |

| 数据类型 | SKILL 对比图 |
|----------|--------------|
| FP32 | [K3](opt_kernel/opt_dwconv/summary/benchmark_fp32_skill_k3.png) · [K5](opt_kernel/opt_dwconv/summary/benchmark_fp32_skill_k5.png) · [K7](opt_kernel/opt_dwconv/summary/benchmark_fp32_skill_k7.png) · [K9](opt_kernel/opt_dwconv/summary/benchmark_fp32_skill_k9.png) · [K11](opt_kernel/opt_dwconv/summary/benchmark_fp32_skill_k11.png) |
| FP16 | [K3](opt_kernel/opt_dwconv/summary/benchmark_fp16_skill_k3.png) · [K5](opt_kernel/opt_dwconv/summary/benchmark_fp16_skill_k5.png) · [K7](opt_kernel/opt_dwconv/summary/benchmark_fp16_skill_k7.png) · [K9](opt_kernel/opt_dwconv/summary/benchmark_fp16_skill_k9.png) · [K11](opt_kernel/opt_dwconv/summary/benchmark_fp16_skill_k11.png) |

完整 sweep 也补充暴露了代表配置报告没有覆盖的范围：

- FP32 K3/K5/K7/K11 的 288 个配置没有超过 2% 的回退；K9 未做专用优化，
  72 个配置中有 35 个相对初版回退超过 2%，最差为 `0.843x`。因此 FP32
  最终 dispatch 不能表述为对所有 kernel size 都提升；相对 cuDNN 的最差配置
  为 K9 N32 C32 H80 的 `0.958x`。
- FP16 的 360 个配置均未回退超过 2%；相对初版的最低值为 `0.984x`。
  相对 cuDNN 的最低值为 `1.176x`。
- 初版、最终版和 cuDNN 使用相同 shape 和 CUDA Event 聚合计时，但由不同
  executable 顺序测量；完整 sweep 中的 cuDNN 只计 convolution、不含 bias，
  对 cuDNN 更有利。

### FP32 K3 H160 三条曲线重合分析

[FP32 K3 SKILL 对比图](opt_kernel/opt_dwconv/summary/benchmark_fp32_skill_k3.png)
中，C64/C128/C256、H160 的大 batch 配置上，`initial`、`skill_final` 和
`cuDNN NCHW` 收敛到约 `0.82-0.84 TFLOP/s`。原始 CSV 的 latency 也相互接近，
因此这不是绘图或数据合并错误。

K3 的名义算术强度约为 `0.9 FLOP/Byte`，远低于 RTX 4090 的 FP32 ridge point
`40.95 FLOP/Byte`。H160、stride 2 时，忽略很小的 weight 和 bias，单次工作集为：

```text
bytes ≈ N * C * (160^2 + 80^2) * sizeof(float)
      = N * C * 128000
```

当前 RTX 4090 的 L2 为 72 MiB，查询到的理论显存带宽为 `1008.1 GB/s`：

| `N*C` | 近似工作集 | 缓存状态 |
|------:|-----------:|----------|
| 512 | 62.5 MiB | 基本可以放入 L2 |
| 1024 | 125 MiB | 明显超过 L2 |

因此三个子图分别在 C64 N16、C128 N8、C256 N4 开始进入相同的 DRAM-bound
区间，即共同满足 `N*C>=1024`。代表点如下：

| 配置 | Initial ms | SKILL final ms | cuDNN ms |
|------|-----------:|---------------:|---------:|
| K3 N16 C64 H160 | 0.144712 | 0.142459 | 0.142838 |
| K3 N16 C128 H160 | 0.284741 | 0.282416 | 0.282675 |
| K3 N32 C256 H160 | 1.125050 | 1.121922 | 1.122210 |

这些点对应约 `906-935 GB/s` 的有效带宽，即理论带宽的约 `90%-93%`，也与
此前 NCU 中 K3 final 的 DRAM throughput `93.77%` 一致。初版和最终版需要搬运
近似相同的 input/output bytes；SKILL 消除的动态地址计算、循环控制和部分 cache
latency 在 DRAM 饱和后不再决定总时间。cuDNN 在该范围也形成了接近最小显存
流量的数据流，因此三者达到同一个带宽 roofline。

本 benchmark 连续 100 次处理同一组 buffer。工作集不超过 72 MiB 时会获得热
L2 cache 收益；超过 L2 后才回落到 DRAM roofline，所以图中会出现明显的容量
拐点。该重合表示大尺寸 K3 已接近当前数据流的合理硬件上限，而不是 SKILL
优化失效。若需要评估单次冷启动场景，应增加样本间 L2 flush 或轮换多组 buffer，
并与当前 steady-state 结果分开报告。

复现命令：

```bash
python3 op/dwconv/plot_benchmark_fp32.py \
    --csv docs/opt_kernel/opt_dwconv/summary/benchmark_results_fp32.csv \
    --out-dir docs/opt_kernel/opt_dwconv/summary
python3 op/dwconv/plot_benchmark_fp16.py \
    --csv docs/opt_kernel/opt_dwconv/summary/benchmark_results_fp16.csv \
    --out-dir docs/opt_kernel/opt_dwconv/summary

./build/opt_conv2d_groups_fp32 --sweep-csv \
    docs/opt_kernel/opt_dwconv/summary/benchmark_skill_final_fp32.csv
python3 op/dwconv/plot_benchmark_fp32.py \
    --csv docs/opt_kernel/opt_dwconv/summary/benchmark_results_fp32.csv \
    --skill-csv \
    docs/opt_kernel/opt_dwconv/summary/benchmark_skill_final_fp32.csv \
    --merged-csv \
    docs/opt_kernel/opt_dwconv/summary/benchmark_skill_comparison_fp32.csv \
    --out-dir docs/opt_kernel/opt_dwconv/summary

./build/opt_conv2d_groups_fp16 --sweep-csv \
    docs/opt_kernel/opt_dwconv/summary/benchmark_skill_final_fp16.csv
python3 op/dwconv/plot_benchmark_fp16.py \
    --csv docs/opt_kernel/opt_dwconv/summary/benchmark_results_fp16.csv \
    --skill-csv \
    docs/opt_kernel/opt_dwconv/summary/benchmark_skill_final_fp16.csv \
    --merged-csv \
    docs/opt_kernel/opt_dwconv/summary/benchmark_skill_comparison_fp16.csv \
    --out-dir docs/opt_kernel/opt_dwconv/summary
```

## 代码、报告与分支状态

- FP32 实现：[opt_conv2d_groups_fp32.cu](../op/dwconv/opt_conv2d_groups_fp32.cu)
- FP16 实现：[opt_conv2d_groups_fp16.cu](../op/dwconv/opt_conv2d_groups_fp16.cu)
- SKILL：[opt_kernel.md](../.claude/skills/opt_kernel/opt_kernel.md)
- FP32 完整报告：[RTX 4090 FP32 优化报告](opt_kernel/opt_dwconv/opt_kernel_conv2d_groups_rtx4090_20260711.md)
- FP16 完整报告：[RTX 4090 FP16 优化报告](opt_kernel/opt_dwconv/opt_kernel_conv2d_groups_fp16_rtx4090_20260712.md)
- `feature/opt_kernel`：SKILL 工作线，远端最新为 `492bfd5`。
- `feat/opt_dwconv`：FP32 优化工作线，远端最新为 `ad7d3bd`。
- `feat/dwconv_fp16_opt`：FP16 优化工作线，本地最新为 `2d88b73`，当前尚未推送。

`[SUCCESS]`



---

## 简历表述

### CUDA Depthwise Convolution 算子与优化工作流 [4090D | CUDA 13.1 | cuDNN 9.17.1]

`CUDA C++` · `cuDNN` · `Nsight Compute` · `Compute Sanitizer` · `CMake`

#### 项目定位

面向多 kernel size、多 batch/channel/spatial shape 的 FP32/FP16 DWConv，实现优于 cuDNN 的 CUDA kernel，
并构建可复用的 `/opt_kernel` Agent Skill，形成从硬件识别、稳定基线、瓶颈分析、结构性实验、条件 dispatch 到最终验证和报告交付的完整优化闭环。

#### 核心贡献

- 自实现初版：FP32 采用 256-thread spatial 映射、每个 block 计算四个 channel、shared weight staging、向量化访存；FP16 扩展为 `half2` 计算，相应每个 block 处理八个 channel。
- 编写 `/opt_kernel` Skill，固化原生 SM 构建、三组 median 与噪声判定、单变量实验、有效优化独立 commit 和 sanitizer 验证等规范，形成可复现的 CUDA kernel 优化工作流：
- 使用 NCU 的 DRAM/SM throughput、occupancy、指令数和 scoreboard 分析不同 workload 的性能限制，针对性采用编译期 kernel 特化、小 grid channel 映射、16×16 shared-input staging 及运行时 dispatch/fallback，并通过多配置回归确定各优化的适用范围。

#### 量化成果

- 自实现初版在代表配置上已超过 cuDNN NCHW：FP32 加速 `1.084x-2.788x`，FP16 加速 `1.194x-4.063x`；在此基础上继续使用 `/opt_kernel` 进行二次优化：
- FP32，最高相对初版加速 `2.724x`；代表配置相对 cuDNN 达到 `1.597x-4.423x`，K3 DRAM throughput 达到 `93.77%`。
- FP16，相对初版最高加速 `3.556x`，相对 cuDNN 达到 `1.176x-5.465x`。
