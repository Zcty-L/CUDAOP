# conv2d_4x128x256_groups_kernel 阶段 0-4 报告

## 执行范围

- 日期：2026-07-10。
- 分支：`feature/opt_kernel`。
- 目标文件：`op/dwconv/opt_conv2d_groups_fp32.cu`。
- 目标 kernel：`conv2d_4x128x256_groups_kernel`。
- CMake target：`opt_conv2d_groups_fp32`。
- 执行到阶段 4；不执行阶段 5 最终验证。
- 用户允许在 Laptop GPU 动态性能噪声下继续探索。

本报告只包含本轮重新采集的数据。旧执行记录、旧 cuDNN 对比和旧实验数据已删除。

## 代码与测试清理

目标文件现在只包含一个 `__global__` kernel。已删除文件内的 cuDNN、
`biasopt`、`db`、NHWC、CSV 全组合等其他实现和测试。

保留的测试能力：

- CPU depthwise convolution 独立正确性参考。
- throughput、main、common、aligned、boundary 五个固定 case。
- 固定随机种子 `20260703`。
- FP32 容差 `atol=1e-4`、`rtol=1e-4`。
- CUDA Event：持续预热 1000 ms、20 个 warmup batch、100 个正式样本、
  每个 case 3 组。
- 独立 `--profile` 入口。

重新分析时发现自适应 `launches_per_sample` 会使基线与实验采用不同批量，
产生假收益。因此批量已冻结：

| 配置 | launches/sample |
|------|----------------:|
| throughput | 16 |
| main | 64 |
| common | 512 |
| aligned | 512 |
| boundary | 1024 |

## 环境

- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU。
- Compute Capability：12.0，46 SM。
- 编译架构：`sm_120`。
- CUDA Driver / Runtime：13.2 / 13.2。
- Nsight Compute：2026.1.1.0。
- Shared Memory / SM：100 KB。
- Registers / SM：65536。
- 估算显存带宽：672.05 GB/s。
- FP32 ridge point：28.24 FLOPs/Byte。
- GPU 为 Laptop 型号，动态功耗和时钟造成明显组间波动。

CMake 修复：

- 删除不存在的 `op/dwconv/conv2d_groups_fp32.cu` 源条目。
- 注册已有 `op/device_query.cu` 为 `device_query` target。

## 基准 kernel

编译资源：

- 40 registers/thread。
- 2064 B static shared memory。
- 0 B dynamic shared memory。
- 0 B stack frame。
- 0 spill stores / loads。
- 1 个 barrier 资源。

Launch：

```text
block = 256
grid = (ceil(outHW / 256), C / 4, N)
```

throughput 配置 grid 为 3584 blocks，在 46 SM、最多 6 blocks/SM 下为
12.99 waves/SM。

### 正确性

| 配置 | max_abs_error | max_rel_error | errors |
|------|--------------:|--------------:|-------:|
| throughput | 2.384186e-06 | 2.222222e-01 | 0/3276800 |
| main | 2.861023e-06 | 9.057971e-02 | 0/819200 |
| common | 1.430511e-06 | 1.288660e-03 | 0/51200 |
| aligned | 4.768372e-07 | 2.066116e-03 | 0/131072 |
| boundary | 4.768372e-07 | 1.814882e-03 | 0/15488 |

五个配置全部通过组合容差。较大的最大相对误差来自接近零的参考值。

### 固定批量性能基线

| 配置 | median ms（3 组） | mean ms（3 组） |
|------|-------------------|-----------------|
| throughput | 0.359506 / 0.379334 / 0.452004 | 0.374024 / 0.425894 / 0.449542 |
| main | 0.080037 / 0.082108 / 0.078448 | 0.079605 / 0.081991 / 0.079792 |
| common | 0.012642 / 0.011605 / 0.010351 | 0.012568 / 0.012344 / 0.010507 |
| aligned | 0.011022 / 0.011473 / 0.011291 | 0.011251 / 0.011546 / 0.011528 |
| boundary | 0.010712 / 0.011137 / 0.010398 | 0.010982 / 0.011329 / 0.010834 |

throughput 三组 median 波动约 25.7%。用户允许继续探索，但所有性能结论
只能标记为暂定。

## 重新瓶颈分析

本节仅使用恢复基准后重新采集的
`conv2d_4x128x256_groups_kernel_reanalysis.ncu-rep`。

### NCU 指标

| 指标 | 数值 |
|------|-----:|
| Duration | 322.08 us |
| Compute (SM) Throughput | 77.17% |
| Memory Throughput（复合） | 74.59% |
| DRAM Throughput | 43.20% |
| L1/TEX Throughput | 75.73% |
| L2 Throughput | 48.47% |
| Memory Throughput | 186.47 GB/s |
| L1/TEX Hit Rate | 82.56% |
| L2 Hit Rate | 73.58% |
| Theoretical / Achieved Occupancy | 100.00% / 96.68% |
| Active / Eligible Warps per Scheduler | 11.60 / 3.79 |
| No Eligible | 27.87% |
| Branch Efficiency | 100.00% |
| Predicated-on threads / warp | 28.72 |

### 静态分析

- 每线程计算同一空间位置的 4 个 channel。
- K=7 时循环执行 52 个 FMA tap，最后 3 个 tap 使用零权重，属于尾部无效工作。
- throughput 有效计算约 324.4 MFLOPs；理论最小 global 流量约 65.6 MB，
  理论最小算术强度约 4.95 FLOPs/Byte，低于 FP32 ridge point。
- 按每输出重复读取 52 个 input float 估算，input load 请求约 681.6 MB；
  这是静态 load 请求量，不是 DRAM 实测流量。
- stride=2 时相邻线程主要按 8 B 间隔读取 input，四个 channel 又相隔
  `inHW`，不适合直接做连续向量 load。
- 权重一次写入 shared 后不再修改；循环同步从数据依赖上看可能冗余。
- throughput 的 `outHW=1600`，x-grid 容量 1792，约 10.7% tail 线程不写
  output，但仍执行地址计算和部分 load/FMA。

### 瓶颈分类

- 主瓶颈：计算与 memory pipe 的混合发射压力。
- 次要瓶颈：调度等待、input 重复加载和同步开销。
- 不是纯 DRAM-bound：DRAM throughput 只有 43.20%。
- 不是 occupancy-bound：achieved occupancy 为 96.68%。
- 不是 launch-bound：单次 kernel 约 0.32 ms。
- 置信度：中等。NCU 证据稳定，但 CUDA Event 动态噪声较大。

## 高级机制适配性

| 机制 | 当前条件 | 判断 |
|------|----------|------|
| register prefetch / software pipeline | 有 load latency，但当前 40 registers 且已有较高 occupancy | 后置；寄存器增长风险高 |
| shared input tile / multi-stage | 邻近窗口有复用，但 256 个线性输出跨行，tile 不规则 | 仅设计评估 |
| `cp.async` | 当前只有约 2 KB weight 搬运，且只搬一次 | 数据量不足，不实验 |
| vectorized global load | stride=2，channel 间隔为 `inHW` | 连续性不满足，不实验 |
| cache hint / ldg variant | L1/L2 hit 已为 82.56%/73.58% | 预期低，后置 |
| TMA / block cluster | 无大粒度规则 tile 或跨 block 共享 | 不满足收益条件 |

已核对 `references/cuda-architecture-features.md`。`sm_120` 支持部分高级搬运
机制，但当前数据流不满足使用条件。

## 阶段 4 实验

### 实验 1：删除循环内 barrier

- 单变量：删除 `k` 循环中 weight shared load 后的 `__syncthreads()`。
- 正确性：五个配置全部 PASS。
- 资源：40 registers、2064 B shared、0 spill，不变。

| 配置 | baseline median ms | experiment median ms | baseline mean ms | experiment mean ms |
|------|--------------------|----------------------|------------------|--------------------|
| throughput | 0.359506 / 0.379334 / 0.452004 | 0.384964 / 0.415928 / 0.408910 | 0.374024 / 0.425894 / 0.449542 | 0.419381 / 0.431862 / 0.438250 |
| main | 0.080037 / 0.082108 / 0.078448 | 0.080208 / 0.078229 / 0.079007 | 0.079605 / 0.081991 / 0.079792 | 0.081578 / 0.079736 / 0.079432 |

结果在三组间混合升降，median 与 mean 没有一致改善。决策：`REVERT`。

在冻结批量之前，此实验曾表现出约 15%–20% 的表面提升；当时基线和实验的
批量不同，该结果已判定为测试框架伪差，不纳入优化结论。

### 实验 2：在初始同步前加载 bias

- 单变量：使用 shared 数组末尾预留的 4 个 float 保存 bias，让初始 weight
  同步同时发布 bias，删除末尾 bias 同步。
- 正确性：五个配置全部 PASS。
- 资源：40 registers、2064 B shared、0 spill，不变。

| 配置 | baseline median ms | experiment median ms | baseline mean ms | experiment mean ms |
|------|--------------------|----------------------|------------------|--------------------|
| throughput | 0.359506 / 0.379334 / 0.452004 | 0.340768 / 0.393164 / 0.341730 | 0.374024 / 0.425894 / 0.449542 | 0.353759 / 0.388005 / 0.334830 |
| main | 0.080037 / 0.082108 / 0.078448 | 0.076145 / 0.078067 / 0.079970 | 0.079605 / 0.081991 / 0.079792 | 0.076529 / 0.079212 / 0.081254 |

部分组改善，但 throughput median 有一组回退，main 也没有跨三组一致改善；
收益未稳定超过约 25.7% 的基线噪声。决策：`REVERT`。

### 实验 3：K=3/5/7 编译期特化

- 单变量：将 kernel size 从运行时参数改为模板常量，K=3/5/7 专门实例化，
  其他尺寸保留通用 fallback。
- 正确性：五个配置全部 PASS。
- 资源：所有实例均为 40 registers、2064 B shared、0 spill。
- throughput median：`0.378270 / 0.350988 / 0.405690 ms`。
- main median：`0.077923 / 0.087586 / 0.087116 ms`。

K=7 throughput 没有一致改善，main 后两组明显回退；K=3 小配置也不稳定。
决策：`REVERT`。

### 实验 4：128-thread block 与权重加载重排

- 单变量：block 从 256 改为 128，每线程协作加载 4 个 weight，减少 x-tail
  无效线程。
- throughput tail 比例从约 10.7% 降到约 3.8%，总线程数减少约 7.1%。
- 正确性：五个配置全部 PASS。
- 资源：40 registers、2064 B shared、0 spill。
- throughput median：`0.403174 / 0.373034 / 0.417478 ms`。
- main median：`0.078277 / 0.078713 / 0.078847 ms`。

tail 减少的收益被 block 数和 weight 重复加载增加抵消，主要配置没有可靠提升。
决策：`REVERT`。

### 实验 5：每线程双输出与输入窗口并集复用

- 单变量：128-thread block 中每线程计算相邻两个输出；K=7、stride=2 时，
  两个窗口按 9 列并集加载，替代原本 14 列独立加载，理论减少约 35.7%
  input load。
- 正确性：五个配置全部 PASS。
- 资源：40 registers、2064 B shared、0 spill。
- throughput median：`0.945810 / 0.935836 / 0.882566 ms`。
- main median：`0.205480 / 0.204384 / 0.199723 ms`。

动态循环、分支与标量 shared weight 读取成本远高于减少的 input load，主要配置
退化约 2.5 倍。决策：`REVERT`。

### 实验 6：8x20 二维 spatial tile 与 channel-major shared staging

- 保留原 kernel，新增 `conv2d_4x8x20_shared_spatial_kernel`。
- 每个 160-thread block 计算 `8x20` 输出 tile，将四通道 input tile 和 weight
  搬入 shared memory。
- K=7、stride=2 时 input tile 为 `21x45x4`，动态 shared 为 17184 B。
- 正确性：五个配置全部 PASS。
- 资源：37 registers、0 spill、1 barrier。
- throughput median：`0.500850 / 0.471594 / 0.476484 ms`。
- 同轮 baseline median：`0.421924 / 0.422760 / 0.445028 ms`。

input staging 降低了 DRAM/L2 压力，但标量 shared load、低 occupancy 和分支成本
导致稳定回退。决策：`REVERT`，保留实验 kernel 继续验证邻近 shared 布局。

### 实验 7：四通道交错 shared layout 与 lds128

- shared input 改为 spatial-major、四通道交错布局。
- shared weight 改为 tap-major、四通道交错布局。
- 每 tap 用两条 `lds128` 取代 8 条标量 shared load。
- 正确性：五个配置全部 PASS。
- 资源：42 registers、17184 B dynamic shared、0 spill。

第一次独立 A/B，K=7 throughput：

| 实现 | median ms（3 组） | mean ms（3 组） |
|------|-------------------|-----------------|
| baseline | 0.422512 / 0.437072 / 0.450014 | 0.462457 / 0.452958 / 0.455373 |
| shared-spatial | 0.409760 / 0.399588 / 0.380230 | 0.414827 / 0.402837 / 0.393082 |

K=7 throughput 和 main 均整体改善；K=3 回退、K=5 结果混合。因此只对
`K=7、stride=2` 启用 shared-spatial，其他配置回落 baseline。

第二次独立 A/B，条件 dispatch 后 K=7 throughput：

| 实现 | median ms（3 组） | mean ms（3 组） |
|------|-------------------|-----------------|
| baseline | 0.435750 / 0.473828 / 0.494348 | 0.504869 / 0.510368 / 0.515387 |
| shared-spatial | 0.395724 / 0.379862 / 0.379978 | 0.401389 / 0.384616 / 0.387358 |

第二次复测仍一致改善。决策：`KEEP`，作为 K=7 条件路径继续调优。

### 实验 8：10x20 spatial tile

- 将输出 tile 从 `8x20` 调整为 `10x20`，block 从 160 调整为 200。
- 对 40x40 输出，spatial grid 从 10 blocks 降为 8 blocks，均无输出 tail。
- 动态 shared：20064 B；资源：42 registers、0 spill。
- NCU theoretical/achieved occupancy：58.33% / 56.48%。
- 正确性：五个配置全部 PASS。

第三次独立 A/B，K=7 throughput：

| 实现 | median ms（3 组） | mean ms（3 组） |
|------|-------------------|-----------------|
| baseline | 0.417244 / 0.430826 / 0.433680 | 0.453313 / 0.442584 / 0.444949 |
| 10x20 shared-spatial | 0.402652 / 0.390646 / 0.375064 | 0.402647 / 0.392430 / 0.373455 |

throughput median 提升约 `3.5% / 9.3% / 13.5%`，mean 提升约
`11.2% / 11.3% / 16.1%`。main median 提升约 `3.0% / 13.0% / 6.9%`。
相较 `8x20`，throughput 接近、main 更好。决策：暂定 `KEEP`。

预热 NCU 中，10x20 kernel Duration 为 437.12 us，baseline 为 321.92 us；
NCU 固定时钟单 launch 与 Laptop 动态功耗下重复 CUDA Event 的结论不一致。
因此收益只适用于当前设备的重复预热负载，仍需阶段 5 锁频复测。

### 实验 9：8x16 warp-aligned spatial tile

- 将 block 调整为 `8x16=128` threads，消除非 warp 整数倍 block。
- 增加 `32x32`、`48x48`、`22x22` 输出，避免只针对 `40x40` 调参。
- K=7 动态 shared 为 14496 B；资源为 42 registers、0 spill。
- 全部配置正确性 PASS。

多尺寸独立 A/B 结果：

| 输出 | 结果 |
|------|------|
| 40x40 throughput | 变化混合，只有部分组小幅改善 |
| 40x40 main | 第一组回退，另外两组改善 |
| 32x32 | median 稳定回退约 3%-5% |
| 48x48 | 基本持平、结果混合 |
| 22x22 | 有改善，但绝对时间很短且噪声较大 |

`128 threads` 虽满足 warp 对齐，但每个 block 的 staging 成本分摊不足，不能作为
通用替换。决策：`REVERT`。

### 实验 10：16x16 warp-aligned spatial tile

- 使用 `16x16=256` threads，每线程计算一个输出位置。
- K=7、stride=2 时动态 shared 为 23968 B；资源为 42 registers、0 spill。
- 全部配置正确性 PASS。

多尺寸独立 A/B 的三组结果：

| 输出 | baseline median ms | 16x16 median ms | 结论 |
|------|--------------------|-----------------|------|
| 40x40 throughput | 0.451064 / 0.473652 / 0.399560 | 0.378780 / 0.395196 / 0.372326 | 三组提升约 6.8%-16.6% |
| 40x40 main | 0.081850 / 0.091887 / 0.092514 | 0.076317 / 0.080487 / 0.077942 | 三组提升约 6.8%-15.8% |
| 32x32 | 0.049831 / 0.049555 / 0.049791 | 0.043538 / 0.043949 / 0.045124 | 三组提升约 9%-13% |
| 48x48 | 0.108794 / 0.111916 / 0.107725 | 0.090350 / 0.086570 / 0.084804 | 三组提升约 17%-23% |
| 22x22 | 0.013860 / 0.010604 / 0.013155 | 0.013819 / 0.013218 / 0.010795 | 结果混合，第二组明显回退 |

因此不能按固定 `40x40` 输出选择 tile，也不能仅凭 block 是 warp 整数倍判定
性能。`16x16` 在已测的大输出 `32/40/48` 上均改善，而 `8x16` 没有；收益来自
warp 对齐、staging 成本分摊、block 数量和 occupancy 的共同作用。

预热 NCU 中，16x16 Duration 为 412.13 us，资源为 42 registers、23968 B
dynamic shared、0 spill，theoretical/achieved occupancy 为 66.67% / 62.17%。
固定时钟单 launch 仍慢于 baseline 的 321.92 us，与重复 CUDA Event 结论不一致。
决策：对 `K=7、stride=2、out_h/out_w >= 32` 暂定 `KEEP`；较小输出及其他
配置回退原 kernel。

## 阶段 4 结论

| 轮次 | 改动 | 正确性 | 资源 | 决策 |
|-----:|------|:------:|------|:----:|
| 0 | 原始 kernel + 固定 benchmark 批量 | PASS | 40 regs / 2064 B / 0 spill | KEEP baseline |
| 1 | 删除循环 barrier | PASS | 不变 | REVERT |
| 2 | 初始同步预加载 bias | PASS | 不变 | REVERT |
| 3 | K=3/5/7 编译期特化 | PASS | 不变 | REVERT |
| 4 | 128-thread block 与权重加载重排 | PASS | 不变 | REVERT |
| 5 | 双输出输入并集复用 | PASS | 不变 | REVERT |
| 6 | 8x20 channel-major shared staging | PASS | 37 regs / 17184 B | REVERT |
| 7 | 四通道交错 shared layout + lds128 | PASS | 42 regs / 17184 B | KEEP K=7 |
| 8 | 10x20 shared-spatial tile | PASS | 42 regs / 20064 B | 被实验 10 替代 |
| 9 | 8x16 warp-aligned tile | PASS | 42 regs / 14496 B | REVERT |
| 10 | 16x16 warp-aligned tile | PASS | 42 regs / 23968 B | TENTATIVE KEEP |

当前 target 使用条件 dispatch：`K=7、stride=2、out_h/out_w >= 32` 使用
16x16 shared-spatial，其他配置使用原 kernel。该条件来自当前测试边界，不假定
实际输出一定是 `40x40`；尚未覆盖的新尺寸默认走 fallback。原 kernel 仍是有效
fallback 和同轮性能参考。

本轮状态：`[TENTATIVE_KEEP]`。由于按用户要求只执行到阶段 4，尚未运行阶段 5
的 compute-sanitizer、锁频统一复测和最终 NCU 对比。

## 硬件上限与最佳性判断

### 当前结论

当前 kernel 可以称为“在现有 NCHW、每线程四通道输出、256-thread 线性映射
及已验证策略下的局部最优”，但不能称为“达到硬件合理上限”，也不能证明是
全局最佳。

原因：

- CUDA Event 的 throughput 三组 median 波动约 25.7%，测量稳定性不足。
- NCU 没有显示任一主要硬件单元接近明确 ceiling。
- 文件内 cuDNN 已按要求删除，目前没有同语义、同布局、同融合范围的独立
  性能参考实现。
- 前五个实验无收益，但二维 tile + 四通道交错 shared layout 在当前 Laptop GPU
  的重复预热负载下获得暂定收益；这仍不能证明不存在更优 tile 或数据布局。

### 不能直接用 SM Throughput 或 Occupancy 判定最佳

NCU 的 throughput 是其组成计数器中最高的利用率，不等于取得相同比例的 FP32
理论峰值。`Compute Throughput=77.17%` 必须展开到具体 FP32、INT、LSU 等
pipe 后才能解释。NVIDIA 对 throughput metric 的定义见
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure)。

同样，`Achieved Occupancy=96.68%` 只说明驻留 warp 数量充足。高 occupancy
不保证执行单元被有效使用；当前 `No Eligible=27.87%` 表明约 28% scheduler
周期没有可发射 warp，延迟隐藏仍不完整。

| 指标 | 当前值 | 对上限判断的意义 |
|------|-------:|------------------|
| Compute Throughput | 77.17% | 较忙，但未接近 90%；需展开 pipe breakdown |
| Memory Throughput（复合） | 74.59% | 不能当作 DRAM 带宽利用率 |
| DRAM Throughput | 43.20% | 没有达到显存带宽 ceiling |
| L1/TEX Throughput | 75.73% | L1/LSU 压力较高，但仍有明显距离 |
| L2 Throughput | 48.47% | 没有达到 L2 ceiling |
| Achieved Occupancy | 96.68% | 驻留充分，不代表吞吐最优 |
| Issued warp/scheduler | 0.72 | 仍有未使用的 issue slot |
| No Eligible | 27.87% | 存在依赖或延迟隐藏不足 |
| Predicated-on threads | 28.72/32 | 约 10% lane 工作被 predicate 屏蔽 |

### 有效性能与简单 Roofline

throughput 配置：

- 输出元素：`16 * 128 * 1600 = 3,276,800`。
- 有效计算：每输出约 `49 FMA + bias = 99 FLOPs`。
- 总有效计算：约 324.4 MFLOPs。
- NCU Duration：322.08 us。
- 有效性能：约 1.01 TFLOP/s。

`device_query` 粗估 FP32 峰值为 18.98 TFLOP/s，因此有效 FLOP 约为粗估
峰值的 5.3%。这不表示 kernel 能直接加速约 19 倍，因为当前还包含整数地址
计算、同步、load 和无效 tap；但它说明 77.17% Compute Throughput 不能解释
为“取得 77.17% FP32 峰值”。

按理论最小 global 流量约 65.6 MB 和算术强度约 4.95 FLOPs/Byte 估算，
简单 DRAM Roofline 约为：

```text
672 GB/s * 4.95 FLOPs/Byte ~= 3.33 TFLOP/s
```

当前约 1.01 TFLOP/s，为该乐观 ceiling 的约 30%。但 input 有大量 L1/L2
命中，实际限制可能位于 L1、LSU、整数 pipe 或调度依赖，因此必须使用 L1、
L2、DRAM 分层 Roofline。Roofline achieved point 到相应 ceiling 的距离才代表
对应层级的优化空间，见
[NVIDIA Roofline 分析说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)。

### “达到合理上限”的项目判据

以下阈值是本项目建议采用的工程标准，不是 NVIDIA 强制标准。

#### 1. 测量可信

- 固定 GPU 时钟，或将三组 median 波动控制在 2% 内。
- baseline/final 使用相同预热、固定 batch、正式样本数和 stream。
- 至少 3 次独立进程复测。
- 测试期间温度、功耗、SM clock 和 memory clock 没有明显漂移。

当前 25.7% 波动不满足该条件，所以即使出现约 10% 的变化也不能直接判定为
稳定收益。

#### 2. 找到并接近唯一主 ceiling

至少有一个条件成立：

- 主要 FP32/FMA pipe 达到约 85%-90% sustained peak；
- 或 L1、L2、DRAM 中某一层达到约 85%-90% ceiling；
- 或 scheduler issue 接近上限，主要 stall 为 `Not Selected`；
- 或极短 kernel 已接近独立测得的 launch latency 下限。

当前 SM 77%、L1 76%、DRAM 43%、issued warp 0.72，不满足上述条件。

#### 3. 建立实际时间下界

```text
T_lower_bound = max(
    FP32 instructions / FP32 issue capacity,
    INT/address instructions / ALU capacity,
    LDST instructions / LSU capacity,
    L1 bytes / L1 bandwidth,
    L2 bytes / L2 bandwidth,
    DRAM bytes / DRAM bandwidth,
    necessary synchronization latency,
    launch latency
)
```

当实际时间只比该下界高约 10% 时，才可认为接近硬件合理上限。当前报告还
缺少完整的 FP32/INT/LDST 指令数量、pipe breakdown、各层实际字节数、
transaction efficiency 和 shared bank conflict 数据，因此还不能建立可信下界。

#### 4. 使用独立参考实现

参考实现不需要放回目标 `.cu` 文件，可以建立单独 benchmark：

- cuDNN Frontend fused depthwise convolution + bias；
- CUTLASS/CuTe 专用实现；
- Triton 或 PyTorch 编译实现；
- 针对固定 shape 的独立手写 kernel。

必须保证输入布局、bias 融合、精度、stream 和 kernel-only 计时范围一致。
建议只有同时满足以下条件，才写“达到合理上限”：

- 与最佳公平参考的差距不超过约 5%，或快于参考；
- 距相应 Roofline ceiling 不超过约 10%；
- 多轮 block、tile、数据流搜索没有稳定超过 2% 的收益；
- 全部正确性和业务 shape 无回退。

### 还需采集的 NCU 数据

本机 NCU 2026.1 提供 `roofline`、`ComputeWorkloadAnalysis`、
`InstructionStats`、`MemoryWorkloadAnalysis_Tables` 等 section。建议运行：

```bash
ncu -f \
  --section SpeedOfLight_HierarchicalSingleRooflineChart \
  --section ComputeWorkloadAnalysis \
  --section InstructionStats \
  --section MemoryWorkloadAnalysis_Tables \
  --section SchedulerStats \
  --section WarpStateStats \
  --section SourceCounters \
  --kernel-name regex:conv2d_4x128x256_groups_kernel \
  --launch-count 1 \
  --kill yes \
  -o build/conv2d_4x128x256_groups_kernel_limit \
  ./build/opt_conv2d_groups_fp32 --profile 7 16 128 80
```

需要回答：

1. `Compute Throughput=77.17%` 由 FP32、INT、LSU 还是其他 pipe 决定？
2. achieved point 距 L1、L2、DRAM 和 FP32 ceiling 分别多远？
3. 每输出执行了多少非必要 INT、LD/ST、branch 和 predicated-off 指令？
4. `No Eligible=27.87%` 主要来自 long scoreboard、barrier、wait 还是
   math throttle？
5. 实际 L1/L2/DRAM bytes 与理论最小流量相差多少？

在完成上述上限审计和公平参考对比之前，正式结论应保持：

> 当前 K=7 大输出条件路径在重复预热负载下优于原 kernel，属于暂定局部最优；
> 尚未证明达到硬件合理上限，也不能宣称为全局最佳。

## 对 opt_kernel.md 的新增建议

1. 实验 0 自适应得到的 `launches_per_sample` 必须写入冻结配置；后续实验不得
   重新自适应，否则批量变化会制造假收益。
2. 用户明确指定工作分支时，应允许覆盖默认 `feat/{op_name}_opt` 规则。
3. 用户明确移除 cuDNN/CUTLASS 时，应允许跳过性能参考，但必须保留独立
   正确性 reference。
4. `device_query` 前应先探测 CMake target，缺失时允许注册仓库已有源或使用
   明确的替代查询。
5. 为动态时钟设备增加 `strict/explore` 模式，并规定 explore 的配对测试、
   噪声估计和暂定 KEEP 标准。
6. 明确持续预热时长、组间波动公式和最多重试次数。
7. 相对误差报告应同时记录对应 reference 的绝对值，避免近零值造成误解。
8. 增加“性能上限审计”阶段：要求分层 Roofline、pipe breakdown、时间下界和
   公平参考对比，避免仅凭 occupancy 或 SpeedOfLight 百分比宣称最佳。

[TENTATIVE_KEEP]
