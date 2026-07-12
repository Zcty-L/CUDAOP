# conv2d_4x128x256_groups_kernel 优化报告

## 2026-07-11 RTX 4090 平台复核（阶段 0-6）

### 执行范围与状态

- 目标文件：`op/dwconv/opt_conv2d_groups_fp32.cu:37`。
- 目标 kernel：`conv2d_4x128x256_groups_kernel`。
- CMake target：`opt_conv2d_groups_fp32`。
- 已完成阶段 0-6；最终状态为 `[SUCCESS]`。
- 初次检查时项目固定编译架构为 `sm_120`，当前 GPU 为 `sm_89`，因此当时没有
  运行 CUDA Event 基线或 Nsight Compute profile。随后已将构建规则改为默认
  查询当前 GPU、用户显式参数优先，并验证目标可按 `sm_89` 编译。本轮已补跑
  CUDA Event、Nsight Compute 和单变量实验；没有实验达到 KEEP 阈值。

### 阶段 0：预检

#### 仓库与构建

- 当前分支：`feature/opt_kernel`，跟踪 `origin/feature/opt_kernel`。
- 预检时工作树无已跟踪文件修改。
- 目标文件未引用已弃用的 `op/cuda_utils.cuh`，未使用 `printf`。
- CMake 已注册 `opt_conv2d_groups_fp32` 和 `device_query` target，并保留
  `--ptxas-options=-v`。
- CUDA Toolkit 13.1（NVCC 13.1.80），Nsight Compute 2025.4.0.0。
- 初次构建产物只针对 `sm_120`；更新构建规则后已按查询值重新构建为 `sm_89`。

#### 当前 GPU

- 两张 NVIDIA GeForce RTX 4090，Compute Capability 8.9，每张 128 SM。
- 每卡约 24 GB 显存、72 MB L2、100 KB shared memory/SM、65536
  registers/SM。
- 默认 shared memory/block 为 48 KB，opt-in 上限为 99 KB。
- 实测查询时两卡 GPU utilization 均为 2%，memory utilization 为 0%，温度
  为 38/36 摄氏度，SM clock 为 2520 MHz，memory clock 为 10501 MHz；
  查询时没有持续高负载迹象。
- GPU 不是 Laptop 或 Max-Q 型号。
- 项目 `device_query` 对 `sm_89` 的估算有误：`cores_per_sm()` 将所有 8.x
  架构按 64 CUDA cores/SM 计算，因而输出 41.28 TFLOP/s 和
  40.95 FLOPs/Byte ridge point。RTX 4090 实际为 16384 CUDA cores，按查询到
  的 2520 MHz 和 1008.10 GB/s 估算，FP32 峰值约 82.58 TFLOP/s，ridge
  point 约 81.91 FLOPs/Byte。该修正值只用于静态分析，后续仍应以 NCU 的
  sustained peak 模型为准。

#### 架构迁移结果

初次检查时 `CMakeLists.txt` 固定为：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 120)
```

现在顶层 CMake 默认查询当前 GPU 的 compute capability，并提供
`CUDAOP_CUDA_ARCHITECTURES` 作为用户显式覆盖参数；`/opt_kernel` 会先查询实际
测试 GPU，再以查询值重新配置，避免旧 cache 沿用其他平台的 SM。RTX 4090 上
已验证 ptxas 的编译目标为 `sm_89`。

### 阶段 1：基线与测试矩阵审计

#### 编译资源

RTX 4090 的 `sm_89` 编译资源为：

- baseline kernel：40 registers/thread、2064 B static shared memory、1 barrier、
  0 spill stores/loads。
- shared-spatial kernel：37 registers/thread、动态 shared memory 由输入决定、
  1 barrier、0 spill stores/loads。

后续仍需用 occupancy API 或 NCU 核对 active blocks/SM。

#### 噪声阈值审计

skill 第 1.4 节当前规则是：

- 20 次 warmup、100 个正式样本；
- 同一配置至少重复 3 个完整 group；
- group median 波动超过 2% 时先排查负载、温度、时钟和缓存状态；
- 预热后仍超过 2% 则阻塞性能实验。

结论：**暂不修改 2% 阈值**。旧报告中的 25.7% 波动来自 RTX 5070 Ti
Laptop 的动态功耗环境，不能迁移为桌面 RTX 4090 的新阈值。当前 4090 查询时
低负载、固定在 2520 MHz，2% 作为进入实验阶段的上限是合理且偏宽松的工程
门槛。架构修复后应先实测新基线噪声，再决定是否需要更严格的阈值。

为避免“波动”定义含糊，后续建议统一记录：

```text
group_median_spread =
    (max(group_median) - min(group_median)) / median(group_median)
```

建议保留 2% 作为阻塞阈值；若新平台连续三次独立进程测得 spread 小于 0.5%，
实验的 KEEP 条件仍按 skill 的至少 2% 提升执行，不因单次低噪声而降低接受门槛。

现有批量计时设置可保留：不同 case 的单个 CUDA Event 内累计约数毫秒，能降低
极短 kernel 的 Event 分辨率和 launch 抖动影响。baseline 与 shared-spatial
虽分开预热和测试，但执行顺序恒为 baseline 在前；正式复测应支持独立
`--impl` 进程，并增加正序/逆序复测以检查温度和缓存顺序偏差。

#### 最大吞吐配置

baseline launch 为：

```text
block = 256
grid = (ceil(outHW / 256), C / 4, N)
```

按当前 40 registers/thread、2064 B shared memory 和 RTX 4090 的资源上限静态
估算，每个 SM 最多驻留 6 个 block，即 48 warps，线程和寄存器共同形成上限。

| 配置 | grid blocks | 估算 active blocks/SM | full waves | 判断 |
|------|------------:|-------------------------:|-----------:|------|
| 当前 throughput：K7/N16/C128/H80/S2 | 3584 | 6 | 4.67 | 不足 8-10 waves |
| 建议 throughput：K7/N32/C128/H80/S2 | 7168 | 6 | 9.33 | 满足建议范围 |
| 当前 main：K7/N4/C128/H80/S2 | 896 | 6 | 1.17 | 仅作业务配置 |

因此新平台的主要吞吐 case 应把 `N` 从 16 提高到 32。其 input 约 100 MiB、
output 约 25 MiB，加上权重、bias 和 CPU/GPU reference 缓冲，远低于 24 GB
显存容量。若后续 `sm_89` 编译后的 active blocks/SM 不再是 6，应按实际 occupancy
重新计算 N，而不是固定沿用 32。

#### 特殊用例覆盖

现有矩阵已覆盖：K=3/5/7、stride=2、对齐 `outHW=1024`、非对齐
`outHW=484/1600/2304`、小 grid、K7 输出 22/32/40/48，以及固定随机种子和
FP32 组合容差。

仍缺少以下重要覆盖：

| 类别 | 建议 case | 目的 |
|------|-----------|------|
| 最大支持卷积核 | K11/N1/C32/H43/S2 | 覆盖 `KhKw=121` 和最后一个四 tap 尾组 |
| stride=1 | K3/N1/C32/H43/S1 | 验证地址计算与高输入复用路径 |
| 最小通道组 | K7/N1/C4/H43/S2 | 覆盖 `grid.y=1` 和 launch-bound 小网格 |
| 非 2 的幂通道 | K7/N1/C36/H43/S2 | 覆盖多 channel group 的尾部规模 |
| dispatch 下边界 | K7/N1/C32/H61/S2，out=31 | 验证回落 baseline |
| dispatch 上边界 | K7/N1/C32/H63/S2，out=32 | 验证进入 shared-spatial |
| K7 尾线程 | K7/N1/C32/H65/S2，out=33 | 同时覆盖 shared-spatial 与 x-tail |
| 矩形输入 | K7/N1/C32/H65/W81/S2 | 验证 `out_h != out_w` 与跨行地址计算 |

当前 `CaseConfig` 只有一个 `h` 字段，无法构造矩形输入；正式阶段 1 应先扩展测试
入口，但不改变目标 kernel 语义。kernel 还隐含要求 `C % 4 == 0` 且
`KhKw <= 128`，测试入口应显式拒绝不满足条件的配置，避免把越界或漏算当成
普通精度失败。

本次架构规则修改后只验证了编译，没有继续运行 RTX 4090 正确性、CUDA Event
latency 或参考实现；旧 Laptop 数据不作为新平台 baseline。

### 阶段 2：暂定瓶颈分析

#### 静态数据流

以 K=7 为例，每个有效空间线程计算 4 个 channel：

- 有效计算为 49 tap × 4 channel × 2 FLOPs，加 4 次 bias，约 396 FLOPs/thread。
- 实际循环按 4 tap 展开到 52 tap，最后 3 tap 的权重为零，仍执行地址计算、
  input 读取和 FMA，tap 维度无效工作约 5.8%。
- 不考虑缓存复用时，每线程 input 请求约 52 × 4 × 4 = 832 B，另有 16 B
  output；weight 和 bias 由 block 共享，摊销后较小。
- 相邻线程在 stride=2 时主要按 8 B 间隔读同一 channel，四个 channel 又相隔
  `inHW`，不能直接形成自然的四 float 连续向量 load。
- block 内相邻输出窗口有显著 input 重叠，但 baseline 只把约 2 KB weight 放入
  shared memory，没有复用 input。
- `outHW=1600` 时 x 维分配 1792 个线程，约 10.7% 线程不写 output，但这些线程
  仍执行地址计算、循环和部分 input/FMA 工作。
- 40 registers/thread 在 256-thread block 下刚好允许静态估算的 6 blocks/SM；
  寄存器上升到 43 以上可能把驻留降到 5 blocks/SM，预取实验必须重点观察。

按无缓存的 load 请求计算，算术强度低于 0.5 FLOPs/Byte；按跨线程理想复用后的
理论最小 global 流量计算会明显更高。两者分别是请求侧上界与乐观下界，均不是
profiler 实测 DRAM 字节数。修正后的 RTX 4090 FP32 ridge point 约
81.91 FLOPs/Byte，因此静态上不可能把该 kernel 判为纯 compute-bound。

#### 暂定分类

- 主瓶颈假设：input load 请求、地址生成和 L1/LSU 发射压力共同限制。
- 次要瓶颈假设：窗口重复加载、K=7 尾 tap 无效工作、x-tail 和同步延迟。
- Resource-bound：当前 baseline 静态 occupancy 可达 100%，暂不支持该分类；
  但增加寄存器或 shared memory 后容易成为次要限制。
- Launch-bound：throughput case 不是；最小通道和小输出 case 可能是。
- 置信度：低。旧 `sm_120` Laptop 的 NCU 数据只作为历史参考，不能证明
  `sm_89` RTX 4090 的当前瓶颈。

架构修复后应先用 NCU basic 收集 SpeedOfLight、LaunchStats 和 Occupancy，再按
假设补充 MemoryWorkloadAnalysis、SchedulerStats、WarpStateStats、
InstructionStats 和 SourceCounters；未采集前不把复合 `Memory Throughput`
直接解释为 DRAM 带宽利用率。

### 阶段 3：制定计划

#### 高级机制适配性判断

| 机制 | 适用条件 | 当前 kernel 是否满足 | 预期收益 | 风险 | 是否实验 |
|------|----------|----------------------|----------|------|----------|
| register prefetch / software pipeline | 下一轮 load 可与本轮计算重叠，寄存器有余量 | 有 49-tap 循环可重叠，但 40 regs 已接近保持 6 blocks/SM 的上限 | 降低 long scoreboard，约 2%-8% | 超过 42 regs 可能降为 5 blocks/SM | 是，后置 |
| shared input tile / multi-stage | block 内输入复用显著，shared 与同步成本可控 | 相邻卷积窗口复用显著；仓库已有 16x16 单 stage 实现，但只在旧 GPU 暂定 KEEP | 降低 L1/L2 请求，潜在 5%-20% | 约 24 KB shared 降低 occupancy，边界搬运成本高 | 是，优先重新验证并调 tile |
| `cp.async` / async copy | 较大的 global-to-shared 搬运可与计算重叠 | `sm_89` 支持；baseline 只有约 2 KB 一次性 weight 搬运，不足，shared-spatial 虽约 24 KB 但当前没有跨 tile pipeline | 单纯替换预期不足 2% | source 为 channel-major、shared 为交错布局，难以向量化且同步复杂 | 暂不实验；仅当 NCU 证明 staging 指令/延迟主导时转为实验 |
| vectorized global load | 地址连续、对齐且边界成本可控 | stride=2 空间访问和跨 channel 间隔均不满足连续四 float | 收益不足以覆盖重排 | 需要改变线程映射或 staging 布局 | 否 |
| cache hint / ldg variant | 只读语义明确，且 cache 行为是瓶颈证据之一 | input/weight 只读，但尚无 `sm_89` cache 指标 | 潜在 1%-5% | `.nc`/L1 策略可能增加 L2/DRAM 压力 | 是，取得 NCU 证据后 |
| TMA / block cluster | `sm_90+`，且大 tile 或跨 block 复用值得协作 | RTX 4090 为 `sm_89`，架构不支持 | 无 | 无法编译或运行 | 否 |

补充判断：Tensor Core/WGMMA 不列入实验。WGMMA 要求 `sm_90a`，RTX 4090
不支持；该 FP32 depthwise 计算还缺少可高效映射的大 K/C 矩阵结构，打包成本和
数值语义风险高。

#### 前置条件

进入任一实验前必须完成以下非 kernel 工作：

1. 已完成：按查询值将本次构建设为 `sm_89`，重新构建并记录 ptxas 资源。
2. 将 throughput 的 N 调为 32，并补齐上表中的必测特殊用例。
3. 用 CPU reference 跑完整正确性矩阵。
4. 至少三组 CUDA Event 基线，确认 `group_median_spread <= 2%`。
5. 重新采集 `sm_89` NCU basic 与定向 section，冻结实验 0。

#### 实验 1：重新验证现有 shared-spatial 条件路径

- 假设：RTX 4090 上 baseline 的重复 input load 仍是主要成本，16x16 shared
  input staging 能减少 L1/L2/LSU 压力。
- 改动：不改算法，单独比较现有 `launch_custom` 与现有 16x16
  shared-spatial 路径；先把“是否保留旧平台实现”作为一个平台迁移变量。
- 预期：K7 大输出提升 5%-15%，L1/L2 load 请求下降；若没有稳定收益，则删除
  或禁用旧平台条件 dispatch，恢复纯 baseline 作为新平台实验起点。
- 风险：约 23968 B dynamic shared memory 将静态 occupancy 降至约 4 blocks/SM；
  旧 Laptop 的暂定收益可能不适用于 Ada。
- 验收：全部正确性 case 通过；throughput/main 的 median 与 mean 三组均改善
  至少 5%，其他必测配置不回退；2%-5% 仅在三组稳定时暂定保留。

#### 实验 2：K=7 精确 tap 数特化

- 假设：运行 52 tap 而有效 tap 只有 49，使约 5.8% 的 tap 地址计算、input load
  和 FMA 无效，Ada 的整数/LSU 发射压力会放大这一成本。
- 改动：只对 K=7 路径使用编译期精确尾部处理，其他 K 保持通用 fallback。
- 预期：指令数和 input load 请求下降，性能提升约 2%-6%。
- 风险：尾部代码增加指令体积；旧 Laptop 上模板特化未获稳定收益。
- 验收：所有 K 配置正确；K7 throughput/main 稳定提升至少 2%，K3/K5/K11
  fallback 不回退，ptxas 无 spill。

#### 实验 3：bias 与初始 weight 同步合并

- 假设：bias 可在初始 weight staging 时写入 shared，并由同一个初始 barrier
  发布，从而删除 kernel 尾部 barrier。
- 改动：只调整 bias load 时机并复用初始同步，不改变主循环。
- 预期：减少一次 block barrier，提升约 1%-3%，小 grid case更敏感。
- 风险：shared 地址重叠或发布顺序错误；旧 Laptop 结果受噪声影响未被接受。
- 验收：全部正确性 case 通过；主要配置稳定提升至少 2%，无资源增长和回退。

#### 实验 4：shared-spatial tile 单变量搜索

- 假设：`16x16` 是针对旧 46-SM GPU 选择的 tile；128-SM Ada 上 shared
  占用、block 数和 staging 分摊的最优点不同。
- 改动：每轮只改变一个 tile shape，候选按 `8x16`、`10x20`、`16x16` 比较，
  其他布局、向量 load 和 dispatch 条件冻结。
- 预期：在保留 input 复用的同时提高 active blocks 或减少 tile 数，较当前最佳
  再提升 2%-10%。
- 风险：边缘 tile 无效线程、非 warp 整数倍 block、动态 shared memory和同步成本。
- 验收：K7 的 out=32/33/40/48 都正确；吞吐和 main 稳定改善，任何必测大输出
  稳定回退超过 2% 即拒绝该 tile。

#### 实验 5：受寄存器约束的 software pipeline

- 假设：若 NCU 显示 long scoreboard/no eligible 较高，可预取下一组 input tap，
  与当前组 FMA 重叠。
- 改动：只增加一阶段 register prefetch，不同时改变 tile 或 cache hint。
- 预期：eligible warps 和 issue slot 利用率提高，提升约 2%-8%。
- 风险：寄存器超过 42/thread 后可能从 6 降到 5 blocks/SM，抵消延迟隐藏收益。
- 验收：无 spill；若寄存器导致 occupancy 降档，必须由 NCU 证明 stall 降低且
  端到端仍稳定提升至少 5%，否则拒绝。

#### 实验 6：只读 cache 策略

- 假设：若 `sm_89` NCU 显示 L1 miss、L2/DRAM 压力或 replay 主导，调整 input
  的只读/cache hint 可改善实际 transaction 行为。
- 改动：每轮只比较一种 input load cache 策略；新增 PTX 时统一放到
  `op/ptx_utils.cuh` 并注明官方来源和用途。
- 预期：降低相关 cache miss 或 replay，提升约 1%-5%。
- 风险：工作集和 case 不同会导致相反效果，可能污染 cache 或增加 DRAM 流量。
- 验收：对应 NCU 指标按假设改善；主要配置稳定提升至少 2%，所有业务和边界
  case 不回退。

### 阶段 4：RTX 4090 单变量实验（2026-07-11）

#### 测试矩阵落地

进入实验前先扩展非 kernel 测试入口：

- throughput 从 `N=16` 调整为 `N=32`，grid 为 7168 blocks，NCU 实测
  `Waves Per SM=9.33`；批量计时固定为每样本 8 次 launch。
- `CaseConfig` 增加独立的 `w`，能够验证矩形输入。
- 新增 K11 尾 tap、stride=1、C=4、C=36、dispatch out=31/32/33、矩形输入。
- 新增 `--correctness-only` 和 `--case <name>`，实验轮次可以先跑完整正确性，
  再只测主要性能配置。

16 个配置对 CPU reference 全部通过，最大绝对误差不超过 `3.814697e-06`，
错误元素均为 0。最终保留版本的编译资源仍为 40 registers/thread、2064 B
static shared memory、0 spill。

#### 实验 1：复核旧平台 16x16 shared-spatial 路径

- 假设：二维 shared input staging 可以在 Ada 上减少重复 input load。
- 正确性：16 个配置全部 PASS。
- 资源：37 registers/thread、23968 B dynamic shared、0 spill。
- NCU：baseline 为 202.53 us，shared-spatial 为 221.50 us，回退 9.4%。
- occupancy：baseline achieved 96.02%，shared-spatial 为 62.49%。
- 小业务配置也稳定回退：out=32 约 5.4%，out=33 约 3.4%，矩形配置约 3.6%。

虽然 `N=4` 的 32x32/40x40/48x48 输出可见局部收益，但主要吞吐 N=32 不提升，
且多个必测特殊配置回退超过 2%。决策：`REVERT`，删除旧平台实验 kernel 和
条件 dispatch。

#### 实验 2：K7 尾部无效 input load

- 单变量：K7 最后一组 4 taps 中，仅对有效的第 49 个 tap 读取 input，跳过
  其余 3 个零权重 tap 的 input load。
- 正确性：16 个配置全部 PASS。
- 资源：40 registers/thread、2064 B shared、0 spill，与 baseline 相同。

| 配置 | baseline median ms（3 组） | 实验 median ms（3 组） |
|------|-----------------------------|-------------------------|
| throughput | 0.210432 / 0.210284 / 0.202624 | 0.198784 / 0.201468 / 0.202368 |
| main | 0.026142 / 0.026064 / 0.025648 | 0.024144 / 0.024368 / 0.024704 |
| out32 | 0.015072 / 0.015072 / 0.015072 | 0.014784 / 0.014736 / 0.014736 |
| out48 | 0.031616 / 0.031553 / 0.031392 | 0.030752 / 0.030393 / 0.030403 |

业务配置改善，但 throughput 第三组只改善约 0.1%；NCU 仅从 202.53 us 改善到
201.82 us，提升约 0.35%。主要配置没有稳定超过 2% 接受阈值。决策：`REVERT`。

#### 实验 3：bias 预加载并合并同步

- 单变量：把 bias 写入 shared 数组预留的末尾 4 个 float，复用初始 weight
  staging 的 barrier，删除 kernel 尾部 barrier。
- 正确性：16 个配置全部 PASS。
- 资源：40 registers/thread、2064 B shared、0 spill，与 baseline 相同。

| 配置 | baseline median ms（3 组） | 实验 median ms（3 组） |
|------|-----------------------------|-------------------------|
| throughput | 0.210432 / 0.210432 / 0.201856 | 0.200704 / 0.204928 / 0.209152 |
| main | 0.026144 / 0.026144 / 0.025728 | 0.024828 / 0.025072 / 0.025422 |
| aligned K3 | 0.003184 / 0.003184 / 0.003184 | 0.003010 / 0.003010 / 0.003008 |
| C=4 | 0.007293 / 0.007293 / 0.007294 | 0.006913 / 0.006913 / 0.006913 |

小配置稳定改善，但 throughput 三组出现混合升降；NCU 仅从 202.53 us 改善到
201.34 us，提升约 0.59%。主要配置没有稳定超过 2% 接受阈值。决策：`REVERT`。

#### 阶段 4 决策

| 轮次 | 单变量 | 正确性 | 主要证据 | 决策 |
|-----:|----------|:------:|----------|:----:|
| 0 | RTX 4090 原始 kernel | PASS | NCU 202.02 us | KEEP baseline |
| 1 | 16x16 shared-spatial | PASS | NCU 回退 9.4%，特殊配置回退 | REVERT |
| 2 | K7 跳过无效尾 load | PASS | NCU 提升 0.35% | REVERT |
| 3 | bias 与初始同步合并 | PASS | NCU 提升 0.59% | REVERT |
| 4 | 10x20 shared-spatial | PASS | throughput 回退 4%-9%，out32 回退约 29% | REVERT |
| 5 | 删除循环内冗余 barrier | PASS | NCU 提升 2.76%，必测配置均改善 | KEEP |
| 6 | 128-thread block | PASS | out32 回退最高约 3%，throughput 混合 | REVERT |
| 7 | 在实验 5 上合并 bias 同步 | PASS | 增量 NCU 仅 0.15%，Event 混合 | REVERT |
| 8 | 通用 8x16 合并加载 shared-input | PASS | K5/K7 回退，K11 Event 提升 | 被实验 9 替代 |
| 9 | 通用 16x16 K11 shared-input | PASS | K11 NCU 提升 5.04%，多尺寸均改善 | KEEP K11 |

实验 2 和实验 3 后曾按旧停止规则结束；停止规则修订后继续验证结构性方案，得到
实验 5 的有效优化。

#### 实验 4：10x20 shared-spatial

- 资源：38 registers/thread、20064 B dynamic shared、0 spill。
- throughput candidate median 为
  `0.217432 / 0.217436 / 0.217344 ms`，同轮 baseline 为
  `0.208000 / 0.208932 / 0.199928 ms`。
- main 提升约 19.6%，但 out32、out48、矩形和主要 throughput 均稳定回退。

减少 shared memory 和 40x40 的 tile 数仍无法覆盖 staging、低 occupancy 和额外
grid 成本。决策：`REVERT`。

#### 实验 5：删除循环内冗余 barrier

weight 只在循环前写入 shared，循环内只读，因此相邻 4-tap 组之间不需要 barrier。
阶段 5 的 `racecheck` 进一步发现，原代码在循环结束后把 bias 写回 weight 区起始
地址，可能覆盖仍被其他 warp 读取的最后一组 weight。最终实现把 bias 放到声明时
已预留的末尾 4 个 float，消除该别名竞争；不恢复循环内 barrier。资源保持
40 registers/thread、2064 B shared、0 spill。

| 指标 | 原始 baseline | 实验 5 / final |
|------|--------------:|----------------:|
| NCU Duration | 202.02 us | 196.45 us |
| Compute Throughput | 75.57% | 77.22% |
| Eligible Warps/Scheduler | 3.91 | 4.12 |
| No Eligible | 22.97% | 21.19% |

首次 A/B 中，main 三组提升约 8.4%-10.2%，out32 提升约 4.1%-5.0%，out48
提升约 4.7%-5.6%；K3、K5、K11、stride=1、最小通道、非 2 次幂通道、
dispatch 边界和矩形输入均稳定改善。throughput 三次独立 A/B 的方向一致，但
组内仍存在约 3% 漂移；NCU 提升 2.76%。决策：`KEEP`。

#### 实验 6：128-thread block

保持实验 5，block 从 256 改为 128，并让每个 block 两轮 staging weight。
两种实例均为 40 registers/thread、2064 B shared、0 spill。out32 第一组回退约
3%，out48 无收益，throughput 随组次反转。额外 block 和 weight staging 抵消
了尾线程减少。决策：`REVERT`。

#### 实验 7：在实验 5 上合并 bias 同步

完整正确性通过，aligned K3 有局部收益，但 throughput/main 结果混合；NCU 从
196.93 us 变为 196.64 us，增量仅约 0.15%。决策：`REVERT`。

#### 实验 8：通用 8x16 shared-input

该实现不使用固定 H/W：tile grid 根据运行时 `out_h/out_w` 生成。global staging
按 channel-major 连续读取，再写为 spatial-major 四通道交错 shared 布局，计算
阶段用 `lds128` 读取四通道。

- 资源：37 registers/thread；K7 dynamic shared 为 14496 B，K11 为 18464 B。
- K5/K7 在 throughput、main、out22/32/48、dispatch 边界和矩形输入上均回退。
- K11 在 H=43、H=64、H=80 和矩形输入上有局部收益。
- K11 NCU theoretical/achieved occupancy 仅为 41.67%/40.67%，grid 为
  15360 blocks；NCU Duration 452.74 us，慢于 baseline 442.08 us。

结论：input 复用只在 K11 足以覆盖 staging 成本，但 8x16 block 的 warp 数和
shared 限制导致 occupancy 太低。转入实验 9，不保留 8x16。

#### 实验 9：通用 16x16 K11 shared-input

只把 tile 从 8x16 改为 16x16，global 合并加载和 shared 交错布局保持不变。
dispatch 仅判断 `Kh=Kw=11` 和 shared 资源是否可用，不判断 H/W、N 或固定输出
尺寸；其他 kernel size 使用实验 5。

| 配置 | baseline median ms（3 组） | target median ms（3 组） |
|------|-----------------------------|--------------------------|
| K11 N32 H80 | 0.429696 / 0.411008 / 0.429680 | 0.368000 / 0.366588 / 0.370068 |
| K11 N4 H80 | 0.053392 / 0.052268 / 0.053388 | 0.048085 / 0.048080 / 0.048080 |
| K11 N4 H64 | 0.032304 / 0.032240 / 0.031408 | 0.027105 / 0.027104 / 0.027108 |
| K11 N1 H43 | 0.013228 / 0.013228 / 0.013228 | 0.011077 / 0.011079 / 0.011080 |
| K11 N1 65x81 | 0.013392 / 0.013392 / 0.013392 | 0.011302 / 0.011302 / 0.011302 |

NCU Duration 从 442.08 us 降到 419.81 us，提升 5.04%。target 使用 37
registers/thread、28960 B dynamic shared、0 spill，theoretical/achieved
occupancy 为 50.00%/46.77%。K11 多尺寸 Event 提升约 8.0%-16.4%，没有固定
shape 回退。决策：`KEEP K11`。

#### 阶段 4 继续优化（2026-07-12）

实验 9 的 NCU 进一步定位到两类具体问题：input staging 的 global `LDG` 为
6666240 sectors、理想值 5403648；计算阶段 input `LDS.128` 为理想 wavefronts 的
2 倍，而 input staging scalar `STS` 约为理想值的 4 倍。继续执行以下结构实验：

| 轮次 | 单变量 | 主要结果 | 决策 |
|-----:|--------|----------|:----:|
| 10 | shared 行宽补齐并做 bank swizzle | shared excessive 降至 7.52M，但 shared 33.55 KB 使 occupancy 50%→33%，Event 回退约 18% | REVERT |
| 11 | 只增加一个 padding vector 的 compact swizzle | 计算阶段全部 `LDS.128` 达到 ideal，NCU 419.65→382.72 us | KEEP |
| 12 | warp 固定 channel/row 的 global staging | global excessive 1.35M→0.50M，但 divergent branches 增至 1.55M，Event 回退约 3% | REVERT |
| 13 | 每 spatial 读取四通道并用 `sts128` staging input | 指令 251.12M→143.82M，throughput 约 0.343→0.220 ms | KEEP |
| 14 | weight 四通道 `sts128` staging | throughput 增量约 1.4%，低于接受阈值 | REVERT |
| 15 | 仅对 shared 路径编译期特化 K=11 | 指令 143.82M→94.74M，throughput 约 0.220→0.202 ms | KEEP |
| 16 | input 使用只读 `.L2::128B` load | L2 hit 52.56%→77.45%，throughput 约 0.202→0.194 ms | KEEP |
| 17 | 在新数据流上重测 8x16 tile | throughput 约 0.197 ms，慢于 16x16 的约 0.194 ms | REVERT |

compact swizzle 使用 `x ^ ((x >> 3) & 1)` 交换奇数 8-column chunk 内的相邻列，
shared row 只增加一个 vector，K11/S2 动态 shared 为 29616 B，仍允许 3 blocks/SM。
它不依赖固定输入或输出尺寸。input staging 改为每线程处理一个 spatial 位置、读取
四个 channel，并以一条 `sts128` 写入交错布局；因此边界和地址计算只做一次，
同时保留各 channel 在 warp 内的连续 global 访问。

`cp.async` 没有继续实现：source 为四个相距 `inHW` 的 channel，destination 为
交错 float4，单条 async copy 不能完成该转置；改回 channel-major shared 会把当前
已达到 ideal 的 `LDS.128` 拆成更多 shared 指令。手写 software pipeline 也没有
继续，因为最终 SASS 已把多个后续 `LDS.128` 提前并与当前 tap 的 FMA 交错。

### 阶段 5：最终验证

#### 构建与资源

- 使用 CMake 按查询到的架构重新构建，ptxas 明确输出 `sm_89`。
- fallback kernel：40 registers/thread、2064 B static shared、1 barrier、0 spill。
- K11 shared-input kernel：43 registers/thread、K11/S2 动态 shared 29616 B、
  K11/S1 动态 shared 13296 B、1 barrier、0 spill。
- 最终源码仅保留接受的 fallback 与 K11 shared-input 两条路径；dispatch 只依赖
  `Kh == 11 && Kw == 11`，不依赖固定 H/W、N 或输出尺寸。

#### 正确性与 sanitizer

- 最终代码重新运行全部 21 个冻结配置，CPU reference 均通过，错误元素为 0，
  最大绝对误差为 `5.245209e-06`。
- `compute-sanitizer 2025.4.0 memcheck` 覆盖全部 21 个正确性配置：0 errors。
- `racecheck` 与 `synccheck` 使用 K7 fallback 矩形配置，以及 K11 shared-input
  的矩形和 stride=1 配置，最终均为 0 errors/hazards。
- 首次 `racecheck` 曾报告 fallback 末尾 weight 读取与 bias 写入之间有 768 个
  hazards。修复为 bias 使用共享内存预留尾部后，重新完成编译、21 配置正确性、
  三类 sanitizer 和完整性能矩阵，未隐藏该验证问题。

#### 最终 CUDA Event 结果

latency 取 3 个 group median 的中位数；`Speedup = Baseline / Final`。同语义 GPU
参考库未在阶段 1 建立，因此 Reference 明确记为 `N/A`，CPU reference 只用于
正确性，不参与 GPU latency 对比。

| 配置 | Baseline group medians ms | Final group medians ms | 降低 | Speedup | Final spread | Reference ms |
|------|--------------------------:|-----------------------:|-----:|--------:|-------------:|-------------:|
| K11 N32 H80 throughput | 0.429696 / 0.413692 / 0.429568 | 0.195200 / 0.195200 / 0.195200 | 54.56% | 2.201x | 0.00% | N/A |
| K11 N4 H80 main | 0.053680 / 0.052589 / 0.053696 | 0.025504 / 0.025493 / 0.025502 | 52.49% | 2.105x | 0.04% | N/A |
| K11 N4 H64 out32 | 0.032381 / 0.031952 / 0.031472 | 0.014208 / 0.014208 / 0.014222 | 55.53% | 2.249x | 0.10% | N/A |
| K11 N1 H43 tail | 0.013252 / 0.013252 / 0.013252 | 0.005112 / 0.005112 / 0.005112 | 61.43% | 2.592x | 0.00% | N/A |
| K11 N1 65x81 rectangular | 0.013480 / 0.013480 / 0.013480 | 0.005276 / 0.005276 / 0.005277 | 60.86% | 2.555x | 0.02% | N/A |
| K11 N1 H43 stride1 | 0.012748 / 0.012751 / 0.012748 | 0.005260 / 0.005260 / 0.005260 | 58.74% | 2.424x | 0.00% | N/A |

baseline 的 K11 throughput 中间组仍有约 3.7% 的可复现漂移，但 final 三组完全
一致；逐组最小收益超过 52%，远大于基线漂移，不影响接受结论。

非 K11 配置的 baseline 与 target 实际调用同一 fallback kernel，A/B 差异只用于
检查测试顺序噪声，不解释为两种实现的性能差。相对阶段 0 冻结的原始 K7 kernel，
最终 fallback 的同条件 NCU Duration 从 202.02 us 降至 196.74 us，提升 2.61%。

#### 最终 Nsight Compute

定向 profile 使用同一 RTX 4090、相同输入、相同 section、跳过前 100 次 launch
并采集第 101 次：

| 路径 | Baseline Duration | Final Duration | 变化 | 关键指标 |
|------|------------------:|---------------:|-----:|----------|
| K7 fallback | 202.02 us | 196.74 us | -2.61% | Compute 75.57%→77.12%，eligible warps 3.91→4.10 |
| K11 shared-input | 439.14 us | 204.10 us | -53.52% | 执行指令 397.38M→90.57M，0 spill |

K11 final 的 L1/TEX throughput 为 84.00%、复合 memory throughput 为 82.32%、
DRAM throughput 为 59.69%，已进入 NCU 的 `>80%` 高利用率区间。shared excessive
为 1.50M/40.88M wavefronts（约 3.7%），计算阶段所有 input/weight `LDS.128`
均达到 ideal；剩余 shared excessive 主要来自一次性 staging store。global
excessive 仍为 1.35M/8.18M sectors（约 16.6%），但 warp-row 方案虽然把它降至
0.50M，却因分支和负载不均衡而回退。当前主要剩余限制是 L1/MIO 发射以及
long/short scoreboard；结合已验证的失败方案，可将当前实现视为该数据流在 RTX
4090 上的合理硬件上限，而不是理论绝对上限。

### 阶段 6：交付

- 接受：删除 fallback 循环内冗余 barrier；隔离共享 bias 尾部；对所有运行时
  H/W 的 K11 使用 16x16 shared-input staging；加入 compact bank swizzle、
  四通道 `sts128` input staging、K11 编译期循环特化和只读 `.L2::128B` load。
- 放弃：旧 16x16 和 10x20 K7 shared-spatial、K7 尾 tap 特化、bias 同步合并、
  128-thread block、padded swizzle、warp-row staging、weight `sts128`、8x16
  shared-input、手写 software pipeline 与 `cp.async`；原因和数据保留在实验记录中。
- 修改文件：`op/dwconv/opt_conv2d_groups_fp32.cu`、
  `.claude/skills/opt_kernel/opt_kernel.md`、本报告。
- 未执行项：同语义 GPU 参考库对比（阶段 1 未建立，记为 `N/A`）；其余阶段 5
  要求的构建、正确性、性能、ptxas、NCU 和 sanitizer 均已执行。

`[SUCCESS]`

---

## 2026-07-10 RTX 5070 Ti Laptop 历史报告（阶段 0-4）

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
