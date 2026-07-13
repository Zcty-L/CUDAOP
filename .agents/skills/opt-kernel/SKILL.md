---
name: opt-kernel
description: >-
  分析并迭代优化 CUDAOP 的 .cu 文件中用户指定的 CUDA kernel，完成目标定位、环境检查、基线冻结、瓶颈定位、
  单变量实验、正确性与性能验证及报告交付。用于 $opt-kernel、CUDA kernel 性能分析、kernel 调优、
  Nsight Compute 分析或要求优化 op/{name} 下 CUDA 算子的任务。
---

# opt-kernel - CUDA Kernel 优化工作流

按“检查、基线、分析、规划、实验、验证、交付”的顺序执行。不得跳过基线正确性验证，也不得用未经测量的经验判断代替性能数据。

## 用法

直接使用自然语言描述优化目标，例如：

```text
$opt-kernel 优化 op/dwconv/example.cu:128 的 dwconv_forward_kernel。
```
能够根据输入定位唯一目标 kernel 的 launch site、launch wrapper 和模板实例化，包括输入、输出、数据类型、布局、形状范围和数值语义，
如果用户没有提供足够的定位信息，先列出文件中的候选 kernel 名称与定义行号，请用户指定目标，不得自行选择。找不到唯一匹配时输出 `[BLOCKED]`。
架构选择使用查询到的当前 GPU 计算能力，除非用户显式指定目标 SM。

## 阶段 0 - 预检

### 0.1 检查仓库

执行并记录：

```bash
git status --short --branch
rg -n "__global__|<kernel名称>" <kernel文件路径>
rg -n "cuda_utils\\.cuh|printf\\s*\\(" <kernel文件路径>
rg -n "<源文件名>|<cmake目标>" CMakeLists.txt
```

修改代码前，从稳定基线为每个 kernel 优化任务创建一个 `feat/opt_{op_name}` 分支。实验轮次不另建分支；每次确认 `KEEP` 后立即创建一个独立的 `perf:` commit，一次 commit 只包含一个有效优化及其必要测试和实验记录。仅分析时无需新建分支。
工作树已有用户修改时，只暂存本任务且能明确归属的文件或 hunk，不得把无关修改带入优化 commit；若与实验改动无法分离，先请求用户处理。
结合定义行号阅读目标 kernel、函数签名、相邻注释及全部 launch site。若目标是模板 kernel，记录被测配置实际使用的模板参数和编译后 kernel 名称。

### 0.2 检查 GPU

先查询 GPU，再用查询到的原生 SM 通过 CMake 构建并运行设备查询：

```bash
nvidia-smi --query-gpu=index,name,compute_cap,utilization.gpu,utilization.memory --format=csv
cmake -S . -B build -DCUDAOP_CUDA_ARCHITECTURES=<查询值>
cmake --build build --target device_query -j
./build/device_query
```

若存在多张不同计算能力的 GPU，先确定 benchmark 实际使用的 device，再使用该 device 的计算能力。
若需要跨设备发布，由用户显式给出架构列表，这与单设备 kernel 优化基线分开处理。

记录：
- GPU 名称、计算能力和 SM 数量；
- GPU 名称是否包含 Laptop、Max-Q 或其他移动端标识；
- Shared Memory / SM、寄存器 / SM、occupancy 上限；
- 显存带宽和各数据类型 ridge point；
- 测试前 GPU 是否存在其他负载。

不能获得 GPU 或 GPU 正被持续占用时，不得给出可信性能结论。可以继续静态分析，但最终状态必须标记 `[BLOCKED]`。

### 0.3 检查测试入口

测试程序必须输出：
测试配置和数据类型、正确性验证阶段、warmup 次数和正式测试次数、latency 统计和参考实现性能、关键结果、成功时的 `[SUCCESS]`。

## 阶段 1 - 冻结基线

基线是第一次不修改 kernel 的完整运行，记为实验 0。如果基线测试不完整，允许修复 CMake 和非目标测试框架，但必须先记录目标 kernel 快照。
允许补充 benchmark 入口参数，如 `--case=<name>`、`--profile-only`、`--impl=<name>`，用于单独运行目标配置或 profile；不得改变目标 kernel 语义。

### 1.1 冻结测试

至少定义：

- 一个吞吐饱和配置，grid 足够大，用于主要性能对比和瓶颈判断；
- 常用业务配置，用于验证真实场景不退化；若 grid 较小，不单独判断硬件利用率上限；
- 对齐尺寸和非对齐尺寸，会触发边界处理的配置；
- 固定随机种子；
- 每种数据类型对应的 `atol` 和 `rtol`。

选择吞吐饱和配置时，记录 grid blocks、active blocks/SM 和 waves/SM；建议至少 8-10 个 full waves，不满足时说明小 grid 或 tail wave 影响。
性能提升和瓶颈判断默认以吞吐饱和配置为主；业务配置用于记录真实场景表现和检查防回退。
若用户指定唯一业务 shape 为优化目标，将其作为主要性能配置，吞吐饱和配置仅作辅助分析。
分别冻结每个配置的基线。测试矩阵用于选择不同条件路径，不要求一个 kernel 实现对所有配置同时最优。

若已有 cuBLAS、cuDNN、CUTLASS 或其他参考实现，在相同输入、布局、精度、stream 和计时范围下测量并记录。参考库不是正确性基线的替代品。

### 1.2 编译基线

```bash
cmake -S . -B build -DCUDAOP_CUDA_ARCHITECTURES=<目标SM>
cmake --build build --target <cmake目标> -j
```

必须读取编译输出中的：registers / thread、static 和 dynamic shared memory、spill stores / loads。
编译命令必须保留 `--ptxas-options=-v`。编译失败时先修复构建，不得进入优化实验。

### 1.3 验证正确性

对测试矩阵逐项比较参考输出，至少报告：最大绝对误差、最大相对误差，错误元素数量/总元素数量；CUDA API、kernel launch 和同步错误。
基线正确性失败时停止并输出 `[BLOCKED]`。优化工作不得掩盖原始正确性问题。

### 1.4 测量性能

严格使用 CUDA Event 测量 kernel，其中warmup 20 次，正式测试 100 次；
报告 100 次结果的 mean、median、min、max 和标准差。极短 kernel 可一次计时批量 launch，但必须除以 launch 次数，并在报告中写明批量大小。批量计时时，warmup 与正式样本使用相同批量大小。
目标实现和参考实现分开预热、分开测试，不在同一组 benchmark 内交替执行。每个实现记录预热时长、批量大小和正式样本数。
同一配置至少重复整组 benchmark 3 次。若组间中位数波动超过 2%，先排查 GPU 负载、温度、时钟和缓存状态，不进入实验阶段。
组间中位数波动统一按以下公式计算：group_median_spread = (max(group_median) - min(group_median)) / median(group_median)，报告百分比时将该结果乘以 100%。
默认不要求锁定 GPU 频率；若发现动态频率、功耗限制或 Laptop GPU 导致波动，在记录和最终报告中注明，不把锁频结果当作默认性能上限。
若动态时钟设备经预热仍无法将组间中位数波动控制在 2% 内，默认标记 `[BLOCKED]`，仅继续静态分析和制定暂定计划。
若用户明确接受噪声阈值策略，可继续探索实验；实验结果只能标记为暂定结论，判定规则见阶段 4。

### 1.5 创建实验记录

```text
docs/opt_kernel/opt_<op_name>/opt_kernel_<kernel_id>_<gpu>_<YYYYMMDD>.md
```

- `kernel_id` 使用简短唯一标识；完整 kernel 名、源码位置写入正文。
- `gpu` 使用具体型号，日期使用基线冻结日。
- 每个 GPU、日期和基线单独建报告；同日新轮次添加 `_02`，历史报告只引用链接。

```text
docs/opt_kernel/opt_dwconv/opt_kernel_conv2d_groups_rtx4090_20260712.md
```

优化期间可使用同名 `/tmp` 草稿；停止或交付前必须写入仓库，且不提交临时 profile。

```markdown
| 轮次 | 假设/改动 | 正确性 | median ms | mean ms | 相对当前基线 | 决策 | Commit |
|-----:|-----------|:------:|----------:|--------:|---------------:|:----:|:------:|
| 0 | baseline | PASS | 0.000000 | 0.000000 | 0.00% | KEEP | abc1234 |
```

## 阶段 2 - 瓶颈分析

必须同时使用代码、编译资源数据和运行指标。算术强度只用于形成初步假设。

### 2.1 静态分析

分析并记录：

- grid、block、动态 shared memory 和 launch 次数；
- 每个输出元素的 FLOPs、整数指令和特殊函数；
- 理论 global memory 读写字节；
- global memory 合并访问、对齐和重复加载；
- shared memory 布局、复用和 bank conflict 风险；
- 分支发散、循环、同步和原子操作；
- 寄存器中间结果的生命周期；
- 边界线程和无效工作比例。

算术强度估算：

```text
I = Total FLOPs / Minimum Global Memory Bytes
```

根据实际数据类型与 `device_query` 的 ridge point 比较。必须说明字节数是理论最小流量还是 profiler 实测流量。

### 2.2 Profile 分析

先收集低开销概览，再根据假设收集详细 section，避免每轮无条件使用 `--set full`：

```bash
ncu -f --set basic --kernel-name regex:<kernel名称正则> \
  --launch-count 1 --kill yes --check-exit-code 0 \
  -o build/<kernel_name>_basic ./build/<cmake目标> <运行参数>
```

每次 profile 必须重新采集；允许使用 `-f` 覆盖旧文件。
先记录 `ncu --version`。不同版本支持的 `--page` 选项可能不同；导出报告时用 `ncu --help` 核对，通常使用 `--page details` 或 `--page raw`，汇总可加 `--print-summary per-kernel`。
记录各层实测吞吐、访问量及相对 `peak_sustained` 的利用率。`% Peak` 的分母来自 NCU 性能模型，不是本次运行实测上限；`Memory Throughput` 是复合指标，不得直接视为 DRAM 带宽利用率。

按需关注：
- `SpeedOfLight`：SM 与 memory throughput；
- `MemoryWorkloadAnalysis`：DRAM、L2、L1/TEX 和 shared memory；
- `LaunchStats`、`Occupancy`：block、warp 和资源限制；
- `SchedulerStats`、`WarpStateStats`：eligible warp 和 stall reason；
- `SourceCounters`：热点源代码与指令。

若 Nsight Compute 不可用，使用 CUDA Event、`ptxas -v` 和静态分析继续，但明确标记证据不完整，进一步提醒用户安装 Nsight Compute。

### 2.3 分类规则

只在证据支持时给出主瓶颈：

| 类型 | 主要证据 |
|------|----------|
| Memory-bound | memory throughput 接近上限，SM throughput 较低，且有效字节流量主导 |
| Compute-bound | 计算管线利用率较高，算术强度高于对应 ridge point |
| Latency-bound | memory 和 compute 均未饱和，eligible warp 低或 stall 明显 |
| Launch-bound | kernel 极短，launch 开销占比较高 |
| Resource-bound | registers、shared memory 或 block 限制 active warps / blocks |

输出“主瓶颈、次要瓶颈、证据、置信度”。不能仅凭 occupancy 低判定性能差，也不能把 occupancy 最大化当作目标。

## 阶段 3 - 制定计划

根据证据生成按优先级排序的实验，不得机械应用优化清单。

每个实验必须写明：

```markdown
### 实验 N：<名称>
- 假设：<哪项指标限制性能>
- 改动：<只描述一个主要变量>
- 预期：<预计改善的指标和大致幅度>
- 风险：<精度、边界、资源或架构风险>
- 验收：<正确性条件和性能条件>
```

优先级原则：

1. 算法和数据流：减少工作量、内存流量或不必要的中间结果。
2. 内存访问：合并访问、复用、布局、向量化和 bank conflict。
3. 并行映射：block、warp、tile 和每线程工作量。
4. 延迟隐藏：指令级并行、预取、异步 global-to-shared copy、双缓冲。
5. 计算单元：warp primitive、Tensor Core 或架构专用指令。
6. 微调：展开、编译属性和较小参数搜索。

制定计划时必须先输出“高级机制适配性判断”，不得只给微调项。至少覆盖：
```markdown
| 机制 | 适用条件 | 当前 kernel 是否满足 | 预期收益 | 风险 | 是否实验 |
|------|----------|----------------------|----------|------|----------|
| register prefetch / software pipeline | 有可重叠的下一轮 load，寄存器余量足够 |  |  |  |  |
| shared input tile / multi-stage | block 内存在显著输入复用，shared 占用和同步成本可控 |  |  |  |  |
| `cp.async` / async copy | global-to-shared 搬运足够大，能与计算重叠 |  |  |  |  |
| vectorized global load | 地址连续、对齐且边界处理成本可控 |  |  |  |  |
| cache hint / ldg variant | 访问复用或只读语义明确，cache 行为是瓶颈证据之一 |  |  |  |  |
| TMA / block cluster | tile 搬运粒度大，存在跨线程块复用或 cluster 协作收益 |  |  |  |  |
```
若判定不实验，必须写明“不满足的具体条件”或“收益不足以覆盖的成本”。若判定实验，必须转化为单变量实验项。

涉及架构专用机制时，先读取 [CUDA 架构特性表](references/cuda-architecture-features.md)，核对目标架构、PTX ISA、CUDA Toolkit、使用约束和 fallback。不得仅因架构支持而应用某项机制。

## 阶段 4 - 单变量实验循环

每轮只验证一个主要假设。“单变量”限制假设数量，不限制改动规模。保持接口、数值语义和任务范围不变时，允许完整重构算法、数据流、线程映射和存储层级：
1. 检查 `git status`，确认不会覆盖用户修改。
2. 记录预期收益和验收条件。
3. 修改允许范围内的代码。
4. 使用 CMake 编译，记录资源变化。
5. 运行完整正确性测试。
6. 运行主要配置 benchmark。
7. 必要时运行定向 Nsight Compute profile。
8. 记录结果并决定 `KEEP`、`REVERT` 或 `CRASH`。
9. `KEEP` 前补齐适用条件、fallback 和 dispatch 边界测试，再重跑完整正确性与必测性能配置。
10. 将一次有效优化及其必要测试、报告记录提交为独立 `perf:` commit；记录短 commit hash，该 commit 成为下一轮基线。

```bash
git commit -m "perf: <kernel名称> <有效优化>"
git rev-parse --short HEAD
```

实现成本较高时，保留当前最近的最佳 kernel，新建以机制命名的实验 kernel 实现复杂方案，并在同一测试中对比。决策后，`KEEP` 替换原实现，`REVERT` 删除实验 kernel。

每轮实验从最近的 `KEEP` commit 开始。失败实验只撤销本轮尚未提交的改动，不得改写已有 `KEEP` commit，也不得使用会覆盖用户修改的 Git 命令。
普通实验不保留 `#ifdef` 基线、旧 kernel 或 `_v2` 变体；最终代码只保留接受的实现，历史由 Git commit、diff 和实验表承担。

多配置下分别判断收益。若某一配置或一类配置获得稳定大幅提升，而其他配置不适合该实现，不得整体撤销；保留优化 kernel，并按运行时参数增加条件 dispatch，让未受益配置回落到最近的已提交基线。
优先用 K、stride、对齐、布局或尺寸范围等可解释条件；只有用户明确把某个固定 shape 列为目标时，才使用精确 H/W 条件。提交前必须补测 dispatch 条件两侧及边界值，确保最终组合在全部必测配置上不回退。

### 决策规则

若环境噪声无法完全消除，先用基线重复组估计噪声范围；只有 median 和 mean 均稳定改善且幅度明显大于噪声，才可暂定 `KEEP` 并要求后续复测。`min` 只作为下界诊断，不作为接受优化的主依据。

| 结果 | 决策 |
|------|------|
| 任一正确性配置失败 | `REVERT` |
| 任一必测配置稳定回退超过 2%，且无法用可靠条件隔离 | `REVERT` |
| 任一配置提升至少 5% 且至少为噪声上界的 3 倍，可由运行时条件可靠选择 | `KEEP` 条件路径，其他配置 fallback |
| 主要配置提升至少 5%，且其他配置不回退 | `KEEP`，作为新基线继续 |
| 提升 2% 到 5%，跨 3 组测试稳定，且最终 dispatch 无必测回退 | `KEEP`，降低后续优先级 |
| 提升不足 2% | 视为噪声，`REVERT` |
| 编译、运行或 profile 失败 | 记录 `CRASH`，修复或尝试下一假设 |

`KEEP` 表示最终 dispatch 组合通过验证，不表示候选 kernel 必须覆盖全部配置。每次 `KEEP` 只提交一次；同一实验为补齐正确性、dispatch 或报告所做的必要修正并入该 commit，不拆成多个微小 commit。

连续两个实验没有提升时只重新分析瓶颈和优先级，不得直接停止；存在未验证的高优先级结构性方案时必须继续。除非用户另有目标，仅在满足以下任一条件时停止：
- 所有高优先级假设均已验证或排除，且每个主要瓶颈至少验证过一个结构性方案；
- 已用硬件吞吐、有效带宽、指令利用率或同语义参考实现量化证明达到合理上限，且没有尚未验证的可信优化路径；
- 环境不再满足稳定测量条件，无法继续形成可信决策。

进入实验循环后应自主推进，不因每轮结果暂停询问；只有需要改变语义、扩大修改范围或缺少关键输入时才请求用户决策。

## 阶段 5 - 最终验证

对最终保留版本重新执行：

1. 阶段 1 冻结的全部必测正确性配置，并覆盖每条接受的条件路径及 fallback。
2. 阶段 1 冻结的全部必测性能配置，沿用相同的预热、正式样本、批量和 3 组重复。
3. 用冻结的原始基线与最终实现对比；仅当阶段 1 已建立同语义 GPU 参考实现时才对比参考库，否则在报告中记为 `N/A`。
4. 通过干净的 CMake 构建确认目标 SM，并记录每条最终 kernel 路径的 `ptxas -v` registers、shared memory 和 spill。
5. 对主要吞吐配置及每条接受的条件路径做 Nsight Compute 对比。基线与最终实现必须使用相同 GPU、输入、section、时钟策略和 launch 位置；
可复用同环境下冻结的基线报告，不得为了 profile 把旧 kernel 留在最终源码中。

```bash
compute-sanitizer --tool memcheck ./build/<cmake目标> --correctness-only
```
`memcheck` 必须覆盖完整正确性矩阵。每条接受的 shared memory、异步复制或显式同步路径，还必须选择一个包含边界或尾部的代表配置运行 `racecheck` 和 `synccheck`：

```bash
compute-sanitizer --tool racecheck ./build/<cmake目标> --correctness-only --case <代表配置>
compute-sanitizer --tool synccheck ./build/<cmake目标> --correctness-only --case <代表配置>
```
任一 sanitizer 报错都视为阶段 5 失败；修复后必须对最终代码重跑受影响测试和完整正确性矩阵。工具不可用时记录工具、版本和阻塞原因，不得写成“已通过”。
最终代码必须只包含被接受的实现，不保留失败实验、临时可执行文件、profile 报告或备份文件。

## 阶段 6 - 交付

将报告写入阶段 1.5 确定的独立路径，不得覆盖或追加到其他 GPU、日期或基线的报告。
若存在对应的 `/tmp` 草稿，将其中有效内容整理或同步到上述路径；删除或省略临时命令噪声、绝对临时 profile 路径和失败实验中不影响结论的冗余日志。

阶段 0-4 可按执行顺序维护草稿；进入阶段 6 后必须按决策重新组织，不能继续追加阶段流水账。最终报告遵循以下规则：
- 开头先给出最终实现、dispatch/fallback、主要性能和状态。
- 环境、测试配置、正确性和 sanitizer 各总结一次。
- 将计划和执行合并为实验表，每个实验只保留改动、关键结果、决策和 Commit。
- NCU 只保留 baseline、final 及支撑关键决策的指标，不复制原始日志或完整命令。
- 不重复代码 diff、阶段摘要和已被最终数据取代的暂定分析；代码细节由 Commit 承担。
- 不设硬性行数；以读者能快速定位结论且结果可复现为准。

报告中的 latency 取 3 个 group median 的中位数，并同时保留各组 median 和 `group_median_spread`；若没有同语义 GPU 参考实现，`Reference ms` 明确写 `N/A`。
条件优化必须写明 dispatch 条件及 fallback，验证项必须列出实际命令范围、配置数、sanitizer 错误数和 NCU 对比条件。
进入阶段 6 时读取并使用 [优化报告模板](references/report-template.md)，按实际结果补充内容，不得保留空占位符。

只有同时满足以下条件才能输出 `[SUCCESS]`：

- 最终代码通过全部必测正确性配置；
- 阶段 5 要求的 sanitizer 均已执行且为零错误；
- 至少一个主要配置获得超过噪声阈值的提升；
- 报告完整记录环境、基线、实验和最终结果。

如果没有找到有效优化，保留原基线代码，提交分析报告并输出 `[NO_GAIN]`。如果环境、输入或正确性阻止闭环，输出 `[BLOCKED]`。

## 执行摘要

每个阶段结束时向用户简要报告：

```text
[阶段 N] <名称>
- 已完成：
- 关键数据：
- 判断：
- 下一步：
```

最终回答必须包含：

- 修改文件；
- 正确性结果；
- baseline 与 final latency；
- 加速比；
- 保留和放弃的实验；
- 未能执行的工具或测试。
