# GPU Kernel 优化公共工作流

本文件定义所有后端共同遵守的流程。构建、JIT、生成代码、计时和安全工具的具体命令由对应后端文件规定。

## 目录

1. 阶段 0：预检
2. 阶段 1：冻结基线
3. 阶段 2：瓶颈分析
4. 阶段 3：制定计划
5. 阶段 4：单变量实验
6. 阶段 5：最终验证
7. 阶段 6：交付

## 阶段 0：预检

### 检查仓库

首先执行 `git status --short --branch`。禁止在 `main` 分支修改文件。

修改代码前，从稳定基线为每个优化任务创建符合仓库规范的优化分支。实验轮次不另建分支；每次确认 `KEEP` 后立即创建独立的 `perf:` commit，一次 commit 只包含一个有效优化及必要测试和实验记录。仅分析时无需新建分支。
工作树已有用户修改时，只暂存本任务且能明确归属的文件或 hunk。若实验改动与用户修改无法安全分离，先请求用户处理。
定位并记录：
- kernel 定义、launch site、wrapper、dispatch 和模板/编译参数；
- 输入、输出、dtype、布局、shape 范围、数值语义和 stream；
- 正确性 reference、benchmark 入口、构建或 JIT 入口；

### 检查环境和 GPU

先查询 GPU 名称、compute capability、利用率和显存占用。使用 benchmark 实际运行的 device；多 GPU 或多架构环境不得默认选择。记录：
- GPU、SM、驱动、CUDA Toolkit/runtime 和后端版本；
- Shared Memory/SM、寄存器/SM、occupancy 上限和适用 dtype 的 ridge point；
- GPU 是否为 Laptop、Max-Q 或其他动态频率明显的设备；
- 测试前是否存在持续 GPU 负载。
不能获得 GPU 或 GPU 持续繁忙时，可以继续静态分析，但不得给出可信性能结论，最终标记 `[BLOCKED]`。

### 检查测试入口

测试必须输出配置与 dtype、主要阶段、正确性结果、warmup 与正式样本数、latency 统计、reference 性能、关键结论和成功时的 `[SUCCESS]`。Python 使用 `logging`，C++ 使用 `std::cout`。

## 阶段 1：冻结基线

基线是第一次不修改目标 kernel 的完整运行，记为实验 0。若测试框架不完整，允许先补齐非目标测试框架，但必须记录目标实现快照，不得改变其语义。

### 冻结测试矩阵

至少定义：
- 一个吞吐饱和配置；
- 常用业务配置；
- 对齐和非对齐尺寸及边界/尾部配置；
- 固定随机种子；
- 每种 dtype 的 `atol` 和 `rtol`；
- 统一的 benchmark 计时范围。

用户指定唯一业务 shape 时，将其作为主要配置，吞吐饱和配置只作辅助分析。若任务包含训练路径，分别定义 forward、backward 和直接 forward+backward；最终端到端结果必须直接测量。
有 cuBLAS、cuDNN、CUTLASS、PyTorch 或其他同语义实现时，在相同输入、布局、精度、stream 和计时范围下测量。参考库不能替代正确性 reference。

### 验证正确性

逐项比较 reference，至少报告：
- 最大绝对误差与最大相对误差；
- 错误元素数/总元素数；
- launch、同步和运行时错误；
- 训练路径的输出、输入梯度和所有可训练参数梯度。
基线正确性失败时停止并输出 `[BLOCKED]`，不得通过放宽容差掩盖原始错误。

### 测量性能

默认 warmup 20 次、正式样本 100 次，报告 mean、median、min、max 和标准差。极短 kernel 可批量 launch 后除以批量大小，但 warmup 与正式样本必须使用相同批量，并在报告中记录。
目标实现与 reference 分开预热、分开测量，不在同一采样组内交替执行。同一配置至少重复完整 benchmark 3 组。最终 latency 取 3 个 group median 的中位数，并保留各组 median。
统一计算：`group_median_spread = (max(group_median) - min(group_median)) / median(group_median)`
组间 spread 超过 2% 时，先排查负载、温度、时钟、功耗、缓存和 JIT 状态，不进入实验。若仍不能控制，默认 `[BLOCKED]`；用户明确接受噪声策略时才可继续，结论必须标记为暂定。

### 创建实验记录

使用路径：
```text
docs/opt_kernel/opt_<op_name>/opt_kernel_<kernel_id>_<backend>_<gpu>_<YYYYMMDD>.md
```
每个 GPU、日期、后端组合和基线使用独立报告；同日新轮次追加 `_02`。优化期间可写 `/tmp` 草稿，但停止或交付前必须整理到仓库，且不提交临时 profile/dump。

## 阶段 2：瓶颈分析

必须同时使用源码、编译/生成代码资源数据和运行指标。算术强度只用于初步假设。

### 静态分析

分析并记录：
- grid、block/warp、tile、shared memory 和 launch 次数；
- FLOPs、整数指令、特殊函数和同步；
- 理论最小 global memory 读写字节；
- 合并访问、对齐、重复加载、缓存复用和 bank conflict 风险；
- 分支发散、原子操作、边界线程和无效工作；
- 寄存器中间值生命周期、spill 风险和 occupancy 限制；
- fusion 前后的 launch、内存流量和中间张量变化。
使用 `I = Total FLOPs / Minimum Global Memory Bytes` 估计算术强度，并明确流量是理论下界还是 profiler 实测值。

### Profile 分析

先收集低开销概览，再按假设收集详细指标，避免每轮无条件使用 full profile。记录 profiler 版本、kernel 筛选方式、launch 位置、输入配置和重放影响。按需关注：
- SM 与 memory throughput；
- DRAM、L2、L1/TEX 和 shared memory 流量；
- occupancy、active warps 和资源限制；
- eligible warps、stall reason 和关键执行 pipe；
- 热点源代码、生成代码与指令。
profiler 不可用时使用后端计时、编译资源数据和静态分析继续，但明确标记证据不完整。

### 分类规则

只在证据支持时判断 Memory-bound、Compute-bound、Latency-bound、Launch-bound 或 Resource-bound。输出主瓶颈、次要瓶颈、证据和置信度。不得仅凭 occupancy 低判定性能差，也不得把 occupancy 最大化本身作为目标。

## 阶段 3：制定计划

根据证据生成有优先级的实验。每个实验写明假设、单一主要变量、预期、风险和验收条件。优先级通常为：
1. 算法、fusion 和数据流；
2. 内存流量、布局、复用与向量化；
3. block/warp/tile 并行映射；
4. software pipeline、预取和延迟隐藏；
5. Tensor Core 或架构专用机制；
6. 编译参数和较小参数搜索。

先输出高级机制适配性判断，至少覆盖软件流水、shared/multi-stage、异步复制、向量化、cache 策略以及目标后端支持的 TMA/cluster/persistent/fusion 机制。若不实验，写明不满足的条件或成本；若实验，转化为单变量实验项。

## 阶段 4：单变量实验

每轮只验证一个主要假设；保持接口和语义时，允许完整重构数据流、映射和存储层级。每轮执行：
1. 检查工作树并记录预期与验收条件。
2. 修改一个主要变量。
3. 按后端规则构建/JIT，记录资源与生成代码变化。
4. 运行完整正确性测试。
5. 运行主要 benchmark，必要时定向 profile。
6. 记录 `KEEP`、`REVERT` 或 `CRASH`。
7. `KEEP` 前补齐 dispatch、fallback 和边界测试，重跑完整必测项。
8. 将一次有效优化及必要测试和报告提交为独立 `perf:` commit。

修改线程/tile 映射、shared 布局或向量化时，写出代表性逻辑坐标到实际地址的映射，审计合并访问、对齐、逻辑流量、bank conflict、同步、寄存器和 occupancy 成本。候选实现若引入可修正的次生问题，只能否定当前实现，不能直接关闭原始方向。
失败实验不得改写已有 `KEEP` commit，也不得覆盖用户修改。最终源码不保留普通 `_v2`、失败实验、临时 dump 或无用基线路径；历史由 commit 和实验表承担。

### 决策规则

| 结果 | 决策 |
|------|------|
| 任一正确性配置失败 | `REVERT` |
| 任一必测配置稳定回退超过 2%，且无法可靠 dispatch | `REVERT` |
| 某类配置提升至少 5% 且至少为噪声上界 3 倍 | `KEEP` 条件路径 |
| 主要配置提升至少 5%，其他配置不回退 | `KEEP` |
| 提升 2% 到 5%，跨 3 组稳定且无必测回退 | `KEEP`，降低后续优先级 |
| 提升不足 2% | `REVERT`，视为噪声 |
| 构建、运行或 profile 失败 | `CRASH`，修复或进入下一假设 |

### 候选方向闭环

把所有表述为值得尝试、建议尝试、下一步或高优先级的方向加入清单，逐项标记：
- `VERIFIED`：已实现并完成正确性、性能和有效性审计；
- `EXCLUDED`：由 profiler、生成代码、硬件约束或数学上界明确排除；
- `BLOCKED`：需要改变语义、扩大授权或缺少关键条件。
存在未闭环方向时不得进入最终验证。连续两个实验无提升时重新分析，不得直接停止。进入实验后自主推进，只在需要改变语义、扩大范围或缺少关键输入时询问用户。

## 阶段 5：最终验证

对最终代码重新执行：
1. 全部正确性矩阵及每条 dispatch/fallback 边界。
2. 全部性能配置，沿用冻结的 warmup、样本、批量和 3 组重复。
3. 原始 baseline 与 final 的同环境比较。
4. 后端规定的干净构建/JIT、资源和生成代码检查。
5. 主要配置及每条接受路径的同条件 profiler 对比。
6. 后端规定的内存、竞争、同步或运行安全检查。
工具不可用时记录工具、版本和原因，不得写成“已通过”。任一安全检查报错都视为失败，修复后重跑受影响测试和完整正确性矩阵。

## 阶段 6：交付

完整读取并使用 [优化报告模板](report-template.md)。按最终决策重组报告，不追加阶段流水账。报告必须：
- 开头给出最终实现、dispatch/fallback、主要性能和状态；
- 汇总一次环境、测试矩阵、正确性和安全检查；
- 合并计划与实验记录，每个实验只保留关键结果、决策和 commit；
- 只保留 baseline、final 和支撑关键决策的 profiler 指标；
- 记录 latency 的 3 组 median、最终 median 和 spread；
- 明确 JIT/预处理/Autograd/fusion/forward/backward 的计时边界；
- 删除临时命令噪声、绝对临时路径和无效 dump。

只有全部正确性与必需安全检查通过、至少一个主要配置提升超过噪声阈值且报告完整时输出 `[SUCCESS]`。没有有效优化时保留原基线、提交分析报告并输出 `[NO_GAIN]`；环境或正确性阻止闭环时输出 `[BLOCKED]`。
