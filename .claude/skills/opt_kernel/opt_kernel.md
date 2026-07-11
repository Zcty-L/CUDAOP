---
name: opt_kernel
description: >-
  分析并迭代优化 CUDAOP 的 .cu 文件中用户指定的 CUDA kernel，完成目标定位、环境检查、基线冻结、瓶颈定位、
  单变量实验、正确性与性能验证及报告交付。用于 /opt_kernel、CUDA kernel 性能分析、kernel 调优、
  Nsight Compute 分析或要求优化 op/{name} 下 CUDA 算子的任务。
---

# /opt_kernel - CUDA Kernel 优化工作流

按“检查、基线、分析、规划、实验、验证、交付”的顺序执行。不得跳过基线正确性验证，也不得用未经测量的经验判断代替性能数据。

## 用法

```text
/opt_kernel <kernel文件路径> --kernel=<kernel名称> [--line=<定义行号>] [--arch=sm_xx] [--target=<cmake目标>]
```
- `kernel文件路径`：必填，包含目标 kernel 的 `.cu` 文件。
- `--kernel=<kernel名称>`：必填，待优化的 kernel 名称。
- `--line=<定义行号>`：推荐，用于区分重载、模板、宏生成或同名定义。
- `--arch=sm_xx`：可选，显式指定目标计算能力。
- `--target=<cmake目标>`：可选，显式指定 CMake target；默认使用文件名去掉 `.cu`。

也接受自然语言目标描述，例如：
```text
需要优化的 kernel 位于 op/dwconv/example.cu:128，kernel name 为 dwconv_forward_kernel。
```
能够根据输入定位唯一目标 kernel 的 launch site、launch wrapper 和模板实例化，包括输入、输出、数据类型、布局、形状范围和数值语义，
如果用户没有提供足够的定位信息，先列出文件中的候选 kernel 名称与定义行号，请用户指定目标，不得自行选择。找不到唯一匹配时输出 `[BLOCKED]`。

架构选择优先级为：用户显式参数、`CMAKE_CUDA_ARCHITECTURES`、当前 GPU 计算能力。三者不一致时停止并说明，不得静默使用默认架构。

## 阶段 0 - 预检

### 0.1 检查仓库

执行并记录：

```bash
git status --short --branch
rg -n "__global__|<kernel名称>" <kernel文件路径>
rg -n "cuda_utils\\.cuh|printf\\s*\\(" <kernel文件路径>
rg -n "<源文件名>|<cmake目标>" CMakeLists.txt
```

修改代码前，从稳定基线为每个 kernel 优化任务创建一个 `feat/{op_name}_opt` 分支。实验轮次不另建分支；只提交正确且有收益的版本。仅分析时无需新建分支。
结合定义行号阅读目标 kernel、函数签名、相邻注释及全部 launch site。若目标是模板 kernel，记录被测配置实际使用的模板参数和编译后 kernel 名称。

### 0.2 检查 GPU

通过 CMake 构建并运行设备查询：

```bash
cmake -S . -B build
cmake --build build --target device_query -j
./build/device_query
nvidia-smi --query-gpu=name,compute_cap,utilization.gpu,utilization.memory --format=csv
```

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

若已有 cuBLAS、cuDNN、CUTLASS 或其他参考实现，在相同输入、布局、精度、stream 和计时范围下测量并记录。参考库不是正确性基线的替代品。

### 1.2 编译基线

```bash
cmake -S . -B build
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
默认不要求锁定 GPU 频率；若发现动态频率、功耗限制或 Laptop GPU 导致波动，在记录和最终报告中注明，不把锁频结果当作默认性能上限。
若动态时钟设备经预热仍无法将组间中位数波动控制在 2% 内，默认标记 `[BLOCKED]`，仅继续静态分析和制定暂定计划。
若用户明确接受噪声阈值策略，可继续探索实验；实验结果只能标记为暂定结论，判定规则见阶段 4。

### 1.5 创建实验记录

在最终报告中维护以下表格。优化期间可先记录在 `/tmp/opt_kernel_<kernel_name>.md`，但 `/tmp` 只作为运行中草稿。
阶段性停止、`[BLOCKED]` 或最终交付前，必须将可读报告写入 `docs/opt_kernel/<kernel_name>.md`。不得只交付 `/tmp` 路径；临时 profile 文件不得提交到仓库。

```markdown
| 轮次 | 假设/改动 | 正确性 | median ms | mean ms | 相对当前基线 | 决策 |
|-----:|-----------|:------:|----------:|----------:|---------------:|:----:|
| 0 | baseline | PASS | 0.000000 | 0.000000 | 0.00% | KEEP |
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
9. 只有 `KEEP` 的实现才能成为下一轮基线。

实现成本较高时，保留当前最近的最佳 kernel，新建以机制命名的实验 kernel 实现复杂方案，并在同一测试中对比。决策后，`KEEP` 替换原实现，`REVERT` 删除实验 kernel。

失败实验只撤销本轮由执行者引入的改动，不得使用会覆盖用户修改的 Git 命令。
普通实验不保留 `#ifdef` 基线、旧 kernel 或 `_v2` 变体；最终代码只保留接受的实现，历史由 Git commit、diff 和实验表承担。

### 决策规则

若环境噪声无法完全消除，先用基线重复组估计噪声范围；只有 median 和 mean 均稳定改善且幅度明显大于噪声，才可暂定 `KEEP` 并要求后续复测。`min` 只作为下界诊断，不作为接受优化的主依据。

| 结果 | 决策 |
|------|------|
| 任一正确性配置失败 | `REVERT` |
| 任一必测配置稳定回退超过 2% | `REVERT` |
| 主要配置提升至少 5%，且其他配置不回退 | `KEEP`，作为新基线继续 |
| 提升 2% 到 5%，跨 3 组测试稳定 | `KEEP`，降低后续优先级 |
| 提升不足 2% | 视为噪声，`REVERT` |
| 编译、运行或 profile 失败 | 记录 `CRASH`，修复或尝试下一假设 |

除非用户另有目标，满足任一条件时停止：

- 连续两个有效实验没有提升；
- 所有高优先级假设已验证；
- 性能达到硬件或参考实现的合理上限；
- 环境不再满足稳定测量条件。

进入实验循环后应自主推进，不因每轮结果暂停询问；只有需要改变语义、扩大修改范围或缺少关键输入时才请求用户决策。

## 阶段 5 - 最终验证

对最终保留版本重新执行：

1. 全部正确性测试。
2. 全部性能配置， warmup、正式测试、3 组重复。
3. 基线、最终实现和参考库的统一对比。
4. `ptxas -v` 资源对比。
5. 定向 Nsight Compute 对比。

```bash
compute-sanitizer --tool memcheck ./build/<cmake目标> <运行参数>
```

如项目使用 race-prone shared memory、异步复制或复杂同步，按需增加 `racecheck` 和 `synccheck`。工具不可用时在报告中说明。

最终代码必须只包含被接受的实现，不保留失败实验、临时可执行文件、profile 报告或备份文件。

## 阶段 6 - 交付

将报告写入仓库可见路径：`docs/opt_kernel/<kernel_name>.md`。
若存在 `/tmp/opt_kernel_<kernel_name>.md`，将其中有效内容整理或同步到上述路径；删除或省略临时命令噪声、绝对临时 profile 路径和失败实验中不影响结论的冗余日志。

报告模板：

```markdown
# <kernel_name> 优化报告

## 环境
- GPU：
- 计算能力：
- CUDA / Driver：
- CMake target：
- 编译架构：
- GPU 时钟与波动：

## 测试配置
- 输入与布局：
- 数据类型：
- 正确性容差：
- warmup / iterations：

## 基线
- 正确性：
- median latency：
- 参考实现：
- registers / shared memory：

## 瓶颈结论
- 主瓶颈：
- 证据：
- 次要瓶颈：
- 分析限制：

## 实验记录
| 轮次 | 假设/改动 | 正确性 | median ms | 相对当前基线 | 决策 |
|-----:|-----------|:------:|----------:|---------------:|:----:|

## 最终结果
| 配置 | Baseline ms | Final ms | Speedup | Reference ms |
|------|------------:|---------:|--------:|-------------:|

## 验证
- 边界测试：
- compute-sanitizer：
- Nsight Compute：

## 结论
- 接受的优化：
- 最终加速比：
- 剩余瓶颈：
- 未执行项及原因：
- 性能可信度限制：

[SUCCESS]
```

只有同时满足以下条件才能输出 `[SUCCESS]`：

- 最终代码通过全部必测正确性配置；
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
