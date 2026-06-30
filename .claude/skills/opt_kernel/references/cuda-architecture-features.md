# CUDA 架构特性表

用于查阅 CUDA kernel 优化中可用的架构特性，不替代对应指令的完整语义文档。先由 profiler 证明确有瓶颈，再使用本表检查兼容性。

核对日期：2026-07-03。依据 PTX ISA 9.3；当前项目工具链为 CUDA 13.2，对应 PTX ISA 9.2。表格空白表示尚未核实，不表示不支持。

## Target 规则

| Target | 兼容性 |
|--------|--------|
| `sm_XY` | baseline target。通常遵循逐代保留能力的 onion model。 |
| `sm_XYa` | architecture-specific target。仅保证指定架构支持，不能假定后续架构兼容。 |
| `sm_XYf` | family-specific target。仅保证同一架构族内的后续 target 兼容。 |

不得将 `sm_90a`、`sm_100a` 或 `sm_100f` 的特性推断为 `sm_120` 支持。必须按具体指令 variant 的 Target ISA Notes 判断。

## PTX 与工具链

| PTX ISA | 最低 CUDA Toolkit | 相关变化 |
|---------|-------------------|----------|
| 6.3 | 10.0 | 首次支持 `sm_75` target |
| 6.5 | 10.2 | 引入 `ldmatrix` |
| 7.0 | 11.0 | `sm_80` |
| 7.8 | 11.8 | `sm_89`、`sm_90` |
| 8.0 | 12.0 | `sm_90a` |
| 8.6 | 12.7 | `sm_100`、`sm_100a` |
| 8.7 | 12.8 | `sm_120`、`sm_120a` |
| 8.8 | 12.9 | `sm_100f`、`sm_120f` 等 family target |
| 9.0 | 13.0 | `sm_110`、`sm_110f`、`sm_110a` |
| 9.1 | 13.1 | |
| 9.2 | 13.2 | |
| 9.3 | 13.3 | |

Toolkit 支持某个 PTX ISA 不等于目标 GPU 支持其中全部指令。

## 数据搬运与同步

| 机制 | 最低 PTX | 支持 target | 作用与关键限制 | Fallback |
|------|----------|-------------|----------------|----------|
| `cp.async` | 7.0 | `sm_80+` baseline | global 到 shared 的非阻塞复制；单条复制 4、8 或 16 B；使用 commit/wait group 或 mbarrier 建立完成与可见性。 | 合并 global load、寄存器中转和 shared store |
| `mbarrier` 基础操作 | 7.0 | `sm_80+` baseline | shared memory 中的异步阶段同步对象；必须正确初始化、等待并在复用存储前失效。 | `__syncthreads()` 或对应 CUDA C++ barrier |
| Thread Block Cluster | 7.8 | `sm_90+` baseline | block 协同调度、cluster 同步和 Distributed Shared Memory；grid 必须满足 cluster 维度，实际 cluster 上限需查询。 | 独立 block、重复加载或多 kernel |
| `cp.async.bulk` | 8.0 | `sm_90+` baseline | TMA bulk copy；支持 global、shared::cta 和 shared::cluster 间的规定方向，使用 mbarrier 或 bulk group 完成机制。 | `cp.async` 或普通 load/store |
| `cp.async.bulk.tensor` | 8.0 | `sm_90+` baseline | TMA tensor copy；通过 tensor map 搬运 1D 至 5D tile，支持边界与 swizzle 配置。 | `cp.async` 或显式多维地址计算 |
| TMA multicast | 8.0 | `sm_90+` baseline | global tile 复制到 cluster 内多个 block。当前文档建议在 `sm_90a`、`sm_100f/a`、`sm_103f/a`、`sm_110f/a` 使用；其他 target 可能显著降速。 | 各 block 独立 TMA 或普通复制 |

`cp.async`、TMA 和 cluster 只在数据流存在重叠或跨线程复用时列入计划。支持某项指令不代表使用后一定更快。

## 矩阵数据与计算

| 机制 | 最低 PTX | 支持 target | 关键限制 |
|------|----------|-------------|----------|
| `ldmatrix` 基础形态 | 6.5 | `sm_75+` baseline | warp 协作从 shared memory 加载矩阵 fragment；地址、布局和参与线程必须满足对应 variant 约束。 |
| `wgmma.mma_async` 基础形态 | 8.0 | `sm_90a` | architecture-specific，不得外推到 `sm_90`、`sm_100` 或 `sm_120`；warpgroup 必须一致参与并正确使用 fence、commit 和 wait。 |
| `tcgen05` 基础操作 | 8.6 | `sm_100a`；PTX 8.8 起支持 `sm_100f`；PTX 9.0 起支持 `sm_110f` | variant、数据类型、shape 和 target 组合复杂；使用前必须再次读取具体指令的 Target ISA Notes。 |

一些指令的支持范围随 shape、数据类型、累加类型和 qualifier 变化，本表暂不做统一推断，必要时自行搜索。

## 使用步骤

1. 按用户参数、`CMAKE_CUDA_ARCHITECTURES`、GPU 计算能力的优先级确定唯一目标。
2. 使用 `nvcc --version` 确认 Toolkit，再由 PTX 发布表确认可用 PTX ISA。
3. 先按本表排除不兼容机制，再读取具体指令的 Target ISA Notes、对齐、同步和内存一致性要求。
4. 优先使用 CUDA C++、libcu++ 或库级接口。必须使用 PTX 时，仅在 `op/ptx_utils.cuh` 添加，并注明指令、官方来源和用途。
5. 提供 baseline fallback，分别编译目标架构，检查生成代码并用正确性测试和定向 NCU 指标验证。

## 官方来源

- [PTX ISA：Target 与 PTX 发布历史](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ptx-module-directives-target)
- [PTX ISA：指令集与 Target ISA Notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-set)
- [CUDA Programming Guide：Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)
- [CUDA C++ Programming Guide：TMA](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)
