# CUDA C++ 后端规则

本文件适用于 `.cu`/`.cuh`、`__global__` kernel、CUDA C++ wrapper，以及 CUTLASS 等通过 CUDA C++ 编译的实现。

## 预检与定位

执行并按目标替换占位符：
```bash
rg -n "__global__|<kernel名称>" <kernel文件>
rg -n "cuda_utils\\.cuh|printf\\s*\\(" <kernel文件>
rg -n "<源文件名>|<cmake目标>" CMakeLists.txt
```
新代码禁止引用已弃用的 `op/cuda_utils.cuh`。新增或读取 PTX 指令统一通过 `op/ptx_utils.cuh`，新增指令需注明名称、来源和用途。

## 构建和编译资源

使用实际 device 的原生 SM 通过 CMake 构建：
```bash
cmake --build build --target <cmake目标> -j
```
最终验证必须通过 CMake 目标。保留 `--ptxas-options=-v`，记录每条 kernel 路径的 registers/thread、static/dynamic shared memory 和 spill stores/loads。构建失败时不得进入实验。

## 正确性与计时

使用 CUDA Event 测量 GPU 时间，确保 event 与 kernel 位于同一 stream，并明确是否包含 launch wrapper、workspace 清零、预处理或多 kernel 序列。检查 CUDA API、launch 和同步错误。
训练路径分别验证输出、输入梯度和参数梯度；分别测 forward、backward 和直接 forward+backward 时，不得把复用 retained graph 的 backward 与重新建图的端到端结果混淆。

## Profile

先记录 `ncu --version`，再收集 basic 概览：
```bash
ncu -f --set basic --kernel-name regex:<kernel正则> \
  --launch-count 1 --kill yes --check-exit-code 0 -o build/<kernel>_basic ./build/<cmake目标> <参数>
```

按假设选择 `SpeedOfLight`、`MemoryWorkloadAnalysis`、`LaunchStats`、`Occupancy`、`SchedulerStats`、`WarpStateStats` 和 `SourceCounters`。每次重新采集，不提交 `.ncu-rep`。
`% Peak` 的分母来自 NCU `peak_sustained` 模型；`Memory Throughput` 是复合指标，不得当作 DRAM 带宽利用率。分别记录 DRAM、L2、L1/TEX 和 shared memory 指标。

## CUDA 专项实验审计

涉及架构专用机制时读取 `cuda-architecture-features.md`。重点审计：
- lane/warp 到逻辑坐标和字节地址的映射；
- global load/store 的合并、对齐和向量宽度；
- shared memory 的 bank、broadcast、冲突、padding 和 swizzle；
- `cp.async`、TMA、cluster、barrier 的架构要求和 fallback；
- registers、spill、active blocks/SM、waves/SM 和 tail wave。

吞吐饱和配置建议至少包含 8 到 10 个 full waves；业务 grid 不足时明确记录 tail wave 影响。

## 最终安全检查

`memcheck` 覆盖完整正确性矩阵：
```bash
compute-sanitizer --tool memcheck ./build/<cmake目标> --correctness-only
```

每条接受的 shared memory、异步复制或显式同步路径，选择包含边界/尾部的配置运行：
```bash
compute-sanitizer --tool racecheck ./build/<cmake目标> --correctness-only --case <配置>
compute-sanitizer --tool synccheck ./build/<cmake目标> --correctness-only --case <配置>
```

工具不可用时在报告中标记未执行及原因，不得声称通过。
