# cuTile 后端规则

本文件适用于使用 cuTile Python DSL、JIT 编译器及其 Python wrapper 的 kernel。cuTile API 和架构支持随版本变化，所有命令和能力必须以当前安装版本实测为准。

## 环境与兼容性预检

优先使用用户指定的 Conda 环境，不得擅自安装、升级或降级 cuTile、PyTorch 或 CUDA 包。记录：

- Python、cuTile、PyTorch、CUDA runtime、Toolkit 和驱动版本；
- GPU compute capability；
- cuTile 编译器声明和实测支持的目标架构；
- import、最小 kernel JIT 和实际目标 kernel JIT 是否成功。

如果编译器不支持当前 GPU 架构，保留完整错误摘要，可继续静态分析，但性能任务必须输出 `[BLOCKED]`，不得用其他架构结果代替。

使用 `rg` 定位 import、JIT 装饰器、kernel、launch、Autograd wrapper、fused/unfused 路径、fallback 和测试入口。因为 API 可能变化，不假定固定模块名或装饰器名称，以仓库源码和安装包实际接口为准。

## 构建与 JIT

Python cuTile 算子不强制增加 CMake 编译目标。若已有 Python `custom_target`，最终使用该目标；否则在指定 Conda 环境直接运行测试。

冻结并记录 kernel 源码、编译参数、tile/layout、目标架构和版本。区分：

- import/初始化时间；
- 首次 JIT 时间；
- 稳定态 latency；
- cache 策略和实际命中；
- 生成 kernel 名称及可获得的 IR/PTX/SASS。

baseline 与 final 必须使用一致缓存策略。稳定态加速比排除 JIT；冷启动只作为独立指标。

## 正确性与训练路径

使用独立同语义 reference，覆盖对齐、非对齐、尾部、stride/layout、dtype 和边界输入。训练算子验证 forward 输出、输入梯度、全部参数梯度以及 fused/unfused 语义。

分别测 forward、backward-only 和直接 forward+backward。backward-only 的图复用、梯度清零和同步方式必须写清；端到端加速比必须来自直接 forward+backward 测量。

## 性能和 Profile

使用 CUDA Event 或当前版本提供的稳定 benchmark 工具。遵守公共 warmup、样本和重复组要求，并统一 stream、同步、分配、metadata 和预处理边界。

profile 前先用独立路径完成 JIT，按生成 kernel 名筛选 NCU。重点检查：

- tile 到线程/warp/CTA 的映射；
- launch 数、fusion 和中间 tensor 流量；
- global/shared/local memory 布局与向量化；
- Tensor Core、pipeline、同步、registers 和 spill；
- 尾部 mask、无效工作和 waves/SM。

若无法获得稳定的生成代码或 profiler 映射，记录限制并降低结论置信度，不得从 DSL 源码直接断言底层指令行为。

## cuTile 专项实验规则

优先验证算法/fusion 与数据流，再实验 tile/layout、并行映射、pipeline/stage、缓存和编译参数。每个参数实验保持单一主要假设，并在业务及边界 shape 上验证，避免只对固定输入过拟合。

使用异步复制、TMA、cluster 或其他架构机制前，必须同时确认 GPU、CUDA Toolkit 和当前 cuTile 编译器支持；编译器缺少能力时标记 `EXCLUDED` 或 `[BLOCKED]`，不得假设 DSL 会自动生成目标机制。

## 最终验证

在指定环境运行语法检查、完整正确性和 3 组性能测试。Python 使用 `logging` 并输出 `[SUCCESS]`。若已有 CMake Python 测试目标，额外通过该目标。

记录当前版本可执行的内存、竞争与同步检查。工具无法稳定处理生成 kernel 时，明确列为未执行及原因，不得写成通过。
