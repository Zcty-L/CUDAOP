# Triton 后端规则

本文件适用于使用 `@triton.jit`、`triton.language`、Triton Autotuner 或 Python Autograd wrapper 的 kernel。

## 环境和目标定位

优先使用用户指定的 Conda 环境；未指定时先从项目测试入口、文档和已安装环境确定，不得擅自安装或升级 PyTorch、Triton、CUDA 包。

记录 Python、PyTorch、Triton、CUDA runtime、驱动和目标 GPU：

```bash
conda run --no-capture-output -n <env> python --version
conda run --no-capture-output -n <env> python -c \
  "import torch, triton; print(torch.__version__); print(triton.__version__); print(torch.version.cuda)"
```

使用 `rg` 定位：

- `@triton.jit`、`@triton.autotune`、kernel 定义和所有 launch；
- `torch.autograd.Function` 的 forward/backward；
- fused/unfused 路径、fallback 和 dispatch；
- benchmark、reference、shape、dtype 与 tolerance。

## 构建与 JIT

Python 构建算子不强制增加 CMake 编译目标。若仓库已有 Python `custom_target`，最终测试使用该目标；否则在指定 Conda 环境直接运行测试入口，并在报告中记录完整命令。
首次调用前冻结 kernel 源码、meta-parameters、constexpr、`num_warps`、`num_stages`、目标架构和版本。分别记录：
- 首次 JIT/Autotune 时间；
- 稳定态 kernel latency；
- Triton cache 的使用或清理策略；
- 命中的配置和生成 kernel 名称。

稳定态性能默认排除首次 JIT 和 Autotune。不得在 baseline 与 final 之间只清理一方缓存。若用户关注冷启动，作为独立指标测量，不与稳定态加速比混合。
检查编译后的 TTIR、TTGIR、LLVM IR、PTX 或 SASS 时，使用当前 Triton 版本实际支持的公开接口或调试选项，并记录获取方式。只保留关键资源数据和指令结论，不提交批量 dump 或缓存目录。

## 正确性

使用独立同语义 reference，不把另一个待优化 Triton 实现当作唯一 reference。覆盖：
- 对齐、非对齐、零长度允许场景和尾部 mask；
- stride、layout、非连续输入和 dtype；
容差必须按 dtype 和累积长度设置并冻结。发生 NaN/Inf 时单独统计，不得只依赖 `allclose`。

## 计时边界

使用 CUDA Event 或当前 Triton 版本提供的稳定 benchmark 工具测量 GPU 时间，明确同步和 stream。默认遵守公共工作流的 20 次 warmup、100 个样本和 3 组重复；工具若内部采用不同统计方式，显式配置或在报告中说明。

## Profile 和生成代码分析

先运行一次独立 warmup/profile-only 路径完成 JIT，再启动 NCU；避免把编译、随机初始化和大量无关 PyTorch kernel 纳入 profile。按实际生成 kernel 名筛选，并核对 launch 次数。

除 NCU 通用指标外，重点比较：
- 实际 program 数、warps、stages 和每个 program 的 tile；
- load/store 向量化、mask、布局转换和冗余计算；
- Tensor Core 指令、software pipeline、shared memory 和 registers/spill；
- Autotune 命中配置与业务 shape 的稳定性。
若源码行映射不足，以 TTIR/TTGIR/PTX/SASS 和 NCU 指标交叉验证，不得仅凭 Triton 源码推断实际访存或资源占用。

## Triton 专项实验规则

优先按证据依次考虑：算法、内存流量、program/tile 映射、layout、`num_warps`、`num_stages`、持久化策略和较小参数搜索。参数搜索必须：
- 先保留固定配置基线，再单独记录搜索空间；
- 在业务 shape 与边界 shape 上验证命中配置；
- 把 Autotune 时间排除在稳定态 benchmark 外；
- 防止只对单一输入过拟合；
- 将最终配置或可解释 dispatch 固化并验证 fallback。

调整 block pointer、layout、mask、warp/stage 或持久化映射时，审计逻辑 tile 到地址的映射、数据复用、尾部浪费、shared/register 成本和 waves/SM。

## 最终验证

在指定 Conda 环境中运行语法检查和完整测试。Python 输出使用 `logging`，包含配置、主要阶段、结果与 `[SUCCESS]`。
若项目已有 CMake Python 测试目标，最终额外通过该目标；否则直接测试入口符合项目“Python 构建算子不要求 CMake”的约定。最终报告明确列出未执行的 compute-sanitizer 检查；若 Triton 生成 kernel 可被当前工具稳定检查，则运行适用检查并记录命令和结果。
