---
name: opt-kernel
description: >-
  分析并迭代优化 CUDAOP 中使用 CUDA C++、Triton 或 cuTile 实现的 GPU kernel，完成目标定位、环境检查、
  基线冻结、瓶颈定位、单变量实验、正确性与性能验证及报告交付。用于 $opt-kernel、CUDA/Triton/cuTile
  kernel 调优、GPU 算子性能分析、Nsight Compute 分析或要求优化 op/{name} 下算子的任务。
---

# opt-kernel - GPU Kernel 优化工作流

按“检查、基线、分析、规划、实验、验证、交付”的顺序执行。不得跳过基线正确性验证，也不得用未经测量的经验判断代替性能数据。

## 使用方式

直接描述目标实现、源码位置、业务配置和目标设备，例如：

```text
$opt-kernel 优化 op/dwconv/example.cu:128 的 dwconv_forward_kernel。
$opt-kernel 优化 op/grouped_gemm/grouped_gemm.py 中的 Triton fused kernel。
$opt-kernel 优化 op/example/cutile_impl.py 中的 cuTile kernel。
```

若输入不能唯一定位实现，列出候选 kernel、定义位置和后端，请用户指定，不得自行选择。找不到唯一匹配时输出 `[BLOCKED]`。

## 必读文件与后端路由

开始任务后：

1. 完整读取 [公共优化工作流](references/workflow-common.md)。
2. 根据目标实现完整读取下列一个或多个后端文件。
3. 进入交付阶段时完整读取 [优化报告模板](references/report-template.md)。

| 识别信号 | 后端文件 |
|----------|----------|
| `.cu`、`.cuh`、`__global__`、CUDA C++ launch | [CUDA C++ 后端](references/backend-cuda.md) |
| `.py`、`@triton.jit`、`triton.language` | [Triton 后端](references/backend-triton.md) |
| `.py`、`cutile`/`cuTile` import 或对应 JIT 装饰器 | [cuTile 后端](references/backend-cutile.md) |

只加载本次任务涉及的后端。若一个测试同时包含 CUTLASS/CUDA、Triton、cuTile 等实现，读取所有相关后端文件，并在报告中分别记录构建、JIT、计时和 profile 边界。

涉及架构专用机制时，还必须读取 [CUDA 架构特性表](references/cuda-architecture-features.md)，核对目标架构、CUDA Toolkit、PTX ISA、后端支持范围和 fallback。

## 规则优先级

依次遵守：

1. 用户明确给出的目标、环境、shape、精度、测试范围和停止条件。
2. 仓库根目录及目标目录中的 `AGENTS.md`。
3. 本文件的公共路由规则。
4. `workflow-common.md` 的公共优化规则。
5. 对应后端文件的工具链规则。

后端文件只替换构建、编译产物、计时和工具链细节，不得放宽公共正确性、重复测量、单变量实验、提交和报告要求。

## 混合后端比较

比较多个后端时必须统一：

- 输入、输出、布局、dtype、数值语义与随机种子；
- device、stream、warmup、正式样本、批量大小和同步边界；
- 是否包含权重预处理、JIT、Autograd、前向、反向和中间张量分配；
- reference、baseline 与优化目标的计时范围。

首次 JIT/编译时间单独记录，不计入稳定态 kernel latency，除非用户明确要求冷启动性能。端到端测试必须额外保留直接测得的端到端 latency，不得用各阶段独立计时结果相加替代。

## 执行摘要

每个阶段结束时向用户报告：

```text
[阶段 N] <名称>
- 已完成：
- 关键数据：
- 判断：
- 下一步：
```

最终回答必须包含修改文件、正确性结果、baseline 与 final latency、加速比、保留和放弃的实验，以及未执行工具或测试及原因。
