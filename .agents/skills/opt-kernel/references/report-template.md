# GPU Kernel 优化报告模板

```markdown
# <kernel_name> 优化报告

## 结论
- 状态：
- 后端与最终实现：
- 主要结果：
- 适用范围：
- 关联历史报告：

## 最终实现与 Dispatch
| 条件 | 后端/实现 | 关键改动 | Fallback |
|------|-----------|----------|----------|

## 最终性能
| 配置/计时范围 | Baseline group medians | Final group medians | Speedup | Final spread | Reference |
|---------------|------------------------|---------------------|--------:|-------------:|----------:|

## 正确性与安全检查
- 输出正确性：
- 梯度正确性：
- 边界路径：
- sanitizer/后端安全工具：

## 环境与测试
- 日期 / GPU / SM：
- 后端 / Python / 编译器：
- CUDA / Driver / NCU：
- Conda 环境 / CMake target / 测试入口：
- 源文件 / kernel：
- dtype / 输入范围 / 容差：
- warmup / samples / groups：
- JIT / cache / Autotune 策略：
- 计时边界：

## 构建、JIT 与资源
| 路径 | 构建/JIT 参数 | Registers | Shared memory | Spill | 生成代码证据 |
|------|--------------|----------:|--------------:|------:|--------------|

## Profiler Baseline / Final 概览
只保留 Baseline 和实际接受的 KEEP。所有行使用相同 GPU、输入、section、时钟策略、cache/JIT 状态和 launch 位置。

| 版本 | Commit | Duration | SM GHz | DRAM GHz | Compute % | Memory composite % | DRAM % | L2 % | L1/TEX % | Occupancy % | Active Warps/SM |
|------|:------:|---------:|-------:|---------:|----------:|-------------------:|-------:|-----:|---------:|------------:|----------------:|
| Baseline |  |  |  |  |  |  |  |  |  |  |  |
| KEEP / Final |  |  |  |  |  |  |  |  |  |  |  |

百分比统一使用 profiler 对应性能模型的定义。NCU `Memory Throughput` 是复合指标，不得解释为 DRAM 带宽利用率。Profiler duration 只用于同条件分析，最终 latency 使用 benchmark 结果。

## 定向证据（按需）
| 指标 | 单位 | Baseline | Final | 变化 | 判断 |
|------|------|---------:|------:|-----:|------|

可按后端加入 TTIR/TTGIR/PTX/SASS、load/store 流量、Tensor Core、eligible warps、stall、bank conflict、launch 数或 fusion 中间张量等指标。删除无关示例，不使用大量 `N/A`。

## 实验记录
| 轮次 | 假设/改动 | 正确性 | 关键性能 | 决策 | Commit |
|-----:|-----------|:------:|----------|:----:|:------:|

## 候选方向闭环
| 方向 | 状态 | 证据 |
|------|:----:|------|

## 剩余限制
- 剩余瓶颈：
- 未执行项及原因：
- 性能可信度限制：

[SUCCESS]
```
