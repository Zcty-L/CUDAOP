# CUDA Kernel 优化报告模板

```markdown
# <kernel_name> 优化报告

## 结论
- 状态：
- 主要结果：
- 适用范围：
- 关联历史报告：

## 最终实现与 Dispatch
| 条件 | 实现 | 关键改动 | Fallback |
|------|------|----------|----------|

## 最终性能
| 配置 | Baseline group medians | Final group medians | Speedup | Final spread | Reference |
|------|------------------------|---------------------|--------:|-------------:|----------:|

## 正确性与安全检查
- 正确性：
- compute-sanitizer：
- 边界路径：

## 环境与测试
- 日期 / GPU / SM：
- CUDA / Driver / NCU：
- 源文件 / kernel / CMake target：
- 数据类型 / 输入范围 / 容差：
- warmup / samples / groups：

## 资源与关键性能证据
| 路径 | Registers | Shared memory | Spill | Baseline / Final NCU | 关键指标 |
|------|----------:|--------------:|------:|----------------------|----------|

## NCU Baseline / KEEP 概览
只保留 Baseline 和实际接受的 KEEP；不记录 REVERT 或 CRASH。每个 KEEP 增加一行，最后一个 KEEP 标记为 Final。所有行必须使用相同 GPU、输入、NCU section、时钟策略和 kernel launch 位置重新采集。

| 版本 | Commit | NCU Duration ms | SM Frequency GHz | DRAM Frequency GHz | Compute (SM) % | Memory Throughput % (composite) | DRAM % | L2 % | L1/TEX % | Achieved Occupancy % | Active Warps/SM |
|------|:------:|----------------:|-----------------:|-------------------:|---------------:|---------:|-------:|-----:|---------:|---------------------:|----------------:|
| Baseline |  |  |  |  |  |  |  |  |  |  |  |
| KEEP / Final |  |  |  |  |  |  |  |  |  |  |  |

表中百分比统一使用 NCU 相对 `peak_sustained` 的结果。`Memory Throughput` 是 NCU 复合吞吐指标，不得解释为 DRAM 带宽利用率；DRAM、L2 和 L1/TEX 必须分别记录。
NCU Duration 只用于同条件 profile 对比，最终 latency 仍以 CUDA Event 为准。若存在多条 KEEP dispatch 路径，为每条路径分别建表。

## NCU 定向证据（按需）
只添加支撑关键假设和 KEEP 决策的指标，并为 Baseline 与 KEEP 使用完全相同的 metric 集重新采集。删除不相关的示例行，不用大量 `N/A` 填充。

| 指标 | 单位 | Baseline | KEEP / Final | 变化 | 支撑的判断 |
|------|------|---------:|-------------:|-----:|------------|
| Eligible Warps Per Scheduler | warp |  |  |  | 延迟隐藏能力 |
| One or More Eligible | % |  |  |  | 调度器可发射程度 |
| 主要 Warp Stall Reason | % |  |  |  | 主要等待来源 |
| 关键执行 pipe 利用率 | % peak_sustained |  |  |  | 计算管线变化 |
| L2 Hit Rate | % |  |  |  | cache 复用变化 |
| DRAM Read / Write Bytes | byte |  |  |  | 实测显存流量变化 |
| Shared Bank Conflicts | count |  |  |  | shared memory 冲突变化 |
| Branch Efficiency | % |  |  |  | 分支发散变化 |

## 实验记录
| 轮次 | 改动 | 关键结果 | 决策 | Commit |
|-----:|------|----------|:----:|:------:|

## 剩余限制
- 剩余瓶颈：
- 未执行项及原因：
- 性能可信度限制：

[SUCCESS]
```
