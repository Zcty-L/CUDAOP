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

## 实验记录
| 轮次 | 改动 | 关键结果 | 决策 | Commit |
|-----:|------|----------|:----:|:------:|

## 剩余限制
- 剩余瓶颈：
- 未执行项及原因：
- 性能可信度限制：

[SUCCESS]
```
