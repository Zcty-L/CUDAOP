# CUDA Kernel 优化报告模板

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
| 轮次 | 假设/改动 | 正确性 | median ms | 相对当前基线 | 决策 | Commit |
|-----:|-----------|:------:|----------:|---------------:|:----:|:------:|

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
