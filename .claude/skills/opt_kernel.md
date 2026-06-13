---
name: opt_kernel
description: CUDA Kernel 优化工作流 — 分析、规划、执行、验证
---

# /opt_kernel — CUDA Kernel 优化工作流

## 用法

```
/opt_kernel <kernel文件路径> [--target=<优化目标>] [--arch=<计算能力>]
```

- `kernel文件路径`：必填，要优化的 .cu / .cuh 文件路径
- `--target=latency|memory|occupancy`：可选，指定优化侧重点，默认自动判断
- `--arch=sm_xx`：可选，目标架构，默认 `sm_75`

---

## 阶段 1 — 分析 (Analyze)

读取目标文件后，执行以下分析：

### 1.1 Kernel Launch 配置分析
- Grid / Block 维度
- 动态共享内存大小
- Kernel 参数类型和数量

### 1.2 内存访问分析
- Global memory 访问模式是否 coalesced？
- 是否使用 shared memory？tile size 是否合理？
- 是否存在 bank conflict？
- 是否存在未合并的读写？

### 1.3 计算分析
- 算术强度（arithmetic intensity）
- 是否使用 vectorized load/store（int4, float4 等）？
- 是否存在冗余计算？
- 寄存器使用量（推测）

### 1.4 输出
**瓶颈分析报告**，明确主要瓶颈类型：
- **Memory-bound**：访存是主要瓶颈
- **Compute-bound**：计算是主要瓶颈
- **Latency-bound**：warp 调度延迟是主要瓶颈

---

## 阶段 2 — 规划 (Plan)

根据瓶颈类型生成优化策略，按优先级排序：

### Memory-bound 策略
| 优先级 | 策略 | 说明 |
|--------|------|------|
| P0 | 合并全局内存访问 | 确保相邻线程访问相邻地址 |
| P1 | Shared memory tiling | 分块加载到 SMEM，减少 GMEM 访问 |
| P2 | Vectorized load/store | 使用 `float4` / `int4` 提升带宽利用 |
| P3 | 减少 bank conflict | 调整 SMEM 布局（padding 等） |
| P4 | 使用 `__ldg()` | 只读数据使用 read-only cache |

### Compute-bound 策略
| 优先级 | 策略 | 说明 |
|--------|------|------|
| P0 | 降低寄存器压力 | 减少每个线程的寄存器使用，提高 occupancy |
| P1 | 循环展开 | `#pragma unroll` 减少循环开销 |
| P2 | Intrinsic 替换 | `__fmaf_rn()`、`__sinf()` 等硬件指令 |
| P3 | 指令级并行 | 混合同一 warp 内的独立计算，隐藏延迟 |
| P4 | 使用更快的数学近似 | 如 `__sinf()` 替代 `sinf()` |

### Latency-bound 策略
| 优先级 | 策略 | 说明 |
|--------|------|------|
| P0 | 提高 occupancy | 调整 block size / registers，让更多 warp 并行 |
| P1 | 增大 block size | 提高每个 SM 的 active warp 数 |
| P2 | 减少同步开销 | `__syncthreads()` 是否必要？能否减少？ |
| P3 | Async 拷贝 | `cp.async` 隐藏 H2D/D2H 延迟 |

---

## 阶段 3 — 执行 (Execute)

逐项应用优化策略，**每应用一项后**：

1. 只改动目标文件（或按需创建新变体）
2. 注释说明优化意图
3. 保留原始的对照代码（用 `#ifdef` 或单独文件）

### 优化实践规范
- 每次只做一项优化，**不要一次性应用多项**
- 每项优化后检查代码正确性（数值一致性）
- 使用 `std::cout` 输出性能日志（禁止 printf）

---

## 阶段 4 — 验证 (Verify)

### 4.1 正确性验证
- 优化前后输出对比（相对误差 / 绝对误差）
- 针对边界测试（最小/最大尺寸、对齐/非对齐）
- 随机输入测试多次

### 4.2 性能对比
- 编译优化前后版本
- 记录 latency（ms）和 bandwidth（GB/s）
- 对比 baseline 的加速比

### 4.3 输出报告
```markdown
## Kernel: <kernel_name> 优化报告

### 优化总结
| 策略 | 状态 | 加速比 |
|------|------|--------|
| shared memory tiling | ✅ | 2.3x |
| vectorized load | ✅ | 1.4x |

### 性能数据
- Baseline: XX ms
- Optimized: XX ms
- Speedup: X.XXx

### 瓶颈分析
<分析结论>
```

---

## 多 AI 兼容说明

本 skill 定义为工具无关的标准化流程，适用于：
- **Claude Code**: `.claude/skills/opt_kernel.md` 注册为 `/opt_kernel` 命令
- **Gemini CLI**: 参考 `AGENTS.md` 中"可用 Skill"部分
- **Codex CLI**: 按此流程编写等效的 prompt 规则
- **其他 AI**: 遵循相同的 4 阶段分析 - 规划 - 执行 - 验证流程

> 核心原则：**流程标准化，工具仅作为执行者。** 无论在哪个 AI 环境中，优化方法论保持一致。

---

## 快速参考

```bash
# 编译优化后的 kernel
cd build && cmake .. && make

# 运行 benchmark
./build/<target>

# 性能分析（可选）
ncu --set full -o <output> ./build/<target>
```
