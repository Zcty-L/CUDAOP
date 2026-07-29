# smem_float4_broadcast_kernel 分析报告

## 结论

- 状态：`VERIFIED`。本任务是 shared-memory transaction 诊断，不包含
  kernel 优化，因此优化状态为 `[NO_GAIN]`。
- 后端与实现：CUDA C++，单 warp 的 32 个 lane 使用同一个 shared 地址，
  每个 lane 执行一条 `ld.shared.v4.b32`。
- 主要结果：RTX 4090 / SM89 上，NCU 记录 1 条 shared load instruction、
  2 个 shared load wavefront、0 个 shared load bank conflict。
- 直接结论：如果这里的 transaction 指 NCU
  `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`，结果确实是 2。
  这两个 wavefront 不是 bank conflict 产生的额外 transaction。
- 适用范围：结论来自 RTX 4090（SM89）与 CUDA 13.1 / NCU 2025.4；
  不直接外推到其他 GPU 架构。

## 最终实现与 Dispatch

| 条件 | 实现 | 用途 |
|------|------|------|
| 默认、`--correctness-only` | `smem_float4_broadcast_kernel` | 目标 LDS.128 正确性与稳定延迟 |
| `--profile-only` | `smem_float4_broadcast_kernel` | 单次目标 kernel profile |
| `--profile-scalar` | `smem_scalar_broadcast_control_kernel` | LDS 32-bit 控制组 |
| `--profile-float2` | `smem_float2_broadcast_control_kernel` | LDS.64 控制组 |

## 地址与 Bank 映射

- block 为 1 个 warp，共 32 个线程。
- shared 对象为一个 16-byte 对齐的 `float4`。
- lane 0 至 lane 31 使用完全相同的 shared byte address。
- 每个 lane 请求相同的四个 32-bit word，对应相同的四个 bank word。
- 初始化由 lane 0 完成，`__syncthreads()` 后所有 lane 执行同址读取。

该映射构成 shared broadcast，不存在不同地址争用同一 bank 的冲突模式。

## 正确性与安全检查

- 输出正确性：128/128 个 float 精确匹配，错误数 0，最大绝对误差 0。
- CTest：`smem_float4_broadcast` 通过。
- Compute Sanitizer memcheck：0 errors。
- Compute Sanitizer racecheck：0 errors、0 warnings、0 hazards。
- Compute Sanitizer synccheck：0 errors。

## 环境与测试

- 日期 / GPU / SM：2026-07-29，NVIDIA GeForce RTX 4090，SM89。
- GPU：使用 device 0；采样前 GPU utilization 为 0%。
- Driver / CUDA / NCU：590.44.01 / CUDA 13.1 / NCU 2025.4.0。
- CMake target：`smem_float4_broadcast`，原生
  `CUDAOP_CUDA_ARCHITECTURES=89`，Release。
- dtype / shape：shared `float4[1]`，grid 1，block 32。
- 计时：每组 warmup 20 次，每次 1000 launches；100 samples；
  3 groups；CUDA Event 包含 kernel launch 与执行，不包含分配和拷贝。

## 构建与资源

| 路径 | SASS | Registers | Static shared | Barrier | Spill |
|------|------|----------:|--------------:|--------:|------:|
| scalar 控制 | `LDS` | 8 | 4 B | 1 | 0 |
| float2 控制 | `LDS.64` | 10 | 8 B | 1 | 0 |
| float4 目标 | `LDS.128` | 14 | 16 B | 1 | 0 |

目标 SASS 中只有一条目标 load：

```text
LDS.128 R4, [RZ]
```

## 稳定延迟基线

| 计时范围 | Group medians | Final median | Spread |
|----------|---------------|-------------:|-------:|
| 单 block、单 warp kernel | 2.256 / 2.263 / 2.290 us | 2.263 us | 1.48% |

该 latency 主要包含极短 kernel 的 launch 固定成本，只用于确认采样稳定，
不用于推导 shared-memory 指令延迟。

## Nsight Compute 定向证据

| 同址读取 | Shared load instructions | Load wavefronts | Load bank conflicts |
|----------|-------------------------:|----------------:|--------------------:|
| scalar / `LDS` | 1 | 1 | 0 |
| float2 / `LDS.64` | 1 | 1 | 0 |
| float4 / `LDS.128` | 1 | 2 | 0 |

NCU 将 wavefront 指标定义为 Data-Stage 处理的 shared-memory wavefront
数量。控制组说明：相同地址的 scalar 与 float2 广播只需要 1 个 wavefront，
而 float4 广播需要 2 个 wavefront；三者 bank conflict 均为 0。因此 SM89
上的结果与 LDS.128 因向量宽度被拆成两个 data-stage wavefront 一致，
不是 2-way bank conflict。

`.ncu-rep` 仅保存在 `build/` 中，不纳入版本控制。

## 实验记录

| 轮次 | 假设/测试 | 正确性 | 关键结果 | 决策 |
|-----:|-----------|:------:|----------|:----:|
| 0 | 32 lane 同址 LDS.128 | PASS | 1 inst / 2 wavefront / 0 conflict | VERIFIED |
| 1 | scalar 同址控制 | profile | 1 inst / 1 wavefront / 0 conflict | VERIFIED |
| 2 | float2 同址控制 | profile | 1 inst / 1 wavefront / 0 conflict | VERIFIED |

## 候选方向闭环

| 方向 | 状态 | 证据 |
|------|:----:|------|
| 检查是否为 bank conflict | `EXCLUDED` | NCU conflict 指标为 0 |
| 检查是否由 128-bit 宽度拆分 | `VERIFIED` | scalar/float2 为 1，float4 为 2 |
| 在其他 GPU 架构复测 | `BLOCKED` | 当前任务只指定并使用 SM89 |
| shared store transaction | `EXCLUDED` | 已有独立 `smem_sts128_transactions` 测试 |

## 剩余限制

- “transaction”在不同文档和工具中可能指 request、wavefront 或 bank
  transaction；本报告的数值 2 特指 NCU shared load wavefront 指标。
- 没有测量同址 LDS.128 的纯指令依赖延迟；当前 kernel latency 受 launch
  开销主导。
- 未做性能优化，因此没有 baseline-to-final speedup。

[NO_GAIN]
