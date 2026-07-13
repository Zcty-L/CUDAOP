# `snn_conv2d_lif_64x64_k16` 融合对比报告

## 结论

- 状态：KEEP。FP32 Conv2D 与 LIF 融合在三个测试形状上均有提升。
- 主要结果：吞吐形状提升 `6.523x`，业务形状提升 `1.083x`，
  边界形状提升 `1.094x`；GPU、CPU 和两条 GPU 路径的输出逐字节一致。
- 适用范围：输入为 packed `uint8_t` spike，权重和卷积累加为 FP32，
  输出为 packed `uint8_t` spike，时间步 `T <= 8`。
- 关联历史报告：无；LIF 语义实现见提交 `26988e7`。

## 最终实现与 Dispatch

| 条件 | 实现 | 关键改动 | Fallback |
|------|------|----------|----------|
| 1x1/s1/p0 | `snn_conv2d_lif_64x64_k16` | Conv 累加后在寄存器内执行 LIF，并直接写 packed 输出 | `snn_conv2d_64x64_k16_u8` + 独立 LIF |
| 3x3/s1/p1 | `snn_conv2d_lif_64x64_k16` | 同上 | 同上 |
| 3x3/s2/p1 | `snn_conv2d_lif_64x64_k16` | 同上 | 同上 |

测试目标 `conv2d_lif_fusion_cmp` 复用两个生产 kernel 的 launch wrapper；
独立路径新增仅供测试使用的 `lif_hard_reset_packed_kernel`。

## 最终性能

| 测试配置 | Baseline group medians (ms) | Final group medians (ms) | Speedup | 提升 | Final spread | Reference |
|----------|-----------------------------|--------------------------|--------:|-----:|-------------:|----------:|
| `throughput_1x1` | 0.315362 / 0.315262 / 0.315290 | 0.048333 / 0.048333 / 0.048333 | 6.523x | 84.67% | 0.000% | CPU + separate |
| `business_3x3_s1` | 0.088957 / 0.088896 / 0.088912 | 0.082122 / 0.082118 / 0.082123 | 1.083x | 7.64% | 0.006% | CPU + separate |
| `boundary_3x3_s2` | 0.038912 / 0.038912 / 0.038912 | 0.035608 / 0.035558 / 0.035539 | 1.094x | 8.62% | 0.193% | CPU + separate |

这里的 Baseline 是原 Conv kernel 加独立 LIF kernel 的端到端 CUDA Event
时间，Final 是融合 kernel 的端到端时间。每组结果取 100 个样本的中位数，
最终结果取三组中位数的中位数。

### 测试配置明细

| 测试配置 | T | C_in | 输入 HxW | C_out | K / stride / padding | 输出 HxW | v_th | v_reset | tau | Grid blocks | Waves/SM |
|----------|--:|-----:|----------:|------:|----------------------|----------:|-----:|--------:|----:|------------:|---------:|
| `throughput_1x1` | 4 | 16 | 128x128 | 512 | 1x1 / 1 / 0 | 128x128 | 0.50 | 0.00 | 0.50 | 2048 | 8.000 |
| `business_3x3_s1` | 4 | 64 | 80x80 | 64 | 3x3 / 1 / 1 | 80x80 | 1.00 | 0.00 | 0.50 | 100 | 0.391 |
| `boundary_3x3_s2` | 3 | 32 | 43x43 | 48 | 3x3 / 2 / 1 | 22x22 | 0.75 | -0.10 | 0.25 | 8 | 0.031 |

因此，提升 `7.64%` 对应 `business_3x3_s1`，提升 `8.62%` 对应
`boundary_3x3_s2`；二者不是吞吐饱和配置。

## 正确性与安全检查

- 正确性：三个对比形状中，separate 对 CPU、fused 对 CPU、fused 对
  separate 均为 `0` errors，`max_abs=0`、`max_rel=0`。
- 原始回归：`conv2d_k64_u8` 的 13 个形状和 `conv2d_sn_k64_u8` 的
  16 个形状全部通过，均为 0 errors。
- `compute-sanitizer --tool memcheck`：三个对比形状全部通过，0 errors。
- `compute-sanitizer --tool synccheck`：3x3/s2 边界形状通过，0 errors。
- `compute-sanitizer --tool racecheck`：3x3/s2 边界形状未清零。
  工具在原 Conv 和融合 Conv 的同一异步 shared-memory 流水位置各报告
  一组 write/read hazard，共 2 errors；两条实现均有此现象，数值结果仍为
  0 errors。该问题不由本次测试代码引入，但需要后续结合带 lineinfo 的构建
  单独确认 `cp.async` 管线同步，当前不能把 racecheck 记为通过。
- 边界路径：覆盖非 64 倍数输出通道、非整 tile 空间尺寸、T=3、stride=2、
  `v_reset=-0.1` 和 `tau=0.25`。

## 环境与测试

- 日期 / GPU / SM：2026-07-13，NVIDIA GeForce RTX 4090，SM 8.9，128 SM。
- CUDA / Driver / NCU：CUDA 13.1、Driver 590.44.01、NCU 2025.4.0.0。
- 源文件 / kernel / CMake target：`op/conv/conv2d_k64_u8.cu` /
  `snn_conv2d_64x64_k16_u8`，`op/conv/conv2d_sn_k64_u8.cu` /
  `snn_conv2d_lif_64x64_k16`，target 为 `conv2d_lif_fusion_cmp`。
- 数据类型 / 输入范围 / 容差：packed uint8 输入、FP32 权重与偏置、packed
  uint8 输出；固定随机种子 42；逐字节比较，容差为 0。
- 计时：每种实现先执行 5000 次稳态预运行；每组 20 次预热，每次预热
  包含 10 次 launch；正式采样 100 次、每个样本 10 次 launch，共 3 组。
- 构建：`cmake -S . -B build -DCUDAOP_CUDA_ARCHITECTURES=89`，随后
  `cmake --build build --target conv2d_lif_fusion_cmp -j 4`。

## 资源与关键性能证据

| 路径 | Registers | Shared memory | Spill | Baseline / Final NCU | 关键指标 |
|------|----------:|--------------:|------:|----------------------|----------|
| 原 Conv，T=4 | 122 | 10 KiB | 0 | Baseline | 128.86 us；DRAM 76.84%；SM 34.11% |
| 独立 LIF，T=4 | 18 | 0 | 0 | Baseline | 151.74 us；DRAM 94.54%；SM 9.31% |
| 融合 Conv+LIF，T=4 | 91 | 10 KiB | 0 | Final | 59.20 us；DRAM 0.61%；SM 76.99% |

NCU 使用相同的吞吐形状和 `--set basic`，profiler 多 pass 的绝对时间不用于
正式 speedup，只用于路径间的瓶颈定性。独立 LIF 明显受 DRAM 限制；融合后
数据保留在寄存器中，kernel 转为计算侧主导。

融合路径避免了 FP32 中间张量的一次写回和一次读取：

| 配置 | 中间张量单向大小 | 消除的聚合流量 |
|------|-----------------:|-----------------:|
| T4/C16/128x128/C512 | 128 MiB | 256 MiB |
| T4/C64/80x80/C64 | 6.25 MiB | 12.5 MiB |
| T3/C32/43x43/C48 | 272.25 KiB | 544.5 KiB |

吞吐形状的中间流量最大，因此端到端收益也最大。业务和边界形状 grid 分别
只有 100 和 8 个 block，launch 与低 wave 数的固定成本占比更高，收益较小。

## 实验记录

| 轮次 | 改动 | 关键结果 | 决策 | Commit |
|-----:|------|----------|:----:|:------:|
| 0 | LIF hard-reset 语义替换 | 原 kernel 回归通过 | KEEP | `26988e7` |
| 1 | separate Conv + LIF 与 fused 对比框架 | 三形状均 0 errors | KEEP | 本次提交 |
| 2 | 每种实现增加 5000 次稳态预运行 | 业务基线 spread 从 7.09% 降至 0.07% 以下 | KEEP | 本次提交 |
| 3 | 三形状完整性能回归 | 6.523x / 1.083x / 1.094x | KEEP | 本次提交 |

## 剩余限制

- 剩余瓶颈：吞吐形状下融合 kernel 的 NCU SM Throughput 为 76.99%，
  当前主要受计算侧限制；理论 occupancy 为 33.33%，寄存器限制为 2 blocks/SM。
- 未执行项及原因：本轮只验证用户指定的 FP32 路径，未测试 FP16。
- 安全检查限制：racecheck 在两个生产 Conv kernel 的既有 `cp.async`
  shared-memory 管线报告 hazard，尚未完成源码级归因与修复。
- 性能可信度限制：结果来自单台 RTX 4090 的 device 0；NCU 会改变时序，
  所以只把独立 CUDA Event 数据用于 speedup。

[SUCCESS]
