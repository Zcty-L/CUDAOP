# FP16 Conv2D + LIF 融合对比报告

## 结论

- 对比对象：独立 `snn_conv2d_64x64_k16_fp16_u8` + FP16 LIF，
  与融合 `snn_conv2d_lif_64x64_k16_fp16`。
- 大吞吐配置提升 `3.334x`，即延迟降低 `70.01%`。
- 常用 3x3/s1 业务配置提升 `1.048x`，即延迟降低 `4.55%`。
- 小型 3x3/s2 边界配置提升 `1.089x`，即延迟降低 `8.14%`。
- 三项配置中，CPU、独立 GPU 路径和融合 GPU 路径均逐字节一致。
- 结论与 FP32 一致：融合对大中间张量收益显著，一般配置收益较小。

## 测试配置明细

所有配置均使用 packed uint8 spike 输入、FP16 权重、FP16 卷积累加和
packed uint8 spike 输出。FP16 融合实现固定 `v_reset=0`。

| 测试配置 | T | C_in | 输入 HxW | C_out | K / stride / padding | 输出 HxW | v_th | v_reset | tau | Grid blocks |
|----------|--:|-----:|----------:|------:|----------------------|----------:|-----:|--------:|----:|------------:|
| `throughput_1x1` | 4 | 16 | 128x128 | 512 | 1x1 / 1 / 0 | 128x128 | 0.50 | 0.00 | 0.50 | 2048 |
| `business_3x3_s1` | 4 | 64 | 80x80 | 64 | 3x3 / 1 / 1 | 80x80 | 1.00 | 0.00 | 0.50 | 100 |
| `boundary_3x3_s2` | 3 | 32 | 43x43 | 48 | 3x3 / 2 / 1 | 22x22 | 0.75 | 0.00 | 0.25 | 8 |

## 性能结果

| 测试配置 | Separate group medians (ms) | Fused group medians (ms) | Separate ms | Fused ms | Speedup | 提升 | Fused spread |
|----------|-----------------------------|--------------------------|------------:|---------:|--------:|-----:|-------------:|
| `throughput_1x1` | 0.139981 / 0.139962 / 0.140061 | 0.041890 / 0.041984 / 0.042086 | 0.139981 | 0.041984 | 3.334x | 70.01% | 0.469% |
| `business_3x3_s1` | 0.084696 / 0.084770 / 0.084755 | 0.080896 / 0.080896 / 0.080896 | 0.084755 | 0.080896 | 1.048x | 4.55% | 0.000% |
| `boundary_3x3_s2` | 0.037682 / 0.037581 / 0.037450 | 0.034560 / 0.034522 / 0.034509 | 0.037581 | 0.034522 | 1.089x | 8.14% | 0.148% |

Separate 和 Fused 分开预运行、分开测试。每种实现先执行 5000 次不计时
稳态预运行；每组 warmup 20 个批次，正式采样 100 个批次，每批 10 次
launch，共重复 3 组。表中的最终延迟为三个 group median 的中位数。

## 数据流解释

独立路径先写出 `[T, C_out, H_out, W_out]` FP16 卷积结果，再由 LIF
读取；融合路径在寄存器中完成 LIF。融合消除的中间张量聚合读写量为：

`4 * T * C_out * H_out * W_out` 字节。

| 测试配置 | FP16 中间张量单向大小 | 消除的聚合流量 |
|----------|-----------------------:|-----------------:|
| `throughput_1x1` | 64 MiB | 128 MiB |
| `business_3x3_s1` | 3.125 MiB | 6.25 MiB |
| `boundary_3x3_s2` | 136.125 KiB | 272.25 KiB |

大吞吐配置的中间数据流量远高于另外两项，因此融合收益最大。业务配置只有
100 个 Conv block，边界配置只有 8 个 Conv block，固定 launch 开销、低并行度
和卷积主体占比更高，融合收益分别只有 4.55% 和 8.14%。

## 正确性与安全检查

- `throughput_1x1`：`0 / 8,388,608` errors。
- `business_3x3_s1`：`0 / 409,600` errors。
- `boundary_3x3_s2`：`0 / 23,232` errors。
- 每项均完成 separate 对 CPU、fused 对 CPU、fused 对 separate 三次比较，
  `max_abs=0`、`max_rel=0`。
- 原 `conv2d_k64_u8_fp16` 的 13 个配置全部通过。
- 原 `conv2d_sn_k64_u8_fp16` 的 15 个配置全部通过。
- `compute-sanitizer --tool memcheck`：完整三配置矩阵，0 errors。
- `compute-sanitizer --tool synccheck`：3x3/s2 边界配置，0 errors。

## 构建与运行

```bash
cmake -S . -B build -DCUDAOP_CUDA_ARCHITECTURES=89
cmake --build build --target conv2d_lif_fusion_cmp_fp16 -j 4
./build/conv2d_lif_fusion_cmp_fp16
```

测试环境为 NVIDIA GeForce RTX 4090、SM 8.9、CUDA 13.1、
Driver 590.44.01。

[SUCCESS]
