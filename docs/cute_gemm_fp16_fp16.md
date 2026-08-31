# CuTe FP16 accumulate GEMM

## 语义

`cute_gemm_fp16_fp16` 验证真正的 FP16 Tensor Core 累加：

```text
D_FP16 = A_FP16 * B_FP16 + C_FP16
```

CuTe atom 为 `SM80_16x8x16_F16F16F16F16_TN`，对应 PTX：

```text
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
```

SM120 SASS 已确认为 `HMMA.16816.F16`。这与
`cute_gemm_fp16_fp32` 的 FP32 accumulator 不同；FP16 accumulator 在长 K
归约过程中持续发生 FP16 舍入。cuBLAS 参考使用 FP16 A/B/C、FP16 alpha/beta
以及 `CUBLAS_COMPUTE_16F`。

## 实现

- CTA tile：`128x128x64`，128 threads，3-stage pipeline。
- GMEM → SMEM：每线程 16B `cp.async`。
- SMEM：`Swizzle<3,3,3>`。
- SMEM → registers：`ldmatrix.x4`。
- MMA：FP16 input、FP16 accumulate。
- Epilogue：FP16 accumulator 直接通过 `stmatrix.b16` 写入复用的 swizzled
  SMEM，再用 `LDS.128 + STG.E.128` 写回 FP16 C。
- stmatrix epilogue 要求 compute capability 9.0+；Blackwell 使用 `sm_120f`。

## 指标与结果

编译资源：166 registers/thread，0 spill。`128x128x128` 单 CTA 的写回指标：

| 指标 | 结果 |
|---|---:|
| global-store requests | 64 |
| global-store sectors | 1024（理论最小） |
| 有效字节/sector | 32B |
| STSM wavefronts | 256 |
| STSM bank conflict | 0 |
| epilogue LDS bank conflict | 0 |

`8192x8192x8192`，1 次 warmup、3 次 iteration 的一次验证结果：

| 实现 | Median ms | TFLOP/s |
|---|---:|---:|
| Identity | 22.6480 | 48.5479 |
| BlockSwizzle8 | 15.0898 | 72.8644 |
| cuBLAS FP16 compute | 13.7743 | 79.8234 |

Identity/BlockSwizzle8 相对 cuBLAS 均为 0 mismatch，两种 CTA 映射之间为
0 bitwise mismatch。CPU 表格中的 `CPU FP16 once` 是 FP64 dot product 最终
只舍入一次，不能作为逐步 FP16 accumulator 的 bitwise 参考。

## 回归

- CMake target 构建通过。
- CTest 通过。
- Compute Sanitizer memcheck：0 errors。
- Compute Sanitizer racecheck：0 errors，0 warnings。
