# CuTe BF16/BF16 GEMM epilogue 实验

## 计算语义

`cute_gemm_bf16_bf16` 的接口为 BF16 A/B 输入和 BF16 C 输出。Tensor Core
使用 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`，先完成 FP32
累加，再在 epilogue 中一次性舍入为 BF16：

```text
C_BF16 = cast_BF16(accumulate_FP32(A_BF16 * B_BF16))
```

测试环境为 RTX 5070 Ti Laptop GPU（SM120），CTA tile 为 `128x128x64`，
128 threads，3-stage `cp.async` pipeline。

## Epilogue 单变量实验

使用 `128x128x128` 单 CTA 隔离输出写回。C tile 包含 16384 个 BF16，
有效数据量为 32 KiB，因此理论最少需要 1024 个 32B global sectors。

| Epilogue | registers/thread | 静态写回指令 | 动态 store requests | sectors | 有效字节/sector |
|---|---:|---:|---:|---:|---:|
| 标量 BF16 | 244 | 128 × `STG.E.U16` | 512 | 4096 | 8B |
| BF16x2 | 244 | 64 × `STG.E.32` | 256 | 2048 | 16B |
| stmatrix | 248 | 16 × `STG.E.128` | 64 | 1024 | 32B |

BF16x2 直接写回只能利用 MMA fragment 每个 lane 内相邻的两个结果，无法跨
不同 store 指令合并同一 sector，因此 sector 利用率为 50%。最终实现执行：

1. FP32 accumulator 转换成同 layout 的 BF16 register fragment。
2. 等待 `cp.async` 完成并复用 A pipeline 的前 32 KiB SMEM。
3. 使用 `stmatrix.sync.aligned.x4.m8n8.shared.b16` 写入
   `Swizzle<3,3,3>` C tile。
4. 每线程通过 `LDS.128 + STG.E.128` 写回连续的 row-major GMEM。

单 CTA 实测 `STSM` 为 256 wavefront、0 bank conflict，epilogue `LDS.128`
也为 0 bank conflict；global store 达到理论最小的 1024 sectors。

## 性能

三个冻结二进制交替执行三轮，每个进程 3 次 warmup、9 次 iteration。以下为
三轮中位数；`M=N=8192, K=128` 用于放大 epilogue 占比。

| Epilogue | Identity ms | BlockSwizzle8 ms |
|---|---:|---:|
| 标量 BF16 | 0.914496 | 0.933568 |
| BF16x2 `STG.32` | 0.479040 | 0.479584 |
| stmatrix + `STG.128` | 0.438496 | 0.437472 |

- stmatrix 相对标量：Identity `2.086x`，BlockSwizzle8 `2.134x`。
- stmatrix 相对 BF16x2：Identity `1.092x`，BlockSwizzle8 `1.096x`。

在 `8192x8192x8192` 下，主循环占主导。每个进程 2 次 warmup、5 次
iteration，交替三轮后的中位数如下：

| Epilogue | Identity ms | BlockSwizzle8 ms |
|---|---:|---:|
| 标量 BF16 | 31.8277 | 30.9753 |
| BF16x2 `STG.32` | 31.5000 | 31.3384 |
| stmatrix + `STG.128` | 31.8200 | 30.6970 |

大 K 下差异约为 1% 量级，易受 GPU 时钟波动影响；确定性的收益是 global
store sectors 从 4096 降至理论最小的 1024。最终默认保留 stmatrix 版本。

## 验证

- CMake target 和 CTest 通过。
- `256x256x128` 与 `8192x8192x8192` 相对 cuBLAS 均为 0 mismatch。
- Identity 与 BlockSwizzle8 bitwise mismatch 为 0。
- Compute Sanitizer memcheck：0 errors。
- Compute Sanitizer racecheck：0 errors，0 warnings。
- stmatrix epilogue 要求 compute capability 9.0+；Blackwell target 使用
  family-specific `sm_120f` 编译。
