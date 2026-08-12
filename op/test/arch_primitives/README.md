# CUDA 架构原语验证

本目录用于独立验证 CUDA 架构原语，不放置完整算子实现。

可复用的生产代码 PTX wrapper 仍应统一放在 `op/ptx_utils.cuh`。本目录是
学习验证例外：`ptx_*.cu`、`tma_swizzle.cu` 和 `cluster_*.cu` 为了直接观察
指令 qualifier、操作数与硬件降低结果，刻意在各自文件内保留 raw inline
PTX；这些本地 wrapper 不是公共接口。

## 基础架构原语

- `tma_swizzle.cu`：直接发出 raw `cp.async.bulk.tensor` 与 mbarrier，通过
  load/store 闭环验证 None、32B、64B 和 128B swizzle/unswizzle。
- `cluster_launch.cu`：直接读取 cluster 特殊寄存器并发出 raw cluster
  barrier，验证显式 thread block cluster 启动。
- `cluster_dsm.cu`：直接发出 raw `mapa.shared::cluster`、
  `ld.shared::cluster` 与 cluster barrier，验证 distributed shared memory。

## PTX 指令拆解

学习文件对应 `feature/linear` 分支提交 `0562768` 中的 CuTe GEMM。
每个文件都直接发出 raw PTX，用 CPU reference 验证结果，并可以通过
`cuobjdump` 观察最终 SASS。

| 学习文件 | 主要 PTX | 对应 linear 实现 | SM120 SASS |
| --- | --- | --- | --- |
| `ptx_cp_async.cu` | `cp.async.ca.shared.global` 16B、`commit_group`、`wait_group 0` | FP16/BF16/FP8/INT8/W4A16 的 16B G2S | `LDGSTS`、`LDGDEPBAR`、`DEPBAR` |
| `ptx_tma.cu` | 1D/2D `cp.async.bulk.tensor` load、2D store、`mbarrier`、bulk group、proxy fence | 原 `tma_copy` 与 TMA G2S pipeline 的统一学习入口 | `UTMALDG.1D/.2D`、`UTMASTG.2D`、`UTMACMDFLUSH` |
| `ptx_mma_float.cu` | FP16、BF16 `m16n8k16` 和 TF32 `m16n8k8` `mma.sync` | `cute_gemm_fp16_fp32.cu`、`cute_gemm_bf16_fp32.cu`、`cute_gemm_tf32.cu` | `HMMA` |
| `ptx_mma_fp4.cu` | NVFP4 `mxf4nvf4.block_scale.scale_vec::4X` | `cute_gemm_fp4_fp16.cu` | `OMMA.SF...UE4M3.4X` |
| `ptx_mma_fp8.cu` | E4M3 `f8f6f4` 与 MXFP8 `mxf8f6f4` | `cute_gemm_fp8_fp32.cu`、`cute_gemm_mxfp8_fp16.cu` | `QMMA`、`QMMA.SF...E8` |
| `ptx_mma_fp6.cu` | E3M2 `f8f6f4` 与 MXFP6 `mxf8f6f4` | `cute_gemm_fp6_mxfp6.cu` | `QMMA`、`QMMA.SF...E8` |
| `ptx_mma_integer.cu` | S8/S4 `m16n8k32` `mma.sync` | `cute_gemm_int8_int32.cu`、`cute_gemm_int4_int32.cu` | S8 `IMMA`；S4 在 SM120 上降为两条 S8 `IMMA` |
| `ptx_mma_sparse.cu` | FP16 2:4 `mma.sp::ordered_metadata` | `cute_gemm_sparse.cu` | `HMMA.SP` |

### `cute_gemm_fp4_fp16.cu` 的数据路径

`cute_gemm_fp4_fp16.cu` 没有使用 `cp.async`。它通过 CuTe `copy` 把 packed
E2M1 直接从 GMEM 搬到 MMA 寄存器 fragment，然后执行：

```text
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.
    m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

因此：

- `ptx_mma_fp4.cu` 隔离的是该 GEMM 真正使用的 NVFP4 MMA。
- `ptx_cp_async.cu` 来自其他已实现 SMEM pipeline 的 GEMM，用于单独学习
  GMEM 到 SMEM 搬运，不应解读为 FP4 GEMM 已使用 `cp.async`。

每个微基准应作为独立目标注册到项目根目录的 `CMakeLists.txt`，测试输出应包含：

- 当前 GPU 和测试配置。
- 主要执行阶段。
- 正确性及关键性能结果。
- `[SUCCESS]` 成功标记。

当前 RTX 5070 Ti Laptop（SM120）可运行 TMA、TMA swizzle、thread block
cluster、DSM 和上述 warp-level MMA 实验。TMEM/`tcgen05` 实验需要
SM100/SM101 环境，不在本机运行目标范围内。

构建并运行全部架构原语测试：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build \
    --target \
        tma_swizzle cluster_launch cluster_dsm \
        ptx_cp_async ptx_tma ptx_mma_float ptx_mma_fp4 ptx_mma_fp8 \
        ptx_mma_fp6 ptx_mma_integer ptx_mma_sparse
ctest --test-dir build \
    -R '^(tma_swizzle|cluster_launch|cluster_dsm|ptx_.*)$' \
    --output-on-failure
```

检查编译器实际保留的 PTX 和硬件指令：

```bash
cuobjdump --dump-ptx build/ptx_mma_fp4
cuobjdump --dump-sass build/ptx_mma_fp4
```

`tma_swizzle` 验证 Tensor Map swizzle/unswizzle 的硬件闭环正确性。shared
memory bank conflict 的性能差异需要后续使用具有相同访存模式的 kernel
和 Nsight Compute 单独测量，不能从这个闭环测试的 kernel latency 推断。
