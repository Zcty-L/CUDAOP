# MxN 矩阵转置

## 功能

`op/transpose` 提供 row-major `M x N` 到 row-major `N x M` 的 CUDA 转置，
支持以下数据格式：

- `float`：32-bit 容器；
- `bf16`：CUDA `__nv_bfloat16` 16-bit 容器；
- `fp8`：CUDA `__nv_fp8_e4m3` 8-bit 容器；
- `fp4`：CUDA NVFP4 E2M1 的双元素打包格式。

转置只搬运元素编码，不做数值转换，因此正确性测试采用逐 bit 比较。

FP4 在 global memory 中按行独立打包：偶数列位于低 nibble，奇数列位于高
nibble，每行占用 `ceil(N / 2)` 字节。输出每行占用 `ceil(M / 2)` 字节；当
输出宽度为奇数时，最后一个字节未使用的高 nibble 固定清零。

## Shared memory 布局

两个 kernel 都使用 `32 x 32` 逻辑 tile 和 `32 x 8` 线程块。

### Pad

Pad 版本在每个 shared-memory 行尾添加一个 4-byte bank：

```text
pad_elements = 4 / sizeof(element)
shared_columns = 32 + pad_elements
```

因此转置后的列读取相邻行时，bank 起点每行旋转一个 bank。

### XOR Swizzle

Swizzle 版本保持紧凑的 `32 x 32` shared tile。设一个 4-byte bank 能容纳
`E = 4 / sizeof(element)` 个元素，逻辑坐标 `(row, column)` 映射为：

```text
logical_bank  = column / E
intra_bank    = column % E
physical_bank = logical_bank XOR (row / E)
physical_col  = physical_bank * E + intra_bank
```

行 stride 自然提供 `row` 的低位，XOR 部分提供其余高位，使同一逻辑列的
32 行分散到 32 个 bank。FP4 在 shared memory 内临时展开为一个 byte 一个
逻辑元素，global memory 仍保持双元素打包；输出阶段由一个线程完整写入一
个 byte，避免两个 nibble 之间产生写竞争。

| 类型 | Pad shared memory | Swizzle shared memory |
|---|---:|---:|
| float | 4224 B | 4096 B |
| bf16 | 2176 B | 2048 B |
| fp8-e4m3 | 1152 B | 1024 B |
| fp4-e2m1 | 1152 B | 1024 B |

## BF16/FP8 32-bit 向量化实现

`op/transpose/transpose_vectorized.cu` 提供独立的 32-bit pack 实现：

- BF16 使用一个 4-byte 对齐的 `Bfloat16x2Pack` 搬运两个元素；
- FP8 E4M3 使用一个 4-byte 对齐的 `Float8E4M3x4Pack` 搬运四个元素；
- BF16 和 FP8 线程块分别为 `16 x 8` 和 `8 x 8`，仍覆盖一个
  `32 x 32` 逻辑 tile；
- 输入侧执行一次 32-bit global load 和一次 32-bit shared store；
- 输出侧从 shared memory gather 两个或四个元素，在寄存器中重组后执行一次
  32-bit global store；
- 行首地址未对齐或 tile 边界不足一个 pack 时，仅对应 pack 使用逐元素 bit
  搬运，不改变任意 `M x N` 尺寸支持。

Pad 布局仍使用普通 row-major 行尾 padding：BF16 每行 17 个 32-bit word，
FP8 每行 9 个 word。输入时调整线程访问逻辑行的顺序，使同一 warp 访问的
BF16 行相隔 16 行、FP8 行相隔 8 行，避免多个短行的 bank 区间重叠。Swizzle
布局直接对 32-bit word 下标执行：

```text
physical_word = logical_word XOR (row / pack_elements)
```

在 RTX 4090、CUDA 13.1、SM89 上对独立完整 `32 x 32` CTA 使用 Nsight
Compute 检查，结果如下：

| 类型 | 布局 | shared load conflicts | shared store conflicts | load/store wavefronts |
|---|---|---:|---:|---:|
| BF16x2 | Pad | 0 | 0 | 32 / 16 |
| BF16x2 | Swizzle | 0 | 0 | 32 / 16 |
| FP8x4 | Pad | 0 | 0 | 32 / 8 |
| FP8x4 | Swizzle | 0 | 0 | 32 / 8 |

反汇编的完整对齐路径包含 32-bit `LDG.E` 和 `STG.E`；同一 kernel 中保留的
`LDG.E.U8`/`STG.E.U8` 指令用于未对齐和尾部 pack。

## 接口

接口声明位于 `op/transpose/transpose.h`：

```cpp
cudaError_t transpose_cuda(
    const void *input,
    void *output,
    uint32_t rows,
    uint32_t columns,
    TransposeDataType data_type,
    TransposeSharedMemoryLayout shared_memory_layout,
    cudaStream_t stream = nullptr);

cudaError_t transpose_vectorized_cuda(
    const void *input,
    void *output,
    uint32_t rows,
    uint32_t columns,
    TransposeDataType data_type,
    TransposeSharedMemoryLayout shared_memory_layout,
    cudaStream_t stream = nullptr);
```

`transpose_vectorized_cuda` 只接受 BF16 和 FP8 E4M3；其他数据格式返回
`cudaErrorInvalidValue`。原有 `transpose_cuda` 保留为覆盖四种数据格式的标量
基线接口。

`transpose_storage_bytes` 可分别计算输入 `M x N` 和输出 `N x M` 的存储大小，
这对奇数宽度的 packed FP4 尤其必要。当前实现不支持原地转置，输入输出地址
不得相同。

## 构建与验证

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target transpose_test -j
ctest --test-dir build -R '^transpose_test$' --output-on-failure

# 只运行 API 与正确性验证（适合 compute-sanitizer）
./build/transpose_test --validate-only

compute-sanitizer --tool memcheck \
    ./build/transpose_test --validate-only
compute-sanitizer --tool racecheck \
    ./build/transpose_test --validate-only
```

测试覆盖 `1x1`、完整 `32x32` tile、奇数尺寸、跨 tile 边界和非方阵。标量
实现验证 8 种 shape、4 种数据格式和 2 种 shared-memory 布局，向量实现验证
8 种 shape、2 种数据格式和 2 种布局，共 96 组逐 bit 正确性组合；同一程序
还会输出 `1024x1024`、`3072x4096` 和 `4096x4096` 的 CUDA Event 性能结果。

2026-08-22 在 RTX 4090、CUDA 13.1、SM89 上的一次 `4096x4096` 示例结果如下。
有效带宽按 `(input_bytes + output_bytes) / time` 计算；数据经过 warmup 后可能
命中 L2，因此该数值不能直接等同于 DRAM 带宽。

| 类型 | Pad ms / GB/s | Swizzle ms / GB/s |
|---|---:|---:|
| float | 0.1558 / 861.58 | 0.1552 / 864.53 |
| bf16 | 0.0219 / 3061.00 | 0.0229 / 2927.02 |
| fp8-e4m3 | 0.0175 / 1919.63 | 0.0223 / 1506.57 |
| fp4-e2m1 | 0.0227 / 739.38 | 0.0293 / 572.07 |

2026-08-24 在同一设备上的一次 `4096x4096` 标量/32-bit 向量化对比如下：

| 类型 | 布局 | 标量 ms / GB/s | vector32 ms / GB/s |
|---|---|---:|---:|
| bf16 | Pad | 0.0221 / 3042.53 | 0.0225 / 2978.91 |
| bf16 | Swizzle | 0.0229 / 2928.33 | 0.0230 / 2919.20 |
| fp8-e4m3 | Pad | 0.0176 / 1910.67 | 0.0152 / 2214.05 |
| fp8-e4m3 | Swizzle | 0.0223 / 1502.43 | 0.0160 / 2095.14 |

该次测试中 BF16 基本持平，FP8 的 32-bit pack 收益更明显；这些数据用于展示
本机相对趋势，不作为跨设备性能保证。
