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
```

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

测试覆盖 `1x1`、奇数尺寸、跨 tile 边界和非方阵，共验证 7 种 shape、4 种
数据格式和 2 种 shared-memory 布局，即 56 组逐 bit 正确性组合；同一程序
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
