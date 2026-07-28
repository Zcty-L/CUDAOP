# Top-K 算子

## 接口语义

本实现处理行主序 `float32 [rows, columns]` 矩阵，沿最后一维选择每行最大的
`k` 个元素，输出 `float32 [rows, k]` 值和 `int32 [rows, k]` 列索引。

排序规则如下：

1. 数值降序。
2. 数值相同时，列索引升序。
3. `NaN` 排在所有数值之前，多个 `NaN` 仍按列索引升序。

`topk_cpu` 是 C++ 参考实现，使用 `std::partial_sort`。
`topk_cuda` 是异步 CUDA 接口，在调用方给出的 CUDA stream 上执行。
输入和输出设备内存不可重叠。

## CUDA 算法

每一行由一个 256 线程的 CUDA block 处理：

1. 每个线程扫描自己负责的列，保存线程局部最大元素。
2. 使用 warp shuffle 和共享内存完成 block 级归约。
3. 输出当前全局最大元素。
4. 只有命中线程继续寻找自己负责数据中的下一个候选，重复到得到 `k` 个结果。

该算法不需要额外 workspace，适合较小 `k` 的批量行 Top-K。它是当前实现基线，
不是面向超大 `k` 的完整排序替代方案。

## PyTorch radix-select 适配

`op/topk/SortingRadixSelect.cuh` 是对
[PyTorch 2.9.1 SortingRadixSelect.cuh](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/SortingRadixSelect.cuh)
的独立 float32 适配。
它去除了 ATen/c10 内部依赖，保留以下核心过程：

1. 将 IEEE 754 float32 映射为保持数值顺序的无符号 radix key。
2. 从最高位开始，每轮使用 2 个 bit 和 4 个桶统计候选分布。
3. 根据剩余排名选择包含第 K 个元素的桶，逐轮收窄 radix 前缀。
4. 目标桶只有一个元素时提前返回，否则确定全部 32 bit 后反变换。

该头文件中的 `radix_select` 只返回第 K 大或第 K 小的阈值。
`topk_radix_cuda` 参考
[PyTorch 2.9.1 TensorTopK.cu](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/TensorTopK.cu)
补全以下阶段：

1. 通过 block exclusive binary prefix scan 收集严格优于阈值的元素。
2. 按原始列顺序收集足够数量的等阈值元素，使结果数量严格等于 K。
3. 同时写出 `float32 [rows, k]` values 和 `int32 [rows, k]` indices。
4. `sorted=true` 时对每行收集到的 K 个二元组执行确定性排序。

`topk_radix_cuda` 支持第 K 大、第 K 小以及 sorted/unsorted 输出。其
float radix key 与 PyTorch 一样，将所有 `NaN` 视为最大 key，并区分
`+0` 和 `-0`。相同 radix key 按原始列索引升序处理。

当前排序阶段使用不申请 workspace 的 block 选择排序，适合较小 K；
复杂度约为 `O(K² / blockDim.x)`。PyTorch 正式实现会为 sorted 输出调用
专门的排序 kernel，因此当前实现用于学习和正确性验证，不代表 PyTorch
大 K 场景的完整性能策略。

测试覆盖阈值原语和完整 values/indices 输出，包括连续与跨步切片、
第 K 大、第 K 小、sorted/unsorted、重复值、`NaN`、无穷、带符号零及
`k=columns`。改编许可证保存在 `op/topk/PYTORCH_LICENSE`。

## NVIDIA 官方库支持

截至 2026-07-23：

- cuBLAS/cuBLASLt 没有通用 Top-K API。它们提供 BLAS、矩阵乘和相关线性代数接口。
- cuDNN 的通用 Frontend/Backend 算子列表没有任意张量 Top-K。cuDNN Frontend
  的实验性 NSA 模块有专用 `TopKReduction`，但它面向稀疏注意力、要求 SM100+，
  不是通用 Top-K。
- 新版 CCCL/CUB 文档已经列出 `cub::DeviceTopK` 和
  `cub::DeviceBatchedTopK`，它们才是最接近本算子的 NVIDIA 官方 CUDA 原语。
  当前项目环境中的 CUDA 13.1 / CCCL 3.1.2 还没有这些头文件。
- TensorRT 提供 `ITopKLayer`，适合推理网络图，不是独立的 cuBLAS/cuDNN 函数。

参考：

- [cuDNN 操作目录](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html)
- [cuDNN NSA Top-K Reduction](https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/nsa.html)
- [cuBLAS API 目录](https://docs.nvidia.com/cuda/cublas/contents.html)
- [CUB Device-wide Primitives](https://nvidia.github.io/cccl/unstable/cub/api/device.html)
- [TensorRT TopK](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/TopK.html)

## 构建与测试

```bash
cmake -S . -B build
cmake --build build \
    --target topk_test sorting_radix_select_test topk_radix_test -j
ctest --test-dir build \
    -R '^(topk_test|sorting_radix_select_test|topk_radix_test)$' \
    --output-on-failure
```

性能测试只计量输入和输出已经驻留 GPU 时的 kernel 执行时间，不包含主机与设备之间的
内存传输。
