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
cmake --build build --target topk_test -j
ctest --test-dir build -R '^topk_test$' --output-on-failure
```

性能测试只计量输入和输出已经驻留 GPU 时的 kernel 执行时间，不包含主机与设备之间的
内存传输。
