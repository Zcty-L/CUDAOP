# CUDAOP

CUDAOP 是一个面向 CUDA 算子实现、性能分析与优化的实验项目。仓库包含独立的CUDA/C++ benchmark、与 cuDNN 等实现的性能对比，以及部分可由 PyTorch 调用的扩展算子。

## 算子

- 卷积：Conv2D、Depthwise Conv2D 与融合算子
- 线性代数：Linear、Grouped GEMM、LoRA MoE、MMA/WGMMA
- 神经元：LIF 等脉冲神经网络算子
- 其他：Resize、逐元素乘法、QK Attention 与异步拷贝实验

核心实现位于 [`op/`](op)，优化记录与项目文档位于 [`docs/`](docs)。

## 构建

项目需要 CMake 3.18+、支持的 NVIDIA GPU、CUDA Toolkit 和 cuDNN。

```bash
cmake -S . -B build
cmake --build build -j
```

CMake 默认查询当前 GPU 的计算能力并据此构建。需要指定目标架构时，可使用：

```bash
cmake -S . -B build -DCUDAOP_CUDA_ARCHITECTURES=89
```

每个主要 `.cu` 文件会生成同名可执行程序，例如：

```bash
./build/opt_conv2d_groups_fp32
```

PyTorch 扩展和对应测试可按需单独构建：

```bash
cmake --build build --target lora_moe_ops
cmake --build build --target neuron_ops
cmake --build build --target cudaop_grouped_gemm_test
```

具体算子的输入配置、正确性验证与性能结果请查阅其源码及 [`docs/`](docs) 中的记录。
