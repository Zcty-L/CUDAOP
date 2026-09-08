# CUDAOP

CUDAOP 是一个面向 CUDA 算子实现、性能分析与优化的实验项目。仓库包含独立的 CUDA/C++ benchmark、与 cuDNN 等实现的性能对比，以及部分可由 PyTorch 调用的扩展算子。

## 算子

- 卷积
  - [Conv2D](op/conv)：FP32/FP16 卷积、脉冲卷积以及 Conv2D + LIF 融合实现
  - [Depthwise Conv2D](op/dwconv)：FP32/FP16、脉冲神经网络与分组卷积优化实现
- 线性代数与 MoE
  - [Linear](op/linear)：全精度与脉冲神经网络线性层
  - [Grouped GEMM](op/grouped_gemm)：CUTLASS、Triton 和 cuTile 实现，包含 LoRA 前向与反向路径
  - [LoRA MoE](op/lora_moe)：路由辅助算子以及标准、非标准 LoRA-MoE 实现
- 注意力与生成
  - [Flash Attention](op/flashattn)：Flash Nano CUDA 实现以及 MHA、GQA、MQA、MLA、Linear Attention 的 PyTorch 参考实现
  - [QK Attention](op/qk_attn)：面向脉冲数据的 QK Attention 实现
  - [Decoder Sampling](op/decoder_sampling)：支持 temperature、top-k、top-p、min-p 和历史惩罚的单步 token 采样
- 神经元
  - [IF、LIF 与 PLIF](op/neuron)：PyTorch C++/CUDA 扩展和 CuPy 实现，支持 FP32/FP16 前向与反向
- 通用算子
  - [Softmax](op/softmax)：FP32 与 INT8 路径
  - [Top-K](op/topk)：CUDA 实现及 CPU 参考实现
  - [Resize](op/resize)：UINT8 图像缩放
  - [逐元素乘法](op/mul)：UINT8 逐元素乘法

底层原语与系统实验包括 [MMA](op/mma)、[WGMMA](op/wgmma)、[异步拷贝](op/cp_async)、[NCCL 多卡 MLP](op/nccl)，以及 [`op/test/`](op/test) 中的 GPU 存储层次和 CUDA 架构原语验证。

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
