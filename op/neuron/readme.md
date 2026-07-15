# Neuron CUDA 算子

本目录包含 IF、LIF 和 PLIF 神经元的三套实现：

- `cpp_cuda_full`：PyTorch C++/CUDA 扩展，输出 spike 和 voltage。
- `cupy_full`：CuPy 完整版，输出 spike 和 voltage。
- `cupy_lite`：CuPy 精简版，只输出 spike，减少 voltage 写回。

## 前向与反向性能对比

### 测试配置

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，SM 8.9 |
| 输入形状 | `[T, N, C, H, W] = [4, 32, 64, 64, 64]` |
| 总元素数 | 33,554,432 |
| 输入大小 | FP32 128 MiB，FP16 64 MiB |
| 数据类型 | FP32、FP16 |
| 神经元 | IF、LIF、PLIF |
| 计时范围 | PyTorch autograd 前向与反向端到端 GPU 时间 |
| 统计方式 | 预热 10 次，采样 30 次，取中位数 |

该配置固定 `T=4`，通过增大 batch、通道和空间尺寸，使每个时间步包含
8,388,608 个元素。full 前向需要读取 input 并写回 spike、hidden 和
voltage，FP32、FP16 工作集均超过 GPU 缓存容量，适合测试持续吞吐。

延迟单位为毫秒。`GElem/s` 按前向和反向各处理一次全部元素计算：

```text
2 × 输入元素数 / 前向与反向总延迟
```

`C++ 加速比` 的计算方式为：

```text
CuPy 前向与反向总延迟 / C++ 前向与反向总延迟
```

数值大于 1 表示 C++/CUDA 更快。

| 神经元 | 精度 | 实现 | 输出 | 前向 | 反向 | 总延迟 | GElem/s | C++ 加速比 |
|---|---|---|---|---:|---:|---:|---:|---:|
| IF | FP32 | C++/CUDA | spike+voltage | 0.6091 | 0.6914 | 1.3005 | 51.601 | 1.000× |
| IF | FP32 | CuPy full | spike+voltage | 0.6492 | 0.7510 | 1.4002 | 47.929 | 1.077× |
| IF | FP32 | CuPy lite | spike | 0.5052 | 0.5744 | 1.0796 | 62.159 | 0.830× |
| IF | FP16 | C++/CUDA | spike+voltage | 0.3154 | 0.3973 | 0.7127 | 94.161 | 1.000× |
| IF | FP16 | CuPy full | spike+voltage | 0.3793 | 0.4663 | 0.8456 | 79.359 | 1.187× |
| IF | FP16 | CuPy lite | spike | 0.2810 | 0.3492 | 0.6302 | 106.490 | 0.884× |
| LIF | FP32 | C++/CUDA | spike+voltage | 0.6093 | 0.6912 | 1.3005 | 51.603 | 1.000× |
| LIF | FP32 | CuPy full | spike+voltage | 0.6492 | 0.7475 | 1.3967 | 48.047 | 1.074× |
| LIF | FP32 | CuPy lite | spike | 0.5028 | 0.5775 | 1.0803 | 62.119 | 0.831× |
| LIF | FP16 | C++/CUDA | spike+voltage | 0.3154 | 0.3984 | 0.7138 | 94.020 | 1.000× |
| LIF | FP16 | CuPy full | spike+voltage | 0.3758 | 0.4694 | 0.8452 | 79.395 | 1.184× |
| LIF | FP16 | CuPy lite | spike | 0.2786 | 0.3512 | 0.6298 | 106.549 | 0.882× |
| PLIF | FP32 | C++/CUDA | spike+voltage | 0.6921 | 0.8642 | 1.5563 | 43.122 | 1.000× |
| PLIF | FP32 | CuPy full | spike+voltage | 0.7209 | 0.9605 | 1.6814 | 39.912 | 1.080× |
| PLIF | FP32 | CuPy lite | spike | 0.7292 | 0.7648 | 1.4940 | 44.918 | 0.960× |
| PLIF | FP16 | C++/CUDA | spike+voltage | 0.3645 | 0.5110 | 0.8755 | 76.650 | 1.000× |
| PLIF | FP16 | CuPy full | spike+voltage | 0.4469 | 0.6401 | 1.0870 | 61.737 | 1.242× |
| PLIF | FP16 | CuPy lite | spike | 0.4589 | 0.5253 | 0.9842 | 68.188 | 1.124× |

### 结果总结

- 在相同的 spike+voltage 输出语义下，C++/CUDA 相对 CuPy full 加速
  `1.074×–1.242×`。
- C++/CUDA 的 IF/LIF 吞吐为 FP32 约 `51.6 GElem/s`、FP16 约
  `94.0 GElem/s`；PLIF 分别为 `43.1 GElem/s` 和 `76.7 GElem/s`。
- CuPy lite 不写回 voltage，工作量与 full 实现不同。它在 IF/LIF 和
  PLIF FP32 上快于 C++ full，在 PLIF FP16 上慢于 C++ full，不能用于
  判断同工作量实现的优劣。
- FP32、FP16 下，full 实现的 spike、voltage，以及 lite 实现的 spike 和
  输入梯度均已通过精度验证。

## 构建与复现

CuPy 的 CUDA 主版本必须与 PyTorch 一致。例如 PyTorch 使用 CUDA 13 时，
应安装 `cupy-cuda13x`。

```bash
cmake -S . -B build
cmake --build build --target neuron_ops_test --parallel 4
cmake --build build --target neuron_ops_benchmark --parallel 4
```

基准默认将原始结果保存到 `build/neuron_benchmark.csv`。也可以直接运行
`op/neuron/benchmark_ops.py`，通过命令行参数修改输入形状、预热次数、
采样次数和输出路径。
