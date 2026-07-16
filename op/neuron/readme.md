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

## 自研 CuPy 与 SpikingJelly 对比

该对比参考 `SNN-Neuron-CUDA/main.py` 的训练路径，统一使用多步模式、
ATan 替代梯度、`detach_reset=True` 和 spike-only 输出。SpikingJelly 使用
CuPy 后端，每轮前重置状态；状态重置发生在 GPU 计时区间外。输入形状、
预热配置与上文相同；每种实现测量 5 组，每组采样 30 次，最终取组中位数
的中位数，并记录组间噪声。

`自研加速比` 的计算方式为：

```text
SpikingJelly 前向与反向总延迟 / 自研 CuPy 前向与反向总延迟
```

| 神经元 | 精度 | 自研总延迟/ms | SpikingJelly 总延迟/ms | 自研 GElem/s | 自研加速比 | 最大组间噪声 |
|---|---|---:|---:|---:|---:|---:|
| IF | FP32 | 1.4561 | 2.8886 | 46.089 | 1.984× | 0.57% |
| IF | FP16 | 0.8744 | 1.9343 | 76.747 | 2.212× | 2.34% |
| LIF | FP32 | 1.4565 | 2.9255 | 46.075 | 2.009× | 1.87% |
| LIF | FP16 | 0.8837 | 2.0026 | 75.940 | 2.266× | 2.97% |
| PLIF | FP32 | 1.8313 | 3.2040 | 36.645 | 1.750× | 3.07% |
| PLIF | FP16 | 1.1161 | 2.9864 | 60.129 | 2.676× | 24.37% |

IF、LIF 和 PLIF FP32 的前向与反向总吞吐相对 SpikingJelly CuPy 提升
`1.750×–2.266×`。PLIF FP16 的中位数加速比为 `2.676×`，但
SpikingJelly 组间噪声达到 `24.37%`；扩大到 9 组、每组 50 次后，加速比为
`2.941×`，组间噪声仍为 `16.82%`，因此该项只能判断自研实现明显更快，
不应把单个加速比作为稳定值。六组 spike、输入梯度和 PLIF 参数梯度均通过
精度验证。

本次测试使用 `py311` 环境：PyTorch 2.9.1+cu130、CuPy 13.6.0、
SpikingJelly 0.0.0.0.15。当前 CuPy 实际加载 CUDA 12.9 runtime，与
PyTorch CUDA 13.0 的主版本不同；两组实现共用该 CuPy runtime，但复测前
仍建议清理重复安装的 `cupy-cuda12x` 和 `cupy-cuda13x`。

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

自研 CuPy 与 SpikingJelly 的结果默认保存到
`build/neuron_spikingjelly_benchmark.csv`，在 `py311` 环境中运行：

```bash
conda run -n py311 python op/neuron/benchmark_spikingjelly.py
```

若 CMake 配置时使用的是 `py311` 的 Python，也可以运行：

```bash
cmake --build build --target neuron_cupy_spikingjelly_benchmark
```
