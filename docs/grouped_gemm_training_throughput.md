# Grouped GEMM 前向与反向吞吐测试

## 结论

在 RTX 4090（SM89）上，Triton 分离实现和 Triton 融合实现的完整
前向与反向吞吐均高于 CUTLASS Grouped GEMM 基线。

- `hidden_size=2048`：Triton 分离实现加速 `1.186x`，融合实现加速
  `1.442x`。
- `hidden_size=8192`：Triton 分离实现加速 `1.041x`，融合实现加速
  `1.016x`。
- `hidden_size=2048` 时融合可明显受益于 kernel 数量从 6 个减少到
  4 个；`hidden_size=8192` 时计算量占比提高，融合实现略慢于分离
  实现，但仍快于 CUTLASS 基线。

cuTile 编译器不支持当前 GPU 的 `sm_89` 目标，因此按测试要求不纳入
本报告。

在 RTX 5070 Ti Laptop GPU（SM120）上补测后，cuTile 分离实现和
cuTile 融合实现均高于本机 CUTLASS Grouped GEMM 基线：

- `hidden_size=2048`：cuTile 分离实现加速 `1.654x`，融合实现加速
  `1.584x`。
- `hidden_size=8192`：cuTile 分离实现加速 `1.281x`，融合实现加速
  `1.189x`。
- cuTile 分离实现是本机两个配置中最快的实现，三次独立运行波动分别为
  `1.246%` 和 `1.479%`。
- cuTile 融合实现的波动分别为 `1.085%` 和 `3.188%`；后一个配置超过
  `2%` 稳定性阈值，因此其精确 latency 需结合三次原始结果解读。

两台机器的 GPU、软件栈和动态时钟行为不同，平台间只比较各自相对
CUTLASS 基线的结果，不直接比较绝对 latency。

## 测试环境

| 项目 | 配置 |
|---|---|
| 测试日期 | 2026-07-14 |
| GPU | NVIDIA GeForce RTX 4090，SM89，GPU 0 |
| NVIDIA Driver | 590.44.01 |
| Python | 3.11.11，Conda 环境 `py311` |
| PyTorch | 2.9.1+cu130 |
| PyTorch CUDA | 13.0 |
| CUDA Toolkit | 13.1 |
| Triton | 3.5.1 |
| 数据类型 | BF16 |

## 测试配置

固定参数如下：

| 参数 | 值 |
|---|---|
| experts | 8 |
| rank | 16 |
| hidden size | 2048、8192 |
| expert token 数 | `[3840, 4710, 2910, 3000, 3330, 3870, 4140, 3030]` |
| 总 token 数 | 28830 |
| 预热次数 | 20 |
| 单组迭代次数 | 100 |
| 采样组数 | 5 |
| 独立运行次数 | 3 |
| 统计量 | 每次取 5 组中位数，最终取 3 次运行的中位数 |

测试范围为完整 down/up 前向与反向，不包含优化器更新：

| 实现 | 前向 kernel | 反向 kernel | 合计 |
|---|---:|---:|---:|
| CUTLASS Grouped GEMM | 2 | 4 | 6 |
| Triton separate | 2 | 4 | 6 |
| Triton fused | 1 | 3 | 4 |

Triton separate 的反向分别计算 `grad_hidden`、`grad_input`、
`grad_down_weight` 和 `grad_up_weight`。Triton fused 将前两项输入梯度
融合为一个 kernel，并分别计算两项权重梯度。

权重预打包、路由 metadata 首次构建和 JIT 编译在预热阶段完成，不计入
稳态吞吐。前向、反向和前向+反向分别独立计时。单独反向计时预先构建
计算图，并使用 `retain_graph=True` 重复执行反向；每轮将叶子梯度设为
`None`，避免梯度累加。前向+反向时间通过每轮重新执行完整 Autograd
路径直接测得，并非前向与反向两列的简单相加。

吞吐和加速比均只使用完整前向+反向时间计算：

```text
throughput(Mtoken/s) = total_tokens / forward_backward_latency(us)
speedup = CUTLASS forward_backward_latency
          / implementation forward_backward_latency
```

## 测试结果

### hidden_size=2048

| 实现 | 前向（us） | 反向（us） | 前向+反向（us） | 完整路径范围（us） | 吞吐（Mtoken/s） | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| CUTLASS Grouped GEMM | 310.804 | 613.303 | 1640.335 | 1637.919–1762.224 | 17.576 | 1.000x |
| Triton separate | 280.083 | 516.997 | 1383.585 | 1327.289–1394.340 | 20.837 | 1.186x |
| Triton fused | 274.779 | 530.207 | 1137.674 | 1108.151–1141.064 | 25.341 | 1.442x |

该配置下 CUTLASS 的三次运行波动较大，因此同时给出完整运行范围；两种
Triton 实现的三次运行结果相对稳定。

### hidden_size=8192

| 实现 | 前向（us） | 反向（us） | 前向+反向（us） | 完整路径范围（us） | 吞吐（Mtoken/s） | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| CUTLASS Grouped GEMM | 1154.202 | 2229.012 | 3388.508 | 3384.392–3389.010 | 8.508 | 1.000x |
| Triton separate | 1109.535 | 2150.287 | 3253.952 | 3253.750–3254.066 | 8.860 | 1.041x |
| Triton fused | 1097.841 | 2242.743 | 3334.062 | 3333.458–3334.192 | 8.647 | 1.016x |

## 精度验证

性能测试前使用 `hidden_size=256` 验证前向和反向。Triton separate、
Triton fused 均通过 `rtol=2e-2`、`atol=5e-1` 的 BF16 反向精度检查。

| Tensor | Triton separate/CUTLASS 最大绝对差 | Triton fused/CUTLASS 最大绝对差 |
|---|---:|---:|
| output | 0.0 | 0.0 |
| grad input | 0.0 | 0.0 |
| grad down weight | 2.0 | 2.0 |
| grad up weight | 8.0 | 8.0 |

权重梯度的绝对值随 28830 个 token 累加而增大，因此正确性判定同时使用
相对误差和绝对误差，以上结果均通过断言。

## 复现方式

```bash
conda activate py311
cmake -S . -B build \
  -DPython3_EXECUTABLE=/home/lsbing/.conda/envs/py311/bin/python
CUDA_VISIBLE_DEVICES=0 \
  cmake --build build --target cudaop_grouped_gemm_test
```

报告数据来自一次 CMake 目标运行和两次脚本独立运行。需要复核运行间波动
时，可在相同空闲 GPU 上重复执行测试目标三次。

最终 CMake 测试目标输出：

```text
[SUCCESS] cudaop_grouped_gemm 对比测试通过
[100%] Built target cudaop_grouped_gemm_test
```

## 本平台补测：RTX 5070 Ti Laptop GPU（SM120）

### 测试环境

| 项目 | 配置 |
|---|---|
| 测试日期 | 2026-07-14 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU，SM120，GPU 0 |
| SM 数量 | 46 |
| NVIDIA Driver | 596.21 |
| Python | 3.11.15，Conda 环境 `py311` |
| PyTorch | 2.12.0+cu132 |
| PyTorch CUDA | 13.2 |
| CUDA Toolkit | 13.2（nvcc 13.2.78） |
| Triton | 3.7.0 |
| cuda-tile | 1.4.0 |
| 数据类型 | BF16 |

设备查询得到显存带宽估算值约为 `672.05 GB/s`。该 GPU 为 Laptop
型号，空闲时处于 P8，测试前后 SM 时钟会降至约 217–262 MHz，运行时
由动态频率提升；未锁定 GPU 时钟。测试前后 GPU 计算利用率均为 0%，
但显存占用约 1.25 GiB。

测试参数、计时范围、随机种子和精度容差与平台一相同。每个脚本内每项
实现预热 20 次，每个采样执行 100 次，共取 5 个采样的中位数；完整
脚本独立运行 3 次。下表 latency 使用 3 次完整运行结果的中位数。

运行间波动统一按以下公式计算：

```text
group_median_spread =
    (max(group_median) - min(group_median))
    / median(group_median)
```

### hidden_size=2048

| 实现 | 三次前向+反向（us） | 中位数（us） | 吞吐（Mtoken/s） | 加速比 | 波动 |
|---|---:|---:|---:|---:|---:|
| CUTLASS Grouped GEMM | 2527.382 / 2536.219 / 2899.726 | 2536.219 | 11.367 | 1.000x | 14.681% |
| Triton separate | 1549.330 / 1624.277 / 1959.266 | 1624.277 | 17.750 | 1.561x | 25.238% |
| Triton fused | 1754.487 / 1770.865 / 1628.459 | 1754.487 | 16.432 | 1.446x | 8.117% |
| cuTile separate | 1525.732 / 1544.833 / 1533.127 | 1533.127 | 18.805 | 1.654x | 1.246% |
| cuTile fused | 1596.159 / 1613.524 / 1601.139 | 1601.139 | 18.006 | 1.584x | 1.085% |

该配置的 CUTLASS 与 Triton 运行间波动明显，cuTile 两条路径自身的波动
低于 2%。因此 cuTile 的绝对 latency 较稳定，但相对 CUTLASS 的精确
加速比仍受基线波动影响。

### hidden_size=8192

| 实现 | 三次前向+反向（us） | 中位数（us） | 吞吐（Mtoken/s） | 加速比 | 波动 |
|---|---:|---:|---:|---:|---:|
| CUTLASS Grouped GEMM | 7693.706 / 7571.088 / 7580.677 | 7580.677 | 3.803 | 1.000x | 1.618% |
| Triton separate | 7423.511 / 6228.884 / 6511.338 | 6511.338 | 4.428 | 1.164x | 18.347% |
| Triton fused | 7516.813 / 6887.728 / 6748.214 | 6887.728 | 4.186 | 1.101x | 11.159% |
| cuTile separate | 5918.940 / 6003.600 / 5916.077 | 5918.940 | 4.871 | 1.281x | 1.479% |
| cuTile fused | 6378.010 / 6403.998 / 6200.660 | 6378.010 | 4.520 | 1.189x | 3.188% |

该配置的 CUTLASS 基线和 cuTile 分离实现波动低于 2%，`1.281x` 加速结果
较稳定。cuTile 融合实现波动为 3.188%，但三次结果均明显快于 CUTLASS；
其性能方向可信，精确加速比建议在固定功耗和时钟条件下复测。

### 精度验证

性能测试前使用 `hidden_size=256` 验证完整前向和反向。cuTile separate
和 cuTile fused 均通过 `rtol=2e-2`、`atol=5e-1` 的 BF16 反向精度
检查。

| Tensor | cuTile separate/CUTLASS 最大绝对差 | cuTile fused/CUTLASS 最大绝对差 |
|---|---:|---:|
| output | 0.0 | 0.0 |
| grad input | 0.0 | 0.0 |
| grad down weight | 16.0 | 16.0 |
| grad up weight | 0.5 | 0.5 |

权重梯度包含 28830 个 token 的累加，最大绝对差需与相对误差共同判断；
上述四项均通过断言。cuTile 分阶段前向、融合前向与 Triton 输出逐元素
一致，本次测试的最大绝对差为 0。

### 复现方式

```bash
conda activate py311
cmake -S . -B build \
  -DCUDAOP_CUDA_ARCHITECTURES=120 \
  -DPython3_EXECUTABLE=/home/if/miniforge3/envs/py311/bin/python
cmake --build build --target cudaop_grouped_gemm_test -j
```

本平台 CMake 测试目标和另外两次脚本独立运行均输出：

```text
[SUCCESS] cudaop_grouped_gemm 对比测试通过
```

## Kernel fusion 专项复测（RTX 4090）

2026-09-08 在 RTX 4090（SM89）上对 `fused_lora.cuh` 进行专项复测，
只比较 CUTLASS 默认非融合路径与 CUTLASS/CuTe 融合路径：

测试参数为 BF16、8 experts、rank 16、hidden size 2048/8192，expert token 数为 `[2560, 3140, 1940, 2000, 2220, 2580, 2760, 2020]`，
总 token 数为 19220。每个采样预热 100 次、计时 1000 次，融合与非融合交替执行，共取 10 个采样的中位数。

纯反向在计时前构建好 Autograd graph，并通过 `retain_graph=True` 重复执行，因此不包含前向。端到端测试每轮重新执行前向和反向。
融合权重预打包和首次 metadata 构建均在计时外；CUDA Event 结果仍包含每个入口自身的 metadata、Autograd 和 kernel launch 开销。
融合/非融合精度、空 expert、非整块 M/N/K 尾块和 bgrad K=256/33 回归全部通过。

### CUTLASS 非融合 vs fusion

| 计时口径 | hidden | 阶段 | 非融合（us） | 融合（us） | 加速比 |
|---|---:|---|---:|---:|---:|
| CUDA Event | 2048 | 前向 | 236.310 | 190.334 | 1.242x |
| CUDA Event | 2048 | 纯反向 | 456.619 | 347.252 | 1.315x |
| CUDA Event | 2048 | 前向+反向 | 1737.381 | 844.424 | 2.057x |
| CUDA Event | 8192 | 前向 | 907.509 | 764.643 | 1.187x |
| CUDA Event | 8192 | 纯反向 | 1647.916 | 1461.882 | 1.127x |
| CUDA Event | 8192 | 前向+反向 | 2554.622 | 2242.184 | 1.139x |
| Nsight 纯计算 | 2048 | 前向 | 229.382 | 191.906 | 1.195x |
| Nsight 纯计算 | 2048 | 纯反向 | 440.305 | 336.124 | 1.310x |
| Nsight 纯计算 | 2048 | 前向+反向 | 604.149 | 558.670 | 1.081x |
| Nsight 纯计算 | 8192 | 前向 | 896.246 | 765.123 | 1.171x |
| Nsight 纯计算 | 8192 | 纯反向 | 1608.875 | 1437.234 | 1.119x |
| Nsight 纯计算 | 8192 | 前向+反向 | 2505.444 | 2223.066 | 1.127x |

### CUTLASS 非融合 vs Triton fusion

`重建` 表示每次调用执行 `clear_metadata_cache()`；`复用` 表示输入信息不变，首次 metadata 构建已在 warmup 完成。

| 计时口径 | metadata | hidden | 阶段 | CUTLASS 非融合（us） | Triton fusion（us） | 加速比 |
|---|---|---:|---|---:|---:|---:|
| CUDA Event | 重建 | 2048 | 前向 | 235.869 | 261.829 | 0.901x |
| CUDA Event | 重建 | 2048 | 纯反向 | 457.636 | 325.740 | 1.405x |
| CUDA Event | 重建 | 2048 | 前向+反向 | 1666.220 | 1536.047 | 1.085x |
| CUDA Event | 重建 | 8192 | 前向 | 909.293 | 803.863 | 1.131x |
| CUDA Event | 重建 | 8192 | 纯反向 | 1643.180 | 1494.391 | 1.100x |
| CUDA Event | 重建 | 8192 | 前向+反向 | 2548.558 | 2319.074 | 1.099x |
| CUDA Event | 复用 | 2048 | 前向 | 237.824 | 182.376 | 1.304x |
| CUDA Event | 复用 | 2048 | 纯反向 | 456.912 | 321.967 | 1.419x |
| CUDA Event | 复用 | 2048 | 前向+反向 | 1732.645 | 1129.045 | 1.535x |
| CUDA Event | 复用 | 8192 | 前向 | 909.863 | 726.543 | 1.252x |
| CUDA Event | 复用 | 8192 | 纯反向 | 1647.500 | 1492.660 | 1.104x |
| CUDA Event | 复用 | 8192 | 前向+反向 | 2557.574 | 2218.912 | 1.153x |
| Nsight 纯计算 | 重建 | 2048 | 前向 | 228.689 | 178.981 | 1.278x |
| Nsight 纯计算 | 重建 | 2048 | 纯反向 | 440.415 | 319.490 | 1.379x |
| Nsight 纯计算 | 重建 | 2048 | 前向+反向 | 604.594 | 561.706 | 1.076x |
| Nsight 纯计算 | 重建 | 8192 | 前向 | 895.111 | 722.408 | 1.239x |
| Nsight 纯计算 | 重建 | 8192 | 纯反向 | 1607.582 | 1490.073 | 1.079x |
| Nsight 纯计算 | 重建 | 8192 | 前向+反向 | 2503.795 | 2219.389 | 1.128x |
| Nsight 纯计算 | 复用 | 2048 | 前向 | 228.986 | 183.094 | 1.251x |
| Nsight 纯计算 | 复用 | 2048 | 纯反向 | 439.378 | 323.177 | 1.360x |
| Nsight 纯计算 | 复用 | 2048 | 前向+反向 | 614.696 | 563.515 | 1.091x |
| Nsight 纯计算 | 复用 | 8192 | 前向 | 906.738 | 729.592 | 1.243x |
| Nsight 纯计算 | 复用 | 8192 | 纯反向 | 1605.521 | 1494.133 | 1.075x |
| Nsight 纯计算 | 复用 | 8192 | 前向+反向 | 2509.204 | 2222.077 | 1.129x |

- “前向”使用的 Tensor 没有开启梯度，不构建完整 Autograd graph，见 op/grouped_gemm/test_grouped_gemm.py:1666。
- “纯反向”的前向和计算图在计时前已经创建；计时期间反复对同一个 graph 执行 retain_graph=True，见 op/grouped_gemm/test_grouped_gemm.py:1682。
- “前向+反向”每轮都会重新构建 Autograd graph、保存中间 Tensor、执行反向并销毁 graph，见 op/grouped_gemm/test_grouped_gemm.py:1743。
- hidden=2048 时 kernel 较短，Python/Autograd 调度和 kernel launch 间隙会造成 GPU 等待。CUDA Event 的起止区间会包含这些 GPU 空闲间隙，因此差异特别明显。
- hidden=8192 时 GPU 计算时间占主导，CPU 调度开销大部分被计算覆盖，所以完整路径基本接近两者之和。

### 复现方式

```bash
conda activate py311
cmake -S . -B build \
  -DPython3_EXECUTABLE=/home/lsbing/.conda/envs/py311/bin/python
CUDA_VISIBLE_DEVICES=1 \
  cmake --build build --target cudaop_grouped_gemm_test -j 4
```

复测 hidden size 8192 时使用：

```bash
CUDA_VISIBLE_DEVICES=1 CUDAOP_GROUPED_GEMM_HIDDEN_SIZE=8192 \
  cmake --build build --target cudaop_grouped_gemm_test -j 4
```

动态 metadata 对比时使用（默认 `clear=1`）：

```bash
CUDA_VISIBLE_DEVICES=1 CUDAOP_GROUPED_GEMM_HIDDEN_SIZE=8192 \
  CUDAOP_GROUPED_GEMM_FUSION_COMPARISON=triton \
  cmake --build build --target cudaop_grouped_gemm_test -j 4
```

固定复用 metadata 时增加 `clear=0`：

```bash
CUDA_VISIBLE_DEVICES=1 CUDAOP_GROUPED_GEMM_HIDDEN_SIZE=8192 \
  CUDAOP_GROUPED_GEMM_FUSION_COMPARISON=triton \
  CUDAOP_GROUPED_GEMM_CLEAR_METADATA_CACHE=0 \
  cmake --build build --target cudaop_grouped_gemm_test -j 4
```

最终目标输出：

```text
[SUCCESS] cudaop_grouped_gemm 对比测试通过
[100%] Built target cudaop_grouped_gemm_test
```
