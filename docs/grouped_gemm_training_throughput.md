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
