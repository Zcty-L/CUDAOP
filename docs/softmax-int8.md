# INT8 量化 Softmax 设计说明

## 工程结论

Softmax 的输入是 logits。常见 GPU 推理链路中，Softmax 的直接输入通常是
FP16、BF16 或 FP32；即使上游 GEMM 使用 INT8，点积也先累加到 INT32，
再经过 scale 转换后进入浮点 Softmax。TensorRT 的 SoftMax 算子当前公开的
输入和输出类型是 FP16、FP32 和 BF16，INT8 张量通过显式 Q/DQ 表达量化边界。

INT8 也可以作为 Softmax 的直接存储输入，但它不是没有量化参数的普通
整数。
本实现采用按张量仿射量化：

```text
real_logit = (quantized_logit - input_zero_point) * input_scale
```

由于 Softmax 会先减去行最大值，同一行使用统一 zero point 时：

```text
real_logit - real_max
    = (quantized_logit - quantized_max) * input_scale
```

因此 kernel 不需要实际执行 zero point 减法，但 API 仍保留并校验
`input_zero_point`，以完整表达输入张量的量化语义。当前实现不支持沿 Softmax
归约轴变化的 per-axis scale 或 zero point。

## 常见实现路线

1. INT8 输入、浮点输出：融合反量化、最大值归约、FP32 `exp` 和求和。
   这适合 GPU 和较长的 attention 维度，也是本项目的推荐路径。
2. INT8 输入、INT8 输出：移动端和 TinyML 中常见。TensorFlow Lite 规定
   Softmax 输出为按张量 INT8，`scale=1/256`、`zero_point=-128`。
3. 全整数近似：CMSIS-NN 提供 S8、S8-to-S16 和 U8 Softmax，使用量化
   multiplier、shift、`diff_min` 或查找表。这类实现适合没有高效浮点单元的
   MCU，不等同于本项目当前使用 FP32 指数计算的 INT8 I/O kernel。

INT8 概率输出只有 `1/256` 的分辨率。类别或序列长度较大时，大量小概率会
量化为零，因此 attention 等长归约场景应优先使用 INT8-to-FP32，而不是
INT8-to-INT8。

## 本项目接口

- `launch_softmax_int8_to_float`：INT8 logits，FP32 概率输出。
- `launch_softmax_int8_to_int8`：INT8 logits，兼容 TensorFlow Lite
  `scale=1/256`、`zero_point=-128` 的 INT8 概率输出。
- 两条路径均支持 `1 <= cols <= 1024`，输入采用按张量 scale 和 zero point，
  内部最大值、指数与求和使用 FP32。

## 参考

- TensorFlow Lite INT8 量化规范：
  https://www.tensorflow.org/lite/performance/quantization_spec
- CMSIS-NN Softmax：
  https://arm-software.github.io/CMSIS-NN/latest/group__Softmax.html
- TensorRT SoftMax 数据类型：
  https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/SoftMax.html
- TensorRT 量化类型与 Q/DQ：
  https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/capabilities.html
- ONNX Runtime XNNPACK 支持的 QLinearSoftmax：
  https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html
