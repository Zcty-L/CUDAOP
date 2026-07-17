"""Validate the low-level C++/CUDA interfaces exported by ``neuron_ops``."""

import logging

import torch

import neuron_ops


def reference_if(
    inputs: torch.Tensor,
    v_threshold: float,
    v_reset: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spike_seq = torch.empty_like(inputs)
    h_seq = torch.empty_like(inputs)
    v_seq = torch.empty_like(inputs)
    last_v = torch.zeros_like(inputs[0])

    for time_step in range(inputs.size(0)):
        h = last_v + inputs[time_step]
        spike = (h >= v_threshold).to(inputs.dtype)
        last_v = h - spike * h + spike * v_reset
        spike_seq[time_step] = spike
        h_seq[time_step] = h
        v_seq[time_step] = last_v

    return spike_seq, h_seq, v_seq


def reference_lif(
    inputs: torch.Tensor,
    v_threshold: float,
    v_reset: float,
    decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spike_seq = torch.empty_like(inputs)
    h_seq = torch.empty_like(inputs)
    v_seq = torch.empty_like(inputs)
    last_v = torch.zeros_like(inputs[0])

    for time_step in range(inputs.size(0)):
        h = last_v + (inputs[time_step] - (last_v - v_reset)) * decay
        spike = (h >= v_threshold).to(inputs.dtype)
        last_v = h - spike * h + spike * v_reset
        spike_seq[time_step] = spike
        h_seq[time_step] = h
        v_seq[time_step] = last_v

    return spike_seq, h_seq, v_seq


def assert_outputs_close(
    actual: tuple[torch.Tensor, ...] | list[torch.Tensor],
    expected: tuple[torch.Tensor, ...],
    dtype: torch.dtype,
) -> None:
    tolerance = 1e-3 if dtype == torch.float16 else 1e-6
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=tolerance,
            rtol=tolerance,
        )


def validate_dtype(dtype: torch.dtype) -> None:
    values = torch.linspace(
        -0.5,
        1.5,
        steps=64,
        device="cuda",
        dtype=dtype,
    ).reshape(4, 2, 8)

    if_args = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32)
    if_outputs = neuron_ops.IFNodeForward(values, if_args, True, 256)
    assert_outputs_close(if_outputs, reference_if(values, 1.0, 0.0), dtype)

    grad_spike = torch.ones_like(values)
    grad_v = torch.ones_like(values)
    if_grad = neuron_ops.IFNodeBackward(
        grad_spike,
        grad_v,
        if_outputs[1],
        if_args,
        True,
        True,
        0,
        256,
    )
    if len(if_grad) != 1 or not torch.isfinite(if_grad[0]).all():
        raise AssertionError("IFNodeBackward returned invalid gradients")

    lif_args = torch.tensor([1.0, 0.0, 0.5, 2.0], dtype=torch.float32)
    lif_outputs = neuron_ops.LIFNodeForward(
        values,
        lif_args,
        True,
        True,
        256,
    )
    assert_outputs_close(
        lif_outputs,
        reference_lif(values, 1.0, 0.0, 0.5),
        dtype,
    )

    lif_grad = neuron_ops.LIFNodeBackward(
        grad_spike,
        grad_v,
        lif_outputs[1],
        lif_args,
        True,
        True,
        True,
        1,
        256,
    )
    if len(lif_grad) != 1 or not torch.isfinite(lif_grad[0]).all():
        raise AssertionError("LIFNodeBackward returned invalid gradients")

    plif_outputs = neuron_ops.PLIFNodeForward(
        values,
        lif_args,
        True,
        True,
        256,
    )
    plif_grad = neuron_ops.PLIFNodeBackward(
        grad_spike,
        grad_v,
        plif_outputs[1],
        plif_outputs[2],
        lif_args,
        True,
        True,
        True,
        1,
        256,
    )
    if len(plif_grad) != 2:
        raise AssertionError("PLIFNodeBackward must return input and decay gradients")
    if not all(torch.isfinite(tensor).all() for tensor in plif_grad):
        raise AssertionError("PLIFNodeBackward returned invalid gradients")

    torch.cuda.synchronize()
    logging.info("dtype=%s: forward/backward 验证通过", dtype)


def validate_alignment_guard() -> None:
    inputs = torch.zeros((2, 5), device="cuda", dtype=torch.float32)
    args = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32)

    try:
        neuron_ops.IFNodeForward(inputs, args, True, 256)
    except RuntimeError as error:
        if "multiple of 4" not in str(error):
            raise
    else:
        raise AssertionError("非对齐输入未被 C++ 参数检查拦截")

    logging.info("非对齐输入保护验证通过")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("neuron_ops 测试需要 CUDA GPU")

    logging.info(
        "测试配置: device=%s, capability=%s",
        torch.cuda.get_device_name(),
        torch.cuda.get_device_capability(),
    )
    validate_alignment_guard()
    validate_dtype(torch.float32)
    validate_dtype(torch.float16)
    logging.info("[SUCCESS] neuron_ops C++/CUDA 接口验证通过")


if __name__ == "__main__":
    main()
