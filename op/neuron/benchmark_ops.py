"""Benchmark CuPy neuron implementations against the C++/CUDA extension."""

import argparse
import csv
from dataclasses import dataclass
import logging
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

import cupy as cp
import torch

import neuron_cupy
import neuron_cupy_lite
import neuron_ops


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent.parent / "build" / "neuron_benchmark.csv"
NEURON_TYPES = ("IF", "LIF", "PLIF")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


@dataclass
class RunState:
    inputs: torch.Tensor
    outputs: tuple[torch.Tensor, ...]


@dataclass
class Timing:
    forward_ms: float
    backward_ms: float

    @property
    def total_ms(self) -> float:
        return self.forward_ms + self.backward_ms


class CppIFNode(torch.autograd.Function):
    """Autograd adapter for the C++ IF interfaces."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        args: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spike, hidden, voltage = neuron_ops.IFNodeForward(
            inputs,
            args,
            True,
            256,
        )
        ctx.save_for_backward(hidden)
        ctx.args = args
        return spike, voltage

    @staticmethod
    def backward(
        ctx: Any,
        grad_spike: torch.Tensor,
        grad_voltage: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        hidden = ctx.saved_tensors[0]
        gradients = neuron_ops.IFNodeBackward(
            grad_spike.contiguous(),
            grad_voltage.contiguous(),
            hidden,
            ctx.args,
            True,
            True,
            0,
            256,
        )
        return gradients[0], None


class CppLIFNode(torch.autograd.Function):
    """Autograd adapter for the C++ LIF interfaces."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        args: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spike, hidden, voltage = neuron_ops.LIFNodeForward(
            inputs,
            args,
            True,
            True,
            256,
        )
        ctx.save_for_backward(hidden)
        ctx.args = args
        return spike, voltage

    @staticmethod
    def backward(
        ctx: Any,
        grad_spike: torch.Tensor,
        grad_voltage: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        hidden = ctx.saved_tensors[0]
        gradients = neuron_ops.LIFNodeBackward(
            grad_spike.contiguous(),
            grad_voltage.contiguous(),
            hidden,
            ctx.args,
            True,
            True,
            True,
            0,
            256,
        )
        return gradients[0], None


class CppPLIFNode(torch.autograd.Function):
    """Autograd adapter for the C++ PLIF interfaces."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        decay: torch.Tensor,
        args: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spike, hidden, voltage = neuron_ops.PLIFNodeForward(
            inputs,
            args,
            True,
            True,
            256,
        )
        ctx.save_for_backward(hidden, voltage)
        ctx.args = args
        return spike, voltage

    @staticmethod
    def backward(
        ctx: Any,
        grad_spike: torch.Tensor,
        grad_voltage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        hidden, voltage = ctx.saved_tensors
        gradients = neuron_ops.PLIFNodeBackward(
            grad_spike.contiguous(),
            grad_voltage.contiguous(),
            hidden,
            voltage,
            ctx.args,
            True,
            True,
            True,
            0,
            256,
        )
        return gradients[0], gradients[1].sum(), None


class CppRunner:
    """Run C++/CUDA through an autograd path matching the CuPy interface."""

    implementation = "cpp_cuda_full"
    output_mode = "spike+voltage"

    def __init__(self, neuron_type: str) -> None:
        self.neuron_type = neuron_type
        values = [1.0, 0.0, 4.0]
        if neuron_type != "IF":
            values = [1.0, 0.0, 0.5, 4.0]
        self.args = torch.tensor(values, dtype=torch.float32)
        self.weight = None
        if neuron_type == "PLIF":
            self.weight = torch.zeros((), device="cuda", requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> RunState:
        leaf_inputs = inputs.detach().requires_grad_(True)
        if self.neuron_type == "IF":
            outputs = CppIFNode.apply(leaf_inputs, self.args)
        elif self.neuron_type == "LIF":
            outputs = CppLIFNode.apply(leaf_inputs, self.args)
        else:
            if self.weight is None:
                raise RuntimeError("PLIF runner 缺少可学习参数")
            self.weight.grad = None
            decay = self.weight.sigmoid()
            self.args[2] = decay.detach().cpu().item()
            outputs = CppPLIFNode.apply(leaf_inputs, decay, self.args)
        return RunState(leaf_inputs, tuple(outputs))

    def backward(
        self,
        state: RunState,
        grad_spike: torch.Tensor,
        grad_voltage: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        torch.autograd.backward(
            state.outputs,
            (grad_spike, grad_voltage),
        )
        if state.inputs.grad is None:
            raise RuntimeError("C++ backward 未生成输入梯度")
        return (state.inputs.grad,)


class CuPyRunner:
    """Run one of the imported CuPy autograd implementations."""

    def __init__(
        self,
        source: ModuleType,
        neuron_type: str,
        lite: bool,
    ) -> None:
        self.neuron_type = neuron_type
        self.lite = lite
        self.implementation = "cupy_lite" if lite else "cupy_full"
        self.output_mode = "spike" if lite else "spike+voltage"

        node_types = {
            "IF": source.IFNode,
            "LIF": source.LIFNode,
            "PLIF": source.ParametricLIFNode,
        }
        node_type = node_types[neuron_type]
        options: dict[str, Any] = {
            "detach_reset": True,
            "store_v_seq": not lite,
        }
        self.node = node_type(**options).cuda()

    def forward(self, inputs: torch.Tensor) -> RunState:
        self.node.zero_grad(set_to_none=True)
        leaf_inputs = inputs.detach().requires_grad_(True)
        output = self.node(leaf_inputs)
        outputs = output if isinstance(output, tuple) else (output,)
        return RunState(leaf_inputs, outputs)

    def backward(
        self,
        state: RunState,
        grad_spike: torch.Tensor,
        grad_voltage: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.lite:
            state.outputs[0].backward(grad_spike)
        else:
            torch.autograd.backward(
                state.outputs,
                (grad_spike, grad_voltage),
            )
        if state.inputs.grad is None:
            raise RuntimeError("CuPy backward 未生成输入梯度")
        return (state.inputs.grad,)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
    description: str,
) -> None:
    tolerance = 2e-2 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(
        actual,
        expected,
        atol=tolerance,
        rtol=tolerance,
        msg=lambda message: f"{description}: {message}",
    )


def validate_runners(
    cpp_runner: CppRunner,
    full_runner: CuPyRunner,
    lite_runner: CuPyRunner,
    inputs: torch.Tensor,
    grad_spike: torch.Tensor,
    grad_voltage: torch.Tensor,
) -> None:
    cpp_state = cpp_runner.forward(inputs)
    cpp_full_grad = cpp_runner.backward(
        cpp_state,
        grad_spike,
        grad_voltage,
    )[0]

    full_state = full_runner.forward(inputs)
    full_grad = full_runner.backward(
        full_state,
        grad_spike,
        grad_voltage,
    )[0]
    assert_close(
        full_state.outputs[0],
        cpp_state.outputs[0],
        inputs.dtype,
        "CuPy full spike",
    )
    assert_close(
        full_state.outputs[1],
        cpp_state.outputs[1],
        inputs.dtype,
        "CuPy full voltage",
    )
    assert_close(full_grad, cpp_full_grad, inputs.dtype, "CuPy full gradient")

    zero_voltage_grad = torch.zeros_like(grad_voltage)
    cpp_lite_state = cpp_runner.forward(inputs)
    cpp_lite_grad = cpp_runner.backward(
        cpp_lite_state,
        grad_spike,
        zero_voltage_grad,
    )[0]
    lite_state = lite_runner.forward(inputs)
    lite_grad = lite_runner.backward(
        lite_state,
        grad_spike,
        zero_voltage_grad,
    )[0]
    assert_close(
        lite_state.outputs[0],
        cpp_state.outputs[0],
        inputs.dtype,
        "CuPy lite spike",
    )
    assert_close(lite_grad, cpp_lite_grad, inputs.dtype, "CuPy lite gradient")
    torch.cuda.synchronize()


def measure_runner(
    runner: CppRunner | CuPyRunner,
    inputs: torch.Tensor,
    grad_spike: torch.Tensor,
    grad_voltage: torch.Tensor,
    warmup: int,
    iterations: int,
) -> Timing:
    for _ in range(warmup):
        state = runner.forward(inputs)
        runner.backward(state, grad_spike, grad_voltage)
    torch.cuda.synchronize()

    forward_samples = []
    backward_samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start.record()
        state = runner.forward(inputs)
        end.record()
        end.synchronize()
        forward_samples.append(start.elapsed_time(end))

        start.record()
        runner.backward(state, grad_spike, grad_voltage)
        end.record()
        end.synchronize()
        backward_samples.append(start.elapsed_time(end))

    return Timing(
        forward_ms=median(forward_samples),
        backward_ms=median(backward_samples),
    )


def create_runners(
    neuron_type: str,
) -> tuple[CppRunner, CuPyRunner, CuPyRunner]:
    return (
        CppRunner(neuron_type),
        CuPyRunner(neuron_cupy, neuron_type, lite=False),
        CuPyRunner(neuron_cupy_lite, neuron_type, lite=True),
    )


def write_results(rows: list[dict[str, str | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(args: argparse.Namespace) -> None:
    shape = tuple(args.shape)
    total_elements = 1
    for dimension in shape:
        total_elements *= dimension
    elements_per_step = 1
    for dimension in shape[1:]:
        elements_per_step *= dimension
    if elements_per_step % 8 != 0:
        raise ValueError("每个时间步的元素数必须是 8 的倍数")

    logging.info(
        "测试配置: device=%s, capability=%s, shape=%s",
        torch.cuda.get_device_name(),
        torch.cuda.get_device_capability(),
        shape,
    )
    logging.info(
        "工作量: elements=%d, FP32 input=%.1f MiB, FP16 input=%.1f MiB",
        total_elements,
        total_elements * 4 / (1024 * 1024),
        total_elements * 2 / (1024 * 1024),
    )
    logging.info(
        "计时配置: warmup=%d, iterations=%d",
        args.warmup,
        args.iterations,
    )
    logging.info(
        "输出语义: full/C++=spike+voltage, lite=spike；lite 对比仅作参考"
    )

    rows: list[dict[str, str | float]] = []
    for neuron_type in args.neurons:
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            inputs = torch.randn(shape, device="cuda", dtype=dtype) * 0.5
            grad_spike = torch.randn_like(inputs)
            grad_voltage = torch.randn_like(inputs)
            runners = create_runners(neuron_type)

            logging.info(
                "\n验证阶段: neuron=%s, dtype=%s",
                neuron_type,
                dtype_name,
            )
            validate_runners(
                runners[0],
                runners[1],
                runners[2],
                inputs,
                grad_spike,
                grad_voltage,
            )

            timings: dict[str, Timing] = {}
            for runner in runners:
                timings[runner.implementation] = measure_runner(
                    runner,
                    inputs,
                    grad_spike,
                    grad_voltage,
                    args.warmup,
                    args.iterations,
                )

            cpp_total = timings["cpp_cuda_full"].total_ms
            logging.info(
                "%-15s %11s %11s %11s %11s %14s",
                "implementation",
                "forward/ms",
                "backward/ms",
                "total/ms",
                "GElem/s",
                "C++ speedup",
            )
            for runner in runners:
                timing = timings[runner.implementation]
                speedup = timing.total_ms / cpp_total
                throughput = 2 * total_elements / (timing.total_ms * 1e6)
                logging.info(
                    "%-15s %11.4f %11.4f %11.4f %11.3f %13.3fx",
                    runner.implementation,
                    timing.forward_ms,
                    timing.backward_ms,
                    timing.total_ms,
                    throughput,
                    speedup,
                )
                rows.append(
                    {
                        "neuron": neuron_type,
                        "dtype": dtype_name,
                        "shape": "x".join(str(value) for value in shape),
                        "implementation": runner.implementation,
                        "output_mode": runner.output_mode,
                        "forward_ms": timing.forward_ms,
                        "backward_ms": timing.backward_ms,
                        "total_ms": timing.total_ms,
                        "throughput_gelem_s": throughput,
                        "cpp_speedup": speedup,
                    }
                )

    output_path = Path(args.output).resolve()
    write_results(rows, output_path)
    logging.info("\n原始结果: %s", output_path)
    logging.info("[SUCCESS] neuron 前向+反向性能对比完成")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        nargs=5,
        type=int,
        default=(4, 32, 64, 64, 64),
        metavar=("T", "N", "C", "H", "W"),
    )
    parser.add_argument(
        "--neurons",
        nargs="+",
        choices=NEURON_TYPES,
        default=NEURON_TYPES,
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=tuple(DTYPES),
        default=tuple(DTYPES),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def check_cuda_compatibility() -> None:
    torch_cuda = torch.version.cuda
    if torch_cuda is None:
        raise RuntimeError("当前 PyTorch 未启用 CUDA")

    torch_cuda_major = int(torch_cuda.split(".", maxsplit=1)[0])
    cupy_cuda_version = cp.cuda.runtime.runtimeGetVersion()
    cupy_cuda_major = cupy_cuda_version // 1000
    if cupy_cuda_major != torch_cuda_major:
        raise RuntimeError(
            "CuPy 与 PyTorch 的 CUDA 主版本不一致："
            f"CuPy={cupy_cuda_major}, PyTorch={torch_cuda}；"
            f"请安装 cupy-cuda{torch_cuda_major}x"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("neuron 性能测试需要 CUDA GPU")
    check_cuda_compatibility()
    benchmark(parse_args())


if __name__ == "__main__":
    main()
