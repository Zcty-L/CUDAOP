"""Benchmark the custom CuPy neurons against SpikingJelly's CuPy backend."""

import argparse
import csv
from dataclasses import dataclass
from importlib import metadata
import logging
from pathlib import Path
from statistics import median
from typing import Any

import cupy as cp
import torch
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import surrogate as sj_surrogate

import neuron_cupy
import neuron_surrogate


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT.parent.parent / "build" / "neuron_spikingjelly_benchmark.csv"
)
NEURON_TYPES = ("IF", "LIF", "PLIF")
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


@dataclass
class RunState:
    inputs: torch.Tensor
    outputs: torch.Tensor


@dataclass
class Timing:
    forward_ms: float
    backward_ms: float
    noise_ratio: float

    @property
    def total_ms(self) -> float:
        return self.forward_ms + self.backward_ms


class NeuronRunner:
    """Common training path for one multi-step neuron implementation."""

    def __init__(
        self,
        implementation: str,
        neuron_type: str,
        node: torch.nn.Module,
        reset_state: bool,
    ) -> None:
        self.implementation = implementation
        self.neuron_type = neuron_type
        self.node = node.cuda()
        self.reset_state = reset_state

    def prepare(self) -> None:
        self.node.zero_grad(set_to_none=True)
        if self.reset_state:
            functional.reset_net(self.node)

    def forward(self, inputs: torch.Tensor) -> RunState:
        leaf_inputs = inputs.detach().requires_grad_(True)
        outputs = self.node(leaf_inputs)
        if not isinstance(outputs, torch.Tensor):
            raise RuntimeError(
                f"{self.implementation} 应仅返回 spike tensor"
            )
        return RunState(leaf_inputs, outputs)

    def backward(
        self,
        state: RunState,
        grad_spike: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        state.outputs.backward(grad_spike)
        if state.inputs.grad is None:
            raise RuntimeError(
                f"{self.implementation} backward 未生成输入梯度"
            )
        weight = getattr(self.node, "w", None)
        weight_grad = None if weight is None else weight.grad
        return state.inputs.grad, weight_grad


def create_runners(neuron_type: str) -> tuple[NeuronRunner, NeuronRunner]:
    custom_types = {
        "IF": neuron_cupy.IFNode,
        "LIF": neuron_cupy.LIFNode,
        "PLIF": neuron_cupy.ParametricLIFNode,
    }
    spikingjelly_types = {
        "IF": sj_neuron.IFNode,
        "LIF": sj_neuron.LIFNode,
        "PLIF": sj_neuron.ParametricLIFNode,
    }
    common_options: dict[str, Any] = {
        "detach_reset": True,
        "store_v_seq": False,
    }
    custom_node = custom_types[neuron_type](
        surrogate_function=neuron_surrogate.ATan(),
        **common_options,
    )
    spikingjelly_node = spikingjelly_types[neuron_type](
        surrogate_function=sj_surrogate.ATan(),
        step_mode="m",
        backend="cupy",
        **common_options,
    )
    return (
        NeuronRunner(
            "custom_cupy",
            neuron_type,
            custom_node,
            reset_state=False,
        ),
        NeuronRunner(
            "spikingjelly_cupy",
            neuron_type,
            spikingjelly_node,
            reset_state=True,
        ),
    )


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
    runners: tuple[NeuronRunner, NeuronRunner],
    inputs: torch.Tensor,
    grad_spike: torch.Tensor,
) -> None:
    results: dict[
        str,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    ] = {}
    for runner in runners:
        runner.prepare()
        state = runner.forward(inputs)
        input_grad, weight_grad = runner.backward(state, grad_spike)
        results[runner.implementation] = (
            state.outputs.detach(),
            input_grad.detach(),
            None if weight_grad is None else weight_grad.detach().clone(),
        )

    custom = results["custom_cupy"]
    reference = results["spikingjelly_cupy"]
    assert_close(custom[0], reference[0], inputs.dtype, "spike")
    assert_close(custom[1], reference[1], inputs.dtype, "input gradient")
    if custom[2] is not None and reference[2] is not None:
        assert_close(
            custom[2],
            reference[2],
            inputs.dtype,
            "parameter gradient",
        )
    torch.cuda.synchronize()


def measure_runner(
    runner: NeuronRunner,
    inputs: torch.Tensor,
    grad_spike: torch.Tensor,
    warmup: int,
    iterations: int,
    groups: int,
) -> Timing:
    for _ in range(warmup):
        runner.prepare()
        state = runner.forward(inputs)
        runner.backward(state, grad_spike)
    torch.cuda.synchronize()

    forward_group_medians: list[float] = []
    backward_group_medians: list[float] = []
    total_group_medians: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(groups):
        forward_samples: list[float] = []
        backward_samples: list[float] = []
        for _ in range(iterations):
            runner.prepare()
            start.record()
            state = runner.forward(inputs)
            end.record()
            end.synchronize()
            forward_samples.append(start.elapsed_time(end))

            start.record()
            runner.backward(state, grad_spike)
            end.record()
            end.synchronize()
            backward_samples.append(start.elapsed_time(end))
        forward_group_medians.append(median(forward_samples))
        backward_group_medians.append(median(backward_samples))
        total_group_medians.append(
            forward_group_medians[-1] + backward_group_medians[-1]
        )

    total_median = median(total_group_medians)
    noise_ratio = (
        max(total_group_medians) - min(total_group_medians)
    ) / total_median

    return Timing(
        forward_ms=median(forward_group_medians),
        backward_ms=median(backward_group_medians),
        noise_ratio=noise_ratio,
    )


def element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def write_results(rows: list[dict[str, str | int | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def benchmark(args: argparse.Namespace) -> None:
    shape = tuple(args.shape)
    validation_shape = tuple(args.validation_shape)
    total_elements = element_count(shape)
    elements_per_step = element_count(shape[1:])
    if elements_per_step % 8 != 0:
        raise ValueError("每个时间步的元素数必须是 8 的倍数")

    torch_cuda = torch.version.cuda or "unknown"
    cupy_runtime = cp.cuda.runtime.runtimeGetVersion()
    versions = {
        "torch": torch.__version__,
        "torch_cuda": torch_cuda,
        "cupy": cp.__version__,
        "cupy_runtime": cupy_runtime,
        "spikingjelly": package_version("spikingjelly"),
    }
    logging.info(
        "测试配置: device=%s, capability=%s, shape=%s",
        torch.cuda.get_device_name(),
        torch.cuda.get_device_capability(),
        shape,
    )
    logging.info(
        "软件环境: PyTorch=%s, PyTorch CUDA=%s, CuPy=%s, "
        "CuPy runtime=%d, SpikingJelly=%s",
        versions["torch"],
        versions["torch_cuda"],
        versions["cupy"],
        versions["cupy_runtime"],
        versions["spikingjelly"],
    )
    torch_cuda_major = int(torch_cuda.split(".", maxsplit=1)[0])
    cupy_cuda_major = cupy_runtime // 1000
    if torch_cuda_major != cupy_cuda_major:
        logging.warning(
            "PyTorch 与 CuPy 的 CUDA 主版本不同: PyTorch=%s, CuPy=%d",
            torch_cuda,
            cupy_cuda_major,
        )
    logging.info(
        "工作量: elements=%d, FP32 input=%.1f MiB, FP16 input=%.1f MiB",
        total_elements,
        total_elements * 4 / (1024 * 1024),
        total_elements * 2 / (1024 * 1024),
    )
    logging.info(
        "计时配置: warmup=%d, groups=%d, iterations/group=%d, "
        "validation_shape=%s",
        args.warmup,
        args.groups,
        args.iterations,
        validation_shape,
    )
    logging.info(
        "统一语义: multi-step、ATan、detach_reset=True、spike-only"
    )

    rows: list[dict[str, str | int | float]] = []
    for neuron_type in args.neurons:
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            runners = create_runners(neuron_type)
            validation_inputs = (
                torch.randn(validation_shape, device="cuda", dtype=dtype)
                * 0.5
            )
            validation_grad = torch.full_like(
                validation_inputs,
                1.0 / validation_inputs.numel(),
            )
            logging.info(
                "\n验证阶段: neuron=%s, dtype=%s",
                neuron_type,
                dtype_name,
            )
            validate_runners(
                runners,
                validation_inputs,
                validation_grad,
            )

            inputs = torch.randn(shape, device="cuda", dtype=dtype) * 0.5
            grad_spike = torch.randn_like(inputs)
            timings: dict[str, Timing] = {}
            for runner in runners:
                timings[runner.implementation] = measure_runner(
                    runner,
                    inputs,
                    grad_spike,
                    args.warmup,
                    args.iterations,
                    args.groups,
                )

            reference_total = timings["spikingjelly_cupy"].total_ms
            logging.info(
                "%-20s %11s %11s %11s %11s %11s %16s",
                "implementation",
                "forward/ms",
                "backward/ms",
                "total/ms",
                "GElem/s",
                "noise",
                "custom speedup",
            )
            for runner in runners:
                timing = timings[runner.implementation]
                throughput = (
                    2 * total_elements / (timing.total_ms * 1e6)
                )
                speedup = reference_total / timing.total_ms
                logging.info(
                    "%-20s %11.4f %11.4f %11.4f %11.3f %10.2f%% "
                    "%15.3fx",
                    runner.implementation,
                    timing.forward_ms,
                    timing.backward_ms,
                    timing.total_ms,
                    throughput,
                    timing.noise_ratio * 100,
                    speedup,
                )
                rows.append(
                    {
                        "neuron": neuron_type,
                        "dtype": dtype_name,
                        "shape": "x".join(str(value) for value in shape),
                        "implementation": runner.implementation,
                        "forward_ms": timing.forward_ms,
                        "backward_ms": timing.backward_ms,
                        "total_ms": timing.total_ms,
                        "throughput_gelem_s": throughput,
                        "noise_ratio": timing.noise_ratio,
                        "custom_speedup": speedup,
                        "gpu": torch.cuda.get_device_name(),
                        "torch": versions["torch"],
                        "torch_cuda": versions["torch_cuda"],
                        "cupy": versions["cupy"],
                        "cupy_runtime": versions["cupy_runtime"],
                        "spikingjelly": versions["spikingjelly"],
                    }
                )

    output_path = Path(args.output).resolve()
    write_results(rows, output_path)
    logging.info("\n原始结果: %s", output_path)
    logging.info("[SUCCESS] CuPy neuron 与 SpikingJelly 吞吐对比完成")


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
        "--validation-shape",
        nargs=5,
        type=int,
        default=(4, 2, 4, 8, 8),
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
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("neuron 性能测试需要 CUDA GPU")
    benchmark(parse_args())


if __name__ == "__main__":
    main()
