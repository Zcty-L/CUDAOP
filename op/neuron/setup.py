"""Build the neuron CUDA kernels as a PyTorch extension."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "csrc"

required_sources = (
    SOURCE_DIR / "ops.cpp",
    ROOT / "if.cu",
    ROOT / "lif.cu",
    ROOT / "plif.cu",
)
missing_sources = [str(source) for source in required_sources if not source.is_file()]
if missing_sources:
    raise FileNotFoundError(
        "neuron_ops source files are missing: " + ", ".join(missing_sources)
    )


setup(
    name="neuron_ops",
    version="0.1.0",
    description="CUDA neuron operators for PyTorch",
    ext_modules=[
        CUDAExtension(
            name="neuron_ops",
            sources=[str(source) for source in required_sources],
            include_dirs=[str(ROOT)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                ],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True),
    },
)
