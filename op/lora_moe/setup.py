"""Build the LoRA-MoE CUDA extension."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "csrc"

required_sources = (
    "ops.cu",
    "common.h",
    "cumsum.h",
    "histogram.h",
    "indices.h",
    "replicate.h",
    "sort.h",
)
missing_sources = [
    source for source in required_sources
    if not (SOURCE_DIR / source).is_file()
]
if missing_sources:
    missing = ", ".join(missing_sources)
    raise FileNotFoundError(
        f"LoRA-MoE CUDA source files are missing from {SOURCE_DIR}: {missing}"
    )


setup(
    name="lora_moe_ops",
    version="0.1.0",
    description="CUDA operators used by LoRA-MoE",
    ext_modules=[
        CUDAExtension(
            name="lora_moe_ops",
            sources=[str(SOURCE_DIR / "ops.cu")],
            include_dirs=[str(SOURCE_DIR)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True),
    },
)
