"""构建 cudaop_grouped_gemm CUDA 扩展。"""

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
CUTLASS_INCLUDE = ROOT.parents[2] / "cutlass" / "include"

for source in ("ops.cu", "grouped_gemm.cuh", "fused_lora.cuh"):
    if not (CSRC / source).is_file():
        raise FileNotFoundError(f"缺少 CUDA 源文件：{CSRC / source}")
if not CUTLASS_INCLUDE.is_dir():
    raise FileNotFoundError(f"缺少 CUTLASS 头文件目录：{CUTLASS_INCLUDE}")


setup(
    name="cudaop_grouped_gemm",
    version="0.1.0",
    description="CUDAOP CUTLASS grouped GEMM",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cudaop_grouped_gemm._C",
            sources=[str(CSRC / "ops.cu")],
            include_dirs=[
                str(CSRC),
                str(CUTLASS_INCLUDE),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                ],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True),
    },
)
