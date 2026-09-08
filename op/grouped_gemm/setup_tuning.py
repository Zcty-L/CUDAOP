"""构建 CUTLASS Grouped GEMM 配置扫描扩展。"""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
CUTLASS_INCLUDE = ROOT.parents[2] / "cutlass" / "include"
SOURCES = (
    "tuning_ops.cpp",
    "tuning_up.cu",
    "tuning_down.cu",
    "tuning_bgrad.cu",
)

for source in SOURCES:
    if not (CSRC / source).is_file():
        raise FileNotFoundError(f"缺少配置扫描源文件：{CSRC / source}")
if not CUTLASS_INCLUDE.is_dir():
    raise FileNotFoundError(f"缺少 CUTLASS 头文件目录：{CUTLASS_INCLUDE}")


setup(
    name="cudaop_grouped_gemm_tuning",
    version="0.1.0",
    description="CUDAOP CUTLASS grouped GEMM configuration tuning",
    ext_modules=[
        CUDAExtension(
            name="cudaop_grouped_gemm._tuning",
            sources=[str(CSRC / source) for source in SOURCES],
            include_dirs=[
                str(CSRC),
                str(CUTLASS_INCLUDE),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-std=c++17",
                ],
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True),
    },
)
