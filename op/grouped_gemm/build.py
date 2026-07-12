"""原地构建扩展，并验证 Python 包可导入。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def resolve_arch_list(requested: str | None) -> str:
    if requested:
        return requested

    configured = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if configured:
        return configured

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA GPU；请使用 --arch-list 显式指定目标架构"
        )

    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch-list",
        default=None,
        help="TORCH_CUDA_ARCH_LIST，默认查询当前 GPU",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = resolve_arch_list(args.arch_list)
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "--inplace",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    sys.path.insert(0, str(ROOT))
    import cudaop_grouped_gemm

    if not callable(cudaop_grouped_gemm.gmm):
        raise RuntimeError("cudaop_grouped_gemm.gmm 不可调用")


if __name__ == "__main__":
    main()
