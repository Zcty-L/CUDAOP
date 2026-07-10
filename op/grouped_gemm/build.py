"""原地构建扩展，并验证 Python 包可导入。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch-list",
        default=None,
        help="TORCH_CUDA_ARCH_LIST，默认使用 12.0",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = (
        args.arch_list
        or env.get("TORCH_CUDA_ARCH_LIST")
        or "12.0"
    )
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
