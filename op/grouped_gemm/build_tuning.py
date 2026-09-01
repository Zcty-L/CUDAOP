"""构建并验证 CUTLASS Grouped GEMM 配置扫描扩展。"""

import argparse
import importlib
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
    env.setdefault("MAX_JOBS", "4")
    tuning_build_root = ROOT / "build" / "tuning"
    subprocess.run(
        [
            sys.executable,
            "setup_tuning.py",
            "build_ext",
            "--inplace",
            "--build-temp",
            str(tuning_build_root / "temp"),
            "--build-lib",
            str(tuning_build_root / "lib"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    sys.path.insert(0, str(ROOT))
    tuning = importlib.import_module("cudaop_grouped_gemm._tuning")
    required = (
        "up_tb128x128x16_w64x64x16_i8_s2",
        "down_tb128x32x16_w64x32x16_i8_s4",
        "bgrad_tb32x128x16_w32x64x16_i8_s4",
    )
    for operation_name in required:
        if not callable(getattr(tuning, operation_name, None)):
            raise RuntimeError(f"配置扫描接口不可调用：{operation_name}")


if __name__ == "__main__":
    main()
