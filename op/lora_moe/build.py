"""Build ``lora_moe_ops`` in place and verify that Python can import it."""

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


def build(arch_list: str | None) -> None:
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = resolve_arch_list(arch_list)

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
    # Importing torch first loads the shared libraries required by the
    # extension (for example libc10.so) into the current process.
    _ = torch.__version__
    module = importlib.import_module("lora_moe_ops")
    expected_ops = (
        "exclusive_cumsum",
        "histogram",
        "inclusive_cumsum",
        "indices",
        "replicate_forward",
        "replicate_backward",
        "sort",
    )
    missing_ops = [name for name in expected_ops if not hasattr(module, name)]
    if missing_ops:
        raise RuntimeError(
            "lora_moe_ops is missing symbols: " + ", ".join(missing_ops)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch-list",
        help=(
            "CUDA architectures passed through TORCH_CUDA_ARCH_LIST "
            "(default: existing environment value, otherwise current GPU)"
        ),
    )
    args = parser.parse_args()
    build(args.arch_list)


if __name__ == "__main__":
    main()
