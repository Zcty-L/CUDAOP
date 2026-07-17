"""Build ``neuron_ops`` in place and verify its exported C++ interfaces."""

import argparse
import importlib
import logging
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parent
EXPECTED_OPS = (
    "IFNodeForward",
    "IFNodeBackward",
    "LIFNodeForward",
    "LIFNodeBackward",
    "PLIFNodeForward",
    "PLIFNodeBackward",
)


def resolve_arch_list(requested: str | None) -> str:
    if requested:
        return requested

    configured = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if configured:
        return configured

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA GPU；请通过 --arch-list 显式指定目标架构"
        )

    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def build(arch_list: str | None) -> None:
    selected_arch = resolve_arch_list(arch_list)
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = selected_arch
    env.pop("MAKEFLAGS", None)
    env.pop("MFLAGS", None)

    logging.info("构建配置: TORCH_CUDA_ARCH_LIST=%s", selected_arch)
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    sys.path.insert(0, str(ROOT))
    _ = torch.__version__
    module = importlib.import_module("neuron_ops")
    missing_ops = [name for name in EXPECTED_OPS if not hasattr(module, name)]
    if missing_ops:
        raise RuntimeError(
            "neuron_ops 缺少导出接口: " + ", ".join(missing_ops)
        )

    logging.info("[SUCCESS] neuron_ops 构建成功，C++ 接口完整")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch-list",
        help=(
            "传给 TORCH_CUDA_ARCH_LIST 的 CUDA 架构；"
            "默认使用已有环境变量，否则查询当前 GPU"
        ),
    )
    args = parser.parse_args()
    build(args.arch_list)


if __name__ == "__main__":
    main()
