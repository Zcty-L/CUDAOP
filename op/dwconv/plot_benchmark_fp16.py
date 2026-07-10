import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="benchmark_results_fp16.csv")
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

csv_path = Path(args.csv)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

sns.set_style("whitegrid")
colors = {
    "custom": "#2196F3",
    "biasopt": "#9C27B0",
    "db": "#FF5722",
    "cudnn_nchw": "#FF9800",
    "cudnn_nhwc": "#4CAF50",
}

kernel_sizes = sorted(df["k_size"].unique())
hs = sorted(df["h"].unique())
cs = sorted(df["c"].unique())

for kernel_size in kernel_sizes:
    kernel_data = df[df["k_size"] == kernel_size]
    fig, axes = plt.subplots(
        len(hs),
        len(cs),
        figsize=(5 * len(cs), 3.5 * len(hs)),
        sharex="col",
        sharey="row",
    )
    axes = np.asarray(axes).reshape(len(hs), len(cs))

    for row, h_value in enumerate(hs):
        for column, c_value in enumerate(cs):
            ax = axes[row][column]
            case_data = kernel_data[
                (kernel_data["h"] == h_value)
                & (kernel_data["c"] == c_value)
            ]
            for implementation in [
                "custom",
                "biasopt",
                "db",
                "cudnn_nchw",
                "cudnn_nhwc",
            ]:
                implementation_data = case_data[
                    case_data["kernel"] == implementation
                ].sort_values("n")
                if implementation_data.empty:
                    continue
                ax.plot(
                    implementation_data["n"],
                    implementation_data["gflops"] / 1e3,
                    marker="o",
                    label=implementation,
                    color=colors[implementation],
                )
            if row == len(hs) - 1:
                ax.set_xlabel("batch n")
            if column == 0:
                ax.set_ylabel("TFlops/s")
            ax.set_title(f"c={c_value}, h={h_value}")
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Depthwise Conv FP16 Performance (k_size={kernel_size}): "
        "custom vs cuDNN",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    output_path = out_dir / f"benchmark_fp16_k{kernel_size}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
