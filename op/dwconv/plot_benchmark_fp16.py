import argparse
import logging
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="benchmark_results_fp16.csv")
parser.add_argument("--skill-csv")
parser.add_argument("--merged-csv")
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

csv_path = Path(args.csv)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
skill_mode = args.skill_csv is not None
if skill_mode:
    initial = df[df["kernel"] == "custom"].copy()
    initial["kernel"] = "initial"
    cudnn = df[df["kernel"] == "cudnn_nchw"].copy()
    skill_final = pd.read_csv(args.skill_csv)
    df = pd.concat([initial, skill_final, cudnn], ignore_index=True)
    if args.merged_csv:
        merged_csv_path = Path(args.merged_csv)
        merged_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(merged_csv_path, index=False)
        logger.info("Saved %s", merged_csv_path)
has_explicit_case = "case" in df.columns
if not has_explicit_case:
    df["case"] = "c" + df["c"].astype(str) + "_h" + df["h"].astype(str)

sns.set_style("whitegrid")
colors = {
    "initial": "#2196F3",
    "skill_final": "#D32F2F",
    "custom": "#2196F3",
    "biasopt": "#9C27B0",
    "db": "#FF5722",
    "cudnn_nchw": "#FF9800",
    "cudnn_nhwc": "#4CAF50",
}
implementation_order = [
    "initial",
    "skill_final",
    "custom",
    "biasopt",
    "db",
    "cudnn_nchw",
    "cudnn_nhwc",
]

kernel_sizes = sorted(df["k_size"].unique())

for kernel_size in kernel_sizes:
    kernel_data = df[df["k_size"] == kernel_size]
    if has_explicit_case:
        cases = list(kernel_data["case"].drop_duplicates())
        columns = min(3, len(cases))
    else:
        cases = [
            f"c{c_value}_h{h_value}"
            for h_value in sorted(kernel_data["h"].unique())
            for c_value in sorted(kernel_data["c"].unique())
        ]
        columns = len(kernel_data["c"].unique())
    rows = math.ceil(len(cases) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
    )

    for index, case_name in enumerate(cases):
        ax = axes[index // columns][index % columns]
        case_data = kernel_data[kernel_data["case"] == case_name]
        for implementation in implementation_order:
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
        ax.set_xlabel("batch n")
        ax.set_ylabel("TFlops/s")
        first_row = case_data.iloc[0]
        if has_explicit_case:
            ax.set_title(case_name)
        else:
            ax.set_title(f"c={first_row['c']}, h={first_row['h']}")
        if ax.lines:
            ax.legend(fontsize=7)

    for index in range(len(cases), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    comparison = "initial vs skill_final vs cuDNN" \
        if skill_mode or has_explicit_case else "custom vs cuDNN"
    fig.suptitle(
        f"Depthwise Conv FP16 Performance (k_size={kernel_size}): "
        f"{comparison}",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    suffix = "_skill" if skill_mode else ""
    output_path = out_dir / (
        f"benchmark_fp16{suffix}_k{kernel_size}.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
