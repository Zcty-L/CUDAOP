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
parser.add_argument("--csv", default="benchmark_results_fp32.csv")
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
    "dbuf": "#FF5722",
    "cudnn_nchw": "#FF9800",
    "cudnn_nhwc": "#4CAF50",
}
implementation_order = [
    "initial",
    "skill_final",
    "custom",
    "biasopt",
    "db",
    "dbuf",
    "cudnn_nchw",
    "cudnn_nhwc",
]

ksizes = sorted(df["k_size"].unique())

for ks in ksizes:
    dk = df[df["k_size"] == ks]
    if has_explicit_case:
        cases = list(dk["case"].drop_duplicates())
        columns = min(3, len(cases))
    else:
        cases = [
            f"c{c_value}_h{h_value}"
            for h_value in sorted(dk["h"].unique())
            for c_value in sorted(dk["c"].unique())
        ]
        columns = len(dk["c"].unique())
    rows = math.ceil(len(cases) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
    )

    for index, case_name in enumerate(cases):
        ax = axes[index // columns][index % columns]
        d = dk[dk["case"] == case_name]
        for kernel in implementation_order:
            dd = d[d["kernel"] == kernel].sort_values("n")
            if dd.empty:
                continue
            ax.plot(
                dd["n"],
                dd["gflops"] / 1e3,
                marker="o",
                label=kernel,
                color=colors[kernel],
            )
        ax.set_xlabel("batch n")
        ax.set_ylabel("TFlops/s")
        first_row = d.iloc[0]
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
        f"Depthwise Conv FP32 Performance (k_size={ks}): {comparison}",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    suffix = "_skill" if skill_mode else ""
    output_path = out_dir / f"benchmark_fp32{suffix}_k{ks}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
