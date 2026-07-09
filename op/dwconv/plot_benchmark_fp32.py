import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="benchmark_results_fp32.csv")
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

csv_path = Path(args.csv)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

sns.set_style("whitegrid")
colors = {
    "custom": "#2196F3",
    "biasopt": "#9C27B0",
    "dbuf": "#FF5722",
    "cudnn_nchw": "#FF9800",
    "cudnn_nhwc": "#4CAF50",
}

ksizes = sorted(df["k_size"].unique())
hs = sorted(df["h"].unique())
cs = sorted(df["c"].unique())

for ks in ksizes:
    dk = df[df["k_size"] == ks]
    fig, axes = plt.subplots(
        len(hs),
        len(cs),
        figsize=(5 * len(cs), 3.5 * len(hs)),
        sharex="col",
        sharey="row",
    )
    axes = np.asarray(axes).reshape(len(hs), len(cs))

    for ri, h_val in enumerate(hs):
        for ci, c_val in enumerate(cs):
            ax = axes[ri][ci]
            d = dk[(dk["h"] == h_val) & (dk["c"] == c_val)]
            for kernel in [
                "custom",
                "biasopt",
                "dbuf",
                "cudnn_nchw",
                "cudnn_nhwc",
            ]:
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
            if ri == len(hs) - 1:
                ax.set_xlabel("batch n")
            if ci == 0:
                ax.set_ylabel("TFlops/s")
            ax.set_title(f"c={c_val}, h={h_val}")
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Depthwise Conv FP32 Performance (k_size={ks}): custom vs cuDNN",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    output_path = out_dir / f"benchmark_fp32_k{ks}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
