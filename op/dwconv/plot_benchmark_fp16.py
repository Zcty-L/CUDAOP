import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("benchmark_results_fp16.csv")

sns.set_style("whitegrid")
hs = sorted(df["h"].unique())
cs = sorted(df["c"].unique())
colors = {"custom": "#2196F3", "cudnn_nchw": "#FF9800", "cudnn_nhwc": "#4CAF50"}

fig, axes = plt.subplots(len(hs), len(cs), figsize=(5 * len(cs), 3.5 * len(hs)), sharex="col", sharey="row")

for ri, h_val in enumerate(hs):
    for ci, c_val in enumerate(cs):
        ax = axes[ri][ci]
        d = df[(df["h"] == h_val) & (df["c"] == c_val)]
        for k in ["custom", "cudnn_nchw", "cudnn_nhwc"]:
            dd = d[d["kernel"] == k].sort_values("n")
            ax.plot(dd["n"], dd["gflops"] / 1e3, marker="o", label=k, color=colors[k])
        if ri == len(hs) - 1:
            ax.set_xlabel("batch n")
        if ci == 0:
            ax.set_ylabel("TFlops/s")
        ax.set_title(f"c={c_val}, h={h_val}")
        ax.legend(fontsize=7)

fig.suptitle("Depthwise Conv FP16 Performance: custom vs cuDNN (NCHW/NHWC)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("benchmark_fp16.png", dpi=150, bbox_inches="tight")
print("Saved benchmark_fp16.png")
