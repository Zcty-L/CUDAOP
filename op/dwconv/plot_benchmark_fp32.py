import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("benchmark_results_fp32.csv")

sns.set_style("whitegrid")
colors = {"custom": "#2196F3", "biasopt": "#9C27B0", "dbuf": "#FF5722",
          "cudnn_nchw": "#FF9800", "cudnn_nhwc": "#4CAF50"}

ksizes = sorted(df["k_size"].unique())
hs = sorted(df["h"].unique())
cs = sorted(df["c"].unique())

for ks in ksizes:
    dk = df[df["k_size"] == ks]
    fig, axes = plt.subplots(len(hs), len(cs), figsize=(5 * len(cs), 3.5 * len(hs)),
                             sharex="col", sharey="row")
    for ri, h_val in enumerate(hs):
        for ci, c_val in enumerate(cs):
            ax = axes[ri][ci] if len(hs) > 1 and len(cs) > 1 else axes
            d = dk[(dk["h"] == h_val) & (dk["c"] == c_val)]
            for k in ["custom", "biasopt", "dbuf", "cudnn_nchw", "cudnn_nhwc"]:
                dd = d[d["kernel"] == k].sort_values("n")
                if dd.empty:
                    continue
                ax.plot(dd["n"], dd["gflops"] / 1e3, marker="o", label=k, color=colors[k])
            if ri == len(hs) - 1:
                ax.set_xlabel("batch n")
            if ci == 0:
                ax.set_ylabel("TFlops/s")
            ax.set_title(f"c={c_val}, h={h_val}")
            ax.legend(fontsize=7)

    fig.suptitle(f"Depthwise Conv FP32 Performance (k_size={ks}): custom vs cuDNN", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"benchmark_fp32_k{ks}.png", dpi=150, bbox_inches="tight")
    print(f"Saved benchmark_fp32_k{ks}.png")
