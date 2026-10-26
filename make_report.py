"""Builds paper tables + figures from results/runs_v2.csv (+ results/loro_summary.csv).
    python make_report.py"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("report", exist_ok=True)
df = pd.read_csv("results/runs_v2.csv")
M = ["accuracy", "mcc", "auprc", "sens_at_spec99", "fa_per_hour_at_05"]
ps = df.groupby(["model", "variant", "split", "seed"])[M].mean().reset_index()   # mean over folds per seed
t = ps.groupby(["variant", "split"])[M].agg(["mean", "std"])
fmt = pd.DataFrame({(s, m): [f"{t.loc[(v, s), (m, 'mean')]:.4f} +/- {t.loc[(v, s), (m, 'std')]:.4f}"
                             if (v, s) in t.index else "" for v in sorted(df.variant.unique())]
                    for s in sorted(df.split.unique()) for m in M}, index=sorted(df.variant.unique()))
fmt.to_csv("report/table_protocols.csv"); print(fmt.to_string())
if os.path.exists("results/loro_summary.csv"):
    print(pd.read_csv("results/loro_summary.csv").round(4).to_string(index=False))
v = sorted(df.variant.unique()); sp = [s for s in ["random_segment", "recording_wise"] if s in df.split.unique()]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, m in zip(axes, ["accuracy", "mcc"]):
    w = 0.8 / len(sp)
    for i, s in enumerate(sp):
        mu = [t.loc[(x, s), (m, "mean")] if (x, s) in t.index else np.nan for x in v]
        sd = [t.loc[(x, s), (m, "std")] if (x, s) in t.index else 0 for x in v]
        ax.bar(np.arange(len(v)) + i * w, mu, w, yerr=sd, capsize=3, label=s)
    ax.set_xticks(np.arange(len(v)) + w / 2 * (len(sp) - 1)); ax.set_xticklabels(v, rotation=30, ha="right")
    ax.set_title(m); ax.set_ylim(max(0, np.nanmin(t[(m, "mean")]) - 0.05), 1.0); ax.legend()
plt.tight_layout(); plt.savefig("report/protocol_comparison.png", dpi=300)
print("wrote report/table_protocols.csv, report/protocol_comparison.png")
