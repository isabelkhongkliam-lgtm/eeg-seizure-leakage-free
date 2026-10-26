"""
Publication-quality figures from corrected results.

    python visualize_results.py

Outputs to report/ directory:
  - roc_curves.png            ROC curves (recording-wise, per model)
  - pr_curves.png             Precision-Recall curves (recording-wise)
  - confusion_matrices.png    Confusion matrices (LORO, per model)
  - metric_boxplots.png       Box plots: per-seed variation (recording-wise)
  - feature_importance.png    Top-20 features bar chart
  - calibration.png           Reliability diagram (LORO)
"""

from __future__ import annotations
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

OUT = "report"
os.makedirs(OUT, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────
COLORS = {
    "logreg":        "#4e79a7",
    "lightgbm":      "#59a14f",
    "svm_rbf":       "#f28e2b",
    "random_forest": "#e15759",
    "full":          "#76b7b2",
    "no_attn":       "#b07aa1",
}
NICE = {
    "logreg": "Logistic Regression",
    "lightgbm": "LightGBM",
    "svm_rbf": "SVM-RBF",
    "random_forest": "Random Forest",
    "full": "Deep (full)",
    "no_attn": "Deep (no attn)",
}


def _style():
    """Common plot styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ═══════════════════════════════════════════════════════════════════════
# 1.  ROC + PR curves  (from LORO predictions, segment-level)
# ═══════════════════════════════════════════════════════════════════════

def plot_roc_pr():
    """Plot ROC and PR curves from LORO per-segment predictions."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for name in ["logreg", "lightgbm", "svm_rbf"]:
        path = f"results/loro_preds_{name}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        y, p = df["y"].values, df["p"].values

        # ROC
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax_roc.plot(fpr, tpr, color=COLORS[name], lw=2,
                    label=f"{NICE[name]} (AUC={auc:.4f})")

        # PR
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax_pr.plot(rec, prec, color=COLORS[name], lw=2,
                   label=f"{NICE[name]} (AP={ap:.4f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC Curves (LORO)")
    ax_roc.legend(loc="lower right", fontsize=9)

    prevalence = None
    for name in ["logreg", "lightgbm", "svm_rbf"]:
        path = f"results/loro_preds_{name}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            prevalence = df["y"].mean()
            break
    if prevalence is not None:
        ax_pr.axhline(prevalence, color="grey", ls="--", lw=0.8, alpha=0.5,
                      label=f"Prevalence ({prevalence:.2f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Precision–Recall Curves (LORO)")
    ax_pr.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{OUT}/roc_curves.png")
    fig.savefig(f"{OUT}/pr_curves.png".replace("pr_", "pr_"))  # same fig, both curves
    print(f"  wrote {OUT}/roc_curves.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 2.  Confusion matrices  (LORO, threshold=0.5)
# ═══════════════════════════════════════════════════════════════════════

def plot_confusion_matrices():
    models = ["logreg", "lightgbm", "svm_rbf"]
    available = [m for m in models if os.path.exists(f"results/loro_preds_{m}.csv")]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    for ax, name in zip(axes, available):
        df = pd.read_csv(f"results/loro_preds_{name}.csv")
        y, p = df["y"].values, (df["p"].values > 0.5).astype(int)
        cm = confusion_matrix(y, p, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Non-seizure", "Seizure"])
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(NICE[name], fontsize=12, fontweight="bold")

    plt.suptitle("Confusion Matrices — LORO Evaluation", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT}/confusion_matrices.png")
    print(f"  wrote {OUT}/confusion_matrices.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Box plots — per-seed variation (recording-wise, from runs_v2)
# ═══════════════════════════════════════════════════════════════════════

def plot_boxplots():
    if not os.path.exists("results/runs_v2.csv"):
        return
    df = pd.read_csv("results/runs_v2.csv")
    rw = df[df["split"] == "recording_wise"].copy()
    if rw.empty:
        return

    # Aggregate folds → per-seed means, then box-plot across seeds
    metrics = ["accuracy", "mcc", "auprc", "sens_at_spec99"]
    nice_metric = {"accuracy": "Accuracy", "mcc": "MCC",
                   "auprc": "AUPRC", "sens_at_spec99": "Sensitivity\n@Spec 99%"}
    seed_means = rw.groupby(["variant", "seed"])[metrics].mean().reset_index()

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    variants = sorted(seed_means["variant"].unique())
    colors = [COLORS.get(v, "#999") for v in variants]
    labels = [NICE.get(v, v) for v in variants]

    for ax, m in zip(axes, metrics):
        data = [seed_means[seed_means["variant"] == v][m].values for v in variants]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(nice_metric.get(m, m), fontsize=12)
        ax.set_ylim(max(0, ax.get_ylim()[0] - 0.01), 1.0)

    plt.suptitle("Per-Seed Variation — Recording-Wise Split", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT}/metric_boxplots.png")
    print(f"  wrote {OUT}/metric_boxplots.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 4.  Feature importance  (top 20 bar chart)
# ═══════════════════════════════════════════════════════════════════════

def plot_feature_importance():
    path = "results/feature_importance_named.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path).sort_values("mean_drop_in_auprc", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df["mean_drop_in_auprc"],
                   xerr=df["sd_across_folds"], capsize=3,
                   color="#4e79a7", alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Drop in AUPRC (permutation importance)", fontsize=11)
    ax.set_title("Top-20 Features — LightGBM, Recording-Wise", fontsize=13,
                 fontweight="bold")
    ax.axvline(0, color="grey", lw=0.8, ls="--")

    plt.tight_layout()
    fig.savefig(f"{OUT}/feature_importance.png")
    print(f"  wrote {OUT}/feature_importance.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Calibration / Reliability diagram  (LORO)
# ═══════════════════════════════════════════════════════════════════════

def plot_calibration():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfectly calibrated")

    for name in ["logreg", "lightgbm", "svm_rbf"]:
        path = f"results/loro_preds_{name}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        y, p = df["y"].values, df["p"].values

        n_bins = 10
        edges = np.linspace(0, 1, n_bins + 1)
        mid, frac_pos = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
            if mask.sum() == 0:
                continue
            mid.append((lo + hi) / 2)
            frac_pos.append(y[mask].mean())

        ax.plot(mid, frac_pos, "o-", color=COLORS[name], lw=2,
                markersize=6, label=NICE[name])

    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
           title="Calibration — LORO Evaluation")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{OUT}/calibration.png")
    print(f"  wrote {OUT}/calibration.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 6.  Protocol comparison (upgraded from make_report.py bar chart)
# ═══════════════════════════════════════════════════════════════════════

def plot_protocol_comparison():
    if not os.path.exists("results/runs_v2.csv"):
        return
    df = pd.read_csv("results/runs_v2.csv")
    metrics = ["accuracy", "mcc", "auprc", "sens_at_spec99"]
    nice_m = {"accuracy": "Accuracy", "mcc": "MCC", "auprc": "AUPRC",
              "sens_at_spec99": "Sens @Spec99%"}
    splits = ["random_segment", "recording_wise"]
    nice_s = {"random_segment": "Random Segment", "recording_wise": "Recording-Wise"}

    ps = df.groupby(["variant", "split", "seed"])[metrics].mean().reset_index()
    t = ps.groupby(["variant", "split"])[metrics].agg(["mean", "std"])
    variants = sorted(df["variant"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5))
    w = 0.35

    for ax, m in zip(axes, metrics):
        for i, s in enumerate(splits):
            mu = [t.loc[(v, s), (m, "mean")] if (v, s) in t.index else np.nan
                  for v in variants]
            sd = [t.loc[(v, s), (m, "std")] if (v, s) in t.index else 0
                  for v in variants]
            x = np.arange(len(variants))
            ax.bar(x + i * w, mu, w, yerr=sd, capsize=3,
                   label=nice_s[s], alpha=0.85,
                   color=["#4e79a7", "#e15759"][i])
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels([NICE.get(v, v) for v in variants],
                           rotation=35, ha="right", fontsize=8)
        ax.set_title(nice_m[m], fontsize=12)
        ymin = np.nanmin([t.loc[idx, (m, "mean")] for idx in t.index
                          if idx in t.index]) - 0.03
        ax.set_ylim(max(0, ymin), 1.0)
        if m == metrics[0]:
            ax.legend(fontsize=9)

    plt.suptitle("Random Segment vs Recording-Wise Split", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT}/protocol_comparison.png")
    print(f"  wrote {OUT}/protocol_comparison.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _style()
    print("Generating publication figures...")
    plot_roc_pr()
    plot_confusion_matrices()
    plot_boxplots()
    plot_feature_importance()
    plot_calibration()
    plot_protocol_comparison()
    print("Done.  All figures saved to report/")
