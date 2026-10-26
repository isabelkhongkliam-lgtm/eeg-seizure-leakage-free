"""
Metrics and statistics.

Accuracy is reported ONLY for comparability with prior work. It is the headline
number in the literature you are critiquing and it is close to meaningless at
clinical prevalence, so it must not lead your results tables.

Lead with: MCC, AUPRC, sensitivity at fixed specificity, false alarms per hour.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
)

SEGMENT_SECONDS = 178 / 173.61


def sensitivity_at_specificity(y, p, target_spec=0.99):
    """Sensitivity at a specificity the clinic would actually tolerate.

    Threshold-free metrics hide the operating point that matters; a detector
    with lovely AUROC can be useless at 99% specificity."""
    neg = p[y == 0]
    if len(neg) == 0:
        return np.nan, np.nan
    thr = np.quantile(neg, target_spec)
    pred = (p > thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    sens = tp / max((y == 1).sum(), 1)
    return float(sens), float(thr)


def false_alarms_per_hour(y, p, thr, segment_seconds=SEGMENT_SECONDS):
    """FP per hour of non-seizure recording -- the number that decides whether
    clinicians would switch the system off."""
    pred = (p > thr).astype(int)
    fp = ((pred == 1) & (y == 0)).sum()
    neg_hours = (y == 0).sum() * segment_seconds / 3600.0
    return float(fp / max(neg_hours, 1e-9))


def expected_calibration_error(y, p, n_bins=15):
    """ECE with equal-width bins. Feeds the deferred calibration paper too."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i > 0 else (p >= 0) & (p <= edges[1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def evaluate(y, p, thr=0.5, target_spec=0.99):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p > thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens_at, thr_at = sensitivity_at_specificity(y, p, target_spec)

    return {
        # reported for comparability only -- do not lead with this
        "accuracy": accuracy_score(y, pred),
        # the metrics that should lead
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "auprc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
        f"sens_at_spec{int(target_spec * 100)}": sens_at,
        "fa_per_hour_at_spec": false_alarms_per_hour(y, p, thr_at),
        "fa_per_hour_at_05": false_alarms_per_hour(y, p, 0.5),
        "ece": expected_calibration_error(y, p),
        "brier": brier_score_loss(y, p),
        "n_test": len(y),
        "n_pos": int(y.sum()),
    }


# ------------------------------------------------------------- statistics

def bootstrap_ci(y, p, metric_fn, n_boot=2000, alpha=0.05, seed=0):
    """Stratified bootstrap CI for a single metric on one test set."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        try:
            vals.append(metric_fn(y[idx], p[idx]))
        except ValueError:
            continue
    if not vals:
        return np.nan, np.nan
    return (float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def compare_across_seeds(a, b, paired=True):
    """
    Compare two configurations across seeds/folds.

    Wilcoxon rather than a t-test: with five seeds, normality is untestable and
    unassumable. Reports Cliff's delta as a distribution-free effect size --
    a p-value on n=5 says very little on its own.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired and len(a) == len(b):
        diff = a - b
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = sps.wilcoxon(a, b).pvalue
    else:
        p = sps.mannwhitneyu(a, b, alternative="two-sided").pvalue

    gt = sum((x > y_) for x in a for y_ in b)
    lt = sum((x < y_) for x in a for y_ in b)
    cliffs = (gt - lt) / (len(a) * len(b))

    return {
        "mean_a": float(a.mean()), "sd_a": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "mean_b": float(b.mean()), "sd_b": float(b.std(ddof=1)) if len(b) > 1 else 0.0,
        "delta": float(a.mean() - b.mean()),
        "p_value": float(p),
        "cliffs_delta": float(cliffs),
    }


def holm_bonferroni(pvals):
    """Holm correction. You will run many ablation comparisons; correct them,
    and say so in the paper."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        prev = max(prev, min(val, 1.0))
        adj[i] = prev
    return adj
