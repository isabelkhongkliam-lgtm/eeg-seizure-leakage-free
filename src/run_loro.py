"""
Leave-one-recording-out with POOLED metrics.

A single held-out recording contains one class, so MCC/AUPRC are undefined per
fold. We therefore collect every held-out prediction and compute metrics once.

    python -m src.run_loro --csv data/uci_full.csv --model lightgbm
    python -m src.run_loro --csv data/uci_full.csv --model logreg --max-groups 100

Classical models use FIXED hyperparameters here (tuned once on a recording-
grouped split beforehand) -- a grid search inside 500 folds is not affordable.
Deep LORO (500 trainings) is not practical on a laptop; use --max-groups
(random subset of recordings, all models on the SAME subset) and say so.
"""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd
from .data_uci import load_uci, assert_no_group_leakage
from .metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True, choices=["logreg", "svm_rbf", "lightgbm"])
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    import lightgbm as lgb
    from .baselines import extract_features

    X, y, groups = load_uci(a.csv)
    F = extract_features(X)
    mk = {
        "logreg": lambda: Pipeline([("sc", StandardScaler()), ("c", LogisticRegression(
            C=1, max_iter=2000, class_weight="balanced"))]),
        "svm_rbf": lambda: Pipeline([("sc", StandardScaler()), ("c", SVC(
            C=10, gamma="scale", probability=True, class_weight="balanced"))]),
        "lightgbm": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
            num_leaves=31, class_weight="balanced", verbose=-1, n_jobs=1, random_state=a.seed),
    }[a.model]

    recs = np.unique(groups)
    if a.max_groups and a.max_groups < len(recs):       # same subset for every model (fixed seed)
        recs = np.sort(np.random.default_rng(12345).choice(recs, a.max_groups, replace=False))
    ys, ps, gs = [], [], []
    for i, r in enumerate(recs):
        te = np.where(groups == r)[0]; tr = np.where(groups != r)[0]
        assert_no_group_leakage(tr, te, groups)
        p = mk().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        ys.append(y[te]); ps.append(p); gs.append(np.full(len(te), r))
        if (i + 1) % 25 == 0: print(f"  {i+1}/{len(recs)} recordings", flush=True)
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    row = dict(model=a.model, protocol="loro_pooled", n_recordings=len(recs), **evaluate(y_all, p_all))
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({"group": np.concatenate(gs), "y": y_all, "p": p_all}).to_csv(
        f"results/loro_preds_{a.model}.csv", index=False)
    pd.DataFrame([row]).to_csv("results/loro_summary.csv", mode="a",
        header=not os.path.exists("results/loro_summary.csv"), index=False)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


if __name__ == "__main__":
    main()
