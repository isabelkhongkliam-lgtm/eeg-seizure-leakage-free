"""
Experiment grid runner.

    python -m src.run_experiments --csv data/uci.csv --stage reproduce
    python -m src.run_experiments --csv data/uci.csv --stage main
    python -m src.run_experiments --csv data/uci.csv --stage baselines
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from .data_uci import (SPLITTERS, assert_no_group_leakage, load_uci,
                       split_random, split_recording_wise)
from .metrics import evaluate
from .models import (VARIANTS, HybridSeizureNet, predict_proba, set_seed,
                     standardise, train_model, wavelet_denoise)

RESULTS = "results/runs_v2.csv"   # new file: runs.csv has shifted columns and invalid groups
COLUMNS = ["model", "variant", "split", "seed", "fold", "val_auprc", "train_seconds",
           "n_params", "best_params", "accuracy", "mcc", "f1", "balanced_acc", "auroc",
           "auprc", "sensitivity", "specificity", "precision", "sens_at_spec99",
           "fa_per_hour_at_spec", "fa_per_hour_at_05", "ece", "brier", "n_test", "n_pos"]


def _append(row: dict, path: str = RESULTS) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    missing = [k for k in row if k not in COLUMNS]
    assert not missing, f"unknown columns {missing}"
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in COLUMNS}])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _done(path: str = RESULTS) -> set:
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    keys = ("model", "variant", "split", "seed", "fold")
    if not all(k in df.columns for k in keys):
        return set()
    return set(map(tuple, df[list(keys)].values))


def run_deep(X, y, groups, split_name, variant, seed, epochs, val_frac=0.15,
             max_groups=None, resume=True):
    cfg = VARIANTS[variant]
    splitter = SPLITTERS[split_name]
    folds = (splitter(X, y, groups, max_groups=max_groups, seed=seed)
             if split_name == "logo_recording"
             else splitter(X, y, groups, seed=seed))
    already = _done() if resume else set()

    for k, (tr, te) in enumerate(folds):
        key = ("deep", variant, split_name, seed, k)
        if key in already:
            print(f"[skip] {key}")
            continue
        if split_name != "random_segment" and groups is not None:
            assert_no_group_leakage(tr, te, groups)

        set_seed(seed + 1000 * k)

        # carve a validation set out of TRAIN
        if groups is not None:
            gtr = groups[tr]
            uniq = np.unique(gtr)
            rng = np.random.default_rng(seed)
            val_groups = set(rng.choice(uniq, max(1, int(len(uniq) * val_frac)),
                                        replace=False))
            va_mask = np.isin(gtr, list(val_groups))
            tr_idx, va_idx = tr[~va_mask], tr[va_mask]
        else:
            rng = np.random.default_rng(seed)
            va_idx = rng.choice(tr, size=max(1, int(len(tr) * val_frac)), replace=False)
            tr_idx = np.setdiff1d(tr, va_idx)

        Xtr, Xva, Xte = X[tr_idx], X[va_idx], X[te]
        if cfg["use_wavelet"]:
            Xtr, Xva, Xte = (wavelet_denoise(Xtr), wavelet_denoise(Xva),
                             wavelet_denoise(Xte))
        Xtr, Xva, Xte = standardise(Xtr, Xva, Xte)

        t0 = time.time()
        model = HybridSeizureNet(**cfg)
        model, best_val = train_model(model, Xtr, y[tr_idx], Xva, y[va_idx],
                                      epochs=epochs)
        p = predict_proba(model, Xte)

        row = dict(model="deep", variant=variant, split=split_name, seed=seed,
                   fold=k, val_auprc=best_val, train_seconds=round(time.time() - t0, 1),
                   n_params=sum(q.numel() for q in model.parameters()),
                   **evaluate(y[te], p))
        _append(row)
        print(f"[deep] {variant:11s} {split_name:15s} seed={seed} fold={k} "
              f"acc={row['accuracy']:.4f} mcc={row['mcc']:.4f} "
              f"auprc={row['auprc']:.4f} fa/h={row['fa_per_hour_at_spec']:.2f}")


def run_baselines(X, y, groups, split_name, seed, resume=True):
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from .baselines import build_baselines, extract_features

    print("[feat] extracting classical features ...")
    F = extract_features(X)
    folds = SPLITTERS[split_name](X, y, groups, seed=seed)
    already = _done() if resume else set()

    for name, (est, grid) in build_baselines(seed=seed).items():
        for k, (tr, te) in enumerate(folds):
            key = ("classical", name, split_name, seed, k)
            if key in already:
                continue
            if split_name != "random_segment" and groups is not None:
                assert_no_group_leakage(tr, te, groups)

            if groups is not None:
                inner = GroupKFold(n_splits=3)
                gs = GridSearchCV(est, grid, scoring="average_precision",
                                  cv=inner.split(F[tr], y[tr], groups=groups[tr]),
                                  n_jobs=-1, refit=True)
            else:
                gs = GridSearchCV(est, grid, scoring="average_precision",
                                  cv=3, n_jobs=-1, refit=True)

            gs.fit(F[tr], y[tr])
            p = gs.predict_proba(F[te])[:, 1]

            row = dict(model="classical", variant=name, split=split_name,
                       seed=seed, fold=k, val_auprc=gs.best_score_,
                       best_params=json.dumps(gs.best_params_),
                       **evaluate(y[te], p))
            _append(row)
            print(f"[base] {name:13s} {split_name:15s} seed={seed} fold={k} "
                  f"acc={row['accuracy']:.4f} mcc={row['mcc']:.4f} "
                  f"auprc={row['auprc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to UCI CSV")
    ap.add_argument("--stage", default="main",
                    choices=["reproduce", "main", "baselines", "logo"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-groups", type=int, default=50,
                    help="LOGO folds to sample (500 is usually too many)")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    X, y, groups = load_uci(a.csv)
    resume = not a.no_resume

    if a.stage == "reproduce":
        for seed in a.seeds:
            run_deep(X, y, groups, "random_segment", "full", seed, a.epochs,
                     resume=resume)

    elif a.stage == "main":
        for split, variant, seed in itertools.product(
            ["random_segment", "recording_wise"], list(VARIANTS), a.seeds
        ):
            run_deep(X, y, groups, split, variant, seed, a.epochs, resume=resume)

    elif a.stage == "logo":
        for seed in a.seeds[:1]:
            run_deep(X, y, groups, "logo_recording", "full", seed, a.epochs,
                     max_groups=a.max_groups, resume=resume)

    elif a.stage == "baselines":
        for split, seed in itertools.product(
            ["random_segment", "recording_wise"], a.seeds
        ):
            run_baselines(X, y, groups, split, seed, resume=resume)

    print(f"\nresults -> {RESULTS}")


if __name__ == "__main__":
    main()
