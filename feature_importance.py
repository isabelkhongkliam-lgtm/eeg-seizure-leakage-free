"""
Feature importance that can actually be interpreted.

Fixes vs. the original script:
  * binary seizure-vs-rest label (the original used the 5-class y -> 0.70 acc)
  * TRUE recording groups parsed from the ID (original used index // 23)
  * importances tied to NAMED engineered features, via permutation importance
    on held-out recordings (impurity importance over 178 raw sample positions
    is flat and not interpretable as frequency content)
    python feature_importance.py data/uci_full.csv
"""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
import lightgbm as lgb
from src.data_uci import load_uci
from src.baselines import extract_features

NAMES = (["mean", "std", "skew", "kurtosis", "peak_to_peak", "rms", "mean_abs_diff",
          "zero_crossings", "p5", "p95", "hjorth_mobility", "hjorth_complexity", "sample_entropy"]
         + [f"abs_{b}" for b in ("delta", "theta", "alpha", "beta", "gamma")]
         + [f"rel_{b}" for b in ("delta", "theta", "alpha", "beta", "gamma")]
         + ["beta/theta", "gamma/beta", "spectral_entropy", "sef50", "sef95", "peak_freq"]
         + [f"wav_energy_{i}" for i in range(5)] + [f"wav_rel_{i}" for i in range(5)])

X, y, groups = load_uci(sys.argv[1])
F = extract_features(X)
assert F.shape[1] == len(NAMES), (F.shape, len(NAMES))
imps = []
for tr, te in GroupKFold(5).split(F, y, groups):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                           verbose=-1, n_jobs=1, random_state=0).fit(F[tr], y[tr])
    r = permutation_importance(m, F[te], y[te], scoring="average_precision",
                               n_repeats=5, random_state=0, n_jobs=1)
    imps.append(r.importances_mean)
imps = np.array(imps)
out = pd.DataFrame({"feature": NAMES, "mean_drop_in_auprc": imps.mean(0),
                    "sd_across_folds": imps.std(0)}).sort_values("mean_drop_in_auprc", ascending=False)
out.to_csv("results/feature_importance_named.csv", index=False)
print(out.head(15).round(4).to_string(index=False))
