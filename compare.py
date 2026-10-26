import numpy as np, pandas as pd
df = pd.read_csv("results/runs_v2.csv")
M = ["accuracy", "mcc", "auprc"]
CLASSICAL = ["lightgbm", "svm_rbf", "random_forest", "logreg"]
for split in ["recording_wise", "random_segment"]:
    d = df[df.split == split]
    f = d.groupby(["variant", "fold"])[M].mean()
    have = set(f.index.get_level_values(0))
    ref = max([v for v in CLASSICAL if v in have], key=lambda v: f.loc[v, "mcc"].mean())
    print(f"\n=== {split}: reference = {ref} (best classical by mean MCC) ===")
    for v in sorted(have):
        if v == ref:
            continue
        a, b = f.loc[v], f.loc[ref]
        row = [f"{v:13s}"]
        for m in M:
            diff = (a[m] - b[m]).values
            row.append(f"{m}: {diff.mean():+.4f} (sd {diff.std(ddof=1):.4f}, wins {int((diff > 0).sum())}/{len(diff)})")
        print("  ".join(row))
print("\nwins = folds where the model beats the reference.")