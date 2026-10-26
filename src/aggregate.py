"""
Turns results/runs.csv into the paper's tables.

    python -m src.aggregate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compare_across_seeds, holm_bonferroni

LEAD = ["mcc", "auprc", "sens_at_spec99", "fa_per_hour_at_spec", "accuracy"]


def table_main(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: performance by split protocol and variant, mean +/- sd over seeds."""
    per_seed = (df.groupby(["model", "variant", "split", "seed"])[LEAD]
                  .mean().reset_index())
    agg = (per_seed.groupby(["model", "variant", "split"])[LEAD]
                   .agg(["mean", "std"]).round(4))
    return agg


def table_leakage_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2: the headline. What each metric loses when grouping is respected."""
    per_seed = (df.groupby(["model", "variant", "split", "seed"])[LEAD]
                  .mean().reset_index())
    rows = []
    for (model, variant), g in per_seed.groupby(["model", "variant"]):
        rand = g[g.split == "random_segment"]
        grp = g[g.split == "recording_wise"]
        if rand.empty or grp.empty:
            continue
        for m in LEAD:
            c = compare_across_seeds(rand[m].values, grp[m].values)
            rows.append(dict(model=model, variant=variant, metric=m,
                             random_segment=f"{c['mean_a']:.4f}+/-{c['sd_a']:.4f}",
                             recording_wise=f"{c['mean_b']:.4f}+/-{c['sd_b']:.4f}",
                             drop=round(c["delta"], 4),
                             p_raw=c["p_value"],
                             cliffs_delta=round(c["cliffs_delta"], 3)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm_bonferroni(out.p_raw.values).round(5)
    return out


def table_ablation(df: pd.DataFrame, split="recording_wise") -> pd.DataFrame:
    """Table 3: each component against the full model, under the HONEST split.

    Ablating under the leaky split is not informative -- if leakage dominates,
    every variant scores ~1.0 and the components look equally irrelevant."""
    d = df[(df.model == "deep") & (df.split == split)]
    per_seed = d.groupby(["variant", "seed"])[LEAD].mean().reset_index()
    full = per_seed[per_seed.variant == "full"]
    rows = []
    for variant, g in per_seed.groupby("variant"):
        if variant == "full":
            continue
        for m in LEAD:
            c = compare_across_seeds(full[m].values, g[m].values)
            rows.append(dict(ablated=variant, metric=m,
                             full=f"{c['mean_a']:.4f}+/-{c['sd_a']:.4f}",
                             ablated_score=f"{c['mean_b']:.4f}+/-{c['sd_b']:.4f}",
                             delta=round(c["delta"], 4),
                             p_raw=c["p_value"],
                             cliffs_delta=round(c["cliffs_delta"], 3)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm_bonferroni(out.p_raw.values).round(5)
    return out


def main():
    df = pd.read_csv("results/runs.csv")
    print(f"loaded {len(df)} fold-level runs\n")

    for name, t in [("TABLE 1  performance by protocol", table_main(df)),
                    ("TABLE 2  leakage effect", table_leakage_effect(df)),
                    ("TABLE 3  ablation (honest split)", table_ablation(df))]:
        print("=" * 78)
        print(name)
        print("=" * 78)
        print(t.to_string() if len(t) else "  (insufficient runs yet)")
        print()

    with pd.ExcelWriter("results/tables.xlsx") as w:
        table_main(df).to_excel(w, sheet_name="performance")
        table_leakage_effect(df).to_excel(w, sheet_name="leakage", index=False)
        table_ablation(df).to_excel(w, sheet_name="ablation", index=False)
    print("wrote results/tables.xlsx")


if __name__ == "__main__":
    main()
