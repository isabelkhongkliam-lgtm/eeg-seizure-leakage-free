# What was wrong, and how to rerun

## Findings from the uploaded project
1. **Grouping was wrong.** `load_uci` used `df.index // 23`. The CSV is shuffled, so every such "group"
   mixes ~22 different recordings (checked: 500/500 groups impure; ~496 of 500 test recordings also in train
   for every "recording-wise" fold). Your logged "recording_wise" results equal the random ones
   (deep full: acc 0.9937 vs 0.9943) -- no leakage was removed. `data_final_with_group.csv` and
   `uci_full_grouped.csv` carry the same wrong grouping. The true recording key is the text after the
   chunk number in the ID (`X21.V1.791` -> `V1.791`): 500 recordings x 23 segments, label-pure.
2. **Preview with true groups (classical, 5-fold, fixed params, 1 seed):**
   logreg  acc 0.977 -> 0.964, MCC 0.931 -> 0.892;  lightgbm acc 0.991 -> 0.971, MCC 0.973 -> 0.908.
   (Leakage inflates; the drop is real.)
3. **No LORO was ever run** (`runs.csv` has 0 logo rows). Deep ablations other than `full` ran only on the
   random split; deep `full` recording-wise has 2 seeds.
4. **`runs.csv` is corrupted:** classical rows have 23 fields vs a 24-field header, so pandas shifts every
   metric (e.g. "mcc" > "accuracy"). New code writes a fixed schema to `results/runs_v2.csv`.
5. **FA/h of 36.26 is an artefact.** `fa_per_hour_at_spec` fixes specificity at 99%, so it is ~36 for every
   model by construction. Report `fa_per_hour_at_05` or sensitivity at fixed specificity instead.
6. **The paper does not match the code.** Paper tables (e.g. classical 91-94% acc) match nothing in runs.csv.
   Code uses wavelet denoising (not in paper), BatchNorm+MaxPool, AdamW + class-weighted CE + cosine LR
   (paper says Adam + BCE), gamma 30-70 Hz (paper 30-45), grid search for LightGBM (paper: random search),
   a 4th classical model (random forest), 5 seeds x 5 folds. Rewrite the Methods from the code.
7. **Feature importance:** root `top20_features.csv` is the first DATA row read as header (values 48, 9.2,
   -12 ... are amplitudes, not feature indices). `results/top20_features.csv` is the 5-class problem
   (acc 0.70) on index//23 groups with flat importances (~0.008). Replaced by `feature_importance.py`.

## Rerun (from project root, copy these files over yours)
    python -m src.run_experiments --csv data/uci_full.csv --stage baselines
    python -m src.run_experiments --csv data/uci_full.csv --stage main          # deep, random + recording-wise
    python -m src.run_loro --csv data/uci_full.csv --model lightgbm             # pooled LORO (classical)
    python -m src.run_loro --csv data/uci_full.csv --model logreg
    python -m src.run_loro --csv data/uci_full.csv --model svm_rbf --max-groups 100
    python feature_importance.py data/uci_full.csv
    python make_report.py
Deep LORO (500 trainings) is impractical; use a 50-100 recording subset (same subset for all models) or
report grouped 10-fold instead, and state that in the paper. Delete the old results/runs.csv.
Untested here: the deep training path (no torch in my sandbox); only its data loading/CSV schema changed.
Still true after the fix: Bonn recordings come from few people, so this is recording- not subject-independent.
