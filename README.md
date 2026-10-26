# Leakage-free evaluation of classical vs. deep models for EEG seizure detection

Code for the paper **"Does the Deep Learning Advantage Survive Recording-Independent Evaluation? Classical and Deep Models for EEG Seizure Detection"**.

We compare logistic regression, SVM-RBF, random forest and LightGBM (39 hand-crafted features) with HybridSeizureNet (a wavelet-denoised 1D CNN, with and without multi-head attention) on the UCI Epileptic Seizure Recognition dataset, under three protocols: random segment-wise 5-fold, recording-wise 5-fold, and leave-one-recording-out (LORO, classical models, pooled metrics).

## Main findings
- Under all protocols a tuned LightGBM matches HybridSeizureNet within fold-to-fold variation (recording-wise: LightGBM 98.54% / MCC 0.955; HybridSeizureNet Full 98.20% / MCC 0.944).
- Random segment-wise splitting inflates accuracy by about 0.3-1.2 points, slightly more for the deep models.
- Attention gives no consistent benefit.
- Permutation importance ranks Hjorth complexity, sample entropy and amplitude features highest.

Limits: the Bonn data come from few individuals, so evaluation is recording-independent, not subject-independent. Single channel, one-second segments, 20% seizure rate. No deep LORO was run.

## Important: recording IDs
The UCI file is not sorted by recording. The recording is the text after the chunk number in the ID (`X21.V1.791` -> `V1.791`), giving 500 recordings x 23 segments. Never group by row position (`index // 23`): that mixes ~22 recordings per group and leaves the split leaky. The loader (`src/data_uci.py`) asserts the correct structure.

## Setup
```
pip install torch lightgbm PyWavelets scikit-learn pandas scipy matplotlib openpyxl
```
Data: UCI Machine Learning Repository, "Epileptic Seizure Recognition" (derived from Andrzejak et al., Phys. Rev. E 64, 061907, 2001). Put the CSV at `data/uci_full.csv` (first column = ID string, columns X1..X178, label `y`).

## Reproduce
```
python -c "from src.data_uci import load_uci; load_uci('data/uci_full.csv')"
python -m src.run_experiments --csv data/uci_full.csv --stage baselines --seeds 0 1 2
python -c "from src.run_experiments import *; X,y,g=load_uci('data/uci_full.csv'); [run_deep(X,y,g,sp,v,s,60) for sp in ['recording_wise','random_segment'] for s in [0,1,2] for v in ['full','no_attn']]"
python -m src.run_loro --csv data/uci_full.csv --model logreg
python -m src.run_loro --csv data/uci_full.csv --model lightgbm
python -m src.run_loro --csv data/uci_full.csv --model svm_rbf --max-groups 100
python feature_importance.py data/uci_full.csv
python compare.py
python make_report.py
```
The deep runs take several hours on a laptop CPU; they resume if interrupted.

## Files
| Path | Purpose |
|---|---|
| `src/data_uci.py` | Loader, recording-ID parsing, split functions, leakage assertions |
| `src/baselines.py` | 39 features and classical models with grouped inner CV |
| `src/models.py` | HybridSeizureNet (Full / No-Attention) |
| `src/run_experiments.py` | Baseline and deep experiment runner |
| `src/run_loro.py` | Pooled leave-one-recording-out |
| `compare.py` | Paired fold-level comparison against the best classical model |
| `feature_importance.py` | Permutation importance on named features |
| `make_report.py` | Tables and figures |
| `results/` | `runs_v2.csv`, `loro_summary.csv`, `feature_importance_named.csv` |

## Citation
[Add the paper citation / DOI when available.]

## License
Code: MIT (see LICENSE). The dataset has its own terms; see the UCI repository page.
