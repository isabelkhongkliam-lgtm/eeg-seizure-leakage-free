"""
Classical feature baselines -- the comparison that decides whether the deep
model's complexity is earning anything.

Tune these PROPERLY. A deliberately weak baseline is the most common way
"deep learning wins" papers mislead, and a reviewer who suspects you sandbagged
the baseline will not believe the rest of the paper either.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy import signal, stats

SFREQ = 173.61
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}   # Bonn data are band-limited to ~0.5-40 Hz


def _spectral_entropy(psd: np.ndarray) -> float:
    p = psd / (psd.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def _hjorth(sig: np.ndarray):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    v0 = sig.var() + 1e-12
    v1 = d1.var() + 1e-12
    v2 = d2.var() + 1e-12
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return mobility, complexity


def _sample_entropy(sig: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """Sample entropy. O(n^2) but n=178, so it is fine."""
    n = len(sig)
    r = r_mult * (sig.std() + 1e-12)

    def _phi(mm):
        emb = np.lib.stride_tricks.sliding_window_view(sig, mm)
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(axis=-1)
        np.fill_diagonal(d, np.inf)
        return (d <= r).sum()

    a, b = _phi(m + 1), _phi(m)
    return float(-np.log((a + 1e-12) / (b + 1e-12)))


def extract_features(X: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """
    ~40 features per segment: time-domain statistics, Hjorth parameters,
    band powers and ratios, spectral entropy and edge frequencies, and
    wavelet sub-band energies. This is a standard, defensible EEG feature set.
    """
    feats = []
    for sig in X:
        f = []

        # time domain
        f += [sig.mean(), sig.std(), stats.skew(sig), stats.kurtosis(sig),
              np.ptp(sig), np.sqrt((sig ** 2).mean()),
              np.abs(np.diff(sig)).mean(),
              float(((sig[:-1] * sig[1:]) < 0).sum()),           # zero crossings
              np.percentile(sig, 5), np.percentile(sig, 95)]
        f += list(_hjorth(sig))
        f.append(_sample_entropy(sig))

        # frequency domain
        nper = min(128, len(sig))
        fr, psd = signal.welch(sig, fs=sfreq, nperseg=nper)
        total = psd.sum() + 1e-12
        bp = []
        for lo, hi in BANDS.values():
            m = (fr >= lo) & (fr < hi)
            bp.append(psd[m].sum())
        f += bp
        f += [b / total for b in bp]                              # relative power
        f += [bp[3] / (bp[1] + 1e-12), bp[4] / (bp[3] + 1e-12)]   # beta/theta, gamma/beta
        f.append(_spectral_entropy(psd))
        csum = np.cumsum(psd) / total
        f += [fr[np.searchsorted(csum, q)] for q in (0.5, 0.95)]  # spectral edges
        f.append(fr[np.argmax(psd)])                              # peak frequency

        # wavelet sub-band energies
        coeffs = pywt.wavedec(sig, "db4", level=4)
        energies = np.array([float((c ** 2).sum()) for c in coeffs])
        f += list(energies)
        f += list(energies / (energies.sum() + 1e-12))

        feats.append(f)

    out = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_baselines(seed: int = 0, n_jobs: int = -1):
    """
    Returns {name: (estimator, param_grid)} for tuned comparison.

    Grids are small on purpose so this stays runnable; widen them if you have
    the compute, and report the search budget in the paper so the comparison
    against the deep model is transparently fair.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import lightgbm as lgb

    return {
        "logreg": (
            Pipeline([("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000,
                                                 class_weight="balanced"))]),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "svm_rbf": (
            Pipeline([("sc", StandardScaler()),
                      ("clf", SVC(kernel="rbf", probability=True,
                                  class_weight="balanced", random_state=seed))]),
            {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                   random_state=seed, n_jobs=n_jobs),
            {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 5]},
        ),
        "lightgbm": (
            lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05,
                               class_weight="balanced", random_state=seed,
                               n_jobs=n_jobs, verbose=-1),
            {"num_leaves": [15, 31, 63], "min_child_samples": [10, 30]},
        ),
    }
