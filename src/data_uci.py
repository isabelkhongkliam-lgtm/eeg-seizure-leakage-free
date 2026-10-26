from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, LeaveOneGroupOut

SFREQ = 173.61          # Hz, original Bonn sampling rate
SEGMENT_SAMPLES = 178   # samples per row
SEGMENT_SECONDS = SEGMENT_SAMPLES / SFREQ   # ~1.025 s


ID_COLUMNS = ("recording_id", "Unnamed", "Unnamed: 0", "id")


def parse_recording_ids(ids: pd.Series) -> np.ndarray:
    """'X21.V1.791' -> 'V1.791' (text after the chunk number).

    UCI IDs look like X<chunk>.V1.<n> (chunk 1..23) or X<chunk>.V<n>.
    Everything after the first '.' identifies the original 23.6 s recording.
    The file is NOT sorted by recording, so row position (index // 23) is NOT
    a valid grouping -- every such 'group' mixes ~22 different recordings.
    """
    ids = ids.astype(str)
    if not ids.str.fullmatch(r"X\d+\..+").all():
        bad = ids[~ids.str.fullmatch(r"X\d+\..+")].head(3).tolist()
        raise ValueError(f"unexpected ID format, e.g. {bad}")
    return ids.str.split(".", n=1).str[1].to_numpy()


def load_uci(csv_path: str, binary: bool = True):
    """Returns X (n, 178) float32, y (n,) int, groups (n,) int recording index.

    binary=True gives the standard task: class 1 (seizure) vs classes 2-5.
    Groups come from the ID column, with hard checks (500 recordings x 23).
    """
    df = pd.read_csv(csv_path)
    if "y" not in df.columns:
        raise ValueError(f"expected a 'y' label column, got {list(df.columns)[-5:]}")
    y_raw = df["y"].astype(int)

    feature_cols = [c for c in df.columns if re.fullmatch(r"X\d+", str(c))]
    if len(feature_cols) != SEGMENT_SAMPLES:
        raise ValueError(f"expected {SEGMENT_SAMPLES} feature columns X1..X178, "
                         f"found {len(feature_cols)}")

    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
    if id_col is None:
        raise ValueError("no recording-ID column found (need e.g. 'Unnamed' or "
                         "'recording_id' holding strings like X21.V1.791)")
    rec = parse_recording_ids(df[id_col])
    groups = pd.factorize(rec)[0]

    # hard checks: this is the guarantee the whole paper rests on
    sizes = np.bincount(groups)
    assert len(sizes) == 500 and (sizes == 23).all(), \
        f"expected 500 recordings x 23 segments, got {len(sizes)} / {np.unique(sizes)}"
    assert (pd.Series(y_raw.values).groupby(groups).nunique() == 1).all(), \
        "a recording contains more than one class label"

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = (y_raw == 1).astype(int).to_numpy() if binary else y_raw.to_numpy()
    print(f"[data] X={X.shape}  positives={y.sum()}/{len(y)} "
          f"({y.mean():.1%})  recordings={len(sizes)} (parsed from '{id_col}')")
    return X, y, groups


# ---------------------------------------------------------------- splitters

def split_random(X, y, groups, n_splits=5, seed=0):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


def split_recording_wise(X, y, groups, n_splits=5, seed=0):
    cv = GroupKFold(n_splits=n_splits)
    return list(cv.split(X, y, groups=groups))


def split_leave_one_group_out(X, y, groups, max_groups=None, seed=0):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    if max_groups is not None and max_groups < len(folds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(folds), size=max_groups, replace=False)
        folds = [folds[i] for i in sorted(idx)]
    return folds


SPLITTERS = {
    "random_segment": split_random,
    "recording_wise": split_recording_wise,
    "logo_recording": split_leave_one_group_out,
}


def assert_no_group_leakage(train_idx, test_idx, groups) -> None:
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        raise AssertionError(
            f"GROUP LEAKAGE: {len(overlap)} recordings in both train and test, "
            f"e.g. {sorted(overlap)[:3]}"
        )
