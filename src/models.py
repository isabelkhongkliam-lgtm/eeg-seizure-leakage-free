"""
Hybrid wavelet + CNN + multi-head attention model, with every component the
original paper claims is important behind an on/off flag.

Ablation variants are therefore one keyword argument each:
    full        = Model(use_wavelet=True,  use_attention=True,  use_skip=True)
    no_attn     = Model(use_wavelet=True,  use_attention=False, use_skip=True)
    no_wavelet  = Model(use_wavelet=False, use_attention=True,  use_skip=True)
    no_skip     = Model(use_wavelet=True,  use_attention=True,  use_skip=False)

Wavelet denoising is a PREPROCESSING step (numpy/pywt), not a layer, so it is
applied in+988/.............0///////// the data pipeline and the flag here only records what was done.
Keeping it out of the graph avoids a per-batch CPU round-trip.
"""

from __future__ import annotations

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------- wavelet denoising

def wavelet_denoise(X: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """
    Universal-threshold soft denoising, per segment.

    Threshold uses the robust MAD estimate of noise sigma from the finest
    detail band -- the standard Donoho-Johnstone choice. Report the wavelet,
    level, and threshold rule in the paper; these are free parameters that
    quietly matter and are usually left unstated.
    """
    out = np.empty_like(X, dtype=np.float32)
    n = X.shape[1]
    for i, sig in enumerate(X):
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-12
        thr = sigma * np.sqrt(2.0 * np.log(n))
        coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, wavelet)
        out[i] = rec[:n]
    return out


def standardise(X_train: np.ndarray, *others: np.ndarray):
    """Per-channel z-scoring with statistics fitted on TRAIN ONLY.

    Fitting the scaler on the full dataset is a second, subtler leak that
    several published EEG pipelines contain. Do not move this."""
    mu = X_train.mean()
    sd = X_train.std() + 1e-8
    return ((X_train - mu) / sd, *[(o - mu) / sd for o in others])


# ------------------------------------------------------------------ model

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=7, pool=2, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2)
        self.bn = nn.BatchNorm1d(c_out)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


class HybridSeizureNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        channels=(32, 64, 128),
        n_heads: int = 4,
        dropout: float = 0.3,
        use_wavelet: bool = True,     # recorded for provenance; applied upstream
        use_attention: bool = True,
        use_skip: bool = True,
    ):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_attention = use_attention
        self.use_skip = use_skip

        blocks, c_in = [], 1
        for c_out in channels:
            blocks.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        d = c_in

        if use_attention:
            self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                              batch_first=True)
            self.norm = nn.LayerNorm(d)

        self.head = nn.Sequential(
            nn.Linear(d * 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):                    # x: (B, 1, T)
        h = self.conv(x)                     # (B, D, T')
        h = h.transpose(1, 2)                # (B, T', D)

        if self.use_attention:
            a, _ = self.attn(h, h, h, need_weights=False)
            h = self.norm(a + h) if self.use_skip else self.norm(a)

        # concat mean+max pooling: cheap and consistently better than either
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(pooled)


VARIANTS = {
    "full":       dict(use_wavelet=True,  use_attention=True,  use_skip=True),
    "no_attn":    dict(use_wavelet=True,  use_attention=False, use_skip=True),
    "no_wavelet": dict(use_wavelet=False, use_attention=True,  use_skip=True),
    "no_skip":    dict(use_wavelet=True,  use_attention=True,  use_skip=False),
}


# --------------------------------------------------------------- training

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model, Xtr, ytr, Xva, yva,
    epochs=60, batch_size=128, lr=1e-3, weight_decay=1e-4,
    patience=10, device=None, verbose=False,
):
    """Trains with early stopping on validation AUPRC.

    AUPRC not accuracy for model selection: under imbalance, accuracy-based
    early stopping happily selects a model that predicts all-negative.
    """
    from sklearn.metrics import average_precision_score

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xtr).unsqueeze(1), torch.from_numpy(ytr).long()
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                     drop_last=False)

    # class weights: the honest splits leave the minority class genuinely rare
    counts = np.bincount(ytr, minlength=2).astype(np.float32)
    w = torch.tensor((counts.sum() / (2 * np.maximum(counts, 1))), device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        p = predict_proba(model, Xva, device=device)
        score = average_precision_score(yva, p) if yva.sum() > 0 else 0.0
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"  ep{ep:03d} val_auprc={score:.4f} best={best:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict_proba(model, X, batch_size=512, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).unsqueeze(1).to(device)
        out.append(F.softmax(model(xb), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)
