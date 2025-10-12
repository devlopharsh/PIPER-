from __future__ import annotations
from typing import Callable, Literal, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, average_precision_score, fbeta_score
from scipy.special import expit

from utils.constants import (
    METER_ID_COL, TIMESTAMP_COL, LEAK_FLAG_COL, DEVICE
)
from utils.DataAggregation import scale_on_train

# --- tabular helper ---
def to_xy(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LEAK_FLAG_COL].to_numpy(dtype=np.float32)  # BCEWithLogits expects float {0,1}
    return X, y
# ----------------------------- Common dataset (sequence) -----------------------------
class SeqDatasetCls(Dataset):
    """
    Sequence dataset for classification (per-hour labels).
    Builds sliding windows [T,F] and label sequences [T].
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], window: int = 24, stride: int = 1):
        self.X, self.y = self._build_sequences(df, feature_cols, window, stride)

    @staticmethod
    def _build_sequences(df: pd.DataFrame, feature_cols: List[str], window: int, stride: int):
        Xs, ys = [], []
        for _, g in df.sort_values([METER_ID_COL, TIMESTAMP_COL]).groupby(METER_ID_COL, sort=False):
            g = g.reset_index(drop=True)
            arr = g[feature_cols].to_numpy(dtype=np.float32)    # [N, F]
            labels = g[LEAK_FLAG_COL].to_numpy(dtype=np.int64)  # [N]
            n = len(g)
            for start in range(0, n - window + 1, stride):
                end = start + window
                Xs.append(arr[start:end])      # [T, F]
                ys.append(labels[start:end])   # [T]
        if not Xs:
            return (
                np.zeros((0, window, len(feature_cols)), np.float32),
                np.zeros((0, window), np.int64),
            )
        return np.stack(Xs), np.stack(ys)

    def __len__(self): 
        return len(self.y)

    def __getitem__(self, i): 
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])


# ----------------------------- Folds & threshold helpers -----------------------------
def _make_group_forward_time_folds(df_in: pd.DataFrame, n_folds: int = 10):
    """
    Forward-chaining folds per meter (χωρίς data leakage).
    """
    df_in = df_in.sort_values([METER_ID_COL, TIMESTAMP_COL]).copy()
    folds = [( [], [] ) for _ in range(n_folds)]
    for _, g in df_in.groupby(METER_ID_COL, sort=False):
        n = len(g)
        if n < n_folds + 1:
            continue
        cuts = [int(np.floor((k+1)/n_folds * n)) for k in range(n_folds)]
        cuts[-1] = n
        for k in range(n_folds):
            cut_k  = cuts[k]
            next_k = cuts[k+1] if k+1 < n_folds else n
            tr = g.iloc[:cut_k].index.to_list()
            va = g.iloc[cut_k:next_k].index.to_list()
            if va:
                folds[k][0].extend(tr)
                folds[k][1].extend(va)

    out = []
    for tr, va in folds:
        tr = np.unique(np.asarray(tr, int))
        va = np.unique(np.asarray(va, int))
        va = va[~np.isin(va, tr)]
        if len(tr) and len(va):
            out.append((tr, va))
    return out


def _choose_threshold_fbeta(y_true: np.ndarray, probs: np.ndarray, beta: float = 1.0):
    """
    Επιλογή threshold που μεγιστοποιεί F_beta πάνω στην PR καμπύλη.
    """
    p, r, thr = precision_recall_curve(y_true, probs)
    thr = np.append(thr, 1.0)  # align με p/r μήκη
    f = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    i = int(np.nanargmax(f))
    return {
        "thr": float(thr[i]),
        "fbeta": float(f[i]),
        "prec": float(p[i]),
        "rec": float(r[i]),
        "ap": float(average_precision_score(y_true, probs)),
    }


# ----------------------------- Public API -----------------------------
def get_cv_threshold_on_train(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    *,
    # CV
    k_folds: int = 10,
    # Mode: 'seq' for RNNs (LSTM/GRU), 'tab' for MLP-like
    mode: Literal["seq", "tab"] = "seq",
    window: int = 24, stride: int = 1,
    # Model factory: given in_dim -> returns nn.Module with logits output [B,T] or [B]
    make_model: Callable[[int], nn.Module],
    # Optimization
    batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
    max_epochs: int = 5, early_patience: int = 2,
    # Others
    seed: int = 42, beta_for_thr: float = 1.0,
) -> tuple[float, List[float]]:
    """
    Forward-chaining K-fold CV ΜΟΝΟ επάνω στο training split για robust threshold.

    Επιστρέφει: (median_threshold, [per_fold_thresholds])
    """
    torch.manual_seed(seed); np.random.seed(seed)

    folds = _make_group_forward_time_folds(df_train, n_folds=k_folds)
    thr_list: List[float] = []

    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        tr = df_train.loc[tr_idx].sort_values([METER_ID_COL, TIMESTAMP_COL])
        va = df_train.loc[va_idx].sort_values([METER_ID_COL, TIMESTAMP_COL])

        # Scale στον fold-train μόνο
        tr_scaled, _, scaler_f = scale_on_train(tr, tr, feature_cols)
        va_scaled = va.copy()
        va_scaled[feature_cols] = scaler_f.transform(va_scaled[feature_cols])

        # Datasets / loaders
        if mode == "seq":
            tr_ds = SeqDatasetCls(tr_scaled, feature_cols, window=window, stride=stride)
            va_ds = SeqDatasetCls(va_scaled, feature_cols, window=window, stride=stride)
            if len(tr_ds) == 0 or len(va_ds) == 0:
                continue
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
            va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            y_seq = tr_ds.y.ravel()
            pos_w = (len(y_seq) - y_seq.sum()) / (y_seq.sum() + 1e-6)
        else:  # "tab"
            Xtr, ytr = to_xy(tr_scaled, feature_cols)
            Xva, yva = to_xy(va_scaled, feature_cols)
            tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                                   batch_size=batch_size, shuffle=True, num_workers=0)
            va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                                   batch_size=batch_size, shuffle=False, num_workers=0)
            pos_w = (len(ytr) - ytr.sum()) / (ytr.sum() + 1e-6)

        # Model/loss/optim
        model_f = make_model(len(feature_cols)).to(DEVICE)
        opt_f = torch.optim.Adam(model_f.parameters(), lr=lr, weight_decay=weight_decay)
        crit_f = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=DEVICE)
        )

        # Early stop πάνω στο fold-val F_beta του threshold που προκύπτει κάθε epoch
        bestF, patience, best_state = -1.0, 0, None
        for _ in range(max_epochs):
            model_f.train()
            for xb, yb in tr_loader:
                xb = xb.to(DEVICE, dtype=torch.float32)
                yb = yb.to(DEVICE, dtype=torch.float32)
                loss = crit_f(model_f(xb), yb)
                opt_f.zero_grad(); loss.backward(); opt_f.step()

            # fold-val: provisional threshold
            model_f.eval()
            with torch.no_grad():
                vs, yt = [], []
                for xb, yb in va_loader:
                    vs.append(model_f(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())  # [B,T]
                    yt.append(yb.numpy())                                                # [B,T]
            yt_bt = np.concatenate(yt, axis=0)                          # [Nw, T]
            vp_bt = expit(np.clip(np.concatenate(vs, axis=0), -20, 20)) # [Nw, T]
            yt_1h = yt_bt[:, -1].ravel()                                 # last step
            vp_1h = vp_bt[:, -1].ravel()

            info = _choose_threshold_fbeta(yt_1h, vp_1h, beta=beta_for_thr)
            vpred = (vp_1h >= info["thr"]).astype(int)
            vF = fbeta_score(yt_1h, vpred, beta=beta_for_thr, average="binary", zero_division=0)

            if vF > bestF + 1e-4:
                bestF, patience = vF, 0
                best_state = {k: v.clone() for k, v in model_f.state_dict().items()}
            else:
                patience += 1
                if patience >= early_patience:
                    break

        # Evaluate στο καλύτερο state για να κρατήσουμε το fold-threshold (last-step)
        if best_state is not None:
            model_f.load_state_dict(best_state)
        model_f.eval()
        with torch.no_grad():
            vs, yt = [], []
            for xb, yb in va_loader:
                vs.append(model_f(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                yt.append(yb.numpy())
        yt_bt = np.concatenate(yt, axis=0)
        vp_bt = expit(np.clip(np.concatenate(vs, axis=0), -20, 20))
        yt_1h = yt_bt[:, -1].ravel()
        vp_1h = vp_bt[:, -1].ravel()
        info = _choose_threshold_fbeta(yt_1h, vp_1h, beta=beta_for_thr)

        thr_list.append(float(info["thr"]))
        print(f"[CV fold {k:02d}] thr={info['thr']:.3f}  Fβ={info['fbeta']:.4f}  P={info['prec']:.4f}  R={info['rec']:.4f}")

    if not thr_list:
        return 0.5, []

    robust_thr = float(np.median(thr_list))
    print(f"[CV] thresholds={np.round(thr_list,3).tolist()} → robust_thr={robust_thr:.3f}")
    return robust_thr, thr_list
