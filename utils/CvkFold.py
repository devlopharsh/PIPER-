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


# ----------------------------- Tabular helper -----------------------------
def to_xy(df: pd.DataFrame, feature_cols: List[str]):
    """
    Convert a dataframe into (X, y) numpy arrays for tabular models.

    Returns:
        X: float32 array of shape [N, F]
        y: float32 array of shape [N] with values in {0, 1}
           (BCEWithLogitsLoss expects float targets)
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LEAK_FLAG_COL].to_numpy(dtype=np.float32)
    return X, y


# ----------------------------- Sequence dataset -----------------------------
class SeqDatasetCls(Dataset):
    """
    Sequence dataset for per-hour classification.

    It builds sliding windows of length `window` over each meter's time
    series and returns:
      - X[i] : windowed features  [T, F]
      - y[i] : windowed labels    [T]

    Notes:
      - Windows slide within each meter only (no cross-meter mixing).
      - Labels are integer {0,1}; during training we cast them to float.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], window: int = 24, stride: int = 1):
        self.X, self.y = self._build_sequences(df, feature_cols, window, stride)

    @staticmethod
    def _build_sequences(df: pd.DataFrame, feature_cols: List[str], window: int, stride: int):
        Xs, ys = [], []
        # Sort by (meter, time) to enforce causal order before windowing
        for _, g in df.sort_values([METER_ID_COL, TIMESTAMP_COL]).groupby(METER_ID_COL, sort=False):
            g = g.reset_index(drop=True)
            arr = g[feature_cols].to_numpy(dtype=np.float32)     # [N, F]
            labels = g[LEAK_FLAG_COL].to_numpy(dtype=np.int64)   # [N]
            n = len(g)
            # Create [start, start+window) slices with step = stride
            for start in range(0, n - window + 1, stride):
                end = start + window
                Xs.append(arr[start:end])     # [T, F]
                ys.append(labels[start:end])  # [T]
        if not Xs:
            # Keep shapes consistent even when no windows are formed
            return (
                np.zeros((0, window, len(feature_cols)), dtype=np.float32),
                np.zeros((0, window), dtype=np.int64),
            )
        return np.stack(Xs), np.stack(ys)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        # Return tensors; the training loop will cast target to float32
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])


# ----------------------------- Folds & threshold helpers -----------------------------
def _make_group_forward_time_folds(df_in: pd.DataFrame, n_folds: int = 10):
    """
    Build forward-chaining folds per meter to avoid temporal leakage.

    For each meter's time series:
      - Split its index into `n_folds` contiguous chunks in time order.
      - For fold k: train = everything up to split k, valid = the next chunk.
      - Then aggregate (extend) indices across meters for each fold.

    Returns:
        List of (train_idx, valid_idx) tuples with global dataframe indices.
        Each (train_idx, valid_idx) covers *all meters* for that fold.
    """
    df_in = df_in.sort_values([METER_ID_COL, TIMESTAMP_COL]).copy()
    folds = [([], []) for _ in range(n_folds)]

    for _, g in df_in.groupby(METER_ID_COL, sort=False):
        n = len(g)
        # Require minimal length so we can create at least 1 train and 1 valid slice
        if n < n_folds + 1:
            continue

        # 'cuts' are the right-end boundaries for each fold chunk
        cuts = [int(np.floor((k + 1) / n_folds * n)) for k in range(n_folds)]
        cuts[-1] = n  # ensure last cut lands exactly at the end

        for k in range(n_folds):
            cut_k = cuts[k]
            next_k = cuts[k + 1] if (k + 1) < n_folds else n

            tr_idx = g.iloc[:cut_k].index.to_list()       # everything up to the cut
            va_idx = g.iloc[cut_k:next_k].index.to_list() # the next contiguous chunk

            if va_idx:
                folds[k][0].extend(tr_idx)
                folds[k][1].extend(va_idx)

    # Deduplicate, remove overlaps, and keep only non-empty splits
    out = []
    for tr, va in folds:
        tr = np.unique(np.asarray(tr, dtype=int))
        va = np.unique(np.asarray(va, dtype=int))
        # Ensure no leakage from train into valid
        va = va[~np.isin(va, tr)]
        if len(tr) and len(va):
            out.append((tr, va))
    return out


def _choose_threshold_fbeta(y_true: np.ndarray, probs: np.ndarray, beta: float = 1.0):
    """
    Choose the probability threshold that maximizes F_beta on the PR curve.

    We work on (precision, recall, thresholds) from precision_recall_curve,
    align lengths, compute F_beta for each operating point, and keep the best.

    Returns:
        dict with 'thr', 'fbeta', 'prec', 'rec', and 'ap' (Average Precision).
    """
    p, r, thr = precision_recall_curve(y_true, probs)
    # precision_recall_curve returns len(thr) = len(p) - 1; append 1.0 for alignment
    thr = np.append(thr, 1.0)

    # F_beta = (1+β^2) * (P*R) / (β^2*P + R)
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
    # Cross-validation settings
    k_folds: int = 10,
    # Mode:
    #   'seq' -> RNN-style models that output logits per timestep [B, T]
    #   'tab' -> MLP/logistic models that output logits per row/record [B]
    mode: Literal["seq", "tab"] = "seq",
    window: int = 24,
    stride: int = 1,
    # Model factory: given in_dim -> returns nn.Module with logits of shape [B, T] or [B]
    make_model: Callable[[int], nn.Module],
    # Optimization
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 5,
    early_patience: int = 2,
    # Others
    seed: int = 42,
    beta_for_thr: float = 1.0,
) -> tuple[float, List[float]]:
    """
    Run forward-chaining K-fold CV **only on the training split** to obtain
    a robust decision threshold (median across folds).

    Returns:
        (robust_thr, per_fold_thresholds)
        - robust_thr: median of thresholds across folds
        - per_fold_thresholds: the individual fold thresholds (for debugging)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    folds = _make_group_forward_time_folds(df_train, n_folds=k_folds)
    if not folds:
        # Fallback: no folds could be built; return a neutral default threshold
        return 0.5, []

    thr_list: List[float] = []

    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        # Keep chronological order inside each split
        tr = df_train.loc[tr_idx].sort_values([METER_ID_COL, TIMESTAMP_COL])
        va = df_train.loc[va_idx].sort_values([METER_ID_COL, TIMESTAMP_COL])

        # Fit scaler on *fold-train* only; transform both train & valid
        # Note: scale_on_train(train_df, test_df, feature_cols) -> (train_scaled, test_scaled, scaler)
        tr_scaled, _, scaler_f = scale_on_train(tr, tr, feature_cols)
        va_scaled = va.copy()
        va_scaled[feature_cols] = scaler_f.transform(va_scaled[feature_cols])

        # Build loaders based on mode
        if mode == "seq":
            tr_ds = SeqDatasetCls(tr_scaled, feature_cols, window=window, stride=stride)
            va_ds = SeqDatasetCls(va_scaled, feature_cols, window=window, stride=stride)
            if len(tr_ds) == 0 or len(va_ds) == 0:
                # Not enough data to make windows in this fold; skip
                continue

            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
            va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            # Class imbalance handling via pos_weight: (#neg / #pos)
            y_seq = tr_ds.y.ravel()  # [Nw * T]
            pos_w = (len(y_seq) - y_seq.sum()) / (y_seq.sum() + 1e-6)

        else:  # "tab"
            Xtr, ytr = to_xy(tr_scaled, feature_cols)
            Xva, yva = to_xy(va_scaled, feature_cols)

            tr_loader = DataLoader(
                TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                batch_size=batch_size, shuffle=True, num_workers=0
            )
            va_loader = DataLoader(
                TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                batch_size=batch_size, shuffle=False, num_workers=0
            )
            pos_w = (len(ytr) - ytr.sum()) / (ytr.sum() + 1e-6)

        # Model / optimizer / loss
        model_f = make_model(len(feature_cols)).to(DEVICE)
        opt_f = torch.optim.Adam(model_f.parameters(), lr=lr, weight_decay=weight_decay)
        crit_f = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=DEVICE)
        )

        # Early stopping on fold-valid F_beta (computed on last timestep for seq)
        bestF, patience, best_state = -1.0, 0, None

        for _ in range(max_epochs):
            model_f.train()
            for xb, yb in tr_loader:
                xb = xb.to(DEVICE, dtype=torch.float32)
                yb = yb.to(DEVICE, dtype=torch.float32)  # BCEWithLogitsLoss expects float

                logits = model_f(xb)
                # Sanity check: shape must match target
                if logits.shape != yb.shape:
                    # Common cases: [B, T, 1] -> [B, T] or [B, 1] -> [B]
                    if logits.ndim == yb.ndim + 1 and logits.size(-1) == 1:
                        logits = logits.squeeze(-1)
                loss = crit_f(logits, yb)

                opt_f.zero_grad()
                loss.backward()
                opt_f.step()

            # ---- Fold validation: compute provisional threshold and F_beta ----
            model_f.eval()
            with torch.no_grad():
                vs, yt = [], []
                for xb, yb in va_loader:
                    logits = model_f(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy()
                    # Keep shapes consistent with targets collected below
                    if logits.ndim == 3 and logits.shape[-1] == 1:
                        logits = np.squeeze(logits, axis=-1)
                    vs.append(logits)          # [B,T] or [B]
                    yt.append(yb.numpy())      # [B,T] or [B]

            yt_bt = np.concatenate(yt, axis=0)                           # [Nw,T] or [N]
            vp_bt = expit(np.clip(np.concatenate(vs, axis=0), -20, 20))  # apply sigmoid safely

            # For sequence models we evaluate on the *last* timestep of each window
            if yt_bt.ndim == 2:
                yt_1h = yt_bt[:, -1].ravel()
                vp_1h = vp_bt[:, -1].ravel()
            else:
                yt_1h = yt_bt.ravel()
                vp_1h = vp_bt.ravel()

            info = _choose_threshold_fbeta(yt_1h, vp_1h, beta=beta_for_thr)
            vpred = (vp_1h >= info["thr"]).astype(int)
            vF = fbeta_score(yt_1h, vpred, beta=beta_for_thr,
                             average="binary", zero_division=0)

            # Early stop bookkeeping
            if vF > bestF + 1e-4:
                bestF, patience = vF, 0
                # Save the best weights so far
                best_state = {k: v.clone() for k, v in model_f.state_dict().items()}
            else:
                patience += 1
                if patience >= early_patience:
                    break

        # ---- Final evaluation at best checkpoint to record this fold's threshold ----
        if best_state is not None:
            model_f.load_state_dict(best_state)
        model_f.eval()
        with torch.no_grad():
            vs, yt = [], []
            for xb, yb in va_loader:
                logits = model_f(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy()
                if logits.ndim == 3 and logits.shape[-1] == 1:
                    logits = np.squeeze(logits, axis=-1)
                vs.append(logits)
                yt.append(yb.numpy())

        yt_bt = np.concatenate(yt, axis=0)
        vp_bt = expit(np.clip(np.concatenate(vs, axis=0), -20, 20))

        if yt_bt.ndim == 2:
            yt_1h = yt_bt[:, -1].ravel()
            vp_1h = vp_bt[:, -1].ravel()
        else:
            yt_1h = yt_bt.ravel()
            vp_1h = vp_bt.ravel()

        info = _choose_threshold_fbeta(yt_1h, vp_1h, beta=beta_for_thr)

        thr_list.append(float(info["thr"]))
        print(f"[CV fold {k:02d}] thr={info['thr']:.3f}  Fβ={info['fbeta']:.4f}  "
              f"P={info['prec']:.4f}  R={info['rec']:.4f}  AP={info['ap']:.4f}")

    if not thr_list:
        # If all folds were skipped (e.g., insufficient data), return default
        return 0.5, []

    robust_thr = float(np.median(thr_list))
    print(f"[CV] thresholds={np.round(thr_list, 3).tolist()} → robust_thr={robust_thr:.3f}")
    return robust_thr, thr_list
