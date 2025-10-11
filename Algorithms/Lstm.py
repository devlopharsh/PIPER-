from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score,
    average_precision_score, fbeta_score
)
from scipy.special import expit
from utils.ThresholdComputer import find_stable_threshold

# ---- Project utilities / constants
from utils.constants import (
    DATA_DIR, MODELS_DIR,
    METER_ID_COL, TIMESTAMP_COL,
    LEAK_FLAG_COL, DEVICE,
    CLS_FEATURE_COLS, LSTM_DEFAULTS,
    TEST_SIZE_DEFAULT, RANDOM_SEED_DEFAULT
)
from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)


#region Datasets for classification
class SeqDatasetCls(Dataset):
    """
    Classification dataset (per hour).
    - Each window is [T, F]
    - Labels are [T] (one label for each timestep, not only the last).
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], window: int = 24, stride: int = 1):
        self.X, self.y = self._build_sequences(df, feature_cols, window, stride)

    @staticmethod
    def _build_sequences(df: pd.DataFrame, feature_cols: list[str], window: int, stride: int):
        Xs, ys = [], []
        # Group by meter so sequences from different meters do not mix
        for _, g in df.sort_values([METER_ID_COL, TIMESTAMP_COL]).groupby(METER_ID_COL, sort=False):
            g = g.reset_index(drop=True)
            arr = g[feature_cols].to_numpy(dtype=np.float32)    # [N, F]
            labels = g[LEAK_FLAG_COL].to_numpy(dtype=np.int64)  # [N]
            n = len(g)
            # Sliding windows
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

#endregion


#region LSTM Model
class LSTMHead(nn.Module):
    """
    A minimal LSTM → Linear head:
    - LSTM encodes the sequence into hidden states.
    - We take the output at all time steps and pass them through a linear layer.
    - For classification: we’ll apply BCEWithLogitsLoss (so output is a logit).
    - For regression: we’ll apply MSELoss directly on the scalar output.
    """
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        # Dropout only applies between stacked LSTM layers; it’s ignored for num_layers=1
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):            # x: [B, T, F]
        out, _ = self.lstm(x)        # [B, T, H]
        out = self.fc(out)           # [B, T, 1]
        return out.squeeze(-1)       # [B, T]
#endregion


# --- K-FOLD CV (forward-chaining per meter) helpers ---
def _make_group_forward_time_folds(df_in: pd.DataFrame, n_folds: int = 10):
    df_in = df_in.sort_values([METER_ID_COL, TIMESTAMP_COL]).copy()
    folds = [( [], [] ) for _ in range(n_folds)]
    for _, g in df_in.groupby(METER_ID_COL, sort=False):
        n = len(g)
        if n < n_folds + 1:  # skip very short series
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
    from sklearn.metrics import precision_recall_curve, average_precision_score
    p, r, thr = precision_recall_curve(y_true, probs)
    thr = np.append(thr, 1.0)
    f = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    i = int(np.nanargmax(f))
    return {
        "thr": float(thr[i]),
        "fbeta": float(f[i]),
        "prec": float(p[i]),
        "rec": float(r[i]),
        "ap": float(average_precision_score(y_true, probs)),
    }


def _cv_threshold_on_train(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    *,
    k_folds: int = 10,
    window: int = 24, stride: int = 1,
    hidden: int = 64, num_layers: int = 1, bidirectional: bool = False,
    batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
    max_epochs: int = 5, early_patience: int = 2,
    dropout: float = 0.1, seed: int = 42, beta_for_thr: float = 1.0,
) -> tuple[float, list[float]]:
    """
    Forward-chaining K-fold CV ONLY on the training split to pick a robust probability threshold.
    Returns (robust_threshold, list_of_per_fold_thresholds).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    folds = _make_group_forward_time_folds(df_train, n_folds=k_folds)
    thr_list: list[float] = []

    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        tr = df_train.loc[tr_idx].sort_values([METER_ID_COL, TIMESTAMP_COL])
        va = df_train.loc[va_idx].sort_values([METER_ID_COL, TIMESTAMP_COL])

        # scale on fold-train only (no leakage)
        tr_scaled, _, scaler_f = scale_on_train(tr, tr, feature_cols)
        va_scaled = va.copy()
        va_scaled[feature_cols] = scaler_f.transform(va_scaled[feature_cols])

        tr_ds = SeqDatasetCls(tr_scaled, feature_cols, window=window, stride=stride)
        va_ds = SeqDatasetCls(va_scaled, feature_cols, window=window, stride=stride)
        if len(tr_ds) == 0 or len(va_ds) == 0:
            continue

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # imbalance weight from fold-train
        y_seq = tr_ds.y.ravel()
        pos_w = (len(y_seq) - y_seq.sum()) / (y_seq.sum() + 1e-6)
        pos_weight_f = torch.tensor([pos_w], dtype=torch.float32, device=DEVICE)

        model_f = LSTMHead(
            in_dim=len(feature_cols), hidden=hidden, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout
        ).to(DEVICE)
        opt_f = torch.optim.Adam(model_f.parameters(), lr=lr, weight_decay=weight_decay)
        crit_f = nn.BCEWithLogitsLoss(pos_weight=pos_weight_f)

        bestF, patience, best_state = -1.0, 0, None

        for _ in range(max_epochs):
            model_f.train()
            for xb, yb in tr_loader:
                xb = xb.to(DEVICE, dtype=torch.float32)
                yb = yb.to(DEVICE, dtype=torch.float32)
                loss = crit_f(model_f(xb), yb)
                opt_f.zero_grad(); loss.backward(); opt_f.step()

            # fold-val → provisional threshold
            model_f.eval()
            with torch.no_grad():
                vs, yt = [], []
                for xb, yb in va_loader:
                    vs.append(model_f(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                    yt.append(yb.numpy())
            yt = np.concatenate(yt).ravel()
            vp = expit(np.clip(np.concatenate(vs).ravel(), -20, 20))
            info = _choose_threshold_fbeta(yt, vp, beta=beta_for_thr)
            vpred = (vp >= info["thr"]).astype(int)
            vF = fbeta_score(yt, vpred, beta=beta_for_thr, average="binary", zero_division=0)

            if vF > bestF + 1e-4:
                bestF, patience = vF, 0
                best_state = {k: v.clone() for k, v in model_f.state_dict().items()}
            else:
                patience += 1
                if patience >= early_patience:
                    break

        # evaluate at best weights for the fold to get the fold's threshold
        if best_state is not None:
            model_f.load_state_dict(best_state)
        model_f.eval()
        with torch.no_grad():
            vs, yt = [], []
            for xb, yb in va_loader:
                vs.append(model_f(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                yt.append(yb.numpy())
        yt = np.concatenate(yt).ravel()
        vp = expit(np.clip(np.concatenate(vs).ravel(), -20, 20))
        info = _choose_threshold_fbeta(yt, vp, beta=beta_for_thr)

        thr_list.append(float(info["thr"]))
        print(f"[CV fold {k:02d}] thr={info['thr']:.3f}  Fβ={info['fbeta']:.4f}  P={info['prec']:.4f}  R={info['rec']:.4f}")

    if not thr_list:
        return 0.5, []

    robust_thr = float(np.median(thr_list))
    print(f"[CV] thresholds={np.round(thr_list,3).tolist()} → robust_thr={robust_thr:.3f}")
    return robust_thr, thr_list


#region Model Training & Evaluation
def train_and_eval_lstm(
    data_dir: Path,
    window: int = 24, stride: int = 1,
    hidden: int = 64, num_layers: int = 1, bidirectional: bool = False,
    batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
    max_epochs: int = 20, early_patience: int = 4,   # kept for CV block; unused in final retrain
    seed: int = 42, test_size: float = 0.30, dropout = 0.1,
    n_segments_min: int = 2, n_segments_max: int = 5, min_len_frac: float = 0.05,
):
    torch.manual_seed(seed); np.random.seed(seed)

    # 1) Build classification dataframe
    df = build_aggregated_dataset(
        data_dir=data_dir,
        leak_mode="variable",
        test_fraction=test_size,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )
    feature_cols = list(CLS_FEATURE_COLS)

    # 2) Holdout split (per meter, time-ordered)
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])

    # === Inner CV on TRAIN: get robust threshold (forward-chaining, per-meter) ===
    cv_thr, cv_thr_list = _cv_threshold_on_train(
        df_train=train_df,
        feature_cols=feature_cols,
        k_folds=10,
        window=window, stride=stride,
        hidden=hidden, num_layers=num_layers, bidirectional=bidirectional,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, early_patience=early_patience, dropout=dropout,
        seed=seed, beta_for_thr=1.0,
    )

    print(f"[CV] thresholds={np.round(cv_thr_list,3).tolist()} → robust_thr={cv_thr:.3f}")

    # 3) Scale features on the FULL TRAIN (no val split now)
    tr_scaled, _, scaler = scale_on_train(train_df, train_df, feature_cols)
    test_scaled = test_df.copy()
    test_scaled[feature_cols] = scaler.transform(test_scaled[feature_cols])

    # 4) Build datasets/loaders (train on all train; test untouched)
    train_ds = SeqDatasetCls(tr_scaled, feature_cols, window=window, stride=stride)
    test_ds  = SeqDatasetCls(test_scaled,  feature_cols, window=window, stride=stride)
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Empty split after windowing. Reduce window/stride or check data coverage.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # 5) Imbalance weight from FULL TRAIN
    y_seq = train_ds.y.ravel()
    pos_weight = torch.tensor([(len(y_seq) - y_seq.sum()) / (y_seq.sum() + 1e-6)],
                              dtype=torch.float32, device=DEVICE)

    # 6) Final model: retrain EXACTLY median_best_epoch on FULL TRAIN (no early stopping)
    model = LSTMHead(in_dim=len(feature_cols), hidden=hidden, num_layers=num_layers,
                     bidirectional=bidirectional, dropout=dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.float32)
            loss = criterion(model(xb), yb)
            optim.zero_grad(); loss.backward(); optim.step()

    # Save final weights + scaler
    torch.save(model.state_dict(), MODELS_DIR / "lstm_cls.pt")
    joblib.dump(scaler, MODELS_DIR / "lstm_cls_scaler.joblib")

    # 7) Test evaluation (threshold from CV)
    model.load_state_dict(torch.load(MODELS_DIR / "lstm_cls.pt", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        scores, ytrue = [], []
        for xb, yb in test_loader:
            scores.append(model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
            ytrue.append(yb.numpy())
    scores = np.concatenate(scores).ravel()
    ytrue  = np.concatenate(ytrue).ravel()
    probs  = expit(np.clip(scores, -20, 20))

    best_thr = cv_thr
    print(f"[INFO] Using CV robust threshold={best_thr:.3f} on TEST")
    ypred = (probs >= best_thr).astype(int)

    # --- Metrics ---
    acc  = accuracy_score(ytrue, ypred)
    f1m  = f1_score(ytrue, ypred, average="binary")
    prec = precision_score(ytrue, ypred, average="binary", zero_division=0)
    rec  = recall_score(ytrue, ypred, average="binary", zero_division=0)
    f2_bin = fbeta_score(ytrue, ypred, beta=2.0, average="binary", zero_division=0)
    ap = average_precision_score(ytrue, probs)

    print("Classification report:\n", classification_report(ytrue, ypred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(ytrue, ypred))
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1:       {f1m:.4f}")
    print(f"Precision:{prec:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"[TEST] PR-AUC={ap:.4f} | F2={f2_bin:.4f} | thr={best_thr:.3f}")

    return {
        "task": "cls",
        "model": "LSTM",
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "precision": float(prec),
        "recall": float(rec),
        "f2_leak": float(f2_bin)
    }
    #endregion
#endregion


#region CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LSTM for leak detection — Classification.")
    p.add_argument("--data_dir", type=str, default=str(DATA_DIR))

    # Shared hyperparameters:
    # - window: sequence length (also lags in forecast mode)
    # - stride: how far we move the window each step (classification only)
    p.add_argument("--window", type=int, default=LSTM_DEFAULTS.window)
    p.add_argument("--stride", type=int, default=LSTM_DEFAULTS.stride)
    p.add_argument("--hidden", type=int, default=LSTM_DEFAULTS.hidden)
    p.add_argument("--num_layers", type=int, default=LSTM_DEFAULTS.num_layers)
    p.add_argument("--bidirectional", action="store_true", default=LSTM_DEFAULTS.bidirectional)
    p.add_argument("--dropout", type=float, default=LSTM_DEFAULTS.dropout)
    p.add_argument("--batch_size", type=int, default=LSTM_DEFAULTS.batch_size)
    p.add_argument("--lr", type=float, default=LSTM_DEFAULTS.lr)
    p.add_argument("--weight_decay", type=float, default=LSTM_DEFAULTS.weight_decay)
    p.add_argument("--max_epochs", type=int, default=LSTM_DEFAULTS.max_epochs)
    p.add_argument("--early_patience", type=int, default=LSTM_DEFAULTS.early_patience)
    p.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    p.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)

    # Classification knobs (affect how synthetic leaks are injected):
    p.add_argument("--n_segments_min", type=int, default=1)
    p.add_argument("--n_segments_max", type=int, default=3)
    p.add_argument("--min_len_frac", type=float, default=0.05)

    args = p.parse_args()
    
    res = train_and_eval_lstm(
        data_dir=Path(args.data_dir),
        window=args.window,
        stride=args.stride,
        hidden=args.hidden,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_patience=args.early_patience,
        seed=args.seed,
        test_size=args.test_size,
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
    )
    print(res)
#endregion