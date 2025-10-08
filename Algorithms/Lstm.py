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

#region Model Training & Evaluation
def train_and_eval_lstm(
    data_dir: Path,
    window: int = 24, stride: int = 1,
    hidden: int = 64, num_layers: int = 1, bidirectional: bool = False,
    batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
    max_epochs: int = 20, early_patience: int = 4,
    seed: int = 42, test_size: float = 0.30, dropout = 0.1,
    # classification builder knobs
    n_segments_min: int = 2,             # number of leak segments per meter
    n_segments_max: int = 5,
    min_len_frac: float = 0.05,          # each leak segment must be at least this fraction of the series
):
    """
    High-level flow:
    1) Build dataframe (either classification or forecasting features).
    2) Split by time per meter (train/val/test in chronological order).
    3) Scale on TRAIN ONLY and apply to val/test (prevents data leakage).
    4) Train with early stopping based on val metric.
    5) Evaluate on test. In forecasting, detect leaks via residual threshold.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    
    #region Classification Workflow
    # 1) Build classification frame: random leak placement across the series
    df = build_aggregated_dataset(
        data_dir=data_dir,
        leak_mode="variable",                  # use variable leak profile
        test_fraction=test_size,               # how many samples become leaks overall
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )
    feature_cols = list(CLS_FEATURE_COLS)      # e.g., ["Volume_Reading","Hour","DayOfWeek","Month","Delta_Volume"]

    # 2) Time-based split per meter (prevents "peeking into the future")
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
    # Make a small validation set from the tail of train (again, per meter)
    train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

    # 3) Scale features on TRAIN-CORE only; transform VAL & TEST
    tr_core_scaled, _val_dummy, scaler = scale_on_train(train_core_df, train_core_df, feature_cols)
    val_scaled = val_df.copy()
    val_scaled[feature_cols] = scaler.transform(val_scaled[feature_cols])
    test_scaled = test_df.copy()
    test_scaled[feature_cols] = scaler.transform(test_scaled[feature_cols])

    # 4) Build sliding-window datasets AFTER scaling
    train_ds = SeqDatasetCls(tr_core_scaled, feature_cols, window=window, stride=stride)
    val_ds   = SeqDatasetCls(val_scaled,     feature_cols, window=window, stride=stride)
    test_ds  = SeqDatasetCls(test_scaled,    feature_cols, window=window, stride=stride)

    # If window is too large compared to series length, you may end up with zero samples
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Empty split after windowing. Reduce window/stride or check data coverage.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # 5) Handle class imbalance by weighting positive class in BCE
    y_seq = train_ds.y.ravel()  # flatten all labels [B*T]
    pos_weight = torch.tensor([(len(y_seq) - y_seq.sum()) / (y_seq.sum() + 1e-6)],
                            dtype=torch.float32, device=DEVICE)

    model = LSTMHead(in_dim=len(feature_cols), hidden=hidden, num_layers=num_layers,
                        bidirectional=bidirectional, dropout=dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 6) Train with early stopping (based on val F1/F2 score)
    best, patience = -1.0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.float32)
            loss = criterion(model(xb), yb)
            optim.zero_grad(); loss.backward(); optim.step()

        # Validate
        model.eval()
        with torch.no_grad():
            vscores, vtrue = [], []
            for xb, yb in val_loader:
                vscores.append(model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                vtrue.append(yb.numpy())

            vscores = np.concatenate(vscores).ravel()  # flatten
            vtrue   = np.concatenate(vtrue).ravel()

        vprobs = expit(np.clip(vscores, -20, 20))  # numerically stable sigmoid

        # Use CV-averaged threshold for stability
        beta = 1
        stable_thr, thr_list = find_stable_threshold(vtrue, vprobs, beta=beta, n_splits=5)
        vpred = (vprobs >= stable_thr).astype(int)

        vF = fbeta_score(vtrue, vpred, beta=beta, average="binary", zero_division=0)
        vAP = average_precision_score(vtrue, vprobs)

        # keep AP for early stopping (threshold-free)
        metric_for_early_stop = vF

        print(f"[Epoch {ep:02d}] Val AP={vAP:.4f} | Val F{beta}={vF:.4f} @thr={stable_thr:.3f}")

        if metric_for_early_stop > best + 1e-4:
            best, patience = metric_for_early_stop, 0
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "lstm_cls.pt")
            joblib.dump(scaler, MODELS_DIR / "lstm_cls_scaler.joblib")
            with open(MODELS_DIR / "lstm_cls_best_thr.txt", "w") as f:
                f.write(str(stable_thr))
        else:
            patience += 1
            if patience >= early_patience:
                print("[INFO] Early stopping."); break

    # 7) Test evaluation
    model.load_state_dict(torch.load(MODELS_DIR / "lstm_cls.pt", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        scores, ytrue = [], []
        for xb, yb in test_loader:
            scores.append(model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
            ytrue.append(yb.numpy())

        scores = np.concatenate(scores).ravel()
        ytrue  = np.concatenate(ytrue).ravel()

    probs = expit(np.clip(scores, -20, 20))

    # Use the best threshold computed from val
    with open(MODELS_DIR / "lstm_cls_best_thr.txt") as f:
        best_thr = float(f.read().strip())

    print(f"[INFO] CV thresholds={thr_list}, using stable_thr={best_thr:.3f}")
    ypred = (probs >= best_thr).astype(int)


    # --- Metrics ---
    acc  = accuracy_score(ytrue, ypred)
    f1m  = f1_score(ytrue, ypred, average="binary")
    prec = precision_score(ytrue, ypred, average="binary", zero_division=0)
    rec  = recall_score(ytrue, ypred, average="binary", zero_division=0)
    f2_bin   = fbeta_score(ytrue, ypred, beta=2.0, average="binary", zero_division=0)
    ap = average_precision_score(ytrue, probs)

    print("Classification report:\n", classification_report(ytrue, ypred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(ytrue, ypred))
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-macro: {f1m:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"[TEST] PR-AUC(Leak)={ap:.4f} | F2(Leak)={f2_bin:.4f} | F1(Leak)={f1m:.4f} | "
    f"Precision(Leak)={prec:.4f} | Recall(Leak)={rec:.4f} | thr={best_thr:.3f}")

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