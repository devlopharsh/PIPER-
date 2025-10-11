from __future__ import annotations
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score,
    average_precision_score, fbeta_score
)
import joblib

from utils.constants import (
    DATA_DIR, MODELS_DIR,
    METER_ID_COL, TIMESTAMP_COL,
    LEAK_FLAG_COL,
    CLS_FEATURE_COLS, GRU_DEFAULTS, RANDOM_SEED_DEFAULT,
    TEST_SIZE_DEFAULT, DEVICE
    
)
from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)
from utils.CvkFold import SeqDatasetCls, get_cv_threshold_on_train

class GRUHead(nn.Module):
    """
    Official PyTorch GRU → Linear head.
    For classification: use BCEWithLogitsLoss on [B, T].
    For forecasting (regression): use MSELoss on [B, T].
    """
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else dropout
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):           # x: [B, T, F]
        out, _ = self.gru(x)        # [B, T, H or 2H]
        out = self.fc(out)          # [B, T, 1]
        return out.squeeze(-1)      # [B, T]


# class SeqDatasetCls(torch.utils.data.Dataset):
#     def __init__(self, df, feature_cols, window=24, stride=1):
#         self.X, self.y = self._build_sequences(df, feature_cols, window, stride)

#     @staticmethod
#     def _build_sequences(df, feature_cols, window, stride):
#         Xs, ys = [], []
#         for _, g in df.sort_values([METER_ID_COL, TIMESTAMP_COL]).groupby(METER_ID_COL, sort=False):
#             g = g.reset_index(drop=True)
#             arr = g[feature_cols].to_numpy(dtype=np.float32)
#             labels = g[LEAK_FLAG_COL].to_numpy(dtype=np.int64)
#             n = len(g)
#             for start in range(0, n - window + 1, stride):
#                 end = start + window
#                 Xs.append(arr[start:end])
#                 ys.append(labels[start:end])
#         if not Xs:
#             return (np.zeros((0, window, len(feature_cols)), np.float32),
#                     np.zeros((0, window), np.int64))
#         return np.stack(Xs), np.stack(ys)

#     def __len__(self): return len(self.y)
#     def __getitem__(self, i):
#         return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])



def train_and_eval_gru(
    data_dir: Path,
    # shared
    window: int = 24, stride: int = 1,
    hidden: int = 64, num_layers: int = 1, bidirectional: bool = False,
    batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
    max_epochs: int = 20, early_patience: int = 6,
    seed: int = 42, test_size: float = 0.30, dropout: float = 0.1,
    # classification data knobs
    n_segments_min: int = 2, n_segments_max: int = 5, min_len_frac: float = 0.05,
):
    torch.manual_seed(seed); np.random.seed(seed)

    # 1) Build classification df with synthetic persistent leaks
    df = build_aggregated_dataset(
        data_dir=data_dir,
        leak_mode="variable",
        test_fraction=test_size,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )
    feature_cols = list(CLS_FEATURE_COLS)

    # 2) Time-based per-meter split → train/test; then train_core/val
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
    # === Inner CV on TRAIN: get robust threshold (forward-chaining, per-meter) ===
    cv_thr, cv_thr_list = get_cv_threshold_on_train(
        df_train=train_df,
        feature_cols=feature_cols,
        k_folds=10,
        mode="seq",
        window=window, stride=stride,
        make_model=lambda in_dim: GRUHead(
            in_dim=in_dim, hidden=hidden, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout
        ),
        batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, early_patience=early_patience,
        seed=seed, beta_for_thr=1.0,
    )

    print(f"[CV] thresholds={np.round(cv_thr_list,3).tolist()} → robust_thr={cv_thr:.3f}")

    # 3) Scale features on the FULL TRAIN
    tr_scaled, _, scaler = scale_on_train(train_df, train_df, feature_cols)
    test_scaled = test_df.copy()
    test_scaled[feature_cols] = scaler.transform(test_scaled[feature_cols])

    # 4) Build datasets/loaders
    train_ds = SeqDatasetCls(tr_scaled, feature_cols, window=window, stride=stride)
    test_ds  = SeqDatasetCls(test_scaled, feature_cols, window=window, stride=stride)
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Empty split after windowing. Reduce window/stride or check data coverage.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # 5) Imbalance weight on FULL TRAIN
    y_seq = train_ds.y.ravel()
    pos_weight = torch.tensor([(len(y_seq) - y_seq.sum()) / (y_seq.sum() + 1e-6)],
                            dtype=torch.float32, device=DEVICE)

    # 6) Final model: train for max_epochs on FULL TRAIN (no early stopping)
    model = GRUHead(in_dim=len(feature_cols), hidden=hidden, num_layers=num_layers,
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

    # Save model + scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "gru_cls.pt")
    joblib.dump(scaler, MODELS_DIR / "gru_cls_scaler.joblib")

    # 7) Test evaluation (threshold from CV)
    model.load_state_dict(torch.load(MODELS_DIR / "gru_cls.pt", map_location=DEVICE))
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
    
    print(f"[INFO] CV thresholds={cv_thr_list}, using stable_thr={best_thr:.3f}")

    # --- Metrics ---
    acc  = accuracy_score(ytrue, ypred)
    f1m  = f1_score(ytrue, ypred, average="binary", zero_division=0)
    prec = precision_score(ytrue, ypred, average="binary", zero_division=0)
    rec  = recall_score(ytrue, ypred, average="binary", zero_division=0)
    f2b  = fbeta_score(ytrue, ypred, beta=2.0, average="binary", zero_division=0)
    ap   = average_precision_score(ytrue, probs)

    print("Classification report:\n", classification_report(ytrue, ypred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(ytrue, ypred))
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-macro: {f1m:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"[TEST] PR-AUC(Leak)={ap:.4f} | F2(Leak)={f2b:.4f} | F1(Leak)={f1m:.4f} | "
    f"Precision(Leak)={prec:.4f} | Recall(Leak)={rec:.4f} | thr={best_thr:.3f}")

    return {
        "task": "cls", "model": "GRU",
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1m), "f2": float(f2b), "ap": float(ap)
    }

# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PyTorch GRU — classification or forecasting")

    ap.add_argument("--data_dir", type=Path, default=Path(str(DATA_DIR)))

    # --- shared RNN hyperparams (align with train_and_eval_gru signature) ---
    ap.add_argument("--window", type=int, default=GRU_DEFAULTS.window)
    ap.add_argument("--stride", type=int, default=GRU_DEFAULTS.stride)
    ap.add_argument("--hidden", type=int, default=GRU_DEFAULTS.hidden)
    ap.add_argument("--num_layers", type=int, default=GRU_DEFAULTS.num_layers)
    ap.add_argument("--bidirectional", action="store_true", default=GRU_DEFAULTS.bidirectional)
    ap.add_argument("--dropout", type=float, default=GRU_DEFAULTS.dropout)
    ap.add_argument("--batch_size", type=int, default=GRU_DEFAULTS.batch_size)
    ap.add_argument("--lr", type=float, default=GRU_DEFAULTS.lr)
    ap.add_argument("--weight_decay", type=float, default=GRU_DEFAULTS.weight_decay)
    ap.add_argument("--max_epochs", type=int, default=GRU_DEFAULTS.max_epochs)
    ap.add_argument("--early_patience", type=int, default=GRU_DEFAULTS.early_patience)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    ap.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)

    # --- classification data knobs ---
    ap.add_argument("--n_segments_min", type=int, default=2)
    ap.add_argument("--n_segments_max", type=int, default=5)
    ap.add_argument("--min_len_frac", type=float, default=0.05)

    args = ap.parse_args()

    results = train_and_eval_gru(
        data_dir=args.data_dir,
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
        dropout=args.dropout,
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
    )

    print(results)

