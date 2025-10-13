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
import json
from utils.ParserCreator import create_common_parser, attach_additional_args, AlgoKind


class GRUHead(nn.Module):
    """
    Official PyTorch GRU → Linear head.
    For classification: use BCEWithLogitsLoss on [B, T].
    For forecasting (regression): use MSELoss on [B, T].
    """
    def __init__(self, in_dim: int, hidden: int = GRU_DEFAULTS.hidden, num_layers: int = GRU_DEFAULTS.num_layers,
                 bidirectional: bool = GRU_DEFAULTS.bidirectional, dropout: float = GRU_DEFAULTS.dropout):
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
        test_fraction=test_size,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )
    feature_cols = list(CLS_FEATURE_COLS)

    # 2) Time-based per-meter split → train/test; then train_core/val
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
    ART = MODELS_DIR / "artifacts"
    ART.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(ART / "train_agg.parquet", index=False) # optional
    test_df.to_parquet(ART / "test_agg.parquet", index=False)
    print(f"[ARTIFACTS] Saved frozen splits under {ART}")
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
    # Save robust threshold για deployment (προαιρετικό)
    (MODELS_DIR / "gru_cls_thr.txt").write_text(f"{cv_thr:.6f}")
    joblib.dump(scaler, MODELS_DIR / "gru_cls_scaler.joblib")
    meta = {
    "model": "GRUHead",
    "task": "cls",
    "thr": float(cv_thr),
    "window": int(window),
    "stride": int(stride),
    "in_dim": len(feature_cols),
    "feature_cols": list(feature_cols),
    "hidden": int(hidden),
    "num_layers": int(num_layers),
    "bidirectional": bool(bidirectional),
    "dropout": float(dropout),
    }
    (MODELS_DIR / "gru_cls_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[META] Saved {MODELS_DIR/'gru_cls_meta.json'}")

    # 7) Test evaluation (threshold from CV)
    model.load_state_dict(torch.load(MODELS_DIR / "gru_cls.pt", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        s_list, y_list = [], []
        for xb, yb in test_loader:
            logits_bt = model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy()  # [B,T]
            s_list.append(logits_bt)
            y_list.append(yb.numpy())

    scores_bt = np.concatenate(s_list, axis=0)   # [Nw, T]
    y_bt      = np.concatenate(y_list, axis=0)   # [Nw, T]

    scores_last = scores_bt[:, -1]               # [Nw]
    ytrue_last  = y_bt[:, -1]                    # [Nw]
    probs       = expit(np.clip(scores_last, -20, 20))
    ytrue = ytrue_last

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
    p = create_common_parser(
        "PyTorch GRU — classification",
        data_dir_default=DATA_DIR,
        test_size_default=TEST_SIZE_DEFAULT,
        seed_default=RANDOM_SEED_DEFAULT,
        n_segments_min_default=2,
        n_segments_max_default=5,
        min_len_frac_default=0.05,
    )
    attach_additional_args(p, AlgoKind.GRU,
        window=GRU_DEFAULTS.window, stride=GRU_DEFAULTS.stride,
        hidden=GRU_DEFAULTS.hidden, num_layers=GRU_DEFAULTS.num_layers,
        bidirectional=GRU_DEFAULTS.bidirectional, dropout=GRU_DEFAULTS.dropout,
        batch_size=GRU_DEFAULTS.batch_size, lr=GRU_DEFAULTS.lr,
        weight_decay=GRU_DEFAULTS.weight_decay, max_epochs=GRU_DEFAULTS.max_epochs,
        early_patience=GRU_DEFAULTS.early_patience,
    )
    args = p.parse_args()

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