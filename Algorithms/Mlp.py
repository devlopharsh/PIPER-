from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import argparse
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error,
)
import pandas as pd
from utils.constants import (
    DATA_DIR, MODELS_DIR,
    METER_ID_COL, LEAK_FLAG_COL,
    CLS_FEATURE_COLS, MLP_DEFAULTS,
    RANDOM_SEED_DEFAULT, TEST_SIZE_DEFAULT,
    DEVICE
)
from utils.ThresholdComputer import find_stable_threshold
from sklearn.metrics import average_precision_score, fbeta_score, precision_recall_curve
from scipy.special import expit


from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)


# ----------------------------- Helpers -----------------------------
def set_seed(seed: int = RANDOM_SEED_DEFAULT) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_xy(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LEAK_FLAG_COL].to_numpy(dtype=np.float32)  # BCEWithLogits expects float {0,1}
    return X, y


# ----------------------------- Model -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = (128, 64), p_drop: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers += [nn.Linear(prev, 1)]  # binary logit (cls) or scalar (reg)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------------------- Train / Eval -----------------------------
def train_and_eval_mlp(
    data_dir: Path,
    # Architecture
    hidden: List[int] = tuple(MLP_DEFAULTS["hidden"]),
    dropout: float = MLP_DEFAULTS["dropout"],
    # Optimization
    batch_size: int = MLP_DEFAULTS["batch_size"],
    lr: float = MLP_DEFAULTS["lr"],
    weight_decay: float = MLP_DEFAULTS["weight_decay"],
    max_epochs: int = MLP_DEFAULTS["max_epochs"],
    early_patience: int = MLP_DEFAULTS["early_patience"],
    seed: int = RANDOM_SEED_DEFAULT,
    # Split
    test_size: float = TEST_SIZE_DEFAULT,
    # Classification data knobs
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
):
    set_seed(seed)

    # ------------------------- BUILD DATA -------------------------
    # Build classification frame with synthetic persistent leaks
    df = build_aggregated_dataset(
        data_dir=data_dir,
        leak_mode="variable",
        test_fraction=test_size,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )

    # Time split per meter → train/test; then train_core/val
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
    train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

    # Feature set (classification)
    feat_cols = [c for c in CLS_FEATURE_COLS if c in train_core_df.columns]

    # Scale X on TRAIN-CORE only
    tr_core_scaled, _dummy, scaler = scale_on_train(train_core_df, train_core_df, feat_cols)
    val_scaled = val_df.copy();  val_scaled[feat_cols] = scaler.transform(val_scaled[feat_cols])
    te_scaled  = test_df.copy(); te_scaled[feat_cols]  = scaler.transform(te_scaled[feat_cols])

    # Lock to common columns (defensive)
    common_cols = [c for c in feat_cols
                    if c in tr_core_scaled.columns
                    and c in val_scaled.columns
                    and c in te_scaled.columns]
    feat_cols = common_cols

    # Dataloaders
    Xtr, ytr = to_xy(tr_core_scaled, feat_cols)
    Xva, yva = to_xy(val_scaled,     feat_cols)
    Xte, yte = to_xy(te_scaled,      feat_cols)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                                batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                                batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                                batch_size=batch_size, shuffle=False, num_workers=0)

    # Infer input dim from batch
    xb0, _ = next(iter(train_loader))
    in_dim = xb0.shape[1]
    model = MLP(in_dim=in_dim, hidden=list(hidden), p_drop=dropout).to(DEVICE)

    # Loss/optim (class imbalance)
    pos = float(ytr.sum())
    pos_weight = (len(ytr) - pos) / (pos + 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE))
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train + early stop on Val F1 (via stable threshold)
    best_metric = -1.0; patience = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            optim.zero_grad(); loss.backward(); optim.step()

        # Validate
        model.eval()
        with torch.no_grad():
            logits, ys = [], []
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits.append(model(xb).cpu().numpy())
                ys.append(yb.numpy())
        y_scores = np.concatenate(logits).ravel()
        y_true   = np.concatenate(ys).ravel()
        y_probs  = expit(np.clip(y_scores, -20, 20))

        beta = 1
        stable_thr, thr_list = find_stable_threshold(y_true, y_probs, beta=beta, n_splits=5)
        y_pred = (y_probs >= stable_thr).astype(int)

        val_fbeta = fbeta_score(y_true, y_pred, beta=beta, average="binary", zero_division=0)
        val_ap    = average_precision_score(y_true, y_probs)
        print(f"[Epoch {ep:02d}] Val AP={val_ap:.4f} | Val F{beta}={val_fbeta:.4f} @thr={stable_thr:.3f}")

        if val_fbeta > best_metric + 1e-4:
            best_metric = val_fbeta; patience = 0
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "mlp_cls.pt")
            joblib.dump(scaler, MODELS_DIR / "mlp_cls_scaler.joblib")
            with open(MODELS_DIR / "mlp_cls_best_thr.txt", "w") as f:
                f.write(str(stable_thr))
        else:
            patience += 1
            if patience >= early_patience:
                print("[INFO] Early stopping."); break

    # ---- Test ----
    model.load_state_dict(torch.load(MODELS_DIR / "mlp_cls.pt", map_location=DEVICE))
    with open(MODELS_DIR / "mlp_cls_best_thr.txt") as f:
        best_thr = float(f.read().strip())

    model.eval()
    with torch.no_grad():
        logits, ys = [], []
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits.append(model(xb).cpu().numpy())
            ys.append(yb.numpy())
    y_scores = np.concatenate(logits).ravel()
    y_true   = np.concatenate(ys).ravel()
    y_probs  = expit(np.clip(y_scores, -20, 20))
    y_pred   = (y_probs >= best_thr).astype(int)

    acc  = float(accuracy_score(y_true, y_pred))
    f1b  = float(f1_score(y_true, y_pred, average="binary"))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    ap   = float(average_precision_score(y_true, y_probs))
    f2   = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return {
        "task": "cls",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_binary": f1b,
        "f2_binary": f2,
        "ap": ap,
        "cm": cm,
        "report": report,
        "thr": best_thr,
        "model_path": str(MODELS_DIR / "mlp_cls.pt"),
        "scaler_path": str(MODELS_DIR / "mlp_cls_scaler.joblib"),
    }



# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP — classification")
    parser.add_argument("--data_dir", type=Path, default=str(DATA_DIR))

    # Arch
    parser.add_argument("--hidden", nargs="+", type=int, default=MLP_DEFAULTS["hidden"])
    parser.add_argument("--dropout", type=float, default=MLP_DEFAULTS["dropout"])

    # Opt
    parser.add_argument("--batch_size", type=int, default=MLP_DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=MLP_DEFAULTS["lr"])
    parser.add_argument("--weight_decay", type=float, default=MLP_DEFAULTS["weight_decay"])
    parser.add_argument("--max_epochs", type=int, default=MLP_DEFAULTS["max_epochs"])
    parser.add_argument("--early_patience", type=int, default=MLP_DEFAULTS["early_patience"])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)

    # Split
    parser.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)

    # Classification builder knobs
    parser.add_argument("--n_segments_min", type=int, default=2)
    parser.add_argument("--n_segments_max", type=int, default=5)
    parser.add_argument("--min_len_frac", type=float, default=0.05)

    args = parser.parse_args()

    results = train_and_eval_mlp(
        data_dir=args.data_dir,
        hidden=args.hidden,
        dropout=args.dropout,
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

    print(json.dumps(results, indent=2))
