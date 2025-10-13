from __future__ import annotations
from pathlib import Path
from typing import List
import argparse
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, fbeta_score
from scipy.special import expit

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from utils.constants import (
    DATA_DIR, MODELS_DIR,
    METER_ID_COL, LEAK_FLAG_COL,
    CLS_FEATURE_COLS, MLP_DEFAULTS,
    RANDOM_SEED_DEFAULT, TEST_SIZE_DEFAULT,
    DEVICE
)
from utils.CvkFold import get_cv_threshold_on_train

from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)
from utils.ParserCreator import create_common_parser, attach_additional_args, AlgoKind


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
    hidden: List[int] = tuple(MLP_DEFAULTS.hidden),
    dropout: float = MLP_DEFAULTS.dropout,
    # Optimization
    batch_size: int = MLP_DEFAULTS.batch_size,
    lr: float = MLP_DEFAULTS.lr,
    weight_decay: float = MLP_DEFAULTS.weight_decay,
    max_epochs: int = MLP_DEFAULTS.max_epochs,
    early_patience: int = MLP_DEFAULTS.early_patience,
    seed: int = RANDOM_SEED_DEFAULT,
    # Split
    test_size: float = TEST_SIZE_DEFAULT,
    # Classification data knobs
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
):
    torch.manual_seed(seed); np.random.seed(seed)

    # ------------------------- BUILD DATA -------------------------
    # Build classification frame with synthetic persistent leaks
    df = build_aggregated_dataset(
        data_dir=data_dir,
        test_fraction=test_size,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )

    # Time split per meter → train/test; then train_core/val
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])

    # Feature set (classification)
    feat_cols = [c for c in CLS_FEATURE_COLS if c in train_df.columns]

    # === Inner CV on TRAIN: robust threshold (tabular mode) ===
    cv_thr, cv_thr_list = get_cv_threshold_on_train(
        df_train=train_df,
        feature_cols=feat_cols,
        k_folds=10,
        mode="tab",
        make_model=lambda in_dim: MLP(in_dim=in_dim, hidden=list(hidden), p_drop=dropout),
        batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        max_epochs=max_epochs, early_patience=early_patience,
        seed=seed, beta_for_thr=1.0,
    )
    print(f"[CV] thresholds={np.round(cv_thr_list,3).tolist()} → robust_thr={cv_thr:.3f}")

    # Scale στο FULL TRAIN
    tr_scaled, _dummy, scaler = scale_on_train(train_df, train_df, feat_cols)
    te_scaled  = test_df.copy(); te_scaled[feat_cols]  = scaler.transform(te_scaled[feat_cols])

    # Dataloaders για final train/test
    Xtr = tr_scaled[feat_cols].to_numpy(np.float32)
    ytr = tr_scaled[LEAK_FLAG_COL].to_numpy(np.float32)
    Xte = te_scaled[feat_cols].to_numpy(np.float32)
    yte = te_scaled[LEAK_FLAG_COL].to_numpy(np.float32)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                            batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
                            batch_size=batch_size, shuffle=False, num_workers=0)

    # Model / loss
    in_dim = Xtr.shape[1]
    model = MLP(in_dim=in_dim, hidden=list(hidden), p_drop=dropout).to(DEVICE)
    pos = float(ytr.sum())
    pos_weight = (len(ytr) - pos) / (pos + 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE))
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Final train σε FULL TRAIN (χωρίς early stop εδώ — το threshold το έχουμε από CV)
    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            optim.zero_grad(); loss.backward(); optim.step()

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "mlp_cls.pt")
    joblib.dump(scaler, MODELS_DIR / "mlp_cls_scaler.joblib")

    # ---- Test ----
    model.load_state_dict(torch.load(MODELS_DIR / "mlp_cls.pt", map_location=DEVICE))
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

    best_thr = cv_thr
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
    p = create_common_parser(
        "MLP — classification",
        data_dir_default=DATA_DIR,
        test_size_default=TEST_SIZE_DEFAULT,
        seed_default=RANDOM_SEED_DEFAULT,
        n_segments_min_default=2,
        n_segments_max_default=5,
        min_len_frac_default=0.05,
    )
    attach_additional_args(p, AlgoKind.MLP,
        hidden=MLP_DEFAULTS.hidden,
        dropout=MLP_DEFAULTS.dropout,
        batch_size=MLP_DEFAULTS.batch_size,
        lr=MLP_DEFAULTS.lr,
        weight_decay=MLP_DEFAULTS.weight_decay,
        max_epochs=MLP_DEFAULTS.max_epochs,
        early_patience=MLP_DEFAULTS.early_patience,
    )
    args = p.parse_args()

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