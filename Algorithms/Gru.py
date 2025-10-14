from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    average_precision_score, fbeta_score,
)

from utils.constants import (
    DATA_DIR, MODELS_DIR,
    METER_ID_COL, DEVICE,
    CLS_FEATURE_COLS, GRU_DEFAULTS,
    RANDOM_SEED_DEFAULT, TEST_SIZE_DEFAULT,
)
from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)
from utils.CvkFold import SeqDatasetCls, get_cv_threshold_on_train
from utils.ParserCreator import create_common_parser, attach_additional_args, AlgoKind


# ---------------- GRU Model ----------------
class GRUHead(nn.Module):
    """Minimal GRU → Linear head for sequence labeling (one score per timestep)."""
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        # Dropout applies only between stacked GRU layers (ignored for num_layers=1)
        self.gru = nn.GRU(
            input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        out, _ = self.gru(x)                             # [B, T, H or 2H]
        out = self.fc(out)                               # [B, T, 1]
        return out.squeeze(-1)                           # [B, T]


# ---------------- Train & Eval ----------------
def train_and_eval_gru(
    data_dir: Path,
    # sequence windowing
    window: int = 24, stride: int = 1,
    # model
    hidden: int = 64, num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.1,
    # optimization
    batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
    max_epochs: int = 20, early_patience: int = 6,  # used inside CV helper, not in final fit
    # split/control
    seed: int = RANDOM_SEED_DEFAULT, test_size: float = TEST_SIZE_DEFAULT,
    # synthetic leak knobs (dataset builder)
    n_segments_min: int = 2, n_segments_max: int = 5, min_len_frac: float = 0.05,
):
    torch.manual_seed(seed); np.random.seed(seed)

    # 1) Build classification dataframe (adds synthetic persistent leaks)
    df = build_aggregated_dataset(
        data_dir=data_dir,
        test_fraction=test_size,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )
    feature_cols = list(CLS_FEATURE_COLS)

    # 2) Time-aware split per meter (head → train, tail → test)
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])

    # 3) Inner CV on TRAIN → robust probability threshold (forward-chaining, per meter)
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

    # 4) Scale features on FULL TRAIN; transform TEST with the same scaler
    tr_scaled, _, scaler = scale_on_train(train_df, train_df, feature_cols)
    test_scaled = test_df.copy()
    test_scaled[feature_cols] = scaler.transform(test_scaled[feature_cols])

    # 5) Build datasets/loaders (windowed sequences)
    train_ds = SeqDatasetCls(tr_scaled, feature_cols, window=window, stride=stride)
    test_ds  = SeqDatasetCls(test_scaled,  feature_cols, window=window, stride=stride)
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Empty split after windowing. Reduce window/stride or check data coverage.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # 6) Imbalance-aware loss (pos_weight from FULL TRAIN sequences)
    y_seq = train_ds.y.ravel()
    pos_weight = torch.tensor([(len(y_seq) - y_seq.sum()) / (y_seq.sum() + 1e-6)],
                              dtype=torch.float32, device=DEVICE)
    model = GRUHead(in_dim=len(feature_cols), hidden=hidden, num_layers=num_layers,
                    bidirectional=bidirectional, dropout=dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 7) Final train on FULL TRAIN (no early stopping here; threshold comes from CV)
    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.float32)
            loss = criterion(model(xb), yb)
            optim.zero_grad(); loss.backward(); optim.step()

    # Save model + scaler for inference
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "gru_cls.pt")
    joblib.dump(scaler, MODELS_DIR / "gru_cls_scaler.joblib")

    # 8) Test evaluation (apply CV-derived threshold to last-timestep probabilities)
    model.load_state_dict(torch.load(MODELS_DIR / "gru_cls.pt", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        s_list, y_list = [], []
        for xb, yb in test_loader:
            logits_bt = model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy()  # [B, T]
            s_list.append(logits_bt)
            y_list.append(yb.numpy())

    scores_bt = np.concatenate(s_list, axis=0)     # [Nw, T]
    y_bt      = np.concatenate(y_list, axis=0)     # [Nw, T]
    scores_last = scores_bt[:, -1]                 # take last step
    ytrue_last  = y_bt[:, -1]
    probs       = expit(np.clip(scores_last, -20, 20))

    ypred = (probs >= cv_thr).astype(int)
    ytrue = ytrue_last

    # Metrics
    print("Classification report:\n", classification_report(ytrue, ypred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(ytrue, ypred))
    print(f"Accuracy:  {accuracy_score(ytrue, ypred):.4f}")
    print(f"F1:       {f1_score(ytrue, ypred, average='binary'):.4f}")
    print(f"Precision:{precision_score(ytrue, ypred, zero_division=0):.4f}")
    print(f"Recall:   {recall_score(ytrue, ypred, zero_division=0):.4f}")
    print(f"[TEST] PR-AUC={average_precision_score(ytrue, probs):.4f} | "
          f"F2={fbeta_score(ytrue, ypred, beta=2.0, zero_division=0):.4f} | thr={cv_thr:.3f}")

    return {
        "task": "cls", "model": "GRU",
        "accuracy": float(accuracy_score(ytrue, ypred)),
        "precision": float(precision_score(ytrue, ypred, zero_division=0)),
        "recall": float(recall_score(ytrue, ypred, zero_division=0)),
        "f1": float(f1_score(ytrue, ypred, average="binary", zero_division=0)),
        "f2": float(fbeta_score(ytrue, ypred, beta=2.0, zero_division=0)),
        "ap": float(average_precision_score(ytrue, probs)),
        "thr": float(cv_thr),
    }


# ---------------- CLI ----------------
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
        data_dir=Path(args.data_dir),
        window=args.window,
        stride=args.stride,
        hidden=args.hidden,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
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
    print(results)