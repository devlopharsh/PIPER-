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
    METER_ID_COL, TIMESTAMP_COL,
    LEAK_FLAG_COL, FORECAST_Y_COL,
    CLS_FEATURE_COLS, CALENDAR_COLS,
    MLP_DEFAULTS,
    RANDOM_SEED_DEFAULT, TEST_SIZE_DEFAULT,
    RESIDUAL_THRESHOLD_PCT_DEFAULT,VOLUME_READING_COL, DELTA_VOLUME_COL
)

from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- Helpers -----------------------------
def set_seed(seed: int = RANDOM_SEED_DEFAULT) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_feature_cols_for_task(df: pd.DataFrame, task: str) -> List[str]:
    """
    For cls: fixed baseline features (CLS_FEATURE_COLS).
    For forecast: all lag_* and roll* (+ calendar if present).
    """
    if task == "cls":
        return list(CLS_FEATURE_COLS)

    cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll")]
    for cal in CALENDAR_COLS:
        if cal in df.columns:
            cols.append(cal)
    return cols


def to_xy(df: pd.DataFrame, feature_cols: List[str], task: str):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    if task == "cls":
        y = df[LEAK_FLAG_COL].to_numpy(dtype=np.float32)  # BCEWithLogits expects float {0,1}
    else:
        y = df[FORECAST_Y_COL].to_numpy(dtype=np.float32)
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
    task: str = "cls",                    # "cls" | "forecast"

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

    # Forecasting data knobs
    target_col: str = DELTA_VOLUME_COL,
    lags: int = 24,
    rolling_windows: Tuple[int, ...] = (6, 24),
    residual_threshold_pct: float = RESIDUAL_THRESHOLD_PCT_DEFAULT,
):
    import pandas as pd  # local to keep import order tidy

    set_seed(seed)

    # ------------------------- 1) Build dataset -------------------------
    if task == "cls":
        df = build_aggregated_dataset(
            data_dir=data_dir,
            use_forecast=False,
            leak_mode="variable",
            test_fraction=test_size,
            n_segments_range=(n_segments_min, n_segments_max),
            min_len_frac=min_len_frac,
        )
    else:
        df = build_aggregated_dataset(
            data_dir=data_dir,
            use_forecast=True,
            target_col=target_col,
            lags=lags,
            rolling_windows=rolling_windows,
            forecast_include_labels=True,     # keep Leak_Flag for train filtering & test detection
        )

    # ------------------------- 2) Time-based split -------------------------
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])

    # Small validation from TRAIN (last 10% per meter)
    train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

    # ------------------------- 3) Scale on TRAIN-CORE only -------------------------
    feat_cols = get_feature_cols_for_task(df, task)
    tr_core_scaled, _dummy, scaler = scale_on_train(train_core_df, train_core_df, feat_cols)
    val_scaled = val_df.copy();  val_scaled[feat_cols]  = scaler.transform(val_scaled[feat_cols])
    te_scaled  = test_df.copy(); te_scaled[feat_cols]   = scaler.transform(te_scaled[feat_cols])

    # ------------------------- 4) DataLoaders -------------------------
    Xtr, ytr = to_xy(tr_core_scaled, feat_cols, task)
    Xva, yva = to_xy(val_scaled,     feat_cols, task)
    Xte, yte = to_xy(te_scaled,      feat_cols, task)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=batch_size, shuffle=False, num_workers=0)

    # ------------------------- 5) Model / Loss / Optim -------------------------
    model = MLP(in_dim=len(feat_cols), hidden=list(hidden), p_drop=dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if task == "cls":
        # class imbalance weight from TRAIN only
        pos = float(ytr.sum())
        pos_weight = (len(ytr) - pos) / (pos + 1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE))

        best_metric = -1.0
        patience = 0

        for ep in range(1, max_epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                loss = criterion(model(xb), yb)
                optim.zero_grad(); loss.backward(); optim.step()

            # ---- validation F1 ----
            model.eval()
            with torch.no_grad():
                logits, ys = [], []
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    logits.append(model(xb).cpu().numpy())
                    ys.append(yb.numpy())
                y_scores = np.concatenate(logits)
                y_true   = np.concatenate(ys)
                y_pred   = (1/(1+np.exp(-y_scores)) >= 0.5).astype(int)
                val_f1m  = f1_score(y_true, y_pred, average="macro")

            print(f"[Epoch {ep:02d}] Val F1-macro={val_f1m:.4f}")
            if val_f1m > best_metric + 1e-4:
                best_metric = val_f1m; patience = 0
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / "mlp_cls.pt")
                joblib.dump(scaler, MODELS_DIR / "mlp_cls_scaler.joblib")
            else:
                patience += 1
                if patience >= early_patience:
                    print("[INFO] Early stopping.")
                    break

        # ---- Final test ----
        model.load_state_dict(torch.load(MODELS_DIR / "mlp_cls.pt", map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            logits, ys = [], []
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                logits.append(model(xb).cpu().numpy())
                ys.append(yb.numpy())
            y_scores = np.concatenate(logits)
            y_true   = np.concatenate(ys)
            y_pred   = (1/(1+np.exp(-y_scores)) >= 0.5).astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, zero_division=0))
        cm   = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)

        return {
            "task": "cls",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_macro": f1m,
            "cm": cm,
            "report": report,
            "model_path": str(MODELS_DIR / "mlp_cls.pt"),
            "scaler_path": str(MODELS_DIR / "mlp_cls_scaler.joblib"),
        }

    # ---------------------- Forecast (regression) ----------------------
    criterion = nn.MSELoss()
    best_metric = np.inf
    patience = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optim.zero_grad(); loss.backward(); optim.step()

        # ---- validation MAE ----
        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                preds.append(model(xb).cpu().numpy())
                ys.append(yb.numpy())
            yhat = np.concatenate(preds)
            yval = np.concatenate(ys)
            val_mae = mean_absolute_error(yval, yhat)

        print(f"[Epoch {ep:02d}] Val MAE={val_mae:.4f}")
        if val_mae + 1e-6 < best_metric:
            best_metric = val_mae; patience = 0
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODELS_DIR / "mlp_forecast.pt")
            joblib.dump(scaler, MODELS_DIR / "mlp_forecast_scaler.joblib")
        else:
            patience += 1
            if patience >= early_patience:
                print("[INFO] Early stopping.")
                break

    # ---- Final test ----
    model.load_state_dict(torch.load(MODELS_DIR / "mlp_forecast.pt", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        preds_tr, ys_tr = [], []
        for xb, yb in DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch_size, shuffle=False):
            xb = xb.to(DEVICE)
            preds_tr.append(model(xb).cpu().numpy()); ys_tr.append(yb.numpy())
        yhat_tr = np.concatenate(preds_tr); ytr_all = np.concatenate(ys_tr)

        preds_te, ys_te = [], []
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds_te.append(model(xb).cpu().numpy()); ys_te.append(yb.numpy())
        yhat_te = np.concatenate(preds_te); yte_all = np.concatenate(ys_te)

    mae  = float(mean_absolute_error(yte_all, yhat_te))
    rmse = float(np.sqrt(mean_squared_error(yte_all, yhat_te)))

    # Residual threshold from TRAIN
    res_tr = np.abs(ytr_all - yhat_tr)
    thr = float(np.percentile(res_tr, residual_threshold_pct))

    out = {
        "task": "forecast",
        "mae": mae,
        "rmse": rmse,
        "train_residual_threshold": thr,
        "model_path": str(MODELS_DIR / "mlp_forecast.pt"),
        "scaler_path": str(MODELS_DIR / "mlp_forecast_scaler.joblib"),
    }

    # Optional residual-based leak detection if labels exist in TEST
    if LEAK_FLAG_COL in te_scaled.columns:
        res_te = np.abs(yte_all - yhat_te)
        pred_leak = (res_te >= thr).astype(int)
        gt = te_scaled[LEAK_FLAG_COL].to_numpy()
        out.update({
            "det_acc": float(accuracy_score(gt, pred_leak)),
            "det_prec": float(precision_score(gt, pred_leak, zero_division=0)),
            "det_rec": float(recall_score(gt, pred_leak, zero_division=0)),
            "det_f1": float(f1_score(gt, pred_leak, zero_division=0)),
            "det_cm": confusion_matrix(gt, pred_leak).tolist(),
        })

    # Persist threshold
    thr_path = MODELS_DIR / "mlp_forecast_residual_threshold.txt"
    with open(thr_path, "w") as f:
        f.write(str(thr))
    out["threshold_path"] = str(thr_path)
    return out


# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-mode MLP — classification or forecasting")
    parser.add_argument("--data_dir", type=Path, default=str(DATA_DIR))
    parser.add_argument("--task", type=str, default="cls", choices=["cls", "forecast"])

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
    parser.add_argument("--n_segments_min", type=int, default=1)
    parser.add_argument("--n_segments_max", type=int, default=3)
    parser.add_argument("--min_len_frac", type=float, default=0.05)

    # Forecasting builder knobs
    parser.add_argument("--target_col", type=str, default=DELTA_VOLUME_COL, choices=[VOLUME_READING_COL, DELTA_VOLUME_COL])
    parser.add_argument("--lags", type=int, default=3*24)
    parser.add_argument("--rolling_windows", type=str, default="6,24")
    parser.add_argument("--residual_threshold_pct", type=float, default=RESIDUAL_THRESHOLD_PCT_DEFAULT)

    args = parser.parse_args()
    rw = tuple(int(x) for x in args.rolling_windows.split(",") if x.strip())

    results = train_and_eval_mlp(
        data_dir=args.data_dir,
        task=args.task,
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
        target_col=args.target_col,
        lags=args.lags,
        rolling_windows=rw,
        residual_threshold_pct=args.residual_threshold_pct,
    )

    print(json.dumps(results, indent=2))
