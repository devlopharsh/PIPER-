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
from utils.ThresholdComputer import find_stable_threshold
from sklearn.metrics import average_precision_score, fbeta_score, precision_recall_curve
from scipy.special import expit


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
    set_seed(seed)

    # ------------------------- BUILD DATA -------------------------
    if task == "cls":
        # Build classification frame with synthetic persistent leaks
        df = build_aggregated_dataset(
            data_dir=data_dir,
            use_forecast=False,
            leak_mode="variable",
            test_fraction=test_size,
            n_segments_range=(n_segments_min, n_segments_max),
            min_len_frac=min_len_frac,
        )

        # Time split per meter → train/test; then train_core/val
        train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
        train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

        # Feature set (classification)
        feat_cols = [c for c in CLS_FEATURE_COLS if c in train_core_df.columns and c != FORECAST_Y_COL]

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
        Xtr, ytr = to_xy(tr_core_scaled, feat_cols, task="cls")
        Xva, yva = to_xy(val_scaled,     feat_cols, task="cls")
        Xte, yte = to_xy(te_scaled,      feat_cols, task="cls")

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

    # ------------------------- FORECAST -------------------------
    else:
        # Build forecasting frame (with labels kept for detection)
        df = build_aggregated_dataset(
            data_dir=data_dir,
            use_forecast=True,
            target_col=target_col,
            lags=lags,
            rolling_windows=rolling_windows,
            forecast_include_labels=True,
        )

        # Time split per meter
        train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
        train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

        # Feature set (forecast): lags + rolling + calendar
        lag_cols  = [c for c in df.columns if c.startswith("lag_")]
        roll_cols = [c for c in df.columns if c.startswith("roll")]
        cal_cols  = [c for c in CALENDAR_COLS if c in df.columns]
        feat_cols = [c for c in (lag_cols + roll_cols + cal_cols) if c != FORECAST_Y_COL]

        # Scale X on TRAIN-CORE only
        tr_core_scaled, _dummy, scaler = scale_on_train(train_core_df, train_core_df, feat_cols)
        val_scaled = val_df.copy();  val_scaled[feat_cols] = scaler.transform(val_scaled[feat_cols])
        te_scaled  = test_df.copy(); te_scaled[feat_cols]  = scaler.transform(te_scaled[feat_cols])

        # y-scaling (recommended)
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        tr_core_scaled[FORECAST_Y_COL] = y_scaler.fit_transform(tr_core_scaled[[FORECAST_Y_COL]])
        val_scaled[FORECAST_Y_COL]     = y_scaler.transform(val_scaled[[FORECAST_Y_COL]])
        te_scaled[FORECAST_Y_COL]      = y_scaler.transform(te_scaled[[FORECAST_Y_COL]])

        # Lock to common feature columns
        common_cols = [c for c in feat_cols
                       if c in tr_core_scaled.columns
                       and c in val_scaled.columns
                       and c in te_scaled.columns]
        feat_cols = common_cols

        # Dataloaders
        Xtr, ytr = to_xy(tr_core_scaled, feat_cols, task="forecast")
        Xva, yva = to_xy(val_scaled,     feat_cols, task="forecast")
        Xte, yte = to_xy(te_scaled,      feat_cols, task="forecast")

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

        # Optim / loss
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        # Train + early stop on Val MAE
        best_metric = np.inf; patience = 0
        for ep in range(1, max_epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                optim.zero_grad(); loss.backward(); optim.step()

            model.eval()
            with torch.no_grad():
                preds, ys = [], []
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    preds.append(model(xb).cpu().numpy())
                    ys.append(yb.numpy())
                yhat = np.concatenate(preds); yval = np.concatenate(ys)
                val_mae = mean_absolute_error(yval, yhat)

            print(f"[Epoch {ep:02d}] Val MAE={val_mae:.4f}")
            if val_mae + 1e-6 < best_metric:
                best_metric = val_mae; patience = 0
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / "mlp_forecast.pt")
                joblib.dump(scaler,   MODELS_DIR / "mlp_forecast_x_scaler.joblib")
                joblib.dump(y_scaler, MODELS_DIR / "mlp_forecast_y_scaler.joblib")
            else:
                patience += 1
                if patience >= early_patience:
                    print("[INFO] Early stopping."); break

        # ---- Final test ----
        model.load_state_dict(torch.load(MODELS_DIR / "mlp_forecast.pt", map_location=DEVICE))
        model.eval()

        def batched_predict(X_np, bs):
            preds = []
            with torch.no_grad():
                for i in range(0, len(X_np), bs):
                    xb = torch.from_numpy(X_np[i:i+bs]).to(DEVICE)
                    preds.append(model(xb).cpu().numpy())
            return np.concatenate(preds)

        # Arrays (scaled)
        yhat_tr_sc = batched_predict(Xtr, batch_size)
        yhat_te_sc = batched_predict(Xte, batch_size)

        # Human-readable metrics in original units
        y_te    = y_scaler.inverse_transform(yte.reshape(-1,1)).ravel()
        yhat_te = y_scaler.inverse_transform(yhat_te_sc.reshape(-1,1)).ravel()
        mae  = float(mean_absolute_error(y_te, yhat_te))
        rmse = float(np.sqrt(mean_squared_error(y_te, yhat_te)))

        # Residual threshold from TRAIN (scaled space)
        res_tr_sc = np.abs(ytr - yhat_tr_sc)
        thr = float(np.percentile(res_tr_sc, residual_threshold_pct))

        # TEST detection (scaled space)
        res_te_sc = np.abs(yte - yhat_te_sc)
        pred_leak = (res_te_sc >= thr).astype(int)

        out = {
            "task": "forecast",
            "mae": mae,
            "rmse": rmse,
            "train_residual_threshold": thr,
            "model_path": str(MODELS_DIR / "mlp_forecast.pt"),
            "x_scaler_path": str(MODELS_DIR / "mlp_forecast_x_scaler.joblib"),
            "y_scaler_path": str(MODELS_DIR / "mlp_forecast_y_scaler.joblib"),
        }

        # Optional detection metrics if labels exist
        if LEAK_FLAG_COL in te_scaled.columns:
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
    parser.add_argument("--n_segments_min", type=int, default=2)
    parser.add_argument("--n_segments_max", type=int, default=5)
    parser.add_argument("--min_len_frac", type=float, default=0.05)

    # Forecasting builder knobs
    parser.add_argument("--target_col", type=str, default=DELTA_VOLUME_COL, choices=[ DELTA_VOLUME_COL])
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
