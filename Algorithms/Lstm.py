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
    mean_absolute_error, mean_squared_error, precision_score, recall_score,
)
import matplotlib.pyplot as plt

# ---- Project utilities / constants
from utils.constants import (
    DATA_DIR, MODELS_DIR,
    METER_ID_COL, TIMESTAMP_COL,
    LEAK_FLAG_COL, FORECAST_Y_COL,
    CLS_FEATURE_COLS, DELTA_VOLUME_COL, VOLUME_READING_COL
)

from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)

# Use GPU if available; otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#region Datasets for cls & forecast
class SeqDatasetCls(Dataset):
    """
    Classification dataset: builds rolling windows per meter.
    - Each window is a sequence of length `window` over `feature_cols`.
    - The label (0/1) is the LEAK_FLAG at the *last* timestamp in the window.

    Why label at the last step? Because we want the model to say:
    "Given the past `window` steps, is there a leak now?"
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], window: int = 24, stride: int = 1):
        # Build all windows up front so __getitem__ is simple and fast.
        self.X, self.y = self._build_sequences(df, feature_cols, window, stride)

    @staticmethod
    def _build_sequences(df: pd.DataFrame, feature_cols: list[str], window: int, stride: int):
        Xs, ys = [], []
        # Group by meter so we never mix different meters in one window
        for _, g in df.sort_values([METER_ID_COL, TIMESTAMP_COL]).groupby(METER_ID_COL, sort=False):
            g = g.reset_index(drop=True)
            arr = g[feature_cols].to_numpy(dtype=np.float32)    # features for this meter
            labels = g[LEAK_FLAG_COL].to_numpy(dtype=np.int64)  # 0/1 leak per row
            n = len(g)
            # Sliding window: start → end with step = stride
            for start in range(0, n - window + 1, stride):
                end = start + window
                Xs.append(arr[start:end])      # shape [T, F]
                ys.append(labels[end - 1])     # label at the last time index in the window
        # Handle edge case: not enough data to form a single window
        if not Xs:
            return np.zeros((0, window, len(feature_cols)), np.float32), np.zeros((0,), np.int64)
        return np.stack(Xs), np.array(ys, dtype=np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


class LagSeqDatasetForecast(Dataset):
    """
    Forecasting dataset: each row is already a *lag vector*:
      X = [lag_k,..., lag_2, lag_1] for the chosen target (ΔVolume here),
      y = FORECAST_Y_COL (what we want to predict at time t).

    We reshape X to [T, F] = [lags, 1] to look like a univariate sequence for the LSTM.
    """
    def __init__(self, df: pd.DataFrame, lag_cols: list[str]):
        X = df[lag_cols].to_numpy(dtype=np.float32)                     # shape [N, L]
        self.X = torch.from_numpy(X.reshape(len(df), len(lag_cols), 1)) # shape [N, T=L, F=1]
        self.y = torch.from_numpy(df[FORECAST_Y_COL].to_numpy(dtype=np.float32))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
#endregion


#region LSTM Model
class LSTMHead(nn.Module):
    """
    A minimal LSTM → Linear head:
    - LSTM encodes the sequence into hidden states.
    - We take the last time step's hidden state and pass it through a linear layer.
    - For classification: we’ll apply BCEWithLogitsLoss (so output is a logit).
    - For regression: we’ll apply MSELoss directly on the scalar output.
    """
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.1):
        super().__init__()
        # dropout only applies between stacked LSTM layers; it’s ignored for num_layers=1
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):            # x: [B, T, F]
        out, _ = self.lstm(x)        # out: [B, T, H]
        last = out[:, -1, :]         # take the last time step representation
        return self.fc(last).squeeze(-1)  # [B]
#endregion


#region Model Training & Evaluation
def train_and_eval_lstm(
    data_dir: Path,
    task: str = "cls",                   # "cls" or "forecast"
    # shared
    window: int = 24, stride: int = 1,   # window length and step (stride used in classification only)
    hidden: int = 64, num_layers: int = 1, bidirectional: bool = False,
    batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
    max_epochs: int = 20, early_patience: int = 4,
    seed: int = 42, test_size: float = 0.30,
    # classification builder knobs
    leak_fraction: float = 0.30,         # fraction of the WHOLE series that will be leak points
    n_segments_min: int = 1,             # we create this many (or up to max) leak segments per meter
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,          # each segment must be at least this fraction of the series
    # forecasting builder knobs
    residual_threshold_pct: float = 97.5 # percentile for residual-based detection threshold
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
    if task == "cls":
        # 1) Build classification frame: random leak placement across the series
        df = build_aggregated_dataset(
            data_dir=data_dir, use_forecast=False,
            leak_mode="variable",                  # use variable leak profile
            leak_fraction=leak_fraction,           # how many samples become leaks overall
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

        # If your window is too large vs. series length, you can end up with zero samples
        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            raise RuntimeError("Empty split after windowing. Reduce window/stride or check data coverage.")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

        # 5) Handle class imbalance by weighting positive class in BCE
        y_seq = train_ds.y
        pos_weight = torch.tensor([(len(y_seq) - y_seq.sum()) / (y_seq.sum() + 1e-6)],
                                  dtype=torch.float32, device=DEVICE)

        model = LSTMHead(in_dim=len(feature_cols), hidden=hidden, num_layers=num_layers,
                         bidirectional=bidirectional, dropout=0.1).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 6) Train with early stopping (val F1-macro improves)
        best, patience = -1.0, 0
        for ep in range(1, max_epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, dtype=torch.float32)
                yb = yb.to(DEVICE, dtype=torch.float32)
                loss = criterion(model(xb), yb)
                optim.zero_grad(); loss.backward(); optim.step()

            # Validate using F1-macro (balances precision/recall and classes)
            model.eval()
            with torch.no_grad():
                vscores, vtrue = [], []
                for xb, yb in val_loader:
                    vscores.append(model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                    vtrue.append(yb.numpy())
                vscores = np.concatenate(vscores); vtrue = np.concatenate(vtrue)
                # Convert logits → probabilities via sigmoid, then threshold at 0.5
                vpred = (1/(1+np.exp(-vscores)) >= 0.5).astype(int)
                val_f1m = f1_score(vtrue, vpred, average="macro")

            print(f"[Epoch {ep:02d}] Val F1-macro={val_f1m:.4f}")
            if val_f1m > best + 1e-4:
                best, patience = val_f1m, 0
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / "lstm_cls.pt")
                joblib.dump(scaler, MODELS_DIR / "lstm_cls_scaler.joblib")
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
            scores = np.concatenate(scores); ytrue = np.concatenate(ytrue)
            ypred = (1/(1+np.exp(-scores)) >= 0.5).astype(int)

        acc = accuracy_score(ytrue, ypred)
        f1m = f1_score(ytrue, ypred, average="macro")
        print("Classification report:\n", classification_report(ytrue, ypred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(ytrue, ypred))
        return {"task":"cls","model":"LSTM","accuracy":float(acc),"f1_macro":float(f1m)}
    #endregion

    #region Forecast Workflow
    else:
        # 1) Build forecasting frame with lags; keep labels to evaluate detection later 
        df = build_aggregated_dataset(
            data_dir=data_dir, use_forecast=True,
            forecast_include_labels=True,       # keep Leak_Flag so we can compute detection metrics
            lags=window,                        # match lag length to LSTM window
            target_col=DELTA_VOLUME_COL,        # predict ΔVolume (works well for residual anomalies)
        )

        # 2) Time-based split per meter
        train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
        train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

        # Keep *only* lag columns as sequence features
        LAG_COLS = [f"lag_{k}" for k in range(window, 0, -1)]

        # 3) SCALE features on TRAIN-CORE ONLY (avoid leakage)
        tr_scaled, _dummy, scaler_X = scale_on_train(train_core_df, train_core_df, LAG_COLS)
        val_scaled = val_df.copy()
        val_scaled[LAG_COLS] = scaler_X.transform(val_scaled[LAG_COLS])
        test_scaled = test_df.copy()
        test_scaled[LAG_COLS] = scaler_X.transform(test_scaled[LAG_COLS])

        # 4) SCALE target (y) as well (helps LSTM training stability)
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        tr_scaled[FORECAST_Y_COL]  = y_scaler.fit_transform(tr_scaled[[FORECAST_Y_COL]])
        val_scaled[FORECAST_Y_COL] = y_scaler.transform(val_scaled[[FORECAST_Y_COL]])
        test_scaled[FORECAST_Y_COL]= y_scaler.transform(test_scaled[[FORECAST_Y_COL]])

        # 5) Build datasets/loaders
        train_ds = LagSeqDatasetForecast(tr_scaled, LAG_COLS)
        val_ds   = LagSeqDatasetForecast(val_scaled, LAG_COLS)
        test_ds  = LagSeqDatasetForecast(test_scaled, LAG_COLS)

        if len(train_ds)==0 or len(val_ds)==0 or len(test_ds)==0:
            raise RuntimeError("Empty split in forecast mode. Check window/lags or data coverage.")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

        # 6) Model / optimizer / loss for regression
        model = LSTMHead(in_dim=1, hidden=hidden, num_layers=num_layers,
                         bidirectional=bidirectional, dropout=0.1).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        # 7) Train with early stopping on Val RMSE
        best_rmse, patience = np.inf, 0
        for ep in range(1, max_epochs + 1):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, dtype=torch.float32)
                yb = yb.to(DEVICE, dtype=torch.float32)
                pred = model(xb)
                loss = criterion(pred, yb)
                optim.zero_grad(); loss.backward(); optim.step()
                train_losses.append(loss.item())
            avg_loss = np.mean(train_losses)

            # Validation: compute MAE & RMSE in *scaled* space (consistent during training)
            model.eval()
            with torch.no_grad():
                pv, tv = [], []
                for xb, yb in val_loader:
                    pv.append(model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                    tv.append(yb.numpy())
                pv = np.concatenate(pv); tv = np.concatenate(tv)
                mae = mean_absolute_error(tv, pv)
                rmse = float(np.sqrt(mean_squared_error(tv, pv)))

            print(f"[Epoch {ep:02d}] Train Loss={avg_loss:.4f} | Val MAE={mae:.4f} RMSE={rmse:.4f}")
            if rmse + 1e-8 < best_rmse:
                best_rmse, patience = rmse, 0
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / "lstm_forecast.pt")
                joblib.dump(scaler_X, MODELS_DIR / "lstm_forecast_scaler.joblib")
            else:
                patience += 1
                if patience >= early_patience:
                    print("[INFO] Early stopping."); break

        # 8) Test regression metrics (in ORIGINAL UNITS for interpretability)
        model.load_state_dict(torch.load(MODELS_DIR / "lstm_forecast.pt", map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            pt, tt = [], []
            for xb, yb in test_loader:
                pred = model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy()
                pt.append(pred)
                tt.append(yb.numpy())
            yhat_te_scaled = np.concatenate(pt)
            y_test_scaled  = np.concatenate(tt)

            # Inverse-transform predictions and targets to original units
            yhat_te = y_scaler.inverse_transform(yhat_te_scaled.reshape(-1,1)).ravel()
            y_test  = y_scaler.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()

            mae  = mean_absolute_error(y_test, yhat_te)
            rmse = float(np.sqrt(mean_squared_error(y_test, yhat_te)))
            print(f"[TEST] MAE={mae:.4f} RMSE={rmse:.4f}")


            # Residual-based leak detection
            # 1) Compute residuals on TRAIN (scaled) to set a threshold at a percentile.
            # 2) Apply same threshold to TEST residuals (scaled) to get leak flags.
            # Why scaled? Because threshold was learned in scaled space; keep it consistent.
            det = {}
            with torch.no_grad():
                pt, tt = [], []
                for xb, yb in train_loader:
                    pred = model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy()
                    pt.append(pred)
                    tt.append(yb.numpy())
                yhat_tr_scaled = np.concatenate(pt)
                y_tr_scaled    = np.concatenate(tt)

            res_tr = np.abs(y_tr_scaled - yhat_tr_scaled)          # TRAIN residuals (scaled)
            thr = float(np.percentile(res_tr, residual_threshold_pct))  # e.g., 97.5th percentile

            res_te = np.abs(y_test_scaled - yhat_te_scaled)        # TEST residuals (scaled)
            pred_leak = (res_te >= thr).astype(int)                 # 1 = detected leak

        # 9) If true leak labels are present, compute detection metrics
        if LEAK_FLAG_COL in test_scaled.columns:
            gt_leak = test_scaled[LEAK_FLAG_COL].to_numpy()
            det = dict(
                det_acc=float(accuracy_score(gt_leak, pred_leak)),
                det_prec=float(precision_score(gt_leak, pred_leak, zero_division=0)),
                det_rec=float(recall_score(gt_leak, pred_leak, zero_division=0)),
                det_f1=float(f1_score(gt_leak, pred_leak, zero_division=0)),
                det_cm=confusion_matrix(gt_leak, pred_leak).tolist(),
                train_residual_threshold=thr,
            )
            print(f"[Leak via residuals] Acc={det['det_acc']:.4f} F1={det['det_f1']:.4f}")

        out = {"task":"forecast","model":"LSTM","mae":float(mae),"rmse":float(rmse)}
        out.update(det)

        # 10) Visualization: Actual vs Predicted + TP/FP/FN markers
        gt_all   = test_scaled[LEAK_FLAG_COL].to_numpy()
        res_te   = np.abs(y_test_scaled - yhat_te_scaled)           # scaled residuals (used for detection)
        pred_all = (res_te >= thr).astype(int)

        # Try to pick a meter that actually has some mistakes (FN/FP) so the plot is informative
        bad_meter = None
        for mid, idx in test_scaled.groupby(METER_ID_COL, sort=False).indices.items():
            m_mask = (test_scaled[METER_ID_COL] == mid).values
            y_true_m = gt_all[m_mask]
            y_pred_m = pred_all[m_mask]
            if ((y_true_m == 1) & (y_pred_m == 0)).any() or ((y_true_m == 0) & (y_pred_m == 1)).any():
                bad_meter = mid
                break

        meter_id = bad_meter if bad_meter is not None else test_scaled[METER_ID_COL].iloc[0]
        mask = (test_scaled[METER_ID_COL] == meter_id).values

        # Time axis for plotting
        timestamps = test_scaled.loc[mask, TIMESTAMP_COL].reset_index(drop=True)

        # ORIGINAL-space series for nicer y-axis units
        y_true_meter = y_test[mask]
        y_pred_meter = yhat_te[mask]

        # SCALED-space residuals for detection (consistent with threshold)
        y_true_meter_sc = y_test_scaled[mask]
        y_pred_meter_sc = yhat_te_scaled[mask]
        residuals_meter_sc = np.abs(y_true_meter_sc - y_pred_meter_sc)
        predicted_leaks = (residuals_meter_sc >= thr).astype(int)

        # Ground-truth leak flags from the dataframe
        actual_leaks = test_scaled.loc[mask, LEAK_FLAG_COL].to_numpy()

        # Confusion breakdown (indices for plotting)
        TP_idx = np.where((actual_leaks == 1) & (predicted_leaks == 1))[0]
        FN_idx = np.where((actual_leaks == 1) & (predicted_leaks == 0))[0]
        FP_idx = np.where((actual_leaks == 0) & (predicted_leaks == 1))[0]

        plt.figure(figsize=(15, 6))
        plt.plot(timestamps, y_true_meter, label="Actual ΔVolume", color="blue")
        plt.plot(timestamps, y_pred_meter, label="Predicted ΔVolume", color="orange")
        plt.scatter(timestamps.iloc[TP_idx], y_true_meter[TP_idx], color="green",
                    label="TP (Detected Leak)", s=40, alpha=0.8)
        plt.scatter(timestamps.iloc[FN_idx], y_true_meter[FN_idx], color="red", marker="o",
                    edgecolors="black", label="FN (Missed Leak)", s=80, alpha=0.9)
        plt.scatter(timestamps.iloc[FP_idx], y_pred_meter[FP_idx], color="purple", marker="x",
                    label="FP (False Alarm)", s=60, alpha=0.8)

        plt.title(f"Meter {meter_id} — Forecast vs Actual with Leak Detection (scaled-space threshold)")
        plt.xlabel("Time")
        plt.ylabel("ΔVolume (liters)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return out
    #endregion
#endregion


# CLI
def str2bool(v: str) -> bool:
    # Convenience parser for true/false flags in CLI
    return v.lower() in {"1","true","t","yes","y"}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LSTM for leak detection — Classification or Forecasting.")
    p.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    p.add_argument("--task", type=str, default="cls", choices=["cls","forecast"])

    # Shared hyperparameters:
    # - window: sequence length (also lags in forecast mode)
    # - stride: how far we move the window each step (classification only)
    p.add_argument("--window", type=int, default=3*24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--bidirectional", type=str, default="false")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--early_patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.30)

    # Classification knobs (affect how synthetic leaks are injected):
    p.add_argument("--leak_fraction", type=float, default=0.30)
    p.add_argument("--n_segments_min", type=int, default=1)
    p.add_argument("--n_segments_max", type=int, default=3)
    p.add_argument("--min_len_frac", type=float, default=0.05)

    # Forecasting detection knob:
    # - residual_threshold_pct: what percentile of TRAIN residuals sets the alarm threshold.
    #   Higher => fewer but stronger alerts (higher precision, lower recall).
    p.add_argument("--residual_threshold_pct", type=float, default=85)

    args = p.parse_args()

    res = train_and_eval_lstm(
        data_dir=Path(args.data_dir),
        task=args.task,
        window=args.window,
        stride=args.stride,
        hidden=args.hidden,
        num_layers=args.num_layers,
        bidirectional=str2bool(args.bidirectional),
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_patience=args.early_patience,
        seed=args.seed,
        test_size=args.test_size,
        leak_fraction=args.leak_fraction,
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
        residual_threshold_pct=args.residual_threshold_pct,
    )
    print(res)
