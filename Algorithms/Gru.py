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
    LEAK_FLAG_COL, FORECAST_Y_COL,
    CLS_FEATURE_COLS, DELTA_VOLUME_COL, GRU_DEFAULTS, RESIDUAL_THRESHOLD_PCT_DEFAULT, RANDOM_SEED_DEFAULT,
    TEST_SIZE_DEFAULT
    
)
from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)
from utils.ThresholdComputer import find_stable_threshold
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reuse your existing dataset classes from LSTM code ---
class SeqDatasetCls(torch.utils.data.Dataset):
    def __init__(self, df, feature_cols, window=24, stride=1):
        self.X, self.y = self._build_sequences(df, feature_cols, window, stride)

    @staticmethod
    def _build_sequences(df, feature_cols, window, stride):
        Xs, ys = [], []
        for _, g in df.sort_values([METER_ID_COL, TIMESTAMP_COL]).groupby(METER_ID_COL, sort=False):
            g = g.reset_index(drop=True)
            arr = g[feature_cols].to_numpy(dtype=np.float32)
            labels = g[LEAK_FLAG_COL].to_numpy(dtype=np.int64)
            n = len(g)
            for start in range(0, n - window + 1, stride):
                end = start + window
                Xs.append(arr[start:end])
                ys.append(labels[start:end])
        if not Xs:
            return (np.zeros((0, window, len(feature_cols)), np.float32),
                    np.zeros((0, window), np.int64))
        return np.stack(Xs), np.stack(ys)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

class LagSeqDatasetForecast(torch.utils.data.Dataset):
    def __init__(self, df, lag_cols):
        X = df[lag_cols].to_numpy(dtype=np.float32)
        self.X = torch.from_numpy(X.reshape(len(df), len(lag_cols), 1))
        self.y = torch.from_numpy(df[FORECAST_Y_COL].to_numpy(dtype=np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_and_eval_gru(
    data_dir: Path,
    task: str = "cls",
    # shared
    window: int = 24, stride: int = 1,
    hidden: int = 64, num_layers: int = 1, bidirectional: bool = False,
    batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
    max_epochs: int = 20, early_patience: int = 6,
    seed: int = 42, test_size: float = 0.30, dropout: float = 0.1,
    # classification data knobs
    n_segments_min: int = 2, n_segments_max: int = 5, min_len_frac: float = 0.05,
    # forecast detection knob
    residual_threshold_pct: float = 85.0
):
    torch.manual_seed(seed); np.random.seed(seed)

    if task == "cls":
        # 1) Build classification df with synthetic persistent leaks
        df = build_aggregated_dataset(
            data_dir=data_dir, use_forecast=False,
            leak_mode="variable",
            test_fraction=test_size,
            n_segments_range=(n_segments_min, n_segments_max),
            min_len_frac=min_len_frac,
        )
        feature_cols = list(CLS_FEATURE_COLS)

        # 2) Time-based per-meter split → train/test; then train_core/val
        train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
        train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

        # 3) Scale on TRAIN-CORE only
        tr_core_scaled, _dummy, scaler = scale_on_train(train_core_df, train_core_df, feature_cols)
        val_scaled = val_df.copy();   val_scaled[feature_cols]  = scaler.transform(val_scaled[feature_cols])
        test_scaled = test_df.copy(); test_scaled[feature_cols] = scaler.transform(test_scaled[feature_cols])

        # 4) Sequences
        train_ds = SeqDatasetCls(tr_core_scaled, feature_cols, window=window, stride=stride)
        val_ds   = SeqDatasetCls(val_scaled,     feature_cols, window=window, stride=stride)
        test_ds  = SeqDatasetCls(test_scaled,    feature_cols, window=window, stride=stride)
        if len(train_ds)==0 or len(val_ds)==0 or len(test_ds)==0:
            raise RuntimeError("Empty split after windowing. Adjust window/stride or check data coverage.")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        # 5) Handle imbalance via pos_weight in BCE
        y_seq = train_ds.y.ravel()
        pos_w = (len(y_seq) - y_seq.sum()) / (y_seq.sum() + 1e-6)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=DEVICE))

        model = GRUHead(in_dim=len(feature_cols), hidden=hidden, num_layers=num_layers,
                        bidirectional=bidirectional, dropout=dropout).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
                vscores = np.concatenate(vscores).ravel()
                vtrue   = np.concatenate(vtrue).ravel()

            vprobs = expit(np.clip(vscores, -20, 20))
            beta = 1
            stable_thr, thr_list = find_stable_threshold(vtrue, vprobs, beta=beta, n_splits=5)
            vpred = (vprobs >= stable_thr).astype(int)
            vF = fbeta_score(vtrue, vpred, beta=beta, average="binary", zero_division=0)
            vAP = average_precision_score(vtrue, vprobs)
            print(f"[Epoch {ep:02d}] Val AP={vAP:.4f} | Val F{beta}={vF:.4f} @thr={stable_thr:.3f}")
            
            metric_for_early_stop = vF

            if metric_for_early_stop > best + 1e-4:
                best, patience = vF, 0
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / "gru_cls.pt")
                joblib.dump(scaler, MODELS_DIR / "gru_cls_scaler.joblib")
                with open(MODELS_DIR / "gru_cls_best_thr.txt", "w") as f:
                    f.write(str(stable_thr))
            else:
                patience += 1
                if patience >= early_patience:
                    print("[INFO] Early stopping."); break

        # Test
        model.load_state_dict(torch.load(MODELS_DIR / "gru_cls.pt", map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            scores, ytrue = [], []
            for xb, yb in test_loader:
                scores.append(model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                ytrue.append(yb.numpy())
            scores = np.concatenate(scores).ravel()
            ytrue  = np.concatenate(ytrue).ravel()

        probs = expit(np.clip(scores, -20, 20))
        with open(MODELS_DIR / "gru_cls_best_thr.txt") as f:
            best_thr = float(f.read().strip())
        ypred = (probs >= best_thr).astype(int)
        
        print(f"[INFO] CV thresholds={thr_list}, using stable_thr={best_thr:.3f}")

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

    # ---------------- Forecast (regression + residual detection) ----------------
    else:
        # 1) Build forecasting frame with lags
        df = build_aggregated_dataset(
            data_dir=data_dir, use_forecast=True,
            forecast_include_labels=True,
            lags=window, target_col=DELTA_VOLUME_COL,
        )

        # 2) Time-based per-meter split
        train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
        train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

        LAG_COLS = [f"lag_{k}" for k in range(window, 0, -1)]

        # 3) Scale X on TRAIN-CORE only
        tr_scaled, _dummy, scaler_X = scale_on_train(train_core_df, train_core_df, LAG_COLS)
        val_scaled = val_df.copy();   val_scaled[LAG_COLS]  = scaler_X.transform(val_scaled[LAG_COLS])
        test_scaled = test_df.copy(); test_scaled[LAG_COLS] = scaler_X.transform(test_scaled[LAG_COLS])

        # Scale y for stability
        from sklearn.preprocessing import StandardScaler
        y_scaler = StandardScaler()
        for d in [tr_scaled, val_scaled, test_scaled]:
            d[FORECAST_Y_COL] = d[FORECAST_Y_COL].astype(float)
        tr_scaled[FORECAST_Y_COL]  = y_scaler.fit_transform(tr_scaled[[FORECAST_Y_COL]])
        val_scaled[FORECAST_Y_COL] = y_scaler.transform(val_scaled[[FORECAST_Y_COL]])
        test_scaled[FORECAST_Y_COL]= y_scaler.transform(test_scaled[[FORECAST_Y_COL]])

        # 4) Datasets
        train_ds = LagSeqDatasetForecast(tr_scaled, LAG_COLS)
        val_ds   = LagSeqDatasetForecast(val_scaled, LAG_COLS)
        test_ds  = LagSeqDatasetForecast(test_scaled, LAG_COLS)
        if len(train_ds)==0 or len(val_ds)==0 or len(test_ds)==0:
            raise RuntimeError("Empty split in forecast mode. Check window/lags or data coverage.")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        # 5) Model + MSE
        model = GRUHead(in_dim=1, hidden=hidden, num_layers=num_layers,
                        bidirectional=bidirectional, dropout=dropout).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_rmse, patience = np.inf, 0
        for ep in range(1, max_epochs + 1):
            model.train()
            tloss = []
            for xb, yb in train_loader:
                xb = xb.to(DEVICE, dtype=torch.float32)
                yb = yb.to(DEVICE, dtype=torch.float32)
                pred = model(xb)
                loss = criterion(pred, yb)
                optim.zero_grad(); loss.backward(); optim.step()
                tloss.append(loss.item())

            # Validate in scaled space (consistent)
            model.eval()
            with torch.no_grad():
                pv, tv = [], []
                for xb, yb in val_loader:
                    pv.append(model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                    tv.append(yb.numpy())
                pv = np.concatenate(pv); tv = np.concatenate(tv)
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                rmse = float(np.sqrt(mean_squared_error(tv, pv)))
                mae  = float(mean_absolute_error(tv, pv))
            print(f"[Epoch {ep:02d}] Val MAE={mae:.4f} RMSE={rmse:.4f}")

            if rmse + 1e-8 < best_rmse:
                best_rmse, patience = rmse, 0
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / "gru_forecast.pt")
                joblib.dump(scaler_X, MODELS_DIR / "gru_forecast_scaler.joblib")
                joblib.dump(y_scaler,  MODELS_DIR / "gru_forecast_y_scaler.joblib")
            else:
                patience += 1
                if patience >= early_patience:
                    print("[INFO] Early stopping."); break

        # 6) Test in ORIGINAL units
        model.load_state_dict(torch.load(MODELS_DIR / "gru_forecast.pt", map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            pt, tt = [], []
            for xb, yb in test_loader:
                pt.append(model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                tt.append(yb.numpy())
            yhat_te_sc = np.concatenate(pt); y_test_sc = np.concatenate(tt)

        y_scaler = joblib.load(MODELS_DIR / "gru_forecast_y_scaler.joblib")
        yhat_te = y_scaler.inverse_transform(yhat_te_sc.reshape(-1,1)).ravel()
        y_test  = y_scaler.inverse_transform(y_test_sc.reshape(-1,1)).ravel()

        mae  = float(mean_absolute_error(y_test, yhat_te))
        rmse = float(np.sqrt(mean_squared_error(y_test, yhat_te)))

        # Residual-based detection with TRAIN threshold (scaled space)
        with torch.no_grad():
            pt, tt = [], []
            for xb, yb in train_loader:
                pt.append(model(xb.to(DEVICE, dtype=torch.float32)).cpu().numpy())
                tt.append(yb.numpy())
            yhat_tr_sc = np.concatenate(pt); y_tr_sc = np.concatenate(tt)
        res_tr = np.abs(y_tr_sc - yhat_tr_sc)
        thr = float(np.percentile(res_tr, residual_threshold_pct))

        res_te = np.abs(y_test_sc - yhat_te_sc)
        pred_leak = (res_te >= thr).astype(int)

        det = {}
        if LEAK_FLAG_COL in test_scaled.columns:
            gt = test_scaled[LEAK_FLAG_COL].to_numpy()
            det = dict(
                det_acc=float(accuracy_score(gt, pred_leak)),
                det_prec=float(precision_score(gt, pred_leak, zero_division=0)),
                det_rec=float(recall_score(gt, pred_leak, zero_division=0)),
                det_f1=float(f1_score(gt, pred_leak, zero_division=0)),
                det_cm=confusion_matrix(gt, pred_leak).tolist(),
                train_residual_threshold=thr
            )

        return {"task":"forecast","model":"GRU","mae":mae,"rmse":rmse, **det}

# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PyTorch GRU — classification or forecasting")

    ap.add_argument("--data_dir", type=Path, default=Path(str(DATA_DIR)))
    ap.add_argument("--task", type=str, default="cls", choices=["cls", "forecast"])

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

    # --- forecasting knobs ---
    ap.add_argument("--residual_threshold_pct", type=float, default=RESIDUAL_THRESHOLD_PCT_DEFAULT)

    args = ap.parse_args()

    results = train_and_eval_gru(
        data_dir=args.data_dir,
        task=args.task,
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
        residual_threshold_pct=args.residual_threshold_pct,
    )

    print(results)

