from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error,
)

from utils.constants import (
    DATA_DIR, MODELS_DIR,
    METER_ID_COL, TIMESTAMP_COL,
    LEAK_FLAG_COL, FORECAST_Y_COL,
    CLS_FEATURE_COLS,
    GRU_DEFAULTS,
    RANDOM_SEED_DEFAULT, TEST_SIZE_DEFAULT,
    RESIDUAL_THRESHOLD_PCT_DEFAULT,DELTA_VOLUME_COL, VOLUME_READING_COL
)

from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)

# -------------------------
# Seeds
# -------------------------
def set_seed(seed: int = RANDOM_SEED_DEFAULT) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# Sequence helpers
# =========================
def make_sequences_per_meter(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classification windows per meter.
    X: (n_seq, seq_len, n_feat), y: (n_seq,)
    """
    X_list, y_list = [], []
    for _, g in df.groupby(METER_ID_COL, sort=False):
        g = g.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        Xg = g[feature_cols].to_numpy(dtype=np.float32)
        yg = g[target_col].to_numpy(dtype=np.int64)
        n = len(g)
        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            X_list.append(Xg[start:end])
            y_list.append(yg[end - 1])    # label at window end
    if not X_list:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    return np.stack(X_list), np.array(y_list)


def make_lag_sequences(
    df: pd.DataFrame,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Forecasting sequences from lag columns.
    X: (N, T=seq_len, F=1), y: (N,), leak (optional): (N,)
    Uses columns lag_seq_len .. lag_1 in that order (oldest->newest).
    """
    lag_cols = [f"lag_{k}" for k in range(seq_len, 0, -1)]
    X = df[lag_cols].to_numpy(dtype=np.float32).reshape(len(df), seq_len, 1)
    y = df[FORECAST_Y_COL].to_numpy(dtype=np.float32)
    leak = df[LEAK_FLAG_COL].to_numpy(dtype=np.int64) if LEAK_FLAG_COL in df.columns else None
    return X, y, leak


def class_weight_from_labels(y: np.ndarray) -> Dict[int, float]:
    """Balanced weights: N / (K * count)."""
    counts = Counter(y.tolist())
    n_total, n_classes = len(y), len(counts)
    return {cls: n_total / (n_classes * cnt) for cls, cnt in counts.items()}


# =========================
# Models
# =========================
def build_gru_classifier(input_len: int, n_features: int, lr: float) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, n_features))
    x = layers.GRU(64)(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_gru_regressor(input_len: int, n_features: int, lr: float) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, n_features))
    x = layers.GRU(128)(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),],
    )
    return model


# =========================
# Train / Eval (dual mode)
# =========================
def run_gru(
    data_dir: Path,
    task: str = "cls",          # "cls" or "forecast"
    # shared
    seq_len: int = GRU_DEFAULTS["seq_len"],
    stride: int = GRU_DEFAULTS["stride"],       # classification only
    batch_size: int = GRU_DEFAULTS["batch_size"],
    epochs: int = GRU_DEFAULTS["epochs"],
    lr: float = GRU_DEFAULTS["lr"],
    patience: int = GRU_DEFAULTS["patience"],
    test_size: float = TEST_SIZE_DEFAULT,
    seed: int = RANDOM_SEED_DEFAULT,
    # classification builder knobs
    leak_fraction: float = 0.30,
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
    # forecasting builder knobs
    residual_threshold_pct: float = RESIDUAL_THRESHOLD_PCT_DEFAULT,
):
    set_seed(seed)

    if task == "cls":
        # 1) Build classification frame (random leaks, labels present)
        df = build_aggregated_dataset(
            data_dir=data_dir,
            use_forecast=False,
            leak_mode="variable",
            leak_fraction=leak_fraction,
            n_segments_range=(n_segments_min, n_segments_max),
            min_len_frac=min_len_frac,
        )

        # 2) Time-based per-meter split, then make VAL from TRAIN tail
        train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
        train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

        # 3) Scale on TRAIN-CORE only
        tr_scaled, _dummy, scaler = scale_on_train(train_core_df, train_core_df, CLS_FEATURE_COLS)
        val_scaled = val_df.copy()
        val_scaled[CLS_FEATURE_COLS] = scaler.transform(val_scaled[CLS_FEATURE_COLS])
        test_scaled = test_df.copy()
        test_scaled[CLS_FEATURE_COLS] = scaler.transform(test_scaled[CLS_FEATURE_COLS])

        # 4) Build sequences AFTER scaling
        X_train, y_train = make_sequences_per_meter(tr_scaled, CLS_FEATURE_COLS, LEAK_FLAG_COL, seq_len, stride)
        X_val,   y_val   = make_sequences_per_meter(val_scaled, CLS_FEATURE_COLS, LEAK_FLAG_COL, seq_len, stride)
        X_test,  y_test  = make_sequences_per_meter(test_scaled, CLS_FEATURE_COLS, LEAK_FLAG_COL, seq_len, stride)

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError("Empty split after windowing. Reduce seq_len/stride or check data coverage.")

        # 5) Model + training
        model = build_gru_classifier(seq_len, X_train.shape[-1], lr=lr)
        es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        class_weight = class_weight_from_labels(y_train)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[es],
            verbose=1,
        )

        # 6) Test evaluation
        probs = model.predict(X_test, batch_size=batch_size).ravel()
        y_pred = (probs >= 0.5).astype(np.int64)
        metrics = {
            "task": "cls",
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, zero_division=0)),
            "cm": confusion_matrix(y_test, y_pred).tolist(),
        }
        # Save
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "gru_cls.h5"
        scaler_path = MODELS_DIR / "gru_cls_scaler.joblib"
        model.save(model_path)
        import joblib; joblib.dump(scaler, scaler_path)
        return metrics

    # ----------------------- FORECAST -----------------------
    else:
        # 1) Build forecasting frame:
        #    - lags = seq_len (so each row's lag_* becomes the GRU sequence)
        #    - keep labels to filter train and compute optional residual detection
        df = build_aggregated_dataset(
            data_dir=data_dir,
            use_forecast=True,
            lags=seq_len,
            leak_fraction=leak_fraction,
            forecast_include_labels=True,   # keep Leak_Flag for train filter & test eval
            target_col=DELTA_VOLUME_COL,
        )

        # 2) Split; then filter leaks from TRAIN/VAL if requested
        train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
        train_core_df, val_df = time_based_group_split(train_df, test_size=0.10, group_cols=[METER_ID_COL])

        # 3) Scale only the lag columns on TRAIN-CORE
        lag_cols = [f"lag_{k}" for k in range(seq_len, 0, -1)]
        tr_scaled, _dummy, scaler = scale_on_train(train_core_df, train_core_df, lag_cols)
        val_scaled = val_df.copy()
        val_scaled[lag_cols] = scaler.transform(val_scaled[lag_cols])
        test_scaled = test_df.copy()
        test_scaled[lag_cols] = scaler.transform(test_scaled[lag_cols])

        # 4) Build sequences from lag rows (no windowing)
        X_train, y_train, _     = make_lag_sequences(tr_scaled, seq_len)
        X_val,   y_val, _       = make_lag_sequences(val_scaled, seq_len)
        X_test,  y_test, leak_t = make_lag_sequences(test_scaled, seq_len)

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError("Empty split in forecast mode. Check seq_len/lags or data coverage.")

        # 5) Model + training (regression)
        model = build_gru_regressor(seq_len, 1, lr=lr)
        es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1,
        )

        # 6) Test regression metrics
        yhat_te = model.predict(X_test, batch_size=batch_size).ravel()
        mae = float(mean_absolute_error(y_test, yhat_te))
        rmse = float(np.sqrt(mean_squared_error(y_test, yhat_te)))
        out = {"task": "forecast", "mae": mae, "rmse": rmse}

        # 7) Optional residual-based leak detection (if labels exist on test)
        yhat_tr = model.predict(X_train, batch_size=batch_size).ravel()
        res_tr = np.abs(y_train - yhat_tr)
        thr = float(np.percentile(res_tr, residual_threshold_pct))
        out["train_residual_threshold"] = thr

        res_te = np.abs(y_test - yhat_te)
        pred_leak = (res_te >= thr).astype(int)
        if leak_t is not None:
            out.update({
                "det_acc": float(accuracy_score(leak_t, pred_leak)),
                "det_prec": float(precision_score(leak_t, pred_leak, zero_division=0)),
                "det_rec": float(recall_score(leak_t, pred_leak, zero_division=0)),
                "det_f1": float(f1_score(leak_t, pred_leak, zero_division=0)),
                "det_cm": confusion_matrix(leak_t, pred_leak).tolist(),
            })

        # Save
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "gru_forecast.h5"
        scaler_path = MODELS_DIR / "gru_forecast_scaler.joblib"
        model.save(model_path)
        import joblib; joblib.dump(scaler, scaler_path)
        return out


# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GRU — classification or forecasting")
    ap.add_argument("--data_dir", type=Path, default=str(DATA_DIR))
    ap.add_argument("--task", type=str, default="cls", choices=["cls", "forecast"])

    # shared
    ap.add_argument("--seq_len", type=int, default=GRU_DEFAULTS["seq_len"])
    ap.add_argument("--stride", type=int, default=GRU_DEFAULTS["stride"])  # cls only
    ap.add_argument("--batch_size", type=int, default=GRU_DEFAULTS["batch_size"])
    ap.add_argument("--epochs", type=int, default=GRU_DEFAULTS["epochs"])
    ap.add_argument("--lr", type=float, default=GRU_DEFAULTS["lr"])
    ap.add_argument("--patience", type=int, default=GRU_DEFAULTS["patience"])
    ap.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)

    # classification knobs
    ap.add_argument("--leak_fraction", type=float, default=0.3)
    ap.add_argument("--n_segments_min", type=int, default=1)
    ap.add_argument("--n_segments_max", type=int, default=3)
    ap.add_argument("--min_len_frac", type=float, default=0.05)

    # forecasting knobs
    ap.add_argument("--residual_threshold_pct", type=float, default=RESIDUAL_THRESHOLD_PCT_DEFAULT)

    args = ap.parse_args()

    results = run_gru(
        data_dir=args.data_dir,
        task=args.task,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        test_size=args.test_size,
        seed=args.seed,
        leak_fraction=args.leak_fraction,
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
        residual_threshold_pct=args.residual_threshold_pct,
    )

    print(json.dumps(results, indent=2))
