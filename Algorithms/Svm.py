from __future__ import annotations
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, precision_score, recall_score
)
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

from utils.constants import (
    DATA_DIR, MODELS_DIR,
    CLS_FEATURE_COLS, CALENDAR_COLS,
    METER_ID_COL, FORECAST_Y_COL, LEAK_FLAG_COL,
    VOLUME_READING_COL, DELTA_VOLUME_COL,
)
from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)

# ----------------------------
# Helpers
# ----------------------------
def parse_int_list(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())

def prepare_Xy(df: pd.DataFrame, feature_cols: list[str], task: str):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    if task == "cls":
        y = df[LEAK_FLAG_COL].to_numpy()
    else:
        y = df[FORECAST_Y_COL].to_numpy(dtype=np.float32)
    return X, y


def train_and_eval_svm(
    data_dir: Path,
    task: str = "cls",                     # "cls" or "forecast"
    kernel: str = "linear",                # "linear", "rbf", "rbf_approx" (cls only)
    C: float = 1.0,
    gamma: str | float = "scale",
    cache_size: int = 200,
    max_iter: int = 10000,
    max_train_samples: int | None = None,
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
    seed: int = 42,
    test_size: float = 0.30,
    target_col: str = DELTA_VOLUME_COL,      # only for forecast
    lags: int = 24,
    rolling_windows: tuple[int, ...] = (6, 24),
    residual_threshold_pct: float = 97.5,
    use_mlflow: bool = False,
    mlflow_uri: str = "file:./mlruns",
    experiment: str = "LeakDetection-SVM",
    run_name: str | None = None,
):

    np.random.seed(seed)

    # ----------------------------
    # CLASSIFICATION BLOCK
    # ----------------------------
    if task == "cls":
        # 1) Build dataset with synthetic leak segments
        df = build_aggregated_dataset(
            data_dir=data_dir,
            use_forecast=False,
            leak_mode="variable",
            test_fraction=test_size,
            n_segments_range=(n_segments_min, n_segments_max),
            min_len_frac=min_len_frac,
        )

        # 2) Split per meter (time-aware)
        train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])

        # 3) Feature set (classification)
        feature_cols = [c for c in CLS_FEATURE_COLS if c in train_df.columns and c != FORECAST_Y_COL]

        # 4) Scale on TRAIN only (avoid leakage)
        train_df_scaled, test_df_scaled, scaler = scale_on_train(train_df, test_df, feature_cols)
        X_train, y_train = prepare_Xy(train_df_scaled, feature_cols, task="cls")
        X_test,  y_test  = prepare_Xy(test_df_scaled,  feature_cols, task="cls")

        # (Optional) subsample train if requested
        if max_train_samples is not None and len(X_train) > max_train_samples:
            idx = np.random.permutation(len(X_train))[:max_train_samples]
            X_train, y_train = X_train[idx], y_train[idx]

        # 5) Model
        if kernel == "linear":
            model = LinearSVC(C=C, max_iter=max_iter, class_weight="balanced", random_state=42)
        elif kernel == "rbf":
            model = SVC(C=C, gamma=gamma, kernel="rbf", cache_size=cache_size,
                        max_iter=max_iter, class_weight="balanced",
                        probability=False, random_state=42)
        elif kernel == "rbf_approx":
            rbf_feature = RBFSampler(gamma=0.1 if gamma == "scale" else float(gamma), random_state=42)
            base_clf = SGDClassifier(loss="hinge", alpha=1.0/(C),
                                     max_iter=1000, tol=1e-3, random_state=42)
            model = make_pipeline(rbf_feature, base_clf)
        else:
            raise ValueError("Unsupported kernel for classification: 'linear', 'rbf', 'rbf_approx'.")

        # 6) Fit
        print(f"[INFO] Training SVM classifier kernel='{kernel}'...")
        model.fit(X_train, y_train)

        # 7) Evaluate (hard labels, like your current SVM flow)
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1b = float(f1_score(y_test, y_pred, average="binary"))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec  = float(recall_score(y_test, y_pred, zero_division=0))
        cm   = confusion_matrix(y_test, y_pred)
        cr_dict = classification_report(y_test, y_pred, output_dict=True)

        print(f"[CLS] Acc={acc:.4f} F1={f1b:.4f} Prec={prec:.4f} Rec={rec:.4f}")
        print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
        print("Confusion matrix:\n", cm)

        # 8) Persist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"{task}_{kernel}"
        model_path  = MODELS_DIR / f"svm_{suffix}.joblib"
        scaler_path = MODELS_DIR / f"svm_{suffix}_scaler.joblib"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        cm_path = MODELS_DIR / f"svm_{suffix}_confusion_matrix.csv"
        pd.DataFrame(cm, columns=["pred_0", "pred_1"], index=["true_0", "true_1"]).to_csv(cm_path, index=True)
        cr_path = MODELS_DIR / f"svm_{suffix}_classif_report.json"
        with open(cr_path, "w") as f:
            json.dump(cr_dict, f, indent=2)

        return {
            "model": f"SVM-{kernel}",
            "task": "cls",
            "accuracy": acc,
            "f1_binary": f1b,
            "precision": prec,
            "recall": rec,
            "cm": cm.tolist(),
            "cr_dict": cr_dict,
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "cm_path": str(cm_path),
            "cr_path": str(cr_path),
        }

    # ----------------------------
    # FORECAST BLOCK
    # ----------------------------
    else:
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler

        # 1) Build forecasting frame (labels kept for detection)
        df = build_aggregated_dataset(
            data_dir=data_dir,
            use_forecast=True,
            leak_mode="variable",
            test_fraction=test_size,
            n_segments_range=(n_segments_min, n_segments_max),
            min_len_frac=min_len_frac,
            target_col=target_col,
            lags=lags,
            rolling_windows=rolling_windows,
            forecast_include_labels=True
        )

        # 2) Split per meter
        train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])

        # 3) Feature set (forecast): lags + rolling + calendar (no leakage of target)
        lag_cols  = [c for c in df.columns if c.startswith("lag_")]
        roll_cols = [c for c in df.columns if c.startswith("roll")]
        cal_cols  = [c for c in CALENDAR_COLS if c in df.columns]
        feature_cols = [c for c in (lag_cols + roll_cols + cal_cols) if c != FORECAST_Y_COL]

        # 4) Scale X on TRAIN only
        train_df_scaled, test_df_scaled, scaler = scale_on_train(train_df, test_df, feature_cols)
        X_train, y_train = prepare_Xy(train_df_scaled, feature_cols, task="forecast")
        X_test,  y_test  = prepare_Xy(test_df_scaled,  feature_cols, task="forecast")

        # (Optional) subsample train if requested
        if max_train_samples is not None and len(X_train) > max_train_samples:
            idx = np.random.permutation(len(X_train))[:max_train_samples]
            X_train, y_train = X_train[idx], y_train[idx]

        # 5) Regressor + y-scaling (official way for SVR stability)
        if kernel == "linear":
            base_reg = LinearSVR(C=C, max_iter=max_iter, random_state=42)
        elif kernel == "rbf":
            g = gamma if isinstance(gamma, str) else float(gamma)
            base_reg = SVR(C=C, gamma=g, kernel="rbf", cache_size=cache_size, max_iter=max_iter)
        else:
            raise ValueError("For forecasting, choose kernel 'linear' or 'rbf'.")

        y_scaler = StandardScaler()
        model = TransformedTargetRegressor(regressor=base_reg, transformer=y_scaler)

        # 6) Fit
        print(f"[INFO] Training SVR forecaster kernel='{kernel}'...")
        model.fit(X_train, y_train)

        # 7) Evaluate (original units, thanks to TTR inverse-transform)
        yhat_tr = model.predict(X_train)
        yhat_te = model.predict(X_test)

        mae  = float(mean_absolute_error(y_test, yhat_te))
        rmse = float(np.sqrt(mean_squared_error(y_test, yhat_te)))

        # Residual threshold in original units (interpretable)
        res_tr = np.abs(y_train - yhat_tr)
        thr    = float(np.percentile(res_tr, residual_threshold_pct))

        res_te = np.abs(y_test - yhat_te)
        pred_leak = (res_te >= thr).astype(int)

        det_metrics = {}
        if LEAK_FLAG_COL in test_df.columns:
            gt_leak = test_df[LEAK_FLAG_COL].to_numpy()
            det_metrics = {
                "det_acc": float(accuracy_score(gt_leak, pred_leak)),
                "det_prec": float(precision_score(gt_leak, pred_leak, zero_division=0)),
                "det_rec": float(recall_score(gt_leak, pred_leak, zero_division=0)),
                "det_f1": float(f1_score(gt_leak, pred_leak, zero_division=0)),
                "det_cm": confusion_matrix(gt_leak, pred_leak).tolist(),
            }

        print(f"[Forecast] MAE={mae:.4f} RMSE={rmse:.4f} | thr={thr:.4f}")
        if det_metrics:
            print(f"[Leak via residuals] Acc={det_metrics['det_acc']:.4f} F1={det_metrics['det_f1']:.4f}")

        # 8) Persist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"{task}_{kernel}"
        model_path   = MODELS_DIR / f"svm_{suffix}.joblib"
        scaler_path  = MODELS_DIR / f"svm_{suffix}_scaler.joblib"              # X-scaler
        ytr_path     = MODELS_DIR / f"svm_{suffix}_y_transformer.joblib"       # y-scaler
        thr_path     = MODELS_DIR / f"svm_{suffix}_residual_threshold.txt"

        joblib.dump(model, model_path)               # contains regressor + transformer
        joblib.dump(scaler, scaler_path)             # X scaler
        joblib.dump(model.transformer_, ytr_path)    # y scaler (also embedded in model)
        with open(thr_path, "w") as f:
            f.write(str(thr))

        return {
            "model": f"SVR-{kernel}",
            "task": "forecast",
            "mae": mae,
            "rmse": rmse,
            "train_residual_threshold": thr,
            **det_metrics,
            "model_path": str(model_path),
            "x_scaler_path": str(scaler_path),
            "y_transformer_path": str(ytr_path),
            "threshold_path": str(thr_path),
        }

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM for Leak Detection — Classification or Forecasting.")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))

    # TASK & common SVM
    parser.add_argument("--task", type=str, default="cls", choices=["cls", "forecast"])
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf", "rbf_approx"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--cache_size", type=int, default=200)
    parser.add_argument("--max_iter", type=int, default=30000)

    # dataset builder knobs (shared)
    parser.add_argument("--n_segments_min", type=int, default=2)
    parser.add_argument("--n_segments_max", type=int, default=5)
    parser.add_argument("--min_len_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.30)

    # forecasting only
    parser.add_argument("--target_col", type=str, default=DELTA_VOLUME_COL,
                        choices=[VOLUME_READING_COL, DELTA_VOLUME_COL])
    parser.add_argument("--lags", type=int, default=24)
    parser.add_argument("--rolling_windows", type=str, default="6,24")
    parser.add_argument("--residual_threshold_pct", type=float, default=85)

    # MLflow
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow_uri", type=str, default="file:./mlruns")
    parser.add_argument("--experiment", type=str, default="LeakDetection-SVM")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    results = train_and_eval_svm(
        data_dir=Path(args.data_dir),
        task=args.task,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        cache_size=args.cache_size,
        max_iter=args.max_iter,
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
        seed=args.seed,
        test_size=args.test_size,
        target_col=args.target_col,
        lags=args.lags,
        rolling_windows=parse_int_list(args.rolling_windows),
        residual_threshold_pct=args.residual_threshold_pct,
        use_mlflow=args.mlflow,
        mlflow_uri=args.mlflow_uri,
        experiment=args.experiment,
        run_name=args.run_name,
    )
