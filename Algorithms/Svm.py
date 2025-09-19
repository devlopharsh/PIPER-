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
import mlflow
import mlflow.sklearn

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

def eval_forecasting(
    model, X_train, y_train, X_test, y_test,
    train_df: pd.DataFrame, test_df: pd.DataFrame, pct: float
):
    """Point-forecast metrics + residual->leak detection if labels exist."""
    yhat_tr = model.predict(X_train)
    yhat_te = model.predict(X_test)

    mae = mean_absolute_error(y_test, yhat_te)
    rmse = float(np.sqrt(mean_squared_error(y_test, yhat_te)))

    res_tr = np.abs(y_train - yhat_tr)
    thr = float(np.percentile(res_tr, pct))

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

    return {
        "mae": float(mae),
        "rmse": rmse,
        "train_residual_threshold": thr,
        **det_metrics
    }
def train_and_eval_svm(
    data_dir: Path,
    # --- TASK SWITCH ---
    task: str = "cls",                     # "cls" or "forecast"
    # --- SVM knobs (cls & forecast) ---
    kernel: str = "linear",                # "linear", "rbf", "rbf_approx"(cls only)
    C: float = 1.0,
    gamma: str | float = "scale",
    cache_size: int = 200,
    max_iter: int = 10000,
    max_train_samples: int | None = None,
    # --- Dataset knobs (shared, used by unified builder) ---
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
    seed: int = 42,
    # --- Split knob ---
    test_size: float = 0.30,
    # --- Forecasting-only knobs ---
    target_col: str = DELTA_VOLUME_COL,      # VOLUME_READING_COL or DELTA_VOLUME_COL
    lags: int = 24,
    rolling_windows: tuple[int, ...] = (6, 24),
    residual_threshold_pct: float = 97.5,
    # --- MLflow ---
    use_mlflow: bool = False,
    mlflow_uri: str = "file:./mlruns",
    experiment: str = "LeakDetection-SVM",
    run_name: str | None = None,
):
    np.random.seed(seed)
    # ----------------------------
    # 1) Build dataset
    # ----------------------------
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
            leak_mode="variable",
            test_fraction=test_size,
            n_segments_range=(n_segments_min, n_segments_max),
            min_len_frac=min_len_frac,
            target_col=target_col,
            lags=lags,
            rolling_windows=rolling_windows,
            forecast_include_labels=True
        )

    feature_cols = list(CLS_FEATURE_COLS)
    # ----------------------------
    # 2) Split
    # ----------------------------
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])
    # ----------------------------
    # 3) Scale on TRAIN only
    # ----------------------------
    train_df_scaled, test_df_scaled, scaler = scale_on_train(train_df, test_df, feature_cols)
    X_train, y_train = prepare_Xy(train_df_scaled, feature_cols, task)
    X_test,  y_test  = prepare_Xy(test_df_scaled,  feature_cols, task)

    # ----------------------------
    # 4) Model
    # ----------------------------
    if task == "cls":
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
    else:
        if kernel == "linear":
            model = LinearSVR(C=C, max_iter=max_iter, random_state=42)
        elif kernel == "rbf":
            g = gamma if isinstance(gamma, str) else float(gamma)
            model = SVR(C=C, gamma=g, kernel="rbf", cache_size=cache_size, max_iter=max_iter)
        else:
            raise ValueError("For forecasting, choose kernel 'linear' or 'rbf'.")

    # ----------------------------
    # 5) Fit
    # ----------------------------
    print(f"[INFO] Training SVM ({'classifier' if task=='cls' else 'regressor'}) kernel='{kernel}'...")
    model.fit(X_train, y_train)

    # ----------------------------
    # 6) Evaluate
    # ----------------------------
    results: dict = {}
    if task == "cls":
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="binary")
        cr_dict = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="binary", zero_division=0)

        print(f"[CLS] Accuracy: {acc:.4f} | F1-macro: {f1m:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
        print("Confusion matrix:\n", cm)

        results.update({
            "model": f"SVM-{kernel}",
            "task": "cls",
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "cm": cm,
            "cr_dict": cr_dict
        })
    else:
        metrics = eval_forecasting(
            model, X_train, y_train, X_test, y_test,
            train_df=train_df, test_df=test_df,
            pct=residual_threshold_pct
        )
        print(f"[Forecast] MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} | thr={metrics['train_residual_threshold']:.4f}")
        if "det_f1" in metrics:
            print(f"[Leak via residuals] Acc={metrics['det_acc']:.4f} F1={metrics['det_f1']:.4f}")

        results.update({
            "model": f"SVR-{kernel}",
            "task": "forecast",
            **metrics
        })

    # ----------------------------
    # 7) Persist artifacts
    # ----------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{task}_{kernel}"
    model_path = MODELS_DIR / f"svm_{suffix}.joblib"
    scaler_path = MODELS_DIR / f"svm_{suffix}_scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    if task == "cls":
        cm_path = MODELS_DIR / f"svm_{suffix}_confusion_matrix.csv"
        pd.DataFrame(results["cm"],
            columns=["pred_0", "pred_1"], index=["true_0", "true_1"]).to_csv(cm_path, index=True)
        cr_path = MODELS_DIR / f"svm_{suffix}_classif_report.json"
        with open(cr_path, "w") as f:
            json.dump(results["cr_dict"], f, indent=2)
        results["cm_path"] = str(cm_path)
        results["cr_path"] = str(cr_path)
    else:
        thr_path = MODELS_DIR / f"svm_{suffix}_residual_threshold.txt"
        with open(thr_path, "w") as f:
            f.write(str(results["train_residual_threshold"]))
        results["threshold_path"] = str(thr_path)

    return results


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

    # Quick console summary
    if args.task == "cls":
        print(json.dumps({
            "task": "cls",
            "accuracy": results["accuracy"],
            "f1_macro": results["f1_macro"],
        }, indent=2))
    else:
        out = {k: v for k, v in results.items()
               if k in ("mae","rmse","train_residual_threshold","det_acc","det_prec","det_rec","det_f1")}
        print(json.dumps({"task": "forecast", **out}, indent=2))
