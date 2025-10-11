from __future__ import annotations
from pathlib import Path
from typing import List
import argparse
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

import xgboost as xgb

from utils.constants import (
    DATA_DIR, MODELS_DIR,
    METER_ID_COL, LEAK_FLAG_COL,
    CLS_FEATURE_COLS,
    RANDOM_SEED_DEFAULT, TEST_SIZE_DEFAULT,
)
from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
)
from xgboost.callback import EarlyStopping
callbacks=[EarlyStopping(rounds=50, save_best=True)]


# -----------------------------
# Feature helpers
# -----------------------------
def prepare_Xy(df: pd.DataFrame, feature_cols: List[str]):
    """Return X, y with float32 features and 0/1 labels."""
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LEAK_FLAG_COL].to_numpy()
    return X, y


# -----------------------------
# Train / Eval
# -----------------------------
def train_and_eval_xgboost(
    data_dir: Path,
    # ---- Core XGB hyperparameters (robust defaults) ----
    n_estimators: int = 1500,             # large, rely on early stopping
    learning_rate: float = 0.05,          # conservative learning rate
    max_depth: int = 6,                   # tree depth
    min_child_weight: float = 1.0,        # min Hessian / leaf weight
    subsample: float = 0.8,               # row sampling
    colsample_bytree: float = 0.8,        # feature sampling
    reg_lambda: float = 1.0,              # L2 regularization
    reg_alpha: float = 0.0,               # L1 regularization
    gamma: float = 0.0,                   # split gain threshold
    # ---- Imbalance handling ----
    scale_pos_weight: str | float = "auto",  # "auto" computes neg/pos from train
    # ---- Early stopping ----
    early_stopping_rounds: int = 50,
    # ---- Execution ----
    random_state: int = RANDOM_SEED_DEFAULT,
    use_gpu: bool = False,                # True to use GPU if available
    # ---- Split / dataset build knobs ----
    test_size: float = TEST_SIZE_DEFAULT,
    test_fraction: float = 0.30,
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
):
    """
    Train an XGBoost classifier for leak detection with time-based per-meter split,
    class imbalance handling, and early stopping on a validation set.
    """
    # 1) Build dataset with synthetic leak segments + rolling features
    df = build_aggregated_dataset(
        data_dir=data_dir,
        leak_mode="variable",
        test_fraction=test_fraction,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )

    # 2) Time-based split (per meter)
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])

    # 3) Features / targets
    feat_cols = list(CLS_FEATURE_COLS)
    X_train, y_train = prepare_Xy(train_df, feat_cols)
    X_test,  y_test  = prepare_Xy(test_df,  feat_cols)

    # Guards for classification sanity
    assert train_df[LEAK_FLAG_COL].nunique() == 2, "Train has a single class — adjust leak_fraction/test_size."
    assert test_df[LEAK_FLAG_COL].nunique() == 2, "Test has a single class — adjust split/segment params."

    # 4) Robust imbalance handling (scale_pos_weight)
    if isinstance(scale_pos_weight, str) and scale_pos_weight.lower() == "auto":
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        spw_val = float(neg / pos)
    else:
        spw_val = float(scale_pos_weight)

    # 5) Device / tree method handling for xgboost v1.x vs v2.x
    try:
        major = int(xgb.__version__.split(".")[0])
    except Exception:
        major = 2  # be permissive

    params_device: dict = {}
    if use_gpu:
        if major >= 2:
            # XGBoost >= 2.0 uses 'device'
            params_device.update(dict(device="cuda", tree_method="hist"))
        else:
            # XGBoost 1.x uses gpu_hist
            params_device.update(dict(tree_method="gpu_hist"))
    else:
        if major >= 2:
            params_device.update(dict(device="cpu", tree_method="hist"))
        else:
            params_device.update(dict(tree_method="hist"))

    # 6) Define model (sklearn API)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        gamma=gamma,
        objective="binary:logistic",
        eval_metric=["logloss", "aucpr"],  # stable for imbalance
        scale_pos_weight=spw_val,
        random_state=random_state,
        **params_device,
    )

    # 7) Fit with early stopping against test set (time-based holdout)
        # 7) Fit with early stopping against test set (time-based holdout)
    if major >= 2:
        # ✅ XGBoost ≥ 2.0 (including 3.x)
        try:
            early_cb = EarlyStopping(
                rounds=early_stopping_rounds,
                minimize=True,   # 3.x compatible
                save_best=True,  # still accepted in 2.x
                maximize=False   # keep consistent with logloss
            )
            clf.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
                callbacks=[early_cb],
            )
        except TypeError:
            # fallback if API signature changed
            clf.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False,
            )



    # 8) Evaluate
    y_pred = clf.predict(X_test)
    acc  = float(accuracy_score(y_test, y_pred))
    f1b  = float(f1_score(y_test, y_pred, average="binary"))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    # 9) Save model + artifacts
        # 9) Save model + artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save sklearn wrapper (for Python inference)
    skl_path = MODELS_DIR / "xgb_cls_sklearn.joblib"
    joblib.dump(clf, skl_path)

    # Save booster-only JSON (for deployment / cloud use)
    booster_path = MODELS_DIR / "xgb_cls.json"
    clf.save_model(booster_path.as_posix())

    # --- Feature importances (safe mapping) ---
    booster = clf.get_booster()
    imp_dict = booster.get_score(importance_type="gain")

    importances = []
    for i, col in enumerate(feat_cols):
        key = f"f{i}"
        importance = imp_dict.get(key, 0.0)
        importances.append({"feature": col, "importance_gain": importance})

    fi = pd.DataFrame(importances)
    fi_path = MODELS_DIR / "xgb_cls_feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    # --- Best iteration tracking ---
    best_iter = getattr(clf, "best_iteration", None)
    best_score = getattr(clf, "best_score", None)

    return {
        "task": "cls",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_binary": f1b,
        "cm": cm,
        "report": report,
        "model_path_sklearn": str(skl_path),
        "model_path_booster": str(booster_path),
        "feature_importances_csv": str(fi_path),
        "scale_pos_weight": spw_val,
        "best_iteration": best_iter,
        "best_score": best_score,
        "xgboost_version": xgb.__version__,
        "device": params_device.get("device", "cpu") if major >= 2 else ("gpu" if use_gpu else "cpu"),
    }



# ---------------- CLI ----------------
def float_or_str(v: str) -> float | str:
    try:
        return float(v)
    except ValueError:
        return v

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost — leak classification")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)

    # Core hparams
    parser.add_argument("--n_estimators", type=int, default=1500)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)

    # Imbalance
    parser.add_argument("--scale_pos_weight", type=float_or_str, default="auto")

    # Early stopping
    parser.add_argument("--early_stopping_rounds", type=int, default=50)

    # Execution
    parser.add_argument("--random_state", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--use_gpu", action="store_true")

    # Split
    parser.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)

    # Classification builder knobs
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--n_segments_min", type=int, default=2)
    parser.add_argument("--n_segments_max", type=int, default=5)
    parser.add_argument("--min_len_frac", type=float, default=0.05)

    args = parser.parse_args()

    results = train_and_eval_xgboost(
        data_dir=args.data_dir,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        gamma=args.gamma,
        scale_pos_weight=args.scale_pos_weight,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.random_state,
        use_gpu=args.use_gpu,
        test_size=args.test_size,
        test_fraction=args.test_fraction,
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
    )

    print(json.dumps(results, indent=2))