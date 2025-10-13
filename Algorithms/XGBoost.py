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
from xgboost.callback import EarlyStopping

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
from utils.CvkFold import _make_group_forward_time_folds, _choose_threshold_fbeta
from utils.ParserCreator import create_common_parser, attach_additional_args, AlgoKind


# -----------------------------
# Feature helpers
# -----------------------------
def prepare_Xy(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LEAK_FLAG_COL].to_numpy()
    return X, y


def _cv_threshold_xgb(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    n_folds: int = 10,
    build_xgb: callable,               # () -> xgb.XGBClassifier
    early_stopping_rounds: int = 50,
) -> tuple[float, list[float], int | None]:
    folds = _make_group_forward_time_folds(train_df, n_folds=n_folds)
    thr_list: list[float] = []
    best_iters: list[int] = []

    for tr_idx, va_idx in folds:
        tr = train_df.loc[tr_idx].sort_values([METER_ID_COL])
        va = train_df.loc[va_idx].sort_values([METER_ID_COL])

        Xtr, ytr = prepare_Xy(tr, feature_cols)
        Xva, yva = prepare_Xy(va, feature_cols)

        model = build_xgb()
        # Early stopping πάνω στο fold-val
        try:
            es = EarlyStopping(rounds=early_stopping_rounds, minimize=True, save_best=True, maximize=False)
            model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False, callbacks=[es])
        except TypeError:
            model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)

        probs = model.predict_proba(Xva)[:, 1]
        info = _choose_threshold_fbeta(yva, probs, beta=1.0)
        thr_list.append(float(info["thr"]))

        bi = getattr(model, "best_iteration", None)
        if bi is not None:
            best_iters.append(int(bi))

    if not thr_list:
        return 0.5, [], None

    robust_thr = float(np.median(thr_list))
    median_best_iter = int(np.median(best_iters)) if best_iters else None
    print(f"[CV-XGB] thresholds={np.round(thr_list,3).tolist()} → robust_thr={robust_thr:.3f}"
          + (f" | median_best_iter={median_best_iter}" if median_best_iter is not None else ""))
    return robust_thr, thr_list, median_best_iter


# -----------------------------
# Train / Eval
# -----------------------------
def train_and_eval_xgboost(
    data_dir: Path,
    # Core hyperparameters
    n_estimators: int = 1500,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    gamma: float = 0.0,
    # Imbalance
    scale_pos_weight: str | float = "auto",
    # Early stopping (used only inside CV folds)
    early_stopping_rounds: int = 50,
    # Execution
    random_state: int = RANDOM_SEED_DEFAULT,
    use_gpu: bool = False,
    # Split / dataset knobs
    test_size: float = TEST_SIZE_DEFAULT,
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
):
    # 1) Dataset
    df = build_aggregated_dataset(
        data_dir=data_dir,
        leak_mode="variable",
        test_fraction=test_size,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )

    # 2) Split per meter (time-aware)
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])

    # 3) Features/targets
    feat_cols = list(CLS_FEATURE_COLS)
    X_train, y_train = prepare_Xy(train_df, feat_cols)
    X_test,  y_test  = prepare_Xy(test_df,  feat_cols)

    assert train_df[LEAK_FLAG_COL].nunique() == 2, "Train has a single class."
    assert test_df[LEAK_FLAG_COL].nunique() == 2, "Test has a single class."

    # 4) Imbalance
    if isinstance(scale_pos_weight, str) and scale_pos_weight.lower() == "auto":
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        spw_val = float(neg / pos)
    else:
        spw_val = float(scale_pos_weight)

    # 5) Device params — απλά & ανθεκτικά
    params_device = {"tree_method": "gpu_hist" if use_gpu else "hist"}

    # 6) Builder για reuse
    def _build_xgb(n_estimators_override: int | None = None):
        return xgb.XGBClassifier(
            n_estimators=n_estimators if n_estimators_override is None else n_estimators_override,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            gamma=gamma,
            objective="binary:logistic",
            eval_metric=["logloss", "aucpr"],
            scale_pos_weight=spw_val,
            random_state=random_state,
            **params_device,
        )

    # 7) Inner CV → robust threshold & median best_iter
    cv_thr, cv_thr_list, median_best_iter = _cv_threshold_xgb(
        train_df=train_df,
        feature_cols=feat_cols,
        n_folds=10,
        build_xgb=lambda: _build_xgb(),
        early_stopping_rounds=early_stopping_rounds,
    )

    # 8) Final fit στο FULL TRAIN
    final_n_estimators = int(max(50, median_best_iter)) if median_best_iter else n_estimators
    clf = _build_xgb(n_estimators_override=final_n_estimators)
    clf.fit(X_train, y_train, verbose=False)

    # 9) Test με robust threshold
    probs = clf.predict_proba(X_test)[:, 1]
    y_pred = (probs >= cv_thr).astype(int)

    acc  = float(accuracy_score(y_test, y_pred))
    f1b  = float(f1_score(y_test, y_pred, average="binary"))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    # 10) Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    skl_path = MODELS_DIR / "xgb_cls_sklearn.joblib"
    joblib.dump(clf, skl_path)
    booster_path = MODELS_DIR / "xgb_cls.json"
    clf.save_model(booster_path.as_posix())

    # Feature importances (gain)
    booster = clf.get_booster()
    imp_dict = booster.get_score(importance_type="gain")
    fi = pd.DataFrame(
        [{"feature": col, "importance_gain": imp_dict.get(f"f{i}", 0.0)} for i, col in enumerate(feat_cols)]
    )
    fi_path = MODELS_DIR / "xgb_cls_feature_importance.csv"
    fi.to_csv(fi_path, index=False)

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
        "final_n_estimators": final_n_estimators,
        "thr": cv_thr,
        "thr_list": [float(x) for x in cv_thr_list],
    }


# ---------------- CLI ----------------
def float_or_str(v: str) -> float | str:
    try:
        return float(v)
    except ValueError:
        return v

if __name__ == "__main__":
    p = create_common_parser(
        "XGBoost — leak classification",
        data_dir_default=DATA_DIR,
        test_size_default=TEST_SIZE_DEFAULT,
        seed_default=RANDOM_SEED_DEFAULT,
        n_segments_min_default=2,
        n_segments_max_default=5,
        min_len_frac_default=0.05,
    )
    attach_additional_args(p, AlgoKind.XGB,
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        scale_pos_weight="auto",
        early_stopping_rounds=50,
        random_state=RANDOM_SEED_DEFAULT,
        use_gpu=False,
    )

    args = p.parse_args()

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
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
    )
    print(json.dumps(results, indent=2))