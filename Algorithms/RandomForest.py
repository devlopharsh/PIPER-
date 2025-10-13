from __future__ import annotations
from pathlib import Path
from typing import List
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

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
    y = (df[LEAK_FLAG_COL].to_numpy())
    return X, y

def _cv_threshold_rf(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    n_folds: int = 10,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced_subsample",
    max_features: str | int | float = "sqrt",
    random_state: int = RANDOM_SEED_DEFAULT,
) -> tuple[float, list[float]]:
    folds = _make_group_forward_time_folds(train_df, n_folds=n_folds)
    thr_list: list[float] = []
    for tr_idx, va_idx in folds:
        tr = train_df.loc[tr_idx].sort_values([METER_ID_COL])
        va = train_df.loc[va_idx].sort_values([METER_ID_COL])

        Xtr, ytr = prepare_Xy(tr, feature_cols)
        Xva, yva = prepare_Xy(va, feature_cols)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            max_features=max_features,
            n_jobs=-1,
            random_state=random_state,
        )
        clf.fit(Xtr, ytr)
        probs = clf.predict_proba(Xva)[:, 1]
        info = _choose_threshold_fbeta(yva, probs, beta=1.0)
        thr_list.append(float(info["thr"]))

    if not thr_list:
        return 0.5, []
    robust_thr = float(np.median(thr_list))
    print(f"[CV-RF] thresholds={np.round(thr_list,3).tolist()} → robust_thr={robust_thr:.3f}")
    return robust_thr, thr_list


# -----------------------------
# Train / Eval
# -----------------------------
def train_and_eval_random_forest(
    data_dir: Path,
    # RF hyperparams (shared)
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = RANDOM_SEED_DEFAULT,
    # cls-only
    class_weight: str | None = "balanced_subsample",
    max_features: str | int | float = "sqrt",
    # split
    test_size: float = TEST_SIZE_DEFAULT,
    # classification data knobs
    test_fraction: float = 0.30,
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
):
    # 1) Build dataset
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

    # Guards for classification
    assert train_df[LEAK_FLAG_COL].nunique() == 2, "Train has a single class — adjust leak_fraction/test_size."
    assert test_df[LEAK_FLAG_COL].nunique() == 2, "Test has a single class — adjust split/segment params."
    
    # 4) Inner CV on TRAIN → robust threshold
    cv_thr, cv_thr_list = _cv_threshold_rf(
        train_df, feat_cols,
        n_folds=10,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        max_features=max_features,
        random_state=random_state,
    )
    print(f"[INFO] Using robust_thr={cv_thr:.3f} on TEST")

    # 5) Fit final on FULL TRAIN
    clf = RandomForestClassifier(
         n_estimators=n_estimators,
         max_depth=max_depth,
         min_samples_leaf=min_samples_leaf,
         class_weight=class_weight,
         max_features=max_features,
         n_jobs=-1,
         random_state=random_state,
     )
    clf.fit(X_train, y_train)

    # 6) Evaluate with CV-threshold
    probs = clf.predict_proba(X_test)[:, 1]
    y_pred = (probs >= cv_thr).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    f1b = float(f1_score(y_test, y_pred, average="binary"))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_cls.joblib"
    import joblib; joblib.dump(clf, model_path)

    return {
        "task": "cls",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_binary": f1b,
        "cm": cm,
        "report": report,
        "model_path": str(model_path),
        "thr": cv_thr,
        "thr_list": [float(x) for x in cv_thr_list],
    }


# ---------------- CLI ----------------
def _int_or_none(v: str) -> int | None:
    return None if v.lower() == "none" else int(v)

if __name__ == "__main__":
    p = create_common_parser(
        "RandomForest — classification",
        data_dir_default=DATA_DIR,
        test_size_default=TEST_SIZE_DEFAULT,
        seed_default=RANDOM_SEED_DEFAULT,
        n_segments_min_default=2,
        n_segments_max_default=5,
        min_len_frac_default=0.05,
    )
    # include test_fraction here via defaults so it’s added by ParserCreator
    attach_additional_args(p, AlgoKind.RF,
        n_estimators=300,
        max_depth="None",
        min_samples_leaf=1,
        random_state=RANDOM_SEED_DEFAULT,
        class_weight="balanced_subsample",
        max_features="sqrt",
        test_fraction=0.30,
    )
    args = p.parse_args()

    md = _int_or_none(args.max_depth)
    # try to parse max_features numeric if provided
    if args.max_features.replace(".", "", 1).isdigit():
        mf: str | int | float = float(args.max_features) if "." in args.max_features else int(args.max_features)
    else:
        mf = args.max_features

    results = train_and_eval_random_forest(
        data_dir=args.data_dir,
        n_estimators=args.n_estimators,
        max_depth=md,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        class_weight=None if (isinstance(args.class_weight, str) and args.class_weight.lower() == "none") else args.class_weight,
        max_features=mf,
        test_size=args.test_size,
        test_fraction=args.test_fraction,
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
    )
    print(json.dumps(results, indent=2))
