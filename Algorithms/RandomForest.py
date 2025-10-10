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

# -----------------------------
# Feature helpers
# -----------------------------

def prepare_Xy(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = (df[LEAK_FLAG_COL].to_numpy())
    return X, y


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

    # 4) Fit
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

    # 5) Evaluate (classification)
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))
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
        "f1_macro": f1m,
        "cm": cm,
        "report": report,
        "model_path": str(model_path),
    }


# ---------------- CLI ----------------
def int_or_none(v: str) -> int | None:
    return None if v.lower() == "none" else int(v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RandomForest — classification or forecasting")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)

    # RF hparams
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=str, default="None")  # int or 'None'
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=RANDOM_SEED_DEFAULT)

    # cls-only
    parser.add_argument("--class_weight", type=str, default="balanced_subsample")
    parser.add_argument("--max_features", type=str, default="sqrt")  # int/float/sqrt/log2

    # split
    parser.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)

    # classification builder knobs
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--n_segments_min", type=int, default=1)
    parser.add_argument("--n_segments_max", type=int, default=3)
    parser.add_argument("--min_len_frac", type=float, default=0.05)

    args = parser.parse_args()

    # parse rolling_windows and max_depth / max_features
    md = int_or_none(args.max_depth)

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
        class_weight=None if args.class_weight.lower() == "none" else args.class_weight,
        max_features=mf,
        test_size=args.test_size,
        test_fraction=args.test_fraction,
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
    )

    print(json.dumps(results, indent=2))
