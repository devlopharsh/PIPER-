from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

from constants import DATA_DIR, MODELS_DIR
from DataAggregation import (
    build_dataset_random_segments,
    time_based_group_split,
    scale_on_train,
    )

# Feature set we’ll use (simple but effective baseline)
FEATURE_COLS = ["Volume_Reading", "Hour", "DayOfWeek", "Month", "Delta_Volume"]

def prepare_flat_Xy(df: pd.DataFrame, feature_cols: list[str], target_col: str = "Leak_Flag"):
    """Extract numpy arrays for sklearn (keeps the training code clean)."""
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy()
    return X, y

def train_and_eval_svm(
    data_dir: Path,
    # --- SVM knobs ---
    kernel: str = "linear",             # "linear", "rbf", or "rbf_approx"
    C: float = 1.0,                     # regularization (higher C -> less regularization)
    gamma: str | float = "scale",       # RBF width, "scale" is a sensible default
    cache_size: int = 200,
    max_iter: int = 10000,
    max_train_samples: int | None = None,
    # --- Dataset knobs (propagate down to builder) ---
    leak_fraction: float = 0.30,
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
    seed: int = 42,
    # --- Split knob ---
    test_size: float = 0.30             # 30% test so train reliably contains some leak
):
    print(f"[INFO] Building random-segments dataset from: {data_dir}")
    combined = build_dataset_random_segments(
        data_dir=data_dir,
        leak_mode="variable",
        leak_fraction=leak_fraction,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
        seed=seed
    )
    print(f"[INFO] Combined shape: {combined.shape}")

    # Per-meter time split (prevents temporal leakage)
    train_df, test_df = time_based_group_split(
        combined, test_size=test_size, group_cols=["Meter_ID"]
    )

    # Safety nets: ensure we have both classes in train and test
    assert train_df["Leak_Flag"].nunique() == 2, \
        "Train has no leak samples — increase leak_fraction or reduce test_size."
    assert test_df["Leak_Flag"].nunique() == 2, \
        "Test has no leak samples — adjust split or segment parameters."

    # Optional: class-balanced subsampling to speed up experimentation
    if max_train_samples is not None and len(train_df) > max_train_samples:
        per_cls = max(1, max_train_samples // 2)
        train_df = (train_df.groupby("Leak_Flag", group_keys=False)
                             .apply(lambda x: x.sample(min(len(x), per_cls), random_state=42)))
        print(f"[INFO] Subsampled training set to {len(train_df)} rows (balanced by Leak_Flag)")

    # Scale features on train only (correct practice)
    train_df, test_df, scaler = scale_on_train(train_df, test_df, FEATURE_COLS)

    # Extract arrays for sklearn
    X_train, y_train = prepare_flat_Xy(train_df, FEATURE_COLS, target_col="Leak_Flag")
    X_test,  y_test  = prepare_flat_Xy(test_df,  FEATURE_COLS, target_col="Leak_Flag")

    # Pick the classifier
    if kernel == "linear":
        clf = LinearSVC(C=C, max_iter=max_iter, class_weight="balanced", random_state=42)
    elif kernel == "rbf":
        clf = SVC(C=C, gamma=gamma, kernel="rbf", cache_size=cache_size,
                  max_iter=max_iter, class_weight="balanced", probability=False, random_state=42)
    elif kernel == "rbf_approx":
        # Approximate RBF with Random Fourier Features, then linear classifier
        rbf_feature = RBFSampler(gamma=0.1 if gamma == "scale" else float(gamma), random_state=42)
        X_train = rbf_feature.fit_transform(X_train)
        X_test  = rbf_feature.transform(X_test)
        clf = SGDClassifier(loss="hinge", alpha=1.0/(C*len(X_train)),
                            max_iter=1000, tol=1e-3, random_state=42)
    else:
        raise ValueError("Unsupported kernel: choose from 'linear', 'rbf', 'rbf_approx'")

    print(f"[INFO] Training SVM with kernel='{kernel}'...")
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Persist artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"svm_{kernel}.joblib"
    scaler_path = MODELS_DIR / f"svm_{kernel}_scaler.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved scaler to: {scaler_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "rbf", "rbf_approx"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--cache_size", type=int, default=200)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--max_train_samples", type=int, default=None)
    # Dataset builder knobs (surfaced here so you can experiment from CLI)
    parser.add_argument("--leak_fraction", type=float, default=0.30)
    parser.add_argument("--n_segments_min", type=int, default=1)
    parser.add_argument("--n_segments_max", type=int, default=3)
    parser.add_argument("--min_len_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    # Split knob
    parser.add_argument("--test_size", type=float, default=0.30)
    args = parser.parse_args()

    train_and_eval_svm(
        data_dir=Path(args.data_dir),
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        cache_size=args.cache_size,
        max_iter=args.max_iter,
        max_train_samples=args.max_train_samples,
        leak_fraction=args.leak_fraction,
        n_segments_min=args.n_segments_min,
        n_segments_max=args.n_segments_max,
        min_len_frac=args.min_len_frac,
        seed=args.seed,
        test_size=args.test_size,
    )
