from __future__ import annotations
from pathlib import Path
from typing import List
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

from utils.constants import (
    DATA_DIR, MODELS_DIR, CLS_FEATURE_COLS,
    METER_ID_COL, LEAK_FLAG_COL, TEST_SIZE_DEFAULT, RANDOM_SEED_DEFAULT
)
from utils.DataAggregation import (
    build_aggregated_dataset,
    time_based_group_split,
    scale_on_train,
)
from utils.ParserCreator import create_common_parser, attach_additional_args, AlgoKind


# ----------------------------
# Helpers
# ----------------------------
def prepare_Xy(df: pd.DataFrame, feature_cols: List[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return features X and labels y with consistent dtypes."""
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[LEAK_FLAG_COL].to_numpy()
    return X, y


def train_and_eval_svm(
    data_dir: Path,
    kernel: str = "linear",                # "linear", "rbf", "rbf_approx"
    C: float = 1.0,
    gamma: str | float = "scale",
    cache_size: int = 200,
    max_iter: int = 10000,
    max_train_samples: int | None = None,
    n_segments_min: int = 1,
    n_segments_max: int = 3,
    min_len_frac: float = 0.05,
    seed: int = 42,
    test_size: float = TEST_SIZE_DEFAULT,
):
    np.random.seed(seed)

    # 1) Build dataset with synthetic leaks
    df = build_aggregated_dataset(
        data_dir=data_dir,
        test_fraction=test_size,
        n_segments_range=(n_segments_min, n_segments_max),
        min_len_frac=min_len_frac,
    )

    # 2) Time-aware split per meter (first part train, tail part test)
    train_df, test_df = time_based_group_split(df, test_size=test_size, group_cols=[METER_ID_COL])

    # 3) Feature set
    feature_cols = [c for c in CLS_FEATURE_COLS if c in train_df.columns]

    # 4) Scale on TRAIN only (avoid leakage), then transform TEST
    train_df_scaled, test_df_scaled, scaler = scale_on_train(train_df, test_df, feature_cols)
    X_train, y_train = prepare_Xy(train_df_scaled, feature_cols)
    X_test,  y_test  = prepare_Xy(test_df_scaled,  feature_cols)

    # (Optional) subsample train for speed
    if max_train_samples is not None and len(X_train) > max_train_samples:
        idx = np.random.permutation(len(X_train))[:max_train_samples]
        X_train, y_train = X_train[idx], y_train[idx]

    # 5) Choose model
    if kernel == "linear":
        model = LinearSVC(C=C, max_iter=max_iter, class_weight="balanced", random_state=42)
    elif kernel == "rbf":
        # Note: sklearn.svm.SVC has no random_state argument
        model = SVC(C=C, gamma=gamma, kernel="rbf", cache_size=cache_size,
                    max_iter=max_iter, class_weight="balanced", probability=False)
    elif kernel == "rbf_approx":
        # Approximate RBF via random Fourier features + linear classifier
        rbf_feature = RBFSampler(gamma=0.1 if gamma == "scale" else float(gamma), random_state=42)
        base_clf = SGDClassifier(loss="hinge", alpha=1.0 / C,
                                 max_iter=1000, tol=1e-3, random_state=42)
        model = make_pipeline(rbf_feature, base_clf)
    else:
        raise ValueError("Unsupported kernel: choose 'linear', 'rbf', or 'rbf_approx'.")

    # 6) Fit
    print(f"[INFO] Training SVM classifier kernel='{kernel}'...")
    model.fit(X_train, y_train)

    # 7) Evaluate (hard labels)
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

    # 8) Persist artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{kernel}"
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
# CLI
# ----------------------------
if __name__ == "__main__":
    p = create_common_parser(
        "SVM for Leak Detection — Classification.",
        data_dir_default=DATA_DIR,
        test_size_default=TEST_SIZE_DEFAULT,
        seed_default=RANDOM_SEED_DEFAULT,
        n_segments_min_default=2,
        n_segments_max_default=5,
        min_len_frac_default=0.05,
    )
    attach_additional_args(p, AlgoKind.SVM,
        kernel="linear",
        C=1.0,
        gamma="scale",
        cache_size=200,
        max_iter=30000,
    )
    args = p.parse_args()

    results = train_and_eval_svm(
        data_dir=Path(args.data_dir),
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
    )
