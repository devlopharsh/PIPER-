from __future__ import annotations
from pathlib import Path
import argparse
import json
import pandas as pd

from utils.constants import DATA_DIR, TEST_SIZE_DEFAULT, RANDOM_SEED_DEFAULT
from Algorithms.Svm import train_and_eval_svm
from Algorithms.RandomForest import train_and_eval_random_forest
from Algorithms.Mlp import train_and_eval_mlp
from Algorithms.Lstm import train_and_eval_lstm
from Algorithms.Gru import run_gru  # keras version

ALGOS = ("svm", "rf", "mlp", "lstm", "gru")

def normalize_row(algorithm: str, res: dict) -> dict:
    """
    Bring all result dicts to a common schema so they can be compared in one DataFrame.
    Unknown fields are left out; missing ones are filled with None.
    """
    row = {
        "algorithm": algorithm,
        "task": res.get("task"),
        # classification metrics (None for forecast)
        "accuracy": res.get("accuracy"),
        "precision": res.get("precision"),
        "recall": res.get("recall"),
        "f1_macro": res.get("f1_macro", res.get("f1")),  # fallback for older returns
        # forecasting metrics (None for cls)
        "mae": res.get("mae"),
        "rmse": res.get("rmse"),
        "train_residual_threshold": res.get("train_residual_threshold"),
        # residual-based detection (forecast, if available)
        "det_acc": res.get("det_acc"),
        "det_prec": res.get("det_prec"),
        "det_rec": res.get("det_rec"),
        "det_f1": res.get("det_f1"),
    }
    return row

def run_one(algorithm: str, task: str, data_dir: Path, test_size: float, seed: int) -> dict:
    if algorithm == "svm":
        return train_and_eval_svm(
            data_dir=data_dir,
            task=task,
            kernel="linear",              # tweak if you want, e.g. "rbf"
            C=1.0,
            gamma="scale",
            test_size=test_size,
            seed=seed,
            leak_fraction=0.30,
        )
    if algorithm == "rf":
        return train_and_eval_random_forest(
            data_dir=data_dir,
            task=task,
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample" if task == "cls" else None,
            test_size=test_size,
            random_state=seed,
            leak_fraction=0.30,
            drop_calendar=True if task == "forecast" else False,
        )
    if algorithm == "mlp":
        return train_and_eval_mlp(
            data_dir=data_dir,
            task=task,
            hidden=[128, 64],
            dropout=0.2,
            max_epochs=30,
            test_size=test_size,
            seed=seed,
            leak_fraction=0.30,
            # sensible forecast defaults
            target_col="Delta_Volume",
            lags=24,
            rolling_windows=(6, 24),
            drop_calendar=False,              # MLP can benefit from calendar
            train_on_normal_only=True,
        )
    if algorithm == "lstm":
        return train_and_eval_lstm(
            data_dir=data_dir,
            task=task,
            window=24,
            stride=1,
            hidden=64,
            num_layers=1,
            test_size=test_size,
            seed=seed,
            leak_fraction=0.30,
        )
    if algorithm == "gru":
        return run_gru(
            data_dir=data_dir,
            task=task,
            seq_len=48 if task == "cls" else 24,   # cls window can be longer; forecast uses lags
            stride=1,
            batch_size=256,
            epochs=30,
            patience=5,
            test_size=test_size,
            leak_fraction=0.30,
            train_on_normal_only=True,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")

def main():
    ap = argparse.ArgumentParser(description="Run experiments across algorithms in cls/forecast modes.")
    ap.add_argument("--data_dir", type=Path, default=Path(DATA_DIR))
    ap.add_argument("--task", choices=["cls", "forecast", "both"], default="cls")
    ap.add_argument("--algos", type=str, default=",".join(ALGOS),
                    help="Comma-separated subset, e.g. 'rf,svm,mlp'")
    ap.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    ap.add_argument("--out_csv", type=Path, default=Path("experiment_summary.csv"))
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    tasks = [args.task] if args.task != "both" else ["cls", "forecast"]

    rows = []
    for t in tasks:
        for algo in algos:
            print(f"\n=== Running {algo.upper()} [{t}] ===")
            res = run_one(algo, t, args.data_dir, args.test_size, args.seed)
            row = normalize_row(algo, res)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("\n=== Summary ===")
    # show best comparator depending on task selection
    if args.task == "cls":
        print(df.sort_values(["task", "f1_macro", "accuracy"], ascending=[True, False, False]).to_string(index=False))
    elif args.task == "forecast":
        print(df.sort_values(["task", "rmse"], ascending=[True, True]).to_string(index=False))
    else:
        # both: show side-by-side by task
        print(df.sort_values(["task", "f1_macro", "rmse"], ascending=[True, False, True]).to_string(index=False))

if __name__ == "__main__":
    main()
