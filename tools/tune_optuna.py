from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict
import optuna
from Algorithms.Gru import train_and_eval_gru
from Algorithms.Lstm import train_and_eval_lstm
from Algorithms.Mlp import train_and_eval_mlp
from Algorithms.RandomForest import train_and_eval_random_forest
from Algorithms.Svm import train_and_eval_svm
from Algorithms.XGBoost import train_and_eval_xgboost
from utils.constants import DATA_DIR, TEST_SIZE_DEFAULT, RANDOM_SEED_DEFAULT, MODELS_DIR


# ----------------------- helpers -----------------------
def get_f1(results: Dict[str, Any]) -> float:
    """Extract F1 from model results (handles both 'f1' and 'f1_binary')."""
    if "f1" in results and results["f1"] is not None:
        return float(results["f1"])
    if "f1_binary" in results and results["f1_binary"] is not None:
        return float(results["f1_binary"])
    raise RuntimeError("F1 not present in results dict.")

def save_best(study: optuna.Study, algo_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {
        "algo": algo_name,
        "best_f1": float(study.best_value),
        "best_params": study.best_params,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
    }
    (out_dir / f"optuna_{algo_name}_best.json").write_text(json.dumps(best, indent=2))
    try:
        df = study.trials_dataframe()
        df.to_csv(out_dir / f"optuna_{algo_name}_trials.csv", index=False)
    except Exception:
        pass

def tobool(trial: optuna.Trial, name: str) -> bool:
    return trial.suggest_categorical(name, [False, True])


# ----------------------- objectives (maximize F1) -----------------------
def make_obj_gru(data_dir: Path, seed: int, test_size: float):
    def _obj(trial: optuna.Trial) -> float:
        params = dict(
            window=trial.suggest_int("window", 12, 48, step=6),   # commonly tuned
            stride=trial.suggest_int("stride", 1, 3),
            hidden=trial.suggest_int("hidden", 32, 256, log=True),
            num_layers=trial.suggest_int("num_layers", 1, 2),
            bidirectional=tobool(trial, "bidirectional"),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            lr=trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            max_epochs=10, early_patience=6,
            seed=seed, test_size=test_size,
        )
        try:
            res = train_and_eval_gru(data_dir=data_dir, **params)
            return get_f1(res)
        except Exception as e:
            trial.set_user_attr("error", str(e)); return -1e9
    return _obj

def make_obj_lstm(data_dir: Path, seed: int, test_size: float):
    def _obj(trial: optuna.Trial) -> float:
        params = dict(
            window=trial.suggest_int("window", 12, 48, step=6),
            stride=trial.suggest_int("stride", 1, 3),
            hidden=trial.suggest_int("hidden", 32, 256, log=True),
            num_layers=trial.suggest_int("num_layers", 1, 2),
            bidirectional=tobool(trial, "bidirectional"),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            lr=trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            max_epochs=10, early_patience=4,
            seed=seed, test_size=test_size,
        )
        try:
            res = train_and_eval_lstm(data_dir=data_dir, **params)
            return get_f1(res)
        except Exception as e:
            trial.set_user_attr("error", str(e)); return -1e9
    return _obj

def make_obj_mlp(data_dir: Path, seed: int, test_size: float):
    def _obj(trial: optuna.Trial) -> float:
        h1 = trial.suggest_int("h1", 64, 512, log=True)
        h2 = trial.suggest_int("h2", 32, 256, log=True)
        use_h3 = tobool(trial, "use_h3")
        hidden = [h1, h2] + ([trial.suggest_int("h3", 16, 128, log=True)] if use_h3 else [])
        params = dict(
            hidden=hidden,
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            max_epochs=10, early_patience=6,
            seed=seed, test_size=test_size,
        )
        try:
            res = train_and_eval_mlp(data_dir=data_dir, **params)
            return get_f1(res)
        except Exception as e:
            trial.set_user_attr("error", str(e)); return -1e9
    return _obj

def make_obj_rf(data_dir: Path, seed: int, test_size: float):
    def _obj(trial: optuna.Trial) -> float:
        max_depth_opt = trial.suggest_categorical("max_depth", ["None", 6, 8, 10, 12, 16, 20])
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=None if max_depth_opt == "None" else int(max_depth_opt),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            class_weight=trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.75, 1.0]),
            random_state=seed, test_size=test_size,
        )
        try:
            res = train_and_eval_random_forest(data_dir=data_dir, **params)
            return get_f1(res)
        except Exception as e:
            trial.set_user_attr("error", str(e)); return -1e9
    return _obj

def make_obj_svm(data_dir: Path, seed: int, test_size: float):
    def _obj(trial: optuna.Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["linear"])
        C = trial.suggest_float("C", 1e-3, 100.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True) if kernel == "rbf" else "scale"
        params = dict(kernel=kernel, C=C, gamma=gamma, cache_size=200, max_iter=30000,
                      seed=seed, test_size=test_size)
        try:
            res = train_and_eval_svm(data_dir=data_dir, **params)
            return get_f1(res)
        except Exception as e:
            trial.set_user_attr("error", str(e)); return -1e9
    return _obj

def make_obj_xgb(data_dir: Path, seed: int, test_size: float):
    def _obj(trial: optuna.Trial) -> float:
        use_auto_spw = tobool(trial, "use_auto_spw")
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            scale_pos_weight=("auto" if use_auto_spw else float(trial.suggest_float("spw", 1.0, 50.0, log=True))),
            early_stopping_rounds=50, random_state=seed, use_gpu=True,
            test_size=test_size,
        )
        try:
            res = train_and_eval_xgboost(data_dir=data_dir, **params)
            return get_f1(res)
        except Exception as e:
            trial.set_user_attr("error", str(e)); return -1e9
    return _obj


OBJECTIVES = {
    "gru": make_obj_gru,
    "lstm": make_obj_lstm,
    "mlp": make_obj_mlp,
    "rf": make_obj_rf,
    "svm": make_obj_svm,
    "xgb": make_obj_xgb,
}


# ----------------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser(description="Optuna tuning (maximize F1) for leak detection models.")
    parser.add_argument("--algos", type=str, required=True,
                        help="Comma-separated subset of {gru,lstm,mlp,rf,svm,xgb}")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)
    parser.add_argument("--storage", type=str, default="", help="Optional Optuna storage URL")
    parser.add_argument("--study_prefix", type=str, default="leaks_f1")
    args = parser.parse_args()

    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    for a in algos:
        if a not in OBJECTIVES:
            raise SystemExit(f"Unknown algo '{a}'. Valid: {list(OBJECTIVES)}")

    data_dir = Path(args.data_dir)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for algo in algos:
        print(f"\n=== Tuning {algo.upper()} (maximize F1) for {args.trials} trials ===")
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        # (Pruner is not very useful here since objective returns once per trial.)
        study_name = f"{args.study_prefix}_{algo}"
        if args.storage:
            study = optuna.create_study(direction="maximize", sampler=sampler,
                                        storage=args.storage, study_name=study_name, load_if_exists=True)
        else:
            study = optuna.create_study(direction="maximize", sampler=sampler)

        objective = OBJECTIVES[algo](data_dir=data_dir, seed=args.seed, test_size=args.test_size)
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

        print(f"[{algo}] Best F1: {study.best_value:.6f}")
        print(f"[{algo}] Best params:\n{json.dumps(study.best_params, indent=2)}")
        save_best(study, algo, MODELS_DIR)

    print("\nAll studies finished.")

if __name__ == "__main__":
    main()
