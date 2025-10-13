from __future__ import annotations
import argparse
from enum import Enum
from pathlib import Path
from typing import Any


class AlgoKind(Enum):
    """Marker enum for the supported algorithm families."""
    MLP  = "mlp"
    GRU  = "gru"
    LSTM = "lstm"
    RF   = "rf"
    SVM  = "svm"
    XGB  = "xgb"


def create_common_parser(
    description: str,
    *,
    data_dir_default: Path,
    test_size_default: float,
    seed_default: int,
    n_segments_min_default: int = 2,
    n_segments_max_default: int = 5,
    min_len_frac_default: float = 0.05,
) -> argparse.ArgumentParser:
    """
    Build a parser with flags used by ALL algorithms.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data_dir", type=Path, default=Path(str(data_dir_default)),
                   help="Root data directory.")
    p.add_argument("--seed", type=int, default=int(seed_default),
                   help="Random seed for reproducibility.")
    p.add_argument("--test_size", type=float, default=float(test_size_default),
                   help="Hold-out fraction for test split (per-meter, time-aware).")
    # Synthetic leak generator knobs (shared across scripts)
    p.add_argument("--n_segments_min", type=int, default=int(n_segments_min_default),
                   help="Minimum number of leak segments per meter.")
    p.add_argument("--n_segments_max", type=int, default=int(n_segments_max_default),
                   help="Maximum number of leak segments per meter.")
    p.add_argument("--min_len_frac", type=float, default=float(min_len_frac_default),
                   help="Minimum fraction of a series length used when creating segments.")
    return p


def attach_additional_args(
    parser: argparse.ArgumentParser,
    algo: AlgoKind,
    **defaults: Any,
) -> None:
    """
    Add ONLY the extra args specific to the chosen algorithm.
    Usage:
      attach_additional_args(p, AlgoKind.LSTM, window=24, stride=1, hidden=64, ...)
    """
    if not isinstance(algo, AlgoKind):
        raise TypeError(f"'algo' must be AlgoKind, got {type(algo).__name__}")

    # ---------- MLP ----------
    if algo is AlgoKind.MLP:
        parser.add_argument("--hidden", nargs="+", type=int, default=list(defaults.get("hidden", (128, 64))),
                            help="MLP hidden sizes, e.g., --hidden 128 64.")
        parser.add_argument("--dropout", type=float, default=float(defaults.get("dropout", 0.2)),
                            help="Dropout after each hidden layer.")
        # Optim
        parser.add_argument("--batch_size", type=int, default=int(defaults.get("batch_size", 256)))
        parser.add_argument("--lr", type=float, default=float(defaults.get("lr", 1e-3)))
        parser.add_argument("--weight_decay", type=float, default=float(defaults.get("weight_decay", 1e-5)))
        parser.add_argument("--max_epochs", type=int, default=int(defaults.get("max_epochs", 10)))
        parser.add_argument("--early_patience", type=int, default=int(defaults.get("early_patience", 4)))
        return

    # ---------- GRU / LSTM (shared) ----------
    if algo in (AlgoKind.GRU, AlgoKind.LSTM):
        parser.add_argument("--window", type=int, default=int(defaults.get("window", 24)), help="Sequence length (hours per window).")
        parser.add_argument("--stride", type=int, default=int(defaults.get("stride", 1)),
                            help="Window step in hours for sliding-window datasets.")
        parser.add_argument("--hidden", type=int, default=int(defaults.get("hidden", 64)),
                            help="RNN hidden size.")
        parser.add_argument("--num_layers", type=int, default=int(defaults.get("num_layers", 1)),
                            help="Number of stacked RNN layers.")
        try:
            action = argparse.BooleanOptionalAction  # py3.9+
            parser.add_argument("--bidirectional", action=action, default=bool(defaults.get("bidirectional", False)),
                                help="Use bidirectional RNN (use --no-bidirectional to force False).")
        except AttributeError:
            parser.add_argument("--bidirectional", action="store_true", default=bool(defaults.get("bidirectional", False)),
                                help="Use bidirectional RNN (no --no-bidirectional on older Python).")
        parser.add_argument("--dropout", type=float, default=float(defaults.get("dropout", 0.2)),
                            help="Dropout between stacked RNN layers (ignored if num_layers=1).")
        # Optim
        parser.add_argument("--batch_size", type=int, default=int(defaults.get("batch_size", 256)))
        parser.add_argument("--lr", type=float, default=float(defaults.get("lr", 1e-3)))
        parser.add_argument("--weight_decay", type=float, default=float(defaults.get("weight_decay", 1e-5)))
        parser.add_argument("--max_epochs", type=int, default=int(defaults.get("max_epochs", 20)))
        parser.add_argument("--early_patience", type=int, default=int(defaults.get("early_patience", 6)))
        return

    # ---------- Random Forest ----------
    if algo is AlgoKind.RF:
        parser.add_argument("--n_estimators", type=int, default=int(defaults.get("n_estimators", 300)))
        parser.add_argument("--max_depth", type=str, default=str(defaults.get("max_depth", "None")))
        parser.add_argument("--min_samples_leaf", type=int, default=int(defaults.get("min_samples_leaf", 1)))
        parser.add_argument("--random_state", type=int, default=int(defaults.get("random_state", defaults.get("seed", 42))))
        parser.add_argument("--class_weight", type=str, default=str(defaults.get("class_weight", "balanced_subsample")))
        parser.add_argument("--max_features", type=str, default=str(defaults.get("max_features", "sqrt")))
        if "test_fraction" in defaults:
            parser.add_argument("--test_fraction", type=float, default=float(defaults["test_fraction"]))
        return

    # ---------- SVM ----------
    if algo is AlgoKind.SVM:
        parser.add_argument("--kernel", type=str, choices=["linear", "rbf", "rbf_approx"],
                            default=str(defaults.get("kernel", "linear")),
                            help="SVM kernel.")
        parser.add_argument("--C", type=float, default=float(defaults.get("C", 1.0)))
        parser.add_argument("--gamma", type=str, default=str(defaults.get("gamma", "scale")),
                            help="For RBF: 'scale'/'auto' or numeric string.")
        parser.add_argument("--cache_size", type=int, default=int(defaults.get("cache_size", 200)))
        parser.add_argument("--max_iter", type=int, default=int(defaults.get("max_iter", 30000)))
        return

    # ---------- XGBoost ----------
    if algo is AlgoKind.XGB:
        parser.add_argument("--n_estimators", type=int, default=int(defaults.get("n_estimators", 1500)))
        parser.add_argument("--learning_rate", type=float, default=float(defaults.get("learning_rate", 0.05)))
        parser.add_argument("--max_depth", type=int, default=int(defaults.get("max_depth", 6)))
        parser.add_argument("--min_child_weight", type=float, default=float(defaults.get("min_child_weight", 1.0)))
        parser.add_argument("--subsample", type=float, default=float(defaults.get("subsample", 0.8)))
        parser.add_argument("--colsample_bytree", type=float, default=float(defaults.get("colsample_bytree", 0.8)))
        parser.add_argument("--reg_lambda", type=float, default=float(defaults.get("reg_lambda", 1.0)))
        parser.add_argument("--reg_alpha", type=float, default=float(defaults.get("reg_alpha", 0.0)))
        parser.add_argument("--gamma", type=float, default=float(defaults.get("gamma", 0.0)))
        parser.add_argument("--scale_pos_weight", type=str, default=str(defaults.get("scale_pos_weight", "auto")))
        parser.add_argument("--early_stopping_rounds", type=int, default=int(defaults.get("early_stopping_rounds", 50)))
        parser.add_argument("--random_state", type=int, default=int(defaults.get("random_state", defaults.get("seed", 42))))
        parser.add_argument("--use_gpu", action="store_true", default=bool(defaults.get("use_gpu", False)))
        return

    # Safety catch (should never hit)
    raise RuntimeError(f"Unhandled AlgoKind: {algo}")
