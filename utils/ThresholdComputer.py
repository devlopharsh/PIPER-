import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Tuple, List


def choose_thresholds(y_true: np.ndarray, probs: np.ndarray, beta: float = 1.0) -> Dict:
    """
    Compute precision-recall curve and find the best threshold 
    that maximizes the F-beta score.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels.
    probs : array-like, shape (n_samples,)
        Predicted probabilities (or decision function values).
    beta : float, default=1.0
        Weight of recall in F-beta score (beta > 1 favours recall).
    
    Returns
    -------
    dict with best threshold and metrics at that threshold
    """
    p, r, thr = precision_recall_curve(y_true, probs)
    thr = np.append(thr, 1.0)  # align lengths

    f_beta = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    i = int(np.nanargmax(f_beta))
    best_thr = thr[i]

    return {
        "best_thr": float(best_thr),
        "best_fbeta": float(f_beta[i]),
        "best_prec": float(p[i]),
        "best_rec": float(r[i]),
        "ap": float(average_precision_score(y_true, probs)),
    }


def find_stable_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    beta: float = 1.0,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[float, List[float]]:
    """
    Find a robust threshold by averaging best thresholds across CV folds.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    probs : array-like
        Predicted probabilities (or decision function values).
    beta : float, default=1.0
        Weight of recall in F-beta score.
    n_splits : int, default=5
        Number of folds in StratifiedKFold.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    (mean_threshold, list_of_thresholds)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    thresholds = []
    for _, val_idx in skf.split(probs, y_true):
        y_val, p_val = y_true[val_idx], probs[val_idx]
        out = choose_thresholds(y_val, p_val, beta=beta)
        thresholds.append(out["best_thr"])
    return float(np.mean(thresholds)), thresholds
