from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import boto3

# Project constants (column names, feature defaults, etc.)
from utils.constants import (
    MODELS_DIR,
    METER_ID_COL, TIMESTAMP_COL, LEAK_FLAG_COL,
    CLS_FEATURE_COLS
)

def load_meta(meta_path: Path) -> dict:
    """Load model metadata (feature order, window length, threshold, etc.)."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))

def load_frozen_test_df(test_path: Path) -> pd.DataFrame:
    """
    Load the frozen test parquet used during training/validation to guarantee
    the same schema and distribution. We sort and reindex for deterministic windowing.
    """
    if not test_path.exists():
        raise FileNotFoundError(f"Frozen test file not found: {test_path}")
    df = pd.read_parquet(test_path)
    # Basic schema checks
    needed = list(CLS_FEATURE_COLS) + [METER_ID_COL, TIMESTAMP_COL, LEAK_FLAG_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Frozen test is missing columns: {missing}")
    return df.sort_values([METER_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)

def build_test_windows_with_meta(
    test_df: pd.DataFrame,
    feature_cols: list[str],
    window: int,
    stride: int = 1,
):
    """
    Convert long-form time series into sliding windows per meter.

    Returns:
        X_windows_raw: np.ndarray [N, window, F]  (unscaled raw features)
        meta_df: DataFrame with per-window metadata:
                 - Meter_ID
                 - Timestamp_end  (timestamp of the window's last row)
                 - Hour           (if available; -1 otherwise)
                 - y_true         (label at the last row of the window)
    """
    X_windows, meta_rows = [], []
    for m, g in test_df.groupby(METER_ID_COL, sort=False):
        g = g.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        if len(g) < window:
            continue
        # Slide over the series with [window] length, [stride] step
        for end in range(window - 1, len(g), stride):
            sl = g.iloc[end - window + 1 : end + 1]
            X_windows.append(sl[feature_cols].to_numpy(dtype=np.float32))
            meta_rows.append({
                "Meter_ID": m,
                "Timestamp_end": sl.iloc[-1][TIMESTAMP_COL],
                "Hour": int(sl.iloc[-1]["Hour"]) if "Hour" in sl.columns else -1,
                "y_true": int(sl.iloc[-1][LEAK_FLAG_COL]),
            })
    if not X_windows:
        empty_X = np.empty((0, window, len(feature_cols)), dtype=np.float32)
        empty_meta = pd.DataFrame(columns=["Meter_ID", "Timestamp_end", "Hour", "y_true"])
        return empty_X, empty_meta

    return np.stack(X_windows, axis=0), pd.DataFrame(meta_rows)


def pick_balanced_sample(
    Xw_raw: np.ndarray,           # kept for signature parity (not used directly)
    meta_df: pd.DataFrame,
    n: int,
    seed: int,
):
    """
    Try to pick a small, balanced sample by label (one leak, one normal) to sanity-check
    the endpoint. If we can't balance, fall back to random selection.

    Returns:
        idx_list (list[int]): indices into Xw_raw/meta_df
        meta_sample_df (DataFrame): subset with matching order
    """
    rng = np.random.default_rng(seed)
    leak_idx = meta_df.index[meta_df["y_true"] == 1].to_numpy()
    norm_idx = meta_df.index[meta_df["y_true"] == 0].to_numpy()

    idx_pick: list[int] = []

    if n >= 2 and len(leak_idx) > 0 and len(norm_idx) > 0:
        # Ensure we include at least one positive and one negative
        idx_pick.append(rng.choice(leak_idx))
        idx_pick.append(rng.choice(norm_idx))

        # Fill remaining slots from the rest (without replacement)
        rest_pool = np.setdiff1d(meta_df.index.to_numpy(), np.array(idx_pick))
        if len(rest_pool) > 0:
            extra = min(n - 2, len(rest_pool))
            if extra > 0:
                idx_pick.extend(rng.choice(rest_pool, size=extra, replace=False).tolist())
    else:
        # Pure random sample if we can't balance (e.g., class extremely imbalanced)
        all_idx = meta_df.index.to_numpy()
        if len(all_idx) == 0:
            idx_pick = []
        else:
            k = min(n, len(all_idx))
            idx_pick = rng.choice(all_idx, size=k, replace=False).tolist()

    meta_sample = meta_df.loc[idx_pick].reset_index(drop=True) if idx_pick else meta_df.iloc[[]]
    return idx_pick, meta_sample


def invoke_endpoint(
    endpoint: str,
    region: str,
    windows: np.ndarray,
    feature_cols: list[str],
    thr: float | None,
):
    """
    Call the SageMaker endpoint (sagemaker-runtime) with a JSON payload.

    The endpoint expects:
      - 'windows': [B, W, F] float32
      - 'feature_cols': list[str] enforcing column order
      - 'thr' (optional): decision threshold override
    """
    rt = boto3.client("sagemaker-runtime", region_name=region)
    payload = {
        "windows": windows.tolist(),
        "feature_cols": feature_cols
    }
    if thr is not None:
        payload["thr"] = float(thr)

    resp = rt.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    # Body is a streaming object; read and decode JSON result
    body = resp["Body"].read().decode("utf-8")
    return json.loads(body)


def main():
    ap = argparse.ArgumentParser(description="Send a few test windows to a SageMaker endpoint for E2E demo.")
    ap.add_argument("--endpoint_name", type=str, required=True, help="SageMaker endpoint name (e.g. leak-gru-endpoint)")
    ap.add_argument("--region", type=str, default="eu-central-1")
    ap.add_argument("--test_path", type=str, default=str(MODELS_DIR / "artifacts/test_agg.parquet"))
    ap.add_argument("--meta_path", type=str, default=str(MODELS_DIR / "gru_cls_meta.json"))
    ap.add_argument("--n", type=int, default=2, help="number of windows to send")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--thr", type=float, default=None, help="optional threshold override per request")
    args = ap.parse_args()

    # Load meta & frozen test
    meta = load_meta(Path(args.meta_path))
    feature_cols = list(meta.get("feature_cols", CLS_FEATURE_COLS))
    window = int(meta.get("window", 48))

    test_df = load_frozen_test_df(Path(args.test_path))

    # Build sliding windows over the test set
    Xw_raw, meta_df = build_test_windows_with_meta(test_df, feature_cols, window=window, stride=1)
    if Xw_raw.shape[0] == 0:
        print("[ERROR] No windows built from test set. Check window length vs data coverage.")
        return

    # Pick a small sample (try balanced by class if possible)
    idx_pick, meta_sample = pick_balanced_sample(Xw_raw, meta_df, n=args.n, seed=args.seed)
    if not idx_pick:
        print("[ERROR] Could not pick any sample windows.")
        return
    X_send = Xw_raw[idx_pick]

    # Call endpoint
    result = invoke_endpoint(
        endpoint=args.endpoint_name,
        region=args.region,
        windows=X_send,
        feature_cols=feature_cols,
        thr=args.thr,
    )

    probs = np.array(result.get("probs", []), dtype=float)
    preds = np.array(result.get("preds", []), dtype=int)
    thr_used = float(result.get("thr", meta.get("thr", 0.5)))

    # Print predictions, probs & ground truth leak flag
    print("\n=== E2E PREDICTIONS ===")
    print(f"endpoint = {args.endpoint_name} | region = {args.region} | thr_used = {thr_used:.3f}")
    print("Meter_ID | Timestamp_end        | y_true |  prob  | pred")
    for i, r in meta_sample.iterrows():
        # Keep meta_sample order aligned with probs/preds
        yi = int(r["y_true"]) if i < len(preds) else -1
        pi = probs[i] if i < len(probs) else float("nan")
        hi = int(preds[i]) if i < len(preds) else -1
        print(f"{str(r['Meter_ID'])} | {r['Timestamp_end']} |   {yi:>1}    | {pi:6.3f} |  {hi}")

    if len(probs) == len(meta_sample):
        acc = float((preds == meta_sample["y_true"].to_numpy()).mean())
        print(f"\n[SAMPLE ACC] {acc:.3f} on {len(preds)} windows")

if __name__ == "__main__":
    main()
