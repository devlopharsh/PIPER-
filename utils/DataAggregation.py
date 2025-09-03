# utils/DataAggregation.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.constants import (
    METER_ID_COL, TIMESTAMP_COL,
    VOLUME_COL, VOLUME_READING_COL, DELTA_VOLUME_COL,
    LEAK_FLAG_COL, LEAK_TYPE_COL, FORECAST_Y_COL,
    CALENDAR_COLS,
)

#region Find CSVs
def find_csvs(data_dir: Path) -> List[Path]:
    """Return all CSV files under data_dir (non-recursive)."""
    paths = sorted([p for p in Path(data_dir).glob("*.csv") if p.is_file()])
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return paths
#endregion


#region Load CSVs
def load_and_merge(csv_paths: List[Path]) -> pd.DataFrame:
    """
    Load all CSVs, stack them together, and sort by (METER_ID_COL, TIMESTAMP_COL).
    Expected columns also include the synthetic leak variants:
      - 'Volume_with_Constant_Leakage'
      - 'Volume_with_Variable_Leakage'
    """
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df[METER_ID_COL] = df[METER_ID_COL].astype(str)
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data.sort_values([METER_ID_COL, TIMESTAMP_COL], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data
#endregion


#region Feature helpers
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features listed in CALENDAR_COLS."""
    df = df.copy()
    if "Hour" in CALENDAR_COLS:
        df["Hour"] = df[TIMESTAMP_COL].dt.hour
    if "DayOfWeek" in CALENDAR_COLS:
        df["DayOfWeek"] = df[TIMESTAMP_COL].dt.dayofweek
    if "Month" in CALENDAR_COLS:
        df["Month"] = df[TIMESTAMP_COL].dt.month
    return df


def add_delta_volume(
    df: pd.DataFrame,
    col: str = VOLUME_READING_COL,
    new_col: str = DELTA_VOLUME_COL,
) -> pd.DataFrame:
    """
    Per-meter first difference of the given column (default: VOLUME_READING_COL).
    Creates/overwrites `new_col` (default: DELTA_VOLUME_COL).
    Group ONLY by METER_ID_COL to avoid label leakage.
    """
    df = df.copy()
    df.sort_values([METER_ID_COL, TIMESTAMP_COL], inplace=True)
    df[new_col] = df.groupby(METER_ID_COL)[col].diff().fillna(0.0)
    return df
#endregion


#region  Classification builder
def inject_random_leak_segments(
    df_raw: pd.DataFrame,
    leak_mode: Literal["constant", "variable"] = "variable",
    leak_fraction: float = 0.30,
    n_segments_range: Tuple[int, int] = (1, 3),
    min_len_frac: float = 0.05,
    seed: int = 42,
    leak_placement: Literal["random", "tail"] = "random",
    tail_fraction: float = 0.30,
) -> pd.DataFrame:
    """
    Create labeled leak episodes per meter and set VOLUME_READING_COL accordingly.

    leak_placement:
      - "random": (default) leak episodes can appear anywhere
      - "tail":   leak episodes restricted to last `tail_fraction` of each meter's series
    """
    assert leak_mode in {"constant", "variable"}
    assert 0 < tail_fraction <= 1.0, "tail_fraction must be in (0, 1]"
    rng = np.random.default_rng(seed)
    rows = []

    for _, g in df_raw.groupby(METER_ID_COL, sort=False):
        g = g.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue

        target_leak = int(np.round(leak_fraction * n))
        target_leak = max(0, min(target_leak, n))

        k_min, k_max = n_segments_range
        k = int(rng.integers(k_min, k_max + 1))
        min_len = max(1, int(np.floor(min_len_frac * n)))

        if target_leak < min_len:
            k, target_leak = 1, min_len

        slack = max(0, target_leak - k * min_len)
        parts = rng.multinomial(slack, [1.0 / k] * k) + min_len  # segment lengths

        if leak_placement == "tail":
            tail_start = max(0, n - int(np.floor(tail_fraction * n)))
        else:
            tail_start = 0

        taken = np.zeros(n, dtype=bool)
        segs = []
        for L in parts:
            placed = False
            start_low, start_high = tail_start, n - L
            if start_high < start_low:
                start_low, start_high = 0, n - L

            for _ in range(200):
                start = int(rng.integers(start_low, start_high + 1))
                window = slice(start, start + L)
                if not taken[window].any():
                    taken[window] = True
                    segs.append((start, start + L))
                    placed = True
                    break

            if not placed:
                for s in range(start_low, start_high + 1):
                    window = slice(s, s + L)
                    if not taken[window].any():
                        taken[window] = True
                        segs.append((s, s + L))
                        placed = True
                        break

        leak_mask = np.zeros(n, dtype=bool)
        for s, e in segs:
            leak_mask[s:e] = True

        out = g.copy()
        out[LEAK_FLAG_COL] = leak_mask.astype(int)

        if leak_mode == "constant":
            out[VOLUME_READING_COL] = np.where(
                leak_mask, out["Volume_with_Constant_Leakage"], out[VOLUME_COL]
            )
            out[LEAK_TYPE_COL] = np.where(leak_mask, 1, 0)
        else:
            # Scale up variable leak effect
            leak_increase = out["Volume_with_Variable_Leakage"] - out[VOLUME_COL]
            amplified = out[VOLUME_COL] + leak_increase * 100  # 3–10 L becomes 300–1000 L

            out[VOLUME_READING_COL] = np.where(
                leak_mask, amplified, out[VOLUME_COL]
            )
            out[LEAK_TYPE_COL] = np.where(leak_mask, 2, 0)

        out = add_time_features(out)
        out = add_delta_volume(out, col=VOLUME_READING_COL, new_col=DELTA_VOLUME_COL)
        rows.append(out)

    combined = pd.concat(rows, ignore_index=True)
    combined.sort_values([METER_ID_COL, TIMESTAMP_COL], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined
#endregion


#region  Forecasting builder
def build_forecasting_frame(
    df_raw: pd.DataFrame,
    target_col: Literal[VOLUME_READING_COL, DELTA_VOLUME_COL] = VOLUME_READING_COL,
    lags: int = 24,
    rolling_windows: Tuple[int, ...] = (6, 24),
    include_labels: bool = False,   # label-free by default
) -> pd.DataFrame:
    """
    Create past-only features for forecasting:
      - FORECAST_Y_COL: target at time t (VOLUME_READING_COL or DELTA_VOLUME_COL)
      - lag_1..lag_L of the chosen target
      - rolling (mean/std) with shift(1)
      - optional calendar features
    """
    df = df_raw.sort_values([METER_ID_COL, TIMESTAMP_COL]).copy()

    if VOLUME_READING_COL not in df.columns:
        df[VOLUME_READING_COL] = df[VOLUME_COL]

    if target_col == DELTA_VOLUME_COL:
        tmp = add_delta_volume(df, col=VOLUME_READING_COL, new_col=DELTA_VOLUME_COL)
        df["TargetBase"] = tmp[DELTA_VOLUME_COL]
    else:
        df["TargetBase"] = df[VOLUME_READING_COL]

    out = []
    for _, g in df.groupby(METER_ID_COL, sort=False):
        g = g.sort_values(TIMESTAMP_COL).reset_index(drop=True)

        # NEW (drop-in)
        tb = g["TargetBase"]  # shorthand

        # build lags all at once
        lag_df = (
            pd.DataFrame({f"lag_{k}": tb.shift(k) for k in range(1, lags + 1)}, index=g.index)
            if lags > 0 else pd.DataFrame(index=g.index)
        )

        # build rolling stats all at once (past-only via shift(1))
        past = tb.shift(1)
        roll_dict = {}
        for w in rolling_windows:
            roll_dict[f"roll{w}_mean"] = past.rolling(w, min_periods=w).mean()
            roll_dict[f"roll{w}_std"]  = past.rolling(w, min_periods=w).std()
        roll_df = pd.DataFrame(roll_dict, index=g.index) if rolling_windows else pd.DataFrame(index=g.index)

        # attach features in one shot
        g = pd.concat([g, lag_df, roll_df], axis=1)


        g[FORECAST_Y_COL] = g["TargetBase"]

        warmup = max(lags, max(rolling_windows) if rolling_windows else 0)
        g = g.iloc[warmup:].reset_index(drop=True)
        out.append(g)

    full = pd.concat(out, ignore_index=True)

    keep = [METER_ID_COL, TIMESTAMP_COL, FORECAST_Y_COL]
    keep += [c for c in full.columns if c.startswith("lag_") or c.startswith("roll")]
    for cal in CALENDAR_COLS:
        if cal in full.columns:
            keep.append(cal)

    if include_labels:
        for maybe in [LEAK_FLAG_COL, LEAK_TYPE_COL]:
            if maybe in full.columns:
                keep.append(maybe)

    return full[keep]
#endregion


#region  Task-agnostic split/scale
def time_based_group_split(
    df: pd.DataFrame,
    test_size: float = 0.30,
    time_col: str = TIMESTAMP_COL,
    group_cols: List[str] = [METER_ID_COL],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each group (meter), take the first (1 - test_size) fraction for train and the rest for test.
    """
    train_parts, test_parts = [], []
    df = df.sort_values(group_cols + [time_col])
    for _, g in df.groupby(group_cols, sort=False):
        n = len(g)
        split_idx = int(np.floor((1.0 - test_size) * n))
        train_parts.append(g.iloc[:split_idx])
        test_parts.append(g.iloc[split_idx:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )



def scale_on_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit scaler on TRAIN ONLY (to avoid leakage), apply to both train and test.
    """
    scaler = StandardScaler()
    tr = train_df.copy()
    te = test_df.copy()
    tr[feature_cols] = scaler.fit_transform(tr[feature_cols])
    te[feature_cols] = scaler.transform(te[feature_cols])
    return tr, te, scaler
#endregion


#region  Build Aggregated Dataset
def build_aggregated_dataset(
    data_dir: Path,
    use_forecast: bool,
    # classification knobs
    leak_mode: Literal["constant", "variable"] = "variable",
    leak_fraction: float = 0.30,
    n_segments_range: Tuple[int, int] = (1, 3),
    min_len_frac: float = 0.05,
    # forecasting knobs
    target_col: Literal[VOLUME_READING_COL, DELTA_VOLUME_COL] = VOLUME_READING_COL,
    lags: int = 24,
    rolling_windows: Tuple[int, ...] = (6, 24),
    # placement knob for forecasting test contamination
    tail_fraction: float = 0.30,
    # control label presence in forecast frames
    forecast_include_labels: bool = False,
) -> pd.DataFrame:
    """
    Classification (use_forecast=False):
        Returns columns needed for classification:
          VOLUME_READING_COL, *CALENDAR_COLS, DELTA_VOLUME_COL, LEAK_FLAG_COL, LEAK_TYPE_COL, ...
        Leaks are placed randomly along the series.

    Forecasting (use_forecast=True):
        Returns columns: [METER_ID_COL, TIMESTAMP_COL, FORECAST_Y_COL, lag_*, roll*, (optional calendar), (optional labels)]
        Leaks are injected only in the tail (last `tail_fraction`) so time-based splitting
        can yield clean train and leaky test if you set test_size ≈ tail_fraction.
    """
    raw = load_and_merge(find_csvs(data_dir))

    if not use_forecast:
        return inject_random_leak_segments(
            raw,
            leak_mode=leak_mode,
            leak_fraction=leak_fraction,
            n_segments_range=n_segments_range,
            min_len_frac=min_len_frac,
            leak_placement="random",
        )

    # forecasting: place leaks only in the tail
    df_with_labels = inject_random_leak_segments(
        raw,
        leak_mode=leak_mode,
        leak_fraction= 0.5 * tail_fraction, # half of the tail contains leaks,
        n_segments_range=n_segments_range,
        min_len_frac=min_len_frac,
        leak_placement="tail",
        tail_fraction=tail_fraction,
    )

    # Build lag/rolling features.
    # Keep labels only if forecast_include_labels=True.
    return build_forecasting_frame(
        df_with_labels,
        target_col=target_col,
        lags=lags,
        rolling_windows=rolling_windows,
        include_labels=forecast_include_labels,
    )
#endregion
