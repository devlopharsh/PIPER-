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
    LEAK_FLAG_COL, FORECAST_Y_COL,
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
    leak_share_total: float = 0.20,
    n_segments_range: Tuple[int, int] = (1, 3),
    min_len_frac: float = 0.05,
    seed: int = 42,
    placement: Literal["random", "tail"] = "random",
    test_fraction: float = 0.30,
    ensure_test_leak_abs_share: float = 0.0,
) -> pd.DataFrame:
    """
    Inject persistent leak episodes per meter.

    Guarantees (per meter):
      • ~leak_share_total of rows are leaks (default 20%).
      • If placement=='tail' → push (almost) all leaked points into the last test_fraction of the series.
      • If placement=='random' and ensure_test_leak_abs_share>0 → ensure at least that absolute share is leaked in test.
    """
    assert 0.0 < leak_share_total <= 1.0
    assert 0.0 <= ensure_test_leak_abs_share <= leak_share_total
    assert 0.0 < test_fraction <= 1.0

    rng = np.random.default_rng(seed)
    rows = []

    for _, g in df_raw.groupby(METER_ID_COL, sort=False):
        g = g.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue

        total_leak_pts = int(round(leak_share_total * n))

        test_len   = int(round(test_fraction * n))
        test_start = max(0, n - test_len)

        # Decide how many leaked points MUST land in test vs train
        if placement == "tail":
            desired_test_pts  = min(total_leak_pts, test_len)  # put all we can into test
            desired_train_pts = total_leak_pts - desired_test_pts
        else:
            # Allocate exactly 5% of all rows as leaks in test
            desired_test_pts = int(round(ensure_test_leak_abs_share * n))
            desired_test_pts = min(desired_test_pts, test_len)

            # The rest go to train, no spillover allowed
            desired_train_pts = total_leak_pts - desired_test_pts
            desired_train_pts = min(desired_train_pts, n - test_len)  # safety clamp

            # If rounding/clamping caused mismatch, fix it here
            if desired_test_pts + desired_train_pts != total_leak_pts:
                # Force adjustment into train, never test
                desired_train_pts = total_leak_pts - desired_test_pts


        # Helper to sample segment lengths that sum ~ target_pts
        def sample_parts(target_pts: int) -> list[int]:
            if target_pts <= 0: return []
            k_min, k_max = n_segments_range
            # minimal segment length based on series size
            seg_min = max(1, int(np.floor(min_len_frac * n)))
            # pick number of segments
            k = int(rng.integers(k_min, k_max + 1))
            if target_pts < seg_min:
                return [target_pts]
            # clamp k so base fits
            k = max(1, min(k, target_pts // seg_min))
            base = seg_min * k
            slack = max(0, target_pts - base)
            parts = (rng.multinomial(slack, [1.0 / k] * k) + seg_min).astype(int)
            return list(map(int, parts))

        test_parts  = sample_parts(desired_test_pts)
        train_parts = sample_parts(desired_train_pts)
        rng.shuffle(test_parts); rng.shuffle(train_parts)

        taken = np.zeros(n, dtype=bool)
        segs: list[tuple[int,int]] = []

        def place_one(L: int, start_low: int, start_high: int) -> tuple[bool, tuple[int,int] | None]:
            for _ in range(200):
                if start_high < start_low: break
                s = int(rng.integers(start_low, start_high + 1))
                e = s + L
                if e > n: continue
                w = slice(s, e)
                if not taken[w].any():
                    taken[w] = True
                    return True, (s, e)
            for s in range(start_low, max(start_high, start_low) + 1):
                e = s + L
                if e > n: break
                w = slice(s, e)
                if not taken[w].any():
                    taken[w] = True
                    return True, (s, e)
            return False, None

        # 1) Place test segments strictly inside test slice
        for L in test_parts:
            ok, seg = place_one(L, test_start, n - L)
            if ok: segs.append(seg)

        # 2) Place train segments strictly inside train slice
        for L in train_parts:
            ok, seg = place_one(L, 0, max(0, test_start - L))
            if ok: segs.append(seg)

        # If still short (due to packing), try anywhere as last resort
        placed_pts = sum((e - s) for s, e in segs)
        deficit = total_leak_pts - placed_pts
        while deficit > 0:
            L = min(deficit, max(1, int(np.floor(min_len_frac * n))))
            ok, seg = place_one(L, 0, n - L)
            if not ok: break
            segs.append(seg); deficit -= L

        # Build mask
        leak_mask = np.zeros(n, dtype=bool)
        for s, e in segs:
            leak_mask[s:e] = True
        # Hard cap to exact total_leak_pts if we overshot
        if leak_mask.sum() > total_leak_pts:
            excess = leak_mask.sum() - total_leak_pts
            idx = np.where(leak_mask)[0]
            leak_mask[idx[-excess:]] = False

        out = g.copy()
        out[LEAK_FLAG_COL] = leak_mask.astype(int)

        if leak_mode == "constant":
            out[VOLUME_READING_COL] = np.where(leak_mask, out["Volume_with_Constant_Leakage"], out[VOLUME_COL])
        else:
            out = add_delta_volume(out, col=VOLUME_COL, new_col="__DeltaBase")
            base_flow = out["__DeltaBase"].clip(lower=0)
            avg_flow  = float(base_flow[base_flow > 0].median() or base_flow.mean() or 1.0)
            alpha     = float(rng.uniform(0.1, 0.20))     # 10–20% of typical hourly usage
            leak_rate = alpha * avg_flow
            extra_delta = np.where(leak_mask, leak_rate, 0.0).astype(np.float64)
            extra_cum   = np.cumsum(extra_delta)
            out[VOLUME_READING_COL] = out[VOLUME_COL] + extra_cum
            out.drop(columns=["__DeltaBase"], errors="ignore", inplace=True)

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
        for maybe in [LEAK_FLAG_COL]:
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
    n_segments_range: Tuple[int, int] = (1, 3),
    min_len_frac: float = 0.05,
    # forecasting knobs
    target_col: Literal[VOLUME_READING_COL, DELTA_VOLUME_COL] = VOLUME_READING_COL,
    lags: int = 24,
    rolling_windows: Tuple[int, ...] = (6, 24),
    test_fraction: float = 0.30,
    forecast_include_labels: bool = False,
) -> pd.DataFrame:
    raw = load_and_merge(find_csvs(data_dir))

    if not use_forecast:
        df = inject_random_leak_segments(
            raw,
            leak_mode=leak_mode,
            leak_share_total=0.20,
            n_segments_range=n_segments_range,
            min_len_frac=min_len_frac,
            placement="random",
            test_fraction=test_fraction,
            ensure_test_leak_abs_share=0.05,       # 5% of ALL rows must be leaked in test
        )
        return df

    df_with_labels = inject_random_leak_segments(
        raw,
        leak_mode=leak_mode,
        leak_share_total=0.20,
        n_segments_range=n_segments_range,
        min_len_frac=min_len_frac,
        placement="tail",
        test_fraction=test_fraction,
        ensure_test_leak_abs_share=0.0,
    )

    return build_forecasting_frame(
        df_with_labels,
        target_col=target_col,
        lags=lags,
        rolling_windows=rolling_windows,
        include_labels=forecast_include_labels,
    )
#endregion
