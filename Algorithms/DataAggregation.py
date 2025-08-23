from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ----------------------
# File discovery / IO
# ----------------------
def find_csvs(data_dir: Path) -> List[Path]:
    """Return all CSV files under data_dir (non-recursive)."""
    paths = sorted([p for p in Path(data_dir).glob("*.csv") if p.is_file()])
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return paths

def load_and_merge(csv_paths: List[Path]) -> pd.DataFrame:
    """
    Load all CSVs, stack them together, and sort by (Meter_ID, Timestamp).
    Expected columns:
      - Meter_ID, Timestamp, Volume
      - Volume_with_Constant_Leakage
      - Volume_with_Variable_Leakage
    """
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        # Normalize types to avoid surprises downstream
        df["Meter_ID"] = df["Meter_ID"].astype(str)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data.sort_values(["Meter_ID", "Timestamp"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

# ----------------------
# Feature helpers
# ----------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple calendar features (good baseline, cheap to compute)."""
    df = df.copy()
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Month"] = df["Timestamp"].dt.month
    return df

def add_delta_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-meter delta of Volume_Reading.
    NOTE: We group ONLY by Meter_ID (NOT by labels) to avoid label leakage.
    """
    df = df.copy()
    df.sort_values(["Meter_ID", "Timestamp"], inplace=True)
    df["Delta_Volume"] = df.groupby("Meter_ID")["Volume_Reading"].diff().fillna(0.0)
    return df

# ----------------------
# Random-segments dataset builder (BINARY)
# ----------------------
def build_dataset_random_segments(
    data_dir: Path,
    leak_mode: str = "variable",            # "variable" -> use Volume_with_Variable_Leakage inside leak segments
    leak_fraction: float = 0.30,             # ~30% of each meter’s timeline will be labeled as leak
    n_segments_range: Tuple[int,int] = (1, 3),  # choose a random number of leak episodes per meter, inclusive
    min_len_frac: float = 0.05,              # each leak episode is at least 5% of the meter’s length
    seed: int = 42                           # RNG seed for reproducibility (change to vary leak placement)
) -> pd.DataFrame:
    """
    For each meter, create LEAK EPISODES at random times (contiguous blocks):
      - Total leaked samples ≈ leak_fraction * series_len.
      - Number of episodes k is chosen uniformly from n_segments_range.
      - Each episode length >= min_len_frac * series_len.
      - Episodes do NOT overlap.

    Output columns (key ones):
      - Volume_Reading: either Volume (no leak) or one of the leakage columns (inside leak)
      - Leak_Flag: 0 (no leak), 1 (leak)
      - Leak_Type: 0 (none) / 1 (constant) / 2 (variable)
      - Hour, DayOfWeek, Month, Delta_Volume: features
    """
    assert leak_mode in {"constant", "variable"}
    rng = np.random.default_rng(seed)

    data = load_and_merge(find_csvs(data_dir))
    rows = []

    for meter, g in data.groupby("Meter_ID", sort=False):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue

        # How many total "leak" samples do we want?
        target_leak = int(np.round(leak_fraction * n))
        target_leak = max(0, min(target_leak, n))

        # Pick how many distinct leak episodes for this meter
        k_min, k_max = n_segments_range
        k = int(rng.integers(k_min, k_max + 1))

        # Enforce a minimum length for each episode (as a fraction of series)
        min_len = max(1, int(np.floor(min_len_frac * n)))

        # If the target total is too small, force a single minimal episode
        if target_leak < min_len:
            k, target_leak = 1, min_len

        # Allocate k episode lengths that sum to target_leak:
        # give each at least min_len, then spread the remaining "slack" randomly
        slack = max(0, target_leak - k * min_len)
        parts = rng.multinomial(slack, [1.0 / k] * k) + min_len  # array of k lengths

        # Place episodes at random, ensuring no overlap
        taken = np.zeros(n, dtype=bool)
        segs = []
        for L in parts:
            placed = False
            # Try random starts; if none fit, fall back to greedy
            for _ in range(200):
                start = int(rng.integers(0, n - L + 1))
                window = slice(start, start + L)
                if not taken[window].any():
                    taken[window] = True
                    segs.append((start, start + L))
                    placed = True
                    break
            if not placed:
                for s in range(0, n - L + 1):
                    window = slice(s, s + L)
                    if not taken[window].any():
                        taken[window] = True
                        segs.append((s, s + L))
                        placed = True
                        break
            # If still not placed, skip (rare for reasonable params)

        # Build the leak mask
        leak_mask = np.zeros(n, dtype=bool)
        for s, e in segs:
            leak_mask[s:e] = True

        out = g.copy()
        out["Leak_Flag"] = leak_mask.astype(int)

        if leak_mode == "constant":
            out["Volume_Reading"] = np.where(leak_mask,
                                             out["Volume_with_Constant_Leakage"],
                                             out["Volume"])
            out["Leak_Type"] = np.where(leak_mask, 1, 0)
        else:
            out["Volume_Reading"] = np.where(leak_mask,
                                             out["Volume_with_Variable_Leakage"],
                                             out["Volume"])
            out["Leak_Type"] = np.where(leak_mask, 2, 0)

        out = add_time_features(out)
        out = add_delta_volume(out)
        rows.append(out)

    combined = pd.concat(rows, ignore_index=True)
    combined.sort_values(["Meter_ID", "Timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined

# ----------------------
# Split & scale
# ----------------------
def time_based_group_split(
    df: pd.DataFrame,
    test_size: float = 0.20,                 # 20% test by default to keep some leak in train
    time_col: str = "Timestamp",
    group_cols = ["Meter_ID"],               # split per meter (leak-safe)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each meter, take the first (1 - test_size) fraction for train, rest for test.
    This preserves temporal order and prevents future information leaking into train.
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
        pd.concat(test_parts, ignore_index=True)
    )

def scale_on_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str]
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
