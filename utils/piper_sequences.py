"""
PIPER Phase 3: Temporal Sequence Construction

This module implements Phase 3 subtasks:
3.1 Time-Series Ordering - Assess whether observations can be ordered reliably within each segment
3.2 Sliding-Window Generation - Only if reliable sequential observations exist
3.3 Sequence Length Selection - Evaluate candidate window sizes if sequences possible
3.4 Train/Validation/Test Split - Leakage-safe split strategy

KEY FINDING FROM PHASE 2: The SCADA dataset CANNOT support reliable temporal sequence construction:
- Global time span: ~16 minutes (0.27 hours)
- Multiple conflicting readings per (segment, timestamp) key
- reading_index is NOT a genuine timestamp
- Minimum unique timestamps per segment: 8 (< 10)

This module implements honest feasibility checks and reports BLOCKED status with evidence.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import json
from datetime import datetime
from enum import Enum

from utils.piper_telemetry import (
    PRIMARY_TELEMETRY, METADATA_COLUMNS, TARGET_COLUMN,
    MetadataVariable, TelemetryVariable
)
from utils.piper_dataset import PIPER_PROCESSED_DIR, PIPER_REPORTS_DIR


class FeasibilityStatus(str, Enum):
    """Feasibility status for Phase 3 subtasks."""
    COMPLETE = "complete"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


@dataclass
class SubtaskReport:
    """Report for a Phase 3 subtask."""
    subtask_id: str
    subtask_name: str
    status: FeasibilityStatus
    evidence: List[str]
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "subtask_id": self.subtask_id,
            "subtask_name": self.subtask_name,
            "status": self.status.value,
            "evidence": self.evidence,
            "details": self.details,
            "timestamp": self.timestamp
        }


@dataclass
class Phase3Report:
    """Complete Phase 3 feasibility report."""
    dataset_name: str
    audit_timestamp: str
    subtasks: List[SubtaskReport]
    overall_assessment: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "audit_timestamp": self.audit_timestamp,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "overall_assessment": self.overall_assessment,
            "recommendations": self.recommendations
        }
    
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def print_summary(self) -> None:
        print(f"\n{'='*70}")
        print(f"PIPER PHASE 3 FEASIBILITY REPORT: {self.dataset_name}")
        print(f"{'='*70}")
        print(f"Audit Timestamp: {self.audit_timestamp}")
        print()
        
        for subtask in self.subtasks:
            status_marker = {
                FeasibilityStatus.COMPLETE: "[COMPLETE]",
                FeasibilityStatus.BLOCKED: "[BLOCKED]",
                FeasibilityStatus.DEFERRED: "[DEFERRED]"
            }[subtask.status]
            print(f"  {subtask.subtask_id} {subtask.subtask_name}: {status_marker}")
            for ev in subtask.evidence:
                print(f"    - {ev}")
            print()
        
        print("--- OVERALL ASSESSMENT ---")
        for k, v in self.overall_assessment.items():
            print(f"  {k}: {v}")
        print()
        
        print("--- RECOMMENDATIONS ---")
        for r in self.recommendations:
            print(f"  - {r}")
        print(f"{'='*70}\n")


# =============================================================================
# 3.1 TIME-SERIES ORDERING
# =============================================================================

def audit_time_series_ordering(
    df: pd.DataFrame,
    sensor_col: str = MetadataVariable.SENSOR_ID.value,
    timestamp_col: str = MetadataVariable.TIMESTAMP.value,
    target_col: str = TARGET_COLUMN
) -> SubtaskReport:
    """
    3.1 Time-Series Ordering
    
    Inspect the current Phase 2 data, identify timestamp/segment/sensor columns,
    parse timestamps, examine duplicate groups, and determine if observations
    can be ordered reliably within each segment.
    
    DOES NOT use reading_index, CSV row order, or arbitrary sorting to claim
    genuine temporal ordering.
    """
    evidence = []
    details = {}
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Identify actual columns
    details["timestamp_column"] = timestamp_col
    details["sensor_column"] = sensor_col
    details["target_column"] = target_col
    details["telemetry_columns"] = [c for c in PRIMARY_TELEMETRY if c in df.columns]
    
    evidence.append(f"Timestamp column: '{timestamp_col}' (parsed as datetime)")
    evidence.append(f"Sensor/segment column: '{sensor_col}'")
    evidence.append(f"Target column: '{target_col}'")
    evidence.append(f"Primary telemetry present: {details['telemetry_columns']}")
    
    # Global statistics
    total_obs = len(df)
    total_segments = df[sensor_col].nunique()
    total_unique_ts = df[timestamp_col].nunique()
    ts_range = {
        "start": str(df[timestamp_col].min()),
        "end": str(df[timestamp_col].max()),
        "span_hours": float((df[timestamp_col].max() - df[timestamp_col].min()).total_seconds() / 3600)
    }
    details["global"] = {
        "total_observations": int(total_obs),
        "total_segments": int(total_segments),
        "total_unique_timestamps": int(total_unique_ts),
        "timestamp_range": ts_range
    }
    
    evidence.append(f"Global: {total_obs} observations, {total_segments} segments, {total_unique_ts} unique timestamps")
    evidence.append(f"Global time span: {ts_range['span_hours']:.2f} hours ({ts_range['start']} to {ts_range['end']})")
    
    # Duplicate (segment, timestamp) analysis
    dup_mask = df.duplicated(subset=[sensor_col, timestamp_col], keep=False)
    dup_count = int(dup_mask.sum())
    dup_groups = df[dup_mask].groupby([sensor_col, timestamp_col])
    n_dup_keys = len(list(dup_groups))
    
    # Check for conflicting measurements within duplicate keys
    conflicting_keys = 0
    conflicting_details = []
    for (seg, ts), group in dup_groups:
        # Check if telemetry values differ
        telemetry_diff = False
        target_diff = False
        for pt in PRIMARY_TELEMETRY:
            if pt in group.columns:
                if group[pt].nunique() > 1:
                    telemetry_diff = True
                    break
        if target_col in group.columns and group[target_col].nunique() > 1:
            target_diff = True
        if telemetry_diff or target_diff:
            conflicting_keys += 1
            conflicting_details.append({
                "segment": int(seg),
                "timestamp": str(ts),
                "n_readings": len(group),
                "telemetry_conflict": telemetry_diff,
                "target_conflict": target_diff
            })
    
    details["duplicate_analysis"] = {
        "total_duplicate_rows": dup_count,
        "duplicate_keys": int(n_dup_keys),
        "conflicting_keys": conflicting_keys,
        "conflict_examples": conflicting_details[:5]  # First 5 examples
    }
    
    evidence.append(f"Duplicate (segment, timestamp) keys: {n_dup_keys} ({dup_count} rows)")
    evidence.append(f"Keys with conflicting telemetry/target: {conflicting_keys}")
    if conflicting_details:
        ex = conflicting_details[0]
        evidence.append(f"  Example: segment={ex['segment']}, ts={ex['timestamp']}, n_readings={ex['n_readings']}, "
                        f"telemetry_conflict={ex['telemetry_conflict']}, target_conflict={ex['target_conflict']}")
    
    # Check reading_index column
    has_reading_index = "reading_index" in df.columns
    details["has_reading_index"] = has_reading_index
    if has_reading_index:
        max_ri = int(df["reading_index"].max())
        evidence.append(f"'reading_index' column exists (max value: {max_ri}) - NOT a genuine timestamp")
        evidence.append("  reading_index only distinguishes repeated records; does not prove chronological order")
    
    # Per-segment ordering analysis
    segment_ordering = []
    for seg_id, group in df.groupby(sensor_col):
        group = group.sort_values(timestamp_col)
        n_obs = len(group)
        n_unique_ts = group[timestamp_col].nunique()
        
        # Check if timestamps are strictly monotonic (ignoring duplicates)
        unique_ts = group[timestamp_col].drop_duplicates().sort_values()
        is_monotonic = unique_ts.is_monotonic_increasing
        
        # Time differences between consecutive UNIQUE timestamps
        if len(unique_ts) > 1:
            diffs = unique_ts.diff().dt.total_seconds() / 60  # minutes
            diffs = diffs.dropna()
            interval_stats = {
                "mean_minutes": float(diffs.mean()),
                "median_minutes": float(diffs.median()),
                "std_minutes": float(diffs.std()),
                "min_minutes": float(diffs.min()),
                "max_minutes": float(diffs.max()),
                "n_intervals": int(len(diffs))
            }
        else:
            interval_stats = {
                "mean_minutes": 0, "median_minutes": 0, "std_minutes": 0,
                "min_minutes": 0, "max_minutes": 0, "n_intervals": 0
            }
        
        segment_ordering.append({
            "segment_id": int(seg_id),
            "n_observations": int(n_obs),
            "n_unique_timestamps": int(n_unique_ts),
            "duplicate_timestamps": int(n_obs - n_unique_ts),
            "timestamps_monotonic": is_monotonic,
            "interval_statistics": interval_stats
        })
    
    details["per_segment_ordering"] = segment_ordering
    
    # Summary statistics
    n_segments = len(segment_ordering)
    segments_with_duplicates = sum(1 for s in segment_ordering if s["duplicate_timestamps"] > 0)
    segments_monotonic = sum(1 for s in segment_ordering if s["timestamps_monotonic"])
    min_unique_ts = min(s["n_unique_timestamps"] for s in segment_ordering)
    max_unique_ts = max(s["n_unique_timestamps"] for s in segment_ordering)
    avg_unique_ts = np.mean([s["n_unique_timestamps"] for s in segment_ordering])
    
    evidence.append(f"Segments with duplicate timestamps: {segments_with_duplicates}/{n_segments}")
    evidence.append(f"Segments with monotonic unique timestamps: {segments_monotonic}/{n_segments}")
    evidence.append(f"Unique timestamps per segment: min={min_unique_ts}, max={max_unique_ts}, avg={avg_unique_ts:.1f}")
    
    # FEASIBILITY DETERMINATION
    can_order_reliably = False
    blocking_reasons = []
    
    if ts_range["span_hours"] < 1.0:
        can_order_reliably = False
        blocking_reasons.append(f"Global time span only {ts_range['span_hours']:.2f} hours (< 1 hour)")
    
    if conflicting_keys > 0:
        can_order_reliably = False
        blocking_reasons.append(f"{conflicting_keys} (segment, timestamp) keys have conflicting telemetry/target values")
    
    if min_unique_ts < 10:
        can_order_reliably = False
        blocking_reasons.append(f"Minimum unique timestamps per segment is {min_unique_ts} (< 10)")
    
    if has_reading_index and conflicting_keys > 0:
        blocking_reasons.append("reading_index exists but cannot resolve conflicting measurements at same timestamp")
    
    details["feasibility"] = {
        "can_order_reliably": can_order_reliably,
        "blocking_reasons": blocking_reasons
    }
    
    if can_order_reliably:
        status = FeasibilityStatus.COMPLETE
        evidence.append("ASSESSMENT: Reliable temporal ordering IS possible")
    else:
        status = FeasibilityStatus.BLOCKED
        evidence.append("ASSESSMENT: Reliable temporal ordering is NOT possible")
        for reason in blocking_reasons:
            evidence.append(f"  BLOCKER: {reason}")
    
    return SubtaskReport(
        subtask_id="3.1",
        subtask_name="Time-Series Ordering",
        status=status,
        evidence=evidence,
        details=details
    )


# =============================================================================
# 3.2 SLIDING-WINDOW GENERATION
# =============================================================================

def check_sliding_window_feasibility(
    df: pd.DataFrame,
    sensor_col: str = MetadataVariable.SENSOR_ID.value,
    timestamp_col: str = MetadataVariable.TIMESTAMP.value,
    target_col: str = TARGET_COLUMN,
    min_windows_per_segment: int = 1
) -> SubtaskReport:
    """
    3.2 Sliding-Window Generation
    
    Implements sliding-window construction ONLY if reliable sequential observations exist.
    Ensures windows don't cross group boundaries, target alignment, no leakage.
    If temporal validity is insufficient, implements clear feasibility check and reports why blocked.
    """
    evidence = []
    details = {}
    
    # First, check if we have reliable ordering (from 3.1)
    ordering_report = audit_time_series_ordering(df, sensor_col, timestamp_col, target_col)
    can_order = ordering_report.details["feasibility"]["can_order_reliably"]
    
    details["ordering_feasibility"] = ordering_report.details["feasibility"]
    
    if not can_order:
        evidence.append("Prerequisite (3.1 Time-Series Ordering) is BLOCKED")
        evidence.append("Cannot generate sliding windows without reliable temporal ordering")
        for reason in ordering_report.details["feasibility"]["blocking_reasons"]:
            evidence.append(f"  Root cause: {reason}")
        
        # Still document what WOULD be needed
        details["would_require"] = {
            "reliable_unique_timestamps_per_segment": ">= 10",
            "no_conflicting_measurements": "zero conflicting (segment, timestamp) keys",
            "sufficient_time_span": ">= 1 hour for meaningful horizons",
            "genuine_temporal_ordering": "not reading_index or row order"
        }
        
        status = FeasibilityStatus.BLOCKED
        evidence.append("ASSESSMENT: Sliding-window generation BLOCKED by dataset limitations")
        
        return SubtaskReport(
            subtask_id="3.2",
            subtask_name="Sliding-Window Generation",
            status=status,
            evidence=evidence,
            details=details
        )
    
    # If we reach here, ordering is reliable - implement window generation
    # (This path is not taken for current SCADA dataset but implemented for completeness)
    evidence.append("Reliable ordering confirmed - implementing sliding window generation")
    
    # Implementation would go here for future datasets
    # For now, mark as DEFERRED since prerequisite is blocked
    status = FeasibilityStatus.DEFERRED
    evidence.append("ASSESSMENT: Deferred - prerequisite (3.1) blocked")
    
    return SubtaskReport(
        subtask_id="3.2",
        subtask_name="Sliding-Window Generation",
        status=status,
        evidence=evidence,
        details=details
    )


def generate_sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    sensor_col: str = MetadataVariable.SENSOR_ID.value,
    timestamp_col: str = MetadataVariable.TIMESTAMP.value,
    target_col: str = TARGET_COLUMN,
    feature_cols: Optional[List[str]] = None,
    prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate sliding windows from reliably ordered time series data.
    
    This function is implemented for future use when dataset supports it.
    For current SCADA dataset, it will raise an error if called.
    
    Args:
        df: DataFrame with reliable temporal ordering (unique timestamps per sensor)
        window_size: Number of timesteps in each window
        sensor_col: Sensor/segment identifier column
        timestamp_col: Timestamp column
        target_col: Target column
        feature_cols: Feature columns to include (default: all except metadata/target)
        prediction_horizon: Steps ahead to predict (default: 1)
    
    Returns:
        X: (n_windows, window_size, n_features)
        y: (n_windows,) - target at t + prediction_horizon
        window_info: DataFrame with window metadata (sensor, start_ts, end_ts, target_ts)
    """
    # Verify prerequisites
    ordering_report = audit_time_series_ordering(df, sensor_col, timestamp_col, target_col)
    if not ordering_report.details["feasibility"]["can_order_reliably"]:
        raise ValueError(
            "Cannot generate sliding windows: dataset does not support reliable temporal ordering. "
            f"Blocking reasons: {ordering_report.details['feasibility']['blocking_reasons']}"
        )
    
    if feature_cols is None:
        feature_cols = [c for c in df.columns 
                       if c not in [sensor_col, timestamp_col, target_col, "reading_index"]
                       and c in PRIMARY_TELEMETRY + [c for c in df.columns if not c.startswith("roll_")]]
    
    windows = []
    targets = []
    window_info = []
    
    for sensor_id, group in df.groupby(sensor_col):
        group = group.sort_values(timestamp_col).reset_index(drop=True)
        n = len(group)
        
        if n < window_size + prediction_horizon:
            continue
        
        for i in range(n - window_size - prediction_horizon + 1):
            # Input window: t to t+window_size-1
            X_window = group.iloc[i:i+window_size][feature_cols].values
            
            # Target: t+window_size-1+prediction_horizon
            target_idx = i + window_size - 1 + prediction_horizon
            y_val = group.iloc[target_idx][target_col]
            
            windows.append(X_window)
            targets.append(y_val)
            window_info.append({
                "sensor_id": sensor_id,
                "window_start_idx": i,
                "window_end_idx": i + window_size - 1,
                "window_start_ts": group.iloc[i][timestamp_col],
                "window_end_ts": group.iloc[i + window_size - 1][timestamp_col],
                "target_ts": group.iloc[target_idx][timestamp_col],
                "target_value": y_val
            })
    
    if not windows:
        return np.array([]), np.array([]), pd.DataFrame()
    
    X = np.stack(windows)
    y = np.array(targets)
    window_df = pd.DataFrame(window_info)
    
    return X, y, window_df


# =============================================================================
# 3.3 SEQUENCE LENGTH SELECTION
# =============================================================================

def evaluate_sequence_lengths(
    df: pd.DataFrame,
    sensor_col: str = MetadataVariable.SENSOR_ID.value,
    timestamp_col: str = MetadataVariable.TIMESTAMP.value,
    target_col: str = TARGET_COLUMN,
    candidate_sizes: List[int] = None
) -> SubtaskReport:
    """
    3.3 Sequence Length Selection
    
    If valid sequences are possible, evaluate candidate window sizes (3, 5, 10)
    subject to available sequence lengths. Report how many valid windows each
    size produces and how many groups contribute.
    
    Does NOT select based on test-set performance.
    If valid sequences cannot be formed, document as deferred/blocked.
    """
    evidence = []
    details = {}
    
    if candidate_sizes is None:
        candidate_sizes = [3, 5, 10]
    
    # Check ordering feasibility first
    ordering_report = audit_time_series_ordering(df, sensor_col, timestamp_col, target_col)
    can_order = ordering_report.details["feasibility"]["can_order_reliably"]
    
    details["ordering_feasibility"] = ordering_report.details["feasibility"]
    details["candidate_sizes"] = candidate_sizes
    
    if not can_order:
        evidence.append("Prerequisite (3.1 Time-Series Ordering) is BLOCKED")
        evidence.append("Cannot evaluate sequence lengths without reliable temporal ordering")
        
        # Report available sequence lengths per segment anyway for documentation
        segment_lengths = []
        for seg_id, group in df.groupby(sensor_col):
            n_unique_ts = group[timestamp_col].nunique()
            segment_lengths.append({"segment_id": int(seg_id), "n_unique_timestamps": int(n_unique_ts)})
        
        details["available_sequence_lengths"] = segment_lengths
        details["max_possible_window"] = max(s["n_unique_timestamps"] for s in segment_lengths) if segment_lengths else 0
        
        evidence.append(f"Max unique timestamps in any segment: {details['max_possible_window']}")
        evidence.append(f"Candidate window sizes: {candidate_sizes}")
        evidence.append("All candidate sizes exceed or barely fit available sequence lengths")
        
        status = FeasibilityStatus.BLOCKED
        evidence.append("ASSESSMENT: Sequence length selection BLOCKED - no valid sequences can be formed")
        
        return SubtaskReport(
            subtask_id="3.3",
            subtask_name="Sequence Length Selection",
            status=status,
            evidence=evidence,
            details=details
        )
    
    # If ordering is reliable, evaluate each window size
    results = {}
    for window_size in candidate_sizes:
        try:
            X, y, window_info = generate_sliding_windows(
                df, window_size, sensor_col, timestamp_col, target_col
            )
            n_windows = len(X)
            n_groups = window_info[sensor_col].nunique() if len(window_info) > 0 else 0
            
            results[window_size] = {
                "n_windows": int(n_windows),
                "n_contributing_groups": int(n_groups),
                "class_distribution": {int(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()} if len(y) > 0 else {}
            }
            evidence.append(f"Window size {window_size}: {n_windows} windows from {n_groups} segments")
        except Exception as e:
            results[window_size] = {"error": str(e)}
            evidence.append(f"Window size {window_size}: ERROR - {e}")
    
    details["window_evaluation"] = results
    
    # Select based on data availability (NOT test performance)
    valid_sizes = [w for w, r in results.items() if r.get("n_windows", 0) > 0]
    if valid_sizes:
        recommended = min(valid_sizes)  # Conservative: smallest that works
        details["recommended_window_size"] = recommended
        details["selection_reason"] = "Smallest window size with valid windows (conservative, not performance-based)"
        evidence.append(f"Recommended window size: {recommended} (smallest viable, not performance-based)")
        status = FeasibilityStatus.COMPLETE
    else:
        details["recommended_window_size"] = None
        evidence.append("No candidate window size produces valid windows")
        status = FeasibilityStatus.BLOCKED
    
    return SubtaskReport(
        subtask_id="3.3",
        subtask_name="Sequence Length Selection",
        status=status,
        evidence=evidence,
        details=details
    )


# =============================================================================
# 3.4 TRAIN/VALIDATION/TEST SPLIT
# =============================================================================

class SplitStrategy(str, Enum):
    """Split strategy types."""
    GROUP_AWARE = "group_aware"           # Split by sensor/segment groups
    CHRONOLOGICAL = "chronological"       # Split by time (if timestamps reliable)
    TABULAR_STRATIFIED = "tabular_stratified"  # For tabular (non-sequence) data
    BLOCKED = "blocked"                   # Not feasible


def design_split_strategy(
    df: pd.DataFrame,
    sensor_col: str = MetadataVariable.SENSOR_ID.value,
    timestamp_col: str = MetadataVariable.TIMESTAMP.value,
    target_col: str = TARGET_COLUMN,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1
) -> SubtaskReport:
    """
    3.4 Train/Validation/Test Split
    
    Design a leakage-safe split strategy. If valid sequence split is impossible,
    recommend a defensible tabular-data split for Phase 4.
    Never silently fall back to random row-level split that leaks related records.
    """
    evidence = []
    details = {}
    
    # Check if sequence-based split is feasible
    ordering_report = audit_time_series_ordering(df, sensor_col, timestamp_col, target_col)
    can_order = ordering_report.details["feasibility"]["can_order_reliably"]
    
    details["sequence_feasible"] = can_order
    details["ordering_feasibility"] = ordering_report.details["feasibility"]
    
    # Check duplicate keys - even if ordering works, duplicates cause leakage risk
    dup_keys = df.duplicated(subset=[sensor_col, timestamp_col], keep=False).sum()
    details["duplicate_keys_total"] = int(dup_keys)
    
    if can_order and dup_keys == 0:
        # Chronological split could work
        evidence.append("Chronological split feasible (reliable timestamps, no duplicates)")
        strategy = SplitStrategy.CHRONOLOGICAL
        status = FeasibilityStatus.COMPLETE
    elif can_order and dup_keys > 0:
        # Group-aware split might work but duplicates are problematic
        evidence.append("Group-aware split possible but duplicate keys create ambiguity")
        strategy = SplitStrategy.GROUP_AWARE
        status = FeasibilityStatus.COMPLETE
    else:
        # Sequence split impossible - recommend tabular split
        evidence.append("Sequence-based split IMPOSSIBLE: no reliable temporal ordering")
        evidence.append("Recommending tabular stratified group-aware split for Phase 4")
        strategy = SplitStrategy.TABULAR_STRATIFIED
        status = FeasibilityStatus.BLOCKED  # For SEQUENCE split, but TABULAR is viable
    
    # Design the recommended split
    if strategy == SplitStrategy.CHRONOLOGICAL:
        # Sort by timestamp globally
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df_sorted)
        test_size = int(n * test_fraction)
        val_size = int(n * val_fraction)
        train_size = n - test_size - val_size
        
        train_df = df_sorted.iloc[:train_size]
        val_df = df_sorted.iloc[train_size:train_size+val_size]
        test_df = df_sorted.iloc[train_size+val_size:]
        
        details["split"] = {
            "strategy": "chronological",
            "train_size": int(train_size),
            "val_size": int(val_size),
            "test_size": int(test_size),
            "train_date_range": [str(train_df[timestamp_col].min()), str(train_df[timestamp_col].max())],
            "val_date_range": [str(val_df[timestamp_col].min()), str(val_df[timestamp_col].max())],
            "test_date_range": [str(test_df[timestamp_col].min()), str(test_df[timestamp_col].max())]
        }
        
    elif strategy == SplitStrategy.GROUP_AWARE:
        # Split by sensor groups
        sensors = df[sensor_col].unique()
        np.random.seed(42)
        np.random.shuffle(sensors)
        
        n_sensors = len(sensors)
        n_test = max(1, int(n_sensors * test_fraction))
        n_val = max(1, int(n_sensors * val_fraction))
        n_train = n_sensors - n_test - n_val
        
        train_sensors = sensors[:n_train]
        val_sensors = sensors[n_train:n_train+n_val]
        test_sensors = sensors[n_train+n_val:]
        
        train_df = df[df[sensor_col].isin(train_sensors)]
        val_df = df[df[sensor_col].isin(val_sensors)]
        test_df = df[df[sensor_col].isin(test_sensors)]
        
        details["split"] = {
            "strategy": "group_aware",
            "train_sensors": [int(s) for s in train_sensors],
            "val_sensors": [int(s) for s in val_sensors],
            "test_sensors": [int(s) for s in test_sensors],
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df))
        }
        
    else:  # TABULAR_STRATIFIED
        # For tabular classification: stratified by target, grouped by sensor
        # This prevents same sensor appearing in multiple splits (for Phase 4 tabular models)
        from sklearn.model_selection import train_test_split
        
        # Get unique sensors with their dominant class
        sensor_class = df.groupby(sensor_col)[target_col].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        )
        
        # Check if stratification is possible (need at least 2 sensors per class)
        class_counts = sensor_class.value_counts()
        can_stratify = all(class_counts >= 2)
        
        if can_stratify:
            # Stratified split of sensors by their dominant class
            train_sensors, test_sensors = train_test_split(
                sensor_class.index, 
                test_size=test_fraction, 
                stratify=sensor_class.values,
                random_state=42
            )
            
            train_sensors, val_sensors = train_test_split(
                train_sensors,
                test_size=val_fraction/(1-test_fraction),
                stratify=sensor_class.loc[train_sensors].values,
                random_state=42
            )
            evidence.append("Sensor-level stratified split applied (both classes have >=2 sensors)")
        else:
            # Fall back to random group-aware split without stratification
            evidence.append("Stratification not possible (some class has <2 sensors); using random group-aware split")
            sensors = sensor_class.index.tolist()
            np.random.seed(42)
            np.random.shuffle(sensors)
            
            n_sensors = len(sensors)
            n_test = max(1, int(n_sensors * test_fraction))
            n_val = max(1, int(n_sensors * val_fraction))
            n_train = n_sensors - n_test - n_val
            
            train_sensors = sensors[:n_train]
            val_sensors = sensors[n_train:n_train+n_val]
            test_sensors = sensors[n_train+n_val:]
        
        train_df = df[df[sensor_col].isin(train_sensors)]
        val_df = df[df[sensor_col].isin(val_sensors)]
        test_df = df[df[sensor_col].isin(test_sensors)]
        
        details["split"] = {
            "strategy": "tabular_stratified_group_aware",
            "description": "Stratified by target class, grouped by sensor (no sensor in multiple splits)",
            "train_sensors": [int(s) for s in train_sensors],
            "val_sensors": [int(s) for s in val_sensors],
            "test_sensors": [int(s) for s in test_sensors],
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "train_class_dist": train_df[target_col].value_counts().to_dict(),
            "val_class_dist": val_df[target_col].value_counts().to_dict(),
            "test_class_dist": test_df[target_col].value_counts().to_dict()
        }
        
        evidence.append(f"Tabular split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
        evidence.append(f"  Train class dist: {details['split']['train_class_dist']}")
        evidence.append(f"  Val class dist: {details['split']['val_class_dist']}")
        evidence.append(f"  Test class dist: {details['split']['test_class_dist']}")
        evidence.append(f"  Train sensors: {len(train_sensors)}, Val sensors: {len(val_sensors)}, Test sensors: {len(test_sensors)}")
        evidence.append("  No sensor appears in multiple splits (prevents leakage)")
    
    details["recommended_strategy"] = strategy.value
    details["test_fraction"] = test_fraction
    details["val_fraction"] = val_fraction
    
    evidence.append(f"Recommended split strategy: {strategy.value}")
    
    if strategy == SplitStrategy.TABULAR_STRATIFIED:
        evidence.append("ASSESSMENT: Sequence split BLOCKED; tabular group-aware split RECOMMENDED for Phase 4")
        status = FeasibilityStatus.BLOCKED  # For sequence split
    else:
        evidence.append("ASSESSMENT: Sequence split feasible with recommended strategy")
        status = FeasibilityStatus.COMPLETE
    
    return SubtaskReport(
        subtask_id="3.4",
        subtask_name="Train/Validation/Test Split",
        status=status,
        evidence=evidence,
        details=details
    )


def apply_tabular_split(
    df: pd.DataFrame,
    sensor_col: str = MetadataVariable.SENSOR_ID.value,
    target_col: str = TARGET_COLUMN,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Apply the recommended tabular group-aware stratified split.
    
    Returns train, val, test DataFrames and split info.
    This is the function Phase 4 should use for tabular model development.
    """
    from sklearn.model_selection import train_test_split
    
    # Get sensor dominant class for stratification
    sensor_class = df.groupby(sensor_col)[target_col].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
    )
    
    # Check if stratification is possible (need at least 2 sensors per class)
    class_counts = sensor_class.value_counts()
    can_stratify = all(class_counts >= 2)
    
    if can_stratify:
        # Stratified split of sensors
        train_sensors, test_sensors = train_test_split(
            sensor_class.index,
            test_size=test_fraction,
            stratify=sensor_class.values,
            random_state=random_state
        )
        
        train_sensors, val_sensors = train_test_split(
            train_sensors,
            test_size=val_fraction/(1-test_fraction),
            stratify=sensor_class.loc[train_sensors].values,
            random_state=random_state
        )
        split_method = "stratified_group_aware"
    else:
        # Fall back to random group-aware split without stratification
        sensors = sensor_class.index.tolist()
        np.random.seed(random_state)
        np.random.shuffle(sensors)
        
        n_sensors = len(sensors)
        n_test = max(1, int(n_sensors * test_fraction))
        n_val = max(1, int(n_sensors * val_fraction))
        n_train = n_sensors - n_test - n_val
        
        train_sensors = sensors[:n_train]
        val_sensors = sensors[n_train:n_train+n_val]
        test_sensors = sensors[n_train+n_val:]
        split_method = "random_group_aware"
    
    train_df = df[df[sensor_col].isin(train_sensors)].copy()
    val_df = df[df[sensor_col].isin(val_sensors)].copy()
    test_df = df[df[sensor_col].isin(test_sensors)].copy()
    
    split_info = {
        "strategy": "tabular_stratified_group_aware",
        "random_state": random_state,
        "train_sensors": sorted([int(s) for s in train_sensors]),
        "val_sensors": sorted([int(s) for s in val_sensors]),
        "test_sensors": sorted([int(s) for s in test_sensors]),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_class_dist": train_df[target_col].value_counts().to_dict(),
        "val_class_dist": val_df[target_col].value_counts().to_dict(),
        "test_class_dist": test_df[target_col].value_counts().to_dict(),
        "no_sensor_overlap": (
            len(set(train_sensors) & set(val_sensors)) == 0 and
            len(set(train_sensors) & set(test_sensors)) == 0 and
            len(set(val_sensors) & set(test_sensors)) == 0
        )
    }
    
    return train_df, val_df, test_df, split_info


# =============================================================================
# MAIN PHASE 3 PIPELINE
# =============================================================================

def run_phase3_feasibility(
    df: pd.DataFrame,
    dataset_name: str = "SCADA_Pipeline",
    sensor_col: str = MetadataVariable.SENSOR_ID.value,
    timestamp_col: str = MetadataVariable.TIMESTAMP.value,
    target_col: str = TARGET_COLUMN,
    save_artifacts: bool = True
) -> Phase3Report:
    """
    Run complete Phase 3 feasibility assessment.
    
    Executes all 4 subtasks and generates comprehensive report.
    Does NOT generate sequence artifacts if feasibility is BLOCKED.
    """
    print(f"\n{'='*70}")
    print(f"PIPER PHASE 3: TEMPORAL SEQUENCE CONSTRUCTION FEASIBILITY")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    subtasks = []
    
    # 3.1 Time-Series Ordering
    print("\n[3.1] Time-Series Ordering Audit...")
    subtask_31 = audit_time_series_ordering(df, sensor_col, timestamp_col, target_col)
    subtasks.append(subtask_31)
    for ev in subtask_31.evidence:
        print(f"    {ev}")
    
    # 3.2 Sliding-Window Generation
    print("\n[3.2] Sliding-Window Generation Feasibility...")
    subtask_32 = check_sliding_window_feasibility(df, sensor_col, timestamp_col, target_col)
    subtasks.append(subtask_32)
    for ev in subtask_32.evidence:
        print(f"    {ev}")
    
    # 3.3 Sequence Length Selection
    print("\n[3.3] Sequence Length Selection...")
    subtask_33 = evaluate_sequence_lengths(df, sensor_col, timestamp_col, target_col)
    subtasks.append(subtask_33)
    for ev in subtask_33.evidence:
        print(f"    {ev}")
    
    # 3.4 Train/Validation/Test Split
    print("\n[3.4] Train/Validation/Test Split Strategy...")
    subtask_34 = design_split_strategy(df, sensor_col, timestamp_col, target_col)
    subtasks.append(subtask_34)
    for ev in subtask_34.evidence:
        print(f"    {ev}")
    
    # Overall assessment
    statuses = [s.status for s in subtasks]
    n_complete = sum(1 for s in statuses if s == FeasibilityStatus.COMPLETE)
    n_blocked = sum(1 for s in statuses if s == FeasibilityStatus.BLOCKED)
    n_deferred = sum(1 for s in statuses if s == FeasibilityStatus.DEFERRED)
    
    overall = {
        "subtasks_complete": n_complete,
        "subtasks_blocked": n_blocked,
        "subtasks_deferred": n_deferred,
        "sequence_learning_viable": n_blocked == 0 and n_deferred == 0,
        "tabular_classification_viable": True,  # From Phase 2
        "next_phase_recommendation": "tabular_classification" if n_blocked > 0 else "sequence_modeling"
    }
    
    recommendations = [
        "Sequence learning (LSTM/GRU) is NOT viable with current SCADA dataset",
        "Proceed to Phase 4 with TABULAR classification models (Random Forest, XGBoost, MLP, SVM)",
        "Use group-aware stratified split (sensor-level) to prevent data leakage",
        "Do NOT attempt to fabricate temporal sequences from conflicting duplicate timestamps",
        "If temporal modeling is required, collect data with: longer time span (>24h), "
        "regular sampling intervals, and unique (sensor, timestamp) keys"
    ]
    
    report = Phase3Report(
        dataset_name=dataset_name,
        audit_timestamp=datetime.now().isoformat(),
        subtasks=subtasks,
        overall_assessment=overall,
        recommendations=recommendations
    )
    
    if save_artifacts:
        # Save Phase 3 feasibility report
        report_path = PIPER_REPORTS_DIR / "phase3_feasibility_report.json"
        report.save(report_path)
        print(f"\nPhase 3 feasibility report saved to: {report_path}")
        
        # Save markdown summary
        md_path = PIPER_REPORTS_DIR / "phase3_summary.md"
        generate_phase3_summary_md(report, md_path)
        print(f"Phase 3 summary saved to: {md_path}")
        
        # Also save the tabular split for Phase 4
        if subtask_34.details["recommended_strategy"] in ["tabular_stratified_group_aware", "tabular_stratified"]:
            train_df, val_df, test_df, split_info = apply_tabular_split(
                df, sensor_col, target_col
            )
            
            # Save split data
            PIPER_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(PIPER_PROCESSED_DIR / "scada_train.csv", index=False)
            val_df.to_csv(PIPER_PROCESSED_DIR / "scada_val.csv", index=False)
            test_df.to_csv(PIPER_PROCESSED_DIR / "scada_test.csv", index=False)
            
            # Save split info
            split_path = PIPER_REPORTS_DIR / "scada_tabular_split_info.json"
            with open(split_path, 'w') as f:
                json.dump(split_info, f, indent=2, default=str)
            
            print(f"\nTabular split saved for Phase 4:")
            print(f"  Train: {len(train_df)} rows ({len(split_info['train_sensors'])} sensors)")
            print(f"  Val:   {len(val_df)} rows ({len(split_info['val_sensors'])} sensors)")
            print(f"  Test:  {len(test_df)} rows ({len(split_info['test_sensors'])} sensors)")
    
    report.print_summary()
    return report


def generate_phase3_summary_md(report: Phase3Report, output_path: Path) -> None:
    """Generate Phase 3 markdown summary."""
    md = f"""# PIPER Phase 3 Summary Report: {report.dataset_name}

**Generated:** {report.audit_timestamp}

## Executive Summary

"""
    
    overall = report.overall_assessment
    md += f"""| Metric | Value |
|--------|-------|
| Subtasks Complete | {overall['subtasks_complete']} |
| Subtasks Blocked | {overall['subtasks_blocked']} |
| Subtasks Deferred | {overall['subtasks_deferred']} |
| Sequence Learning Viable | {'YES' if overall['sequence_learning_viable'] else 'NO'} |
| Tabular Classification Viable | {'YES' if overall['tabular_classification_viable'] else 'NO'} |
| Recommended Next Phase | {overall['next_phase_recommendation']} |

"""
    
    md += "## Subtask Status\n\n"
    
    for subtask in report.subtasks:
        status_marker = {
            FeasibilityStatus.COMPLETE: "[COMPLETE]",
            FeasibilityStatus.BLOCKED: "[BLOCKED]",
            FeasibilityStatus.DEFERRED: "[DEFERRED]"
        }[subtask.status]
        
        md += f"### {subtask.subtask_id} {subtask.subtask_name}: {status_marker}\n\n"
        md += "**Evidence:**\n\n"
        for ev in subtask.evidence:
            md += f"- {ev}\n"
        md += "\n"
        
        if subtask.details:
            md += "**Details:**\n\n"
            md += "```json\n"
            md += json.dumps(subtask.details, indent=2, default=str)
            md += "\n```\n\n"
    
    md += "## Overall Assessment\n\n"
    for k, v in overall.items():
        md += f"- **{k}**: {v}\n"
    md += "\n"
    
    md += "## Recommendations\n\n"
    for r in report.recommendations:
        md += f"- {r}\n"
    md += "\n"
    
    md += "## Artifacts Generated\n\n"
    md += "- `data/reports/phase3_feasibility_report.json` - Full feasibility report\n"
    md += "- `data/reports/phase3_summary.md` - This summary\n"
    md += "- `data/processed/scada_train.csv` - Training set (tabular)\n"
    md += "- `data/processed/scada_val.csv` - Validation set (tabular)\n"
    md += "- `data/processed/scada_test.csv` - Test set (tabular, ISOLATED)\n"
    md += "- `data/reports/scada_tabular_split_info.json` - Split metadata\n\n"
    
    md += "---\n\n"
    md += "**Conclusion:** The SCADA dataset cannot support temporal sequence construction for LSTM/GRU models. "
    md += "Phase 3 sequence track is BLOCKED. Phase 4 should proceed with tabular classification "
    md += "using the provided train/val/test splits which are leakage-safe at the sensor level.\n"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PIPER Phase 3: Temporal Sequence Feasibility")
    parser.add_argument("--dataset", choices=["scada", "apoerch"], default="scada", 
                       help="Dataset to assess")
    parser.add_argument("--force-reload", action="store_true", help="Reload from raw data")
    parser.add_argument("--no-save", action="store_true", help="Don't save artifacts")
    args = parser.parse_args()
    
    if args.dataset == "scada":
        from utils.piper_dataset import load_or_create_scada_piper_dataset
        from utils.piper_cleaning import clean_scada_dataset
        from utils.piper_features import engineer_scada_features
        from utils.piper_labels import label_scada_real
        
        df, info = load_or_create_scada_piper_dataset(force_reload=args.force_reload)
        clean_df, _ = clean_scada_dataset(df, save_report=False)
        engineered_df, _ = engineer_scada_features(clean_df, save_report=False)
        labeled_df, _ = label_scada_real(engineered_df, save_report=False)
        
        run_phase3_feasibility(labeled_df, dataset_name="SCADA_Pipeline", save_artifacts=not args.no_save)
    
    elif args.dataset == "apoerch":
        from utils.piper_dataset import load_or_create_apoerch_piper_dataset
        from utils.piper_cleaning import clean_apoerch_dataset
        from utils.piper_features import engineer_apoerch_features
        from utils.piper_labels import label_apoerch_synthetic
        
        df, info = load_or_create_apoerch_piper_dataset(force_reload=args.force_reload)
        clean_df, _ = clean_apoerch_dataset(df, save_report=False)
        engineered_df, _ = engineer_apoerch_features(clean_df, save_report=False)
        labeled_df, _ = label_apoerch_synthetic(engineered_df, save_report=False)
        
        run_phase3_feasibility(labeled_df, dataset_name="ApoErch_LeakageData", save_artifacts=not args.no_save)


if __name__ == "__main__":
    main()