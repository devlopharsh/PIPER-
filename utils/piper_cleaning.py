"""
PIPER Data Cleaning Pipeline

Implements reproducible cleaning operations with detailed reporting.
Each operation tracks rows affected and preserves raw data.

Cleaning operations:
1. Timestamp parsing and validation
2. Chronological ordering
3. Duplicate record detection/removal
4. Missing value detection
5. Invalid/non-numeric telemetry detection
6. Impossible sensor value detection
7. Outlier inspection (report only, no auto-delete)
8. Sensor/pipeline consistency checks
9. Sampling interval consistency
10. Data integrity validation

IMPORTANT: Abnormal observations (potential leaks) are NOT automatically removed.
Cleaning distinguishes between invalid measurements and legitimate abnormal conditions.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import json
from datetime import datetime

from utils.piper_telemetry import (
    PRIMARY_TELEMETRY, METADATA_COLUMNS, TARGET_COLUMN,
    TelemetryVariable, MetadataVariable, TargetVariable,
    is_primary_telemetry
)
from utils.constants import METER_ID_COL as LEGACY_METER_ID, TIMESTAMP_COL as LEGACY_TIMESTAMP


# =============================================================================
# CLEANING REPORT STRUCTURES
# =============================================================================

@dataclass
class CleaningStepReport:
    """Report for a single cleaning operation."""
    step_name: str
    description: str
    rows_before: int
    rows_after: int
    rows_affected: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


@dataclass
class CleaningReport:
    """Complete cleaning pipeline report."""
    dataset_name: str
    total_rows_raw: int
    total_rows_clean: int = 0
    steps: List[CleaningStepReport] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    
    def add_step(self, step: CleaningStepReport) -> None:
        self.steps.append(step)
    
    def finalize(self) -> None:
        self.end_time = datetime.now().isoformat()
        self.total_rows_clean = self.steps[-1].rows_after if self.steps else self.total_rows_raw
    
    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "total_rows_raw": self.total_rows_raw,
            "total_rows_clean": self.total_rows_clean,
            "total_rows_removed": self.total_rows_raw - self.total_rows_clean,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "steps": [asdict(s) for s in self.steps]
        }
    
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def print_summary(self) -> None:
        print(f"\n{'='*70}")
        print(f"CLEANING REPORT: {self.dataset_name}")
        print(f"{'='*70}")
        print(f"Raw rows:     {self.total_rows_raw:,}")
        print(f"Clean rows:   {self.total_rows_clean:,}")
        print(f"Removed:      {self.total_rows_raw - self.total_rows_clean:,}")
        print(f"Duration:     {self.start_time} to {self.end_time}")
        print()
        for step in self.steps:
            print(f"  [{step.step_name}] {step.description}")
            print(f"    Before: {step.rows_before:,}  After: {step.rows_after:,}  Affected: {step.rows_affected:,}")
            if step.details:
                for k, v in step.details.items():
                    print(f"    {k}: {v}")
        print(f"{'='*70}\n")


# =============================================================================
# CLEANING OPERATIONS
# =============================================================================

class PiperCleaner:
    """PIPER data cleaning pipeline with reporting."""
    
    def __init__(self, dataset_name: str = "unknown"):
        self.dataset_name = dataset_name
        self.report = CleaningReport(dataset_name=dataset_name, total_rows_raw=0)
        self._df: Optional[pd.DataFrame] = None
    
    def load(self, df: pd.DataFrame) -> "PiperCleaner":
        """Load dataframe for cleaning."""
        self._df = df.copy()
        self.report.total_rows_raw = len(df)
        return self
    
    def get_data(self) -> pd.DataFrame:
        """Get current dataframe."""
        if self._df is None:
            raise ValueError("No data loaded. Call load() first.")
        return self._df
    
    def _record_step(
        self,
        step_name: str,
        description: str,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        details: Dict[str, Any] = None
    ) -> CleaningStepReport:
        """Record a cleaning step."""
        step = CleaningStepReport(
            step_name=step_name,
            description=description,
            rows_before=len(df_before),
            rows_after=len(df_after),
            rows_affected=len(df_before) - len(df_after),
            details=details or {}
        )
        self.report.add_step(step)
        return step
    
    # -------------------------------------------------------------------------
    # 1. Timestamp parsing and validation
    # -------------------------------------------------------------------------
    def validate_timestamps(
        self,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        format_hint: str = "%d/%m/%Y %H:%M",
        dayfirst: bool = True
    ) -> "PiperCleaner":
        """Validate and parse timestamps."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if timestamp_col not in self._df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")
        
        # Parse timestamps
        original_dtype = self._df[timestamp_col].dtype
        self._df[timestamp_col] = pd.to_datetime(
            self._df[timestamp_col].astype(str).str.strip(),
            format=format_hint,
            dayfirst=dayfirst,
            errors="coerce"
        )
        
        # Detect parsing failures
        nat_count = self._df[timestamp_col].isna().sum()
        total_count = len(self._df)
        
        # Remove rows with unparseable timestamps (these are invalid measurements)
        self._df = self._df.dropna(subset=[timestamp_col]).copy()
        
        details = {
            "original_dtype": str(original_dtype),
            "format_used": format_hint,
            "parsing_failures": int(nat_count),
            "rows_with_invalid_timestamp_removed": int(nat_count),
            "timestamp_range": {
                "min": str(self._df[timestamp_col].min()) if len(self._df) > 0 else None,
                "max": str(self._df[timestamp_col].max()) if len(self._df) > 0 else None
            }
        }
        
        self._record_step(
            "validate_timestamps",
            f"Parse and validate timestamps ({format_hint})",
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # 2. Chronological ordering
    # -------------------------------------------------------------------------
    def enforce_chronological_order(
        self,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value
    ) -> "PiperCleaner":
        """Sort by sensor then timestamp."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        required_cols = [sensor_col, timestamp_col]
        missing = [c for c in required_cols if c not in self._df.columns]
        if missing:
            raise ValueError(f"Missing columns for sorting: {missing}")
        
        self._df = self._df.sort_values(required_cols).reset_index(drop=True)
        
        details = {
            "sort_columns": required_cols,
            "was_already_sorted": df_before.equals(self._df)
        }
        
        self._record_step(
            "enforce_chronological_order",
            f"Sort by {sensor_col}, then {timestamp_col}",
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # 3. Duplicate record detection/removal
    # -------------------------------------------------------------------------
    def handle_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: str = "first",
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value
    ) -> "PiperCleaner":
        """
        Detect and handle duplicates:
        - Exact duplicate rows (all columns identical): remove, keep first
        - Non-identical rows with same (sensor_id, timestamp) key: report but preserve all
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if subset is None:
            subset = [sensor_col, timestamp_col]
        
        missing = [c for c in subset if c not in self._df.columns]
        if missing:
            raise ValueError(f"Missing columns for duplicate detection: {missing}")
        
        # 1. Exact duplicates (all columns)
        exact_dup_mask = self._df.duplicated(keep=keep)
        exact_dup_count = int(exact_dup_mask.sum())
        
        # 2. Key duplicates (same sensor_id, timestamp) - check if non-identical
        key_dup_mask = self._df.duplicated(subset=subset, keep=False)
        key_dup_groups = self._df[key_dup_mask].groupby(subset)
        
        conflicting_keys = 0
        identical_keys = 0
        conflicting_rows = 0
        
        for _, group in key_dup_groups:
            # Check if all rows in group are identical (excluding index)
            unique_rows = group.drop_duplicates()
            if len(unique_rows) == 1:
                identical_keys += 1
            else:
                conflicting_keys += 1
                conflicting_rows += len(group)
        
        # conflicting_rows_preserved = extra rows beyond first per key
        # = total rows in conflicting groups - conflicting_keys (1 kept per key)
        conflicting_rows_preserved = conflicting_rows - conflicting_keys if conflicting_keys > 0 else 0
        
        # Remove exact duplicates only
        if exact_dup_count > 0:
            self._df = self._df.drop_duplicates(keep=keep).reset_index(drop=True)
        
        # Add reading_index to distinguish multiple readings per (sensor_id, timestamp)
        if conflicting_keys > 0:
            self._df["reading_index"] = self._df.groupby(subset).cumcount()
        
        details = {
            "subset_columns": subset,
            "keep_policy": keep,
            "exact_duplicate_rows_found": exact_dup_count,
            "exact_duplicate_rows_removed": exact_dup_count,
            "key_duplicate_groups_found": int(key_dup_mask.sum() / 2) if key_dup_mask.any() else 0,  # approximate
            "conflicting_key_groups": conflicting_keys,
            "identical_key_groups": identical_keys,
            "conflicting_rows_preserved": conflicting_rows_preserved,
            "reading_index_added": conflicting_keys > 0,
            "action": "Exact duplicates removed; conflicting key duplicates preserved with reading_index"
        }
        
        self._record_step(
            "handle_duplicates",
            f"Handle duplicates on {subset}: exact removed, conflicts preserved",
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # 4. Missing value detection
    # -------------------------------------------------------------------------
    def detect_missing_values(
        self,
        columns: Optional[List[str]] = None
    ) -> "PiperCleaner":
        """Detect and report missing values. Does NOT remove rows."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if columns is None:
            columns = [c for c in self._df.columns if is_primary_telemetry(c)] + METADATA_COLUMNS
            columns = [c for c in columns if c in self._df.columns]
        
        missing_counts = self._df[columns].isna().sum().to_dict()
        total_missing = sum(missing_counts.values())
        
        # Report rows with ANY missing in checked columns
        rows_with_missing = int(self._df[columns].isna().any(axis=1).sum())
        
        details = {
            "columns_checked": columns,
            "missing_per_column": missing_counts,
            "total_missing_values": int(total_missing),
            "rows_with_any_missing": rows_with_missing,
            "action": "REPORT_ONLY - no rows removed"
        }
        
        self._record_step(
            "detect_missing_values",
            f"Detect missing values in {len(columns)} columns",
            df_before, self._df, details
        )
        return self
    
    def handle_missing_values(
        self,
        strategy: str = "drop",
        columns: Optional[List[str]] = None,
        fill_value: Any = 0.0
    ) -> "PiperCleaner":
        """Handle missing values (drop or fill)."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if columns is None:
            columns = [c for c in self._df.columns if is_primary_telemetry(c)] + METADATA_COLUMNS
            columns = [c for c in columns if c in self._df.columns]
        
        rows_before = len(self._df)
        
        if strategy == "drop":
            self._df = self._df.dropna(subset=columns).reset_index(drop=True)
            action = f"dropped rows with missing values in {columns}"
        elif strategy == "fill":
            self._df[columns] = self._df[columns].fillna(fill_value)
            action = f"filled missing values in {columns} with {fill_value}"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        rows_removed = rows_before - len(self._df)
        
        details = {
            "strategy": strategy,
            "columns": columns,
            "rows_removed": rows_removed,
            "fill_value": fill_value if strategy == "fill" else None
        }
        
        self._record_step(
            "handle_missing_values",
            action,
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # 5. Invalid/non-numeric telemetry detection
    # -------------------------------------------------------------------------
    def validate_telemetry_numeric(
        self,
        columns: Optional[List[str]] = None
    ) -> "PiperCleaner":
        """Check that telemetry columns are numeric."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if columns is None:
            columns = [c for c in self._df.columns if is_primary_telemetry(c)]
            columns = [c for c in columns if c in self._df.columns]
        
        non_numeric_counts = {}
        for col in columns:
            if col in self._df.columns:
                # Try to convert to numeric, count failures
                numeric_series = pd.to_numeric(self._df[col], errors="coerce")
                non_numeric = int(self._df[col].notna().sum() - numeric_series.notna().sum())
                non_numeric_counts[col] = non_numeric
        
        total_non_numeric = sum(non_numeric_counts.values())
        
        details = {
            "columns_checked": columns,
            "non_numeric_per_column": non_numeric_counts,
            "total_non_numeric": total_non_numeric,
            "action": "REPORT_ONLY - no conversion performed"
        }
        
        self._record_step(
            "validate_telemetry_numeric",
            f"Validate numeric type for telemetry columns",
            df_before, self._df, details
        )
        return self
    
    def coerce_telemetry_numeric(
        self,
        columns: Optional[List[str]] = None,
        errors: str = "coerce"
    ) -> "PiperCleaner":
        """Coerce telemetry columns to numeric, handling errors."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if columns is None:
            columns = [c for c in self._df.columns if is_primary_telemetry(c)]
            columns = [c for c in columns if c in self._df.columns]
        
        coerced_counts = {}
        for col in columns:
            if col in self._df.columns:
                original_notna = int(self._df[col].notna().sum())
                self._df[col] = pd.to_numeric(self._df[col], errors=errors)
                new_notna = int(self._df[col].notna().sum())
                coerced_counts[col] = original_notna - new_notna
        
        total_coerced = sum(coerced_counts.values())
        
        # Drop rows where primary telemetry became NaN
        if errors == "coerce":
            telemetry_cols = [c for c in columns if c in self._df.columns and is_primary_telemetry(c)]
            if telemetry_cols:
                self._df = self._df.dropna(subset=telemetry_cols).reset_index(drop=True)
        
        rows_removed = len(df_before) - len(self._df)
        
        details = {
            "columns": columns,
            "errors_policy": errors,
            "values_coerced_to_nan_per_column": coerced_counts,
            "total_coerced": total_coerced,
            "rows_removed_due_to_nan_telemetry": rows_removed
        }
        
        self._record_step(
            "coerce_telemetry_numeric",
            f"Coerce telemetry columns to numeric ({errors})",
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # 6. Impossible sensor value detection
    # -------------------------------------------------------------------------
    def detect_impossible_values(
        self,
        rules: Optional[Dict[str, Dict]] = None
    ) -> "PiperCleaner":
        """
        Detect impossible sensor values based on physical constraints.
        
        Rules format:
        {
            "pressure": {"min": 0, "max": 1000},  # psi
            "flow_rate": {"min": 0, "max": 10000}, # gpm
            "temperature": {"min": -50, "max": 150}, # celsius
            "vibration": {"min": 0, "max": 100}  # mm/s
        }
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if rules is None:
            # Default physical constraints for water pipeline telemetry
            rules = {
                TelemetryVariable.PRESSURE.value: {"min": 0, "max": 2000},      # psi
                TelemetryVariable.FLOW_RATE.value: {"min": 0, "max": 50000},    # gpm (adjust per pipe size)
                TelemetryVariable.TEMPERATURE.value: {"min": -40, "max": 200},   # celsius
                TelemetryVariable.VIBRATION.value: {"min": 0, "max": 500},       # mm/s RMS
            }
        
        impossible_counts = {}
        impossible_masks = {}
        
        for col, bounds in rules.items():
            if col in self._df.columns:
                mask = pd.Series(False, index=self._df.index)
                if "min" in bounds:
                    mask |= (self._df[col] < bounds["min"])
                if "max" in bounds:
                    mask |= (self._df[col] > bounds["max"])
                impossible_counts[col] = int(mask.sum())
                impossible_masks[col] = mask
        
        total_impossible = sum(impossible_counts.values())
        any_impossible_mask = pd.Series(False, index=self._df.index)
        for mask in impossible_masks.values():
            any_impossible_mask |= mask
        
        details = {
            "rules_applied": rules,
            "impossible_per_column": impossible_counts,
            "total_impossible_values": total_impossible,
            "rows_with_any_impossible": int(any_impossible_mask.sum()),
            "action": "REPORT_ONLY - abnormal values preserved for leak detection"
        }
        
        self._record_step(
            "detect_impossible_values",
            f"Check physical bounds on telemetry",
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # 7. Outlier inspection (report only)
    # -------------------------------------------------------------------------
    def inspect_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        iqr_multiplier: float = 1.5
    ) -> "PiperCleaner":
        """Inspect outliers using IQR or z-score method. REPORT ONLY - no removal."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if columns is None:
            columns = [c for c in self._df.columns if is_primary_telemetry(c)]
            columns = [c for c in columns if c in self._df.columns]
        
        outlier_counts = {}
        outlier_details = {}
        
        for col in columns:
            if col not in self._df.columns:
                continue
            
            series = self._df[col].dropna()
            if len(series) == 0:
                outlier_counts[col] = 0
                continue
            
            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr
                mask = (self._df[col] < lower) | (self._df[col] > upper)
                outlier_counts[col] = int(mask.sum())
                outlier_details[col] = {
                    "method": "iqr",
                    "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                    "lower_bound": float(lower), "upper_bound": float(upper),
                    "multiplier": iqr_multiplier
                }
            elif method == "zscore":
                mean = series.mean()
                std = series.std()
                if std > 0:
                    z = (self._df[col] - mean).abs() / std
                    mask = z > 3.0
                    outlier_counts[col] = int(mask.sum())
                    outlier_details[col] = {
                        "method": "zscore",
                        "mean": float(mean), "std": float(std),
                        "threshold": 3.0
                    }
                else:
                    outlier_counts[col] = 0
                    outlier_details[col] = {"method": "zscore", "std": 0}
        
        total_outliers = sum(outlier_counts.values())
        
        details = {
            "method": method,
            "columns_checked": columns,
            "outliers_per_column": outlier_counts,
            "total_outliers": total_outliers,
            "outlier_details": outlier_details,
            "action": "REPORT_ONLY - outliers preserved (may indicate leaks)"
        }
        
        self._record_step(
            "inspect_outliers",
            f"Outlier inspection using {method}",
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # 8. Sensor/pipeline consistency
    # -------------------------------------------------------------------------
    def check_sensor_consistency(
        self,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value
    ) -> "PiperCleaner":
        """Check for sensor data consistency (gaps, single-reading sensors, etc.)."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if sensor_col not in self._df.columns or timestamp_col not in self._df.columns:
            raise ValueError(f"Missing required columns: {sensor_col}, {timestamp_col}")
        
        sensor_stats = []
        for sensor_id, group in self._df.groupby(sensor_col):
            group = group.sort_values(timestamp_col)
            n_readings = len(group)
            date_range = (group[timestamp_col].min(), group[timestamp_col].max())
            time_span_hours = (date_range[1] - date_range[0]).total_seconds() / 3600
            
            # Expected readings if hourly
            expected_readings = int(time_span_hours) + 1 if time_span_hours > 0 else 1
            completeness = n_readings / expected_readings if expected_readings > 0 else 0
            
            # Gap analysis
            diffs = group[timestamp_col].diff().dt.total_seconds() / 3600
            gaps = diffs[diffs > 1.5]  # gaps > 1.5 hours
            n_gaps = len(gaps)
            max_gap_hours = float(gaps.max()) if n_gaps > 0 else 0
            
            sensor_stats.append({
                "sensor_id": sensor_id,
                "n_readings": n_readings,
                "date_range_start": str(date_range[0]),
                "date_range_end": str(date_range[1]),
                "time_span_hours": time_span_hours,
                "expected_readings_hourly": expected_readings,
                "completeness": completeness,
                "n_gaps_gt_1h": n_gaps,
                "max_gap_hours": max_gap_hours
            })
        
        sensors_with_gaps = sum(1 for s in sensor_stats if s["n_gaps_gt_1h"] > 0)
        sensors_low_completeness = sum(1 for s in sensor_stats if s["completeness"] < 0.5)
        single_reading_sensors = sum(1 for s in sensor_stats if s["n_readings"] == 1)
        
        details = {
            "total_sensors": len(sensor_stats),
            "sensor_statistics": sensor_stats,
            "sensors_with_gaps": sensors_with_gaps,
            "sensors_low_completeness": sensors_low_completeness,
            "single_reading_sensors": single_reading_sensors,
            "action": "REPORT_ONLY"
        }
        
        self._record_step(
            "check_sensor_consistency",
            f"Per-sensor consistency check",
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # 9. Sampling interval consistency
    # -------------------------------------------------------------------------
    def check_sampling_interval(
        self,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        expected_interval_hours: float = 1.0,
        tolerance_hours: float = 0.1
    ) -> "PiperCleaner":
        """Check sampling interval consistency per sensor."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if sensor_col not in self._df.columns or timestamp_col not in self._df.columns:
            raise ValueError(f"Missing required columns")
        
        interval_stats = []
        
        for sensor_id, group in self._df.groupby(sensor_col):
            group = group.sort_values(timestamp_col)
            diffs = group[timestamp_col].diff().dt.total_seconds() / 3600
            diffs = diffs.dropna()
            
            if len(diffs) == 0:
                continue
            
            # Categorize intervals
            on_time = ((diffs >= expected_interval_hours - tolerance_hours) & 
                      (diffs <= expected_interval_hours + tolerance_hours)).sum()
            early = (diffs < expected_interval_hours - tolerance_hours).sum()
            late = (diffs > expected_interval_hours + tolerance_hours).sum()
            
            interval_stats.append({
                "sensor_id": sensor_id,
                "n_intervals": len(diffs),
                "mean_interval_hours": float(diffs.mean()),
                "median_interval_hours": float(diffs.median()),
                "std_interval_hours": float(diffs.std()),
                "on_time_count": int(on_time),
                "early_count": int(early),
                "late_count": int(late),
                "on_time_pct": float(on_time / len(diffs) * 100) if len(diffs) > 0 else 0
            })
        
        total_intervals = sum(s["n_intervals"] for s in interval_stats)
        overall_on_time = sum(s["on_time_count"] for s in interval_stats)
        
        details = {
            "expected_interval_hours": expected_interval_hours,
            "tolerance_hours": tolerance_hours,
            "total_intervals_analyzed": total_intervals,
            "overall_on_time_pct": float(overall_on_time / total_intervals * 100) if total_intervals > 0 else 0,
            "per_sensor_statistics": interval_stats,
            "action": "REPORT_ONLY"
        }
        
        self._record_step(
            "check_sampling_interval",
            f"Sampling interval consistency (expected: {expected_interval_hours}h)",
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # 10. Data integrity validation
    # -------------------------------------------------------------------------
    def validate_data_integrity(
        self,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        primary_telemetry: Optional[List[str]] = None
    ) -> "PiperCleaner":
        """Comprehensive data integrity validation."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        if primary_telemetry is None:
            primary_telemetry = PRIMARY_TELEMETRY
        
        checks = {}
        
        # Check 1: Required columns present
        required = [sensor_col, timestamp_col] + primary_telemetry + [TARGET_COLUMN]
        missing = [c for c in required if c not in self._df.columns]
        checks["required_columns_present"] = len(missing) == 0
        checks["missing_required_columns"] = missing
        
        # Check 2: No duplicate timestamps per sensor
        dup_ts = self._df.duplicated(subset=[sensor_col, timestamp_col]).sum()
        checks["no_duplicate_timestamps_per_sensor"] = dup_ts == 0
        checks["duplicate_timestamp_count"] = int(dup_ts)
        
        # Check 3: Timestamp monotonicity per sensor
        non_monotonic = 0
        for _, group in self._df.groupby(sensor_col):
            if not group[timestamp_col].is_monotonic_increasing:
                non_monotonic += 1
        checks["timestamps_monotonic_per_sensor"] = non_monotonic == 0
        checks["sensors_non_monotonic"] = non_monotonic
        
        # Check 4: Primary telemetry coverage
        telemetry_coverage = {}
        for pt in primary_telemetry:
            if pt in self._df.columns:
                coverage = self._df[pt].notna().mean()
                telemetry_coverage[pt] = float(coverage)
        checks["telemetry_coverage"] = telemetry_coverage
        
        # Check 5: Target column validity (if present)
        if TARGET_COLUMN in self._df.columns:
            unique_labels = sorted(self._df[TARGET_COLUMN].dropna().unique())
            checks["target_column_present"] = True
            checks["target_unique_values"] = unique_labels
            checks["target_valid_binary"] = set(unique_labels).issubset({0, 1})
        else:
            checks["target_column_present"] = False
        
        all_passed = all(
            v for k, v in checks.items() 
            if isinstance(v, bool)
        )
        
        details = {
            "checks": checks,
            "all_checks_passed": all_passed,
            "action": "VALIDATION_REPORT"
        }
        
        self._record_step(
            "validate_data_integrity",
            "Comprehensive data integrity validation",
            df_before, self._df, details
        )
        return self
    
    # -------------------------------------------------------------------------
    # Run full pipeline
    # -------------------------------------------------------------------------
    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        timestamp_format: str = "%d/%m/%Y %H:%M",
        dayfirst: bool = True,
        handle_missing: str = "report_only",  # "drop", "fill", "report_only"
        coerce_numeric: bool = True,
        check_outliers: bool = True,
        outlier_method: str = "iqr"
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Run the complete cleaning pipeline."""
        self.load(df)
        
        # 1. Timestamp validation
        self.validate_timestamps(format_hint=timestamp_format, dayfirst=dayfirst)
        
        # 2. Chronological order
        self.enforce_chronological_order()
        
        # 3. Duplicates
        self.handle_duplicates()
        
        # 4. Missing values detection
        self.detect_missing_values()
        if handle_missing != "report_only":
            self.handle_missing_values(strategy=handle_missing)
        
        # 5. Numeric validation
        self.validate_telemetry_numeric()
        if coerce_numeric:
            self.coerce_telemetry_numeric()
        
        # 6. Impossible values
        self.detect_impossible_values()
        
        # 7. Outliers
        if check_outliers:
            self.inspect_outliers(method=outlier_method)
        
        # 8. Sensor consistency
        self.check_sensor_consistency()
        
        # 9. Sampling interval
        self.check_sampling_interval()
        
        # 10. Data integrity
        self.validate_data_integrity()
        
        self.report.finalize()
        return self._df.copy(), self.report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def clean_apoerch_dataset(
    df: pd.DataFrame,
    dataset_name: str = "ApoErch_LeakageData",
    save_report: bool = True
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Clean the ApoErch dataset with PIPER cleaning pipeline.
    
    Note: This dataset has Volume (cumulative) and derived flow_rate.
    Pressure, Temperature, Vibration are NOT available.
    """
    cleaner = PiperCleaner(dataset_name=dataset_name)
    clean_df, report = cleaner.run_full_pipeline(
        df,
        timestamp_format="%Y-%m-%d %H:%M:%S",  # already parsed to datetime
        dayfirst=True,
        handle_missing="report_only",
        coerce_numeric=True,
        check_outliers=True,
        outlier_method="iqr"
    )
    
    if save_report:
        report_path = PIPER_REPORTS_DIR / f"cleaning_report_{dataset_name}.json"
        report.save(report_path)
        print(f"Cleaning report saved to: {report_path}")
    
    report.print_summary()
    
    return clean_df, report


# Import path constants
from utils.piper_dataset import PIPER_REPORTS_DIR


if __name__ == "__main__":
    # Test cleaning on ApoErch dataset
    from utils.piper_dataset import load_or_create_apoerch_piper_dataset
    
    df, info = load_or_create_apoerch_piper_dataset()
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    clean_df, report = clean_apoerch_dataset(df)
    print(f"Cleaned: {len(clean_df)} rows")


# =============================================================================
# SCADA CLEANING
# =============================================================================

def clean_scada_dataset(
    df: pd.DataFrame,
    dataset_name: str = "SCADA_Pipeline",
    save_report: bool = True
) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Clean the SCADA dataset with PIPER cleaning pipeline.
    
    SCADA has: pressure, flow_rate, temperature (primary telemetry)
    segment_id as sensor_id
    target as label
    """
    cleaner = PiperCleaner(dataset_name=dataset_name)
    clean_df, report = cleaner.run_full_pipeline(
        df,
        timestamp_format="%m/%d/%Y %H:%M",  # SCADA timestamp format
        dayfirst=False,  # month/day/year
        handle_missing="report_only",
        coerce_numeric=True,
        check_outliers=True,
        outlier_method="iqr"
    )
    
    if save_report:
        report_path = PIPER_REPORTS_DIR / f"cleaning_report_{dataset_name}.json"
        report.save(report_path)
        print(f"Cleaning report saved to: {report_path}")
    
    report.print_summary()
    
    return clean_df, report