"""
PIPER Label Generation Interface

Implements a reproducible label interface with clear distinction between:
- REAL labels: from actual leak/abnormal condition records
- SYNTHETIC labels: generated for development/testing ONLY

Canonical target:
    0 = Normal
    1 = Abnormal / Leak

IMPORTANT: Synthetic labels must NEVER be represented as real experimental ground truth.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
import json
from datetime import datetime

from utils.piper_telemetry import (
    PRIMARY_TELEMETRY, METADATA_COLUMNS, TARGET_COLUMN,
    TelemetryVariable, MetadataVariable, TargetVariable
)
from utils.piper_dataset import PIPER_REPORTS_DIR


# =============================================================================
# LABEL SOURCE TYPES
# =============================================================================

class LabelSourceType(str, Enum):
    """Source of label - MUST be explicitly declared."""
    REAL_GROUND_TRUTH = "real_ground_truth"      # From actual leak records, SCADA alarms, maintenance logs
    REAL_OPERATOR_ANNOTATED = "real_operator_annotated"  # Manually labeled by domain experts
    SYNTHETIC_INJECTED = "synthetic_injected"    # Algorithmically injected (development only)
    SYNTHETIC_RULE_BASED = "synthetic_rule_based"  # Rule-based generation (development only)
    UNKNOWN = "unknown"


class LabelMethod(str, Enum):
    """Label generation method."""
    EXISTING_COLUMN = "existing_column"           # Column already has 0/1 labels
    THRESHOLD_BASED = "threshold_based"           # Derived from telemetry thresholds
    RULE_BASED = "rule_based"                     # Domain rules (e.g., pressure drop + flow increase)
    SYNTHETIC_RANDOM = "synthetic_random"         # Random injection (development only)
    SYNTHETIC_PROGRESSIVE = "synthetic_progressive"  # Progressive leak model (development only)
    EXTERNAL_MERGE = "external_merge"             # Merged from external leak database


# =============================================================================
# LABEL GENERATION REPORT
# =============================================================================

@dataclass
class LabelGenerationReport:
    """Complete label generation report."""
    dataset_name: str
    label_source: LabelSourceType
    label_method: LabelMethod
    description: str
    
    # Class distribution
    n_normal: int = 0
    n_abnormal: int = 0
    total_labeled: int = 0
    class_balance: float = 0.0  # abnormal / total
    
    # Source column info
    source_column: Optional[str] = None
    source_values: Optional[List[Any]] = None
    mapping_applied: Optional[Dict[Any, int]] = None
    
    # Generation parameters
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def is_synthetic(self) -> bool:
        return self.label_source in [
            LabelSourceType.SYNTHETIC_INJECTED,
            LabelSourceType.SYNTHETIC_RULE_BASED
        ]
    
    @property
    def is_real(self) -> bool:
        return self.label_source in [
            LabelSourceType.REAL_GROUND_TRUTH,
            LabelSourceType.REAL_OPERATOR_ANNOTATED
        ]
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['label_source'] = self.label_source.value
        d['label_method'] = self.label_method.value
        d['is_synthetic'] = self.is_synthetic
        d['is_real'] = self.is_real
        return d
    
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def print_summary(self) -> None:
        source_label = "SYNTHETIC" if self.is_synthetic else "REAL" if self.is_real else "UNKNOWN"
        print(f"\n{'='*70}")
        print(f"LABEL GENERATION REPORT: {self.dataset_name} [{source_label}]")
        print(f"{'='*70}")
        print(f"Label Source:     {self.label_source.value}")
        print(f"Label Method:     {self.label_method.value}")
        print(f"Description:      {self.description}")
        print()
        print(f"Class Distribution:")
        print(f"  Normal (0):     {self.n_normal:,}")
        print(f"  Abnormal (1):   {self.n_abnormal:,}")
        print(f"  Total:          {self.total_labeled:,}")
        print(f"  Abnormal Rate:  {self.class_balance:.4f} ({self.class_balance*100:.2f}%)")
        print()
        if self.source_column:
            print(f"Source Column:    {self.source_column}")
            print(f"Source Values:    {self.source_values}")
            print(f"Mapping Applied:  {self.mapping_applied}")
        print()
        if self.generation_params:
            print(f"Generation Parameters:")
            for k, v in self.generation_params.items():
                print(f"  {k}: {v}")
        print()
        if self.warnings:
            print(f"!!! WARNINGS !!!")
            for w in self.warnings:
                print(f"  - {w}")
        print()
        if self.is_synthetic:
            print(f"!!! THIS DATASET USES SYNTHETIC LABELS !!!")
            print(f"    These labels are for DEVELOPMENT/TESTING only.")
            print(f"    They do NOT represent real leak events.")
            print(f"    Do NOT use for final experimental validation.")
        print(f"{'='*70}\n")


# =============================================================================
# LABEL GENERATORS
# =============================================================================

class PiperLabelGenerator:
    """PIPER label generation with explicit source tracking."""
    
    def __init__(self, dataset_name: str = "unknown"):
        self.dataset_name = dataset_name
        self.report: Optional[LabelGenerationReport] = None
        self._df: Optional[pd.DataFrame] = None
    
    def load(self, df: pd.DataFrame) -> "PiperLabelGenerator":
        self._df = df.copy()
        return self
    
    def get_data(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("No data loaded")
        return self._df
    
    # -------------------------------------------------------------------------
    # REAL LABELS: Map existing column to canonical target
    # -------------------------------------------------------------------------
    def map_existing_labels(
        self,
        source_column: str,
        mapping: Dict[Any, int],
        target_column: str = TARGET_COLUMN,
        label_source: LabelSourceType = LabelSourceType.REAL_GROUND_TRUTH,
        label_method: LabelMethod = LabelMethod.EXISTING_COLUMN,
        description: str = "Mapped existing label column to canonical target"
    ) -> "PiperLabelGenerator":
        """
        Map an existing label column to canonical 0/1 target.
        
        Args:
            source_column: Column containing original labels
            mapping: Dict mapping source values -> {0, 1}
            target_column: Output column name (default: 'label')
            label_source: Source type (must be REAL_* for real labels)
            label_method: Method used
            description: Human-readable description
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        if source_column not in self._df.columns:
            raise ValueError(f"Source column '{source_column}' not found")
        
        df_before = self._df.copy()
        
        # Apply mapping
        source_values = self._df[source_column].copy()
        self._df[target_column] = source_values.map(mapping)
        
        # Check for unmapped values
        unmapped_mask = source_values.notna() & self._df[target_column].isna()
        n_unmapped = int(unmapped_mask.sum())
        
        if n_unmapped > 0:
            unmapped_values = self._df.loc[unmapped_mask, source_column].unique().tolist()
            raise ValueError(
                f"Unmapped values in {source_column}: {unmapped_values}. "
                f"Add them to mapping dict."
            )
        
        # Drop rows where target is NaN (no label available)
        n_before = len(self._df)
        self._df = self._df.dropna(subset=[target_column]).copy()
        self._df[target_column] = self._df[target_column].astype(int)
        n_after = len(self._df)
        
        # Validate binary
        unique_vals = set(self._df[target_column].unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(f"Target contains non-binary values: {unique_vals}")
        
        n_normal = int((self._df[target_column] == 0).sum())
        n_abnormal = int((self._df[target_column] == 1).sum())
        
        self.report = LabelGenerationReport(
            dataset_name=self.dataset_name,
            label_source=label_source,
            label_method=label_method,
            description=description,
            n_normal=n_normal,
            n_abnormal=n_abnormal,
            total_labeled=len(self._df),
            class_balance=n_abnormal / len(self._df) if len(self._df) > 0 else 0,
            source_column=source_column,
            source_values=sorted(self._df[source_column].dropna().unique().tolist()),
            mapping_applied=mapping,
            generation_params={},
            warnings=[]
        )
        
        if n_unmapped > 0:
            self.report.warnings.append(f"{n_unmapped} rows had unmapped source values (dropped)")
        if n_before > n_after:
            self.report.warnings.append(f"{n_before - n_after} rows dropped due to missing labels")
        
        return self
    
    # -------------------------------------------------------------------------
    # REAL LABELS: Merge from external leak database
    # -------------------------------------------------------------------------
    def merge_external_labels(
        self,
        leak_df: pd.DataFrame,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        leak_start_col: str = "leak_start",
        leak_end_col: str = "leak_end",
        leak_type_col: Optional[str] = None,
        target_column: str = TARGET_COLUMN,
        label_source: LabelSourceType = LabelSourceType.REAL_GROUND_TRUTH,
        label_method: LabelMethod = LabelMethod.EXTERNAL_MERGE,
        description: str = "Merged labels from external leak database"
    ) -> "PiperLabelGenerator":
        """
        Merge labels from external leak event database.
        
        leak_df must have: sensor_id, leak_start, leak_end (and optionally leak_type)
        Creates binary label: 1 during leak window, 0 otherwise.
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        df_before = self._df.copy()
        
        # Validate leak_df
        required = [sensor_col, leak_start_col, leak_end_col]
        missing = [c for c in required if c not in leak_df.columns]
        if missing:
            raise ValueError(f"Leak dataframe missing columns: {missing}")
        
        # Ensure datetime
        leak_df = leak_df.copy()
        leak_df[leak_start_col] = pd.to_datetime(leak_df[leak_start_col])
        leak_df[leak_end_col] = pd.to_datetime(leak_df[leak_end_col])
        
        # Initialize target column
        self._df[target_column] = 0
        
        # For each leak event, mark rows in window
        n_leak_rows = 0
        for _, leak in leak_df.iterrows():
            sensor = leak[sensor_col]
            start = leak[leak_start_col]
            end = leak[leak_end_col]
            
            mask = (
                (self._df[sensor_col] == sensor) &
                (self._df[timestamp_col] >= start) &
                (self._df[timestamp_col] <= end)
            )
            self._df.loc[mask, target_column] = 1
            n_leak_rows += int(mask.sum())
        
        n_normal = int((self._df[target_column] == 0).sum())
        n_abnormal = int((self._df[target_column] == 1).sum())
        
        self.report = LabelGenerationReport(
            dataset_name=self.dataset_name,
            label_source=label_source,
            label_method=label_method,
            description=description,
            n_normal=n_normal,
            n_abnormal=n_abnormal,
            total_labeled=len(self._df),
            class_balance=n_abnormal / len(self._df) if len(self._df) > 0 else 0,
            source_column="external_leak_database",
            generation_params={
                "n_leak_events": len(leak_df),
                "n_leak_rows_marked": n_leak_rows,
                "leak_types": leak_df[leak_type_col].unique().tolist() if leak_type_col else None
            },
            warnings=[]
        )
        
        return self
    
    # -------------------------------------------------------------------------
    # SYNTHETIC LABELS: Threshold-based (development only)
    # -------------------------------------------------------------------------
    def generate_threshold_labels(
        self,
        telemetry_col: str,
        threshold: float,
        direction: Literal["above", "below"] = "above",
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        target_column: str = TARGET_COLUMN,
        label_source: LabelSourceType = LabelSourceType.SYNTHETIC_RULE_BASED,
        label_method: LabelMethod = LabelMethod.THRESHOLD_BASED,
        description: str = "Threshold-based synthetic labels (DEVELOPMENT ONLY)"
    ) -> "PiperLabelGenerator":
        """
        Generate synthetic labels based on telemetry threshold.
        
        WARNING: This creates SYNTHETIC labels for development only.
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        if telemetry_col not in self._df.columns:
            raise ValueError(f"Telemetry column '{telemetry_col}' not found")
        
        if label_source not in [LabelSourceType.SYNTHETIC_INJECTED, LabelSourceType.SYNTHETIC_RULE_BASED]:
            raise ValueError("Threshold-based labels must use SYNTHETIC label source")
        
        df_before = self._df.copy()
        
        if direction == "above":
            self._df[target_column] = (self._df[telemetry_col] > threshold).astype(int)
        else:
            self._df[target_column] = (self._df[telemetry_col] < threshold).astype(int)
        
        n_normal = int((self._df[target_column] == 0).sum())
        n_abnormal = int((self._df[target_column] == 1).sum())
        
        self.report = LabelGenerationReport(
            dataset_name=self.dataset_name,
            label_source=label_source,
            label_method=label_method,
            description=description,
            n_normal=n_normal,
            n_abnormal=n_abnormal,
            total_labeled=len(self._df),
            class_balance=n_abnormal / len(self._df) if len(self._df) > 0 else 0,
            generation_params={
                "telemetry_col": telemetry_col,
                "threshold": threshold,
                "direction": direction
            },
            warnings=[
                "SYNTHETIC LABELS - for development/testing only",
                "Not suitable for final experimental validation",
                f"Threshold-based on {telemetry_col} {direction} {threshold}"
            ]
        )
        
        return self
    
    # -------------------------------------------------------------------------
    # SYNTHETIC LABELS: Random injection (matches ApoErch approach)
    # -------------------------------------------------------------------------
    def generate_synthetic_leak_segments(
        self,
        leak_share_total: float = 0.20,
        n_segments_range: Tuple[int, int] = (2, 5),
        min_len_frac: float = 0.05,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        target_column: str = TARGET_COLUMN,
        seed: int = 42,
        test_fraction: float = 0.30,
        ensure_test_leak_abs_share: float = 0.05,
        label_source: LabelSourceType = LabelSourceType.SYNTHETIC_INJECTED,
        label_method: LabelMethod = LabelMethod.SYNTHETIC_PROGRESSIVE,
        description: str = "Synthetic progressive leak injection (ApoErch compatible, DEVELOPMENT ONLY)"
    ) -> "PiperLabelGenerator":
        """
        Generate synthetic leak segments per sensor (ApoErch compatible).
        
        This replicates the inject_random_leak_segments logic from DataAggregation.py
        but with explicit SYNTHETIC labeling and reporting.
        
        WARNING: Creates SYNTHETIC labels for development only.
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        if sensor_col not in self._df.columns or timestamp_col not in self._df.columns:
            raise ValueError(f"Missing required columns: {sensor_col}, {timestamp_col}")
        
        # This is the exact logic from DataAggregation.inject_random_leak_segments
        # but we apply it to the already-cleaned data
        rng = np.random.default_rng(seed)
        rows_out = []
        
        for _, g in self._df.groupby(sensor_col, sort=False):
            g = g.sort_values(timestamp_col).reset_index(drop=True)
            n = len(g)
            if n == 0:
                continue
            
            total_leak_pts = int(round(leak_share_total * n))
            test_len = int(round(test_fraction * n))
            test_start = max(0, n - test_len)
            
            desired_test_pts = int(round(ensure_test_leak_abs_share * n))
            desired_test_pts = min(desired_test_pts, test_len)
            desired_train_pts = total_leak_pts - desired_test_pts
            desired_train_pts = min(desired_train_pts, n - test_len)
            
            if desired_test_pts + desired_train_pts != total_leak_pts:
                desired_train_pts = total_leak_pts - desired_test_pts
            
            def sample_parts(target_pts: int) -> list[int]:
                if target_pts <= 0:
                    return []
                k_min, k_max = n_segments_range
                seg_min = max(1, int(np.floor(min_len_frac * n)))
                k = int(rng.integers(k_min, k_max + 1))
                if target_pts < seg_min:
                    return [target_pts]
                k = max(1, min(k, target_pts // seg_min))
                base = seg_min * k
                slack = max(0, target_pts - base)
                parts = (rng.multinomial(slack, [1.0 / k] * k) + seg_min).astype(int)
                return list(map(int, parts))
            
            test_parts = sample_parts(desired_test_pts)
            train_parts = sample_parts(desired_train_pts)
            rng.shuffle(test_parts)
            rng.shuffle(train_parts)
            
            taken = np.zeros(n, dtype=bool)
            segs = []
            
            def place_one(L: int, start_low: int, start_high: int):
                for _ in range(200):
                    if start_high < start_low:
                        break
                    s = int(rng.integers(start_low, start_high + 1))
                    e = s + L
                    if e > n:
                        continue
                    if not taken[s:e].any():
                        taken[s:e] = True
                        return True, (s, e)
                for s in range(start_low, max(start_high, start_low) + 1):
                    e = s + L
                    if e > n:
                        break
                    if not taken[s:e].any():
                        taken[s:e] = True
                        return True, (s, e)
                return False, None
            
            for L in test_parts:
                ok, seg = place_one(L, test_start, n - L)
                if ok:
                    segs.append(seg)
            
            for L in train_parts:
                ok, seg = place_one(L, 0, max(0, test_start - L))
                if ok:
                    segs.append(seg)
            
            placed_pts = sum((e - s) for s, e in segs)
            deficit = total_leak_pts - placed_pts
            while deficit > 0:
                L = min(deficit, max(1, int(np.floor(min_len_frac * n))))
                ok, seg = place_one(L, 0, n - L)
                if not ok:
                    break
                segs.append(seg)
                deficit -= L
            
            leak_mask = np.zeros(n, dtype=bool)
            for s, e in segs:
                leak_mask[s:e] = True
            
            if leak_mask.sum() > total_leak_pts:
                excess = leak_mask.sum() - total_leak_pts
                idx = np.where(leak_mask)[0]
                leak_mask[idx[-excess:]] = False
            
            out = g.copy()
            out[target_column] = leak_mask.astype(int)
            rows_out.append(out)
        
        combined = pd.concat(rows_out, ignore_index=True)
        combined.sort_values([sensor_col, timestamp_col], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        
        self._df = combined
        
        n_normal = int((self._df[target_column] == 0).sum())
        n_abnormal = int((self._df[target_column] == 1).sum())
        
        self.report = LabelGenerationReport(
            dataset_name=self.dataset_name,
            label_source=label_source,
            label_method=label_method,
            description=description,
            n_normal=n_normal,
            n_abnormal=n_abnormal,
            total_labeled=len(self._df),
            class_balance=n_abnormal / len(self._df) if len(self._df) > 0 else 0,
            generation_params={
                "leak_share_total": leak_share_total,
                "n_segments_range": n_segments_range,
                "min_len_frac": min_len_frac,
                "seed": seed,
                "test_fraction": test_fraction,
                "ensure_test_leak_abs_share": ensure_test_leak_abs_share
            },
            warnings=[
                "SYNTHETIC LABELS - for development/testing only",
                "Not suitable for final experimental validation",
                "Labels generated by algorithmic injection, not from real events",
                "Matches ApoErch inject_random_leak_segments logic"
            ]
        )
        
        return self
    
    # -------------------------------------------------------------------------
    # FINALIZE
    # -------------------------------------------------------------------------
    def finalize(self) -> Tuple[pd.DataFrame, LabelGenerationReport]:
        if self._df is None:
            raise ValueError("No data loaded")
        if self.report is None:
            raise ValueError("No label generation method called")
        
        # Ensure target column exists
        if TARGET_COLUMN not in self._df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not created")
        
        return self._df.copy(), self.report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def label_apoerch_synthetic(
    df: pd.DataFrame,
    dataset_name: str = "ApoErch_LeakageData",
    save_report: bool = True
) -> Tuple[pd.DataFrame, LabelGenerationReport]:
    """
    Generate synthetic labels for ApoErch dataset (development only).
    
    Uses the same progressive leak injection as the original ApoErch code.
    Explicitly marked as SYNTHETIC.
    """
    generator = PiperLabelGenerator(dataset_name=dataset_name)
    labeled_df, report = generator.load(df).generate_synthetic_leak_segments().finalize()
    
    if save_report:
        report_path = PIPER_REPORTS_DIR / f"label_report_{dataset_name}.json"
        report.save(report_path)
        print(f"Label generation report saved to: {report_path}")
    
    report.print_summary()
    
    return labeled_df, report


def label_from_existing_column(
    df: pd.DataFrame,
    source_column: str,
    mapping: Dict[Any, int],
    dataset_name: str = "unknown",
    label_source: LabelSourceType = LabelSourceType.REAL_GROUND_TRUTH,
    description: str = "Mapped existing label column",
    save_report: bool = True
) -> Tuple[pd.DataFrame, LabelGenerationReport]:
    """Map existing label column to canonical target."""
    generator = PiperLabelGenerator(dataset_name=dataset_name)
    labeled_df, report = (
        generator.load(df)
        .map_existing_labels(
            source_column=source_column,
            mapping=mapping,
            label_source=label_source,
            description=description
        )
        .finalize()
    )
    
    if save_report:
        report_path = PIPER_REPORTS_DIR / f"label_report_{dataset_name}.json"
        report.save(report_path)
        print(f"Label generation report saved to: {report_path}")
    
    report.print_summary()
    
    return labeled_df, report


if __name__ == "__main__":
    # Test label generation on engineered ApoErch dataset
    from utils.piper_dataset import load_or_create_apoerch_piper_dataset
    from utils.piper_cleaning import clean_apoerch_dataset
    from utils.piper_features import engineer_apoerch_features
    
    df, info = load_or_create_apoerch_piper_dataset()
    clean_df, clean_report = clean_apoerch_dataset(df)
    engineered_df, feat_report = engineer_apoerch_features(clean_df)
    
    labeled_df, label_report = label_apoerch_synthetic(engineered_df)
    print(f"Labeled: {len(labeled_df)} rows")
    print(f"Label distribution:")
    print(labeled_df[TARGET_COLUMN].value_counts())


# =============================================================================
# SCADA LABEL GENERATION
# =============================================================================

def label_scada_real(
    df: pd.DataFrame,
    dataset_name: str = "SCADA_Pipeline",
    save_report: bool = True
) -> Tuple[pd.DataFrame, LabelGenerationReport]:
    """
    Map existing target column to canonical label for SCADA dataset.
    
    SCADA has a real 'target' column with 0=normal, 1=abnormal.
    After column mapping, this becomes 'label' (canonical target name).
    This is REAL ground truth, not synthetic.
    """
    generator = PiperLabelGenerator(dataset_name=dataset_name)
    labeled_df, report = (
        generator.load(df)
        .map_existing_labels(
            source_column="label",  # Already mapped to canonical 'label'
            mapping={0: 0, 1: 1},  # Already 0/1, but explicit mapping for clarity
            target_column=TARGET_COLUMN,
            label_source=LabelSourceType.REAL_GROUND_TRUTH,
            label_method=LabelMethod.EXISTING_COLUMN,
            description="Mapped SCADA target column (0=normal, 1=abnormal) to canonical label"
        )
        .finalize()
    )
    
    if save_report:
        report_path = PIPER_REPORTS_DIR / f"label_report_{dataset_name}.json"
        report.save(report_path)
        print(f"Label generation report saved to: {report_path}")
    
    report.print_summary()
    
    return labeled_df, report