"""
PIPER Feature Engineering

Creates derived features from primary telemetry while maintaining clear separation:
- PRIMARY TELEMETRY (current adapted scope): pressure, flow_rate, temperature (3 variables)
- DERIVED FEATURES: temporal features, pressure-flow relationships, metadata encoding
- Rolling statistics, rate-of-change, and relative features with long windows are NOT supported
  by the current SCADA dataset (16-minute span) and are excluded.

The final feature configuration must clearly distinguish primary from derived.
No derived feature silently replaces primary telemetry.
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
    is_primary_telemetry, is_derived_feature, DERIVED_FEATURE_PREFIXES
)
from utils.piper_cleaning import PIPER_REPORTS_DIR


# =============================================================================
# FEATURE ENGINEERING REPORT
# =============================================================================

@dataclass
class FeatureEngineeringStepReport:
    """Report for a single feature engineering operation."""
    step_name: str
    description: str
    features_created: List[str]
    feature_types: Dict[str, str]  # feature -> "primary" | "derived" | "temporal" | "metadata"
    parameters: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FeatureEngineeringReport:
    """Complete feature engineering pipeline report."""
    dataset_name: str
    input_columns: List[str]
    output_columns: List[str]
    primary_telemetry_present: List[str]
    primary_telemetry_missing: List[str]
    derived_features: List[str]
    temporal_features: List[str]
    metadata_features: List[str]
    steps: List[FeatureEngineeringStepReport] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    
    def add_step(self, step: FeatureEngineeringStepReport) -> None:
        self.steps.append(step)
    
    def finalize(self) -> None:
        self.end_time = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "primary_telemetry_present": self.primary_telemetry_present,
            "primary_telemetry_missing": self.primary_telemetry_missing,
            "derived_features": self.derived_features,
            "temporal_features": self.temporal_features,
            "metadata_features": self.metadata_features,
            "total_features": len(self.output_columns),
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
        print(f"FEATURE ENGINEERING REPORT: {self.dataset_name}")
        print(f"{'='*70}")
        print(f"Input columns:  {len(self.input_columns)}")
        print(f"Output columns: {len(self.output_columns)}")
        print()
        print(f"--- PRIMARY TELEMETRY PRESENT ({len(self.primary_telemetry_present)}/3) ---")
        for pt in self.primary_telemetry_present:
            print(f"  [OK] {pt}")
        print(f"--- PRIMARY TELEMETRY MISSING ({len(self.primary_telemetry_missing)}/3) ---")
        for pt in self.primary_telemetry_missing:
            print(f"  [MISSING] {pt}")
        print()
        print(f"--- DERIVED FEATURES ({len(self.derived_features)}) ---")
        for f in self.derived_features:
            print(f"  + {f}")
        print()
        print(f"--- TEMPORAL FEATURES ({len(self.temporal_features)}) ---")
        for f in self.temporal_features:
            print(f"  + {f}")
        print()
        print(f"--- METADATA FEATURES ({len(self.metadata_features)}) ---")
        for f in self.metadata_features:
            print(f"  + {f}")
        print(f"{'='*70}\n")


# =============================================================================
# FEATURE ENGINEERING PIPELINE
# =============================================================================

class PiperFeatureEngineer:
    """PIPER feature engineering pipeline with reporting."""
    
    def __init__(self, dataset_name: str = "unknown"):
        self.dataset_name = dataset_name
        self.report = FeatureEngineeringReport(
            dataset_name=dataset_name,
            input_columns=[],
            output_columns=[],
            primary_telemetry_present=[],
            primary_telemetry_missing=[],
            derived_features=[],
            temporal_features=[],
            metadata_features=[]
        )
        self._df: Optional[pd.DataFrame] = None
    
    def load(self, df: pd.DataFrame) -> "PiperFeatureEngineer":
        """Load dataframe for feature engineering."""
        self._df = df.copy()
        self.report.input_columns = df.columns.tolist()
        
        # Check primary telemetry presence
        present = [pt for pt in PRIMARY_TELEMETRY if pt in df.columns]
        missing = [pt for pt in PRIMARY_TELEMETRY if pt not in df.columns]
        self.report.primary_telemetry_present = present
        self.report.primary_telemetry_missing = missing
        
        return self
    
    def get_data(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("No data loaded. Call load() first.")
        return self._df
    
    def _record_step(
        self,
        step_name: str,
        description: str,
        features_created: List[str],
        feature_types: Dict[str, str],
        parameters: Dict[str, Any]
    ) -> None:
        step = FeatureEngineeringStepReport(
            step_name=step_name,
            description=description,
            features_created=features_created,
            feature_types=feature_types,
            parameters=parameters
        )
        self.report.add_step(step)
    
    # -------------------------------------------------------------------------
    # PRIMARY TELEMETRY PASSTHROUGH (ensure they're preserved)
    # -------------------------------------------------------------------------
    def preserve_primary_telemetry(self) -> "PiperFeatureEngineer":
        """Ensure primary telemetry columns are preserved and documented."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        preserved = [pt for pt in PRIMARY_TELEMETRY if pt in self._df.columns]
        
        self._record_step(
            "preserve_primary_telemetry",
            "Primary telemetry columns preserved as-is (no transformation)",
            features_created=preserved,
            feature_types={pt: "primary" for pt in preserved},
            parameters={"primary_telemetry": PRIMARY_TELEMETRY}
        )
        return self
    
    # -------------------------------------------------------------------------
    # TEMPORAL FEATURES
    # -------------------------------------------------------------------------
    def add_temporal_features(
        self,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        features: Optional[List[str]] = None
    ) -> "PiperFeatureEngineer":
        """
        Add temporal features from timestamp.
        
        Available features:
        - hour: Hour of day (0-23)
        - dayofweek: Day of week (0-6, Monday=0)
        - month: Month (1-12)
        - quarter: Quarter (1-4)
        - dayofyear: Day of year (1-366)
        - weekofyear: Week of year (1-53)
        - is_weekend: Boolean (Saturday/Sunday)
        - sin_hour, cos_hour: Cyclical encoding of hour
        - sin_dayofweek, cos_dayofweek: Cyclical encoding of day of week
        - sin_month, cos_month: Cyclical encoding of month
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        if timestamp_col not in self._df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")
        
        if features is None:
            features = [
                "hour", "dayofweek", "month", "quarter", "dayofyear", "weekofyear",
                "is_weekend", "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek",
                "sin_month", "cos_month"
            ]
        
        created = []
        feature_types = {}
        
        ts = self._df[timestamp_col]
        
        if "hour" in features:
            self._df["hour"] = ts.dt.hour
            created.append("hour")
            feature_types["hour"] = "temporal"
        
        if "dayofweek" in features:
            self._df["dayofweek"] = ts.dt.dayofweek
            created.append("dayofweek")
            feature_types["dayofweek"] = "temporal"
        
        if "month" in features:
            self._df["month"] = ts.dt.month
            created.append("month")
            feature_types["month"] = "temporal"
        
        if "quarter" in features:
            self._df["quarter"] = ts.dt.quarter
            created.append("quarter")
            feature_types["quarter"] = "temporal"
        
        if "dayofyear" in features:
            self._df["dayofyear"] = ts.dt.dayofyear
            created.append("dayofyear")
            feature_types["dayofyear"] = "temporal"
        
        if "weekofyear" in features:
            self._df["weekofyear"] = ts.dt.isocalendar().week
            created.append("weekofyear")
            feature_types["weekofyear"] = "temporal"
        
        if "is_weekend" in features:
            self._df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
            created.append("is_weekend")
            feature_types["is_weekend"] = "temporal"
        
        # Cyclical encodings
        if "sin_hour" in features:
            self._df["sin_hour"] = np.sin(2 * np.pi * ts.dt.hour / 24)
            created.append("sin_hour")
            feature_types["sin_hour"] = "temporal"
        
        if "cos_hour" in features:
            self._df["cos_hour"] = np.cos(2 * np.pi * ts.dt.hour / 24)
            created.append("cos_hour")
            feature_types["cos_hour"] = "temporal"
        
        if "sin_dayofweek" in features:
            self._df["sin_dayofweek"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
            created.append("sin_dayofweek")
            feature_types["sin_dayofweek"] = "temporal"
        
        if "cos_dayofweek" in features:
            self._df["cos_dayofweek"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
            created.append("cos_dayofweek")
            feature_types["cos_dayofweek"] = "temporal"
        
        if "sin_month" in features:
            self._df["sin_month"] = np.sin(2 * np.pi * ts.dt.month / 12)
            created.append("sin_month")
            feature_types["sin_month"] = "temporal"
        
        if "cos_month" in features:
            self._df["cos_month"] = np.cos(2 * np.pi * ts.dt.month / 12)
            created.append("cos_month")
            feature_types["cos_month"] = "temporal"
        
        self._record_step(
            "add_temporal_features",
            f"Extract temporal features from {timestamp_col}",
            features_created=created,
            feature_types=feature_types,
            parameters={"timestamp_col": timestamp_col, "features_requested": features}
        )
        return self
    
    # -------------------------------------------------------------------------
    # ROLLING STATISTICS (per sensor)
    # -------------------------------------------------------------------------
    def add_rolling_statistics(
        self,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        telemetry_cols: Optional[List[str]] = None,
        windows: List[int] = None,
        statistics: List[str] = None
    ) -> "PiperFeatureEngineer":
        """
        Add rolling window statistics per sensor.
        
        Args:
            sensor_col: Sensor identifier column
            timestamp_col: Timestamp column (for sorting)
            telemetry_cols: Columns to compute rolling stats on (default: primary telemetry present)
            windows: Window sizes in hours (default: [6, 24, 168, 720] = 6h, 24h, 1w, 30d)
            statistics: Statistics to compute (default: ['mean', 'std', 'min', 'max'])
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        if telemetry_cols is None:
            telemetry_cols = [pt for pt in PRIMARY_TELEMETRY if pt in self._df.columns]
        
        if windows is None:
            windows = [6, 24, 168, 720]  # 6h, 24h, 1 week, 30 days
        
        if statistics is None:
            statistics = ["mean", "std", "min", "max"]
        
        # Ensure sorted
        self._df = self._df.sort_values([sensor_col, timestamp_col]).reset_index(drop=True)
        
        created = []
        feature_types = {}
        
        for col in telemetry_cols:
            if col not in self._df.columns:
                continue
            
            for window in windows:
                window_str = f"{window}h"
                
                # Rolling mean
                if "mean" in statistics:
                    feat_name = f"roll_{window_str}_mean_{col}"
                    self._df[feat_name] = self._df.groupby(sensor_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    created.append(feat_name)
                    feature_types[feat_name] = "derived"
                
                # Rolling std
                if "std" in statistics:
                    feat_name = f"roll_{window_str}_std_{col}"
                    self._df[feat_name] = self._df.groupby(sensor_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std().fillna(0.0)
                    )
                    created.append(feat_name)
                    feature_types[feat_name] = "derived"
                
                # Rolling min
                if "min" in statistics:
                    feat_name = f"roll_{window_str}_min_{col}"
                    self._df[feat_name] = self._df.groupby(sensor_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).min()
                    )
                    created.append(feat_name)
                    feature_types[feat_name] = "derived"
                
                # Rolling max
                if "max" in statistics:
                    feat_name = f"roll_{window_str}_max_{col}"
                    self._df[feat_name] = self._df.groupby(sensor_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
                    created.append(feat_name)
                    feature_types[feat_name] = "derived"
        
        self._record_step(
            "add_rolling_statistics",
            f"Rolling statistics per sensor for {telemetry_cols}",
            features_created=created,
            feature_types=feature_types,
            parameters={
                "sensor_col": sensor_col,
                "telemetry_cols": telemetry_cols,
                "windows_hours": windows,
                "statistics": statistics
            }
        )
        return self
    
    # -------------------------------------------------------------------------
    # RATE OF CHANGE / DIFFERENCES
    # -------------------------------------------------------------------------
    def add_rate_of_change(
        self,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        telemetry_cols: Optional[List[str]] = None,
        periods: List[int] = None
    ) -> "PiperFeatureEngineer":
        """
        Add rate-of-change (difference) features per sensor.
        
        Args:
            sensor_col: Sensor identifier
            timestamp_col: Timestamp (for sorting)
            telemetry_cols: Columns to compute diffs on
            periods: Lag periods for difference (default: [1, 6, 24])
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        if telemetry_cols is None:
            telemetry_cols = [pt for pt in PRIMARY_TELEMETRY if pt in self._df.columns]
        
        if periods is None:
            periods = [1, 6, 24]  # 1h, 6h, 24h
        
        self._df = self._df.sort_values([sensor_col, timestamp_col]).reset_index(drop=True)
        
        created = []
        feature_types = {}
        
        for col in telemetry_cols:
            if col not in self._df.columns:
                continue
            
            for period in periods:
                feat_name = f"diff_{period}h_{col}"
                self._df[feat_name] = self._df.groupby(sensor_col)[col].diff(periods=period)
                created.append(feat_name)
                feature_types[feat_name] = "derived"
        
        self._record_step(
            "add_rate_of_change",
            f"Rate of change (diff) for {telemetry_cols}",
            features_created=created,
            feature_types=feature_types,
            parameters={
                "sensor_col": sensor_col,
                "telemetry_cols": telemetry_cols,
                "periods_hours": periods
            }
        )
        return self
    
    # -------------------------------------------------------------------------
    # RELATIVE / NORMALIZED FEATURES
    # -------------------------------------------------------------------------
    def add_relative_features(
        self,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        telemetry_cols: Optional[List[str]] = None,
        reference_windows: List[int] = None
    ) -> "PiperFeatureEngineer":
        """
        Add relative features (current value / rolling reference).
        
        Args:
            sensor_col: Sensor identifier
            timestamp_col: Timestamp (for sorting)
            telemetry_cols: Columns to normalize
            reference_windows: Rolling window sizes for reference (default: [24, 168])
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        if telemetry_cols is None:
            telemetry_cols = [pt for pt in PRIMARY_TELEMETRY if pt in self._df.columns]
        
        if reference_windows is None:
            reference_windows = [24, 168]  # 24h, 1 week
        
        self._df = self._df.sort_values([sensor_col, timestamp_col]).reset_index(drop=True)
        
        created = []
        feature_types = {}
        
        for col in telemetry_cols:
            if col not in self._df.columns:
                continue
            
            for window in reference_windows:
                window_str = f"{window}h"
                
                # Rolling mean as reference
                ref_col = f"roll_{window_str}_mean_{col}"
                if ref_col not in self._df.columns:
                    self._df[ref_col] = self._df.groupby(sensor_col)[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    if ref_col not in created:
                        created.append(ref_col)
                        feature_types[ref_col] = "derived"
                
                # Relative to rolling mean
                rel_name = f"rel_to_roll_{window_str}_{col}"
                self._df[rel_name] = np.where(
                    self._df[ref_col] != 0,
                    self._df[col] / self._df[ref_col],
                    0.0
                )
                created.append(rel_name)
                feature_types[rel_name] = "derived"
        
        self._record_step(
            "add_relative_features",
            f"Relative features (value / rolling mean) for {telemetry_cols}",
            features_created=created,
            feature_types=feature_types,
            parameters={
                "sensor_col": sensor_col,
                "telemetry_cols": telemetry_cols,
                "reference_windows_hours": reference_windows
            }
        )
        return self
    
    # -------------------------------------------------------------------------
    # PRESSURE-FLOW RELATIONSHIPS (domain-specific)
    # -------------------------------------------------------------------------
    def add_pressure_flow_features(
        self,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value
    ) -> "PiperFeatureEngineer":
        """
        Add pressure-flow relationship features (when both are available).
        
        For the current adapted scope (tabular classification, no temporal history):
        - flow_pressure_ratio: flow_rate / pressure (hydraulic indicator, current observation)
        
        Features requiring historical windows (pressure_drop_from_24h_mean, pressure_velocity)
        are NOT generated as they require temporal depth not available in current dataset.
        """
        if self._df is None:
            raise ValueError("No data loaded")
        
        has_pressure = TelemetryVariable.PRESSURE.value in self._df.columns
        has_flow = TelemetryVariable.FLOW_RATE.value in self._df.columns
        
        if not (has_pressure and has_flow):
            self._record_step(
                "add_pressure_flow_features",
                "Skipped: pressure and/or flow_rate not available",
                features_created=[],
                feature_types={},
                parameters={"has_pressure": has_pressure, "has_flow": has_flow}
            )
            return self
        
        self._df = self._df.sort_values([sensor_col, timestamp_col]).reset_index(drop=True)
        
        created = []
        feature_types = {}
        
        pressure_col = TelemetryVariable.PRESSURE.value
        flow_col = TelemetryVariable.FLOW_RATE.value
        
        # Flow-pressure ratio (hydraulic indicator) - current observation only
        self._df["flow_pressure_ratio"] = np.where(
            self._df[pressure_col] != 0,
            self._df[flow_col] / self._df[pressure_col],
            0.0
        )
        created.append("flow_pressure_ratio")
        feature_types["flow_pressure_ratio"] = "derived"
        
        self._record_step(
            "add_pressure_flow_features",
            "Pressure-flow relationship features (current observation only)",
            features_created=created,
            feature_types=feature_types,
            parameters={"pressure_col": pressure_col, "flow_col": flow_col}
        )
        return self
    
    # -------------------------------------------------------------------------
    # METADATA FEATURES
    # -------------------------------------------------------------------------
    def add_metadata_features(
        self,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        pipeline_col: str = MetadataVariable.PIPELINE_ID.value
    ) -> "PiperFeatureEngineer":
        """Add metadata-based features (sensor encoding, etc.)."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        created = []
        feature_types = {}
        
        # Sensor ID encoding (if not already numeric)
        if sensor_col in self._df.columns:
            # Create integer encoding for sensor
            sensor_codes, uniques = pd.factorize(self._df[sensor_col])
            self._df["sensor_id_encoded"] = sensor_codes
            created.append("sensor_id_encoded")
            feature_types["sensor_id_encoded"] = "metadata"
            
            # Number of unique sensors
            self._df["n_sensors"] = len(uniques)
            created.append("n_sensors")
            feature_types["n_sensors"] = "metadata"
        
        # Pipeline ID encoding (if available)
        if pipeline_col in self._df.columns:
            pipeline_codes, _ = pd.factorize(self._df[pipeline_col])
            self._df["pipeline_id_encoded"] = pipeline_codes
            created.append("pipeline_id_encoded")
            feature_types["pipeline_id_encoded"] = "metadata"
        
        self._record_step(
            "add_metadata_features",
            "Metadata encoding features",
            features_created=created,
            feature_types=feature_types,
            parameters={"sensor_col": sensor_col, "pipeline_col": pipeline_col}
        )
        return self
    
    # -------------------------------------------------------------------------
    # FINALIZE
    # -------------------------------------------------------------------------
    def finalize(self) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
        """Finalize and return engineered dataframe and report."""
        if self._df is None:
            raise ValueError("No data loaded")
        
        self.report.output_columns = self._df.columns.tolist()
        
        # Categorize output features
        self.report.derived_features = [
            c for c in self.report.output_columns
            if (is_derived_feature(c) or c.startswith(("roll_", "diff_", "rel_", "pressure_", "flow_", "sensor_id_encoded", "pipeline_id_encoded", "n_sensors")))
            and not is_primary_telemetry(c)
        ]
        self.report.temporal_features = [
            c for c in self.report.output_columns
            if c in ["hour", "dayofweek", "month", "quarter", "dayofyear", "weekofyear", "is_weekend",
                     "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "sin_month", "cos_month"]
        ]
        self.report.metadata_features = [
            c for c in self.report.output_columns
            if c in METADATA_COLUMNS or c in ["sensor_id_encoded", "pipeline_id_encoded", "n_sensors"]
        ]
        
        self.report.finalize()
        return self._df.copy(), self.report
    
    # -------------------------------------------------------------------------
    # RUN FULL PIPELINE
    # -------------------------------------------------------------------------
    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        timestamp_col: str = MetadataVariable.TIMESTAMP.value,
        sensor_col: str = MetadataVariable.SENSOR_ID.value,
        add_temporal: bool = True,
        add_rolling: bool = True,
        add_rate_of_change: bool = True,
        add_relative: bool = True,
        add_pressure_flow: bool = True,
        add_metadata: bool = True,
        rolling_windows: List[int] = None,
        rolling_stats: List[str] = None,
        diff_periods: List[int] = None,
        relative_windows: List[int] = None
    ) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
        """Run complete feature engineering pipeline."""
        self.load(df)
        
        # 1. Preserve primary telemetry
        self.preserve_primary_telemetry()
        
        # 2. Temporal features
        if add_temporal:
            self.add_temporal_features(timestamp_col=timestamp_col)
        
        # 3. Rolling statistics
        if add_rolling:
            telemetry_cols = [pt for pt in PRIMARY_TELEMETRY if pt in self._df.columns]
            self.add_rolling_statistics(
                sensor_col=sensor_col,
                timestamp_col=timestamp_col,
                telemetry_cols=telemetry_cols,
                windows=rolling_windows,
                statistics=rolling_stats
            )
        
        # 4. Rate of change
        if add_rate_of_change:
            telemetry_cols = [pt for pt in PRIMARY_TELEMETRY if pt in self._df.columns]
            self.add_rate_of_change(
                sensor_col=sensor_col,
                timestamp_col=timestamp_col,
                telemetry_cols=telemetry_cols,
                periods=diff_periods
            )
        
        # 5. Relative features
        if add_relative:
            telemetry_cols = [pt for pt in PRIMARY_TELEMETRY if pt in self._df.columns]
            self.add_relative_features(
                sensor_col=sensor_col,
                timestamp_col=timestamp_col,
                telemetry_cols=telemetry_cols,
                reference_windows=relative_windows
            )
        
        # 6. Pressure-flow relationships (if both available)
        if add_pressure_flow:
            self.add_pressure_flow_features(
                sensor_col=sensor_col,
                timestamp_col=timestamp_col
            )
        
        # 7. Metadata features
        if add_metadata:
            self.add_metadata_features(
                sensor_col=sensor_col,
                pipeline_col=MetadataVariable.PIPELINE_ID.value
            )
        
        return self.finalize()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def engineer_apoerch_features(
    df: pd.DataFrame,
    dataset_name: str = "ApoErch_LeakageData",
    save_report: bool = True
) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
    """Engineer features for ApoErch dataset (has flow_rate derived from Volume)."""
    engineer = PiperFeatureEngineer(dataset_name=dataset_name)
    
    engineered_df, report = engineer.run_full_pipeline(
        df,
        timestamp_col=MetadataVariable.TIMESTAMP.value,
        sensor_col=MetadataVariable.SENSOR_ID.value,
        add_temporal=True,
        add_rolling=True,
        add_rate_of_change=True,
        add_relative=True,
        add_pressure_flow=False,  # pressure not available
        add_metadata=True,
        rolling_windows=[6, 24, 168],
        rolling_stats=["mean", "std"],
        diff_periods=[1, 6, 24],
        relative_windows=[24, 168]
    )
    
    if save_report:
        report_path = PIPER_REPORTS_DIR / f"feature_report_{dataset_name}.json"
        report.save(report_path)
        print(f"Feature engineering report saved to: {report_path}")
    
    report.print_summary()
    
    return engineered_df, report


if __name__ == "__main__":
    # Test feature engineering on cleaned ApoErch dataset
    from utils.piper_dataset import load_or_create_apoerch_piper_dataset
    from utils.piper_cleaning import clean_apoerch_dataset
    
    df, info = load_or_create_apoerch_piper_dataset()
    clean_df, clean_report = clean_apoerch_dataset(df)
    
    engineered_df, feat_report = engineer_apoerch_features(clean_df)
    print(f"Engineered: {len(engineered_df)} rows, {len(engineered_df.columns)} columns")


# =============================================================================
# SCADA FEATURE ENGINEERING
# =============================================================================

def engineer_scada_features(
    df: pd.DataFrame,
    dataset_name: str = "SCADA_Pipeline",
    save_report: bool = True
) -> Tuple[pd.DataFrame, FeatureEngineeringReport]:
    """
    Engineer features for SCADA dataset (has pressure, flow_rate, temperature).
    
    NOTE: The SCADA dataset spans only ~16 minutes with multiple readings per timestamp.
    Rolling statistics, rate-of-change, and relative features with historical windows
    are NOT supported and are DISABLED. Only features valid for the current observation
    are generated: temporal features, pressure-flow ratio, metadata encoding.
    """
    engineer = PiperFeatureEngineer(dataset_name=dataset_name)
    
    engineered_df, report = engineer.run_full_pipeline(
        df,
        timestamp_col=MetadataVariable.TIMESTAMP.value,
        sensor_col=MetadataVariable.SENSOR_ID.value,
        add_temporal=True,
        add_rolling=False,       # Not supported: dataset span ~16 min, multiple readings per timestamp
        add_rate_of_change=False, # Not supported: irregular intervals, multiple readings per timestamp
        add_relative=False,       # Not supported: no meaningful historical reference
        add_pressure_flow=True,   # Current observation only: pressure-flow ratio
        add_metadata=True,
        # Parameters below are ignored when corresponding features are disabled
        rolling_windows=[6, 24, 168],
        rolling_stats=["mean", "std"],
        diff_periods=[1, 6, 24],
        relative_windows=[24, 168]
    )
    
    if save_report:
        report_path = PIPER_REPORTS_DIR / f"feature_report_{dataset_name}.json"
        report.save(report_path)
        print(f"Feature engineering report saved to: {report_path}")
    
    report.print_summary()
    
    return engineered_df, report