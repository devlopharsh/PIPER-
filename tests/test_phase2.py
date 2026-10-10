"""
PIPER Phase 2 Tests

Tests for:
1. Telemetry schema validation (adapted scope: 3 primary telemetry)
2. Required-feature detection
3. Column mapping
4. Timestamp parsing
5. Duplicate detection (conflicting keys preserved)
6. Missing-value handling
7. Feature generation (no unsupported long-window features)
8. Label validation
9. Raw-data preservation
10. No train/test preprocessing leakage where applicable

Tests MUST fail clearly when mandatory PIPER telemetry is absent
rather than silently substituting other variables.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Import PIPER modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.piper_telemetry import (
    PRIMARY_TELEMETRY, METADATA_COLUMNS, TARGET_COLUMN, REQUIRED_COLUMNS,
    ColumnMapping, APOERCH_COLUMN_MAPPING, SCADA_COLUMN_MAPPING, validate_mapping,
    TelemetryVariable, MetadataVariable, TargetVariable,
    is_primary_telemetry, is_derived_feature
)
from utils.piper_dataset import (
    load_apoerch_raw, apply_column_mapping, derive_flow_rate_from_volume,
    create_apoerch_piper_dataset, load_or_create_scada_piper_dataset, DatasetCatalog
)
from utils.piper_cleaning import (
    PiperCleaner, clean_apoerch_dataset, clean_scada_dataset
)
from utils.piper_features import (
    PiperFeatureEngineer, engineer_apoerch_features, engineer_scada_features
)
from utils.piper_labels import (
    PiperLabelGenerator, label_apoerch_synthetic, label_scada_real,
    LabelSourceType, LabelMethod, TARGET_COLUMN
)
from utils.constants import DATA_DIR as LEGACY_DATA_DIR


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def raw_apoerch_df():
    """Load raw ApoErch dataset (immutable)."""
    return load_apoerch_raw(LEGACY_DATA_DIR)


@pytest.fixture(scope="session")
def mapped_apoerch_df():
    """Load ApoErch dataset mapped to PIPER schema with derived flow_rate."""
    df, _ = create_apoerch_piper_dataset(save_raw_copy=False)
    return df


@pytest.fixture(scope="session")
def cleaned_apoerch_df():
    """Cleaned ApoErch dataset."""
    df, _ = create_apoerch_piper_dataset(save_raw_copy=False)
    clean_df, _ = clean_apoerch_dataset(df, save_report=False)
    return clean_df


@pytest.fixture(scope="session")
def engineered_apoerch_df():
    """Feature-engineered ApoErch dataset."""
    df, _ = create_apoerch_piper_dataset(save_raw_copy=False)
    clean_df, _ = clean_apoerch_dataset(df, save_report=False)
    engineered_df, _ = engineer_apoerch_features(clean_df, save_report=False)
    return engineered_df


@pytest.fixture(scope="session")
def labeled_apoerch_df():
    """Labeled ApoErch dataset (synthetic labels)."""
    df, _ = create_apoerch_piper_dataset(save_raw_copy=False)
    clean_df, _ = clean_apoerch_dataset(df, save_report=False)
    engineered_df, _ = engineer_apoerch_features(clean_df, save_report=False)
    labeled_df, _ = label_apoerch_synthetic(engineered_df, save_report=False)
    return labeled_df


# SCADA fixtures
@pytest.fixture(scope="session")
def scada_mapped_df():
    """Load SCADA dataset mapped to PIPER schema."""
    df, _ = load_or_create_scada_piper_dataset(force_reload=False)
    return df


@pytest.fixture(scope="session")
def scada_cleaned_df():
    """Cleaned SCADA dataset."""
    df, _ = load_or_create_scada_piper_dataset(force_reload=False)
    clean_df, _ = clean_scada_dataset(df, save_report=False)
    return clean_df


@pytest.fixture(scope="session")
def scada_engineered_df():
    """Feature-engineered SCADA dataset."""
    df, _ = load_or_create_scada_piper_dataset(force_reload=False)
    clean_df, _ = clean_scada_dataset(df, save_report=False)
    engineered_df, _ = engineer_scada_features(clean_df, save_report=False)
    return engineered_df


@pytest.fixture(scope="session")
def scada_labeled_df():
    """Labeled SCADA dataset (real labels)."""
    df, _ = load_or_create_scada_piper_dataset(force_reload=False)
    clean_df, _ = clean_scada_dataset(df, save_report=False)
    engineered_df, _ = engineer_scada_features(clean_df, save_report=False)
    labeled_df, _ = label_scada_real(engineered_df, save_report=False)
    return labeled_df


# =============================================================================
# 2.1 DATASET ACQUISITION TESTS
# =============================================================================

class TestDatasetAcquisition:
    """Test dataset discovery and loading."""
    
    def test_legacy_data_directory_exists(self):
        """Legacy data directory must exist."""
        assert LEGACY_DATA_DIR.exists(), f"Legacy data directory not found: {LEGACY_DATA_DIR}"
    
    def test_legacy_csv_files_exist(self):
        """At least one CSV file must exist in legacy data directory."""
        csv_files = list(LEGACY_DATA_DIR.glob("*.csv"))
        assert len(csv_files) > 0, f"No CSV files found in {LEGACY_DATA_DIR}"
    
    def test_load_raw_apoerch(self, raw_apoerch_df):
        """Raw ApoErch data loads correctly."""
        assert len(raw_apoerch_df) > 0
        assert "Meter_ID" in raw_apoerch_df.columns
        assert "Timestamp" in raw_apoerch_df.columns
        assert "Volume" in raw_apoerch_df.columns
        assert pd.api.types.is_datetime64_any_dtype(raw_apoerch_df["Timestamp"])
    
    def test_raw_data_immutability(self, raw_apoerch_df):
        """Raw data must not be modified by loading."""
        csv_files = list(LEGACY_DATA_DIR.glob("*.csv"))
        total_csv_rows = sum(len(pd.read_csv(f)) for f in csv_files)
        assert len(raw_apoerch_df) == total_csv_rows
    
    def test_create_apoerch_piper_dataset(self, mapped_apoerch_df):
        """PIPER-mapped dataset creation works."""
        assert len(mapped_apoerch_df) > 0
        assert "sensor_id" in mapped_apoerch_df.columns
        assert "timestamp" in mapped_apoerch_df.columns
        assert "flow_rate" in mapped_apoerch_df.columns  # derived
    
    def test_raw_copy_preserved(self):
        """Raw copy is saved to data/raw."""
        raw_copy = Path("./data/raw/apoerch_raw.parquet")
        assert raw_copy.exists(), "Raw copy not preserved in data/raw"
        raw_df = pd.read_parquet(raw_copy)
        assert len(raw_df) == 270499
    
    def test_scada_raw_file_exists(self):
        """SCADA raw CSV exists."""
        scada_path = Path("./data/raw/scada_pipeline.csv")
        assert scada_path.exists(), f"SCADA raw file not found: {scada_path}"
    
    def test_load_scada_raw(self, scada_mapped_df):
        """SCADA raw data loads and maps correctly."""
        assert len(scada_mapped_df) == 1000  # raw rows before cleaning
        assert "sensor_id" in scada_mapped_df.columns
        assert "timestamp" in scada_mapped_df.columns
        assert "pressure" in scada_mapped_df.columns
        assert "flow_rate" in scada_mapped_df.columns
        assert "temperature" in scada_mapped_df.columns
        assert "label" in scada_mapped_df.columns
    
    def test_scada_raw_copy_preserved(self):
        """SCADA raw copy is saved to data/raw."""
        raw_copy = Path("./data/raw/scada_raw.parquet")
        assert raw_copy.exists(), "SCADA raw copy not preserved in data/raw"
        raw_df = pd.read_parquet(raw_copy)
        assert len(raw_df) == 1000


# =============================================================================
# 2.2 TELEMETRY SELECTION TESTS
# =============================================================================

class TestTelemetrySelection:
    """Test PIPER telemetry schema and column mapping."""
    
    def test_primary_telemetry_constants(self):
        """Primary telemetry constants are defined correctly (adapted scope: 3 variables)."""
        assert PRIMARY_TELEMETRY == [
            "pressure", "flow_rate", "temperature"
        ]
        assert len(PRIMARY_TELEMETRY) == 3
        assert "vibration" not in PRIMARY_TELEMETRY
    
    def test_metadata_constants(self):
        """Metadata constants are defined correctly."""
        assert METADATA_COLUMNS == ["pipeline_id", "sensor_id", "timestamp"]
        assert TARGET_COLUMN == "label"
    
    def test_required_columns(self):
        """REQUIRED_COLUMNS includes all mandatory fields (adapted scope)."""
        expected = PRIMARY_TELEMETRY + METADATA_COLUMNS + [TARGET_COLUMN]
        assert REQUIRED_COLUMNS == expected
        assert len(REQUIRED_COLUMNS) == 7  # 3 telemetry + 3 metadata + 1 target
        assert "vibration" not in REQUIRED_COLUMNS
    
    def test_apoerch_mapping_structure(self):
        """ApoErch mapping has correct structure."""
        assert isinstance(APOERCH_COLUMN_MAPPING, ColumnMapping)
        assert APOERCH_COLUMN_MAPPING.dataset_name == "ApoErch_LeakageData"
        assert "Meter_ID" in APOERCH_COLUMN_MAPPING.source_to_canonical
        assert "Timestamp" in APOERCH_COLUMN_MAPPING.source_to_canonical
    
    def test_apoerch_mapping_validation_fails_on_missing_telemetry(self):
        """Validation MUST fail for missing mandatory telemetry (ApoErch missing 3 of 3 + target)."""
        report = validate_mapping(APOERCH_COLUMN_MAPPING)
        
        # Check all mandatory telemetry are flagged as missing
        for pt in PRIMARY_TELEMETRY:
            assert pt in report["primary_telemetry"]
            assert report["primary_telemetry"][pt]["mapped"] == False
        
        # Check blockers list - should have 4 (3 telemetry + target)
        # ApoErch doesn't map target from source (synthetic labels generated later)
        assert len(report["blockers"]) == 4
        for pt in PRIMARY_TELEMETRY:
            assert any(pt in b for b in report["blockers"])
        # Target also missing
        assert any("target" in b.lower() or "label" in b.lower() for b in report["blockers"])
    
    def test_scada_mapping_structure(self):
        """SCADA mapping has correct structure."""
        assert isinstance(SCADA_COLUMN_MAPPING, ColumnMapping)
        assert SCADA_COLUMN_MAPPING.dataset_name == "SCADA_Pipeline"
        assert "timestamp" in SCADA_COLUMN_MAPPING.source_to_canonical
        assert "segment_id" in SCADA_COLUMN_MAPPING.source_to_canonical
        assert "pressure" in SCADA_COLUMN_MAPPING.source_to_canonical
        assert "flow_rate" in SCADA_COLUMN_MAPPING.source_to_canonical
        assert "temperature" in SCADA_COLUMN_MAPPING.source_to_canonical
        assert "target" in SCADA_COLUMN_MAPPING.source_to_canonical
    
    def test_scada_mapping_validation_passes(self):
        """SCADA validation should pass for required telemetry (3/3 present)."""
        report = validate_mapping(SCADA_COLUMN_MAPPING)
        
        # All 3 primary telemetry should be mapped
        for pt in PRIMARY_TELEMETRY:
            assert pt in report["primary_telemetry"]
            assert report["primary_telemetry"][pt]["mapped"] == True
            assert report["primary_telemetry"][pt]["status"] == "MAPPED"
        
        # Target should be mapped
        assert report["target"]["label"]["mapped"] == True
        
        # pipeline_id is optional metadata - not a blocker
        assert report["metadata"]["pipeline_id"]["mapped"] == False
        assert "OPTIONAL" in report["metadata"]["pipeline_id"]["status"]
        
        # No blockers for required telemetry/target
        assert len(report["blockers"]) == 0
        assert report["all_required_mapped"] == True
    
    def test_scada_has_all_three_telemetry(self, scada_mapped_df):
        """SCADA dataset has all 3 required primary telemetry."""
        for pt in PRIMARY_TELEMETRY:
            assert pt in scada_mapped_df.columns, f"{pt} missing from SCADA dataset"
    
    def test_scada_vibration_not_required(self, scada_mapped_df):
        """Vibration is not in SCADA dataset and not required for current scope."""
        assert "vibration" not in scada_mapped_df.columns
        assert "vibration" not in PRIMARY_TELEMETRY
    
    def test_column_mapping_application(self, raw_apoerch_df):
        """Column mapping correctly renames columns."""
        mapped = apply_column_mapping(raw_apoerch_df, APOERCH_COLUMN_MAPPING)
        assert "sensor_id" in mapped.columns
        assert "timestamp" in mapped.columns
        assert "Meter_ID" not in mapped.columns
        assert "Timestamp" not in mapped.columns
    
    def test_is_primary_telemetry_function(self):
        """is_primary_telemetry correctly identifies primary columns."""
        for pt in PRIMARY_TELEMETRY:
            assert is_primary_telemetry(pt)
        assert not is_primary_telemetry("vibration")
        assert not is_primary_telemetry("hour")
        assert not is_primary_telemetry("roll_24h_mean_flow_rate")
    
    def test_dataset_catalog(self):
        """Dataset catalog can register and validate datasets."""
        catalog = DatasetCatalog()
        df, info = create_apoerch_piper_dataset(save_raw_copy=False)
        catalog.register(info)
        
        assert "ApoErch_LeakageData" in catalog.list_datasets()
        assert len(catalog.get_piper_compatible()) == 0  # ApoErch not compatible
        
        # SCADA should be compatible
        df_scada, info_scada = load_or_create_scada_piper_dataset(force_reload=False)
        catalog.register(info_scada)
        assert "SCADA_Pipeline" in catalog.list_datasets()
        assert "SCADA_Pipeline" in catalog.get_piper_compatible()


# =============================================================================
# 2.3 DATA CLEANING TESTS
# =============================================================================

class TestDataCleaning:
    """Test data cleaning pipeline."""
    
    def test_cleaning_preserves_raw_data(self, mapped_apoerch_df):
        """Cleaning does not modify original dataframe."""
        original_len = len(mapped_apoerch_df)
        original_cols = set(mapped_apoerch_df.columns)
        
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(mapped_apoerch_df)
        
        # Original unchanged
        assert len(mapped_apoerch_df) == original_len
        assert set(mapped_apoerch_df.columns) == original_cols
    
    def test_cleaning_report_generated(self, mapped_apoerch_df):
        """Cleaning produces detailed report."""
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(mapped_apoerch_df)
        
        assert isinstance(report, object)
        assert report.dataset_name == "test"
        assert report.total_rows_raw == 270499
        assert len(report.steps) > 0
        assert report.end_time is not None
    
    def test_cleaning_no_rows_removed_by_default(self, mapped_apoerch_df):
        """Default cleaning preserves all rows (report only)."""
        clean_df, report = clean_apoerch_dataset(mapped_apoerch_df, save_report=False)
        assert len(clean_df) == 270499
        assert report.total_rows_raw - report.total_rows_clean == 0
    
    def test_timestamp_validation(self, mapped_apoerch_df):
        """Timestamp validation step works."""
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(
            mapped_apoerch_df,
            timestamp_format="%Y-%m-%d %H:%M:%S"
        )
        
        ts_step = next(s for s in report.steps if s.step_name == "validate_timestamps")
        assert ts_step.rows_affected == 0
        assert ts_step.details["parsing_failures"] == 0
    
    def test_duplicate_detection_apoerch(self, mapped_apoerch_df):
        """Duplicate detection works for ApoErch (no duplicates expected)."""
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(
            mapped_apoerch_df,
            timestamp_format="%Y-%m-%d %H:%M:%S"
        )
        
        dup_step = next(s for s in report.steps if s.step_name == "handle_duplicates")
        assert dup_step.details["exact_duplicate_rows_found"] == 0
        assert dup_step.details["conflicting_key_groups"] == 0
    
    def test_scada_duplicate_handling_preserves_conflicts(self, scada_mapped_df):
        """SCADA cleaning preserves conflicting duplicate keys with reading_index."""
        clean_df, report = clean_scada_dataset(scada_mapped_df, save_report=False)
        
        # Should NOT drop conflicting rows - all 1000 rows preserved
        assert len(clean_df) == 1000
        
        # Check duplicate step details
        dup_step = next(s for s in report.steps if s.step_name == "handle_duplicates")
        assert dup_step.details["exact_duplicate_rows_found"] == 0
        assert dup_step.details["conflicting_key_groups"] == 270  # 270 keys with conflicts
        assert dup_step.details["conflicting_rows_preserved"] == 408  # 1000 - 592 = 408 extra rows
        assert dup_step.details["reading_index_added"] == True
        
        # reading_index column should exist
        assert "reading_index" in clean_df.columns
    
    def test_scada_duplicate_reading_index_values(self, scada_cleaned_df):
        """Reading index should distinguish multiple readings per key."""
        # Check that reading_index values are 0, 1, 2... for each key
        max_idx = scada_cleaned_df.groupby(["sensor_id", "timestamp"])["reading_index"].max()
        # Some keys should have reading_index > 0 (multiple readings)
        assert (max_idx > 0).any()
    
    def test_missing_value_detection(self, mapped_apoerch_df):
        """Missing value detection works."""
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(
            mapped_apoerch_df,
            timestamp_format="%Y-%m-%d %H:%M:%S"
        )
        
        missing_step = next(s for s in report.steps if s.step_name == "detect_missing_values")
        assert missing_step.details["total_missing_values"] == 0
    
    def test_impossible_value_detection(self, mapped_apoerch_df):
        """Impossible value detection identifies out-of-bounds values."""
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(
            mapped_apoerch_df,
            timestamp_format="%Y-%m-%d %H:%M:%S"
        )
        
        impossible_step = next(s for s in report.steps if s.step_name == "detect_impossible_values")
        assert impossible_step.details["impossible_per_column"]["flow_rate"] == 52
        assert impossible_step.details["action"] == "REPORT_ONLY - abnormal values preserved for leak detection"
    
    def test_outlier_inspection(self, mapped_apoerch_df):
        """Outlier inspection reports outliers without removing."""
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(
            mapped_apoerch_df,
            timestamp_format="%Y-%m-%d %H:%M:%S"
        )
        
        outlier_step = next(s for s in report.steps if s.step_name == "inspect_outliers")
        assert outlier_step.details["outliers_per_column"]["flow_rate"] > 0
        assert "REPORT_ONLY" in outlier_step.details["action"]
    
    def test_sensor_consistency_check(self, mapped_apoerch_df):
        """Per-sensor consistency check reports gaps and completeness."""
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(
            mapped_apoerch_df,
            timestamp_format="%Y-%m-%d %H:%M:%S"
        )
        
        sensor_step = next(s for s in report.steps if s.step_name == "check_sensor_consistency")
        assert sensor_step.details["total_sensors"] == 10
        assert sensor_step.details["sensors_with_gaps"] == 10
        assert all(s["completeness"] < 1.0 for s in sensor_step.details["sensor_statistics"])
    
    def test_sampling_interval_check(self, mapped_apoerch_df):
        """Sampling interval consistency check works."""
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(
            mapped_apoerch_df,
            timestamp_format="%Y-%m-%d %H:%M:%S"
        )
        
        interval_step = next(s for s in report.steps if s.step_name == "check_sampling_interval")
        assert interval_step.details["overall_on_time_pct"] > 0
        assert interval_step.details["overall_on_time_pct"] < 100
    
    def test_data_integrity_validation(self, mapped_apoerch_df):
        """Data integrity validation identifies missing required columns."""
        cleaner = PiperCleaner("test")
        clean_df, report = cleaner.load(mapped_apoerch_df).run_full_pipeline(mapped_apoerch_df)
        
        integrity_step = next(s for s in report.steps if s.step_name == "validate_data_integrity")
        checks = integrity_step.details["checks"]
        assert checks["required_columns_present"] == False
        assert "pressure" in checks["missing_required_columns"]
        assert "temperature" in checks["missing_required_columns"]
        assert "label" in checks["missing_required_columns"]
        assert checks["no_duplicate_timestamps_per_sensor"] == True
        assert checks["timestamps_monotonic_per_sensor"] == True
    
    def test_cleaning_report_saved(self):
        """Cleaning report is saved to disk."""
        df, _ = create_apoerch_piper_dataset(save_raw_copy=False)
        clean_df, report = clean_apoerch_dataset(df, save_report=True)
        
        report_path = Path("./data/reports/cleaning_report_ApoErch_LeakageData.json")
        assert report_path.exists()
        
        with open(report_path) as f:
            saved = json.load(f)
        assert saved["dataset_name"] == "ApoErch_LeakageData"
        assert saved["total_rows_raw"] == 270499


# =============================================================================
# 2.4 FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    """Test feature engineering pipeline."""
    
    def test_apoerch_primary_telemetry_preserved(self, cleaned_apoerch_df):
        """ApoErch primary telemetry columns are preserved."""
        engineer = PiperFeatureEngineer("test")
        engineered_df, report = engineer.load(cleaned_apoerch_df).run_full_pipeline(cleaned_apoerch_df)
        
        assert "flow_rate" in engineered_df.columns
        assert report.primary_telemetry_present == ["flow_rate"]
        # vibration no longer in PRIMARY_TELEMETRY for adapted scope
        assert set(report.primary_telemetry_missing) == {"pressure", "temperature"}
    
    def test_scada_primary_telemetry_preserved(self, scada_cleaned_df):
        """SCADA primary telemetry columns are preserved (all 3 present)."""
        engineer = PiperFeatureEngineer("test")
        engineered_df, report = engineer.load(scada_cleaned_df).run_full_pipeline(
            scada_cleaned_df,
            add_rolling=False,  # Disabled for SCADA
            add_rate_of_change=False,
            add_relative=False
        )
        
        for pt in PRIMARY_TELEMETRY:
            assert pt in engineered_df.columns
        assert set(report.primary_telemetry_present) == set(PRIMARY_TELEMETRY)
        assert len(report.primary_telemetry_missing) == 0
    
    def test_temporal_features_created(self, scada_cleaned_df):
        """Temporal features are created from timestamp."""
        engineer = PiperFeatureEngineer("test")
        engineered_df, report = engineer.load(scada_cleaned_df).run_full_pipeline(
            scada_cleaned_df,
            add_rolling=False,
            add_rate_of_change=False,
            add_relative=False
        )
        
        temporal = ["hour", "dayofweek", "month", "quarter", "dayofyear", "weekofyear",
                   "is_weekend", "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek",
                   "sin_month", "cos_month"]
        
        for feat in temporal:
            assert feat in engineered_df.columns, f"Missing temporal feature: {feat}"
        
        assert len(report.temporal_features) == 13
    
    def test_scada_no_rolling_statistics(self, scada_engineered_df):
        """SCADA feature engineering does NOT create rolling statistics (disabled)."""
        rolling_feats = [c for c in scada_engineered_df.columns if c.startswith("roll_")]
        assert len(rolling_feats) == 0, f"Rolling features should not exist: {rolling_feats}"
    
    def test_scada_no_rate_of_change(self, scada_engineered_df):
        """SCADA feature engineering does NOT create rate-of-change features (disabled)."""
        diff_feats = [c for c in scada_engineered_df.columns if c.startswith("diff_")]
        assert len(diff_feats) == 0, f"Diff features should not exist: {diff_feats}"
    
    def test_scada_no_relative_features(self, scada_engineered_df):
        """SCADA feature engineering does NOT create relative features (disabled)."""
        rel_feats = [c for c in scada_engineered_df.columns if c.startswith("rel_")]
        assert len(rel_feats) == 0, f"Relative features should not exist: {rel_feats}"
    
    def test_scada_pressure_flow_ratio_created(self, scada_engineered_df):
        """SCADA creates pressure-flow ratio (current observation only)."""
        assert "flow_pressure_ratio" in scada_engineered_df.columns
        
        # Verify it's computed correctly: flow_rate / pressure
        expected = np.where(
            scada_engineered_df["pressure"] != 0,
            scada_engineered_df["flow_rate"] / scada_engineered_df["pressure"],
            0.0
        )
        np.testing.assert_allclose(scada_engineered_df["flow_pressure_ratio"], expected)
    
    def test_scada_no_pressure_drop_or_velocity(self, scada_engineered_df):
        """SCADA does NOT create pressure_drop_from_24h_mean or pressure_velocity."""
        assert "pressure_drop_from_24h_mean" not in scada_engineered_df.columns
        assert "pressure_velocity" not in scada_engineered_df.columns
    
    def test_metadata_features_created(self, scada_cleaned_df):
        """Metadata encoding features are created."""
        engineer = PiperFeatureEngineer("test")
        engineered_df, report = engineer.load(scada_cleaned_df).run_full_pipeline(
            scada_cleaned_df,
            add_rolling=False,
            add_rate_of_change=False,
            add_relative=False
        )
        
        assert "sensor_id_encoded" in engineered_df.columns
        assert "n_sensors" in engineered_df.columns
        assert engineered_df["n_sensors"].iloc[0] == 50
    
    def test_derived_features_distinguished_from_primary(self, scada_cleaned_df):
        """Derived features are clearly distinguished from primary telemetry."""
        engineer = PiperFeatureEngineer("test")
        engineered_df, report = engineer.load(scada_cleaned_df).run_full_pipeline(
            scada_cleaned_df,
            add_rolling=False,
            add_rate_of_change=False,
            add_relative=False
        )
        
        for pt in PRIMARY_TELEMETRY:
            assert pt in report.primary_telemetry_present
        
        # Derived features should NOT include primary telemetry
        for feat in report.derived_features:
            assert feat not in PRIMARY_TELEMETRY, f"Primary telemetry {feat} should not be in derived"
    
    def test_feature_report_saved(self):
        """Feature engineering report is saved."""
        df, _ = create_apoerch_piper_dataset(save_raw_copy=False)
        clean_df, _ = clean_apoerch_dataset(df, save_report=False)
        engineered_df, report = engineer_apoerch_features(clean_df, save_report=True)
        
        report_path = Path("./data/reports/feature_report_ApoErch_LeakageData.json")
        assert report_path.exists()
        
        with open(report_path) as f:
            saved = json.load(f)
        assert saved["dataset_name"] == "ApoErch_LeakageData"
        assert len(saved["output_columns"]) > len(saved["input_columns"])


# =============================================================================
# 2.5 LABEL GENERATION TESTS
# =============================================================================

class TestLabelGeneration:
    """Test label generation interface."""
    
    def test_label_source_types(self):
        """Label source types are defined."""
        assert LabelSourceType.REAL_GROUND_TRUTH.value == "real_ground_truth"
        assert LabelSourceType.SYNTHETIC_INJECTED.value == "synthetic_injected"
    
    def test_synthetic_label_generation(self, engineered_apoerch_df):
        """Synthetic label generation works and is marked as synthetic."""
        generator = PiperLabelGenerator("test")
        labeled_df, report = (
            generator.load(engineered_apoerch_df)
            .generate_synthetic_leak_segments()
            .finalize()
        )
        
        assert TARGET_COLUMN in labeled_df.columns
        assert set(labeled_df[TARGET_COLUMN].unique()) == {0, 1}
        assert report.is_synthetic == True
        assert report.is_real == False
        assert report.label_source == LabelSourceType.SYNTHETIC_INJECTED
        assert report.n_normal + report.n_abnormal == len(labeled_df)
    
    def test_synthetic_labels_warning(self, engineered_apoerch_df):
        """Synthetic labels have explicit warnings."""
        generator = PiperLabelGenerator("test")
        labeled_df, report = (
            generator.load(engineered_apoerch_df)
            .generate_synthetic_leak_segments()
            .finalize()
        )
        
        assert len(report.warnings) > 0
        assert any("SYNTHETIC" in w for w in report.warnings)
        assert any("development" in w.lower() for w in report.warnings)
    
    def test_scada_real_labels(self, scada_engineered_df):
        """SCADA real label mapping works correctly."""
        generator = PiperLabelGenerator("test")
        labeled_df, report = (
            generator.load(scada_engineered_df)
            .map_existing_labels(
                source_column="label",
                mapping={0: 0, 1: 1},
                label_source=LabelSourceType.REAL_GROUND_TRUTH,
                label_method=LabelMethod.EXISTING_COLUMN,
                description="Test SCADA label mapping"
            )
            .finalize()
        )
        
        assert report.is_real == True
        assert report.is_synthetic == False
        assert report.label_source == LabelSourceType.REAL_GROUND_TRUTH
        assert set(labeled_df[TARGET_COLUMN].unique()) == {0, 1}
    
    def test_scada_label_distribution(self, scada_labeled_df):
        """SCADA labeled dataset has correct class distribution (1000 rows, all preserved)."""
        assert TARGET_COLUMN in scada_labeled_df.columns
        counts = scada_labeled_df[TARGET_COLUMN].value_counts()
        assert counts[0] == 694  # normal (from raw 1000 rows)
        assert counts[1] == 306  # abnormal (from raw 1000 rows)
        assert counts[1] / len(scada_labeled_df) == pytest.approx(0.306, abs=0.01)
    
    def test_real_label_mapping_fails_on_unmapped(self):
        """Real label mapping fails if values not in mapping."""
        df = pd.DataFrame({
            "sensor_id": ["s1"],
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="h"),
            "flow_rate": [10],
            "pressure": [50],
            "temperature": [20],
            "label": [99]  # unmapped value
        })
        
        generator = PiperLabelGenerator("test")
        with pytest.raises(ValueError, match="Unmapped values"):
            generator.load(df).map_existing_labels(
                source_column="label",
                mapping={0: 0, 1: 1}
            ).finalize()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhase2Integration:
    """End-to-end Phase 2 pipeline tests."""
    
    def test_apoerch_full_pipeline_runs(self):
        """Complete ApoErch Phase 2 pipeline runs without errors."""
        df, _ = create_apoerch_piper_dataset(save_raw_copy=False)
        clean_df, clean_report = clean_apoerch_dataset(df, save_report=False)
        engineered_df, feat_report = engineer_apoerch_features(clean_df, save_report=False)
        labeled_df, label_report = label_apoerch_synthetic(engineered_df, save_report=False)
        
        assert len(labeled_df) == 270499
        assert TARGET_COLUMN in labeled_df.columns
    
    def test_scada_full_pipeline_runs(self, scada_labeled_df):
        """Complete SCADA Phase 2 pipeline runs without errors."""
        assert len(scada_labeled_df) == 1000  # All rows preserved
        assert TARGET_COLUMN in scada_labeled_df.columns
        assert set(scada_labeled_df[TARGET_COLUMN].unique()) == {0, 1}
        
        # All 3 primary telemetry present
        for pt in PRIMARY_TELEMETRY:
            assert pt in scada_labeled_df.columns
        
        # Vibration not required
        assert "vibration" not in scada_labeled_df.columns
        
        # Temporal features present
        assert "hour" in scada_labeled_df.columns
        assert "flow_pressure_ratio" in scada_labeled_df.columns
        
        # No long-window features
        for prefix in ["roll_", "diff_", "rel_"]:
            matching = [c for c in scada_labeled_df.columns if c.startswith(prefix)]
            assert len(matching) == 0, f"Long-window features should not exist: {matching}"
    
    def test_raw_data_preserved_through_pipeline(self):
        """Raw data is preserved at each stage."""
        raw_path = Path("./data/raw/apoerch_raw.parquet")
        assert raw_path.exists()
        
        raw_df = pd.read_parquet(raw_path)
        assert len(raw_df) == 270499
        assert list(raw_df.columns) == ["Meter_ID", "Timestamp", "Volume", "Unnamed: 3", "Unnamed: 4"]
        
        scada_raw_path = Path("./data/raw/scada_raw.parquet")
        assert scada_raw_path.exists()
        scada_raw = pd.read_parquet(scada_raw_path)
        assert len(scada_raw) == 1000
    
    def test_all_reports_exist(self):
        """All Phase 2 reports are generated."""
        reports_dir = Path("./data/reports")
        expected_json = [
            "cleaning_report_ApoErch_LeakageData.json",
            "feature_report_ApoErch_LeakageData.json",
            "label_report_ApoErch_LeakageData.json",
            "cleaning_report_SCADA_Pipeline.json",
            "feature_report_SCADA_Pipeline.json",
            "label_report_SCADA_Pipeline.json",
            "scada_temporal_feasibility.json",
        ]
        expected_md = [
            "phase2_summary.md",
        ]
        
        for report_file in expected_json:
            path = reports_dir / report_file
            assert path.exists(), f"Missing report: {report_file}"
            
            with open(path) as f:
                data = json.load(f)
            assert "dataset_name" in data
        
        for report_file in expected_md:
            path = reports_dir / report_file
            assert path.exists(), f"Missing report: {report_file}"
            # Markdown files don't need JSON parsing
            content = path.read_text()
            assert len(content) > 0
            assert "PIPER Phase 2" in content or "Phase 2" in content
    
    def test_missing_telemetry_blocker_documented_apoerch(self):
        """ApoErch missing mandatory telemetry is clearly documented as BLOCKER."""
        report = validate_mapping(APOERCH_COLUMN_MAPPING)
        
        for pt in PRIMARY_TELEMETRY:
            assert not report["primary_telemetry"][pt]["mapped"]
        
        # ApoErch has 3 missing telemetry + missing target = 4 blockers
        assert len(report["blockers"]) == 4
    
    def test_scada_no_telemetry_blockers(self):
        """SCADA has no telemetry blockers for adapted scope."""
        report = validate_mapping(SCADA_COLUMN_MAPPING)
        assert len(report["blockers"]) == 0
        assert report["all_required_mapped"] == True
    
    def test_synthetic_labels_explicitly_marked(self):
        """Synthetic labels are explicitly marked, never hidden."""
        df, _ = create_apoerch_piper_dataset(save_raw_copy=False)
        clean_df, _ = clean_apoerch_dataset(df, save_report=False)
        engineered_df, _ = engineer_apoerch_features(clean_df, save_report=False)
        labeled_df, report = label_apoerch_synthetic(engineered_df, save_report=False)
        
        assert report.is_synthetic
        assert not report.is_real
        
        report_path = Path("./data/reports/label_report_ApoErch_LeakageData.json")
        with open(report_path) as f:
            saved = json.load(f)
        assert saved["is_synthetic"] == True
    
    def test_scada_labels_are_real_not_synthetic(self, scada_labeled_df):
        """SCADA labels are real ground truth, not synthetic."""
        report_path = Path("./data/reports/label_report_SCADA_Pipeline.json")
        with open(report_path) as f:
            saved = json.load(f)
        assert saved["is_synthetic"] == False
        assert saved["label_source"] == "real_ground_truth"
    
    def test_no_vibration_column_or_fabricated_feature(self, scada_labeled_df):
        """No vibration column or fabricated vibration feature exists."""
        feat_cols = scada_labeled_df.columns.tolist()
        vibration_cols = [c for c in feat_cols if 'vibration' in c.lower()]
        assert len(vibration_cols) == 0, f"Vibration columns found: {vibration_cols}"
    
    def test_target_leakage_prevention(self, scada_labeled_df):
        """Target and unsafe metadata are excluded from predictive features."""
        # Check feature manifest
        manifest_path = Path("./data/processed/scada_feature_manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        excluded = [c for c, info in manifest["features"].items() 
                   if info["excluded_from_predictive_features"]]
        
        assert "label" in excluded
        assert "event_type" in excluded
        assert "alarm_triggered" in excluded
        
        # Verify reasons
        assert manifest["features"]["label"]["exclusion_reason"] == "Target variable - not a predictive feature"
        assert manifest["features"]["event_type"]["exclusion_reason"] == "Post-event information - would cause target leakage"
        assert manifest["features"]["alarm_triggered"]["exclusion_reason"] == "Directly encodes abnormal condition - target leakage risk"


# =============================================================================
# REGRESSION TESTS (ensure existing functionality not broken)
# =============================================================================

class TestRegression:
    """Ensure Phase 2 doesn't break existing ApoErch functionality."""
    
    def test_legacy_constants_accessible(self):
        """Legacy constants still accessible."""
        from utils.constants import (
            METER_ID_COL, TIMESTAMP_COL, VOLUME_COL, VOLUME_READING_COL,
            DELTA_VOLUME_COL, LEAK_FLAG_COL, CLS_FEATURE_COLS
        )
        
        assert METER_ID_COL == "Meter_ID"
        assert TIMESTAMP_COL == "Timestamp"
        assert VOLUME_COL == "Volume"
        assert LEAK_FLAG_COL == "Leak_Flag"
    
    def test_legacy_data_aggregation_importable(self):
        """Legacy DataAggregation module still importable."""
        from utils.DataAggregation import build_aggregated_dataset
        assert callable(build_aggregated_dataset)
    
    def test_legacy_algorithms_importable(self):
        """Legacy algorithm modules still importable."""
        from Algorithms import Gru, Lstm, XGBoost
        assert hasattr(Gru, "train_and_eval_gru")
        assert hasattr(Lstm, "train_and_eval_lstm")
        assert hasattr(XGBoost, "train_and_eval_xgboost")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])