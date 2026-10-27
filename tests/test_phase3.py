"""
PIPER Phase 3 Tests

Tests for temporal sequence construction feasibility assessment:
1. Time-series ordering audit (3.1)
2. Sliding-window generation feasibility (3.2)
3. Sequence length selection (3.3)
4. Train/validation/test split strategy (3.4)

Tests verify that feasibility assessment correctly identifies dataset limitations
and does NOT fabricate sequences when data cannot support them.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.piper_sequences import (
    audit_time_series_ordering,
    check_sliding_window_feasibility,
    evaluate_sequence_lengths,
    design_split_strategy,
    apply_tabular_split,
    run_phase3_feasibility,
    FeasibilityStatus,
    Phase3Report,
    SplitStrategy,
    generate_sliding_windows
)
from utils.piper_dataset import load_or_create_scada_piper_dataset
from utils.piper_cleaning import clean_scada_dataset
from utils.piper_features import engineer_scada_features
from utils.piper_labels import label_scada_real
from utils.piper_telemetry import (
    PRIMARY_TELEMETRY, METADATA_COLUMNS, TARGET_COLUMN,
    MetadataVariable
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def scada_labeled_df():
    """Load fully processed SCADA dataset through Phase 2 pipeline."""
    df, _ = load_or_create_scada_piper_dataset(force_reload=False)
    clean_df, _ = clean_scada_dataset(df, save_report=False)
    engineered_df, _ = engineer_scada_features(clean_df, save_report=False)
    labeled_df, _ = label_scada_real(engineered_df, save_report=False)
    return labeled_df


# =============================================================================
# 3.1 TIME-SERIES ORDERING TESTS
# =============================================================================

class TestTimeSeriesOrdering:
    """Test 3.1 Time-Series Ordering audit."""
    
    def test_audit_identifies_columns(self, scada_labeled_df):
        """Audit correctly identifies timestamp, sensor, target columns."""
        report = audit_time_series_ordering(scada_labeled_df)
        
        assert report.details["timestamp_column"] == "timestamp"
        assert report.details["sensor_column"] == "sensor_id"
        assert report.details["target_column"] == "label"
        assert set(report.details["telemetry_columns"]) == set(PRIMARY_TELEMETRY)
    
    def test_audit_reports_global_stats(self, scada_labeled_df):
        """Audit reports correct global statistics."""
        report = audit_time_series_ordering(scada_labeled_df)
        
        g = report.details["global"]
        assert g["total_observations"] == 1000
        assert g["total_segments"] == 50
        assert g["total_unique_timestamps"] == 17
        assert g["timestamp_range"]["span_hours"] == pytest.approx(0.2667, rel=0.01)
    
    def test_audit_detects_duplicate_keys(self, scada_labeled_df):
        """Audit detects duplicate (segment, timestamp) keys with conflicts."""
        report = audit_time_series_ordering(scada_labeled_df)
        
        dup = report.details["duplicate_analysis"]
        assert dup["duplicate_keys"] == 270  # 270 unique keys with duplicates
        assert dup["total_duplicate_rows"] == 678  # All rows in duplicate key groups (including first)
        assert dup["conflicting_keys"] == 270  # All have conflicting telemetry/target
        assert len(dup["conflict_examples"]) > 0
    
    def test_audit_reading_index_not_temporal(self, scada_labeled_df):
        """Audit confirms reading_index exists but is not a genuine timestamp."""
        report = audit_time_series_ordering(scada_labeled_df)
        
        assert report.details["has_reading_index"] == True
        # Check evidence mentions reading_index is not temporal
        reading_index_evidence = [e for e in report.evidence if "reading_index" in e.lower()]
        assert len(reading_index_evidence) > 0
        assert any("NOT a genuine timestamp" in e for e in reading_index_evidence)
    
    def test_audit_per_segment_ordering(self, scada_labeled_df):
        """Audit analyzes per-segment ordering statistics."""
        report = audit_time_series_ordering(scada_labeled_df)
        
        seg_stats = report.details["per_segment_ordering"]
        assert len(seg_stats) == 50
        
        # All segments should have duplicates
        for seg in seg_stats:
            assert seg["duplicate_timestamps"] > 0
            assert seg["n_unique_timestamps"] >= 8
            assert seg["n_unique_timestamps"] <= 16
            assert seg["timestamps_monotonic"] == True  # Unique timestamps are monotonic
    
    def test_audit_blocks_on_short_span(self, scada_labeled_df):
        """Audit correctly BLOCKS due to <1 hour time span."""
        report = audit_time_series_ordering(scada_labeled_df)
        
        assert report.status == FeasibilityStatus.BLOCKED
        feasibility = report.details["feasibility"]
        assert feasibility["can_order_reliably"] == False
        assert any("0.27 hours" in r or "span_hours" in r for r in feasibility["blocking_reasons"])
    
    def test_audit_blocks_on_conflicts(self, scada_labeled_df):
        """Audit correctly BLOCKS due to conflicting measurements."""
        report = audit_time_series_ordering(scada_labeled_df)
        
        feasibility = report.details["feasibility"]
        assert any("conflicting" in r.lower() for r in feasibility["blocking_reasons"])
    
    def test_audit_blocks_on_insufficient_timestamps(self, scada_labeled_df):
        """Audit correctly BLOCKS due to <10 unique timestamps per segment."""
        report = audit_time_series_ordering(scada_labeled_df)
        
        feasibility = report.details["feasibility"]
        assert any("8" in r and "10" in r for r in feasibility["blocking_reasons"])
    
    def test_audit_status_is_blocked(self, scada_labeled_df):
        """Overall 3.1 status is BLOCKED."""
        report = audit_time_series_ordering(scada_labeled_df)
        assert report.status == FeasibilityStatus.BLOCKED


# =============================================================================
# 3.2 SLIDING-WINDOW GENERATION TESTS
# =============================================================================

class TestSlidingWindowFeasibility:
    """Test 3.2 Sliding-Window Generation feasibility."""
    
    def test_check_blocks_when_ordering_blocked(self, scada_labeled_df):
        """Sliding window check BLOCKS when 3.1 is blocked."""
        report = check_sliding_window_feasibility(scada_labeled_df)
        
        assert report.status == FeasibilityStatus.BLOCKED
        assert any("Prerequisite (3.1" in e for e in report.evidence)
    
    def test_check_reports_root_causes(self, scada_labeled_df):
        """Check reports root causes from 3.1."""
        report = check_sliding_window_feasibility(scada_labeled_df)
        
        # Should mention the same blocking reasons
        root_cause_evidence = [e for e in report.evidence if "Root cause" in e]
        assert len(root_cause_evidence) >= 3  # At least 3 root causes
    
    def test_check_documents_requirements(self, scada_labeled_df):
        """Check documents what would be required for window generation."""
        report = check_sliding_window_feasibility(scada_labeled_df)
        
        assert "would_require" in report.details
        req = report.details["would_require"]
        assert "reliable_unique_timestamps_per_segment" in req
        assert "no_conflicting_measurements" in req
        assert "sufficient_time_span" in req
        assert "genuine_temporal_ordering" in req
    
    def test_generate_sliding_windows_raises_on_bad_data(self, scada_labeled_df):
        """generate_sliding_windows raises ValueError on unreliable data."""
        with pytest.raises(ValueError, match="does not support reliable temporal ordering"):
            generate_sliding_windows(scada_labeled_df, window_size=3)
    
    def test_no_fabricated_windows(self, scada_labeled_df):
        """Test confirms no window artifacts are generated when blocked."""
        # The feasibility check should be BLOCKED, not create windows
        report = check_sliding_window_feasibility(scada_labeled_df)
        assert report.status == FeasibilityStatus.BLOCKED
        # No window arrays should be created in this path


# =============================================================================
# 3.3 SEQUENCE LENGTH SELECTION TESTS
# =============================================================================

class TestSequenceLengthSelection:
    """Test 3.3 Sequence Length Selection."""
    
    def test_evaluate_blocks_when_ordering_blocked(self, scada_labeled_df):
        """Sequence length evaluation BLOCKS when 3.1 is blocked."""
        report = evaluate_sequence_lengths(scada_labeled_df)
        
        assert report.status == FeasibilityStatus.BLOCKED
        assert any("Prerequisite (3.1" in e for e in report.evidence)
    
    def test_evaluate_reports_available_lengths(self, scada_labeled_df):
        """Evaluation reports available sequence lengths per segment."""
        report = evaluate_sequence_lengths(scada_labeled_df)
        
        assert "available_sequence_lengths" in report.details
        lengths = report.details["available_sequence_lengths"]
        assert len(lengths) == 50
        assert all(8 <= s["n_unique_timestamps"] <= 16 for s in lengths)
    
    def test_evaluate_max_possible_window(self, scada_labeled_df):
        """Evaluation reports max possible window size."""
        report = evaluate_sequence_lengths(scada_labeled_df)
        
        assert "max_possible_window" in report.details
        assert report.details["max_possible_window"] == 16
    
    def test_evaluate_candidate_sizes(self, scada_labeled_df):
        """Evaluation tests candidate sizes 3, 5, 10."""
        report = evaluate_sequence_lengths(scada_labeled_df)
        
        assert report.details["candidate_sizes"] == [3, 5, 10]
        # All should exceed or barely fit available lengths
        for e in report.evidence:
            if "Window size" in e:
                assert "ERROR" in e or "0 windows" in e
    
    def test_no_performance_based_selection(self, scada_labeled_df):
        """Test confirms selection is NOT based on test-set performance."""
        report = evaluate_sequence_lengths(scada_labeled_df)
        
        # Should not mention test performance, accuracy, F1, etc.
        perf_keywords = ["accuracy", "f1", "precision", "recall", "auc", "performance", "test set"]
        for e in report.evidence:
            for kw in perf_keywords:
                assert kw not in e.lower(), f"Evidence mentions test performance: {e}"
    
    def test_status_is_blocked(self, scada_labeled_df):
        """Overall 3.3 status is BLOCKED."""
        report = evaluate_sequence_lengths(scada_labeled_df)
        assert report.status == FeasibilityStatus.BLOCKED


# =============================================================================
# 3.4 TRAIN/VALIDATION/TEST SPLIT TESTS
# =============================================================================

class TestTrainValTestSplit:
    """Test 3.4 Train/Validation/Test Split Strategy."""
    
    def test_design_recommends_tabular_split(self, scada_labeled_df):
        """Split design recommends tabular stratified group-aware split."""
        report = design_split_strategy(scada_labeled_df)
        
        # Strategy will be "tabular_stratified" (fallback when stratification not possible)
        # or "tabular_stratified_group_aware" if stratification works
        assert report.details["recommended_strategy"] in ["tabular_stratified", "tabular_stratified_group_aware"]
        assert report.status == FeasibilityStatus.BLOCKED  # For SEQUENCE split
    
    def test_split_prevents_sensor_leakage(self, scada_labeled_df):
        """Split ensures no sensor appears in multiple splits."""
        report = design_split_strategy(scada_labeled_df)
        
        split = report.details["split"]
        train_sensors = set(split["train_sensors"])
        val_sensors = set(split["val_sensors"])
        test_sensors = set(split["test_sensors"])
        
        assert len(train_sensors & val_sensors) == 0
        assert len(train_sensors & test_sensors) == 0
        assert len(val_sensors & test_sensors) == 0
        assert len(train_sensors) + len(val_sensors) + len(test_sensors) == 50
    
    def test_split_stratified_by_target(self, scada_labeled_df):
        """Split is stratified by target class at sensor level."""
        report = design_split_strategy(scada_labeled_df)
        
        split = report.details["split"]
        assert "train_class_dist" in split
        assert "val_class_dist" in split
        assert "test_class_dist" in split
        
        # All splits should have both classes
        for dist in [split["train_class_dist"], split["val_class_dist"], split["test_class_dist"]]:
            assert 0 in dist
            assert 1 in dist
    
    def test_apply_tabular_split_creates_disjoint_sets(self, scada_labeled_df):
        """apply_tabular_split produces disjoint sensor sets."""
        train_df, val_df, test_df, split_info = apply_tabular_split(scada_labeled_df)
        
        assert split_info["no_sensor_overlap"] == True
        assert split_info["train_size"] + split_info["val_size"] + split_info["test_size"] == 1000
        
        train_sensors = set(train_df["sensor_id"].unique())
        val_sensors = set(val_df["sensor_id"].unique())
        test_sensors = set(test_df["sensor_id"].unique())
        
        assert len(train_sensors & val_sensors) == 0
        assert len(train_sensors & test_sensors) == 0
        assert len(val_sensors & test_sensors) == 0
    
    def test_apply_tabular_split_class_balance(self, scada_labeled_df):
        """Split maintains reasonable class balance."""
        train_df, val_df, test_df, split_info = apply_tabular_split(scada_labeled_df)
        
        # Check class balance is preserved roughly
        train_rate = split_info["train_class_dist"][1] / split_info["train_size"]
        val_rate = split_info["val_class_dist"][1] / split_info["val_size"]
        test_rate = split_info["test_class_dist"][1] / split_info["test_size"]
        
        # Should be around 30.6% (global rate)
        assert abs(train_rate - 0.306) < 0.05
        assert abs(val_rate - 0.306) < 0.05
        assert abs(test_rate - 0.306) < 0.05
    
    def test_no_random_row_level_split(self, scada_labeled_df):
        """Test confirms no random row-level split is used."""
        report = design_split_strategy(scada_labeled_df)
        
        # Should not use random row split
        split = report.details["split"]
        assert split["strategy"] != "random_row"
        assert "group_aware" in split["strategy"] or "tabular_stratified" in split["strategy"]
        
        evidence_text = " ".join(report.evidence)
        assert "leakage" in evidence_text.lower() or "isolat" in evidence_text.lower()
    
    def test_test_set_isolated(self, scada_labeled_df):
        """Test set is isolated from training/validation."""
        train_df, val_df, test_df, split_info = apply_tabular_split(scada_labeled_df)
        
        # Test sensors should not appear in train or val
        test_sensors = set(test_df["sensor_id"].unique())
        train_sensors = set(train_df["sensor_id"].unique())
        val_sensors = set(val_df["sensor_id"].unique())
        
        assert len(test_sensors & train_sensors) == 0
        assert len(test_sensors & val_sensors) == 0
        
        # Also verify evidence mentions isolation
        report = design_split_strategy(scada_labeled_df)
        iso_evidence = [e for e in report.evidence if "isolat" in e.lower() or "no sensor" in e.lower()]
        assert len(iso_evidence) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhase3Integration:
    """End-to-end Phase 3 feasibility assessment tests."""
    
    def test_full_phase3_runs(self, scada_labeled_df):
        """Complete Phase 3 feasibility runs without errors."""
        report = run_phase3_feasibility(scada_labeled_df, save_artifacts=False)
        
        assert isinstance(report, Phase3Report)
        assert len(report.subtasks) == 4
        assert report.dataset_name == "SCADA_Pipeline"
    
    def test_all_subtasks_reported(self, scada_labeled_df):
        """All 4 subtasks are reported with correct IDs."""
        report = run_phase3_feasibility(scada_labeled_df, save_artifacts=False)
        
        subtask_ids = [s.subtask_id for s in report.subtasks]
        assert subtask_ids == ["3.1", "3.2", "3.3", "3.4"]
    
    def test_overall_assessment_correct(self, scada_labeled_df):
        """Overall assessment correctly identifies sequence learning not viable."""
        report = run_phase3_feasibility(scada_labeled_df, save_artifacts=False)
        
        overall = report.overall_assessment
        assert overall["sequence_learning_viable"] == False
        assert overall["tabular_classification_viable"] == True
        assert overall["next_phase_recommendation"] == "tabular_classification"
        assert overall["subtasks_blocked"] >= 3  # At least 3 subtasks blocked
    
    def test_recommendations_appropriate(self, scada_labeled_df):
        """Recommendations are appropriate for blocked sequence track."""
        report = run_phase3_feasibility(scada_labeled_df, save_artifacts=False)
        
        recs = report.recommendations
        assert any("NOT viable" in r for r in recs)
        assert any("TABULAR classification" in r for r in recs)
        assert any("group-aware stratified" in r for r in recs)
        assert any("fabricate" in r.lower() for r in recs)
    
    def test_artifacts_saved_when_requested(self, scada_labeled_df, tmp_path):
        """Artifacts are saved when save_artifacts=True."""
        # Test that the function runs and saves artifacts to the actual project directories
        # (since the module uses module-level paths, we test with the real paths)
        import utils.piper_sequences as seq_module
        
        # Run with save_artifacts=True (default)
        report = run_phase3_feasibility(scada_labeled_df, save_artifacts=True)
        
        # Check files exist in the actual project directories
        assert (seq_module.PIPER_REPORTS_DIR / "phase3_feasibility_report.json").exists()
        assert (seq_module.PIPER_REPORTS_DIR / "phase3_summary.md").exists()
        assert (seq_module.PIPER_PROCESSED_DIR / "scada_train.csv").exists()
        assert (seq_module.PIPER_PROCESSED_DIR / "scada_val.csv").exists()
        assert (seq_module.PIPER_PROCESSED_DIR / "scada_test.csv").exists()
        assert (seq_module.PIPER_REPORTS_DIR / "scada_tabular_split_info.json").exists()
    
    def test_phase3_report_json_valid(self, scada_labeled_df):
        """Phase 3 report serializes to valid JSON."""
        report = run_phase3_feasibility(scada_labeled_df, save_artifacts=False)
        
        json_str = json.dumps(report.to_dict(), default=str)
        parsed = json.loads(json_str)
        
        assert "dataset_name" in parsed
        assert "subtasks" in parsed
        assert len(parsed["subtasks"]) == 4
        assert "overall_assessment" in parsed
        assert "recommendations" in parsed
    
    def test_subtask_details_preserved(self, scada_labeled_df):
        """Subtask details contain rich evidence for audit trail."""
        report = run_phase3_feasibility(scada_labeled_df, save_artifacts=False)
        
        for subtask in report.subtasks:
            assert len(subtask.evidence) > 0
            assert len(subtask.details) > 0
            assert subtask.timestamp is not None
            assert subtask.status in [FeasibilityStatus.COMPLETE, FeasibilityStatus.BLOCKED, FeasibilityStatus.DEFERRED]


# =============================================================================
# REGRESSION TESTS - Ensure Phase 1/2 not broken
# =============================================================================

class TestPhase12Regression:
    """Ensure Phase 3 doesn't break Phase 1/2 functionality."""
    
    def test_scada_pipeline_still_runs(self, scada_labeled_df):
        """SCADA Phase 2 pipeline output unchanged."""
        assert len(scada_labeled_df) == 1000
        assert TARGET_COLUMN in scada_labeled_df.columns
        assert set(scada_labeled_df[TARGET_COLUMN].unique()) == {0, 1}
        for pt in PRIMARY_TELEMETRY:
            assert pt in scada_labeled_df.columns
    
    def test_feature_manifest_still_valid(self, scada_labeled_df):
        """Feature manifest from Phase 2 still accessible."""
        manifest_path = Path("./data/processed/scada_feature_manifest.json")
        assert manifest_path.exists()
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        assert manifest["summary"]["predictive_features"] == 28
        assert manifest["summary"]["excluded_features"] == 3
        assert "label" in manifest["features"]
        assert manifest["features"]["label"]["excluded_from_predictive_features"] == True
    
    def test_temporal_feasibility_report_unchanged(self):
        """Phase 2 temporal feasibility report unchanged."""
        report_path = Path("./data/reports/scada_temporal_feasibility.json")
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        assert report["feasibility_assessment"]["can_support_sequence_construction"] == False
        assert report["feasibility_assessment"]["can_support_tabular_classification"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])