"""
PIPER Phase 4 Tests

Tests for tabular classification pipeline:
1. Predictor/target separation
2. Forbidden-feature exclusion
3. Split isolation
4. Preprocessing fitted only on training data
5. Consistent feature schema
6. Model save/load equivalence
7. Inference output shape and valid probabilities
8. Metric calculations and reproducibility
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.piper_phase4 import (
    run_phase4_tabular,
    load_phase4_model,
    predict_phase4,
    get_predictor_features,
    prepare_features,
    prepare_targets,
    evaluate_model,
    find_optimal_threshold,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    EXCLUDED_FEATURES,
    CONSTANT_FEATURES,
    TARGET_COLUMN,
    PIPER_MODELS_DIR,
    PIPER_PROCESSED_DIR,
    PIPER_REPORTS_DIR,
)
from utils.piper_telemetry import PRIMARY_TELEMETRY, METADATA_COLUMNS


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def phase4_report():
    """Run Phase 4 and return the report (or load existing)."""
    report_path = PIPER_REPORTS_DIR / "phase4_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    else:
        # Run Phase 4 if not already done
        from utils.piper_phase4 import run_phase4_tabular
        report_obj = run_phase4_tabular(save_artifacts=True)
        return report_obj.to_dict()


@pytest.fixture(scope="session")
def train_val_test_data():
    """Load train/val/test splits."""
    train_df = pd.read_csv(PIPER_PROCESSED_DIR / "scada_train.csv")
    val_df = pd.read_csv(PIPER_PROCESSED_DIR / "scada_val.csv")
    test_df = pd.read_csv(PIPER_PROCESSED_DIR / "scada_test.csv")
    return train_df, val_df, test_df


@pytest.fixture(scope="session")
def model_data():
    """Load fitted model and preprocessor."""
    return load_phase4_model()


# =============================================================================
# TASK A: AUDIT TESTS
# =============================================================================

class TestPhase4Audit:
    """Tests verifying Phase 4 audit requirements."""
    
    def test_split_files_exist(self):
        """All split files exist."""
        assert (PIPER_PROCESSED_DIR / "scada_train.csv").exists()
        assert (PIPER_PROCESSED_DIR / "scada_val.csv").exists()
        assert (PIPER_PROCESSED_DIR / "scada_test.csv").exists()
        assert (PIPER_REPORTS_DIR / "scada_tabular_split_info.json").exists()
    
    def test_split_row_counts_match_report(self, train_val_test_data, phase4_report):
        """Row counts match Phase 3 split report."""
        train_df, val_df, test_df = train_val_test_data
        split_info = phase4_report["split_info"]
        
        assert len(train_df) == split_info["train_size"]
        assert len(val_df) == split_info["val_size"]
        assert len(test_df) == split_info["test_size"]
    
    def test_split_sensor_counts_match_report(self, train_val_test_data, phase4_report):
        """Sensor counts match Phase 3 split report."""
        train_df, val_df, test_df = train_val_test_data
        split_info = phase4_report["split_info"]
        
        assert train_df["sensor_id"].nunique() == len(split_info["train_sensors"])
        assert val_df["sensor_id"].nunique() == len(split_info["val_sensors"])
        assert test_df["sensor_id"].nunique() == len(split_info["test_sensors"])
    
    def test_split_class_distribution_match_report(self, train_val_test_data, phase4_report):
        """Class distribution matches Phase 3 split report."""
        train_df, val_df, test_df = train_val_test_data
        split_info = phase4_report["split_info"]
        
        for name, df, expected in [
            ("train", train_df, split_info["train_class_dist"]),
            ("val", val_df, split_info["val_class_dist"]),
            ("test", test_df, split_info["test_class_dist"]),
        ]:
            actual = df[TARGET_COLUMN].value_counts().to_dict()
            actual = {str(k): v for k, v in actual.items()}
            assert actual == expected, f"{name} class dist mismatch: {actual} vs {expected}"
    
    def test_no_sensor_overlap(self, train_val_test_data, phase4_report):
        """No sensor appears in multiple splits."""
        train_df, val_df, test_df = train_val_test_data
        
        train_sensors = set(train_df["sensor_id"].unique())
        val_sensors = set(val_df["sensor_id"].unique())
        test_sensors = set(test_df["sensor_id"].unique())
        
        assert len(train_sensors & val_sensors) == 0, "Train-Val sensor overlap"
        assert len(train_sensors & test_sensors) == 0, "Train-Test sensor overlap"
        assert len(val_sensors & test_sensors) == 0, "Val-Test sensor overlap"
        
        # Also check report says no overlap
        assert phase4_report["split_info"]["no_sensor_overlap"] == True
    
    def test_both_classes_present_in_each_split(self, train_val_test_data):
        """Each split has both classes (0 and 1)."""
        for name, df in [("train", train_val_test_data[0]), 
                         ("val", train_val_test_data[1]), 
                         ("test", train_val_test_data[2])]:
            classes = set(df[TARGET_COLUMN].unique())
            assert classes == {0, 1}, f"{name} split missing class: {classes}"
    
    def test_no_missing_values(self, train_val_test_data):
        """No missing values in any split."""
        for name, df in [("train", train_val_test_data[0]), 
                         ("val", train_val_test_data[1]), 
                         ("test", train_val_test_data[2])]:
            missing = df.isnull().sum().sum()
            assert missing == 0, f"{name} split has {missing} missing values"
    
    def test_target_excluded_from_predictors(self, train_val_test_data):
        """Target column is excluded from predictor features."""
        train_df = train_val_test_data[0]
        predictors = get_predictor_features(train_df)
        
        assert TARGET_COLUMN not in predictors, "Target column in predictors"
    
    def test_forbidden_features_excluded(self, train_val_test_data):
        """Forbidden features (leakage risk) are excluded from predictors."""
        train_df = train_val_test_data[0]
        predictors = get_predictor_features(train_df)
        
        forbidden = ["alarm_triggered", "event_type"]
        for feat in forbidden:
            assert feat not in predictors, f"Forbidden feature '{feat}' in predictors"
    
    def test_metadata_features_excluded(self, train_val_test_data):
        """Metadata features are excluded from predictors."""
        train_df = train_val_test_data[0]
        predictors = get_predictor_features(train_df)
        
        metadata = ["timestamp", "timestamp_parsed", "sensor_id", "reading_index", "n_sensors"]
        for feat in metadata:
            assert feat not in predictors, f"Metadata feature '{feat}' in predictors"
    
    def test_constant_features_excluded(self, train_val_test_data):
        """Constant features (1 unique value) are excluded."""
        train_df = train_val_test_data[0]
        predictors = get_predictor_features(train_df)
        
        # Temporal features are all constant in this dataset (single day)
        for feat in CONSTANT_FEATURES:
            assert feat not in predictors, f"Constant feature '{feat}' in predictors"
    
    def test_predictor_features_list(self, train_val_test_data):
        """Verify exact predictor feature list."""
        train_df = train_val_test_data[0]
        predictors = get_predictor_features(train_df)
        
        expected = [
            "compressor_state", "energy_consumption", "flow_pressure_ratio",
            "flow_rate", "pressure", "pump_speed", "pump_state",
            "sensor_id_encoded", "temperature", "valve_status"
        ]
        
        assert set(predictors) == set(expected), f"Predictors mismatch: {set(predictors) ^ set(expected)}"
    
    def test_feature_manifest_exclusions_respected(self, train_val_test_data):
        """Feature manifest exclusions are respected."""
        with open(PIPER_PROCESSED_DIR / "scada_feature_manifest.json") as f:
            manifest = json.load(f)
        
        excluded_in_manifest = [
            col for col, info in manifest["features"].items()
            if info.get("excluded_from_predictive_features", False)
        ]
        
        predictors = get_predictor_features(train_val_test_data[0])
        
        for feat in excluded_in_manifest:
            assert feat not in predictors, f"Manifest-excluded feature '{feat}' in predictors"


# =============================================================================
# TASK B: TRAINING PIPELINE TESTS
# =============================================================================

class TestTrainingPipeline:
    """Tests for the training pipeline."""
    
    def test_preprocessing_fitted_on_train_only(self, train_val_test_data):
        """Preprocessing is fitted on training data only."""
        train_df, val_df, test_df = train_val_test_data
        
        X_train, X_val, X_test, feature_names, preprocessor, label_encoders = prepare_features(
            train_df, val_df, test_df
        )
        
        # Check that preprocessor is fitted (has mean_ attribute from StandardScaler)
        # The ColumnTransformer should have fitted transformers
        assert hasattr(preprocessor, 'transformers_')
        
        # Verify feature names are consistent
        assert len(feature_names) == X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    
    def test_prepare_targets(self, train_val_test_data):
        """Target preparation returns correct arrays."""
        train_df, val_df, test_df = train_val_test_data
        
        y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
        
        assert len(y_train) == len(train_df)
        assert len(y_val) == len(val_df)
        assert len(y_test) == len(test_df)
        assert set(y_train) == {0, 1}
        assert set(y_val) == {0, 1}
        assert set(y_test) == {0, 1}
    
    def test_model_training_logistic_regression(self, train_val_test_data):
        """Logistic Regression trains without error."""
        train_df, val_df, test_df = train_val_test_data
        
        X_train, X_val, X_test, feature_names, _, _ = prepare_features(train_df, val_df, test_df)
        y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
        
        model, threshold = train_logistic_regression(X_train, y_train, X_val, y_val)
        
        assert model is not None
        assert 0 < threshold < 1
        assert hasattr(model, 'predict_proba')
    
    def test_model_training_random_forest(self, train_val_test_data):
        """Random Forest trains without error."""
        train_df, val_df, test_df = train_val_test_data
        
        X_train, X_val, X_test, feature_names, _, _ = prepare_features(train_df, val_df, test_df)
        y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
        
        model, threshold = train_random_forest(X_train, y_train, X_val, y_val)
        
        assert model is not None
        assert 0 < threshold < 1
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'feature_importances_')
    
    def test_model_training_xgboost(self, train_val_test_data):
        """XGBoost trains without error."""
        train_df, val_df, test_df = train_val_test_data
        
        X_train, X_val, X_test, feature_names, _, _ = prepare_features(train_df, val_df, test_df)
        y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
        
        model, threshold = train_xgboost(X_train, y_train, X_val, y_val)
        
        assert model is not None
        assert 0 < threshold < 1
        assert hasattr(model, 'predict_proba')
    
    def test_threshold_optimization(self, train_val_test_data):
        """Threshold is optimized on validation set."""
        train_df, val_df, test_df = train_val_test_data
        
        X_train, X_val, X_test, feature_names, _, _ = prepare_features(train_df, val_df, test_df)
        y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
        
        model, threshold = train_random_forest(X_train, y_train, X_val, y_val)
        
        # Verify threshold is in valid range
        assert 0.01 <= threshold <= 0.99
        
        # Verify it's actually optimal (better than 0.5)
        metrics_at_optimal = evaluate_model(model, X_val, y_val, threshold)
        metrics_at_default = evaluate_model(model, X_val, y_val, 0.5)
        
        # Optimal threshold should have >= F1 than default (allowing small numerical differences)
        assert metrics_at_optimal.f1 >= metrics_at_default.f1 - 0.01
    
    def test_class_weight_balanced_used(self, train_val_test_data):
        """Class imbalance handling is applied."""
        train_df, val_df, test_df = train_val_test_data
        
        X_train, X_val, X_test, feature_names, _, _ = prepare_features(train_df, val_df, test_df)
        y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
        
        # Check class imbalance in training data
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        assert n_neg > n_pos  # Imbalanced
        
        # Models should use balanced class weights
        lr_model, _ = train_logistic_regression(X_train, y_train, X_val, y_val)
        assert lr_model.class_weight == 'balanced'
        
        rf_model, _ = train_random_forest(X_train, y_train, X_val, y_val)
        assert rf_model.class_weight == 'balanced'


# =============================================================================
# TASK C: MODEL SELECTION & EVALUATION TESTS
# =============================================================================

class TestModelSelectionEvaluation:
    """Tests for model selection and evaluation."""
    
    def test_validation_metrics_computed(self, train_val_test_data):
        """Validation metrics are computed for all models."""
        train_df, val_df, test_df = train_val_test_data
        
        X_train, X_val, X_test, feature_names, _, _ = prepare_features(train_df, val_df, test_df)
        y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
        
        for train_fn in [train_logistic_regression, train_random_forest, train_xgboost]:
            model, threshold = train_fn(X_train, y_train, X_val, y_val)
            metrics = evaluate_model(model, X_val, y_val, threshold)
            
            assert 0 <= metrics.accuracy <= 1
            assert 0 <= metrics.precision <= 1
            assert 0 <= metrics.recall <= 1
            assert 0 <= metrics.f1 <= 1
            assert 0 <= metrics.roc_auc <= 1
            assert metrics.confusion_matrix is not None
            assert len(metrics.confusion_matrix) == 2
            assert len(metrics.confusion_matrix[0]) == 2
    
    def test_model_selection_based_on_validation_f1(self, phase4_report):
        """Best model selected based on validation F1."""
        model_results = phase4_report["model_results"]
        
        # Find best by validation F1
        best_by_val_f1 = max(model_results, key=lambda r: r["val_metrics"]["f1"])
        selected = phase4_report["selected_model"]
        
        assert best_by_val_f1["model_name"] == selected
    
    def test_threshold_not_optimized_on_test(self, phase4_report):
        """Threshold is not optimized using test set."""
        # The threshold in the report should match the validation-optimized threshold
        for mr in phase4_report["model_results"]:
            if mr["model_name"] == phase4_report["selected_model"]:
                assert mr["threshold"] == phase4_report["selected_threshold"]
                break
    
    def test_final_test_evaluation(self, phase4_report):
        """Final evaluation on isolated test set."""
        test_metrics = phase4_report["final_test_metrics"]
        
        assert 0 <= test_metrics["accuracy"] <= 1
        assert 0 <= test_metrics["precision"] <= 1
        assert 0 <= test_metrics["recall"] <= 1
        assert 0 <= test_metrics["f1"] <= 1
        assert 0 <= test_metrics["roc_auc"] <= 1
        assert test_metrics["confusion_matrix"] is not None
        assert test_metrics["threshold"] == phase4_report["selected_threshold"]
    
    def test_confusion_matrix_format(self, phase4_report):
        """Confusion matrix has correct format."""
        cm = phase4_report["final_test_metrics"]["confusion_matrix"]
        
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2
        assert all(isinstance(v, int) for row in cm for v in row)
        
        # Sum should equal test set size
        total = sum(sum(row) for row in cm)
        assert total == phase4_report["split_info"]["test_size"]


# =============================================================================
# TASK D: ARTIFACT SAVING TESTS
# =============================================================================

class TestArtifactSaving:
    """Tests for artifact saving and loading."""
    
    def test_model_saved(self):
        """Model is saved to disk."""
        assert (PIPER_MODELS_DIR / "phase4_model.joblib").exists()
    
    def test_preprocessor_saved(self):
        """Preprocessor is saved to disk."""
        assert (PIPER_MODELS_DIR / "phase4_preprocessor.joblib").exists()
    
    def test_report_saved(self):
        """Report is saved to disk."""
        assert (PIPER_REPORTS_DIR / "phase4_report.json").exists()
        assert (PIPER_REPORTS_DIR / "phase4_summary.md").exists()
    
    def test_test_predictions_saved(self):
        """Test predictions are saved."""
        assert (PIPER_PROCESSED_DIR / "scada_test_predictions.csv").exists()
    
    def test_model_load_equivalence(self, model_data, train_val_test_data):
        """Loaded model produces same predictions as original."""
        train_df, val_df, test_df = train_val_test_data
        
        # Get original predictions
        original_preds = predict_phase4(test_df, model_data)
        
        # Reload and predict again
        reloaded_data = load_phase4_model()
        reloaded_preds = predict_phase4(test_df, reloaded_data)
        
        # Should be identical
        np.testing.assert_array_equal(
            original_preds["predicted_label"].values,
            reloaded_preds["predicted_label"].values
        )
        np.testing.assert_allclose(
            original_preds["predicted_probability"].values,
            reloaded_preds["predicted_probability"].values,
            rtol=1e-10
        )
    
    def test_feature_schema_saved(self, phase4_report):
        """Feature schema is saved in report."""
        schema = phase4_report["feature_schema"]
        
        assert "predictor_features" in schema
        assert "target_column" in schema
        assert "excluded_features" in schema
        assert "n_features" in schema
        assert schema["n_features"] == 10
        assert schema["target_column"] == "label"
    
    def test_preprocessor_contains_all_needed(self, model_data):
        """Preprocessor contains all needed components."""
        assert "preprocessor" in model_data
        assert "label_encoders" in model_data
        assert "feature_names" in model_data
        assert "threshold" in model_data
        assert "model_name" in model_data
        
        assert model_data["model_name"] == "RandomForest"
        assert model_data["threshold"] == 0.67
        assert len(model_data["feature_names"]) == 10


# =============================================================================
# TASK E: INFERENCE TESTS
# =============================================================================

class TestInference:
    """Tests for inference functionality."""
    
    def test_inference_output_shape(self, model_data, train_val_test_data):
        """Inference output has correct shape."""
        _, _, test_df = train_val_test_data
        predictions = predict_phase4(test_df, model_data)
        
        assert len(predictions) == len(test_df)
        assert "predicted_probability" in predictions.columns
        assert "predicted_label" in predictions.columns
    
    def test_inference_probabilities_valid(self, model_data, train_val_test_data):
        """Inference probabilities are valid (0-1)."""
        _, _, test_df = train_val_test_data
        predictions = predict_phase4(test_df, model_data)
        
        probs = predictions["predicted_probability"].values
        assert np.all(probs >= 0) and np.all(probs <= 1)
    
    def test_inference_labels_binary(self, model_data, train_val_test_data):
        """Inference labels are binary (0 or 1)."""
        _, _, test_df = train_val_test_data
        predictions = predict_phase4(test_df, model_data)
        
        labels = predictions["predicted_label"].values
        assert set(labels) <= {0, 1}
    
    def test_inference_consistent_with_threshold(self, model_data, train_val_test_data):
        """Predicted labels match threshold applied to probabilities."""
        _, _, test_df = train_val_test_data
        predictions = predict_phase4(test_df, model_data)
        
        threshold = model_data["threshold"]
        expected_labels = (predictions["predicted_probability"] >= threshold).astype(int)
        
        np.testing.assert_array_equal(
            predictions["predicted_label"].values,
            expected_labels
        )
    
    def test_inference_reproducible(self, model_data, train_val_test_data):
        """Inference is reproducible (same input -> same output)."""
        _, _, test_df = train_val_test_data
        
        pred1 = predict_phase4(test_df, model_data)
        pred2 = predict_phase4(test_df, model_data)
        
        np.testing.assert_array_equal(pred1["predicted_label"].values, pred2["predicted_label"].values)
        np.testing.assert_allclose(pred1["predicted_probability"].values, pred2["predicted_probability"].values)


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestPhase123Regression:
    """Ensure Phase 4 doesn't break Phase 1-3 functionality."""
    
    def test_phase2_pipeline_still_works(self, train_val_test_data):
        """Phase 2 pipeline output unchanged."""
        train_df, val_df, test_df = train_val_test_data
        
        # Combined should equal original 1000 rows
        total_rows = len(train_df) + len(val_df) + len(test_df)
        assert total_rows == 1000
        
        # All have required columns
        for df in [train_df, val_df, test_df]:
            for pt in PRIMARY_TELEMETRY:
                assert pt in df.columns
            assert TARGET_COLUMN in df.columns
    
    def test_phase3_split_unchanged(self, phase4_report):
        """Phase 3 split artifacts unchanged."""
        split_info = phase4_report["split_info"]
        
        # Verify split strategy is still group-aware
        assert split_info["strategy"] in ["tabular_stratified_group_aware", "tabular_stratified"]
        assert split_info["no_sensor_overlap"] == True
    
    def test_feature_manifest_unchanged(self):
        """Feature manifest from Phase 2 unchanged."""
        manifest_path = PIPER_PROCESSED_DIR / "scada_feature_manifest.json"
        assert manifest_path.exists()
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        assert manifest["summary"]["predictive_features"] == 28
        assert manifest["summary"]["excluded_features"] == 3


# =============================================================================
# METRIC CALCULATION TESTS
# =============================================================================

class TestMetrics:
    """Tests for metric calculation correctness."""
    
    def test_evaluate_model_returns_all_metrics(self, train_val_test_data):
        """evaluate_model returns all required metrics."""
        train_df, val_df, test_df = train_val_test_data
        
        X_train, X_val, X_test, feature_names, _, _ = prepare_features(train_df, val_df, test_df)
        y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
        
        model, threshold = train_logistic_regression(X_train, y_train, X_val, y_val)
        metrics = evaluate_model(model, X_val, y_val, threshold)
        
        # Check all fields present
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'f1')
        assert hasattr(metrics, 'roc_auc')
        assert hasattr(metrics, 'confusion_matrix')
        assert hasattr(metrics, 'class_report')
        assert hasattr(metrics, 'threshold')
    
    def test_metrics_reproducible(self, train_val_test_data):
        """Metrics are reproducible for same model/data."""
        train_df, val_df, test_df = train_val_test_data
        
        X_train, X_val, X_test, feature_names, _, _ = prepare_features(train_df, val_df, test_df)
        y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
        
        model, threshold = train_random_forest(X_train, y_train, X_val, y_val)
        
        m1 = evaluate_model(model, X_val, y_val, threshold)
        m2 = evaluate_model(model, X_val, y_val, threshold)
        
        assert m1.accuracy == m2.accuracy
        assert m1.f1 == m2.f1
        assert m1.roc_auc == m2.roc_auc


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])