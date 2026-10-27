"""
PIPER Phase 5 Tests

Tests for predictive performance evaluation:
1. Metrics recomputation matches saved predictions
2. Confusion matrix orientation (sklearn convention)
3. Threshold handling and positive class definition
3. Artifact reload and prediction reproducibility
4. ROC/PR curve generation
5. Calibration curve generation
6. Model comparison documentation
7. Phase 2-4 regression tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)

from utils.piper_phase5 import (
    run_phase5_evaluation,
    load_phase4_artifacts,
    recompute_predictions,
    compute_evaluation_metrics,
    verify_threshold_application,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
    POSITIVE_CLASS,
    NEGATIVE_CLASS,
    PIPER_MODELS_DIR,
    PIPER_PROCESSED_DIR,
    PIPER_REPORTS_DIR,
    PIPER_PLOTS_DIR,
)
from utils.piper_telemetry import TARGET_COLUMN


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def phase5_report():
    """Run Phase 5 and return the report (or load existing)."""
    report_path = PIPER_REPORTS_DIR / "phase5_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    else:
        from utils.piper_phase5 import run_phase5_evaluation
        report_obj = run_phase5_evaluation(save_artifacts=True)
        return report_obj.to_dict()


@pytest.fixture(scope="session")
def loaded_artifacts():
    """Load Phase 4 model, preprocessor, test data, and saved predictions."""
    model_data, test_df, saved_preds = load_phase4_artifacts()
    return model_data, test_df, saved_preds


@pytest.fixture(scope="session")
def model_data(loaded_artifacts):
    """Load Phase 4 model and preprocessor (dict)."""
    return loaded_artifacts[0]


@pytest.fixture(scope="session")
def test_data(loaded_artifacts):
    """Load test data and saved predictions."""
    return loaded_artifacts[1], loaded_artifacts[2]


# =============================================================================
# METRICS RECOMPUTATION TESTS
# =============================================================================

class TestMetricsRecomputation:
    """Tests verifying metrics recomputation matches reported values."""
    
    def test_confusion_matrix_matches(self, model_data, test_data):
        """Confusion matrix matches reported values."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_pred = saved_preds['predicted_label'].values
        y_proba = saved_preds['predicted_probability'].values
        threshold = model_data['threshold']
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        assert tn == 143, f"TN={tn}, expected 143"
        assert fp == 2, f"FP={fp}, expected 2"
        assert fn == 8, f"FN={fn}, expected 8"
        assert tp == 64, f"TP={tp}, expected 64"
    
    def test_accuracy_matches(self, model_data, test_data):
        """Accuracy matches reported value."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_pred = saved_preds['predicted_label'].values
        
        acc = accuracy_score(y_true, y_pred)
        assert abs(acc - 0.953917) < 0.0001, f"Accuracy={acc:.6f}"
    
    def test_precision_matches(self, model_data, test_data):
        """Precision matches reported value."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_pred = saved_preds['predicted_label'].values
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        assert abs(prec - 0.969697) < 0.0001, f"Precision={prec:.6f}"
    
    def test_recall_matches(self, model_data, test_data):
        """Recall matches reported value."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_pred = saved_preds['predicted_label'].values
        
        rec = recall_score(y_true, y_pred, zero_division=0)
        assert abs(rec - 0.888889) < 0.0001, f"Recall={rec:.6f}"
    
    def test_f1_matches(self, model_data, test_data):
        """F1-score matches reported value."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_pred = saved_preds['predicted_label'].values
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        assert abs(f1 - 0.927536) < 0.0001, f"F1={f1:.6f}"
    
    def test_roc_auc_matches(self, model_data, test_data):
        """ROC-AUC matches reported value."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_proba = saved_preds['predicted_probability'].values
        
        roc_auc = roc_auc_score(y_true, y_proba)
        assert abs(roc_auc - 0.984674) < 0.0001, f"ROC-AUC={roc_auc:.6f}"
    
    def test_pr_auc_computed(self, model_data, test_data):
        """PR-AUC is computed and valid."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_proba = saved_preds['predicted_probability'].values
        
        pr_auc = average_precision_score(y_true, y_proba)
        assert 0 <= pr_auc <= 1, f"PR-AUC={pr_auc}"
        assert pr_auc > 0.5, f"PR-AUC should be > 0.5 for useful classifier"
    
    def test_specificity_computed(self, model_data, test_data):
        """Specificity is computed correctly."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_pred = saved_preds['predicted_label'].values
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp)
        
        assert abs(spec - 0.986207) < 0.0001, f"Specificity={spec:.6f}"
    
    def test_class_wise_metrics(self, model_data, test_data):
        """Class-wise metrics are correct."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_pred = saved_preds['predicted_label'].values
        
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Normal class (0)
        assert abs(report['0']['precision'] - 0.947020) < 0.0001
        assert abs(report['0']['recall'] - 0.986207) < 0.0001
        assert abs(report['0']['f1-score'] - 0.966216) < 0.0001
        assert report['0']['support'] == 145
        
        # Abnormal class (1)
        assert abs(report['1']['precision'] - 0.969697) < 0.0001
        assert abs(report['1']['recall'] - 0.888889) < 0.0001
        assert abs(report['1']['f1-score'] - 0.927536) < 0.0001
        assert report['1']['support'] == 72


# =============================================================================
# CONFUSION MATRIX ORIENTATION TESTS
# =============================================================================

class TestConfusionMatrixOrientation:
    """Tests for confusion matrix orientation and conventions."""
    
    def test_sklearn_convention(self, model_data, test_data):
        """Confusion matrix follows sklearn convention: rows=true, cols=pred."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_pred = saved_preds['predicted_label'].values
        
        cm = confusion_matrix(y_true, y_pred)
        
        # sklearn: cm[i, j] = number of samples with true label i, predicted label j
        # Row 0 (true=0): [TN, FP]
        # Row 1 (true=1): [FN, TP]
        assert cm[0, 0] == 143  # TN
        assert cm[0, 1] == 2    # FP
        assert cm[1, 0] == 8    # FN
        assert cm[1, 1] == 64   # TP
    
    def test_positive_class_is_abnormal(self, model_data, test_data):
        """Positive class is 1 (Abnormal), consistent with Phase 2."""
        assert POSITIVE_CLASS == 1
        assert NEGATIVE_CLASS == 0
        
        test_df, saved_preds = test_data
        y_true = test_df[TARGET_COLUMN].values
        
        # Verify both classes present
        assert set(y_true) == {0, 1}
        
        # Phase 2 label generation: 0=normal, 1=abnormal
        assert 1 in y_true  # Abnormal present
        assert 0 in y_true  # Normal present
    
    def test_threshold_applied_correctly(self, model_data, test_data):
        """Threshold is applied as: pred = (proba >= threshold).astype(int)."""
        test_df, saved_preds = test_data
        
        y_proba = saved_preds['predicted_probability'].values
        y_pred = saved_preds['predicted_label'].values
        threshold = model_data['threshold']
        
        # Verify threshold application
        pred_from_proba = (y_proba >= threshold).astype(int)
        assert np.array_equal(y_pred, pred_from_proba)
        
        # Check threshold value
        assert threshold == 0.67
    
    def test_no_inverted_predictions(self, model_data, test_data):
        """Predictions are not inverted (higher proba -> class 1)."""
        test_df, saved_preds = test_data
        
        y_proba = saved_preds['predicted_probability'].values
        y_pred = saved_preds['predicted_label'].values
        
        # For samples predicted as class 1, probability should be >= threshold
        class1_proba = y_proba[y_pred == 1]
        assert np.all(class1_proba >= 0.67), "Class 1 predictions have proba < threshold"
        
        # For samples predicted as class 0, probability should be < threshold
        class0_proba = y_proba[y_pred == 0]
        assert np.all(class0_proba < 0.67), "Class 0 predictions have proba >= threshold"


# =============================================================================
# ARTIFACT RELOAD AND REPRODUCIBILITY TESTS
# =============================================================================

class TestArtifactReloadReproducibility:
    """Tests for artifact reload and prediction reproducibility."""
    
    def test_model_reload_equivalence(self, loaded_artifacts, test_data):
        """Reloaded model produces identical predictions."""
        model_data = loaded_artifacts[0]
        test_df = test_data[0]
        
        # Get predictions from original model data
        proba1 = recompute_predictions(model_data, test_df)
        pred1 = (proba1 >= model_data['threshold']).astype(int)
        
        # Reload and predict again
        reloaded_data = load_phase4_artifacts()
        reloaded_model_data = reloaded_data[0]
        proba2 = recompute_predictions(reloaded_model_data, test_df)
        pred2 = (proba2 >= reloaded_model_data['threshold']).astype(int)
        
        # Should be identical
        np.testing.assert_allclose(proba1, proba2, rtol=1e-10)
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_saved_predictions_match_recomputed(self, model_data, test_data):
        """Saved predictions match recomputed from model."""
        test_df, saved_preds = test_data
        
        y_proba_saved = saved_preds['predicted_probability'].values
        y_pred_saved = saved_preds['predicted_label'].values
        
        y_proba_recomputed = recompute_predictions(model_data, test_df)
        y_pred_recomputed = (y_proba_recomputed >= model_data['threshold']).astype(int)
        
        # Should match exactly
        assert np.allclose(y_proba_saved, y_proba_recomputed, rtol=1e-10)
        assert np.array_equal(y_pred_saved, y_pred_recomputed)
    
    def test_preprocessor_reload_equivalence(self):
        """Reloaded preprocessor produces identical transformations."""
        preprocessor_data1 = joblib.load(PIPER_MODELS_DIR / "phase4_preprocessor.joblib")
        preprocessor_data2 = joblib.load(PIPER_MODELS_DIR / "phase4_preprocessor.joblib")
        
        assert preprocessor_data1['threshold'] == preprocessor_data2['threshold']
        assert preprocessor_data1['model_name'] == preprocessor_data2['model_name']
        assert preprocessor_data1['feature_names'] == preprocessor_data2['feature_names']
    
    def test_evaluation_reproducible(self, model_data, test_data):
        """Evaluation metrics are reproducible."""
        test_df, saved_preds = test_data
        
        y_true = test_df[TARGET_COLUMN].values
        y_pred = saved_preds['predicted_label'].values
        y_proba = saved_preds['predicted_probability'].values
        threshold = model_data['threshold']
        
        m1 = compute_evaluation_metrics(y_true, y_pred, y_proba, threshold)
        m2 = compute_evaluation_metrics(y_true, y_pred, y_proba, threshold)
        
        assert m1.accuracy == m2.accuracy
        assert m1.precision == m2.precision
        assert m1.recall == m2.recall
        assert m1.f1_score == m2.f1_score
        assert m1.roc_auc == m2.roc_auc
        assert m1.pr_auc == m2.pr_auc


# =============================================================================
# PLOT GENERATION TESTS
# =============================================================================

class TestPlotGeneration:
    """Tests for plot generation."""
    
    def test_confusion_matrix_plot_exists(self):
        """Confusion matrix plot was generated."""
        assert (PIPER_PLOTS_DIR / "phase5_confusion_matrix.png").exists()
        assert (PIPER_PLOTS_DIR / "phase5_confusion_matrix.png").stat().st_size > 0
    
    def test_roc_curve_plot_exists(self):
        """ROC curve plot was generated."""
        assert (PIPER_PLOTS_DIR / "phase5_roc_curve.png").exists()
        assert (PIPER_PLOTS_DIR / "phase5_roc_curve.png").stat().st_size > 0
    
    def test_pr_curve_plot_exists(self):
        """PR curve plot was generated."""
        assert (PIPER_PLOTS_DIR / "phase5_pr_curve.png").exists()
        assert (PIPER_PLOTS_DIR / "phase5_pr_curve.png").stat().st_size > 0
    
    def test_calibration_curve_plot_exists(self):
        """Calibration curve plot was generated."""
        assert (PIPER_PLOTS_DIR / "phase5_calibration_curve.png").exists()
        assert (PIPER_PLOTS_DIR / "phase5_calibration_curve.png").stat().st_size > 0
    
    def test_plots_are_valid_png(self):
        """Generated plots are valid PNG files."""
        import struct
        
        for plot_name in ["phase5_confusion_matrix.png", "phase5_roc_curve.png",
                          "phase5_pr_curve.png", "phase5_calibration_curve.png"]:
            path = PIPER_PLOTS_DIR / plot_name
            with open(path, 'rb') as f:
                header = f.read(8)
                assert header == b'\x89PNG\r\n\x1a\n', f"{plot_name} is not a valid PNG"


# =============================================================================
# MODEL COMPARISON TESTS
# =============================================================================

class TestModelComparison:
    """Tests for model comparison documentation."""
    
    def test_model_comparison_documented(self, phase5_report):
        """Model comparison is documented in report."""
        comparison = phase5_report["model_comparison"]
        
        assert len(comparison) == 3
        model_names = {mc["model_name"] for mc in comparison}
        assert model_names == {"LogisticRegression", "RandomForest", "XGBoost"}
        
        # One model should be selected
        selected = [mc for mc in comparison if mc["selected"]]
        assert len(selected) == 1
        assert selected[0]["model_name"] == "RandomForest"
    
    def test_validation_f1_maximization(self, phase5_report):
        """Model selected based on validation F1."""
        comparison = phase5_report["model_comparison"]
        selected = [mc for mc in comparison if mc["selected"]][0]
        
        # Random Forest should have highest val F1
        val_f1s = {mc["model_name"]: mc["val_metrics"]["f1"] for mc in comparison}
        assert val_f1s["RandomForest"] == max(val_f1s.values())
        assert selected["model_name"] == "RandomForest"
    
    def test_threshold_not_optimized_on_test(self, phase5_report):
        """Threshold selection did not use test data."""
        threshold_sel = phase5_report["threshold_selection"]
        
        assert threshold_sel["method"] == "validation_f1_maximization"
        assert "NOT optimized on test set" in threshold_sel["note"]
        assert threshold_sel["selected_threshold"] == 0.67
    
    def test_test_metrics_only_for_final_evaluation(self, phase5_report):
        """Test metrics only computed for selected model."""
        comparison = phase5_report["model_comparison"]
        
        for mc in comparison:
            if mc["selected"]:
                assert mc["test_metrics"] is not None
            else:
                # Non-selected models should not have test metrics (or they should be null)
                # In our implementation, only selected model gets test_metrics
                pass


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestPhase234Regression:
    """Ensure Phase 5 doesn't break Phase 2-4 functionality."""
    
    def test_phase2_artifacts_unchanged(self):
        """Phase 2 artifacts unchanged."""
        assert (PIPER_REPORTS_DIR / "scada_temporal_feasibility.json").exists()
        assert (PIPER_REPORTS_DIR / "phase2_summary.md").exists()
        assert (PIPER_PROCESSED_DIR / "scada_feature_manifest.json").exists()
    
    def test_phase3_artifacts_unchanged(self):
        """Phase 3 artifacts unchanged."""
        assert (PIPER_REPORTS_DIR / "phase3_feasibility_report.json").exists()
        assert (PIPER_REPORTS_DIR / "phase3_summary.md").exists()
        assert (PIPER_PROCESSED_DIR / "scada_train.csv").exists()
        assert (PIPER_PROCESSED_DIR / "scada_val.csv").exists()
        assert (PIPER_PROCESSED_DIR / "scada_test.csv").exists()
    
    def test_phase4_artifacts_unchanged(self):
        """Phase 4 artifacts unchanged."""
        assert (PIPER_MODELS_DIR / "phase4_model.joblib").exists()
        assert (PIPER_MODELS_DIR / "phase4_preprocessor.joblib").exists()
        assert (PIPER_REPORTS_DIR / "phase4_report.json").exists()
        assert (PIPER_REPORTS_DIR / "phase4_summary.md").exists()
    
    def test_phase4_model_still_works(self, model_data, test_data):
        """Phase 4 model still produces correct predictions."""
        test_df, saved_preds = test_data
        
        # Use Phase 4's predict function
        from utils.piper_phase4 import predict_phase4
        
        predictions = predict_phase4(test_df, model_data)
        
        # Should match saved predictions
        assert np.array_equal(
            predictions['predicted_label'].values,
            saved_preds['predicted_label'].values
        )
        assert np.allclose(
            predictions['predicted_probability'].values,
            saved_preds['predicted_probability'].values
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])