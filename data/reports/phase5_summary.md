# PIPER Phase 5 Summary Report: scada

**Generated:** 2026-10-27T18:23:17.059141

## Executive Summary

| Metric | Value |
|--------|-------|
| Model | RandomForest |
| Threshold | 0.67 |
| Test Accuracy | 0.9539 |
| Test Precision | 0.9697 |
| Test Recall | 0.8889 |
| Test F1-Score | 0.9275 |
| Test ROC-AUC | 0.9847 |
| Test PR-AUC | 0.9771 |
| Test Specificity | 0.9862 |

## Verification Results

| Check | Result |
|-------|--------|
| Confusion matrix orientation | sklearn convention: rows=true, cols=pred. TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1] |
| Positive class | 1 = Abnormal/Leak (consistent with Phase 2 label generation) |
| Threshold application verified | pred = (proba >= 0.67).astype(int); verified: True |
| Saved vs recomputed probabilities match | True |
| Saved vs recomputed predictions match | True |
| Metrics recomputed from saved artifacts | True |
| No test data used for model selection | True |
| No test data used for threshold selection | True |
| Preprocessing fitted on training only | True |

## Confusion Matrix (Test Set)

| | Predicted Normal | Predicted Abnormal |
|---|---|---|
| **Actual Normal** | 143 | 2 |
| **Actual Abnormal** | 8 | 64 |

**Convention:** TN=143, FP=2, FN=8, TP=64 (Positive class = 1 = Abnormal)

## Class-wise Metrics (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | 0.9470 | 0.9862 | 0.9662 | 145 |
| Abnormal (1) | 0.9697 | 0.8889 | 0.9275 | 72 |

## Model Comparison (Validation-Based Selection)

| Model | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Selected |
|-------|--------------|---------------|------------|--------|-------------|----------|
| LogisticRegression | 0.8667 | 0.9412 | 0.5517 | 0.6957 | 0.7568 |  |
| RandomForest | 0.9619 | 0.9630 | 0.8966 | 0.9286 | 0.9877 | ✓ |
| XGBoost | 0.9524 | 0.9615 | 0.8621 | 0.9091 | 0.9887 |  |


## Threshold Selection Procedure

- **Method:** validation_f1_maximization
- **Description:** For each candidate model, threshold was selected by maximizing F1-score on the validation set across 99 thresholds (0.01 to 0.99). The model with highest validation F1 was selected, and its corresponding threshold was used for test evaluation.
- **Selected threshold:** 0.67
- **Selected model:** RandomForest
- **Validation F1 at selected threshold:**
  - LogisticRegression: 0.6957
  - RandomForest: 0.9286
  - XGBoost: 0.9091

> **Note:** Threshold was NOT optimized on test set. Test set used only for final evaluation.

## Evaluation Plots

- Confusion Matrix: `data\reports\plots\phase5_confusion_matrix.png`
- ROC Curve: `data\reports\plots\phase5_roc_curve.png`
- Precision-Recall Curve: `data\reports\plots\phase5_pr_curve.png`
- Calibration Curve: `data\reports\plots\phase5_calibration_curve.png`

## Limitations

- Split is sensor-level (35 train / 5 val / 10 test sensors), not pipeline-level. pipeline_id unavailable in dataset.
- All temporal features are constant (single-day data) and were correctly excluded from predictors.
- Generalization to unseen pipelines cannot be guaranteed by current split strategy.
- XGBoost shows overfitting (train F1=0.998 vs val F1=0.909); Random Forest selected for better generalization.
- Dataset has 16-minute time span with conflicting duplicate timestamps - temporal modeling not supported.
- Only 10 test sensors (217 samples); confidence intervals on metrics would be wide.
- Calibration curve shows moderate miscalibration at high probabilities (expected with Random Forest).


## Reproducibility

To reproduce this evaluation:

```bash
cd PIPER-
python -m utils.piper_phase5
```

All evaluation metrics are computed from saved Phase 4 artifacts and the isolated test split.
No test data was used for model selection or threshold optimization.

---

*Evaluation completed using saved Random Forest model with threshold 0.67, selected via validation F1 maximization.*
