# PIPER Phase 4 Summary Report: scada

**Generated:** 2026-10-27T18:12:03.080245

## Executive Summary

| Metric | Value |
|--------|-------|
| Selected Model | RandomForest |
| Selected Threshold | 0.6700 |
| Test Accuracy | 0.9539 |
| Test Precision | 0.9697 |
| Test Recall | 0.8889 |
| Test F1 | 0.9275 |
| Test ROC-AUC | 0.9847 |

## Split Information

| Split | Rows | Sensors | Normal (0) | Abnormal (1) | Abnormal Rate |
|-------|------|---------|------------|--------------|---------------|
| Train | 678 | 35 | 473 | 205 | 0.3024 |
| Val | 105 | 5 | 76 | 29 | 0.2762 |
| Test | 217 | 10 | 145 | 72 | 0.3318 |

**No sensor overlap between splits:** True

## Feature Schema

- **Predictor features (10)**: compressor_state, energy_consumption, flow_pressure_ratio, flow_rate, pressure, pump_speed, pump_state, sensor_id_encoded, temperature, valve_status
- **Target column**: label
- **Excluded features**: label, alarm_triggered, event_type, timestamp, timestamp_parsed, sensor_id, reading_index, n_sensors, hour, dayofweek, month, quarter, dayofyear, weekofyear, is_weekend, sin_hour, cos_hour, sin_dayofweek, cos_dayofweek, sin_month, cos_month

## Model Comparison (Validation Metrics)

| Model | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Threshold |
|-------|--------------|---------------|------------|--------|-------------|-----------|
| LogisticRegression | 0.8667 | 0.9412 | 0.5517 | 0.6957 | 0.7568 | 0.7000 |
| RandomForest | 0.9619 | 0.9630 | 0.8966 | 0.9286 | 0.9877 | 0.6700 |
| XGBoost | 0.9524 | 0.9615 | 0.8621 | 0.9091 | 0.9887 | 0.9000 |


## Selected Model: RandomForest

**Configuration:** {
  "n_estimators": 200,
  "max_depth": 10,
  "min_samples_split": 5,
  "min_samples_leaf": 2,
  "class_weight": "balanced",
  "random_state": 42
}

**Threshold:** 0.6700 (optimized on validation set)

## Final Test Evaluation

| Metric | Value |
|--------|-------|
| Accuracy | 0.9539 |
| Precision | 0.9697 |
| Recall | 0.8889 |
| F1-Score | 0.9275 |
| ROC-AUC | 0.9847 |
| Threshold | 0.6700 |

### Confusion Matrix (Test)

| | Predicted Normal | Predicted Abnormal |
|---|---|---|
| **Actual Normal** | 143 | 2 |
| **Actual Abnormal** | 8 | 64 |

### Class-wise Metrics (Test)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | 0.9470 | 0.9862 | 0.9662 | 145 |
| Abnormal (1) | 0.9697 | 0.8889 | 0.9275 | 72 |

## Artifacts Saved

- `models/phase4_model.joblib` - Fitted model
- `models/phase4_preprocessor.joblib` - Preprocessing pipeline + encoders
- `data/reports/phase4_report.json` - Full report
- `data/reports/phase4_summary.md` - This summary
- `data/processed/scada_test_predictions.csv` - Test set predictions

## Recommendations

- Selected model: RandomForest with threshold 0.6700
- Test F1: 0.9275, ROC-AUC: 0.9847
- Split is sensor-level (no sensor appears in multiple splits)
- Preprocessing fitted on training data only (no leakage)
- Class imbalance handled via class_weight='balanced' and scale_pos_weight
- Threshold optimized on validation set (not test set)
- For production: monitor feature drift and retrain periodically


## Reproducibility

To reproduce this pipeline:

```bash
cd PIPER-
python -m utils.piper_phase4
```

All preprocessing (imputation, scaling, encoding) is fitted on training data only.
Validation set used for model selection and threshold optimization.
Test set remains isolated until final evaluation.

---
*Note: Split is sensor-level (35 train / 5 val / 10 test sensors). 
This does not guarantee generalization to unseen pipelines since pipeline_id is not available.*
