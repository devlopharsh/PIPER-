# PIPER LSTM Experiment Evaluation Report

**Generated:** 2026-11-14 00:12:45

## 1. Dataset Information

- **Source:** scada_pipeline_segment_even_timestamps.csv
- **Total Rows:** 4500
- **Total Columns:** 13
- **Unique Segments:** 30
- **Target Distribution:** Normal (0): 3000, Abnormal (1): 1500
- **Segment Lengths:** Min=150, Max=150, Mean=150.0
- **Missing Values:** 0
- **Duplicates:** 0

## 2. Preprocessing Methodology

- **Timestamp Parsing:** `%d-%m-%Y %H:%M:%S` format
- **Time Features Added:** Hour, DayOfWeek, Month
- **Rolling Features:** 6h and 24h windows (mean, std) for pressure, flow_rate, temperature, energy_consumption
- **Delta Features:** First differences for pressure, flow_rate, temperature, energy_consumption
- **Categorical Handling:** valve_status, pump_state, compressor_state, alarm_triggered kept as numeric (ordinal)
- **Normalization:** StandardScaler fitted on training data only (no leakage)
- **Sequence Construction:** Sliding windows within each segment (no cross-segment mixing)

## 3. LSTM Architecture and Parameters

- **Input Features:** 32 (pressure, flow_rate, temperature, valve_status, pump_state, pump_speed, compressor_state, energy_consumption, alarm_triggered, Hour, DayOfWeek, Month, pressure_roll6_mean, pressure_roll6_std, pressure_roll24_mean, pressure_roll24_std, flow_rate_roll6_mean, flow_rate_roll6_std, flow_rate_roll24_mean, flow_rate_roll24_std, temperature_roll6_mean, temperature_roll6_std, temperature_roll24_mean, temperature_roll24_std, energy_consumption_roll6_mean, energy_consumption_roll6_std, energy_consumption_roll24_mean, energy_consumption_roll24_std, pressure_delta, flow_rate_delta, temperature_delta, energy_consumption_delta)
- **Sequence Length (window):** 24
- **Stride:** 1
- **Hidden Units:** 96
- **Number of Layers:** 2
- **Bidirectional:** False
- **Dropout:** 0.36
- **Loss Function:** BCEWithLogitsLoss with pos_weight (class imbalance handling)
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Weight Decay:** 1e-05
- **Batch Size:** 256
- **Max Epochs:** 10
- **Random Seed:** 42

## 4. Train/Validation/Test Split

- **Test Fraction:** 0.3 (chronological, per segment)
- **Train Rows:** 3150
- **Test Rows:** 660
- **Train Windows:** 2460
- **Test Windows:** 660
- **CV Folds:** 10 (forward-chaining, per segment)
- **CV Threshold Selection:** Median of per-fold F1-optimal thresholds
- **Robust Threshold:** 0.500

## 5. Confusion Matrix

| | Predicted Normal (0) | Predicted Abnormal (1) |
|---|---:|---:|
| **Actual Normal (0)** | 426 | 14 |
| **Actual Abnormal (1)** | 14 | 206 |

## 6. Evaluation Metrics

- **Accuracy:** 0.9576
- **Precision:** 0.9364
- **Recall (Sensitivity):** 0.9364
- **F1-score:** 0.9364
- **F2-score:** 0.9364
- **Specificity:** 0.9682
- **Balanced Accuracy:** 0.9523
- **PR-AUC:** 0.9616
- **Classification Threshold:** 0.500
- **Test Samples (windows):** 660

## 7. Performance Objective Assessment

- **Target Accuracy (>= 97%):** 95.76% -- NO
- **Target Precision (>= 97%):** 93.64% -- NO

## 8. Execution Status

- **Status:** Completed successfully
- **Device:** cuda

## 9. Limitations

1. **Synthetic CV Threshold:** The decision threshold is derived from cross-validation on the training set using a synthetic leak injection methodology adapted from the original PIPER codebase.
2. **Class Imbalance:** The dataset has imbalanced classes (Normal: 68.9%, Abnormal: 31.1%), handled via pos_weight in loss function.
3. **Segment Length Variability:** Segments range from 36-112 timesteps; shorter segments may not generate enough windows for robust training.
4. **Temporal Scope:** All data is from a single day (2024-01-01); generalization to other days/conditions is untested.
5. **Feature Engineering:** Rolling and delta features are computed per-segment; no cross-segment features used.
6. **Event Type Not Used:** The `event_type` column (normal/degradation/leak/surge/blockage) was not used as a feature; target is binary.
