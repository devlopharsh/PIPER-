# PIPER Valve Status Correction Report

## Section A: Initial Dataset Analysis

### A.1 Original Valve Status Distribution (New Dataset: `scada_pipeline_1hour_updated.csv`)

| Valve Status | Count | Percentage |
|--------------|-------|------------|
| 0 | 709 | 19.69% |
| 1 | 2,492 | 69.22% |
| 2 | 399 | 11.08% |
| **Total** | **3,600** | **100.00%** |

### A.2 Segment-Specific Anomalies

The new dataset contains **3,600 rows** across **50 segments** with **2,066 unique (segment_id, timestamp) groups**. Critically, **1,017 groups (49.2%) have multiple rows per timestamp** (2-7 rows), and **543 groups (26.3%) have inconsistent valve_status values** within the same (segment_id, timestamp) group.

**Examples of inconsistent groups:**
- Segment 1, 00:04:00: valve_status = [0, 1, 1] (3 rows)
- Segment 2, 00:03:00: valve_status = [0, 0, 1, 1] (4 rows)
- Segment 42, 00:14:00: valve_status = [1, 1, 1, 2, 1, 1, 1] (7 rows)

### A.3 Temporal Transition Observations

**Old dataset (reference):** 1,000 rows, 1,000 unique (segment, timestamp) pairs, 487 transitions, **no zero-time-diff transitions**.

**New dataset:** 3,600 rows, 2,066 unique (segment, timestamp) pairs, 1,695 transitions, **692 transitions with 0 time diff** (same timestamp).

**Transition type distribution (new dataset):**
| Transition | Count |
|------------|-------|
| 0→1 | 487 |
| 1→0 | 466 |
| 1→2 | 272 |
| 2→1 | 255 |
| 2→0 | 95 |
| 0→2 | 70 |

**Isolated transitions** (single-point changes surrounded by identical state): 623 (36.8% of transitions)
- 1→0→1: 308
- 1→2→1: 173
- 0→1→0: 92

### A.4 Relationship with Telemetry

| Valve Status | Pressure (mean) | Flow Rate (mean) | Temp (mean) | Pump Speed (mean) | Energy (mean) | Pump=0% |
|--------------|-----------------|------------------|-------------|-------------------|---------------|---------|
| 0 | 75.01 | 4.50 | 31.61 | 889.98 | 26.09 | 31.5% |
| 1 | 74.69 | 4.43 | 31.60 | 894.34 | 26.23 | 30.5% |
| 2 | 75.36 | 4.38 | 31.62 | 899.72 | 26.29 | 29.8% |

Telemetry differences between valve statuses are minimal (<1% for pressure, flow, temperature), indicating valve_status has weak predictive signal for telemetry.

### A.5 Relationship with Event Types

| Event Type | Valve=0% | Valve=1% | Valve=2% |
|------------|----------|----------|----------|
| Normal | 19.1% | 69.3% | 11.5% |
| Degradation | 20.9% | 68.5% | 10.6% |
| Leak | 24.7% | 65.5% | 9.8% |
| Surge | 19.5% | 69.5% | 11.0% |
| Blockage | 17.9% | 74.4% | 7.7% |

Valve status distributions are similar across event types, with no strong diagnostic pattern.

---

## Section B: Correction Summary

### B.1 Correction Strategy

**Root cause identified:** The new dataset has multiple rows per (segment_id, timestamp) group (up to 7 rows) with conflicting valve_status values. This is a data generation artifact, not physically meaningful.

**Correction rules applied:**
1. For groups with consistent valve_status: preserve original value
2. For groups with clear majority (>50%): use majority value
3. For groups without clear majority: apply physical consistency scoring based on:
   - pump_state (0→prefer 0/1, 1→prefer 1/2)
   - compressor_state (0→prefer 1/2, 1→prefer 0/1)
   - pump_speed (0→prefer 0/1, >0→prefer 1/2)

### B.2 Correction Results

| Metric | Value |
|--------|-------|
| Original valve_status entries | 3,600 |
| Entries modified | 618 (17.17%) |
| Entries preserved | 2,982 (82.83%) |
| Groups made consistent | 543 (100% of inconsistent groups) |

### B.3 Before vs After Distribution

| Valve Status | Before | After | Change |
|--------------|--------|-------|--------|
| 0 | 709 (19.69%) | 379 (10.53%) | -330 |
| 1 | 2,492 (69.22%) | 2,942 (81.72%) | +450 |
| 2 | 399 (11.08%) | 279 (7.75%) | -120 |

The correction consolidates valve_status toward the dominant state (1) which aligns with the physical expectation that valves are predominantly open during normal pipeline operation.

---

## Section C: Model Comparison

### C.1 Evaluation Setup

- **Split strategy:** Sensor-level (35 train / 5 val / 10 test sensors), no sensor overlap
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Threshold selection:** Validation F1 maximization (99 thresholds: 0.01-0.99)
- **Test set:** Held out until final evaluation
- **Features:** pressure, flow_rate, temperature, pump_speed, pump_state, compressor_state, energy_consumption, valve_status, flow_pressure_ratio, sensor_id_encoded
- **Target:** label (0=Normal, 1=Abnormal)

### C.2 Performance Comparison

| Metric | Original Dataset | Corrected Dataset | Difference |
|--------|------------------|-------------------|------------|
| **Accuracy** | 0.3433 | 0.3303 | **-0.0131** |
| **Precision** | 0.3205 | 0.3152 | **-0.0053** |
| **Recall** | 0.9710 | 0.9627 | **-0.0083** |
| **F1-Score** | 0.4820 | 0.4749 | **-0.0071** |
| **ROC-AUC** | 0.5238 | 0.5093 | **-0.0146** |
| **PR-AUC** | 0.3561 | 0.3418 | **-0.0143** |
| **Specificity** | 0.0552 | 0.0400 | -0.0152 |

### C.3 Confusion Matrices

**Original Dataset (Best: XGBoost, threshold=0.06):**
| | Predicted Normal | Predicted Abnormal |
|---|---|---|
| **Actual Normal** | 29 | 496 |
| **Actual Abnormal** | 7 | 234 |

**Corrected Dataset (Best: XGBoost, threshold=0.05):**
| | Predicted Normal | Predicted Abnormal |
|---|---|---|
| **Actual Normal** | 21 | 504 |
| **Actual Abnormal** | 9 | 232 |

### C.4 All Models Test Comparison

| Model | Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------|----------|-----------|--------|-----|---------|
| LogisticRegression | Original | 0.3159 | 0.3150 | 1.0000 | 0.4791 | 0.5177 |
| RandomForest | Original | 0.3146 | 0.3146 | 1.0000 | 0.4786 | 0.5197 |
| **XGBoost** | **Original** | **0.3433** | **0.3205** | **0.9710** | **0.4820** | **0.5238** |
| LogisticRegression | Corrected | 0.3159 | 0.3150 | 1.0000 | 0.4791 | 0.5005 |
| RandomForest | Corrected | 0.3146 | 0.3146 | 1.0000 | 0.4786 | 0.5046 |
| **XGBoost** | **Corrected** | **0.3303** | **0.3152** | **0.9627** | **0.4749** | **0.5093** |

---

## Section D: Integrity Verification

| Check | Result |
|-------|--------|
| Row count preserved (3,600) | ✅ PASS |
| Column count and order preserved | ✅ PASS |
| All non-valve_status columns identical | ✅ PASS |
| Row order preserved | ✅ PASS |
| No missing values introduced | ✅ PASS |
| No unexpected valve_status categories | ✅ PASS (0, 1, 2 only) |
| target column unchanged | ✅ PASS |
| alarm_triggered column unchanged | ✅ PASS |
| event_type column unchanged | ✅ PASS |
| Original datasets untouched | ✅ PASS |
| Output saved as new file | ✅ PASS |

**Output file:** `C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_valve_corrected.csv`

---

## Section E: Final Recommendation

### E.1 Does the corrected valve_status improve PIPER's predictive performance?

**No.** The valve_status correction does not improve predictive performance. All metrics slightly degrade:
- ROC-AUC decreases from 0.5238 → 0.5093 (-0.0146)
- F1-Score decreases from 0.4820 → 0.4749 (-0.0071)
- Accuracy decreases from 0.3433 → 0.3303 (-0.0131)

### E.2 Why doesn't it improve performance?

1. **Dataset limitations dominate:** The SCADA dataset spans only **16 minutes** with **17 unique timestamps** and **multiple conflicting readings per timestamp**. This fundamentally limits any model's ability to learn temporal patterns.

2. **Valve status has minimal signal:** Telemetry differences between valve states are negligible (<1% for pressure/flow/temperature). Valve status distribution is nearly identical across event types (Normal, Degradation, Leak, Surge, Blockage).

3. **Correction introduces bias:** The correction consolidates valve_status toward state 1 (81.7% vs 69.2%), reducing the already minimal variance in this feature. This slightly degrades the feature's discriminative power.

4. **Near-random performance:** All models achieve ROC-AUC ≈ 0.50-0.52, indicating the feature set (including corrected valve_status) has **no meaningful predictive power** for this dataset.

### E.3 Honest Assessment

The PIPER LSTM/tabular pipeline cannot achieve the target metrics (97-98% Accuracy/Precision) on this dataset because:

- **Temporal span insufficient:** 16 minutes with 17 timestamps vs. required hours/days for LSTM sequences
- **Duplicate timestamps:** Multiple conflicting readings per (segment, timestamp) prevent reliable sequence construction
- **Label provenance unknown:** Phase 1.1 audit verdict was FAIL - target may be synthetic/derived from event_type
- **Leakage features present:** event_type and alarm_triggered perfectly/near-perfectly predict target

### E.4 Recommendations

1. **Do not use valve_status as a predictive feature** for this dataset - it adds noise without signal
2. **Address fundamental data limitations** before pursuing feature engineering:
   - Obtain longer time-series data (hours/days)
   - Resolve duplicate timestamp conflicts
   - Document label generation process
3. **If valve_status must be used:** Keep the corrected version (physically consistent within timestamp groups) but expect no performance gain
4. **Consider excluding valve_status entirely** from the feature set for this dataset

---

## Appendix: Correction Algorithm Details

```python
# For each (segment_id, timestamp) group:
if len(unique_valve_status) == 1:
    corrected = unique_valve_status[0]
elif max_count > total_rows / 2:
    corrected = majority_value
else:
    # Physical consistency scoring
    score = 0
    if pump_state_mode == 0 and candidate in [0, 1]: score += 1
    if pump_state_mode == 1 and candidate in [1, 2]: score += 1
    if compressor_state_mode == 0 and candidate in [1, 2]: score += 0.5
    if compressor_state_mode == 1 and candidate in [0, 1]: score += 0.5
    if pump_speed_mean == 0 and candidate in [0, 1]: score += 1
    if pump_speed_mean > 0 and candidate in [1, 2]: score += 1
    corrected = argmax(score, tiebreak=count)
```

**Files generated:**
- `scada_pipeline_1hour_valve_corrected.csv` - Corrected dataset
- `data/reports/valve_correction_comparison.json` - Comparison metrics
- `evaluate_valve_correction.py` - Evaluation script
- `fix_valve_correction.py` - Correction script
- `verify_integrity.py` - Integrity verification script