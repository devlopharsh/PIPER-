# PIPER Phase 1.2 — Dataset Schema and Feature Inventory

**Dataset:** `data/raw/scada_pipeline.csv`  
**Phase 1.1 Hash Verification:** MATCH (minute-precision SHA-256: `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`)  
**Current SHA-256 (seconds-precision):** `1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c`  
**Date:** 2026-11-05 (original), 2026-11-05 (seconds-precision update)

---

## ⚠️ VERSION NOTICE

This document reflects **two dataset versions**:

1. **Minute-precision (original, Phase 1.1–1.4):** SHA-256 `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`, timestamp format `%m/%d/%Y %H:%M`, 17 unique timestamps.
2. **Seconds-precision (reconstructed):** SHA-256 `1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c`, timestamp format `%m/%d/%Y %H:%M:%S`, 1,000 unique timestamps.

The seconds-precision version was created by **reconstructing seconds from original CSV row order within each minute group**. This is a derived timestamp, **not independently verified sensor-recorded time**. The original minute-precision dataset is not available as a separate file.

**All findings below reflect the original minute-precision audit unless explicitly marked [SECONDS-PRECISION UPDATE].**

---

## 1. Dataset File Identity and Hash Verification

| Property | Value (Minute-Precision) | Value (Seconds-Precision) |
|----------|--------------------------|---------------------------|
| **File path** | `data/raw/scada_pipeline.csv` | `data/raw/scada_pipeline.csv` |
| **File size** | 64,185 bytes | 69,184 bytes |
| **SHA-256** | `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a` | `1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c` |
| **Phase 1.1 hash** | `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a` | N/A (reconstructed) |
| **Hash match** | ✅ YES | N/A — different version |
| **Read status** | ✅ Successfully loaded as CSV | ✅ Successfully loaded as CSV |
| **Row count** | 1,000 | 1,000 |
| **Column count** | 13 | 13 |

---

## 2. Dataset Dimensions

| Metric | Minute-Precision | Seconds-Precision [UPDATE] |
|--------|------------------|----------------------------|
| Total rows | 1,000 | 1,000 |
| Total columns | 13 | 13 |
| Total missing values | 0 (0.0%) | 0 (0.0%) |
| Unique pipeline segments (`segment_id`) | 50 (IDs 1–50) | 50 (IDs 1–50) |
| Unique timestamps | 17 (`1/1/2024 0:00` through `1/1/2024 0:16`) | 1,000 (`01/01/2024 00:00:01` through `01/01/2024 00:16:40`) |
| Unique target values | 2 (0, 1) | 2 (0, 1) |
| Target class distribution | 0: 694 (69.4%), 1: 306 (30.6%) | 0: 694 (69.4%), 1: 306 (30.6%) |
| Exact duplicate rows | 0 | 0 |
| Duplicate `(segment_id, timestamp)` keys | 270 groups, 678 rows | 0 |

---

## 3. Full Column Inventory

| # | Column Name | Loaded dtype | Interpreted dtype | Count | Null Count | Null % | Distinct Values | Min | Max | Mean | Std | Constant? |
|---|-------------|--------------|-------------------|-------|------------|--------|-----------------|-----|-----|------|-----|-----------|
| 1 | timestamp | object | datetime (string) | 1000 | 0 | 0.0% | 17 / 1000 [UPDATE] | — | — | — | — | No |
| 2 | segment_id | int64 | categorical (segment ID) | 1000 | 0 | 0.0% | 50 | 1 | 50 | 25.75 | 14.32 | No |
| 3 | pressure | float64 | continuous numeric | 1000 | 0 | 0.0% | 833 | 43.11 | 115.30 | 76.20 | 9.48 | No |
| 4 | flow_rate | float64 | continuous numeric | 1000 | 0 | 0.0% | 312 | 1.01 | 7.53 | 4.36 | 0.89 | No |
| 5 | temperature | float64 | continuous numeric | 1000 | 0 | 0.0% | 553 | 25.92 | 38.52 | 32.10 | 1.98 | No |
| 6 | valve_status | int64 | ordinal categorical (0,1,2) | 1000 | 0 | 0.0% | 3 | 0 | 2 | 0.92 | 0.55 | No |
| 7 | pump_state | int64 | binary categorical (0,1) | 1000 | 0 | 0.0% | 2 | 0 | 1 | 0.69 | 0.46 | No |
| 8 | pump_speed | float64 | continuous numeric | 1000 | 0 | 0.0% | 643 | 0.00 | 1678.80 | 937.19 | 640.75 | No |
| 9 | compressor_state | int64 | binary categorical (0,1) | 1000 | 0 | 0.0% | 2 | 0 | 1 | 0.60 | 0.49 | No |
| 10 | energy_consumption | float64 | continuous numeric | 1000 | 0 | 0.0% | 852 | 5.42 | 58.18 | 26.85 | 9.27 | No |
| 11 | alarm_triggered | int64 | binary categorical (0,1) | 1000 | 0 | 0.0% | 2 | 0 | 1 | 0.22 | 0.41 | No |
| 12 | event_type | object | nominal categorical (5 values) | 1000 | 0 | 0.0% | 5 | — | — | — | — | No |
| 13 | target | int64 | binary label (0,1) | 1000 | 0 | 0.0% | 2 | 0 | 1 | 0.31 | 0.46 | No |

**Notes:**
- No column is constant or nearly constant (all have ≥2 distinct values).
- All 1,000 rows are complete — no missing values in any column.
- **Minute-precision:** `timestamp` stored as string; format `M/D/YYYY H:MM` (e.g., `1/1/2024 0:00`).
- **[SECONDS-PRECISION UPDATE]** `timestamp` stored as string; format `MM/DD/YYYY HH:MM:SS` (e.g., `01/01/2024 00:00:01`). **Reconstructed from CSV row order; not sensor-verified.**

---

## 4. Target Distribution

| Target Value | Count | Percentage | Interpretation |
|--------------|-------|------------|----------------|
| 0 (Normal) | 694 | 69.4% | Normal operation |
| 1 (Abnormal) | 306 | 30.6% | Abnormal event (leak, blockage, degradation, surge) |

**Target is binary.** (Unchanged between versions)

---

## 5. Column Classification

### A. Identifier Fields

| Column | Type | Notes |
|--------|------|-------|
| `segment_id` | Integer (1–50) | Identifies pipeline segment / sensor location. 50 unique segments. Should be treated as grouping key, not direct model input. |

### B. Timestamp Fields

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | String (datetime format) | **Minute-precision:** 17 unique values spanning 16 minutes. Format: `%m/%d/%Y %H:%M`. Multiple rows per `(segment_id, timestamp)` — not a regular time series.<br>**[SECONDS-PRECISION UPDATE]** 1,000 unique values spanning 16 min 39 sec. Format: `%m/%d/%Y %H:%M:%S`. **Reconstructed from CSV row order within each minute group (seconds 01–60 per minute, final minute 00–40). NOT sensor-verified.** Globally continuous 1-second sequence is an artifact of reconstruction. Per-segment gaps remain highly irregular (1–339 sec). |

### C. Telemetry Input Candidates (Primary — paper-specified)

| Column | Type | Range | Distinct | Notes |
|--------|------|-------|----------|-------|
| `pressure` | Float | 43.11 – 115.30 | 833 | Primary telemetry. Continuous. Unit unknown. |
| `flow_rate` | Float | 1.01 – 7.53 | 312 | Primary telemetry. Continuous. Unit unknown. |
| `temperature` | Float | 25.92 – 38.52 | 553 | Primary telemetry. Continuous. Unit unknown. |
| **vibration** | — | — | — | **NOT PRESENT IN DATASET** |

### D. Equipment-State Input Candidates (Auxiliary)

| Column | Type | Values | Distinct | Notes |
|--------|------|--------|----------|-------|
| `valve_status` | Integer (ordinal) | 0, 1, 2 | 3 | Equipment state. Meaning of 0/1/2 unknown. |
| `pump_state` | Binary | 0, 1 | 2 | Pump on/off. |
| `pump_speed` | Float | 0.0 – 1678.8 | 643 | Continuous when pump on; 0 when off. |
| `compressor_state` | Binary | 0, 1 | 2 | Compressor on/off. |
| `energy_consumption` | Float | 5.42 – 58.18 | 852 | Continuous. Unit unknown. |

### E. Target Field

| Column | Type | Values | Distribution | Notes |
|--------|------|--------|--------------|-------|
| `target` | Binary | 0, 1 | 0: 69.4%, 1: 30.6% | Ground-truth label. 0=Normal, 1=Abnormal. |

### F. Potentially Leakage-Prone Metadata

| Column | Type | Values | Leakage Risk | Evidence |
|--------|------|--------|--------------|----------|
| `event_type` | Categorical (5) | normal, blockage, degradation, leak, surge | **CRITICAL** | **Perfectly predicts target**: `normal` → target=0 (100%), all others → target=1 (100%). Using this as input would be direct label leakage. |
| `alarm_triggered` | Binary | 0, 1 | **HIGH** | `alarm_triggered=1` → target=1 (100%). `alarm_triggered=0` includes both targets (694 normal, 90 abnormal). Strong proxy for target. |

---

## 6. Actual Model Feature Inventory (LSTM Candidate Features)

The PIPER framework uses **LSTM as the only predictive algorithm**. The following are candidate features for future LSTM input, classified by suitability.

| Column | Category | Data Type | Observed Range / Values | Encoding Needed | LSTM Candidate? | Concerns / Restrictions |
|--------|----------|-----------|------------------------|-----------------|-----------------|-------------------------|
| `pressure` | Telemetry | Float | 43.11 – 115.30 | Normalization | **YES** (Primary) | Core paper variable. Unit unknown. |
| `flow_rate` | Telemetry | Float | 1.01 – 7.53 | Normalization | **YES** (Primary) | Core paper variable. Unit unknown. |
| `temperature` | Telemetry | Float | 25.92 – 38.52 | Normalization | **YES** (Primary) | Core paper variable. Unit unknown. |
| **vibration** | Telemetry | — | **NOT PRESENT** | — | **NO** | **Absent from dataset. Must not be fabricated.** |
| `valve_status` | Equipment | Ordinal (0,1,2) | 3 values | Ordinal / One-hot | **MAYBE** | Meaning of states unknown. Low cardinality. |
| `pump_state` | Equipment | Binary (0,1) | 2 values | None (binary) | **MAYBE** | May be redundant with pump_speed=0. |
| `pump_speed` | Equipment | Float | 0.0 – 1678.8 | Normalization | **MAYBE** | Correlated with pump_state. Zero-inflated. |
| `compressor_state` | Equipment | Binary (0,1) | 2 values | None (binary) | **MAYBE** | Auxiliary equipment state. |
| `energy_consumption` | Equipment | Float | 5.42 – 58.18 | Normalization | **MAYBE** | Aggregated measure; may correlate with pump/compressor. |
| `alarm_triggered` | Metadata | Binary (0,1) | 2 values | None | **EXCLUDE** | **Leakage risk: alarm=1 ⇒ target=1 (100%)** |
| `event_type` | Metadata | Categorical (5) | 5 values | One-hot | **EXCLUDE** | **Critical leakage: perfectly predicts target** |
| `segment_id` | Identifier | Categorical (50) | 50 values | Embedding / One-hot | **NO** | Grouping key, not predictive feature. Use for sequence grouping only. |
| `timestamp` | Timestamp | Datetime string | 17 / 1000 [UPDATE] | Temporal encoding | **NO** | Used for ordering/windowing, not direct input. **[SECONDS-PRECISION]** Reconstructed timestamps not sensor-verified. |
| `target` | Target | Binary (0,1) | 2 values | — | **NO** | Label column. |

### Summary of Candidate Features for LSTM

**Primary (paper-mandated, available):**
1. `pressure`
2. `flow_rate`
3. `temperature`

**Primary (paper-mandated, MISSING):**
4. `vibration` — **NOT AVAILABLE**

**Auxiliary equipment-state (candidate, pending leakage/temporal audit):**
5. `valve_status`
6. `pump_state`
7. `pump_speed`
8. `compressor_state`
9. `energy_consumption`

**Excluded due to leakage:**
- `event_type` — perfectly determines target
- `alarm_triggered` — alarm=1 deterministically implies target=1

**Not model inputs (structural):**
- `segment_id` — grouping identifier
- `timestamp` — temporal index
- `target` — label

---

## 7. Findings and Limitations

### 7.1 Confirmed Observations (Minute-Precision)

1. **File integrity confirmed** — SHA-256 matches Phase 1.1 exactly; file unmodified.
2. **1,000 complete rows** — zero missing values across all 13 columns.
3. **50 pipeline segments** — `segment_id` ranges 1–50, each with 8–16 unique timestamps (mean 11.84).
4. **17 unique timestamps** — spanning only 16 minutes (2024-01-01 00:00 to 00:16).
5. **Three primary telemetry variables present** — pressure, flow_rate, temperature.
6. **Vibration is absent** — not a column in the CSV; cannot be used for LSTM input.
7. **Target is binary** — 69.4% normal (0), 30.6% abnormal (1).
8. **Five equipment-state variables** — valve_status, pump_state, pump_speed, compressor_state, energy_consumption.
9. **Two leakage-prone metadata fields** — `event_type` (critical), `alarm_triggered` (high).
10. **Duplicate timestamp keys** — 270 `(segment_id, timestamp)` groups contain 678 rows (multiple readings per segment per timestamp).
11. **Target conflicts in duplicates** — 130 of 270 duplicate groups have conflicting target labels (both 0 and 1 for same segment-timestamp).

### 7.2 Seconds-Precision Updates [UPDATE]

1. **Timestamp reconstruction** — Seconds assigned from CSV row order within each minute (first 16 minutes: 60 rows → seconds 01–60; final minute: 41 rows → seconds 00–40).
2. **Global 1-second continuity** — 1,000 unique timestamps form continuous sequence 00:00:01 to 00:16:40. **This is synthetic** — reflects CSV row order, not sensor sampling.
3. **Repeated groups eliminated** — 270 duplicate `(segment_id, timestamp)` groups reduced to 0. The 130 target conflicts are "resolved" by assigning different seconds, but **underlying label ambiguity remains**.
4. **Per-segment gaps unchanged** — Despite unique timestamps, per-segment intervals remain highly irregular (1–339 sec gaps, median 12–74.5 sec). **No segment has consistent 1-second sampling.**

### 7.3 Interpretations

1. **Units are unknown** — No documentation specifies units for pressure, flow_rate, temperature, pump_speed, or energy_consumption. Values are consistent with typical industrial ranges (pressure ~psi/bar, temperature ~°C, flow_rate ~m³/h) but this is inference only.
2. **`valve_status` ordinal meaning unknown** — Values 0, 1, 2 could represent closed/partial/open or other states.
3. **`pump_speed` zero-inflation** — 0.0 values correspond to `pump_state=0`; non-zero values to `pump_state=1`. Redundancy likely.
4. **Temporal structure is irregular** — Multiple readings per timestamp per segment (minute-precision); not a regular time series. **[SECONDS-PRECISION]** Unique timestamps but per-segment gaps highly irregular (1–339 sec), no segment has consistent 1-second sampling.

### 7.4 Unverified Assumptions

1. The dataset originates from the claimed Kaggle source (https://www.kaggle.com/datasets/zara2099/scada-pipeline-operations-dataset) — **not independently verified**.
2. `target` represents genuine ground truth — Phase 1.1 provenance was FAIL; label generation process unknown.
3. The 16-minute window represents a snapshot, not continuous monitoring.
4. `segment_id` maps 1:1 to physical sensors — not documented.
5. **[SECONDS-PRECISION]** Reconstructed timestamps represent actual sensor sampling times — **NOT SUPPORTED**; reconstruction is based on CSV row order assumption.

### 7.5 Outstanding Questions for Phase 1.3 and 1.4

| Question | Phase |
|----------|-------|
| What is the exact timestamp parsing format and timezone? | 1.3 |
| Why are there multiple readings per `(segment_id, timestamp)`? Are they multi-sensor, repeated samples, or export duplicates? | 1.3 |
| How should duplicate-timestamp groups be handled for temporal sequencing? | 1.3 |
| Can segments be synchronized across timestamps? | 1.3 |
| What is the label generation process for `target`? | 1.4 |
| Why does `event_type` perfectly determine `target`? Is `event_type` a post-hoc label? | 1.4 |
| Why does `alarm_triggered=1` always coincide with `target=1`? | 1.4 |
| How to handle 130 duplicate groups with conflicting target labels? | 1.4 |
| Are equipment-state variables (`valve_status`, etc.) causes or effects of abnormal events? | 1.4 |
| **[SECONDS-PRECISION]** What is the sensor sampling rate? Are reconstructed seconds accurate? | 1.3 |
| **[SECONDS-PRECISION]** How to handle target conflicts now hidden by unique-second assignment? | 1.4 |

### 7.5 Schema Sufficiency for Next Phase

**YES** — The schema is sufficiently documented to begin Phase 1.3 (Timestamp and Segment Audit). All columns are inventoried, classified, and leakage risks identified. The critical issues (duplicate timestamps, target conflicts, leakage fields) are documented and ready for investigation. **[SECONDS-PRECISION]** The reconstruction has changed the temporal semantics; Phase 1.3 audit must be revisited.

---

## 8. Phase 1.2 Completion Status

| Criterion | Minute-Precision | Seconds-Precision [UPDATE] |
|-----------|------------------|----------------------------|
| Dataset file inspected | ✅ PASS | ✅ PASS |
| Hash verified against Phase 1.1 | ✅ PASS | N/A (different version) |
| All columns inventoried | ✅ PASS | ✅ PASS |
| Data types and null counts recorded | ✅ PASS | ✅ PASS |
| Unique segments, timestamps, targets counted | ✅ PASS | ✅ PASS |
| Every column classified | ✅ PASS | ✅ PASS |
| LSTM feature candidates identified | ✅ PASS | ✅ PASS |
| Missing vibration explicitly addressed | ✅ PASS | ✅ PASS |
| Units documented only when supported | ✅ PASS | ✅ PASS |
| Data dictionary generated (separate file) | ✅ PASS | ✅ PASS |
| JSON report generated (separate file) | ✅ PASS | ✅ PASS |
| Raw CSV unchanged | ✅ PASS | ✅ PASS |
| No model trained, no source code modified | ✅ PASS | ✅ PASS |

**Phase 1.2 Verdict: PASS** (both versions documented)

---

## Appendix: Timestamp Reconstruction Convention [SECONDS-PRECISION]

The seconds-precision dataset was created by assigning seconds based on CSV row order within each minute group:

- **Minutes 00:00–00:15:** 60 rows each → seconds 01–60 (minute 00:00 has 59 rows, seconds 01–59)
- **Minute 00:16:** 41 rows → seconds 00–40
- **Total:** 59 + 15×60 + 41 = 1,000 rows

**This is a derived timestamp reconstruction. It is NOT independently verified sensor-recorded time. The globally continuous 1-second sequence is an artifact of the reconstruction method.**