# PIPER Phase 1.4 — Target and Leakage Audit

**Dataset:** `data/raw/scada_pipeline.csv`  
**SHA-256:** `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`  
**Phase 1.1–1.3 Hash Match:** ✅ CONFIRMED — file unchanged  
**Audit Date:** 2026-11-05

---

## 1. Audit Objective

This audit investigates the relationship between the target label (`target`) and all available dataset fields, identifies direct and indirect target leakage, quantifies label conflicts in repeated observations, and establishes a documented leakage-exclusion policy for the future PIPER LSTM model.

---

## 2. Dataset Integrity and Target Validation

### 2.1 Target Column Validation

| Metric | Value |
|--------|-------|
| Total rows | 1,000 |
| Null/invalid target values | 0 |
| Unique target values | {0, 1} |
| Target = 0 (Normal) | 694 (69.4%) |
| Target = 1 (Abnormal) | 306 (30.6%) |
| Binary target | ✅ YES |

**Semantic meaning of target:** The dataset documentation does not establish whether `target` represents genuine ground truth from a physical pipeline, simulated anomalies, or synthetically injected labels. Phase 1.1 provenance verdict was FAIL. The label generation process is undocumented.

### 2.2 Per-Segment Target Distribution

| Segment Range | Abnormal Rate Range | Notes |
|---------------|---------------------|-------|
| All 50 segments | 9.1% – 61.1% | Segment 25: 9.1% (2/22), Segment 11: 61.1% (11/18) |
| Mean per segment | 30.6% | Matches global distribution |

No segment has 0% or 100% abnormal rate — all segments contain both classes.

### 2.3 Per-Timestamp Target Distribution

| Timestamp | Abnormal Count | Total | Abnormal Rate |
|-----------|----------------|-------|---------------|
| 0:00 | 13 | 60 | 21.7% |
| 0:01 | 25 | 60 | 41.7% |
| 0:02 | 21 | 60 | 35.0% |
| 0:03 | 17 | 60 | 28.3% |
| 0:04 | 20 | 60 | 33.3% |
| 0:05 | 26 | 60 | 43.3% |
| 0:06 | 21 | 60 | 35.0% |
| 0:07 | 20 | 60 | 33.3% |
| 0:08 | 19 | 60 | 31.7% |
| 0:09 | 14 | 60 | 23.3% |
| 0:10 | 20 | 60 | 33.3% |
| 0:11 | 14 | 60 | 23.3% |
| 0:12 | 17 | 60 | 28.3% |
| 0:13 | 16 | 60 | 26.7% |
| 0:14 | 13 | 60 | 21.7% |
| 0:15 | 16 | 60 | 26.7% |
| 0:16 | 14 | 40 | 35.0% |

Abnormal rate varies across timestamps (21.7%–43.3%), indicating non-stationary event distribution.

---

## 3. Target Cross-Tabulation

### 3.1 Categorical Fields

#### `event_type` vs `target` (Critical)

| event_type | target=0 | target=1 | Total | Target=1 Rate |
|------------|----------|----------|-------|---------------|
| normal | 694 | 0 | 694 | 0.0% |
| blockage | 0 | 45 | 45 | 100.0% |
| degradation | 0 | 135 | 135 | 100.0% |
| leak | 0 | 65 | 65 | 100.0% |
| surge | 0 | 61 | 61 | 100.0% |
| **Total** | **694** | **306** | **1,000** | **30.6%** |

**Finding:** `event_type` **perfectly determines `target`**. Every non-"normal" event_type maps to target=1; "normal" maps to target=0.

#### `alarm_triggered` vs `target` (Critical)

| alarm_triggered | target=0 | target=1 | Total | Target=1 Rate |
|-----------------|----------|----------|-------|---------------|
| 0 | 694 | 90 | 784 | 11.5% |
| 1 | 0 | 216 | 216 | 100.0% |
| **Total** | **694** | **306** | **1,000** | **30.6%** |

**Finding:** `alarm_triggered = 1` **perfectly implies `target = 1`** (216/216). `alarm_triggered = 0` includes both classes (694 normal, 90 abnormal). One-direction perfect leakage.

#### Equipment Status Fields

| Field | Target=1 Rate by Category | Perfect Mapping? |
|-------|---------------------------|------------------|
| `valve_status` (0,1,2) | 0: 32.1%, 1: 30.2%, 2: 30.4% | ❌ No |
| `pump_state` (0,1) | 0: 33.4%, 1: 29.3% | ❌ No |
| `compressor_state` (0,1) | 0: 29.8%, 1: 31.2% | ❌ No |

No equipment status field perfectly determines target.

### 3.2 Numeric Fields by Target Class

| Variable | Target=0 (Normal) | Target=1 (Abnormal) | Distribution Overlap |
|----------|-------------------|---------------------|----------------------|
| `pressure` | mean=74.85, median=74.88, std=4.93, range=[61.37, 88.82] | mean=79.26, median=77.32, std=15.02, range=[43.11, 115.30] | **Substantial** — abnormal has wider spread, lower min, higher max |
| `flow_rate` | mean=4.52, median=4.54, std=0.49, range=[3.07, 6.15] | mean=3.99, median=4.26, std=1.37, range=[1.01, 7.53] | **Substantial** — abnormal has wider spread |
| `temperature` | mean=32.11, median=32.13, std=2.00, range=[27.08, 38.52] | mean=32.07, median=32.12, std=1.93, range=[25.92, 37.26] | **Nearly identical** — minimal separation |
| `pump_speed` | mean=979.3, median=1336.8, std=644.6, range=[0, 1678.8] | mean=841.8, median=1132.6, std=622.6, range=[0, 1623.7] | **Substantial** — zero-inflated, pump-off correlated with target |
| `energy_consumption` | mean=25.78, median=25.25, std=8.41, range=[5.42, 39.49] | mean=29.28, median=30.19, std=10.60, range=[7.02, 58.18] | **Substantial** — abnormal slightly higher on average |

**No deterministic threshold** found in any numeric variable that perfectly separates target classes.

---

## 4. Event Type Leakage Investigation

### 4.1 Evidence

- **Unique values:** `normal`, `blockage`, `degradation`, `leak`, `surge` (5 categories)
- **Perfect deterministic mapping:** Each event_type category contains **only one target class**
- **No mixed categories:** Zero event_type categories contain both target=0 and target=1

### 4.2 Temporal Availability Assessment

| Aspect | Assessment |
|--------|------------|
| Prediction-time availability | **UNKNOWN** — No documentation on when `event_type` is generated |
| Semantic meaning | Appears to be a **post-hoc event classification** (leak, blockage, degradation, surge are outcomes) |
| Generation process | Undocumented — could be manual annotation, automated detection, or synthetic injection |

### 4.3 Classification

**EXCLUDE — DIRECT LEAKAGE**

**Reasoning:** `event_type` perfectly determines `target` (100% deterministic mapping). Even if available at prediction time, it encodes the outcome. Using it as input would be direct label leakage.

**Condition for reconsideration:** Only if proven to be a pre-event sensor-derived feature with documented generation timing (highly unlikely given semantic names).

---

## 5. Alarm Triggered Leakage Investigation

### 5.1 Evidence

| Metric | Value |
|--------|-------|
| `alarm_triggered = 1` → `target = 1` | 216/216 (100%) |
| `alarm_triggered = 1` → `target = 0` | 0/216 (0%) |
| `alarm_triggered = 0` → `target = 1` | 90/784 (11.5%) |
| `alarm_triggered = 0` → `target = 0` | 694/784 (88.5%) |
| P(target=1 \| alarm=1) | **1.000** |
| P(target=1 \| alarm=0) | **0.115** |

### 5.2 Disagreement Analysis

- `target=1, alarm=0`: **90 rows (9.0%)** — all are `event_type=degradation`
- `target=0, alarm=1`: **0 rows (0.0%)** — never occurs

### 5.3 Temporal Availability Assessment

| Aspect | Assessment |
|--------|------------|
| Prediction-time availability | **UNKNOWN** — No documentation on alarm system timing |
| Semantic meaning | **Alarm outcome** — triggered *by* an abnormal condition |
| Generation process | Undocumented — likely post-detection |

### 5.4 Classification

**EXCLUDE — DIRECT LEAKAGE**

**Reasoning:** `alarm_triggered = 1` perfectly implies `target = 1`. It is an alarm *outcome*, not a pre-event predictor. The one-direction perfect relationship (alarm=1 ⇒ target=1) means any model seeing `alarm=1` would trivially know the target.

**Condition for reconsideration:** Only if proven to be a predictive early-warning alarm with documented lead time (no evidence supports this).

---

## 6. Equipment Status and Indirect Leakage Audit

### 6.1 Summary of Equipment Fields

| Field | Type | Target=1 Rate Variation | Varies in Repeated Groups? | Classification |
|-------|------|------------------------|---------------------------|----------------|
| `valve_status` | Ordinal (0,1,2) | 30.2%–32.1% (narrow) | Yes (53.3% of repeated groups) | **Category 3** |
| `pump_state` | Binary (0,1) | 29.3%–33.4% | Yes (52.2%) | **Category 3** |
| `compressor_state` | Binary (0,1) | 29.8%–31.2% | Yes (57.8%) | **Category 3** |
| `pump_speed` | Continuous (0–1678) | Mean differs (979 vs 842) | Yes (94.8%) | **Category 3** |
| `energy_consumption` | Continuous (5.4–58.2) | Mean differs (25.8 vs 29.3) | Yes (100%) | **Category 3** |

### 6.2 Relationship with `event_type` and `alarm_triggered`

Equipment status fields show modest association with `event_type` and `alarm_triggered` but **no perfect mapping**. For example:

- `valve_status=2` has slightly higher alarm rate (23.2% vs 20.4% for status 0)
- `pump_state=0` has slightly higher alarm rate (23.5% vs 20.8% for state 1)

These are **predictive associations**, not leakage — the equipment states could plausibly be causal factors or concurrent indicators.

### 6.3 Repeated-Group Variation (from Phase 1.3)

All equipment fields **vary within repeated `(segment_id, timestamp)` groups**:
- `valve_status`: 144/270 groups (53.3%)
- `pump_state`: 141/270 (52.2%)
- `compressor_state`: 156/270 (57.8%)
- `pump_speed`: 256/270 (94.8%)
- `energy_consumption`: 270/270 (100%)

This confirms they are **measurement variables**, not static labels.

### 6.4 Classification

**CATEGORY 3 — Potentially admissible candidates** (pending temporal validation)

**Reasoning:** No deterministic target encoding found. Associations are statistical, not perfect. Prediction-time availability undocumented — must be established before use.

---

## 7. Numeric Telemetry Leakage Analysis

### 7.1 Primary Telemetry (Paper-Mandated)

| Variable | Target=0 Distribution | Target=1 Distribution | Leakage Assessment |
|----------|----------------------|----------------------|-------------------|
| `pressure` | N(74.85, 4.93), range [61.4, 88.8] | N(79.26, 15.02), range [43.1, 115.3] | **No deterministic encoding** — substantial overlap, wider abnormal spread |
| `flow_rate` | N(4.52, 0.49), range [3.07, 6.15] | N(3.99, 1.37), range [1.01, 7.53] | **No deterministic encoding** — substantial overlap |
| `temperature` | N(32.11, 2.00), range [27.1, 38.5] | N(32.07, 1.93), range [25.9, 37.3] | **No deterministic encoding** — nearly identical distributions |

### 7.2 Equipment Telemetry

| Variable | Assessment |
|----------|------------|
| `pump_speed` | Zero-inflated (pump off = 0). Mean differs but distributions overlap substantially. Correlated with `pump_state`. |
| `energy_consumption` | Aggregated measure. Abnormal slightly higher mean but substantial overlap. |

### 7.3 Repeated-Group Variation

All primary telemetry **vary in 100% (pressure, temperature) or 99.3% (flow_rate)** of repeated groups — confirming they are sensor measurements, not static labels.

### 7.4 Classification

**CATEGORY 3 — Potentially admissible candidates** (pending temporal validation)

**Reasoning:** These are the core physical telemetry variables. No deterministic target boundaries found. Distributions overlap substantially. Must establish prediction-time availability and units.

---

## 8. Repeated-Group Target Conflict Analysis

### 8.1 Conflict Statistics

| Metric | Value |
|--------|-------|
| Total `(segment_id, timestamp)` groups | 592 |
| Single-row groups | 322 (54.4%) |
| Repeated groups (>1 row) | 270 (45.6%) |
| Rows in repeated groups | 678 (67.8%) |
| Groups with target=0 only | 332 |
| Groups with target=1 only | 130 |
| **Groups with conflicting targets (0 and 1)** | **130** |
| **Rows in conflicting groups** | **349 (34.9% of dataset)** |

### 8.2 Conflict Characteristics

In the 130 conflicting groups:
- **All 130 have `event_type` conflicts** — mix of "normal" and abnormal types
- **All 130 have `alarm_triggered` conflicts** — mix of 0 and 1
- **Telemetry varies substantially** — pressure range up to 49.9, flow_rate up to 4.5, temperature up to 7.9°C
- **Equipment states vary** — pump_state, compressor_state, valve_status all flip

### 8.3 Target–Alarm Disagreement

| Disagreement Type | Count | Percentage | Event Types |
|-------------------|-------|------------|-------------|
| `target=1, alarm=0` | 90 | 9.0% | All `degradation` |
| `target=0, alarm=1` | 0 | 0.0% | — |

**Critical:** 90 abnormal rows have `alarm_triggered=0`, all classified as `degradation`. This means `alarm_triggered` **misses all degradation events**.

### 8.4 Impact on Supervised Sequence Modeling

1. **Label ambiguity:** 34.9% of rows belong to groups where the "correct" label is undefined
2. **Sequence contamination:** Any window containing a conflicting timestamp inherits label uncertainty
3. **No resolution policy:** Cannot arbitrarily choose majority, max, first, or last without source justification
4. **Leakage fields disagree:** `event_type` and `alarm_triggered` conflict with `target` within same timestamp

---

## 9. Direct and Indirect Target Encoding

### 9.1 Perfect Deterministic Encoding

| Field(s) | Encoding Type | Evidence |
|----------|---------------|----------|
| `event_type` | **Perfect bidirectional** | Each category maps to exactly one target class |
| `alarm_triggered` | **Perfect one-direction** | alarm=1 ⇒ target=1 (100%); alarm=0 ⇒ mixed |

### 9.2 Joint Encoding

| Combination | Target Revelation |
|-------------|-------------------|
| `event_type` + `alarm_triggered` | Jointly reveal target with certainty; `degradation` with alarm=0 is the only ambiguous case |

### 9.3 Suspicious Patterns Checked

- **Numeric thresholds:** No deterministic pressure/flow/temperature threshold perfectly separates classes
- **Equipment combinations:** No valve+pump+compressor combination perfectly predicts target
- **Segment/time shortcuts:** No segment_id or timestamp pattern perfectly predicts target
- **Single-field encoding:** Only `event_type` and `alarm_triggered` (one-direction)

---

## 10. Prediction-Time Availability Assessment

| Field | Availability Category | Rationale |
|-------|----------------------|-----------|
| `pressure` | **B** — Potentially available, undocumented | Physical sensor; no timing docs |
| `flow_rate` | **B** — Potentially available, undocumented | Physical sensor; no timing docs |
| `temperature` | **B** — Potentially available, undocumented | Physical sensor; no timing docs |
| `valve_status` | **B** — Potentially available, undocumented | Equipment status; no timing docs |
| `pump_state` | **B** — Potentially available, undocumented | Equipment status; no timing docs |
| `pump_speed` | **B** — Potentially available, undocumented | Equipment sensor; no timing docs |
| `compressor_state` | **B** — Potentially available, undocumented | Equipment status; no timing docs |
| `energy_consumption` | **B** — Potentially available, undocumented | Aggregated meter; no timing docs |
| `alarm_triggered` | **C** — Likely post-event | Alarm outcome; perfect one-direction leakage |
| `event_type` | **D** — Derived from label | Post-hoc classification; perfect leakage |
| `segment_id` | **A** — Clearly available | Static identifier |
| `timestamp` | **A** — Clearly available | Observation time |

**Key concern:** All physical telemetry and equipment fields are **Category B** — availability is plausible but undocumented. No field has documented prediction-time latency or sampling guarantee.

---

## 11. Frozen Leakage-Exclusion List

| Field | Category | Evidence | Reconsideration Condition |
|-------|----------|----------|---------------------------|
| `event_type` | **Cat 1: Mandatory Exclusion** | Perfectly determines target (100% deterministic) | Proven pre-event sensor input with documented timing |
| `alarm_triggered` | **Cat 1: Mandatory Exclusion** | P(target=1\|alarm=1)=1.0; alarm outcome reveals target | Proven predictive early-warning with documented lead time |
| `valve_status` | **Cat 3: Potentially Admissible** | Statistical association only; varies in repeated groups | Document prediction-time availability |
| `pump_state` | **Cat 3: Potentially Admissible** | Statistical association only; varies in repeated groups | Document prediction-time availability |
| `pump_speed` | **Cat 3: Potentially Admissible** | Continuous sensor; zero-inflated; varies in repeated groups | Document availability; address zero-inflation |
| `compressor_state` | **Cat 3: Potentially Admissible** | Statistical association only; varies in repeated groups | Document prediction-time availability |
| `energy_consumption` | **Cat 3: Potentially Admissible** | Aggregated measure; overlapping distributions | Document prediction-time availability |
| `pressure` | **Cat 3: Potentially Admissible** | Primary telemetry; no deterministic encoding; overlaps | Document sensor availability, units, latency |
| `flow_rate` | **Cat 3: Potentially Admissible** | Primary telemetry; no deterministic encoding; overlaps | Document sensor availability, units, latency |
| `temperature` | **Cat 3: Potentially Admissible** | Primary telemetry; no deterministic encoding; overlaps | Document sensor availability, units, latency |
| `segment_id` | **Cat 4: Metadata Only** | Static grouping key | N/A |
| `timestamp` | **Cat 4: Metadata Only** | Chronological index | N/A |
| `target` | **Cat 4: Metadata Only** | Supervision label | N/A |

**Policy:** This list is **frozen as a conservative Phase 1.4 audit decision**. Category 3 fields are not permanently excluded — they require temporal validation before Phase 2.

---

## 12. Provisional LSTM Input Policy

### 12.1 Mandatory Excluded Features
- `event_type` (Category 1)
- `alarm_triggered` (Category 1)

### 12.2 Excluded Pending Further Evidence
- None (all remaining are Category 3 or 4)

### 12.3 Candidate Physical Telemetry Inputs
1. `pressure`
2. `flow_rate`
3. `temperature`
4. **`vibration` — MISSING FROM DATASET** (paper-mandated, cannot be fabricated)

### 12.4 Candidate Equipment-State Inputs
1. `valve_status`
2. `pump_state`
3. `pump_speed`
4. `compressor_state`
5. `energy_consumption`

### 12.5 Metadata-Only Fields
- `segment_id` (grouping key)
- `timestamp` (chronological index)
- `target` (label)

### 12.6 Target Definition Status
**NOT SUPPORTED** — The current evidence does not support a defensible prediction target definition because:
- 130 repeated groups have conflicting target labels (34.9% of rows)
- `alarm_triggered` misses all `degradation` events (90 rows)
- `event_type` is a post-hoc label, not a pre-event input
- 16-minute span with no segment synchronization prevents temporal forecasting
- Label provenance unknown (Phase 1.1 FAIL)

### 12.7 Unresolved Questions Before Phase 2/Model Development

| # | Question | Required Resolution |
|---|----------|---------------------|
| 1 | Label policy for 130 conflicting repeated groups | Source-backed rule (exclude? flag? majority?) |
| 2 | Prediction-time availability of all Cat 3 fields | Vendor/docs/specs with latency guarantees |
| 3 | Physical units for pressure, flow, temperature, etc. | Calibration documentation |
| 4 | Vibration data source | Must obtain or formally exclude from paper scope |
| 5 | Segment synchronization mechanism | Evidence of coordinated sampling |
| 6 | Prediction task definition | Current-state classification? 1-min ahead? Horizon? |
| 7 | Label generation provenance | Phase 1.1 provenance gap |

---

## 13. Limitations and Unresolved Questions

1. **Target conflicts in 34.9% of rows** — No source-backed resolution policy exists
2. **All equipment/telemetry availability undocumented** — Category B assessment is speculative
3. **Vibration absent** — Paper requires 4 telemetry variables; only 3 available
4. **16-minute duration** — Insufficient for any meaningful forecast horizon
5. **No segment synchronization** — Multi-segment sequences not feasible
6. **Label provenance unknown** — Phase 1.1 FAIL means target may be synthetic
7. **`degradation` events invisible to alarm system** — 90 rows with target=1, alarm=0

---

## 14. Phase 1.4 Verdict

**PASS WITH OBSERVATIONS**

### Rationale

The audit successfully completed all tasks (A–K) and produced all five required deliverables. The leakage-exclusion list is frozen with clear evidence-based reasoning.

**However, critical observations restrict subsequent phases:**

1. **Target definition is not established** — 34.9% label conflict rate, alarm misses degradation events, post-hoc `event_type`
2. **Prediction-time availability unproven** for all candidate features (all Category B)
3. **Vibration missing** — Cannot satisfy paper's 4-variable LSTM requirement
4. **16-minute span with repeated-timestamp ambiguity** — Temporal sequence construction semantically invalid

The audit execution **PASSES**; the dataset's suitability for LSTM temporal modeling remains **NOT ESTABLISHED**. Phase 1.5 (Suitability Decision) must formally address these gaps.

---

## 15. Recommendations for Phase 1.5

1. **Formally resolve target conflicts** — Define exclusion/flagging policy for 130 conflicting groups with source justification
2. **Obtain vibration data** or formally scope paper to 3 telemetry variables
3. **Document prediction-time availability** for pressure, flow_rate, temperature (vendor specs, latency budgets)
4. **Define prediction task** explicitly: current-state classification vs. forecasting; horizon; segment-level vs. global
5. **If dataset retained**: Explicitly label as synthetic/benchmark of unknown origin in all publications
6. **If dataset replaced**: Re-run Phase 1.1–1.4 on new dataset

---

## 16. Deliverables Generated

| File | Description | Size |
|------|-------------|------|
| `PIPER_phase1_4_target_leakage_audit.md` | This report | ~18 KB |
| `PIPER_phase1_4_target_leakage_report.json` | Machine-readable cross-tabs, conflicts, classifications | ~35 KB |
| `PIPER_phase1_4_leakage_exclusion_list.csv` | Frozen field-level exclusion policy (13 fields) | ~4 KB |
| `PIPER_phase1_4_target_conflict_examples.csv` | 20 representative conflicting groups with differing fields | ~8 KB |
| `PIPER_phase1_4_feature_availability.csv` | Field-level prediction-time availability (12 fields) | ~1 KB |

---

## 17. Integrity Confirmation

✅ **Raw CSV unchanged** — SHA-256: `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a` (matches Phases 1.1–1.3)  
✅ **Phase 1.1 files preserved** — All 3 files intact  
✅ **Phase 1.2 files preserved** — All 3 files intact  
✅ **Phase 1.3 files preserved** — All 4 files intact  
✅ **No model trained** — Read-only analytical audit  
✅ **No source code modified** — Analysis scripts only  

---

## 18. Deviations from Prompt

None. All tasks (A–K) executed per specification. The "predictive tests" for numeric thresholds were descriptive (sorted transition inspection) as permitted, not model-based.