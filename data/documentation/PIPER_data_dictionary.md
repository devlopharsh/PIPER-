# PIPER Data Dictionary — `scada_pipeline.csv`

**Source File:** `data/raw/scada_pipeline.csv`  
**Rows:** 1,000 | **Columns:** 13 | **Missing Values:** 0  
**Hash (SHA-256):** `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`  
**Generated:** 2026-11-05

---

## Column-Level Data Dictionary

| Column Name | Data Type | Category | Description | Observed Values / Range | Unit | Model Input Status | Notes |
|---|---|---|---|---|---|---|---|
| `timestamp` | String (datetime format) | Timestamp | Event timestamp for the reading. Multiple rows share the same timestamp within and across segments. | 17 unique values: `1/1/2024 0:00` through `1/1/2024 0:16` (16-minute span) | Unknown / Not documented | **Exclude** (temporal index only) | Format: `%m/%d/%Y %H:%M`. Not a regular time series — multiple readings per segment per timestamp. Use for ordering/windowing. |
| `segment_id` | Integer | Identifier | Pipeline segment / sensor location identifier. 50 unique segments. | 1 – 50 (50 distinct) | N/A | **Exclude** (grouping key) | Use to group sequences per segment. Not a direct predictive feature. |
| `pressure` | Float | Telemetry (Primary) | Pipeline pressure reading. One of three paper-mandated primary telemetry variables. | 43.11 – 115.30 (833 distinct values) | Unknown / Not documented | **Candidate** (Primary) | Core LSTM input per paper. Requires normalization. Unit not documented (likely psi or bar). |
| `flow_rate` | Float | Telemetry (Primary) | Pipeline flow rate reading. One of three paper-mandated primary telemetry variables. | 1.01 – 7.53 (312 distinct values) | Unknown / Not documented | **Candidate** (Primary) | Core LSTM input per paper. Requires normalization. Unit not documented (likely m³/h or similar). |
| `temperature` | Float | Telemetry (Primary) | Pipeline temperature reading. One of three paper-mandated primary telemetry variables. | 25.92 – 38.52 (553 distinct values) | Unknown / Not documented | **Candidate** (Primary) | Core LSTM input per paper. Requires normalization. Unit not documented (likely °C). |
| `valve_status` | Integer (ordinal) | Equipment State | Valve operational status. Meaning of ordinal values not documented. | 0, 1, 2 (3 distinct) | N/A | **Candidate** (Auxiliary, pending audit) | Low cardinality ordinal. May need one-hot or ordinal encoding. Semantic meaning unknown. |
| `pump_state` | Integer (binary) | Equipment State | Pump on/off state. | 0 (off), 1 (on) | N/A | **Candidate** (Auxiliary, pending audit) | Binary. May be redundant with `pump_speed=0`. |
| `pump_speed` | Float | Equipment State | Pump rotational speed. Zero when pump off. | 0.0 – 1678.8 (643 distinct) | Unknown / Not documented | **Candidate** (Auxiliary, pending audit) | Continuous when pump on. Zero-inflated (correlates with pump_state). Unit likely RPM. |
| `compressor_state` | Integer (binary) | Equipment State | Compressor on/off state. | 0 (off), 1 (on) | N/A | **Candidate** (Auxiliary, pending audit) | Binary. Auxiliary equipment. |
| `energy_consumption` | Float | Equipment State | Total energy consumption reading. | 5.42 – 58.18 (852 distinct) | Unknown / Not documented | **Candidate** (Auxiliary, pending audit) | Aggregated measure. May correlate with pump/compressor states. Unit likely kW. |
| `alarm_triggered` | Integer (binary) | Metadata (Leakage-Prone) | Alarm status flag. **Critical leakage risk**: `alarm_triggered=1` ⇒ `target=1` in 100% of cases (216/216). | 0 (784), 1 (216) | N/A | **Exclude** (Leakage) | Using as input would leak target information. `alarm=0` includes both targets. |
| `event_type` | String (categorical) | Metadata (Leakage-Prone) | Event classification. **Critical leakage risk**: perfectly predicts target — `normal`→0 (100%), all others→1 (100%). | `normal` (694), `degradation` (135), `leak` (65), `surge` (61), `blockage` (45) | N/A | **Exclude** (Leakage) | Post-event or deterministic label. Using as input would be direct label leakage. |
| `target` | Integer (binary) | Target | Ground-truth label for abnormal condition detection. | 0 (Normal, 694), 1 (Abnormal, 306) | N/A | **Target** (Label) | Binary classification target. 0=Normal operation, 1=Abnormal event. |

---

## Summary Tables

### Primary Telemetry (Paper-Mandated for LSTM)

| Variable | Column | Present? | Range | Unit |
|---|---|---|---|---|
| Pressure | `pressure` | ✅ Yes | 43.11 – 115.30 | Unknown |
| Flow Rate | `flow_rate` | ✅ Yes | 1.01 – 7.53 | Unknown |
| Temperature | `temperature` | ✅ Yes | 25.92 – 38.52 | Unknown |
| Vibration | — | ❌ **No** | — | — |

### Equipment-State Variables (Auxiliary)

| Variable | Column | Type | Values | Unit |
|---|---|---|---|---|
| Valve Status | `valve_status` | Ordinal | 0, 1, 2 | N/A |
| Pump State | `pump_state` | Binary | 0, 1 | N/A |
| Pump Speed | `pump_speed` | Continuous | 0.0 – 1678.8 | Unknown |
| Compressor State | `compressor_state` | Binary | 0, 1 | N/A |
| Energy Consumption | `energy_consumption` | Continuous | 5.42 – 58.18 | Unknown |

### Leakage-Prone Fields (Exclude from Model Inputs)

| Field | Leakage Severity | Evidence |
|---|---|---|
| `event_type` | **Critical** | `normal`→target=0 (100%), all others→target=1 (100%) |
| `alarm_triggered` | **High** | `alarm=1`→target=1 (100%), `alarm=0`→mixed |

### Structural Fields (Not Model Inputs)

| Field | Role |
|---|---|
| `segment_id` | Sequence grouping identifier (50 segments) |
| `timestamp` | Temporal index for windowing/ordering |
| `target` | Label / ground truth |

---

## Important Notes

1. **No units are documented** in the source file or any project documentation. All unit assignments above are inferences from value ranges only and must not be treated as authoritative.

2. **Vibration is absent** — The paper specifies four primary telemetry variables (pressure, flow rate, temperature, vibration). This dataset contains only three. Vibration must not be fabricated, imputed, or included in the experimental feature list.

3. **`event_type` and `alarm_triggered` are leakage hazards** — Both fields deterministically or near-deterministically reveal the target. They must be excluded from any model input and flagged for Phase 1.4 leakage audit.

4. **Duplicate timestamps** — 270 `(segment_id, timestamp)` groups contain multiple rows (678 total). 130 of these groups have conflicting target labels. This must be resolved in Phase 1.3/1.4 before temporal sequence construction.

5. **Temporal span is 16 minutes** — Only 17 unique timestamps across all segments. Insufficient for meaningful LSTM sequence modeling with standard window sizes. This is a fundamental dataset limitation.

6. **Label provenance unknown** — Phase 1.1 provenance verdict was FAIL. The `target` column's generation process is not documented. It may be synthetic, simulated, or derived from `event_type`/`alarm_triggered`.

---

## Encoding Recommendations (for future phases)

| Column | Recommended Encoding | Rationale |
|---|---|---|
| `pressure`, `flow_rate`, `temperature`, `pump_speed`, `energy_consumption` | Standardization (z-score) or Min-Max | Continuous telemetry; LSTM expects normalized inputs |
| `valve_status` | One-hot (3 classes) or Ordinal (0,1,2) | Ordinal meaning unknown; one-hot safer |
| `pump_state`, `compressor_state` | Binary (0/1) | Already binary |
| `segment_id` | Not encoded as feature; use for grouping | 50 segments — too many for one-hot; use for sequence separation |
| `timestamp` | Not encoded as feature; use for sorting/windowing | Convert to datetime for ordering; cyclic time features only if regular |
| `event_type`, `alarm_triggered` | **Do not encode** — exclude | Leakage |
| `target` | Label (0/1) | Binary classification target |