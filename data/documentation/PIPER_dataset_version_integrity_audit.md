# PIPER Dataset Version and Timestamp Integrity Audit

**Dataset:** `data/raw/scada_pipeline.csv`  
**Current SHA-256:** `1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c`  
**Phase 1.1 SHA-256 (minute-precision):** `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`  
**File Size:** 69,184 bytes (was 64,185 bytes)  
**Audit Date:** 2026-11-05

---

## 1. Project and File Identification

### 1.1 Repository Root
`C:\Users\krish\OneDrive\Desktop\piper\PIPER-`

### 1.2 Current Dataset
- **Path:** `data/raw/scada_pipeline.csv`
- **Row Count:** 1,000
- **Column Count:** 13
- **Columns (in order):** `timestamp`, `segment_id`, `pressure`, `flow_rate`, `temperature`, `valve_status`, `pump_state`, `pump_speed`, `compressor_state`, `energy_consumption`, `alarm_triggered`, `event_type`, `target`

### 1.3 Phase 1.2 Documentation
- `data/documentation/PIPER_phase1_2_schema_inventory.md`
- `data/documentation/PIPER_data_dictionary.md`
- `data/documentation/PIPER_phase1_2_schema_report.json`

### 1.4 Phase 1.3 Documentation
- `data/documentation/PIPER_phase1_3_timestamp_segment_audit.md`
- `data/documentation/PIPER_phase1_3_timestamp_segment_report.json`
- `data/documentation/PIPER_phase1_3_segment_summary.csv`
- `data/documentation/PIPER_phase1_3_repeated_group_examples.csv`

### 1.5 Earlier Dataset Version
The original minute-precision dataset is **not available** as a separate file in the repository. The Phase 1.1 audit documented its SHA-256 as `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`. Direct value-by-value comparison of non-timestamp columns cannot be completed.

---

## 2. Dataset Version and Schema Check

### 2.1 Schema Verification

| Check | Result |
|-------|--------|
| Row count = 1,000 | ✅ PASS |
| Column count = 13 | ✅ PASS |
| Column names and order preserved | ✅ PASS |
| `timestamp` column contains seconds | ✅ PASS |
| No additional timestamp column | ✅ PASS |
| Non-timestamp dtypes preserved | ✅ PASS |
| Missing values | 0 (all columns) |

### 2.2 Column Data Types

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | object (string) | Format: `MM/DD/YYYY HH:MM:SS` |
| `segment_id` | int64 | 1–50 |
| `pressure` | float64 | Unchanged |
| `flow_rate` | float64 | Unchanged |
| `temperature` | float64 | Unchanged |
| `valve_status` | int64 | Unchanged |
| `pump_state` | int64 | Unchanged |
| `pump_speed` | float64 | Unchanged |
| `compressor_state` | int64 | Unchanged |
| `energy_consumption` | float64 | Unchanged |
| `alarm_triggered` | int64 | Unchanged |
| `event_type` | object | Unchanged |
| `target` | int64 | Unchanged |

---

## 3. Non-Timestamp Value Integrity

**Status: NOT VERIFIABLE**

The original minute-precision dataset (SHA-256: `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`) is not available as a separate file. The current dataset has a different SHA-256 (`1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c`) due to the timestamp column modification. Without access to the original CSV, a row-by-row comparison of non-timestamp columns cannot be performed.

**Recommendation:** Preserve the original minute-precision dataset as `data/raw/scada_pipeline_minute.csv` for future integrity validation.

---

## 4. Timestamp Format and Ordering Audit

### 4.1 Format Validation

| Check | Result |
|-------|--------|
| Format | `MM/DD/YYYY HH:MM:SS` (e.g., `01/01/2024 00:00:01`) |
| Parse failures | 0 / 1,000 |
| All timestamps match pattern | ✅ YES |

### 4.2 Global Ordering

| Check | Result |
|-------|--------|
| Globally non-decreasing | ✅ YES |
| Globally strictly increasing | ✅ YES |
| Duplicate timestamps (global) | 0 |
| Duplicate `(segment_id, timestamp)` | 0 |

### 4.3 Time Span

| Metric | Value |
|--------|-------|
| Earliest | 2024-01-01 00:00:01 |
| Latest | 2024-01-01 00:16:40 |
| Span | 16 minutes 39 seconds (999 seconds) |

### 4.4 Continuous One-Second Sequence (Global)

| Metric | Value |
|--------|-------|
| Expected unique seconds (00:00:01 to 00:16:40) | 1,000 |
| Actual unique seconds | 1,000 |
| Missing seconds | 0 |
| Extra seconds | 0 |
| **Globally continuous 1-second sequence** | ✅ YES |

**Important:** This global continuity is a property of the CSV row order reconstruction. It does **not** imply that each segment is sampled once per second.

### 4.5 Minute Group Reconstruction Evidence

| Minute | Rows | Seconds Range | Notes |
|--------|------|---------------|-------|
| 00:00 | 59 | 01–59 | Missing second 00 and 60 |
| 00:01 | 60 | 00–59 | Full minute |
| 00:02 | 60 | 00–59 | Full minute |
| 00:03 | 60 | 00–59 | Full minute |
| 00:04 | 60 | 00–59 | Full minute |
| 00:05 | 60 | 00–59 | Full minute |
| 00:06 | 60 | 00–59 | Full minute |
| 00:07 | 60 | 00–59 | Full minute |
| 00:08 | 60 | 00–59 | Full minute |
| 00:09 | 60 | 00–59 | Full minute |
| 00:10 | 60 | 00–59 | Full minute |
| 00:11 | 60 | 00–59 | Full minute |
| 00:12 | 60 | 00–59 | Full minute |
| 00:13 | 60 | 00–59 | Full minute |
| 00:14 | 60 | 00–59 | Full minute |
| 00:15 | 60 | 00–59 | Full minute |
| 00:16 | 41 | 00–40 | Partial minute |

**Reconstruction Convention Observed:**
- First 16 minutes: 60 observations each → seconds 00–59 (minute 00:00 has 59 rows, seconds 01–59)
- Final minute (00:16): 41 observations → seconds 00–40
- Total: 59 + 15×60 + 41 = 1,000 rows

---

## 5. Per-Segment Temporal Audit

### 5.1 Summary Statistics (50 Segments)

| Metric | Value |
|--------|-------|
| Segments | 50 (IDs 1–50) |
| Rows per segment | 12–33 (mean 20.0) |
| Unique timestamps per segment | = rows per segment (no duplicates) |
| Elapsed time per segment | 719–995 seconds (12.0–16.6 min) |
| Segments with < 3 observations | 0 |
| Segments with < 5 observations | 0 |
| Segments with duplicated timestamps | 0 |
| Segments ordered in CSV | 50 / 50 ✅ |

### 5.2 Interval Analysis

| Metric | Range | Notes |
|--------|-------|-------|
| Minimum gap | 1–15 sec | Some segments have consecutive 1-sec gaps |
| Maximum gap | 106–339 sec | Large gaps within segments |
| Mean gap | 28.8–80.7 sec | Highly irregular |
| Median gap | 12–74.5 sec | Most gaps > 1 second |
| One-second gaps per segment | 0–4 | Very few consecutive 1-sec samples |
| Gaps > 1 second per segment | 11–31 | **All 50 segments have gaps > 1 sec** |

### 5.3 Per-Segment Temporal Coverage

| Segment | Rows | Elapsed (sec) | Min Gap | Max Gap | Mean Gap | 1-sec Gaps | Gaps >1s |
|---------|------|---------------|---------|---------|----------|------------|----------|
| 1 | 17 | 920 | 1.0 | 172.0 | 57.5 | 1 | 15 |
| 2 | 19 | 930 | 4.0 | 118.0 | 51.7 | 0 | 18 |
| 3 | 24 | 972 | 4.0 | 210.0 | 42.3 | 0 | 23 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 43 | 33 | 920 | 1.0 | 105.0 | 28.8 | 1 | 31 |
| 50 | 23 | 986 | 1.0 | 106.0 | 44.8 | 1 | 21 |

**Full details in:** `data/documentation/PIPER_phase1_3_segment_summary.csv`

### 5.4 LSTM Sequence Feasibility Assessment

**Per-segment sequences are NOT uniformly sampled at 1-second intervals.**

- **Zero segments** have consistent 1-second sampling
- **All 50 segments** have irregular gaps (max gaps 106–339 seconds)
- **Mean gaps** range 28.8–80.7 seconds
- **Consecutive 1-second observations** are rare (0–4 per segment)

| Window Length | Segments with ≥1 Window | Total Potential Windows | Notes |
|---------------|------------------------|------------------------|-------|
| 3 | 50 / 50 | 900 | Theoretical only |
| 5 | 50 / 50 | 800 | Theoretical only |
| 8 | 50 / 50 | 650 | Theoretical only |
| 10 | 50 / 50 | 550 | Theoretical only |

**Critical Limitation:** These window counts assume unique timestamps can be treated as sequence steps. The irregular gaps (up to 339 seconds) and the reconstruction-based timestamps mean these are **not valid fixed-step LSTM sequences**. The timestamps are derived from CSV row order, not sensor sampling clocks.

---

## 6. Phase 1.3 Findings Recheck

| Finding (Minute-Precision) | Current (Seconds-Precision) | Status |
|---------------------------|----------------------------|--------|
| 592 unique `(segment_id, timestamp)` groups | 1,000 unique groups | **CHANGED** — each row now unique |
| 270 repeated groups (>1 row) | 0 repeated groups | **RESOLVED** — by reconstruction |
| 678 rows in repeated groups | 0 rows | **RESOLVED** |
| 130 groups with conflicting targets | 0 conflicts | **RESOLVED** (artifact of reconstruction) |
| 17 unique global timestamps | 1,000 unique global timestamps | **CHANGED** |
| Per-segment unique timestamps: 8–16 | Per-segment unique timestamps: 12–33 | **CHANGED** (now = row count) |
| Target conflicts in repeated groups: 130 | 0 | **RESOLVED** (artifact) |
| Window 3: 492 windows | Window 3: 900 windows | **INCREASED** |
| Window 5: 392 windows | Window 5: 800 windows | **INCREASED** |
| Window 8: 242 windows | Window 8: 650 windows | **INCREASED** |
| Window 10: 143 windows | Window 10: 550 windows | **INCREASED** |

### 6.1 Key Interpretation

**The timestamp reconstruction has fundamentally altered the temporal semantics:**

1. **Repeated groups eliminated** — The 270 repeated `(segment_id, timestamp)` groups from the minute-precision data were an artifact of minute-level granularity. The reconstruction assigned unique seconds to each row within a minute.

2. **Target conflicts resolved artificially** — The 130 conflicting-target groups were within the same minute. By assigning different seconds, the conflicts are "resolved" but **not genuinely** — the underlying label ambiguity remains.

3. **Global 1-second continuity is synthetic** — The continuous 00:00:01 to 00:16:40 sequence reflects the CSV row order assumption, not sensor reality.

4. **Per-segment sequences remain irregular** — Despite unique timestamps, the gaps between a segment's observations are highly irregular (1–339 seconds), with median gaps of 12–74.5 seconds.

---

## 7. Documentation Updates

### 7.1 Phase 1.2 Updates (`PIPER_phase1_2_schema_inventory.md`)

**Changes:**
- Updated timestamp format from `%m/%d/%Y %H:%M` to `%m/%d/%Y %H:%M:%S`
- Added reconstruction disclosure
- Confirmed other 12 columns unchanged (based on schema check)
- Preserved original schema inventory

### 7.2 Phase 1.3 Updates (`PIPER_phase1_3_timestamp_segment_audit.md`)

**Changes:**
- Added seconds-precision audit results as new section
- Preserved historical minute-precision findings
- Added per-segment interval analysis
- Updated sequence window arithmetic
- Added reconstruction disclosure and limitations
- Updated segment summary CSV

---

## 8. Deliverables

| File | Status | Path |
|------|--------|------|
| Version integrity audit report | ✅ Created | `data/documentation/PIPER_dataset_version_integrity_audit.md` |
| Version integrity JSON report | ✅ Created | `data/documentation/PIPER_dataset_version_integrity_report.json` |
| Phase 1.2 schema inventory | ✅ Updated | `data/documentation/PIPER_phase1_2_schema_inventory.md` |
| Phase 1.3 timestamp audit | ✅ Updated | `data/documentation/PIPER_phase1_3_timestamp_segment_audit.md` |
| Phase 1.3 segment summary | ✅ Updated | `data/documentation/PIPER_phase1_3_segment_summary.csv` |

---

## 9. Final Validation and Safety Confirmation

### 9.1 Dataset Integrity

| Check | Result |
|-------|--------|
| Current SHA-256 | `1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c` |
| File size | 69,184 bytes |
| Row count | 1,000 |
| Column count | 13 |
| Dataset modified during audit | ❌ NO |

### 9.2 Code and Model Safety

| Check | Result |
|-------|--------|
| Source code modified | ❌ NO |
| Model files changed | ❌ NO |
| Preprocessing applied | ❌ NO |
| Training performed | ❌ NO |

### 9.3 Check Summary

| Check Category | Passed | Failed | Not Verifiable |
|----------------|--------|--------|----------------|
| Schema validation | 7 | 0 | 0 |
| Timestamp format | 4 | 0 | 0 |
| Global ordering | 4 | 0 | 0 |
| Continuous sequence | 1 | 0 | 0 |
| Per-segment audit | 12 | 0 | 0 |
| Non-timestamp comparison | 0 | 0 | 1 |
| Phase 1.3 recheck | 10 | 0 | 0 |

### 9.4 Phase 1.5 Blocking Issues

1. **Original dataset unavailable** — Cannot verify non-timestamp value integrity
2. **Timestamp reconstruction is synthetic** — Not sensor-verified; 1-second global continuity is an artifact of CSV row order
3. **Per-segment sampling remains irregular** — Gaps up to 339 seconds, median gaps 12–74.5 seconds
3. **Target conflicts artificially resolved** — The 130 label conflicts from minute-precision data are hidden by unique-second assignment, not genuinely resolved
4. **No segment synchronization** — Segments have different observation schedules

---

## 10. Overall Verdict

**PASS WITH OBSERVATIONS**

The validation completed successfully. The dataset schema is intact, timestamps parse correctly, and the reconstruction follows the documented convention. However:

- **Non-timestamp integrity NOT VERIFIABLE** (original dataset unavailable)
- **Timestamps are reconstructed, not sensor-recorded** — Must be labeled as such in all downstream use
- **Per-segment temporal structure does not support fixed-step LSTM sequences** — Irregular gaps, no 1-second sampling
- **Phase 1.3 target conflicts artificially resolved** — Underlying label ambiguity remains

**Phase 1.5 should not proceed** until:
1. Original minute-precision dataset is archived
2. Timestamp reconstruction limitations are formally documented in all downstream artifacts
3. LSTM sequence feasibility is assessed with irregular-time methods (not fixed-step)
4. Target conflict resolution policy is established

---

## 11. Files Created/Updated

1. `data/documentation/PIPER_dataset_version_integrity_audit.md` (this report)
2. `data/documentation/PIPER_dataset_version_integrity_report.json`
3. `data/documentation/PIPER_phase1_2_schema_inventory.md` (updated)
4. `data/documentation/PIPER_phase1_3_timestamp_segment_audit.md` (updated)
5. `data/documentation/PIPER_phase1_3_segment_summary.csv` (updated)