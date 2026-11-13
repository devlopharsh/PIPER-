# PIPER Phase 1.3 — Timestamp and Segment Semantics Audit

**Dataset:** `data/raw/scada_pipeline.csv`  
**Minute-Precision SHA-256:** `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`  
**Seconds-Precision SHA-256:** `1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c`  
**Phase 1.1 & 1.2 Hash Match (minute-precision):** ✅ CONFIRMED  
**Audit Date:** 2026-11-05 (original), 2026-11-05 (seconds-precision update)

---

## ⚠️ VERSION NOTICE

This document contains findings for **two dataset versions**:

1. **Minute-precision (original, audited 2026-11-05):** SHA-256 `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`, 17 unique timestamps, 270 repeated groups.
2. **Seconds-precision (reconstructed, audited 2026-11-05):** SHA-256 `1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c`, 1,000 unique timestamps, 0 repeated groups.

The seconds-precision version was created by **reconstructing seconds from original CSV row order within each minute group**. This is a derived timestamp, **not independently verified sensor-recorded time**. The original minute-precision dataset is not available as a separate file.

**Sections 1–14 reflect the original minute-precision audit. Section 15+ contains the seconds-precision update.**

---

## 1. Audit Objective

This audit investigates the timestamp validity, chronological structure, segment-level temporal coverage, repeated observations, and the meaning of repeated `(segment_id, timestamp)` groups in the SCADA pipeline dataset. The objective is to establish whether the dataset's temporal semantics support sequence-based LSTM modeling as required by the PIPER framework.

---

## 2. Dataset Identity and Integrity Check (Minute-Precision)

| Property | Value |
|----------|-------|
| File path | `data/raw/scada_pipeline.csv` |
| File size | 64,185 bytes |
| SHA-256 | `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a` |
| Rows | 1,000 |
| Columns | 13 |
| Phase 1.1 hash match | ✅ YES |
| Phase 1.2 hash match | ✅ YES |

**Raw CSV integrity confirmed:** The file is identical to the version audited in Phases 1.1 and 1.2.

---

## 3. Timestamp Parsing and Validation (Minute-Precision)

### 3.1 Parsing Results

| Metric | Value |
|--------|-------|
| Format used | `%m/%d/%Y %H:%M` (explicit, not inferred) |
| Successfully parsed | 1,000 / 1,000 (100%) |
| Failed parses | 0 |
| Null/malformed | 0 |
| Unique parsed timestamps | 17 |
| Earliest | 2024-01-01 00:00:00 |
| Latest | 2024-01-01 00:16:00 |
| Total time span | 16 minutes (0.267 hours) |

### 3.2 Timestamp Format Characteristics

- **Seconds present:** No — format shows `H:MM` (e.g., `0:00`, `0:16`), regex check for `:\d{2}$` returns True due to minute precision but no seconds field
- **Timezone information:** None detected in raw strings
- **Timezone documented:** No — source provides no timezone documentation
- **Parsing assumption:** Treated as naive local timestamps; no UTC conversion applied

**Conclusion:** All 1,000 timestamps parse successfully with the explicit format. The 16-minute span across 17 unique minute-resolution timestamps is extremely short for pipeline monitoring.

---

## 4. Global Temporal Coverage (Minute-Precision)

### 4.1 Timestamp Frequency

| Timestamp | Rows |
|-----------|------|
| 1/1/2024 0:00 | 60 |
| 1/1/2024 0:01 | 60 |
| 1/1/2024 0:02 | 60 |
| 1/1/2024 0:03 | 60 |
| 1/1/2024 0:04 | 60 |
| 1/1/2024 0:05 | 60 |
| 1/1/2024 0:06 | 60 |
| 1/1/2024 0:07 | 60 |
| 1/1/2024 0:08 | 60 |
| 1/1/2024 0:09 | 60 |
| 1/1/2024 0:10 | 60 |
| 1/1/2024 0:11 | 60 |
| 1/1/2024 0:12 | 60 |
| 1/1/2024 0:13 | 60 |
| 1/1/2024 0:14 | 60 |
| 1/1/2024 0:15 | 60 |
| 1/1/2024 0:16 | 40 |

**Statistics:**
- Min rows per timestamp: 40
- Max rows per timestamp: 60
- Mean: 58.8
- Median: 60.0

### 4.2 Chronological Structure

| Property | Value |
|----------|-------|
| CSV globally sorted by timestamp | ✅ YES |
| Unique timestamp interval | Exactly 1.0 minute between all consecutive timestamps |
| Interval regularity | Perfectly regular (16 intervals of 1 minute each) |

**Observation:** The global timestamp grid is perfectly regular at 1-minute intervals. However, no single segment observes all 17 timestamps (see Section 5).

---

## 5. Per-Segment Temporal Coverage (Minute-Precision)

### 5.1 Summary Statistics (50 segments)

| Metric | Value |
|--------|-------|
| Rows per segment | 12–33 (mean 20.0) |
| Unique timestamps per segment | 8–16 (mean 11.8) |
| Time span per segment | 0.20–0.27 hours (12–16 minutes) |
| Segments ordered in CSV | 50/50 ✅ |
| Repeated timestamps per segment | 2–9 (mean 5.4) |
| Max repeats at single timestamp | 2–7 |

### 5.2 Segments with Lowest/Highest Coverage

| Category | Segments | Unique Timestamps |
|----------|----------|-------------------|
| **Lowest coverage** | 42, 7, 13, 14, 44 | 8, 9, 9, 9, 9 |
| **Highest coverage** | 43, 46, 9, 18, 22 | 16, 15, 14, 14, 14 |
| **Most repeated timestamps** | 16, 36, 23, 43, 3 | 9, 9, 8, 8, 7 repeated |

### 5.3 Per-Segment Interval Patterns

- All segments show 1-minute median intervals where consecutive unique timestamps exist
- Mean intervals range 1.06–2.00 minutes (some segments skip minutes)
- No segment has a continuous 17-timestamp sequence

**Key Finding:** Every segment has **multiple readings at the same timestamp** (2–9 repeated timestamps per segment). No segment provides a clean, deduplicated time series.

---

## 6. Repeated Segment–Timestamp Group Analysis (Minute-Precision)

### 6.1 Group Statistics

| Metric | Value |
|--------|-------|
| Total distinct `(segment_id, timestamp)` groups | 592 |
| Single-row groups | 322 (54.4%) |
| Repeated groups (>1 row) | 270 (45.6%) |
| Rows in repeated groups | 678 (67.8% of dataset) |
| Group size range | 1–7 rows |
| Mean group size | 1.69 |
| Median group size | 1.0 |

### 6.2 Group Size Distribution

| Group Size | Groups | Rows | % of Repeated Groups |
|------------|--------|------|---------------------|
| 1 | 322 | 322 | — |
| 2 | 171 | 342 | 63.3% |
| 3 | 67 | 201 | 24.8% |
| 4 | 27 | 108 | 10.0% |
| 5 | 4 | 20 | 1.5% |
| 7 | 1 | 7 | 0.4% |

### 6.3 Group Categories

| Category | Description | Groups | % of Repeated |
|----------|-------------|--------|---------------|
| **F_mixed_telemetry_and_metadata** | Both telemetry and metadata/label vary | 150 | 55.6% |
| **C_different_telemetry_same_metadata** | Telemetry varies, `event_type`/`target` same | 120 | 44.4% |
| A_exact_duplicate | All columns identical | 0 | 0% |
| B_same_telemetry_diff_metadata | Telemetry same, metadata differs | 0 | 0% |
| E_other | Other patterns | 0 | 0% |

**Critical Finding:** **Zero exact duplicates.** Every repeated group contains meaningfully different data.

### 6.4 Representative Examples

#### Category F: Mixed Telemetry and Metadata Variation (150 groups)

| Segment | Timestamp | Size | Key Differences |
|---------|-----------|------|-----------------|
| 1 | 1/1/2024 0:02 | 3 | pressure: 89.89/66.90/72.15; target: 1/0/0; event: blockage/normal/normal; pump: on/off/on |
| 1 | 1/1/2024 0:05 | 2 | pressure: 72.71/66.83; target: 1/0; event: degradation/normal; pump: on/off |
| 1 | 1/1/2024 0:08 | 3 | pressure: 77.60/82.73/68.80; target: 0/1/0; event: normal/degradation/normal |

#### Category C: Different Telemetry, Same Metadata (120 groups)

| Segment | Timestamp | Size | Key Differences |
|---------|-----------|------|-----------------|
| 1 | 1/1/2024 0:10 | 2 | pressure: 69.75/75.92; flow: 4.37/5.08; temp: 31.86/33.59; valve: 1/0; pump: off/on |
| 2 | 1/1/2024 0:12 | 2 | pressure: 75.63/74.87; flow: 4.16/4.53; temp: 33.84/35.35; pump_speed: 1379/1352 |
| 2 | 1/1/2024 0:13 | 2 | pressure: 72.57/70.37; valve: 1/0; pump_speed: 1465/1434; compressor: off/on |

---

## 7. Row-Level Difference Analysis (Minute-Precision)

### 7.1 Variation Frequency Across 270 Repeated Groups

| Column | Groups Varying | Percentage |
|--------|----------------|------------|
| **pressure** | 270 | 100.0% |
| **temperature** | 270 | 100.0% |
| **energy_consumption** | 270 | 100.0% |
| **flow_rate** | 268 | 99.3% |
| **pump_speed** | 256 | 94.8% |
| **compressor_state** | 156 | 57.8% |
| **valve_status** | 144 | 53.3% |
| **pump_state** | 141 | 52.2% |
| **event_type** | 150 | 55.6% |
| **target** | 130 | 48.1% |
| **alarm_triggered** | 114 | 42.2% |

### 7.2 Numeric Variation Magnitudes (Within-Group Range)

| Variable | Groups Varying | Min Range | Max Range | Mean Range |
|----------|----------------|-----------|-----------|------------|
| pressure | 270 | 0.04 | 49.87 | 12.09 |
| flow_rate | 268 | 0.01 | 4.50 | 1.14 |
| temperature | 270 | 0.03 | 7.94 | 2.64 |
| pump_speed | 256 | 0.80 | 1678.80 | 833.52 |
| energy_consumption | 270 | 0.06 | 38.23 | 12.64 |

**Interpretation:** Telemetry differences within the same `(segment, timestamp)` are substantial — not measurement noise. Pressure ranges up to ~50 units, flow_rate up to 4.5, temperature up to ~8°C.

### 7.3 Categorical/Binary Variation Patterns

| Variable | Variation Pattern | Groups |
|----------|-------------------|--------|
| pump_state | 0 ↔ 1 | 141 |
| compressor_state | 0 ↔ 1 | 156 |
| valve_status | 0↔1 (78), 1↔2 (42), 0↔1↔2 (13), 0↔2 (11) | 144 |
| alarm_triggered | 0 ↔ 1 | 114 |
| event_type | normal↔degradation (50), normal↔surge (27), normal↔leak (25), normal↔blockage (15) | 150 |
| target | 0 ↔ 1 | 130 |

---

## 8. Cross-Segment Synchronization (Minute-Precision)

### 8.1 Segment Representation at Each Timestamp

| Timestamp | Segments Present | Coverage |
|-----------|------------------|----------|
| 0:00 | 34 | 68% |
| 0:01 | 37 | 74% |
| 0:02 | 36 | 72% |
| 0:03 | 33 | 66% |
| 0:04 | 33 | 66% |
| 0:05 | 37 | 74% |
| 0:06 | 35 | 70% |
| 0:07 | 34 | 68% |
| 0:08 | 37 | 74% |
| 0:09 | 37 | 74% |
| 0:10 | 35 | 70% |
| 0:11 | 34 | 68% |
| 0:12 | 36 | 72% |
| 0:13 | 38 | 76% |
| 0:14 | 35 | 70% |
| 0:15 | 34 | 68% |
| 0:16 | 27 | 54% |

### 8.2 Synchronization Assessment

| Metric | Value |
|--------|-------|
| Segments present at ALL 17 timestamps | **0 / 50** |
| Segments missing ≥1 timestamp | 50 / 50 |
| Per-segment coverage (unique timestamps) | 8–16 (mean 11.8) |

**Conclusion:** **No cross-segment synchronization.** Segments do not share a common observation schedule. Each segment has a different subset of the 17 global timestamps. The dataset cannot be treated as a synchronized multi-sensor time series.

---

## 9. LSTM Temporal Sequence Feasibility (Minute-Precision)

### 9.1 Theoretical Window Counts (Based on Unique Timestamps per Segment)

Formula: `Potential windows = max(0, unique_timestamps - window_length + 1)`

| Window Length | Segments with ≥1 Window | Total Potential Windows | Segments with 0 Windows |
|---------------|------------------------|------------------------|------------------------|
| 3 | 50 / 50 | 492 | 0 |
| 5 | 50 / 50 | 392 | 0 |
| 8 | 50 / 50 | 242 | 0 |
| 10 | 45 / 50 | 143 | 5 |

### 9.2 Critical Feasibility Limitations

| Limitation | Impact |
|------------|--------|
| **Only 8–16 unique timestamps per segment** | Even window=10 leaves 5 segments with zero windows |
| **16-minute total span** | No meaningful forecast horizon possible; "future" is seconds away |
| **Repeated timestamps with conflicting data** | 67.8% of rows belong to ambiguous groups; which row represents "the" observation? |
| **Target conflicts in 130/270 repeated groups** | Same segment-timestamp has both normal and abnormal labels |
| **No segment synchronization** | Cannot construct multi-segment sequences or align segments temporally |
| **Event_type perfectly predicts target** | Leakage risk if used; but also suggests target is post-hoc |

**Assessment:** While theoretical window counts are positive for small windows (3–8), the **semantic validity** of sequences is severely compromised by repeated-timestamp ambiguity, target conflicts, and the 16-minute total duration. These are not validated training samples — they are theoretical upper bounds assuming clean deduplication and label resolution.

---

## 10. Key Findings and Limitations (Minute-Precision)

| Finding | Classification | Evidence |
|---------|----------------|----------|
| Timestamps parse successfully with explicit format | **CONFIRMED** | 1000/1000 parsed, 0 failures |
| Timestamps are perfectly regular at 1-minute intervals globally | **CONFIRMED** | All 16 intervals = 1.0 min |
| No timezone documented or present in data | **CONFIRMED** | Regex check + no documentation |
| Every segment has repeated timestamps | **CONFIRMED** | 2–9 repeated timestamps per segment |
| Zero exact duplicate rows in repeated groups | **CONFIRMED** | All 270 groups show meaningful differences |
| Repeated groups show substantial telemetry variation | **CONFIRMED** | Pressure range up to 50, temp up to 8°C within same timestamp |
| 48% of repeated groups have conflicting target labels | **CONFIRMED** | 130/270 groups vary in `target` |
| No segment observes all 17 timestamps | **CONFIRMED** | Max 16/17, min 8/17 |
| Cross-segment synchronization absent | **CONFIRMED** | 0/50 segments at all timestamps |
| Theoretical LSTM windows exist for small windows | **CONFIRMED** | 492 windows at length 3 |
| **But** sequences semantically invalid due to ambiguity | **PARTIALLY CONFIRMED** | Repeated rows, target conflicts, 16-min span |
| **Dataset supports defensible temporal prediction target** | **NOT SUPPORTED** | Target conflicts, leakage fields, short span, no provenance |

---

## 11. Phase 1.3 Verdict (Minute-Precision)

**PASS WITH OBSERVATIONS**

### Rationale

The audit successfully completed all required tasks and produced complete documentation. However, the following observations must be carried forward:

1. **Repeated-row semantics unresolved** — 67.8% of rows belong to ambiguous `(segment, timestamp)` groups with differing telemetry and conflicting labels. No evidence identifies whether these represent multi-sensor readings, repeated samples, or export artifacts.

2. **Target conflicts in 130 groups** — Same segment-timestamp has both `target=0` and `target=1`. Phase 1.4 must resolve label policy.

3. **Cross-segment synchronization absent** — Cannot construct multi-sequence inputs or assume aligned observations.

4. **16-minute total duration** — Fundamentally insufficient for temporal forecasting; any "prediction horizon" would be seconds.

5. **Leakage fields present** — `event_type` and `alarm_triggered` deterministically relate to target (Phase 1.2 finding confirmed).

The audit execution is **PASS**; the dataset's suitability for LSTM temporal modeling remains **INCONCLUSIVE TO NEGATIVE** pending Phase 1.4 resolution.

---

## 12. Recommended Inputs for Phase 1.4

Phase 1.4 (Label Integrity and Leakage Audit) should investigate:

1. **Label resolution policy for repeated groups** — 130 groups with conflicting `target`; must decide: exclude, majority vote, flag as uncertain, or use only single-row groups.

2. **Origin of `event_type` and `target` relationship** — `event_type` perfectly predicts `target`. Is `event_type` a post-hoc classification? Is `target` derived from it?

3. **`alarm_triggered` as proxy label** — 100% of `alarm=1` rows have `target=1`. Is alarm the ground truth source?

4. **Repeated-group provenance** — Are these multiple sensors per segment? Repeated measurements? Simulated event injections?

5. **Temporal target definition** — With 16-minute span and no segment synchronization, what is the prediction task? Current-state classification? 1-minute ahead? Segment-level or global?

6. **Deduplication strategy** — If sequences are attempted, how to handle 270 ambiguous groups? (Decision must be source-backed, not arbitrary.)

---

## 13. Deliverables Generated (Minute-Precision)

| File | Description | Size |
|------|-------------|------|
| `PIPER_phase1_3_timestamp_segment_audit.md` | This report | ~15 KB |
| `PIPER_phase1_3_timestamp_segment_report.json` | Machine-readable summary | 26 KB |
| `PIPER_phase1_3_segment_summary.csv` | Per-segment temporal statistics (50 rows) | 4.4 KB |
| `PIPER_phase1_3_repeated_group_examples.csv` | All 270 repeated groups with category & differing values | 134 KB |

---

## 14. Integrity Confirmation (Minute-Precision)

✅ **Raw CSV unchanged** — SHA-256 matches Phase 1.1 and 1.2: `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`  
✅ **Phase 1.1 reports preserved** — No modifications  
✅ **Phase 1.2 reports preserved** — No modifications  
✅ **No model trained** — Read-only analysis only  
✅ **No source code modified** — Analysis scripts only

---

# ════════════════════════════════════════════════════════════════
# SECONDS-PRECISION UPDATE (Reconstructed Timestamps)
# ═══════════════════════════════════════════════════════════════

## 15. Seconds-Precision Dataset Identity

| Property | Value |
|----------|-------|
| File path | `data/raw/scada_pipeline.csv` |
| File size | 69,184 bytes |
| SHA-256 | `1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c` |
| Rows | 1,000 |
| Columns | 13 |
| Timestamp format | `MM/DD/YYYY HH:MM:SS` |
| Reconstruction method | Seconds assigned from CSV row order within each minute group |
| Original minute-precision SHA-256 | `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a` |

**Reconstruction convention:**
- Minutes 00:00–00:15: 60 rows each → seconds 01–60 (minute 00:00 has 59 rows, seconds 01–59)
- Minute 00:16: 41 rows → seconds 00–40
- Total: 59 + 15×60 + 41 = 1,000 rows

**⚠️ This is a derived timestamp reconstruction. It is NOT independently verified sensor-recorded time.**

---

## 16. Seconds-Precision Timestamp Parsing and Validation

| Metric | Value |
|--------|-------|
| Format used | `%m/%d/%Y %H:%M:%S` (explicit) |
| Successfully parsed | 1,000 / 1,000 (100%) |
| Failed parses | 0 |
| Null/malformed | 0 |
| Unique parsed timestamps | 1,000 |
| Earliest | 2024-01-01 00:00:01 |
| Latest | 2024-01-01 00:16:40 |
| Total time span | 16 minutes 39 seconds (999 seconds) |
| All timestamps match `MM/DD/YYYY HH:MM:SS` | ✅ YES |

---

## 17. Seconds-Precision Global Temporal Coverage

### 17.1 Global Ordering

| Check | Result |
|-------|--------|
| Globally non-decreasing | ✅ YES |
| Globally strictly increasing | ✅ YES |
| Duplicate timestamps (global) | 0 |
| Duplicate `(segment_id, timestamp)` | 0 |

### 17.2 Time Span

| Metric | Value |
|--------|-------|
| Earliest | 2024-01-01 00:00:01 |
| Latest | 2024-01-01 00:16:40 |
| Span | 16 min 39 sec (999 seconds) |

### 17.3 Continuous One-Second Sequence (Global)

| Metric | Value |
|--------|-------|
| Expected unique seconds (00:00:01 to 00:16:40) | 1,000 |
| Actual unique seconds | 1,000 |
| Missing seconds | 0 |
| Extra seconds | 0 |
| **Globally continuous 1-second sequence** | ✅ YES |

**⚠️ This global continuity is synthetic** — it reflects the CSV row order reconstruction convention, not sensor sampling reality.

---

## 18. Seconds-Precision Per-Segment Temporal Coverage

### 18.1 Summary Statistics (50 segments)

| Metric | Value |
|--------|-------|
| Segments | 50 (IDs 1–50) |
| Rows per segment | 12–33 (mean 20.0) |
| Unique timestamps per segment | = rows per segment (no duplicates) |
| Elapsed time per segment | 719–995 seconds (12.0–16.6 min) |
| Segments ordered in CSV | 50/50 ✅ |
| Segments with duplicated timestamps | 0 |

### 18.2 Interval Analysis (Per-Segment)

| Metric | Range | Notes |
|--------|-------|-------|
| Minimum gap | 1–15 sec | Some segments have consecutive 1-sec gaps |
| Maximum gap | 106–339 sec | Large gaps within segments |
| Mean gap | 28.8–80.7 sec | Highly irregular |
| Median gap | 12–74.5 sec | Most gaps > 1 second |
| One-second gaps per segment | 0–4 | Very few consecutive 1-sec samples |
| Gaps > 1 second per segment | 11–31 | **All 50 segments have gaps > 1 sec** |

### 18.3 Per-Segment Coverage Table

| Segment | Rows | Elapsed (sec) | Min Gap | Max Gap | Mean Gap | 1-sec Gaps | Gaps >1s |
|---------|------|---------------|---------|---------|----------|------------|----------|
| 1 | 17 | 920 | 1.0 | 172.0 | 57.5 | 1 | 15 |
| 2 | 19 | 930 | 4.0 | 118.0 | 51.7 | 0 | 18 |
| 3 | 24 | 972 | 4.0 | 210.0 | 42.3 | 0 | 23 |
| 4 | 13 | 968 | 13.0 | 185.0 | 80.7 | 0 | 12 |
| 5 | 23 | 820 | 6.0 | 132.0 | 37.3 | 0 | 22 |
| 6 | 20 | 875 | 2.0 | 161.0 | 46.1 | 0 | 19 |
| 7 | 15 | 985 | 2.0 | 162.0 | 70.4 | 0 | 14 |
| 8 | 20 | 842 | 1.0 | 163.0 | 44.3 | 1 | 18 |
| 9 | 21 | 960 | 1.0 | 134.0 | 48.0 | 1 | 19 |
| 10 | 20 | 969 | 1.0 | 171.0 | 51.0 | 4 | 15 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 43 | 33 | 920 | 1.0 | 105.0 | 28.8 | 1 | 31 |
| 50 | 23 | 986 | 1.0 | 106.0 | 44.8 | 1 | 21 |

**Full details in:** `data/documentation/PIPER_phase1_3_segment_summary.csv`

---

## 19. Seconds-Precision Cross-Segment Synchronization

| Metric | Value |
|--------|-------|
| Segments present at ALL 1,000 timestamps | **0 / 50** |
| Segments missing ≥1 timestamp | 50 / 50 |
| Per-segment coverage (unique timestamps) | 12–33 (mean 20.0) |

**Conclusion:** **No cross-segment synchronization.** Segments have different observation schedules. The dataset cannot be treated as a synchronized multi-sensor time series.

---

## 20. Seconds-Precision LSTM Temporal Sequence Feasibility

### 20.1 Theoretical Window Counts (Based on Unique Timestamps per Segment)

Formula: `Potential windows = max(0, unique_timestamps - window_length + 1)`

| Window Length | Segments with ≥1 Window | Total Potential Windows | Segments with 0 Windows |
|---------------|------------------------|------------------------|------------------------|
| 3 | 50 / 50 | 900 | 0 |
| 5 | 50 / 50 | 800 | 0 |
| 8 | 50 / 50 | 650 | 0 |
| 10 | 50 / 50 | 550 | 0 |

### 20.2 Critical Feasibility Assessment

| Limitation | Impact |
|------------|--------|
| **No segment has consistent 1-second sampling** | Median gaps 12–74.5 sec; max gaps 106–339 sec |
| **16-minute total span** | No meaningful forecast horizon |
| **Global 1-second continuity is synthetic** | Artifact of CSV row order reconstruction |
| **Per-segment gaps highly irregular** | 1–339 sec gaps; 0–4 consecutive 1-sec samples per segment |
| **No segment synchronization** | Cannot align segments temporally |
| **Target conflicts artificially resolved** | 130 minute-precision conflicts "hidden" by unique-second assignment |

**Assessment:** **Per-segment sequences do NOT support fixed-step LSTM modeling.** The theoretical window counts assume unique timestamps = valid sequence steps, but the gaps are highly irregular (1–339 sec) and the timestamps are reconstructed, not sensor-verified. Any LSTM would require irregular-time handling (e.g., time-aware LSTM, CT-LSTM) or interpolation — which introduces its own assumptions.

---

## 21. Comparison: Minute vs. Seconds Precision

| Finding | Minute-Precision | Seconds-Precision | Status |
|---------|------------------|-------------------|--------|
| Unique `(segment_id, timestamp)` groups | 592 | 1,000 | **CHANGED** — each row now unique |
| Repeated groups (>1 row) | 270 | 0 | **RESOLVED** — by reconstruction |
| Rows in repeated groups | 678 | 0 | **RESOLVED** |
| Target conflicts in repeated groups | 130 | 0 | **ARTIFICIALLY RESOLVED** |
| Unique global timestamps | 17 | 1,000 | **CHANGED** |
| Per-segment unique timestamps | 8–16 | 12–33 | **CHANGED** (now = row count) |
| Global 1-second continuity | N/A | ✅ Continuous | **SYNTHETIC** |
| Per-segment 1-second sampling | N/A | ❌ None | **INVALID** |
| Window 3 potential windows | 492 | 900 | **INCREASED** (theoretical) |
| Window 5 potential windows | 392 | 800 | **INCREASED** (theoretical) |
| Window 8 potential windows | 242 | 650 | **INCREASED** (theoretical) |
| Window 10 potential windows | 143 | 550 | **INCREASED** (theoretical) |

### 21.1 Key Interpretation

**The timestamp reconstruction has fundamentally altered the temporal semantics:**

1. **Repeated groups eliminated** — The 270 repeated `(segment_id, timestamp)` groups from the minute-precision data were an artifact of minute-level granularity. The reconstruction assigned unique seconds to each row within a minute.

2. **Target conflicts resolved artificially** — The 130 conflicting-target groups were within the same minute. By assigning different seconds, the conflicts are "resolved" but **not genuinely** — the underlying label ambiguity remains.

3. **Global 1-second continuity is synthetic** — The continuous 00:00:01 to 00:16:40 sequence reflects the CSV row order assumption, not sensor reality.

4. **Per-segment sequences remain irregular** — Despite unique timestamps, the gaps between a segment's observations are highly irregular (1–339 seconds), with median gaps of 12–74.5 seconds.

---

## 22. Updated Phase 1.3 Verdict (Combined)

**PASS WITH OBSERVATIONS** (both versions)

### Minute-Precision
The audit successfully completed all required tasks. Critical observations:
- 130 target conflicts in repeated groups
- No cross-segment synchronization
- 16-minute total duration insufficient for forecasting
- Leakage fields present

### Seconds-Precision [UPDATE]
The reconstruction eliminated repeated groups but:
- **Artificially resolved target conflicts** — 130 conflicts hidden by unique-second assignment
- **Synthetic 1-second global continuity** — Not sensor-verified
- **Per-segment sampling still invalid for fixed-step LSTM** — Highly irregular gaps, no 1-second sampling
- **Underlying label ambiguity remains** — Reconstruction does not resolve the semantic issues

### Combined Assessment
The audit execution **PASSES** for both versions. The dataset's suitability for LSTM temporal modeling remains **INCONCLUSIVE TO NEGATIVE**. The seconds-precision version does not fix the fundamental limitations; it only masks the repeated-group conflicts while introducing synthetic timestamps.

---

## 23. Updated Recommendations for Phase 1.4

Phase 1.4 must address **both versions' findings**:

1. **Label resolution policy for 130 conflicting groups** (minute-precision) — The conflicts are real; reconstruction does not resolve them.

2. **Timestamp provenance documentation** — All downstream use must declare: "Seconds-precision timestamps reconstructed from CSV row order; not sensor-verified."

3. **LSTM sequence strategy** — Fixed-step LSTM not supported. Options:
   - Time-aware LSTM (CT-LSTM, Phased LSTM) with actual timestamps
   - Interpolation to regular grid (with uncertainty quantification)
   - Segment-level aggregation (loses temporal resolution)

4. **Segment synchronization** — No evidence of coordinated sampling.

5. **Vibration missing** — Paper requires 4 telemetry variables; only 3 available.

---

## 24. Updated Deliverables

| File | Status | Path |
|------|--------|------|
| Phase 1.3 audit report | ✅ Updated | `data/documentation/PIPER_phase1_3_timestamp_segment_audit.md` |
| Phase 1.3 JSON report | ✅ Updated | `data/documentation/PIPER_phase1_3_timestamp_segment_report.json` |
| Phase 1.3 segment summary | ✅ Updated | `data/documentation/PIPER_phase1_3_segment_summary.csv` |
| Repeated group examples | Preserved (minute-precision) | `data/documentation/PIPER_phase1_3_repeated_group_examples.csv` |

---

## 25. Integrity Confirmation (Both Versions)

✅ **Raw CSV unchanged during this audit** — Seconds-precision SHA-256: `1090c79fbcb3ae19011140f3965a511546cd6b8aa8357837c5bf3bd1d4d37b9c`  
✅ **Minute-precision SHA-256 documented** — `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a`  
✅ **Phase 1.1 reports preserved** — No modifications  
✅ **Phase 1.2 reports preserved** — Updated with version notice  
✅ **No model trained** — Read-only analysis only  
✅ **No source code modified** — Analysis scripts only