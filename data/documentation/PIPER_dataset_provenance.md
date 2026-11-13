# PIPER Dataset Provenance Record

## 1. Dataset Identity

| Field | Value | Evidence Source |
|-------|-------|-----------------|
| **Exact dataset name** | SCADA_Pipeline | `data/reports/scada_dataset_profile.json`, `data/reports/phase2_summary.md` |
| **Original dataset title** | Not found in available project evidence | — |
| **Dataset provider / publisher / author** | Not found in available project evidence | — |
| **Hosting platform / repository** | Local repository only (PIPER- project) | Repository inspection; not found in GitHub ApoErch/Water-Leakage-Detection |
| **Original download / source URL** | Not found in available project evidence | — |
| **Version / release number / publication date** | Not found in available project evidence | — |
| **Download date** | Not found in available project evidence | — |
| **Local filename** | scada_pipeline.csv | Repository file listing |
| **Exact repository path** | `data/raw/scada_pipeline.csv` | Repository file listing |
| **File format** | CSV (comma-separated values) | File extension; `data/reports/scada_dataset_profile.json` |
| **File size** | 64,185 bytes | Local measurement (SHA-256 verification) |
| **Number of records** | 1,000 rows | `data/reports/scada_dataset_profile.json`; local verification |
| **Number of columns** | 13 columns | `data/reports/scada_dataset_profile.json`; local verification |
| **Licence and usage restrictions** | Not found in available project evidence | — |

**Column names:** `timestamp`, `segment_id`, `pressure`, `flow_rate`, `temperature`, `valve_status`, `pump_state`, `pump_speed`, `compressor_state`, `energy_consumption`, `alarm_triggered`, `event_type`, `target`

---

## 2. Provider and Hosting Platform

| Aspect | Finding | Evidence |
|--------|---------|----------|
| **Original data provider** | Unknown — no source documentation identifies the creator | No README, metadata, or citation file references a provider |
| **Hosting platform** | Local repository only (`PIPER-/data/raw/`) | File exists locally; not present in upstream GitHub repo (ApoErch/Water-Leakage-Detection) |
| **Relationship to ApoErch repository** | Distinct dataset — ApoErch repo contains only `LeakageData/` (water meter CSVs) | GitHub API inspection of ApoErch/Water-Leakage-Detection shows no `scada_pipeline.csv` |

---

## 3. Original Source URL

**Not found in available project evidence.** No URL, DOI, or repository reference is documented in any project file.

---

## 4. Version and Download Date Information

| Field | Value |
|-------|-------|
| Version / release | Not found in available project evidence |
| Publication date | Not found in available project evidence |
| Download date | Not found in available project evidence |
| Local file creation date (filesystem) | 2026-10-11 16:40 (from `dir` output) |

---

## 5. Licence Information

**Not found in available project evidence.** No LICENSE file, README notice, or metadata field specifies licence terms for `scada_pipeline.csv`.

---

## 6. Dataset File Details

| Property | Value |
|----------|-------|
| Local path | `data/raw/scada_pipeline.csv` |
| Format | CSV |
| Size | 64,185 bytes |
| Rows | 1,000 |
| Columns | 13 |
| Column names | timestamp, segment_id, pressure, flow_rate, temperature, valve_status, pump_state, pump_speed, compressor_state, energy_consumption, alarm_triggered, event_type, target |
| SHA-256 checksum | `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a` |
| Timestamp range | 2024-01-01 00:00:00 to 2024-01-01 00:16:00 (16 minutes) |
| Unique segments (sensor_id) | 50 (segment_id 1–50) |
| Target distribution | 694 normal (0), 306 abnormal (1) |
| Event types | normal (694), degradation (135), leak (65), surge (61), blockage (45) |

---

## 7. Measurement Origin Classification

| Classification | Assessment | Evidence |
|----------------|------------|----------|
| **Real industrial measurements** | **No** | No documentation of physical pipeline, sensor deployment, or SCADA system |
| **Simulated measurements** | **Likely** | Extremely short time span (16 min), multiple conflicting readings per (segment, timestamp), uniform segment count (50), all segments have both target classes |
| **Synthetic / generated observations** | **Strong indicators** | Perfect 50 segments, 17 unique timestamps globally, duplicate timestamps with conflicting telemetry and targets in all 50 segments, no missing values, no sensor gaps |
| **Benchmark / laboratory measurements** | **No evidence** | No documentation of test facility, experimental protocol, or benchmark suite |
| **Mixture** | **Not supported** | No evidence of mixed provenance |

**Conclusion:** The dataset exhibits multiple hallmarks of synthetic/generated data: implausibly short collection window (16 minutes for 50 sensors), multiple conflicting readings at identical timestamps for the same sensor, uniform sensor coverage, and absence of any collection metadata. It should **not** be described as real industrial SCADA measurements.

---

## 8. Collection Context

| Aspect | Finding | Evidence Source |
|--------|---------|-----------------|
| Pipeline type and application | Unknown — not documented | No source documentation |
| Physical or simulated pipeline environment | Unknown — not documented | No source documentation |
| Sensor types and placement | Unknown — segment_id (1–50) used as sensor proxy | `data/reports/scada_dataset_profile.json` maps segment_id → sensor_id |
| Measurement units | Not explicitly documented | Pressure values ~68–102 (likely psi/bar), flow_rate ~1.5–6 (likely m³/h or similar), temperature ~27–37 (likely °C) — inferred from value ranges only |
| Operating conditions | Unknown | No documentation |
| Data acquisition system | Unknown — filename suggests "SCADA" but no system identified | Filename only |
| Sampling process and frequency | Irregular; ~1 minute median interval but multiple readings per timestamp | `data/reports/scada_temporal_feasibility.json`, `data/reports/cleaning_report_SCADA_Pipeline.json` |
| Collection duration | 16 minutes (2024-01-01 00:00 to 00:16) | `data/reports/scada_temporal_feasibility.json` |
| Number of pipelines / segments / sensors | 1 pipeline (assumed), 50 segments/sensors | `data/reports/scada_dataset_profile.json`; `pipeline_id` missing |
| Fault / event generation process | Unknown — event_type column exists (normal, degradation, leak, surge, blockage) but generation method not documented | `event_type` column present; no documentation of how events were generated |
| Known limitations (source-reported) | None — no source documentation exists | — |
| Documented assumptions (source-reported) | None — no source documentation exists | — |

---

## 9. Sensor and Measurement Information

| Variable | Column | Values Observed | Units (Inferred) | Notes |
|----------|--------|-----------------|------------------|-------|
| Pressure | `pressure` | 66.4 – 101.8 | Likely psi or bar | Primary telemetry |
| Flow rate | `flow_rate` | 1.4 – 5.9 | Likely m³/h or similar | Primary telemetry |
| Temperature | `temperature` | 27.4 – 37.2 | Likely °C | Primary telemetry |
| Vibration | — | **Not present** | — | Missing from dataset |
| Valve status | `valve_status` | 0, 1, 2 | Categorical (ordinal?) | Auxiliary |
| Pump state | `pump_state` | 0, 1 | Binary | Auxiliary |
| Pump speed | `pump_speed` | 0 – ~1669 | Likely RPM | Auxiliary |
| Compressor state | `compressor_state` | 0, 1 | Binary | Auxiliary |
| Energy consumption | `energy_consumption` | 8 – 39.5 | Likely kW | Auxiliary |
| Alarm triggered | `alarm_triggered` | 0, 1 | Binary | Auxiliary; **excluded from predictive features (target leakage)** |
| Event type | `event_type` | normal, degradation, leak, surge, blockage | Categorical | **Excluded from predictive features (post-event info)** |
| Target | `target` | 0 (normal), 1 (abnormal) | Binary | Label; mapped from existing column |

---

## 10. Sampling Information

| Property | Value | Evidence |
|----------|-------|----------|
| Global time span | 0.267 hours (16 minutes) | `scada_temporal_feasibility.json` |
| Unique timestamps (global) | 17 | `scada_temporal_feasibility.json` |
| Median sampling interval | ~1 minute (but highly irregular) | `cleaning_report_SCADA_Pipeline.json` |
| Sampling regularity | **Not regular** — 0% on-time rate for 1-hour expected interval; multiple readings per timestamp | `cleaning_report_SCADA_Pipeline.json` (check_sampling_interval step) |
| Duplicate (segment_id, timestamp) keys | 270 groups, 408 conflicting rows | `cleaning_report_SCADA_Pipeline.json` (handle_duplicates step) |
| Per-segment unique timestamps | 8–16 | `scada_temporal_feasibility.json` |
| Timestamp format | `%m/%d/%Y %H:%M` (e.g., `1/1/2024 0:00`) | `inspect_scada.py`, `cleaning_report` |

---

## 11. Known Limitations

1. **No provenance documentation** — No source URL, provider, licence, version, or collection metadata.
2. **Not real industrial data** — All evidence points to synthetic/generated data.
3. **Extremely short time span** — Only 16 minutes of data across all 50 sensors.
4. **Duplicate timestamps with conflicts** — Every segment has multiple readings at the same timestamp with different telemetry values and sometimes different target labels.
5. **No vibration telemetry** — The paper's fourth primary variable is absent.
6. **No pipeline_id** — Single pipeline assumed; segment_id used as sensor_id.
7. **Insufficient for temporal sequence modeling** — Phase 2 feasibility assessment: `can_support_sequence_construction: false`, `can_support_prediction_horizon: false`.
8. **Target leakage risk** — `alarm_triggered` and `event_type` columns deterministically relate to target; excluded from predictive features.
9. **Irregular sampling** — Not a regular time series; multiple readings per timestamp per sensor.

---

## 12. Evidence References

| Evidence File | Path | Type |
|---------------|------|------|
| Dataset profile | `data/reports/scada_dataset_profile.json` | Generated report |
| Cleaning report | `data/reports/cleaning_report_SCADA_Pipeline.json` | Generated report |
| Label report | `data/reports/label_report_SCADA_Pipeline.json` | Generated report |
| Temporal feasibility | `data/reports/scada_temporal_feasibility.json` | Generated report |
| Phase 2 summary | `data/reports/phase2_summary.md` | Generated report |
| Dataset mapping code | `utils/piper_telemetry.py` (SCADA_COLUMN_MAPPING) | Source code |
| Dataset loading code | `utils/piper_dataset.py` (load_scada_raw) | Source code |
| Raw data file | `data/raw/scada_pipeline.csv` | Primary artifact |
| GitHub ApoErch repo | https://github.com/ApoErch/Water-Leakage-Detection | External reference (checked; SCADA file not present) |

---

## 13. Unknown or Unverified Information

| Field | Status |
|-------|--------|
| Original dataset creator / author | **Unknown** — no attribution found |
| Original publication / release venue | **Unknown** |
| Data generation process (if synthetic) | **Unknown** — no generation script or methodology documented |
| Licence / usage rights | **Unknown** |
| Whether dataset was downloaded or created locally | **Unknown** — file appears in repo with no download script |
| Physical sensor specifications | **Unknown** |
| SCADA system vendor / model | **Unknown** |
| Geographic location of pipeline | **Unknown** |
| Fluid / product transported | **Unknown** |
| Event injection methodology (if simulated) | **Unknown** |

---

## 14. Overall Provenance Assessment

**VERDICT: FAIL — Dataset identity and origin cannot be established sufficiently to support research claims.**

### Summary

| Criterion | Status |
|-----------|--------|
| Dataset identity (name, provider, source URL) | **Not established** — only local filename known |
| Measurement origin (real vs. simulated) | **Established as likely synthetic** — multiple strong indicators |
| Collection context | **Almost entirely unknown** — no source documentation |
| Licence | **Unknown** |
| Raw data integrity | **Confirmed** — SHA-256 stable, file unchanged |

The dataset `scada_pipeline.csv` exists only in the local repository with no traceable provenance. It exhibits definitive characteristics of synthetic data (16-minute span, 50 perfectly balanced sensors, universal timestamp conflicts, no missing values, no gaps). No documentation, README, metadata, or citation links it to any real pipeline, SCADA system, benchmark suite, or publication. It cannot be described as "pipeline telemetry" in the sense of real industrial measurements.

**Recommendation:** Do not use this dataset for temporal LSTM training claiming real-world pipeline monitoring. If retained, it must be explicitly labelled as **synthetic/benchmark data of unknown origin** in all reports and the paper. A replacement dataset with verifiable provenance should be sought for Phase 1.5 suitability decision.