# PIPER Provenance Evidence Register

This register documents every provenance claim made in the dataset provenance record, its evidence source, and verification status.

## Verification Status Definitions

| Status | Meaning |
|--------|---------|
| **Verified** | Claim directly supported by traceable evidence in the repository |
| **Partially verified** | Claim supported by evidence but with gaps or inferred elements |
| **Unverified** | Claim made based on inference or observation without direct source documentation |
| **Not available** | Information sought but no evidence exists in the repository |

---

## Evidence Register

| # | Claim | Evidence Source | File Path or URL | Evidence Type | Verification Status |
|---|-------|-----------------|------------------|---------------|---------------------|
| 1 | Dataset name is "SCADA_Pipeline" | Generated reports | `data/reports/scada_dataset_profile.json`, `data/reports/phase2_summary.md` | JSON/Markdown report | Verified |
| 2 | Local filename is `scada_pipeline.csv` | Repository file listing | `data/raw/scada_pipeline.csv` | Filesystem | Verified |
| 3 | Exact repository path is `data/raw/scada_pipeline.csv` | Repository file listing | `data/raw/scada_pipeline.csv` | Filesystem | Verified |
| 4 | File format is CSV | File extension; report | `data/reports/scada_dataset_profile.json` | File + report | Verified |
| 5 | File size is 64,185 bytes | Local measurement | `data/raw/scada_pipeline.csv` | Filesystem (measured) | Verified |
| 6 | SHA-256 checksum is `c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a` | Local calculation | `data/raw/scada_pipeline.csv` | Cryptographic hash | Verified |
| 7 | Record count is 1,000 rows | Local verification + reports | `data/reports/scada_dataset_profile.json` | Code + report | Verified |
| 8 | Column count is 13 | Local verification + reports | `data/reports/scada_dataset_profile.json` | Code + report | Verified |
| 9 | Column names are: timestamp, segment_id, pressure, flow_rate, temperature, valve_status, pump_state, pump_speed, compressor_state, energy_consumption, alarm_triggered, event_type, target | Local verification + reports | `data/reports/scada_dataset_profile.json` | Code + report | Verified |
| 10 | Timestamp range: 2024-01-01 00:00:00 to 2024-01-01 00:16:00 | Generated reports | `data/reports/scada_temporal_feasibility.json`, `data/reports/cleaning_report_SCADA_Pipeline.json` | JSON report | Verified |
| 11 | 50 unique segment_ids (1–50) | Generated reports | `data/reports/scada_dataset_profile.json`, `data/reports/scada_temporal_feasibility.json` | JSON report | Verified |
| 12 | Target distribution: 694 normal (0), 306 abnormal (1) | Generated reports | `data/reports/label_report_SCADA_Pipeline.json`, `data/reports/phase2_summary.md` | JSON/Markdown report | Verified |
| 13 | Event type distribution: normal (694), degradation (135), leak (65), surge (61), blockage (45) | Local verification | `data/raw/scada_pipeline.csv` | Code analysis | Verified |
| 14 | 270 duplicate (segment_id, timestamp) key groups with 408 conflicting rows | Generated reports | `data/reports/cleaning_report_SCADA_Pipeline.json` (handle_duplicates step) | JSON report | Verified |
| 15 | No exact duplicate rows | Generated reports | `data/reports/cleaning_report_SCADA_Pipeline.json` | JSON report | Verified |
| 16 | Sampling is irregular; 0% on-time rate for 1-hour expected interval | Generated reports | `data/reports/cleaning_report_SCADA_Pipeline.json` (check_sampling_interval step) | JSON report | Verified |
| 17 | Per-segment unique timestamps: 8–16 | Generated reports | `data/reports/scada_temporal_feasibility.json` | JSON report | Verified |
| 18 | Vibration telemetry not present | Generated reports + local check | `data/reports/scada_dataset_profile.json`, `data/reports/phase2_summary.md`, `data/raw/scada_pipeline.csv` | Report + code | Verified |
| 19 | pipeline_id column missing | Generated reports | `data/reports/scada_dataset_profile.json`, `data/reports/phase2_summary.md` | JSON/Markdown report | Verified |
| 20 | label_report classifies target as "real_ground_truth" with is_synthetic=false | Generated report | `data/reports/label_report_SCADA_Pipeline.json` | JSON report | Verified (but claim contradicted by other evidence) |
| 21 | Dataset NOT present in upstream GitHub repo (ApoErch/Water-Leakage-Detection) | GitHub API check | https://github.com/ApoErch/Water-Leakage-Detection | External API | Verified |
| 22 | No source documentation (README, metadata, licence, citation) for scada_pipeline.csv | Repository inspection | `data/raw/`, project root | Filesystem inspection | Verified |
| 23 | No download script or acquisition record for scada_pipeline.csv | Repository inspection | Project root, `utils/`, `tools/` | Code inspection | Verified |
| 24 | Dataset exhibits hallmarks of synthetic data: 16-min span, 50 uniform segments, universal timestamp conflicts, perfect coverage | Local analysis of raw data | `data/raw/scada_pipeline.csv`, `data/reports/scada_temporal_feasibility.json` | Code analysis + report | Partially verified (inference from data characteristics) |
| 25 | Measurement units not documented; inferred from value ranges | Local analysis | `data/raw/scada_pipeline.csv` | Code analysis | Unverified (inference only) |
| 26 | Original dataset provider / author unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 27 | Original dataset title unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 28 | Original download/source URL unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 29 | Version/release/publication date unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 30 | Download date unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 31 | Licence and usage restrictions unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 32 | Pipeline type and application unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 33 | Physical or simulated pipeline environment unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 34 | Sensor types and placement unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 35 | Operating conditions unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 36 | Data acquisition system unknown (filename only) | Absence of evidence | Repository inspection | Negative finding | Not available |
| 37 | Collection duration: 16 minutes (from timestamps) | Generated reports | `data/reports/scada_temporal_feasibility.json` | JSON report | Verified (timestamp-derived) |
| 38 | Number of pipelines: 1 (assumed) | Generated reports | `data/reports/scada_dataset_profile.json` | JSON report | Partially verified (assumption documented) |
| 39 | Number of segments/sensors: 50 | Generated reports | `data/reports/scada_dataset_profile.json` | JSON report | Verified |
| 40 | Fault/event generation process unknown | Absence of evidence | Repository inspection | Negative finding | Not available |
| 41 | Source-reported limitations: none (no source docs) | Absence of evidence | Repository inspection | Negative finding | Not available |
| 42 | Source-reported assumptions: none (no source docs) | Absence of evidence | Repository inspection | Negative finding | Not available |
| 43 | Timestamp format: %m/%d/%Y %H:%M | Code + reports | `inspect_scada.py`, `cleaning_report_SCADA_Pipeline.json` | Code + report | Verified |
| 44 | Timezone not specified | Absence of evidence | Repository inspection | Negative finding | Not available |
| 45 | Phase 2 feasibility: tabular classification YES, sequence construction NO, prediction horizon NO | Generated report | `data/reports/scada_temporal_feasibility.json`, `data/reports/phase2_summary.md` | JSON/Markdown report | Verified |
| 46 | alarm_triggered and event_type excluded from predictive features (target leakage risk) | Generated reports | `data/reports/phase2_summary.md`, `data/processed/scada_feature_manifest.json` | Markdown + JSON | Verified |
| 47 | File creation timestamp (filesystem): 2026-10-11 16:40 | Filesystem | `dir data/raw` output | Filesystem | Verified |

---

## Summary of Verification Status

| Status | Count |
|--------|-------|
| Verified | 26 |
| Partially verified | 3 |
| Unverified | 1 |
| Not available | 17 |

---

## Key Gaps

1. **No primary source documentation exists** for the dataset — no README, no metadata file, no licence, no citation, no download record.
2. **All collection context fields are unknown** — pipeline, sensors, environment, acquisition system, event generation.
3. **Measurement origin is inferred, not source-stated** — the dataset itself provides no origin claim; classification as synthetic is based on data characteristics.
4. **label_report claims "real_ground_truth" but this is a project-internal label** — not an assertion from the original data provider (which is unknown).
5. **Upstream repository (ApoErch) does not contain this dataset** — it only contains the LeakageData water-meter CSVs.