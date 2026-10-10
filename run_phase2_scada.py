#!/usr/bin/env python
"""
PIPER Phase 2: SCADA Dataset Preparation Pipeline (Adapted Tabular Scope)

This script runs the complete Phase 2 pipeline on the SCADA dataset:
1. Dataset acquisition and validation
2. Telemetry selection (pressure, flow_rate, temperature - 3 primary telemetry)
3. Data cleaning (handles conflicting duplicate keys properly)
4. Feature engineering (temporal, pressure-flow ratio, metadata - NO long-window features)
5. Label generation (real labels from target column)
6. Temporal feasibility audit (honest assessment - sequence construction not supported)
7. Report generation

Current adapted scope: Tabular abnormal-condition classification.
Vibration is not required. Temporal forecasting not supported by current dataset.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

from utils.piper_dataset import (
    load_or_create_scada_piper_dataset,
    PIPER_PROCESSED_DIR,
    PIPER_REPORTS_DIR
)
from utils.piper_cleaning import clean_scada_dataset
from utils.piper_features import engineer_scada_features
from utils.piper_labels import label_scada_real
from utils.piper_telemetry import (
    PRIMARY_TELEMETRY, METADATA_COLUMNS, TARGET_COLUMN,
    TelemetryVariable, MetadataVariable, TargetVariable,
    validate_mapping, SCADA_COLUMN_MAPPING
)


# =============================================================================
# TEMPORAL FEASIBILITY AUDIT
# =============================================================================

def audit_temporal_feasibility(
    df: pd.DataFrame,
    sensor_col: str = MetadataVariable.SENSOR_ID.value,
    timestamp_col: str = MetadataVariable.TIMESTAMP.value,
    target_col: str = TARGET_COLUMN
) -> dict:
    """
    Comprehensive temporal feasibility audit for sequence construction.
    
    Reports per-segment and global statistics to determine if data
    can support chronological sequence construction for LSTM/GRU models.
    
    Current dataset limitation: ~16 minute span, multiple readings per timestamp.
    Sequence construction is NOT supported.
    """
    report = {
        "dataset_name": "SCADA_Pipeline",
        "audit_timestamp": datetime.now().isoformat(),
        "global": {},
        "per_segment": {},
        "feasibility_assessment": {}
    }
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Global statistics
    report["global"]["total_observations"] = int(len(df))
    report["global"]["total_segments"] = int(df[sensor_col].nunique())
    report["global"]["total_unique_timestamps"] = int(df[timestamp_col].nunique())
    report["global"]["timestamp_range"] = {
        "start": str(df[timestamp_col].min()),
        "end": str(df[timestamp_col].max()),
        "span_hours": float((df[timestamp_col].max() - df[timestamp_col].min()).total_seconds() / 3600)
    }
    
    # Duplicate timestamp analysis
    dup_keys = df.duplicated(subset=[sensor_col, timestamp_col])
    report["global"]["duplicate_segment_timestamp_keys"] = int(dup_keys.sum())
    report["global"]["exact_duplicate_rows"] = int(df.duplicated().sum())
    
    # Per-segment analysis
    segment_stats = []
    for seg_id, group in df.groupby(sensor_col):
        group = group.sort_values(timestamp_col)
        n_obs = len(group)
        n_unique_ts = group[timestamp_col].nunique()
        first_ts = group[timestamp_col].min()
        last_ts = group[timestamp_col].max()
        time_span_hours = (last_ts - first_ts).total_seconds() / 3600
        
        # Time differences between consecutive unique timestamps
        unique_ts = group[timestamp_col].drop_duplicates().sort_values()
        if len(unique_ts) > 1:
            diffs = unique_ts.diff().dt.total_seconds() / 60  # minutes
            diffs = diffs.dropna()
            interval_stats = {
                "mean_minutes": float(diffs.mean()),
                "median_minutes": float(diffs.median()),
                "std_minutes": float(diffs.std()),
                "min_minutes": float(diffs.min()),
                "max_minutes": float(diffs.max()),
                "n_intervals": int(len(diffs))
            }
        else:
            interval_stats = {
                "mean_minutes": 0,
                "median_minutes": 0,
                "std_minutes": 0,
                "min_minutes": 0,
                "max_minutes": 0,
                "n_intervals": 0
            }
        
        # Target distribution
        target_dist = group[target_col].value_counts().to_dict() if target_col in group.columns else {}
        
        segment_stats.append({
            "segment_id": int(seg_id),
            "n_observations": int(n_obs),
            "n_unique_timestamps": int(n_unique_ts),
            "duplicate_timestamps": int(n_obs - n_unique_ts),
            "first_timestamp": str(first_ts),
            "last_timestamp": str(last_ts),
            "time_span_hours": float(time_span_hours),
            "interval_statistics": interval_stats,
            "target_distribution": {str(k): int(v) for k, v in target_dist.items()},
            "has_both_classes": len(set(target_dist.keys())) > 1
        })
    
    report["per_segment"] = segment_stats
    
    # Feasibility assessment
    n_segments = len(segment_stats)
    segments_with_duplicates = sum(1 for s in segment_stats if s["duplicate_timestamps"] > 0)
    segments_with_both_classes = sum(1 for s in segment_stats if s["has_both_classes"])
    min_unique_ts = min(s["n_unique_timestamps"] for s in segment_stats)
    max_unique_ts = max(s["n_unique_timestamps"] for s in segment_stats)
    avg_unique_ts = np.mean([s["n_unique_timestamps"] for s in segment_stats])
    global_span_hours = report["global"]["timestamp_range"]["span_hours"]
    
    report["feasibility_assessment"] = {
        "total_segments": n_segments,
        "segments_with_duplicate_timestamps": segments_with_duplicates,
        "segments_with_both_target_classes": segments_with_both_classes,
        "min_unique_timestamps_per_segment": int(min_unique_ts),
        "max_unique_timestamps_per_segment": int(max_unique_ts),
        "avg_unique_timestamps_per_segment": float(avg_unique_ts),
        "global_time_span_hours": float(global_span_hours),
        "can_support_tabular_classification": True,
        "can_support_sequence_construction": False,  # Dataset span ~16 min, multiple readings per timestamp
        "can_support_prediction_horizon": False,      # Insufficient temporal depth
        "limitations": []
    }
    
    # Add limitations
    if segments_with_duplicates > 0:
        report["feasibility_assessment"]["limitations"].append(
            f"{segments_with_duplicates}/{n_segments} segments have multiple readings per timestamp "
            "(conflicting telemetry/target values). Preserved with reading_index; not a regular time series."
        )
    
    if global_span_hours < 1.0:
        report["feasibility_assessment"]["limitations"].append(
            f"Global time span is only {global_span_hours:.2f} hours "
            "(< 1 hour). Insufficient for temporal sequence modeling with meaningful horizons."
        )
    
    if min_unique_ts < 10:
        report["feasibility_assessment"]["limitations"].append(
            f"Minimum unique timestamps per segment is {min_unique_ts} "
            "(< 10). Insufficient for meaningful sequence windows."
        )
    
    if not all(s["has_both_classes"] for s in segment_stats):
        report["feasibility_assessment"]["limitations"].append(
            f"{n_segments - segments_with_both_classes}/{n_segments} segments have only one target class. "
            "Cannot train per-segment classifiers without both classes."
        )
    
    # Missing pipeline_id (not vibration - vibration not required for current scope)
    if MetadataVariable.PIPELINE_ID.value not in df.columns:
        report["feasibility_assessment"]["limitations"].append(
            "Metadata 'pipeline_id' is missing. Assuming single pipeline; segment_id used as sensor_id."
        )
    
    # Note about vibration (not a blocker for current adapted scope)
    if TelemetryVariable.VIBRATION.value not in df.columns:
        report["feasibility_assessment"]["limitations"].append(
            "Telemetry 'vibration' not present in dataset. Not required for current tabular classification scope "
            "(adapted PIPER scope uses 3 primary telemetry: pressure, flow_rate, temperature)."
        )
    
    return report


def save_temporal_feasibility_report(report: dict, output_path: Path) -> None:
    """Save temporal feasibility report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Temporal feasibility report saved to: {output_path}")


def print_temporal_feasibility_summary(report: dict) -> None:
    """Print human-readable temporal feasibility summary."""
    print(f"\n{'='*70}")
    print(f"TEMPORAL FEASIBILITY AUDIT: {report['dataset_name']}")
    print(f"{'='*70}")
    
    g = report["global"]
    print(f"\n--- GLOBAL STATISTICS ---")
    print(f"  Total observations:        {g['total_observations']:,}")
    print(f"  Total segments:            {g['total_segments']}")
    print(f"  Total unique timestamps:   {g['total_unique_timestamps']:,}")
    print(f"  Timestamp range:           {g['timestamp_range']['start']} to {g['timestamp_range']['end']}")
    print(f"  Time span:                 {g['timestamp_range']['span_hours']:.2f} hours")
    print(f"  Exact duplicate rows:      {g['exact_duplicate_rows']}")
    print(f"  Duplicate (seg, ts) keys:  {g['duplicate_segment_timestamp_keys']}")
    
    print(f"\n--- PER-SEGMENT SUMMARY ---")
    fa = report["feasibility_assessment"]
    print(f"  Segments:                    {fa['total_segments']}")
    print(f"  With duplicate timestamps:   {fa['segments_with_duplicate_timestamps']}")
    print(f"  With both target classes:    {fa['segments_with_both_target_classes']}")
    print(f"  Min unique timestamps/seg:   {fa['min_unique_timestamps_per_segment']}")
    print(f"  Max unique timestamps/seg:   {fa['max_unique_timestamps_per_segment']}")
    print(f"  Avg unique timestamps/seg:   {fa['avg_unique_timestamps_per_segment']:.1f}")
    
    print(f"\n--- FEASIBILITY ASSESSMENT ---")
    print(f"  Tabular classification:      {'YES' if fa['can_support_tabular_classification'] else 'NO'}")
    print(f"  Sequence construction:       {'YES' if fa['can_support_sequence_construction'] else 'NO'}")
    print(f"  Prediction horizon support:  {'YES' if fa['can_support_prediction_horizon'] else 'NO'}")
    
    if fa["limitations"]:
        print(f"\n--- LIMITATIONS ---")
        for lim in fa["limitations"]:
            print(f"  ! {lim}")
    else:
        print(f"\n  No significant limitations identified.")
    
    print(f"{'='*70}\n")


# =============================================================================
# FEATURE MANIFEST GENERATION
# =============================================================================

def generate_feature_manifest(
    engineered_df: pd.DataFrame,
    original_df: pd.DataFrame,
    dataset_name: str = "SCADA_Pipeline"
) -> dict:
    """
    Generate feature manifest documenting all features, their types,
    derivation rules, and exclusions with reasons.
    """
    from utils.piper_telemetry import (
        is_primary_telemetry, is_derived_feature, PRIMARY_TELEMETRY,
        METADATA_COLUMNS, TARGET_COLUMN
    )
    
    manifest = {
        "dataset_name": dataset_name,
        "generated_at": datetime.now().isoformat(),
        "features": {},
        "excluded_columns": {},
        "summary": {}
    }
    
    # Categorize all columns
    for col in engineered_df.columns:
        if is_primary_telemetry(col):
            feature_type = "primary_telemetry"
            derivation = "Direct sensor measurement (no transformation)"
            excluded = False
        elif col in METADATA_COLUMNS:
            feature_type = "metadata"
            derivation = "Identifier/timestamp column"
            excluded = False
        elif col == TARGET_COLUMN:
            feature_type = "target"
            derivation = "Ground truth label (0=normal, 1=abnormal)"
            excluded = True
            exclusion_reason = "Target variable - not a predictive feature"
        elif col == "event_type":
            feature_type = "event_metadata"
            derivation = "Original event type classification"
            excluded = True
            exclusion_reason = "Post-event information - would cause target leakage"
        elif col in ["alarm_triggered"]:
            feature_type = "auxiliary"
            derivation = "Auxiliary operational indicator"
            excluded = True
            exclusion_reason = "Directly encodes abnormal condition - target leakage risk"
        elif col in ["valve_status", "pump_state", "pump_speed", "compressor_state", "energy_consumption"]:
            feature_type = "auxiliary"
            derivation = "Operational state measurement"
            excluded = False  # Could be used as features with caution
        elif is_derived_feature(col) or col.startswith(("roll_", "diff_", "rel_", "pressure_", "flow_", "sensor_id_encoded", "pipeline_id_encoded", "n_sensors")):
            feature_type = "derived"
            derivation = _get_derivation_description(col)
            excluded = False
        elif col in ["hour", "dayofweek", "month", "quarter", "dayofyear", "weekofyear", "is_weekend",
                     "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "sin_month", "cos_month"]:
            feature_type = "temporal"
            derivation = f"Extracted from {MetadataVariable.TIMESTAMP.value}"
            excluded = False
        else:
            feature_type = "unknown"
            derivation = "Unclassified"
            excluded = False
        
        manifest["features"][col] = {
            "type": feature_type,
            "derivation": derivation,
            "excluded_from_predictive_features": excluded,
            "exclusion_reason": exclusion_reason if excluded else None,
            "dtype": str(engineered_df[col].dtype),
            "n_unique": int(engineered_df[col].nunique()),
            "missing_pct": float(engineered_df[col].isna().mean() * 100)
        }
    
    # Summary counts
    type_counts = {}
    for info in manifest["features"].values():
        t = info["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    
    manifest["summary"] = {
        "total_features": len(manifest["features"]),
        "by_type": type_counts,
        "predictive_features": sum(1 for v in manifest["features"].values() if not v["excluded_from_predictive_features"]),
        "excluded_features": sum(1 for v in manifest["features"].values() if v["excluded_from_predictive_features"]),
        "primary_telemetry_present": [pt for pt in PRIMARY_TELEMETRY if pt in engineered_df.columns],
        "primary_telemetry_missing": [pt for pt in PRIMARY_TELEMETRY if pt not in engineered_df.columns]
    }
    
    return manifest


def _get_derivation_description(col: str) -> str:
    """Get human-readable derivation description for derived features."""
    if col.startswith("roll_"):
        parts = col.split("_")
        if len(parts) >= 4:
            window = parts[1]
            stat = parts[2]
            base = "_".join(parts[3:])
            return f"Rolling {stat} of {base} over {window} window (per sensor)"
    elif col.startswith("diff_"):
        parts = col.split("_")
        if len(parts) >= 3:
            period = parts[1]
            base = "_".join(parts[2:])
            return f"{period} period difference of {base} (per sensor)"
    elif col.startswith("rel_to_roll_"):
        parts = col.split("_")
        if len(parts) >= 4:
            window = parts[2]
            base = "_".join(parts[3:])
            return f"{base} divided by {window} rolling mean (per sensor)"
    elif col.startswith("pressure_") or col.startswith("flow_"):
        return f"Pressure-flow relationship feature: {col}"
    elif col == "sensor_id_encoded":
        return "Integer encoding of sensor_id"
    elif col == "pipeline_id_encoded":
        return "Integer encoding of pipeline_id"
    elif col == "n_sensors":
        return "Total number of unique sensors in dataset"
    return "Derived feature"


def save_feature_manifest(manifest: dict, output_path: Path) -> None:
    """Save feature manifest to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Feature manifest saved to: {output_path}")


def print_feature_manifest_summary(manifest: dict) -> None:
    """Print feature manifest summary."""
    print(f"\n{'='*70}")
    print(f"FEATURE MANIFEST: {manifest['dataset_name']}")
    print(f"{'='*70}")
    print(f"Total features: {manifest['summary']['total_features']}")
    print(f"Predictive features: {manifest['summary']['predictive_features']}")
    print(f"Excluded features: {manifest['summary']['excluded_features']}")
    print(f"\nPrimary telemetry present: {manifest['summary']['primary_telemetry_present']}")
    print(f"Primary telemetry missing: {manifest['summary']['primary_telemetry_missing']}")
    print(f"\nBy type:")
    for t, count in manifest['summary']['by_type'].items():
        print(f"  {t}: {count}")
    
    print(f"\n--- EXCLUDED FEATURES ---")
    for col, info in manifest["features"].items():
        if info["excluded_from_predictive_features"]:
            print(f"  {col} ({info['type']}): {info['exclusion_reason']}")
    print(f"{'='*70}\n")


# =============================================================================
# PHASE 2 SUMMARY REPORT
# =============================================================================

def generate_phase2_summary(
    dataset_info,
    cleaning_report,
    feature_report,
    label_report,
    temporal_report,
    feature_manifest
) -> str:
    """Generate Phase 2 summary markdown report."""
    
    md = f"""# PIPER Phase 2 Summary Report: {dataset_info.name}

**Generated:** {datetime.now().isoformat()}

## Dataset Overview

| Property | Value |
|----------|-------|
| Dataset Name | {dataset_info.name} |
| Source Path | {dataset_info.path} |
| Format | {dataset_info.format} |
| Total Rows | {dataset_info.total_rows:,} |
| Total Columns (raw) | {len(dataset_info.columns)} |
| Date Range | {dataset_info.date_range[0]} to {dataset_info.date_range[1]} |
| Unique Sensors | {len(dataset_info.sensors)} |

## PIPER Telemetry Validation

"""
    
    val = dataset_info.validation_report
    md += f"**All Required Mapped:** {val['all_required_mapped']}\n\n"
    
    md += "### Primary Telemetry (Mandatory)\n\n"
    for var, info in val['primary_telemetry'].items():
        status = "[OK] MAPPED" if info['mapped'] else "[MISSING] MISSING - BLOCKER"
        md += f"- **{var}**: {status} (source: {info['source_column'] or 'N/A'})\n"
    
    md += "\n### Metadata\n\n"
    for var, info in val['metadata'].items():
        status = "[OK] MAPPED" if info['mapped'] else "[MISSING] MISSING"
        md += f"- **{var}**: {status} (source: {info['source_column'] or 'N/A'})\n"
    
    md += "\n### Target\n\n"
    for var, info in val['target'].items():
        status = "[OK] MAPPED" if info['mapped'] else "[MISSING] MISSING"
        md += f"- **{var}**: {status} (source: {info['source_column'] or 'N/A'})\n"
    
    if val['blockers']:
        md += "\n### Blockers\n\n"
        for b in val['blockers']:
            md += f"- {b}\n"
    
    md += f"""

## Data Cleaning

| Metric | Value |
|--------|-------|
| Raw Rows | {cleaning_report.total_rows_raw:,} |
| Clean Rows | {cleaning_report.total_rows_clean:,} |
| Rows Removed | {cleaning_report.total_rows_raw - cleaning_report.total_rows_clean:,} |
| Cleaning Steps | {len(cleaning_report.steps)} |

### Cleaning Steps Detail

"""
    
    for step in cleaning_report.steps:
        md += f"#### {step.step_name}\n"
        md += f"- **Description:** {step.description}\n"
        md += f"- **Rows Before:** {step.rows_before:,}\n"
        md += f"- **Rows After:** {step.rows_after:,}\n"
        md += f"- **Rows Affected:** {step.rows_affected:,}\n"
        if step.details:
            md += "- **Details:**\n"
            for k, v in step.details.items():
                md += f"  - {k}: {v}\n"
        md += "\n"
    
    md += f"""

## Feature Engineering

| Metric | Value |
|--------|-------|
| Input Columns | {len(feature_report.input_columns)} |
| Output Columns | {len(feature_report.output_columns)} |
| Primary Telemetry Present | {len(feature_report.primary_telemetry_present)}/3 |
| Primary Telemetry Missing | {len(feature_report.primary_telemetry_missing)}/3 |
| Derived Features | {len(feature_report.derived_features)} |
| Temporal Features | {len(feature_report.temporal_features)} |
| Metadata Features | {len(feature_report.metadata_features)} |

### Primary Telemetry Status

"""
    
    for pt in feature_report.primary_telemetry_present:
        md += f"- [OK] {pt}\n"
    for pt in feature_report.primary_telemetry_missing:
        md += f"- [MISSING] {pt} (BLOCKER)\n"
    
    md += f"""

### Feature Manifest Summary

| Category | Count |
|----------|-------|
| Total Features | {feature_manifest['summary']['total_features']} |
| Predictive Features | {feature_manifest['summary']['predictive_features']} |
| Excluded Features | {feature_manifest['summary']['excluded_features']} |

### Excluded Features (with reasons)

"""
    
    for col, info in feature_manifest["features"].items():
        if info["excluded_from_predictive_features"]:
            md += f"- **{col}** ({info['type']}): {info['exclusion_reason']}\n"
    
    md += f"""

## Label Generation

| Metric | Value |
|--------|-------|
| Label Source | {label_report.label_source.value} |
| Label Method | {label_report.label_method.value} |
| Normal (0) | {label_report.n_normal:,} |
| Abnormal (1) | {label_report.n_abnormal:,} |
| Total Labeled | {label_report.total_labeled:,} |
| Abnormal Rate | {label_report.class_balance:.4f} ({label_report.class_balance*100:.2f}%) |
| Is Synthetic | {label_report.is_synthetic} |

"""
    
    if label_report.warnings:
        md += "### Warnings\n\n"
        for w in label_report.warnings:
            md += f"- {w}\n"
    
    md += f"""

## Temporal Feasibility Audit

### Global Statistics

| Metric | Value |
|--------|-------|
| Total Observations | {temporal_report['global']['total_observations']:,} |
| Total Segments | {temporal_report['global']['total_segments']} |
| Total Unique Timestamps | {temporal_report['global']['total_unique_timestamps']:,} |
| Time Span (hours) | {temporal_report['global']['timestamp_range']['span_hours']:.2f} |
| Duplicate (segment, timestamp) Keys | {temporal_report['global']['duplicate_segment_timestamp_keys']} |
| Exact Duplicate Rows | {temporal_report['global']['exact_duplicate_rows']} |

### Per-Segment Summary

| Metric | Value |
|--------|-------|
| Segments with Duplicate Timestamps | {temporal_report['feasibility_assessment']['segments_with_duplicate_timestamps']} |
| Segments with Both Target Classes | {temporal_report['feasibility_assessment']['segments_with_both_target_classes']} |
| Min Unique Timestamps/Segment | {temporal_report['feasibility_assessment']['min_unique_timestamps_per_segment']} |
| Max Unique Timestamps/Segment | {temporal_report['feasibility_assessment']['max_unique_timestamps_per_segment']} |
| Avg Unique Timestamps/Segment | {temporal_report['feasibility_assessment']['avg_unique_timestamps_per_segment']:.1f} |

### Feasibility Assessment

| Capability | Supported |
|------------|-----------|
| Tabular Classification | {'[YES]' if temporal_report['feasibility_assessment']['can_support_tabular_classification'] else '[NO]'} |
| Sequence Construction | {'[YES]' if temporal_report['feasibility_assessment']['can_support_sequence_construction'] else '[NO]'} |
| Prediction Horizon Support | {'[YES]' if temporal_report['feasibility_assessment']['can_support_prediction_horizon'] else '[NO]'} |

### Limitations

"""
    
    if temporal_report['feasibility_assessment']['limitations']:
        for lim in temporal_report['feasibility_assessment']['limitations']:
            md += f"- {lim}\n"
    else:
        md += "No significant limitations identified.\n"
    
    md += f"""

## Phase 2 Completion Status

"""
    
    # Determine completion status
    blockers = len(val['blockers'])
    
    # For adapted scope: only 3 primary telemetry required, vibration not required
    # pipeline_id is optional metadata
    if blockers == 0:
        status = "COMPLETE (adapted tabular-data scope)"
    else:
        status = "PARTIALLY COMPLETE"
    
    md += f"**Status:** {status}\n\n"
    
    md += "### Artifacts Generated\n\n"
    artifacts = [
        "data/raw/scada_pipeline.csv (preserved unchanged)",
        "data/processed/scada_pipeline_cleaned.csv",
        "data/processed/scada_pipeline_features.csv",
        "data/processed/scada_feature_manifest.json",
        "data/reports/scada_dataset_profile.json",
        "data/reports/scada_cleaning_report.json",
        "data/reports/scada_label_report.json",
        "data/reports/scada_temporal_feasibility.json",
        "data/reports/phase2_summary.md"
    ]
    for a in artifacts:
        md += f"- {a}\n"
    
    md += f"""

## Next Steps (Phase 3)

"""
    
    if status == "COMPLETE (adapted tabular-data scope)":
        md += """Phase 2 requirements for the adapted tabular classification scope are satisfied.

**Current scope:** Tabular abnormal-condition classification using 3 primary telemetry (pressure, flow_rate, temperature).

**Dataset limitations acknowledged:**
- Sequence construction (LSTM/GRU) NOT supported by current dataset (~16 min span, multiple readings per timestamp)
- Temporal forecasting NOT supported
- Vibration telemetry not required for current scope

**Phase 3 can begin** for tabular model development (Random Forest, XGBoost, MLP, SVM).
Sequence modeling track should be deferred until richer temporal data is available.
"""
    else:
        md += "Phase 2 has known limitations. Review the following before proceeding to Phase 3:\n"
        for lim in temporal_report['feasibility_assessment']['limitations']:
            md += f"- {lim}\n"
        md += "\nPhase 3 can begin with awareness of these limitations, but sequence construction may not yield meaningful results.\n"
    
    return md


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("="*70)
    print("PIPER PHASE 2: SCADA DATASET PREPARATION")
    print("="*70)
    
    # 1. Dataset Acquisition
    print("\n[1/7] Loading SCADA dataset...")
    df_mapped, dataset_info = load_or_create_scada_piper_dataset(force_reload=False)
    print(f"    Loaded {len(df_mapped):,} rows, {len(df_mapped.columns)} columns")
    print(f"    Columns: {df_mapped.columns.tolist()}")
    
    # Validate telemetry
    print("\n[2/7] Validating PIPER telemetry mapping...")
    validation_report = validate_mapping(SCADA_COLUMN_MAPPING)
    dataset_info.validation_report = validation_report
    print(f"    All required mapped: {validation_report['all_required_mapped']}")
    for pt in PRIMARY_TELEMETRY:
        status = validation_report['primary_telemetry'][pt]['status']
        print(f"    {pt}: {status}")
    
    # Save dataset profile
    profile_path = PIPER_REPORTS_DIR / "scada_dataset_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_data = {
        "dataset_name": dataset_info.name,
        "source_path": str(dataset_info.path),
        "format": dataset_info.format,
        "n_files": dataset_info.n_files,
        "total_rows": dataset_info.total_rows,
        "columns": dataset_info.columns,
        "date_range": dataset_info.date_range,
        "sensors": dataset_info.sensors,
        "validation_report": validation_report
    }
    with open(profile_path, 'w') as f:
        json.dump(profile_data, f, indent=2, default=str)
    print(f"    Dataset profile saved to: {profile_path}")
    
    # 3. Data Cleaning
    print("\n[3/7] Running data cleaning pipeline...")
    clean_df, cleaning_report = clean_scada_dataset(df_mapped, save_report=True)
    print(f"    Cleaned: {len(clean_df):,} rows")
    
    # Save cleaned data
    cleaned_path = PIPER_PROCESSED_DIR / "scada_pipeline_cleaned.csv"
    PIPER_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(cleaned_path, index=False)
    print(f"    Cleaned data saved to: {cleaned_path}")
    
    # 4. Feature Engineering
    print("\n[4/7] Running feature engineering...")
    engineered_df, feature_report = engineer_scada_features(clean_df, save_report=True)
    print(f"    Engineered: {len(engineered_df):,} rows, {len(engineered_df.columns)} columns")
    
    # Save feature-engineered data
    features_path = PIPER_PROCESSED_DIR / "scada_pipeline_features.csv"
    engineered_df.to_csv(features_path, index=False)
    print(f"    Feature data saved to: {features_path}")
    
    # 5. Label Generation (real labels from target column)
    print("\n[5/7] Generating labels from existing target column...")
    labeled_df, label_report = label_scada_real(engineered_df, save_report=True)
    print(f"    Labeled: {len(labeled_df):,} rows")
    print(f"    Class distribution: {labeled_df[TARGET_COLUMN].value_counts().to_dict()}")
    
    # 6. Temporal Feasibility Audit
    print("\n[6/7] Running temporal feasibility audit...")
    temporal_report = audit_temporal_feasibility(labeled_df)
    temporal_path = PIPER_REPORTS_DIR / "scada_temporal_feasibility.json"
    save_temporal_feasibility_report(temporal_report, temporal_path)
    print_temporal_feasibility_summary(temporal_report)
    
    # 7. Feature Manifest
    print("\n[7/7] Generating feature manifest...")
    feature_manifest = generate_feature_manifest(labeled_df, df_mapped)
    manifest_path = PIPER_PROCESSED_DIR / "scada_feature_manifest.json"
    save_feature_manifest(feature_manifest, manifest_path)
    print_feature_manifest_summary(feature_manifest)
    
    # Phase 2 Summary
    print("\n" + "="*70)
    print("GENERATING PHASE 2 SUMMARY")
    print("="*70)
    
    summary_md = generate_phase2_summary(
        dataset_info, cleaning_report, feature_report,
        label_report, temporal_report, feature_manifest
    )
    
    summary_path = PIPER_REPORTS_DIR / "phase2_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary_md)
    print(f"Phase 2 summary saved to: {summary_path}")
    print("\n" + summary_md)
    
    print("\n" + "="*70)
    print("PHASE 2 PIPELINE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()