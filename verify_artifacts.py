import os
import pandas as pd
import json

print('=== Verifying artifacts exist ===')
artifacts = [
    'data/raw/scada_pipeline.csv',
    'data/raw/scada_raw.parquet',
    'data/processed/scada_pipeline_cleaned.csv',
    'data/processed/scada_pipeline_features.csv',
    'data/processed/scada_feature_manifest.json',
    'data/processed/scada_piper_mapped.parquet',
    'data/reports/scada_dataset_profile.json',
    'data/reports/cleaning_report_SCADA_Pipeline.json',
    'data/reports/label_report_SCADA_Pipeline.json',
    'data/reports/scada_temporal_feasibility.json',
    'data/reports/phase2_summary.md'
]
for a in artifacts:
    exists = os.path.exists(a)
    size = os.path.getsize(a) if exists else 0
    status = "OK" if exists else "MISSING"
    print(f'  {status}: {a} ({size} bytes)')

print()
print('=== Verify raw CSV unchanged ===')
raw_df = pd.read_csv('data/raw/scada_pipeline.csv')
print(f'Raw CSV shape: {raw_df.shape}')
print(f'Raw CSV columns: {raw_df.columns.tolist()}')

print()
print('=== Verify cleaned data ===')
clean_df = pd.read_csv('data/processed/scada_pipeline_cleaned.csv')
print(f'Cleaned shape: {clean_df.shape}')
print(f'Cleaned columns: {clean_df.columns.tolist()}')

print()
print('=== Verify feature data ===')
feat_df = pd.read_csv('data/processed/scada_pipeline_features.csv')
print(f'Feature shape: {feat_df.shape}')
print(f'Feature columns count: {len(feat_df.columns)}')

print()
print('=== Verify feature manifest ===')
with open('data/processed/scada_feature_manifest.json') as f:
    manifest = json.load(f)
print(f'Total features: {manifest["summary"]["total_features"]}')
print(f'Predictive: {manifest["summary"]["predictive_features"]}')
print(f'Excluded: {manifest["summary"]["excluded_features"]}')
print(f'Primary present: {manifest["summary"]["primary_telemetry_present"]}')
print(f'Primary missing: {manifest["summary"]["primary_telemetry_missing"]}')

print()
print('=== Verify label report ===')
with open('data/reports/label_report_SCADA_Pipeline.json') as f:
    label = json.load(f)
print(f'Label source: {label["label_source"]}')
print(f'Is synthetic: {label["is_synthetic"]}')
print(f'Normal: {label["n_normal"]}')
print(f'Abnormal: {label["n_abnormal"]}')
print(f'Class balance: {label["class_balance"]:.4f}')

print()
print('=== Verify cleaning report ===')
with open('data/reports/cleaning_report_SCADA_Pipeline.json') as f:
    cleaning = json.load(f)
print(f'Dataset: {cleaning["dataset_name"]}')
print(f'Raw rows: {cleaning["total_rows_raw"]}')
print(f'Clean rows: {cleaning["total_rows_clean"]}')
print(f'Rows removed: {cleaning["total_rows_raw"] - cleaning["total_rows_clean"]}')
print(f'Steps: {len(cleaning["steps"])}')

print()
print('=== Verify temporal feasibility ===')
with open('data/reports/scada_temporal_feasibility.json') as f:
    temporal = json.load(f)
fa = temporal['feasibility_assessment']
print(f'Tabular: {fa["can_support_tabular_classification"]}')
print(f'Sequence: {fa["can_support_sequence_construction"]}')
print(f'Horizon: {fa["can_support_prediction_horizon"]}')
print(f'Limitations: {len(fa["limitations"])}')
for lim in fa["limitations"]:
    print(f'  - {lim}')

print()
print('=== Verify no vibration column or fabricated vibration feature ===')
feat_cols = feat_df.columns.tolist()
vibration_cols = [c for c in feat_cols if 'vibration' in c.lower()]
print(f'Vibration columns in features: {vibration_cols}')
if not vibration_cols:
    print('CONFIRMED: No vibration column or fabricated vibration feature')
else:
    print('WARNING: Vibration column found!')

print()
print('=== Verify target leakage prevention ===')
excluded = [c for c, info in manifest["features"].items() if info["excluded_from_predictive_features"]]
print(f'Excluded from predictive features: {excluded}')
leakage_cols = ['label', 'event_type', 'alarm_triggered']
for col in leakage_cols:
    if col in feat_cols:
        info = manifest["features"].get(col, {})
        excluded_flag = info.get("excluded_from_predictive_features", "UNKNOWN")
        print(f'  {col}: excluded={excluded_flag}, reason={info.get("exclusion_reason", "N/A")}')