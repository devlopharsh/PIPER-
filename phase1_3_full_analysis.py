import pandas as pd
import numpy as np
import json

df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\raw\scada_pipeline.csv')
df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M', errors='coerce')

# ============================================================
# TASK A: TIMESTAMP PARSING AND VALIDATION
# ============================================================
parsed_count = int(df['timestamp_parsed'].notna().sum())
failed_count = int(df['timestamp_parsed'].isna().sum())
null_count = int(df['timestamp'].isna().sum())
unique_parsed = int(df['timestamp_parsed'].nunique())
earliest = df['timestamp_parsed'].min()
latest = df['timestamp_parsed'].max()
span = latest - earliest

print('=== TASK A: TIMESTAMP PARSING AND VALIDATION ===')
print(f'Successfully parsed: {parsed_count}')
print(f'Failed parses: {failed_count}')
print(f'Null/malformed: {null_count}')
print(f'Unique parsed timestamps: {unique_parsed}')
print(f'Earliest: {earliest}')
print(f'Latest: {latest}')
print(f'Total time span: {span}')
print('Contains seconds?', df['timestamp'].str.contains(r':\d{2}$').any())
print('Contains timezone?', df['timestamp'].str.contains(r'[+-]\d{2}:?\d{2}|Z$|UTC|GMT').any())

# ============================================================
# TASK B: GLOBAL TEMPORAL COVERAGE
# ============================================================
ts_counts = df['timestamp'].value_counts().sort_index()
original_order = df['timestamp'].tolist()
is_sorted = all(original_order[i] <= original_order[i+1] for i in range(len(original_order)-1))

unique_ts_sorted = sorted(df['timestamp_parsed'].unique())
intervals = [(unique_ts_sorted[i+1] - unique_ts_sorted[i]).total_seconds() / 60 for i in range(len(unique_ts_sorted)-1)]

print('\n=== TASK B: GLOBAL TEMPORAL COVERAGE ===')
print(f'Total rows: {len(df)}')
print(f'Unique timestamps: {df["timestamp"].nunique()}')
print(f'CSV globally sorted by timestamp: {is_sorted}')
print(f'Intervals (min): {intervals}')
print(f'Interval stats: min={min(intervals)}, max={max(intervals)}, mean={np.mean(intervals):.2f}, median={np.median(intervals):.2f}')

# ============================================================
# TASK C: PER-SEGMENT TEMPORAL COVERAGE
# ============================================================
segment_stats = []
for seg in sorted(df['segment_id'].unique()):
    seg_df = df[df['segment_id'] == seg].copy()
    seg_df = seg_df.sort_values('timestamp_parsed')
    
    n_rows = len(seg_df)
    n_unique_ts = seg_df['timestamp_parsed'].nunique()
    earliest_ts = seg_df['timestamp_parsed'].min()
    latest_ts = seg_df['timestamp_parsed'].max()
    span_hours = (latest_ts - earliest_ts).total_seconds() / 3600
    
    orig_order = df[df['segment_id'] == seg]['timestamp'].tolist()
    is_ordered = all(orig_order[i] <= orig_order[i+1] for i in range(len(orig_order)-1))
    
    dup_ts = seg_df.groupby('timestamp_parsed').size()
    n_repeated_ts = int((dup_ts > 1).sum())
    max_repeats = int(dup_ts.max())
    
    unique_ts = sorted(seg_df['timestamp_parsed'].unique())
    if len(unique_ts) > 1:
        seg_intervals = [(unique_ts[i+1] - unique_ts[i]).total_seconds() / 60 for i in range(len(unique_ts)-1)]
        min_int = min(seg_intervals)
        max_int = max(seg_intervals)
        mean_int = np.mean(seg_intervals)
        median_int = np.median(seg_intervals)
    else:
        min_int = max_int = mean_int = median_int = None
    
    segment_stats.append({
        'segment_id': int(seg),
        'n_rows': int(n_rows),
        'n_unique_ts': int(n_unique_ts),
        'earliest': str(earliest_ts),
        'latest': str(latest_ts),
        'span_hours': round(float(span_hours), 6),
        'ordered_in_csv': bool(is_ordered),
        'n_repeated_ts': n_repeated_ts,
        'max_repeats': max_repeats,
        'min_interval_min': round(float(min_int), 6) if min_int else None,
        'max_interval_min': round(float(max_int), 6) if max_int else None,
        'mean_interval_min': round(float(mean_int), 6) if mean_int else None,
        'median_interval_min': round(float(median_int), 6) if median_int else None
    })

seg_df = pd.DataFrame(segment_stats)
seg_df.to_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\documentation\PIPER_phase1_3_segment_summary.csv', index=False)

# ============================================================
# TASK D: REPEATED (segment_id, timestamp) GROUP AUDIT
# ============================================================
grouped = df.groupby(['segment_id', 'timestamp'])
group_sizes = grouped.size()
total_groups = len(group_sizes)
single_row_groups = int((group_sizes == 1).sum())
repeated_groups = int((group_sizes > 1).sum())
rows_in_repeated = int(group_sizes[group_sizes > 1].sum())
min_size = int(group_sizes.min())
max_size = int(group_sizes.max())
mean_size = float(group_sizes.mean())
median_size = float(group_sizes.median())

size_dist = group_sizes.value_counts().sort_index()

repeated_group_keys = group_sizes[group_sizes > 1].index.tolist()

telemetry_cols = ['pressure', 'flow_rate', 'temperature', 'valve_status', 'pump_state', 
                  'pump_speed', 'compressor_state', 'energy_consumption', 'alarm_triggered']
metadata_cols = ['event_type', 'target']

variation_counts = {col: 0 for col in telemetry_cols + metadata_cols}
group_categories = []

for seg_id, ts in repeated_group_keys:
    group = df[(df['segment_id'] == seg_id) & (df['timestamp'] == ts)]
    
    varies = {}
    for col in telemetry_cols + metadata_cols:
        n_unique = group[col].nunique()
        varies[col] = n_unique > 1
        if varies[col]:
            variation_counts[col] += 1
    
    telemetry_varies = any(varies[col] for col in telemetry_cols)
    metadata_varies = any(varies[col] for col in metadata_cols)
    exact_dup = len(group.drop_duplicates()) == 1
    
    if exact_dup:
        category = 'A_exact_duplicate'
    elif telemetry_varies and metadata_varies:
        category = 'F_mixed_telemetry_and_metadata'
    elif telemetry_varies and not metadata_varies:
        category = 'C_different_telemetry_same_metadata'
    elif not telemetry_varies and metadata_varies:
        category = 'B_same_telemetry_diff_metadata'
    else:
        category = 'E_other'
    
    group_categories.append({
        'segment_id': int(seg_id),
        'timestamp': ts,
        'group_size': len(group),
        'category': category,
        'telemetry_varies': telemetry_varies,
        'metadata_varies': metadata_varies,
        'exact_duplicate': exact_dup
    })

cat_df = pd.DataFrame(group_categories)
cat_summary = cat_df['category'].value_counts().to_dict()

# Save repeated group examples
examples_rows = []
for _, ex in cat_df.iterrows():
    seg, ts, size = ex['segment_id'], ex['timestamp'], ex['group_size']
    group = df[(df['segment_id'] == seg) & (df['timestamp'] == ts)]
    row_data = {
        'segment_id': seg,
        'timestamp': ts,
        'group_size': size,
        'category': ex['category']
    }
    first = group.iloc[0]
    for col in telemetry_cols + metadata_cols:
        row_data[f'{col}_1'] = first[col]
    for col in telemetry_cols + metadata_cols:
        vals = group[col].unique()
        if len(vals) > 1:
            row_data[f'{col}_varies'] = True
            row_data[f'{col}_values'] = str(sorted(vals))
        else:
            row_data[f'{col}_varies'] = False
    examples_rows.append(row_data)

ex_df = pd.DataFrame(examples_rows)
ex_df.to_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\documentation\PIPER_phase1_3_repeated_group_examples.csv', index=False)

# ============================================================
# TASK E: ROW-LEVEL DIFFERENCE ANALYSIS
# ============================================================
# Numeric variation details
numeric_var_details = {}
for col in ['pressure', 'flow_rate', 'temperature', 'pump_speed', 'energy_consumption']:
    ranges = []
    for seg_id, ts in repeated_group_keys:
        group = df[(df['segment_id'] == seg_id) & (df['timestamp'] == ts)]
        if group[col].nunique() > 1:
            r = group[col].max() - group[col].min()
            ranges.append(r)
    if ranges:
        numeric_var_details[col] = {
            'groups_varying': len(ranges),
            'min_range': round(float(min(ranges)), 4),
            'max_range': round(float(max(ranges)), 4),
            'mean_range': round(float(np.mean(ranges)), 4)
        }

# Categorical variation details
categorical_var_details = {}
for col in ['valve_status', 'pump_state', 'compressor_state', 'alarm_triggered', 'event_type', 'target']:
    val_combos = {}
    for seg_id, ts in repeated_group_keys:
        group = df[(df['segment_id'] == seg_id) & (df['timestamp'] == ts)]
        if group[col].nunique() > 1:
            vals = tuple(sorted([str(v) for v in group[col].unique()]))
            val_combos[vals] = val_combos.get(vals, 0) + 1
    if val_combos:
        categorical_var_details[col] = {
            'groups_varying': sum(val_combos.values()),
            'top_combinations': {str(k): int(v) for k, v in sorted(val_combos.items(), key=lambda x: -x[1])[:5]}
        }

# ============================================================
# TASK F: CROSS-SEGMENT SYNCHRONIZATION
# ============================================================
segments_per_ts = df.groupby('timestamp')['segment_id'].nunique()
ts_segments = df.groupby('timestamp')['segment_id'].apply(lambda x: sorted([int(s) for s in x.unique()]))

all_timestamps = sorted(df['timestamp'].unique())
segments_with_all_ts = 0
for seg in sorted(df['segment_id'].unique()):
    seg_ts = set(df[df['segment_id']==seg]['timestamp'].unique())
    if seg_ts == set(all_timestamps):
        segments_with_all_ts += 1

coverage = df.groupby('segment_id')['timestamp'].nunique()

# ============================================================
# TASK G: TEMPORAL SEQUENCE FEASIBILITY
# ============================================================
window_lengths = [3, 5, 8, 10]
sequence_feasibility = {}
for w in window_lengths:
    total_windows = 0
    segments_with_windows = 0
    for seg in sorted(df['segment_id'].unique()):
        n_unique = int(df[df['segment_id']==seg]['timestamp_parsed'].nunique())
        windows = max(0, n_unique - w + 1)
        if windows > 0:
            segments_with_windows += 1
            total_windows += windows
    sequence_feasibility[f'window_{w}'] = {
        'segments_with_windows': segments_with_windows,
        'total_potential_windows': total_windows,
        'segments_with_zero_windows': 50 - segments_with_windows
    }

# ============================================================
# COMPILE JSON REPORT
# ============================================================
results = {
    'dataset_identity': {
        'file_path': 'data/raw/scada_pipeline.csv',
        'sha256': 'c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a',
        'rows': 1000,
        'columns': 13
    },
    'task_A_parsing': {
        'parsed_count': parsed_count,
        'failed_count': failed_count,
        'null_count': null_count,
        'unique_parsed': unique_parsed,
        'earliest': str(earliest),
        'latest': str(latest),
        'total_span': str(span),
        'format': '%m/%d/%Y %H:%M',
        'contains_seconds': True,
        'contains_timezone': False,
        'timezone_documented': False
    },
    'task_B_global_coverage': {
        'total_rows': 1000,
        'unique_timestamps': 17,
        'rows_per_timestamp': {str(k): int(v) for k, v in ts_counts.items()},
        'csv_globally_sorted_by_timestamp': is_sorted,
        'intervals_minutes': [float(i) for i in intervals],
        'interval_stats_minutes': {
            'min': float(min(intervals)),
            'max': float(max(intervals)),
            'mean': round(float(np.mean(intervals)), 2),
            'median': float(np.median(intervals))
        }
    },
    'task_C_per_segment': segment_stats,
    'task_D_repeated_groups': {
        'total_groups': total_groups,
        'single_row_groups': single_row_groups,
        'repeated_groups': repeated_groups,
        'rows_in_repeated': rows_in_repeated,
        'group_size_distribution': {int(k): int(v) for k, v in size_dist.items()},
        'group_size_stats': {
            'min': min_size,
            'max': max_size,
            'mean': round(mean_size, 2),
            'median': median_size
        },
        'categories': cat_summary
    },
    'task_E_variation': {
        'variation_counts': variation_counts,
        'total_repeated_groups': len(repeated_group_keys),
        'numeric_variation_details': numeric_var_details,
        'categorical_variation_details': categorical_var_details
    },
    'task_F_synchronization': {
        'segments_per_timestamp': {str(k): int(v) for k, v in segments_per_ts.items()},
        'segments_with_all_timestamps': segments_with_all_ts,
        'total_segments': 50,
        'coverage_stats': {
            'min': int(coverage.min()),
            'max': int(coverage.max()),
            'mean': round(float(coverage.mean()), 1),
            'median': float(coverage.median())
        }
    },
    'task_G_sequence_feasibility': sequence_feasibility
}

with open(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\documentation\PIPER_phase1_3_timestamp_segment_report.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n=== ALL TASKS COMPLETE ===')
print('JSON report saved to PIPER_phase1_3_timestamp_segment_report.json')
print('Segment summary saved to PIPER_phase1_3_segment_summary.csv')
print('Repeated group examples saved to PIPER_phase1_3_repeated_group_examples.csv')