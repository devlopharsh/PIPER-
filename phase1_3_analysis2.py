import pandas as pd
import numpy as np
import json

df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\raw\scada_pipeline.csv')
df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M', errors='coerce')

# Task D: Repeated (segment_id, timestamp) group audit
print('=== TASK D: REPEATED (segment_id, timestamp) GROUP AUDIT ===')

# Group by segment_id and timestamp
grouped = df.groupby(['segment_id', 'timestamp'])
group_sizes = grouped.size()
total_groups = len(group_sizes)
single_row_groups = (group_sizes == 1).sum()
repeated_groups = (group_sizes > 1).sum()
rows_in_repeated = group_sizes[group_sizes > 1].sum()
min_size = group_sizes.min()
max_size = group_sizes.max()
mean_size = group_sizes.mean()
median_size = group_sizes.median()

print(f'Total distinct segment-timestamp groups: {total_groups}')
print(f'Groups with exactly 1 row: {single_row_groups}')
print(f'Groups with >1 row: {repeated_groups}')
print(f'Rows in repeated groups: {rows_in_repeated}')
print(f'Group size stats: min={min_size}, max={max_size}, mean={mean_size:.2f}, median={median_size:.2f}')

# Distribution of group sizes
size_dist = group_sizes.value_counts().sort_index()
print('\nGroup size distribution:')
for size, count in size_dist.items():
    print(f'  Size {size}: {count} groups ({count*size} rows)')

# Detailed analysis of repeated groups
repeated_group_keys = group_sizes[group_sizes > 1].index.tolist()
print(f'\nAnalyzing {len(repeated_group_keys)} repeated groups...')

# For each repeated group, check which columns vary
telemetry_cols = ['pressure', 'flow_rate', 'temperature', 'valve_status', 'pump_state', 
                  'pump_speed', 'compressor_state', 'energy_consumption', 'alarm_triggered']
metadata_cols = ['event_type', 'target']

variation_counts = {col: 0 for col in telemetry_cols + metadata_cols}
group_categories = []

for seg_id, ts in repeated_group_keys:
    group = df[(df['segment_id'] == seg_id) & (df['timestamp'] == ts)]
    
    # Check variations
    varies = {}
    for col in telemetry_cols + metadata_cols:
        n_unique = group[col].nunique()
        varies[col] = n_unique > 1
        if varies[col]:
            variation_counts[col] += 1
    
    # Categorize the group
    telemetry_varies = any(varies[col] for col in telemetry_cols)
    metadata_varies = any(varies[col] for col in metadata_cols)
    
    # Check if exact duplicate (all columns same)
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
print('\nGroup categories:')
cat_summary = cat_df['category'].value_counts()
for cat, count in cat_summary.items():
    pct = count / len(cat_df) * 100
    print(f'  {cat}: {count} groups ({pct:.1f}%)')

# Representative examples for each category
print('\nRepresentative examples per category:')
for cat in cat_df['category'].unique():
    examples = cat_df[cat_df['category'] == cat].head(3)
    print(f'\n  {cat} ({len(cat_df[cat_df["category"]==cat])} groups):')
    for _, ex in examples.iterrows():
        seg, ts, size = ex['segment_id'], ex['timestamp'], ex['group_size']
        group = df[(df['segment_id'] == seg) & (df['timestamp'] == ts)]
        print(f'    Segment {seg}, {ts}, size={size}')
        # Show differing columns
        for col in telemetry_cols + metadata_cols:
            vals = group[col].unique()
            if len(vals) > 1:
                print(f'      {col}: {vals}')

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
    # Add first row values
    first = group.iloc[0]
    for col in telemetry_cols + metadata_cols:
        row_data[f'{col}_1'] = first[col]
    # Add differing columns info
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
print('\nRepeated group examples saved to PIPER_phase1_3_repeated_group_examples.csv')

# Task E: Row-level difference analysis
print('\n=== TASK E: ROW-LEVEL DIFFERENCE ANALYSIS ===')
print('\nVariation frequency across repeated groups:')
for col in telemetry_cols + metadata_cols:
    count = variation_counts[col]
    pct = count / len(repeated_group_keys) * 100
    print(f'  {col}: {count}/{len(repeated_group_keys)} groups ({pct:.1f}%)')

# Numeric variation details
print('\nNumeric variation details (within-group range):')
for col in ['pressure', 'flow_rate', 'temperature', 'pump_speed', 'energy_consumption']:
    ranges = []
    for seg_id, ts in repeated_group_keys:
        group = df[(df['segment_id'] == seg_id) & (df['timestamp'] == ts)]
        if group[col].nunique() > 1:
            r = group[col].max() - group[col].min()
            ranges.append(r)
    if ranges:
        print(f'  {col}: {len(ranges)} groups vary, range: min={min(ranges):.4f}, max={max(ranges):.4f}, mean={np.mean(ranges):.4f}')

# Categorical/binary variation details
print('\nCategorical/binary variation details:')
for col in ['valve_status', 'pump_state', 'compressor_state', 'alarm_triggered', 'event_type', 'target']:
    val_combos = {}
    for seg_id, ts in repeated_group_keys:
        group = df[(df['segment_id'] == seg_id) & (df['timestamp'] == ts)]
        if group[col].nunique() > 1:
            vals = tuple(sorted(group[col].unique()))
            val_combos[vals] = val_combos.get(vals, 0) + 1
    if val_combos:
        print(f'  {col}: varies in {sum(val_combos.values())} groups')
        for vals, cnt in sorted(val_combos.items(), key=lambda x: -x[1])[:5]:
            print(f'    Values {vals}: {cnt} groups')

# Task F: Cross-segment synchronization
print('\n=== TASK F: CROSS-SEGMENT SYNCHRONIZATION ===')

# Number of segments at each timestamp
segments_per_ts = df.groupby('timestamp')['segment_id'].nunique()
print('\nSegments represented at each timestamp:')
for ts, cnt in segments_per_ts.items():
    print(f'  {ts}: {cnt} segments')

# Segments present at each timestamp
ts_segments = df.groupby('timestamp')['segment_id'].apply(lambda x: sorted(x.unique()))
print('\nSegment presence at each timestamp:')
for ts in sorted(df['timestamp'].unique()):
    segs = ts_segments[ts]
    print(f'  {ts}: {len(segs)} segments - {segs[:10]}{"..." if len(segs)>10 else ""}')

# Check if all segments appear at all timestamps
all_timestamps = sorted(df['timestamp'].unique())
segments_with_all_ts = 0
for seg in sorted(df['segment_id'].unique()):
    seg_ts = set(df[df['segment_id']==seg]['timestamp'].unique())
    if seg_ts == set(all_timestamps):
        segments_with_all_ts += 1

print(f'\nSegments present at ALL 17 timestamps: {segments_with_all_ts}/50')
print(f'Segments missing at least 1 timestamp: {50 - segments_with_all_ts}/50')

# Coverage per segment
coverage = df.groupby('segment_id')['timestamp'].nunique()
print(f'\nPer-segment timestamp coverage:')
print(f'  Min: {coverage.min()} / 17')
print(f'  Max: {coverage.max()} / 17')
print(f'  Mean: {coverage.mean():.1f} / 17')
print(f'  Median: {coverage.median():.1f} / 17')

# Task G: Temporal sequence feasibility
print('\n=== TASK G: TEMPORAL SEQUENCE FEASIBILITY ===')

window_lengths = [3, 5, 8, 10]
for w in window_lengths:
    print(f'\nWindow length = {w}:')
    total_windows = 0
    segments_with_windows = 0
    for seg in sorted(df['segment_id'].unique()):
        n_unique = df[df['segment_id']==seg]['timestamp_parsed'].nunique()
        windows = max(0, n_unique - w + 1)
        if windows > 0:
            segments_with_windows += 1
            total_windows += windows
    print(f'  Segments with >=1 window: {segments_with_windows}/50')
    print(f'  Total potential windows: {total_windows}')
    print(f'  Segments with 0 windows: {50 - segments_with_windows}')

# Save all results to JSON
results = {
    'task_A_parsing': {
        'parsed_count': int(parsed_count),
        'failed_count': int(failed_count),
        'null_count': int(null_count),
        'unique_parsed': int(unique_parsed),
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
        'csv_globally_sorted_by_timestamp': True,
        'intervals_minutes': [float(i) for i in intervals],
        'interval_stats_min': {
            'min': float(min(intervals)),
            'max': float(max(intervals)),
            'mean': float(np.mean(intervals)),
            'median': float(np.median(intervals))
        }
    },
    'task_C_per_segment': segment_stats,
    'task_D_repeated_groups': {
        'total_groups': int(total_groups),
        'single_row_groups': int(single_row_groups),
        'repeated_groups': int(repeated_groups),
        'rows_in_repeated': int(rows_in_repeated),
        'group_size_distribution': {int(k): int(v) for k, v in size_dist.items()},
        'group_size_stats': {
            'min': int(min_size),
            'max': int(max_size),
            'mean': round(float(mean_size), 2),
            'median': float(median_size)
        },
        'categories': cat_df['category'].value_counts().to_dict()
    },
    'task_E_variation': {
        'variation_counts': variation_counts,
        'total_repeated_groups': len(repeated_group_keys)
    },
    'task_F_synchronization': {
        'segments_per_timestamp': {str(k): int(v) for k, v in segments_per_ts.items()},
        'segments_with_all_timestamps': int(segments_with_all_ts),
        'coverage_stats': {
            'min': int(coverage.min()),
            'max': int(coverage.max()),
            'mean': round(float(coverage.mean()), 1),
            'median': float(coverage.median())
        }
    },
    'task_G_sequence_feasibility': {
        f'window_{w}': {
            'segments_with_windows': 0,
            'total_potential_windows': 0,
            'segments_with_zero_windows': 50
        } for w in window_lengths
    }
}

# Fill in task_G properly
for w in window_lengths:
    total_windows = 0
    segments_with_windows = 0
    for seg in sorted(df['segment_id'].unique()):
        n_unique = int(df[df['segment_id']==seg]['timestamp_parsed'].nunique())
        windows = max(0, n_unique - w + 1)
        if windows > 0:
            segments_with_windows += 1
            total_windows += windows
    results['task_G_sequence_feasibility'][f'window_{w}'] = {
        'segments_with_windows': segments_with_windows,
        'total_potential_windows': total_windows,
        'segments_with_zero_windows': 50 - segments_with_windows
    }

with open(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\documentation\PIPER_phase1_3_timestamp_segment_report.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\nFull JSON report saved to PIPER_phase1_3_timestamp_segment_report.json')