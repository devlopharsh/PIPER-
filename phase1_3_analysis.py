import pandas as pd
import numpy as np
from datetime import datetime
import json

df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\raw\scada_pipeline.csv')

# Task A: Timestamp parsing and validation
print('=== TASK A: TIMESTAMP PARSING AND VALIDATION ===')
print('Original timestamp column dtype:', df['timestamp'].dtype)
print('Sample values:', df['timestamp'].head(10).tolist())

# Parse with explicit format
df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M', errors='coerce')

parsed_count = df['timestamp_parsed'].notna().sum()
failed_count = df['timestamp_parsed'].isna().sum()
null_count = df['timestamp'].isna().sum()
unique_parsed = df['timestamp_parsed'].nunique()
earliest = df['timestamp_parsed'].min()
latest = df['timestamp_parsed'].max()
span = latest - earliest

print(f'Successfully parsed: {parsed_count}')
print(f'Failed parses: {failed_count}')
print(f'Null/malformed: {null_count}')
print(f'Unique parsed timestamps: {unique_parsed}')
print(f'Earliest: {earliest}')
print(f'Latest: {latest}')
print(f'Total time span: {span}')

# Check for seconds, timezone info
print('\nOriginal timestamp format check:')
print('Contains seconds?', df['timestamp'].str.contains(r':\d{2}$').any())
print('Contains timezone?', df['timestamp'].str.contains(r'[+-]\d{2}:?\d{2}|Z$|UTC|GMT').any())
print('Sample raw:', df['timestamp'].iloc[0])

# Task B: Global temporal coverage
print('\n=== TASK B: GLOBAL TEMPORAL COVERAGE ===')
print(f'Total rows: {len(df)}')
print(f'Unique timestamps: {df["timestamp"].nunique()}')

# Timestamp frequency
ts_counts = df['timestamp'].value_counts().sort_index()
print('\nRows per timestamp:')
for ts, cnt in ts_counts.items():
    print(f'  {ts}: {cnt} rows')

print(f'\nTimestamp frequency stats:')
print(f'  Min rows per timestamp: {ts_counts.min()}')
print(f'  Max rows per timestamp: {ts_counts.max()}')
print(f'  Mean rows per timestamp: {ts_counts.mean():.1f}')
print(f'  Median rows per timestamp: {ts_counts.median():.1f}')

# Chronological ordering in original file
original_order = df['timestamp'].tolist()
is_sorted = all(original_order[i] <= original_order[i+1] for i in range(len(original_order)-1))
print(f'\nCSV globally sorted by timestamp: {is_sorted}')

# Timestamp intervals
unique_ts_sorted = sorted(df['timestamp_parsed'].unique())
intervals = [(unique_ts_sorted[i+1] - unique_ts_sorted[i]).total_seconds() / 60 for i in range(len(unique_ts_sorted)-1)]
print(f'\nConsecutive unique timestamp intervals (minutes):')
print(f'  Intervals: {intervals}')
print(f'  Min: {min(intervals)} min')
print(f'  Max: {max(intervals)} min')
print(f'  Mean: {np.mean(intervals):.2f} min')
print(f'  Median: {np.median(intervals):.2f} min')

# Task C: Per-segment temporal coverage
print('\n=== TASK C: PER-SEGMENT TEMPORAL COVERAGE ===')
segment_stats = []
for seg in sorted(df['segment_id'].unique()):
    seg_df = df[df['segment_id'] == seg].copy()
    seg_df = seg_df.sort_values('timestamp_parsed')
    
    n_rows = len(seg_df)
    n_unique_ts = seg_df['timestamp_parsed'].nunique()
    earliest_ts = seg_df['timestamp_parsed'].min()
    latest_ts = seg_df['timestamp_parsed'].max()
    span_hours = (latest_ts - earliest_ts).total_seconds() / 3600
    
    # Check if ordered in original CSV
    orig_order = df[df['segment_id'] == seg]['timestamp'].tolist()
    is_ordered = all(orig_order[i] <= orig_order[i+1] for i in range(len(orig_order)-1))
    
    # Repeated timestamps within segment
    dup_ts = seg_df.groupby('timestamp_parsed').size()
    n_repeated_ts = (dup_ts > 1).sum()
    max_repeats = dup_ts.max()
    
    # Intervals between unique timestamps
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
        'n_repeated_ts': int(n_repeated_ts),
        'max_repeats': int(max_repeats),
        'min_interval_min': round(float(min_int), 6) if min_int else None,
        'max_interval_min': round(float(max_int), 6) if max_int else None,
        'mean_interval_min': round(float(mean_int), 6) if mean_int else None,
        'median_interval_min': round(float(median_int), 6) if median_int else None
    })

seg_df = pd.DataFrame(segment_stats)
print(seg_df[['segment_id', 'n_rows', 'n_unique_ts', 'span_hours', 'ordered_in_csv', 'n_repeated_ts', 'max_repeats']].to_string(index=False))

print(f'\nSegments with lowest unique timestamps: {seg_df.nsmallest(5, "n_unique_ts")["segment_id"].tolist()}')
print(f'Segments with highest unique timestamps: {seg_df.nlargest(5, "n_unique_ts")["segment_id"].tolist()}')
print(f'Segments with most repeated timestamps: {seg_df.nlargest(5, "n_repeated_ts")["segment_id"].tolist()}')

# Save segment summary for later use
seg_df.to_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\documentation\PIPER_phase1_3_segment_summary.csv', index=False)
print('\nSegment summary saved to PIPER_phase1_3_segment_summary.csv')