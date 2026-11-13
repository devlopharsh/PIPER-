import pandas as pd

df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_segment_even_timestamps.csv')
print(f'Rows: {len(df)}, Cols: {len(df.columns)}')
print(f'Columns: {list(df.columns)}')
print(f'Unique segments: {df["segment_id"].nunique()}')
print(f'Target distribution:\n{df["target"].value_counts()}')
print(f'Event type distribution:\n{df["event_type"].value_counts()}')
print(f'Alarm triggered distribution:\n{df["alarm_triggered"].value_counts()}')
print(f'Valve status distribution:\n{df["valve_status"].value_counts()}')
print(f'Missing values:\n{df.isnull().sum()}')
print(f'Duplicates: {df.duplicated().sum()}')
print(f'\nFirst few timestamps:\n{df["timestamp"].head()}')
print(f'\nLast few timestamps:\n{df["timestamp"].tail()}')

# Check segment lengths
segment_lengths = df.groupby('segment_id').size()
print(f'\nSegment lengths:\n{segment_lengths}')
print(f'Min segment length: {segment_lengths.min()}')
print(f'Max segment length: {segment_lengths.max()}')
print(f'Mean segment length: {segment_lengths.mean():.1f}')

# Check timestamp format and ordering per segment
for seg_id in sorted(df['segment_id'].unique()):
    seg_df = df[df['segment_id'] == seg_id].copy()
    seg_df['timestamp'] = pd.to_datetime(seg_df['timestamp'], format='%d-%m-%Y %H:%M:%S')
    is_sorted = seg_df['timestamp'].is_monotonic_increasing
    print(f'Segment {seg_id}: sorted={is_sorted}, time range: {seg_df["timestamp"].min()} to {seg_df["timestamp"].max()}')