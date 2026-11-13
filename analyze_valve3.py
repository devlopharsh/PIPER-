import pandas as pd
import numpy as np

# New dataset - detailed zero time diff analysis
new_df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_updated.csv')
new_df['timestamp_parsed'] = pd.to_datetime(new_df['timestamp'], format='mixed', dayfirst=False)
new_df = new_df.sort_values(['segment_id', 'timestamp_parsed']).reset_index(drop=True)

new_df['valve_status_prev'] = new_df.groupby('segment_id')['valve_status'].shift()
new_df['valve_changed'] = new_df['valve_status'] != new_df['valve_status_prev']
new_df['time_diff'] = new_df.groupby('segment_id')['timestamp_parsed'].diff().dt.total_seconds()

transitions = new_df[new_df['valve_changed'] == True].copy()
transitions['transition_type'] = transitions['valve_status_prev'].astype(str) + '->' + transitions['valve_status'].astype(str)
zero_time = transitions[transitions['time_diff'] == 0]
print('Transitions with 0 time diff:', len(zero_time))
print('Transition types with 0 time diff:')
print(zero_time['transition_type'].value_counts())

# Show details of zero time diff transitions
print()
print('Details of zero time diff transitions (first 20):')
for idx, row in zero_time.head(20).iterrows():
    seg = row['segment_id']
    ts = row['timestamp_parsed']
    print(f'  Row {idx}: segment {seg}, time {ts}, transition {row["valve_status_prev"]}->{row["valve_status"]}')

# Check how many rows per (segment_id, timestamp) group
print()
print('=== ROWS PER (SEGMENT, TIMESTAMP) GROUP ===')
group_counts = new_df.groupby(['segment_id', 'timestamp_parsed']).size()
print('Group size distribution:')
print(group_counts.value_counts().sort_index())
print()
print('Groups with >1 row:', (group_counts > 1).sum())
print('Total rows in multi-row groups:', group_counts[group_counts > 1].sum())

# Check if valve_status differs within same (segment, timestamp) groups
multi_groups = new_df.groupby(['segment_id', 'timestamp_parsed']).filter(lambda x: len(x) > 1)
print()
print('Multi-row groups analysis:')
for (seg, ts), group in multi_groups.groupby(['segment_id', 'timestamp_parsed']):
    valve_vals = group['valve_status'].tolist()
    if len(set(valve_vals)) > 1:
        print(f'  Segment {seg}, time {ts}: valve_status values = {valve_vals}')
        # Show all columns for this group
        for col in group.columns:
            if col not in ['timestamp_parsed', 'valve_status_prev', 'valve_changed', 'time_diff', 'transition_type']:
                vals = group[col].tolist()
                if len(set(vals)) > 1:
                    print(f'    {col}: {vals}')