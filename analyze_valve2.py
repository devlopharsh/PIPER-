import pandas as pd
import numpy as np

# Load old dataset
old_df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline.csv')
old_df['timestamp_parsed'] = pd.to_datetime(old_df['timestamp'], format='%m/%d/%Y %H:%M:%S')
old_df = old_df.sort_values(['segment_id', 'timestamp_parsed']).reset_index(drop=True)

# Add valve_status shift column for analysis
old_df['valve_status_prev'] = old_df.groupby('segment_id')['valve_status'].shift()
old_df['valve_changed'] = old_df['valve_status'] != old_df['valve_status_prev']
old_df['time_diff'] = old_df.groupby('segment_id')['timestamp_parsed'].diff().dt.total_seconds()

# Analyze all transitions
transitions = old_df[old_df['valve_changed'] == True].copy()
print('OLD DATASET')
print('Total transitions:', len(transitions))
print('Time diff stats for transitions:')
print(transitions['time_diff'].describe())

# Transition types
transitions['transition_type'] = transitions['valve_status_prev'].astype(str) + '->' + transitions['valve_status'].astype(str)
print()
print('Transition type counts:')
print(transitions['transition_type'].value_counts())

# Check telemetry changes at transitions
print()
print('=== TELEMETRY CHANGES AT TRANSITIONS (Old) ===')
for col in ['pressure', 'flow_rate', 'temperature', 'pump_speed', 'energy_consumption', 'pump_state', 'compressor_state']:
    if col in transitions.columns:
        transitions[f'{col}_change'] = transitions[col] - transitions.groupby('segment_id')[col].shift()
        print(f'{col} change at transitions: mean={transitions[f"{col}_change"].mean():.4f}, std={transitions[f"{col}_change"].std():.4f}')

# Isolated transitions
print()
print('=== ISOLATED TRANSITIONS ANALYSIS (Old) ===')
old_df['valve_status_next'] = old_df.groupby('segment_id')['valve_status'].shift(-1)
old_df['isolated'] = (old_df['valve_changed'] == True) & (old_df['valve_status_next'] == old_df['valve_status_prev'])
isolated = old_df[old_df['isolated'] == True]
print('Isolated transitions:', len(isolated))
if len(isolated) > 0:
    isolated['iso_type'] = isolated['valve_status_prev'].astype(str) + '->' + isolated['valve_status'].astype(str) + '->' + isolated['valve_status_next'].astype(str)
    print(isolated['iso_type'].value_counts())

# Check for zero time diff transitions (same timestamp)
print()
print('=== ZERO TIME DIFF TRANSITIONS ===')
zero_time = transitions[transitions['time_diff'] == 0]
print('Transitions with 0 time diff:', len(zero_time))
if len(zero_time) > 0:
    print('Transition types with 0 time diff:')
    print(zero_time['transition_type'].value_counts())

# Now new dataset - detailed zero time diff analysis
print()
print('=== NEW DATASET - ZERO TIME DIFF TRANSITIONS ===')
new_df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_updated.csv')
new_df['timestamp_parsed'] = pd.to_datetime(new_df['timestamp'], format='mixed', dayfirst=False)
new_df = new_df.sort_values(['segment_id', 'timestamp_parsed']).reset_index(drop=True)

new_df['valve_status_prev'] = new_df.groupby('segment_id')['valve_status'].shift()
new_df['valve_changed'] = new_df['valve_status'] != new_df['valve_status_prev']
new_df['time_diff'] = new_df.groupby('segment_id')['timestamp_parsed'].diff().dt.total_seconds()

transitions = new_df[new_df['valve_changed'] == True].copy()
zero_time = transitions[transitions['time_diff'] == 0]
print('Transitions with 0 time diff:', len(zero_time))
if len(zero_time) > 0:
    print('Transition types with 0 time diff:')
    print(zero_time['transition_type'].value_counts())
    print()
    # Show details of zero time diff transitions
    for idx, row in zero_time.head(10).iterrows():
        seg = row['segment_id']
        ts = row['timestamp_parsed']
        print(f'  Row {idx}: segment {seg}, time {ts}, transition {row["valve_status_prev"]}->{row["valve_status"]}')