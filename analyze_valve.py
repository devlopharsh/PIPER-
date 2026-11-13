import pandas as pd
import numpy as np

# Load new dataset
new_df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_updated.csv')
new_df['timestamp_parsed'] = pd.to_datetime(new_df['timestamp'], format='mixed', dayfirst=False)
new_df = new_df.sort_values(['segment_id', 'timestamp_parsed']).reset_index(drop=True)

# Add valve_status shift column for analysis
new_df['valve_status_prev'] = new_df.groupby('segment_id')['valve_status'].shift()
new_df['valve_changed'] = new_df['valve_status'] != new_df['valve_status_prev']
new_df['time_diff'] = new_df.groupby('segment_id')['timestamp_parsed'].diff().dt.total_seconds()

# Analyze all transitions
transitions = new_df[new_df['valve_changed'] == True].copy()
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
print('=== TELEMETRY CHANGES AT TRANSITIONS ===')
for col in ['pressure', 'flow_rate', 'temperature', 'pump_speed', 'energy_consumption', 'pump_state', 'compressor_state']:
    if col in transitions.columns:
        transitions[f'{col}_change'] = transitions[col] - transitions.groupby('segment_id')[col].shift()
        print(f'{col} change at transitions: mean={transitions[f"{col}_change"].mean():.4f}, std={transitions[f"{col}_change"].std():.4f}')

# Compare with non-transitions
non_transitions = new_df[new_df['valve_changed'] == False].copy()
for col in ['pressure', 'flow_rate', 'temperature', 'pump_speed', 'energy_consumption']:
    non_transitions[f'{col}_change'] = non_transitions[col] - non_transitions.groupby('segment_id')[col].shift()

print()
print('=== TELEMETRY CHANGES AT NON-TRANSITIONS (for comparison) ===')
for col in ['pressure', 'flow_rate', 'temperature', 'pump_speed', 'energy_consumption']:
    print(f'{col} change at non-transitions: mean={non_transitions[f"{col}_change"].mean():.4f}, std={non_transitions[f"{col}_change"].std():.4f}')

# Check isolated transitions (single point changes surrounded by same state)
print()
print('=== ISOLATED TRANSITIONS ANALYSIS ===')
new_df['valve_status_next'] = new_df.groupby('segment_id')['valve_status'].shift(-1)
new_df['isolated'] = (new_df['valve_changed'] == True) & (new_df['valve_status_next'] == new_df['valve_status_prev'])
isolated = new_df[new_df['isolated'] == True]
print('Isolated transitions:', len(isolated))
if len(isolated) > 0:
    print('Isolated transition types:')
    isolated['iso_type'] = isolated['valve_status_prev'].astype(str) + '->' + isolated['valve_status'].astype(str) + '->' + isolated['valve_status_next'].astype(str)
    print(isolated['iso_type'].value_counts())

# Check for suspiciously rapid switching (multiple transitions in short time)
print()
print('=== RAPID SWITCHING ANALYSIS ===')
# For each segment, check if there are multiple transitions within a short window
for seg in sorted(new_df['segment_id'].unique())[:5]:
    seg_df = new_df[new_df['segment_id'] == seg].reset_index(drop=True)
    seg_df['valve_change'] = seg_df['valve_status'] != seg_df['valve_status'].shift()
    transitions_idx = seg_df[seg_df['valve_change'] == True].index.tolist()
    if len(transitions_idx) > 1:
        time_diffs = []
        for i in range(1, len(transitions_idx)):
            time_diffs.append((seg_df.loc[transitions_idx[i], 'timestamp_parsed'] - seg_df.loc[transitions_idx[i-1], 'timestamp_parsed']).total_seconds())
        if time_diffs:
            print(f'Segment {seg}: {len(transitions_idx)} transitions, min interval between transitions: {min(time_diffs):.0f}s, max: {max(time_diffs):.0f}s')