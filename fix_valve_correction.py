import pandas as pd
import numpy as np

# Load the updated dataset (with pump_state/pump_speed fixes)
updated_path = r"C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_updated.csv"
new_df = pd.read_csv(updated_path)

print('Loaded updated dataset:', new_df.shape)

# Preserve original row order
new_df['original_index'] = range(len(new_df))

# Parse timestamp for analysis
new_df['timestamp_parsed'] = pd.to_datetime(new_df['timestamp'], format='mixed', dayfirst=False)

# For analysis, we need to sort by segment_id and timestamp
# But we'll restore original order afterwards
new_df_sorted = new_df.sort_values(['segment_id', 'timestamp_parsed']).reset_index(drop=True)

# Build group-level corrections on sorted data
# For each (segment_id, timestamp) group, determine corrected valve_status
corrected_values = {}
for (seg, ts), group in new_df_sorted.groupby(['segment_id', 'timestamp_parsed']):
    vs_counts = group['valve_status'].value_counts()
    if len(vs_counts) == 1:
        # Already consistent
        corrected_values[(seg, ts)] = vs_counts.index[0]
    else:
        # Multiple values - need to decide
        max_count = vs_counts.max()
        total = len(group)
        if max_count > total / 2:
            # Clear majority
            corrected_values[(seg, ts)] = vs_counts.idxmax()
        else:
            # No clear majority - use physical consistency
            pump_state_mode = group['pump_state'].mode().iloc[0]
            compressor_state_mode = group['compressor_state'].mode().iloc[0]
            pump_speed_mean = group['pump_speed'].mean()
            
            candidate_vs = list(vs_counts.index)
            scores = {}
            for cand in candidate_vs:
                score = 0
                if pump_state_mode == 0 and cand in [0, 1]:
                    score += 1
                elif pump_state_mode == 1 and cand in [1, 2]:
                    score += 1
                if compressor_state_mode == 0 and cand in [1, 2]:
                    score += 0.5
                elif compressor_state_mode == 1 and cand in [0, 1]:
                    score += 0.5
                if pump_speed_mean == 0 and cand in [0, 1]:
                    score += 1
                elif pump_speed_mean > 0 and cand in [1, 2]:
                    score += 1
                scores[cand] = score
            
            best_vs = max(scores.keys(), key=lambda k: (scores[k], vs_counts[k]))
            corrected_values[(seg, ts)] = best_vs

# Now apply corrections to the ORIGINAL (unsorted) dataframe
def get_corrected_vs(row):
    return corrected_values[(row['segment_id'], row['timestamp_parsed'])]

# Apply to original order dataframe
new_df['valve_status_corrected'] = new_df.apply(get_corrected_vs, axis=1)

# Verify
print('Original valve_status distribution:')
print(new_df['valve_status'].value_counts().sort_index())
print()
print('Corrected valve_status distribution:')
print(new_df['valve_status_corrected'].value_counts().sort_index())

changed = (new_df['valve_status'] != new_df['valve_status_corrected']).sum()
print(f'\nRows changed: {changed} ({changed/len(new_df)*100:.2f}%)')

# Replace valve_status with corrected
new_df['valve_status'] = new_df['valve_status_corrected']
new_df = new_df.drop(columns=['timestamp_parsed', 'valve_status_corrected', 'original_index'])

# Verify row order is preserved by checking first few rows
print('\nFirst 5 rows of corrected dataset:')
print(new_df.head())

# Save corrected dataset
output_path = r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_valve_corrected.csv'
new_df.to_csv(output_path, index=False)
print(f'\nSaved corrected dataset to: {output_path}')