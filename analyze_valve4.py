import pandas as pd
import numpy as np

# Load both datasets
old_df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline.csv')
old_df['timestamp_parsed'] = pd.to_datetime(old_df['timestamp'], format='%m/%d/%Y %H:%M:%S')
old_df = old_df.sort_values(['segment_id', 'timestamp_parsed']).reset_index(drop=True)

new_df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_updated.csv')
new_df['timestamp_parsed'] = pd.to_datetime(new_df['timestamp'], format='mixed', dayfirst=False)
new_df = new_df.sort_values(['segment_id', 'timestamp_parsed']).reset_index(drop=True)

# Check unique (segment_id, timestamp) pairs
old_pairs = set(zip(old_df['segment_id'], old_df['timestamp_parsed']))
new_pairs = set(zip(new_df['segment_id'], new_df['timestamp_parsed']))

print('Old dataset unique (segment, timestamp) pairs:', len(old_pairs))
print('New dataset unique (segment, timestamp) pairs:', len(new_pairs))
print('Old pairs in new:', len(old_pairs & new_pairs))
print('New pairs not in old:', len(new_pairs - old_pairs))
print('Old pairs not in new:', len(old_pairs - new_pairs))

# For pairs that exist in both, compare valve_status
print()
print('=== VALVE_STATUS COMPARISON FOR COMMON PAIRS ===')
common_pairs = old_pairs & new_pairs
mismatches = 0
for seg, ts in sorted(common_pairs)[:50]:
    old_vs = old_df[(old_df['segment_id'] == seg) & (old_df['timestamp_parsed'] == ts)]['valve_status'].values
    new_vs = new_df[(new_df['segment_id'] == seg) & (new_df['timestamp_parsed'] == ts)]['valve_status'].values
    # Old dataset has unique pairs, so old_vs should be single value
    if len(old_vs) == 1:
        old_val = old_vs[0]
        # New dataset may have multiple values
        new_vals = set(new_vs)
        if old_val not in new_vals:
            mismatches += 1
            print(f'  Segment {seg}, {ts}: old={old_val}, new={new_vals}')

print(f'Total mismatches in first 50: {mismatches}')

# Now let's look at the new dataset groups and determine correction strategy
print()
print('=== NEW DATASET GROUP ANALYSIS ===')
# For each (segment, timestamp) group in new dataset, check valve_status consistency
group_vs = new_df.groupby(['segment_id', 'timestamp_parsed'])['valve_status'].apply(lambda x: list(x.unique())).reset_index()
group_vs['n_unique'] = group_vs['valve_status'].apply(len)
group_vs['n_rows'] = new_df.groupby(['segment_id', 'timestamp_parsed']).size().values

print('Groups with multiple unique valve_status:', (group_vs['n_unique'] > 1).sum())
print('Total multi-row groups:', (group_vs['n_rows'] > 1).sum())

# For groups with multiple valve_status, what's the distribution?
multi_vs = group_vs[group_vs['n_unique'] > 1]
print()
print('Multi-value groups valve_status combinations:')
for _, row in multi_vs.iterrows():
    print(f'  Segment {row["segment_id"]}, {row["timestamp_parsed"]}: {row["valve_status"]} (n_rows={row["n_rows"]})')

# Strategy: For each group, use majority vote or most physically consistent
# Let's check what the old dataset says for these timestamps
print()
print('=== CORRECTION STRATEGY: Use old dataset as reference ===')
# Build a reference from old dataset
old_ref = old_df[['segment_id', 'timestamp_parsed', 'valve_status']].drop_duplicates()
old_ref_dict = dict(zip(zip(old_ref['segment_id'], old_ref['timestamp_parsed']), old_ref['valve_status']))

# For each group in new dataset, determine corrected valve_status
corrections = 0
for _, row in group_vs.iterrows():
    seg = row['segment_id']
    ts = row['timestamp_parsed']
    key = (seg, ts)
    
    if key in old_ref_dict:
        ref_vs = old_ref_dict[key]
        current_vs = row['valve_status']
        if ref_vs not in current_vs:
            # Reference value not present in current group - needs correction
            corrections += 1
            print(f'  Segment {seg}, {ts}: reference={ref_vs}, current={current_vs} -> CORRECT TO {ref_vs}')
        elif len(current_vs) > 1:
            # Multiple values, but reference is one of them - standardize to reference
            corrections += 1
            print(f'  Segment {seg}, {ts}: reference={ref_vs}, current={current_vs} -> STANDARDIZE TO {ref_vs}')

print(f'Total groups needing correction: {corrections}')

# Now let's build the corrected dataset
print()
print('=== BUILDING CORRECTED DATASET ===')
new_df_corrected = new_df.copy()

# For each row, determine the correct valve_status based on its (segment_id, timestamp) group
def get_corrected_vs(row):
    key = (row['segment_id'], row['timestamp_parsed'])
    if key in old_ref_dict:
        return old_ref_dict[key]
    else:
        # For timestamps not in old dataset, use majority vote within group
        group_key = (row['segment_id'], row['timestamp_parsed'])
        group_vals = new_df[(new_df['segment_id'] == row['segment_id']) & 
                           (new_df['timestamp_parsed'] == row['timestamp_parsed'])]['valve_status'].values
        # Return most common
        unique, counts = np.unique(group_vals, return_counts=True)
        return unique[np.argmax(counts)]

new_df_corrected['valve_status_corrected'] = new_df_corrected.apply(get_corrected_vs, axis=1)

print('Original valve_status distribution:')
print(new_df['valve_status'].value_counts().sort_index())
print()
print('Corrected valve_status distribution:')
print(new_df_corrected['valve_status_corrected'].value_counts().sort_index())

# Check how many changed
changed = (new_df['valve_status'] != new_df_corrected['valve_status_corrected']).sum()
print(f'\nRows changed: {changed} ({changed/len(new_df)*100:.2f}%)')
print(f'Rows preserved: {len(new_df) - changed}')

# Verify integrity - all other columns should be identical
print('\n=== INTEGRITY CHECK ===')
for col in new_df.columns:
    if col != 'valve_status' and col != 'timestamp_parsed':
        if not new_df[col].equals(new_df_corrected[col]):
            print(f'MISMATCH in {col}')
        else:
            print(f'OK: {col} identical')

# Save corrected dataset
output_path = r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_valve_corrected.csv'
# Drop the helper column
new_df_corrected.drop(columns=['timestamp_parsed', 'valve_status_corrected'], inplace=True)
# Replace valve_status with corrected
new_df_corrected['valve_status'] = new_df_corrected.apply(get_corrected_vs, axis=1)
new_df_corrected.to_csv(output_path, index=False)
print(f'\nSaved corrected dataset to: {output_path}')