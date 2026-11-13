import pandas as pd
import numpy as np

# Load new dataset
new_df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_updated.csv')
new_df['timestamp_parsed'] = pd.to_datetime(new_df['timestamp'], format='mixed', dayfirst=False)
new_df = new_df.sort_values(['segment_id', 'timestamp_parsed']).reset_index(drop=True)

print('Original shape:', new_df.shape)
print('Original valve_status distribution:')
print(new_df['valve_status'].value_counts().sort_index())

# Strategy: For each (segment_id, timestamp) group, determine the most physically consistent valve_status
# Based on telemetry patterns and majority vote

# First, let's analyze the physical relationships
# Check how valve_status correlates with pump_state, compressor_state, pump_speed

print()
print('=== PHYSICAL CONSISTENCY ANALYSIS ===')
for vs in [0, 1, 2]:
    subset = new_df[new_df['valve_status'] == vs]
    print(f'valve_status={vs}:')
    print(f'  pump_state=0: {(subset["pump_state"]==0).mean()*100:.1f}%, pump_state=1: {(subset["pump_state"]==1).mean()*100:.1f}%')
    print(f'  compressor_state=0: {(subset["compressor_state"]==0).mean()*100:.1f}%, compressor_state=1: {(subset["compressor_state"]==1).mean()*100:.1f}%')
    print(f'  pump_speed=0: {(subset["pump_speed"]==0).mean()*100:.1f}%, pump_speed>0: {(subset["pump_speed"]>0).mean()*100:.1f}%')
    print(f'  pressure mean: {subset["pressure"].mean():.1f}')
    print(f'  flow_rate mean: {subset["flow_rate"].mean():.2f}')

# The old dataset showed: when pump_speed=0 -> pump_state=0, compressor_state=1
# when pump_speed>0 -> pump_state=1, compressor_state=0 or 1

# Now let's determine the correct valve_status for each group
# We'll use a combination of:
# 1. Majority vote within group
# 2. Physical consistency with pump_state/compressor_state/pump_speed

# Build group-level statistics
group_stats = new_df.groupby(['segment_id', 'timestamp_parsed']).agg(
    n_rows=('valve_status', 'count'),
    valve_status_mode=('valve_status', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
    valve_status_unique=('valve_status', lambda x: list(x.unique())),
    pump_state_mode=('pump_state', lambda x: x.mode().iloc[0]),
    compressor_state_mode=('compressor_state', lambda x: x.mode().iloc[0]),
    pump_speed_mean=('pump_speed', 'mean'),
    pump_speed_max=('pump_speed', 'max'),
    pressure_mean=('pressure', 'mean'),
    flow_rate_mean=('flow_rate', 'mean'),
    event_type_mode=('event_type', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
    target_mode=('target', lambda x: x.mode().iloc[0])
).reset_index()

print()
print('Group stats shape:', group_stats.shape)
print('Groups with multiple valve_status:', (group_stats['valve_status_unique'].apply(len) > 1).sum())

# For groups with consistent valve_status, keep it
# For groups with inconsistent valve_status, determine the best one
# Based on physical rules:
# - If pump_speed == 0 (pump off): valve_status likely 0 (closed) or 1 (transitioning)
# - If pump_speed > 0 (pump on): valve_status likely 1 (open) or 2 (throttled)
# But we need to understand what 0, 1, 2 mean physically

# From old dataset analysis:
# valve_status 0: 19.6%, pump_speed mean=976, pump_state=0 prob=?
# valve_status 1: 69.2%, pump_speed mean=932
# valve_status 2: 11.2%, pump_speed mean=904

# The means are similar, so valve_status doesn't strongly correlate with pump_speed
# Let's check the old dataset's physical pattern more carefully

# Load old dataset for reference
old_df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline.csv')
old_df['timestamp_parsed'] = pd.to_datetime(old_df['timestamp'], format='%m/%d/%Y %H:%M:%S')

print()
print('=== OLD DATASET: valve_status vs pump_state/compressor_state ===')
for vs in [0, 1, 2]:
    subset = old_df[old_df['valve_status'] == vs]
    print(f'valve_status={vs}:')
    print(f'  pump_state=0: {(subset["pump_state"]==0).mean()*100:.1f}%, pump_state=1: {(subset["pump_state"]==1).mean()*100:.1f}%')
    print(f'  compressor_state=0: {(subset["compressor_state"]==0).mean()*100:.1f}%, compressor_state=1: {(subset["compressor_state"]==1).mean()*100:.1f}%')
    print(f'  pump_speed=0: {(subset["pump_speed"]==0).mean()*100:.1f}%, pump_speed>0: {(subset["pump_speed"]>0).mean()*100:.1f}%')

# Now let's use a more principled approach:
# For each group, we have multiple rows. The "true" valve_status should be consistent with the equipment states.
# Since the group has multiple rows with potentially different equipment states (pump_state, compressor_state),
# we need to decide which valve_status is most representative.

# Approach: For each group, use the majority valve_status, but if there's a tie or near-tie,
# use physical consistency rules.

# Let's check the groups with multiple valve_status and see the distribution of pump_state etc.
inconsistent_groups = group_stats[group_stats['valve_status_unique'].apply(len) > 1]
print()
print('=== INCONSISTENT GROUPS ANALYSIS ===')
print('Number of inconsistent groups:', len(inconsistent_groups))

# For each inconsistent group, let's see the row-level detail
def analyze_group(row):
    seg = row['segment_id']
    ts = row['timestamp_parsed']
    group_rows = new_df[(new_df['segment_id'] == seg) & (new_df['timestamp_parsed'] == ts)]
    print(f'\nSegment {seg}, {ts}:')
    print(f'  valve_status: {group_rows["valve_status"].tolist()}')
    print(f'  pump_state: {group_rows["pump_state"].tolist()}')
    print(f'  compressor_state: {group_rows["compressor_state"].tolist()}')
    print(f'  pump_speed: {group_rows["pump_speed"].tolist()}')
    print(f'  pressure: {group_rows["pressure"].tolist()}')
    print(f'  flow_rate: {group_rows["flow_rate"].tolist()}')
    print(f'  event_type: {group_rows["event_type"].tolist()}')
    print(f'  target: {group_rows["target"].tolist()}')

# Analyze first 10 inconsistent groups
for _, row in inconsistent_groups.head(10).iterrows():
    analyze_group(row)

# Now let's develop a correction rule:
# The valve_status should be consistent within a (segment_id, timestamp) group.
# We'll assign a single valve_status per group based on:
# 1. If there's a clear majority (>50%), use that
# 2. If tied, use the valve_status that's most physically consistent with the dominant equipment state
#    - pump_state=0, pump_speed=0 -> likely valve_status=0 (closed) or 1
#    - pump_state=1, pump_speed>0 -> likely valve_status=1 (open) or 2
# 3. If still ambiguous, use the mode

# Let's implement this correction
new_df_corrected = new_df.copy()

# For each group, determine corrected valve_status
corrected_values = {}
for (seg, ts), group in new_df.groupby(['segment_id', 'timestamp_parsed']):
    vs_counts = group['valve_status'].value_counts()
    if len(vs_counts) == 1:
        # Already consistent
        corrected_values[(seg, ts)] = vs_counts.index[0]
    else:
        # Multiple values - need to decide
        # Check if there's a clear majority
        max_count = vs_counts.max()
        total = len(group)
        if max_count > total / 2:
            # Clear majority
            corrected_values[(seg, ts)] = vs_counts.idxmax()
        else:
            # No clear majority - use physical consistency
            # Get dominant equipment states
            pump_state_mode = group['pump_state'].mode().iloc[0]
            compressor_state_mode = group['compressor_state'].mode().iloc[0]
            pump_speed_mean = group['pump_speed'].mean()
            
            # Physical rules based on old dataset patterns
            # From old data: valve_status 0 associated with pump_state=0 (39%), valve_status 1 with pump_state=1 (59%)
            # But let's use a simpler rule: if pump off, prefer 0 or 1; if pump on, prefer 1 or 2
            
            # Check which valve_status in the group is most consistent
            candidate_vs = list(vs_counts.index)
            
            # Score each candidate
            scores = {}
            for cand in candidate_vs:
                score = 0
                # Consistency with pump_state
                if pump_state_mode == 0 and cand in [0, 1]:
                    score += 1
                elif pump_state_mode == 1 and cand in [1, 2]:
                    score += 1
                # Consistency with compressor_state  
                if compressor_state_mode == 0 and cand in [1, 2]:
                    score += 0.5
                elif compressor_state_mode == 1 and cand in [0, 1]:
                    score += 0.5
                # Consistency with pump_speed
                if pump_speed_mean == 0 and cand in [0, 1]:
                    score += 1
                elif pump_speed_mean > 0 and cand in [1, 2]:
                    score += 1
                scores[cand] = score
            
            # Pick highest scoring, break ties by count
            best_vs = max(scores.keys(), key=lambda k: (scores[k], vs_counts[k]))
            corrected_values[(seg, ts)] = best_vs

# Apply corrections
def get_corrected_vs(row):
    return corrected_values[(row['segment_id'], row['timestamp_parsed'])]

new_df_corrected['valve_status'] = new_df_corrected.apply(get_corrected_vs, axis=1)

print()
print('=== CORRECTION RESULTS ===')
print('Original valve_status distribution:')
print(new_df['valve_status'].value_counts().sort_index())
print()
print('Corrected valve_status distribution:')
print(new_df_corrected['valve_status'].value_counts().sort_index())

changed = (new_df['valve_status'] != new_df_corrected['valve_status']).sum()
print(f'\nRows changed: {changed} ({changed/len(new_df)*100:.2f}%)')
print(f'Rows preserved: {len(new_df) - changed}')

# Verify group consistency
new_df_corrected['valve_status_check'] = new_df_corrected.groupby(['segment_id', 'timestamp_parsed'])['valve_status'].transform('nunique')
inconsistent_after = (new_df_corrected['valve_status_check'] > 1).sum()
print(f'Groups with inconsistent valve_status after correction: {inconsistent_after}')

# Integrity check
print('\n=== INTEGRITY CHECK ===')
for col in new_df.columns:
    if col not in ['valve_status', 'timestamp_parsed']:
        if new_df[col].equals(new_df_corrected[col]):
            print(f'OK: {col} identical')
        else:
            print(f'MISMATCH in {col}')

# Save corrected dataset
output_path = r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_valve_corrected.csv'
save_df = new_df_corrected.drop(columns=['timestamp_parsed', 'valve_status_check'])
save_df.to_csv(output_path, index=False)
print(f'\nSaved corrected dataset to: {output_path}')