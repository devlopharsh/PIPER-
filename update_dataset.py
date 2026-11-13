import pandas as pd
import numpy as np

# Read the new dataset
df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour.csv')

print("Original dataset shape:", df.shape)
print("\nOriginal event_type distribution:")
print(df['event_type'].value_counts())
print("\nOriginal target distribution:")
print(df['target'].value_counts())
print("\nOriginal alarm_triggered distribution:")
print(df['alarm_triggered'].value_counts())

# ============ TARGET COLUMN ============
# target = 0 for "normal", 1 for everything else
df['target'] = df['event_type'].apply(lambda x: 0 if x == 'normal' else 1)

# ============ ALARM_TRIGGERED COLUMN ============
# Start with same pattern: 0 for "normal", 1 for everything else
df['alarm_triggered'] = df['event_type'].apply(lambda x: 0 if x == 'normal' else 1)

# EXCEPTION for degradation: 66.67% should have opposite values
# For degradation: target=1, so opposite means alarm_triggered=0
# 66.67% of degradation rows should have alarm_triggered=0
# 33.33% should keep alarm_triggered=1

degradation_mask = df['event_type'] == 'degradation'
degradation_indices = df[degradation_mask].index.tolist()
n_degradation = len(degradation_indices)

# Calculate 66.67% for opposite values
n_opposite = int(n_degradation * 0.6667)
n_same = n_degradation - n_opposite

print(f"\nDegradation rows: {n_degradation}")
print(f"Rows with opposite values (alarm=0): {n_opposite}")
print(f"Rows with same values (alarm=1): {n_same}")

# Randomly select indices for opposite values (alarm=0)
np.random.seed(42)  # For reproducibility
opposite_indices = np.random.choice(degradation_indices, size=n_opposite, replace=False)

# Set alarm_triggered=0 for selected degradation rows (opposite of target=1)
df.loc[opposite_indices, 'alarm_triggered'] = 0

# The remaining degradation rows keep alarm_triggered=1 (same as target)

print("\n=== AFTER CHANGES ===")
print("\nNew target distribution:")
print(df['target'].value_counts())
print("\nNew alarm_triggered distribution:")
print(df['alarm_triggered'].value_counts())

# Verify degradation exception
deg_target = df[degradation_mask]['target'].values
deg_alarm = df[degradation_mask]['alarm_triggered'].values
opposite_count = np.sum(deg_target != deg_alarm)
same_count = np.sum(deg_target == deg_alarm)
print(f"\nDegradation verification:")
print(f"  Target=1, Alarm=0 (opposite): {opposite_count} ({opposite_count/n_degradation*100:.2f}%)")
print(f"  Target=1, Alarm=1 (same): {same_count} ({same_count/n_degradation*100:.2f}%)")

# Save the updated dataset
output_path = r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Updated dataset saved to: {output_path}")

# Also print cross-tabulation for verification
print("\n=== CROSS-TABULATION: event_type vs target ===")
print(pd.crosstab(df['event_type'], df['target']))

print("\n=== CROSS-TABULATION: event_type vs alarm_triggered ===")
print(pd.crosstab(df['event_type'], df['alarm_triggered']))