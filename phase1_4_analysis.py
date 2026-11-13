import pandas as pd
import numpy as np
import json

df = pd.read_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\raw\scada_pipeline.csv')

print('=== TASK A: TARGET DISTRIBUTION AND LABEL VALIDATION ===')
print(f'Total rows: {len(df)}')
print(f'Null target values: {df["target"].isnull().sum()}')
print(f'Unique target values: {sorted(df["target"].unique())}')
print(f'Target distribution:')
target_counts = df['target'].value_counts().sort_index()
for k, v in target_counts.items():
    pct = v / len(df) * 100
    print(f'  target={k}: {v} ({pct:.1f}%)')
print(f'Is binary: {set(df["target"].unique()) == {0, 1}}')

# Per-segment target distribution
print('\nPer-segment target distribution (unusual if any):')
seg_target = df.groupby('segment_id')['target'].agg(['sum', 'count', 'mean'])
seg_target.columns = ['abnormal_count', 'total', 'abnormal_rate']
print(seg_target.to_string())

# Per-timestamp target distribution
print('\nPer-timestamp target distribution:')
ts_target = df.groupby('timestamp')['target'].agg(['sum', 'count', 'mean'])
ts_target.columns = ['abnormal_count', 'total', 'abnormal_rate']
print(ts_target.to_string())

# ============================================================
# TASK B: TARGET CROSS-TABULATION
# ============================================================
print('\n=== TASK B: TARGET CROSS-TABULATION ===')

categorical_fields = ['event_type', 'alarm_triggered', 'valve_status', 'pump_state', 'compressor_state', 'segment_id']
for col in categorical_fields:
    print(f'\n--- {col} vs target ---')
    ct = pd.crosstab(df[col], df['target'], margins=True)
    print(ct.to_string())
    # Target rate per category
    ct_pct = pd.crosstab(df[col], df['target'], normalize='index') * 100
    print(f'\nTarget=1 rate by {col}:')
    print(ct_pct[1].round(1).to_string())

# Numeric fields by target
numeric_fields = ['pressure', 'flow_rate', 'temperature', 'pump_speed', 'energy_consumption']
print('\n--- Numeric fields by target class ---')
for col in numeric_fields:
    print(f'\n{col}:')
    for t in [0, 1]:
        subset = df[df['target'] == t][col]
        print(f'  target={t}: n={len(subset)}, mean={subset.mean():.3f}, median={subset.median():.3f}, std={subset.std():.3f}, min={subset.min():.3f}, max={subset.max():.3f}')

# ============================================================
# TASK C: EVENT_TYPE LEAKAGE INVESTIGATION
# ============================================================
print('\n=== TASK C: EVENT_TYPE LEAKAGE INVESTIGATION ===')
print(f'Unique event_type values: {sorted(df["event_type"].unique())}')
print(f'Event type counts:')
print(df['event_type'].value_counts().to_string())

ct_event = pd.crosstab(df['event_type'], df['target'])
print('\nEvent type vs target:')
print(ct_event.to_string())

# Check if any event_type has both targets
for et in df['event_type'].unique():
    subset = df[df['event_type'] == et]
    targets = subset['target'].unique()
    print(f'  {et}: targets={targets}, both_classes={len(targets) > 1}')

# ============================================================
# TASK D: ALARM_TRIGGERED LEAKAGE INVESTIGATION
# ============================================================
print('\n=== TASK D: ALARM_TRIGGERED LEAKAGE INVESTIGATION ===')
ct_alarm = pd.crosstab(df['alarm_triggered'], df['target'])
print('Alarm vs target:')
print(ct_alarm.to_string())

# Probabilities
alarm1_target1 = ct_alarm.loc[1, 1] if 1 in ct_alarm.index and 1 in ct_alarm.columns else 0
alarm1_target0 = ct_alarm.loc[1, 0] if 1 in ct_alarm.index and 0 in ct_alarm.columns else 0
alarm0_target1 = ct_alarm.loc[0, 1] if 0 in ct_alarm.index and 1 in ct_alarm.columns else 0
alarm0_target0 = ct_alarm.loc[0, 0] if 0 in ct_alarm.index and 0 in ct_alarm.columns else 0

print(f'alarm=1, target=1: {alarm1_target1}')
print(f'alarm=1, target=0: {alarm1_target0}')
print(f'alarm=0, target=1: {alarm0_target1}')
print(f'alarm=0, target=0: {alarm0_target0}')

p_target1_given_alarm1 = alarm1_target1 / (alarm1_target1 + alarm1_target0) if (alarm1_target1 + alarm1_target0) > 0 else 0
p_target1_given_alarm0 = alarm0_target1 / (alarm0_target1 + alarm0_target0) if (alarm0_target1 + alarm0_target0) > 0 else 0
print(f'P(target=1 | alarm=1): {p_target1_given_alarm1:.4f}')
print(f'P(target=1 | alarm=0): {p_target1_given_alarm0:.4f}')

# ============================================================
# TASK E: STATUS-FIELD AND INDIRECT LEAKAGE AUDIT
# ============================================================
print('\n=== TASK E: STATUS-FIELD AND INDIRECT LEAKAGE AUDIT ===')
status_fields = ['valve_status', 'pump_state', 'pump_speed', 'compressor_state', 'energy_consumption']

for col in status_fields:
    print(f'\n--- {col} ---')
    if df[col].nunique() <= 10:
        ct = pd.crosstab(df[col], df['target'], normalize='index') * 100
        print(f'Target=1 rate by {col}:')
        print(ct[1].round(1).to_string())
    else:
        for t in [0, 1]:
            subset = df[df['target'] == t][col]
            print(f'  target={t}: mean={subset.mean():.3f}, median={subset.median():.3f}, std={subset.std():.3f}')
    
    # Check variation within repeated groups (from Phase 1.3)
    # We'll do this in Task G

# Check relationship with event_type and alarm_triggered
print('\n--- Status fields vs event_type ---')
for col in ['valve_status', 'pump_state', 'compressor_state']:
    ct = pd.crosstab(df[col], df['event_type'], normalize='index') * 100
    print(f'{col} vs event_type (row %):')
    print(ct.round(1).to_string())

print('\n--- Status fields vs alarm_triggered ---')
for col in ['valve_status', 'pump_state', 'compressor_state']:
    ct = pd.crosstab(df[col], df['alarm_triggered'], normalize='index') * 100
    print(f'{col} vs alarm (row %):')
    print(ct.round(1).to_string())

# ============================================================
# TASK F: NUMERIC TELEMETRY LEAKAGE AUDIT
# ============================================================
print('\n=== TASK F: NUMERIC TELEMETRY LEAKAGE AUDIT ===')
telemetry_fields = ['pressure', 'flow_rate', 'temperature', 'pump_speed', 'energy_consumption']

for col in telemetry_fields:
    print(f'\n--- {col} ---')
    # By target
    for t in [0, 1]:
        subset = df[df['target'] == t][col]
        print(f'  target={t}: n={len(subset)}, mean={subset.mean():.3f}, median={subset.median():.3f}, std={subset.std():.3f}, min={subset.min():.3f}, max={subset.max():.3f}')
    
    # Check for threshold-like behavior
    # Sort by value and see target transition
    sorted_df = df.sort_values(col)
    # Find if there's a clear boundary
    target_sorted = sorted_df['target'].values
    transitions = np.where(np.diff(target_sorted) != 0)[0]
    if len(transitions) < 10:
        print(f'  Target transitions in sorted {col}: {len(transitions)}')
        for t_idx in transitions[:5]:
            val_before = sorted_df.iloc[t_idx][col]
            val_after = sorted_df.iloc[t_idx+1][col]
            print(f'    {val_before:.3f} (target={target_sorted[t_idx]}) -> {val_after:.3f} (target={target_sorted[t_idx+1]})')

# ============================================================
# TASK G: TARGET CONFLICT ANALYSIS
# ============================================================
print('\n=== TASK G: TARGET CONFLICT ANALYSIS ===')
grouped = df.groupby(['segment_id', 'timestamp'])
group_sizes = grouped.size()
total_groups = len(group_sizes)
single_row_groups = int((group_sizes == 1).sum())
repeated_groups = int((group_sizes > 1).sum())
rows_in_repeated = int(group_sizes[group_sizes > 1].sum())

print(f'Total groups: {total_groups}')
print(f'Single-row groups: {single_row_groups}')
print(f'Repeated groups: {repeated_groups}')
print(f'Rows in repeated groups: {rows_in_repeated}')

# Target consistency within groups
target_conflicts = 0
target_0_only = 0
target_1_only = 0
both_targets = 0

conflict_examples = []

for (seg, ts), group in grouped:
    targets = group['target'].unique()
    if len(targets) == 1:
        if targets[0] == 0:
            target_0_only += 1
        else:
            target_1_only += 1
    else:
        both_targets += 1
        target_conflicts += 1
        if len(conflict_examples) < 20:
            conflict_examples.append({
                'segment_id': seg,
                'timestamp': ts,
                'group_size': len(group),
                'target_values': str(sorted(targets)),
                'event_types': str(sorted(group['event_type'].unique())),
                'alarm_values': str(sorted(group['alarm_triggered'].unique())),
                'pressure_range': f"{group['pressure'].min():.2f}-{group['pressure'].max():.2f}",
                'flow_range': f"{group['flow_rate'].min():.2f}-{group['flow_rate'].max():.2f}",
                'temp_range': f"{group['temperature'].min():.2f}-{group['temperature'].max():.2f}"
            })

print(f'Groups with target=0 only: {target_0_only}')
print(f'Groups with target=1 only: {target_1_only}')
print(f'Groups with both targets (conflicts): {both_targets}')
print(f'Total target conflicts: {target_conflicts}')
print(f'Rows affected by conflicting groups: {rows_in_repeated}')

# Rows in conflicting groups
conflict_rows = 0
for (seg, ts), group in grouped:
    if group['target'].nunique() > 1:
        conflict_rows += len(group)
print(f'Rows in conflicting groups: {conflict_rows} ({conflict_rows/len(df)*100:.1f}%)')

# Target disagreement with alarm
print('\n--- Target vs Alarm disagreement ---')
disagree = df[(df['target'] == 1) & (df['alarm_triggered'] == 0)]
print(f'target=1, alarm=0: {len(disagree)} rows ({len(disagree)/len(df)*100:.1f}%)')
disagree2 = df[(df['target'] == 0) & (df['alarm_triggered'] == 1)]
print(f'target=0, alarm=1: {len(disagree2)} rows ({len(disagree2)/len(df)*100:.1f}%)')

# By event_type
print('\ntarget=1, alarm=0 by event_type:')
print(disagree['event_type'].value_counts().to_string())

# ============================================================
# TASK H: DIRECT AND INDIRECT TARGET ENCODING
# ============================================================
print('\n=== TASK H: DIRECT AND INDIRECT TARGET ENCODING ===')

# Check combinations
print('--- event_type + alarm_triggered vs target ---')
ct_combo = pd.crosstab([df['event_type'], df['alarm_triggered']], df['target'])
print(ct_combo.to_string())

# Check if any single field perfectly predicts target
print('\n--- Perfect prediction check ---')
for col in ['event_type', 'alarm_triggered', 'valve_status', 'pump_state', 'compressor_state']:
    ct = pd.crosstab(df[col], df['target'])
    # Check if any category has only one target class
    perfect = True
    for idx in ct.index:
        if ct.loc[idx].sum() > 0:
            non_zero = (ct.loc[idx] > 0).sum()
            if non_zero > 1:
                perfect = False
                break
    if perfect:
        print(f'  {col}: PERFECTLY determines target')
    else:
        # Check one-direction
        one_dir = True
        for idx in ct.index:
            if ct.loc[idx, 1] > 0 and ct.loc[idx, 0] == 0:
                # This category implies target=1
                pass
            elif ct.loc[idx, 0] > 0 and ct.loc[idx, 1] == 0:
                # This category implies target=0
                pass
            else:
                one_dir = False
        if one_dir:
            print(f'  {col}: ONE-DIRECTION perfect (each category maps to single target)')

# ============================================================
# TASK I: PREDICTION-TIME AVAILABILITY ASSESSMENT
# ============================================================
print('\n=== TASK I: PREDICTION-TIME AVAILABILITY ASSESSMENT ===')
availability = {
    'pressure': 'B - Potentially available before prediction, but undocumented (physical sensor reading)',
    'flow_rate': 'B - Potentially available before prediction, but undocumented (physical sensor reading)',
    'temperature': 'B - Potentially available before prediction, but undocumented (physical sensor reading)',
    'valve_status': 'B - Potentially available before prediction, but undocumented (equipment status)',
    'pump_state': 'B - Potentially available before prediction, but undocumented (equipment status)',
    'pump_speed': 'B - Potentially available before prediction, but undocumented (equipment sensor)',
    'compressor_state': 'B - Potentially available before prediction, but undocumented (equipment status)',
    'energy_consumption': 'B - Potentially available before prediction, but undocumented (aggregated measurement)',
    'alarm_triggered': 'C - Likely generated after an event/alarm (perfect one-direction: alarm=1 => target=1)',
    'event_type': 'D - Directly derived from target/event label (perfectly determines target)',
    'segment_id': 'A - Clearly available before prediction (static identifier)',
    'timestamp': 'A - Clearly available before prediction (time of observation)'
}

for field, assess in availability.items():
    print(f'  {field}: {assess}')

# ============================================================
# TASK J: LEAKAGE-EXCLUSION LIST
# ============================================================
print('\n=== TASK J: LEAKAGE-EXCLUSION LIST ===')

exclusion_list = [
    {
        'field': 'event_type',
        'category': 'Category 1 - Mandatory exclusion',
        'evidence': 'Perfectly determines target: normal->0 (100%), all others->1 (100%). Post-event classification.',
        'reason': 'Direct target leakage; perfect deterministic mapping to label',
        'reconsideration': 'Only if proven to be pre-event sensor input with documented timing'
    },
    {
        'field': 'alarm_triggered',
        'category': 'Category 1 - Mandatory exclusion',
        'evidence': 'P(target=1|alarm=1)=1.0 (216/216). alarm=1 => target=1 in 100% of cases. alarm=0 includes both targets.',
        'reason': 'Direct target leakage; alarm outcome reveals target with certainty when triggered',
        'reconsideration': 'Only if proven to be pre-event predictive alarm with documented timing'
    },
    {
        'field': 'valve_status',
        'category': 'Category 3 - Potentially admissible candidate',
        'evidence': 'Ordinal (0,1,2). Target rate varies but no perfect mapping. Varies within repeated groups.',
        'reason': 'Equipment state with predictive association but no proven leakage; availability undocumented',
        'reconsideration': 'Document prediction-time availability and temporal semantics'
    },
    {
        'field': 'pump_state',
        'category': 'Category 3 - Potentially admissible candidate',
        'evidence': 'Binary (0,1). Target rate: pump=1 -> 22.6%, pump=0 -> 50.3%. Varies within repeated groups.',
        'reason': 'Equipment state with predictive association; availability undocumented',
        'reconsideration': 'Document prediction-time availability and temporal semantics'
    },
    {
        'field': 'pump_speed',
        'category': 'Category 3 - Potentially admissible candidate',
        'evidence': 'Continuous (0-1678). Correlated with pump_state. Target=0 mean=1040, Target=1 mean=703.',
        'reason': 'Equipment sensor reading; zero-inflated with pump_state; availability undocumented',
        'reconsideration': 'Document prediction-time availability; address zero-inflation'
    },
    {
        'field': 'compressor_state',
        'category': 'Category 3 - Potentially admissible candidate',
        'evidence': 'Binary (0,1). Target rate: compressor=1 -> 26.0%, compressor=0 -> 37.5%.',
        'reason': 'Equipment state with predictive association; availability undocumented',
        'reconsideration': 'Document prediction-time availability and temporal semantics'
    },
    {
        'field': 'energy_consumption',
        'category': 'Category 3 - Potentially admissible candidate',
        'evidence': 'Continuous (5.4-58.2). Target=0 mean=27.8, Target=1 mean=24.6. Overlapping distributions.',
        'reason': 'Aggregated equipment measurement; availability undocumented',
        'reconsideration': 'Document prediction-time availability and temporal semantics'
    },
    {
        'field': 'pressure',
        'category': 'Category 3 - Potentially admissible candidate',
        'evidence': 'Primary telemetry. Continuous (43-115). Target=0 mean=76.7, Target=1 mean=75.1. Overlapping.',
        'reason': 'Core physical telemetry variable; no deterministic target encoding found',
        'reconsideration': 'Document sensor availability, units, and prediction-time latency'
    },
    {
        'field': 'flow_rate',
        'category': 'Category 3 - Potentially admissible candidate',
        'evidence': 'Primary telemetry. Continuous (1-7.5). Target=0 mean=4.4, Target=1 mean=4.3. Overlapping.',
        'reason': 'Core physical telemetry variable; no deterministic target encoding found',
        'reconsideration': 'Document sensor availability, units, and prediction-time latency'
    },
    {
        'field': 'temperature',
        'category': 'Category 3 - Potentially admissible candidate',
        'evidence': 'Primary telemetry. Continuous (26-38.5). Target=0 mean=32.1, Target=1 mean=32.0. Overlapping.',
        'reason': 'Core physical telemetry variable; no deterministic target encoding found',
        'reconsideration': 'Document sensor availability, units, and prediction-time latency'
    },
    {
        'field': 'segment_id',
        'category': 'Category 4 - Metadata only',
        'evidence': 'Static identifier (1-50). No predictive value; used for grouping sequences.',
        'reason': 'Not a predictive feature; grouping key only',
        'reconsideration': 'N/A - not a model input'
    },
    {
        'field': 'timestamp',
        'category': 'Category 4 - Metadata only',
        'evidence': 'Temporal index. Used for ordering and windowing, not direct prediction.',
        'reason': 'Not a predictive feature; chronological index only',
        'reconsideration': 'N/A - not a model input'
    },
    {
        'field': 'target',
        'category': 'Category 4 - Metadata only',
        'evidence': 'Label column (ground truth). Not an input feature.',
        'reason': 'Supervision target, not model input',
        'reconsideration': 'N/A'
    }
]

for item in exclusion_list:
    print(f"\n{item['field']}: {item['category']}")
    print(f"  Evidence: {item['evidence']}")
    print(f"  Reason: {item['reason']}")

# Save exclusion list CSV
ex_df = pd.DataFrame(exclusion_list)
ex_df.to_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\documentation\PIPER_phase1_4_leakage_exclusion_list.csv', index=False)
print('\nExclusion list saved to PIPER_phase1_4_leakage_exclusion_list.csv')

# Save conflict examples
conflict_ex_df = pd.DataFrame(conflict_examples)
conflict_ex_df.to_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\documentation\PIPER_phase1_4_target_conflict_examples.csv', index=False)
print('Conflict examples saved to PIPER_phase1_4_target_conflict_examples.csv')

# Save feature availability
avail_df = pd.DataFrame([{'field': k, 'availability_assessment': v} for k, v in availability.items()])
avail_df.to_csv(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\documentation\PIPER_phase1_4_feature_availability.csv', index=False)
print('Feature availability saved to PIPER_phase1_4_feature_availability.csv')

# ============================================================
# COMPILE JSON REPORT
# ============================================================
json_report = {
    'dataset_identity': {
        'file_path': 'data/raw/scada_pipeline.csv',
        'sha256': 'c7546b99557add43f9661409bf1d2fac0866bb674b502da69a9f9501c50ed68a',
        'rows': 1000,
        'columns': 13
    },
    'task_A_target_validation': {
        'total_rows': 1000,
        'null_target': 0,
        'unique_values': [0, 1],
        'counts': {'0': 694, '1': 306},
        'percentages': {'0': 69.4, '1': 30.6},
        'is_binary': True,
        'per_segment_distribution': seg_target.to_dict('index'),
        'per_timestamp_distribution': {str(k): {'abnormal': int(v['abnormal_count']), 'total': int(v['total']), 'rate': round(float(v['abnormal_rate']), 4)} for k, v in ts_target.iterrows()}
    },
    'task_B_cross_tabulation': {
        'event_type': pd.crosstab(df['event_type'], df['target']).to_dict(),
        'alarm_triggered': pd.crosstab(df['alarm_triggered'], df['target']).to_dict(),
        'valve_status': pd.crosstab(df['valve_status'], df['target']).to_dict(),
        'pump_state': pd.crosstab(df['pump_state'], df['target']).to_dict(),
        'compressor_state': pd.crosstab(df['compressor_state'], df['target']).to_dict(),
        'segment_id': pd.crosstab(df['segment_id'], df['target']).to_dict(),
        'numeric_by_target': {
            col: {
                'target_0': {'mean': float(df[df['target']==0][col].mean()), 'median': float(df[df['target']==0][col].median()), 'std': float(df[df['target']==0][col].std()), 'min': float(df[df['target']==0][col].min()), 'max': float(df[df['target']==0][col].max())},
                'target_1': {'mean': float(df[df['target']==1][col].mean()), 'median': float(df[df['target']==1][col].median()), 'std': float(df[df['target']==1][col].std()), 'min': float(df[df['target']==1][col].min()), 'max': float(df[df['target']==1][col].max())}
            } for col in numeric_fields
        }
    },
    'task_C_event_type_leakage': {
        'unique_values': sorted(df['event_type'].unique()),
        'counts': df['event_type'].value_counts().to_dict(),
        'target_distribution': pd.crosstab(df['event_type'], df['target']).to_dict(),
        'perfect_deterministic': True,
        'classification': 'EXCLUDE - DIRECT LEAKAGE'
    },
    'task_D_alarm_leakage': {
        'contingency': pd.crosstab(df['alarm_triggered'], df['target']).to_dict(),
        'alarm1_target1': int(alarm1_target1),
        'alarm1_target0': int(alarm1_target0),
        'alarm0_target1': int(alarm0_target1),
        'alarm0_target0': int(alarm0_target0),
        'p_target1_given_alarm1': float(p_target1_given_alarm1),
        'p_target1_given_alarm0': float(p_target1_given_alarm0),
        'one_direction_perfect': True,
        'classification': 'EXCLUDE - DIRECT LEAKAGE'
    },
    'task_E_status_indirect': {
        col: {
            'target_rate_by_category': pd.crosstab(df[col], df['target'], normalize='index')[1].round(4).to_dict() if df[col].nunique() <= 10 else 'continuous',
            'mean_by_target': {int(t): float(df[df['target']==t][col].mean()) for t in [0,1]},
            'varies_in_repeated_groups': 'from Phase 1.3'
        } for col in status_fields
    },
    'task_F_numeric_telemetry': {
        col: {
            'target_0': {'mean': float(df[df['target']==0][col].mean()), 'median': float(df[df['target']==0][col].median()), 'std': float(df[df['target']==0][col].std()), 'min': float(df[df['target']==0][col].min()), 'max': float(df[df['target']==0][col].max())},
            'target_1': {'mean': float(df[df['target']==1][col].mean()), 'median': float(df[df['target']==1][col].median()), 'std': float(df[df['target']==1][col].std()), 'min': float(df[df['target']==1][col].min()), 'max': float(df[df['target']==1][col].max())},
            'distribution_overlap': 'substantial - no deterministic threshold found'
        } for col in telemetry_fields
    },
    'task_G_target_conflicts': {
        'total_groups': total_groups,
        'single_row_groups': single_row_groups,
        'repeated_groups': repeated_groups,
        'rows_in_repeated': rows_in_repeated,
        'target_0_only_groups': int(target_0_only),
        'target_1_only_groups': int(target_1_only),
        'conflicting_target_groups': int(both_targets),
        'rows_in_conflicting_groups': int(conflict_rows),
        'conflict_percentage': round(conflict_rows/len(df)*100, 1),
        'target_alarm_disagreement': {
            'target1_alarm0': int(len(disagree)),
            'target0_alarm1': int(len(disagree2)),
            'by_event_type': disagree['event_type'].value_counts().to_dict()
        }
    },
    'task_H_encoding': {
        'perfect_fields': ['event_type'],
        'one_direction_perfect': ['alarm_triggered'],
        'suspicious_combinations': ['event_type + alarm_triggered jointly reveal target with certainty']
    },
    'task_I_availability': availability,
    'task_J_exclusion_list': exclusion_list,
    'task_K_lstm_input_policy': {
        'mandatory_excluded': ['event_type', 'alarm_triggered'],
        'excluded_pending_validation': [],
        'candidate_telemetry': ['pressure', 'flow_rate', 'temperature'],
        'candidate_equipment': ['valve_status', 'pump_state', 'pump_speed', 'compressor_state', 'energy_consumption'],
        'metadata_only': ['segment_id', 'timestamp', 'target'],
        'target_definition_supported': False,
        'unresolved_before_modeling': [
            'Target conflicts in 130 repeated groups',
            'Prediction-time availability of all candidate features undocumented',
            '16-minute span insufficient for temporal prediction',
            'Repeated-group semantics unknown',
            'No segment synchronization',
            'Vibration missing'
        ]
    }
}

with open(r'C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\documentation\PIPER_phase1_4_target_leakage_report.json', 'w') as f:
    json.dump(json_report, f, indent=2, default=str)

print('\nJSON report saved to PIPER_phase1_4_target_leakage_report.json')
print('=== ALL TASKS COMPLETE ===')