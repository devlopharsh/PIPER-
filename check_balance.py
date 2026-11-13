import pandas as pd

df = pd.read_csv('segment_analysis_results.csv')

# Calculate class balance ratio (closer to 1.0 = more balanced)
df['class_balance_ratio'] = df.apply(lambda row: min(row['n_positive'], row['n_negative']) / max(row['n_positive'], row['n_negative']), axis=1)

# Sort by class balance ratio (most balanced first)
print("=== MOST BALANCED BY CLASS DISTRIBUTION (50/50 split) ===")
print(df.nlargest(15, 'class_balance_ratio')[['segment_id', 'n_samples', 'n_positive', 'n_negative', 'class_balance_ratio', 'balanced_accuracy', 'f1_score']].to_string(index=False))

print("\n=== BEST COMBINED: Class Balance + Performance ===")
# Combined score: class_balance * balanced_accuracy
df['combined_score'] = df['class_balance_ratio'] * df['balanced_accuracy']
print(df.nlargest(15, 'combined_score')[['segment_id', 'n_samples', 'n_positive', 'n_negative', 'class_balance_ratio', 'balanced_accuracy', 'f1_score', 'combined_score']].to_string(index=False))

print("\n=== SEGMENT 11 DETAIL (Best class balance among high performers) ===")
seg11 = df[df['segment_id'] == 11].iloc[0]
print(f"Segment 11: {seg11['n_positive']} pos, {seg11['n_negative']} neg, ratio={seg11['class_balance_ratio']:.3f}")
print(f"  Balanced Acc: {seg11['balanced_accuracy']:.4f}, F1: {seg11['f1_score']:.4f}")
print(f"  CM: [[{int(seg11['tn'])}, {int(seg11['fp'])}], [{int(seg11['fn'])}, {int(seg11['tp'])}]]")

print("\n=== SEGMENT 6 DETAIL (Perfect 50/50 split) ===")
seg6 = df[df['segment_id'] == 6].iloc[0]
print(f"Segment 6: {seg6['n_positive']} pos, {seg6['n_negative']} neg, ratio={seg6['class_balance_ratio']:.3f}")
print(f"  Balanced Acc: {seg6['balanced_accuracy']:.4f}, F1: {seg6['f1_score']:.4f}")
print(f"  CM: [[{int(seg6['tn'])}, {int(seg6['fp'])}], [{int(seg6['fn'])}, {int(seg6['tp'])}]]")

print("\n=== SEGMENT 45 DETAIL (Near 50/50 + High Performance) ===")
seg45 = df[df['segment_id'] == 45].iloc[0]
print(f"Segment 45: {seg45['n_positive']} pos, {seg45['n_negative']} neg, ratio={seg45['class_balance_ratio']:.3f}")
print(f"  Balanced Acc: {seg45['balanced_accuracy']:.4f}, F1: {seg45['f1_score']:.4f}")
print(f"  CM: [[{int(seg45['tn'])}, {int(seg45['fp'])}], [{int(seg45['fn'])}, {int(seg45['tp'])}]]")