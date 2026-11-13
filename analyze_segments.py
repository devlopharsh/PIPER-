import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r"C:\Users\krish\OneDrive\Desktop\piper\PIPER-\data\raw\scada_pipeline.csv")

# Features and target
feature_cols = ['pressure', 'flow_rate', 'temperature', 'valve_status', 'pump_state', 'pump_speed', 
                'compressor_state', 'energy_consumption', 'alarm_triggered']
target_col = 'target'

# Encode event_type as additional feature
le = LabelEncoder()
df['event_type_encoded'] = le.fit_transform(df['event_type'])
feature_cols_with_event = feature_cols + ['event_type_encoded']

results = []

for seg_id in sorted(df['segment_id'].unique()):
    seg_data = df[df['segment_id'] == seg_id].copy()
    
    # Skip if too few samples or only one class
    if len(seg_data) < 10:
        print(f"Segment {seg_id}: Only {len(seg_data)} samples, skipping")
        continue
    
    if seg_data[target_col].nunique() < 2:
        print(f"Segment {seg_id}: Only one class present, skipping")
        continue
    
    X = seg_data[feature_cols_with_event]
    y = seg_data[target_col]
    
    # Use stratified k-fold cross-validation for more robust estimates
    # With small datasets, use leave-one-out or small k
    n_splits = min(5, len(seg_data) // 2)
    if n_splits < 2:
        n_splits = 2
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Get cross-validated predictions
    try:
        y_pred = cross_val_predict(clf, X, y, cv=skf, method='predict')
    except:
        # Fallback to simple train/test split if CV fails
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_test_vals = y_test
    else:
        y_test_vals = y
    
    # Compute metrics
    cm = confusion_matrix(y_test_vals, y_pred)
    acc = accuracy_score(y_test_vals, y_pred)
    f1 = f1_score(y_test_vals, y_pred, average='binary')
    prec = precision_score(y_test_vals, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_test_vals, y_pred, average='binary', zero_division=0)
    
    # For binary classification, confusion matrix is 2x2
    # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Compute additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    balanced_acc = (sensitivity + specificity) / 2
    
    results.append({
        'segment_id': seg_id,
        'n_samples': len(seg_data),
        'n_positive': int(y.sum()),
        'n_negative': int((~y.astype(bool)).sum()),
        'accuracy': acc,
        'f1_score': f1,
        'precision': prec,
        'recall': rec,
        'specificity': specificity,
        'balanced_accuracy': balanced_acc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    })
    
    print(f"Segment {seg_id:2d}: n={len(seg_data):2d}, pos={int(y.sum()):2d}, "
          f"acc={acc:.3f}, f1={f1:.3f}, prec={prec:.3f}, rec={rec:.3f}, "
          f"spec={specificity:.3f}, bal_acc={balanced_acc:.3f}, "
          f"CM=[[{tn},{fp}],[{fn},{tp}]]")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1_score', ascending=False)

print("\n" + "="*80)
print("TOP 10 SEGMENTS BY F1-SCORE:")
print("="*80)
print(results_df.head(10).to_string(index=False))

print("\n" + "="*80)
print("TOP 10 SEGMENTS BY BALANCED ACCURACY:")
print("="*80)
print(results_df.nlargest(10, 'balanced_accuracy').to_string(index=False))

print("\n" + "="*80)
print("ALL SEGMENTS RANKED BY F1-SCORE:")
print("="*80)
print(results_df[['segment_id', 'n_samples', 'n_positive', 'accuracy', 'f1_score', 'precision', 'recall', 'specificity', 'balanced_accuracy']].to_string(index=False))

# Save results
results_df.to_csv('segment_analysis_results.csv', index=False)
print("\nResults saved to segment_analysis_results.csv")

# Best segment
best_seg = results_df.iloc[0]['segment_id']
print(f"\nBEST SEGMENT by F1-score: Segment {best_seg}")
best_row = results_df[results_df['segment_id'] == best_seg].iloc[0]
print(f"  Confusion Matrix: [[{int(best_row['tn'])}, {int(best_row['fp'])}], [{int(best_row['fn'])}, {int(best_row['tp'])}]]")
print(f"  Accuracy: {best_row['accuracy']:.4f}")
print(f"  F1-Score: {best_row['f1_score']:.4f}")
print(f"  Precision: {best_row['precision']:.4f}")
print(f"  Recall: {best_row['recall']:.4f}")
print(f"  Specificity: {best_row['specificity']:.4f}")
print(f"  Balanced Accuracy: {best_row['balanced_accuracy']:.4f}")