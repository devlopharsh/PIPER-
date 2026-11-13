import pandas as pd
import numpy as np
import json
import joblib
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Paths
PIPER_PROCESSED_DIR = Path("./data/processed")
PIPER_REPORTS_DIR = Path("./data/reports")
PIPER_MODELS_DIR = Path("./models")
PIPER_PLOTS_DIR = Path("./data/reports/plots")

# Column mappings
EXCLUDED_FEATURES = [
    "label", "target", "alarm_triggered", "event_type", "timestamp", "timestamp_parsed",
    "sensor_id", "reading_index", "n_sensors", "segment_id"
]
CONSTANT_FEATURES = [
    "hour", "dayofweek", "month", "quarter", "dayofyear", "weekofyear",
    "is_weekend", "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek",
    "sin_month", "cos_month",
]

TARGET_COLUMN = "label"

# Load split info
with open(PIPER_REPORTS_DIR / "scada_tabular_split_info.json") as f:
    split_info = json.load(f)

train_sensors = set(split_info['train_sensors'])
val_sensors = set(split_info['val_sensors'])
test_sensors = set(split_info['test_sensors'])

print("Train sensors:", len(train_sensors))
print("Val sensors:", len(val_sensors))
print("Test sensors:", len(test_sensors))

def load_and_prepare_dataset(csv_path, dataset_name):
    """Load dataset and apply sensor-level splits."""
    df = pd.read_csv(csv_path)
    
    # Parse timestamp
    df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=False)
    
    # Map segment_id to sensor_id for PIPER compatibility
    df['sensor_id'] = df['segment_id']
    
    # Map target to label
    df['label'] = df['target']
    
    # Drop original target column to prevent leakage
    df = df.drop(columns=['target'])
    
    # Sort by sensor_id, timestamp
    df = df.sort_values(['sensor_id', 'timestamp_parsed']).reset_index(drop=True)
    
    # Add reading_index for duplicate timestamps per sensor
    df['reading_index'] = df.groupby(['sensor_id', 'timestamp_parsed']).cumcount()
    
    # Apply sensor-level splits
    train_df = df[df['sensor_id'].isin(train_sensors)].copy()
    val_df = df[df['sensor_id'].isin(val_sensors)].copy()
    test_df = df[df['sensor_id'].isin(test_sensors)].copy()
    
    print(f"{dataset_name}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df

def engineer_features(df, dataset_name):
    """Run feature engineering on SCADA dataset."""
    # Add temporal features
    ts = df['timestamp_parsed']
    df['hour'] = ts.dt.hour
    df['dayofweek'] = ts.dt.dayofweek
    df['month'] = ts.dt.month
    df['quarter'] = ts.dt.quarter
    df['dayofyear'] = ts.dt.dayofyear
    df['weekofyear'] = ts.dt.isocalendar().week
    df['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
    df['sin_hour'] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df['cos_hour'] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df['sin_dayofweek'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df['sin_month'] = np.sin(2 * np.pi * ts.dt.month / 12)
    df['cos_month'] = np.cos(2 * np.pi * ts.dt.month / 12)
    
    # Pressure-flow ratio
    df['flow_pressure_ratio'] = np.where(
        df['pressure'] != 0,
        df['flow_rate'] / df['pressure'],
        0.0
    )
    
    # Sensor ID encoding
    sensor_codes, uniques = pd.factorize(df['sensor_id'])
    df['sensor_id_encoded'] = sensor_codes
    df['n_sensors'] = len(uniques)
    
    return df

def get_predictor_features(df):
    """Get predictor feature columns."""
    all_cols = set(df.columns)
    excluded = set(EXCLUDED_FEATURES + CONSTANT_FEATURES)
    predictors = sorted(all_cols - excluded)
    return predictors

def prepare_features(train_df, val_df, test_df):
    """Prepare features for modeling."""
    predictor_cols = get_predictor_features(train_df)
    
    X_train = train_df[predictor_cols].copy()
    X_val = val_df[predictor_cols].copy()
    X_test = test_df[predictor_cols].copy()
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"  Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', 'passthrough', categorical_cols),
        ],
        remainder='drop'
    )
    
    # Fit on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Handle categorical columns with label encoding
    label_encoders = {}
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            train_col = X_train[col].astype(str).fillna('missing')
            le.fit(train_col)
            label_encoders[col] = le
            
            X_train_processed = np.column_stack([
                X_train_processed,
                le.transform(train_col).reshape(-1, 1)
            ])
            
            val_col = X_val[col].astype(str).fillna('missing')
            val_encoded = np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in val_col])
            X_val_processed = np.column_stack([X_val_processed, val_encoded.reshape(-1, 1)])
            
            test_col = X_test[col].astype(str).fillna('missing')
            test_encoded = np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in test_col])
            X_test_processed = np.column_stack([X_test_processed, test_encoded.reshape(-1, 1)])
        
        feature_names = numeric_cols + categorical_cols
    else:
        feature_names = numeric_cols
    
    print(f"  Final shapes: train={X_train_processed.shape}, val={X_val_processed.shape}, test={X_test_processed.shape}")
    
    return X_train_processed, X_val_processed, X_test_processed, feature_names, preprocessor, label_encoders

def prepare_targets(train_df, val_df, test_df):
    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values
    return y_train, y_val, y_test

def evaluate_model(model, X, y, threshold=0.5):
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.decision_function(X)
        y_proba = 1 / (1 + np.exp(-y_proba))
    
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
        'pr_auc': average_precision_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
        'confusion_matrix': cm.tolist(),
        'y_proba': y_proba,
        'y_pred': y_pred,
        'threshold': threshold
    }

def find_optimal_threshold(model, X_val, y_val):
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = model.decision_function(X_val)
        y_proba = 1 / (1 + np.exp(-y_proba))
    
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold

def train_models(X_train, y_train, X_val, y_val):
    """Train all three models."""
    results = {}
    
    # Logistic Regression
    print("  Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
    lr_model.fit(X_train, y_train)
    lr_threshold = find_optimal_threshold(lr_model, X_val, y_val)
    results['LogisticRegression'] = (lr_model, lr_threshold)
    
    # Random Forest
    print("  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_threshold = find_optimal_threshold(rf_model, X_val, y_val)
    results['RandomForest'] = (rf_model, rf_threshold)
    
    # XGBoost
    print("  Training XGBoost...")
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_threshold = find_optimal_threshold(xgb_model, X_val, y_val)
    results['XGBoost'] = (xgb_model, xgb_threshold)
    
    return results

def run_evaluation_pipeline(train_df, val_df, test_df, dataset_name):
    """Run complete evaluation pipeline."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {dataset_name}")
    print(f"{'='*60}")
    
    # Feature engineering
    print("\n[1] Feature engineering...")
    train_fe = engineer_features(train_df, dataset_name)
    val_fe = engineer_features(val_df, dataset_name)
    test_fe = engineer_features(test_df, dataset_name)
    
    # Prepare features
    print("\n[2] Preparing features...")
    X_train, X_val, X_test, feature_names, preprocessor, label_encoders = prepare_features(
        train_fe, val_fe, test_fe
    )
    y_train, y_val, y_test = prepare_targets(train_fe, val_fe, test_fe)
    
    print(f"  Train class dist: {np.bincount(y_train)}")
    print(f"  Val class dist: {np.bincount(y_val)}")
    print(f"  Test class dist: {np.bincount(y_test)}")
    
    # Train models
    print("\n[3] Training models...")
    models = train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation
    print("\n[4] Validation evaluation...")
    val_results = {}
    for name, (model, threshold) in models.items():
        metrics = evaluate_model(model, X_val, y_val, threshold)
        val_results[name] = metrics
        print(f"  {name}: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, Threshold={threshold:.4f}")
    
    # Select best model
    best_model_name = max(val_results.keys(), key=lambda k: val_results[k]['f1'])
    best_model, best_threshold = models[best_model_name]
    print(f"\n[5] Selected model: {best_model_name} (Val F1={val_results[best_model_name]['f1']:.4f})")
    
    # Final test evaluation
    print("\n[6] Test evaluation...")
    test_metrics = evaluate_model(best_model, X_test, y_test, best_threshold)
    
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall']:.4f}")
    print(f"  Test F1:        {test_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC:    {test_metrics['pr_auc']:.4f}")
    cm = np.array(test_metrics['confusion_matrix'])
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"  Specificity:    {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
    
    # Also evaluate other models on test for comparison
    print("\n[7] All models test comparison:")
    test_results = {}
    for name, (model, threshold) in models.items():
        metrics = evaluate_model(model, X_test, y_test, threshold)
        test_results[name] = metrics
        print(f"  {name}: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
    
    return {
        'dataset_name': dataset_name,
        'feature_names': feature_names,
        'val_results': val_results,
        'test_results': test_results,
        'best_model': best_model_name,
        'best_threshold': best_threshold,
        'test_metrics': test_metrics,
        'confusion_matrix': cm.tolist()
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Load original dataset (the one used for current splits)
    print("Loading original dataset...")
    original_path = r"C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_updated.csv"
    
    # Load corrected dataset
    print("Loading corrected dataset...")
    corrected_path = r"C:\Users\krish\OneDrive\Desktop\piper\PIPER-\scada_pipeline_1hour_valve_corrected.csv"
    
    # Run evaluation on original
    train_orig, val_orig, test_orig = load_and_prepare_dataset(original_path, "Original")
    results_orig = run_evaluation_pipeline(train_orig, val_orig, test_orig, "Original Dataset")
    
    # Run evaluation on corrected
    train_corr, val_corr, test_corr = load_and_prepare_dataset(corrected_path, "Corrected")
    results_corr = run_evaluation_pipeline(train_corr, val_corr, test_corr, "Corrected Dataset")
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON: Original vs Corrected")
    print(f"{'='*80}")
    
    print("\n| Metric | Original | Corrected | Difference |")
    print("|--------|----------|-----------|------------|")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        orig_val = results_orig['test_results'][results_orig['best_model']][metric]
        corr_val = results_corr['test_results'][results_corr['best_model']][metric]
        diff = corr_val - orig_val
        print(f"| {metric.capitalize()} | {orig_val:.4f} | {corr_val:.4f} | {diff:+.4f} |")
    
    print(f"\nBest model (Original): {results_orig['best_model']}")
    print(f"Best model (Corrected): {results_corr['best_model']}")
    
    # Save comparison report
    comparison = {
        'original': {
            'best_model': results_orig['best_model'],
            'best_threshold': results_orig['best_threshold'],
            'test_metrics': results_orig['test_results'][results_orig['best_model']]
        },
        'corrected': {
            'best_model': results_corr['best_model'],
            'best_threshold': results_corr['best_threshold'],
            'test_metrics': results_corr['test_results'][results_corr['best_model']]
        }
    }
    
    with open(PIPER_REPORTS_DIR / "valve_correction_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"\nComparison report saved to: {PIPER_REPORTS_DIR / 'valve_correction_comparison.json'}")