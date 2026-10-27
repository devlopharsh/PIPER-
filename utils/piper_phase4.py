"""
PIPER Phase 4: Tabular Prediction Module

Implements tabular classification for leak/abnormal condition detection using
the sensor-level train/validation/test splits from Phase 3.

Models: Logistic Regression (baseline), Random Forest, XGBoost
Preprocessing: Imputation, Scaling, Encoding (fitted on training data only)
Evaluation: Validation for model/threshold selection, Test for final report
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from xgboost import XGBClassifier

from utils.piper_telemetry import TARGET_COLUMN, PRIMARY_TELEMETRY, METADATA_COLUMNS


# =============================================================================
# CONFIGURATION
# =============================================================================

PIPER_MODELS_DIR = Path("./models")
PIPER_PROCESSED_DIR = Path("./data/processed")
PIPER_REPORTS_DIR = Path("./data/reports")

# Features to EXCLUDE from predictors (target leakage or non-predictive)
EXCLUDED_FEATURES = [
    TARGET_COLUMN,           # "label" - target variable
    "alarm_triggered",       # Directly encodes abnormal condition
    "event_type",            # Post-event information
    "timestamp",             # Metadata - not predictive
    "timestamp_parsed",      # Metadata - not predictive
    "sensor_id",             # Metadata - identifier (use sensor_id_encoded instead)
    "reading_index",         # Not a genuine temporal feature
    "n_sensors",             # Constant (always 50)
]

# Features that are constant in this dataset (1 unique value) - should be dropped
CONSTANT_FEATURES = [
    "hour", "dayofweek", "month", "quarter", "dayofyear", "weekofyear",
    "is_weekend", "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek",
    "sin_month", "cos_month",
]


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    class_report: Dict
    threshold: float = 0.5


@dataclass
class ModelResult:
    """Complete model training and evaluation result."""
    model_name: str
    model_config: Dict[str, Any]
    train_metrics: ModelMetrics
    val_metrics: ModelMetrics
    test_metrics: Optional[ModelMetrics] = None
    threshold: float = 0.5
    feature_names: List[str] = field(default_factory=list)
    training_time_seconds: float = 0.0


@dataclass
class Phase4Report:
    """Complete Phase 4 report."""
    dataset_name: str
    timestamp: str
    split_info: Dict[str, Any]
    feature_schema: Dict[str, Any]
    model_results: List[ModelResult]
    selected_model: str
    selected_threshold: float
    final_test_metrics: ModelMetrics
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "split_info": self.split_info,
            "feature_schema": self.feature_schema,
            "model_results": [
                {
                    "model_name": mr.model_name,
                    "model_config": mr.model_config,
                    "train_metrics": asdict(mr.train_metrics),
                    "val_metrics": asdict(mr.val_metrics),
                    "test_metrics": asdict(mr.test_metrics) if mr.test_metrics else None,
                    "threshold": mr.threshold,
                    "feature_names": mr.feature_names,
                    "training_time_seconds": mr.training_time_seconds,
                }
                for mr in self.model_results
            ],
            "selected_model": self.selected_model,
            "selected_threshold": self.selected_threshold,
            "final_test_metrics": asdict(self.final_test_metrics),
            "recommendations": self.recommendations,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# =============================================================================
# FEATURE ENGINEERING / PREPROCESSING
# =============================================================================

def get_predictor_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of predictor feature columns from dataframe.
    Excludes target, leakage features, metadata, and constant features.
    """
    all_cols = set(df.columns)
    excluded = set(EXCLUDED_FEATURES + CONSTANT_FEATURES)
    predictors = sorted(all_cols - excluded)
    return predictors


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], ColumnTransformer]:
    """
    Prepare features for modeling.
    
    Returns:
        X_train, X_val, X_test, feature_names, fitted_preprocessor
    """
    predictor_cols = get_predictor_features(train_df)
    
    X_train = train_df[predictor_cols].copy()
    X_val = val_df[predictor_cols].copy()
    X_test = test_df[predictor_cols].copy()
    
    # Identify column types
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"  Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', LabelEncoder()),  # Will be handled differently
    ])
    
    # For categorical, we'll use a simple approach: label encode each column
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', 'passthrough', categorical_cols),  # Handle separately
        ],
        remainder='drop'
    )
    
    # Fit on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Handle categorical columns with label encoding
    if categorical_cols:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on training data
            train_col = X_train[col].astype(str).fillna('missing')
            le.fit(train_col)
            label_encoders[col] = le
            
            # Transform all splits
            X_train_processed = np.column_stack([
                X_train_processed,
                le.transform(train_col).reshape(-1, 1)
            ])
            
            val_col = X_val[col].astype(str).fillna('missing')
            # Handle unseen categories
            val_encoded = np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in val_col])
            X_val_processed = np.column_stack([X_val_processed, val_encoded.reshape(-1, 1)])
            
            test_col = X_test[col].astype(str).fillna('missing')
            test_encoded = np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in test_col])
            X_test_processed = np.column_stack([X_test_processed, test_encoded.reshape(-1, 1)])
        
        # Update feature names
        feature_names = numeric_cols + categorical_cols
    else:
        feature_names = numeric_cols
        label_encoders = {}
    
    print(f"  Final feature matrix shapes: train={X_train_processed.shape}, val={X_val_processed.shape}, test={X_test_processed.shape}")
    
    return X_train_processed, X_val_processed, X_test_processed, feature_names, preprocessor, label_encoders


def prepare_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract target arrays."""
    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values
    return y_train, y_val, y_test


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5
) -> ModelMetrics:
    """Evaluate model with given threshold."""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        # For models without predict_proba (e.g., SVM without probability)
        y_proba = model.decision_function(X)
        # Normalize to [0, 1] range approximately
        y_proba = 1 / (1 + np.exp(-y_proba))
    
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y, y_pred)
    
    return ModelMetrics(
        accuracy=accuracy_score(y, y_pred),
        precision=precision_score(y, y_pred, zero_division=0),
        recall=recall_score(y, y_pred, zero_division=0),
        f1=f1_score(y, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
        confusion_matrix=cm.tolist(),
        class_report=classification_report(y, y_pred, output_dict=True, zero_division=0),
        threshold=threshold
    )


def find_optimal_threshold(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Find optimal threshold on validation set using F1 score."""
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


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight: str = 'balanced',
    max_iter: int = 1000,
    random_state: int = 42
) -> Tuple[LogisticRegression, float]:
    """Train Logistic Regression with balanced class weights."""
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    threshold = find_optimal_threshold(model, X_val, y_val)
    return model, threshold


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 200,
    max_depth: Optional[int] = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    class_weight: str = 'balanced',
    random_state: int = 42
) -> Tuple[RandomForestClassifier, float]:
    """Train Random Forest with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    threshold = find_optimal_threshold(model, X_val, y_val)
    return model, threshold


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42
) -> Tuple[XGBClassifier, float]:
    """Train XGBoost with class imbalance handling."""
    if scale_pos_weight is None:
        # Calculate from training data
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    threshold = find_optimal_threshold(model, X_val, y_val)
    return model, threshold


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_phase4_tabular(
    dataset_name: str = "SCADA_Pipeline",
    save_artifacts: bool = True
) -> Phase4Report:
    """
    Run complete Phase 4 tabular classification pipeline.
    """
    print(f"\n{'='*70}")
    print(f"PIPER PHASE 4: TABULAR CLASSIFICATION")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # 1. Load split data
    print("\n[1/6] Loading train/validation/test splits...")
    train_df = pd.read_csv(PIPER_PROCESSED_DIR / "scada_train.csv")
    val_df = pd.read_csv(PIPER_PROCESSED_DIR / "scada_val.csv")
    test_df = pd.read_csv(PIPER_PROCESSED_DIR / "scada_test.csv")
    
    # Load split metadata
    with open(PIPER_REPORTS_DIR / "scada_tabular_split_info.json") as f:
        split_info = json.load(f)
    
    print(f"  Train: {len(train_df)} rows, {train_df['sensor_id'].nunique()} sensors")
    print(f"  Val:   {len(val_df)} rows, {val_df['sensor_id'].nunique()} sensors")
    print(f"  Test:  {len(test_df)} rows, {test_df['sensor_id'].nunique()} sensors")
    
    # 2. Load feature manifest
    print("\n[2/6] Loading feature manifest...")
    with open(PIPER_PROCESSED_DIR / "scada_feature_manifest.json") as f:
        feature_manifest = json.load(f)
    
    # 3. Prepare features and targets
    print("\n[3/6] Preparing features and targets...")
    X_train, X_val, X_test, feature_names, preprocessor, label_encoders = prepare_features(
        train_df, val_df, test_df
    )
    y_train, y_val, y_test = prepare_targets(train_df, val_df, test_df)
    
    print(f"  Feature names ({len(feature_names)}): {feature_names}")
    print(f"  Train class distribution: {np.bincount(y_train)}")
    print(f"  Val class distribution: {np.bincount(y_val)}")
    print(f"  Test class distribution: {np.bincount(y_test)}")
    
    # 4. Train models
    print("\n[4/6] Training models...")
    model_results = []
    
    # Logistic Regression
    print("  Training Logistic Regression...")
    import time
    start = time.time()
    lr_model, lr_threshold = train_logistic_regression(X_train, y_train, X_val, y_val)
    lr_time = time.time() - start
    
    lr_train_metrics = evaluate_model(lr_model, X_train, y_train, lr_threshold)
    lr_val_metrics = evaluate_model(lr_model, X_val, y_val, lr_threshold)
    
    lr_result = ModelResult(
        model_name="LogisticRegression",
        model_config={
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42
        },
        train_metrics=lr_train_metrics,
        val_metrics=lr_val_metrics,
        threshold=lr_threshold,
        feature_names=feature_names,
        training_time_seconds=lr_time
    )
    model_results.append(lr_result)
    print(f"    Val F1: {lr_val_metrics.f1:.4f}, Val AUC: {lr_val_metrics.roc_auc:.4f}, Threshold: {lr_threshold:.4f}")
    
    # Random Forest
    print("  Training Random Forest...")
    start = time.time()
    rf_model, rf_threshold = train_random_forest(X_train, y_train, X_val, y_val)
    rf_time = time.time() - start
    
    rf_train_metrics = evaluate_model(rf_model, X_train, y_train, rf_threshold)
    rf_val_metrics = evaluate_model(rf_model, X_val, y_val, rf_threshold)
    
    rf_result = ModelResult(
        model_name="RandomForest",
        model_config={
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42
        },
        train_metrics=rf_train_metrics,
        val_metrics=rf_val_metrics,
        threshold=rf_threshold,
        feature_names=feature_names,
        training_time_seconds=rf_time
    )
    model_results.append(rf_result)
    print(f"    Val F1: {rf_val_metrics.f1:.4f}, Val AUC: {rf_val_metrics.roc_auc:.4f}, Threshold: {rf_threshold:.4f}")
    
    # XGBoost
    print("  Training XGBoost...")
    start = time.time()
    xgb_model, xgb_threshold = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_time = time.time() - start
    
    xgb_train_metrics = evaluate_model(xgb_model, X_train, y_train, xgb_threshold)
    xgb_val_metrics = evaluate_model(xgb_model, X_val, y_val, xgb_threshold)
    
    xgb_result = ModelResult(
        model_name="XGBoost",
        model_config={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        train_metrics=xgb_train_metrics,
        val_metrics=xgb_val_metrics,
        threshold=xgb_threshold,
        feature_names=feature_names,
        training_time_seconds=xgb_time
    )
    model_results.append(xgb_result)
    print(f"    Val F1: {xgb_val_metrics.f1:.4f}, Val AUC: {xgb_val_metrics.roc_auc:.4f}, Threshold: {xgb_threshold:.4f}")
    
    # 5. Select best model based on validation F1
    print("\n[5/6] Selecting best model...")
    best_result = max(model_results, key=lambda r: r.val_metrics.f1)
    selected_model_name = best_result.model_name
    selected_threshold = best_result.threshold
    selected_model = {"LogisticRegression": lr_model, "RandomForest": rf_model, "XGBoost": xgb_model}[selected_model_name]
    
    print(f"  Selected: {selected_model_name} (Val F1: {best_result.val_metrics.f1:.4f})")
    print(f"  Threshold: {selected_threshold:.4f}")
    
    # 6. Final evaluation on test set
    print("\n[6/6] Final evaluation on test set...")
    test_metrics = evaluate_model(selected_model, X_test, y_test, selected_threshold)
    print(f"  Test Accuracy: {test_metrics.accuracy:.4f}")
    print(f"  Test Precision: {test_metrics.precision:.4f}")
    print(f"  Test Recall: {test_metrics.recall:.4f}")
    print(f"  Test F1: {test_metrics.f1:.4f}")
    print(f"  Test ROC-AUC: {test_metrics.roc_auc:.4f}")
    print(f"  Confusion Matrix:\n{np.array(test_metrics.confusion_matrix)}")
    
    # Update best result with test metrics
    best_result.test_metrics = test_metrics
    
    # Feature schema for saving
    feature_schema = {
        "predictor_features": feature_names,
        "target_column": TARGET_COLUMN,
        "excluded_features": EXCLUDED_FEATURES + CONSTANT_FEATURES,
        "preprocessor_type": "ColumnTransformer(StandardScaler + LabelEncoder)",
        "n_features": len(feature_names),
        "feature_types": {
            "numeric": [c for c in feature_names if c in X_train.shape and True],  # Will be corrected
            "categorical": [c for c in feature_names if c in ['valve_status', 'pump_state', 'compressor_state']]
        }
    }
    
    # Recommendations
    recommendations = [
        f"Selected model: {selected_model_name} with threshold {selected_threshold:.4f}",
        f"Test F1: {test_metrics.f1:.4f}, ROC-AUC: {test_metrics.roc_auc:.4f}",
        "Split is sensor-level (no sensor appears in multiple splits)",
        "Preprocessing fitted on training data only (no leakage)",
        "Class imbalance handled via class_weight='balanced' and scale_pos_weight",
        "Threshold optimized on validation set (not test set)",
        "For production: monitor feature drift and retrain periodically"
    ]
    
    # Create report
    report = Phase4Report(
        dataset_name=dataset_name,
        timestamp=datetime.now().isoformat(),
        split_info=split_info,
        feature_schema={
            "predictor_features": feature_names,
            "target_column": TARGET_COLUMN,
            "excluded_features": EXCLUDED_FEATURES + CONSTANT_FEATURES,
            "n_features": len(feature_names)
        },
        model_results=model_results,
        selected_model=selected_model_name,
        selected_threshold=selected_threshold,
        final_test_metrics=test_metrics,
        recommendations=recommendations
    )
    
    if save_artifacts:
        print("\n[Saving artifacts]")
        PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = PIPER_MODELS_DIR / "phase4_model.joblib"
        joblib.dump(selected_model, model_path)
        print(f"  Model saved to: {model_path}")
        
        # Save preprocessor
        preprocessor_path = PIPER_MODELS_DIR / "phase4_preprocessor.joblib"
        joblib.dump({
            'preprocessor': preprocessor,
            'label_encoders': label_encoders,
            'feature_names': feature_names,
            'threshold': selected_threshold,
            'model_name': selected_model_name
        }, preprocessor_path)
        print(f"  Preprocessor saved to: {preprocessor_path}")
        
        # Save report
        report_path = PIPER_REPORTS_DIR / "phase4_report.json"
        report.save(report_path)
        print(f"  Report saved to: {report_path}")
        
        # Save test predictions
        test_preds_path = PIPER_PROCESSED_DIR / "scada_test_predictions.csv"
        if hasattr(selected_model, 'predict_proba'):
            test_proba = selected_model.predict_proba(X_test)[:, 1]
        else:
            test_proba = selected_model.decision_function(X_test)
            test_proba = 1 / (1 + np.exp(-test_proba))
        test_pred = (test_proba >= selected_threshold).astype(int)
        
        test_predictions = test_df[['sensor_id', 'timestamp', TARGET_COLUMN]].copy()
        test_predictions['predicted_probability'] = test_proba
        test_predictions['predicted_label'] = test_pred
        test_predictions.to_csv(test_preds_path, index=False)
        print(f"  Test predictions saved to: {test_preds_path}")
        
        # Save markdown summary
        md_path = PIPER_REPORTS_DIR / "phase4_summary.md"
        generate_phase4_summary_md(report, md_path)
        print(f"  Summary saved to: {md_path}")
    
    print(f"\n{'='*70}")
    print("PHASE 4 COMPLETE")
    print(f"{'='*70}")
    
    return report


def generate_phase4_summary_md(report: Phase4Report, output_path: Path) -> None:
    """Generate Phase 4 markdown summary."""
    md = f"""# PIPER Phase 4 Summary Report: {report.dataset_name}

**Generated:** {report.timestamp}

## Executive Summary

| Metric | Value |
|--------|-------|
| Selected Model | {report.selected_model} |
| Selected Threshold | {report.selected_threshold:.4f} |
| Test Accuracy | {report.final_test_metrics.accuracy:.4f} |
| Test Precision | {report.final_test_metrics.precision:.4f} |
| Test Recall | {report.final_test_metrics.recall:.4f} |
| Test F1 | {report.final_test_metrics.f1:.4f} |
| Test ROC-AUC | {report.final_test_metrics.roc_auc:.4f} |

## Split Information

| Split | Rows | Sensors | Normal (0) | Abnormal (1) | Abnormal Rate |
|-------|------|---------|------------|--------------|---------------|
| Train | {report.split_info['train_size']} | {len(report.split_info['train_sensors'])} | {report.split_info['train_class_dist']['0']} | {report.split_info['train_class_dist']['1']} | {report.split_info['train_class_dist']['1']/report.split_info['train_size']:.4f} |
| Val | {report.split_info['val_size']} | {len(report.split_info['val_sensors'])} | {report.split_info['val_class_dist']['0']} | {report.split_info['val_class_dist']['1']} | {report.split_info['val_class_dist']['1']/report.split_info['val_size']:.4f} |
| Test | {report.split_info['test_size']} | {len(report.split_info['test_sensors'])} | {report.split_info['test_class_dist']['0']} | {report.split_info['test_class_dist']['1']} | {report.split_info['test_class_dist']['1']/report.split_info['test_size']:.4f} |

**No sensor overlap between splits:** {report.split_info['no_sensor_overlap']}

## Feature Schema

- **Predictor features ({report.feature_schema['n_features']})**: {', '.join(report.feature_schema['predictor_features'])}
- **Target column**: {report.feature_schema['target_column']}
- **Excluded features**: {', '.join(report.feature_schema['excluded_features'])}

## Model Comparison (Validation Metrics)

| Model | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Threshold |
|-------|--------------|---------------|------------|--------|-------------|-----------|
"""
    
    for mr in report.model_results:
        md += f"| {mr.model_name} | {mr.val_metrics.accuracy:.4f} | {mr.val_metrics.precision:.4f} | {mr.val_metrics.recall:.4f} | {mr.val_metrics.f1:.4f} | {mr.val_metrics.roc_auc:.4f} | {mr.threshold:.4f} |\n"
    
    md += f"""

## Selected Model: {report.selected_model}

**Configuration:** {json.dumps(report.model_results[[mr.model_name for mr in report.model_results].index(report.selected_model)].model_config, indent=2)}

**Threshold:** {report.selected_threshold:.4f} (optimized on validation set)

## Final Test Evaluation

| Metric | Value |
|--------|-------|
| Accuracy | {report.final_test_metrics.accuracy:.4f} |
| Precision | {report.final_test_metrics.precision:.4f} |
| Recall | {report.final_test_metrics.recall:.4f} |
| F1-Score | {report.final_test_metrics.f1:.4f} |
| ROC-AUC | {report.final_test_metrics.roc_auc:.4f} |
| Threshold | {report.final_test_metrics.threshold:.4f} |

### Confusion Matrix (Test)

| | Predicted Normal | Predicted Abnormal |
|---|---|---|
| **Actual Normal** | {report.final_test_metrics.confusion_matrix[0][0]} | {report.final_test_metrics.confusion_matrix[0][1]} |
| **Actual Abnormal** | {report.final_test_metrics.confusion_matrix[1][0]} | {report.final_test_metrics.confusion_matrix[1][1]} |

### Class-wise Metrics (Test)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | {report.final_test_metrics.class_report['0']['precision']:.4f} | {report.final_test_metrics.class_report['0']['recall']:.4f} | {report.final_test_metrics.class_report['0']['f1-score']:.4f} | {int(report.final_test_metrics.class_report['0']['support'])} |
| Abnormal (1) | {report.final_test_metrics.class_report['1']['precision']:.4f} | {report.final_test_metrics.class_report['1']['recall']:.4f} | {report.final_test_metrics.class_report['1']['f1-score']:.4f} | {int(report.final_test_metrics.class_report['1']['support'])} |

## Artifacts Saved

- `models/phase4_model.joblib` - Fitted model
- `models/phase4_preprocessor.joblib` - Preprocessing pipeline + encoders
- `data/reports/phase4_report.json` - Full report
- `data/reports/phase4_summary.md` - This summary
- `data/processed/scada_test_predictions.csv` - Test set predictions

## Recommendations

"""
    
    for rec in report.recommendations:
        md += f"- {rec}\n"
    
    md += f"""

## Reproducibility

To reproduce this pipeline:

```bash
cd PIPER-
python -m utils.piper_phase4
```

All preprocessing (imputation, scaling, encoding) is fitted on training data only.
Validation set used for model selection and threshold optimization.
Test set remains isolated until final evaluation.

---
*Note: Split is sensor-level (35 train / 5 val / 10 test sensors). 
This does not guarantee generalization to unseen pipelines since pipeline_id is not available.*
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)


# =============================================================================
# INFERENCE FUNCTION
# =============================================================================

def load_phase4_model(model_path: Path = None, preprocessor_path: Path = None) -> Dict[str, Any]:
    """Load fitted model and preprocessor for inference."""
    if model_path is None:
        model_path = PIPER_MODELS_DIR / "phase4_model.joblib"
    if preprocessor_path is None:
        preprocessor_path = PIPER_MODELS_DIR / "phase4_preprocessor.joblib"
    
    model = joblib.load(model_path)
    preprocessor_data = joblib.load(preprocessor_path)
    
    return {
        'model': model,
        'preprocessor': preprocessor_data['preprocessor'],
        'label_encoders': preprocessor_data['label_encoders'],
        'feature_names': preprocessor_data['feature_names'],
        'threshold': preprocessor_data['threshold'],
        'model_name': preprocessor_data['model_name']
    }


def predict_phase4(
    df: pd.DataFrame,
    model_data: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Run inference on new data using fitted Phase 4 model.
    
    Args:
        df: DataFrame with same feature columns as training data
        model_data: Output from load_phase4_model() (loads from disk if None)
    
    Returns:
        DataFrame with predicted_probability and predicted_label columns
    """
    if model_data is None:
        model_data = load_phase4_model()
    
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']
    threshold = model_data['threshold']
    
    # Prepare features
    X = df[feature_names].copy()
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Transform numeric
    X_numeric = preprocessor.transform(X)
    
    # Transform categorical
    if categorical_cols and label_encoders:
        for col in categorical_cols:
            le = label_encoders[col]
            col_data = X[col].astype(str).fillna('missing')
            encoded = np.array([le.transform([v])[0] if v in le.classes_ else -1 for v in col_data])
            X_numeric = np.column_stack([X_numeric, encoded.reshape(-1, 1)])
    
    # Predict
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_numeric)[:, 1]
    else:
        proba = model.decision_function(X_numeric)
        proba = 1 / (1 + np.exp(-proba))
    
    pred = (proba >= threshold).astype(int)
    
    result = df.copy()
    result['predicted_probability'] = proba
    result['predicted_label'] = pred
    
    return result


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PIPER Phase 4: Tabular Classification")
    parser.add_argument("--dataset", default="scada", help="Dataset name")
    parser.add_argument("--no-save", action="store_true", help="Don't save artifacts")
    parser.add_argument("--inference", action="store_true", help="Run inference on test set")
    args = parser.parse_args()
    
    if args.inference:
        # Load test data and run inference
        test_df = pd.read_csv(PIPER_PROCESSED_DIR / "scada_test.csv")
        model_data = load_phase4_model()
        predictions = predict_phase4(test_df, model_data)
        
        print("Inference results on test set:")
        print(predictions[['sensor_id', 'timestamp', 'label', 'predicted_probability', 'predicted_label']].head(20))
        
        from sklearn.metrics import accuracy_score, f1_score
        print(f"\nAccuracy: {accuracy_score(predictions['label'], predictions['predicted_label']):.4f}")
        print(f"F1: {f1_score(predictions['label'], predictions['predicted_label']):.4f}")
    else:
        run_phase4_tabular(dataset_name=args.dataset, save_artifacts=not args.no_save)


if __name__ == "__main__":
    main()