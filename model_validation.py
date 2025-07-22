#!/usr/bin/env python3
"""
Model Validation Script for Credit Risk XGBoost Model
Performs comprehensive validation and generates detailed report
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, log_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """Load the XGBoost model and test data"""
    print("Loading model and data...")
    
    # Load model
    model = xgb.Booster()
    model.load_model('tune_best.xgb')
    
    # Load test data
    data_path = "/mnt/data/Credit-Risk-Model/data/test_data.csv"
    test_df = pd.read_csv(data_path)
    
    # Separate features and target
    X_test = test_df.drop('credit', axis=1)
    y_test = test_df['credit']
    
    # Convert to DMatrix for XGBoost
    dtest = xgb.DMatrix(X_test)
    
    return model, X_test, y_test, dtest

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'average_precision': average_precision_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba)
    }
    
    # Add confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp),
        'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
    })
    
    return metrics

def analyze_threshold_performance(y_true, y_pred_proba):
    """Analyze performance at different thresholds"""
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_metrics = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        threshold_metrics.append(metrics)
    
    return pd.DataFrame(threshold_metrics)

def analyze_business_impact(y_true, y_pred_proba):
    """Analyze business impact using the three-tier system"""
    # Define business thresholds
    approved = y_pred_proba >= 0.6
    manual_review = (y_pred_proba >= 0.4) & (y_pred_proba < 0.6)
    denied = y_pred_proba < 0.4
    
    # Calculate distribution
    distribution = {
        'approved': np.sum(approved),
        'manual_review': np.sum(manual_review),
        'denied': np.sum(denied)
    }
    
    # Calculate accuracy within each tier
    tier_performance = {
        'approved_accuracy': accuracy_score(y_true[approved], (y_pred_proba[approved] >= 0.5).astype(int)) if np.sum(approved) > 0 else 0,
        'manual_review_accuracy': accuracy_score(y_true[manual_review], (y_pred_proba[manual_review] >= 0.5).astype(int)) if np.sum(manual_review) > 0 else 0,
        'denied_accuracy': accuracy_score(y_true[denied], (y_pred_proba[denied] >= 0.5).astype(int)) if np.sum(denied) > 0 else 0
    }
    
    # Calculate bad rate in each tier
    bad_rates = {
        'approved_bad_rate': 1 - np.mean(y_true[approved]) if np.sum(approved) > 0 else 0,
        'manual_review_bad_rate': 1 - np.mean(y_true[manual_review]) if np.sum(manual_review) > 0 else 0,
        'denied_bad_rate': 1 - np.mean(y_true[denied]) if np.sum(denied) > 0 else 0
    }
    
    return distribution, tier_performance, bad_rates

def create_visualizations(model, X_test, y_test, y_pred_proba, report_dir):
    """Create and save validation visualizations"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig(f'{report_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'{report_dir}/pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix
    plt.figure(figsize=(8, 6))
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'{report_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Calibration Plot
    plt.figure(figsize=(8, 6))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.savefig(f'{report_dir}/calibration_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Score Distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Bad Credit', density=True)
    plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Good Credit', density=True)
    plt.axvline(x=0.4, color='red', linestyle='--', label='Deny Threshold')
    plt.axvline(x=0.6, color='green', linestyle='--', label='Approve Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Score Distribution by Class')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(y_pred_proba, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0.4, color='red', linestyle='--', label='Deny Threshold')
    plt.axvline(x=0.6, color='green', linestyle='--', label='Approve Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Overall Score Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{report_dir}/score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Feature Importance (Top 20)
    plt.figure(figsize=(10, 8))
    importance = model.get_score(importance_type='gain')
    if importance:
        features = list(importance.keys())[:20]
        scores = [importance[f] for f in features]
        y_pos = np.arange(len(features))
        plt.barh(y_pos, scores)
        plt.yticks(y_pos, features)
        plt.xlabel('Gain')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{report_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(model, X_test, y_test, y_pred_proba, report_dir):
    """Generate comprehensive validation report"""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Threshold analysis
    threshold_df = analyze_threshold_performance(y_test, y_pred_proba)
    
    # Business impact analysis
    distribution, tier_performance, bad_rates = analyze_business_impact(y_test, y_pred_proba)
    
    # Create visualizations
    create_visualizations(model, X_test, y_test, y_pred_proba, report_dir)
    
    # Generate text report
    report = f"""
# Credit Risk Model Validation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Model File: tune_best.xgb
- Model Type: XGBoost Binary Classifier
- Test Set Size: {len(y_test)} samples
- Class Distribution: {dict(y_test.value_counts())}

## Overall Performance Metrics

### Classification Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Balanced Accuracy**: {metrics['balanced_accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
- **Matthews Correlation Coefficient**: {metrics['matthews_corrcoef']:.4f}

### Probabilistic Metrics
- **ROC-AUC**: {metrics['roc_auc']:.4f}
- **Average Precision**: {metrics['average_precision']:.4f}
- **Log Loss**: {metrics['log_loss']:.4f}

### Confusion Matrix Analysis
- **True Positives**: {metrics['true_positives']} (Correctly identified good credit)
- **True Negatives**: {metrics['true_negatives']} (Correctly identified bad credit)
- **False Positives**: {metrics['false_positives']} (Bad credit classified as good)
- **False Negatives**: {metrics['false_negatives']} (Good credit classified as bad)
- **Specificity**: {metrics['specificity']:.4f}
- **Negative Predictive Value**: {metrics['negative_predictive_value']:.4f}

## Business Impact Analysis

### Decision Distribution
- **Approved (≥0.6)**: {distribution['approved']} ({distribution['approved']/len(y_test)*100:.1f}%)
- **Manual Review (0.4-0.6)**: {distribution['manual_review']} ({distribution['manual_review']/len(y_test)*100:.1f}%)
- **Denied (<0.4)**: {distribution['denied']} ({distribution['denied']/len(y_test)*100:.1f}%)

### Bad Rate by Decision Tier
- **Approved Applications**: {bad_rates['approved_bad_rate']:.2%} bad rate
- **Manual Review Applications**: {bad_rates['manual_review_bad_rate']:.2%} bad rate
- **Denied Applications**: {bad_rates['denied_bad_rate']:.2%} bad rate

## Threshold Analysis

{threshold_df.to_string(index=False)}

## Model Characteristics

### Classification Report
{classification_report(y_test, y_pred, target_names=['Bad Credit', 'Good Credit'])}

### Recommendations
1. The model shows {'strong' if metrics['roc_auc'] > 0.8 else 'moderate'} discriminatory power with an AUC of {metrics['roc_auc']:.3f}
2. The business thresholds effectively separate risk levels with bad rates of {bad_rates['approved_bad_rate']:.1%} for approved vs {bad_rates['denied_bad_rate']:.1%} for denied
3. Consider adjusting thresholds based on business risk appetite and manual review capacity

## Visualizations

### ROC Curve
![ROC Curve](roc_curve.png)

### Precision-Recall Curve
![Precision-Recall Curve](pr_curve.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Calibration Plot
![Calibration Plot](calibration_plot.png)

### Score Distribution
![Score Distribution](score_distribution.png)

### Feature Importance
![Feature Importance](feature_importance.png)
"""
    
    # Save report
    with open(f'{report_dir}/validation_report.md', 'w') as f:
        f.write(report)
    
    # Save metrics as JSON
    full_metrics = {
        'metrics': metrics,
        'threshold_analysis': threshold_df.to_dict('records'),
        'business_impact': {
            'distribution': {k: int(v) for k, v in distribution.items()},
            'tier_performance': tier_performance,
            'bad_rates': bad_rates
        }
    }
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    full_metrics = convert_to_serializable(full_metrics)
    
    with open(f'{report_dir}/validation_metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2)
    
    print(f"\nValidation complete! Results saved to {report_dir}/")
    print(report)

def main():
    """Main validation function"""
    # Create report directory
    report_dir = 'validation_report'
    os.makedirs(report_dir, exist_ok=True)
    
    # Load model and data
    model, X_test, y_test, dtest = load_model_and_data()
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict(dtest)
    
    # Generate report
    print("Generating validation report...")
    generate_report(model, X_test, y_test, y_pred_proba, report_dir)

if __name__ == "__main__":
    main()