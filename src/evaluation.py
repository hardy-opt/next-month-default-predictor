"""
Model Evaluation Module
Contains functions for evaluating machine learning models with proper visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
import joblib
import os


def plot_confusion_matrix(model, X_test, y_test, model_name, save_path=None):
    """
    Plot confusion matrix using sklearn's ConfusionMatrixDisplay
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with predict method
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    model_name : str
        Name of the model for title
    save_path : str, optional
        Path to save the figure
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create the display
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['No Default', 'Default']
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add percentage annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / cm.sum() * 100
            ax.text(j, i + 0.3, f'({percentage:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cm


def plot_roc_curve(model, X_test, y_test, model_name, save_path=None):
    """
    Plot ROC curve using sklearn's RocCurveDisplay
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with predict_proba method
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    model_name : str
        Name of the model for title
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(model, 'predict_proba'):
        print(f"Model {model_name} does not have predict_proba method. Skipping ROC curve.")
        return None
    
    # Create the display
    disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    disp.ax_.set_title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
    disp.ax_.grid(True, alpha=0.3)
    
    # Add AUC score to the plot
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    disp.ax_.text(0.6, 0.2, f'AUC = {auc_score:.3f}', fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return auc_score


def plot_precision_recall_curve(model, X_test, y_test, model_name, save_path=None):
    """
    Plot precision-recall curve using sklearn's PrecisionRecallDisplay
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with predict_proba method
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    model_name : str
        Name of the model for title
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(model, 'predict_proba'):
        print(f"Model {model_name} does not have predict_proba method. Skipping PR curve.")
        return None
    
    # Create the display
    disp = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    disp.ax_.set_title(f'{model_name} - Precision-Recall Curve', fontsize=14, fontweight='bold')
    disp.ax_.grid(True, alpha=0.3)
    
    # Add average precision score
    avg_precision = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    disp.ax_.text(0.6, 0.2, f'Avg Precision = {avg_precision:.3f}', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return avg_precision


def plot_feature_importance(model, feature_names, model_name, top_n=20, save_path=None):
    """
    Plot feature importance for models that support it
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model for title
    top_n : int
        Number of top features to display
    save_path : str, optional
        Path to save the figure
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not have feature_importances_ attribute.")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)
    
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'{model_name} - Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return feature_importance_df


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive classification metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities for positive class
        
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'specificity': None,
        'auc_score': None,
        'avg_precision': None
    }
    
    # Calculate specificity (True Negative Rate)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate AUC and Average Precision if probabilities are provided
    if y_pred_proba is not None:
        metrics['auc_score'] = roc_auc_score(y_true, y_pred_proba)
        metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
    
    return metrics


def evaluate_model(model, X_test, y_test, model_name, feature_names=None, save_dir=None):
    """
    Comprehensive model evaluation with all metrics and visualizations
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        True labels
    model_name : str
        Name of the model
    feature_names : list, optional
        List of feature names for feature importance plot
    save_dir : str, optional
        Directory to save all plots
        
    Returns:
    --------
    dict : Dictionary containing all evaluation results
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"{'='*60}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Print metrics
    print(f"\n=== PERFORMANCE METRICS ===")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1_score']:.4f}")
    print(f"Specificity:   {metrics['specificity']:.4f}")
    
    if metrics['auc_score'] is not None:
        print(f"AUC Score:     {metrics['auc_score']:.4f}")
        print(f"Avg Precision: {metrics['avg_precision']:.4f}")
    
    # Detailed classification report
    print(f"\n=== DETAILED CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    # Create visualizations
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Confusion Matrix
    cm_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png') if save_dir else None
    cm = plot_confusion_matrix(model, X_test, y_test, model_name, cm_path)
    
    # ROC Curve
    if hasattr(model, 'predict_proba'):
        roc_path = os.path.join(save_dir, f'{model_name}_roc_curve.png') if save_dir else None
        auc_score = plot_roc_curve(model, X_test, y_test, model_name, roc_path)
        
        # Precision-Recall Curve
        pr_path = os.path.join(save_dir, f'{model_name}_precision_recall.png') if save_dir else None
        avg_precision = plot_precision_recall_curve(model, X_test, y_test, model_name, pr_path)
    
    # Feature Importance
    if feature_names and hasattr(model, 'feature_importances_'):
        fi_path = os.path.join(save_dir, f'{model_name}_feature_importance.png') if save_dir else None
        feature_importance_df = plot_feature_importance(model, feature_names, model_name, 
                                                       top_n=20, save_path=fi_path)
    
    # Prepare results dictionary
    results = {
        'model_name': model_name,
        'metrics': metrics,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'predicted_probabilities': y_pred_proba,
        'classification_report': classification_report(y_test, y_pred, 
                                                     target_names=['No Default', 'Default'],
                                                     output_dict=True)
    }
    
    if feature_names and hasattr(model, 'feature_importances_'):
        results['feature_importance'] = feature_importance_df
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE FOR: {model_name}")
    print(f"{'='*60}")
    
    return results


def compare_models(results_list, save_path=None):
    """
    Compare multiple model evaluation results
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries from evaluate_model function
    save_path : str, optional
        Path to save the comparison plot
    """
    if len(results_list) < 2:
        print("Need at least 2 model results for comparison")
        return
    
    # Create comparison dataframe
    comparison_data = []
    for result in results_list:
        metrics = result['metrics'].copy()
        metrics['model'] = result['model_name']
        comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Plot comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    if 'auc_score' in comparison_df.columns and comparison_df['auc_score'].notna().any():
        metrics_to_plot.append('auc_score')
    
    fig, axes = plt.subplots(nrows=len(metrics_to_plot), ncols=1, figsize=(10, 5 * len(metrics_to_plot)))
    
    # Ensure axes is an array even for a single subplot
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x='model', y=metric, data=comparison_df, ax=axes[i], palette='viridis')
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].set_xlabel('Model')
        axes[i].tick_params(axis='x', rotation=45) # Rotate x-axis labels for better readability

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    # Example usage:
    # Dummy results from a hypothetical evaluate_model function
    result1 = {
        'model_name': 'Logistic Regression',
        'metrics': {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'specificity': 0.90,
            'auc_score': 0.88
        }
    }

    result2 = {
        'model_name': 'Random Forest',
        'metrics': {
            'accuracy': 0.88,
            'precision': 0.85,
            'recall': 0.82,
            'f1_score': 0.83,
            'specificity': 0.92,
            'auc_score': 0.91
        }
    }

    result3 = {
        'model_name': 'SVM',
        'metrics': {
            'accuracy': 0.83,
            'precision': 0.78,
            'recall': 0.70,
            'f1_score': 0.74,
            'specificity': 0.89,
            'auc_score': 0.85
        }
    }

    results = [result1, result2, result3]

    # Compare models and display the plot
    compare_models(results)

    # Compare models and save the plot
    # compare_models(results, save_path='model_comparison.png')

    # Example with only two models
    print("\nComparing only two models:")
    compare_models([result1, result2])

    # Example with missing auc_score in one model (should still plot other metrics)
    result4 = {
        'model_name': 'Decision Tree',
        'metrics': {
            'accuracy': 0.80,
            'precision': 0.70,
            'recall': 0.65,
            'f1_score': 0.67,
            'specificity': 0.80,
            # 'auc_score': None # Missing AUC score
        }
    }
    results_with_missing_auc = [result1, result4]
    print("\nComparing models with one missing AUC score:")
    compare_models(results_with_missing_auc)

# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
# from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
# import matplotlib.pyplot as plt

# def plot_confusion_matrix_updated(model, X_test, y_test, model_name):
#     """Plot confusion matrix using new sklearn method"""
#     y_pred = model.predict(X_test)
#     cm = confusion_matrix(y_test, y_pred)
    
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
#                                   display_labels=['No Default', 'Default'])
#     fig, ax = plt.subplots(figsize=(8, 6))
#     disp.plot(ax=ax, cmap='Blues', values_format='d')
#     plt.title(f'{model_name} - Confusion Matrix')
#     plt.show()

# def plot_precision_recall_curve_updated(model, X_test, y_test, model_name):
#     """Plot precision-recall curve using new sklearn method"""
#     disp = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
#     disp.ax_.set_title(f'{model_name} - Precision-Recall Curve')
#     plt.show()

# def evaluate_model(model, X_test, y_test, model_name):
#     """Comprehensive model evaluation with updated plotting"""
#     predictions = model.predict(X_test)
    
#     print(f"=== {model_name} Results ===")
#     print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
#     print("\nClassification Report:")
#     print(classification_report(y_test, predictions))
    
#     # Updated plotting functions
#     plot_confusion_matrix_updated(model, X_test, y_test, model_name)
#     plot_precision_recall_curve_updated(model, X_test, y_test, model_name)
    
#     return predictions

# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import plot_confusion_matrix, plot_precision_recall_curve

# # def evaluate_model(model, X_test, y_test, model_name):
# #     """Comprehensive model evaluation"""
# #     predictions = model.predict(X_test)
    
# #     print(f"=== {model_name} Results ===")
# #     print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
# #     print("\nClassification Report:")
# #     print(classification_report(y_test, predictions))
    
# #     # Confusion Matrix
# #     plot_confusion_matrix(model, X_test, y_test, cmap="Blues_r")
# #     plt.title(f"{model_name} - Confusion Matrix")
# #     plt.show()
    
# #     # Precision-Recall Curve
# #     plot_precision_recall_curve(model, X_test, y_test)
# #     plt.title(f"{model_name} - Precision-Recall Curve")
# #     plt.show()
    
# #     return predictions

# def compare_models(models_dict, X_test, y_test):
#     """Compare multiple models"""
#     results = {}
#     for name, model in models_dict.items():
#         predictions = model.predict(X_test)
#         accuracy = accuracy_score(y_test, predictions)
#         results[name] = accuracy
    
#     return results
