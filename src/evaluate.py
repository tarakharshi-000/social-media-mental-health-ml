"""
Model Evaluation and Results Generation Module

This module handles:
- Comprehensive model evaluation
- Results visualization
- Confusion matrix generation
- Feature importance analysis
- Comparison with paper results
- Results export to CSV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_results_table(results, cv_results):
    """
    Create comprehensive results table comparing all models.

    Args:
        results (dict): Test set evaluation results
        cv_results (dict): Cross-validation results

    Returns:
        pd.DataFrame: Results table
    """
    data = []

    model_order = ['xgboost', 'random_forest', 'naive_bayes', 'baseline']
    model_names = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest',
        'naive_bayes': 'Naïve Bayes',
        'baseline': 'Baseline (Majority Class)'
    }

    for key in model_order:
        if key in results:
            row = {
                'Model': model_names[key],
                'Test Accuracy': results[key]['accuracy'],
                'Test Precision': results[key]['precision'],
                'Test Recall': results[key]['recall'],
                'Test F1-Score': results[key]['f1_score'],
                'Test ROC-AUC': results[key]['roc_auc'],
                'CV Mean Accuracy': cv_results[key]['cv_mean'],
                'CV Std Accuracy': cv_results[key]['cv_std']
            }
            data.append(row)

    df = pd.DataFrame(data)
    return df


def plot_confusion_matrix(cm, model_name, save_dir):
    """
    Plot and save confusion matrix.

    Args:
        cm (np.array): Confusion matrix
        model_name (str): Name of the model
        save_dir (str): Directory to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
    plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix for {model_name}")


def plot_model_comparison(results_df, save_dir):
    """
    Plot model comparison bar chart.

    Args:
        results_df (pd.DataFrame): Results dataframe
        save_dir (str): Directory to save plot
    """
    metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score']
    x = np.arange(len(results_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, results_df[metric], width, label=metric)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved model comparison plot")


def plot_cv_comparison(cv_results, save_dir):
    """
    Plot cross-validation comparison.

    Args:
        cv_results (dict): Cross-validation results
        save_dir (str): Directory to save plot
    """
    model_names = []
    cv_means = []
    cv_stds = []

    model_order = ['xgboost', 'random_forest', 'naive_bayes', 'baseline']
    name_mapping = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest',
        'naive_bayes': 'Naïve Bayes',
        'baseline': 'Baseline'
    }

    for key in model_order:
        if key in cv_results:
            model_names.append(name_mapping[key])
            cv_means.append(cv_results[key]['cv_mean'])
            cv_stds.append(cv_results[key]['cv_std'])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_ylabel('CV Accuracy')
    ax.set_title('Cross-Validation Accuracy Comparison')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean in zip(bars, cv_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}',
                ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved CV comparison plot")


def plot_feature_importance(importance_df, model_name, save_dir, top_n=15):
    """
    Plot feature importance.

    Args:
        importance_df (pd.DataFrame): Feature importance dataframe
        model_name (str): Name of the model
        save_dir (str): Directory to save plot
        top_n (int): Number of top features to show
    """
    if importance_df is None or len(importance_df) == 0:
        print(f"No feature importance available for {model_name}")
        return

    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance for {model_name}")


def create_paper_comparison(results_df, save_dir):
    """
    Create comparison table with paper results.

    Args:
        results_df (pd.DataFrame): Our results
        save_dir (str): Directory to save comparison
    """
    # Paper results (from the paper we're comparing against)
    paper_results = {
        'XGBoost': {'accuracy': 0.90, 'precision': None, 'recall': None, 'f1_score': None},
        'Random Forest': {'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None},
        'Naïve Bayes': {'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None},
    }

    comparison_data = []

    for _, row in results_df.iterrows():
        model_name = row['Model']
        if model_name in paper_results:
            paper = paper_results[model_name]
            comparison_row = {
                'Model': model_name,
                'Paper Accuracy': paper['accuracy'],
                'Our Accuracy': row['Test Accuracy'],
                'Difference': row['Test Accuracy'] - paper['accuracy'] if paper['accuracy'] is not None else 'N/A',
                'Paper Precision': paper['precision'],
                'Our Precision': row['Test Precision'],
                'Paper Recall': paper['recall'],
                'Our Recall': row['Test Recall'],
                'Paper F1-Score': paper['f1_score'],
                'Our F1-Score': row['Test F1-Score']
            }
            comparison_data.append(comparison_row)

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison table
    comparison_df.to_csv(os.path.join(save_dir, 'paper_comparison.csv'), index=False)
    print("Saved paper comparison table")

    return comparison_df


def generate_evaluation_report(results, cv_results, save_dir):
    """
    Generate comprehensive evaluation report.

    Args:
        results (dict): Test set evaluation results
        cv_results (dict): Cross-validation results
        save_dir (str): Directory to save report
    """
    report = []
    report.append("=" * 80)
    report.append("MODEL EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    model_order = ['xgboost', 'random_forest', 'naive_bayes', 'baseline']
    model_names = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest',
        'naive_bayes': 'Naïve Bayes',
        'baseline': 'Baseline (Majority Class)'
    }

    for key in model_order:
        if key in results:
            report.append(f"\n{model_names[key]}")
            report.append("-" * 80)
            report.append(f"Test Accuracy: {results[key]['accuracy']:.4f}")
            report.append(f"Test Precision: {results[key]['precision']:.4f}")
            report.append(f"Test Recall: {results[key]['recall']:.4f}")
            report.append(f"Test F1-Score: {results[key]['f1_score']:.4f}")
            if results[key]['roc_auc'] is not None:
                report.append(f"Test ROC-AUC: {results[key]['roc_auc']:.4f}")
            report.append(f"CV Mean Accuracy: {cv_results[key]['cv_mean']:.4f} (+/- {cv_results[key]['cv_std']:.4f})")
            report.append("")

    report.append("=" * 80)
    report.append("PAPER COMPARISON")
    report.append("=" * 80)
    report.append("")
    report.append("Paper: 'Detecting the Impact of Social Media on Users' Mental Health")
    report.append("       Using Machine Learning and XAI'")
    report.append("Author: Ara Bela Zulfa Laila (2026)")
    report.append("")
    report.append("Paper Results:")
    report.append("  XGBoost: 90% accuracy (best performing model)")
    report.append("  Random Forest: Not explicitly stated")
    report.append("  Naïve Bayes: Not explicitly stated")
    report.append("")
    report.append("Our Results:")
    for key in model_order:
        if key in results and key != 'baseline':
            report.append(f"  {model_names[key]}: {results[key]['accuracy']:.2%} accuracy")
    report.append("")
    report.append("Key Differences:")
    report.append("  1. Different target variable: Mental health perception vs clinical depression")
    report.append("  2. Missing features: No gender or relationship status in our dataset")
    report.append("  3. Platform data quality: Our platform column required significant cleaning")
    report.append("  4. Sample size: 1,195 responses vs unknown paper sample size")
    report.append("  5. Cultural/geographic context: Unknown in paper vs our dataset")
    report.append("  6. Survey design: Different question wording and scales")
    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(report_text)

    print("Saved evaluation report")

    return report_text


def save_all_results(results, cv_results, save_dir='results'):
    """
    Save all results to files.

    Args:
        results (dict): Test set evaluation results
        cv_results (dict): Cross-validation results
        save_dir (str): Directory to save results
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    figures_dir = os.path.join(save_dir, 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Create results table
    results_df = create_results_table(results, cv_results)
    results_df.to_csv(os.path.join(save_dir, 'model_results.csv'), index=False)
    print("Saved model results to CSV")

    # Plot confusion matrices
    model_names = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest',
        'naive_bayes': 'Naïve Bayes',
        'baseline': 'Baseline'
    }

    for key, name in model_names.items():
        if key in results:
            plot_confusion_matrix(results[key]['confusion_matrix'], name, figures_dir)

    # Plot model comparison
    plot_model_comparison(results_df, figures_dir)

    # Plot CV comparison
    plot_cv_comparison(cv_results, figures_dir)

    # Create paper comparison
    paper_comparison_df = create_paper_comparison(results_df, save_dir)

    # Generate evaluation report
    evaluation_report = generate_evaluation_report(results, cv_results, save_dir)

    print("\n" + "=" * 80)
    print("RESULTS SAVED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results directory: {save_dir}")
    print(f"Figures directory: {figures_dir}")

    return results_df, paper_comparison_df


if __name__ == "__main__":
    # Test the evaluation module
    from train import train_all_models, evaluate_all_models, perform_cv_all_models
    from data_preprocessing import full_preprocessing_pipeline

    # Load and preprocess data
    file_path = r'D:\social_media_survey\Fai Datasets Rishu.xlsx'
    processed_data = full_preprocessing_pipeline(file_path)

    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']

    # Train models
    models = train_all_models(X_train, y_train)

    # Evaluate models
    results = evaluate_all_models(models, X_test, y_test)

    # Cross-validation
    cv_results = perform_cv_all_models(models, X_train, y_train)

    # Save results
    results_df, paper_comparison_df = save_all_results(results, cv_results)

    print("\nEvaluation module test completed successfully!")
