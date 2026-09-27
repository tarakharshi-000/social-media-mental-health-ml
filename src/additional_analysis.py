"""
Additional Analysis Module

This module handles:
- Feature importance analysis
- Permutation importance
- Correlation analysis
- Additional insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import os

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def extract_feature_importance(model, feature_names, model_name):
    """
    Extract feature importance from model.

    Args:
        model: Trained model
        feature_names (list): List of feature names
        model_name (str): Name of the model

    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return importance_df
    else:
        print(f"Model {model_name} does not have feature_importances_ attribute")
        return None


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
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, top_features['importance'].values)):
        ax.text(imp + 0.001, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot for {model_name}")


def calculate_permutation_importance(model, X_test, y_test, feature_names, model_name, n_repeats=10, random_state=42):
    """
    Calculate permutation importance.

    Args:
        model: Trained model
        X_test (np.array): Test features
        y_test (np.array): Test labels
        feature_names (list): List of feature names
        model_name (str): Name of the model
        n_repeats (int): Number of permutation repeats
        random_state (int): Random seed

    Returns:
        pd.DataFrame: Permutation importance dataframe
    """
    try:
        perm_importance = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)

        return importance_df
    except Exception as e:
        print(f"Error calculating permutation importance for {model_name}: {e}")
        return None


def plot_permutation_importance(importance_df, model_name, save_dir, top_n=15):
    """
    Plot permutation importance with error bars.

    Args:
        importance_df (pd.DataFrame): Permutation importance dataframe
        model_name (str): Name of the model
        save_dir (str): Directory to save plot
        top_n (int): Number of top features to show
    """
    if importance_df is None or len(importance_df) == 0:
        print(f"No permutation importance available for {model_name}")
        return

    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                   xerr=top_features['std'].values, capsize=5, color='coral', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=10)
    ax.set_xlabel('Permutation Importance (decrease in accuracy)', fontsize=12)
    ax.set_title(f'Top {top_n} Permutation Importance - {model_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'permutation_importance_{model_name.replace(" ", "_").lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved permutation importance plot for {model_name}")


def analyze_feature_correlation(X_train, feature_names, save_dir):
    """
    Analyze feature correlations.

    Args:
        X_train (np.array): Training features
        feature_names (list): List of feature names
        save_dir (str): Directory to save plot
    """
    # Create dataframe
    df = pd.DataFrame(X_train, columns=feature_names)

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved feature correlation matrix")

    # Find highly correlated features
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values('correlation', ascending=False)
        high_corr_df.to_csv(os.path.join(save_dir, 'high_correlations.csv'), index=False)
        print(f"Found {len(high_corr)} highly correlated feature pairs")
        print(high_corr_df.head(10))
    else:
        print("No highly correlated features found (threshold > 0.7)")


def compare_with_paper_findings(importance_df, model_name):
    """
    Compare our feature importance with paper findings.

    Args:
        importance_df (pd.DataFrame): Feature importance dataframe
        model_name (str): Name of the model
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON WITH PAPER FINDINGS - {model_name}")
    print(f"{'='*80}")

    # Paper findings: duration, age, platform most important
    paper_features = ['duration', 'age', 'platform']

    print("\nPaper reported key features:")
    print("  1. Duration of social media use")
    print("  2. Age")
    print("  3. Platform")

    print(f"\nOur top 10 features for {model_name}:")
    print(importance_df.head(10).to_string(index=False))

    # Check if paper features are in our top features
    top_10 = importance_df.head(10)['feature'].tolist()

    duration_features = [f for f in top_10 if 'time' in f.lower() or 'hour' in f.lower() or 'duration' in f.lower()]
    age_features = [f for f in top_10 if 'age' in f.lower()]
    platform_features = [f for f in top_10 if 'platform' in f.lower()]

    print(f"\nAlignment with paper findings:")
    print(f"  Duration-related features in top 10: {len(duration_features)}")
    if duration_features:
        print(f"    {duration_features}")
    print(f"  Age-related features in top 10: {len(age_features)}")
    if age_features:
        print(f"    {age_features}")
    print(f"  Platform-related features in top 10: {len(platform_features)}")
    if platform_features:
        print(f"    {platform_features}")

    alignment_score = len(duration_features) + len(age_features) + len(platform_features)
    print(f"\nAlignment score (out of 3): {alignment_score}/3")


def perform_additional_analysis(models, X_train, X_test, y_train, y_test, feature_names, save_dir):
    """
    Perform comprehensive additional analysis.

    Args:
        models (dict): Dictionary of trained models
        X_train (np.array): Training features
        X_test (np.array): Test features
        y_train (np.array): Training labels
        y_test (np.array): Test labels
        feature_names (list): List of feature names
        save_dir (str): Directory to save results
    """
    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSIS")
    print("=" * 80)

    analysis_dir = os.path.join(save_dir, 'additional_analysis')
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # Feature correlation analysis
    print("\n[1/4] Analyzing feature correlations...")
    analyze_feature_correlation(X_train, feature_names, analysis_dir)

    # Feature importance for tree-based models
    model_names = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest'
    }

    for key, name in model_names.items():
        if key in models:
            print(f"\n[2/4] Extracting feature importance for {name}...")
            importance_df = extract_feature_importance(models[key], feature_names, name)

            if importance_df is not None:
                # Save to CSV
                importance_df.to_csv(os.path.join(analysis_dir, f'feature_importance_{name.replace(" ", "_").lower()}.csv'), index=False)
                print(f"  Saved feature importance to CSV")

                # Plot
                plot_feature_importance(importance_df, name, analysis_dir)

                # Compare with paper
                compare_with_paper_findings(importance_df, name)

    # Permutation importance for best model
    print(f"\n[3/4] Calculating permutation importance for Random Forest (best model)...")
    if 'random_forest' in models:
        perm_importance = calculate_permutation_importance(
            models['random_forest'], X_test, y_test, feature_names, 'Random Forest'
        )

        if perm_importance is not None:
            perm_importance.to_csv(os.path.join(analysis_dir, 'permutation_importance_random_forest.csv'), index=False)
            print("  Saved permutation importance to CSV")
            plot_permutation_importance(perm_importance, 'Random Forest', analysis_dir)

    print("\n[4/4] Additional analysis complete")

    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Analysis directory: {analysis_dir}")


if __name__ == "__main__":
    # Test the additional analysis module
    from train import train_all_models
    from data_preprocessing import full_preprocessing_pipeline

    # Load and preprocess data
    file_path = r'D:\social_media_survey\Fai Datasets Rishu.xlsx'
    processed_data = full_preprocessing_pipeline(file_path)

    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    feature_names = processed_data['feature_names']

    # Train models
    models = train_all_models(X_train, y_train)

    # Perform additional analysis
    perform_additional_analysis(models, X_train, X_test, y_train, y_test, feature_names, 'results')

    print("\nAdditional analysis module test completed successfully!")
