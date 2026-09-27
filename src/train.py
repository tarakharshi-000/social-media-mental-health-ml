"""
Model Training Module for Social Media Mental Health Impact Classification

This module implements:
- XGBoost classifier (primary model - matches paper)
- Random Forest classifier (comparison - matches paper)
- Naïve Bayes classifier (comparison - matches paper)
- Baseline majority class classifier
- Cross-validation
- Model comparison
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
import os

# Set random seed for reproducibility
RANDOM_SEED = 42


def train_xgboost(X_train, y_train):
    """
    Train XGBoost classifier with class weighting for imbalance.

    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels

    Returns:
        xgb.XGBClassifier: Trained XGBoost model
    """
    # Calculate class weight for imbalance
    class_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        eval_metric='logloss',
        scale_pos_weight=class_weight
    )

    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier with class weighting.

    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels

    Returns:
        RandomForestClassifier: Trained Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    """
    Train Gaussian Naïve Bayes classifier.

    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels

    Returns:
        GaussianNB: Trained Naïve Bayes model
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def train_baseline(X_train, y_train):
    """
    Train baseline majority class classifier.

    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels

    Returns:
        DummyClassifier: Trained baseline model
    """
    model = DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance using multiple metrics.

    Args:
        model: Trained model
        X_test (np.array): Test features
        y_test (np.array): Test labels
        model_name (str): Name of the model

    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Probability predictions (for models that support it)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        y_pred_proba = None
        has_proba = False

    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
    }

    # Add ROC-AUC if probability predictions available
    if has_proba:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    else:
        metrics['roc_auc'] = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics['classification_report'] = report

    return metrics


def perform_cross_validation(model, X_train, y_train, model_name, cv=5):
    """
    Perform stratified k-fold cross-validation.

    Args:
        model: Model to evaluate
        X_train (np.array): Training features
        y_train (np.array): Training labels
        model_name (str): Name of the model
        cv (int): Number of folds

    Returns:
        dict: Cross-validation results
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')

    results = {
        'model': model_name,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }

    return results


def train_all_models(X_train, y_train):
    """
    Train all models (XGBoost, Random Forest, Naïve Bayes, Baseline).

    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels

    Returns:
        dict: Dictionary of trained models
    """
    print("=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)

    models = {}

    # Train XGBoost
    print("\n[1/4] Training XGBoost...")
    models['xgboost'] = train_xgboost(X_train, y_train)
    print("  XGBoost trained successfully")

    # Train Random Forest
    print("\n[2/4] Training Random Forest...")
    models['random_forest'] = train_random_forest(X_train, y_train)
    print("  Random Forest trained successfully")

    # Train Naïve Bayes
    print("\n[3/4] Training Naïve Bayes...")
    models['naive_bayes'] = train_naive_bayes(X_train, y_train)
    print("  Naïve Bayes trained successfully")

    # Train Baseline
    print("\n[4/4] Training Baseline (Majority Class)...")
    models['baseline'] = train_baseline(X_train, y_train)
    print("  Baseline trained successfully")

    print("\n" + "=" * 80)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 80)

    return models


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models on test set.

    Args:
        models (dict): Dictionary of trained models
        X_test (np.array): Test features
        y_test (np.array): Test labels

    Returns:
        dict: Dictionary of evaluation results
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    results = {}

    model_names = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest',
        'naive_bayes': 'Naïve Bayes',
        'baseline': 'Baseline (Majority Class)'
    }

    for key, model in models.items():
        print(f"\nEvaluating {model_names[key]}...")
        metrics = evaluate_model(model, X_test, y_test, model_names[key])
        results[key] = metrics

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return results


def perform_cv_all_models(models, X_train, y_train):
    """
    Perform cross-validation on all models.

    Args:
        models (dict): Dictionary of trained models
        X_train (np.array): Training features
        y_train (np.array): Training labels

    Returns:
        dict: Dictionary of cross-validation results
    """
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION")
    print("=" * 80)

    cv_results = {}

    model_names = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest',
        'naive_bayes': 'Naïve Bayes',
        'baseline': 'Baseline (Majority Class)'
    }

    for key, model in models.items():
        print(f"\nCross-validating {model_names[key]}...")
        cv_result = perform_cross_validation(model, X_train, y_train, model_names[key])
        cv_results[key] = cv_result

        print(f"  CV Mean Accuracy: {cv_result['cv_mean']:.4f} (+/- {cv_result['cv_std']:.4f})")

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80)

    return cv_results


def save_models(models, save_dir='models'):
    """
    Save trained models to disk.

    Args:
        models (dict): Dictionary of trained models
        save_dir (str): Directory to save models
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key, model in models.items():
        model_path = os.path.join(save_dir, f'{key}.pkl')
        joblib.dump(model, model_path)
        print(f"Saved {key} model to {model_path}")


def load_models(save_dir='models'):
    """
    Load trained models from disk.

    Args:
        save_dir (str): Directory to load models from

    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    model_keys = ['xgboost', 'random_forest', 'naive_bayes', 'baseline']

    for key in model_keys:
        model_path = os.path.join(save_dir, f'{key}.pkl')
        if os.path.exists(model_path):
            models[key] = joblib.load(model_path)
            print(f"Loaded {key} model from {model_path}")

    return models


def get_feature_importance(model, feature_names, model_name):
    """
    Extract feature importance from model.

    Args:
        model: Trained model
        feature_names (list): List of feature names
        model_name (str): Name of the model

    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if model_name == 'XGBoost':
        importance = model.feature_importances_
    elif model_name == 'Random Forest':
        importance = model.feature_importances_
    else:
        return None

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df


if __name__ == "__main__":
    # Test the training module
    from data_preprocessing import full_preprocessing_pipeline

    # Load and preprocess data
    file_path = r'D:\social_media_survey\Fai Datasets Rishu.xlsx'
    processed_data = full_preprocessing_pipeline(file_path)

    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']

    # Train all models
    models = train_all_models(X_train, y_train)

    # Evaluate all models
    results = evaluate_all_models(models, X_test, y_test)

    # Perform cross-validation
    cv_results = perform_cv_all_models(models, X_train, y_train)

    # Save models
    save_models(models)

    print("\nTraining module test completed successfully!")
