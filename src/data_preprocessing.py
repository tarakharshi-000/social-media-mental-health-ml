"""
Data Preprocessing Module for Social Media Mental Health Impact Classification

This module handles:
- Data loading from Excel file
- Platform column cleaning
- Missing value imputation
- Target variable creation
- Feature engineering
- Train/test splitting
- Categorical encoding
- Feature scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re

# Set random seed for reproducibility
RANDOM_SEED = 42


def load_data(file_path):
    """
    Load dataset from Excel file.

    Args:
        file_path (str): Path to Excel file

    Returns:
        pd.DataFrame: Raw dataset
    """
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def clean_platform(platform):
    """
    Clean and standardize platform names.

    Args:
        platform (str): Raw platform name

    Returns:
        str: Cleaned platform category
    """
    if pd.isna(platform):
        return 'Other'

    # Convert to lowercase and strip whitespace
    platform = str(platform).lower().strip()

    # Extract first platform if multiple mentioned
    if ' and ' in platform or ', ' in platform or '+' in platform:
        # Split by common separators and take first
        parts = re.split(r' and |, |\+|,|\s+', platform)
        platform = parts[0] if parts else platform

    # Direct mapping for exact matches
    platform_mapping = {
        'instagram': 'Instagram',
        'instagarm': 'Instagram',
        'insta': 'Instagram',
        'instgram': 'Instagram',
        'instagram+': 'Instagram',
        'instagram linkedin google': 'Instagram',
        'instagram and spotify': 'Instagram',
        'instagram, snapchat , clash of clans , gmail': 'Instagram',
        'instagram, whatsapp,youtube,netflix': 'Instagram',
        'instagram snapchat': 'Instagram',
        'whatsapp instagram': 'Instagram',
        'isntagram and snap': 'Instagram',
        'facebook': 'Facebook',
        'fb': 'Facebook',
        'youtube': 'YouTube',
        'yt': 'YouTube',
        'you tube': 'YouTube',
        'youtube.': 'YouTube',
        'whatsapp': 'WhatsApp',
        'wtsapp': 'WhatsApp',
        'watsapp': 'WhatsApp',
        'whatsapp and instagram': 'WhatsApp',
        'whatsapp, instagram': 'WhatsApp',
        'whatsapp, youtube': 'WhatsApp',
        'whatsapp, snapchat': 'WhatsApp',
        'whatsapp, snapchat': 'WhatsApp',
        'twitter': 'Twitter',
        'x': 'Twitter',
        'tiktok': 'TikTok',
        'snapchat': 'Snapchat',
        'linkedin': 'LinkedIn',
        'telegram': 'Telegram',
        'pinterest': 'Pinterest',
        'reddit': 'Reddit',
        'twitch': 'Twitch',
        'netflix': 'Streaming',
        'disney+': 'Streaming',
        'amazon prime video': 'Streaming',
        'spotify': 'Streaming',
        'bgmi': 'Gaming',
        'free fire': 'Gaming',
        'crunchyroll': 'Streaming',
        'bgmi, instagram': 'Gaming',
        'free fire, instagram, whatsapp, youtube': 'Gaming',
    }

    # Check for exact match first
    if platform in platform_mapping:
        return platform_mapping[platform]

    # Check for partial matches
    for key, value in platform_mapping.items():
        if key in platform:
            return value

    # Categorize remaining platforms
    social_media = ['instagram', 'facebook', 'twitter', 'tiktok', 'snapchat',
                   'linkedin', 'whatsapp', 'telegram', 'pinterest', 'reddit']

    if any(sm in platform for sm in social_media):
        # Capitalize first letter
        return platform.capitalize()
    elif any(stream in platform for stream in ['netflix', 'disney', 'prime', 'spotify', 'crunchyroll']):
        return 'Streaming'
    elif any(game in platform for game in ['bgmi', 'free fire', 'game']):
        return 'Gaming'
    else:
        return 'Other'


def preprocess_platform(df):
    """
    Clean the platform column in the dataset.

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        pd.DataFrame: Dataset with cleaned platform column
    """
    df_cleaned = df.copy()
    df_cleaned['platform_cleaned'] = df_cleaned['Which platform do you use the most daily?'].apply(clean_platform)

    # Print distribution after cleaning
    print("\nPlatform distribution after cleaning:")
    print(df_cleaned['platform_cleaned'].value_counts())

    return df_cleaned


def handle_missing_values(df):
    """
    Handle missing values by imputing with mode for categorical features.

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        pd.DataFrame: Dataset with imputed missing values
    """
    df_imputed = df.copy()

    # Columns with missing values
    categorical_cols = df_imputed.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0:
            mode_value = df_imputed[col].mode()[0]
            df_imputed[col] = df_imputed[col].fillna(mode_value)
            print(f"Imputed {df_imputed[col].isnull().sum()} missing values in {col} with mode: {mode_value}")

    return df_imputed


def create_target_variable(df):
    """
    Create binary target variable for mental health impact.

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        pd.DataFrame: Dataset with target variable
    """
    df_target = df.copy()

    # Create binary target: 1 if Yes, 0 if No or Maybe
    df_target['mental_health_impact'] = (
        df_target['Do you feel social media affects your mental health?'].apply(
            lambda x: 1 if x == 'Yes' else 0
        )
    )

    print("\nTarget variable distribution:")
    print(df_target['mental_health_impact'].value_counts())
    print(f"Positive class (Yes): {df_target['mental_health_impact'].sum()} ({df_target['mental_health_impact'].mean()*100:.1f}%)")
    print(f"Negative class (No+Maybe): {(1 - df_target['mental_health_impact']).sum()} ({(1 - df_target['mental_health_impact']).mean()*100:.1f}%)")

    return df_target


def select_features(df):
    """
    Select features for modeling and exclude target leakage variables.

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        list: List of feature column names
    """
    # Exclude target leakage variable
    exclude_cols = [
        'Would you ever delete social media permanently?',
        'Do you feel social media affects your mental health?',  # Original target column
        'mental_health_impact',  # New target column
        'Which platform do you use the most daily?',  # Original platform column
    ]

    # Use cleaned platform instead
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\nSelected features ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")

    return feature_cols


def split_data(df, feature_cols, target_col='mental_health_impact', test_size=0.2, random_state=RANDOM_SEED):
    """
    Split data into train and test sets with stratification.

    Args:
        df (pd.DataFrame): Input dataset
        feature_cols (list): List of feature column names
        target_col (str): Target column name
        test_size (float): Proportion of data for test set
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain/test split:")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"\nTrain target distribution:")
    print(y_train.value_counts())
    print(f"\nTest target distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


def create_preprocessor(X_train):
    """
    Create preprocessing pipeline for categorical and numerical features.

    Args:
        X_train (pd.DataFrame): Training data

    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    print(f"\nPreprocessing:")
    print(f"  Numerical columns: {numerical_cols}")
    print(f"  Categorical columns: {categorical_cols}")

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    return preprocessor


def apply_preprocessing(preprocessor, X_train, X_test):
    """
    Apply preprocessing pipeline to train and test data.

    Args:
        preprocessor (ColumnTransformer): Preprocessing pipeline
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Test data

    Returns:
        tuple: X_train_processed, X_test_processed, feature_names
    """
    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after encoding
    feature_names = preprocessor.get_feature_names_out()

    print(f"\nProcessed data shape:")
    print(f"  Train: {X_train_processed.shape}")
    print(f"  Test: {X_test_processed.shape}")
    print(f"  Features: {len(feature_names)}")

    return X_train_processed, X_test_processed, feature_names


def full_preprocessing_pipeline(file_path):
    """
    Execute complete preprocessing pipeline.

    Args:
        file_path (str): Path to Excel file

    Returns:
        dict: Dictionary containing processed data and metadata
    """
    print("=" * 80)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    df = load_data(file_path)

    # Step 2: Clean platform
    print("\n[Step 2] Cleaning platform column...")
    df = preprocess_platform(df)

    # Step 3: Handle missing values
    print("\n[Step 3] Handling missing values...")
    df = handle_missing_values(df)

    # Step 4: Create target variable
    print("\n[Step 4] Creating target variable...")
    df = create_target_variable(df)

    # Step 5: Select features
    print("\n[Step 5] Selecting features...")
    feature_cols = select_features(df)

    # Step 6: Split data
    print("\n[Step 6] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, feature_cols)

    # Step 7: Create preprocessor
    print("\n[Step 7] Creating preprocessor...")
    preprocessor = create_preprocessor(X_train)

    # Step 8: Apply preprocessing
    print("\n[Step 8] Applying preprocessing...")
    X_train_processed, X_test_processed, feature_names = apply_preprocessing(preprocessor, X_train, X_test)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'preprocessor': preprocessor,
        'feature_cols': feature_cols,
        'df_processed': df
    }


if __name__ == "__main__":
    # Test the preprocessing pipeline
    file_path = r'D:\social_media_survey\Fai Datasets Rishu.xlsx'
    processed_data = full_preprocessing_pipeline(file_path)

    print("\nPreprocessing pipeline test completed successfully!")
