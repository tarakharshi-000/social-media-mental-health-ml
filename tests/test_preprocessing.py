"""
Unit tests for data preprocessing module
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.data_preprocessing import clean_platform, handle_missing_values, create_target_variable


def test_clean_platform():
    """Test platform cleaning function"""
    print("Testing platform cleaning...")

    # Test standard platforms
    assert clean_platform('Instagram') == 'Instagram'
    assert clean_platform('instagram') == 'Instagram'
    assert clean_platform('INSTAGRAM') == 'Instagram'
    assert clean_platform('Facebook') == 'Facebook'

    # Test misspellings
    assert clean_platform('instagarm') == 'Instagram'
    assert clean_platform('insta') == 'Instagram'

    # Test streaming
    assert clean_platform('Netflix') == 'Streaming'
    assert clean_platform('Disney+') == 'Streaming'

    # Test gaming
    assert clean_platform('BGMI') == 'Gaming'

    # Test missing
    assert clean_platform(None) == 'Other'
    assert clean_platform(np.nan) == 'Other'

    # Test multi-platform
    assert clean_platform('Instagram and Spotify') == 'Instagram'

    print("[OK] Platform cleaning tests passed")


def test_handle_missing_values():
    """Test missing value handling"""
    print("Testing missing value handling...")

    # Create test dataframe with missing values
    df = pd.DataFrame({
        'col1': ['A', 'B', None, 'A'],
        'col2': ['X', None, 'Z', 'X'],
        'col3': [1, 2, 3, 4]  # No missing values
    })

    df_imputed = handle_missing_values(df)

    # Check no missing values remain
    assert df_imputed.isnull().sum().sum() == 0

    # Check imputation with mode
    assert df_imputed['col1'].iloc[2] == 'A'  # Mode of col1
    assert df_imputed['col2'].iloc[1] == 'X'  # Mode of col2

    print("[OK] Missing value handling tests passed")


def test_create_target_variable():
    """Test target variable creation"""
    print("Testing target variable creation...")

    # Create test dataframe
    df = pd.DataFrame({
        'Do you feel social media affects your mental health?': ['Yes', 'No', 'Maybe', 'Yes', 'No']
    })

    df_target = create_target_variable(df)

    # Check target column exists
    assert 'mental_health_impact' in df_target.columns

    # Check binary transformation
    assert df_target['mental_health_impact'].iloc[0] == 1  # Yes
    assert df_target['mental_health_impact'].iloc[1] == 0  # No
    assert df_target['mental_health_impact'].iloc[2] == 0  # Maybe
    assert df_target['mental_health_impact'].iloc[3] == 1  # Yes
    assert df_target['mental_health_impact'].iloc[4] == 0  # No

    # Check distribution
    assert df_target['mental_health_impact'].sum() == 2  # 2 Yes values
    assert (df_target['mental_health_impact'] == 0).sum() == 3  # 3 No+Maybe values

    print("[OK] Target variable creation tests passed")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING PREPROCESSING TESTS")
    print("=" * 80)
    print()

    test_clean_platform()
    test_handle_missing_values()
    test_create_target_variable()

    print()
    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
