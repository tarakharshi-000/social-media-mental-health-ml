"""
Utility functions for the project
"""

import os
import json
import pandas as pd
import numpy as np


def ensure_directory(directory):
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_json(data, filepath):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath (str): File path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Load data from JSON file.

    Args:
        filepath (str): File path

    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def print_section(title):
    """
    Print a formatted section header.

    Args:
        title (str): Section title
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title):
    """
    Print a formatted subsection header.

    Args:
        title (str): Subsection title
    """
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)


def get_project_root():
    """
    Get the project root directory.

    Returns:
        str: Project root directory
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
