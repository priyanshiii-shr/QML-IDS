"""
data_loader.py
===============
Responsible ONLY for reading the raw NSL-KDD `.txt` files off disk and
returning clean pandas DataFrames with proper column names attached.

WHY SEPARATE FROM preprocessing.py?
------------------------------------
Separating "I/O and raw loading" from "transformation logic" is a standard
software-engineering practice that makes each module independently
testable: you can swap in a different data source (e.g. a different IoT
dataset) by only touching this file, without breaking preprocessing.

NSL-KDD FILE FORMAT
--------------------
Each row is a comma-separated line, no header, with 43 columns:
    41 features, 1 label (e.g. "normal", "neptune", "smurf", ...),
    1 "difficulty" score (an artifact of NSL-KDD's construction, not used
    for classification -- dropped during preprocessing).
"""

import os
import pandas as pd

from src.config import TRAIN_FILE, TEST_FILE, NSL_KDD_COLUMNS


def _load_file(path: str) -> pd.DataFrame:
    """Load a single NSL-KDD raw text file into a labeled DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'.\n"
            "Please download KDDTrain+.txt and KDDTest+.txt from "
            "https://www.unb.ca/cic/datasets/nsl.html and place them "
            "inside the 'data/raw/' folder."
        )
    df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    return df


def load_train_data() -> pd.DataFrame:
    """Load the raw NSL-KDD training set (KDDTrain+.txt)."""
    return _load_file(TRAIN_FILE)


def load_test_data() -> pd.DataFrame:
    """Load the raw NSL-KDD test set (KDDTest+.txt)."""
    return _load_file(TEST_FILE)


def load_raw_data():
    """Convenience function: load both train and test sets at once."""
    train_df = load_train_data()
    test_df = load_test_data()
    return train_df, test_df


if __name__ == "__main__":
    # Quick manual sanity check when running this file directly:
    #   python -m src.data_loader
    train_df, test_df = load_raw_data()
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nSample rows:\n", train_df.head())
    print("\nUnique labels in train set:\n", train_df["label"].unique())
