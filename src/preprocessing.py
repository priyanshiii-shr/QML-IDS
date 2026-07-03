"""
preprocessing.py
==================
Transforms raw NSL-KDD DataFrames into model-ready numeric arrays.

STEPS PERFORMED (in order)
---------------------------
1. Drop the `difficulty` column (not a feature; an NSL-KDD construction
   artifact unrelated to network traffic).
2. Binarize the label: every row labeled "normal" -> 0, every attack
   subtype (neptune, smurf, satan, guess_passwd, ...) -> 1. This converts
   the original 5-class problem into the standard binary IDS formulation.
3. One-hot encode the three categorical columns: `protocol_type`,
   `service`, `flag`. Train and test sets are encoded TOGETHER (after
   concatenation) so that both end up with identical columns even if the
   test set contains service/flag categories absent from the training set
   (this is a well-known NSL-KDD quirk and a common bug source if ignored).
4. Scale all numeric (now all-numeric, since categoricals are one-hot)
   features with `StandardScaler`, fit ONLY on the training set and then
   applied to the test set (avoiding data leakage).
5. Split into X_train, X_test, y_train, y_test numpy arrays.

WHY FIT THE SCALER/ENCODER ONLY ON TRAIN?
-------------------------------------------
Fitting on test data (or train+test combined) leaks information about the
test distribution into training, inflating reported performance -- a
classic methodological error reviewers will flag in a research paper.
One-hot encoding is the one exception where we look at both sets' column
*categories* (not their statistics) to guarantee a consistent feature
schema; this is standard practice and does not leak label information.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import CATEGORICAL_COLUMNS, TARGET_COLUMN


def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map the multi-class 'label' column to binary: 0 = normal, 1 = attack."""
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
        lambda x: 0 if x.strip() == "normal" else 1
    )
    return df


def _drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])
    return df


def one_hot_encode_combined(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    One-hot encode categorical columns using a SHARED schema across train
    and test, so unseen categories in the test set don't crash the
    pipeline and both sets end up with identical column ordering.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["__split__"] = "train"
    test_df["__split__"] = "test"

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLUMNS)

    train_encoded = combined[combined["__split__"] == "train"].drop(
        columns=["__split__"]
    ).reset_index(drop=True)
    test_encoded = combined[combined["__split__"] == "test"].drop(
        columns=["__split__"]
    ).reset_index(drop=True)

    return train_encoded, test_encoded


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standardize features: zero mean, unit variance. Fit on train only."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Full preprocessing pipeline. Returns numpy arrays ready for modeling.

    Returns
    -------
    X_train, X_test : np.ndarray (scaled, numeric, one-hot encoded features)
    y_train, y_test  : np.ndarray (binary labels: 0=normal, 1=attack)
    scaler            : fitted StandardScaler (saved for future inference)
    """
    train_df = _drop_irrelevant_columns(train_df)
    test_df = _drop_irrelevant_columns(test_df)

    train_df = binarize_labels(train_df)
    test_df = binarize_labels(test_df)

    train_encoded, test_encoded = one_hot_encode_combined(train_df, test_df)

    y_train = train_encoded[TARGET_COLUMN].values.astype(np.int64)
    y_test = test_encoded[TARGET_COLUMN].values.astype(np.int64)

    X_train_df = train_encoded.drop(columns=[TARGET_COLUMN])
    X_test_df = test_encoded.drop(columns=[TARGET_COLUMN])

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_df, X_test_df)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train_df.columns.tolist()


if __name__ == "__main__":
    # python -m src.preprocessing
    from src.data_loader import load_raw_data

    train_df, test_df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(
        train_df, test_df
    )
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Train label distribution:", np.bincount(y_train))
    print("Test label distribution:", np.bincount(y_test))
    print("Number of engineered features:", len(feature_names))
