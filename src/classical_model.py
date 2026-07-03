"""
classical_model.py
====================
Classical Machine Learning baseline: a Random Forest classifier.

WHY RANDOM FOREST AS THE BASELINE?
-------------------------------------
- It is the single most widely reported classical baseline in IDS
  literature (robust to feature scaling choices, handles non-linear
  decision boundaries, gives feature importances, rarely overfits badly
  with default settings) -- making your QML comparison directly comparable
  to dozens of prior published results.
- It trains fast even on the full ~125k-row NSL-KDD training set, which
  lets us additionally report "best-case classical" performance (full
  feature set) alongside the "fair, qubit-matched" comparison (PCA-reduced
  feature set used identically by both models).

This module exposes a thin wrapper class so `main.py` and notebooks can
train/evaluate/save/load the model with a clean, consistent API.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.config import RF_PARAMS, MODELS_DIR
import os


class ClassicalIDS:
    """Random Forest based Intrusion Detection classifier."""

    def __init__(self, params: dict = None):
        self.params = params or RF_PARAMS
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive ('attack') class."""
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self):
        return self.model.feature_importances_

    def save(self, filename: str = "random_forest.joblib"):
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(self.model, path)
        print(f"[classical_model] Saved trained Random Forest to {path}")
        return path

    @classmethod
    def load(cls, filename: str = "random_forest.joblib"):
        path = os.path.join(MODELS_DIR, filename)
        instance = cls()
        instance.model = joblib.load(path)
        return instance


if __name__ == "__main__":
    # python -m src.classical_model
    from src.data_loader import load_raw_data
    from src.preprocessing import preprocess

    train_df, test_df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(train_df, test_df)

    clf = ClassicalIDS().fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"Random Forest test accuracy (full feature set): {acc:.4f}")
    clf.save()
