"""
feature_selection.py
======================
Reduces the high-dimensional one-hot-encoded NSL-KDD feature space (~120+
columns) down to `N_QUBITS` dimensions using PCA, AND rescales those PCA
components into the [0, pi] range needed for quantum angle encoding.

WHY PCA SPECIFICALLY (vs. plain feature selection)?
------------------------------------------------------
- PCA is unsupervised, deterministic, and well-understood -- a defensible,
  standard choice that any reviewer will accept without much pushback.
- It captures the directions of maximum variance, which tends to retain the
  most discriminative signal even after compressing to very few dimensions.
- It produces *exactly* N_QUBITS continuous components, perfectly matching
  one-feature-per-qubit angle encoding -- no need for manual feature
  ranking/selection heuristics.

This module is used to build the REDUCED feature set fed to BOTH the
quantum model (mandatory, due to qubit constraints) and a same-size
Random Forest variant (for a fair comparison under identical features),
while the main classical Random Forest can still optionally also be
trained on the FULL feature set as a secondary "best classical effort"
benchmark, per config.USE_FULL_DATA_FOR_RF.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.config import N_QUBITS


def reduce_dimensionality(X_train: np.ndarray, X_test: np.ndarray, n_components: int = N_QUBITS):
    """
    Fit PCA on the (already standardized) training features and apply it
    to both train and test sets. Returns components plus the fitted PCA
    object (useful for reporting explained variance in a paper).
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def scale_to_angle_range(X_train_pca: np.ndarray, X_test_pca: np.ndarray):
    """
    Rescale PCA output (which can be any real number) into [0, pi], the
    natural range for rotation-angle quantum encoding (RY gates rotate
    a qubit's Bloch-sphere state; angles outside [0, 2*pi] simply wrap
    around, so constraining to [0, pi] keeps the mapping well-behaved and
    avoids saturating/wrapping artifacts). Fit the scaler on train only.
    """
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_angles = angle_scaler.fit_transform(X_train_pca)
    X_test_angles = angle_scaler.transform(X_test_pca)
    return X_train_angles, X_test_angles, angle_scaler


def prepare_quantum_features(X_train: np.ndarray, X_test: np.ndarray, n_components: int = N_QUBITS):
    """
    Full pipeline: standardized features -> PCA -> angle-range scaling.
    Returns angle-encoded train/test arrays plus both fitted transformers
    so they can be reused at inference time on new data.
    """
    X_train_pca, X_test_pca, pca = reduce_dimensionality(X_train, X_test, n_components)
    X_train_angles, X_test_angles, angle_scaler = scale_to_angle_range(X_train_pca, X_test_pca)

    explained_variance_ratio = pca.explained_variance_ratio_
    total_variance_retained = float(np.sum(explained_variance_ratio))

    print(
        f"[feature_selection] PCA reduced features to {n_components} "
        f"components, retaining {total_variance_retained:.2%} of variance."
    )

    return X_train_angles, X_test_angles, pca, angle_scaler


if __name__ == "__main__":
    # python -m src.feature_selection
    from src.data_loader import load_raw_data
    from src.preprocessing import preprocess

    train_df, test_df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(train_df, test_df)
    X_train_q, X_test_q, pca, angle_scaler = prepare_quantum_features(X_train, X_test)
    print("Quantum-ready train shape:", X_train_q.shape)
    print("Quantum-ready test shape:", X_test_q.shape)
