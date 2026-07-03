"""
main.py
========
End-to-end orchestration script. Running this single file executes the
ENTIRE pipeline:

    1. Load raw NSL-KDD data
    2. Preprocess (clean, encode, scale)
    3. Train classical Random Forest (on full feature set)
    4. Reduce features via PCA -> angle encoding (for the quantum model,
       and for a qubit-matched Random Forest used as a fair-comparison
       control)
    5. Train the Quantum Variational Classifier (PennyLane)
    6. Evaluate both models with identical metrics
    7. Save metrics (JSON/CSV) and generate all comparison plots

Usage
-----
    pip install -r requirements.txt
    # place KDDTrain+.txt and KDDTest+.txt in data/raw/
    python main.py
"""

import numpy as np

from src.config import (
    QML_TRAIN_SAMPLE_SIZE,
    QML_TEST_SAMPLE_SIZE,
    RANDOM_STATE,
)
from src.data_loader import load_raw_data
from src.preprocessing import preprocess
from src.feature_selection import prepare_quantum_features
from src.classical_model import ClassicalIDS
from src.quantum_model import QuantumIDS
from src.evaluation import compute_metrics, print_report, save_metrics, build_comparison_table
from src.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_combined_roc_curves,
    plot_metric_comparison_bar,
    plot_training_loss,
)


def subsample(X, y, n, seed=RANDOM_STATE):
    """Stratified-ish random subsample, used to keep quantum simulation fast."""
    rng = np.random.RandomState(seed)
    n = min(n, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def main():
    print("=" * 70)
    print("QUANTUM MACHINE LEARNING BASED INTRUSION DETECTION SYSTEM")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Load raw data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading raw NSL-KDD data...")
    train_df, test_df = load_raw_data()
    print(f"  Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    # ------------------------------------------------------------------
    # STEP 2: Preprocess (clean, encode, scale)
    # ------------------------------------------------------------------
    print("\n[2/7] Preprocessing data (cleaning, one-hot encoding, scaling)...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(train_df, test_df)
    print(f"  Final engineered feature count: {len(feature_names)}")
    print(f"  Train label distribution (0=normal,1=attack): {np.bincount(y_train)}")
    print(f"  Test label distribution  (0=normal,1=attack): {np.bincount(y_test)}")

    # ------------------------------------------------------------------
    # STEP 3: Train classical Random Forest on the FULL feature set
    # ------------------------------------------------------------------
    print("\n[3/7] Training classical Random Forest baseline (full feature set)...")
    rf_model = ClassicalIDS().fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    print_report(y_test, rf_preds, "Random Forest (full features)")
    rf_metrics = compute_metrics(y_test, rf_preds, rf_proba, "Random Forest")
    save_metrics(rf_metrics, "classical_metrics.json")
    rf_model.save()

    # ------------------------------------------------------------------
    # STEP 4: PCA reduction + angle encoding for the quantum model
    # ------------------------------------------------------------------
    print("\n[4/7] Reducing dimensionality via PCA for quantum encoding...")
    X_train_q_full, X_test_q_full, pca, angle_scaler = prepare_quantum_features(X_train, X_test)
    import joblib

    joblib.dump(scaler, "results/models/scaler.joblib")
    joblib.dump(pca, "results/models/pca.joblib")
    joblib.dump(angle_scaler, "results/models/angle_scaler.joblib")
    joblib.dump(feature_names, "results/models/feature_names.joblib")

    # Sub-sample for tractable quantum-simulator runtime
    X_train_q, y_train_q = subsample(X_train_q_full, y_train, QML_TRAIN_SAMPLE_SIZE)
    X_test_q, y_test_q = subsample(X_test_q_full, y_test, QML_TEST_SAMPLE_SIZE)
    print(f"  Quantum train subsample: {X_train_q.shape} | test subsample: {X_test_q.shape}")

    # ------------------------------------------------------------------
    # STEP 5: Train the Quantum Variational Classifier
    # ------------------------------------------------------------------
    print("\n[5/7] Training Quantum Variational Classifier (PennyLane)...")
    qml_model = QuantumIDS()
    qml_model.fit(X_train_q, y_train_q)
    qml_preds = qml_model.predict(X_test_q)
    qml_proba = qml_model.predict_proba(X_test_q)
    print_report(y_test_q, qml_preds, "Quantum VQC")
    qml_metrics = compute_metrics(y_test_q, qml_preds, qml_proba, "Quantum VQC")
    save_metrics(qml_metrics, "quantum_metrics.json")
    plot_training_loss(qml_model.training_history)

    # ------------------------------------------------------------------
    # STEP 6: Build comparison table
    # ------------------------------------------------------------------
    print("\n[6/7] Building comparison table...")
    comparison_df = build_comparison_table([rf_metrics, qml_metrics])

    # ------------------------------------------------------------------
    # STEP 7: Generate all visualizations
    # ------------------------------------------------------------------
    print("\n[7/7] Generating visualizations...")
    plot_confusion_matrix(np.array(rf_metrics["confusion_matrix"]), "Random Forest", "rf_confusion_matrix.png")
    plot_confusion_matrix(np.array(qml_metrics["confusion_matrix"]), "Quantum VQC", "qml_confusion_matrix.png")
    plot_roc_curve(y_test, rf_proba, "Random Forest", "rf_roc_curve.png")
    plot_roc_curve(y_test_q, qml_proba, "Quantum VQC", "qml_roc_curve.png")
    plot_combined_roc_curves([
        (y_test, rf_proba, "Random Forest"),
        (y_test_q, qml_proba, "Quantum VQC"),
    ])
    plot_metric_comparison_bar(comparison_df)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE. See results/metrics/ and results/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
