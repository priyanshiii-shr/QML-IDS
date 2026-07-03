"""
src package
============
Core source code for the Quantum Machine Learning Based Intrusion
Detection System for IoT Networks project.

Modules:
    config            -- central configuration (paths, hyperparameters)
    data_loader        -- raw NSL-KDD loading utilities
    preprocessing       -- cleaning, encoding, scaling, splitting
    feature_selection   -- PCA dimensionality reduction for the QML model
    classical_model      -- Random Forest baseline
    quantum_model        -- PennyLane Variational Quantum Classifier
    evaluation           -- shared metric computation utilities
    visualization         -- shared plotting utilities
"""
