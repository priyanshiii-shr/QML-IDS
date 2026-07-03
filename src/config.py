"""
config.py
=========
Central configuration file for the whole pipeline.

WHY THIS FILE EXISTS
---------------------
Hard-coding paths, hyperparameters, and constants throughout multiple
scripts makes a research project fragile and hard to reproduce. Every other
module in `src/` imports its settings from here. If you want to change the
number of qubits, the Random Forest depth, or a file path, change it in
ONE place and the entire pipeline updates consistently. This is standard
practice for reproducible research code.

WHAT TO EDIT
------------
- `N_QUBITS`        -> controls PCA output dimensionality AND circuit width.
- `RANDOM_STATE`     -> seed for full reproducibility (data split, RF, QML).
- `RF_PARAMS`        -> Random Forest hyperparameters.
- `QML_PARAMS`       -> quantum circuit hyperparameters (layers, epochs, lr).
"""

import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

TRAIN_FILE = os.path.join(DATA_RAW_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_RAW_DIR, "KDDTest+.txt")

# Ensure output directories always exist
for _d in [DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Feature reduction
# ---------------------------------------------------------------------
# Number of PCA components = number of qubits used in the quantum circuit.
# Kept small because quantum simulators scale as O(2^N_QUBITS) in memory.
N_QUBITS = 8

# ---------------------------------------------------------------------
# Sampling (NSL-KDD train set has ~125,973 rows; quantum simulation is
# slow, so by default we sub-sample the TRAINING set for the quantum model
# to keep runtime reasonable on a laptop/Colab. The classical model can
# still be trained on the FULL set for a fair "best each model can do"
# comparison -- controlled by USE_FULL_DATA_FOR_RF below.)
# ---------------------------------------------------------------------
QML_TRAIN_SAMPLE_SIZE = 800     # rows used to train the quantum model
QML_TEST_SAMPLE_SIZE = 400      # rows used to test the quantum model
USE_FULL_DATA_FOR_RF = True     # Random Forest trains on the full dataset

# ---------------------------------------------------------------------
# Classical model hyperparameters
# ---------------------------------------------------------------------
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# ---------------------------------------------------------------------
# Quantum model hyperparameters
# ---------------------------------------------------------------------
QML_PARAMS = dict(
    n_qubits=N_QUBITS,
    n_layers=3,            # depth of StronglyEntanglingLayers ansatz
    n_epochs=15,
    learning_rate=0.05,
    batch_size=32,
)

# ---------------------------------------------------------------------
# NSL-KDD official column names (41 features + label + difficulty)
# ---------------------------------------------------------------------
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
TARGET_COLUMN = "label"
