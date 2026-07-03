# Quantum Machine Learning Based Intrusion Detection System for IoT Networks

A research-internship-grade, end-to-end project comparing a **classical Random
Forest baseline** against a **Quantum Machine Learning (QML) Variational
Quantum Classifier (VQC)** built with **PennyLane**, for binary network
intrusion detection on the **NSL-KDD** dataset.

---

## 1. Motivation

IoT networks are resource-constrained and generate high-dimensional,
high-volume traffic. Classical Intrusion Detection Systems (IDS) using
tree-based ensembles (Random Forest, XGBoost) are strong baselines, but
Quantum Machine Learning is an emerging research direction that exploits
quantum superposition and entanglement to construct rich, non-linear feature
spaces using very few qubits. This project benchmarks a QML model against a
classical model on identical preprocessing, identical train/test splits, and
identical evaluation metrics — which is the standard rigorous methodology
expected in a research paper.

## 2. Dataset

**NSL-KDD** (an improved version of the original KDD'99 dataset; no
redundant records, better class balance) — a widely used IDS benchmark.

Download (place in `data/raw/`):
- `KDDTrain+.txt`
- `KDDTest+.txt`

Source: https://www.unb.ca/cic/datasets/nsl.html

The dataset has 41 features + 1 label column (attack type) + 1 difficulty
column. We treat the problem as **binary classification**: `normal` vs
`attack` (all attack subtypes — DoS, Probe, R2L, U2R — collapsed into one
"attack" class). This is the standard formulation used in most IDS papers
and is also the most natural fit for a 2-class quantum classifier.

## 3. Project Pipeline

```
Raw NSL-KDD  →  Data Loading  →  Preprocessing  →  Feature Selection (PCA)
            →  Classical RF Model   →  Evaluation  ↘
            →  Quantum VQC Model    →  Evaluation  →  Comparison + Plots
```

1. **Data Loading** (`src/data_loader.py`) — reads the raw `.txt` files and
   attaches official NSL-KDD column names.
2. **Preprocessing** (`src/preprocessing.py`) — cleans labels into
   binary form, one-hot encodes categorical features (`protocol_type`,
   `service`, `flag`), scales numeric features with `StandardScaler`.
3. **Feature Selection** (`src/feature_selection.py`) — PCA reduces ~120
   one-hot-encoded features down to a small number of components (default:
   8), because current quantum hardware/simulators can only practically
   handle a small number of qubits (1 qubit per feature in our encoding).
4. **Classical Model** (`src/classical_model.py`) — Random Forest trained on
   the full PCA-reduced feature set (it could use all features too; we keep
   it on the same reduced set for a *fair* apples-to-apples comparison).
5. **Quantum Model** (`src/quantum_model.py`) — a Variational Quantum
   Classifier built in PennyLane: angle-encoding of features into qubits,
   a trainable variational circuit (entangling layers), and a Pauli-Z
   expectation-value readout, trained with gradient descent (parameter-shift
   rule, handled automatically by PennyLane's autograd interface).
6. **Evaluation** (`src/evaluation.py`) — Accuracy, Precision, Recall,
   F1-score, ROC-AUC, Confusion Matrix for both models, saved as JSON/CSV.
7. **Visualization** (`src/visualization.py`) — confusion matrix heatmaps,
   ROC curves, and a side-by-side bar chart comparing both models.
8. **`main.py`** — runs the entire pipeline end-to-end with one command.

## 4. How to Run

```bash
pip install -r requirements.txt
# place KDDTrain+.txt and KDDTest+.txt inside data/raw/
python main.py
```

Outputs:
- `results/metrics/classical_metrics.json`
- `results/metrics/quantum_metrics.json`
- `results/metrics/comparison_table.csv`
- `results/figures/*.png`
- `results/models/random_forest.joblib`

## 5. Why PCA + Angle Encoding (Design Justification)

Simulated quantum circuits scale exponentially in classical compute cost
with qubit count (state vector size = 2^n). A laptop/Colab simulator can
comfortably handle 6–10 qubits. NSL-KDD has 41 raw features that balloon to
~120+ after one-hot encoding categorical columns. We therefore use PCA to
compress this to `N_QUBITS` (default 8) components that retain most
variance, then map each component to a rotation angle on one qubit
(angle encoding, `qml.AngleEmbedding`). This is a standard, defensible
choice in QML literature and should be explicitly justified in any paper
(see Section 8 below — this exact justification can go in your "Methodology"
section).

## 6. Models in Detail

### Classical: Random Forest
- `n_estimators=200`, `max_depth=None` (tuned via the config file)
- Trained on PCA-reduced features for fairness, with a note in code on how
  to also benchmark it on the *full* feature set (recommended as an
  additional ablation in your paper).

### Quantum: Variational Quantum Classifier (VQC)
- **Encoding layer**: `AngleEmbedding` — each classical feature (scaled to
  [0, π]) becomes a single-qubit rotation (RY gate).
- **Variational (ansatz) layer**: `qml.StronglyEntanglingLayers` — trainable
  rotation + entangling CNOT layers, depth configurable.
- **Measurement**: expectation value of Pauli-Z on the first qubit, mapped
  through a sigmoid to a [0,1] probability of "attack".
- **Optimizer**: Adam (via PennyLane's `qml.AdamOptimizer` / PyTorch
  interface), trained on binary cross-entropy loss.
- **Device**: `default.qubit` simulator (swap to real IBM/AWS hardware via
  Qiskit/Amazon Braket plugins for future work — see Section 8).

## 7. Evaluation Metrics

Both models are evaluated identically on the held-out NSL-KDD test set
(`KDDTest+.txt`, which famously contains attack patterns *not* seen in
training — making this a genuinely hard generalization benchmark):

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## 8. Suggested Research Contributions / Future Work

These are concrete angles you can position as *novel contributions* for a
paper or thesis chapter:

1. **Quantum feature map ablation study** — compare `AngleEmbedding` vs
   `AmplitudeEmbedding` vs `IQPEmbedding` and quantify how the choice of
   feature map affects separability of NSL-KDD classes (this alone is a
   solid paper contribution, since most QML-IDS papers don't ablate this).
2. **Qubit-count vs accuracy/runtime trade-off curve** — sweep
   `N_QUBITS = {4, 6, 8, 10, 12}` and plot accuracy and wall-clock training
   time, characterizing the practical "quantum advantage frontier" for IDS.
3. **Hybrid quantum-classical IDS** — use the quantum circuit purely as a
   learned feature extractor (its output expectation values) feeding into a
   lightweight classical classifier (logistic regression/SVM), and compare
   against pure-VQC and pure-RF.
4. **Multi-class extension** — extend from binary (normal/attack) to the
   5-class NSL-KDD taxonomy (Normal, DoS, Probe, R2L, U2R) using one-vs-rest
   quantum classifiers, and study how class imbalance (R2L/U2R are rare)
   affects the quantum model differently than Random Forest.
5. **Noise-resilience study** — inject simulated hardware noise (via
   PennyLane's `default.mixed` device or Qiskit noise models) and measure
   how IDS accuracy degrades, which is directly relevant for *deploying*
   QML-IDS on near-term IoT-edge-adjacent quantum hardware.
6. **Real IoT dataset validation** — replicate the pipeline on
   IoT-specific datasets (e.g., **TON_IoT**, **BoT-IoT**, **CICIoT2023**) to
   test whether conclusions from NSL-KDD generalize to genuine IoT traffic.
7. **Lightweight quantum kernel methods** — benchmark a Quantum Kernel SVM
   (`qml.kernels`) as a third model class, since kernel methods often
   outperform VQCs on small/medium tabular data and are a natural addition
   to the comparison table.
8. **Adversarial robustness** — compare how RF vs VQC degrade under
   adversarially perturbed NSL-KDD samples, an increasingly important IDS
   research theme.

## 9. File-by-File Explanation

See the docstring at the top of each file in `src/` — every file begins with
a detailed explanation of its purpose, inputs, outputs, and design choices.
A condensed version is also in Section 3 above.

## 10. Limitations to State Explicitly in a Paper

- Quantum circuits here run on a **classical simulator**, not real quantum
  hardware — report this clearly; "quantum advantage" is *not* claimed,
  only the expressivity/representational comparison is studied.
- PCA dimensionality reduction to fit qubit budgets is a current necessity
  of NISQ-era QML, not a quantum algorithmic advantage — discuss this as a
  limitation/future-work item (see Section 8.2).
- NSL-KDD is a 1999-derived dataset; while it's a standard benchmark, it is
  not raw IoT traffic — Section 8.6 addresses this directly.
