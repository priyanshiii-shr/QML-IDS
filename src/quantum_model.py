"""
quantum_model.py
==================
Quantum Machine Learning model: a Variational Quantum Classifier (VQC)
built with PennyLane, trained via gradient descent using PyTorch as the
autodiff/optimization backend (PennyLane's `torch` interface).

ARCHITECTURE
------------
1. ENCODING (data -> quantum state):
   `qml.AngleEmbedding` -- each of the N_QUBITS classical features
   (already rescaled into [0, pi] by feature_selection.py) is loaded as a
   single-qubit RY rotation angle. This is the standard, simplest, and
   most hardware-efficient encoding scheme in NISQ-era QML literature.

2. VARIATIONAL ANSATZ (trainable quantum "neural network"):
   `qml.StronglyEntanglingLayers` -- alternating trainable single-qubit
   rotations (RX/RY/RZ, the optimizer learns these) and entangling CNOT
   gates connecting neighboring qubits in a ring topology. This is the
   quantum analogue of a dense neural-network layer: it lets the circuit
   learn non-trivial correlations between the encoded features.

3. MEASUREMENT (quantum state -> classical output):
   Expectation value of the Pauli-Z operator on qubit 0, which is a real
   number in [-1, 1]. We map this through a sigmoid-like transform to get
   a [0, 1] "probability of attack" usable with standard binary
   cross-entropy loss and standard scikit-learn metric functions.

4. OPTIMIZATION:
   Adam optimizer (via PyTorch) minimizes binary cross-entropy loss.
   PennyLane automatically computes gradients through the quantum circuit
   using the parameter-shift rule under the hood when the `torch`
   interface is used -- no manual gradient derivation needed.

WHY THIS COUNTS AS "QUANTUM MACHINE LEARNING"
-------------------------------------------------
The trainable parameters live INSIDE the quantum circuit (rotation
angles of the ansatz), and the loss landscape is optimized by actually
running (simulating) the quantum circuit -- this is the standard
definition of a Variational Quantum Algorithm / VQC, the dominant
NISQ-era QML paradigm used in essentially all current QML-IDS papers.
"""

import numpy as np
import torch
import pennylane as qml
from tqdm import trange

from src.config import QML_PARAMS


class QuantumIDS:
    """PennyLane-based Variational Quantum Classifier for IDS."""

    def __init__(self, params: dict = None):
        self.params = params or QML_PARAMS
        self.n_qubits = self.params["n_qubits"]
        self.n_layers = self.params["n_layers"]
        self.n_epochs = self.params["n_epochs"]
        self.lr = self.params["learning_rate"]
        self.batch_size = self.params["batch_size"]

        # `default.qubit` is PennyLane's built-in state-vector simulator.
        # Swap to `qiskit.aer` or a real-hardware plugin for future work
        # (see README Section 8 "noise-resilience study").
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Shape of trainable weights expected by StronglyEntanglingLayers:
        # (n_layers, n_qubits, 3) -- 3 rotation angles per qubit per layer.
        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.n_qubits
        )
        self.weights = torch.tensor(
            0.01 * np.random.randn(*weight_shape),
            requires_grad=True,
            dtype=torch.float64,
        )

        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, inputs, weights):
        """The quantum circuit: encoding + ansatz + measurement."""
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(0))

    def _forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        """Run the circuit on a batch and map output to [0, 1] probability."""
        raw_outputs = torch.stack(
            [self.qnode(x, self.weights) for x in X_batch]
        )
        # Map expectation value in [-1, 1] to probability in [0, 1].
        probs = (raw_outputs + 1.0) / 2.0
        return probs

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        X_tensor = torch.tensor(X_train, dtype=torch.float64)
        y_tensor = torch.tensor(y_train, dtype=torch.float64)

        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        loss_fn = torch.nn.BCELoss()

        n_samples = X_tensor.shape[0]
        history = []

        epoch_iter = trange(self.n_epochs, desc="Training QML model") if verbose else range(self.n_epochs)

        for epoch in epoch_iter:
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                idx = permutation[start:start + self.batch_size]
                X_batch = X_tensor[idx]
                y_batch = y_tensor[idx]

                optimizer.zero_grad()
                preds = self._forward(X_batch)
                preds = torch.clamp(preds, 1e-6, 1 - 1e-6)  # numerical safety
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append(avg_loss)
            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

        self.training_history = history
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.tensor(X, dtype=torch.float64)
        with torch.no_grad():
            probs = self._forward(X_tensor)
        return probs.numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int64)

    def save_weights(self, path: str):
        torch.save(self.weights, path)
        print(f"[quantum_model] Saved trained quantum weights to {path}")

    def load_weights(self, path: str):
        self.weights = torch.load(path)
        return self


if __name__ == "__main__":
    # python -m src.quantum_model
    # Small smoke test on synthetic data to verify the circuit runs.
    np.random.seed(0)
    X_dummy = np.random.uniform(0, np.pi, size=(40, QML_PARAMS["n_qubits"]))
    y_dummy = np.random.randint(0, 2, size=40)

    model = QuantumIDS(params={**QML_PARAMS, "n_epochs": 3, "batch_size": 8})
    model.fit(X_dummy, y_dummy)
    preds = model.predict(X_dummy)
    print("Smoke-test predictions:", preds[:10])
