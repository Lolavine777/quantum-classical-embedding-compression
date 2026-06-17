import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, count_parameters, get_class_weights


def create_qnode(n_qubits: int, n_layers: int):
    """
    Create a PennyLane QNode with angle encoding + variational layers.

    Circuit structure:
        1. Angle encoding: RY(input[i]) on each qubit
        2. For each variational layer:
           a. RY(w[layer,i,0]) and RZ(w[layer,i,1]) on each qubit
           b. CNOT ring entanglement: CNOT(i, (i+1) % n_qubits)
        3. Measurement: <Z_i> for each qubit
    """
    dev = qml.device("default.qubit.torch", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Angle encoding
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')

        # Variational layers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Return list of expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class PQCModel(nn.Module):
    """
    Hybrid classical-quantum model for embedding compression.

    Architecture:
        1. Linear projection: 768 → n_qubits (with tanh to bound inputs to [-1,1] → scale to [-π,π])
        2. Quantum circuit: angle encoding + variational layers → n_qubits expectation values
        3. Classifier: n_qubits → num_classes
    """

    def __init__(self, d_in: int = 768, d_out: int = 8,
                 n_qubits: int = 8, n_layers: int = 2, num_classes: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_out = d_out

        # Classical projection to n_qubits dimensions
        self.projection = nn.Linear(d_in, n_qubits)

        # Quantum circuit as TorchLayer
        circuit = create_qnode(n_qubits, n_layers)
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Classifier head
        self.classifier = nn.Linear(d_out, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (logits, compressed)
        """
        # Project to n_qubits dimensions and scale to [-π, π]
        projected = torch.tanh(self.projection(x)) * np.pi

        # Quantum circuit output
        compressed = self.qlayer(projected)  # (batch, n_qubits)

        # Classification
        logits = self.classifier(compressed)

        return logits, compressed

    def get_compressed(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.tanh(self.projection(x)) * np.pi
        compressed = self.qlayer(projected)
        return compressed


class PQCCompressor:
    """Wrapper for training and using the PQC hybrid model."""

    def __init__(self, d_out: int):
        self.d_out = d_out
        self.model = PQCModel(
            d_in=config.EMBED_DIM,
            d_out=d_out,
            n_qubits=config.N_QUBITS,
            n_layers=config.N_LAYERS,
            num_classes=config.NUM_CLASSES,
        ).to(config.DEVICE)
        self.training_log: list[dict] = []
        self.anomalies: list[str] = []

    def fit(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        val_embeddings: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> None:
        """Train PQC end-to-end with class-weighted cross-entropy loss."""
        set_seed(config.SEED)

        train_dataset = TensorDataset(train_embeddings, train_labels)
        val_dataset = TensorDataset(val_embeddings, val_labels)
        train_loader = DataLoader(
            train_dataset, batch_size=config.PQC_BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.PQC_BATCH_SIZE, shuffle=False
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.PQC_LR)
        
        # Calculate dynamic class weights from training labels
        class_weights = get_class_weights(train_labels).to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.training_log = []
        self.anomalies = []
        total_params = count_parameters(self.model)
        projection_params = self.get_param_count()

        print(f"  Training PQC (d_out={self.d_out})...")
        print(f"  Total trainable params: {total_params}")
        print(f"  Projection head params: {projection_params}")

        # Stagnation detection: track consecutive epochs without val_loss improvement
        best_val_loss = float("inf")
        stagnation_counter = 0
        STAGNATION_THRESHOLD = 5

        for epoch in range(config.PQC_EPOCHS):
            # Training
            self.model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)

                logits, _ = self.model(batch_x)
                loss = criterion(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * batch_x.size(0)
                train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                train_count += batch_x.size(0)

            train_loss = train_loss_sum / train_count
            train_acc = train_correct / train_count

            # Validation
            self.model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_count = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(config.DEVICE)
                    batch_y = batch_y.to(config.DEVICE)

                    logits, _ = self.model(batch_x)
                    loss = criterion(logits, batch_y)

                    val_loss_sum += loss.item() * batch_x.size(0)
                    val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                    val_count += batch_x.size(0)

            val_loss = val_loss_sum / val_count
            val_acc = val_correct / val_count

            self.training_log.append({
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
            })

            # Stagnation detection
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter == STAGNATION_THRESHOLD:
                    msg = (f"PQC gradient stagnation: val_loss did not improve for "
                           f"{STAGNATION_THRESHOLD} consecutive epochs "
                           f"(epochs {epoch+2-STAGNATION_THRESHOLD}-{epoch+1}). "
                           f"Possible vanishing gradient through quantum circuit.")
                    self.anomalies.append(msg)
                    print(f"    [WARNING] {msg}")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{config.PQC_EPOCHS} - "
                      f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                      f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")

        print(f"  PQC training complete. Final val_acc: {val_acc:.4f}")
        if self.anomalies:
            print(f"  [WARNING] {len(self.anomalies)} stagnation warning(s) recorded.")

    def transform(self, embeddings: torch.Tensor) -> np.ndarray:
        """Extract compressed embeddings from trained PQC."""
        self.model.eval()
        all_compressed = []
        dataset = TensorDataset(embeddings)
        loader = DataLoader(dataset, batch_size=config.PQC_BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(config.DEVICE)
                compressed = self.model.get_compressed(batch_x)
                all_compressed.append(compressed.cpu())

        return torch.cat(all_compressed, dim=0).numpy()

    def predict(self, embeddings: torch.Tensor) -> np.ndarray:
        """Predict class labels using the full hybrid model."""
        self.model.eval()
        all_preds = []
        dataset = TensorDataset(embeddings)
        loader = DataLoader(dataset, batch_size=config.PQC_BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(config.DEVICE)
                logits, _ = self.model(batch_x)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())

        return torch.cat(all_preds, dim=0).numpy()

    def get_param_count(self) -> int:
        """
        Return projection head parameter count.
        Includes: projection layer + quantum circuit weights.
        Excludes: classifier head.
        """
        count = sum(p.numel() for p in self.model.projection.parameters())
        count += sum(p.numel() for p in self.model.qlayer.parameters())
        return count

    def get_training_log(self) -> list[dict]:
        return self.training_log

    def get_anomalies(self) -> list[str]:
        """Return list of anomaly messages detected during training."""
        return self.anomalies
