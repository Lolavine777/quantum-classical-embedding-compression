import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, count_parameters, get_class_weights


class BlochEncoder(nn.Module):
    """
    Encode a classical d-dimensional vector into d Bloch-sphere qubit states.

    Each scalar u_j is mapped to a 2D qubit state on the Bloch sphere:
        theta_j = tanh(u_j) * pi/2 + pi/2      # theta in [0, pi]
        state_j = [cos(theta_j / 2), -sin(theta_j / 2)]

    The negative sign on the sin term comes from choosing phi = pi on the
    Bloch sphere (azimuthal angle), which keeps all amplitudes real.

    Input:  (batch, d)
    Output: (batch, d, 2)  — each qubit as a 2D amplitude vector
    """

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Classical embedding vector, shape (batch, d).

        Returns:
            Qubit states, shape (batch, d, 2).
        """
        theta = torch.tanh(u) * (math.pi / 2) + (math.pi / 2)  # (batch, d)
        half_theta = theta / 2
        cos_h = torch.cos(half_theta)  # (batch, d)
        sin_h = torch.sin(half_theta)  # (batch, d)
        # Stack as [cos(θ/2), -sin(θ/2)] per qubit
        states = torch.stack([cos_h, -sin_h], dim=-1)  # (batch, d, 2)
        return states


class VectorizedZYZRotation(nn.Module):
    """
    Trainable ZYZ single-qubit unitaries applied to a batch of qubit states in parallel.

    Instead of defining separate modules for each qubit, we define parameter tensors
    of shape (n_pairs,) to perform vectorized rotation operations.
    """

    def __init__(self, n_pairs: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(n_pairs) * 0.1)
        self.beta = nn.Parameter(torch.randn(n_pairs) * 0.1)
        self.gamma = nn.Parameter(torch.randn(n_pairs) * 0.1)

    def forward(self, qubits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            qubits: Qubit states, shape (batch, n_pairs, 2).

        Returns:
            Rotated qubit states, shape (batch, n_pairs, 2).
        """
        # Apply Ry(beta) rotation to each qubit
        half_beta = self.beta / 2
        cos_b = torch.cos(half_beta).unsqueeze(0)  # (1, n_pairs)
        sin_b = torch.sin(half_beta).unsqueeze(0)  # (1, n_pairs)

        a = qubits[:, :, 0]  # (batch, n_pairs)
        b = qubits[:, :, 1]  # (batch, n_pairs)
        new_a = cos_b * a - sin_b * b  # (batch, n_pairs)
        new_b = sin_b * a + cos_b * b  # (batch, n_pairs)
        return torch.stack([new_a, new_b], dim=-1)  # (batch, n_pairs, 2)


class CascadeLayer(nn.Module):
    """
    One layer of cascaded pairwise compression: d qubits → d/2 qubits.
    Vectorized implementation to avoid slow Python loops over pairs.

    For each adjacent pair (qubit_i, qubit_{i+1}):
      1. Apply trainable ZYZ (Ry) rotation to each qubit in parallel.
      2. Simulate CNOT(i → i+1) classically.
      3. Partial trace over qubit i (measure & discard) in parallel.
      4. Re-encode measured qubit as: [sqrt(p(0)), sqrt(p(1))]
      5. Keep qubit j, drop qubit i → d/2 output qubits.

    Args:
        d_in: Number of input qubits (must be even).
    """

    def __init__(self, d_in: int):
        super().__init__()
        assert d_in % 2 == 0, f"d_in must be even, got {d_in}"
        self.d_in = d_in
        self.n_pairs = d_in // 2

        # Two trainable rotations per layer, each containing parameters for all n_pairs
        self.rotations_i = VectorizedZYZRotation(self.n_pairs)
        self.rotations_j = VectorizedZYZRotation(self.n_pairs)

    def forward(self, qubits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            qubits: Qubit states, shape (batch, d_in, 2).

        Returns:
            Compressed qubit states, shape (batch, d_in // 2, 2).
        """
        # Split into control (even) and target (odd) qubits
        qi = qubits[:, 0::2, :]  # (batch, n_pairs, 2)
        qj = qubits[:, 1::2, :]  # (batch, n_pairs, 2)

        # Step 1: Apply trainable rotations
        qi = self.rotations_i(qi)  # (batch, n_pairs, 2)
        qj = self.rotations_j(qj)  # (batch, n_pairs, 2)

        # Step 2: CNOT(i → j) — classical simulation & Step 3: Partial trace over control
        a_i, b_i = qi[:, :, 0], qi[:, :, 1]  # (batch, n_pairs)
        a_j, b_j = qj[:, :, 0], qj[:, :, 1]  # (batch, n_pairs)

        # Combined state after CNOT: [a_i*a_j, a_i*b_j, b_i*b_j, b_i*a_j]
        # p(0) for qubit j = |a_i*a_j|^2 + |b_i*b_j|^2
        p0 = (a_i * a_j) ** 2 + (b_i * b_j) ** 2  # (batch, n_pairs)
        # p(1) for qubit j = |a_i*b_j|^2 + |b_i*a_j|^2
        p1 = (a_i * b_j) ** 2 + (b_i * a_j) ** 2  # (batch, n_pairs)

        # Step 4: Re-encode as new qubit state
        eps = 1e-8
        new_qubits = torch.stack([
            torch.sqrt(p0 + eps),
            torch.sqrt(p1 + eps),
        ], dim=-1)  # (batch, n_pairs, 2)

        return new_qubits


class QiCascadedModel(nn.Module):
    """
    Quantum-inspired Cascaded Compression model (arXiv:2501.04591).

    Architecture:
        1. Linear projection: 768 → d_start
        2. Bloch encoding: d_start scalars → d_start qubit states
        3. Cascaded compression: d_start → d_start/2 → ... → d_out
           (log2(d_start / d_out) cascade layers)
        4. Flatten qubit states to scalar probabilities p(0) per qubit
        5. Classifier head: d_out → num_classes

    All operations are differentiable PyTorch — no quantum hardware or
    PennyLane needed. This is a purely classical simulation of the
    quantum-inspired compression scheme.
    """

    def __init__(self, d_in: int = 768, d_out: int = 8, num_classes: int = 3):
        super().__init__()
        self.d_out = d_out

        # d_start = 4 * d_out to show cascading (e.g. 32 → 16 → 8)
        self.d_start = 4 * d_out
        n_cascade_layers = int(math.log2(self.d_start // d_out))

        # Linear projection into d_start dimensions
        self.projection = nn.Linear(d_in, self.d_start)

        # Bloch sphere encoder
        self.bloch_encoder = BlochEncoder()

        # Cascade layers: d_start → d_start/2 → ... → d_out
        cascade = []
        dim = self.d_start
        for _ in range(n_cascade_layers):
            cascade.append(CascadeLayer(dim))
            dim //= 2
        assert dim == d_out, f"Cascade did not reach d_out={d_out}, ended at {dim}"
        self.cascade = nn.ModuleList(cascade)

        # Classifier on the compressed representation
        self.classifier = nn.Linear(d_out, num_classes)

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the full compression pipeline (projection → Bloch → cascade → flatten).

        Args:
            x: Input embeddings, shape (batch, d_in).

        Returns:
            Compressed representation, shape (batch, d_out).
        """
        # Project to d_start dimensions
        projected = self.projection(x)  # (batch, d_start)

        # Bloch encode each dimension into a qubit state
        qubits = self.bloch_encoder(projected)  # (batch, d_start, 2)

        # Cascade compress
        for layer in self.cascade:
            qubits = layer(qubits)  # halves qubit count each layer

        # Flatten: extract p(0) = |amplitude_0|^2 for each qubit as scalar feature
        compressed = qubits[:, :, 0] ** 2  # (batch, d_out)
        return compressed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input embeddings, shape (batch, d_in).

        Returns:
            Tuple of (logits, compressed).
        """
        compressed = self._compress(x)
        logits = self.classifier(compressed)
        return logits, compressed

    def get_compressed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract compressed representation without classification."""
        return self._compress(x)


class QiCompressor:
    """
    Wrapper that trains the QiCascadedModel end-to-end.

    Follows the same interface as AutoencoderCompressor:
        - fit(train_embeddings, train_labels, val_embeddings, val_labels)
        - predict(embeddings) -> np.ndarray
        - transform(embeddings) -> np.ndarray
        - get_param_count() -> int
        - get_training_log() -> list[dict]
    """

    def __init__(self, d_out: int):
        self.d_out = d_out
        self.model = QiCascadedModel(
            d_in=config.EMBED_DIM,
            d_out=d_out,
            num_classes=config.NUM_CLASSES,
        ).to(config.DEVICE)
        self.training_log: list[dict] = []

    def fit(self, train_embeddings: torch.Tensor, train_labels: torch.Tensor,
            val_embeddings: torch.Tensor, val_labels: torch.Tensor) -> None:
        """
        Train end-to-end with class-weighted cross-entropy loss.

        Uses Adam optimizer with the same hyperparameters as the autoencoder:
        batch_size=AE_BATCH_SIZE, lr=AE_LR, epochs=AE_EPOCHS.
        """
        set_seed(config.SEED)

        train_dataset = TensorDataset(train_embeddings, train_labels)
        val_dataset = TensorDataset(val_embeddings, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=config.AE_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.AE_BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.AE_LR)

        # Calculate dynamic class weights from training labels
        class_weights = get_class_weights(train_labels).to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.training_log = []

        print(f"  Training QI Cascaded Compressor (d_out={self.d_out}, "
              f"d_start={self.model.d_start}, "
              f"layers={len(self.model.cascade)})...")
        print(f"  Total trainable params: {count_parameters(self.model)}")
        print(f"  Compression head params: {self.get_param_count()}")

        for epoch in range(config.AE_EPOCHS):
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

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{config.AE_EPOCHS} - "
                      f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                      f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")

        print(f"  QI training complete. Final val_loss: {val_loss:.6f}, val_acc: {val_acc:.4f}")

    def transform(self, embeddings: torch.Tensor) -> np.ndarray:
        """Extract compressed representations from the trained model."""
        self.model.eval()
        all_compressed = []
        dataset = TensorDataset(embeddings)
        loader = DataLoader(dataset, batch_size=config.AE_BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(config.DEVICE)
                compressed = self.model.get_compressed(batch_x)
                all_compressed.append(compressed.cpu())

        return torch.cat(all_compressed, dim=0).numpy()

    def predict(self, embeddings: torch.Tensor) -> np.ndarray:
        """Predict class labels using the trained classifier head."""
        self.model.eval()
        all_preds = []
        dataset = TensorDataset(embeddings)
        loader = DataLoader(dataset, batch_size=config.AE_BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(config.DEVICE)
                logits, _ = self.model(batch_x)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())

        return torch.cat(all_preds, dim=0).numpy()

    def get_param_count(self) -> int:
        """
        Return compression head parameter count.

        Includes: projection layer + cascade layer rotations.
        Excludes: classifier head.
        """
        count = sum(p.numel() for p in self.model.projection.parameters())
        count += sum(p.numel() for p in self.model.cascade.parameters())
        return count

    def get_training_log(self) -> list[dict]:
        return self.training_log
