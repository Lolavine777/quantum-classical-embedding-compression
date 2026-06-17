import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, count_parameters, get_class_weights


class AutoencoderClassifier(nn.Module):
    """
    MLP Autoencoder + Classifier for joint end-to-end training.

    Architecture:
        Encoder: 768 → 256 → d_out (8)
        Decoder: d_out (8) → 256 → 768
        Classifier: d_out (8) → num_classes (3)
    """

    def __init__(self, d_in: int = 768, d_out: int = 8, num_classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, d_out),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_out, 256),
            nn.ReLU(),
            nn.Linear(256, d_in),
        )
        self.classifier = nn.Linear(d_out, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input embeddings, shape (batch, d_in)

        Returns:
            Tuple of (reconstructed, latent, logits)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        logits = self.classifier(latent)
        return reconstructed, latent, logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AutoencoderCompressor:
    """Wrapper that trains the AutoencoderClassifier jointly end-to-end."""

    def __init__(self, d_out: int):
        self.d_out = d_out
        self.model = AutoencoderClassifier(d_in=config.EMBED_DIM, d_out=d_out, num_classes=config.NUM_CLASSES).to(config.DEVICE)
        self.training_log: list[dict] = []

    def fit(self, train_embeddings: torch.Tensor, train_labels: torch.Tensor,
            val_embeddings: torch.Tensor, val_labels: torch.Tensor) -> None:
        """
        Train end-to-end using reconstruction MSE loss + class-weighted classification cross-entropy.
        """
        set_seed(config.SEED)

        train_dataset = TensorDataset(train_embeddings, train_labels)
        val_dataset = TensorDataset(val_embeddings, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=config.AE_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.AE_BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.AE_LR)
        
        # Calculate dynamic class weights from training labels
        class_weights = get_class_weights(train_labels).to(config.DEVICE)
        criterion_clf = nn.CrossEntropyLoss(weight=class_weights)
        criterion_rec = nn.MSELoss()
        lambda_recon = config.AE_LAMBDA_RECON

        self.training_log = []

        print(f"  Training Autoencoder jointly end-to-end (d_out={self.d_out})...")
        print(f"  lambda_recon={lambda_recon} (loss = lambda*MSE + CE)")
        print(f"  Total trainable params: {count_parameters(self.model)}")
        print(f"  Encoder/Projection params: {self.get_param_count()}")

        for epoch in range(config.AE_EPOCHS):
            # Training
            self.model.train()
            train_loss_rec_sum = 0.0
            train_loss_clf_sum = 0.0
            train_correct = 0
            train_count = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)

                reconstructed, _, logits = self.model(batch_x)
                
                loss_rec = criterion_rec(reconstructed, batch_x)
                loss_clf = criterion_clf(logits, batch_y)
                
                # Weighted combination: lambda_recon * MSE + CE
                loss = lambda_recon * loss_rec + loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_rec_sum += loss_rec.item() * batch_x.size(0)
                train_loss_clf_sum += loss_clf.item() * batch_x.size(0)
                train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                train_count += batch_x.size(0)

            train_loss_rec = train_loss_rec_sum / train_count
            train_loss_clf = train_loss_clf_sum / train_count
            train_loss = lambda_recon * train_loss_rec + train_loss_clf
            train_acc = train_correct / train_count

            # Validation
            self.model.eval()
            val_loss_rec_sum = 0.0
            val_loss_clf_sum = 0.0
            val_correct = 0
            val_count = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(config.DEVICE)
                    batch_y = batch_y.to(config.DEVICE)

                    reconstructed, _, logits = self.model(batch_x)
                    
                    loss_rec = criterion_rec(reconstructed, batch_x)
                    loss_clf = criterion_clf(logits, batch_y)

                    val_loss_rec_sum += loss_rec.item() * batch_x.size(0)
                    val_loss_clf_sum += loss_clf.item() * batch_x.size(0)
                    val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                    val_count += batch_x.size(0)

            val_loss_rec = val_loss_rec_sum / val_count
            val_loss_clf = val_loss_clf_sum / val_count
            val_loss = lambda_recon * val_loss_rec + val_loss_clf
            val_acc = val_correct / val_count

            self.training_log.append({
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_loss_rec": round(train_loss_rec, 6),
                "train_loss_clf": round(train_loss_clf, 6),
                "val_loss": round(val_loss, 6),
                "val_loss_rec": round(val_loss_rec, 6),
                "val_loss_clf": round(val_loss_clf, 6),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
            })

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{config.AE_EPOCHS} - "
                      f"loss: {train_loss:.4f} (rec={train_loss_rec:.4f}, clf={train_loss_clf:.4f}), "
                      f"val_loss: {val_loss:.4f}, "
                      f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")

        print(f"  Autoencoder training complete. Final val_loss: {val_loss:.6f}, val_acc: {val_acc:.4f}")

    def transform(self, embeddings: torch.Tensor) -> np.ndarray:
        """Extract compressed representations from the trained encoder."""
        self.model.eval()
        with torch.no_grad():
            latent = self.model.encode(embeddings.to(config.DEVICE))
        return latent.cpu().numpy()

    def predict(self, embeddings: torch.Tensor) -> np.ndarray:
        """Predict class labels using the trained classifier head."""
        self.model.eval()
        all_preds = []
        dataset = TensorDataset(embeddings)
        loader = DataLoader(dataset, batch_size=config.AE_BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(config.DEVICE)
                _, _, logits = self.model(batch_x)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())

        return torch.cat(all_preds, dim=0).numpy()

    def get_param_count(self) -> int:
        """Return encoder parameter count (projection head only)."""
        return sum(p.numel() for p in self.model.encoder.parameters())

    def get_training_log(self) -> list[dict]:
        return self.training_log
