import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, evaluate_classifier, get_class_weights


def train_and_evaluate_classifier(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    d_in: int,
) -> dict:
    """
    Train a linear classifier on PCA features with class-weighted cross-entropy loss.
    """
    set_seed(config.SEED)

    train_X_t = torch.tensor(train_X, dtype=torch.float32)
    train_y_t = torch.tensor(train_y, dtype=torch.long)
    val_X_t = torch.tensor(val_X, dtype=torch.float32)
    val_y_t = torch.tensor(val_y, dtype=torch.long)
    test_X_t = torch.tensor(test_X, dtype=torch.float32)
    test_y_t = torch.tensor(test_y, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(train_X_t, train_y_t),
        batch_size=config.CLF_BATCH_SIZE,
        shuffle=True,
    )

    # Linear classifier
    classifier = nn.Linear(d_in, config.NUM_CLASSES).to(config.DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.CLF_LR)
    
    # Class-weighted loss
    class_weights = get_class_weights(train_y_t).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Early stopping
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.CLF_EPOCHS):
        classifier.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)

            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation evaluation
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_X_t.to(config.DEVICE))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = (val_preds == val_y).mean()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.CLF_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}, best val_acc: {best_val_acc:.4f}")
                break

    # Load best state
    if best_state is not None:
        classifier.load_state_dict(best_state)

    # Evaluate on test set
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(test_X_t.to(config.DEVICE))
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    results = evaluate_classifier(test_y, test_preds, config.LABEL_NAMES)
    return results
