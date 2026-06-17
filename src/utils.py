import os
import random
import time
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate balanced class weights to handle dataset class imbalance.
    Formula: total_samples / (num_classes * class_count)
    """
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    weights = total_samples / (len(class_counts) * class_counts.float())
    return weights


def get_environment_info() -> dict:
    """Collect environment information for reproducibility."""
    import pennylane as qml
    import transformers
    import sklearn

    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers_version": transformers.__version__,
        "pennylane_version": qml.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    return info


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict:
    """Compute accuracy, macro-F1, and per-class classification report."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
    }


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


class Timer:
    """Context manager to measure elapsed time."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
