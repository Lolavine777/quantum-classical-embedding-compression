"""
Visualization utilities for the Quantum-Classical Embedding Compression study.

Generates:
  - t-SNE 2D scatter plots of compressed embeddings (colored by sentiment class)
  - Training curve comparison (loss & accuracy per epoch)
  - Bar charts comparing Accuracy & Macro-F1 across methods and dimensions
  - Parameter efficiency chart
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Color palette for 3 sentiment classes
CLASS_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}
CLASS_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Method colors for bar/line charts
METHOD_COLORS = {
    "PCA": "#3498db",
    "Autoencoder": "#2ecc71",
    "PQC": "#e74c3c",
    "PQC_NoEnt": "#e67e22",
    "QiC": "#9b59b6",
    "Fidelity_KNN": "#1abc9c",
}


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    d_out: int,
    save_dir: str,
    perplexity: int = 30,
    seed: int = 42,
) -> str:
    """
    Generate a t-SNE 2D scatter plot of compressed embeddings.

    Args:
        embeddings: Compressed embeddings, shape (N, d_out)
        labels: Integer class labels, shape (N,)
        method_name: Name of the compression method
        d_out: Output dimensionality
        save_dir: Directory to save the plot
        perplexity: t-SNE perplexity parameter
        seed: Random seed for reproducibility

    Returns:
        Path to the saved plot image
    """
    # t-SNE reduction to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_id in sorted(np.unique(labels)):
        mask = labels == cls_id
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=CLASS_COLORS.get(cls_id, "#999999"),
            label=CLASS_LABELS.get(cls_id, f"Class {cls_id}"),
            alpha=0.6, s=15, edgecolors="none",
        )
    ax.set_title(f"t-SNE: {method_name} (d'={d_out})", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"tsne_{method_name.lower().replace(' ', '_')}_d{d_out}.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved t-SNE plot: {filepath}")
    return filepath


def plot_training_curves(
    training_logs: dict[str, list[dict]],
    save_dir: str,
    d_out: int,
) -> str:
    """
    Plot training curves (loss and accuracy) for multiple methods.

    Args:
        training_logs: Dict mapping method_name -> list of epoch dicts
                       Each dict has keys: epoch, train_loss, val_loss, train_acc, val_acc
        save_dir: Directory to save the plot
        d_out: Output dimensionality (for title)

    Returns:
        Path to the saved plot image
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax_loss = axes[0]
    for method, log in training_logs.items():
        if log is None:
            continue
        epochs = [entry["epoch"] for entry in log]
        val_losses = [entry["val_loss"] for entry in log]
        color = METHOD_COLORS.get(method, "#999999")
        ax_loss.plot(epochs, val_losses, label=method, color=color, linewidth=1.5)
    ax_loss.set_title(f"Validation Loss (d'={d_out})", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="best")
    ax_loss.grid(True, alpha=0.3)

    # Accuracy subplot
    ax_acc = axes[1]
    for method, log in training_logs.items():
        if log is None:
            continue
        epochs = [entry["epoch"] for entry in log]
        val_accs = [entry["val_acc"] for entry in log]
        color = METHOD_COLORS.get(method, "#999999")
        ax_acc.plot(epochs, val_accs, label=method, color=color, linewidth=1.5)
    ax_acc.set_title(f"Validation Accuracy (d'={d_out})", fontsize=13, fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc="best")
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"training_curves_d{d_out}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved training curves: {filepath}")
    return filepath


def plot_comparison_bars(
    results_table: list[dict],
    save_dir: str,
) -> str:
    """
    Plot grouped bar charts comparing Accuracy and Macro-F1 across methods.

    Args:
        results_table: List of dicts with keys: method, d_out, accuracy, macro_f1
        save_dir: Directory to save the plot

    Returns:
        Path to the saved plot image
    """
    # Group by d_out
    d_outs = sorted(set(r["d_out"] for r in results_table))
    methods = []
    seen = set()
    for r in results_table:
        if r["method"] not in seen:
            methods.append(r["method"])
            seen.add(r["method"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "macro_f1", "Macro F1-Score"),
    ]:
        x = np.arange(len(d_outs))
        width = 0.8 / len(methods)
        for i, method in enumerate(methods):
            values = []
            for d in d_outs:
                match = [r for r in results_table if r["method"] == method and r["d_out"] == d]
                values.append(match[0][metric] if match else 0)
            color = METHOD_COLORS.get(method, "#999999")
            bars = ax.bar(x + i * width - 0.4 + width / 2, values,
                         width, label=method, color=color, alpha=0.85)
            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                           f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Output Dimension (d')")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in d_outs])
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "comparison_bars.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved comparison bars: {filepath}")
    return filepath


def plot_param_efficiency(
    results_table: list[dict],
    save_dir: str,
) -> str:
    """
    Plot accuracy per 1000 parameters for each method (parameter efficiency).

    Args:
        results_table: List of dicts with keys: method, d_out, accuracy, projection_params
        save_dir: Directory to save the plot

    Returns:
        Path to the saved plot image
    """
    # Filter to a single d_out for clarity (use the default D_OUT=8)
    target_d = config.D_OUT
    filtered = [r for r in results_table if r["d_out"] == target_d and r.get("projection_params", 0) > 0]

    if not filtered:
        print("    No trainable methods found for param efficiency chart.")
        return ""

    methods = [r["method"] for r in filtered]
    accs = [r["accuracy"] for r in filtered]
    params = [r["projection_params"] for r in filtered]
    efficiency = [a / (p / 1000) if p > 0 else 0 for a, p in zip(accs, params)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [METHOD_COLORS.get(m, "#999999") for m in methods]
    bars = ax.bar(methods, efficiency, color=colors, alpha=0.85)
    for bar, eff, p in zip(bars, efficiency, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
               f"{eff:.4f}\n({p} params)", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Parameter Efficiency: Accuracy per 1K Params (d'={target_d})",
                fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy / 1000 params")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "param_efficiency.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved param efficiency: {filepath}")
    return filepath
