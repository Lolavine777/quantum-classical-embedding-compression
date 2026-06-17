import numpy as np
from sklearn.decomposition import PCA


class PCACompressor:
    """PCA-based embedding compression. No trainable parameters."""

    def __init__(self, d_out: int):
        self.d_out = d_out
        self.pca = PCA(n_components=d_out, random_state=42)

    def fit(self, train_embeddings: np.ndarray) -> None:
        """Fit PCA on training embeddings."""
        self.pca.fit(train_embeddings)
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"  PCA(d_out={self.d_out}): explained variance ratio = {explained_var:.4f}")

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to d_out dimensions."""
        return self.pca.transform(embeddings)

    def get_param_count(self) -> int:
        return 0

    def get_training_log(self) -> None:
        return None
