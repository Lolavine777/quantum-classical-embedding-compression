import os
import sys
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import evaluate_classifier


class FidelityClassifier:
    """
    Fidelity-based prototype classifier inspired by arXiv:2501.04591.

    Instead of standard Euclidean/Cosine distance, classification uses the
    quantum Fidelity of fully separable (product) states.

    Pipeline:
        1. Compute class prototypes as the mean embedding of training samples.
        2. Encode each embedding dimension as a Bloch sphere angle:
               theta_j = tanh(x_j) * pi/2 + pi/2
           This maps any real-valued feature to the range (0, pi).
        3. For a test sample u and a class prototype v, compute the product
           fidelity over all dimensions:
               F(|u>, |v>) = prod_j cos^2((theta_u_j - theta_v_j) / 2)
        4. Assign the class whose prototype has the highest fidelity.

    This classifier has zero trainable parameters.
    """

    def __init__(self):
        self.prototypes: np.ndarray | None = None
        self.classes: np.ndarray | None = None

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        """
        Compute class prototypes from training data.

        Each prototype is the element-wise mean of all training embeddings
        belonging to that class.

        Args:
            train_X: Training embeddings of shape (n_samples, d_out).
            train_y: Training labels of shape (n_samples,).
        """
        self.classes = np.unique(train_y)
        self.prototypes = np.zeros((len(self.classes), train_X.shape[1]))

        for i, c in enumerate(self.classes):
            mask = train_y == c
            self.prototypes[i] = train_X[mask].mean(axis=0)
            count = mask.sum()
            print(f"    Class {c}: {count} samples -> prototype computed")

    # ------------------------------------------------------------------
    # Bloch encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bloch_angles(X: np.ndarray) -> np.ndarray:
        """
        Encode real-valued features as Bloch sphere angles.

        theta_j = tanh(x_j) * pi/2 + pi/2

        This maps (-inf, +inf) -> (0, pi), centred at pi/2 for x=0.

        Args:
            X: Array of shape (..., d) with real-valued features.

        Returns:
            Array of the same shape with angles in (0, pi).
        """
        return np.tanh(X) * (math.pi / 2.0) + (math.pi / 2.0)

    # ------------------------------------------------------------------
    # Fidelity computation
    # ------------------------------------------------------------------

    @staticmethod
    def _fidelity(theta_u: np.ndarray, theta_v: np.ndarray) -> np.ndarray:
        """
        Compute the product fidelity between two sets of Bloch angles.

        F(|u>, |v>) = prod_j cos^2((theta_u_j - theta_v_j) / 2)

        To avoid numerical underflow for high-dimensional embeddings the
        computation is done in log-space:
            log F = sum_j 2 * log|cos((theta_u_j - theta_v_j) / 2)|

        Args:
            theta_u: Bloch angles for samples, shape (n_samples, d).
            theta_v: Bloch angles for prototypes, shape (n_prototypes, d).

        Returns:
            Fidelity matrix of shape (n_samples, n_prototypes).
        """
        # theta_u: (N, 1, d), theta_v: (1, C, d) -> broadcast diff: (N, C, d)
        diff = theta_u[:, np.newaxis, :] - theta_v[np.newaxis, :, :]
        cos_half_diff = np.cos(diff / 2.0)

        # Log-space computation to prevent underflow for large d
        log_fidelity = np.sum(
            2.0 * np.log(np.abs(cos_half_diff) + 1e-30), axis=-1
        )
        return np.exp(log_fidelity)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using Fidelity distance to prototypes.

        Args:
            test_X: Test embeddings of shape (n_samples, d_out).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        if self.prototypes is None:
            raise RuntimeError("Classifier has not been fitted. Call fit() first.")

        theta_test = self._to_bloch_angles(test_X)
        theta_proto = self._to_bloch_angles(self.prototypes)

        # Fidelity matrix: (n_samples, n_classes)
        fidelity_matrix = self._fidelity(theta_test, theta_proto)

        # Assign class with highest fidelity
        pred_indices = np.argmax(fidelity_matrix, axis=1)
        return self.classes[pred_indices]

    def predict_cosine(self, test_X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using Cosine similarity to prototypes.

        Provided as a baseline for fair comparison against the Fidelity
        approach under the same prototype-based classification framework.

        Args:
            test_X: Test embeddings of shape (n_samples, d_out).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        if self.prototypes is None:
            raise RuntimeError("Classifier has not been fitted. Call fit() first.")

        # Normalise to unit vectors
        test_norms = np.linalg.norm(test_X, axis=1, keepdims=True) + 1e-30
        proto_norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-30

        test_normed = test_X / test_norms
        proto_normed = self.prototypes / proto_norms

        # Cosine similarity matrix: (n_samples, n_classes)
        cosine_matrix = test_normed @ proto_normed.T

        # Assign class with highest cosine similarity
        pred_indices = np.argmax(cosine_matrix, axis=1)
        return self.classes[pred_indices]

    # ------------------------------------------------------------------
    # Convenience: full evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_X: np.ndarray,
        test_y: np.ndarray,
        label_names: list[str] | None = None,
    ) -> dict:
        """
        Run both Fidelity and Cosine predictions and return evaluation metrics.

        Args:
            test_X: Test embeddings of shape (n_samples, d_out).
            test_y: Ground-truth labels of shape (n_samples,).
            label_names: Optional list of human-readable label names.

        Returns:
            Dictionary with 'fidelity' and 'cosine' sub-dicts, each containing
            accuracy, macro_f1, and classification_report.
        """
        if label_names is None:
            label_names = config.LABEL_NAMES

        fidelity_preds = self.predict(test_X)
        cosine_preds = self.predict_cosine(test_X)

        results = {
            "fidelity": evaluate_classifier(test_y, fidelity_preds, label_names),
            "cosine": evaluate_classifier(test_y, cosine_preds, label_names),
        }
        return results

    def get_param_count(self) -> int:
        """Return the number of trainable parameters (always zero)."""
        return 0

    def get_training_log(self) -> None:
        """No training log for a non-trainable classifier."""
        return None
