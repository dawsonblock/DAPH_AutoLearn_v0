"""DAPH v0.4 — Dimensionality reduction for hidden-state features.

Standardization + PCA pipeline fitted on TRAIN data only.
Never fit on dev or final data (would leak information).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PCAPipeline:
    """Standardization + PCA pipeline.

    Fitted on training data, then applied to all splits.
    """
    mean: np.ndarray = None
    std: np.ndarray = None
    components: np.ndarray = None  # [n_components, n_features]
    explained_variance_ratio: np.ndarray = None
    n_components: int = 0
    n_features: int = 0

    def fit(self, X: np.ndarray, n_components: int) -> "PCAPipeline":
        """Fit the pipeline on training data.

        Parameters
        ----------
        X : np.ndarray  shape [N, D]
            Training features.
        n_components : int
            Number of PCA components.
        """
        self.n_features = X.shape[1]
        self.n_components = min(n_components, self.n_features, X.shape[0])

        # Standardization
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0  # avoid division by zero

        X_centered = (X - self.mean) / self.std

        # PCA via SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components]
        self.explained_variance_ratio = (S[:self.n_components] ** 2) / (S ** 2).sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted pipeline to data."""
        X_centered = (X - self.mean) / self.std
        return X_centered @ self.components.T

    def fit_transform(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, n_components)
        return self.transform(X)

    def save(self, path: str) -> None:
        """Save the pipeline to an NPZ file."""
        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
            components=self.components,
            explained_variance_ratio=self.explained_variance_ratio,
            n_components=self.n_components,
            n_features=self.n_features,
        )

    @classmethod
    def load(cls, path: str) -> "PCAPipeline":
        """Load a pipeline from an NPZ file."""
        data = np.load(path)
        return cls(
            mean=data["mean"],
            std=data["std"],
            components=data["components"],
            explained_variance_ratio=data["explained_variance_ratio"],
            n_components=int(data["n_components"]),
            n_features=int(data["n_features"]),
        )


def select_pca_dimension(
    train_X: np.ndarray,
    dev_X: np.ndarray,
    train_y: np.ndarray,
    dev_y: np.ndarray,
    candidate_dims: list[int],
    policy_cls,
    policy_kwargs: dict | None = None,
) -> dict:
    """Select the best PCA dimension on development data.

    Parameters
    ----------
    train_X : np.ndarray  [N_train, D]
    dev_X : np.ndarray  [N_dev, D]
    train_y : np.ndarray  [N_train]
    dev_y : np.ndarray  [N_dev]
    candidate_dims : list[int]
        PCA dimensions to try.
    policy_cls : type
        Policy class with fit/predict.
    policy_kwargs : dict
        kwargs for policy constructor.

    Returns
    -------
    dict
        Results per dimension, plus the selected dimension.
    """
    if policy_kwargs is None:
        policy_kwargs = {}

    results = {}
    best_dim = candidate_dims[0]
    best_score = -np.inf

    for dim in candidate_dims:
        pipeline = PCAPipeline()
        train_reduced = pipeline.fit_transform(train_X, dim)
        dev_reduced = pipeline.transform(dev_X)

        policy = policy_cls(**policy_kwargs)
        policy.fit(train_reduced, train_y)

        # Score on dev
        dev_pred = policy.predict(dev_reduced)
        score = (dev_pred == dev_y).mean()

        results[dim] = {
            "accuracy": float(score),
            "explained_var": float(pipeline.explained_variance_ratio.sum()),
        }
        print(f"    PCA dim={dim}: dev accuracy={score:.4f}, "
              f"explained_var={pipeline.explained_variance_ratio.sum():.4f}")

        if score > best_score:
            best_score = score
            best_dim = dim

    results["selected_dim"] = best_dim
    results["selected_accuracy"] = best_score
    return results


__all__ = ["PCAPipeline", "select_pca_dimension"]
