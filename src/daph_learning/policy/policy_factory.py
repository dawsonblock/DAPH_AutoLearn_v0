"""v0.3.10.1-alpha — PolicyModel protocol and factory (Section 3).

The config/CLI accepts multiple policy types, but the integrated
learner previously always trained logistic routing. This module
defines the common :class:`PolicyModel` protocol and a factory that
returns the correct implementation for a given ``policy_type``.

Supported types (Section 3):

* ``centroid``         — :class:`CentroidPolicy` (weighted signed projection)
* ``logistic``         — :class:`WeightedLogisticRouter` (default)
* ``mlp_experimental`` — :class:`SmallMLPRouter` (diagnostic, not default)

All three implement the same surface so the integrated learner can
train, predict, and save/load them interchangeably.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import ExperimentConfig
from .targets import TargetMode


@runtime_checkable
class PolicyModel(Protocol):
    """Common protocol for all policy implementations (Section 3)."""

    def fit(
        self,
        train_features: np.ndarray,
        train_delta_u: np.ndarray,
        train_weights: np.ndarray,
        **kwargs: Any,
    ) -> "PolicyModel":
        ...  # pragma: no cover

    def predict_proba(self, h: Any) -> Any:
        ...  # pragma: no cover

    def save(self, path: str) -> None:
        ...  # pragma: no cover


def make_policy(config: ExperimentConfig):
    """Return an un-fit policy model instance for ``config.policy_type``.

    Parameters
    ----------
    config : ExperimentConfig
        ``config.policy_type`` selects the implementation.

    Returns
    -------
    PolicyModel
        ``CentroidPolicy`` | ``WeightedLogisticRouter`` |
        ``SmallMLPRouter``. Note: for torch-backed models the returned
        object is an ``nn.Module`` whose ``input_dim`` is not yet known
        — :func:`fit_policy` constructs the actual nn.Module from the
        training feature dimension.

    Raises
    ------
    ValueError
        On unknown ``policy_type``.
    """
    pt = config.policy_type
    if pt == "centroid":
        from .centroid_policy import CentroidPolicy
        return CentroidPolicy()
    if pt == "logistic":
        from .logistic import WeightedLogisticRouter
        # input_dim filled in by fit_policy
        return WeightedLogisticRouter.__new__(WeightedLogisticRouter)
    if pt == "mlp_experimental":
        from .mlp import SmallMLPRouter
        return SmallMLPRouter.__new__(SmallMLPRouter)
    raise ValueError(
        f"unknown policy_type {pt!r}; expected 'centroid', 'logistic', "
        f"or 'mlp_experimental'")


def fit_policy(
    config: ExperimentConfig,
    train_features: np.ndarray,
    train_delta_u: np.ndarray,
    train_weights: np.ndarray,
    *,
    dev_features: np.ndarray | None = None,
    dev_delta_u: np.ndarray | None = None,
    dev_weights: np.ndarray | None = None,
    dev_tasks: "Sequence[Mapping[str, Any]] | None" = None,
    utility_fn: "Callable[[Mapping[str, Any], str], float] | None" = None,
    seed: int = 0,
):
    """Fit the policy selected by ``config.policy_type``.

    Returns the fitted model. For torch-backed models, the returned
    object is the actual ``nn.Module`` (constructed with the correct
    ``input_dim`` from ``train_features``).
    """
    pt = config.policy_type
    if pt == "centroid":
        from .centroid_policy import train_centroid_policy
        return train_centroid_policy(
            train_features, train_delta_u, train_weights,
            gap_threshold=config.gap_threshold,
        )
    if pt == "logistic":
        from .logistic import (
            LogisticTrainConfig, WeightedLogisticRouter,
            train_weighted_logistic_router,
        )
        input_dim = train_features.shape[1]
        lcfg = LogisticTrainConfig(
            temperature=config.target_temperature,
            early_stopping_metric=config.early_stopping_metric,
            target_mode=config.target_mode,
            gap_threshold=config.gap_threshold,
        )
        return train_weighted_logistic_router(
            train_features, train_delta_u, train_weights,
            config=lcfg,
            dev_features=dev_features,
            dev_delta_u=dev_delta_u,
            dev_weights=dev_weights,
            dev_tasks=dev_tasks,
            utility_fn=utility_fn,
            confidence_threshold=config.confidence_threshold,
            seed=seed,
        )
    if pt == "mlp_experimental":
        from .mlp import MLPTrainConfig, train_small_mlp_router
        lcfg = MLPTrainConfig(
            temperature=config.target_temperature,
            early_stopping_metric=config.early_stopping_metric,
            target_mode=config.target_mode,
            gap_threshold=config.gap_threshold,
            hidden_dim=config.mlp_hidden_dim,
        )
        return train_small_mlp_router(
            train_features, train_delta_u, train_weights,
            config=lcfg,
            dev_features=dev_features,
            dev_delta_u=dev_delta_u,
            dev_weights=dev_weights,
            dev_tasks=dev_tasks,
            utility_fn=utility_fn,
            confidence_threshold=config.confidence_threshold,
            seed=seed,
        )
    raise ValueError(
        f"unknown policy_type {pt!r}; expected 'centroid', 'logistic', "
        f"or 'mlp_experimental'")


def predict_proba(model, features: np.ndarray) -> np.ndarray:
    """Run ``model.predict_proba`` and return a numpy array.

    Handles both torch and numpy backends.
    """
    pt_features = np.asarray(features, dtype=np.float32)
    # CentroidPolicy returns numpy directly.
    from .centroid_policy import CentroidPolicy
    if isinstance(model, CentroidPolicy):
        return np.asarray(model.predict_proba(pt_features), dtype=np.float64)
    # Torch models.
    try:
        import torch
        with torch.no_grad():
            t = torch.as_tensor(pt_features, dtype=torch.float32)
            probs = model.predict_proba(t)
            return probs.cpu().numpy()
    except ImportError:  # pragma: no cover
        return np.asarray(model.predict_proba(pt_features), dtype=np.float64)


__all__ = [
    "PolicyModel",
    "fit_policy",
    "make_policy",
    "predict_proba",
]
