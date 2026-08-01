"""DAPH v0.4 — Generic executive policy models for arbitrary action spaces.

This module provides policy models that work with the generic
:class:`~daph_learning.executive.ActionSpace` instead of the hard-coded
binary ``symbolic`` vs ``llm`` routing.

Two implementations are provided:

* :class:`ExecutiveCentroidPolicy` — generalizes ``CentroidPolicy`` to
  N-action classification using one-vs-rest weighted centroids.
* :class:`ExecutiveLogisticPolicy` — generalizes
  ``WeightedLogisticRouter`` to N-action multinomial logistic regression.

Both implement the :class:`ExecutivePolicyModel` protocol:

    fit(features, action_utilities, weights, action_space) → self
    predict_proba(h) → np.ndarray  shape [N, n_actions]
    predict_decision(h, action_space) → list[ActionDecision]

The key difference from v0.3.x is that ``action_utilities`` is a
``[N, n_actions]`` matrix of per-action utilities (not a 1-D ``ΔU``
array), and ``predict_proba`` returns a full probability distribution
over actions (not a single ``P(symbolic)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from daph_learning.executive.types import (
    ActionSpace,
    ActionDecision,
)


@runtime_checkable
class ExecutivePolicyModel(Protocol):
    """Protocol for generic executive policy models.

    All v0.4 policy models implement this interface. It generalizes
    the v0.3.x ``PolicyModel`` protocol to arbitrary action spaces.
    """

    def fit(
        self,
        train_features: np.ndarray,
        action_utilities: np.ndarray,
        train_weights: np.ndarray,
        action_space: ActionSpace,
        **kwargs: Any,
    ) -> "ExecutivePolicyModel":
        ...  # pragma: no cover

    def predict_proba(self, h: Any) -> Any:
        ...  # pragma: no cover

    def save(self, path: str) -> None:
        ...  # pragma: no cover


# ──────────────────────────────────────────────────────────────────────
# Executive Centroid Policy (N-action one-vs-rest)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExecutiveCentroidPolicy:
    """N-action centroid policy using one-vs-rest weighted centroids.

    For each action ``a``, computes a weighted centroid ``μ_a`` of the
    features where ``a`` is the best action (argmax of utilities).
    At prediction time, scores each action by projection onto its
    centroid and applies softmax to get a probability distribution.

    This generalizes the binary ``CentroidPolicy`` which uses a single
    contrastive direction ``v = μ_S - μ_L``. For 2 actions, the
    one-vs-rest approach reduces to the same binary contrast.
    """

    action_ids: tuple[str, ...] = ()
    centroids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    threshold: float = 0.0
    temperature: float = 1.0
    n_train_: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    weight_fallback: bool = False
    fallback_actions: tuple[str, ...] = ()
    degenerate_train_data: bool = False

    def fit(
        self,
        train_features: np.ndarray,
        action_utilities: np.ndarray,
        train_weights: np.ndarray,
        action_space: ActionSpace,
        *,
        gap_threshold: float = 0.0,
        eps: float = 1e-12,
    ) -> "ExecutiveCentroidPolicy":
        """Fit the N-action centroid policy.

        Parameters
        ----------
        train_features : np.ndarray  shape ``[N, D]``
        action_utilities : np.ndarray  shape ``[N, n_actions]``
            Per-action utility for each training example.
        train_weights : np.ndarray  shape ``[N]``
        action_space : ActionSpace
        gap_threshold : float
            Utility gap below which an example is considered a tie
            (excluded from centroid computation).
        """
        feats = np.asarray(train_features, dtype=np.float32)
        utils = np.asarray(action_utilities, dtype=np.float64)
        w = np.asarray(train_weights, dtype=np.float64)
        n, d = feats.shape
        n_actions = len(action_space.action_ids)

        if utils.shape != (n, n_actions):
            raise ValueError(
                f"action_utilities shape {utils.shape} != expected ({n}, {n_actions})")
        if w.shape != (n,):
            raise ValueError(f"weights shape {w.shape} != expected ({n},)")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

        self.action_ids = action_space.action_ids

        # Determine best action per example (oracle)
        best_actions = np.argmax(utils, axis=1)
        best_utils = np.max(utils, axis=1)

        # Check for ties (gap between best and second-best < gap_threshold)
        if n_actions > 2:
            sorted_utils = np.sort(utils, axis=1)
            gap = sorted_utils[:, -1] - sorted_utils[:, -2]
        else:
            gap = np.abs(utils[:, 0] - utils[:, 1])

        decisive = gap > gap_threshold

        # Compute per-action centroids
        centroids = np.zeros((n_actions, d), dtype=np.float32)
        counts: dict[str, int] = {}
        self.weight_fallback = False
        self.fallback_actions = ()

        for i, action_id in enumerate(self.action_ids):
            mask = decisive & (best_actions == i)
            counts[action_id] = int(mask.sum())
            if mask.sum() == 0:
                continue
            class_w = w[mask]
            if float(class_w.sum()) <= eps:
                # Fallback to unweighted mean
                self.weight_fallback = True
                self.fallback_actions = self.fallback_actions + (action_id,)
                class_w = np.ones_like(class_w)
            weighted_mean = (feats[mask] * class_w[:, None]).sum(axis=0) / max(class_w.sum(), eps)
            centroids[i] = weighted_mean

        # Check degenerate cases
        n_nonempty = sum(1 for c in counts.values() if c > 0)
        if n_nonempty <= 1:
            self.degenerate_train_data = True
        else:
            self.degenerate_train_data = False

        self.centroids = centroids
        self.n_train_ = n
        self.class_counts = counts

        # Calibrate temperature on training scores
        if n_nonempty >= 2:
            scores = feats @ centroids.T  # [N, n_actions]
            # Use the margin between best and second-best score
            sorted_scores = np.sort(scores, axis=1)
            if n_actions > 1:
                margins = sorted_scores[:, -1] - sorted_scores[:, -2]
                self.temperature = float(max(margins.std(), 1e-8))
            else:
                self.temperature = 1.0
            self.threshold = 0.0
        else:
            self.temperature = 1.0
            self.threshold = 0.0

        return self

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        """Return ``P(a | h)`` for each row of ``h``.

        Returns
        -------
        np.ndarray  shape ``[N, n_actions]``
            Probability distribution over actions for each input.
        """
        h = np.asarray(h, dtype=np.float32)
        if self.centroids.shape[0] == 0:
            raise RuntimeError("ExecutiveCentroidPolicy is not fitted")
        single = h.ndim == 1
        if single:
            h = h[np.newaxis, :]
        if h.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                f"feature dim {h.shape[1]} != fitted dim "
                f"{self.centroids.shape[1]}")

        # Score = dot product with each centroid
        scores = h @ self.centroids.T  # [N, n_actions]
        tau = max(self.temperature, 1e-8)

        # Softmax over actions
        logits = scores / tau
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return probs[0] if single else probs

    def predict_decision(
        self,
        h: np.ndarray,
        action_space: ActionSpace,
        *,
        state_ids: list[str] | None = None,
        calibration_applied: bool = False,
    ) -> list[ActionDecision]:
        """Return ActionDecision for each row of ``h``."""
        probs = self.predict_proba(h)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]

        n = probs.shape[0]
        if state_ids is None:
            state_ids = [f"s{i}" for i in range(n)]

        decisions = []
        for i in range(n):
            p = probs[i]
            selected_idx = int(np.argmax(p))
            selected = action_space.action_ids[selected_idx]
            scores = {action_space.action_ids[j]: float(p[j]) for j in range(len(p))}
            decisions.append(ActionDecision(
                state_id=state_ids[i],
                scores=scores,
                probabilities=scores,
                selected_action=selected,
                confidence=float(p[selected_idx]),
                calibration_applied=calibration_applied,
            ))
        return decisions

    def estimator_name(self) -> str:
        if self.weight_fallback:
            return "executive_centroid_with_unweighted_fallback"
        return "executive_centroid"

    def save(self, path: str) -> None:
        import json
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "policy_type": "executive_centroid",
            "action_ids": list(self.action_ids),
            "centroids": np.asarray(self.centroids, dtype=np.float32).tolist(),
            "threshold": float(self.threshold),
            "temperature": float(self.temperature),
            "n_train_": int(self.n_train_),
            "class_counts": dict(self.class_counts),
            "weight_fallback": bool(self.weight_fallback),
            "fallback_actions": list(self.fallback_actions),
            "degenerate_train_data": bool(self.degenerate_train_data),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExecutiveCentroidPolicy":
        import json
        with open(path) as f:
            payload = json.load(f)
        if payload.get("policy_type") != "executive_centroid":
            raise ValueError(
                f"expected policy_type='executive_centroid', got "
                f"{payload.get('policy_type')!r}")
        model = cls()
        model.action_ids = tuple(payload["action_ids"])
        model.centroids = np.asarray(payload["centroids"], dtype=np.float32)
        model.threshold = float(payload["threshold"])
        model.temperature = float(payload["temperature"])
        model.n_train_ = int(payload.get("n_train_", 0))
        model.class_counts = dict(payload.get("class_counts", {}))
        model.weight_fallback = bool(payload.get("weight_fallback", False))
        model.fallback_actions = tuple(payload.get("fallback_actions", []))
        model.degenerate_train_data = bool(payload.get("degenerate_train_data", False))
        return model


# ──────────────────────────────────────────────────────────────────────
# Executive Logistic Policy (N-action multinomial logistic)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExecutiveLogisticPolicy:
    """N-action multinomial logistic regression policy.

    Uses softmax regression with utility-weighted examples. For 2 actions
    this reduces to binary logistic regression (same as
    ``WeightedLogisticRouter``).
    """

    action_ids: tuple[str, ...] = ()
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    n_train_: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    n_iter_: int = 0
    convergence_history: list[float] = field(default_factory=list)

    def fit(
        self,
        train_features: np.ndarray,
        action_utilities: np.ndarray,
        train_weights: np.ndarray,
        action_space: ActionSpace,
        *,
        lr: float = 0.01,
        n_iter: int = 500,
        l2: float = 1e-4,
        tol: float = 1e-6,
    ) -> "ExecutiveLogisticPolicy":
        """Fit the N-action logistic policy via gradient descent.

        Parameters
        ----------
        train_features : np.ndarray  shape ``[N, D]``
        action_utilities : np.ndarray  shape ``[N, n_actions]``
        train_weights : np.ndarray  shape ``[N]``
        action_space : ActionSpace
        lr : float
            Learning rate.
        n_iter : int
            Maximum iterations.
        l2 : float
            L2 regularization strength.
        tol : float
            Convergence tolerance on loss change.
        """
        feats = np.asarray(train_features, dtype=np.float64)
        utils = np.asarray(action_utilities, dtype=np.float64)
        w = np.asarray(train_weights, dtype=np.float64)
        n, d = feats.shape
        n_actions = len(action_space.action_ids)

        if utils.shape != (n, n_actions):
            raise ValueError(
                f"action_utilities shape {utils.shape} != expected ({n}, {n_actions})")

        self.action_ids = action_space.action_ids

        # Target: best action (argmax utility)
        best_actions = np.argmax(utils, axis=1)
        best_utils = np.max(utils, axis=1)

        # One-hot encode targets
        Y = np.zeros((n, n_actions), dtype=np.float64)
        Y[np.arange(n), best_actions] = 1.0

        # Weighted targets (scale by sample weight)
        Y_weighted = Y * w[:, None]

        # Initialize weights
        W = np.zeros((d, n_actions), dtype=np.float64)
        b = np.zeros(n_actions, dtype=np.float64)

        prev_loss = float("inf")
        history = []

        for iteration in range(n_iter):
            logits = feats @ W + b  # [N, n_actions]
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Weighted cross-entropy loss
            loss = -np.sum(w[:, None] * Y * np.log(probs + 1e-12)) / max(w.sum(), 1e-12)
            loss += 0.5 * l2 * np.sum(W ** 2)
            history.append(float(loss))

            if abs(prev_loss - loss) < tol:
                self.n_iter_ = iteration + 1
                break
            prev_loss = loss

            # Gradient
            diff = probs - Y  # [N, n_actions]
            grad_W = (feats.T @ (diff * w[:, None])) / max(w.sum(), 1e-12) + l2 * W
            grad_b = np.sum(diff * w[:, None], axis=0) / max(w.sum(), 1e-12)

            W -= lr * grad_W
            b -= lr * grad_b
        else:
            self.n_iter_ = n_iter

        self.weights = W.astype(np.float32)
        self.bias = b.astype(np.float32)
        self.n_train_ = n
        self.convergence_history = history
        self.class_counts = {
            action_space.action_ids[i]: int(np.sum(best_actions == i))
            for i in range(n_actions)
        }
        return self

    def predict_proba(self, h: np.ndarray) -> np.ndarray:
        """Return ``P(a | h)`` for each row of ``h``.

        Returns
        -------
        np.ndarray  shape ``[N, n_actions]``
        """
        h = np.asarray(h, dtype=np.float32)
        if self.weights.shape[0] == 0:
            raise RuntimeError("ExecutiveLogisticPolicy is not fitted")
        single = h.ndim == 1
        if single:
            h = h[np.newaxis, :]
        if h.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"feature dim {h.shape[1]} != fitted dim "
                f"{self.weights.shape[0]}")

        logits = h @ self.weights + self.bias
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return probs[0] if single else probs

    def predict_decision(
        self,
        h: np.ndarray,
        action_space: ActionSpace,
        *,
        state_ids: list[str] | None = None,
        calibration_applied: bool = False,
    ) -> list[ActionDecision]:
        """Return ActionDecision for each row of ``h``."""
        probs = self.predict_proba(h)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]

        n = probs.shape[0]
        if state_ids is None:
            state_ids = [f"s{i}" for i in range(n)]

        decisions = []
        for i in range(n):
            p = probs[i]
            selected_idx = int(np.argmax(p))
            selected = action_space.action_ids[selected_idx]
            scores = {action_space.action_ids[j]: float(p[j]) for j in range(len(p))}
            decisions.append(ActionDecision(
                state_id=state_ids[i],
                scores=scores,
                probabilities=scores,
                selected_action=selected,
                confidence=float(p[selected_idx]),
                calibration_applied=calibration_applied,
            ))
        return decisions

    def estimator_name(self) -> str:
        return "executive_logistic"

    def save(self, path: str) -> None:
        import json
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "policy_type": "executive_logistic",
            "action_ids": list(self.action_ids),
            "weights": np.asarray(self.weights, dtype=np.float32).tolist(),
            "bias": np.asarray(self.bias, dtype=np.float32).tolist(),
            "n_train_": int(self.n_train_),
            "class_counts": dict(self.class_counts),
            "n_iter_": int(self.n_iter_),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExecutiveLogisticPolicy":
        import json
        with open(path) as f:
            payload = json.load(f)
        if payload.get("policy_type") != "executive_logistic":
            raise ValueError(
                f"expected policy_type='executive_logistic', got "
                f"{payload.get('policy_type')!r}")
        model = cls()
        model.action_ids = tuple(payload["action_ids"])
        model.weights = np.asarray(payload["weights"], dtype=np.float32)
        model.bias = np.asarray(payload["bias"], dtype=np.float32)
        model.n_train_ = int(payload.get("n_train_", 0))
        model.class_counts = dict(payload.get("class_counts", {}))
        model.n_iter_ = int(payload.get("n_iter_", 0))
        return model


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def make_executive_policy(policy_type: str) -> ExecutivePolicyModel:
    """Return an un-fit executive policy model.

    Parameters
    ----------
    policy_type : str
        ``"centroid"`` or ``"logistic"``.
    """
    if policy_type == "centroid":
        return ExecutiveCentroidPolicy()
    if policy_type == "logistic":
        return ExecutiveLogisticPolicy()
    raise ValueError(
        f"unknown executive policy_type {policy_type!r}; "
        f"expected 'centroid' or 'logistic'")


__all__ = [
    "ExecutivePolicyModel",
    "ExecutiveCentroidPolicy",
    "ExecutiveLogisticPolicy",
    "make_executive_policy",
]
