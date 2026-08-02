"""DAPH v0.4 — Q-regression executive policy (B4-B).

Instead of classification (argmax of utilities), this policy learns
to predict the actual counterfactual utility of each action:

    Q_hat(h, a) ≈ U(h, a)

Then selects:
    pi(h) = argmax_a Q_hat(h, a)

This preserves the full counterfactual information and trains for
regret minimization rather than route accuracy.

Training loss:
    L = (1/N) * sum_i sum_a w_ia * (Q_hat(h_i, a) - U_ia)^2

Or Huber loss for robustness.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class QRegressionPolicy:
    """Three-headed utility predictor for executive routing.

    Learns a separate linear regressor for each action's utility.
    Selection is argmax over predicted utilities.

    Attributes
    ----------
    action_ids : list[str]
        Action IDs in order.
    learning_rate : float
        Gradient descent learning rate.
    n_iter : int
        Number of training iterations.
    l2 : float
        L2 regularization strength.
    loss : str
        "mse" or "huber".
    huber_delta : float
        Delta for Huber loss.
    """

    action_ids: list[str] = field(default_factory=list)
    learning_rate: float = 0.01
    n_iter: int = 1000
    l2: float = 0.001
    loss: str = "mse"
    huber_delta: float = 0.5

    # Learned parameters
    weights_: np.ndarray = None  # [n_actions, n_features]
    bias_: np.ndarray = None     # [n_actions]

    def _init_params(self, n_features: int):
        self.weights_ = np.zeros((len(self.action_ids), n_features), dtype=np.float32)
        self.bias_ = np.zeros(len(self.action_ids), dtype=np.float32)

    def fit(
        self,
        X: np.ndarray,
        utilities: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> "QRegressionPolicy":
        """Fit the Q-regression policy.

        Parameters
        ----------
        X : np.ndarray  [N, D]
            Feature matrix.
        utilities : np.ndarray  [N, n_actions]
            Counterfactual utilities for each action.
        sample_weights : np.ndarray  [N] | None
            Optional per-sample weights.
        """
        n, d = X.shape
        n_actions = len(self.action_ids)
        self._init_params(d)

        if sample_weights is None:
            sample_weights = np.ones(n, dtype=np.float32)

        # Gradient descent
        for iteration in range(self.n_iter):
            # Forward pass: predict utilities
            preds = X @ self.weights_.T + self.bias_  # [N, n_actions]

            # Compute errors
            errors = preds - utilities  # [N, n_actions]

            # Loss gradient
            if self.loss == "huber":
                abs_err = np.abs(errors)
                small = abs_err <= self.huber_delta
                grad = np.where(small, errors, self.huber_delta * np.sign(errors))
            else:
                grad = errors

            # Weight gradient with sample weights
            weighted_grad = grad * sample_weights[:, None]  # [N, n_actions]
            grad_w = (weighted_grad.T @ X) / n + self.l2 * self.weights_  # [n_actions, D]
            grad_b = weighted_grad.mean(axis=0)  # [n_actions]

            # Update
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

        return self

    def predict_utilities(self, X: np.ndarray) -> np.ndarray:
        """Predict utilities for all actions.

        Returns
        -------
        np.ndarray  [N, n_actions]
        """
        return X @ self.weights_.T + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the best action index for each sample.

        Returns
        -------
        np.ndarray  [N]  (action indices)
        """
        preds = self.predict_utilities(X)
        return np.argmax(preds, axis=1)

    def predict_action_ids(self, X: np.ndarray) -> list[str]:
        """Predict the best action ID for each sample."""
        idxs = self.predict(X)
        return [self.action_ids[i] for i in idxs]

    def save(self, path: str) -> None:
        """Save policy to JSON."""
        import json
        data = {
            "policy_type": "q_regression",
            "action_ids": self.action_ids,
            "learning_rate": self.learning_rate,
            "n_iter": self.n_iter,
            "l2": self.l2,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "weights_": self.weights_.tolist() if self.weights_ is not None else None,
            "bias_": self.bias_.tolist() if self.bias_ is not None else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QRegressionPolicy":
        """Load policy from JSON."""
        import json
        with open(path) as f:
            data = json.load(f)
        policy = cls(
            action_ids=data["action_ids"],
            learning_rate=data["learning_rate"],
            n_iter=data["n_iter"],
            l2=data["l2"],
            loss=data["loss"],
            huber_delta=data["huber_delta"],
        )
        if data["weights_"] is not None:
            policy.weights_ = np.array(data["weights_"], dtype=np.float32)
            policy.bias_ = np.array(data["bias_"], dtype=np.float32)
        return policy


def compute_regret(
    predicted_actions: np.ndarray,
    utilities: np.ndarray,
) -> np.ndarray:
    """Compute per-task regret.

    Regret = U(best action) - U(selected action)

    Parameters
    ----------
    predicted_actions : np.ndarray  [N]  (action indices)
    utilities : np.ndarray  [N, n_actions]

    Returns
    -------
    np.ndarray  [N]
    """
    best_utilities = utilities.max(axis=1)
    selected_utilities = utilities[np.arange(len(predicted_actions)), predicted_actions]
    return best_utilities - selected_utilities


def mean_regret(predicted_actions: np.ndarray, utilities: np.ndarray) -> float:
    """Compute mean regret."""
    return float(compute_regret(predicted_actions, utilities).mean())


__all__ = ["QRegressionPolicy", "compute_regret", "mean_regret"]
