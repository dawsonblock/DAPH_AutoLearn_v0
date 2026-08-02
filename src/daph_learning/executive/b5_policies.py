"""DAPH v0.4 — B5 policy models for adaptive compute.

Trains only modest policies to predict the full utility vector::

    h_i → [U_fast, U_think, U_retrieve, U_decompose]

Three model classes:

1. **LinearQPolicy** — Linear Q regression: Q_a(h) = w_a^T h + b_a
2. **RidgeQPolicy** — Ridge Q regression (tune ridge on DEV only)
3. **MLPQPolicy** — Small two-layer MLP (128→128→64→4)

All train on full counterfactual utility (regression loss), not a
simple winner classifier.

Optional secondary pairwise-ranking loss::

    L_rank = max(0, m - (Q_{a+} - Q_{a-}))

Do not replace utility regression with a winner classifier.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Base interface
# ──────────────────────────────────────────────────────────────────────

class QPolicyBase:
    """Base class for Q-regression policies."""

    action_ids: list[str]
    weights_: np.ndarray | None
    bias_: np.ndarray | None

    def predict_utilities(self, X: np.ndarray) -> np.ndarray:
        """Predict utilities for all actions. [N, n_actions]"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict best action index for each sample. [N]"""
        preds = self.predict_utilities(X)
        return np.argmax(preds, axis=1)

    def predict_action_ids(self, X: np.ndarray) -> list[str]:
        idxs = self.predict(X)
        return [self.action_ids[i] for i in idxs]

    def predict_margins(self, X: np.ndarray) -> np.ndarray:
        """Predict Q-margin: Q_(1) - Q_(2) for each sample. [N]"""
        preds = self.predict_utilities(X)
        sorted_preds = np.sort(preds, axis=1)
        if preds.shape[1] < 2:
            return np.ones(len(X))
        return sorted_preds[:, -1] - sorted_preds[:, -2]

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "QPolicyBase":
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Linear Q regression policy
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LinearQPolicy(QPolicyBase):
    """Linear Q regression: Q_a(h) = w_a^T h + b_a.

    Uses gradient descent with optional L2 regularization.
    """

    action_ids: list[str] = field(default_factory=list)
    learning_rate: float = 0.01
    n_iter: int = 1000
    l2: float = 0.001
    loss: str = "mse"  # "mse" or "huber"
    huber_delta: float = 0.5
    rank_margin: float = 0.0  # if >0, add pairwise ranking loss

    weights_: np.ndarray = field(default=None, repr=False)
    bias_: np.ndarray = field(default=None, repr=False)

    def _init_params(self, n_features: int):
        self.weights_ = np.zeros((len(self.action_ids), n_features), dtype=np.float32)
        self.bias_ = np.zeros(len(self.action_ids), dtype=np.float32)

    def fit(
        self,
        X: np.ndarray,
        utilities: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> "LinearQPolicy":
        n, d = X.shape
        self._init_params(d)
        if sample_weights is None:
            sample_weights = np.ones(n, dtype=np.float32)

        for _ in range(self.n_iter):
            preds = X @ self.weights_.T + self.bias_
            errors = preds - utilities

            if self.loss == "huber":
                abs_err = np.abs(errors)
                small = abs_err <= self.huber_delta
                grad = np.where(small, errors, self.huber_delta * np.sign(errors))
            else:
                grad = errors

            weighted_grad = grad * sample_weights[:, None]
            grad_w = (weighted_grad.T @ X) / n + self.l2 * self.weights_
            grad_b = weighted_grad.mean(axis=0)

            # Optional ranking loss gradient
            if self.rank_margin > 0:
                rank_grad = self._ranking_gradient(preds, utilities)
                grad_w += rank_grad.T @ X / n

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

        return self

    def _ranking_gradient(self, preds: np.ndarray, utilities: np.ndarray) -> np.ndarray:
        """Pairwise ranking loss gradient (margin-based)."""
        n, n_actions = preds.shape
        grad = np.zeros_like(preds)
        for i in range(n):
            # For each pair (a+, a-) where U[a+] > U[a-]
            for a1 in range(n_actions):
                for a2 in range(n_actions):
                    if utilities[i, a1] > utilities[i, a2]:
                        margin = preds[i, a1] - preds[i, a2]
                        if margin < self.rank_margin:
                            grad[i, a1] -= 1.0
                            grad[i, a2] += 1.0
        return grad

    def predict_utilities(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights_.T + self.bias_

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "policy_type": "linear_q",
                "action_ids": self.action_ids,
                "learning_rate": self.learning_rate,
                "n_iter": self.n_iter,
                "l2": self.l2,
                "loss": self.loss,
                "huber_delta": self.huber_delta,
                "rank_margin": self.rank_margin,
                "weights_": self.weights_.tolist() if self.weights_ is not None else None,
                "bias_": self.bias_.tolist() if self.bias_ is not None else None,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LinearQPolicy":
        with open(path) as f:
            data = json.load(f)
        policy = cls(
            action_ids=data["action_ids"],
            learning_rate=data["learning_rate"],
            n_iter=data["n_iter"],
            l2=data["l2"],
            loss=data["loss"],
            huber_delta=data.get("huber_delta", 0.5),
            rank_margin=data.get("rank_margin", 0.0),
        )
        if data.get("weights_") is not None:
            policy.weights_ = np.array(data["weights_"], dtype=np.float32)
            policy.bias_ = np.array(data["bias_"], dtype=np.float32)
        return policy


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Ridge Q regression policy (closed-form)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RidgeQPolicy(QPolicyBase):
    """Ridge Q regression using closed-form solution.

    Tune the ridge parameter using DEV only.
    """

    action_ids: list[str] = field(default_factory=list)
    alpha: float = 1.0  # ridge strength

    weights_: np.ndarray = field(default=None, repr=False)
    bias_: np.ndarray = field(default=None, repr=False)

    def fit(self, X: np.ndarray, utilities: np.ndarray) -> "RidgeQPolicy":
        """Fit ridge regression for each action independently."""
        n, d = X.shape
        n_actions = len(self.action_ids)

        # Add bias column
        X_aug = np.hstack([X, np.ones((n, 1), dtype=X.dtype)])
        # Ridge: w = (X^T X + alpha I)^-1 X^T y
        reg = self.alpha * np.eye(d + 1, dtype=X.dtype)
        reg[-1, -1] = 0  # don't regularize bias

        XtX = X_aug.T @ X_aug + reg
        Xty = X_aug.T @ utilities
        W = np.linalg.solve(XtX, Xty)  # [d+1, n_actions]

        self.weights_ = W[:d].T  # [n_actions, d]
        self.bias_ = W[d]        # [n_actions]
        return self

    def fit_with_dev_tuning(
        self,
        X_train: np.ndarray,
        U_train: np.ndarray,
        X_dev: np.ndarray,
        U_dev: np.ndarray,
        alphas: Sequence[float] | None = None,
    ) -> "RidgeQPolicy":
        """Fit with ridge strength tuned on DEV only."""
        if alphas is None:
            alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

        best_alpha = alphas[0]
        best_loss = float("inf")

        for alpha in alphas:
            self.alpha = alpha
            self.fit(X_train, U_train)
            preds = self.predict_utilities(X_dev)
            loss = float(np.mean((preds - U_dev) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha

        # Refit with best alpha
        self.alpha = best_alpha
        self.fit(X_train, U_train)
        return self

    def predict_utilities(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights_.T + self.bias_

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "policy_type": "ridge_q",
                "action_ids": self.action_ids,
                "alpha": self.alpha,
                "weights_": self.weights_.tolist() if self.weights_ is not None else None,
                "bias_": self.bias_.tolist() if self.bias_ is not None else None,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RidgeQPolicy":
        with open(path) as f:
            data = json.load(f)
        policy = cls(
            action_ids=data["action_ids"],
            alpha=data["alpha"],
        )
        if data.get("weights_") is not None:
            policy.weights_ = np.array(data["weights_"], dtype=np.float32)
            policy.bias_ = np.array(data["bias_"], dtype=np.float32)
        return policy


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Small MLP Q regression policy
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MLPQPolicy(QPolicyBase):
    """Small two-layer MLP for Q regression.

    Architecture::
        input 128
        → hidden 128, GELU, dropout
        → hidden 64, GELU
        → 4 Q outputs

    Keep parameter count small. Do not use another transformer.
    Do not use a large router.
    """

    action_ids: list[str] = field(default_factory=list)
    hidden1: int = 128
    hidden2: int = 64
    learning_rate: float = 0.001
    n_iter: int = 2000
    l2: float = 0.001
    dropout: float = 0.1
    seed: int = 42

    # Learned parameters
    W1_: np.ndarray = field(default=None, repr=False)
    b1_: np.ndarray = field(default=None, repr=False)
    W2_: np.ndarray = field(default=None, repr=False)
    b2_: np.ndarray = field(default=None, repr=False)
    W3_: np.ndarray = field(default=None, repr=False)
    b3_: np.ndarray = field(default=None, repr=False)

    def _init_params(self, n_features: int):
        rng = np.random.RandomState(self.seed)
        n_actions = len(self.action_ids)
        # He initialization
        self.W1_ = (rng.randn(n_features, self.hidden1) * np.sqrt(2.0 / n_features)).astype(np.float32)
        self.b1_ = np.zeros(self.hidden1, dtype=np.float32)
        self.W2_ = (rng.randn(self.hidden1, self.hidden2) * np.sqrt(2.0 / self.hidden1)).astype(np.float32)
        self.b2_ = np.zeros(self.hidden2, dtype=np.float32)
        self.W3_ = (rng.randn(self.hidden2, n_actions) * np.sqrt(2.0 / self.hidden2)).astype(np.float32)
        self.b3_ = np.zeros(n_actions, dtype=np.float32)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def _forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        h1 = self._gelu(X @ self.W1_ + self.b1_)
        if training and self.dropout > 0:
            mask = (np.random.rand(*h1.shape) > self.dropout) / (1 - self.dropout)
            h1 = h1 * mask
        h2 = self._gelu(h1 @ self.W2_ + self.b2_)
        out = h2 @ self.W3_ + self.b3_
        return out

    def fit(self, X: np.ndarray, utilities: np.ndarray) -> "MLPQPolicy":
        n, d = X.shape
        self._init_params(d)
        rng = np.random.RandomState(self.seed)

        for epoch in range(self.n_iter):
            # Forward
            h1 = self._gelu(X @ self.W1_ + self.b1_)
            h2 = self._gelu(h1 @ self.W2_ + self.b2_)
            preds = h2 @ self.W3_ + self.b3_

            # MSE loss gradient
            errors = preds - utilities  # [N, n_actions]
            grad_out = (2.0 / n) * errors  # [N, n_actions]

            # Backprop
            grad_W3 = h2.T @ grad_out + self.l2 * self.W3_
            grad_b3 = grad_out.mean(axis=0)
            grad_h2 = grad_out @ self.W3_.T
            grad_h2_pre = grad_h2 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h1 @ self.W2_ + self.b2_))) * 0.5  # approx GELU grad
            grad_h2_pre = grad_h2 * self._gelu_grad(h1 @ self.W2_ + self.b2_)

            grad_W2 = h1.T @ grad_h2_pre + self.l2 * self.W2_
            grad_b2 = grad_h2_pre.mean(axis=0)
            grad_h1 = grad_h2_pre @ self.W2_.T
            grad_h1_pre = grad_h1 * self._gelu_grad(X @ self.W1_ + self.b1_)

            grad_W1 = X.T @ grad_h1_pre + self.l2 * self.W1_
            grad_b1 = grad_h1_pre.mean(axis=0)

            # Update
            self.W3_ -= self.learning_rate * grad_W3
            self.b3_ -= self.learning_rate * grad_b3
            self.W2_ -= self.learning_rate * grad_W2
            self.b2_ -= self.learning_rate * grad_b2
            self.W1_ -= self.learning_rate * grad_W1
            self.b1_ -= self.learning_rate * grad_b1

        return self

    def _gelu_grad(self, x: np.ndarray) -> np.ndarray:
        """GELU gradient (approximate)."""
        return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))) + \
               0.5 * x * (1.0 - np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)) ** 2) * \
               np.sqrt(2.0 / np.pi) * (1.0 + 3 * 0.044715 * x ** 2)

    def predict_utilities(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X, training=False)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "policy_type": "mlp_q",
                "action_ids": self.action_ids,
                "hidden1": self.hidden1,
                "hidden2": self.hidden2,
                "learning_rate": self.learning_rate,
                "n_iter": self.n_iter,
                "l2": self.l2,
                "dropout": self.dropout,
                "seed": self.seed,
                "W1_": self.W1_.tolist() if self.W1_ is not None else None,
                "b1_": self.b1_.tolist() if self.b1_ is not None else None,
                "W2_": self.W2_.tolist() if self.W2_ is not None else None,
                "b2_": self.b2_.tolist() if self.b2_ is not None else None,
                "W3_": self.W3_.tolist() if self.W3_ is not None else None,
                "b3_": self.b3_.tolist() if self.b3_ is not None else None,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MLPQPolicy":
        with open(path) as f:
            data = json.load(f)
        policy = cls(
            action_ids=data["action_ids"],
            hidden1=data["hidden1"],
            hidden2=data["hidden2"],
            learning_rate=data["learning_rate"],
            n_iter=data["n_iter"],
            l2=data["l2"],
            dropout=data["dropout"],
            seed=data["seed"],
        )
        for attr in ["W1_", "b1_", "W2_", "b2_", "W3_", "b3_"]:
            if data.get(attr) is not None:
                setattr(policy, attr, np.array(data[attr], dtype=np.float32))
        return policy


# ──────────────────────────────────────────────────────────────────────
# Section 5 — Surface-feature baselines
# ──────────────────────────────────────────────────────────────────────

def compute_surface_features(
    tasks: list[dict[str, Any]],
    *,
    feature_types: Sequence[str] = ("subtype", "prompt_length", "tfidf"),
    vocabulary: dict | None = None,
) -> np.ndarray:
    """Compute surface-information features for baseline policies.

    Parameters
    ----------
    tasks : list of task dicts
    feature_types : which features to include
    vocabulary : dict | None
        If provided, use this fitted vocabulary (from train) for
        consistent feature dimensions across splits. If None,
        vocabulary is built from the current tasks (train-only use).

    Returns
    -------
    np.ndarray  [N, D]
    """
    features: list[np.ndarray] = []

    if "subtype" in feature_types or "family" in feature_types:
        # Use fitted vocabulary or build from current tasks
        if vocabulary and "subtypes" in vocabulary:
            all_types = vocabulary["subtypes"]
        else:
            all_types = sorted(set(
                t.get("subtype", t.get("family", "unknown")) for t in tasks
            ))
        type_to_idx = {s: i for i, s in enumerate(all_types)}
        n_types = len(all_types)
        onehot = np.zeros((len(tasks), n_types), dtype=np.float32)
        for i, t in enumerate(tasks):
            s = t.get("subtype", t.get("family", "unknown"))
            if s in type_to_idx:
                onehot[i, type_to_idx[s]] = 1.0
        features.append(onehot)

    if "prompt_length" in feature_types:
        lengths = np.array([
            len(t.get("prompt", "")) for t in tasks
        ], dtype=np.float32).reshape(-1, 1)
        max_len = vocabulary.get("max_prompt_length", 0) if vocabulary else 0
        if max_len > 0:
            lengths = lengths / max_len
        elif lengths.max() > 0:
            lengths = lengths / lengths.max()
        features.append(lengths)

    if "char_count" in feature_types:
        chars = np.array([
            len(t.get("prompt", "")) for t in tasks
        ], dtype=np.float32).reshape(-1, 1)
        features.append(chars)

    if "digit_count" in feature_types:
        digits = np.array([
            sum(c.isdigit() for c in t.get("prompt", "")) for t in tasks
        ], dtype=np.float32).reshape(-1, 1)
        features.append(digits)

    if "operator_count" in feature_types:
        ops = np.array([
            sum(c in "+-×÷*/" for c in t.get("prompt", "")) for t in tasks
        ], dtype=np.float32).reshape(-1, 1)
        features.append(ops)

    if "tfidf" in feature_types:
        from collections import Counter
        import math
        docs = [t.get("prompt", "").lower().split() for t in tasks]
        if vocabulary and "tfidf_vocab" in vocabulary:
            vocab = vocabulary["tfidf_vocab"]
            idf = vocabulary["tfidf_idf"]
        else:
            vocab = sorted(set(w for doc in docs for w in doc))
            n_docs = len(docs)
            df = Counter()
            for doc in docs:
                for w in set(doc):
                    df[w] += 1
            idf = {w: math.log(n_docs / (df[w] + 1)) for w in vocab}
        vocab_idx = {w: i for i, w in enumerate(vocab)}
        tfidf = np.zeros((len(tasks), len(vocab)), dtype=np.float32)
        for i, doc in enumerate(docs):
            counts = Counter(doc)
            total = sum(counts.values()) or 1
            for w, c in counts.items():
                if w in vocab_idx:
                    tfidf[i, vocab_idx[w]] = (c / total) * idf.get(w, 0.0)
        features.append(tfidf)

    if not features:
        return np.zeros((len(tasks), 1), dtype=np.float32)

    return np.concatenate(features, axis=1)


class SurfaceFeatureExtractor:
    """Fitted surface feature extractor for consistent train/dev/final features.

    Fit on TRAIN only, then transform any split.
    """

    def __init__(self, feature_types: Sequence[str] = ("subtype", "prompt_length", "tfidf")):
        self.feature_types = tuple(feature_types)
        self.vocabulary_: dict = {}
        self.n_features_: int = 0

    def fit(self, tasks: list[dict[str, Any]]) -> "SurfaceFeatureExtractor":
        """Fit vocabulary on training tasks."""
        from collections import Counter
        import math

        self.vocabulary_ = {}
        if "subtype" in self.feature_types or "family" in self.feature_types:
            self.vocabulary_["subtypes"] = sorted(set(
                t.get("subtype", t.get("family", "unknown")) for t in tasks
            ))
        if "prompt_length" in self.feature_types:
            self.vocabulary_["max_prompt_length"] = max(
                len(t.get("prompt", "")) for t in tasks
            ) or 1
        if "tfidf" in self.feature_types:
            docs = [t.get("prompt", "").lower().split() for t in tasks]
            vocab = sorted(set(w for doc in docs for w in doc))
            n_docs = len(docs)
            df = Counter()
            for doc in docs:
                for w in set(doc):
                    df[w] += 1
            self.vocabulary_["tfidf_vocab"] = vocab
            self.vocabulary_["tfidf_idf"] = {
                w: math.log(n_docs / (df[w] + 1)) for w in vocab
            }
        # Compute n_features
        X = self.transform(tasks)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, tasks: list[dict[str, Any]]) -> np.ndarray:
        return compute_surface_features(
            tasks, feature_types=self.feature_types, vocabulary=self.vocabulary_
        )

    def fit_transform(self, tasks: list[dict[str, Any]]) -> np.ndarray:
        self.fit(tasks)
        return self.transform(tasks)


class SurfaceEnsemblePolicy(QPolicyBase):
    """Canonical surface-ensemble baseline.

    Uses no hidden-state vectors. Combines multiple surface features
    (subtype, prompt TF-IDF, prompt length, digit count, operator count)
    with a ridge Q-regression model.

    This is the strongest surface-only control. Scientific comparison
    should prioritize Hidden > SurfaceEnsemble, not merely
    Hidden > SubtypeOnly.
    """

    def __init__(
        self,
        action_ids: list[str] | None = None,
        alpha: float = 1.0,
        feature_types: Sequence[str] = ("subtype", "prompt_length", "tfidf"),
    ):
        self.action_ids = list(action_ids) if action_ids else []
        self.alpha = alpha
        self.feature_extractor = SurfaceFeatureExtractor(feature_types=feature_types)
        self.ridge: RidgeQPolicy | None = None

    def fit(self, X: np.ndarray, utilities: np.ndarray) -> "SurfaceEnsemblePolicy":
        self.ridge = RidgeQPolicy(action_ids=self.action_ids, alpha=self.alpha)
        self.ridge.fit(X, utilities)
        return self

    def fit_with_tasks(
        self, train_tasks: list[dict[str, Any]], utilities: np.ndarray
    ) -> "SurfaceEnsemblePolicy":
        X = self.feature_extractor.fit_transform(train_tasks)
        self.fit(X, utilities)
        return self

    def predict_utilities(self, X: np.ndarray) -> np.ndarray:
        return self.ridge.predict_utilities(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ridge.predict(X)

    def transform_tasks(self, tasks: list[dict[str, Any]]) -> np.ndarray:
        return self.feature_extractor.transform(tasks)


__all__ = [
    "QPolicyBase",
    "LinearQPolicy",
    "RidgeQPolicy",
    "MLPQPolicy",
    "compute_surface_features",
    "SurfaceFeatureExtractor",
    "SurfaceEnsemblePolicy",
]
