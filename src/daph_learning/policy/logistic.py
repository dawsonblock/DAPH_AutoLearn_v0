"""v0.3.10 — weighted soft-target logistic router (Section 5, BASELINE B/F).

This is the **primary policy learner** for v0.3.10. It learns a decision
boundary ``P(S | h) = σ(w^T h + b)`` from the continuous utility
difference ``ΔU``, rather than discarding the utility gap into a hard
class label.

Soft target (Section 5)::

    q_i = σ(ΔU_i / τ)

where ``τ`` controls how sharply utility differences become policy
preference:

    ΔU >> 0  =>  q_i ≈ 1   (strongly prefer symbolic)
    ΔU << 0  =>  q_i ≈ 0   (strongly prefer LLM)
    ΔU ≈ 0   =>  q_i ≈ 0.5 (essentially tied)

Weighted binary cross entropy::

    L = - Σ_i ω_i [ q_i log p_i + (1-q_i) log(1-p_i) ]

where ``ω_i = f(|ΔU_i|, confidence_i)``
(see :mod:`daph_learning.policy.weighting`).

Hard-label mode (Section 6) is also provided for ablation comparison::

    symbolic  if ΔU > ε
    llm       if ΔU < -ε
    abstain   if |ΔU| <= ε

The four-way experiment (Section 6) compares:
    hard weighted centroid  |  soft weighted centroid
    hard logistic           |  soft logistic
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


class HardTargetMode(str, Enum):
    """How to convert ``ΔU`` into a hard target for ablation (Section 6)."""

    SYMBOLIC_LLM = "symbolic_llm"  # ΔU > ε → symbolic, ΔU < -ε → llm
    SYMBOLIC_LLM_ABSTAIN = "symbolic_llm_abstain"  # abstain on ties


def soft_preference_target(
    delta_u: "torch.Tensor | np.ndarray | float",
    temperature: float,
) -> "torch.Tensor | np.ndarray | float":
    """Soft preference target: ``q = σ(ΔU / τ)`` (Section 5).

    Parameters
    ----------
    delta_u : array-like or float
        Utility difference ``U_symbolic - U_llm``.
    temperature : float
        Temperature ``τ``. Must be > 0. Smaller ``τ`` => sharper preference.

    Returns
    -------
    array-like
        Soft target probabilities in ``(0, 1)``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if _HAS_TORCH and isinstance(delta_u, torch.Tensor):
        return torch.sigmoid(delta_u / temperature)
    arr = np.asarray(delta_u, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-arr / temperature))


def hard_preference_target(
    delta_u: "torch.Tensor | np.ndarray | float",
    gap_threshold: float = 0.0,
    mode: HardTargetMode = HardTargetMode.SYMBOLIC_LLM_ABSTAIN,
) -> "torch.Tensor | np.ndarray | float":
    """Hard target for ablation (Section 6).

    Returns 1.0 for symbolic, 0.0 for LLM, and 0.5 for abstain (ties).
    In ``SYMBOLIC_LLM`` mode, ties are set to NaN so they can be filtered
    out of the loss.
    """
    is_torch = _HAS_TORCH and isinstance(
        delta_u, torch.Tensor)
    du = delta_u if is_torch else np.asarray(
        delta_u, dtype=np.float64)
    if mode == HardTargetMode.SYMBOLIC_LLM:
        if is_torch:
            result = torch.where(
                du > gap_threshold, 1.0,
                torch.where(
                    du < -gap_threshold, 0.0,
                    float("nan")))
        else:
            result = np.where(
                du > gap_threshold, 1.0,
                np.where(
                    du < -gap_threshold, 0.0,
                    np.nan))
        return result
    # SYMBOLIC_LLM_ABSTAIN: ties -> 0.5
    if is_torch:
        return torch.where(
            du > gap_threshold, 1.0,
            torch.where(
                du < -gap_threshold, 0.0, 0.5))
    return np.where(
        du > gap_threshold, 1.0,
        np.where(
            du < -gap_threshold, 0.0, 0.5))


if _HAS_TORCH:

    class WeightedLogisticRouter(nn.Module):
        """Weighted logistic routing policy (Section 5, BASELINE B/F).

        ``p = P(S | h) = σ(w^T h + b)``

        Trained with weighted binary cross entropy against soft targets
        ``q = σ(ΔU / τ)``. The continuous utility difference is retained
        as supervision rather than discarded into a hard class label.

        Parameters
        ----------
        input_dim : int
            Feature dimension ``D``.
        """

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            if input_dim <= 0:
                raise ValueError("input_dim must be > 0")
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            """Return the logit (pre-sigmoid) for ``P(S | h)``."""
            if h.dim() == 1:
                h = h.unsqueeze(0)
            return self.linear(h).squeeze(-1)

        def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
            """Return ``P(S | h)`` as a probability in ``(0, 1)``."""
            return torch.sigmoid(self.forward(h))

        def predict_route(
            self,
            h: torch.Tensor,
            confidence_threshold: float = 0.5,
        ) -> list[str]:
            """Predict routes with calibrated abstention (Section 10).

            Abstains when ``max(p, 1-p) < confidence_threshold``.
            """
            probs = self.predict_proba(h)
            routes: list[str] = []
            for p in probs:
                conf = float(max(p.item(), 1.0 - p.item()))
                if conf < confidence_threshold:
                    routes.append("abstain")
                elif p.item() >= 0.5:
                    routes.append("symbolic")
                else:
                    routes.append("llm")
            return routes

    def weighted_policy_loss(
        logits: torch.Tensor,
        target_prob: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted binary cross entropy with logits (Section 5).

        ``L = - Σ_i ω_i [ q_i log p_i + (1-q_i) log(1-p_i) ]``

        Uses ``binary_cross_entropy_with_logits`` for numerical stability.
        The loss is a weighted mean (normalized by total weight).
        """
        if logits.shape != target_prob.shape:
            raise ValueError("logits and target_prob must have the same shape")
        if sample_weight.shape != logits.shape:
            raise ValueError(
                "sample_weight must have the same "
                "shape as logits")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target_prob,
            reduction="none",
        )
        denom = sample_weight.sum().clamp_min(1e-8)
        return (loss * sample_weight).sum() / denom

    @dataclass
    class LogisticTrainConfig:
        """Configuration for training the weighted logistic router.

        All hyperparameters that affect the learned policy. Tune on the
        development split only (Section 34) — never on final test.
        """

        temperature: float = 1.0
        weight_decay: float = 1e-4
        max_epochs: int = 200
        learning_rate: float = 0.01
        early_stopping_patience: int = 10
        early_stopping_metric: str = "dev_regret"

    def train_weighted_logistic_router(
        train_features: np.ndarray,
        train_delta_u: np.ndarray,
        train_weights: np.ndarray,
        *,
        config: "LogisticTrainConfig | None" = None,
        dev_features: np.ndarray | None = None,
        dev_delta_u: np.ndarray | None = None,
        dev_weights: np.ndarray | None = None,
        seed: int = 0,
    ) -> "WeightedLogisticRouter":
        """Train a :class:`WeightedLogisticRouter` on soft targets.

        Parameters
        ----------
        train_features : np.ndarray
            Shape ``[N, D]``.
        train_delta_u : np.ndarray
            Shape ``[N]``. Utility differences.
        train_weights : np.ndarray
            Shape ``[N]``. Sample weights (zero-weight examples are
            effectively excluded).
        config : LogisticTrainConfig | None
        dev_* : optional development set for early stopping.
        seed : int
            Random seed for reproducibility (Section 23).
        """
        if not _HAS_TORCH:
            raise ImportError("torch is required to train the logistic router")
        cfg = config or LogisticTrainConfig()
        torch.manual_seed(seed)
        input_dim = train_features.shape[1]
        model = WeightedLogisticRouter(input_dim)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        h = torch.as_tensor(train_features, dtype=torch.float32)
        du = torch.as_tensor(train_delta_u, dtype=torch.float32)
        w = torch.as_tensor(train_weights, dtype=torch.float32)
        q = soft_preference_target(du, cfg.temperature)

        has_dev = dev_features is not None and dev_delta_u is not None
        if has_dev:
            h_dev = torch.as_tensor(dev_features, dtype=torch.float32)
            du_dev = torch.as_tensor(dev_delta_u, dtype=torch.float32)
            w_dev = torch.as_tensor(
                dev_weights if dev_weights is not None
                else np.ones_like(dev_delta_u),
                dtype=torch.float32,
            )
            q_dev = soft_preference_target(du_dev, cfg.temperature)

        best_loss = float("inf")
        best_state = None
        patience = 0
        for epoch in range(cfg.max_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(h)
            loss = weighted_policy_loss(logits, q, w)
            loss.backward()
            optimizer.step()
            # Early stopping on dev loss (or train loss if no dev).
            if has_dev:
                model.eval()
                with torch.no_grad():
                    dev_logits = model(h_dev)
                    dev_loss = weighted_policy_loss(
                        dev_logits, q_dev, w_dev).item()
                metric = dev_loss
            else:
                metric = loss.item()
            if metric < best_loss - 1e-6:
                best_loss = metric
                best_state = {
                    k: v.clone()
                    for k, v in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model


def numpy_logistic_predict(
    weights: np.ndarray,
    bias: float,
    features: np.ndarray,
) -> np.ndarray:
    """Numpy-only logistic prediction: ``σ(w^T h + b)``.

    Provided for environments without torch and for the synthetic
    closed-loop test environment (Section 35).
    """
    logits = features @ np.asarray(weights, dtype=np.float64) + bias
    return 1.0 / (1.0 + np.exp(-logits))


__all__ = [
    "HardTargetMode",
    "LogisticTrainConfig",
    "WeightedLogisticRouter",
    "hard_preference_target",
    "numpy_logistic_predict",
    "soft_preference_target",
    "train_weighted_logistic_router",
    "weighted_policy_loss",
]
