"""v0.3.10.1-alpha — weighted logistic router (Section 1, BASELINE B/F).

This is the **primary policy learner**. It learns a decision boundary
``P(S | h) = σ(w^T h + b)`` from the continuous utility difference
``ΔU``, using either soft or hard preference targets.

v0.3.10.1 — breaking changes (clean break):

* The trainer now accepts a ``target_mode`` (``TargetMode.SOFT`` or
  ``TargetMode.HARD``) and applies a validity mask to the loss. The old
  ``soft_targets: bool`` flag was a *correctness bug*: the trainer
  always produced ``q_i = sigmoid(ΔU_i / τ)`` regardless of the flag,
  so "hard logistic" was never actually hard-label logistic.
* Early stopping is now selectable (``dev_loss`` | ``dev_regret`` |
  ``dev_utility``), default ``dev_regret``. The previous trainer always
  used dev BCE loss even when the config said ``dev_regret``.
* Hard targets carry a ``valid_mask``. Ties (``|ΔU| <= gap_threshold``)
  are *ignored*, not coerced to 0.5. Coercing ties to 0.5 trains the
  router to output 0.5 on near-ties, which silently changes the learned
  policy geometry.

Soft target (Section 1, SOFT)::

    q_i = sigmoid(ΔU_i / τ)
    valid_mask = all True

Hard target (Section 1, HARD)::

    y_i = 1.0  if ΔU_i >  epsilon_gap
    y_i = 0.0  if ΔU_i < -epsilon_gap
    IGNORE      if |ΔU_i| <= epsilon_gap  (valid_mask = False)

Weighted masked binary cross entropy::

    L = - Σ_i ω_i * m_i [ q_i log p_i + (1-q_i) log(1-p_i) ]
        / max(eps, Σ_i ω_i * m_i)

where ``ω_i = f(|ΔU_i|, confidence_i)`` (see
:mod:`daph_learning.policy.weighting`) and ``m_i`` is the target
validity mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False

from .targets import TargetMode, build_preference_targets


# ------------------------------------------------------------------
# Deprecated v0.3.10 helpers — kept for backwards-compat imports in
# existing tests. New code should use :func:`build_preference_targets`
# with an explicit :class:`TargetMode`.
# ------------------------------------------------------------------

class HardTargetMode(str, Enum):
    """Deprecated v0.3.10 hard-target mode enum.

    New code should use :class:`daph_learning.policy.targets.TargetMode`
    and :func:`build_preference_targets`. This enum is kept so existing
    imports do not break.
    """

    SYMBOLIC_LLM = "symbolic_llm"
    SYMBOLIC_LLM_ABSTAIN = "symbolic_llm_abstain"


def soft_preference_target(
    delta_u,
    temperature: float,
):
    """Deprecated: soft target ``q = sigmoid(ΔU / τ)``.

    New code should use
    :func:`daph_learning.policy.targets.build_preference_targets` with
    ``mode=TargetMode.SOFT``. This function is kept as a thin wrapper.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if _HAS_TORCH and isinstance(delta_u, torch.Tensor):
        return torch.sigmoid(delta_u / temperature)
    arr = np.asarray(delta_u, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-arr / temperature))


def hard_preference_target(
    delta_u,
    gap_threshold: float = 0.0,
    mode: "HardTargetMode | str" = HardTargetMode.SYMBOLIC_LLM_ABSTAIN,
):
    """Deprecated: hard target for ablation (Section 6).

    New code should use
    :func:`daph_learning.policy.targets.build_preference_targets` with
    ``mode=TargetMode.HARD``. This function is kept so existing imports
    do not break. It does NOT return a validity mask — the new
    ``build_preference_targets`` does, and the loss MUST apply it.
    """
    if isinstance(mode, str):
        mode = HardTargetMode(mode)
    is_torch = _HAS_TORCH and isinstance(delta_u, torch.Tensor)
    du = delta_u if is_torch else np.asarray(delta_u, dtype=np.float64)
    if mode == HardTargetMode.SYMBOLIC_LLM:
        if is_torch:
            return torch.where(
                du > gap_threshold, 1.0,
                torch.where(
                    du < -gap_threshold, 0.0, float("nan")))
        return np.where(
            du > gap_threshold, 1.0,
            np.where(du < -gap_threshold, 0.0, np.nan))
    # SYMBOLIC_LLM_ABSTAIN: ties -> 0.5 (legacy behaviour, no mask)
    if is_torch:
        return torch.where(
            du > gap_threshold, 1.0,
            torch.where(du < -gap_threshold, 0.0, 0.5))
    return np.where(
        du > gap_threshold, 1.0,
        np.where(du < -gap_threshold, 0.0, 0.5))


# ------------------------------------------------------------------
# Torch-backed logistic router.
# ------------------------------------------------------------------

if _HAS_TORCH:

    class WeightedLogisticRouter(nn.Module):
        """Weighted logistic routing policy (Section 1, BASELINE B/F).

        ``p = P(S | h) = σ(w^T h + b)``

        Trained with weighted masked binary cross entropy against soft
        or hard preference targets (Section 1). The continuous utility
        difference is retained as supervision rather than discarded into
        a hard class label (in SOFT mode).

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

        # --- PolicyModel protocol (Section 3) ----------------------

        def save(self, path: str) -> None:
            """Save the router state to ``path`` (torch state_dict)."""
            import os
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            payload = {
                "policy_type": "logistic",
                "input_dim": self.linear.in_features,
                "state_dict": self.state_dict(),
            }
            torch.save(payload, path)

        @classmethod
        def load(cls, path: str) -> "WeightedLogisticRouter":
            """Load a router from ``path`` (torch state_dict)."""
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("policy_type") != "logistic":
                raise ValueError(
                    f"expected policy_type='logistic', got "
                    f"{payload.get('policy_type')!r}")
            model = cls(int(payload["input_dim"]))
            model.load_state_dict(payload["state_dict"])
            model.eval()
            return model

    def weighted_policy_loss(
        logits: torch.Tensor,
        target_prob: torch.Tensor,
        sample_weight: torch.Tensor,
        valid_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Weighted masked binary cross entropy with logits (Section 1).

        ``L = - Σ_i ω_i * m_i [ q_i log p_i + (1-q_i) log(1-p_i) ]
            / max(eps, Σ_i ω_i * m_i)``

        Parameters
        ----------
        logits : torch.Tensor  shape ``[N]``
        target_prob : torch.Tensor  shape ``[N]``
            Soft or hard preference target.
        sample_weight : torch.Tensor  shape ``[N]``
            Per-example utility weight (>= 0). Zero-weight examples are
            effectively excluded.
        valid_mask : torch.Tensor | None  shape ``[N]`` dtype bool
            Target validity mask. If ``None``, all examples are valid
            (SOFT mode behaviour). For HARD mode, ties are masked out
            so they do not contribute to the loss.
        """
        if logits.shape != target_prob.shape:
            raise ValueError("logits and target_prob must have the same shape")
        if sample_weight.shape != logits.shape:
            raise ValueError(
                "sample_weight must have the same shape as logits")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target_prob,
            reduction="none",
        )
        if valid_mask is None:
            valid_mask = torch.ones_like(
                logits, dtype=torch.bool, device=logits.device)
        elif valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()
        if valid_mask.shape != logits.shape:
            raise ValueError("valid_mask must have the same shape as logits")
        eff_weight = sample_weight * valid_mask.to(sample_weight.dtype)
        denom = eff_weight.sum().clamp_min(1e-8)
        return (loss * eff_weight).sum() / denom

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
        # v0.3.10.1 — selectable early stopping (Section 9).
        early_stopping_metric: str = "dev_regret"
        # v0.3.10.1 — target mode (Section 1).
        target_mode: str = "soft"
        gap_threshold: float = 0.0

    def _evaluate_dev_metric(
        router: "WeightedLogisticRouter",
        metric: str,
        h_dev: torch.Tensor,
        du_dev: torch.Tensor,
        w_dev: torch.Tensor,
        targets_dev: torch.Tensor,
        mask_dev: torch.Tensor,
        *,
        dev_tasks: "Sequence[Mapping[str, Any]] | None" = None,
        utility_fn: "Callable[[Mapping[str, Any], str], float] | None" = None,
        cfg: "LogisticTrainConfig | None" = None,
    ) -> float:
        """Compute the dev metric used for early stopping.

        ``dev_loss``    : weighted masked BCE on dev (lower is better).
        ``dev_regret``  : mean dev regret (lower is better). Requires
                          ``dev_tasks`` and ``utility_fn``.
        ``dev_utility`` : mean selected-route utility (HIGHER is better,
                          so we return ``-utility`` so that "lower is
                          better" is the universal convention here).
        """
        router.eval()
        with torch.no_grad():
            logits = router(h_dev)
            probs = torch.sigmoid(logits)
        if metric == "dev_loss":
            with torch.no_grad():
                dev_loss = weighted_policy_loss(
                    logits, targets_dev, w_dev, mask_dev).item()
            return dev_loss
        if metric in ("dev_regret", "dev_utility"):
            if dev_tasks is None or utility_fn is None:
                # Fall back to dev_loss if regret/utility cannot be
                # computed (no utility_fn). This is the historical
                # behaviour and keeps the trainer usable without a
                # utility_fn; the integrated learner
                # (policy.learner) always supplies one and validates
                # before calling.
                with torch.no_grad():
                    return weighted_policy_loss(
                        logits, targets_dev, w_dev, mask_dev).item()
            from .types import Route
            from .regret import mean_regret
            from .abstention import choose_route
            p_np = probs.cpu().numpy()
            cand_utils = []
            ora_utils = []
            for i, task in enumerate(dev_tasks):
                p = float(p_np[i])
                route = choose_route(p, 0.5)
                cu = utility_fn(task, route)
                cand_utils.append(cu)
                u_sym = utility_fn(task, Route.SYMBOLIC)
                u_llm = utility_fn(task, Route.LLM)
                ora_utils.append(max(u_sym, u_llm))
            cand_utils = np.asarray(cand_utils, dtype=np.float64)
            ora_utils = np.asarray(ora_utils, dtype=np.float64)
            if metric == "dev_regret":
                return float(mean_regret(cand_utils, ora_utils))
            # dev_utility: higher is better → return -utility.
            return float(-cand_utils.mean())
        raise ValueError(
            f"unknown early_stopping_metric {metric!r}; expected "
            f"'dev_loss', 'dev_regret', or 'dev_utility'")

    def train_weighted_logistic_router(
        train_features: np.ndarray,
        train_delta_u: np.ndarray,
        train_weights: np.ndarray,
        *,
        config: "LogisticTrainConfig | None" = None,
        dev_features: np.ndarray | None = None,
        dev_delta_u: np.ndarray | None = None,
        dev_weights: np.ndarray | None = None,
        dev_tasks: "Sequence[Mapping[str, Any]] | None" = None,
        utility_fn: "Callable[[Mapping[str, Any], str], float] | None" = None,
        seed: int = 0,
    ) -> "WeightedLogisticRouter":
        """Train a :class:`WeightedLogisticRouter` on soft or hard targets.

        Parameters
        ----------
        train_features : np.ndarray  shape ``[N, D]``
        train_delta_u : np.ndarray  shape ``[N]``
        train_weights : np.ndarray  shape ``[N]``
            Per-example utility weights (>= 0).
        config : LogisticTrainConfig | None
        dev_features, dev_delta_u, dev_weights : optional dev set
        dev_tasks, utility_fn : required for ``dev_regret`` /
            ``dev_utility`` early stopping (Section 9). If absent with
            those metrics, falls back to ``dev_loss``.
        seed : int
        """
        if not _HAS_TORCH:
            raise ImportError("torch is required to train the logistic router")
        cfg = config or LogisticTrainConfig()
        if cfg.early_stopping_metric not in (
                "dev_loss", "dev_regret", "dev_utility"):
            raise ValueError(
                "early_stopping_metric must be 'dev_loss', 'dev_regret', "
                "or 'dev_utility'")
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
        # v0.3.10.1 — target mode aware target construction with mask.
        targets, mask = build_preference_targets(
            du, cfg.target_mode, cfg.temperature, cfg.gap_threshold)

        has_dev = dev_features is not None and dev_delta_u is not None
        if has_dev:
            h_dev = torch.as_tensor(dev_features, dtype=torch.float32)
            du_dev = torch.as_tensor(dev_delta_u, dtype=torch.float32)
            w_dev = torch.as_tensor(
                dev_weights if dev_weights is not None
                else np.ones_like(dev_delta_u),
                dtype=torch.float32,
            )
            targets_dev, mask_dev = build_preference_targets(
                du_dev, cfg.target_mode, cfg.temperature, cfg.gap_threshold)
        else:
            h_dev = du_dev = w_dev = targets_dev = mask_dev = None

        # For dev_loss, "lower is better" universally. For dev_utility
        # we return -utility so lower is better here too.
        lower_is_better = True

        best_metric = float("inf")
        best_state = None
        patience = 0
        for epoch in range(cfg.max_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(h)
            loss = weighted_policy_loss(logits, targets, w, mask)
            loss.backward()
            optimizer.step()
            if has_dev:
                metric = _evaluate_dev_metric(
                    model, cfg.early_stopping_metric,
                    h_dev, du_dev, w_dev, targets_dev, mask_dev,
                    dev_tasks=dev_tasks, utility_fn=utility_fn, cfg=cfg,
                )
            else:
                metric = loss.item()
            # Universal convention: lower is better (dev_utility is
            # negated inside _evaluate_dev_metric).
            if metric < best_metric - 1e-6:
                best_metric = metric
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
