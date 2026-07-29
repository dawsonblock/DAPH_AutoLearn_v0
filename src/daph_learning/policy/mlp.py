"""v0.3.10.1-alpha — small nonlinear MLP router (Section 30).

A diagnostic baseline, not a production policy. Its purpose is to
answer: *if MLP >> logistic, routing geometry is nonlinear.*

Trained with the same targets, weights, dev regret, and calibration
path as the logistic router (Section 1). Only enabled by explicit
config (``policy_type="mlp_experimental"``); never the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False

from .logistic import (
    LogisticTrainConfig,
    _evaluate_dev_metric,
    weighted_policy_loss,
)
from .targets import TargetMode, build_preference_targets


if _HAS_TORCH:

    class SmallMLPRouter(nn.Module):
        """Small MLP routing policy (Section 30, experimental).

        ``p = P(S | h) = sigmoid(MLP(h))`` where
        ``MLP(h) = Linear(GELU(Linear(h)))``.

        Parameters
        ----------
        input_dim : int
        hidden_dim : int
            Default 64 (Section 30).
        """

        def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
            super().__init__()
            if input_dim <= 0:
                raise ValueError("input_dim must be > 0")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be > 0")
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            """Return the logit (pre-sigmoid) for ``P(S | h)``."""
            if h.dim() == 1:
                h = h.unsqueeze(0)
            return self.net(h).squeeze(-1)

        def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
            """Return ``P(S | h)`` as a probability in ``(0, 1)``."""
            return torch.sigmoid(self.forward(h))

        def predict_route(
            self,
            h: torch.Tensor,
            confidence_threshold: float = 0.5,
        ) -> list[str]:
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
            import os
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            payload = {
                "policy_type": "mlp_experimental",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "state_dict": self.state_dict(),
            }
            torch.save(payload, path)

        @classmethod
        def load(cls, path: str) -> "SmallMLPRouter":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("policy_type") != "mlp_experimental":
                raise ValueError(
                    f"expected policy_type='mlp_experimental', got "
                    f"{payload.get('policy_type')!r}")
            model = cls(int(payload["input_dim"]), int(payload["hidden_dim"]))
            model.load_state_dict(payload["state_dict"])
            model.eval()
            return model

    @dataclass
    class MLPTrainConfig(LogisticTrainConfig):
        """Configuration for training the small MLP router.

        Inherits all fields from :class:`LogisticTrainConfig` (target
        mode, early stopping metric, etc.) and adds ``hidden_dim``.
        """

        hidden_dim: int = 64

    def train_small_mlp_router(
        train_features: np.ndarray,
        train_delta_u: np.ndarray,
        train_weights: np.ndarray,
        *,
        config: "MLPTrainConfig | None" = None,
        dev_features: np.ndarray | None = None,
        dev_delta_u: np.ndarray | None = None,
        dev_weights: np.ndarray | None = None,
        dev_tasks=None,
        utility_fn=None,
        confidence_threshold: float = 0.5,
        seed: int = 0,
    ) -> "SmallMLPRouter":
        """Train a :class:`SmallMLPRouter` on soft or hard targets.

        Mirrors :func:`train_weighted_logistic_router` but uses the MLP
        architecture. Same target/weight/dev-regret/calibration path.
        """
        if not _HAS_TORCH:
            raise ImportError("torch is required to train the MLP router")
        cfg = config or MLPTrainConfig()
        if cfg.early_stopping_metric not in (
                "dev_loss", "dev_regret", "dev_utility"):
            raise ValueError(
                "early_stopping_metric must be 'dev_loss', 'dev_regret', "
                "or 'dev_utility'")
        torch.manual_seed(seed)
        input_dim = train_features.shape[1]
        model = SmallMLPRouter(input_dim, hidden_dim=cfg.hidden_dim)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        h = torch.as_tensor(train_features, dtype=torch.float32)
        du = torch.as_tensor(train_delta_u, dtype=torch.float32)
        w = torch.as_tensor(train_weights, dtype=torch.float32)
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
                    confidence_threshold=confidence_threshold,
                )
            else:
                metric = loss.item()
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


__all__ = ["SmallMLPRouter"]
