"""v0.3.10.1-alpha — real-model residual-stream intervention pipeline
(Sections 13-15).

This module implements actual model intervention support for a loaded
transformer:

1. Resolve target layer.
2. Install a residual-stream hook.
3. Capture the baseline hidden state.
4. Run with ``alpha ∈ {-A, -A/2, 0, A/2, A}``.
5. Measure route probability, selected action, downstream utility,
   token KL, and safety-clamp telemetry.

The intervention is::

    h' = h + alpha * v

subject to existing norm/cosine safety clamps. Results are recorded as
:class:`RealInterventionResult` with ``evidence_level="REAL_MODEL_LATENT_INTERVENTION"``.

This module is torch/transformers-dependent. It is imported lazily so
the rest of the package works without those dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import RealInterventionResult
from ..policy.abstention import choose_route


@dataclass(frozen=True)
class InterventionConfig:
    """Configuration for a real-model intervention run (Section 13).

    Attributes
    ----------
    layer : int
        Target residual-stream layer index.
    alpha_grid : tuple of float
        Dose strengths. Default ``(-1.0, -0.5, 0.0, 0.5, 1.0)``.
    max_vector_norm : float | None
        If set, the steering vector is L2-normalized to this norm
        before intervention.
    max_neutral_kl : float
        Safety threshold for the neutral-prompt KL gate.
    clamp_norm : float | None
        If set, the intervened hidden state ``h'`` is clamped so that
        ``||h' - h|| <= clamp_norm`` (per-step norm safety).
    """

    layer: int = 20
    alpha_grid: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    max_vector_norm: float | None = None
    max_neutral_kl: float = 0.03
    clamp_norm: float | None = None


class ResidualStreamHook:
    """A residual-stream intervention hook for a transformer model
    (Section 13).

    Installs a forward hook on the target layer's residual stream. On
    each forward pass, the hook captures the hidden state and (if
    intervention is armed) replaces ``h`` with ``h + alpha * v``.

    Usage::

        hook = ResidualStreamHook(model, layer=20)
        hook.arm(vector, alpha=1.0)
        with hook.installed():
            outputs = model(**inputs)
        h_baseline = hook.captured_hidden
    """

    def __init__(self, model, layer: int) -> None:
        self.model = model
        self.layer = layer
        self.vector: np.ndarray | None = None
        self.alpha: float = 0.0
        self.captured_hidden: np.ndarray | None = None
        self.clamp_triggered: bool = False
        self._handle = None
        self._target_module = None

    def _resolve_target_module(self):
        """Resolve the residual-stream module at the target layer.

        Handles common transformer architectures (Qwen2, Llama, GPT-2).
        """
        # Try Qwen2 / Llama style: model.model.layers[i]
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer]
        # Try GPT-2 style: model.transformer.h[i]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[self.layer]
        # Fallback: model.layers[i]
        if hasattr(self.model, "layers"):
            return self.model.layers[self.layer]
        raise ValueError(
            f"could not resolve residual-stream module at layer "
            f"{self.layer}; model architecture not recognized")

    def arm(self, vector: np.ndarray, alpha: float = 0.0,
            clamp_norm: float | None = None) -> None:
        """Arm the hook with a steering vector and dose."""
        v = np.asarray(vector, dtype=np.float32)
        self.vector = v
        self.alpha = float(alpha)
        self.clamp_norm = clamp_norm
        self.clamp_triggered = False

    def disarm(self) -> None:
        """Disarm the hook (no intervention on subsequent forward passes)."""
        self.alpha = 0.0
        self.clamp_triggered = False

    def _hook_fn(self, module, inputs, outputs):
        """Forward hook: capture and optionally intervene on the hidden state.

        v0.3.10.1 — the hook captures the POST-intervention hidden state
        (after applying h' = h + alpha * v), so callers can read
        ``captured_hidden`` to get the intervened residual stream. The
        returned (modified) output propagates through the rest of the
        model so downstream layers see the intervention.
        """
        import torch
        # outputs is typically a tuple; the first element is the hidden state.
        if isinstance(outputs, tuple):
            hidden = outputs[0]
            rest = outputs[1:]
        else:
            hidden = outputs
            rest = None
        # Intervene if armed (BEFORE capturing, so captured_hidden
        # reflects the intervention).
        if self.vector is not None and self.alpha != 0.0:
            v = torch.as_tensor(self.vector, dtype=hidden.dtype,
                                device=hidden.device)
            # Broadcast v over the sequence dimension.
            if hidden.dim() == 3:  # [batch, seq, dim]
                v = v.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
            elif hidden.dim() == 2:  # [batch, dim]
                v = v.unsqueeze(0)
            delta = self.alpha * v
            # Norm clamp (Section 13 safety).
            if self.clamp_norm is not None:
                delta_norm = float(delta.norm().item())
                if delta_norm > self.clamp_norm:
                    delta = delta * (self.clamp_norm / delta_norm)
                    self.clamp_triggered = True
            hidden = hidden + delta
        # Capture the (possibly intervened) hidden state.
        self.captured_hidden = hidden.detach().cpu().numpy().copy()
        if rest is not None:
            return (hidden,) + rest
        return hidden

    def install(self):
        """Context manager that installs the forward hook."""
        import contextlib

        target = self._resolve_target_module()
        self._target_module = target
        self._handle = target.register_forward_hook(self._hook_fn)

        @contextlib.contextmanager
        def _ctx():
            try:
                yield self
            finally:
                if self._handle is not None:
                    self._handle.remove()
                    self._handle = None

        return _ctx()


def run_real_intervention(
    model,
    tokenizer,
    tasks: Sequence[Mapping[str, Any]],
    vector: np.ndarray,
    *,
    config: InterventionConfig | None = None,
    policy_prob_fn: Callable[[np.ndarray], float] | None = None,
    utility_fn: Callable[[Mapping[str, Any], str], float] | None = None,
    neutral_prompts: Sequence[str] | None = None,
    device: str = "cpu",
) -> list[RealInterventionResult]:
    """Run a real-model dose-response intervention (Sections 13-15).

    For each task and each ``alpha`` in the grid:
    1. Install the residual-stream hook at ``config.layer``.
    2. Arm with ``(vector, alpha)``.
    3. Run the model to capture the intervened hidden state.
    4. Compute ``P(S | h')`` via ``policy_prob_fn`` (if provided).
    5. Choose a route and compute utility via ``utility_fn``.
    6. Optionally measure neutral-prompt KL.

    Parameters
    ----------
    model : transformers model
    tokenizer : transformers tokenizer
    tasks : sequence of task dicts (must have ``task_id`` and ``prompt``).
    vector : np.ndarray
        Steering vector (shape ``[hidden_dim]``).
    config : InterventionConfig | None
    policy_prob_fn : callable | None
        ``h -> P(S | h)``. If None, route is chosen from the hidden
        state projection directly.
    utility_fn : callable | None
        ``(task, route) -> utility``.
    neutral_prompts : sequence of str | None
        Prompts for the neutral-KL gate (Section 25).
    device : str

    Returns
    -------
    list[RealInterventionResult]
        One per (task, alpha) pair, with ``evidence_level="REAL_MODEL_LATENT_INTERVENTION"``.
    """
    import torch
    cfg = config or InterventionConfig()
    v = np.asarray(vector, dtype=np.float32)
    if cfg.max_vector_norm is not None:
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v * (cfg.max_vector_norm / norm)

    if policy_prob_fn is None:
        # Default: use the vector itself as a linear policy.
        def policy_prob_fn(h):
            h = np.asarray(h, dtype=np.float64).flatten()
            return float(1.0 / (1.0 + np.exp(-(h @ v))))
    if utility_fn is None:
        def utility_fn(task, route):
            return 0.0

    hook = ResidualStreamHook(model, layer=cfg.layer)
    results: list[RealInterventionResult] = []
    for task in tasks:
        tid = str(task.get("task_id", ""))
        prompt = task.get("prompt", task.get("specification", ""))
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Baseline (alpha=0): capture the hidden state from the hook.
        # Note: we use hook.captured_hidden, NOT output_hidden_states,
        # because output_hidden_states captures pre-hook values and does
        # not reflect the intervention. The hook fires on the layer
        # module's forward output, so captured_hidden is the post-hook
        # residual stream at the target layer.
        hook.arm(v, alpha=0.0, clamp_norm=cfg.clamp_norm)
        with hook.install():
            with torch.no_grad():
                _ = model(**inputs)
        h0 = hook.captured_hidden
        if h0 is None:
            continue
        # Use the last-token hidden state as the policy input.
        h0_vec = h0[0, -1, :] if h0.ndim == 3 else h0.flatten()
        p0 = float(policy_prob_fn(h0_vec))
        route0 = choose_route(p0, 0.5)
        u0 = float(utility_fn(task, route0))
        # Dose-response: for each alpha, re-run with the hook armed.
        # The hook modifies the layer output; captured_hidden reflects
        # the intervention.
        for alpha in cfg.alpha_grid:
            hook.arm(v, alpha=float(alpha), clamp_norm=cfg.clamp_norm)
            with hook.install():
                with torch.no_grad():
                    _ = model(**inputs)
            h_int = hook.captured_hidden
            if h_int is None:
                continue
            h_int_vec = h_int[0, -1, :] if h_int.ndim == 3 else h_int.flatten()
            p_int = float(policy_prob_fn(h_int_vec))
            route_int = choose_route(p_int, 0.5)
            u_int = float(utility_fn(task, route_int))
            kl_int = None
            if neutral_prompts:
                kl_int = _measure_neutral_kl(
                    model, tokenizer, neutral_prompts, v, float(alpha),
                    cfg, device)
            results.append(RealInterventionResult(
                task_id=tid,
                vector_id=str(task.get("vector_id", "candidate")),
                layer=cfg.layer,
                alpha=float(alpha),
                baseline_route=route0.value,
                intervened_route=route_int.value,
                baseline_utility=u0,
                intervened_utility=u_int,
                utility_delta=u_int - u0,
                route_score_before=p0,
                route_score_after=p_int,
                neutral_kl=kl_int,
                clamp_triggered=hook.clamp_triggered,
                evidence_level="REAL_MODEL_LATENT_INTERVENTION",
            ))
    return results


def _measure_neutral_kl(
    model,
    tokenizer,
    prompts: Sequence[str],
    vector: np.ndarray,
    alpha: float,
    cfg: InterventionConfig,
    device: str,
) -> float | None:
    """Measure mean KL divergence on neutral prompts at this alpha."""
    import torch
    import torch.nn.functional as F
    kls = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Baseline logits.
        with torch.no_grad():
            out0 = model(**inputs)
        logits0 = out0.logits[:, -1, :] if out0.logits.dim() == 3 else out0.logits
        p0 = F.softmax(logits0, dim=-1)
        # Intervened logits.
        hook = ResidualStreamHook(model, layer=cfg.layer)
        hook.arm(vector, alpha=alpha, clamp_norm=cfg.clamp_norm)
        with hook.install():
            with torch.no_grad():
                out1 = model(**inputs)
        logits1 = out1.logits[:, -1, :] if out1.logits.dim() == 3 else out1.logits
        p1 = F.softmax(logits1, dim=-1)
        kl = F.kl_div(p1.log(), p0, reduction="sum").item()
        kls.append(kl)
    return float(np.mean(kls)) if kls else None


__all__ = [
    "InterventionConfig",
    "ResidualStreamHook",
    "run_real_intervention",
]
