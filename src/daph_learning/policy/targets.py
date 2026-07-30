"""v0.3.10.1-alpha — preference target construction (Section 1).

The configuration exposes a *target mode* that selects how the
continuous utility difference ``ΔU`` is converted into a training
target for the routing policy. The previous ``soft_targets: bool``
flag was a *correctness bug*: the trainer always produced
``q_i = sigmoid(ΔU_i / τ)`` regardless of the flag, so "hard
logistic" was never actually hard-label logistic.

This module provides two explicit target modes:

Soft target (Section 1, SOFT)::

    q_i = sigmoid(ΔU_i / τ)

    valid_mask = all True

Hard target (Section 1, HARD)::

    y_i = 1.0  if ΔU_i >  epsilon_gap
    y_i = 0.0  if ΔU_i < -epsilon_gap
    IGNORE / ABSTAIN if |ΔU_i| <= epsilon_gap  (valid_mask = False)

Hard targets carry a ``valid_mask``. The loss must apply this mask:
ties are *ignored*, not coerced to 0.5. Coercing ties to 0.5 trains
the router to output 0.5 on near-ties, which silently changes the
learned policy geometry.

This module is torch-aware: when ``delta_u`` is a ``torch.Tensor``,
both ``targets`` and ``valid_mask`` are returned as torch tensors on
the same device/dtype, suitable for direct use in
:func:`daph_learning.policy.logistic.weighted_policy_loss`.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


class TargetMode(str, Enum):
    """How ``ΔU`` is converted into a preference target.

    SOFT : ``q_i = sigmoid(ΔU_i / τ)`` — continuous soft labels (default).
    HARD : ``y_i ∈ {0, 1}`` with ties (``|ΔU| <= gap``) masked out.
    HARD_GAP : Section 14 — same as HARD but with explicit gap threshold
        selected on dev data and frozen.
    SOFT_TEMPERATURE : Section 14 — same as SOFT but with temperature
        selected on dev data and frozen.
    SIGNAL_TO_NOISE : Section 14 — reliability-weighted targets using
        ``w_i = min(w_max, |ΔU_i| / (σ_i + ε))`` where σ_i is the
        per-example uncertainty.
    """

    SOFT = "soft"
    HARD = "hard"
    HARD_GAP = "hard_gap"
    SOFT_TEMPERATURE = "soft_temperature"
    SIGNAL_TO_NOISE = "signal_to_noise"

    @classmethod
    def from_str(cls, value: str) -> "TargetMode":
        """Parse a target-mode string, raising ``ValueError`` on unknown values."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"unknown target_mode {value!r}; expected one of "
            f"{[m.value for m in cls]}"
        )


def build_preference_targets(
    delta_u,
    mode: "TargetMode | str",
    temperature: float,
    gap_threshold: float,
):
    """Construct preference training targets with a validity mask.

    Parameters
    ----------
    delta_u : torch.Tensor | np.ndarray
        Utility difference ``U_symbolic - U_llm``. Shape ``[N]``.
    mode : TargetMode | str
        ``"soft"`` or ``"hard"``.
    temperature : float
        Soft-target temperature ``τ``. Must be ``> 0`` when mode is SOFT.
        Ignored (but still validated >= 0) when mode is HARD.
    gap_threshold : float
        Hard-mode tie band ``epsilon_gap``. Must be ``>= 0``. Tasks with
        ``|ΔU| <= gap_threshold`` are masked out (not coerced to 0.5).

    Returns
    -------
    targets : torch.Tensor | np.ndarray
        Shape ``[N]``, dtype matching ``delta_u`` (float). For HARD mode,
        masked positions are filled with ``0.5`` so they are harmless if
        the caller forgets to apply the mask — but the mask MUST be
        applied for the loss to be mathematically correct.
    valid_mask : torch.Tensor | np.ndarray
        Shape ``[N]``, dtype ``bool`` (torch.bool for torch inputs,
        np.bool_ for numpy inputs). ``True`` for examples that should
        contribute to the loss. For SOFT mode, all entries are ``True``.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, ``temperature <= 0`` for SOFT mode,
        ``temperature < 0`` for HARD mode, or ``gap_threshold < 0``.
    """
    if isinstance(mode, str):
        mode = TargetMode.from_str(mode)
    if not isinstance(mode, TargetMode):
        raise ValueError(f"mode must be a TargetMode or str, got {type(mode)}")
    if gap_threshold < 0:
        raise ValueError("gap_threshold must be >= 0")
    if temperature <= 0:
        if mode == TargetMode.SOFT:
            raise ValueError("temperature must be > 0 for SOFT target mode")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")

    is_torch = _HAS_TORCH and isinstance(delta_u, torch.Tensor)

    if mode == TargetMode.SOFT:
        if is_torch:
            targets = torch.sigmoid(delta_u / temperature)
            valid_mask = torch.ones_like(
                delta_u, dtype=torch.bool, device=delta_u.device)
            return targets, valid_mask
        du = np.asarray(delta_u, dtype=np.float64)
        targets = 1.0 / (1.0 + np.exp(-du / temperature))
        valid_mask = np.ones(du.shape, dtype=np.bool_)
        return targets.astype(delta_u.dtype if hasattr(delta_u, "dtype") else np.float64), valid_mask

    if mode == TargetMode.HARD:
        if is_torch:
            targets = torch.full_like(
                delta_u, 0.5, dtype=delta_u.dtype, device=delta_u.device)
            valid_mask = delta_u.abs() > gap_threshold
            targets = torch.where(
                delta_u > gap_threshold,
                torch.ones_like(delta_u),
                torch.where(
                    delta_u < -gap_threshold,
                    torch.zeros_like(delta_u),
                    targets,
                ),
            )
            return targets, valid_mask
        du = np.asarray(delta_u, dtype=np.float64)
        targets = np.full(du.shape, 0.5, dtype=np.float64)
        valid_mask = np.abs(du) > gap_threshold
        targets = np.where(
            du > gap_threshold, 1.0,
            np.where(du < -gap_threshold, 0.0, targets),
        )
        out_dtype = delta_u.dtype if hasattr(delta_u, "dtype") else np.float64
        return targets.astype(out_dtype), valid_mask

    raise ValueError(f"Unsupported target mode: {mode}")


# ------------------------------------------------------------------
# Section 14: uncertainty-aware targets
# ------------------------------------------------------------------

def estimate_sigma(
    u_symbolic,
    u_llm,
    *,
    n_replicates: int = 1,
) -> np.ndarray | "torch.Tensor":
    """Section 14 — estimate per-example uncertainty σ_i.

    σ_i = sqrt(Var(U_S) + Var(U_L))

    When only one replicate is available, σ_i is estimated from the
    magnitude of the utility gap itself (larger gaps → lower relative
    uncertainty).

    Parameters
    ----------
    u_symbolic, u_llm : array-like
        Utility arrays for symbolic and LLM backends. Shape ``[N]`` for
        single-replicate, or ``[N, R]`` for R replicates.
    n_replicates : int
        Number of replicates. If 1, uses the gap-based fallback.

    Returns
    -------
    sigma : same type as inputs
        Per-example uncertainty, shape ``[N]``.
    """
    is_torch = _HAS_TORCH and isinstance(u_symbolic, torch.Tensor)
    if is_torch:
        if n_replicates > 1 and u_symbolic.dim() > 1:
            var_s = u_symbolic.var(dim=-1)
            var_l = u_llm.var(dim=-1)
            return torch.sqrt(var_s + var_l + 1e-8)
        # Fallback: σ_i = |ΔU| * 0.1 + ε (proportional to gap).
        du = u_symbolic - u_llm
        return du.abs() * 0.1 + 1e-6

    us = np.asarray(u_symbolic, dtype=np.float64)
    ul = np.asarray(u_llm, dtype=np.float64)
    if n_replicates > 1 and us.ndim > 1:
        var_s = us.var(axis=-1)
        var_l = ul.var(axis=-1)
        return np.sqrt(var_s + var_l + 1e-8)
    du = us - ul
    return np.abs(du) * 0.1 + 1e-6


def build_uncertainty_aware_targets(
    delta_u,
    sigma,
    *,
    mode: TargetMode | str = TargetMode.SIGNAL_TO_NOISE,
    temperature: float = 1.0,
    gap_threshold: float = 0.0,
    w_max: float = 1.0,
    epsilon: float = 1e-6,
):
    """Section 14 — build uncertainty-aware training targets.

    For SIGNAL_TO_NOISE mode::

        w_i = min(w_max, |ΔU_i| / (σ_i + ε))
        q_i = sigmoid(ΔU_i / τ) * w_i  (soft)
        y_i = (ΔU_i > gap) * w_i       (hard)

    The reliability weight ``w_i`` down-weights examples with high
    uncertainty relative to their gap.

    Returns
    -------
    targets, valid_mask, weights : same type as delta_u
    """
    if isinstance(mode, str):
        mode = TargetMode.from_str(mode)
    is_torch = _HAS_TORCH and isinstance(delta_u, torch.Tensor)

    if mode == TargetMode.SIGNAL_TO_NOISE:
        if is_torch:
            w = torch.clamp(
                torch.abs(delta_u) / (sigma + epsilon),
                max=w_max)
            targets = torch.sigmoid(delta_u / temperature) * w
            valid_mask = torch.ones_like(delta_u, dtype=torch.bool)
            return targets, valid_mask, w
        du = np.asarray(delta_u, dtype=np.float64)
        sig = np.asarray(sigma, dtype=np.float64)
        w = np.clip(np.abs(du) / (sig + epsilon), 0.0, w_max)
        targets = (1.0 / (1.0 + np.exp(-du / temperature))) * w
        valid_mask = np.ones(du.shape, dtype=np.bool_)
        return targets, valid_mask, w

    if mode == TargetMode.HARD_GAP:
        # Same as HARD but with explicit frozen gap threshold.
        return build_preference_targets(
            delta_u, TargetMode.HARD, temperature, gap_threshold)

    if mode == TargetMode.SOFT_TEMPERATURE:
        # Same as SOFT but with explicit frozen temperature.
        return build_preference_targets(
            delta_u, TargetMode.SOFT, temperature, gap_threshold)

    raise ValueError(f"Unsupported uncertainty-aware mode: {mode}")


__all__ = [
    "TargetMode",
    "build_preference_targets",
    "build_uncertainty_aware_targets",
    "estimate_sigma",
]
