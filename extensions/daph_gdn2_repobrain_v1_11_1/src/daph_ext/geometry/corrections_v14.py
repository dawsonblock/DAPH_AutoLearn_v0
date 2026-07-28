from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class CorrectionStats:
    kind: str
    requested_fraction: float | None = None
    actual_fraction: float | None = None
    lambda_reg: float | None = None


def _validate_pair(candidate: Tensor, fisher_diag: Tensor) -> tuple[Tensor, Tensor]:
    if candidate.shape != fisher_diag.shape:
        raise ValueError("candidate and fisher_diag must have identical shapes")
    if not candidate.is_floating_point() or not fisher_diag.is_floating_point():
        raise TypeError("candidate and fisher_diag must be floating point")
    return candidate, fisher_diag.to(device=candidate.device)


def normalize_fisher(
    fisher_diag: Tensor,
    *,
    mode: str = "median",
    eps: float = 1e-12,
) -> Tensor:
    """Normalize Fisher within one tensor so lambda has interpretable scale."""
    f = fisher_diag.detach().float().clamp_min(0)
    positive = f[f > 0]
    if positive.numel() == 0:
        return torch.zeros_like(f)
    if mode == "median":
        denom = positive.median()
    elif mode == "mean":
        denom = positive.mean()
    elif mode == "none":
        denom = torch.tensor(1.0, device=f.device)
    else:
        raise ValueError("mode must be 'median', 'mean', or 'none'")
    return f / denom.clamp_min(eps)


def hard_fisher_topk(
    candidate: Tensor,
    fisher_diag: Tensor,
    *,
    fraction: float,
    positive_only: bool = True,
) -> tuple[Tensor, CorrectionStats]:
    """Prune a deterministic highest-Fisher budget.

    ``positive_only=True`` avoids arbitrarily deleting coordinates when the
    Fisher estimate is exactly zero. Stable sorting makes tie handling
    reproducible on a fixed backend. ``actual_fraction`` records the true
    fraction removed when fewer than the requested number have positive Fisher.
    """
    candidate, fisher_diag = _validate_pair(candidate, fisher_diag)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    n = candidate.numel()
    requested_k = min(n, max(0, int(round(fraction * n))))
    if requested_k == 0:
        return candidate.clone(), CorrectionStats("hard_fisher", fraction, 0.0)
    flat_f = fisher_diag.detach().float().reshape(-1).clamp_min(0)
    if positive_only:
        requested_k = min(requested_k, int((flat_f > 0).sum().item()))
    if requested_k == 0:
        return candidate.clone(), CorrectionStats("hard_fisher", fraction, 0.0)
    indices = torch.argsort(flat_f, descending=True, stable=True)[:requested_k]
    mask = torch.ones(n, device=candidate.device, dtype=candidate.dtype)
    mask[indices.to(mask.device)] = 0
    corrected = candidate.reshape(-1) * mask
    return corrected.reshape_as(candidate), CorrectionStats("hard_fisher", fraction, requested_k / n)


def soft_fisher_attenuation(
    candidate: Tensor,
    fisher_diag: Tensor,
    *,
    lambda_reg: float,
    gamma: float = 1.0,
    normalization: str = "median",
) -> tuple[Tensor, CorrectionStats]:
    """Continuously attenuate sensitive coordinates instead of deleting them."""
    candidate, fisher_diag = _validate_pair(candidate, fisher_diag)
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be >= 0")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    f = normalize_fisher(fisher_diag, mode=normalization).to(candidate.device)
    attenuation = 1.0 / (1.0 + lambda_reg * f.pow(gamma))
    corrected = candidate.float() * attenuation
    return corrected.to(candidate.dtype), CorrectionStats("soft_fisher", lambda_reg=lambda_reg)


def svd_retained_energy(delta: Tensor, rank: int) -> float:
    if delta.ndim != 2:
        raise ValueError("delta must be a matrix")
    if rank <= 0:
        raise ValueError("rank must be > 0")
    s = torch.linalg.svdvals(delta.float())
    denom = s.square().sum().clamp_min(1e-20)
    keep = min(rank, s.numel())
    return float((s[:keep].square().sum() / denom).item())


def relative_reconstruction_error(target: Tensor, reconstructed: Tensor) -> float:
    if target.shape != reconstructed.shape:
        raise ValueError("target/reconstructed shape mismatch")
    num = torch.linalg.vector_norm((target.float() - reconstructed.float()).reshape(-1))
    den = torch.linalg.vector_norm(target.float().reshape(-1)).clamp_min(1e-12)
    return float((num / den).item())
