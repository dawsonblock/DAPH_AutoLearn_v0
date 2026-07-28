from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from daph_ext.geometry.corrections_v14 import (
    hard_fisher_topk,
    relative_reconstruction_error,
    soft_fisher_attenuation,
    svd_retained_energy,
)
from .materialize import materialize_group_delta
from .refactor import _truncated_balanced_factors
from .repo2lora import LoRAFactors


@dataclass(frozen=True)
class RefactorReport:
    retained_energy: float
    relative_error: float
    spectral_relative_error: float
    frobenius_norm_ratio: float
    corrected_numerical_rank: int
    target_rank: int
    accepted: bool


def correct_and_refactor_group(
    factors: LoRAFactors,
    group_idx: int,
    fisher_diag: Tensor,
    *,
    method: str,
    value: float,
    min_retained_energy: float = 0.95,
    max_relative_error: float = 0.10,
    gamma: float = 1.0,
    normalization: str = "median",
    strict: bool = False,
) -> tuple[LoRAFactors, RefactorReport]:
    """Apply a Fisher correction to one dense group and restore LoRA rank.

    A correction that cannot retain the requested SVD energy is returned with
    ``accepted=False``; callers must not silently deploy it.
    """
    dense = materialize_group_delta(factors, group_idx)
    if dense.ndim != 2:
        raise ValueError("correct_and_refactor_group currently expects one repository (unbatched factors)")
    if fisher_diag.shape != dense.shape:
        raise ValueError("fisher_diag must match materialized group delta")

    if method == "hard_fisher":
        corrected, _ = hard_fisher_topk(dense, fisher_diag, fraction=value)
    elif method == "soft_fisher":
        corrected, _ = soft_fisher_attenuation(
            dense,
            fisher_diag,
            lambda_reg=value,
            gamma=gamma,
            normalization=normalization,
        )
    else:
        raise ValueError("method must be 'hard_fisher' or 'soft_fisher'")

    retained = svd_retained_energy(corrected, factors.rank)
    scale = factors.alpha / factors.rank
    b_new, a_new = _truncated_balanced_factors(corrected, factors.rank, scale)

    a = factors.a.clone()
    b = factors.b.clone()
    a[group_idx] = a_new.to(device=a.device, dtype=a.dtype)
    b[group_idx] = b_new.to(device=b.device, dtype=b.dtype)
    updated = LoRAFactors(a=a, b=b, alpha=factors.alpha)
    reconstructed = materialize_group_delta(updated, group_idx)
    error = relative_reconstruction_error(corrected, reconstructed)
    corrected_f = corrected.float()
    reconstructed_f = reconstructed.float()
    target_norm = torch.linalg.matrix_norm(corrected_f, ord=2).clamp_min(1e-12)
    spectral_error = float((torch.linalg.matrix_norm(corrected_f - reconstructed_f, ord=2) / target_norm).item())
    frob_ratio = float((torch.linalg.vector_norm(reconstructed_f) / torch.linalg.vector_norm(corrected_f).clamp_min(1e-12)).item())
    numerical_rank = int(torch.linalg.matrix_rank(corrected_f).item())
    report = RefactorReport(
        retained_energy=retained, relative_error=error,
        spectral_relative_error=spectral_error, frobenius_norm_ratio=frob_ratio,
        corrected_numerical_rank=numerical_rank, target_rank=factors.rank,
        accepted=(retained >= min_retained_energy and error <= max_relative_error),
    )
    if strict and not report.accepted:
        raise RuntimeError(
            f"corrected adapter failed refactor gate: retained_energy={retained:.6f}, "
            f"relative_error={error:.6f}"
        )
    return updated, report
