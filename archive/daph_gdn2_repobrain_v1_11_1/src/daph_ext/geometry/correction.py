from __future__ import annotations

from torch import Tensor


def remove_reference_projection(
    candidate: Tensor,
    reference: Tensor,
    strength: float = 1.0,
) -> Tensor:
    """Remove a configurable component parallel to ``reference``.

    Supports one matrix/tensor or a leading batch dimension. For batched dense
    deltas [B,Out,In], projection coefficients are computed independently per
    batch element rather than across repositories.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0,1]")
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have identical shapes")
    if candidate.ndim < 1:
        raise ValueError("candidate/reference must have at least one dimension")

    a = candidate.float()
    b = reference.float()
    if a.ndim <= 2:
        denom = (b * b).sum().clamp_min(1e-12)
        projection = ((a * b).sum() / denom) * b
    else:
        reduce_dims = tuple(range(1, a.ndim))
        denom = (b * b).sum(dim=reduce_dims, keepdim=True).clamp_min(1e-12)
        coeff = (a * b).sum(dim=reduce_dims, keepdim=True) / denom
        projection = coeff * b
    return (a - strength * projection).to(candidate.dtype)
