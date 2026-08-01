from __future__ import annotations

import math

import torch
from torch import Tensor

from daph_ext.geometry.correction import remove_reference_projection
from .materialize import materialize_group_delta
from .repo2lora import LoRAFactors


def _truncated_balanced_factors(
    delta: Tensor,
    rank: int,
    lora_scale: float,
) -> tuple[Tensor, Tensor]:
    """Return B,A such that lora_scale * (B @ A) approximates delta.

    Supports a single dense matrix [Out,In] or a batched matrix
    [B,Out,In]. A balanced truncated SVD split keeps both factors on a
    comparable scale.
    """
    if rank <= 0:
        raise ValueError("rank must be > 0")
    if lora_scale <= 0:
        raise ValueError("LoRA alpha/rank scale must be > 0")
    if delta.ndim not in (2, 3):
        raise ValueError("delta must be [Out,In] or [B,Out,In]")

    work = delta.float()
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    keep = min(rank, s.shape[-1])
    root = torch.sqrt(s[..., :keep].clamp_min(1e-12))
    inv_scale_root = 1.0 / math.sqrt(lora_scale)

    if delta.ndim == 2:
        b = (u[:, :keep] * root.unsqueeze(0)) * inv_scale_root
        a = (root.unsqueeze(1) * vh[:keep, :]) * inv_scale_root
        if keep < rank:
            b = torch.cat([b, torch.zeros(b.shape[0], rank - keep, device=b.device, dtype=b.dtype)], dim=1)
            a = torch.cat([a, torch.zeros(rank - keep, a.shape[1], device=a.device, dtype=a.dtype)], dim=0)
        return b, a

    b = (u[..., :keep] * root.unsqueeze(-2)) * inv_scale_root
    a = (root.unsqueeze(-1) * vh[..., :keep, :]) * inv_scale_root
    if keep < rank:
        b = torch.cat([
            b,
            torch.zeros(*b.shape[:-1], rank - keep, device=b.device, dtype=b.dtype),
        ], dim=-1)
        a = torch.cat([
            a,
            torch.zeros(*a.shape[:-2], rank - keep, a.shape[-1], device=a.device, dtype=a.dtype),
        ], dim=-2)
    return b, a


def project_and_refactor_factors(
    factors: LoRAFactors,
    group_idx: int,
    reference_delta: Tensor,
    strength: float = 1.0,
) -> LoRAFactors:
    """Project a dense group delta and map it back to the existing LoRA rank.

    ``reference_delta`` may be one matrix shared across a batch or a batch of
    matrices matching the generated factor batch.
    """
    if not (0 <= group_idx < factors.groups):
        raise IndexError(group_idx)

    dense_delta = materialize_group_delta(factors, group_idx)
    if dense_delta.ndim == 3 and reference_delta.ndim == 2:
        reference_delta = reference_delta.unsqueeze(0).expand_as(dense_delta)
    if dense_delta.shape != reference_delta.shape:
        raise ValueError("reference_delta must match the materialized delta shape")

    corrected_delta = remove_reference_projection(dense_delta, reference_delta, strength)
    lora_scale = factors.alpha / factors.rank
    b_new, a_new = _truncated_balanced_factors(
        corrected_delta,
        factors.rank,
        lora_scale,
    )

    a_updated = factors.a.clone()
    b_updated = factors.b.clone()
    if factors.is_batched:
        a_updated[:, group_idx] = a_new.to(device=factors.a.device, dtype=factors.a.dtype)
        b_updated[:, group_idx] = b_new.to(device=factors.b.device, dtype=factors.b.dtype)
    else:
        a_updated[group_idx] = a_new.to(device=factors.a.device, dtype=factors.a.dtype)
        b_updated[group_idx] = b_new.to(device=factors.b.device, dtype=factors.b.dtype)
    return LoRAFactors(a=a_updated, b=b_updated, alpha=factors.alpha)
