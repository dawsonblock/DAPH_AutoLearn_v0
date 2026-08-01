from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GeometryMetrics:
    cosine: float
    norm_ratio: float
    sign_conflict: float
    projection_overlap: float
    fisher_energy: float | None = None


def _flat(x: Tensor) -> Tensor:
    return x.detach().float().reshape(-1)


def geometry_metrics(candidate: Tensor, reference: Tensor, fisher_diag: Tensor | None = None) -> GeometryMetrics:
    a, b = _flat(candidate), _flat(reference)
    if a.numel() != b.numel():
        raise ValueError("candidate/reference size mismatch")
    eps = 1e-12
    cosine = torch.dot(a, b) / (a.norm() * b.norm() + eps)
    norm_ratio = a.norm() / (b.norm() + eps)
    active = (a != 0) & (b != 0)
    sign_conflict = ((torch.sign(a[active]) != torch.sign(b[active])).float().mean() if active.any() else torch.tensor(0.0))
    proj = torch.dot(a, b) / (torch.dot(b, b) + eps)
    projection_overlap = (proj * b).norm() / (a.norm() + eps)
    fisher_energy = None
    if fisher_diag is not None:
        f = _flat(fisher_diag)
        if f.numel() != a.numel():
            raise ValueError("fisher size mismatch")
        fisher_energy = float(torch.sum(f.clamp_min(0) * a.square()).item())
    return GeometryMetrics(
        cosine=float(cosine.item()),
        norm_ratio=float(norm_ratio.item()),
        sign_conflict=float(sign_conflict.item()),
        projection_overlap=float(projection_overlap.item()),
        fisher_energy=fisher_energy,
    )
