from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from daph_ext.geometry.adapter_basis import JointAdapterBasis
from .repo2lora import LoRAFactors


@dataclass(frozen=True)
class BasisRepoBrainOutput:
    coefficients: Tensor
    factors_by_module: Mapping[str, LoRAFactors]


class RepoCoefficientPredictor(nn.Module):
    """Repository encoder head that predicts signed coordinates c(repo)."""

    def __init__(
        self,
        repo_embedding_dim: int,
        hidden_dim: int,
        basis_size: int,
        *,
        coefficient_scale_init: float = 0.10,
    ) -> None:
        super().__init__()
        if min(repo_embedding_dim, hidden_dim, basis_size) <= 0:
            raise ValueError("dimensions must be > 0")
        if coefficient_scale_init <= 0:
            raise ValueError("coefficient_scale_init must be > 0")
        self.repo_embedding_dim = repo_embedding_dim
        self.basis_size = basis_size
        self.trunk = nn.Sequential(
            nn.Linear(repo_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, basis_size)
        # FSDP-safe 1D scalar. Small initial coefficients keep adaptation close
        # to the frozen base while still allowing signed basis coordinates.
        self.log_scale = nn.Parameter(
            torch.tensor([float(coefficient_scale_init)], dtype=torch.float32).log()
        )
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, repo_embedding: Tensor) -> Tensor:
        if repo_embedding.ndim != 2 or repo_embedding.shape[-1] != self.repo_embedding_dim:
            raise ValueError(
                f"repo_embedding must be [B,{self.repo_embedding_dim}], got {tuple(repo_embedding.shape)}"
            )
        h = self.trunk(repo_embedding)
        return torch.tanh(self.head(h)) * self.log_scale.exp()


class BasisRepoBrain(nn.Module):
    """RepoBrain constrained to a reusable ExFusion-derived capability basis.

    repository -> c(repo) -> sum_k c_k U_k -> runtime LoRA factors
    """

    def __init__(
        self,
        basis: JointAdapterBasis,
        *,
        repo_embedding_dim: int,
        hidden_dim: int = 512,
        coefficient_scale_init: float = 0.10,
        train_basis: bool = False,
    ) -> None:
        super().__init__()
        self.basis = basis
        for p in self.basis.parameters():
            p.requires_grad_(train_basis)
        self.predictor = RepoCoefficientPredictor(
            repo_embedding_dim,
            hidden_dim,
            basis.basis_size,
            coefficient_scale_init=coefficient_scale_init,
        )

    def forward(self, repo_embedding: Tensor) -> BasisRepoBrainOutput:
        coefficients = self.predictor(repo_embedding)
        factors = self.basis.compose(coefficients)
        return BasisRepoBrainOutput(coefficients=coefficients, factors_by_module=factors)

    def forward_factors(self, repo_embedding: Tensor) -> dict[str, LoRAFactors]:
        return dict(self.forward(repo_embedding).factors_by_module)
