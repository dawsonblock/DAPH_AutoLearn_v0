from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

import torch
from torch import Tensor, nn

from daph_ext.config import Repo2LoRAConfig


@dataclass(frozen=True)
class ModuleShape:
    in_features: int
    out_features: int


@dataclass
class LoRAFactors:
    """Generated LoRA factors.

    Single-repository contract:
      A [G, R, In], B [G, Out, R]

    Batched-repository contract:
      A [B, G, R, In], B [B, G, Out, R]
    """

    a: Tensor
    b: Tensor
    alpha: float

    def __post_init__(self) -> None:
        if self.a.ndim not in (3, 4) or self.b.ndim != self.a.ndim:
            raise ValueError("LoRA factors must both be rank-3 or rank-4 tensors")
        if any(dim <= 0 for dim in self.a.shape) or any(dim <= 0 for dim in self.b.shape):
            raise ValueError("LoRA factor dimensions must be non-zero")
        if self.a.shape[:-2] != self.b.shape[:-2]:
            raise ValueError("LoRA A/B batch and group dimensions must match")
        if self.a.shape[-2] != self.b.shape[-1]:
            raise ValueError("LoRA A/B rank dimensions must match")
        if not self.a.is_floating_point() or not self.b.is_floating_point():
            raise TypeError("LoRA factors must be floating point")
        if self.a.device != self.b.device:
            raise ValueError("LoRA A/B factors must be on the same device")
        if not math.isfinite(float(self.alpha)) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and > 0")

    @property
    def rank(self) -> int:
        return self.a.shape[-2]

    @property
    def is_batched(self) -> bool:
        return self.a.ndim == 4

    @property
    def batch_size(self) -> int:
        return self.a.shape[0] if self.is_batched else 1

    @property
    def groups(self) -> int:
        return self.a.shape[1] if self.is_batched else self.a.shape[0]


class _FactorHead(nn.Module):
    """Emit grouped LoRA factors for one or more repository embeddings.

    The learnable initial magnitude is applied to B only so the effective dense
    delta is scaled once, not squared through both factors.
    """

    def __init__(
        self,
        hidden_dim: int,
        groups: int,
        rank: int,
        shape: ModuleShape,
        scale_init: float,
    ) -> None:
        super().__init__()
        if scale_init <= 0:
            raise ValueError("scale_init must be > 0")
        self.groups = groups
        self.rank = rank
        self.shape = shape
        self.a_head = nn.Linear(hidden_dim, groups * rank * shape.in_features)
        self.b_head = nn.Linear(hidden_dim, groups * shape.out_features * rank)
        self.log_scale = nn.Parameter(torch.tensor([float(scale_init)], dtype=torch.float32).log())
        nn.init.zeros_(self.a_head.bias)
        nn.init.zeros_(self.b_head.bias)

    def forward(self, h: Tensor, alpha: float) -> LoRAFactors:
        if h.ndim != 2 or h.shape[0] < 1:
            raise ValueError("Repo2LoRA expects input tensor shape [B, hidden_dim] with B >= 1")
        bsz = h.shape[0]
        a = torch.tanh(self.a_head(h)).view(
            bsz,
            self.groups,
            self.rank,
            self.shape.in_features,
        )
        b = torch.tanh(self.b_head(h)).view(
            bsz,
            self.groups,
            self.shape.out_features,
            self.rank,
        )
        b = b * self.log_scale.exp()

        # Preserve the original v1.x single-repository tensor contract.
        if bsz == 1:
            a = a.squeeze(0)
            b = b.squeeze(0)
        return LoRAFactors(a=a, b=b, alpha=alpha)


class Repo2LoRALite(nn.Module):
    """Small grouped hypernetwork for repository-conditioned LoRA generation."""

    def __init__(self, cfg: Repo2LoRAConfig, module_shapes: Mapping[str, ModuleShape]) -> None:
        super().__init__()
        missing = set(cfg.target_modules) - set(module_shapes)
        if missing:
            raise ValueError(f"missing shapes for target modules: {sorted(missing)}")
        self.cfg = cfg
        self.trunk = nn.Sequential(
            nn.Linear(cfg.repo_embedding_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.heads = nn.ModuleDict(
            {
                name: _FactorHead(
                    cfg.hidden_dim,
                    cfg.layer_groups,
                    cfg.rank,
                    module_shapes[name],
                    cfg.output_scale_init,
                )
                for name in cfg.target_modules
            }
        )

    def forward(self, repo_embedding: Tensor) -> dict[str, LoRAFactors]:
        if (
            repo_embedding.ndim != 2
            or repo_embedding.shape[0] < 1
            or repo_embedding.shape[-1] != self.cfg.repo_embedding_dim
        ):
            raise ValueError(
                f"repo_embedding must be [B,{self.cfg.repo_embedding_dim}] with B >= 1, "
                f"got {tuple(repo_embedding.shape)}"
            )
        h = self.trunk(repo_embedding)
        return {name: head(h, self.cfg.alpha) for name, head in self.heads.items()}


def layer_to_group(layer_idx: int, num_layers: int, groups: int) -> int:
    if not (0 <= layer_idx < num_layers):
        raise IndexError(layer_idx)
    if groups <= 0 or groups > num_layers:
        raise ValueError("groups must be in [1,num_layers]")
    return min(groups - 1, (layer_idx * groups) // num_layers)
