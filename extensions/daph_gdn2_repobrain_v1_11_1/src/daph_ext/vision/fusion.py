from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StructuralSemanticFusion(nn.Module):
    """Fuse semantic and structural tokens into the DAPH hidden size.

    Different visual encoders often expose different token-grid lengths. v1.2
    resamples the structural sequence to the semantic sequence length before the
    gated fusion bridge. This is sequence interpolation, not geometric 2-D grid
    registration; callers that know both HxW grids should preferably align them
    spatially before this module.
    """

    def __init__(self, semantic_dim: int, structural_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.semantic_proj = nn.Linear(semantic_dim, hidden_size)
        self.structural_proj = nn.Linear(structural_dim, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, semantic_tokens: Tensor, structural_tokens: Tensor) -> Tensor:
        if semantic_tokens.ndim != 3 or structural_tokens.ndim != 3:
            raise ValueError("semantic_tokens and structural_tokens must both be [B,N,D]")
        if semantic_tokens.shape[0] != structural_tokens.shape[0]:
            raise ValueError("batch size of semantic and structural tokens must match")
        if semantic_tokens.shape[1] == 0 or structural_tokens.shape[1] == 0:
            raise ValueError("token sequences must be non-empty")

        if semantic_tokens.shape[1] != structural_tokens.shape[1]:
            target_tokens = semantic_tokens.shape[1]
            original_dtype = structural_tokens.dtype
            work = structural_tokens.transpose(1, 2)
            # CPU linear interpolation is not uniformly implemented for every
            # half dtype; FP32 also gives more stable resampling.
            if work.dtype in (torch.float16, torch.bfloat16):
                work = work.float()
            structural_tokens = F.interpolate(
                work, size=target_tokens, mode="linear", align_corners=False
            ).transpose(1, 2).to(dtype=original_dtype)

        s = self.semantic_proj(semantic_tokens)
        d = self.structural_proj(structural_tokens)
        pair = torch.cat([s, d], dim=-1)
        gate = torch.sigmoid(self.gate(pair))
        fused = self.out(pair)
        return gate * fused + (1.0 - gate) * s
