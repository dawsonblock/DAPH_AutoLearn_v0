from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor, nn


class FileEmbedder(Protocol):
    def encode_chunks(self, chunks: Sequence[str]) -> Tensor: ...


@dataclass(frozen=True)
class FileEmbedding:
    vector: Tensor
    token_count: int
    path_weight: float = 1.0
    distinctiveness: float = 1.0


class RepoPooler(nn.Module):
    """Training-free mean+max repository aggregation with FP32 accumulation.

    Input file vectors are ``[N,D]`` and output is ``[2D]``. Weighted
    accumulation and max pooling are performed in FP32 to avoid half-precision
    overflow/roundoff over large repositories, then cast back to input dtype.
    """

    def forward(
        self,
        file_vectors: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        if file_vectors.ndim != 2:
            raise ValueError("file_vectors must be [N,D]")
        if file_vectors.shape[0] == 0:
            raise ValueError("repository must contain at least one file vector")
        if not file_vectors.is_floating_point():
            raise TypeError("file_vectors must use a floating-point dtype")

        if not torch.isfinite(file_vectors).all():
            raise ValueError("file_vectors contain NaN or Inf")
        original_dtype = file_vectors.dtype
        work = file_vectors.float()
        if weights is None:
            work_weights = torch.ones(
                file_vectors.shape[0],
                device=file_vectors.device,
                dtype=torch.float32,
            )
        else:
            if weights.shape != (file_vectors.shape[0],):
                raise ValueError("weights must be [N]")
            work_weights = weights.to(device=file_vectors.device, dtype=torch.float32)

        if not torch.isfinite(work_weights).all():
            raise ValueError("weights contain NaN or Inf")
        work_weights = work_weights.clamp_min(0)
        denom = work_weights.sum()
        if denom <= 0:
            raise ValueError("at least one repository weight must be > 0")
        mean = (work * work_weights.unsqueeze(-1)).sum(dim=0) / denom
        max_pool = work.max(dim=0).values
        return torch.cat([mean, max_pool], dim=-1).to(dtype=original_dtype)


def chunk_token_ranges(total_tokens: int, chunk_size: int = 4096, overlap: int = 512) -> list[tuple[int, int]]:
    if total_tokens < 0 or chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("invalid chunking arguments")
    if total_tokens == 0:
        return []
    stride = chunk_size - overlap
    ranges = []
    start = 0
    while start < total_tokens:
        end = min(total_tokens, start + chunk_size)
        ranges.append((start, end))
        if end == total_tokens:
            break
        start += stride
    return ranges
