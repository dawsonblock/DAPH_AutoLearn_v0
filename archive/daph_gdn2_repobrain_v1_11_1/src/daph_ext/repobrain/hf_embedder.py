from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .encoder import RepoPooler


def chunk_token_ranges(length: int, *, chunk_size: int = 4096, overlap: int = 512) -> list[tuple[int, int]]:
    if length < 0:
        raise ValueError("length must be >= 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if length == 0:
        return []
    step = chunk_size - overlap
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(length, start + chunk_size)
        ranges.append((start, end))
        if end == length:
            break
        start += step
    return ranges


@dataclass(frozen=True)
class RepoFile:
    path: str
    text: str
    weight: float = 1.0


class HuggingFaceFileEmbedder:
    """Frozen Hugging Face embedding-model adapter.

    The dependency is intentionally optional. The model is always placed in
    eval mode and all parameters are frozen. Mean pooling uses the attention
    mask unless the model exposes a native sentence embedding.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = False,
        batch_size: int = 8,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("install the 'real' extra to use HuggingFaceFileEmbedder") from exc
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        kwargs = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        if device is not None:
            self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    @torch.no_grad()
    def encode_chunks(self, chunks: Sequence[str]) -> Tensor:
        if not chunks:
            raise ValueError("chunks must be non-empty")
        batch = self.tokenizer(list(chunks), padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)
        if hasattr(out, "sentence_embedding"):
            emb = out.sentence_embedding
        elif hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
            mask = batch.get("attention_mask")
            if mask is None:
                emb = hidden.mean(dim=1)
            else:
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        elif isinstance(out, tuple) and out and isinstance(out[0], Tensor):
            hidden = out[0]
            emb = hidden.mean(dim=1)
        else:  # pragma: no cover
            raise TypeError("embedding model output does not expose usable hidden states")
        return torch.nn.functional.normalize(emb.float(), dim=-1)

    @torch.no_grad()
    def encode_token_id_chunks(self, chunks: Sequence[Sequence[int]]) -> Tensor:
        if not chunks:
            raise ValueError("chunks must be non-empty")
        if any(len(chunk) == 0 for chunk in chunks):
            raise ValueError("token-id chunks must be non-empty")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")

        outputs: list[Tensor] = []
        for start in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[start : start + self.batch_size]
            max_len = max(len(chunk) for chunk in batch_chunks)
            input_ids = torch.full((len(batch_chunks), max_len), int(pad_id), dtype=torch.long, device=self.device)
            attention_mask = torch.zeros_like(input_ids)
            for i, chunk in enumerate(batch_chunks):
                ids = torch.tensor(list(chunk), dtype=torch.long, device=self.device)
                input_ids[i, : ids.numel()] = ids
                attention_mask[i, : ids.numel()] = 1
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(out, "sentence_embedding"):
                emb = out.sentence_embedding
            elif hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
                weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            elif isinstance(out, tuple) and out and isinstance(out[0], Tensor):
                hidden = out[0]
                weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            else:
                raise TypeError("embedding model output does not expose usable hidden states")
            outputs.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0).to(self.device)

    def tokenize_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)


class RepositoryEmbeddingPipeline:
    """Token-aware file chunking and FP32 mean+max repository pooling."""

    def __init__(self, embedder: HuggingFaceFileEmbedder, *, chunk_size: int = 4096, overlap: int = 512) -> None:
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.pooler = RepoPooler()

    @torch.no_grad()
    def encode_file(self, text: str) -> Tensor:
        ids = self.embedder.tokenize_text(text)
        ranges = chunk_token_ranges(len(ids), chunk_size=self.chunk_size, overlap=self.overlap)
        if not ranges:
            raise ValueError("cannot embed an empty file")
        chunks = [ids[s:e] for s, e in ranges]
        chunk_vectors = self.embedder.encode_token_id_chunks(chunks)
        return chunk_vectors.float().mean(dim=0)

    @torch.no_grad()
    def encode_repository(self, files: Sequence[RepoFile]) -> Tensor:
        if not files:
            raise ValueError("repository must contain at least one file")
        vectors = torch.stack([self.encode_file(f.text) for f in files], dim=0)
        weights = torch.tensor([f.weight for f in files], device=vectors.device, dtype=torch.float32)
        pooled = self.pooler(vectors, weights)
        return pooled.unsqueeze(0)
