from __future__ import annotations

from torch import Tensor, nn


class RepoEvolutionState(nn.Module):
    """Compact recurrent state over commit/diff embeddings.

    Accepts either unbatched vectors ``[D]`` or batched embeddings ``[B,D]``.
    Internally GRU execution is normalized to 2-D tensors so behavior is stable
    across PyTorch versions and external wrappers.
    """

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or state_dim <= 0:
            raise ValueError("input_dim and state_dim must be > 0")
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.in_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, state_dim))
        self.init_proj = nn.Linear(input_dim, state_dim)
        self.gru = nn.GRUCell(state_dim, state_dim)

    def initialize(self, snapshot_embedding: Tensor) -> Tensor:
        if snapshot_embedding.ndim not in (1, 2) or snapshot_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"snapshot_embedding must be [{self.input_dim}] or [B,{self.input_dim}], "
                f"got {tuple(snapshot_embedding.shape)}"
            )
        unbatched = snapshot_embedding.ndim == 1
        x = snapshot_embedding.unsqueeze(0) if unbatched else snapshot_embedding
        state = self.init_proj(x).tanh()
        return state.squeeze(0) if unbatched else state

    def update(self, diff_embedding: Tensor, state: Tensor) -> Tensor:
        if diff_embedding.ndim not in (1, 2) or diff_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"diff_embedding must be [{self.input_dim}] or [B,{self.input_dim}], "
                f"got {tuple(diff_embedding.shape)}"
            )
        if state.ndim not in (1, 2) or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state must be [{self.state_dim}] or [B,{self.state_dim}], got {tuple(state.shape)}"
            )
        diff_unbatched = diff_embedding.ndim == 1
        state_unbatched = state.ndim == 1
        if diff_unbatched != state_unbatched:
            raise ValueError("diff_embedding and state must either both be batched or both be unbatched")

        x = diff_embedding.unsqueeze(0) if diff_unbatched else diff_embedding
        s = state.unsqueeze(0) if state_unbatched else state
        if x.shape[0] != s.shape[0]:
            raise ValueError(f"batch mismatch: diff batch={x.shape[0]} state batch={s.shape[0]}")
        out = self.gru(self.in_proj(x), s)
        return out.squeeze(0) if diff_unbatched else out
