from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor, nn


@dataclass
class MixerOutput:
    hidden_states: Tensor
    recurrent_state: Any | None = None


class SequenceMixer(Protocol):
    def __call__(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        recurrent_state: Any | None = None,
        use_cache: bool = False,
    ) -> MixerOutput: ...


class IdentityMixer(nn.Module):
    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        recurrent_state: Any | None = None,
        use_cache: bool = False,
    ) -> MixerOutput:
        del attention_mask, use_cache
        return MixerOutput(hidden_states=hidden_states, recurrent_state=recurrent_state)
