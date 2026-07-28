from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .base import MixerOutput


class ResidualGDN2Mixer(nn.Module):
    """Identity-preserving wrapper around a GDN2-compatible sequence mixer.

    y = x + tanh(gate) * (GDN2(x) - x)

    With gate_init=0, insertion is exactly identity at initialization, avoiding
    an immediate hidden-state distribution shock to a pretrained Transformer.
    The gate can be trained during continued pretraining/adaptation.
    """

    def __init__(self, mixer: nn.Module, *, gate_init: float = 0.0, trainable: bool = True) -> None:
        super().__init__()
        if not torch.isfinite(torch.tensor(gate_init)):
            raise ValueError("gate_init must be finite")
        self.mixer = mixer
        self.gate = nn.Parameter(torch.tensor([float(gate_init)], dtype=torch.float32), requires_grad=trainable)

    @property
    def mixing_weight(self) -> Tensor:
        return torch.tanh(self.gate)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        recurrent_state: Any | None = None,
        use_cache: bool = False,
    ) -> MixerOutput:
        out = self.mixer(
            hidden_states,
            attention_mask=attention_mask,
            recurrent_state=recurrent_state,
            use_cache=use_cache,
        )
        if not isinstance(out, MixerOutput):
            raise TypeError("wrapped mixer must return MixerOutput")
        weight = self.mixing_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        mixed = hidden_states + weight * (out.hidden_states - hidden_states)
        return MixerOutput(hidden_states=mixed, recurrent_state=out.recurrent_state)
