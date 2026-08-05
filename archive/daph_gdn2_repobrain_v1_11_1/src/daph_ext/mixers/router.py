from __future__ import annotations

from dataclasses import dataclass

from daph_ext.config import MixerKind


@dataclass(frozen=True)
class LayerMixerPlan:
    kinds: tuple[MixerKind, ...]

    def kind_for_layer(self, layer_idx: int) -> MixerKind:
        return self.kinds[layer_idx]


def alternating_gdn2_attention(num_layers: int, every_n_layers: int = 2) -> LayerMixerPlan:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if every_n_layers <= 0:
        raise ValueError("every_n_layers must be positive")
    kinds = tuple(
        MixerKind.GDN2 if layer_idx % every_n_layers == 0 else MixerKind.ATTENTION
        for layer_idx in range(num_layers)
    )
    return LayerMixerPlan(kinds=kinds)


def assert_no_sparse_compaction(original_positions: list[int], compacted_positions: list[int]) -> None:
    """Guard against feeding non-contiguous token subsets into a recurrence.

    Layer-level routing is the supported v1 strategy. A future sparse recurrent
    implementation must preserve skipped-time state transitions explicitly.
    """
    if original_positions != compacted_positions:
        raise RuntimeError(
            "Sparse token compaction is forbidden for recurrent mixers. "
            "Route at layer granularity or implement state-preserving masked updates."
        )
