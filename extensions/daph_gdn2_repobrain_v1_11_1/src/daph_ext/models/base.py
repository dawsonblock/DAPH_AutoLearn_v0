from __future__ import annotations

from typing import Protocol

from torch import nn

from daph_ext.repobrain.repo2lora import ModuleShape


class ModelArchitectureAdapter(Protocol):
    """Architecture contract used by RepoBrain patching.

    Implementations must derive facts from the loaded checkpoint instead of
    silently assuming one model-family naming convention.
    """

    model_type: str

    def num_layers(self, model: nn.Module) -> int: ...
    def hidden_size(self, model: nn.Module) -> int: ...
    def target_linear_names(self, model: nn.Module) -> tuple[str, ...]: ...
    def layer_index(self, module_name: str) -> int | None: ...
    def module_shapes(self, model: nn.Module) -> dict[str, ModuleShape]: ...
