from __future__ import annotations

import re
from collections import defaultdict

from torch import nn

from daph_ext.repobrain.repo2lora import ModuleShape


_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)"),
)


class CommonDecoderArchitectureAdapter:
    """Checkpoint-introspecting adapter for LLaMA/Mistral/Gemma-style decoders."""

    model_type = "common_decoder"
    known_targets = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    def num_layers(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(config, attr, None) if config is not None else None
            if isinstance(value, int) and value > 0:
                return value
        found = [idx for name, _ in model.named_modules() if (idx := self.layer_index(name)) is not None]
        if not found:
            raise ValueError("could not infer decoder layer count")
        return max(found) + 1

    def hidden_size(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, attr, None) if config is not None else None
            if isinstance(value, int) and value > 0:
                return value
        raise ValueError("could not infer hidden size from config")

    def target_linear_names(self, model: nn.Module) -> tuple[str, ...]:
        available = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
        found = tuple(name for name in self.known_targets if name in available)
        if not found:
            raise ValueError("no recognized decoder projection names found")
        return found

    def layer_index(self, module_name: str) -> int | None:
        for pattern in _LAYER_PATTERNS:
            match = pattern.search(module_name)
            if match:
                return int(match.group(1))
        return None

    def module_shapes(self, model: nn.Module) -> dict[str, ModuleShape]:
        shapes: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or self.layer_index(name) is None:
                continue
            suffix = name.rsplit(".", 1)[-1]
            if suffix in self.known_targets:
                shapes[suffix].add((module.in_features, module.out_features))
        result: dict[str, ModuleShape] = {}
        for suffix, variants in shapes.items():
            if len(variants) != 1:
                raise ValueError(f"inconsistent shapes for {suffix}: {sorted(variants)}")
            in_features, out_features = next(iter(variants))
            result[suffix] = ModuleShape(in_features, out_features)
        if not result:
            raise ValueError("no patchable decoder linear modules discovered")
        return result


class LlamaArchitectureAdapter(CommonDecoderArchitectureAdapter):
    model_type = "llama"


class MistralArchitectureAdapter(CommonDecoderArchitectureAdapter):
    model_type = "mistral"


class GemmaArchitectureAdapter(CommonDecoderArchitectureAdapter):
    model_type = "gemma"
