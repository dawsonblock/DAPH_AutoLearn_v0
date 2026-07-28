from __future__ import annotations

from torch import nn

from .base import ModelArchitectureAdapter
from .common_decoder import GemmaArchitectureAdapter, LlamaArchitectureAdapter, MistralArchitectureAdapter
from .qwen import QwenArchitectureAdapter


def get_architecture_adapter(model: nn.Module) -> ModelArchitectureAdapter:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type.startswith("qwen"):
        return QwenArchitectureAdapter()
    if model_type in {"llama", "mllama"}:
        return LlamaArchitectureAdapter()
    if model_type.startswith("mistral") or model_type == "mixtral":
        return MistralArchitectureAdapter()
    if model_type.startswith("gemma"):
        return GemmaArchitectureAdapter()
    raise ValueError(
        f"unsupported model_type={model_type!r}; supported families are Qwen, LLaMA, Mistral/Mixtral, and Gemma"
    )
