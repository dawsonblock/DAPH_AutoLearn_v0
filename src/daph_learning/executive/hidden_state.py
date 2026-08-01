"""DAPH v0.4 — Hidden state capture for executive qualification.

Captures hidden states from a HuggingFace model to use as features for
the executive policy. This is the v0.4 equivalent of the
``_capture_hidden_states`` function in ``run_gate_a_staged.py``,
generalized to work with the executive pipeline.

The capture is done separately from LLM generation (which uses the vLLM
API server). The HF model is loaded in-process for a single forward
pass per task to extract hidden states.

Multiple layers and pooling strategies are supported:
  - last_token: hidden state at the last prompt token
  - mean_prompt: mean of all prompt token hidden states
  - mean_content: mean of non-special tokens

Multiple layers can be captured and concatenated to produce richer
features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class HiddenStateConfig:
    """Configuration for hidden state capture.

    Attributes
    ----------
    model_name : str
        HuggingFace model name (must match the vLLM model).
    layers : list[float]
        Fractional layer positions in [0, 1]. E.g. [0.25, 0.5, 0.75, 1.0]
        captures layers at 25%, 50%, 75%, and 100% of the model depth.
    location : str
        Pooling strategy: "last_token", "mean_prompt", or "mean_content".
    max_length : int
        Maximum token length for tokenization.
    batch_size : int
        Batch size for forward passes.
    """

    model_name: str = ""
    layers: list[float] = field(default_factory=lambda: [0.5])
    location: str = "last_token"
    max_length: int = 512
    batch_size: int = 16


def load_model_for_capture(
    model_name: str,
    device: str = "cuda",
    dtype: str = "auto",
):
    """Load a HuggingFace model for hidden state capture.

    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    if dtype == "auto":
        torch_dtype = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device if device != "cpu" else None,
    )
    model.eval()
    return model, tokenizer


def capture_hidden_states(
    tasks: list[dict[str, Any]],
    model,
    tokenizer,
    config: HiddenStateConfig,
    device: str = "cuda",
) -> np.ndarray:
    """Capture hidden states for a list of tasks.

    Parameters
    ----------
    tasks : list[dict]
        Each task must have a "prompt" field.
    model : HuggingFace model
    tokenizer : HuggingFace tokenizer
    config : HiddenStateConfig
    device : str

    Returns
    -------
    np.ndarray  shape [N, D]
        Where D = hidden_dim * n_layers (concatenated).
    """
    import torch

    # Resolve layer indices from fractional positions
    n_layers = model.config.num_hidden_layers
    # +1 because hidden_states includes the embedding layer (index 0)
    layer_indices = sorted(set(
        min(int(frac * n_layers), n_layers) for frac in config.layers
    ))

    features = []
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    for i in range(0, len(tasks), config.batch_size):
        batch = tasks[i:i + config.batch_size]
        prompts = [str(t.get("prompt", t.get("specification", ""))) for t in batch]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for j in range(len(batch)):
            # Collect features from each requested layer
            task_features = []
            for layer_idx in layer_indices:
                hidden = outputs.hidden_states[layer_idx]  # [batch, seq, dim]

                if config.location == "last_token":
                    attn = inputs.attention_mask[j]
                    last_idx = int(attn.sum().item()) - 1
                    h = hidden[j, last_idx, :]
                elif config.location == "mean_prompt":
                    attn = inputs.attention_mask[j].float().unsqueeze(-1)
                    h = (hidden[j] * attn).sum(0) / attn.sum().clamp(min=1)
                elif config.location == "mean_content":
                    attn = inputs.attention_mask[j]
                    content_mask = attn.clone()
                    input_ids = inputs.input_ids[j]
                    for t_idx in range(int(attn.sum().item())):
                        if int(input_ids[t_idx].item()) in special_ids:
                            content_mask[t_idx] = 0
                    if content_mask.sum().item() == 0:
                        content_mask = attn
                    cm = content_mask.float().unsqueeze(-1)
                    h = (hidden[j] * cm).sum(0) / cm.sum().clamp(min=1)
                else:
                    h = hidden[j, -1, :]

                task_features.append(h.cpu().numpy().astype(np.float32))

            # Concatenate features from all layers
            features.append(np.concatenate(task_features))

        done = min(i + config.batch_size, len(tasks))
        if done % 100 == 0 or done >= len(tasks):
            print(f"    ... captured {done}/{len(tasks)} hidden states "
                  f"({len(layer_indices)} layers × {config.location})")

    return np.array(features, dtype=np.float32)


__all__ = [
    "HiddenStateConfig",
    "load_model_for_capture",
    "capture_hidden_states",
]
