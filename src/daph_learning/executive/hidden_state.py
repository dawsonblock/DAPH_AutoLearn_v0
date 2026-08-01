"""DAPH v0.4 — Hidden state capture for executive qualification.

Captures features from tasks to use as input for the executive policy.

Two capture modes are supported:

1. **HF model mode** (``capture_hidden_states``): Loads a HuggingFace
   model in-process and runs a forward pass to extract hidden states.
   Requires GPU memory separate from vLLM.

2. **vLLM logprob mode** (``capture_logprob_features``): Uses the vLLM
   completions API with ``echo=true`` to get prompt token logprobs.
   Computes statistics (mean, std, min, max, percentiles) from these
   logprobs as features. No additional GPU memory needed — works
   alongside a running vLLM server.

The logprob mode is preferred when the GPU is already fully utilized by
vLLM, as it requires no additional model loading.
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
    """Capture hidden states for a list of tasks using a HF model.

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


def capture_logprob_features(
    tasks: list[dict[str, Any]],
    *,
    vllm_base_url: str = "http://localhost:8000",
    vllm_api_key: str = "",
    model_name: str = "",
    max_tokens: int = 0,
    batch_size: int = 32,
) -> np.ndarray:
    """Capture features from vLLM prompt logprobs.

    Uses the vLLM completions API with ``echo=true`` and ``logprobs=1``
    to get the logprob of each prompt token. Computes statistics from
    these logprobs as features.

    This approach requires no additional GPU memory — it uses the
    already-running vLLM server.

    Features extracted (10 per task):
    - n_tokens: number of prompt tokens
    - mean_logprob: mean logprob across prompt tokens
    - std_logprob: std of logprobs
    - min_logprob: minimum logprob (most surprising token)
    - max_logprob: maximum logprob (most expected token)
    - median_logprob: median logprob
    - p25_logprob: 25th percentile
    - p75_logprob: 75th percentile
    - first_logprob: logprob of first content token
    - last_logprob: logprob of last prompt token

    Parameters
    ----------
    tasks : list[dict]
        Each task must have a "prompt" field.
    vllm_base_url : str
        Base URL of the vLLM server.
    vllm_api_key : str
        API key for the vLLM server.
    model_name : str
        Model name to pass to vLLM.
    max_tokens : int
        Maximum tokens to generate (0 = just echo prompt).
    batch_size : int
        Number of concurrent requests.

    Returns
    -------
    np.ndarray  shape [N, 10]
        Feature array.
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _time

    headers = {
        "Authorization": f"Bearer {vllm_api_key}",
        "Content-Type": "application/json",
    }

    def _capture_one(task):
        prompt = str(task.get("prompt", task.get("specification", "")))
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "echo": True,
            "logprobs": 1,
            "temperature": 0.0,
        }
        try:
            resp = requests.post(
                f"{vllm_base_url}/v1/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            data = resp.json()
            choice = data["choices"][0]
            logprobs_info = choice.get("logprobs")

            if logprobs_info is None:
                return _default_features()

            token_logprobs = logprobs_info.get("token_logprobs", [])
            # Filter out None values (first token has no logprob)
            valid_lps = [lp for lp in token_logprobs if lp is not None]

            if not valid_lps:
                return _default_features()

            lps = np.array(valid_lps, dtype=np.float32)
            features = np.array([
                len(token_logprobs),           # n_tokens
                float(np.mean(lps)),            # mean_logprob
                float(np.std(lps)),             # std_logprob
                float(np.min(lps)),             # min_logprob
                float(np.max(lps)),             # max_logprob
                float(np.median(lps)),          # median_logprob
                float(np.percentile(lps, 25)),  # p25_logprob
                float(np.percentile(lps, 75)),  # p75_logprob
                float(valid_lps[0]),            # first_logprob
                float(valid_lps[-1]),           # last_logprob
            ], dtype=np.float32)
            return features
        except Exception:
            return _default_features()

    def _default_features():
        return np.zeros(10, dtype=np.float32)

    features = []
    t0 = _time.time()

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(_capture_one, t): i for i, t in enumerate(tasks)}
        results = [None] * len(tasks)
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                elapsed = _time.time() - t0
                print(f"    ... captured {completed}/{len(tasks)} logprob features "
                      f"({elapsed:.1f}s)")

    return np.array(results, dtype=np.float32)


__all__ = [
    "HiddenStateConfig",
    "load_model_for_capture",
    "capture_hidden_states",
    "capture_logprob_features",
]
