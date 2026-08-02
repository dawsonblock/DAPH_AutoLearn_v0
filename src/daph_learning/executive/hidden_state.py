"""DAPH v0.4 — Hidden state capture for executive qualification (B4).

Captures frozen Qwen3-8B hidden activations for use as executive policy
features. The model is frozen — no fine-tuning. We only extract internal
representations.

Key design decisions (hardened for B4):

1. **Chat template formatting**: Uses the Qwen3 chat template with
   ``add_generation_prompt=True`` so the representation matches what
   the model sees in deployment.

2. **Robust last-token selection**: Uses ``torch.nonzero(attention_mask)``
   instead of ``attention_mask.sum() - 1`` to handle arbitrary padding.

3. **Pinned revision**: The exact HF commit SHA is recorded so the
   model used for capture is identical to the model used for action
   execution.

4. **Staged execution**: Hidden-state capture is designed to run
   AFTER vLLM is stopped, on a separate Stage B process. This avoids
   GPU memory contention.

5. **Raw activation preservation**: Saves raw 4096-dim vectors per
   layer in NPZ format (~40MB for 1280 tasks × 4 layers).

6. **Multiple pooling strategies**: last_token, mean_prompt,
   mean_content — evaluated independently on dev data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# Pinned Qwen3-8B revision (must match vLLM server)
QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


@dataclass
class HiddenStateConfig:
    """Configuration for hidden state capture.

    Attributes
    ----------
    model_name : str
        HuggingFace model name.
    revision : str
        Exact HF commit SHA. Must match the vLLM model.
    layers : list[float]
        Fractional layer positions in [0, 1]. E.g. [0.25, 0.5, 0.75, 1.0].
    pooling : str
        Pooling strategy: "last_token", "mean_prompt", or "mean_content".
    max_length : int
        Maximum token length for tokenization.
    batch_size : int
        Batch size for forward passes.
    dtype : str
        Model dtype: "bfloat16", "float16", or "auto".
    """

    model_name: str = "Qwen/Qwen3-8B"
    revision: str = QWEN3_8B_REVISION
    layers: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    pooling: str = "last_token"
    max_length: int = 512
    batch_size: int = 8
    dtype: str = "bfloat16"


def load_model_for_capture(
    model_name: str,
    revision: str = QWEN3_8B_REVISION,
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    """Load a frozen HuggingFace model for hidden state capture.

    The model is loaded in eval mode with no gradients. This is
    Stage B — vLLM must be stopped first to free GPU memory.

    Returns (model, tokenizer, config_info).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device if device != "cpu" else None,
    )
    model.eval()

    config_info = {
        "model_name": model_name,
        "revision": revision,
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "dtype": str(torch_dtype),
        "device": device,
    }

    return model, tokenizer, config_info


def _apply_chat_template(tokenizer, prompt: str) -> str:
    """Apply the Qwen3 chat template to a prompt.

    This ensures the representation matches what the model sees in
    deployment (vLLM uses chat completions API).
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def _pool_hidden_state(
    hidden: Any,
    attention_mask: Any,
    input_ids: Any,
    pooling: str,
    special_ids: set,
) -> Any:
    """Pool a hidden state tensor according to the pooling strategy.

    Parameters
    ----------
    hidden : torch.Tensor  shape [seq, dim]
        Hidden states for a single sequence.
    attention_mask : torch.Tensor  shape [seq]
        Attention mask for the sequence.
    input_ids : torch.Tensor  shape [seq]
        Input token IDs.
    pooling : str
        "last_token", "mean_prompt", or "mean_content".
    special_ids : set
        Set of special token IDs to exclude for mean_content.

    Returns
    -------
    torch.Tensor  shape [dim]
    """
    import torch

    if pooling == "last_token":
        # Robust last-token selection: find the last non-padding position
        nonzero = torch.nonzero(attention_mask, as_tuple=False)
        if nonzero.numel() == 0:
            return hidden[-1, :]
        last_idx = nonzero[-1].item()
        return hidden[last_idx, :]

    elif pooling == "mean_prompt":
        # Mean over all non-padding tokens
        mask = attention_mask.float().unsqueeze(-1)
        return (hidden * mask).sum(0) / mask.sum().clamp(min=1)

    elif pooling == "mean_content":
        # Mean over non-special, non-padding tokens
        content_mask = attention_mask.clone()
        for t_idx in range(attention_mask.shape[0]):
            if attention_mask[t_idx] == 0:
                continue
            if int(input_ids[t_idx].item()) in special_ids:
                content_mask[t_idx] = 0
        if content_mask.sum() == 0:
            content_mask = attention_mask
        cm = content_mask.float().unsqueeze(-1)
        return (hidden * cm).sum(0) / cm.sum().clamp(min=1)

    else:
        raise ValueError(f"unknown pooling: {pooling}")


def capture_hidden_states(
    tasks: list[dict[str, Any]],
    model,
    tokenizer,
    config: HiddenStateConfig,
    device: str = "cuda",
    save_raw_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Capture hidden states for a list of tasks using a frozen HF model.

    Returns a dict mapping ``"{pooling}/layer_{idx}"`` to arrays of
    shape ``[N, hidden_dim]``.

    If ``save_raw_path`` is provided, also saves raw per-layer activations
    as an NPZ file (one array per layer, shape [N, hidden_dim]).

    Parameters
    ----------
    tasks : list[dict]
        Each task must have a "prompt" field.
    model : HuggingFace model (frozen, eval mode)
    tokenizer : HuggingFace tokenizer
    config : HiddenStateConfig
    device : str
    save_raw_path : str | None
        If provided, save raw activations to this NPZ path.
    """
    import torch

    # Resolve layer indices from fractional positions
    n_layers = model.config.num_hidden_layers
    layer_indices = sorted(set(
        min(int(frac * n_layers), n_layers) for frac in config.layers
    ))

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    # Collect features for each layer × pooling
    pooling_strategies = [config.pooling] if config.pooling != "all" else [
        "last_token", "mean_prompt", "mean_content"
    ]

    results = {f"{p}/layer_{li:02d}": [] for p in pooling_strategies for li in layer_indices}
    raw_activations = {f"layer_{li:02d}": [] for li in layer_indices}

    for i in range(0, len(tasks), config.batch_size):
        batch = tasks[i:i + config.batch_size]
        prompts = [str(t.get("prompt", t.get("specification", ""))) for t in batch]

        # Apply chat template to each prompt
        chat_texts = [_apply_chat_template(tokenizer, p) for p in prompts]

        # Tokenize
        inputs = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for j in range(len(batch)):
            for layer_idx in layer_indices:
                hidden = outputs.hidden_states[layer_idx]  # [batch, seq, dim]

                # Save raw activation (last_token pooling for raw)
                nonzero = torch.nonzero(inputs.attention_mask[j], as_tuple=False)
                last_idx = nonzero[-1].item() if nonzero.numel() > 0 else -1
                raw_vec = hidden[j, last_idx, :].cpu().to(torch.float16).numpy()
                raw_activations[f"layer_{layer_idx:02d}"].append(raw_vec)

                # Apply each pooling strategy
                for pooling in pooling_strategies:
                    h = _pool_hidden_state(
                        hidden[j],
                        inputs.attention_mask[j],
                        inputs.input_ids[j],
                        pooling,
                        special_ids,
                    )
                    vec = h.cpu().to(torch.float16).numpy().astype(np.float16)
                    results[f"{pooling}/layer_{layer_idx:02d}"].append(vec)

        done = min(i + config.batch_size, len(tasks))
        if done % 50 == 0 or done >= len(tasks):
            print(f"    ... captured {done}/{len(tasks)} hidden states "
                  f"(layers={layer_indices}, pooling={pooling_strategies})")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key], dtype=np.float32)

    # Save raw activations
    if save_raw_path:
        save_path = Path(save_raw_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        raw_dict = {k: np.array(v, dtype=np.float16) for k, v in raw_activations.items()}
        np.savez_compressed(save_path, **raw_dict)
        print(f"    Saved raw activations to {save_path} "
              f"({sum(a.nbytes for a in raw_dict.values()) / 1e6:.1f} MB)")

    return results


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
    to get the logprob of each prompt token. Computes 10 statistics.

    This is the B2/B3 feature extractor, kept for ablation comparison.
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
                return np.zeros(10, dtype=np.float32)
            token_logprobs = logprobs_info.get("token_logprobs", [])
            valid_lps = [lp for lp in token_logprobs if lp is not None]
            if not valid_lps:
                return np.zeros(10, dtype=np.float32)
            lps = np.array(valid_lps, dtype=np.float32)
            return np.array([
                len(token_logprobs),
                float(np.mean(lps)),
                float(np.std(lps)),
                float(np.min(lps)),
                float(np.max(lps)),
                float(np.median(lps)),
                float(np.percentile(lps, 25)),
                float(np.percentile(lps, 75)),
                float(valid_lps[0]),
                float(valid_lps[-1]),
            ], dtype=np.float32)
        except Exception:
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
                print(f"    ... captured {completed}/{len(tasks)} logprob features "
                      f"({_time.time() - t0:.1f}s)")
    return np.array(results, dtype=np.float32)


__all__ = [
    "HiddenStateConfig",
    "QWEN3_8B_REVISION",
    "load_model_for_capture",
    "capture_hidden_states",
    "capture_logprob_features",
]
