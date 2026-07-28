from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Literal, Sequence

from daph_learning.routing.errors import ContextBoundaryError, MultiTokenRouteError
from daph_learning.routing.steered_router import (
    RouteAction,
    detect_route_prompt_alignment,
    resolve_route_token_ids,
    resolve_route_token_ids_contextual,
    route_action_from_logits,
)
from daph_learning.steering.anchors import find_anchor_token_index
from daph_learning.steering.hooks import (
    multi_layer_residual_addition_hook,
    residual_addition_hook,
)

TokenResolver = Literal["isolated", "contextual"]
SequenceNormalization = Literal["sum", "mean"]


def _tokenize_route_batch(
    rendered_prompts: Sequence[str],
    tokenizer: Any,
    *,
    anchor: str,
):
    """Tokenize a left-padded batch and map per-example anchor positions."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs pad_token_id or eos_token_id for batching")
        tokenizer.pad_token = tokenizer.eos_token

    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        unpadded_anchor_indices = [
            find_anchor_token_index(prompt, tokenizer, anchor)
            for prompt in rendered_prompts
        ]
        individual_lengths = [
            len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            for prompt in rendered_prompts
        ]
        encoded = tokenizer(
            list(rendered_prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
    finally:
        tokenizer.padding_side = previous_padding_side

    padded_len = int(encoded["input_ids"].shape[1])
    padded_anchor_indices = [
        anchor_idx + (padded_len - length)
        for anchor_idx, length in zip(unpadded_anchor_indices, individual_lengths)
    ]
    return encoded, padded_anchor_indices


def _steering_context(
    model: Any,
    *,
    vector: Any | None,
    alpha: float | None,
    vectors: Sequence[Any] | None,
    alphas: Sequence[float] | None,
    target_indices: Sequence[int],
    telemetry_sink: list | None,
    safety_limits: Any | None,
):
    """Build one validated steering context for either routing scorer."""
    if vector is not None and vectors is not None:
        raise ValueError("provide either vector or vectors, not both")
    if vectors is not None:
        bundle = list(vectors)
        if not bundle:
            raise ValueError("vectors must be non-empty")
        scales = (
            [float(v.spec.alpha) for v in bundle]
            if alphas is None
            else [float(value) for value in alphas]
        )
        if len(scales) != len(bundle):
            raise ValueError("alphas length must match vectors length")
        return multi_layer_residual_addition_hook(
            model,
            [
                {
                    "layer_index": vec.spec.layer,
                    "vector": vec.values,
                    "alpha": scale,
                    "token_scope": "last",
                    "target_token_index": list(target_indices),
                    "safety_limits": safety_limits,
                }
                for vec, scale in zip(bundle, scales)
            ],
            telemetry_sink=telemetry_sink,
        )
    if vector is not None:
        scale = float(vector.spec.alpha if alpha is None else alpha)
        return residual_addition_hook(
            model,
            layer_index=vector.spec.layer,
            vector=vector.values,
            alpha=scale,
            token_scope="last",
            target_token_index=list(target_indices),
            telemetry_sink=telemetry_sink,
            safety_limits=safety_limits,
        )
    return nullcontext()


def score_route_batch_from_sequences(
    rendered_prompts: Sequence[str],
    model: Any,
    tokenizer: Any,
    *,
    vector: Any | None = None,
    alpha: float | None = None,
    vectors: Sequence[Any] | None = None,
    alphas: Sequence[float] | None = None,
    anchor: str = "ACTION:",
    threshold: float = 0.0,
    symbolic_label: str = "SYMBOLIC",
    llm_label: str = "LLM",
    normalization: SequenceNormalization = "mean",
    device: Any | None = None,
    telemetry_sink: list | None = None,
    safety_limits: Any | None = None,
) -> list[tuple[RouteAction, float]]:
    """Score complete route-label token sequences with teacher forcing.

    Both candidate labels are evaluated on the same prompts, model state,
    layer, alpha, and route scorer in one combined forward pass.  The score is
    either the sum or mean of continuation-token log probabilities.  ``mean``
    is the default because the human-readable labels have different token
    lengths on many BPE/SentencePiece tokenizers.

    The function fails closed when ``T(prompt)`` is not a strict prefix of
    ``T(prompt + label)``.  Silently choosing the first sub-token would make the
    reported decision a different objective from full-label routing.
    """
    import torch

    if normalization not in {"sum", "mean"}:
        raise ValueError("normalization must be 'sum' or 'mean'")
    if not rendered_prompts:
        return []
    if not symbolic_label or not llm_label or symbolic_label == llm_label:
        raise ValueError("route labels must be non-empty and distinct")

    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError("tokenizer needs pad_token_id or eos_token_id for batching")
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    # Entries are interleaved [prompt0/symbolic, prompt0/llm, ...].
    entries: list[dict[str, Any]] = []
    for prompt_index, original_prompt in enumerate(rendered_prompts):
        aligned_prompt, leading_space = detect_route_prompt_alignment(original_prompt)
        base_ids = list(tokenizer.encode(aligned_prompt, add_special_tokens=True))
        if not base_ids:
            raise ContextBoundaryError("tokenizer produced no prompt tokens")
        anchor_index = find_anchor_token_index(aligned_prompt, tokenizer, anchor)
        for label_index, label in enumerate((symbolic_label, llm_label)):
            suffix = (" " if leading_space else "") + label
            full_ids = list(
                tokenizer.encode(
                    aligned_prompt + suffix,
                    add_special_tokens=True,
                )
            )
            if full_ids[: len(base_ids)] != base_ids:
                raise ContextBoundaryError(
                    "route label changes prompt tokenization at the terminal "
                    f"boundary for prompt {prompt_index}, label {label!r}"
                )
            continuation = full_ids[len(base_ids):]
            if not continuation:
                raise MultiTokenRouteError(
                    f"route label {label!r} produced no continuation tokens"
                )
            entries.append(
                {
                    "full_ids": full_ids,
                    "base_len": len(base_ids),
                    "continuation": continuation,
                    "anchor_index": anchor_index,
                    "prompt_index": prompt_index,
                    "label_index": label_index,
                }
            )

    max_len = max(len(entry["full_ids"]) for entry in entries)
    input_ids = torch.full(
        (len(entries), max_len),
        pad_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    target_indices: list[int] = []
    for row, entry in enumerate(entries):
        ids = torch.tensor(entry["full_ids"], dtype=torch.long)
        left_pad = max_len - ids.numel()
        input_ids[row, left_pad:] = ids
        attention_mask[row, left_pad:] = 1
        target_indices.append(int(entry["anchor_index"]) + left_pad)
        entry["left_pad"] = left_pad

    if device is None:
        device = model.get_input_embeddings().weight.device
    encoded = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    ctx = _steering_context(
        model,
        vector=vector,
        alpha=alpha,
        vectors=vectors,
        alphas=alphas,
        target_indices=target_indices,
        telemetry_sink=telemetry_sink,
        safety_limits=safety_limits,
    )

    with torch.inference_mode(), ctx:
        outputs = model(**encoded)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)

    candidate_scores = torch.empty(
        (len(rendered_prompts), 2),
        dtype=torch.float64,
    )
    for row, entry in enumerate(entries):
        start = int(entry["left_pad"]) + int(entry["base_len"]) - 1
        token_ids = entry["continuation"]
        positions = torch.arange(
            start,
            start + len(token_ids),
            device=log_probs.device,
        )
        targets = torch.tensor(token_ids, device=log_probs.device)
        token_scores = log_probs[row, positions, targets]
        score = (
            token_scores.mean()
            if normalization == "mean"
            else token_scores.sum()
        )
        candidate_scores[
            int(entry["prompt_index"]),
            int(entry["label_index"]),
        ] = float(score)

    results: list[tuple[RouteAction, float]] = []
    for symbolic_score, llm_score in candidate_scores.tolist():
        margin = float(symbolic_score - llm_score)
        results.append(
            ("symbolic" if margin > threshold else "llm", margin)
        )
    return results


def score_route_batch_from_logits(
    rendered_prompts: Sequence[str],
    model: Any,
    tokenizer: Any,
    *,
    vector: Any | None = None,
    alpha: float | None = None,
    vectors: Sequence[Any] | None = None,
    alphas: Sequence[float] | None = None,
    anchor: str = "ACTION:",
    threshold: float = 0.0,
    leading_space: bool | None = None,
    token_resolver: TokenResolver = "isolated",
    device: Any | None = None,
    telemetry_sink: list | None = None,
    allow_first_token_fallback: bool = False,
    safety_limits: Any | None = None,
) -> list[tuple[RouteAction, float]]:
    """Score route decisions in one forward pass per batch.

    Supports either a single steering vector or a multi-layer composite vector
    list. Candidate route labels must each be one tokenizer token; callers can
    catch `ValueError` and fall back to autoregressive generation when this is
    not true for a model/tokenizer.

    ``token_resolver`` selects how route-label token IDs are derived:

    - ``"isolated"`` (default, v0.3.4 behavior): tokenizes labels via
      ``tokenizer.encode(label)`` in isolation. Fast, but not guaranteed
      to match the actual continuation token for all BPE/SentencePiece
      tokenizers. See CLAIMS.md §9.
    - ``"contextual"``: derives the continuation token by diffing
      ``T(rendered_prompt + label)`` against ``T(rendered_prompt)`` for
      the first prompt in the batch. Eliminates the boundary assumption.
      Slightly slower (one extra tokenization per label) and requires
      that all prompts in the batch share the same continuation token
      (which is true when they share the same routing-prompt template).

    ``allow_first_token_fallback`` (v0.3.6): when ``True``, multi-token
    route labels (e.g. Qwen2.5's ``"SYMBOLIC"`` → ``[" SY", "MBOL", "IC"]``)
    are reduced to their first continuation token, enabling a first-token
    logit contrast instead of forcing a fall-through to autoregressive
    generation. This is the default routing path for the AutoLearn loop on
    multi-token tokenizers; see CLAIMS.md §9 and the v0.3.6 repair plan
    Phase 1.1.
    """
    import torch

    if vector is not None and vectors is not None:
        raise ValueError("provide either vector or vectors, not both")
    if vectors is not None:
        vectors = list(vectors)
        if not vectors:
            raise ValueError("vectors must be non-empty")
        if alphas is None:
            alphas = [float(v.spec.alpha) for v in vectors]
        if len(alphas) != len(vectors):
            raise ValueError("alphas length must match vectors length")
    elif vector is not None:
        alpha = float(vector.spec.alpha if alpha is None else alpha)

    if token_resolver == "contextual":
        if not rendered_prompts:
            raise ValueError("contextual resolver requires at least one rendered prompt")
        # v0.3.6 Phase 2.2: align the prompt terminal boundary so the
        # contextual resolver sees a canonical "ACTION:" + " LABEL" form.
        # When the caller did not force a leading_space hint, derive it
        # from the rendered prompt's terminal whitespace.
        aligned_prompt = rendered_prompts[0]
        effective_leading_space = leading_space
        if leading_space is None:
            aligned_prompt, effective_leading_space = detect_route_prompt_alignment(
                rendered_prompts[0]
            )
        sym_id, llm_id = resolve_route_token_ids_contextual(
            tokenizer,
            aligned_prompt,
            leading_space=effective_leading_space,
            allow_first_token_fallback=allow_first_token_fallback,
        )
    elif token_resolver == "isolated":
        sym_id, llm_id = resolve_route_token_ids(
            tokenizer,
            leading_space=leading_space,
            allow_first_token_fallback=allow_first_token_fallback,
        )
    else:
        raise ValueError(
            f"token_resolver must be 'isolated' or 'contextual', got {token_resolver!r}"
        )

    encoded, target_indices = _tokenize_route_batch(
        rendered_prompts,
        tokenizer,
        anchor=anchor,
    )

    if device is None:
        device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    if vectors is not None:
        configs = [
            {
                "layer_index": vec.spec.layer,
                "vector": vec.values,
                "alpha": float(scale),
                "token_scope": "last",
                "target_token_index": target_indices,
                "safety_limits": safety_limits,
            }
            for vec, scale in zip(vectors, alphas)
        ]
        ctx = multi_layer_residual_addition_hook(model, configs, telemetry_sink=telemetry_sink)
    elif vector is not None:
        ctx = residual_addition_hook(
            model,
            layer_index=vector.spec.layer,
            vector=vector.values,
            alpha=float(alpha),
            token_scope="last",
            target_token_index=target_indices,
            telemetry_sink=telemetry_sink,
            safety_limits=safety_limits,
        )
    else:
        ctx = nullcontext()

    with torch.inference_mode(), ctx:
        outputs = model(**encoded)
        # Left padding guarantees the final sequence position is the actual
        # prompt end for every example, not a padding position.
        last_logits = outputs.logits[:, -1, :]

    return [
        route_action_from_logits(
            row,
            sym_id,
            llm_id,
            threshold=threshold,
        )
        for row in last_logits
    ]
