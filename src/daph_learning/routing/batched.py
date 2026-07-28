"""Batched steered-route evaluation (V037-002).

Moved out of ``scripts/tune_steering.py`` so the library layer no longer
depends on the CLI layer. The ``scripts/tune_steering.py`` CLI now imports
from here.

Public names (the leading underscores are dropped now that these are part of
the public library API):

- :func:`as_task_map`
- :func:`evaluate_batch_steered_routes`
- :func:`score_key`
- :func:`chunks`

Backward-compat aliases (``_as_task_map``, ``_evaluate_batch_steered_routes``)
are exported so existing callers inside ``scripts/`` keep working unchanged.
"""

from __future__ import annotations

from typing import Any, Sequence

from daph_learning.data.task_utils import format_for_model
from daph_learning.routing.logit_router import (
    score_route_batch_from_logits,
    score_route_batch_from_sequences,
)
from daph_learning.routing.steered_router import build_route_prompt


def as_task_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index task rows by task_id, validating presence and uniqueness."""
    ids = [str(row.get("task_id", "")) for row in rows]
    if any(not tid for tid in ids):
        raise ValueError("validation tasks must all contain task_id")
    if len(ids) != len(set(ids)):
        raise ValueError("validation task IDs must be unique")
    return {row["task_id"]: row for row in rows}


def score_key(record: dict[str, Any]) -> tuple[float, float, float]:
    """Sort key for tuning records: (f1, decision_coverage, route_accuracy)."""
    metrics = record["metrics"]
    return (
        float(metrics["f1"]),
        float(metrics["decision_coverage"]),
        float(metrics["route_accuracy"]),
    )


def chunks(items: Sequence[Any], batch_size: int):
    """Yield successive ``batch_size`` slices of ``items``."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def evaluate_batch_steered_routes(
    tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    vector: Any | None = None,
    alpha: float | None = None,
    prompt_format: str = "chat",
    *,
    vectors: Sequence[Any] | None = None,
    alphas: Sequence[float] | None = None,
    batch_size: int = 32,
    threshold: float = 0.0,
    token_resolver: str = "isolated",
    allow_first_token_fallback: bool = False,
    label_scoring: str = "single_token",
    sequence_normalization: str = "mean",
    safety_limits: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate routes with batched logit scoring.

    Supports both the historical single-vector call shape and a composite
    multi-layer bundle.

    ``label_scoring="sequence"`` uses full teacher-forced label log
    probabilities and is required for protocol-eligible v0.3.8 results.
    ``label_scoring="single_token"`` preserves the historical fast path.

    ``allow_first_token_fallback`` (v0.3.6) is forwarded to
    :func:`score_route_batch_from_logits`; when ``True``, multi-token route
    labels are reduced to their first continuation token so the logit
    contrast path stays available for tokenizers like Qwen2.5 that never
    produce a single-token route label.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if label_scoring not in {"single_token", "sequence"}:
        raise ValueError("label_scoring must be 'single_token' or 'sequence'")
    if label_scoring == "sequence" and allow_first_token_fallback:
        raise ValueError(
            "allow_first_token_fallback is incompatible with full sequence scoring"
        )
    if vector is not None and vectors is not None:
        raise ValueError("provide either vector or vectors, not both")

    bundle = list(vectors or ([] if vector is None else [vector]))
    if not bundle:
        raise ValueError("at least one steering vector is required")

    if alphas is None:
        if vector is not None and alpha is not None:
            effective_alphas = [float(alpha)]
        else:
            effective_alphas = [float(v.spec.alpha) for v in bundle]
    else:
        effective_alphas = [float(a) for a in alphas]

    if len(effective_alphas) != len(bundle):
        raise ValueError("alphas length must match vector bundle length")

    anchors = {v.spec.anchor or "ACTION:" for v in bundle}
    if len(anchors) != 1:
        raise ValueError("all vectors in a composite tool-policy bundle must share one anchor")
    anchor = next(iter(anchors))

    routes: dict[str, dict[str, Any]] = {}
    for batch in chunks(tasks, batch_size):
        rendered = [
            format_for_model(build_route_prompt(task), tokenizer, prompt_format)
            for task in batch
        ]
        if label_scoring == "sequence":
            scored = score_route_batch_from_sequences(
                rendered,
                model,
                tokenizer,
                vectors=bundle,
                alphas=effective_alphas,
                anchor=anchor,
                threshold=threshold,
                normalization=sequence_normalization,
                safety_limits=safety_limits,
            )
            route_method = f"full_sequence_logprob_{sequence_normalization}"
        else:
            scored = score_route_batch_from_logits(
                rendered,
                model,
                tokenizer,
                vectors=bundle,
                alphas=effective_alphas,
                anchor=anchor,
                threshold=threshold,
                leading_space=None,
                token_resolver=token_resolver,
                allow_first_token_fallback=allow_first_token_fallback,
                safety_limits=safety_limits,
            )
            route_method = (
                "first_token_logit_contrast"
                if allow_first_token_fallback
                else "single_token_logit_contrast"
            )
        for task, (action, margin) in zip(batch, scored):
            routes[task["task_id"]] = {
                "task_id": task["task_id"],
                "route": action,
                "route_margin": margin,
                "route_method": route_method,
            }
    return routes


# --- Backward-compat aliases (scripts/ callers used underscore-prefixed names) ---

_as_task_map = as_task_map
_evaluate_batch_steered_routes = evaluate_batch_steered_routes
_score_key = score_key
_chunks = chunks


__all__ = [
    "as_task_map",
    "evaluate_batch_steered_routes",
    "score_key",
    "chunks",
    # backward-compat aliases:
    "_as_task_map",
    "_evaluate_batch_steered_routes",
    "_score_key",
    "_chunks",
]
