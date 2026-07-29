"""Core AutoLearn learning loop.

The loop iteratively improves a steering vector by learning from execution
outcomes. See ``daph_learning.autolearn.__init__`` for the design overview.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from daph_learning.execution.symbolic_executor import (
    execute_plan,
    plan_from_structured_task,
)
from daph_learning.routing.errors import (
    ContextBoundaryError,
    ModelRoutingError,
    MultiTokenRouteError,
    SteeringApplicationError,
)
from daph_learning.steering.extract import contrastive_mean_direction
from daph_learning.steering.hooks import capture_layer_output
from daph_learning.steering.anchors import (
    find_anchor_token_index,
    find_anchor_token_span,
)
from daph_learning.steering.types import SteeringSpec, SteeringVector
from daph_learning.tools.symbolic_math import SymbolicMathError


# --- Outcome classification ---

OutcomeLabel = Literal["correct", "misrouted", "unverifiable"]

# v0.3.8.1: explicit termination reason for the learning loop. ``not updated``
# alone is not a diagnosis — it can mean too few examples, capture failure,
# every example being unverifiable, or genuine convergence. The loop now
# records the actual cause so a stopped run can be triaged rather than
# mislabelled "converged".
StopReason = Literal[
    "converged",
    "insufficient_positive",
    "insufficient_negative",
    "capture_failure",
    "no_verifiable_experiences",
    "validation_rejected",
    "max_iterations",
]

# v0.3.8.1: the activation-pooling methods supported at capture time. The
# default is ``"anchor"`` so the configured ``AutoLearnConfig.anchor`` is
# actually honoured (previously the hook was installed with
# ``target_token_index=-1`` and the anchor was ignored, so metadata claimed
# ``capture_anchor = ACTION:`` while the vector was learned from the last
# prompt token).
CaptureTokenScope = Literal["anchor", "last", "mean_anchor_span", "mean_sequence"]


def classify_outcome(
    task: dict[str, Any],
    route: str | None,
    output: str | None,
    *,
    expected: Any | None = None,
    route_raw: str | None = None,
) -> OutcomeLabel:
    """Classify a routed execution with typed, exact verification.

    ``route_raw`` may recover a malformed route decision, but it is never
    reused as an answer. Numeric correctness accepts only an exact integer or
    ``FINAL: <integer>``; substring matching is intentionally forbidden.

    For tasks without a checkable expected value, an LLM route is credited
    only as bootstrap route suitability when the task carries an explicit LLM
    oracle (or has no symbolic capabilities). This is not a claim that the LLM
    answer itself was correct.
    """
    expected = expected if expected is not None else task.get("expected")

    # v0.3.6: when the caller passes a parsed route of None (generation
    # produced no parseable ACTION), inspect route_raw directly. This keeps
    # multi-token generated outputs (e.g. " SYMBOLIC" emitted as " SY",
    # "MBOL", "IC" then decoded) from being silently treated as LLM.
    if route is None and route_raw:
        from daph_learning.routing.steered_router import parse_route_action
        parsed = parse_route_action(route_raw)
        if parsed is not None:
            route = parsed

    if route not in {"symbolic", "llm"}:
        return "unverifiable"

    if expected is None:
        explicit_oracle = (
            task.get("utility_oracle")
            or task.get("route_label")
        )
        if route == "llm" and (
            explicit_oracle == "llm" or not task.get("capability_ids")
        ):
            return "correct"
        if explicit_oracle in {"symbolic", "llm"} and route != explicit_oracle:
            return "misrouted"
        return "unverifiable"

    from daph_learning.evaluation.verification import (
        NumericVerifier,
        VerificationStatus,
    )

    if output is None and route == "llm":
        # The v0.3.x loop may deliberately skip expensive LLM execution.
        # A skipped backend is unknown, not a measured failure.
        return "unverifiable"

    verification = NumericVerifier().verify(
        {**task, "expected": expected},
        output,
        execution_succeeded=output is not None,
    )
    if verification.status is VerificationStatus.VERIFIED_CORRECT:
        return "correct"
    if verification.status in {
        VerificationStatus.VERIFIED_INCORRECT,
        VerificationStatus.EXECUTION_FAILED,
    }:
        return "misrouted"
    if (
        verification.status is VerificationStatus.INVALID_OUTPUT
        and route == "symbolic"
    ):
        return "misrouted"
    return "unverifiable"


# --- Data structures ---

@dataclass(frozen=True)
class AutoLearnConfig:
    """Configuration for the AutoLearn learning loop.

    Parameters
    ----------
    n_iterations : int
        Maximum number of learning iterations.
    layer : int
        Which transformer layer to capture activations from and steer.
    alpha : float
        Steering alpha to use during routing.
    anchor : str
        The anchor token for activation capture and steering.
    capture_token_scope : str
        Activation-pooling method at capture time. One of:

        - ``"anchor"`` (default): resolve the configured ``anchor`` in the
          rendered prompt and capture that token's activation. This is the
          only scope that actually honours ``anchor``.
        - ``"last"``: capture the final prompt token (the historical v0.3.x
          behaviour; ignores ``anchor``).
        - ``"mean_anchor_span"``: mean-pool the activation over every token
          whose character span overlaps the anchor.
        - ``"mean_sequence"``: mean-pool the activation over the whole
          sequence.

        The scope actually used is recorded in the vector's
        ``SteeringSpec.capture_token_scope`` provenance so a run manifest can
        reproduce the capture position.
    min_examples_per_update : int
        Minimum number of positive and negative examples required to
        update the steering vector. If fewer are available, the loop
        stops early.
    normalization : str
        Normalization for the extracted direction: "l2", "none", or
        "mean_centered".
    seed : int
        Random seed for reproducibility.
    """

    n_iterations: int = 5
    layer: int = 24
    alpha: float = 1.0
    anchor: str = "ACTION"
    capture_token_scope: CaptureTokenScope = "anchor"  # type: ignore[assignment]
    min_examples_per_update: int = 4
    normalization: str = "l2"
    seed: int = 42
    max_relative_perturbation: float = 0.65
    max_cosine_shift: float = 0.25


@dataclass
class IterationMetrics:
    """Metrics for a single iteration of the learning loop.

    v0.3.8.1 adds activation-capture coverage telemetry. A steering vector
    is biased if capture fails at a different rate for the positive
    (should-route-symbolic) and negative (should-route-LLM) classes, so the
    loop records per-class failure rates instead of silently skipping
    capture failures.
    """

    iteration: int
    n_train: int
    n_correct: int
    n_misrouted: int
    n_unverifiable: int
    train_accuracy: float
    val_f1: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    vector_norm: float
    n_positive_examples: int
    n_negative_examples: int
    updated: bool  # whether the vector was updated this iteration
    # --- activation-capture coverage (v0.3.8.1) ---
    n_capture_attempts: int = 0
    n_capture_successes: int = 0
    capture_coverage: float = 1.0  # successes / attempts; 1.0 when no attempts
    capture_failure_rate_positive: float = 0.0  # P(capture failure | positive class)
    capture_failure_rate_negative: float = 0.0  # P(capture failure | negative class)


@dataclass
class AutoLearnResult:
    """Result of the AutoLearn learning loop.

    Attributes
    ----------
    best_vector : SteeringVector | None
        The steering vector from the iteration with the best validation F1.
    best_iteration : int
        The iteration index that produced the best vector.
    best_val_f1 : float
        The best validation F1 achieved.
    iterations : list[IterationMetrics]
        Metrics for each iteration, forming the learning curve.
    config : AutoLearnConfig
        The configuration used.
    """

    best_vector: SteeringVector | None
    best_iteration: int
    best_val_f1: float
    iterations: list[IterationMetrics]
    config: AutoLearnConfig
    # v0.3.8.1: explicit termination reason. ``"converged"`` is reserved for a
    # real stability signal (v0.3.9 candidate promotion); the v0.3.8.1 loop
    # never emits it. A bare "no update" stop is reported as the actual cause
    # (insufficient examples / capture failure / no verifiable experiences).
    stop_reason: StopReason = "max_iterations"


# --- The loop ---

def _resolve_anchor_token_index(
    rendered: str,
    tokenizer: Any,
    anchor: str,
    scope: CaptureTokenScope,
) -> int | None:
    """Resolve the single token index to capture for ``anchor``/``last`` scopes.

    Returns ``None`` if the anchor cannot be resolved, so the caller can
    record a capture failure. ``"last"`` always returns ``-1`` without
    touching the tokenizer.
    """
    if scope == "last":
        return -1
    # ``"anchor"``: the token at the end of the anchor text. Try the anchor
    # as configured, then with a trailing ``:`` (the route prompt terminates
    # with ``ACTION:`` while the config anchor is conventionally ``ACTION``).
    for candidate in (anchor, f"{anchor}:"):
        try:
            return find_anchor_token_index(rendered, tokenizer, candidate)
        except ValueError:
            continue
    return None


def _capture_activations(
    tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    layer: int,
    anchor: str,
    capture_token_scope: CaptureTokenScope = "anchor",
    prompt_format: str = "raw",
    class_label: str = "unknown",
) -> tuple[np.ndarray | None, int, int]:
    """Capture activations from a list of tasks for the configured scope.

    Returns ``(activations, n_attempts, n_successes)`` where ``activations``
    is an ``[N_success, H]`` array (or ``None`` if every capture failed) and
    the counts let the loop report capture coverage per class.

    v0.3.8.1: the configured ``anchor`` is now actually resolved and captured.
    Previously the hook was installed with ``target_token_index=-1`` for
    every scope, so metadata advertised ``capture_anchor = ACTION:`` while
    the vector was learned from the last prompt token (which, under a chat
    template, is the generation-header token rather than the anchor).

    Supported scopes (see :data:`CaptureTokenScope`):

    - ``"anchor"``: resolve the anchor and capture that single token.
    - ``"last"``: capture the final prompt token (legacy behaviour).
    - ``"mean_anchor_span"``: mean-pool over every token overlapping the
      anchor character span.
    - ``"mean_sequence"``: mean-pool over the whole sequence.
    """
    import torch
    from daph_learning.routing.steered_router import build_route_prompt

    needs_full_hidden = capture_token_scope in ("mean_anchor_span", "mean_sequence")
    activations = []
    n_attempts = 0
    for task in tasks:
        prompt = build_route_prompt(task)
        # Apply prompt formatting
        if prompt_format == "chat" and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = prompt

        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=True)
        if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        n_attempts += 1

        # Resolve the capture position for this scope.
        if capture_token_scope == "last":
            token_indices: list[int] | None = [-1]
        elif capture_token_scope == "mean_sequence":
            token_indices = None  # pool everything
        elif capture_token_scope == "mean_anchor_span":
            span: list[int] = []
            for candidate in (anchor, f"{anchor}:"):
                try:
                    span = find_anchor_token_span(rendered, tokenizer, candidate)
                    break
                except ValueError:
                    continue
            token_indices = span or None
        else:  # "anchor"
            idx = _resolve_anchor_token_index(rendered, tokenizer, anchor, capture_token_scope)
            token_indices = [idx] if idx is not None else None

        if token_indices is None and capture_token_scope != "mean_sequence":
            # Anchor resolution failed for this task. Record it as a capture
            # failure rather than silently falling back to the last token
            # (which would reintroduce the v0.3.8 provenance mismatch).
            from daph_learning.telemetry import emit_fallback
            emit_fallback(
                event="activation_capture_failure",
                frm="capture",
                to="skip",
                reason="anchor_resolution_failed",
                task_id=str(task.get("task_id", "")),
                layer=layer,
                scope=capture_token_scope,
                class_label=class_label,
            )
            continue

        sink: list = []
        try:
            if needs_full_hidden:
                with torch.inference_mode(), capture_layer_output(
                    model,
                    layer_index=layer,
                    sink=sink,
                    target_token_index=None,
                    last_token_only=False,
                ):
                    model(**encoded, use_cache=False)
                if sink:
                    act = sink[0]
                    if act.ndim == 3:
                        act = act[0]  # [seq, hidden]
                    if capture_token_scope == "mean_sequence":
                        pooled = act.mean(dim=0)
                    else:
                        pooled = act[token_indices].mean(dim=0)
                    activations.append(pooled.numpy().astype(np.float32))
            else:
                target_index = int(token_indices[0])
                with torch.inference_mode(), capture_layer_output(
                    model,
                    layer_index=layer,
                    sink=sink,
                    target_token_index=target_index,
                ):
                    model(**encoded, use_cache=False)
                if sink:
                    act = sink[0]
                    # capture_layer_output with target_token_index returns
                    # hidden[..., idx, :] which is [batch=1, hidden]; squeeze
                    # the leading dim so the stack is [N, hidden].
                    if act.ndim == 2:
                        act = act[0]
                    activations.append(act.numpy().astype(np.float32))
        except (RuntimeError, ValueError, TypeError, IndexError, SteeringApplicationError) as exc:
            # Expected activation-capture failures: layer index out of range,
            # shape/device mismatch, hook application error, type mismatch
            # from a non-standard tokenizer. The task is skipped (no
            # activation collected). Unexpected exceptions propagate.
            from daph_learning.telemetry import emit_fallback
            emit_fallback(
                event="activation_capture_failure",
                frm="capture",
                to="skip",
                reason=type(exc).__name__,
                task_id=str(task.get("task_id", "")),
                layer=layer,
                scope=capture_token_scope,
                class_label=class_label,
            )
            continue

    n_successes = len(activations)
    if not activations:
        return None, n_attempts, n_successes
    return np.stack(activations, axis=0), n_attempts, n_successes


def _execute_symbolic(task: dict[str, Any]) -> tuple[str | None, bool]:
    """Execute a task symbolically. Returns (output_string, success).

    The output string is "FINAL: <value>" on success, None on failure.
    """
    try:
        plan = plan_from_structured_task(task, reason_code="autolearn")
        result = execute_plan(plan)
        return f"FINAL: {result.value}", result.verified
    except (SymbolicMathError, ValueError, TypeError, KeyError) as exc:
        # Expected symbolic-execution failures: unsafe expression, unsupported
        # operation, resource limit, unsupported capability, missing input.
        # These are normal "symbolic couldn't handle this task" signals, not
        # infrastructure failures. Unexpected exceptions propagate.
        from daph_learning.telemetry import emit_fallback
        emit_fallback(
            event="symbolic_execution_failure",
            frm="symbolic",
            to="none",
            reason=type(exc).__name__,
            task_id=str(task.get("task_id", "")),
        )
        return None, False


def _route_with_steering_generate(
    tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    vector: SteeringVector,
    alpha: float,
    *,
    prompt_format: str = "raw",
    max_new_tokens: int = 5,
    safety_limits: Any | None = None,
) -> list[tuple[str | None, str | None]]:
    """Route tasks using steering + generate mode.

    This is the fallback for tokenizers where direct-logit routing fails
    (e.g. Qwen2.5 tokenizes "SYMBOLIC" as multiple tokens). It applies
    the steering vector via a residual addition hook, generates a few
    tokens, and parses the route action from the generated text.

    Returns a list of ``(route, raw_text)`` tuples where ``route`` is
    "symbolic", "llm", or None if parsing fails, and ``raw_text`` is
    the decoded generation (for downstream route_raw recovery).
    """
    import torch
    from daph_learning.routing.steered_router import build_route_prompt, parse_route_action
    from daph_learning.steering.hooks import residual_addition_hook

    embed_device = model.get_input_embeddings().weight.device
    routes: list[tuple[str | None, str | None]] = []

    for task in tasks:
        prompt = build_route_prompt(task)
        if prompt_format == "chat" and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = prompt

        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {
            k: (v.to(embed_device) if hasattr(v, "to") else torch.tensor(v).to(embed_device))
            for k, v in encoded.items()
        }

        vec_values = np.asarray(vector.values, dtype=np.float32)
        try:
            with torch.inference_mode(), residual_addition_hook(
                model,
                layer_index=vector.spec.layer,
                vector=vec_values,
                alpha=alpha,
                token_scope="last",
                safety_limits=safety_limits,
            ):
                out = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    use_cache=True,
                )
            generated = tokenizer.decode(
                out[0][encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            action = parse_route_action(generated)
            routes.append((action, generated))
        except (RuntimeError, SteeringApplicationError, ValueError, AttributeError) as exc:
            # Expected generate-mode routing failures: model.generate error,
            # steering hook failure, decode/shape error, or model lacks a
            # generate method. The task gets a (None, None) route; unexpected
            # exceptions propagate.
            from daph_learning.telemetry import emit_fallback
            emit_fallback(
                event="route_fallback",
                frm="steered_generate",
                to="none",
                reason=type(exc).__name__,
                task_id=str(task.get("task_id", "")),
            )
            routes.append((None, None))

    return routes


def _route_without_steering_generate(
    tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    prompt_format: str = "raw",
    max_new_tokens: int = 5,
) -> list[str | None]:
    """Route tasks without steering using generate mode.

    Returns a list of route strings ("symbolic", "llm", or None).
    """
    import torch
    from daph_learning.routing.steered_router import build_route_prompt, parse_route_action

    embed_device = model.get_input_embeddings().weight.device
    routes: list[str | None] = []

    for task in tasks:
        prompt = build_route_prompt(task)
        if prompt_format == "chat" and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = prompt

        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {
            k: (v.to(embed_device) if hasattr(v, "to") else torch.tensor(v).to(embed_device))
            for k, v in encoded.items()
        }

        try:
            with torch.inference_mode():
                out = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    use_cache=True,
                )
            generated = tokenizer.decode(
                out[0][encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            action = parse_route_action(generated)
            routes.append(action)
        except (RuntimeError, ValueError, AttributeError) as exc:
            # Expected generate-mode routing failures (no steering): model
            # error, decode error, or model lacks a generate method. The task
            # gets a None route; unexpected exceptions propagate.
            from daph_learning.telemetry import emit_fallback
            emit_fallback(
                event="route_fallback",
                frm="generate",
                to="none",
                reason=type(exc).__name__,
                task_id=str(task.get("task_id", "")),
            )
            routes.append(None)

    return routes


def _evaluate_routes_simple(
    tasks: list[dict[str, Any]],
    routes: list[str | None],
    label_field: str = "route_label",
) -> dict[str, float]:
    """Simple routing evaluation that doesn't require the full
    evaluate_route_records infrastructure. Works with generate-mode
    routing where route records aren't available.

    v0.3.8.1: a missing route (``None``) is no longer scored as an LLM
    prediction. A router that fails to decide is not evidence for the LLM
    backend. Metrics now preserve the distinction between a decided route
    and an undecidable one:

    - ``accuracy_on_decisions``: correctness over tasks where the router
      actually decided (``symbolic``/``llm``). Undefined (0.0) when nothing
      was decided.
    - ``decision_coverage``: fraction of tasks with a real decision.
    - ``invalid_rate``: fraction of tasks with a ``None`` (undecidable)
      route.
    - ``abstain_rate``: fraction of tasks routed to ``"abstain"``.
    - ``overall_utility``: ``accuracy_on_decisions * decision_coverage``,
      a simple proxy that penalizes low-coverage routers. The full
      utility objective (correctness minus latency/cost/risk) arrives in
      v0.3.9; route F1 remains diagnostic, not the master objective.

    The legacy ``accuracy``/``precision``/``recall``/``f1`` keys are kept
    for backward compatibility and are computed over decided routes only.
    """
    expected = [t.get(label_field, "llm") for t in tasks]
    n = len(tasks) if tasks else 0

    decided = [(r, e) for r, e in zip(routes, expected) if r in ("symbolic", "llm")]
    n_decided = len(decided)
    n_invalid = sum(1 for r in routes if r is None)
    n_abstain = sum(1 for r in routes if r == "abstain")

    tp = sum(1 for r, e in decided if r == "symbolic" and e == "symbolic")
    fp = sum(1 for r, e in decided if r == "symbolic" and e == "llm")
    fn = sum(1 for r, e in decided if r == "llm" and e == "symbolic")
    tn = sum(1 for r, e in decided if r == "llm" and e == "llm")

    accuracy_on_decisions = (tp + tn) / n_decided if n_decided else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    decision_coverage = n_decided / n if n else 0.0
    invalid_rate = n_invalid / n if n else 0.0
    abstain_rate = n_abstain / n if n else 0.0
    overall_utility = accuracy_on_decisions * decision_coverage

    return {
        # legacy keys (computed over decided routes only)
        "accuracy": accuracy_on_decisions,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "route_accuracy": accuracy_on_decisions,
        # v0.3.8.1 coverage-aware metrics
        "accuracy_on_decisions": accuracy_on_decisions,
        "decision_coverage": decision_coverage,
        "invalid_rate": invalid_rate,
        "abstain_rate": abstain_rate,
        "overall_utility": overall_utility,
    }


def run_autolearn_loop(
    train_tasks: list[dict[str, Any]],
    val_tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    config: AutoLearnConfig,
    initial_vector: SteeringVector | None = None,
    route_fn: Callable[..., dict[str, Any]] | None = None,
    eval_fn: Callable[..., dict[str, float]] | None = None,
    prompt_format: str = "raw",
    label_field: str | None = "route_label",
    label_oracle_kind: str | None = "policy_heuristic",
    routing_mode: str = "auto",
) -> AutoLearnResult:
    """Run the AutoLearn learning loop.

    The loop iteratively improves a steering vector by:
    1. Routing training tasks with the current vector.
    2. Executing the chosen backend.
    3. Classifying outcomes as correct/misrouted/unverifiable.
    4. Capturing activations from correct (positive) and misrouted (negative) tasks.
    5. Updating the steering vector via contrastive mean difference.
    6. Evaluating on the validation set.

    Parameters
    ----------
    train_tasks, val_tasks : list[dict]
        Training and validation task lists.
    model, tokenizer : Any
        The HuggingFace model and tokenizer.
    config : AutoLearnConfig
        Loop configuration.
    initial_vector : SteeringVector | None
        Starting vector. If None, the first iteration routes without
        steering (baseline) and uses the outcomes to extract the first vector.
    route_fn : callable, optional
        Custom routing function. If None, uses a simple symbolic-only
        execution path (no model routing) for the first pass, then
        switches to steered routing once a vector is available.
    eval_fn : callable, optional
        Custom evaluation function. If None, uses evaluate_route_records.
    prompt_format : str
        "raw" or "chat".
    label_field, label_oracle_kind : str | None
        Passed to the evaluation function.
    routing_mode : str
        "auto" (default): try full-label sequence-logit routing, then fall
        back to generation only when the prompt/label boundary is unstable.
        "logit": force full-label sequence-logit routing.
        "generate": force generate-mode routing (slower but works with
        any tokenizer).

    Returns
    -------
    AutoLearnResult
        The best vector, best iteration, and the full learning curve.
    """
    # V037-002: import reusable routing/evaluation helpers from the library
    # layer instead of from scripts/. The dependency direction is now
    # CLI -> package only.
    from daph_learning.evaluation.routes import evaluate_route_records
    from daph_learning.routing.batched import (
        evaluate_batch_steered_routes as _evaluate_batch_steered_routes,
        as_task_map as _as_task_map,
    )
    from daph_learning.steering.hooks import SteeringSafetyLimits

    if eval_fn is None:
        eval_fn = evaluate_route_records

    current_vector = initial_vector
    iterations: list[IterationMetrics] = []
    best_vector = initial_vector
    best_val_f1 = -1.0
    best_iteration = -1
    # v0.3.8.1: explicit termination reason. Default is ``max_iterations``;
    # the loop overwrites it whenever it stops for a diagnosed cause.
    stop_reason: StopReason = "max_iterations"

    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("cannot determine model hidden_size")

    # Determine which routing method to use.
    # If routing_mode is "auto", try logit first and fall back to generate.
    use_generate_mode = routing_mode == "generate"
    safety_limits = SteeringSafetyLimits(
        max_relative_perturbation=config.max_relative_perturbation,
        max_cosine_shift=config.max_cosine_shift,
    )

    for iteration in range(config.n_iterations):
        # --- Stage 1: Route and execute training tasks ---
        # v0.3.8.1: routes may be ``None`` (undecidable/invalid generation)
        # or ``"abstain"`` (deliberate non-decision). Neither is silently
        # coerced to ``"llm"`` any more — a failed route is not evidence for
        # the LLM backend, and abstention is an honest controller option, not
        # a hidden LLM route. Both flow through as their own categories and
        # produce no backend execution (outcome ``unverifiable``).
        routes: dict[str, str | None] = {}
        route_raws: dict[str, str | None] = {}
        outputs: dict[str, str | None] = {}

        if route_fn is not None:
            # V037-001: actually invoke the user-supplied router. Previously
            # `route_fn` was accepted but never called (the v0.3.6 branch
            # `current_vector is not None and route_fn is None` sent every
            # other case to the capability heuristic). The custom router now
            # takes precedence over both the steered and baseline paths, and
            # invalid results raise InvalidRouteDecisionError instead of being
            # silently dropped. `route_fn` exceptions propagate.
            from daph_learning.routing.policy import route_tasks as _route_tasks
            decisions = _route_tasks(
                train_tasks,
                model=model,
                tokenizer=tokenizer,
                steering=current_vector,
                route_fn=route_fn,
                alpha=config.alpha,
                prompt_format=prompt_format,
            )
            for task, decision in zip(train_tasks, decisions):
                tid = task["task_id"]
                # v0.3.8.1: preserve ``abstain`` as its own route category. The
                # v0.3.7 code collapsed abstain -> "llm" for execution, which
                # destroyed the abstention signal and let a router look good
                # by abstaining on LLM-labelled examples. A real abstain
                # backend arrives in v0.3.9 (V039-003); until then abstain
                # produces no execution and is counted in ``abstain_rate``.
                routes[tid] = decision.route
                route_raws[tid] = None
        elif current_vector is not None:
            if use_generate_mode:
                # Use generate-mode steered routing (works with any tokenizer)
                task_list = train_tasks
                steered_route_list = _route_with_steering_generate(
                    task_list, model, tokenizer, current_vector, config.alpha,
                    prompt_format=prompt_format,
                    safety_limits=safety_limits,
                )
                for task, (route, raw) in zip(task_list, steered_route_list):
                    # v0.3.8.1: keep ``None`` (unparseable generation) as None.
                    # Scoring a failed route as LLM let a router look good by
                    # failing on LLM-labelled examples; the decision is now
                    # recorded as invalid and counted in ``invalid_rate``.
                    routes[task["task_id"]] = route
                    route_raws[task["task_id"]] = raw
            else:
                # v0.3.8: score the complete route-label sequences.  The
                # first-subtoken heuristic is retained only in the legacy
                # public API and is not used by the learning loop.
                task_map = _as_task_map(train_tasks)
                task_list = list(task_map.values())
                routed_via_logit = False
                # v0.3.8.1: only the ``contextual`` resolver is attempted here.
                # Full-sequence scoring (``score_route_batch_from_sequences``)
                # does not take a ``token_resolver`` — ``isolated`` is a
                # single-token-scorer concept and is not a distinct operation
                # for teacher-forced label scoring. The previous loop and
                # error text claimed "contextual -> isolated" was tried, which
                # was false. See RELEASE_NOTES_V0_3_8_1.md §5.
                for resolver in ("contextual",):
                    try:
                        steered_routes = _evaluate_batch_steered_routes(
                            task_list,
                            model,
                            tokenizer,
                            vectors=[current_vector],
                            alphas=[config.alpha],
                            prompt_format=prompt_format,
                            token_resolver=resolver,
                            allow_first_token_fallback=False,
                            label_scoring="sequence",
                            sequence_normalization="mean",
                            safety_limits=safety_limits,
                        )
                        for task in task_list:
                            routes[task["task_id"]] = steered_routes[task["task_id"]]["route"]
                        routed_via_logit = True
                        break
                    except MultiTokenRouteError as exc:
                        # Expected: tokenizer produces multi-token route labels
                        # and full-sequence scoring could not resolve the
                        # prompt/label boundary. Emit structured telemetry and
                        # fall through to the generate-mode fallback.
                        from daph_learning.telemetry import emit_fallback
                        emit_fallback(
                            event="route_fallback",
                            frm=resolver,
                            to="generate",
                            reason="multi_token_label",
                            iteration=iteration,
                            detail=str(exc),
                        )
                        continue
                    except ContextBoundaryError as exc:
                        # Expected: prompt/label boundary unresolvable.
                        from daph_learning.telemetry import emit_fallback
                        emit_fallback(
                            event="route_fallback",
                            frm=resolver,
                            to="generate",
                            reason="context_boundary",
                            iteration=iteration,
                            detail=str(exc),
                        )
                        continue
                if not routed_via_logit:
                    if routing_mode == "logit":
                        raise ModelRoutingError(
                            "logit routing mode requested but the contextual "
                            "sequence resolver failed to score the route labels "
                            "(multi-token label or unresolvable prompt/label "
                            "boundary); no other resolver is attempted for "
                            "full-sequence scoring"
                        )
                    # Fall back to generate mode
                    from daph_learning.telemetry import emit_fallback
                    emit_fallback(
                        event="route_fallback",
                        frm="logit",
                        to="generate",
                        reason="all_resolvers_failed",
                        iteration=iteration,
                    )
                    use_generate_mode = True
                    steered_route_list = _route_with_steering_generate(
                        task_list, model, tokenizer, current_vector, config.alpha,
                        prompt_format=prompt_format,
                        safety_limits=safety_limits,
                    )
                    for task, (route, raw) in zip(task_list, steered_route_list):
                        routes[task["task_id"]] = route
                        route_raws[task["task_id"]] = raw
        else:
            # No vector yet — use heuristic routing
            for task in train_tasks:
                routes[task["task_id"]] = "symbolic" if task.get("capability_ids") else "llm"

        # Execute. v0.3.8.1: only ``"symbolic"`` and ``"llm"`` are real
        # backends. ``None`` (invalid/undecidable route) and ``"abstain"``
        # (deliberate non-decision) run no backend; their outcome is
        # ``unverifiable`` rather than a hidden LLM prediction. LLM execution
        # is still skipped during the learning loop (the counterfactual
        # both-backend execution arrives in v0.3.9); an LLM-routed task with
        # no executed output is honestly classified as ``unverifiable``.
        for task in train_tasks:
            tid = task["task_id"]
            route = routes[tid]
            if route == "symbolic":
                output, _ = _execute_symbolic(task)
                outputs[tid] = output
            elif route == "llm":
                # LLM execution is skipped during the learning loop. The
                # outcome is classified as ``unverifiable`` (not ``correct``),
                # so an LLM-routed task cannot inflate routing metrics by
                # failing to execute. This is the v0.3.8 limitation that v0.3.9
                # removes by running both backends.
                outputs[tid] = None
            else:
                # None (invalid) or "abstain": no backend runs.
                outputs[tid] = None

        # --- Stage 2: Classify outcomes ---
        positive_tasks: list[dict[str, Any]] = []
        negative_tasks: list[dict[str, Any]] = []
        n_correct = n_misrouted = n_unverifiable = 0

        for task in train_tasks:
            tid = task["task_id"]
            route = routes[tid]
            output = outputs[tid]
            outcome = classify_outcome(
                task, route, output, route_raw=route_raws.get(tid),
            )

            if outcome == "correct":
                n_correct += 1
                # For the steering vector, "positive" = should route to symbolic,
                # "negative" = should route to LLM. We use the oracle labels
                # to determine which class each task belongs to, and the
                # execution outcome to determine if the current route was right.
                # A correctly-routed symbolic task is a positive example.
                # A correctly-routed LLM task (non-symbolic) is a negative example.
                if route == "symbolic":
                    positive_tasks.append(task)
                else:
                    negative_tasks.append(task)
            elif outcome == "misrouted":
                n_misrouted += 1
                # A misrouted task that went to symbolic but should have gone
                # to LLM is a negative example. One that went to LLM but should
                # have gone to symbolic is a positive example.
                if route == "symbolic":
                    negative_tasks.append(task)
                else:
                    positive_tasks.append(task)
            else:
                n_unverifiable += 1

        # --- Stage 3: Capture activations and update vector ---
        updated = False
        n_pos = len(positive_tasks)
        n_neg = len(negative_tasks)
        vector_norm = float(np.linalg.norm(current_vector.values)) if current_vector else 0.0

        # v0.3.8.1: capture-coverage telemetry accumulators. A steering vector
        # is biased if capture fails at a different rate for the positive
        # (should-route-symbolic) and negative (should-route-LLM) classes, so
        # failures are tracked per class rather than silently skipped.
        n_capture_attempts = 0
        n_capture_successes = 0
        pos_attempts = pos_successes = 0
        neg_attempts = neg_successes = 0
        capture_failed = False

        if n_pos >= config.min_examples_per_update and n_neg >= config.min_examples_per_update:
            pos_activations, pos_attempts, pos_successes = _capture_activations(
                positive_tasks, model, tokenizer,
                layer=config.layer, anchor=config.anchor,
                capture_token_scope=config.capture_token_scope,
                prompt_format=prompt_format,
                class_label="positive",
            )
            neg_activations, neg_attempts, neg_successes = _capture_activations(
                negative_tasks, model, tokenizer,
                layer=config.layer, anchor=config.anchor,
                capture_token_scope=config.capture_token_scope,
                prompt_format=prompt_format,
                class_label="negative",
            )
            n_capture_attempts = pos_attempts + neg_attempts
            n_capture_successes = pos_successes + neg_successes

            if pos_activations is not None and neg_activations is not None:
                spec = SteeringSpec(
                    vector_id=f"autolearn_iter_{iteration}",
                    family="tool_policy",
                    behavior="invoke_symbolic_tool",
                    layer=config.layer,
                    alpha=config.alpha,
                    anchor=config.anchor,
                    model_id=getattr(model.config, "_name_or_path", "unknown"),
                    hidden_size=hidden_size,
                    extraction_method="contrastive_mean_difference",
                    normalization=config.normalization,
                    positive_n=n_pos,
                    negative_n=n_neg,
                    capture_anchor=config.anchor,
                    capture_prompt_format=prompt_format,
                    # v0.3.8.1: record the pooling method actually used so the
                    # manifest can reproduce the capture position.
                    capture_token_scope=config.capture_token_scope,
                )
                current_vector = contrastive_mean_direction(
                    pos_activations, neg_activations, spec,
                )
                vector_norm = float(np.linalg.norm(current_vector.values))
                updated = True
            else:
                # At least one class produced no usable activations.
                capture_failed = True

        # --- Stage 4: Evaluate on validation set ---
        val_f1 = val_accuracy = val_precision = val_recall = 0.0
        if val_tasks:
            val_list = val_tasks

            if current_vector is not None:
                if use_generate_mode:
                    # Use generate-mode routing for validation
                    val_route_list = _route_with_steering_generate(
                        val_list, model, tokenizer, current_vector, config.alpha,
                        prompt_format=prompt_format,
                        safety_limits=safety_limits,
                    )
                    val_routes_only = [route for route, _raw in val_route_list]
                    val_metrics = _evaluate_routes_simple(
                        val_list, val_routes_only,
                        label_field=label_field or "route_label",
                    )
                    val_f1 = float(val_metrics["f1"])
                    val_accuracy = float(val_metrics["accuracy"])
                    val_precision = float(val_metrics["precision"])
                    val_recall = float(val_metrics["recall"])
                else:
                    val_map = _as_task_map(val_tasks)
                    val_routed_via_logit = False
                    for resolver in ("contextual",):
                        try:
                            val_routes = _evaluate_batch_steered_routes(
                                val_list,
                                model,
                                tokenizer,
                                vectors=[current_vector],
                                alphas=[config.alpha],
                                prompt_format=prompt_format,
                                token_resolver=resolver,
                                allow_first_token_fallback=False,
                                label_scoring="sequence",
                                sequence_normalization="mean",
                                safety_limits=safety_limits,
                            )
                            val_metrics = eval_fn(
                                val_map,
                                val_routes,
                                label_field=label_field,
                                label_oracle_kind=label_oracle_kind,
                            )
                            val_f1 = float(val_metrics["f1"])
                            val_accuracy = float(val_metrics["route_accuracy"])
                            val_precision = float(val_metrics["precision"])
                            val_recall = float(val_metrics["recall"])
                            val_routed_via_logit = True
                            break
                        except MultiTokenRouteError as exc:
                            from daph_learning.telemetry import emit_fallback
                            emit_fallback(
                                event="route_fallback",
                                frm=resolver,
                                to="generate",
                                reason="multi_token_label",
                                iteration=iteration,
                                phase="validation",
                                detail=str(exc),
                            )
                            continue
                        except ContextBoundaryError as exc:
                            from daph_learning.telemetry import emit_fallback
                            emit_fallback(
                                event="route_fallback",
                                frm=resolver,
                                to="generate",
                                reason="context_boundary",
                                iteration=iteration,
                                phase="validation",
                                detail=str(exc),
                            )
                            continue
                    if not val_routed_via_logit:
                        # Fall back to generate mode for validation too
                        from daph_learning.telemetry import emit_fallback
                        emit_fallback(
                            event="route_fallback",
                            frm="logit",
                            to="generate",
                            reason="all_resolvers_failed",
                            iteration=iteration,
                            phase="validation",
                        )
                        use_generate_mode = True
                        val_route_list = _route_with_steering_generate(
                            val_list, model, tokenizer, current_vector, config.alpha,
                            prompt_format=prompt_format,
                            safety_limits=safety_limits,
                        )
                        val_routes_only = [r for r, _ in val_route_list]
                        val_metrics = _evaluate_routes_simple(
                            val_list, val_routes_only,
                            label_field=label_field or "route_label",
                        )
                        val_f1 = float(val_metrics["f1"])
                        val_accuracy = float(val_metrics["accuracy"])
                        val_precision = float(val_metrics["precision"])
                        val_recall = float(val_metrics["recall"])

        # --- Record iteration metrics ---
        train_accuracy = n_correct / len(train_tasks) if train_tasks else 0.0
        capture_coverage = (
            n_capture_successes / n_capture_attempts if n_capture_attempts else 1.0
        )
        pos_failures = pos_attempts - pos_successes
        neg_failures = neg_attempts - neg_successes
        capture_failure_rate_positive = (
            pos_failures / pos_attempts if pos_attempts else 0.0
        )
        capture_failure_rate_negative = (
            neg_failures / neg_attempts if neg_attempts else 0.0
        )
        metrics = IterationMetrics(
            iteration=iteration,
            n_train=len(train_tasks),
            n_correct=n_correct,
            n_misrouted=n_misrouted,
            n_unverifiable=n_unverifiable,
            train_accuracy=train_accuracy,
            val_f1=val_f1,
            val_accuracy=val_accuracy,
            val_precision=val_precision,
            val_recall=val_recall,
            vector_norm=vector_norm,
            n_positive_examples=n_pos,
            n_negative_examples=n_neg,
            updated=updated,
            n_capture_attempts=n_capture_attempts,
            n_capture_successes=n_capture_successes,
            capture_coverage=capture_coverage,
            capture_failure_rate_positive=capture_failure_rate_positive,
            capture_failure_rate_negative=capture_failure_rate_negative,
        )
        iterations.append(metrics)

        # Track best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_iteration = iteration
            best_vector = current_vector

        # v0.3.8.1: replace the misleading "no update => converged" early
        # stop with an explicit, diagnosed stop reason. ``not updated`` can
        # mean too few positive/negative examples, capture failure, or
        # every example being unverifiable; only the actual cause is
        # reported. The loop still stops on the first non-updating iteration
        # after iteration 0 (preserving the v0.3.8 early-stop behaviour) but
        # now labels it honestly. ``"converged"`` is reserved for a real
        # stability signal and is not emitted by the v0.3.8.1 loop.
        if not updated and iteration > 0:
            if n_unverifiable == len(train_tasks) and train_tasks:
                # Everything was unverifiable: that is the root cause of having
                # no positive/negative examples, so report it ahead of the
                # insufficient-example symptoms.
                stop_reason = "no_verifiable_experiences"
            elif capture_failed:
                stop_reason = "capture_failure"
            elif n_pos < config.min_examples_per_update:
                stop_reason = "insufficient_positive"
            elif n_neg < config.min_examples_per_update:
                stop_reason = "insufficient_negative"
            else:
                # Examples were sufficient and capture did not fail, yet no
                # update happened. With the v0.3.8.1 loop this is unreachable
                # (the only remaining cause would be a zero-norm contrastive
                # direction, which raises); fall back to capture_failure as a
                # safe, honest "something in the capture/update path failed".
                stop_reason = "capture_failure"
            break

    return AutoLearnResult(
        best_vector=best_vector,
        best_iteration=best_iteration,
        best_val_f1=best_val_f1,
        iterations=iterations,
        config=config,
        stop_reason=stop_reason,
    )
