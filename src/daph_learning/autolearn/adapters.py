"""v0.3.9 — model-backed adapters for the counterfactual learning loop.

This module provides adapters that bridge the model-optional
:class:`~daph_learning.autolearn.trust_region.CounterfactualLearningLoop` to
real HuggingFace model + tokenizer components:

- :func:`make_model_route_policy`: wraps :func:`daph_learning.routing.policy.route_tasks`
  into a :data:`RoutePolicyFn` that uses a steering vector to route.
- :func:`make_model_capture_fn`: wraps the activation-capture logic from
  :func:`daph_learning.autolearn.loop._capture_activations` into the new
  :class:`~daph_learning.autolearn.experience.CaptureResult` contract.
- :func:`make_model_execute_fn`: wraps the symbolic executor and LLM runner
  into an :data:`ExecuteFn`.

These adapters keep the trust-region optimizer decoupled from the model
implementation (Section 2 of the repair spec).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .experience import CaptureResult, CapturedActivation
from .route_policy import (
    PolicyRouteDecision,
    RoutePolicyFn,
    SafetyClampTelemetry,
    SteeringPolicyConfig,
)


def make_model_route_policy(
    model: Any,
    tokenizer: Any,
    *,
    prompt_format: str = "raw",
) -> RoutePolicyFn:
    """Build a :data:`RoutePolicyFn` backed by a real HuggingFace model.

    The adapter constructs a :class:`daph_learning.steering.types.SteeringVector`
    from the supplied numpy array + :class:`SteeringPolicyConfig`, then calls
    :func:`daph_learning.routing.policy.route_tasks` to get model-mediated
    routing decisions. The results are coerced into
    :class:`PolicyRouteDecision` records with full provenance.
    """

    def _route(
        tasks: Sequence[Mapping[str, Any]],
        steering_vector: np.ndarray,
        config: SteeringPolicyConfig,
    ) -> list[PolicyRouteDecision]:
        from daph_learning.routing.policy import route_tasks
        from daph_learning.steering.types import SteeringSpec, SteeringVector

        vec = np.asarray(steering_vector, dtype=np.float32)
        hidden_size = (
            getattr(model.config, "hidden_size", None)
            or getattr(model.config, "n_embd", None)
            or vec.shape[0]
        )
        spec = SteeringSpec(
            vector_id=config.vector_version,
            family="reasoning_policy",
            behavior="autolearn_counterfactual",
            layer=config.layer,
            alpha=config.alpha,
            anchor=config.token_location if config.token_location != "mean_sequence" else "ACTION",
            model_id=getattr(model.config, "_name_or_path", None),
            hidden_size=hidden_size,
        )
        steering = SteeringVector(spec=spec, values=vec)
        decisions = route_tasks(
            tasks,
            model=model,
            tokenizer=tokenizer,
            steering=steering,
            alpha=config.alpha,
            prompt_format=prompt_format,
        )
        results: list[PolicyRouteDecision] = []
        for d in decisions:
            results.append(PolicyRouteDecision(
                task_id=d.task_id,
                route=d.route,
                confidence=d.confidence,
                vector_id=config.vector_version,
                model_revision=config.model_revision,
                tokenizer_revision=config.tokenizer_revision,
                layer=config.layer,
                token_location=config.token_location,
                alpha=config.alpha,
                raw_scores=dict(d.raw_scores),
            ))
        return results

    return _route


def make_model_capture_fn(
    model: Any,
    tokenizer: Any,
    *,
    layer: int,
    anchor: str = "ACTION",
    capture_token_scope: str = "anchor",
    prompt_format: str = "raw",
) -> Callable[[Sequence[Mapping[str, Any]], str], CaptureResult]:
    """Build a capture_fn backed by a real model.

    Wraps :func:`daph_learning.autolearn.loop._capture_activations` and
    converts the legacy ``(activations, n_attempts, n_successes)`` tuple into
    a :class:`CaptureResult` with per-task :class:`CapturedActivation` records.
    """

    def _capture(
        tasks: Sequence[Mapping[str, Any]],
        class_label: str,
    ) -> CaptureResult:
        from daph_learning.autolearn.loop import _capture_activations

        task_list = [dict(t) for t in tasks]
        activations, n_attempts, n_successes = _capture_activations(
            task_list, model, tokenizer,
            layer=layer, anchor=anchor,
            capture_token_scope=capture_token_scope,
            prompt_format=prompt_format,
            class_label=class_label,
        )
        # Build per-task CapturedActivation records. The legacy capture returns
        # a stacked array of successful captures (positional, no task IDs).
        # We map them back to task IDs positionally — but only for successful
        # captures, which preserves the task-ID alignment invariant.
        captured: list[CapturedActivation] = []
        if activations is not None and activations.ndim == 2:
            n_success = activations.shape[0]
            # The legacy capture skips failed tasks, so successful captures
            # are a subset. We assign task IDs in order (the legacy capture
            # processes tasks in input order and appends on success).
            success_idx = 0
            for task in task_list:
                tid = str(task.get("task_id", ""))
                if success_idx < n_success:
                    captured.append(CapturedActivation(
                        task_id=tid,
                        activation=activations[success_idx],
                        layer=layer,
                        token_location=capture_token_scope,
                        success=True,
                    ))
                    success_idx += 1
                else:
                    captured.append(CapturedActivation(
                        task_id=tid,
                        layer=layer,
                        token_location=capture_token_scope,
                        success=False,
                        failure_reason="capture_skipped",
                    ))
        else:
            # All captures failed.
            for task in task_list:
                tid = str(task.get("task_id", ""))
                captured.append(CapturedActivation(
                    task_id=tid,
                    layer=layer,
                    token_location=capture_token_scope,
                    success=False,
                    failure_reason="capture_failed",
                ))
        return CaptureResult(
            activations=captured,
            n_attempts=n_attempts,
            n_successes=n_successes,
        )

    return _capture


def make_model_execute_fn(
    model: Any,
    tokenizer: Any,
    *,
    prompt_format: str = "raw",
) -> Callable[[Mapping[str, Any], str], Mapping[str, Any]]:
    """Build an execute_fn that runs the symbolic or LLM backend.

    For ``backend == "symbolic"``: uses the bounded symbolic executor from
    :mod:`daph_learning.autolearn.loop`.
    For ``backend == "llm"``: generates a response with the HuggingFace model.
    """

    def _execute(task: Mapping[str, Any], backend: str) -> Mapping[str, Any]:
        if backend == "symbolic":
            from daph_learning.autolearn.loop import _execute_symbolic
            output, success = _execute_symbolic(dict(task))
            return {
                "output": output,
                "execution_succeeded": success,
                "latency_ms": None,
                "compute_cost": None,
                "failure_type": None if success else "execution_error",
            }
        elif backend == "llm":
            import time
            from daph_learning.routing.steered_router import build_route_prompt
            prompt = build_route_prompt(dict(task))
            if prompt_format == "chat" and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                rendered = prompt
            import torch
            encoded = tokenizer(rendered, return_tensors="pt")
            device = next(model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            t0 = time.perf_counter()
            with torch.inference_mode():
                gen_out = model.generate(
                    **encoded, max_new_tokens=128, do_sample=False,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            generated = tokenizer.decode(
                gen_out[0][encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            return {
                "output": generated,
                "execution_succeeded": True,
                "latency_ms": elapsed_ms,
                "compute_cost": float(len(generated)),
                "failure_type": None,
            }
        else:
            return {
                "output": None,
                "execution_succeeded": False,
                "latency_ms": None,
                "compute_cost": None,
                "failure_type": f"unknown_backend:{backend}",
            }

    return _execute


__all__ = [
    "make_model_capture_fn",
    "make_model_execute_fn",
    "make_model_route_policy",
]
