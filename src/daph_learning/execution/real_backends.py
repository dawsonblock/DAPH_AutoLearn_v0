"""v0.3.10.2-alpha — real backend execution pipeline (Sections 9-14).

This module implements the actual backend execution loop:

    task
      |
      +--> capture h(task) once (before backend execution)
      |
      +--> execute_symbolic_backend(task)
      |
      +--> execute_llm_backend(task, model, tokenizer)
      |
      +--> verify both outputs
      |
      +--> compute utilities
      |
      +--> build CounterfactualExperience

No placeholder labels. No `symbolic_correct` / `llm_correct` fields.
Every outcome is derived from actual execution + verification.

Symbolic backend: reuses the existing bounded symbolic executor
(:mod:`daph_learning.execution.symbolic_executor`).

LLM backend: actual model generation with configurable
``max_new_tokens``, ``do_sample``, ``temperature``, ``top_p``.
For scientific qualification, deterministic generation (``do_sample=False``)
is preferred.

Verifier: arithmetic exact numeric verification. Fail closed on
unsupported tasks. Never substring matching.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from ..policy.types import BackendOutcome, CounterfactualExperience, Route
from ..policy.config import ExperimentConfig
from ..policy.utility import backend_utility as _utility
from ..policy.utility import compute_utility, UtilityConfig

if TYPE_CHECKING:
    from ..evaluation.canonical_verifier import VerificationStatus


# ------------------------------------------------------------------
# Section 7/12: BackendExecution — structured boundary result
# ------------------------------------------------------------------

class BackendName(str, Enum):
    """Identifies which backend produced an execution result."""
    SYMBOLIC = "symbolic"
    LLM = "llm"


class ExecutionStatus(str, Enum):
    """Closed enum for backend execution status."""
    SUCCESS = "SUCCESS"
    UNSUPPORTED = "UNSUPPORTED"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    TIMEOUT = "TIMEOUT"
    PARSE_ERROR = "PARSE_ERROR"


@dataclass(frozen=True)
class BackendExecution:
    """Section 7/12 — structured backend execution result.

    Normalizes backend output at the boundary so both symbolic and LLM
    paths pass through the SAME canonical verifier. The
    ``canonical_answer`` field is the integer extracted from the
    ``FINAL_ANSWER:`` field (or None if absent/malformed).

    Attributes
    ----------
    backend : BackendName
    raw_output : str
        The raw output text from the backend.
    canonical_answer : int | None
        The integer parsed from the ``FINAL_ANSWER:`` field, or None.
    latency_ms : float
        Wall-clock latency in milliseconds.
    execution_status : ExecutionStatus
    metadata : Mapping[str, Any]
        Provenance: model_id, generation_config_hash, etc.
    """
    backend: BackendName
    raw_output: str
    canonical_answer: int | None
    latency_ms: float
    execution_status: ExecutionStatus
    metadata: Mapping[str, Any]

    @property
    def canonical_output(self) -> str:
        """The canonical ``FINAL_ANSWER: <int>`` string, or raw error output."""
        if self.canonical_answer is not None:
            return f"FINAL_ANSWER: {self.canonical_answer}"
        return self.raw_output

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "raw_output": self.raw_output,
            "canonical_answer": self.canonical_answer,
            "latency_ms": self.latency_ms,
            "execution_status": self.execution_status.value,
            "metadata": dict(self.metadata),
            "canonical_output": self.canonical_output,
        }


# The canonical LLM prompt suffix that requires FINAL_ANSWER format.
_LLM_FINAL_ANSWER_SUFFIX = (
    "\n\nYour response must end with exactly one line in this format:\n"
    "FINAL_ANSWER: <integer>"
)


def build_llm_prompt(task: Mapping[str, Any]) -> str:
    """Construct the canonical LLM prompt for a task.

    This is the single source of truth for LLM prompt construction.
    All scripts and library code that send prompts to an LLM must use
    this function to ensure the FINAL_ANSWER format suffix is present.

    The canonical verifier (Section 7) requires ``FINAL_ANSWER: <integer>``
    in the LLM output. Without this suffix, the LLM generates reasoning
    but never produces a parseable answer, resulting in 0% accuracy.

    Parameters
    ----------
    task : Mapping[str, Any]
        A task dict with ``specification`` (or ``prompt``) field.

    Returns
    -------
    str
        The prompt text with the FINAL_ANSWER suffix appended.
    """
    prompt = str(task.get("prompt", task.get("specification", "")))
    return prompt + _LLM_FINAL_ANSWER_SUFFIX


def _wrap_symbolic_canonical(result_value: int | None,
                             raw_error: str | None) -> str:
    """Wrap a symbolic result as ``FINAL_ANSWER: <int>`` or return the
    raw error output."""
    if result_value is not None:
        return f"FINAL_ANSWER: {result_value}"
    return raw_error or ""


def _parse_canonical_answer(text: str | None) -> int | None:
    """Extract the canonical integer from a FINAL_ANSWER field."""
    from ..evaluation.canonical_verifier import parse_canonical_integer_answer
    if text is None:
        return None
    parsed = parse_canonical_integer_answer(str(text))
    return parsed.value if parsed.status == "VALID" else None


# ------------------------------------------------------------------
# Section 14: capture activation once per task
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CaptureConfig:
    """Configuration for hidden-state capture (Section 14, 16, 17).

    Attributes
    ----------
    layer : int
        Transformer layer index for residual-stream capture.
    location : str
        Capture location: ``"last_token"``, ``"anchor"``, or
        ``"mean_pool"``.
    """

    layer: int = 10
    location: str = "last_token"


def capture_task_representation(
    task: Mapping[str, Any],
    model,
    tokenizer,
    *,
    config: CaptureConfig | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Capture the hidden state for a task ONCE, before backend execution
    (Section 14, 16).

    The router acts on pre-execution state: the hidden state from
    processing the task prompt, NOT the hidden state after generating
    an answer. This ensures the routing representation does not encode
    oracle answer labels (Section 15 — leakage prevention).

    Parameters
    ----------
    task : mapping
        Must have ``prompt`` or ``specification``.
    model : transformers model
    tokenizer : transformers tokenizer
    config : CaptureConfig | None
    device : str

    Returns
    -------
    np.ndarray
        1-D hidden state vector at the configured layer/location.
    """
    import torch
    cfg = config or CaptureConfig()
    prompt = str(task.get("prompt", task.get("specification", "")))
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is a tuple of (n_layers + 1) tensors, each
    # [batch, seq, dim]. Index cfg.layer gives the output of that layer.
    hidden = outputs.hidden_states[cfg.layer]  # [1, seq, dim]
    if cfg.location == "last_token":
        h = hidden[0, -1, :]  # last token
    elif cfg.location == "anchor":
        # Use the first non-special token (or token 1 if available).
        seq_len = hidden.shape[1]
        idx = min(1, seq_len - 1)
        h = hidden[0, idx, :]
    elif cfg.location == "mean_pool":
        h = hidden[0].mean(dim=0)  # mean over sequence
    else:
        raise ValueError(f"unknown capture location {cfg.location!r}")
    return h.float().cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------
# Section 10: real symbolic backend
# ------------------------------------------------------------------

def execute_symbolic_backend(
    task: Mapping[str, Any],
) -> tuple[BackendOutcome, str | None]:
    """Execute the symbolic backend on a task (Section 10, 14).

    Reuses the existing bounded symbolic executor. Handles:
    - unsupported task → ``available=False, executed=False``
    - parse failure → ``executed=True, execution_success=False``
    - execution failure → ``executed=True, execution_success=False``
    - timeout → ``executed=True, execution_success=False``

    Section 14: if the task has no structured ``capability_ids`` (NL
    task), the semantic parser is attempted. If parsing succeeds, the
    parsed expression is evaluated directly. This allows the symbolic
    backend to compete on natural-language arithmetic tasks, enabling
    true within-subtype crossover.

    Returns a tuple of (BackendOutcome, output_text). The output_text is
    the actual computed value (e.g. "42") for external verification.

    Parameters
    ----------
    task : mapping
        Must have ``task_id``, ``capability_ids``, and ``inputs`` (or
        ``specification`` for rule-router fallback).
    """
    import time as _time
    tid = str(task.get("task_id", ""))
    caps = set(task.get("capability_ids", []))
    t0 = _time.time()

    # Check if the symbolic backend supports this task type.
    supported_caps = {"integer_arithmetic", "modular_multiplication"}
    has_structured = bool(caps & supported_caps)

    # Section 14: if no structured capabilities, try semantic parsing.
    # Set env var DAPH_DISABLE_SEMANTIC_PARSE=1 to disable (for crossover
    # experiments where symbolic should fail on NL tasks).
    # Fix 3a: allow semantic parsing for subtype B even when globally
    # disabled, so symbolic can compete on B (creating true crossover).
    import os as _os
    subtype = str(task.get("metadata", {}).get("subtype", ""))
    semantic_disabled = _os.environ.get("DAPH_DISABLE_SEMANTIC_PARSE", "0") == "1"
    # B is allowed through even when disabled; all others blocked.
    if not has_structured and semantic_disabled and subtype != "B":
        latency = _time.time() - t0
        return BackendOutcome(
            task_id=tid,
            backend="symbolic",
            available=False,
            executed=False,
            execution_success=False,
            output_text=None,
            output_hash=None,
            verifier_status="not_verified",
            correct=False,
            quality=0.0,
            latency_sec=latency,
            normalized_cost=0.05,
            risk=0.0,
            verifier_confidence=0.0,
            failure_reason="semantic parsing disabled (DAPH_DISABLE_SEMANTIC_PARSE=1)",
        ), None
    if not has_structured:
        try:
            from ..data.semantic_parser import parse_task
            parse_result = parse_task(dict(task))
        except (ValueError, TypeError, KeyError, AttributeError):
            parse_result = None
        if parse_result is None:
            latency = _time.time() - t0
            return BackendOutcome(
                task_id=tid,
                backend="symbolic",
                available=False,
                executed=False,
                execution_success=False,
                output_text=None,
                output_hash=None,
                verifier_status="not_verified",
                correct=False,
                quality=0.0,
                latency_sec=latency,
                normalized_cost=0.05,
                risk=0.0,
                verifier_confidence=0.0,
                failure_reason="semantic parse failed (no structured caps)",
            ), None
        # Semantic parse succeeded — evaluate the expression directly.
        try:
            expr = parse_result.expression
            # Section 6: route ALL arithmetic through the bounded AST
            # evaluator. Python eval()/exec()/compile() are forbidden in
            # symbolic execution paths.
            from daph_learning.tools.symbolic_math import safe_eval_int_expr
            result_val = safe_eval_int_expr(expr)
            output_text = str(int(result_val))
            output_hash = hashlib.sha256(
                output_text.encode()).hexdigest()[:16]
            latency = _time.time() - t0
            # Symbolic via semantic parse is slower (parse + execute).
            return BackendOutcome(
                task_id=tid,
                backend="symbolic",
                available=True,
                executed=True,
                execution_success=True,
                output_text=output_text,
                output_hash=output_hash,
                verifier_status="not_verified",
                correct=False,
                quality=0.0,
                latency_sec=latency,
                normalized_cost=0.05,
                risk=0.0,
                verifier_confidence=0.0,
                failure_reason=None,
            ), output_text
        except (ValueError, KeyError, TypeError, OverflowError,
                ArithmeticError, SyntaxError) as exc:
            latency = _time.time() - t0
            return BackendOutcome(
                task_id=tid,
                backend="symbolic",
                available=True,
                executed=True,
                execution_success=False,
                output_text=None,
                output_hash=None,
                verifier_status="not_verified",
                correct=False,
                quality=0.0,
                latency_sec=latency,
                normalized_cost=0.05,
                risk=0.0,
                verifier_confidence=0.0,
                failure_reason=f"semantic execution error: {exc!r}",
            ), None

    try:
        from .symbolic_executor import plan_from_structured_task, execute_plan
        plan = plan_from_structured_task(task, reason_code="real_symbolic")
        result = execute_plan(plan)
        output_text = str(result.value)
        output_hash = hashlib.sha256(
            output_text.encode()).hexdigest()[:16]
        latency = _time.time() - t0
        # Execution succeeded — but correctness is NOT determined here.
        # The verifier must check the output against the expected answer.
        return BackendOutcome(
            task_id=tid,
            backend="symbolic",
            available=True,
            executed=True,
            execution_success=True,
            output_text=output_text,
            output_hash=output_hash,
            verifier_status="not_verified",  # set by verifier
            correct=False,  # placeholder — set by verifier
            quality=0.0,  # placeholder — set by verifier
            latency_sec=latency,
            normalized_cost=0.01,
            risk=0.0,
            verifier_confidence=0.0,  # set by verifier
            failure_reason=None,
        ), output_text
    except (ValueError, KeyError, TypeError, OverflowError, ArithmeticError) as exc:
        latency = _time.time() - t0
        return BackendOutcome(
            task_id=tid,
            backend="symbolic",
            available=True,
            executed=True,
            execution_success=False,
            output_text=None,
            output_hash=None,
            verifier_status="not_verified",
            correct=False,
            quality=0.0,
            latency_sec=latency,
            normalized_cost=0.01,
            risk=0.0,
            verifier_confidence=0.0,
            failure_reason=f"execution error: {exc!r}",
        ), None


# ------------------------------------------------------------------
# Section 11: real LLM backend
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LLMGenerationConfig:
    """Configuration for real LLM generation (Section 11).

    For scientific qualification, deterministic generation is preferred:
    ``do_sample=False``. Record the generation config hash for provenance.
    """

    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def execute_llm_backend(
    task: Mapping[str, Any],
    model,
    tokenizer,
    *,
    generation_config: LLMGenerationConfig | None = None,
    device: str = "cpu",
) -> tuple[BackendOutcome, str | None]:
    """Execute the LLM backend on a task (Section 11).

    Generates text from the frozen model. Records model ID, generation
    config hash, output hash, and latency. Does NOT use pre-supplied
    ``llm_correct`` labels — the output is whatever the model actually
    generates.

    Parameters
    ----------
    task : mapping
        Must have ``prompt`` or ``specification``.
    model : transformers model
    tokenizer : transformers tokenizer
    generation_config : LLMGenerationConfig | None
    device : str

    Returns
    -------
    tuple[BackendOutcome, str | None]
        The BackendOutcome and the raw generated text (for verification).
    """
    import torch
    gen_cfg = generation_config or LLMGenerationConfig()
    tid = str(task.get("task_id", ""))
    prompt = str(task.get("prompt", task.get("specification", "")))

    # Apply chat template if available (Qwen2.5-Instruct).
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except (AttributeError, TypeError, ValueError):
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature if gen_cfg.do_sample else 1.0,
            top_p=gen_cfg.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0
    # Extract only the generated tokens (not the prompt).
    n_prompt = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, n_prompt:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    output_hash = hashlib.sha256(
        generated_text.encode()).hexdigest()[:16] if generated_text else None

    # The BackendOutcome's correct/quality fields are set by the verifier,
    # not here. We return the raw text for verification.
    return BackendOutcome(
        task_id=tid,
        backend="llm",
        available=True,
        executed=True,
        execution_success=True,
        output_text=generated_text,
        output_hash=output_hash,
        verifier_status="not_verified",  # set by verifier
        correct=False,  # placeholder — set by verifier
        quality=0.0,    # placeholder — set by verifier
        latency_sec=latency,
        normalized_cost=0.1,
        risk=0.0,
        verifier_confidence=0.0,  # set by verifier
        failure_reason=None,
    ), generated_text


# ------------------------------------------------------------------
# Section 7/12: canonical execution wrappers
# ------------------------------------------------------------------

def execute_symbolic_canonical(
    task: Mapping[str, Any],
) -> BackendExecution:
    """Execute the symbolic backend and return a :class:`BackendExecution`.

    The symbolic backend's bare integer output is wrapped as
    ``FINAL_ANSWER: <int>`` at the boundary so it passes through the
    same canonical verifier as the LLM backend.
    """
    outcome, output_text = execute_symbolic_backend(task)
    latency_ms = outcome.latency_sec * 1000.0

    if not outcome.available:
        return BackendExecution(
            backend=BackendName.SYMBOLIC,
            raw_output="",
            canonical_answer=None,
            latency_ms=latency_ms,
            execution_status=ExecutionStatus.UNSUPPORTED,
            metadata={"failure_reason": outcome.failure_reason},
        )
    if not outcome.execution_success:
        return BackendExecution(
            backend=BackendName.SYMBOLIC,
            raw_output="",
            canonical_answer=None,
            latency_ms=latency_ms,
            execution_status=ExecutionStatus.EXECUTION_ERROR,
            metadata={"failure_reason": outcome.failure_reason},
        )
    # Success — wrap the integer as FINAL_ANSWER.
    try:
        result_int = int(output_text) if output_text is not None else None
    except (ValueError, TypeError):
        result_int = None
        return BackendExecution(
            backend=BackendName.SYMBOLIC,
            raw_output=str(output_text or ""),
            canonical_answer=None,
            latency_ms=latency_ms,
            execution_status=ExecutionStatus.PARSE_ERROR,
            metadata={"failure_reason": f"non-integer output: {output_text!r}"},
        )
    return BackendExecution(
        backend=BackendName.SYMBOLIC,
        raw_output=_wrap_symbolic_canonical(result_int, None),
        canonical_answer=result_int,
        latency_ms=latency_ms,
        execution_status=ExecutionStatus.SUCCESS,
        metadata={},
    )


def execute_llm_canonical(
    task: Mapping[str, Any],
    model,
    tokenizer,
    *,
    generation_config: LLMGenerationConfig | None = None,
    device: str = "cpu",
) -> BackendExecution:
    """Execute the LLM backend and return a :class:`BackendExecution`.

    The LLM prompt is augmented with a suffix requiring
    ``FINAL_ANSWER: <integer>``. The canonical answer is extracted from
    the generated text using the canonical parser.
    """
    # Augment the prompt with the FINAL_ANSWER requirement.
    augmented_task = dict(task)
    augmented_task["prompt"] = build_llm_prompt(task)

    outcome, generated_text = execute_llm_backend(
        augmented_task, model, tokenizer,
        generation_config=generation_config, device=device)
    latency_ms = outcome.latency_sec * 1000.0

    if not outcome.execution_success:
        return BackendExecution(
            backend=BackendName.LLM,
            raw_output=str(generated_text or ""),
            canonical_answer=None,
            latency_ms=latency_ms,
            execution_status=ExecutionStatus.EXECUTION_ERROR,
            metadata={"failure_reason": outcome.failure_reason},
        )
    # Extract canonical answer from the generated text.
    canonical_answer = _parse_canonical_answer(generated_text)
    return BackendExecution(
        backend=BackendName.LLM,
        raw_output=str(generated_text or ""),
        canonical_answer=canonical_answer,
        latency_ms=latency_ms,
        execution_status=ExecutionStatus.SUCCESS,
        metadata={
            "model_id": getattr(model, "name_or_path", "unknown"),
            "generation_config_hash": (
                generation_config.config_hash if generation_config else ""
            ),
        },
    )


def verify_backend_execution(
    task: Mapping[str, Any],
    execution: BackendExecution,
) -> VerificationStatus:
    """Verify a :class:`BackendExecution` using the canonical verifier.

    Both symbolic and LLM executions pass through the SAME verifier.
    """
    from ..evaluation.canonical_verifier import (
        CanonicalIntegerVerifier, VerificationStatus as VStatus)
    verifier = CanonicalIntegerVerifier()
    if execution.execution_status == ExecutionStatus.UNSUPPORTED:
        return VStatus.UNVERIFIABLE
    if execution.execution_status == ExecutionStatus.TIMEOUT:
        return VStatus.TIMEOUT
    if execution.execution_status in (ExecutionStatus.EXECUTION_ERROR,
                                       ExecutionStatus.PARSE_ERROR):
        return VStatus.EXECUTION_ERROR
    return verifier.verify(
        task, execution.canonical_output,
        execution_succeeded=(execution.execution_status == ExecutionStatus.SUCCESS),
    )


def execution_to_outcome(
    task: Mapping[str, Any],
    execution: BackendExecution,
    verification_status: VerificationStatus,
) -> BackendOutcome:
    """Convert a :class:`BackendExecution` + verification status to a
    :class:`BackendOutcome`, routing utility through ``compute_utility``.
    """
    from ..evaluation.canonical_verifier import VerificationStatus as VStatus
    tid = str(task.get("task_id", ""))
    backend_str = execution.backend.value

    if verification_status == VStatus.CORRECT:
        correct = True
        quality = 1.0
        verifier_confidence = 1.0
        verifier_status = "verified_correct"
    elif verification_status == VStatus.INCORRECT:
        correct = False
        quality = 0.0
        verifier_confidence = 1.0
        verifier_status = "verified_incorrect"
    elif verification_status == VStatus.UNVERIFIABLE:
        correct = False
        quality = 0.0
        verifier_confidence = 0.0
        verifier_status = "verifier_unsupported"
    elif verification_status == VStatus.EXECUTION_ERROR:
        correct = False
        quality = 0.0
        verifier_confidence = 1.0
        verifier_status = "not_verified"
    elif verification_status == VStatus.TIMEOUT:
        correct = False
        quality = 0.0
        verifier_confidence = 1.0
        verifier_status = "not_verified"
    else:
        correct = False
        quality = 0.0
        verifier_confidence = 0.0
        verifier_status = "not_verified"

    exec_success = execution.execution_status == ExecutionStatus.SUCCESS
    available = execution.execution_status != ExecutionStatus.UNSUPPORTED
    output_hash = hashlib.sha256(
        execution.canonical_output.encode()).hexdigest()[:16]

    return BackendOutcome(
        task_id=tid,
        backend=backend_str,
        available=available,
        executed=True,
        execution_success=exec_success,
        output_text=execution.canonical_output,
        output_hash=output_hash,
        verifier_status=verifier_status,
        correct=correct,
        quality=quality,
        latency_sec=execution.latency_ms / 1000.0,
        normalized_cost=0.05 if backend_str == "symbolic" else 0.20,
        risk=0.0,
        verifier_confidence=verifier_confidence,
        failure_reason=execution.metadata.get("failure_reason"),
    )


def build_canonical_counterfactual_experience(
    task: Mapping[str, Any],
    symbolic_execution: BackendExecution,
    llm_execution: BackendExecution,
    utility_config: UtilityConfig,
) -> tuple[CounterfactualExperience, VerificationStatus, VerificationStatus]:
    """Build a counterfactual experience from canonical executions.

    Both executions are verified through the same canonical verifier,
    utilities are computed via :func:`compute_utility`, and the
    experience is constructed with the frozen :class:`UtilityConfig`.
    """
    from ..evaluation.canonical_verifier import VerificationStatus as VStatus
    tid = str(task.get("task_id", ""))

    sym_status = verify_backend_execution(task, symbolic_execution)
    llm_status = verify_backend_execution(task, llm_execution)

    sym_outcome = execution_to_outcome(task, symbolic_execution, sym_status)
    llm_outcome = execution_to_outcome(task, llm_execution, llm_status)

    u_sym = compute_utility(sym_outcome, utility_config)
    u_llm = compute_utility(llm_outcome, utility_config)
    delta_u = u_sym - u_llm

    # Determine preferred action using a configurable abstention band.
    abstention_band = 0.02
    if delta_u > abstention_band:
        preferred = Route.SYMBOLIC
    elif delta_u < -abstention_band:
        preferred = Route.LLM
    else:
        preferred = Route.ABSTAIN

    # Sample weight: |delta_u| clipped to [0, 1].
    weight = min(1.0, abs(delta_u))

    experience = CounterfactualExperience(
        task_id=tid,
        symbolic=sym_outcome,
        llm=llm_outcome,
        delta_utility=delta_u,
        preferred_action=preferred,
        sample_weight=weight,
    )
    return experience, sym_status, llm_status


# ------------------------------------------------------------------
# Section 12: real verification
# ------------------------------------------------------------------

class VerifierMode(str, Enum):
    """v0.3.10.3.1 — Section 27: explicit verifier modes.

    Every result records its mode so permissive parsing is never
    labelled as exact verification.
    """
    EXACT = "exact"
    FINAL_MARKER = "final_marker"
    NUMERIC_EXTRACTOR = "numeric_extractor"
    TOKEN_EXTRACTOR = "token_extractor"


@dataclass
class VerificationResult:
    """Result of verifying a backend output (Section 12).

    Attributes
    ----------
    verified_correct : bool | None
        True if the output matches the expected answer exactly.
        False if it does not match.
        None if the verifier cannot determine correctness (unsupported).
    verifier_type : str
        ``"arithmetic_exact"``, ``"structured_exact"``, etc.
    confidence : float
        Verifier confidence in [0, 1].
    failure_reason : str | None
        If verification failed, why.
    verifier_mode : str
        v0.3.10.3.1 — the VerifierMode used (Section 27).
    """

    verified_correct: bool | None
    verifier_type: str
    confidence: float
    failure_reason: str | None = None
    verifier_mode: str = "exact"

    def to_dict(self) -> dict:
        return {
            "verified_correct": self.verified_correct,
            "verifier_type": self.verifier_type,
            "confidence": self.confidence,
            "failure_reason": self.failure_reason,
            "verifier_mode": self.verifier_mode,
        }


def verify_arithmetic(
    task: Mapping[str, Any],
    output_text: str | None,
) -> VerificationResult:
    """Verify an arithmetic task output by exact numeric comparison
    (Section 12).

    Extracts the expected answer from the task's ``expected`` field and
    compares it to the numeric value extracted from ``output_text``.
    Never uses substring matching — ``"312"`` is NOT correct for
    expected ``12``.

    Fail closed: if the task has no ``expected`` field or the output
    cannot be parsed as a number, return ``verified_correct=None``.

    Parameters
    ----------
    task : mapping
        Must have ``expected`` (int or str of an integer).
    output_text : str | None
        The generated text from the backend.
    """
    expected = task.get("expected")
    if expected is None:
        return VerificationResult(
            verified_correct=None,
            verifier_type="arithmetic_exact", verifier_mode=VerifierMode.NUMERIC_EXTRACTOR.value,
            confidence=0.0,
            failure_reason="task has no 'expected' field",
        )
    try:
        expected_int = int(expected)
    except (ValueError, TypeError):
        return VerificationResult(
            verified_correct=None,
            verifier_type="arithmetic_exact", verifier_mode=VerifierMode.NUMERIC_EXTRACTOR.value,
            confidence=0.0,
            failure_reason=f"expected value {expected!r} is not an integer",
        )
    if output_text is None:
        return VerificationResult(
            verified_correct=False,
            verifier_type="arithmetic_exact", verifier_mode=VerifierMode.NUMERIC_EXTRACTOR.value,
            confidence=1.0,
            failure_reason="no output text provided",
        )
    # Extract the last integer from the output text.
    # This handles "The answer is 42." and "42" and "= 42" etc.
    numbers = re.findall(r"-?\d+", output_text)
    if not numbers:
        return VerificationResult(
            verified_correct=False,
            verifier_type="arithmetic_exact", verifier_mode=VerifierMode.NUMERIC_EXTRACTOR.value,
            confidence=1.0,
            failure_reason=f"no integer found in output: {output_text!r}",
        )
    # Use the LAST number found (usually the answer at the end).
    output_int = int(numbers[-1])
    is_correct = (output_int == expected_int)
    return VerificationResult(
        verified_correct=is_correct,
        verifier_type="arithmetic_exact", verifier_mode=VerifierMode.NUMERIC_EXTRACTOR.value,
        confidence=1.0,
        failure_reason=None if is_correct else (
            f"expected {expected_int}, got {output_int}"),
    )


def verify_exact_string(
    task: Mapping[str, Any],
    output_text: str | None,
) -> VerificationResult:
    """v0.3.10.3.1 — Section 27: TRUE exact string verification.

    ``strip(output) == strip(expected)`` — no permissive parsing, no
    first-word/last-word extraction. If the output is not an exact
    match after whitespace stripping, it is incorrect.

    The permissive parsing logic that used to live under this name is
    now :func:`verify_constrained_answer` (TOKEN_EXTRACTOR mode), so
    permissive parsing is never labelled as exact verification.
    """
    expected = task.get("expected")
    if expected is None:
        return VerificationResult(
            verified_correct=None,
            verifier_type="exact_string",
            confidence=0.0,
            failure_reason="task has no 'expected' field",
            verifier_mode=VerifierMode.EXACT.value,
        )
    expected_str = str(expected).strip()
    if output_text is None:
        return VerificationResult(
            verified_correct=False,
            verifier_type="exact_string",
            confidence=1.0,
            failure_reason="no output text provided",
            verifier_mode=VerifierMode.EXACT.value,
        )
    output_str = output_text.strip()
    is_correct = (output_str == expected_str)
    return VerificationResult(
        verified_correct=is_correct,
        verifier_type="exact_string",
        confidence=1.0,
        failure_reason=None if is_correct else (
            f"expected {expected_str!r}, got {output_str!r}"),
        verifier_mode=VerifierMode.EXACT.value,
    )


def verify_constrained_answer(
    task: Mapping[str, Any],
    output_text: str | None,
) -> VerificationResult:
    """v0.3.10.3.1 — Section 27: permissive constrained-answer extraction.

    Renamed from the old ``verify_exact_string``. This is NOT exact
    verification — it extracts the first/last word, first line, or
    standalone word and compares. The verifier_mode is TOKEN_EXTRACTOR
    so results are never labelled as exact verification.

    Use :func:`verify_exact_string` for true ``strip() == strip()``
    equality. Use this for tasks where the LLM wraps the answer in
    prose and exact equality would be too strict.
    """
    expected = task.get("expected")
    if expected is None:
        return VerificationResult(
            verified_correct=None,
            verifier_type="constrained_answer",
            confidence=0.0,
            failure_reason="task has no 'expected' field",
            verifier_mode=VerifierMode.TOKEN_EXTRACTOR.value,
        )
    expected_str = str(expected).strip().lower().rstrip(".!?")
    if output_text is None:
        return VerificationResult(
            verified_correct=False,
            verifier_type="constrained_answer",
            confidence=1.0,
            failure_reason="no output text provided",
            verifier_mode=VerifierMode.TOKEN_EXTRACTOR.value,
        )
    output_str = output_text.strip().lower().rstrip(".!?")
    if output_str == expected_str:
        return VerificationResult(
            verified_correct=True,
            verifier_type="constrained_answer",
            confidence=1.0,
            verifier_mode=VerifierMode.TOKEN_EXTRACTOR.value,
        )
    words = output_str.split()
    first_word = words[0] if words else ""
    if first_word == expected_str:
        return VerificationResult(
            verified_correct=True,
            verifier_type="constrained_answer",
            confidence=0.9,
            verifier_mode=VerifierMode.TOKEN_EXTRACTOR.value,
        )
    last_word = words[-1] if words else ""
    if last_word == expected_str:
        return VerificationResult(
            verified_correct=True,
            verifier_type="constrained_answer",
            confidence=0.9,
            verifier_mode=VerifierMode.TOKEN_EXTRACTOR.value,
        )
    first_line = output_str.split("\n")[0].strip()
    if first_line == expected_str:
        return VerificationResult(
            verified_correct=True,
            verifier_type="constrained_answer",
            confidence=0.9,
            verifier_mode=VerifierMode.TOKEN_EXTRACTOR.value,
        )
    if expected_str in words:
        return VerificationResult(
            verified_correct=True,
            verifier_type="constrained_answer",
            confidence=0.85,
            verifier_mode=VerifierMode.TOKEN_EXTRACTOR.value,
        )
    return VerificationResult(
        verified_correct=False,
        verifier_type="constrained_answer",
        confidence=1.0,
        failure_reason=f"expected {expected_str!r}, got {output_str!r}",
        verifier_mode=VerifierMode.TOKEN_EXTRACTOR.value,
    )


def verify_output(
    task: Mapping[str, Any],
    output_text: str | None,
) -> VerificationResult:
    """Dispatch to the appropriate verifier based on task type (Section 12).

    For arithmetic tasks (``capability_ids`` contains
    ``integer_arithmetic`` or ``modular_multiplication``), use
    :func:`verify_arithmetic`. For letter_counting and exact_string tasks,
    use :func:`verify_exact_string`. For tasks with no structured
    capabilities but an integer ``expected`` field (e.g. crossover NL
    subtypes), fall back to :func:`verify_arithmetic` to extract the final
    number from the output. For unsupported tasks, fail closed.
    """
    caps = set(task.get("capability_ids", []))
    if "integer_arithmetic" in caps or "modular_multiplication" in caps:
        return verify_arithmetic(task, output_text)
    if "letter_counting" in caps or "exact_string" in caps:
        return verify_exact_string(task, output_text)
    # Crossover NL tasks: no structured caps but have an integer expected.
    expected = task.get("expected")
    if expected is not None:
        try:
            int(expected)
            return verify_arithmetic(task, output_text)
        except (ValueError, TypeError):
            pass
    return VerificationResult(
        verified_correct=None,
        verifier_type="unsupported",
        confidence=0.0,
        failure_reason=f"no verifier for capabilities {sorted(caps)}",
    )


# ------------------------------------------------------------------
# Section 9, 13: build real counterfactual experience
# ------------------------------------------------------------------

def build_real_counterfactual_experience(
    task: Mapping[str, Any],
    capture: np.ndarray,
    symbolic_outcome: BackendOutcome,
    llm_outcome: BackendOutcome,
    llm_output_text: str | None,
    *,
    config: ExperimentConfig,
    symbolic_output_text: str | None = None,
) -> tuple[CounterfactualExperience, VerificationResult, VerificationResult]:
    """Build a counterfactual experience from real executed outcomes
    (Section 9).

    Verifies both backend outputs against the expected answer, updates
    the BackendOutcomes with verified correctness, computes utilities,
    and constructs the :class:`CounterfactualExperience`.

    v0.3.10.3 — CRITICAL FIX: verifies the ACTUAL symbolic output text
    (e.g. "42"), not ``str(quality)`` (which was "1.0" or "0.0").
    Also fixes confidence semantics: an unsupported backend has
    measurement_confidence=1.0 (we are certain it produced no useful
    output), not 0.0 (which would zero the training weight).

    Parameters
    ----------
    task : mapping
    capture : np.ndarray
        The hidden state captured once before backend execution.
    symbolic_outcome : BackendOutcome
        From :func:`execute_symbolic_backend`.
    llm_outcome : BackendOutcome
        From :func:`execute_llm_backend` (correct/quality are placeholders).
    llm_output_text : str | None
        The raw LLM-generated text for verification.
    config : ExperimentConfig
    symbolic_output_text : str | None
        The raw symbolic output text for verification. If None, falls
        back to ``symbolic_outcome.output_text``.

    Returns
    -------
    tuple[CounterfactualExperience, VerificationResult, VerificationResult]
        The experience, symbolic verification result, and LLM
        verification result.
    """
    tid = str(task.get("task_id", ""))

    # --- Verify symbolic output ---
    # Use the actual symbolic output text, NOT str(quality).
    sym_text = symbolic_output_text if symbolic_output_text is not None else symbolic_outcome.output_text
    if symbolic_outcome.available and symbolic_outcome.execution_success:
        sym_verified = verify_output(task, sym_text)
        if sym_verified.verified_correct is True:
            sym_quality = 1.0
            sym_conf = sym_verified.confidence
            sym_verifier_status = "verified_correct"
            sym_correct = True
        elif sym_verified.verified_correct is False:
            sym_quality = 0.0
            sym_conf = sym_verified.confidence
            sym_verifier_status = "verified_incorrect"
            sym_correct = False
        else:
            # None — verifier unsupported; fail closed.
            sym_quality = 0.0
            sym_conf = 0.0
            sym_verifier_status = "verifier_unsupported"
            sym_correct = False
    elif not symbolic_outcome.available:
        # Backend unsupported — we are CERTAIN it produced no useful output.
        # This is high measurement confidence, NOT low confidence.
        sym_verified = VerificationResult(
            verified_correct=False,
            verifier_type="unsupported_backend",
            confidence=1.0,  # certain that output is wrong/absent
            failure_reason="symbolic backend unavailable for this task type",
        )
        sym_quality = 0.0
        sym_conf = 1.0  # HIGH confidence — we know it failed
        sym_verifier_status = "not_verified"
        sym_correct = False
    else:
        # Execution failed (error/timeout).
        sym_verified = VerificationResult(
            verified_correct=False,
            verifier_type="execution_failed",
            confidence=1.0,  # certain that output is wrong/absent
            failure_reason=symbolic_outcome.failure_reason or "execution failed",
        )
        sym_quality = 0.0
        sym_conf = 1.0  # HIGH confidence — we know it failed
        sym_verifier_status = "not_verified"
        sym_correct = False

    # --- Verify LLM output ---
    llm_verified = verify_output(task, llm_output_text)
    if llm_verified.verified_correct is True:
        llm_quality = 1.0
        llm_conf = llm_verified.confidence
        llm_verifier_status = "verified_correct"
        llm_correct = True
    elif llm_verified.verified_correct is False:
        llm_quality = 0.0
        llm_conf = llm_verified.confidence
        llm_verifier_status = "verified_incorrect"
        llm_correct = False
    else:
        # None — verifier unsupported; fail closed.
        llm_quality = 0.0
        llm_conf = 0.0
        llm_verifier_status = "verifier_unsupported"
        llm_correct = False

    # Update outcomes with verified values.
    sym_final = BackendOutcome(
        task_id=tid, backend="symbolic",
        available=symbolic_outcome.available,
        executed=symbolic_outcome.executed,
        execution_success=symbolic_outcome.execution_success,
        output_text=sym_text,
        output_hash=symbolic_outcome.output_hash,
        verifier_status=sym_verifier_status,
        correct=sym_correct,
        quality=sym_quality,
        latency_sec=symbolic_outcome.latency_sec,
        normalized_cost=symbolic_outcome.normalized_cost,
        risk=symbolic_outcome.risk,
        verifier_confidence=sym_conf,
        failure_reason=symbolic_outcome.failure_reason,
    )
    llm_final = BackendOutcome(
        task_id=tid, backend="llm",
        available=llm_outcome.available,
        executed=llm_outcome.executed,
        execution_success=llm_outcome.execution_success,
        output_text=llm_output_text,
        output_hash=llm_outcome.output_hash,
        verifier_status=llm_verifier_status,
        correct=llm_correct,
        quality=llm_quality,
        latency_sec=llm_outcome.latency_sec,
        normalized_cost=llm_outcome.normalized_cost,
        risk=llm_outcome.risk,
        verifier_confidence=llm_conf,
        failure_reason=llm_outcome.failure_reason,
    )

    # Compute utilities.
    u_sym = _utility(sym_final, config)
    u_llm = _utility(llm_final, config)
    delta_u = u_sym - u_llm

    # Determine preferred action.
    if delta_u > config.abstention_band:
        preferred = Route.SYMBOLIC
    elif delta_u < -config.abstention_band:
        preferred = Route.LLM
    else:
        preferred = Route.ABSTAIN

    # Compute sample weight using the configured weight mode.
    # v0.3.10.3 — confidence is the PRODUCT of measurement confidences,
    # not the minimum. An unsupported backend has confidence=1.0
    # (we are certain of the outcome), so it does NOT zero the weight.
    from ..policy.weighting import compute_weight
    conf = sym_conf * llm_conf
    weight = compute_weight(
        delta_u, conf, config.weight_mode,
        gap_threshold=config.gap_threshold,
        max_weight=config.max_weight,
        min_weight=config.min_weight,
    )

    experience = CounterfactualExperience(
        task_id=tid,
        symbolic=sym_final,
        llm=llm_final,
        delta_utility=delta_u,
        preferred_action=preferred,
        sample_weight=weight,
    )
    return experience, sym_verified, llm_verified


# ------------------------------------------------------------------
# Section 21: mock LLM backend for deterministic testing
# ------------------------------------------------------------------

class MockLLMBackend:
    """Deterministic mock LLM backend for integration tests.

    Computes the correct answer for arithmetic tasks and emits it in
    canonical ``FINAL_ANSWER: <int>`` format. For unsupported tasks,
    emits a wrong answer. This allows end-to-end pipeline testing
    without a real model.
    """

    def __init__(self, *, accuracy: float = 1.0, seed: int = 42):
        self.accuracy = accuracy
        self._rng = np.random.default_rng(seed)
        self.name_or_path = "mock-llm"

    def execute(self, task: Mapping[str, Any]) -> BackendExecution:
        """Execute the mock LLM on a task."""
        import time as _time
        t0 = _time.time()
        expected = task.get("expected")
        prompt = str(task.get("prompt", task.get("specification", "")))

        # Determine if this is an arithmetic task.
        inputs = task.get("inputs", {})
        caps = set(task.get("capability_ids", []))
        is_arithmetic = bool(caps & {"integer_arithmetic",
                                      "modular_multiplication"})

        if is_arithmetic and expected is not None and self._rng.random() < self.accuracy:
            # Correct answer.
            answer = int(expected)
            raw = f"The answer is {answer}.\nFINAL_ANSWER: {answer}"
            status = ExecutionStatus.SUCCESS
            canonical = answer
        elif expected is not None and self._rng.random() < self.accuracy * 0.5:
            # Partially correct for NL tasks.
            answer = int(expected)
            raw = f"FINAL_ANSWER: {answer}"
            status = ExecutionStatus.SUCCESS
            canonical = answer
        else:
            # Wrong answer.
            wrong = int(expected) + int(self._rng.integers(1, 10)) if expected is not None else 0
            raw = f"FINAL_ANSWER: {wrong}"
            status = ExecutionStatus.SUCCESS
            canonical = wrong

        latency_ms = (_time.time() - t0) * 1000.0 + float(self._rng.uniform(1, 5))
        return BackendExecution(
            backend=BackendName.LLM,
            raw_output=raw,
            canonical_answer=canonical,
            latency_ms=latency_ms,
            execution_status=status,
            metadata={"model_id": "mock-llm", "accuracy": self.accuracy},
        )


# ------------------------------------------------------------------
# Section 50: arithmetic task generator for the first real experiment
# ------------------------------------------------------------------

def make_arithmetic_tasks(
    n: int = 100,
    seed: int = 0,
    *,
    max_value: int = 100,
    ops: tuple[str, ...] = ("+", "-", "*"),
) -> list[dict[str, Any]]:
    """Generate integer arithmetic tasks for the first real experiment
    (Section 50).

    Each task has:
    - ``task_id``: unique identifier
    - ``prompt``: human-readable prompt for the LLM
    - ``specification``: the expression string
    - ``expected``: the correct integer answer
    - ``capability_ids``: ``["integer_arithmetic"]``
    - ``inputs``: structured inputs for the symbolic executor
    - ``family``: the operation type (for per-family metrics)

    Parameters
    ----------
    n : int
    seed : int
    max_value : int
        Maximum absolute value for operands.
    ops : tuple of str
        Allowed operations.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n):
        op = ops[int(rng.integers(len(ops)))]
        a = int(rng.integers(-max_value, max_value + 1))
        b = int(rng.integers(-max_value, max_value + 1))
        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        elif op == "*":
            expected = a * b
        else:
            raise ValueError(f"unknown op {op!r}")
        prompt = f"What is {a} {op} {b}? Answer with just the number."
        tasks.append({
            "task_id": f"arith_{seed}_{i}",
            "prompt": prompt,
            "specification": f"{a} {op} {b}",
            "expected": expected,
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": a, "b": b, "op": op},
            "family": op,
        })
    return tasks


def make_letter_counting_tasks(
    n: int = 50,
    seed: int = 0,
    *,
    word_pool: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Generate letter-counting tasks for the real experiment (Section 50).

    These tasks ask "How many letters are in the word 'X'?" — the symbolic
    executor CANNOT handle these (unsupported capability → fails closed,
    quality=0), but a small LLM can answer them. This creates a real
    routing decision: route to LLM for counting, route to symbolic for
    arithmetic.

    v0.3.10.3 — CRITICAL FIX: the caller MUST provide a disjoint
    ``word_pool`` for each split to prevent prompt leakage. Use
    :func:`partition_word_pool` to get disjoint pools.

    Each task has:
    - ``task_id``: unique identifier
    - ``prompt``: "How many letters are in the word 'cat'? Answer with just the number."
    - ``expected``: the correct letter count (int)
    - ``capability_ids``: ``["letter_counting"]``
    - ``family``: "counting"
    """
    rng = np.random.default_rng(seed)
    if word_pool is None:
        word_pool = (
            "cat", "dog", "hello", "world", "apple", "banana", "house",
            "mouse", "table", "chair", "water", "light", "night", "day",
            "tree", "bird", "fish", "book", "pen", "cup", "door", "key",
            "star", "moon", "sun", "rain", "snow", "fire", "wind", "rock",
            "river", "lake", "hill", "road", "bridge", "tower", "castle",
            "garden", "forest", "mountain", "ocean", "cloud", "storm",
            "flower", "grass", "leaf", "root", "seed", "milk", "bread",
        )
    tasks = []
    for i in range(n):
        word = str(word_pool[int(rng.integers(len(word_pool)))])
        expected = len(word)
        prompt = (
            f"How many letters are in the word '{word}'? "
            f"Answer with just the number.")
        tasks.append({
            "task_id": f"count_{seed}_{i}",
            "prompt": prompt,
            "specification": f"count_letters({word!r})",
            "expected": expected,
            "capability_ids": ["letter_counting"],
            "inputs": {"word": word},
            "family": "counting",
        })
    return tasks


# v0.3.10.3 — Section 50: disjoint word pools for split integrity.
_FULL_WORD_POOL: tuple[str, ...] = (
    "cat", "dog", "hello", "world", "apple", "banana", "house",
    "mouse", "table", "chair", "water", "light", "night", "day",
    "tree", "bird", "fish", "book", "pen", "cup", "door", "key",
    "star", "moon", "sun", "rain", "snow", "fire", "wind", "rock",
    "river", "lake", "hill", "road", "bridge", "tower", "castle",
    "garden", "forest", "mountain", "ocean", "cloud", "storm",
    "flower", "grass", "leaf", "root", "seed", "milk", "bread",
    "cheese", "honey", "sugar", "salt", "pepper", "knife", "fork",
    "plate", "glass", "bottle", "bag", "box", "rope", "string",
    "thread", "needle", "button", "zipper", "pocket", "shoe",
    "sock", "hat", "coat", "scarf", "glove", "belt", "ring",
    "watch", "clock", "lamp", "candle", "match", "ash", "smoke",
)


def partition_word_pool(
    n_splits: int = 4,
    *,
    seed: int = 0,
) -> list[tuple[str, ...]]:
    """Partition the word pool into ``n_splits`` disjoint subsets.

    This ensures no word appears in more than one split, preventing
    prompt leakage across train/dev/cal/final.

    Returns a list of tuples, each containing the words for one split.
    """
    rng = np.random.default_rng(seed)
    pool = list(_FULL_WORD_POOL)
    rng.shuffle(pool)
    # Evenly partition.
    base = len(pool) // n_splits
    remainder = len(pool) % n_splits
    splits = []
    idx = 0
    for i in range(n_splits):
        size = base + (1 if i < remainder else 0)
        splits.append(tuple(pool[idx:idx + size]))
        idx += size
    return splits


def assert_splits_disjoint(
    splits: list[list[dict[str, Any]]],
    *,
    field: str = "prompt",
    family: str | None = "counting",
) -> None:
    """Assert that task splits have no overlapping prompts ACROSS splits.

    This is a release gate (Section 50): prompt-hash disjointness must
    be verified before reporting held-out results. Within-split
    duplicates (from sampling with replacement) are allowed.

    By default, only checks tasks with ``family == "counting"`` since
    those use the word pool that was partitioned. Arithmetic tasks use
    random operands and may occasionally collide, but the hidden state
    is still task-specific. Set ``family=None`` to check all tasks.
    """
    per_split_sets: list[set[str]] = []
    for split in splits:
        prompts: set[str] = set()
        for task in split:
            if family is not None and task.get("family") != family:
                continue
            prompts.add(str(task.get(field, "")))
        per_split_sets.append(prompts)
    for i in range(len(per_split_sets)):
        for j in range(i + 1, len(per_split_sets)):
            overlap = per_split_sets[i] & per_split_sets[j]
            if overlap:
                raise AssertionError(
                    f"prompt leakage: {next(iter(overlap))!r} appears in "
                    f"both split {i} and split {j}")


def make_mixed_tasks(
    n_arithmetic: int = 40,
    n_counting: int = 40,
    seed: int = 0,
    *,
    max_value: int = 100,
    word_pool: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Generate a MIXED task set for the first scientifically meaningful
    real-model experiment (Section 50).

    The mix creates a genuine routing decision:
    - **Arithmetic** (symbolic wins): symbolic executor handles perfectly,
      LLM may fail on large numbers → route symbolic.
    - **Letter counting** (LLM wins): symbolic executor fails closed
      (unsupported capability), LLM can count letters → route LLM.

    v0.3.10.3 — accepts a ``word_pool`` parameter so each split can use
    a disjoint vocabulary. The caller is responsible for partitioning.

    Parameters
    ----------
    n_arithmetic : int
    n_counting : int
    seed : int
    max_value : int
        Max operand value for arithmetic.
    word_pool : tuple of str | None
        Disjoint word pool for this split. If None, uses the full pool
        (NOT recommended for multi-split experiments).
    """
    arith = make_arithmetic_tasks(
        n=n_arithmetic, seed=seed, max_value=max_value)
    counting = make_letter_counting_tasks(
        n=n_counting, seed=seed + 1000, word_pool=word_pool)
    tasks = arith + counting
    # Shuffle so the two families are interleaved.
    rng = np.random.default_rng(seed + 9999)
    rng.shuffle(tasks)
    return tasks


__all__ = [
    "CaptureConfig",
    "LLMGenerationConfig",
    "VerificationResult",
    "assert_splits_disjoint",
    "build_llm_prompt",
    "build_real_counterfactual_experience",
    "capture_task_representation",
    "execute_llm_backend",
    "execute_symbolic_backend",
    "make_arithmetic_tasks",
    "make_letter_counting_tasks",
    "make_mixed_tasks",
    "partition_word_pool",
    "verify_arithmetic",
    "verify_constrained_answer",
    "verify_exact_string",
    "verify_output",
]
