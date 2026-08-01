"""DAPH v0.4 — Generic action executors for executive qualification.

This module provides the executor framework that runs actions on tasks
and produces :class:`~daph_learning.executive.ActionExecution` records.

Each executor implements the :class:`ActionExecutor` protocol:

    execute(task) → ActionExecution

The executors handle:
- LLM generation (direct, retrieval-augmented, decomposed)
- Verification (via the canonical verifier)
- Latency / cost measurement

For the B1 experiment (direct vs retrieval vs decompose), three
executors are provided. Each can work with either:
- A vLLM API server (preferred for production)
- In-process HuggingFace model (for development)
- A mock generator (for testing)
"""

from __future__ import annotations

import time
import re
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from daph_learning.executive.types import (
    ActionExecution,
    ActionDescriptor,
    ActionSpace,
    ExecutiveState,
    CounterfactualSet,
)


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Action Executor Protocol
# ──────────────────────────────────────────────────────────────────────

@runtime_checkable
class ActionExecutor(Protocol):
    """Protocol for executing an action on a task.

    All v0.4 action executors implement this interface.
    """

    @property
    def action_id(self) -> str:
        ...  # pragma: no cover

    def execute(self, task: Mapping[str, Any]) -> ActionExecution:
        ...  # pragma: no cover


# ──────────────────────────────────────────────────────────────────────
# Section 2 — LLM Generation Backend
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LLMGenerationConfig:
    """Configuration for LLM generation."""
    model_id: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    # vLLM API server
    vllm_port: int = 8000
    vllm_api_key: str = ""
    vllm_api_key_env: str = ""
    vllm_base_url: str = ""
    vllm_max_concurrent: int = 64


# The FINAL_ANSWER suffix (shared with the legacy build_llm_prompt)
_FINAL_ANSWER_SUFFIX = "\n\nProvide your final answer as: FINAL_ANSWER: <integer>"


def _build_prompt(task: Mapping[str, Any], *, suffix: str = _FINAL_ANSWER_SUFFIX) -> str:
    """Build the base prompt from a task."""
    prompt = str(task.get("prompt", task.get("specification", "")))
    return prompt + suffix


def _parse_final_answer(text: str | None) -> int | None:
    """Extract the integer from a FINAL_ANSWER: <int> field."""
    if text is None:
        return None
    from daph_learning.evaluation.canonical_verifier import parse_canonical_integer_answer
    parsed = parse_canonical_integer_answer(str(text))
    return parsed.value if parsed.status == "VALID" else None


def _verify_answer(task: Mapping[str, Any], answer: int | None) -> bool | None:
    """Verify an answer against the task's expected answer.

    Returns True/False/None (None = unverifiable).
    """
    expected = task.get("answer")
    if expected is None:
        return None
    try:
        expected_int = int(expected)
    except (ValueError, TypeError):
        return None
    if answer is None:
        return False
    return answer == expected_int


def _call_vllm_api(
    prompt: str,
    config: LLMGenerationConfig,
) -> tuple[str, float]:
    """Call a vLLM OpenAI-compatible API server for a single prompt.

    Returns (generated_text, latency_ms).
    """
    import os
    api_key = config.vllm_api_key
    if config.vllm_api_key_env:
        api_key = os.environ.get(config.vllm_api_key_env, api_key)

    base_url = config.vllm_base_url or f"http://localhost:{config.vllm_port}/v1"

    payload = json.dumps({
        "model": config.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            latency_ms = (time.time() - t0) * 1000.0
            return text, latency_ms
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000.0
        return f"ERROR: {e}", latency_ms


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Direct Reasoning Executor
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DirectReasoningExecutor:
    """Execute direct LLM reasoning (no decomposition, no retrieval).

    This is the simplest action: send the task prompt to the LLM and
    parse the FINAL_ANSWER from the response.

    Attributes
    ----------
    action_id : str
        Must be ``"action.reasoning.direct"``.
    config : LLMGenerationConfig
    generate_fn : callable | None
        Optional override for generation. If None, uses vLLM API.
        Signature: ``(prompt, config) → (text, latency_ms)``.
    cost_estimate : float
        Normalized compute cost for this action.
    """

    action_id: str = "action.reasoning.direct"
    config: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    generate_fn: Callable[[str, LLMGenerationConfig], tuple[str, float]] | None = None
    cost_estimate: float = 0.15

    def execute(self, task: Mapping[str, Any]) -> ActionExecution:
        prompt = _build_prompt(task)
        t0 = time.time()

        if self.generate_fn is not None:
            text, latency_ms = self.generate_fn(prompt, self.config)
        else:
            text, latency_ms = _call_vllm_api(prompt, self.config)

        answer = _parse_final_answer(text)
        verified = _verify_answer(task, answer)

        failure_type = None
        if text.startswith("ERROR:"):
            failure_type = "execution_error"
        elif answer is None and not text.startswith("ERROR:"):
            failure_type = "parse_error"

        return ActionExecution(
            action_id=self.action_id,
            selected=False,  # set by the caller
            executed=True,
            output=text,
            verified_correct=verified,
            verifier_name="numeric_exact",
            latency_ms=latency_ms,
            compute_cost=self.cost_estimate,
            failure_type=failure_type,
        )


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Retrieval-Augmented Executor
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalVectorExecutor:
    """Execute retrieval-augmented LLM reasoning.

    Retrieves similar examples from a provided example store, appends
    them to the prompt as in-context examples, then generates.

    Attributes
    ----------
    action_id : str
        Must be ``"action.retrieval.vector"``.
    config : LLMGenerationConfig
    examples : list[dict]
        The retrieval store — list of example tasks with ``prompt``
        and ``answer`` fields.
    retrieve_fn : callable | None
        Optional override for retrieval. Signature:
        ``(query, examples, k) → list[dict]``. Default: random.
    generate_fn : callable | None
        Optional override for generation.
    cost_estimate : float
    n_retrieved : int
        Number of examples to retrieve.
    """

    action_id: str = "action.retrieval.vector"
    config: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    examples: list[dict] = field(default_factory=list)
    retrieve_fn: Callable[[str, list[dict], int], list[dict]] | None = None
    generate_fn: Callable[[str, LLMGenerationConfig], tuple[str, float]] | None = None
    cost_estimate: float = 0.10
    n_retrieved: int = 3

    def _default_retrieve(self, query: str, examples: list[dict], k: int) -> list[dict]:
        """Default retrieval: random selection (placeholder).

        In production, this should use vector similarity search.
        For testing, deterministic random is fine.
        """
        import random
        rng = random.Random(hash(query) % 2**31)
        n = min(k, len(examples))
        return rng.sample(examples, n) if n > 0 else []

    def _build_retrieval_prompt(self, task: Mapping[str, Any]) -> str:
        """Build a prompt with retrieved examples as in-context demonstrations."""
        base_prompt = str(task.get("prompt", task.get("specification", "")))
        retrieve = self.retrieve_fn or self._default_retrieve
        retrieved = retrieve(base_prompt, self.examples, self.n_retrieved)

        parts = []
        if retrieved:
            parts.append("Here are some similar examples:\n")
            for ex in retrieved:
                ex_prompt = str(ex.get("prompt", ""))
                ex_answer = str(ex.get("answer", ""))
                parts.append(f"Problem: {ex_prompt}\nAnswer: {ex_answer}\n")
            parts.append("\nNow solve this problem:\n")

        parts.append(base_prompt)
        parts.append(_FINAL_ANSWER_SUFFIX)
        return "\n".join(parts)

    def execute(self, task: Mapping[str, Any]) -> ActionExecution:
        prompt = self._build_retrieval_prompt(task)
        t0 = time.time()

        if self.generate_fn is not None:
            text, latency_ms = self.generate_fn(prompt, self.config)
        else:
            text, latency_ms = _call_vllm_api(prompt, self.config)

        answer = _parse_final_answer(text)
        verified = _verify_answer(task, answer)

        failure_type = None
        if text.startswith("ERROR:"):
            failure_type = "execution_error"
        elif answer is None and not text.startswith("ERROR:"):
            failure_type = "parse_error"

        return ActionExecution(
            action_id=self.action_id,
            selected=False,
            executed=True,
            output=text,
            verified_correct=verified,
            verifier_name="numeric_exact",
            latency_ms=latency_ms,
            compute_cost=self.cost_estimate,
            failure_type=failure_type,
        )


# ──────────────────────────────────────────────────────────────────────
# Section 5 — Decomposed Reasoning Executor
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningDecomposeExecutor:
    """Execute decomposed reasoning: break the problem into sub-problems.

    Step 1: Ask the LLM to decompose the problem into sub-problems.
    Step 2: Solve each sub-problem.
    Step 3: Combine the sub-problem answers into a final answer.

    This is more expensive (multiple LLM calls) but can solve problems
    that direct reasoning cannot.

    Attributes
    ----------
    action_id : str
        Must be ``"action.reasoning.decompose"``.
    config : LLMGenerationConfig
    generate_fn : callable | None
    cost_estimate : float
    max_subproblems : int
        Maximum number of sub-problems to decompose into.
    """

    action_id: str = "action.reasoning.decompose"
    config: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    generate_fn: Callable[[str, LLMGenerationConfig], tuple[str, float]] | None = None
    cost_estimate: float = 0.30
    max_subproblems: int = 3

    _DECOMPOSE_SUFFIX = (
        "\n\nBreak this problem into at most {n} simpler sub-problems. "
        "List each sub-problem on a new line starting with 'SUB:'. "
        "Then solve each sub-problem and provide the final combined answer "
        "as: FINAL_ANSWER: <integer>"
    )

    def execute(self, task: Mapping[str, Any]) -> ActionExecution:
        base_prompt = str(task.get("prompt", task.get("specification", "")))
        decompose_prompt = base_prompt + self._DECOMPOSE_SUFFIX.format(
            n=self.max_subproblems)

        total_latency = 0.0
        gen = self.generate_fn or _call_vllm_api

        # Step 1: Decompose + solve in one call (simplification for v0.4)
        text, latency = gen(decompose_prompt, self.config)
        total_latency += latency

        answer = _parse_final_answer(text)
        verified = _verify_answer(task, answer)

        failure_type = None
        if text.startswith("ERROR:"):
            failure_type = "execution_error"
        elif answer is None and not text.startswith("ERROR:"):
            failure_type = "parse_error"

        return ActionExecution(
            action_id=self.action_id,
            selected=False,
            executed=True,
            output=text,
            verified_correct=verified,
            verifier_name="numeric_exact",
            latency_ms=total_latency,
            compute_cost=self.cost_estimate,
            failure_type=failure_type,
        )


# ──────────────────────────────────────────────────────────────────────
# Section 6 — Executor Registry
# ──────────────────────────────────────────────────────────────────────

class ExecutorRegistry:
    """Registry of action executors for an experiment.

    Maps action_ids to their executors.
    """

    def __init__(self) -> None:
        self._executors: dict[str, ActionExecutor] = {}

    def register(self, executor: ActionExecutor) -> None:
        """Register an executor."""
        self._executors[executor.action_id] = executor

    def get(self, action_id: str) -> ActionExecutor | None:
        return self._executors.get(action_id)

    def execute_all(
        self,
        task: Mapping[str, Any],
        action_space: ActionSpace,
    ) -> CounterfactualSet:
        """Execute all actions on a task (counterfactual execution).

        Returns a CounterfactualSet with all action executions.
        """
        state = ExecutiveState(
            task_id=str(task.get("task_id", "")),
            prompt=str(task.get("prompt", task.get("specification", ""))),
            task_metadata={
                k: v for k, v in task.items()
                if k not in ("prompt", "specification", "task_id")
            },
        )

        executions: dict[str, ActionExecution] = {}
        for action_id in action_space.action_ids:
            executor = self._executors.get(action_id)
            if executor is None:
                # Action not registered — mark as not executed
                executions[action_id] = ActionExecution(
                    action_id=action_id,
                    selected=False,
                    executed=False,
                    failure_type="no_executor",
                )
            else:
                executions[action_id] = executor.execute(task)

        return CounterfactualSet(
            state=state,
            executions=executions,
            selected_action=None,  # set by policy later
        )

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(self._executors.keys())


def build_b1_executors(
    config: LLMGenerationConfig,
    *,
    examples: list[dict] | None = None,
    generate_fn: Callable[[str, LLMGenerationConfig], tuple[str, float]] | None = None,
) -> ExecutorRegistry:
    """Build the executor registry for the B1 experiment.

    B1 = direct reasoning vs retrieval vs decomposition.
    """
    registry = ExecutorRegistry()
    registry.register(DirectReasoningExecutor(
        config=config, generate_fn=generate_fn))
    registry.register(RetrievalVectorExecutor(
        config=config, examples=examples or [], generate_fn=generate_fn))
    registry.register(ReasoningDecomposeExecutor(
        config=config, generate_fn=generate_fn))
    return registry


__all__ = [
    "ActionExecutor",
    "LLMGenerationConfig",
    "DirectReasoningExecutor",
    "RetrievalVectorExecutor",
    "ReasoningDecomposeExecutor",
    "ExecutorRegistry",
    "build_b1_executors",
]
