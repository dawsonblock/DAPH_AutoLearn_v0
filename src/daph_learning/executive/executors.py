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

# Qwen3 supports /no_think to disable reasoning blocks, making generation
# much faster and shorter. Used for direct reasoning and decompose sub-problems.
_NO_THINK_PREFIX = "/no_think\n"


def _build_prompt(task: Mapping[str, Any], *, suffix: str = _FINAL_ANSWER_SUFFIX,
                  no_think: bool = False) -> str:
    """Build the base prompt from a task.

    If ``no_think`` is True, prepend ``/no_think`` to disable Qwen3's
    reasoning mode. This produces shorter, faster responses without
    ``<think>`` blocks — ideal for simple problems where reasoning
    overhead is unnecessary.
    """
    prompt = str(task.get("prompt", task.get("specification", "")))
    full = prompt + suffix
    if no_think:
        full = _NO_THINK_PREFIX + full
    return full


def _parse_final_answer(text: str | None) -> int | None:
    """Extract the integer from an LLM response.

    Tries multiple strategies in order:
    1. Canonical ``FINAL_ANSWER: <integer>`` marker (preferred).
    2. Strip ``...`` reasoning blocks and retry FINAL_ANSWER.
    3. Fallback: extract the last standalone integer from the
       non-thinking portion of the response.

    Returns the integer value, or ``None`` if no answer can be extracted.
    """
    if text is None:
        return None
    raw = str(text)

    # Strategy 1: canonical FINAL_ANSWER marker
    from daph_learning.evaluation.canonical_verifier import parse_canonical_integer_answer
    parsed = parse_canonical_integer_answer(raw)
    if parsed.status == "VALID" and parsed.value is not None:
        return parsed.value

    # Strategy 2: strip <think>...</think> blocks and retry
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    if cleaned != raw:
        parsed = parse_canonical_integer_answer(cleaned)
        if parsed.status == "VALID" and parsed.value is not None:
            return parsed.value

    # Strategy 3: fallback — look for common answer patterns
    # in the non-thinking portion
    search_text = cleaned if cleaned != raw else raw

    # Pattern: "The answer is X" / "answer is X" / "= X"
    for pattern in [
        r"(?:the\s+)?answer\s+is\s*:?\s*(-?\d+)",
        r"=\s*(-?\d+)\s*(?:\.|$)",
        r"(?:result|total|sum|product)\s+is\s*:?\s*(-?\d+)",
    ]:
        m = re.search(pattern, search_text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

    # Strategy 4: last standalone integer on its own line
    lines = search_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        # Skip empty lines and lines that are just punctuation
        if not line or line in ("```", "---", "***"):
            continue
        # Try to find a standalone integer
        m = re.search(r"(?<![\d.]) (-?\d{1,10}) (?![\d.])", " " + line + " ")
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        # Or a number at the end of the line
        m = re.search(r"(-?\d+)\s*\.?\s*$", line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

    return None


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
) -> tuple[str, float, int, int]:
    """Call a vLLM OpenAI-compatible API server for a single prompt.

    Returns (generated_text, latency_ms, prompt_tokens, completion_tokens).
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
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            latency_ms = (time.time() - t0) * 1000.0
            usage = data.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            return text, latency_ms, prompt_tokens, completion_tokens
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000.0
        return f"ERROR: {e}", latency_ms, 0, 0


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Direct Reasoning Executor
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DirectReasoningExecutor:
    """Execute direct LLM reasoning (no decomposition, no retrieval).

    This is the simplest action: send the task prompt to the LLM and
    parse the FINAL_ANSWER from the response.

    Uses ``/no_think`` mode for Qwen3 to disable reasoning blocks,
    making it fast but less capable on complex problems.

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
        # B4: Use /no_think for speed. The scientific question is about
        # representation quality, not latency equalization. /no_think
        # creates cleaner conditional structure: direct fails on hard
        # problems, retrieval/decompose help more.
        prompt = _build_prompt(task, no_think=True)
        t0 = time.time()

        if self.generate_fn is not None:
            text, latency_ms = self.generate_fn(prompt, self.config)
            prompt_tokens, completion_tokens = 0, 0
        else:
            text, latency_ms, prompt_tokens, completion_tokens = _call_vllm_api(prompt, self.config)

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
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            aggregate_inference_latency_ms=latency_ms,
        )


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Retrieval-Augmented Executor
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalLexicalExecutor:
    """Execute retrieval-augmented LLM reasoning.

    Retrieves similar examples from a provided example store, appends
    them to the prompt as in-context examples, then generates.

    Attributes
    ----------
    action_id : str
        Must be ``"action.retrieval.lexical"``.
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

    action_id: str = "action.retrieval.lexical"
    config: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    examples: list[dict] = field(default_factory=list)
    retrieve_fn: Callable[[str, list[dict], int], list[dict]] | None = None
    generate_fn: Callable[[str, LLMGenerationConfig], tuple[str, float]] | None = None
    cost_estimate: float = 0.10
    n_retrieved: int = 3

    def _default_retrieve(self, query: str, examples: list[dict], k: int) -> list[dict]:
        """Default retrieval: lexical keyword overlap.

        Uses simple keyword matching to find examples that share words
        with the query. This is lexical retrieval (keyword overlap),
        NOT vector/embedding retrieval. The action is named
        ``action.retrieval.lexical`` to reflect this accurately.

        For real vector retrieval, use a pinned embedding model with
        cosine similarity (see ``RetrievalEmbeddingExecutor``).
        """
        if not examples:
            return []

        # Tokenize the query
        query_words = set(query.lower().split())
        # Remove common words
        stopwords = {"what", "is", "the", "calculate", "consider", "note:",
                     "not", "their", "sum", "product", "answer", "with", "for",
                     "of", "in", "to", "a", "an", "this", "that", "these",
                     "those", "first", "then", "later", "more", "and", "or",
                     "by", "at", "on", "from", "they", "she", "he", "it"}
        query_words -= stopwords

        # Score each example by keyword overlap
        scored = []
        for ex in examples:
            ex_words = set(str(ex.get("prompt", "")).lower().split())
            ex_words -= stopwords
            overlap = len(query_words & ex_words)
            scored.append((overlap, ex))

        # Sort by score (descending), take top k
        scored.sort(key=lambda x: -x[0])
        return [ex for _, ex in scored[:k]]

    def _build_retrieval_prompt(self, task: Mapping[str, Any]) -> str:
        """Build a prompt with retrieved examples as in-context demonstrations.

        B4: Uses /no_think for speed. The in-context examples provide
        the pattern; the model just needs to apply it, not reason
        extensively about it.
        """
        base_prompt = str(task.get("prompt", task.get("specification", "")))
        retrieve = self.retrieve_fn or self._default_retrieve
        retrieved = retrieve(base_prompt, self.examples, self.n_retrieved)

        parts = ["/no_think"]
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
            prompt_tokens, completion_tokens = 0, 0
        else:
            text, latency_ms, prompt_tokens, completion_tokens = _call_vllm_api(prompt, self.config)

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
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            retrieval_call_count=1,
            aggregate_inference_latency_ms=latency_ms,
        )


# ──────────────────────────────────────────────────────────────────────
# Section 5 — Decomposed Reasoning Executor
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningDecomposeExecutor:
    """Execute decomposed reasoning: break the problem into sub-problems.

    Step 1: Ask the LLM to decompose the problem into sub-problems.
    Step 2: Solve each sub-problem in a separate LLM call.
    Step 3: Combine the sub-problem answers in a final LLM call.

    This is more expensive (multiple LLM calls) but can solve problems
    that direct reasoning cannot, particularly multi-step problems
    where each step is simple but the composition is complex.

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

    _DECOMPOSE_PROMPT = (
        "/no_think\n"
        "{problem}\n\n"
        "Break this problem into at most {n} simpler sub-problems. "
        "Output each sub-problem on its own line in this exact format:\n"
        "SUB: <sub-problem description>\n"
        "Do not solve the sub-problems yet. Just list them."
    )

    _SOLVE_PROMPT = (
        "/no_think\n"
        "Solve this sub-problem and output ONLY the numerical answer:\n"
        "{sub_problem}\n\n"
        "Provide your answer as: FINAL_ANSWER: <integer>"
    )

    _COMBINE_PROMPT = (
        "/no_think\n"
        "Here are the answers to sub-problems that solve a larger problem:\n"
        "{sub_answers}\n\n"
        "The original problem was:\n{original_problem}\n\n"
        "Combine these sub-answers to get the final answer. "
        "Provide your answer as: FINAL_ANSWER: <integer>"
    )

    def _parse_sub_problems(self, text: str) -> list[str]:
        """Parse SUB: lines from the decomposition output."""
        subs = []
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("SUB:"):
                subs.append(line[4:].strip())
            elif line.upper().startswith("SUB :"):
                subs.append(line[5:].strip())
        return subs[:self.max_subproblems]

    def execute(self, task: Mapping[str, Any]) -> ActionExecution:
        base_prompt = str(task.get("prompt", task.get("specification", "")))
        gen = self.generate_fn or _call_vllm_api
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        all_outputs = []

        def _call_gen(prompt_str):
            """Call gen and normalize return to (text, latency, ptoks, ctoks)."""
            result = gen(prompt_str, self.config)
            if len(result) == 4:
                return result
            # generate_fn returns (text, latency)
            return result[0], result[1], 0, 0

        # Step 1: Decompose the problem
        decompose_prompt = self._DECOMPOSE_PROMPT.format(
            problem=base_prompt, n=self.max_subproblems)
        text1, latency1, pt1, ct1 = _call_gen(decompose_prompt)
        total_latency += latency1
        total_prompt_tokens += pt1
        total_completion_tokens += ct1
        all_outputs.append(f"[DECOMPOSE]\n{text1[:200]}")

        sub_problems = self._parse_sub_problems(text1)

        if not sub_problems:
            # Fallback: if decomposition failed, try direct solve
            solve_prompt = base_prompt + _FINAL_ANSWER_SUFFIX
            text, latency, pt, ct = _call_gen(solve_prompt)
            total_latency += latency
            total_prompt_tokens += pt
            total_completion_tokens += ct
            all_outputs.append(f"[FALLBACK DIRECT]\n{text[:200]}")
            answer = _parse_final_answer(text)
        else:
            # Step 2: Solve each sub-problem CONCURRENTLY
            from concurrent.futures import ThreadPoolExecutor

            def _solve_sub(i_sub):
                i, sub = i_sub
                solve_prompt = self._SOLVE_PROMPT.format(sub_problem=sub)
                text_s, latency_s, pt_s, ct_s = _call_gen(solve_prompt)
                return i, sub, text_s, latency_s, pt_s, ct_s, _parse_final_answer(text_s)

            sub_answers = [None] * len(sub_problems)
            with ThreadPoolExecutor(max_workers=len(sub_problems)) as ex:
                futures = [ex.submit(_solve_sub, (i, sub))
                           for i, sub in enumerate(sub_problems)]
                for fut in futures:
                    i, sub, text_s, latency_s, pt_s, ct_s, sub_answer = fut.result()
                    total_latency += latency_s
                    total_prompt_tokens += pt_s
                    total_completion_tokens += ct_s
                    all_outputs.append(f"[SUB {i+1}: {sub[:50]}]\n{text_s[:100]}")
                    if sub_answer is not None:
                        sub_answers[i] = f"Sub-problem {i+1}: {sub} → Answer: {sub_answer}"
                    else:
                        sub_answers[i] = f"Sub-problem {i+1}: {sub} → Answer: unknown"

            # Step 3: Combine
            combine_prompt = self._COMBINE_PROMPT.format(
                sub_answers="\n".join(sub_answers),
                original_problem=base_prompt,
            )
            text_c, latency_c, pt_c, ct_c = _call_gen(combine_prompt)
            total_latency += latency_c
            total_prompt_tokens += pt_c
            total_completion_tokens += ct_c
            all_outputs.append(f"[COMBINE]\n{text_c[:200]}")
            answer = _parse_final_answer(text_c)

        verified = _verify_answer(task, answer)

        failure_type = None
        if any("ERROR:" in o for o in all_outputs):
            failure_type = "execution_error"
        elif answer is None:
            failure_type = "parse_error"

        return ActionExecution(
            action_id=self.action_id,
            selected=False,
            executed=True,
            output="\n\n".join(all_outputs),
            verified_correct=verified,
            verifier_name="numeric_exact",
            latency_ms=total_latency,
            compute_cost=self.cost_estimate,
            failure_type=failure_type,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            llm_call_count=1 + len(sub_problems) + (1 if sub_problems else 0),
            aggregate_inference_latency_ms=total_latency,
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

        Actions are executed CONCURRENTLY since they are independent
        and all hit the same vLLM server (I/O bound, not CPU bound).

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

        from concurrent.futures import ThreadPoolExecutor

        executions: dict[str, ActionExecution] = {}

        def _run_action(action_id):
            executor = self._executors.get(action_id)
            if executor is None:
                return action_id, ActionExecution(
                    action_id=action_id,
                    selected=False,
                    executed=False,
                    failure_type="no_executor",
                )
            return action_id, executor.execute(task)

        # Run all actions concurrently — they're independent I/O calls
        with ThreadPoolExecutor(max_workers=len(action_space.action_ids)) as ex:
            futures = [ex.submit(_run_action, aid) for aid in action_space.action_ids]
            for fut in futures:
                aid, execution = fut.result()
                executions[aid] = execution

        return CounterfactualSet(
            state=state,
            executions=executions,
            selected_action=None,  # set by policy later
        )

    def execute_all_tasks(
        self,
        tasks: list[Mapping[str, Any]],
        action_space: ActionSpace,
        *,
        max_concurrent: int = 64,
        progress_every: int = 50,
    ) -> list[CounterfactualSet]:
        """Execute all actions on all tasks concurrently.

        Uses ThreadPoolExecutor for concurrent API calls.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time

        results: list[CounterfactualSet | None] = [None] * len(tasks)
        t0 = _time.time()

        def _execute_one(idx_task):
            idx, task = idx_task
            return idx, self.execute_all(task, action_space)

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(_execute_one, (i, t)): i
                for i, t in enumerate(tasks)
            }
            completed = 0
            for future in as_completed(futures):
                idx, cf_set = future.result()
                results[idx] = cf_set
                completed += 1
                if progress_every > 0 and completed % progress_every == 0:
                    elapsed = _time.time() - t0
                    rate = completed / max(elapsed, 0.1)
                    print(f"  [{completed}/{len(tasks)}] "
                          f"{elapsed:.1f}s ({rate:.1f} tasks/s)")

        elapsed = _time.time() - t0
        print(f"  Completed {len(tasks)} tasks in {elapsed:.1f}s "
              f"({len(tasks)/max(elapsed,0.1):.1f} tasks/s)")
        return results  # type: ignore[return-value]

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
    registry.register(RetrievalLexicalExecutor(
        config=config, examples=examples or [], generate_fn=generate_fn))
    registry.register(ReasoningDecomposeExecutor(
        config=config, generate_fn=generate_fn))
    return registry


# Backward-compatible alias for code that references the old name.
RetrievalVectorExecutor = RetrievalLexicalExecutor


__all__ = [
    "ActionExecutor",
    "LLMGenerationConfig",
    "DirectReasoningExecutor",
    "RetrievalLexicalExecutor",
    "RetrievalVectorExecutor",  # backward-compat alias
    "ReasoningDecomposeExecutor",
    "ExecutorRegistry",
    "build_b1_executors",
]
