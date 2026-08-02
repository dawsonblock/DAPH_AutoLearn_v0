"""DAPH v0.4 — B5 Adaptive Compute action space and executors.

B5 tests whether a frozen model's internal state can predict whether
additional reasoning compute will improve verified utility enough to
justify its cost.

Action space (exactly four actions)::

    action.reasoning.direct_fast   — one model call, /no_think, low token budget
    action.reasoning.direct_think  — same model, thinking mode on, higher budget
    action.retrieval.examples      — lexical/embedding retrieval + generation
    action.reasoning.decompose     — decompose → solve subproblems → combine → verify

The difference between FAST and THINK represents additional inference
compute, not a different model.
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from daph_learning.executive.types import (
    ActionDescriptor,
    ActionSpace,
    ActionExecution,
)
from daph_learning.executive.executors import (
    LLMGenerationConfig,
    _build_prompt,
    _parse_final_answer,
    _verify_answer,
    _call_vllm_api,
    _FINAL_ANSWER_SUFFIX,
    _NO_THINK_PREFIX,
    RetrievalLexicalExecutor,
    ReasoningDecomposeExecutor,
)


# ──────────────────────────────────────────────────────────────────────
# Section 1 — B5 Action Space
# ──────────────────────────────────────────────────────────────────────

# Canonical B5 action IDs
B5_ACTION_DIRECT_FAST = "action.reasoning.direct_fast"
B5_ACTION_DIRECT_THINK = "action.reasoning.direct_think"
B5_ACTION_RETRIEVE = "action.retrieval.examples"
B5_ACTION_DECOMPOSE = "action.reasoning.decompose"

B5_ACTION_IDS = (
    B5_ACTION_DIRECT_FAST,
    B5_ACTION_DIRECT_THINK,
    B5_ACTION_RETRIEVE,
    B5_ACTION_DECOMPOSE,
)


def b5_action_space() -> ActionSpace:
    """Build the canonical B5 four-action adaptive compute space."""
    return ActionSpace(actions=(
        ActionDescriptor(
            action_id=B5_ACTION_DIRECT_FAST,
            display_name="Direct Fast",
            description=(
                "One model call with /no_think mode and low token budget. "
                "Minimal reasoning compute — fast and cheap."
            ),
            cost_estimate=0.10,
            tags=("reasoning", "direct", "fast", "no_think"),
        ),
        ActionDescriptor(
            action_id=B5_ACTION_DIRECT_THINK,
            display_name="Direct Think",
            description=(
                "Same frozen model, thinking mode enabled, higher token budget. "
                "Additional inference compute — slower but more capable."
            ),
            cost_estimate=0.35,
            tags=("reasoning", "direct", "think", "high_compute"),
        ),
        ActionDescriptor(
            action_id=B5_ACTION_RETRIEVE,
            display_name="Retrieval Examples",
            description=(
                "Retrieve similar training examples (lexical or embedding), "
                "append as in-context demonstrations, then generate."
            ),
            cost_estimate=0.15,
            tags=("retrieval", "examples", "augmented"),
        ),
        ActionDescriptor(
            action_id=B5_ACTION_DECOMPOSE,
            display_name="Decompose",
            description=(
                "Decompose into sub-problems, solve each (optionally parallel), "
                "combine, then verify. Highest compute cost."
            ),
            cost_estimate=0.50,
            tags=("reasoning", "decompose", "multi-step", "high_compute"),
        ),
    ))


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Inference parameter presets (frozen before final)
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InferencePreset:
    """Frozen inference parameters for a B5 action.

    The difference between FAST and THINK must represent additional
    inference compute, not a different model.
    """

    reasoning_mode: str  # "off" or "on"
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    no_think_prefix: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_mode": self.reasoning_mode,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "no_think_prefix": self.no_think_prefix,
        }


# Default frozen presets (must be frozen before final evaluation)
B5_DEFAULT_PRESETS: dict[str, InferencePreset] = {
    B5_ACTION_DIRECT_FAST: InferencePreset(
        reasoning_mode="off",
        max_tokens=512,
        temperature=0.0,
        no_think_prefix=True,
    ),
    B5_ACTION_DIRECT_THINK: InferencePreset(
        reasoning_mode="on",
        max_tokens=2048,
        temperature=0.0,
        no_think_prefix=False,
    ),
    B5_ACTION_RETRIEVE: InferencePreset(
        reasoning_mode="off",
        max_tokens=512,
        temperature=0.0,
        no_think_prefix=True,
    ),
    B5_ACTION_DECOMPOSE: InferencePreset(
        reasoning_mode="off",
        max_tokens=1024,
        temperature=0.0,
        no_think_prefix=True,
    ),
}


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Direct Fast Executor
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DirectFastExecutor:
    """Execute direct fast reasoning: /no_think, low token budget.

    Characteristics:
    * one model call
    * minimal reasoning mode (/no_think)
    * low token budget (default 512)
    * low latency
    * no retrieval, no decomposition
    """

    action_id: str = B5_ACTION_DIRECT_FAST
    config: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    preset: InferencePreset = field(default_factory=lambda: B5_DEFAULT_PRESETS[B5_ACTION_DIRECT_FAST])
    generate_fn: Callable[[str, LLMGenerationConfig], tuple[str, float]] | None = None
    cost_estimate: float = 0.10

    def execute(self, task: Mapping[str, Any]) -> ActionExecution:
        # Apply preset to config
        cfg = LLMGenerationConfig(
            model_id=self.config.model_id,
            max_tokens=self.preset.max_tokens,
            temperature=self.preset.temperature,
            top_p=self.preset.top_p,
            vllm_port=self.config.vllm_port,
            vllm_api_key=self.config.vllm_api_key,
            vllm_api_key_env=self.config.vllm_api_key_env,
            vllm_base_url=self.config.vllm_base_url,
            vllm_max_concurrent=self.config.vllm_max_concurrent,
        )
        prompt = _build_prompt(task, no_think=self.preset.no_think_prefix)
        t0 = time.time()

        if self.generate_fn is not None:
            text, latency_ms = self.generate_fn(prompt, cfg)
            prompt_tokens, completion_tokens = 0, 0
        else:
            text, latency_ms, prompt_tokens, completion_tokens = _call_vllm_api(prompt, cfg)

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
        )


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Direct Think Executor
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DirectThinkExecutor:
    """Execute direct think reasoning: thinking mode on, higher token budget.

    Characteristics:
    * same frozen model as FAST
    * same task
    * no retrieval, no decomposition
    * higher reasoning budget (thinking mode enabled)
    * higher token budget (default 2048)

    The difference from FAST represents additional inference compute,
    not a different model.
    """

    action_id: str = B5_ACTION_DIRECT_THINK
    config: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    preset: InferencePreset = field(default_factory=lambda: B5_DEFAULT_PRESETS[B5_ACTION_DIRECT_THINK])
    generate_fn: Callable[[str, LLMGenerationConfig], tuple[str, float]] | None = None
    cost_estimate: float = 0.35

    def execute(self, task: Mapping[str, Any]) -> ActionExecution:
        cfg = LLMGenerationConfig(
            model_id=self.config.model_id,
            max_tokens=self.preset.max_tokens,
            temperature=self.preset.temperature,
            top_p=self.preset.top_p,
            vllm_port=self.config.vllm_port,
            vllm_api_key=self.config.vllm_api_key,
            vllm_api_key_env=self.config.vllm_api_key_env,
            vllm_base_url=self.config.vllm_base_url,
            vllm_max_concurrent=self.config.vllm_max_concurrent,
        )
        # Thinking mode: do NOT prepend /no_think
        prompt = _build_prompt(task, no_think=self.preset.no_think_prefix)
        t0 = time.time()

        if self.generate_fn is not None:
            text, latency_ms = self.generate_fn(prompt, cfg)
            prompt_tokens, completion_tokens = 0, 0
        else:
            text, latency_ms, prompt_tokens, completion_tokens = _call_vllm_api(prompt, cfg)

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
        )


# ──────────────────────────────────────────────────────────────────────
# Section 5 — B5 Executor Registry Builder
# ──────────────────────────────────────────────────────────────────────

def build_b5_executors(
    config: LLMGenerationConfig,
    *,
    retrieval_examples: list[dict] | None = None,
    generate_fn: Callable[[str, LLMGenerationConfig], tuple[str, float]] | None = None,
    presets: dict[str, InferencePreset] | None = None,
) -> "B5ExecutorRegistry":
    """Build the B5 four-action executor registry.

    Parameters
    ----------
    config : LLMGenerationConfig
        Base LLM config (model_id, vllm settings).
    retrieval_examples : list[dict] | None
        Training examples for the retrieval action.
    generate_fn : callable | None
        Optional mock generation function.
    presets : dict | None
        Optional override of inference presets.
    """
    presets = presets or B5_DEFAULT_PRESETS
    registry = B5ExecutorRegistry()

    fast = DirectFastExecutor(
        config=config, preset=presets[B5_ACTION_DIRECT_FAST],
        generate_fn=generate_fn,
    )
    think = DirectThinkExecutor(
        config=config, preset=presets[B5_ACTION_DIRECT_THINK],
        generate_fn=generate_fn,
    )
    retrieve = RetrievalLexicalExecutor(
        action_id=B5_ACTION_RETRIEVE,
        config=config, examples=retrieval_examples or [],
        generate_fn=generate_fn, n_retrieved=3,
        cost_estimate=0.15,
    )
    decompose = ReasoningDecomposeExecutor(
        action_id=B5_ACTION_DECOMPOSE,
        config=config, generate_fn=generate_fn,
        cost_estimate=0.50,
    )

    registry.register(fast)
    registry.register(think)
    registry.register(retrieve)
    registry.register(decompose)
    return registry


# Re-use the existing ExecutorRegistry but with a B5-specific alias
from daph_learning.executive.executors import ExecutorRegistry as _BaseRegistry


class B5ExecutorRegistry(_BaseRegistry):
    """Executor registry specialized for B5's four-action space."""

    pass


__all__ = [
    "B5_ACTION_DIRECT_FAST",
    "B5_ACTION_DIRECT_THINK",
    "B5_ACTION_RETRIEVE",
    "B5_ACTION_DECOMPOSE",
    "B5_ACTION_IDS",
    "b5_action_space",
    "InferencePreset",
    "B5_DEFAULT_PRESETS",
    "DirectFastExecutor",
    "DirectThinkExecutor",
    "B5ExecutorRegistry",
    "build_b5_executors",
]
