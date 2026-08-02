"""DAPH v0.4.0a3 — Error semantics for experiment execution.

Every execution outcome must have a frozen utility rule. Errors must
not be silently converted into ordinary wrong answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionStatus(str, Enum):
    """Frozen execution outcome statuses."""

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNVERIFIABLE = "UNVERIFIABLE"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    TIMEOUT = "TIMEOUT"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    MODEL_ERROR = "MODEL_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"


# Frozen utility rules for each status
STATUS_UTILITY_RULES: dict[ExecutionStatus, dict] = {
    ExecutionStatus.CORRECT: {
        "quality": 1.0,
        "risk": 0.0,
        "description": "Answer verified correct",
    },
    ExecutionStatus.INCORRECT: {
        "quality": 0.0,
        "risk": 0.0,
        "description": "Answer verified incorrect",
    },
    ExecutionStatus.UNVERIFIABLE: {
        "quality": 0.0,
        "risk": 0.0,
        "description": "Could not verify answer — treated as no benefit",
    },
    ExecutionStatus.EXECUTION_ERROR: {
        "quality": 0.0,
        "risk": 1.0,
        "description": "Execution failed — penalty applied",
    },
    ExecutionStatus.TIMEOUT: {
        "quality": 0.0,
        "risk": 0.5,
        "description": "Execution timed out — partial penalty",
    },
    ExecutionStatus.INVALID_OUTPUT: {
        "quality": 0.0,
        "risk": 0.5,
        "description": "Output format invalid — partial penalty",
    },
    ExecutionStatus.MODEL_ERROR: {
        "quality": 0.0,
        "risk": 1.0,
        "description": "Model API error — full penalty",
    },
    ExecutionStatus.RETRIEVAL_ERROR: {
        "quality": 0.0,
        "risk": 0.5,
        "description": "Retrieval failed — partial penalty",
    },
}


@dataclass(frozen=True)
class ObservedCost:
    """Observed execution cost metrics.

    B5 utility must use real execution measurements, not hand-written
    cost estimates. Raw components are always stored.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    llm_call_count: int = 0
    retrieval_calls: int = 0
    wall_latency_ms: float = 0.0
    inference_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "llm_call_count": self.llm_call_count,
            "retrieval_calls": self.retrieval_calls,
            "wall_latency_ms": self.wall_latency_ms,
            "inference_duration_ms": self.inference_duration_ms,
        }


def compute_normalized_cost(
    cost: ObservedCost,
    *,
    w_prompt: float = 0.001,
    w_output: float = 0.001,
    w_compute: float = 0.01,
    w_latency: float = 0.001,
    time_reference_ms: float = 15000.0,
) -> float:
    """Compute normalized cost from observed metrics.

    C_i = w_p * P_norm + w_o * O_norm + w_c * C_norm + w_l * L_norm

    Weights are frozen before final evaluation.
    """
    p_norm = cost.prompt_tokens / 1000.0
    o_norm = (cost.completion_tokens + cost.reasoning_tokens) / 1000.0
    c_norm = cost.llm_call_count + cost.retrieval_calls
    l_norm = cost.wall_latency_ms / time_reference_ms
    return w_prompt * p_norm + w_output * o_norm + w_compute * c_norm + w_latency * l_norm


def compute_observed_utility(
    status: ExecutionStatus,
    cost: ObservedCost,
    *,
    quality_weight: float = 1.0,
    w_prompt: float = 0.001,
    w_output: float = 0.001,
    w_compute: float = 0.01,
    w_latency: float = 0.001,
    time_reference_ms: float = 15000.0,
    failure_penalty: float = 0.5,
) -> float:
    """Compute utility from observed status and cost.

    U_i = w_A * A_i - C_i - w_F * F_i

    Where:
    - A_i = quality (1.0 if correct, 0.0 otherwise)
    - C_i = normalized cost
    - F_i = failure penalty (from status)
    """
    rule = STATUS_UTILITY_RULES.get(status, STATUS_UTILITY_RULES[ExecutionStatus.EXECUTION_ERROR])
    quality = rule["quality"]
    risk = rule["risk"]
    cost_val = compute_normalized_cost(
        cost, w_prompt=w_prompt, w_output=w_output,
        w_compute=w_compute, w_latency=w_latency,
        time_reference_ms=time_reference_ms,
    )
    return quality_weight * quality - cost_val - failure_penalty * risk


__all__ = [
    "ExecutionStatus",
    "STATUS_UTILITY_RULES",
    "ObservedCost",
    "compute_normalized_cost",
    "compute_observed_utility",
]
