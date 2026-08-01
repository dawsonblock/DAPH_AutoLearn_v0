"""DAPH v0.4 — Generic Executive Qualification types.

This package introduces the generic executive abstraction that replaces
the hard-coded ``symbolic`` vs ``llm`` binary routing of v0.3.x.

Core concepts:

* **ActionDescriptor** — describes an executable action (symbolic_arithmetic,
  llm_direct, retrieval_vector, reasoning_decompose, etc.).
* **ExecutiveState** — the observable state at a decision point (prompt,
  task metadata, optional hidden-state reference).
* **ActionExecution** — the result of executing one action on one state,
  including output, verification, latency, and cost.
* **CounterfactualSet** — the collection of action executions for all
  candidate actions on a single state, enabling counterfactual learning.
* **UtilityModel** — computes ``U(state, action)`` for any action, replacing
  the hard-coded ``ΔU = U_symbolic - U_LLM``.
* **ActionDecision** — the policy's output: per-action scores and a
  selected action, replacing ``symbolic_probability``.

The old binary types (``RouteAction``, ``BACKENDS``, ``CounterfactualExperience``
with ``symbolic``/``llm`` fields) remain in their existing modules for
backward compatibility. The new generic types live here and can represent
the same binary experiment as a special case with two actions.
"""

from daph_learning.executive.types import (
    ActionDescriptor,
    ActionSpace,
    ExecutiveState,
    ActionExecution,
    CounterfactualSet,
    UtilityModel,
    UtilityBreakdown,
    ActionDecision,
    ActionProbability,
    Regret,
)
from daph_learning.executive.adapters import (
    binary_action_space,
    action_from_backend,
    backend_from_action,
    counterfactual_set_from_outcome,
    action_decision_from_symbolic_probability,
    symbolic_probability_from_action_decision,
    utility_model_from_legacy_config,
)
from daph_learning.executive.qualification import (
    ExecutiveTaskRecord,
    ExecutiveQualificationResult,
    compute_oracle_action,
    compute_oracle_utility,
    compute_always_action_utility,
    group_aware_bootstrap,
    permute_action_utilities_within_bins,
    evaluate_qualification,
)
from daph_learning.executive.report import (
    generate_markdown_report,
    generate_json_report,
    write_report,
)
from daph_learning.executive.policy import (
    ExecutivePolicyModel,
    ExecutiveCentroidPolicy,
    ExecutiveLogisticPolicy,
    make_executive_policy,
)
from daph_learning.executive.experience import (
    ExecutiveExperience,
    build_executive_experiences,
    experiences_to_training_arrays,
    combine_action_confidences,
    ExecutiveTrainingTargets,
    build_executive_training_targets,
    estimate_uncertainty,
)
from daph_learning.executive.executors import (
    ActionExecutor,
    LLMGenerationConfig,
    DirectReasoningExecutor,
    RetrievalVectorExecutor,
    ReasoningDecomposeExecutor,
    ExecutorRegistry,
    build_b1_executors,
)

__all__ = [
    "ActionDescriptor",
    "ActionSpace",
    "ExecutiveState",
    "ActionExecution",
    "CounterfactualSet",
    "UtilityModel",
    "UtilityBreakdown",
    "ActionDecision",
    "ActionProbability",
    "Regret",
    # Adapters
    "binary_action_space",
    "action_from_backend",
    "backend_from_action",
    "counterfactual_set_from_outcome",
    "action_decision_from_symbolic_probability",
    "symbolic_probability_from_action_decision",
    "utility_model_from_legacy_config",
    # Qualification
    "ExecutiveTaskRecord",
    "ExecutiveQualificationResult",
    "compute_oracle_action",
    "compute_oracle_utility",
    "compute_always_action_utility",
    "group_aware_bootstrap",
    "permute_action_utilities_within_bins",
    "evaluate_qualification",
    # Report
    "generate_markdown_report",
    "generate_json_report",
    "write_report",
    # Policy
    "ExecutivePolicyModel",
    "ExecutiveCentroidPolicy",
    "ExecutiveLogisticPolicy",
    "make_executive_policy",
    # Experience
    "ExecutiveExperience",
    "build_executive_experiences",
    "experiences_to_training_arrays",
    "combine_action_confidences",
    "ExecutiveTrainingTargets",
    "build_executive_training_targets",
    "estimate_uncertainty",
    # Executors
    "ActionExecutor",
    "LLMGenerationConfig",
    "DirectReasoningExecutor",
    "RetrievalVectorExecutor",
    "ReasoningDecomposeExecutor",
    "ExecutorRegistry",
    "build_b1_executors",
]
