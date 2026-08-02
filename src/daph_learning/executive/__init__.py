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
    RetrievalLexicalExecutor,
    RetrievalVectorExecutor,
    ReasoningDecomposeExecutor,
    ExecutorRegistry,
    build_b1_executors,
)
from daph_learning.executive.task_generator import (
    generate_diverse_tasks,
    build_retrieval_store,
)
from daph_learning.executive.hidden_state import (
    HiddenStateConfig,
    QWEN3_8B_REVISION,
    load_model_for_capture,
    capture_hidden_states,
    capture_logprob_features,
)
from daph_learning.executive.b4_dataset import (
    B4DatasetSplit,
    generate_b4_dataset,
    build_b4_retrieval_store,
)
from daph_learning.executive.dim_reduction import (
    PCAPipeline,
    select_pca_dimension,
)
from daph_learning.executive.q_policy import (
    QRegressionPolicy,
    compute_regret,
    mean_regret,
)
# B4 hardening + B5 adaptive compute
from daph_learning.executive.artifact_integrity import (
    ArtifactCheck,
    IntegrityReport,
    sha256_file,
    sha256_json,
    detect_corruption,
    validate_json_artifact,
    validate_required_tree,
    validate_manifest_hashes,
    B4_REQUIRED_ARTIFACTS,
    B5_REQUIRED_ARTIFACTS,
)
from daph_learning.executive.manifest import (
    ManifestBuilder,
    compute_config_hash,
    write_config_hash,
    capture_environment,
    load_manifest,
)
from daph_learning.executive.leakage import (
    LeakageCheck,
    LeakageReport,
    check_task_id_overlap,
    check_exact_prompt_leakage,
    check_retrieval_store_leakage,
    check_pca_train_only,
    check_policy_leakage,
    check_group_leakage,
    check_representation_sanity,
    run_leakage_checks_from_artifacts,
)
from daph_learning.executive.stats import (
    GroupResult,
    compute_group_local_results,
    positive_group_fraction,
    worst_group_delta,
    BootstrapResult,
    paired_group_bootstrap,
    ShamComparisonResult,
    create_matched_sham_utilities,
    run_matched_sham_evaluation,
    gap_capture,
    selection_accuracy,
    margin_analysis,
)
from daph_learning.executive.lifecycle import (
    ExperimentStatus,
    FrozenConfig,
    ExperimentState,
    RegistryEntry,
    load_registry,
    save_registry,
    register_experiment,
    invalidate_experiment,
)
from daph_learning.executive.b5_actions import (
    B5_ACTION_DIRECT_FAST,
    B5_ACTION_DIRECT_THINK,
    B5_ACTION_RETRIEVE,
    B5_ACTION_DECOMPOSE,
    B5_ACTION_IDS,
    b5_action_space,
    InferencePreset,
    B5_DEFAULT_PRESETS,
    DirectFastExecutor,
    DirectThinkExecutor,
    B5ExecutorRegistry,
    build_b5_executors,
)
from daph_learning.executive.b5_dataset import (
    B5_FAMILIES,
    B5DatasetSplit,
    generate_b5_dataset,
    build_b5_retrieval_store,
    compute_winner_distribution,
)
from daph_learning.executive.b5_policies import (
    LinearQPolicy,
    RidgeQPolicy,
    MLPQPolicy,
    compute_surface_features,
)
from daph_learning.executive.b5_qualification import (
    GateResult,
    QualificationResult,
    GateThresholds,
    DEFAULT_THRESHOLDS,
    evaluate_gates,
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
    "RetrievalLexicalExecutor",
    "RetrievalVectorExecutor",
    "ReasoningDecomposeExecutor",
    "ExecutorRegistry",
    "build_b1_executors",
    # Task generation
    "generate_diverse_tasks",
    "build_retrieval_store",
    # Hidden state capture
    "HiddenStateConfig",
    "QWEN3_8B_REVISION",
    "load_model_for_capture",
    "capture_hidden_states",
    "capture_logprob_features",
    # B4 dataset
    "B4DatasetSplit",
    "generate_b4_dataset",
    "build_b4_retrieval_store",
    # Dimensionality reduction
    "PCAPipeline",
    "select_pca_dimension",
    # Q-regression policy
    "QRegressionPolicy",
    "compute_regret",
    "mean_regret",
    # Artifact integrity
    "ArtifactCheck",
    "IntegrityReport",
    "sha256_file",
    "sha256_json",
    "detect_corruption",
    "validate_json_artifact",
    "validate_required_tree",
    "validate_manifest_hashes",
    "B4_REQUIRED_ARTIFACTS",
    "B5_REQUIRED_ARTIFACTS",
    # Manifest
    "ManifestBuilder",
    "compute_config_hash",
    "write_config_hash",
    "capture_environment",
    "load_manifest",
    # Leakage checks
    "LeakageCheck",
    "LeakageReport",
    "check_task_id_overlap",
    "check_exact_prompt_leakage",
    "check_retrieval_store_leakage",
    "check_pca_train_only",
    "check_policy_leakage",
    "check_group_leakage",
    "check_representation_sanity",
    "run_leakage_checks_from_artifacts",
    # Corrected stats
    "GroupResult",
    "compute_group_local_results",
    "positive_group_fraction",
    "worst_group_delta",
    "BootstrapResult",
    "paired_group_bootstrap",
    "ShamComparisonResult",
    "create_matched_sham_utilities",
    "run_matched_sham_evaluation",
    "gap_capture",
    "selection_accuracy",
    "margin_analysis",
    # Lifecycle
    "ExperimentStatus",
    "FrozenConfig",
    "ExperimentState",
    "RegistryEntry",
    "load_registry",
    "save_registry",
    "register_experiment",
    "invalidate_experiment",
    # B5 actions
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
    # B5 dataset
    "B5_FAMILIES",
    "B5DatasetSplit",
    "generate_b5_dataset",
    "build_b5_retrieval_store",
    "compute_winner_distribution",
    # B5 policies
    "LinearQPolicy",
    "RidgeQPolicy",
    "MLPQPolicy",
    "compute_surface_features",
    # B5 qualification
    "GateResult",
    "QualificationResult",
    "GateThresholds",
    "DEFAULT_THRESHOLDS",
    "evaluate_gates",
]
