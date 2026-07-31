__version__ = "0.3.10.5-alpha"

# Section 2: experiment identity. The current source tree targets a NEW,
# frozen, auditable Gate A experiment. The earlier real-model run is
# preserved as legacy evidence (see artifacts/legacy/).
CURRENT_EXPERIMENT_ID = "daph_gate_a_real_002"
LEGACY_FAILED_EXPERIMENT_ID = "daph_gate_a_real_001_failed"

# Section 7/12: canonical execution + verification exports.
from .execution.real_backends import (
    BackendExecution,
    BackendName,
    ExecutionStatus,
    MockLLMBackend,
    build_canonical_counterfactual_experience,
    execute_llm_canonical,
    execute_symbolic_canonical,
    execution_to_outcome,
    verify_backend_execution,
)

# Section 7: canonical verifier exports.
from .evaluation.canonical_verifier import (
    CanonicalIntegerVerifier,
    ParsedFinalAnswer,
    VerificationStatus,
    parse_canonical_integer_answer,
)

# Section 8: utility config exports.
from .policy.utility import (
    GATE_A_ACCURACY_PRIMARY,
    GATE_A_COST_SENSITIVE_SECONDARY,
    UtilityConfig,
    compute_utility,
    get_protocol,
)

# Section 14: uncertainty-aware targets export.
from .policy.targets import TrainingTargets

# Section 15: sham control exports.
from .evaluation.sham import PolicyTrainingSpec

# Section 10: audit exports.
from .benchmark.audit import EmpiricalCrossoverAudit, audit_empirical_crossover

__all__ = [
    "BackendExecution",
    "BackendName",
    "CanonicalIntegerVerifier",
    "CURRENT_EXPERIMENT_ID",
    "EmpiricalCrossoverAudit",
    "ExecutionStatus",
    "GATE_A_ACCURACY_PRIMARY",
    "GATE_A_COST_SENSITIVE_SECONDARY",
    "LEGACY_FAILED_EXPERIMENT_ID",
    "MockLLMBackend",
    "ParsedFinalAnswer",
    "PolicyTrainingSpec",
    "TrainingTargets",
    "UtilityConfig",
    "VerificationStatus",
    "audit_empirical_crossover",
    "build_canonical_counterfactual_experience",
    "compute_utility",
    "execute_llm_canonical",
    "execute_symbolic_canonical",
    "execution_to_outcome",
    "get_protocol",
    "parse_canonical_integer_answer",
    "verify_backend_execution",
]
