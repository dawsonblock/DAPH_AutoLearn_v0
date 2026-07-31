"""v0.3.9 (V039-001/002/004) — separated repair-outcome semantics.

The historical :func:`daph_learning.autolearn.loop.classify_outcome` collapses
four distinct questions into a single ``"correct" | "misrouted" | "unverifiable"``
label:

1. **backend_selected** — which backend did the router choose?
2. **backend_executed** — which backend(s) were actually run?
3. **verification_status** — did the run output pass an independent verifier?
4. **route_suitability** — does the selected route match the policy oracle?

Collapsing these lets a skipped backend be credited as ``"correct"`` via the
route-suitability path, and lets a route label be presented as measured backend
correctness. This module separates them.

Protocol invariant (enforced structurally):
    **An unexecuted backend can never be correct.**

    :class:`BackendExecutionRecord` raises in ``__post_init__`` if
    ``executed is False`` while ``verification.status is VERIFIED_CORRECT``.
    The :attr:`BackendExecutionRecord.is_correct` property is ``True`` *only*
    when the backend was executed and independently verified correct.

This module is dependency-free (stdlib + the existing verification layer) so it
can be imported anywhere ``daph_learning`` runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from daph_learning.evaluation.verification import (
    NumericVerifier,
    RouteSuitabilityVerifier,
    VerificationResult,
    VerificationStatus,
    OutcomeVerifier,
)

# The two executable backends. ``"abstain"`` is a route *decision* (V039-003),
# not a backend; an abstained task executes neither backend.
BACKENDS = ("symbolic", "llm")
RouteAction = str  # "symbolic" | "llm" | "abstain" | None


@dataclass(frozen=True)
class BackendExecutionRecord:
    """Per-backend execution + verification record.

    Attributes
    ----------
    backend : str
        ``"symbolic"`` or ``"llm"``.
    selected : bool
        Whether the router selected this backend. In counterfactual execution
        both backends are executed regardless of selection; ``selected``
        records the router's preference for route-suitability analysis.
    executed : bool
        Whether this backend was actually run. The protocol invariant is
        enforced on this field: an unexecuted backend can never be correct.
    output : Any | None
        The backend's raw output (e.g. ``"FINAL_ANSWER: 42"`` or generated text).
    verification : VerificationResult
        Independent verifier result. For an unexecuted backend this must be
        ``NOT_EXECUTED`` (never ``VERIFIED_CORRECT``).
    latency_ms : float | None
        Measured latency for this backend run, or ``None`` if not timed.
    compute_cost : float | None
        Normalized compute / token cost, or ``None`` if not measured.
    failure_type : str | None
        ``None`` on success, otherwise a short failure classifier
        (e.g. ``"execution_error"``, ``"invalid_output"``).
    """

    backend: str
    selected: bool
    executed: bool
    output: Any | None = None
    verification: VerificationResult = field(
        default_factory=lambda: VerificationResult(
            status=VerificationStatus.NOT_EXECUTED,
            verifier="none",
        )
    )
    latency_ms: float | None = None
    compute_cost: float | None = None
    failure_type: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(
                f"backend must be one of {BACKENDS!r}, got {self.backend!r}"
            )
        # Structural enforcement of the protocol invariant:
        # an unexecuted backend can never be correct.
        if not self.executed and self.verification.status is VerificationStatus.VERIFIED_CORRECT:
            raise ValueError(
                f"protocol invariant violated: backend {self.backend!r} is "
                f"unexecuted but verification status is VERIFIED_CORRECT"
            )
        if not self.executed and self.verification.status is not VerificationStatus.NOT_EXECUTED:
            # An unexecuted backend must carry NOT_EXECUTED, not a stale
            # verification from a prior run. This keeps the four semantics
            # truly separate.
            raise ValueError(
                f"protocol invariant violated: backend {self.backend!r} is "
                f"unexecuted but verification status is "
                f"{self.verification.status.value!r}, expected NOT_EXECUTED"
            )

    @property
    def is_correct(self) -> bool:
        """``True`` iff executed and independently verified correct.

        This is the crisp boolean used by the utility model. It is **never**
        ``True`` for an unexecuted backend, by construction.
        """
        return bool(
            self.executed
            and self.verification.status is VerificationStatus.VERIFIED_CORRECT
        )

    @property
    def is_executed_correct(self) -> bool:
        """Alias for :attr:`is_correct` for readability at call sites."""
        return self.is_correct


@dataclass(frozen=True)
class OutcomeSemantics:
    """Full separated outcome for a single task.

    The four semantics the protocol requires are all present and distinct:

    - :attr:`backend_selected` — the router's decision
      (``"symbolic"``/``"llm"``/``"abstain"``/``None``).
    - :attr:`backends` — per-backend :class:`BackendExecutionRecord` carrying
      ``executed`` and ``verification``.
    - :attr:`route_suitability` — oracle-based route-suitability result,
      separate from backend correctness.
    """

    task_id: str
    backend_selected: RouteAction
    backends: Mapping[str, BackendExecutionRecord]
    route_suitability: VerificationResult

    def __post_init__(self) -> None:
        for b in self.backends:
            if b not in BACKENDS:
                raise ValueError(f"unknown backend {b!r} in backends mapping")
        # The selected backend, if it is a real backend, must be present and
        # marked selected in its record.
        if self.backend_selected in BACKENDS:
            rec = self.backends.get(self.backend_selected)
            if rec is None:
                raise ValueError(
                    f"selected backend {self.backend_selected!r} has no "
                    f"execution record"
                )
            if not rec.selected:
                raise ValueError(
                    f"selected backend {self.backend_selected!r} record is "
                    f"not marked selected"
                )

    @property
    def selected_backend_record(self) -> BackendExecutionRecord | None:
        if self.backend_selected in BACKENDS:
            return self.backends[self.backend_selected]
        return None

    @property
    def selected_backend_executed(self) -> bool:
        rec = self.selected_backend_record
        return bool(rec and rec.executed)

    @property
    def selected_backend_correct(self) -> bool:
        """The selected backend's verified correctness.

        ``False`` when the selected backend was not executed — an unexecuted
        backend can never be correct.
        """
        rec = self.selected_backend_record
        return bool(rec and rec.is_correct)

    @property
    def both_backends_executed(self) -> bool:
        return all(self.backends[b].executed for b in BACKENDS)

    def backend_correct(self, backend: str) -> bool:
        if backend not in self.backends:
            return False
        return self.backends[backend].is_correct

    def to_legacy_label(self) -> str:
        """Bridge to the v0.3.8 ``OutcomeLabel`` vocabulary.

        Preserves the historical ``"correct" | "misrouted" | "unverifiable"``
        contract for callers that have not migrated to the separated
        semantics. The mapping is intentionally conservative: only a
        *selected and executed and verified-correct* backend yields
        ``"correct"``; a selected backend that executed but verified incorrect
        yields ``"misrouted"``; everything else yields ``"unverifiable"``.
        """
        rec = self.selected_backend_record
        if rec is None or not rec.executed:
            return "unverifiable"
        status = rec.verification.status
        if status is VerificationStatus.VERIFIED_CORRECT:
            return "correct"
        if status in {
            VerificationStatus.VERIFIED_INCORRECT,
            VerificationStatus.EXECUTION_FAILED,
        }:
            return "misrouted"
        if status is VerificationStatus.INVALID_OUTPUT and rec.backend == "symbolic":
            return "misrouted"
        return "unverifiable"


def _not_executed_record(backend: str, selected: bool) -> BackendExecutionRecord:
    return BackendExecutionRecord(
        backend=backend,
        selected=selected,
        executed=False,
        verification=VerificationResult(
            status=VerificationStatus.NOT_EXECUTED,
            verifier="none",
        ),
    )


def build_outcome_semantics(
    task: Mapping[str, Any],
    *,
    backend_selected: RouteAction,
    executed_backends: Mapping[str, dict[str, Any]],
    verifier: OutcomeVerifier | None = None,
    route_suitability_verifier: RouteSuitabilityVerifier | None = None,
    expected: Any | None = None,
    oracle_field: str = "utility_oracle",
) -> OutcomeSemantics:
    """Assemble :class:`OutcomeSemantics` from raw per-backend execution data.

    Parameters
    ----------
    task : Mapping
        The task dict (must contain ``task_id``).
    backend_selected : str | None
        The router's decision (``"symbolic"``/``"llm"``/``"abstain"``/``None``).
    executed_backends : Mapping[str, dict]
        A mapping of backend name -> execution record dict with keys:
        ``output`` (Any|None), ``execution_succeeded`` (bool),
        ``latency_ms`` (float|None), ``compute_cost`` (float|None),
        ``failure_type`` (str|None). Backends not present in this mapping are
        treated as not executed.
    verifier : OutcomeVerifier | None
        Independent answer verifier. Defaults to :class:`NumericVerifier`.
        The router must not be its own verifier.
    route_suitability_verifier : RouteSuitabilityVerifier | None
        Bootstrap route-suitability verifier. Defaults to a fresh
        :class:`RouteSuitabilityVerifier` using ``oracle_field``.
    expected : Any | None
        Override for the task's expected answer.
    oracle_field : str
        Task field holding the policy oracle (``"symbolic"``/``"llm"``).
    """
    if verifier is None:
        verifier = NumericVerifier()
    if route_suitability_verifier is None:
        route_suitability_verifier = RouteSuitabilityVerifier(oracle_field=oracle_field)

    if backend_selected is not None and backend_selected not in BACKENDS and backend_selected != "abstain":
        raise ValueError(
            f"backend_selected must be one of {BACKENDS!r}, 'abstain', or None; "
            f"got {backend_selected!r}"
        )
    task_id = str(task.get("task_id", ""))
    expected_value = expected if expected is not None else task.get("expected")

    records: dict[str, BackendExecutionRecord] = {}
    for backend in BACKENDS:
        selected = backend_selected == backend
        raw = executed_backends.get(backend)
        if raw is None:
            # The backend was never invoked. NOT_EXECUTED, never correct.
            records[backend] = _not_executed_record(backend, selected)
            continue
        # The backend was invoked (executed=True). Whether it *succeeded*
        # in producing an output is a separate axis, driven by
        # ``execution_succeeded`` and passed to the verifier. A failed run
        # is still an executed backend — it just is not correct.
        output = raw.get("output")
        execution_succeeded = bool(
            raw.get("execution_succeeded", output is not None)
        )
        verification = verifier.verify(
            {**task, "expected": expected_value},
            output,
            execution_succeeded=execution_succeeded,
        )
        failure_type = raw.get("failure_type")
        if failure_type is None and verification.status in {
            VerificationStatus.EXECUTION_FAILED,
            VerificationStatus.INVALID_OUTPUT,
        }:
            failure_type = verification.status.value
        records[backend] = BackendExecutionRecord(
            backend=backend,
            selected=selected,
            executed=True,
            output=output,
            verification=verification,
            latency_ms=raw.get("latency_ms"),
            compute_cost=raw.get("compute_cost"),
            failure_type=failure_type,
        )

    route_suitability = route_suitability_verifier.verify_route(task, backend_selected)

    return OutcomeSemantics(
        task_id=task_id,
        backend_selected=backend_selected,
        backends=records,
        route_suitability=route_suitability,
    )


__all__ = [
    "BACKENDS",
    "BackendExecutionRecord",
    "OutcomeSemantics",
    "build_outcome_semantics",
]
