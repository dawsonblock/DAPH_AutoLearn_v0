"""v0.3.9 (V039-009) — immutable, hashed experience records.

Item 3 of the recommended repair sequence: every learning experience is
recorded as an **immutable** row whose content is hashed, and the per-row
hashes are coupled into a ledger hash that also binds the dataset hash and
the frozen utility-config hash. This makes a recorded experience
reproducible and tamper-evident: changing the task, prompt, output,
verifier, timing method, model/tokenizer revision, utility config, or
capture counts changes a hash and breaks the coupling.

Hashed components (per record):
    task, prompts, per-backend outputs, verifier results + verifier config,
    timing method + per-backend timing, model revision, tokenizer revision,
    utility configuration, capture attempts / successes.

Coupling (per ledger):
    ``ledger_hash = sha256(record_hashes ‖ data_hash ‖ config_hash ‖
                           model_revision ‖ tokenizer_revision ‖
                           timing_method)``
    so changing the dataset, the utility config, or the model/tokenizer
    revision changes the ledger hash even if no record content changed.

The module is dependency-free (stdlib + the outcome/counterfactual layers).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from daph_learning.autolearn.counterfactual import UtilityConfig
from daph_learning.autolearn.outcome import OutcomeSemantics


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_json(obj: Any) -> str:
    return _sha256_hex(
        json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    )


def hash_task(task: Mapping[str, Any]) -> str:
    """SHA-256 of a task dict (canonical JSON)."""
    return _hash_json(dict(task))


def hash_prompt(prompt: str) -> str:
    """SHA-256 of a rendered prompt string."""
    return _sha256_hex(prompt.encode("utf-8"))


def hash_output(output: Any) -> str:
    """SHA-256 of a backend output (canonical JSON of the value)."""
    return _hash_json({"output": output})


def hash_verifier(verifier: Any) -> str:
    """SHA-256 of a verifier's identity + configuration.

    The verifier must not be the router itself; recording the verifier
    identity makes that auditable.
    """
    payload = {
        "class": type(verifier).__name__,
        "module": type(verifier).__module__,
        "fields": asdict(verifier) if _is_dataclass(verifier) else None,
    }
    return _hash_json(payload)


def _is_dataclass(instance: Any) -> bool:
    try:
        import dataclasses
        return dataclasses.is_dataclass(instance)
    except ImportError:  # pragma: no cover - dataclasses always present on 3.12
        return False


def timing_method_name(device: str, cuda_event_timing: bool) -> str:
    """Canonical timing-method label recorded in the experience row."""
    if device == "cuda" and cuda_event_timing:
        return "cuda_event"
    return "perf_counter"


@dataclass(frozen=True)
class CaptureProvenance:
    """Capture-step provenance, used to verify capture-count coupling.

    The experience record's ``capture_attempts`` / ``capture_successes``
    must equal the capture step's reported counts. This struct carries the
    capture step's own counts so :func:`verify_capture_counts` can enforce
    the match.
    """

    n_attempts: int
    n_successes: int
    layer: int | None = None
    token_scope: str | None = None
    capture_hash: str | None = None  # hash of the captured activation bytes

    @property
    def coverage(self) -> float:
        return self.n_successes / self.n_attempts if self.n_attempts else 0.0


# --- v0.3.9: structured per-task capture results (Section 6) ---


def _activation_hash(activation: Any) -> str | None:
    """SHA-256 of an activation array (canonical bytes)."""
    if activation is None:
        return None
    try:
        arr = np.asarray(activation)
        return _sha256_hex(arr.tobytes())
    except (TypeError, ValueError, AttributeError):
        return None


@dataclass(frozen=True)
class CapturedActivation:
    """One per-task activation-capture result (Section 6 of the repair spec).

    Activation capture may fail for individual tasks. This struct records the
    outcome per task so that weighted class means can join activations and
    utility weights by ``task_id`` — never by positional index.

    Attributes
    ----------
    task_id : str
        The task this activation was captured for.
    activation : np.ndarray | None
        The captured activation vector, or ``None`` if capture failed.
    layer : int | None
        Layer at which the activation was captured.
    token_location : str | None
        Token position / pooling scope of the capture.
    success : bool
        Whether capture succeeded.
    failure_reason : str | None
        Short reason if capture failed (``None`` on success).
    activation_hash : str | None
        SHA-256 of the activation bytes (``None`` if capture failed).
    """

    task_id: str
    activation: Any = None
    layer: int | None = None
    token_location: str | None = None
    success: bool = False
    failure_reason: str | None = None
    activation_hash: str | None = None

    def __post_init__(self) -> None:
        if self.success and self.activation is None:
            raise ValueError(
                f"CapturedActivation for task {self.task_id!r} marked success "
                f"but activation is None"
            )
        if not self.success and self.activation is not None:
            raise ValueError(
                f"CapturedActivation for task {self.task_id!r} marked failure "
                f"but activation is not None"
            )
        # Compute hash if not provided and activation exists.
        if self.activation_hash is None and self.activation is not None:
            object.__setattr__(self, "activation_hash", _activation_hash(self.activation))


@dataclass(frozen=True)
class CaptureResult:
    """Structured result of a capture call over a set of tasks.

    Replaces the old ``(activations[N,D], n_attempts, n_successes)`` tuple.
    The per-task :class:`CapturedActivation` list lets the loop join
    activations and utility weights by ``task_id``.
    """

    activations: list[CapturedActivation] = field(default_factory=list)
    n_attempts: int = 0
    n_successes: int = 0

    @property
    def coverage(self) -> float:
        return self.n_successes / self.n_attempts if self.n_attempts else 0.0

    def successful_by_task_id(self) -> dict[str, CapturedActivation]:
        """Map task_id -> CapturedActivation for successful captures only."""
        return {a.task_id: a for a in self.activations if a.success}

    def to_provenance(self, layer: int | None = None, token_scope: str | None = None) -> CaptureProvenance:
        return CaptureProvenance(
            n_attempts=self.n_attempts,
            n_successes=self.n_successes,
            layer=layer,
            token_scope=token_scope,
            capture_hash=None,
        )


@dataclass(frozen=True)
class ExperienceRecord:
    """One immutable, hashed learning experience.

    The ``record_hash`` couples every hashed component: changing any field
    that feeds it changes the hash. The hash is computed by
    :func:`build_experience_record`; do not construct this dataclass
    directly with a hand-computed hash.
    """

    record_id: str
    task_id: str
    task_hash: str
    prompt_hash: str
    backend_output_hashes: Mapping[str, str]
    verifier_results: Mapping[str, str]
    verifier_hash: str
    timing_method: str
    timing_hashes: Mapping[str, str]
    model_revision: str | None
    tokenizer_revision: str | None
    utility_config_hash: str
    capture_attempts: int
    capture_successes: int
    record_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_hash_payload(record_without_hash: dict[str, Any]) -> str:
    """Compute the coupling hash over every record field except itself."""
    payload = {k: v for k, v in record_without_hash.items() if k != "record_hash"}
    return _hash_json(payload)


def build_experience_record(
    *,
    record_id: str,
    task: Mapping[str, Any],
    prompt: str,
    outcome: OutcomeSemantics,
    verifier: Any,
    timing_method: str,
    per_backend_timing: Mapping[str, Mapping[str, Any]],
    model_revision: str | None,
    tokenizer_revision: str | None,
    utility_config: UtilityConfig,
    capture_provenance: CaptureProvenance | None = None,
) -> ExperienceRecord:
    """Assemble an :class:`ExperienceRecord` from an executed outcome.

    ``per_backend_timing`` maps backend -> ``{"device": str,
    "cuda_event_timing": bool, "elapsed_ms": float}`` (the fields produced
    by :class:`daph_learning.evaluation.timing.TimingResult`).
    """
    cfg = utility_config.freeze()
    task_id = str(task.get("task_id", ""))
    backend_output_hashes = {
        b: hash_output(outcome.backends[b].output) for b in outcome.backends
    }
    verifier_results = {
        b: outcome.backends[b].verification.status.value
        for b in outcome.backends
    }
    timing_hashes = {
        b: _hash_json({
            "device": per_backend_timing.get(b, {}).get("device"),
            "cuda_event_timing": per_backend_timing.get(b, {}).get("cuda_event_timing"),
            "elapsed_ms": per_backend_timing.get(b, {}).get("elapsed_ms"),
        })
        for b in outcome.backends
    }
    cap = capture_provenance or CaptureProvenance(n_attempts=0, n_successes=0)
    fields_dict = {
        "record_id": record_id,
        "task_id": task_id,
        "task_hash": hash_task(task),
        "prompt_hash": hash_prompt(prompt),
        "backend_output_hashes": dict(backend_output_hashes),
        "verifier_results": dict(verifier_results),
        "verifier_hash": hash_verifier(verifier),
        "timing_method": timing_method,
        "timing_hashes": dict(timing_hashes),
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "utility_config_hash": cfg.config_sha256,  # type: ignore[arg-type]
        "capture_attempts": cap.n_attempts,
        "capture_successes": cap.n_successes,
    }
    record_hash = _record_hash_payload(fields_dict)
    return ExperienceRecord(record_hash=record_hash, **fields_dict)


@dataclass(frozen=True)
class ProvenanceBundle:
    """The coupling bundle: ledger + data + config + revision provenance.

    ``ledger_hash`` incorporates ``data_hash``, ``config_hash``,
    ``model_revision``, ``tokenizer_revision``, and ``timing_method``, so
    changing any of those changes the ledger hash even if the records are
    byte-identical. This is the "ledger/data/config hashes are coupled"
    invariant.
    """

    data_hash: str
    config_hash: str
    model_revision: str | None
    tokenizer_revision: str | None
    timing_method: str
    ledger_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_ledger_hash(
    records: Sequence[ExperienceRecord],
    *,
    data_hash: str,
    config_hash: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    timing_method: str,
) -> str:
    """Couple record hashes with data/config/revision provenance."""
    payload = {
        "record_hashes": [r.record_hash for r in records],
        "data_hash": data_hash,
        "config_hash": config_hash,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "timing_method": timing_method,
    }
    return _hash_json(payload)


@dataclass
class ExperienceLedger:
    """An append-only ledger of experience records + their provenance bundle.

    Immutability is enforced by structural coupling: the ``bundle.ledger_hash``
    binds every record hash to the data/config/revision provenance.
    :meth:`verify_coupling` recomputes the ledger hash and checks every
    record's ``utility_config_hash`` equals the bundle's ``config_hash``.
    """

    bundle: ProvenanceBundle
    records: list[ExperienceRecord] = field(default_factory=list)

    @classmethod
    def open(
        cls,
        *,
        data_hash: str,
        utility_config: UtilityConfig,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        timing_method: str = "perf_counter",
    ) -> "ExperienceLedger":
        cfg = utility_config.freeze()
        bundle = ProvenanceBundle(
            data_hash=data_hash,
            config_hash=cfg.config_sha256,  # type: ignore[arg-type]
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            timing_method=timing_method,
            ledger_hash="",
        )
        ledger = cls(bundle=bundle)
        # recompute ledger hash with zero records to bind the bundle.
        ledger._recompute_ledger_hash()
        return ledger

    def _recompute_ledger_hash(self) -> None:
        # NOTE: frozen ProvenanceBundle would block setattr; we use
        # dataclass_replace to produce a new bundle with the updated hash.
        from daph_learning.autolearn.counterfactual import dataclass_replace
        new_hash = compute_ledger_hash(
            self.records,
            data_hash=self.bundle.data_hash,
            config_hash=self.bundle.config_hash,
            model_revision=self.bundle.model_revision,
            tokenizer_revision=self.bundle.tokenizer_revision,
            timing_method=self.bundle.timing_method,
        )
        self.bundle = dataclass_replace(self.bundle, ledger_hash=new_hash)

    def append(self, record: ExperienceRecord) -> None:
        """Append an immutable record and re-bind the ledger hash."""
        if record.utility_config_hash != self.bundle.config_hash:
            raise ValueError(
                "record utility_config_hash does not match ledger config_hash; "
                "the ledger couples a single frozen utility config"
            )
        self.records.append(record)
        self._recompute_ledger_hash()

    def verify_coupling(self) -> None:
        """Recompute the ledger hash and assert it matches the bundle.

        Also asserts every record's utility config hash equals the bundle's
        config hash (the ledger/config coupling) and that the data hash is
        non-empty. Raises on any mismatch.
        """
        expected = compute_ledger_hash(
            self.records,
            data_hash=self.bundle.data_hash,
            config_hash=self.bundle.config_hash,
            model_revision=self.bundle.model_revision,
            tokenizer_revision=self.bundle.tokenizer_revision,
            timing_method=self.bundle.timing_method,
        )
        if expected != self.bundle.ledger_hash:
            raise ValueError(
                "ledger coupling broken: ledger_hash mismatch "
                f"(expected {expected}, got {self.bundle.ledger_hash})"
            )
        for r in self.records:
            if r.utility_config_hash != self.bundle.config_hash:
                raise ValueError(
                    f"record {r.record_id!r} utility_config_hash does not "
                    f"match ledger config_hash"
                )
        if not self.bundle.data_hash:
            raise ValueError("ledger data_hash is empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "records": [r.to_dict() for r in self.records],
        }


def verify_capture_counts(
    record: ExperienceRecord,
    capture_provenance: CaptureProvenance,
) -> None:
    """Assert the record's capture counts match the capture step's provenance.

    This is the "successful-capture counts match provenance" acceptance test.
    """
    if record.capture_attempts != capture_provenance.n_attempts:
        raise ValueError(
            f"capture-attempts mismatch: record={record.capture_attempts} "
            f"provenance={capture_provenance.n_attempts}"
        )
    if record.capture_successes != capture_provenance.n_successes:
        raise ValueError(
            f"capture-successes mismatch: record={record.capture_successes} "
            f"provenance={capture_provenance.n_successes}"
        )


__all__ = [
    "CaptureProvenance",
    "CaptureResult",
    "CapturedActivation",
    "ExperienceLedger",
    "ExperienceRecord",
    "ProvenanceBundle",
    "build_experience_record",
    "compute_ledger_hash",
    "hash_output",
    "hash_prompt",
    "hash_task",
    "hash_verifier",
    "timing_method_name",
    "verify_capture_counts",
]
