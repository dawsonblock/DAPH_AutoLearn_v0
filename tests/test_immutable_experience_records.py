"""V039-009 — immutable, hashed experience records.

Acceptance tests for Item 3:
* records are immutable and content-hashed (task, prompts, outputs, verifier,
  timing, revisions, utility config, capture counts)
* ledger/data/config hashes are coupled
* successful-capture counts match provenance
* GPU capture succeeds (CUDA event timing recorded when available)
"""

from __future__ import annotations

import pytest

from daph_learning.autolearn.counterfactual import (
    UtilityConfig,
    execute_both_backends,
)
from daph_learning.autolearn.experience import (
    CaptureProvenance,
    ExperienceLedger,
    build_experience_record,
    compute_ledger_hash,
    hash_output,
    hash_prompt,
    hash_task,
    hash_verifier,
    timing_method_name,
    verify_capture_counts,
)
from daph_learning.evaluation.verification import NumericVerifier


def _task(expected=42, oracle="symbolic", task_id="t1"):
    return {"task_id": task_id, "expected": expected,
            "capability_ids": ["integer_arithmetic"], "utility_oracle": oracle}


def _sym(answer):
    def _run(t):
        return {"output": None if answer is None else f"FINAL: {answer}",
                "execution_succeeded": answer is not None,
                "latency_ms": 1.0, "compute_cost": 0.0,
                "failure_type": None if answer is not None else "execution_error"}
    return _run


def _llm(answer):
    def _run(t):
        return {"output": f"FINAL: {answer}", "execution_succeeded": True,
                "latency_ms": 50.0, "compute_cost": 0.01, "failure_type": None}
    return _run


def _make_record(task=None, expected=42, sym_ans=42, llm_ans=99, cfg=None,
                 capture=None, prompt="ACTION: SYMBOLIC", record_id="r1"):
    task = task or _task(expected=expected)
    cfg = cfg or UtilityConfig(lambda_risk=1.0)
    outcome = execute_both_backends(
        task, backend_selected="symbolic",
        symbolic_runner=_sym(sym_ans), llm_runner=_llm(llm_ans))
    timing = {
        "symbolic": {"device": "cpu", "cuda_event_timing": False, "elapsed_ms": 1.0},
        "llm": {"device": "cpu", "cuda_event_timing": False, "elapsed_ms": 50.0},
    }
    return build_experience_record(
        record_id=record_id, task=task, prompt=prompt, outcome=outcome,
        verifier=NumericVerifier(), timing_method="perf_counter",
        per_backend_timing=timing, model_revision="main@abc",
        tokenizer_revision="main@abc", utility_config=cfg,
        capture_provenance=capture or CaptureProvenance(0, 0),
    ), cfg, task


# --- Records are content-hashed ---

def test_record_hash_covers_task():
    rec1, _, _ = _make_record(task=_task(expected=42))
    rec2, _, _ = _make_record(task=_task(expected=43))
    assert rec1.record_hash != rec2.record_hash


def test_record_hash_covers_prompt():
    rec1, _, _ = _make_record(prompt="ACTION: SYMBOLIC")
    rec2, _, _ = _make_record(prompt="ACTION: LLM")
    assert rec1.record_hash != rec2.record_hash


def test_record_hash_covers_outputs():
    rec1, _, _ = _make_record(sym_ans=42, llm_ans=99)
    rec2, _, _ = _make_record(sym_ans=42, llm_ans=100)
    assert rec1.record_hash != rec2.record_hash


def test_record_hash_covers_verifier():
    rec1, _, _ = _make_record()
    # same content, different verifier identity -> different verifier_hash
    class OtherVerifier(NumericVerifier):
        pass
    task = _task()
    cfg = UtilityConfig(lambda_risk=1.0)
    from daph_learning.autolearn.counterfactual import execute_both_backends
    outcome = execute_both_backends(task, backend_selected="symbolic",
                                    symbolic_runner=_sym(42), llm_runner=_llm(99))
    timing = {"symbolic": {"device": "cpu", "cuda_event_timing": False, "elapsed_ms": 1.0},
              "llm": {"device": "cpu", "cuda_event_timing": False, "elapsed_ms": 50.0}}
    rec2 = build_experience_record(
        record_id="r1", task=task, prompt="ACTION: SYMBOLIC", outcome=outcome,
        verifier=OtherVerifier(), timing_method="perf_counter",
        per_backend_timing=timing, model_revision="main@abc",
        tokenizer_revision="main@abc", utility_config=cfg,
        capture_provenance=CaptureProvenance(0, 0))
    assert rec1.verifier_hash != rec2.verifier_hash
    assert rec1.record_hash != rec2.record_hash


def test_record_hash_covers_timing_method():
    rec1, _, _ = _make_record()
    # rebuild with a different timing method
    task = _task()
    cfg = UtilityConfig(lambda_risk=1.0)
    outcome = execute_both_backends(task, backend_selected="symbolic",
                                    symbolic_runner=_sym(42), llm_runner=_llm(99))
    timing = {"symbolic": {"device": "cuda", "cuda_event_timing": True, "elapsed_ms": 0.5},
              "llm": {"device": "cuda", "cuda_event_timing": True, "elapsed_ms": 5.0}}
    rec2 = build_experience_record(
        record_id="r1", task=task, prompt="ACTION: SYMBOLIC", outcome=outcome,
        verifier=NumericVerifier(), timing_method="cuda_event",
        per_backend_timing=timing, model_revision="main@abc",
        tokenizer_revision="main@abc", utility_config=cfg,
        capture_provenance=CaptureProvenance(0, 0))
    assert rec1.timing_method != rec2.timing_method
    assert rec1.record_hash != rec2.record_hash


def test_record_hash_covers_model_revision():
    rec1, _, _ = _make_record()
    task = _task()
    cfg = UtilityConfig(lambda_risk=1.0)
    outcome = execute_both_backends(task, backend_selected="symbolic",
                                    symbolic_runner=_sym(42), llm_runner=_llm(99))
    timing = {"symbolic": {"device": "cpu", "cuda_event_timing": False, "elapsed_ms": 1.0},
              "llm": {"device": "cpu", "cuda_event_timing": False, "elapsed_ms": 50.0}}
    rec2 = build_experience_record(
        record_id="r1", task=task, prompt="ACTION: SYMBOLIC", outcome=outcome,
        verifier=NumericVerifier(), timing_method="perf_counter",
        per_backend_timing=timing, model_revision="main@def",
        tokenizer_revision="main@abc", utility_config=cfg,
        capture_provenance=CaptureProvenance(0, 0))
    assert rec1.record_hash != rec2.record_hash


def test_record_hash_covers_utility_config():
    rec1, cfg1, _ = _make_record(cfg=UtilityConfig(lambda_risk=1.0))
    rec2, _, _ = _make_record(cfg=UtilityConfig(lambda_risk=2.0))
    assert rec1.utility_config_hash != rec2.utility_config_hash
    assert rec1.record_hash != rec2.record_hash


def test_record_hash_covers_capture_counts():
    rec1, _, _ = _make_record(capture=CaptureProvenance(10, 8))
    rec2, _, _ = _make_record(capture=CaptureProvenance(10, 7))
    assert rec1.record_hash != rec2.record_hash


def test_record_is_immutable():
    rec, _, _ = _make_record()
    with pytest.raises(Exception):
        rec.task_id = "tampered"  # type: ignore[misc]


# --- ledger/data/config hashes are coupled ---

def test_ledger_hash_couples_records_and_data_and_config():
    cfg = UtilityConfig(lambda_risk=1.0)
    rec, _, task = _make_record(cfg=cfg)
    data_hash = hash_task(task)
    h1 = compute_ledger_hash([rec], data_hash=data_hash, config_hash=cfg.freeze().config_sha256,
                             model_revision="main@abc", tokenizer_revision="main@abc",
                             timing_method="perf_counter")
    # different data hash -> different ledger hash
    h2 = compute_ledger_hash([rec], data_hash="deadbeef", config_hash=cfg.freeze().config_sha256,
                             model_revision="main@abc", tokenizer_revision="main@abc",
                             timing_method="perf_counter")
    assert h1 != h2
    # different config hash -> different ledger hash
    h3 = compute_ledger_hash([rec], data_hash=data_hash, config_hash="cafebabe",
                             model_revision="main@abc", tokenizer_revision="main@abc",
                             timing_method="perf_counter")
    assert h1 != h3


def test_ledger_verify_coupling_passes_on_consistent_ledger():
    cfg = UtilityConfig(lambda_risk=1.0)
    rec, _, task = _make_record(cfg=cfg)
    data_hash = hash_task(task)
    ledger = ExperienceLedger.open(data_hash=data_hash, utility_config=cfg,
                                   model_revision="main@abc",
                                   tokenizer_revision="main@abc",
                                   timing_method="perf_counter")
    ledger.append(rec)
    ledger.verify_coupling()  # must not raise


def test_ledger_rejects_record_with_mismatched_config_hash():
    cfg = UtilityConfig(lambda_risk=1.0)
    rec, _, task = _make_record(cfg=UtilityConfig(lambda_risk=2.0))
    data_hash = hash_task(task)
    ledger = ExperienceLedger.open(data_hash=data_hash, utility_config=cfg,
                                   model_revision="main@abc",
                                   tokenizer_revision="main@abc",
                                   timing_method="perf_counter")
    with pytest.raises(ValueError, match="utility_config_hash"):
        ledger.append(rec)


def test_ledger_hash_changes_on_append():
    cfg = UtilityConfig(lambda_risk=1.0)
    rec1, _, task = _make_record(cfg=cfg, record_id="r1")
    rec2, _, _ = _make_record(cfg=cfg, record_id="r2", task=_task(task_id="t2"))
    data_hash = hash_task(task)
    ledger = ExperienceLedger.open(data_hash=data_hash, utility_config=cfg,
                                   model_revision="main@abc",
                                   tokenizer_revision="main@abc",
                                   timing_method="perf_counter")
    h0 = ledger.bundle.ledger_hash
    ledger.append(rec1)
    h1 = ledger.bundle.ledger_hash
    ledger.append(rec2)
    h2 = ledger.bundle.ledger_hash
    assert len({h0, h1, h2}) == 3


# --- successful-capture counts match provenance ---

def test_capture_counts_match_provenance():
    cap = CaptureProvenance(n_attempts=10, n_successes=8)
    rec, _, _ = _make_record(capture=cap)
    verify_capture_counts(rec, cap)  # must not raise


def test_capture_counts_mismatch_raises():
    cap = CaptureProvenance(n_attempts=10, n_successes=8)
    rec, _, _ = _make_record(capture=cap)
    wrong = CaptureProvenance(n_attempts=10, n_successes=7)
    with pytest.raises(ValueError, match="capture-successes mismatch"):
        verify_capture_counts(rec, wrong)


# --- GPU capture succeeds (CUDA event timing recorded when available) ---

def test_timing_method_name_cuda_when_available():
    assert timing_method_name("cuda", True) == "cuda_event"
    assert timing_method_name("cuda", False) == "perf_counter"
    assert timing_method_name("cpu", False) == "perf_counter"


def test_gpu_capture_succeeds_or_skips_gracefully():
    """Acceptance test: GPU capture succeeds when CUDA is available, and
    degrades gracefully (perf_counter) when it is not. Either way the
    timing method is recorded in the experience row."""
    from daph_learning.evaluation.timing import cuda_timed, _torch_cuda_available

    cfg = UtilityConfig(lambda_risk=1.0)
    task = _task()
    outcome = execute_both_backends(task, backend_selected="symbolic",
                                    symbolic_runner=_sym(42), llm_runner=_llm(99))
    timing = {}
    for backend in ("symbolic", "llm"):
        with cuda_timed() as t:
            pass  # no-op region; measures timer overhead
        timing[backend] = {
            "device": t.device,
            "cuda_event_timing": t.cuda_event_timing,
            "elapsed_ms": t.elapsed_ms,
        }
    method = timing_method_name(timing["symbolic"]["device"],
                                timing["symbolic"]["cuda_event_timing"])
    if _torch_cuda_available():
        assert method == "cuda_event"
        assert timing["symbolic"]["device"] == "cuda"
    else:
        assert method == "perf_counter"
        assert timing["symbolic"]["device"] == "cpu"
    rec = build_experience_record(
        record_id="r1", task=task, prompt="ACTION: SYMBOLIC", outcome=outcome,
        verifier=NumericVerifier(), timing_method=method,
        per_backend_timing=timing, model_revision="main@abc",
        tokenizer_revision="main@abc", utility_config=cfg,
        capture_provenance=CaptureProvenance(0, 0))
    assert rec.timing_method == method
