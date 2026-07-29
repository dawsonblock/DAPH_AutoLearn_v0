"""v0.3.10.3.1-alpha — Section 3 / G7: one canonical utility function.

The same ``U_b`` definition must be used by training, dev evaluation,
calibration, final evaluation, promotion, steering experiments, random
controls, and hand-router baselines. This test enforces that:

1. No real scoring path duplicates the utility formula with a
   ``cfg.quality_weight * quality`` shortcut.
2. ``backend_utility`` and ``counterfactual.compute_backend_utility``
   agree on identical inputs (they share ``utility_formula``).
3. ``utility_for_route`` centralizes route-utility lookup.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from daph_learning.policy.config import ExperimentConfig
from daph_learning.policy.types import BackendOutcome, CounterfactualExperience, Route
from daph_learning.policy.utility import (
    backend_utility,
    utility_for_route,
    utility_formula,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "daph_learning"


def _outcome(backend: str, *, quality=0.8, latency=0.5, cost=0.1, risk=0.0):
    return BackendOutcome(
        task_id="t1", backend=backend, available=True, executed=True,
        execution_success=True, output_text="42", output_hash="h",
        verifier_status="verified_correct", correct=True,
        quality=quality, latency_sec=latency, normalized_cost=cost,
        risk=risk, verifier_confidence=1.0,
    )


def test_backend_utility_matches_formula():
    cfg = ExperimentConfig(quality_weight=2.0, lambda_time=0.5,
                           lambda_compute=0.25, lambda_risk=1.0)
    o = _outcome("symbolic", quality=0.9, latency=0.2, cost=0.5, risk=0.1)
    expected = utility_formula(
        quality=0.9, latency_ms=200.0, compute_cost=0.5, risk=0.1,
        quality_weight=2.0, lambda_time=0.5, lambda_compute=0.25,
        lambda_risk=1.0, time_reference_ms=1000.0, compute_reference=1.0)
    assert backend_utility(o, cfg) == pytest.approx(expected)


def test_utility_for_route_symbolic_llm_abstain():
    cfg = ExperimentConfig()
    sym = _outcome("symbolic", quality=0.9)
    llm = _outcome("llm", quality=0.4)
    exp = CounterfactualExperience(
        task_id="t1", symbolic=sym, llm=llm,
        delta_utility=backend_utility(sym, cfg) - backend_utility(llm, cfg),
        preferred_action=Route.SYMBOLIC, sample_weight=0.5)
    assert utility_for_route(exp, "symbolic", cfg) == pytest.approx(
        backend_utility(sym, cfg))
    assert utility_for_route(exp, "llm", cfg) == pytest.approx(
        backend_utility(llm, cfg))
    assert utility_for_route(exp, "abstain", cfg) == 0.0
    assert utility_for_route(exp, Route.LLM, cfg) == pytest.approx(
        backend_utility(llm, cfg))


def test_utility_for_route_rejects_unknown():
    cfg = ExperimentConfig()
    exp = CounterfactualExperience(
        task_id="t1", symbolic=_outcome("symbolic"), llm=_outcome("llm"),
        delta_utility=0.5, preferred_action=Route.SYMBOLIC, sample_weight=0.5)
    with pytest.raises(ValueError):
        utility_for_route(exp, "bogus", cfg)


def test_counterfactual_and_backend_utility_agree():
    """Section 3: the two typed entry points must agree because they
    share ``utility_formula``."""
    from daph_learning.autolearn.counterfactual import (
        UtilityConfig, compute_backend_utility, BackendExecutionRecord,
    )
    from daph_learning.evaluation.verification import (
        VerificationResult, VerificationStatus,
    )
    cfg = ExperimentConfig(quality_weight=1.5, lambda_time=0.3,
                           lambda_compute=0.2, lambda_risk=0.8)
    ucfg = UtilityConfig(quality_weight=1.5, lambda_time=0.3,
                         lambda_compute=0.2, lambda_risk=0.8)
    rec = BackendExecutionRecord(
        backend="symbolic", selected=True, executed=True,
        verification=VerificationResult(
            status=VerificationStatus.VERIFIED_CORRECT,
            verifier="exact", reason=""),
        latency_ms=250.0, compute_cost=0.4,
    )
    breakdown = compute_backend_utility(rec, ucfg)
    o = _outcome("symbolic", quality=1.0, latency=0.25, cost=0.4, risk=0.0)
    # _quality for a verified-correct record is 1.0; _risk is 0.0.
    assert breakdown.utility == pytest.approx(backend_utility(o, cfg))


def test_no_quality_weight_shortcut_in_real_scoring_paths():
    """Section 3: grep the source tree for duplicated utility shortcuts.

    A real scoring path must not compute utility as
    ``cfg.quality_weight * <outcome>.quality`` — that ignores
    time/compute/risk and drifts from the canonical formula. The only
    allowed occurrences are in the canonical utility module itself and
    in tests.
    """
    pattern = re.compile(r"quality_weight\s*\*\s*\w+\.(symbolic|llm)\.quality")
    offenders = []
    for py in sorted(SRC.rglob("*.py")):
        rel = py.relative_to(REPO_ROOT)
        text = py.read_text(encoding="utf-8")
        # Skip the canonical utility module and the legacy
        # counterfactual path (which now delegates to utility_formula).
        if "policy/utility.py" in str(rel):
            continue
        for m in pattern.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            offenders.append(f"{rel}:{line_no}: {m.group(0)}")
    assert not offenders, (
        "duplicated utility shortcut found (use utility_for_route/"
        "backend_utility instead):\n" + "\n".join(offenders))
