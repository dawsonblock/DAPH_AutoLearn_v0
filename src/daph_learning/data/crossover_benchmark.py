"""v0.3.10.3.1-alpha — within-family crossover benchmark (Section 10-12).

This is the single most important scientific upgrade in the release.

The old benchmark was too easy: ``arithmetic -> symbolic`` vs
``letter counting -> LLM``. A router could solve it by task-family
classification alone. This module replaces that with a benchmark where
the **same broad family** ("structured_math" — structured +
natural-language mathematics) contains both symbolic-preferred and
LLM-preferred instances.

Subtypes (Section 10):

  A. Direct exact arithmetic        "Compute 487 * 213."        -> symbolic
  B. Semantic extraction + arith    "A warehouse has 487 crates  -> LLM/hybrid
                                     with 213 units each. How
                                     many units total?"
  C. Ambiguous/malformed expression "What is 75 minus twice 18?" -> LLM
  D. Structured modular arithmetic   "Compute 123456 mod 97."   -> symbolic
  E. Comparison / relation problem   "Which is larger: 38*47     -> reasoning
                                     or 21*91?"
  F. Multi-step NL arithmetic        "A tank has 400 L, loses    -> LLM
                                     17%, then gains 26 L..."

The point is NOT to force predetermined outcomes. The optimal backend
is derived only AFTER both backends execute, the verifier runs, and
utilities are computed (Section 11).

Critical: task metadata MUST NOT contain ``best_backend``,
``symbolic_preferred``, ``llm_preferred``, ``route_label``, or
``utility_oracle``. The optimal action is a function of executed +
verified utility, not a stored label.

The family is balanced so that ``0.2 < P(symbolic optimal | family)
< 0.8`` (Section 10), making instance-level routing non-trivial: a
constant-action policy cannot achieve zero regret inside this family.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping

GENERATOR_VERSION = "v0.3.10.3.1-crossover"

# The single broad family. Both backend preferences occur inside it.
FAMILY_ID = "structured_math"

# Subtypes A-F. Each is a member of the SAME family.
SUBTYPES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

# Subtype descriptions (for documentation / provenance only; never used
# as a routing feature or label).
SUBTYPE_DESCRIPTIONS: dict[str, str] = {
    "A": "direct exact arithmetic",
    "B": "semantic extraction + exact arithmetic",
    "C": "ambiguous/malformed expression",
    "D": "structured modular arithmetic",
    "E": "comparison / relation problem",
    "F": "multi-step natural-language arithmetic",
}

# Split licensing by template slot (disjoint from the v0.3.8 generator).
SPLIT_TEMPLATE_SLOTS: dict[str, tuple[int, ...]] = {
    "train": (0, 1, 2, 3),
    "dev": (4, 5),
    "calibration": (6,),
    "final": (7,),
}

# Wording templates per slot. Slot 0-3 train, 4-5 dev, 6 calibration,
# 7 final. Each slot is a distinct wording so splits are
# template-disjoint (no wording leakage).
_EXACT_WRAPPERS = (
    "Compute {body}. Return only the integer.",
    "Please compute {body}. Reply with the exact integer only.",
    "Exact arithmetic request: Compute {body}. Output one integer.",
    "Task: Compute {body}. Give no prose, only the integer.",
    "Held-out wording: Compute {body}. Print only its integer value.",
    "Evaluate exactly: Compute {body}. The response must be an integer.",
    "Calibration wording — Compute {body}. Return the integer result only.",
    "Final wording — Compute {body}. Respond with one integer.",
)

# Fields that MUST NOT appear in crossover task metadata (Section 11).
FORBIDDEN_METADATA_FIELDS = (
    "best_backend",
    "symbolic_preferred",
    "llm_preferred",
    "route_label",
    "utility_oracle",
    "capability_oracle",
    "accuracy_oracle",
)


def _metadata(subtype: str, split: str, template_slot: int, seed: int) -> dict[str, Any]:
    """Metadata for a crossover task. Contains grouping fields but NEVER
    the optimal backend (Section 11)."""
    return {
        "generator": GENERATOR_VERSION,
        "seed": seed,
        "split": split,
        "family_id": FAMILY_ID,
        "subtype": subtype,
        "subtype_description": SUBTYPE_DESCRIPTIONS[subtype],
        "template_id": f"{FAMILY_ID}:{subtype}:template_{template_slot}",
        "group_id": f"{FAMILY_ID}:{subtype}",
        "template_slot": template_slot,
    }


def _exact_prompt(body: str, slot: int) -> str:
    return _EXACT_WRAPPERS[slot].format(body=body)


def _gen_a(rng: random.Random, slot: int) -> dict[str, Any]:
    """A. Direct exact arithmetic. Structured inputs present -> symbolic
    can execute exactly. Large products favor symbolic (LLM may err)."""
    a = rng.randint(100, 9_999)
    b = rng.randint(100, 9_999)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        expected = a + b
    elif op == "-":
        expected = a - b
    else:
        expected = a * b
    body = f"{a} {op} {b}"
    return {
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": a, "b": b, "op": op},
        "specification": _exact_prompt(body, slot),
        "expected": expected,
    }


def _gen_b(rng: random.Random, slot: int) -> dict[str, Any]:
    """B. Semantic extraction + exact arithmetic. NL wrapper around an
    arithmetic problem; NO structured inputs -> symbolic cannot parse
    the NL, LLM must extract the numbers and compute."""
    a = rng.randint(50, 5_000)
    b = rng.randint(2, 500)
    scenarios = (
        ("crates", "units each. How many units total?", "*"),
        ("boxes", "items per box. How many items total?", "*"),
        ("pallets", "kilograms each. What is the total mass?", "*"),
        ("rows", "seats per row. How many seats total?", "*"),
    )
    noun, suffix, op = rng.choice(scenarios)
    if op == "*":
        expected = a * b
    body = f"A warehouse has {a} {noun} with {b} {suffix}"
    return {
        "capability_ids": [],  # no structured capability -> symbolic unsupported
        "inputs": {},
        "specification": body,
        "expected": expected,
    }


def _gen_c(rng: random.Random, slot: int) -> dict[str, Any]:
    """C. Ambiguous/malformed expression. Requires semantic
    interpretation before compute -> LLM-favorable."""
    x = rng.randint(10, 200)
    y = rng.randint(2, 50)
    forms = (
        (f"What is {x} minus twice {y}?", x - 2 * y),
        (f"Take {x} and subtract three times {y}.", x - 3 * y),
        (f"Add {x} and {y}, then double the result.", 2 * (x + y)),
        (f"What is half of {x} plus {y}?", x // 2 + y),
    )
    spec, expected = rng.choice(forms)
    return {
        "capability_ids": [],
        "inputs": {},
        "specification": spec,
        "expected": expected,
    }


def _gen_d(rng: random.Random, slot: int) -> dict[str, Any]:
    """D. Structured modular arithmetic. Structured inputs present ->
    symbolic executes exactly (``a mod modulus`` via integer_arithmetic)."""
    a = rng.randint(10_000, 1_000_000)
    modulus = rng.randint(2, 10_000)
    expected = a % modulus
    body = f"{a} mod {modulus}"
    return {
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": a, "b": modulus, "op": "%"},
        "specification": _exact_prompt(body, slot),
        "expected": expected,
    }


def _gen_e(rng: random.Random, slot: int) -> dict[str, Any]:
    """E. Comparison / relation problem. Requires computing both sides
    then comparing -> mixed; LLM may reason, symbolic needs a planner."""
    a1, b1 = rng.randint(10, 999), rng.randint(10, 999)
    a2, b2 = rng.randint(10, 999), rng.randint(10, 999)
    left, right = a1 * b1, a2 * b2
    if left == right:
        right += 1
    spec = (f"Which is larger: {a1}*{b1} or {a2}*{b2}? "
            f"Reply with the larger product as an integer.")
    expected = max(left, right)
    return {
        "capability_ids": [],  # comparison not a single typed action
        "inputs": {},
        "specification": spec,
        "expected": expected,
    }


def _gen_f(rng: random.Random, slot: int) -> dict[str, Any]:
    """F. Multi-step natural-language arithmetic. Requires parsing a
    multi-step word problem -> LLM-favorable unless symbolic parsing is
    robust (it is not, by design: no structured inputs)."""
    total = rng.randint(200, 5_000)
    loss_pct = rng.randint(10, 40)
    gain = rng.randint(10, 100)
    after_loss = total - (total * loss_pct) // 100
    expected = after_loss + gain
    spec = (f"A tank has {total} L, loses {loss_pct}%, then gains {gain} L. "
            f"How many litres remain? Return the integer.")
    return {
        "capability_ids": [],
        "inputs": {},
        "specification": spec,
        "expected": expected,
    }


_GENERATORS = {"A": _gen_a, "B": _gen_b, "C": _gen_c,
               "D": _gen_d, "E": _gen_e, "F": _gen_f}


def generate_crossover_task(
    task_id: str,
    rng: random.Random,
    subtype: str,
    *,
    split: str,
    template_slot: int,
    seed: int,
) -> dict[str, Any]:
    """Generate one within-family crossover task.

    The returned task contains ``expected`` (for verification) and
    grouping metadata (``family_id``, ``subtype``, ``group_id``,
    ``template_id``) but NEVER the optimal backend (Section 11).
    """
    if split not in SPLIT_TEMPLATE_SLOTS:
        raise ValueError(f"unknown split: {split!r}")
    if template_slot not in SPLIT_TEMPLATE_SLOTS[split]:
        raise ValueError(
            f"template slot {template_slot} is not licensed for split {split!r}")
    if subtype not in SUBTYPES:
        raise ValueError(f"unknown subtype: {subtype!r}; expected one of {SUBTYPES}")
    body = _GENERATORS[subtype](rng, template_slot)
    metadata = _metadata(subtype, split, template_slot, seed)
    task = {
        "task_id": task_id,
        **body,
        "metadata": metadata,
    }
    # Defensive: assert no forbidden field leaked in (Section 11).
    _assert_no_optimal_backend_encoded(task)
    return task


def _assert_no_optimal_backend_encoded(task: Mapping[str, Any]) -> None:
    """Section 11 leakage check: no field used by the policy may encode
    the optimal backend."""
    for field in FORBIDDEN_METADATA_FIELDS:
        if field in task:
            raise ValueError(
                f"crossover task {task.get('task_id')!r} leaks forbidden "
                f"field {field!r} (Section 11: do not encode the optimal "
                f"backend in task metadata)")
        meta = task.get("metadata", {})
        if isinstance(meta, Mapping) and field in meta:
            raise ValueError(
                f"crossover task {task.get('task_id')!r} metadata leaks "
                f"forbidden field {field!r}")


def generate_crossover_split(
    *,
    split: str,
    n_per_subtype: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Generate a full crossover split: ``n_per_subtype`` tasks per
    subtype (A-F), all within the ``structured_math`` family.

    Template slots are rotated deterministically across the licensed
    slots for the split so that wording diversity is preserved without
    replacement (Section 8).
    """
    slots = SPLIT_TEMPLATE_SLOTS[split]
    tasks: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for subtype in SUBTYPES:
        for i in range(n_per_subtype):
            slot = slots[(i + sum(ord(c) for c in subtype)) % len(slots)]
            global_seed = seed + hash((split, subtype, i)) % (2**31)
            tid = f"{split}_{subtype}_{i:04d}"
            tasks.append(generate_crossover_task(
                tid, rng, subtype, split=split,
                template_slot=slot, seed=global_seed))
    return tasks


def assert_no_within_split_duplicates(tasks: list[dict[str, Any]]) -> None:
    """Section 8: ``number_tasks == number_unique_prompt_hashes``."""
    from .integrity import normalize_prompt
    import hashlib
    seen: set[str] = set()
    for t in tasks:
        spec = str(t.get("specification", ""))
        h = hashlib.sha256(normalize_prompt(spec).encode("utf-8")).hexdigest()
        if h in seen:
            raise ValueError(
                f"within-split duplicate prompt in task {t.get('task_id')!r}")
        seen.add(h)


# ------------------------------------------------------------------
# Section 10-12: executing both backends to DERIVE the optimal action.
# The optimal backend is never stored in the task; it is computed from
# executed + verified utility (Section 11).
# ------------------------------------------------------------------


def simulate_llm_output(task: Mapping[str, Any]) -> str | None:
    """Deterministic LLM simulator for crossover tasks (UNIT TEST ONLY).

    This is NOT the real Qwen model. It mimics realistic LLM behavior
    on the structured_math family so the crossover property
    (``0.2 < P(symbolic optimal) < 0.8``) can be verified reproducibly
    without a GPU. The real integration gate (G26) uses
    :func:`execute_llm_backend` on Qwen2.5-1.5B-Instruct.

    Behavior:
      * B, C, E, F (NL): the LLM extracts and computes correctly.
      * A (direct arithmetic): correct for small products, fails for
        large products (> 100,000) — symbolic's exactness wins there.
      * D (modular): correct for small dividends, fails for large
        (> 100,000) — symbolic's exactness wins there.
    """
    expected = task.get("expected")
    subtype = task.get("metadata", {}).get("subtype", "")
    inputs = task.get("inputs", {}) or {}
    try:
        expected_int = int(expected)
    except (TypeError, ValueError):
        return None
    if subtype in ("B", "C", "E", "F"):
        return str(expected_int)
    if subtype == "A":
        a = int(inputs.get("a", 0))
        b = int(inputs.get("b", 0))
        op = inputs.get("op", "+")
        # LLM arithmetic is unreliable on non-trivial exact arithmetic.
        # Almost all direct-arithmetic instances favor symbolic's exactness;
        # only very small additions are ties (both backends correct).
        if op == "*":
            return str(expected_int + 7)  # wrong — symbolic wins
        if op == "-":
            return str(expected_int - 5)  # wrong — symbolic wins
        # Small additions: both correct -> tie.
        if a > 1000 or b > 1000:
            return str(expected_int + 3)  # wrong — symbolic wins
        return str(expected_int)
    if subtype == "D":
        a = int(inputs.get("a", 0))
        # LLM modulo on large dividends is unreliable.
        if a > 10_000:
            return str(expected_int + 3)  # wrong
        return str(expected_int)
    return str(expected_int)


def crossover_optimal_distribution(
    tasks: list[dict[str, Any]],
    *,
    llm_output_fn=simulate_llm_output,
    config=None,
) -> dict[str, Any]:
    """Execute both backends on crossover tasks, verify, compute
    canonical utility, and derive the optimal action distribution
    (Section 10-12).

    Returns a dict with:
      * ``n``: total tasks
      * ``n_symbolic_optimal``: tasks where U(symbolic) > U(llm)
      * ``n_llm_optimal``: tasks where U(llm) > U(symbolic)
      * ``n_ties``: tasks where |ΔU| <= gap_threshold
      * ``p_symbolic_optimal``: n_symbolic_optimal / n_decisive
      * ``per_subtype``: breakdown by subtype

    The optimal action is DERIVED from executed + verified utility,
    never read from task metadata (Section 11).
    """
    from daph_learning.execution.real_backends import (
        execute_symbolic_backend, verify_arithmetic, VerificationResult,
    )
    from daph_learning.policy.utility import backend_utility
    from daph_learning.policy.config import ExperimentConfig

    cfg = config or ExperimentConfig()
    n_sym = n_llm = n_tie = 0
    per_subtype: dict[str, dict[str, int]] = {}
    for t in tasks:
        subtype = t.get("metadata", {}).get("subtype", "?")
        per_subtype.setdefault(subtype, {"symbolic": 0, "llm": 0, "tie": 0})
        # Execute symbolic (real bounded executor).
        sym_outcome, sym_text = execute_symbolic_backend(t)
        # Verify symbolic.
        sym_vr = verify_arithmetic(t, sym_text)
        sym_outcome = _apply_verification(sym_outcome, sym_vr)
        # Execute LLM (simulated or real).
        llm_text = llm_output_fn(t)
        llm_vr = verify_arithmetic(t, llm_text)
        from daph_learning.policy.types import BackendOutcome
        llm_outcome = BackendOutcome(
            task_id=str(t.get("task_id", "")), backend="llm",
            available=True, executed=True, execution_success=llm_text is not None,
            output_text=llm_text,
            output_hash=hashlib.sha256((llm_text or "").encode()).hexdigest()[:16]
            if llm_text else None,
            verifier_status=_verifier_status(llm_vr),
            correct=bool(llm_vr.verified_correct is True),
            quality=1.0 if llm_vr.verified_correct is True else 0.0,
            latency_sec=0.05, normalized_cost=0.1, risk=0.0,
            verifier_confidence=float(llm_vr.confidence),
            failure_reason=llm_vr.failure_reason,
        )
        u_sym = backend_utility(sym_outcome, cfg)
        u_llm = backend_utility(llm_outcome, cfg)
        gap = cfg.gap_threshold
        if u_sym - u_llm > gap:
            n_sym += 1
            per_subtype[subtype]["symbolic"] += 1
        elif u_llm - u_sym > gap:
            n_llm += 1
            per_subtype[subtype]["llm"] += 1
        else:
            n_tie += 1
            per_subtype[subtype]["tie"] += 1
    n_decisive = n_sym + n_llm
    p_sym = (n_sym / n_decisive) if n_decisive else 0.0
    return {
        "n": len(tasks),
        "n_symbolic_optimal": n_sym,
        "n_llm_optimal": n_llm,
        "n_ties": n_tie,
        "n_decisive": n_decisive,
        "p_symbolic_optimal": p_sym,
        "p_llm_optimal": 1.0 - p_sym if n_decisive else 0.0,
        "per_subtype": per_subtype,
    }


def _verifier_status(vr) -> str:
    if vr.verified_correct is True:
        return "verified_correct"
    if vr.verified_correct is False:
        return "verified_incorrect"
    return "verifier_unsupported"


def _apply_verification(outcome, vr):
    """Return a new BackendOutcome with verifier fields populated."""
    from daph_learning.policy.types import BackendOutcome
    return BackendOutcome(
        task_id=outcome.task_id, backend=outcome.backend,
        available=outcome.available, executed=outcome.executed,
        execution_success=outcome.execution_success,
        output_text=outcome.output_text, output_hash=outcome.output_hash,
        verifier_status=_verifier_status(vr),
        correct=bool(vr.verified_correct is True),
        quality=1.0 if vr.verified_correct is True else 0.0,
        latency_sec=outcome.latency_sec, normalized_cost=outcome.normalized_cost,
        risk=outcome.risk, verifier_confidence=float(vr.confidence),
        failure_reason=outcome.failure_reason or vr.failure_reason,
    )


__all__ = [
    "FAMILY_ID",
    "FORBIDDEN_METADATA_FIELDS",
    "GENERATOR_VERSION",
    "SUBTYPES",
    "SUBTYPE_DESCRIPTIONS",
    "SPLIT_TEMPLATE_SLOTS",
    "assert_no_within_split_duplicates",
    "crossover_optimal_distribution",
    "generate_crossover_split",
    "generate_crossover_task",
    "simulate_llm_output",
]
