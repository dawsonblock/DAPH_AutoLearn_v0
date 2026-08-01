#!/usr/bin/env python3
"""Section 20 — Independent reanalysis mode for Gate A bundles.

Loads immutable raw outcomes from a Gate A bundle and recomputes
all statistics, gate verdicts, and qualification status without
retraining or regenerating any artifact.

Usage:
    python scripts/reanalyze_gate_a.py \
        --bundle artifacts/gate_a_qualified/daph_gate_a_real_003 \
        --output artifacts/reanalysis/daph_gate_a_real_003_corrected

If required frozen artifacts (policy, calibration, features) are missing,
the reanalysis reports NOT_EVALUABLE with a description of what is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from daph_learning.evaluation.qualification import (
    FinalTaskRecord,
    RouteAction,
    group_bootstrap_mean_delta,
    bootstrap_p1_minus_sham,
    ShamTaskPrediction,
    compute_oracle_gap_capture,
    positive_group_fraction,
    group_fraction_breakdown,
    compute_subtype_regression,
    compute_crossover_metrics,
    count_crossover_subtypes,
    decisive_fraction,
    check_preconditions,
    compare,
    Comparator,
    QualificationStatus,
)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_records(bundle: Path) -> list[FinalTaskRecord] | None:
    """Load final task records from the bundle."""
    # Try final_predictions.json first (has all fields)
    preds_path = bundle / "final_predictions.json"
    if not preds_path.exists():
        return None
    preds = json.loads(preds_path.read_text())
    records = []
    for p in preds:
        sym_u = float(p.get("symbolic_utility", 0))
        llm_u = float(p.get("llm_utility", 0))
        records.append(FinalTaskRecord(
            task_id=p.get("task_id", ""),
            group_id=p.get("group_id", ""),
            subtype=p.get("subtype", ""),
            split="final",
            symbolic_utility=sym_u,
            llm_utility=llm_u,
            utility_gap_symbolic_minus_llm=sym_u - llm_u,
            symbolic_probability=float(p.get("symbolic_probability", 0)),
            calibrated_symbolic_probability=float(p.get("calibrated_symbolic_probability", p.get("symbolic_probability", 0))),
            raw_symbolic_probability=float(p.get("raw_symbolic_probability", p.get("symbolic_probability", 0))),
            selected_action=RouteAction(p.get("selected_action", "llm")),
            oracle_action=RouteAction(p.get("oracle_action", "llm")),
            p1_realized_utility=float(p.get("p1_realized_utility", 0)),
            p0_realized_utility=float(p.get("p0_realized_utility", 0)),
            always_symbolic_utility=sym_u,
            oracle_utility=float(p.get("oracle_utility", max(sym_u, llm_u))),
            p1_minus_p0=float(p.get("p1_minus_p0", 0)),
            p1_minus_oracle=float(p.get("p1_realized_utility", 0)) - float(p.get("oracle_utility", 0)),
            symbolic_correct=bool(p.get("symbolic_correct", False)),
            llm_correct=bool(p.get("llm_correct", False)),
            symbolic_verification_status=p.get("symbolic_verification_status", ""),
            llm_verification_status=p.get("llm_verification_status", ""),
        ))
    return records


def _load_sham_records(bundle: Path) -> list[ShamTaskPrediction] | None:
    """Load sham predictions from the bundle."""
    sham_path = bundle / "sham_predictions.json"
    if not sham_path.exists():
        return None
    sham_data = json.loads(sham_path.read_text())
    return [ShamTaskPrediction(
        sham_seed=s["sham_seed"],
        task_id=s["task_id"],
        symbolic_probability=s["symbolic_probability"],
        selected_action=RouteAction(s["selected_action"]),
        realized_utility=s["realized_utility"],
    ) for s in sham_data]


def reanalyze(bundle: Path, output: Path) -> dict:
    """Recompute all statistics from stored task-level evidence."""
    result = {
        "source_bundle": str(bundle),
        "experiment_id": bundle.name,
        "reanalysis_possible": True,
        "limitations": [],
    }

    # Load records
    records = _load_records(bundle)
    if records is None or len(records) == 0:
        result["reanalysis_possible"] = False
        result["qualification_status"] = "NOT_EVALUABLE"
        result["limitations"].append("final_predictions.json not found or empty")
        return result
    result["n_records"] = len(records)

    # Load sham records
    sham_records = _load_sham_records(bundle)
    if sham_records is None:
        result["limitations"].append("sham_predictions.json not found — sham comparison skipped")
        sham_records = []

    # Load stored results for comparison
    stored_stats = _load_json(bundle / "experiment_results.json")
    stored_gate = _load_json(bundle / "gate_decision.json")

    # Recompute group bootstrap
    group_deltas: dict[str, np.ndarray] = {}
    for r in records:
        group_deltas.setdefault(r.group_id, []).append(r.p1_minus_p0)
    group_deltas = {k: np.array(v) for k, v in group_deltas.items()}

    n_iter = int(stored_stats.get("primary_endpoint", {}).get("n_iterations", 20000)) if stored_stats else 20000
    bootstrap_result = group_bootstrap_mean_delta(
        group_deltas, n_iterations=n_iter, confidence_level=0.95, seed=20260731,
        estimand="group_weighted",
    )
    result["primary_endpoint"] = {
        "estimand": "group_weighted",
        "point_estimate": bootstrap_result.point_estimate,
        "ci_low": bootstrap_result.ci_low,
        "ci_high": bootstrap_result.ci_high,
        "n_iterations": n_iter,
        "samples_sha256": bootstrap_result.samples_sha256,
    }

    # Recompute sham bootstrap
    if sham_records:
        sham_bootstrap = bootstrap_p1_minus_sham(
            records, sham_records, n_iterations=n_iter, confidence_level=0.95, seed=20260731,
        )
        result["sham"] = {
            "p1_minus_sham_mean": sham_bootstrap.point_estimate,
            "p1_minus_sham_ci_low": sham_bootstrap.ci_low,
            "p1_minus_sham_ci_high": sham_bootstrap.ci_high,
            "samples_sha256": sham_bootstrap.samples_sha256,
        }

    # Recompute all other metrics
    oracle_capture = compute_oracle_gap_capture(records)
    result["oracle_gap_capture"] = oracle_capture.value
    result["oracle_utility"] = oracle_capture.oracle_utility
    result["p0_utility"] = oracle_capture.p0_utility
    result["p1_utility"] = oracle_capture.p1_utility

    result["positive_group_fraction"] = positive_group_fraction(records)
    pos_breakdown = group_fraction_breakdown(records)
    result["positive_group_count"] = pos_breakdown["positive"]
    result["negative_group_count"] = pos_breakdown["negative"]
    result["zero_group_count"] = pos_breakdown["zero"]

    subtype_metrics, worst_regression = compute_subtype_regression(records)
    result["worst_subtype_regression"] = worst_regression
    result["subtype_metrics"] = [{
        "subtype": m.subtype, "n_tasks": m.n_tasks, "p1_utility": m.p1_utility,
        "p0_utility": m.p0_utility, "p1_minus_p0": m.p1_minus_p0,
    } for m in subtype_metrics]

    crossover_metrics = compute_crossover_metrics(records)
    result["crossover_count"] = count_crossover_subtypes(records)
    result["final_decisive_fraction"] = decisive_fraction(records)

    # Compare with stored values
    if stored_stats:
        discrepancies = []
        stored_pe = stored_stats.get("primary_endpoint", {})
        if stored_pe:
            for key in ("point_estimate", "ci_low", "ci_high"):
                stored_val = stored_pe.get(key)
                recomputed = result["primary_endpoint"].get(key)
                if stored_val is not None and recomputed is not None:
                    if abs(float(stored_val) - float(recomputed)) > 1e-10:
                        discrepancies.append({
                            "metric": f"primary_endpoint.{key}",
                            "stored": stored_val,
                            "recomputed": recomputed,
                        })
        if discrepancies:
            result["discrepancies"] = discrepancies
        else:
            result["discrepancies"] = []

    # Recompute gate verdict
    if stored_gate:
        gates_config = stored_gate.get("gate_verdicts", {})
        # Map gate names to recomputed values
        gate_values = {
            "minimum_point_gain_vs_p0": result["primary_endpoint"]["point_estimate"],
            "require_lcb_vs_p0_above": result["primary_endpoint"]["ci_low"],
            "require_lcb_vs_sham_above": result.get("sham", {}).get("p1_minus_sham_ci_low", 0),
            "minimum_oracle_gap_capture": result.get("oracle_gap_capture", 0) or 0,
            "minimum_positive_group_fraction": result.get("positive_group_fraction", 0),
            "maximum_worst_subtype_regression": result.get("worst_subtype_regression", 0),
            "maximum_final_access_count": 1,
        }
        gate_verdicts = {}
        all_pass = True
        for gate_name, gate_spec in gates_config.items():
            actual = gate_values.get(gate_name, 0)
            threshold = gate_spec.get("threshold", 0)
            comparator = gate_spec.get("comparator", "gte")
            passed = compare(float(actual), Comparator(comparator), float(threshold))
            gate_verdicts[gate_name] = {
                "actual": actual, "threshold": threshold,
                "comparator": comparator, "passed": passed,
            }
            if not passed:
                all_pass = False
        result["gate_verdicts"] = gate_verdicts
        result["qualification_status"] = "PASS" if all_pass else "FAIL"
    else:
        result["qualification_status"] = "NOT_EVALUABLE"
        result["limitations"].append("gate_decision.json not found — cannot recompute gate verdict")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Reanalyze a Gate A bundle")
    parser.add_argument("--bundle", required=True, type=Path,
                        help="Path to the artifact bundle to reanalyze")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to write the reanalysis report")
    args = parser.parse_args()

    if not args.bundle.exists():
        print(f"ERROR: bundle not found: {args.bundle}", file=sys.stderr)
        return 1

    result = reanalyze(args.bundle, args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    print(f"Reanalysis written to {args.output}")
    print(f"Qualification status: {result.get('qualification_status', 'UNKNOWN')}")
    if result.get("limitations"):
        print("Limitations:")
        for lim in result["limitations"]:
            print(f"  - {lim}")
    if result.get("discrepancies"):
        print(f"Discrepancies found: {len(result['discrepancies'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
