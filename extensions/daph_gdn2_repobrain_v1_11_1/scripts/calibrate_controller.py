from __future__ import annotations

import argparse
import json
from pathlib import Path

from daph_ext.calibration.controller import CalibrationRecord, calibrate_kl_envelope


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate KL envelope from validation-only empirical adapter records")
    p.add_argument("input", help="JSON array of validation records")
    p.add_argument("output")
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--max-generic-regression", type=float, default=0.01)
    p.add_argument("--max-execution-regression", type=float, default=0.01)
    p.add_argument("--min-safe-count", type=int, default=30, help="Production default; use 5 only for pilot/debug")
    p.add_argument("--min-repository-benefit", type=float, default=0.0)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    raw = json.loads(Path(args.input).read_text())
    if not isinstance(raw, list):
        raise SystemExit("input must be a JSON array")
    records = [CalibrationRecord(**item) for item in raw]
    result = calibrate_kl_envelope(records, percentile=args.percentile,
        max_generic_regression=args.max_generic_regression,
        max_execution_regression=args.max_execution_regression,
        min_repository_benefit=args.min_repository_benefit,
        min_safe_count=args.min_safe_count, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    payload = {"max_kl_drift": result.threshold, "safe_count": result.safe_count,
               "total_count": result.total_count, "safe_fraction": result.safe_fraction,
               "percentile": result.percentile, "bootstrap_ci_95": [result.ci_low, result.ci_high],
               "repository_count": result.repository_count,
               "min_repository_benefit": result.min_repository_benefit}
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
