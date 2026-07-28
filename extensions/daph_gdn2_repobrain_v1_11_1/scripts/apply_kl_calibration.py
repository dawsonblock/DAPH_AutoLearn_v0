from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from daph_ext.config import DAPHIntegrationConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Apply validation-derived KL calibration to a production YAML template")
    p.add_argument("template")
    p.add_argument("calibration_json")
    p.add_argument("output")
    p.add_argument("--min-safe-count", type=int, default=30)
    p.add_argument("--max-ci-width", type=float, default=None)
    args = p.parse_args()
    raw = yaml.safe_load(Path(args.template).read_text())
    calibration = json.loads(Path(args.calibration_json).read_text())
    threshold = calibration.get("max_kl_drift")
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise SystemExit("calibration JSON must contain non-negative max_kl_drift")
    safe_count = calibration.get("safe_count", 0)
    if not isinstance(safe_count, int) or safe_count < args.min_safe_count:
        raise SystemExit(f"calibration has only {safe_count} safe+useful adapters; need >= {args.min_safe_count}")
    ci = calibration.get("bootstrap_ci_95")
    if args.max_ci_width is not None:
        if not (isinstance(ci, list) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci)):
            raise SystemExit("calibration missing bootstrap_ci_95")
        if float(ci[1]) - float(ci[0]) > args.max_ci_width:
            raise SystemExit("KL calibration CI is too wide for production promotion")
    raw.setdefault("controller", {})["max_kl_drift"] = float(threshold)
    raw["controller"]["require_calibrated_kl"] = True
    out = Path(args.output)
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    # Prove the generated config satisfies the strict production contract.
    DAPHIntegrationConfig.from_yaml(out)
    print(f"wrote calibrated production config: {out}")


if __name__ == "__main__":
    main()
