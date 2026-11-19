#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Calculate Adapter Amortization Efficiency (AAE).")
    p.add_argument("--base", type=float, required=True)
    p.add_argument("--direct-lora", type=float, required=True)
    p.add_argument("--repobrain", type=float, required=True)
    args = p.parse_args()
    denom = args.direct_lora - args.base
    if abs(denom) < 1e-12:
        raise SystemExit("AAE undefined: direct LoRA provides no measurable gain over base")
    aae = (args.repobrain - args.base) / denom
    print(f"AAE={aae:.6f}")


if __name__ == "__main__":
    main()
