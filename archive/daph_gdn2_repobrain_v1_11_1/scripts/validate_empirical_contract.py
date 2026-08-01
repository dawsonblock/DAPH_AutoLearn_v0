from __future__ import annotations

import argparse
import json
from pathlib import Path


def _nonempty_str(payload, field, errors):
    if not isinstance(payload.get(field), str) or not payload[field].strip():
        errors.append(f"{field} must be a non-empty string")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--require-hardware", action="store_true")
    args = p.parse_args()
    payload = json.loads(Path(args.path).read_text())
    errors: list[str] = []
    if payload.get("data_origin") != "empirical": errors.append("data_origin must be 'empirical'")
    for field in ("experiment_name", "checkpoint", "dataset_hash"):
        _nonempty_str(payload, field, errors)
    seed = payload.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool): errors.append("seed must be an integer")
    if args.require_hardware:
        hw = payload.get("hardware")
        if not isinstance(hw, dict):
            errors.append("hardware must be an object")
        else:
            for field in ("device", "device_name", "torch_version"):
                if not isinstance(hw.get(field), str) or not hw[field].strip(): errors.append(f"hardware.{field} must be non-empty")
            world_size = hw.get("world_size", 1)
            if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
                errors.append("hardware.world_size must be a positive integer")
            if hw.get("device", "").startswith("cuda") and not isinstance(hw.get("cuda_version"), str):
                errors.append("hardware.cuda_version must be present for CUDA runs")
    if errors:
        raise SystemExit("invalid empirical artifact: " + "; ".join(errors))
    print("empirical artifact contract OK")


if __name__ == "__main__":
    main()
