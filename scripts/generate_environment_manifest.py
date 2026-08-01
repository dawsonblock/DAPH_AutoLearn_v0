#!/usr/bin/env python
"""Section 8.3 — Generate environment_manifest.json for an experiment bundle.

Stores Python version, OS, architecture, PyTorch/transformers versions,
CUDA version, device type, model identifier, random seeds, etc.

Usage::

    python scripts/generate_environment_manifest.py --bundle artifacts/gate_a_qualified/daph_gate_a_real_005_requal
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _safe_version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "not_installed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate environment_manifest.json")
    parser.add_argument("--bundle", required=True, help="Path to experiment bundle directory")
    parser.add_argument("--config", default=None, help="Optional config YAML for model info")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    if not bundle.exists():
        print(f"ERROR: bundle not found: {bundle}", file=sys.stderr)
        return 1

    # Load model info from config if provided.
    model_info = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        model_cfg = config.get("model", {})
        stats_cfg = config.get("statistics", {})
        model_info = {
            "model_id": model_cfg.get("model_id", ""),
            "model_revision": model_cfg.get("revision", ""),
            "tokenizer_revision": model_cfg.get("tokenizer_revision", model_cfg.get("revision", "")),
            "dtype": model_cfg.get("dtype", ""),
            "do_sample": model_cfg.get("do_sample", False),
            "temperature": model_cfg.get("temperature", 0.0),
            "top_p": model_cfg.get("top_p", 1.0),
            "max_new_tokens": model_cfg.get("max_new_tokens", 256),
            "bootstrap_seed": stats_cfg.get("bootstrap_seed", 20260731),
        }

    manifest = {
        "manifest_type": "environment_manifest",
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.release(),
        "architecture": platform.machine(),
        "packages": {
            "torch": _safe_version("torch"),
            "transformers": _safe_version("transformers"),
            "tokenizers": _safe_version("tokenizers"),
            "numpy": _safe_version("numpy"),
            "scikit-learn": _safe_version("scikit-learn"),
            "accelerate": _safe_version("accelerate"),
            "safetensors": _safe_version("safetensors"),
        },
        "cuda": {
            "available": False,
            "version": None,
            "device": None,
        },
        "deterministic_algorithms": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **model_info,
    }

    # Check CUDA availability.
    try:
        import torch
        manifest["cuda"]["available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["cuda"]["version"] = torch.version.cuda
            manifest["cuda"]["device"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        manifest["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except Exception:
        pass

    out_path = bundle / "environment_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
