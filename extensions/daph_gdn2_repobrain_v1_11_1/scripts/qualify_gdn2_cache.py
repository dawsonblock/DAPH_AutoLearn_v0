from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from daph_ext.config import GDN2Config
from daph_ext.mixers.gdn2_adapter import ExternalGDN2Adapter
from daph_ext.mixers.gdn2_reference import relative_l2


def _load_factory(spec: str | None):
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError("--cache-factory must be MODULE:FUNCTION")
    module, name = spec.split(":", 1)
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError("cache factory must be callable")
    return fn


def main() -> None:
    p = argparse.ArgumentParser(description="Qualify ExternalGDN2Adapter cache persistence and incremental decode")
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, required=True)
    p.add_argument("--mode", choices=("chunk", "fused_recurrent"), default="chunk")
    p.add_argument("--external-module", default="lit_gpt.gdn2")
    p.add_argument("--external-class", default="GatedDeltaNet2")
    p.add_argument("--cache-factory", help="Optional MODULE:FUNCTION returning an upstream-compatible empty cache")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    p.add_argument("--prefill-length", type=int, default=8)
    p.add_argument("--decode-steps", type=int, default=8, help="single value for backward compatibility")
    p.add_argument("--decode-grid", default="8,32,128,512", help="comma-separated decode lengths; max value drives run")
    p.add_argument("--full-parity-tol", type=float, default=2e-2)
    p.add_argument("--output")
    args = p.parse_args()
    decode_grid = sorted({int(x) for x in args.decode_grid.split(",") if x.strip()} | {args.decode_steps})
    if args.prefill_length <= 0 or not decode_grid or any(x <= 0 for x in decode_grid):
        raise SystemExit("prefill-length/decode lengths must be > 0")
    max_decode = max(decode_grid)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    cfg = GDN2Config(head_dim=args.head_dim, num_heads=args.num_heads, mode=args.mode,
                     external_module=args.external_module, external_class=args.external_class)
    adapter = ExternalGDN2Adapter(args.hidden_size, cfg, layer_idx=0).to(device=args.device, dtype=dtype).eval()
    factory = _load_factory(args.cache_factory)
    seq = torch.randn(1, args.prefill_length + max_decode, args.hidden_size, device=args.device, dtype=dtype)

    with torch.inference_mode():
        initial_cache = factory() if factory else None
        prefill = adapter(seq[:, :args.prefill_length], recurrent_state=initial_cache, use_cache=True)
        if prefill.recurrent_state is None:
            raise SystemExit("GDN2 cache qualification FAILED: no recurrent cache returned")
        pieces = [prefill.hidden_states]
        cache = prefill.recurrent_state
        cache_type = type(cache).__qualname__
        checkpoint_errors = {}
        for i in range(max_decode):
            out = adapter(seq[:, args.prefill_length+i:args.prefill_length+i+1], recurrent_state=cache, use_cache=True)
            if out.recurrent_state is None:
                raise SystemExit(f"GDN2 cache qualification FAILED at decode step {i}")
            if not torch.isfinite(out.hidden_states).all():
                raise SystemExit(f"GDN2 cache qualification FAILED: non-finite output at decode step {i}")
            cache = out.recurrent_state
            pieces.append(out.hidden_states)
            step_count = i + 1
            if step_count in decode_grid:
                inc_now = torch.cat(pieces, dim=1)
                fresh_now = factory() if factory else None
                full_now = adapter(seq[:, :args.prefill_length + step_count], recurrent_state=fresh_now, use_cache=True)
                checkpoint_errors[str(step_count)] = float(relative_l2(inc_now.float(), full_now.hidden_states.float()).item())
        incremental = torch.cat(pieces, dim=1)
        fresh = factory() if factory else None
        full = adapter(seq, recurrent_state=fresh, use_cache=True)
        if full.recurrent_state is None:
            raise SystemExit("GDN2 cache qualification FAILED: full forward returned no cache")
        if full.hidden_states.shape != incremental.shape:
            raise SystemExit("GDN2 cache qualification FAILED: incremental/full shape mismatch")
        err = float(relative_l2(incremental.float(), full.hidden_states.float()).item())
        passed = bool(torch.isfinite(incremental).all().item() and err <= args.full_parity_tol)

    payload = {
        "data_origin": "empirical", "experiment_name": "gdn2_cache_qualification",
        "mode": args.mode, "dtype": args.dtype, "prefill_length": args.prefill_length,
        "decode_steps": max_decode, "decode_grid": decode_grid, "cache_type": cache_type,
        "checkpoint_relative_l2": checkpoint_errors,
        "incremental_vs_full_relative_l2": err, "threshold": args.full_parity_tol,
        "passed": passed,
    }
    if args.output:
        outp=Path(args.output); outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
    if not passed:
        raise SystemExit(f"GDN2 cache qualification FAILED: relative L2 {err:.6g}")
    print("GDN2 cache qualification PASS")


if __name__ == "__main__":
    main()
