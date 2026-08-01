from __future__ import annotations

import argparse
import gc
import importlib
import json
import time
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from daph_ext.mixers.gdn2_reference import gdn2_recurrence_reference, relative_l2

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
DEFAULT_THRESHOLDS = {"fp32": 1e-5, "bf16": 5e-3, "fp16": 1e-2}
DEFAULT_REGIMES = ("normal", "no_decay", "extreme_decay", "near_zero_write", "high_erase")


def load_kernel(spec: str) -> Callable[..., tuple[Tensor, Tensor]]:
    if ":" not in spec:
        raise ValueError("--kernel must be MODULE:FUNCTION")
    module_name, function_name = spec.split(":", 1)
    fn = getattr(importlib.import_module(module_name), function_name)
    if not callable(fn):
        raise TypeError("kernel adapter must be callable")
    return fn


def make_master_case(batch: int, seq: int, heads: int, key_dim: int, value_dim: int, device: str, regime: str):
    q = torch.randn(batch, seq, heads, key_dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq, heads, value_dim, device=device, dtype=torch.float32)
    erase = torch.sigmoid(torch.randn_like(k))
    write = torch.sigmoid(torch.randn_like(v))
    if regime == "no_decay":
        log_decay = torch.zeros_like(k)
    elif regime == "extreme_decay":
        log_decay = torch.full_like(k, -20.0)
    else:
        log_decay = -torch.nn.functional.softplus(torch.randn_like(k))
    if regime == "near_zero_write":
        write = torch.full_like(write, 1e-5)
    elif regime == "high_erase":
        erase = torch.full_like(erase, 0.999)
    state = torch.randn(batch, heads, key_dim, value_dim, device=device, dtype=torch.float32) * 0.01
    return q, k, v, erase, write, log_decay, state


def _validate_kernel_output(y: Tensor, s: Tensor, ref_y: Tensor, ref_s: Tensor) -> None:
    if not isinstance(y, Tensor) or not isinstance(s, Tensor):
        raise TypeError("kernel_adapter must return (Tensor outputs, Tensor final_state)")
    if y.shape != ref_y.shape:
        raise ValueError(f"kernel output shape {tuple(y.shape)} != reference {tuple(ref_y.shape)}")
    if s.shape != ref_s.shape:
        raise ValueError(f"kernel state shape {tuple(s.shape)} != reference {tuple(ref_s.shape)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Hardware parity harness for an external GDN2 low-level kernel adapter")
    p.add_argument("--kernel", required=True, help="MODULE:FUNCTION accepting q,k,v,erase,write,log_decay,initial_state")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtypes", default="fp32,bf16,fp16")
    p.add_argument("--lengths", default="1,16,64,256,1024,4096,16384,32768")
    p.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full-parity-max-length", type=int, default=4096)
    args = p.parse_args()
    if args.batch <= 0 or args.heads <= 0 or args.key_dim <= 0 or args.value_dim <= 0:
        raise SystemExit("batch/heads/key-dim/value-dim must be > 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(args.seed)
    kernel = load_kernel(args.kernel)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    if not lengths or any(x <= 0 for x in lengths):
        raise SystemExit("all sequence lengths must be > 0")
    dtype_names = [x.strip() for x in args.dtypes.split(",") if x.strip()]
    if any(x not in DTYPES for x in dtype_names):
        raise SystemExit(f"unsupported dtype list: {dtype_names}")
    regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]
    if any(r not in DEFAULT_REGIMES for r in regimes):
        raise SystemExit(f"unsupported regimes: {regimes}")

    results = []
    overall_pass = True
    device_name = torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else str(args.device)
    with torch.inference_mode():
        for dn in dtype_names:
            dtype = DTYPES[dn]
            threshold = DEFAULT_THRESHOLDS[dn]
            for seq in lengths:
                for regime in regimes:
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(args.device)
                    master = make_master_case(args.batch, seq, args.heads, args.key_dim, args.value_dim, args.device, regime)
                    q, k, v, e, w, g, s = master
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    t0 = time.perf_counter()
                    ref_y, ref_s = gdn2_recurrence_reference(k, v, e, w, g, s, q=q)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    reference_seconds = time.perf_counter() - t0
                    low = tuple(x.to(dtype) for x in master)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    t0 = time.perf_counter()
                    ker_y, ker_s = kernel(*low)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    kernel_seconds = time.perf_counter() - t0
                    _validate_kernel_output(ker_y, ker_s, ref_y, ref_s)
                    out_err = float(relative_l2(ker_y.float(), ref_y).item())
                    state_err = float(relative_l2(ker_s.float(), ref_s).item())
                    finite = bool(torch.isfinite(ker_y).all().item() and torch.isfinite(ker_s).all().item())
                    passed = finite and out_err <= threshold and state_err <= threshold
                    overall_pass &= passed
                    peak = int(torch.cuda.max_memory_allocated(args.device)) if args.device.startswith("cuda") else 0
                    results.append({
                        "dtype": dn, "seq_len": seq, "regime": regime,
                        "qualification_tier": "full_parity" if seq <= args.full_parity_max_length else "long_context_stress",
                        "relative_output_l2": out_err, "relative_state_l2": state_err,
                        "threshold": threshold, "finite": finite, "passed": passed,
                        "reference_seconds": reference_seconds, "kernel_seconds": kernel_seconds,
                        "peak_cuda_bytes": peak,
                    })
                    del master, q, k, v, e, w, g, s, low, ref_y, ref_s, ker_y, ker_s
                    gc.collect()
    payload = {
        "data_origin": "empirical",
        "experiment_name": "gdn2_hardware_parity",
        "device": args.device,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "kernel": args.kernel,
        "seed": args.seed,
        "passed": overall_pass,
        "results": results,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if not overall_pass:
        raise SystemExit("GDN2 hardware qualification FAILED")
    print("GDN2 hardware qualification PASS")


if __name__ == "__main__":
    main()
