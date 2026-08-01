#!/usr/bin/env python3
"""Fit a shared DAPH adapter/task basis from independently trained LoRA teachers.

Teacher files are torch.save payloads with:
  {"modules": {"q_proj": {"a": Tensor, "b": Tensor, "alpha": float}, ...}}
A/B must use the single-repository LoRA contract [G,R,In] and [G,Out,R].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from daph_ext.geometry.adapter_basis import export_basis_payload, fit_joint_adapter_basis
from daph_ext.repobrain.repo2lora import LoRAFactors


def load_teacher(path: Path) -> dict[str, LoRAFactors]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = payload.get("modules") if isinstance(payload, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected non-empty 'modules' mapping")
    result = {}
    for name, item in raw.items():
        result[name] = LoRAFactors(a=item["a"], b=item["b"], alpha=float(item["alpha"]))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("teachers", nargs="+", type=Path)
    p.add_argument("--basis-size", type=int, default=16)
    p.add_argument("--basis-rank", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    teachers = [load_teacher(x) for x in args.teachers]
    basis, coefficients, diag = fit_joint_adapter_basis(
        teachers, basis_size=args.basis_size, basis_rank=args.basis_rank
    )
    payload = export_basis_payload(basis)
    payload["teacher_coefficients"] = coefficients
    payload["teacher_files"] = [str(x) for x in args.teachers]
    payload["diagnostics"] = {
        "basis_size": diag.basis_size,
        "teacher_count": diag.teacher_count,
        "explained_energy": diag.explained_energy,
        "cumulative_explained_energy": diag.cumulative_explained_energy,
        "block_relative_errors": diag.block_relative_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(payload["diagnostics"], indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
