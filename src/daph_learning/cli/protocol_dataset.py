"""Build the v0.3.8 template-disjoint benchmark and provenance bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from daph_learning.data.benchmark import generate_protocol_splits
from daph_learning.data.integrity import detect_leakage, split_family


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DAPH v0.3.8 train/dev/calibration/final-test data with "
            "disjoint prompt-template families and a leakage report."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-size", type=int, default=2500)
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--calibration-size", type=int, default=500)
    parser.add_argument("--final-test-size", type=int, default=1000)
    args = parser.parse_args()

    sizes = {
        "train": args.train_size,
        "dev": args.dev_size,
        "calibration": args.calibration_size,
        "final_test": args.final_test_size,
    }
    splits = generate_protocol_splits(sizes, seed=args.seed)
    report = detect_leakage(splits)
    if not report.is_clean:
        raise RuntimeError("generated benchmark failed its own leakage audit")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, rows in splits.items():
        path = output_dir / f"{name}.jsonl"
        digest = _write_jsonl(path, rows)
        files[name] = {
            "path": path.name,
            "sha256": digest,
            "num_tasks": len(rows),
            "categories": sorted(
                {str(row["metadata"]["category"]) for row in rows}
            ),
            "split_families": sorted({split_family(row) for row in rows}),
        }

    leakage_path = output_dir / "leakage_report.json"
    leakage_path.write_text(
        json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "daph.dataset_bundle.v1",
        "daph_version": "0.3.8",
        "seed": args.seed,
        "files": files,
        "leakage_report": {
            "path": leakage_path.name,
            "sha256": hashlib.sha256(leakage_path.read_bytes()).hexdigest(),
            "is_clean": True,
        },
        "claim_boundary": (
            "Synthetic template-disjoint benchmark; not broad real-world "
            "out-of-distribution qualification."
        ),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
