"""Dataset fingerprints, family-disjoint splitting, and leakage detection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import random
import re
import unicodedata
from typing import Any, Mapping, Sequence


_WS_RE = re.compile(r"\s+")
_NON_SEMANTIC_RE = re.compile(r"[^\w+\-*/%]+", re.UNICODE)
_INTEGER_RE = re.compile(r"(?<![\w.])[+-]?\d+(?![\w.])")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def normalize_prompt(prompt: str) -> str:
    """Normalize case, Unicode, punctuation, and whitespace, preserving values."""
    text = unicodedata.normalize("NFKC", str(prompt)).casefold()
    text = _NON_SEMANTIC_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_template(prompt: str) -> str:
    """Normalize a prompt and replace integer parameters with ``<int>``."""
    normalized = normalize_prompt(prompt)
    return _INTEGER_RE.sub("<int>", normalized)


def task_fingerprint(task: Mapping[str, Any]) -> str:
    """Stable identity for task semantics, independent of ``task_id``."""
    payload = {
        "specification": normalize_prompt(str(task.get("specification", ""))),
        "inputs": task.get("inputs") or {},
        "capability_ids": sorted(map(str, task.get("capability_ids") or [])),
        "expected": task.get("expected"),
        "oracles": {
            key: task.get(key)
            for key in (
                "capability_oracle",
                "accuracy_oracle",
                "utility_oracle",
                "route_label",
            )
            if key in task
        },
    }
    return _sha256_text(canonical_json(payload))


def split_family(task: Mapping[str, Any]) -> str:
    """Resolve the indivisible family used by split construction.

    New benchmark rows must carry ``metadata.split_family``.  The fallbacks
    keep the scanner useful on legacy datasets while making the limitation
    visible in the returned family string.
    """
    metadata = task.get("metadata") or {}
    if isinstance(metadata, Mapping):
        for key in ("split_family", "generator_family", "template_id"):
            value = metadata.get(key)
            if value:
                return str(value)
        category = str(metadata.get("category", "unknown"))
    else:
        category = "unknown"
    template = normalize_template(str(task.get("specification", "")))
    return f"legacy:{category}:{_sha256_text(template)[:16]}"


@dataclass(frozen=True)
class Overlap:
    split_a: str
    split_b: str
    kind: str
    count: int
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_a": self.split_a,
            "split_b": self.split_b,
            "kind": self.kind,
            "count": self.count,
            "examples": list(self.examples),
        }


@dataclass
class LeakageReport:
    split_sizes: dict[str, int]
    cross_split: list[Overlap] = field(default_factory=list)
    within_split_duplicates: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return not self.cross_split and all(
            all(count == 0 for count in counts.values())
            for counts in self.within_split_duplicates.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "daph.leakage.v1",
            "is_clean": self.is_clean,
            "split_sizes": dict(self.split_sizes),
            "cross_split": [item.to_dict() for item in self.cross_split],
            "within_split_duplicates": self.within_split_duplicates,
        }


def _duplicate_count(values: Sequence[str]) -> int:
    return len(values) - len(set(values))


def detect_leakage(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_examples: int = 10,
) -> LeakageReport:
    """Scan exact, normalized, semantic, family, and ID overlap.

    Any family overlap is a protocol failure. Categories are intentionally not
    treated as families: every split should cover every benchmark category.
    """
    if len(splits) < 2:
        raise ValueError("leakage detection requires at least two splits")
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")

    indexes: dict[str, dict[str, list[str]]] = {}
    within: dict[str, dict[str, int]] = {}
    for split_name, rows_seq in splits.items():
        rows = list(rows_seq)
        ids = [str(row.get("task_id", "")) for row in rows]
        if any(not value for value in ids):
            raise ValueError(f"split {split_name!r} contains a row without task_id")
        prompts = [str(row.get("specification", "")) for row in rows]
        normalized = [normalize_prompt(value) for value in prompts]
        fingerprints = [task_fingerprint(row) for row in rows]
        families = [split_family(row) for row in rows]
        indexes[split_name] = {
            "task_id": ids,
            "exact_prompt": prompts,
            "normalized_prompt": normalized,
            "task_fingerprint": fingerprints,
            "split_family": families,
        }
        within[split_name] = {
            "task_id": _duplicate_count(ids),
            "exact_prompt": _duplicate_count(prompts),
            "normalized_prompt": _duplicate_count(normalized),
            "task_fingerprint": _duplicate_count(fingerprints),
        }

    overlaps: list[Overlap] = []
    split_names = list(splits)
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1:]:
            left = indexes[left_name]
            right = indexes[right_name]
            for kind in (
                "task_id",
                "exact_prompt",
                "normalized_prompt",
                "task_fingerprint",
                "split_family",
            ):
                shared = sorted(set(left[kind]) & set(right[kind]))
                if shared:
                    overlaps.append(
                        Overlap(
                            split_a=left_name,
                            split_b=right_name,
                            kind=kind,
                            count=len(shared),
                            examples=tuple(shared[:max_examples]),
                        )
                    )

    return LeakageReport(
        split_sizes={name: len(rows) for name, rows in splits.items()},
        cross_split=overlaps,
        within_split_duplicates=within,
    )


def assert_no_leakage(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> LeakageReport:
    report = detect_leakage(splits)
    if not report.is_clean:
        summary = ", ".join(
            f"{item.split_a}/{item.split_b}:{item.kind}={item.count}"
            for item in report.cross_split
        )
        within = {
            split: counts
            for split, counts in report.within_split_duplicates.items()
            if any(counts.values())
        }
        raise ValueError(
            "dataset leakage detected"
            + (f": {summary}" if summary else "")
            + (f"; within={within}" if within else "")
        )
    return report


def family_aware_split(
    tasks: Sequence[Mapping[str, Any]],
    ratios: Mapping[str, float],
    *,
    seed: int,
    stratify_field: str = "category",
) -> dict[str, list[dict[str, Any]]]:
    """Assign indivisible template families to stratified splits.

    Counts are approximate because a family is never fragmented.  The input
    must provide at least one family per non-empty target split in every
    stratum.
    """
    if not tasks:
        raise ValueError("tasks must be non-empty")
    if not ratios or any(float(value) <= 0 for value in ratios.values()):
        raise ValueError("ratios must contain only positive values")
    total_ratio = sum(float(value) for value in ratios.values())
    normalized_ratios = {
        name: float(value) / total_ratio
        for name, value in ratios.items()
    }
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in ratios}

    strata: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for raw in tasks:
        task = dict(raw)
        metadata = task.get("metadata") or {}
        stratum = (
            str(metadata.get(stratify_field, "unknown"))
            if isinstance(metadata, Mapping)
            else "unknown"
        )
        strata[stratum][split_family(task)].append(task)

    rng = random.Random(seed)
    for stratum, families in sorted(strata.items()):
        family_items = list(families.items())
        if len(family_items) < len(output):
            raise ValueError(
                f"stratum {stratum!r} has {len(family_items)} families for "
                f"{len(output)} requested splits"
            )
        rng.shuffle(family_items)
        # Largest groups first makes the greedy size target more stable.
        family_items.sort(key=lambda item: len(item[1]), reverse=True)
        stratum_size = sum(len(rows) for _, rows in family_items)
        targets = {
            name: stratum_size * ratio
            for name, ratio in normalized_ratios.items()
        }
        assigned = {name: 0 for name in output}

        # Seed every split with one family so small strata remain covered.
        split_order = sorted(output, key=lambda name: normalized_ratios[name], reverse=True)
        for split_name, (_family, rows) in zip(split_order, family_items[: len(output)]):
            output[split_name].extend(rows)
            assigned[split_name] += len(rows)

        for _family, rows in family_items[len(output):]:
            split_name = max(
                output,
                key=lambda name: targets[name] - assigned[name],
            )
            output[split_name].extend(rows)
            assigned[split_name] += len(rows)

    for rows in output.values():
        rng.shuffle(rows)
    assert_no_leakage(output)
    return output


__all__ = [
    "LeakageReport",
    "Overlap",
    "assert_no_leakage",
    "canonical_json",
    "detect_leakage",
    "family_aware_split",
    "normalize_prompt",
    "normalize_template",
    "split_family",
    "task_fingerprint",
]
