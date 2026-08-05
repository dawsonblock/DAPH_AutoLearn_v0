from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from daph_ext.experiments.splits import ImmutableRepoSplits, RepositoryIdentity, validate_repository_lineage


def _repo_id(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("repo", "repository", "repo_id", "id", "name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError(f"cannot derive repository identifier from {item!r}")


def _deterministic_partition(items, *, seed: int, n_train: int, n_val: int, n_test: int) -> ImmutableRepoSplits:
    ids = [_repo_id(x).strip() for x in items]
    if len(set(ids)) != len(ids):
        raise ValueError("raw repository list contains duplicate identifiers")
    need = n_train + n_val + n_test
    if len(ids) < need:
        raise ValueError(f"need at least {need} repositories; got {len(ids)}")
    def score(repo_id: str) -> str:
        return hashlib.sha256(f"{seed}:{repo_id}".encode()).hexdigest()
    ordered = sorted(ids, key=lambda r: (score(r), r))
    # Minimum counts are allocated first; any surplus remains in train to avoid
    # shrinking the learning set while keeping val/test frozen.
    test = ordered[:n_test]
    val = ordered[n_test:n_test+n_val]
    train = ordered[n_test+n_val:]
    if len(train) < n_train:
        raise ValueError("insufficient repositories after validation/test allocation")
    return ImmutableRepoSplits.from_sequences(train, val, test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or canonicalize an immutable repository split manifest")
    parser.add_argument("input", help="JSON mapping with train/validation/test arrays, or a raw repository array")
    parser.add_argument("output")
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--min-validation", type=int, default=5)
    parser.add_argument("--min-test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lineage-input", help="optional JSON array of repo identity records: repo_id, content_hash, fork_root, family")
    args = parser.parse_args()
    if min(args.min_train, args.min_validation, args.min_test) <= 0:
        raise SystemExit("split minimums must be > 0")
    raw = json.loads(Path(args.input).read_text())
    if isinstance(raw, dict) and all(k in raw for k in ("train", "validation", "test")):
        splits = ImmutableRepoSplits.from_sequences(raw["train"], raw["validation"], raw["test"])
    elif isinstance(raw, list):
        splits = _deterministic_partition(raw, seed=args.seed, n_train=args.min_train,
                                          n_val=args.min_validation, n_test=args.min_test)
    else:
        raise SystemExit("input must be a split mapping or a raw repository array")
    splits.validate()
    for name, values, minimum in (("train", splits.train, args.min_train), ("validation", splits.validation, args.min_validation), ("test", splits.test, args.min_test)):
        if len(values) < minimum:
            raise SystemExit(f"{name} split requires at least {minimum} repositories; got {len(values)}")

    if args.lineage_input:
        lineage_raw=json.loads(Path(args.lineage_input).read_text())
        if not isinstance(lineage_raw,list):
            raise SystemExit("lineage input must be a JSON array")
        split_of={r:"train" for r in splits.train}|{r:"validation" for r in splits.validation}|{r:"test" for r in splits.test}
        identities=[]
        for item in lineage_raw:
            if not isinstance(item,dict) or not isinstance(item.get("repo_id"),str):
                raise SystemExit("invalid lineage identity record")
            rid=item["repo_id"].strip()
            if rid not in split_of:
                continue
            identities.append(RepositoryIdentity(repo_id=rid,split=split_of[rid],content_hash=item.get("content_hash"),fork_root=item.get("fork_root"),family=item.get("family")))
        covered={x.repo_id for x in identities}
        missing=set(split_of)-covered
        if missing:
            raise SystemExit(f"lineage input missing {len(missing)} split repositories")
        validate_repository_lineage(identities)

    digest = splits.write(args.output)
    print(digest)


if __name__ == "__main__":
    main()
