"""Split the golden set into 1,000 rows for evaluation and 9,000 for training.

The evaluation set is drawn to look like the whole set: rows are grouped by what the request is
and what the answer should be, then every group contributes the same 10% share. An earlier
version took one row per group regardless of group size, which left the eval set at 65.7% DEEP
against a population of 55.9% and over-sampled whichever adversarial families happened to carry
the most sub-tags.

There is no dev split. Models train on all 9,000 with fixed hyperparameters, so nothing selects a
checkpoint by peeking at held-out data.

Usage:  uv run python data/intent_parser_splits/build_train_dev_split.py [--eval-size 1000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "routing_eval_golden.jsonl"
EVAL_OUT = ROOT / "data" / "routing_eval_golden_1k.jsonl"
SPLIT_DIR = ROOT / "data" / "intent_parser_splits"
TRAIN_OUT = SPLIT_DIR / "train.jsonl"
MANIFEST = SPLIT_DIR / "split_manifest.json"


def group_key(case: dict) -> tuple[str, ...]:
    """What the request is, and what the answer should be.

    Deliberately excludes source: a router never sees which corpus a question came from, so
    balancing on it would balance our bookkeeping rather than the request. Dropping it also makes
    groups larger, which means fewer of them round down to zero rows.
    """
    types = tuple(sorted(t for t in case["tags"] if t.startswith("type:")))
    return (case["expected_route"], *types)


def stable_order(case: dict) -> str:
    """Deterministic shuffle within a group, independent of file order."""
    return hashlib.sha256(case["id"].encode("utf-8")).hexdigest()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-size", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    total = len(cases)

    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for case in cases:
        groups[group_key(case)].append(case)

    # proportional allocation: each group gives its own share, largest remainders take the rest
    share = args.eval_size / total
    quota = {k: int(len(v) * share) for k, v in groups.items()}
    remainder = sorted(groups, key=lambda k: (len(groups[k]) * share - quota[k], k), reverse=True)
    index = 0
    while sum(quota.values()) < args.eval_size and index < len(remainder) * 4:
        key = remainder[index % len(remainder)]
        if quota[key] < len(groups[key]):
            quota[key] += 1
        index += 1

    eval_ids: set[str] = set()
    for key, members in groups.items():
        for case in sorted(members, key=stable_order)[: quota[key]]:
            eval_ids.add(case["id"])

    eval_set = [c for c in cases if c["id"] in eval_ids]
    train_set = [c for c in cases if c["id"] not in eval_ids]

    if eval_ids & {c["id"] for c in train_set}:
        raise SystemExit("train and eval overlap")
    if len(eval_set) + len(train_set) != total:
        raise SystemExit("rows lost during the split")

    def deep_pct(rows: list[dict]) -> float:
        return 100 * sum(1 for r in rows if r["expected_route"] == "DEEP") / len(rows)

    print(f"source {total} rows | {len(groups)} groups")
    print(f"  eval   {len(eval_set):>5} rows | DEEP {deep_pct(eval_set):5.1f}%")
    print(f"  train  {len(train_set):>5} rows | DEEP {deep_pct(train_set):5.1f}%")
    print(f"  all    {total:>5} rows | DEEP {deep_pct(cases):5.1f}%")

    base = Counter(t for c in cases for t in c["tags"] if t.startswith("type:"))
    got = Counter(t for c in eval_set for t in c["tags"] if t.startswith("type:"))
    worst = max((abs(100 * got[t] / len(eval_set) - 100 * base[t] / total), t) for t in base)
    print(f"  worst type drift in eval: {worst[0]:.1f} points (on {worst[1]})")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in ((EVAL_OUT, eval_set), (TRAIN_OUT, train_set)):
        with path.open("w", encoding="utf-8") as handle:
            for case in rows:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    MANIFEST.write_text(
        json.dumps(
            {
                "version": "v2",
                "allocation": "proportional, 10% of every (route, type-signature) group",
                "source": "data/routing_eval_golden.jsonl",
                "heldout": "data/routing_eval_golden_1k.jsonl",
                "train": "data/intent_parser_splits/train.jsonl",
                "source_sha256": sha256_of(SOURCE),
                "heldout_sha256": sha256_of(EVAL_OUT),
                "train_sha256": sha256_of(TRAIN_OUT),
                "source_cases": total,
                "heldout_cases": len(eval_set),
                "train_cases": len(train_set),
                "dev_cases": 0,
                "groups": len(groups),
                "overlap": {"train_heldout": 0},
                "train_route_counts": dict(Counter(c["expected_route"] for c in train_set)),
                "heldout_route_counts": dict(Counter(c["expected_route"] for c in eval_set)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nwrote {EVAL_OUT}\nwrote {TRAIN_OUT}\nwrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
