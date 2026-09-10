"""Rewrite the golden set's tags down to the two that were asked for: source and type.

`type:` comes from the judged pass in `type_out/<model>/batch_NNN.json` — a per-row reading of the
request, replacing the old `risk:` tags which were the corpus name renamed (every casehold row was
high-stakes-legal, every pubmedqa row high-stakes-medical, and so on, so they carried no
information beyond `source:`).

Two of the twelve router features are not judged because they are exactly computable, and asking a
model to guess at them produced wrong labels in the smoke run:

  tool_gap      the request needs a capability `available_tools` does not offer
  context_load  the supplied context is large enough that finding the answer is itself work

Everything else that used to be a tag — the generator's own parameters (attack, evasiveness,
placement, payload, carrier, trust, pressure, intensity, ambiguity-style, citation-style,
tool-arm, transform-arm, donor), corpus metadata (meta, subcorpus, lang), origin, and the
label-provenance flag (judged) — moves into `notes`, where nothing can stratify on it.

Usage:  uv run python apply_types.py [--model opus] [--rows 10000] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = 25

JUDGED = ["high_stakes", "freshness", "ambiguity", "citation_need", "multi_step",
          "numerical", "user_pressure", "context_conflict", "evidence_gap", "prompt_injection"]
COMPUTED = ["tool_gap", "context_load"]
VALID = set(JUDGED) | set(COMPUTED) | {"none"}

# context_load fires where the router's own feature reaches half scale. features.py computes it as
# min(1.0, chars / 12_000) -- a ramp on characters, with no item count -- so the binary threshold
# is 6,000 characters. An earlier 4,000-or-4-items rule fired on 42% of rows, most of them pulled
# in by the item clause, which only reflects the 3-decoy trim rather than any real reading burden.
CONTEXT_LOAD_CHARS = 6000

# a request needs a capability when answering it means reaching outside the supplied context.
# numerical is excluded: arithmetic does not require a tool.
NEEDS_TOOL = {"evidence_gap", "freshness", "citation_need"}


def computed_types(case: dict) -> list[str]:
    """The two types that are facts about the request, not judgement calls."""
    found = []
    chars = sum(len(c.get("text", "")) for c in case.get("context", []))
    if chars >= CONTEXT_LOAD_CHARS:
        found.append("context_load")
    return found


def tool_gap(case: dict, judged: set[str]) -> bool:
    """True when the request must look something up and no tool can serve it.

    Derived rather than judged: it is a fact about `available_tools`, and in the smoke run the
    judge invented gaps on rows whose answer was sitting in the context.
    """
    if not (judged & NEEDS_TOOL):
        return False
    return not case.get("available_tools")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opus")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--source", default=os.path.join(DIR, "..", "routing_eval_golden.jsonl"))
    ap.add_argument("--out", default=os.path.join(DIR, "golden.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    order = json.load(open(os.path.join(DIR, "judge_order.json"), encoding="utf-8"))
    id_map = json.load(open(os.path.join(DIR, "id_map.json"), encoding="utf-8"))

    # verdict position -> case id. The bridge that has bitten before: verdicts are positions in
    # judge_order, and id_map is keyed by judge_input line number. id_map[order[pos]], never
    # id_map[pos].
    types_by_id: dict[str, list[str]] = {}
    missing_batches = []
    for b in range(args.rows // BATCH):
        path = os.path.join(DIR, "type_out", args.model, f"batch_{b:03d}.json")
        if not os.path.exists(path):
            missing_batches.append(b)
            continue
        for entry in json.load(open(path, encoding="utf-8")):
            pos = b * BATCH + (entry["request"] - 1)
            cid = id_map[str(order[pos])]
            bad = [t for t in entry["types"] if t not in VALID]
            if bad:
                raise ValueError(f"batch {b} request {entry['request']}: unknown types {bad}")
            types_by_id[cid] = entry["types"]

    if missing_batches:
        print(f"MISSING {len(missing_batches)} batches: {missing_batches[:20]}")
        if not args.dry_run:
            return 1

    cases = [json.loads(line) for line in open(args.source, encoding="utf-8") if line.strip()]
    print(f"source {len(cases)} rows | judged types for {len(types_by_id)}")

    stats = Counter()
    unmatched = 0
    out = []
    for case in cases:
        judged = types_by_id.get(case["id"])
        if judged is None:
            unmatched += 1
            continue

        judged_set = {t for t in judged if t != "none"}
        derived = set(computed_types(case))
        if tool_gap(case, judged_set):
            derived.add("tool_gap")

        final = sorted(judged_set | derived) or ["none"]

        source = next((t for t in case["tags"] if t.startswith("source:")), None)
        dropped = [t for t in case["tags"] if not t.startswith("source:")]

        case["tags"] = ([source] if source else []) + [f"type:{t}" for t in final]
        case["notes"] = f"{case['notes']} | dropped_tags: {' '.join(dropped)}"
        out.append(case)
        stats.update(final)
        stats["_rows"] += 1
        if len(final) > 1:
            stats["_multi"] += 1

    n = stats["_rows"]
    print(f"written {n} | unmatched {unmatched}")
    print(f"\n{'type':<18}{'rows':>7}{'%':>8}   ")
    for t in JUDGED + COMPUTED + ["none"]:
        if stats[t]:
            print(f"{t:<18}{stats[t]:>7}{100*stats[t]/n:>7.1f}%")
    print(f"\nrows with >1 type: {stats['_multi']} ({100*stats['_multi']/n:.0f}%)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    with open(args.out, "w", encoding="utf-8") as fh:
        for case in out:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
