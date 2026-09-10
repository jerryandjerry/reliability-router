"""Phase 2: stratified 10k draw from fullset.jsonl -> golden_real.jsonl.

Two passes so the 1.5 GB corpus never sits in memory: pass 1 indexes line numbers by
stratum, pass 2 re-reads only the selected lines.
"""
import json
import os
import random
from collections import Counter, defaultdict

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(DIR, "fullset.jsonl")
OUT = os.path.join(DIR, "golden_real.jsonl")
random.seed(20260726)

TARGETS = {
    "squad": 1400, "hotpot_qa": 1400, "musique": 1100, "drop": 1100, "casehold": 1100,
    "streamingqa": 1100, "pubmedqa": 1000, "finqa": 800, "asqa": 600, "retrievalqa": 300,
    "crrag_conflict": 100,
}

def ctx_bucket(chars):
    for edge, name in ((500, "xs"), (2000, "s"), (5000, "m"), (12000, "l")):
        if chars < edge:
            return name
    return "xl"

# ---- pass 1: index line numbers by (source, stratum), dedupe on query ----
strata = defaultdict(lambda: defaultdict(list))
seen_queries = set()
dupes = 0
total = 0
with open(SRC, encoding="utf-8") as fh:
    for lineno, line in enumerate(fh):
        rec = json.loads(line)
        total += 1
        src = rec["source"]
        # Dedupe within a source only: the same question carrying different context is a
        # different routing input (and cross-source overlap is real, e.g. RealTimeQA feeds
        # both retrievalqa and crrag_conflict).
        key = (src, rec["query"].strip().lower())
        if key in seen_queries:
            dupes += 1
            continue
        seen_queries.add(key)
        # stratum = descriptive tags (minus source) + context-size bucket
        tags = [t for t in rec["tags"] if t != src]
        chars = sum(len(c["text"]) for c in rec["context"])
        stratum = "|".join(sorted(tags)) + f"#ctx:{ctx_bucket(chars)}"
        strata[src][stratum].append(lineno)

print(f"indexed {total} rows | unique queries {len(seen_queries)} | dropped {dupes} duplicates\n")

# ---- allocate per source across its strata, proportional with a floor, round-robin remainder ----
selected = set()
plan = {}
for src, target in TARGETS.items():
    groups = strata.get(src, {})
    if not groups:
        print(f"WARN {src}: no rows")
        continue
    avail = sum(len(v) for v in groups.values())
    take = min(target, avail)
    # shuffle within each stratum for reproducible randomness
    for v in groups.values():
        random.shuffle(v)
    chosen = []
    # round-robin across strata => maximum scenario spread, rare strata never starved
    keys = sorted(groups)
    random.shuffle(keys)
    idx = {k: 0 for k in keys}
    while len(chosen) < take:
        progressed = False
        for k in keys:
            if len(chosen) >= take:
                break
            i = idx[k]
            if i < len(groups[k]):
                chosen.append(groups[k][i])
                idx[k] = i + 1
                progressed = True
        if not progressed:
            break
    selected.update(chosen)
    plan[src] = (take, len(groups), avail)
    print(f"  {src:16} take={take:>5} from {len(groups):>4} strata (avail {avail})")

print(f"\nselected {len(selected)} rows")

# ---- pass 2: write the selected lines ----
kept = Counter()
qtypes = Counter()
ctxb = Counter()
with open(SRC, encoding="utf-8") as fh, open(OUT, "w", encoding="utf-8") as out:
    for lineno, line in enumerate(fh):
        if lineno not in selected:
            continue
        rec = json.loads(line)
        out.write(line if line.endswith("\n") else line + "\n")
        kept[rec["source"]] += 1
        chars = sum(len(c["text"]) for c in rec["context"])
        ctxb[ctx_bucket(chars)] += 1
        for t in rec["tags"]:
            if t.startswith(("single-hop", "multi-hop", "high-stakes", "numerical", "freshness",
                             "ambiguity", "conflict", "tool-dependent")):
                qtypes[t] += 1

print("\n=== golden_real.jsonl ===")
for s, c in kept.most_common():
    print(f"  {s:16} {c:>5}")
print(f"  {'TOTAL':16} {sum(kept.values()):>5}")
print("\nquestion types:", dict(qtypes.most_common()))
print("context-size spread:", dict(ctxb))
print(f"size: {os.path.getsize(OUT)/1e6:.1f} MB -> {OUT}")
