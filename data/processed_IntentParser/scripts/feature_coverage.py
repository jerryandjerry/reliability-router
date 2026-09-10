"""Measure which of the 12 features actually fire on the real fullset data."""
import json
import os
import random
from collections import defaultdict

from reliability_router.features import FeatureExtractor
from reliability_router.models import ContextItem, RouterRequest

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fullset.jsonl")
PER_SOURCE = 300
random.seed(7)

buckets = defaultdict(list)
with open(PATH, encoding="utf-8") as fh:
    for line in fh:
        rec = json.loads(line)
        b = buckets[rec["source"]]
        if len(b) < PER_SOURCE:
            b.append(rec)

ex = FeatureExtractor()
FEATURES = ["high_stakes", "freshness", "citation_need", "multi_step", "numerical", "user_pressure",
            "ambiguity", "prompt_injection", "context_conflict", "context_load", "tool_gap", "evidence_gap"]

overall = defaultdict(int)
per_source = defaultdict(lambda: defaultdict(int))
totals = defaultdict(int)

for src, rows in buckets.items():
    for rec in rows:
        req = RouterRequest(
            query=rec["query"],
            context=[ContextItem(id=c["id"], text=c["text"], source=c["source"], trust=c["trust"])
                     for c in rec["context"]],
            available_tools=rec["available_tools"],
        )
        f = ex.extract(req)
        totals[src] += 1
        for name in FEATURES:
            v = getattr(f, name)
            if v > 0:
                overall[name] += 1
                per_source[src][name] += 1

n = sum(totals.values())
print(f"sampled {n} rows across {len(buckets)} sources ({PER_SOURCE}/source cap)\n")
print("=== FEATURE ACTIVATION (fraction of rows where feature > 0) ===")
for name in FEATURES:
    pct = 100 * overall[name] / n
    bar = "#" * int(pct / 2)
    flag = "  <-- NEVER FIRES" if overall[name] == 0 else ("  <-- rare" if pct < 2 else "")
    print(f"  {name:18} {pct:5.1f}%  {bar}{flag}")

print("\n=== per-source activation % (blank = 0) ===")
hdr = "source".ljust(16) + "".join(f[:7].rjust(8) for f in FEATURES)
print(hdr)
for src in sorted(totals):
    row = src.ljust(16)
    for name in FEATURES:
        p = 100 * per_source[src][name] / totals[src]
        row += (f"{p:7.0f}%" if p > 0 else "       -").rjust(8)
    print(row)
