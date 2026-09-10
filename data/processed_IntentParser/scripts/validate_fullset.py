"""Validate fullset.jsonl: EvalCase-convertibility, tag histogram, context stats."""
import json
import os
from collections import Counter

from reliability_router.models import EvalCase

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fullset.jsonl")

n = 0
tags = Counter()
sources = Counter()
qtypes = Counter()
ctx_items = []
ctx_chars = []
trusts = Counter()
bad = []
no_answer = 0

QTYPE_PREFIXES = ("single-hop", "multi-hop", "high-stakes", "numerical", "freshness",
                  "ambiguity", "conflict", "tool-dependent")

with open(PATH, encoding="utf-8") as fh:
    for i, line in enumerate(fh, 1):
        rec = json.loads(line)
        n += 1
        sources[rec["source"]] += 1
        for t in rec["tags"]:
            tags[t] += 1
            if t.startswith(QTYPE_PREFIXES):
                qtypes[t] += 1
        ctx_items.append(len(rec["context"]))
        ctx_chars.append(sum(len(c["text"]) for c in rec["context"]))
        for c in rec["context"]:
            trusts[c["trust"]] += 1
        if rec.get("answer") in (None, [], ""):
            no_answer += 1
        # EvalCase convertibility: strip our extra fields, add a placeholder label
        if i % 5000 == 1 or i <= 20:
            probe = {k: v for k, v in rec.items() if k in ("id", "query", "context", "available_tools", "tags", "notes")}
            probe["expected_route"] = "QUICK"
            probe["context"] = [
                {k: v for k, v in c.items() if k in ("id", "text", "source", "trust", "metadata")}
                for c in probe["context"]
            ]
            try:
                EvalCase.model_validate(probe)
            except Exception as exc:
                bad.append((rec["id"], str(exc)[:150]))

print(f"rows: {n}")
print(f"EvalCase probe failures: {len(bad)}")
for b in bad[:5]:
    print("   ", b)
print(f"\nrows with no answer captured: {no_answer} ({100*no_answer/n:.1f}%)")
print(f"\ncontext items/row : min={min(ctx_items)} median={sorted(ctx_items)[n//2]} max={max(ctx_items)} mean={sum(ctx_items)/n:.1f}")
print(f"context chars/row : min={min(ctx_chars)} median={sorted(ctx_chars)[n//2]} max={max(ctx_chars)} mean={int(sum(ctx_chars)/n)}")
print(f"total context chars: {sum(ctx_chars)/1e9:.2f} B")
print(f"\ntrust values used: {dict(sorted(trusts.items()))}")
print("\n=== question_type tags ===")
for t, c in qtypes.most_common():
    print(f"  {t:28} {c:>7}")
print("\n=== per-source ===")
for s, c in sources.most_common():
    print(f"  {s:16} {c:>7}")
