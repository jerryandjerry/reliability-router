"""Assign each of the 10k real rows a Phase-3 transform family; dump the ambiguity subset for LLM rewriting."""
import glob
import json
import os
import random
from collections import Counter

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
random.seed(31415)

ALLOC = [
    ("injection", 2500), ("context_removal", 1500), ("context_shuffle", 1200),
    ("conflict", 1200), ("tool_variant", 800), ("transform_wrap", 800),
    ("user_pressure", 700), ("ambiguity", 700), ("citation", 600),
]
assert sum(n for _, n in ALLOC) == 10000

rows = [json.loads(line) for line in open(os.path.join(DIR, "golden_real.jsonl"), encoding="utf-8")]
print(f"loaded {len(rows)} real rows")

# Families that need context must land on rows that have some; conflict wants >=2 items ideally.
order = list(range(len(rows)))
random.shuffle(order)

assignment = {}

# The ambiguity family needs a per-row LLM rewrite, which cannot be regenerated for free. If the
# sample shifts (e.g. context trimming moves rows between size strata), assign ambiguity to the
# rows whose rewrites we already hold, so none are wasted; the shortfall is dumped for rewriting.
have_rewrite = set()
_amb_dir = os.path.join(SC, "ambiguity")
for _fp in glob.glob(os.path.join(_amb_dir, "amb_out_*.json")):
    for _e in json.load(open(_fp, encoding="utf-8")):
        have_rewrite.add(_e["id"])
amb_target = dict(ALLOC)["ambiguity"]
reused = 0
for i in order:
    if reused >= amb_target:
        break
    rid = rows[i]["id"]
    if rid in have_rewrite:
        assignment[rid] = "ambiguity"
        reused += 1
print(f"ambiguity: reusing {reused}/{amb_target} existing rewrites")

cursor = 0
for family, n in ALLOC:
    if family == "ambiguity":
        n -= reused
    picked = 0
    while picked < n and cursor < len(order):
        i = order[cursor]
        cursor += 1
        rec = rows[i]
        if rec["id"] in assignment:
            continue
        if family in ("context_removal", "context_shuffle", "conflict") and not rec["context"]:
            continue
        assignment[rec["id"]] = family
        picked += 1
    if picked < n:
        print(f"WARN {family}: only {picked}/{n}")

print("\nfamily assignment:", dict(Counter(assignment.values())))
with open(os.path.join(SC, "family_assignment.json"), "w") as f:
    json.dump(assignment, f)

# dump the ambiguity subset for the LLM rewrite agents
amb = [r for r in rows if assignment.get(r["id"]) == "ambiguity" and r["id"] not in have_rewrite]
payload = [{"id": r["id"], "query": r["query"], "source": r["source"]} for r in amb]
CHUNKS = 10
size = (len(payload) + CHUNKS - 1) // CHUNKS
for c in range(CHUNKS):
    chunk = payload[c * size:(c + 1) * size]
    if not chunk:
        continue
    os.makedirs(os.path.join(SC, "ambiguity"), exist_ok=True)
    with open(os.path.join(SC, "ambiguity", f"amb_in_{c}.json"), "w") as f:
        json.dump(chunk, f, ensure_ascii=False, indent=1)
print(f"dumped {len(payload)} ambiguity rows into {CHUNKS} chunks (~{size} each)")
