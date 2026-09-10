"""Phase 4 prep: build judge_input.jsonl containing ONLY what a real request carries.

A judge must see what a caller actually sends and nothing else:
    query, context text, context trust, available_tools.

Everything else is our bookkeeping and MUST NOT reach a judge. Earlier versions leaked the
answer on 100% of rows: the id itself named the corpus or the adversarial family
("pubmedqa_train_0", "synth_injection_02390"), and context.source named the corpus on another
27% ("pubmed:abstract", "court-opinion:excerpt"). Row identity now lives only in id_map.json,
which judges never read. context.source is dropped entirely -- the router never reads it either
(features.py uses only text and trust), so withholding it costs no fidelity.

Context is passed IN FULL: every item, every character (~41k tokens per 50-row batch).
"""
import json
import os

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(DIR, "judge_input.jsonl")
IDMAP = os.path.join(DIR, "id_map.json")

rows = []
id_map = {}
for name in ("golden_real.jsonl", "golden_synth.jsonl"):
    for line in open(os.path.join(DIR, name), encoding="utf-8"):
        r = json.loads(line)
        row_no = len(rows)
        id_map[row_no] = r["id"]
        rows.append({
            "row": row_no,
            "query": r["query"],
            "context": [{"trust": c["trust"], "text": c["text"]} for c in r["context"]],
            "available_tools": r["available_tools"],
        })

with open(OUT, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(IDMAP, "w", encoding="utf-8") as fh:
    json.dump(id_map, fh)

os.makedirs(os.path.join(DIR, "judge_out"), exist_ok=True)

# leak check: nothing in a judge-visible row may name a corpus or a transform family
BANNED = ("squad", "hotpot", "musique", "pubmedqa", "drop", "finqa", "casehold", "streamingqa",
          "asqa", "retrievalqa", "crrag", "synth", "wikipedia:", "pubmed:", "court-opinion",
          "financial-filing", "news:article", "injection", "context_removal", "context_shuffle",
          "tool_variant", "transform_wrap", "user_pressure")
leaks = 0
for r in rows:
    blob = json.dumps({k: v for k, v in r.items() if k != "context"}, ensure_ascii=False).lower()
    if any(b in blob for b in BANNED):
        leaks += 1
print(f"wrote {len(rows)} rows -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
print(f"id map -> {IDMAP} (judges never read this)")
print(f"metadata leaks outside context text: {leaks}")
sizes = [len(json.dumps(r, ensure_ascii=False)) for r in rows]
print(f"~{sum(sizes)/len(sizes)*50/4:.0f} tokens per 50-row batch (full context)")
