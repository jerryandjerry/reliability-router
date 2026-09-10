"""Chunk the contested rows for Fable adjudication.

contested.json holds positions in the judge order; render_batch.py --rows takes judge_input
line indices. order[position] bridges the two, exactly as in build_golden_10k.py.

Fable judges blind: it sees the same content-only rendering under the same policy, and never
sees what Opus or Sonnet said, so its vote breaks the tie instead of anchoring on one side.

Output: adjudication_chunks.json -- [{chunk, rows (positions), lines (judge_input indices)}]
"""
import json
import os

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK = 25

order = json.load(open(os.path.join(DIR, "judge_order.json")))
contested = json.load(open(os.path.join(DIR, "contested.json"), encoding="utf-8"))

positions = [c["row"] for c in contested]
print(f"contested rows: {len(positions)}")

chunks = []
for k in range(0, len(positions), CHUNK):
    rows = positions[k:k + CHUNK]
    chunks.append({
        "chunk": len(chunks),
        "rows": rows,
        "lines": [order[p] for p in rows],
    })

json.dump(chunks, open(os.path.join(DIR, "adjudication_chunks.json"), "w"), indent=1)
print(f"wrote {len(chunks)} chunks -> adjudication_chunks.json")
print(f"  sizes: {[len(c['rows']) for c in chunks][:5]}{' ...' if len(chunks) > 5 else ''} (last {len(chunks[-1]['rows'])})")

# the workflow only needs chunk + lines; keep the args payload small
print("\nargs payload:")
print(json.dumps({"model": "fable",
                  "chunks": [{"chunk": c["chunk"], "lines": c["lines"]} for c in chunks]}))
