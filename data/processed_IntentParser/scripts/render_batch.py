"""Render one batch of requests as a judge sees them.

EVERY judge subagent runs this exact script — no agent slices, filters or formats anything
itself. Output is content only: the user's message, the context text and its trust score, and
the tools available. No ids, no source names, no dataset names, no sizes, no counts.

Requests are numbered 1..N within the batch purely so verdicts can be returned in order; the
pipeline maps that number back to the real row.

Usage:  python3 render_batch.py <batch_number>
        python3 render_batch.py --rows 12,349,17888     (arbitrary rows, same rendering)
"""
from __future__ import annotations  # judges invoke this as bare `python3`; keep it 3.9-safe

import json
import os
import sys

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = 25   # smaller batches: judges were skimming long context at 50/batch (~45k tokens)


def load(wanted: set[int] | None, start: int, end: int) -> list[dict]:
    rows = []
    with open(os.path.join(DIR, "judge_input.jsonl"), encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if wanted is not None:
                if i in wanted:
                    rows.append(json.loads(line))
                if len(rows) == len(wanted):
                    break
            else:
                if i < start:
                    continue
                if i >= end:
                    break
                rows.append(json.loads(line))
    return rows


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--rest":
        # the 18,000 rows not covered by the smoke sample
        b = int(sys.argv[2])
        rest = json.load(open(os.path.join(DIR, "remaining_rows.json")))
        chunk = rest[b * BATCH:(b + 1) * BATCH]
        rows = sorted(load(set(chunk), 0, 0), key=lambda r: chunk.index(r["row"]))
        label = f"rest batch {b}"
    elif len(sys.argv) == 3 and sys.argv[1] == "--smoke":
        # stratified smoke sample: batch b = smoke_rows[b*BATCH : (b+1)*BATCH]
        b = int(sys.argv[2])
        sample = json.load(open(os.path.join(DIR, "smoke_rows.json")))
        chunk = sample[b * BATCH:(b + 1) * BATCH]
        rows = sorted(load(set(chunk), 0, 0), key=lambda r: chunk.index(r["row"]))
        label = f"smoke batch {b}"
    elif len(sys.argv) == 3 and sys.argv[1] == "--rows":
        # keep the caller's order: request N must map back to the Nth row they asked for
        chunk = [int(x) for x in sys.argv[2].split(",") if x.strip()]
        rows = sorted(load(set(chunk), 0, 0), key=lambda r: chunk.index(r["row"]))
        label = f"rows {sys.argv[2][:40]}"
    elif len(sys.argv) == 2:
        # default: the stratified judging order over all 20,000 rows, so any prefix of the
        # batches is a representative cross-section rather than a corpus-ordered slice
        b = int(sys.argv[1])
        order = json.load(open(os.path.join(DIR, "judge_order.json")))
        chunk = order[b * BATCH:(b + 1) * BATCH]
        rows = sorted(load(set(chunk), 0, 0), key=lambda r: chunk.index(r["row"]))
        label = f"batch {b}"
    else:
        print("usage: render_batch.py <batch_number> | --rows a,b,c", file=sys.stderr)
        return 2

    if not rows:
        print(f"no rows for {label}", file=sys.stderr)
        return 1

    out = []
    for n, r in enumerate(rows, start=1):
        out.append(f"===== REQUEST {n} =====")
        out.append("user message:")
        out.append(r["query"])
        if r["context"]:
            out.append("")
            out.append("context already in the request:")
            for c in r["context"]:
                out.append(f"  --- (trust {c['trust']}) ---")
                out.append(f"  {c['text']}")
        else:
            out.append("")
            out.append("context already in the request: (none)")
        out.append("")
        tools = ", ".join(r["available_tools"]) if r["available_tools"] else "(none)"
        out.append(f"tools available: {tools}")
        out.append("")
    print("\n".join(out))
    print(f"===== END: {len(rows)} requests =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
