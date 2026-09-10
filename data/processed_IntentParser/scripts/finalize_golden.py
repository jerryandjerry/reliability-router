"""Fold Fable's adjudications into golden_smoke.jsonl so no judged row is dropped.

Unanimous Opus/Sonnet rows are already in the file. Contested rows get Fable's verdict and a
`judged:boundary` tag, so downstream analysis can report with and without them.
"""
import glob
import json
import os
import re
from collections import Counter

from reliability_router.models import EvalCase

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX_KEYS = {"id", "text", "source", "trust", "metadata"}

contested = json.load(open(os.path.join(DIR, "contested.json"), encoding="utf-8"))
chunks = json.load(open(os.path.join(DIR, "adjudication_chunks.json"), encoding="utf-8"))
id_map = {int(k): v for k, v in json.load(open(os.path.join(DIR, "id_map.json"))).items()}

cases = {}
for name in ("golden_real.jsonl", "golden_synth.jsonl"):
    for line in open(os.path.join(DIR, name), encoding="utf-8"):
        r = json.loads(line)
        cases[r["id"]] = r

# chunk index -> ordered row numbers, so request N maps back to a row
row_of = {}
for c in chunks:
    for i, row in enumerate(c["rows"], start=1):
        row_of[(c["chunk"], i)] = row

verdict = {}
for fp in sorted(glob.glob(os.path.join(DIR, "judge_out", "fable", "chunk_*.json"))):
    ci = int(os.path.basename(fp).split("_")[1].split(".")[0])
    try:
        data = json.load(open(fp, encoding="utf-8"))
    except Exception as exc:
        print(f"  unreadable {os.path.basename(fp)}: {str(exc)[:60]}")
        continue
    for v in data:
        row = row_of.get((ci, v.get("request")))
        if row is not None:
            verdict[row] = v

by_row = {c["row"]: c for c in contested}
print(f"contested {len(contested)} | adjudicated {len(verdict)} | missing {len(contested)-len(verdict)}")

existing = [json.loads(l) for l in open(os.path.join(DIR, "golden_smoke.jsonl"), encoding="utf-8")]
print(f"unanimous rows already written: {len(existing)}")

added, bad = 0, []
sided = Counter()
with open(os.path.join(DIR, "golden_smoke.jsonl"), "a", encoding="utf-8") as fh:
    for row, v in verdict.items():
        c = by_row.get(row)
        if not c:
            continue
        src = cases[id_map[row]]
        case = {
            "id": src["id"],
            "query": src["query"],
            "expected_route": v["route"],
            "context": [{k: x for k, x in item.items() if k in CTX_KEYS} for item in src["context"]],
            "available_tools": src["available_tools"],
            "tags": [*src["tags"], "judged:boundary"],
            "notes": f"{src['notes']} | opus={c['opus']} sonnet={c['sonnet']} "
                     f"fable={v['route']} | {v.get('why','')}",
        }
        try:
            EvalCase.model_validate(case)
        except Exception as exc:
            bad.append((src["id"], str(exc)[:90]))
            continue
        fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        added += 1
        sided["opus" if v["route"] == c["opus"] else "sonnet"] += 1

print(f"\nadded {added} boundary rows | EvalCase failures {len(bad)}")
for b in bad[:3]:
    print("   ", b)
print(f"fable sided with: {dict(sided)}")

total = len(existing) + added
labels = Counter()
for line in open(os.path.join(DIR, "golden_smoke.jsonl"), encoding="utf-8"):
    labels[json.loads(line)["expected_route"]] += 1
print(f"\ngolden_smoke.jsonl now {total} rows | {dict(labels)}")
