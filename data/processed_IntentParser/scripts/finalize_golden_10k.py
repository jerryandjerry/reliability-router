"""Fold Fable's adjudications into golden.jsonl so no judged row is dropped.

Unanimous Opus/Sonnet rows are already written by build_golden_10k.py. Contested rows get
Fable's blind verdict and a `judged:boundary` tag, so downstream analysis can report with and
without them.

adjudication_chunks.json maps (chunk, request) back to a position in the judge order;
order[position] then gives the judge_input line that id_map is keyed by.
"""
import glob
import json
import os
from collections import Counter

from reliability_router.models import EvalCase

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX_KEYS = {"id", "text", "source", "trust", "metadata"}

contested = json.load(open(os.path.join(DIR, "contested.json"), encoding="utf-8"))
chunks = json.load(open(os.path.join(DIR, "adjudication_chunks.json"), encoding="utf-8"))
id_map = {int(k): v for k, v in json.load(open(os.path.join(DIR, "id_map.json"))).items()}
order = json.load(open(os.path.join(DIR, "judge_order.json")))

cases = {}
for name in ("golden_real.jsonl", "golden_synth.jsonl"):
    for line in open(os.path.join(DIR, name), encoding="utf-8"):
        r = json.loads(line)
        cases[r["id"]] = r

# (chunk, request) -> position in the judge order
row_of = {}
for c in chunks:
    for i, row in enumerate(c["rows"], start=1):
        row_of[(c["chunk"], i)] = row

verdict, unreadable = {}, []
for fp in sorted(glob.glob(os.path.join(DIR, "judge_out", "fable", "chunk_*.json"))):
    ci = int(os.path.basename(fp).split("_")[1].split(".")[0])
    try:
        data = json.load(open(fp, encoding="utf-8"))
    except Exception as exc:
        unreadable.append((os.path.basename(fp), str(exc)[:60]))
        continue
    for v in data:
        row = row_of.get((ci, v.get("request")))
        if row is not None:
            verdict[row] = v

by_row = {c["row"]: c for c in contested}
covered = [r for r in by_row if r in verdict]
print(f"contested {len(contested)} | fable verdicts {len(verdict)} | "
      f"contested covered {len(covered)} | missing {len(contested)-len(covered)}")
for u in unreadable[:5]:
    print("   unreadable:", u)
stale = [r for r in verdict if r not in by_row]
if stale:
    print(f"   {len(stale)} fable verdicts for rows no longer contested -> ignored")

existing = [json.loads(l) for l in open(os.path.join(DIR, "golden.jsonl"), encoding="utf-8")]
print(f"unanimous rows already written: {len(existing)}")

added, bad = 0, []
sided, labels_added = Counter(), Counter()
with open(os.path.join(DIR, "golden.jsonl"), "a", encoding="utf-8") as fh:
    for row in sorted(covered):
        v, c = verdict[row], by_row[row]
        src = cases[id_map[order[row]]]
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
        labels_added[v["route"]] += 1
        sided["opus" if v["route"] == c["opus"] else "sonnet"] += 1

print(f"\nadded {added} boundary rows {dict(labels_added)} | EvalCase failures {len(bad)}")
for b in bad[:3]:
    print("   ", b)
print(f"fable sided with: {dict(sided)}")

labels = Counter()
for line in open(os.path.join(DIR, "golden.jsonl"), encoding="utf-8"):
    labels[json.loads(line)["expected_route"]] += 1
print(f"\ngolden.jsonl now {sum(labels.values())} rows | {dict(labels)}")
