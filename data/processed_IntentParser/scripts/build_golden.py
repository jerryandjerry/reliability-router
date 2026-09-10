"""Build golden_smoke.jsonl from the judged smoke sample.

Rows where Opus and Sonnet agree become labelled EvalCases. Rows where they disagree are held
out to contested.json for Fable adjudication rather than being silently resolved.
"""
import glob
import json
import os
import re
from collections import Counter

from reliability_router.models import EvalCase

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = 25
CTX_KEYS = {"id", "text", "source", "trust", "metadata"}

sample = json.load(open(os.path.join(DIR, "smoke_rows.json")))
id_map = {int(k): v for k, v in json.load(open(os.path.join(DIR, "id_map.json"))).items()}

cases = {}
for name in ("golden_real.jsonl", "golden_synth.jsonl"):
    for line in open(os.path.join(DIR, name), encoding="utf-8"):
        r = json.loads(line)
        cases[r["id"]] = r


def load(model):
    out = {}
    for fp in glob.glob(os.path.join(DIR, "judge_out", model, "batch_*.json")):
        b = int(os.path.basename(fp).split("_")[1].split(".")[0])
        chunk = sample[b * BATCH:(b + 1) * BATCH]
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for v in data:
            req = v.get("request")
            if isinstance(req, int) and 1 <= req <= len(chunk):
                out[chunk[req - 1]] = {"route": v.get("route"), "why": v.get("why", "")}
    return out


opus, sonnet = load("opus"), load("sonnet")
both = sorted(set(opus) & set(sonnet))
print(f"opus {len(opus)} | sonnet {len(sonnet)} | overlap {len(both)}")


def stratum(rid):
    if rid.startswith("synth_"):
        return "synth:" + re.sub(r"_\d+$", "", rid[len("synth_"):])
    return "real:" + re.sub(r"_(train|test|validation|dev)_\d+$", "", rid)


agreed, contested = [], []
for row in both:
    (agreed if opus[row]["route"] == sonnet[row]["route"] else contested).append(row)

out_path = os.path.join(DIR, "golden_smoke.jsonl")
label_counts, strat_counts, bad = Counter(), Counter(), []
with open(out_path, "w", encoding="utf-8") as fh:
    for row in agreed:
        rid = id_map[row]
        src = cases[rid]
        case = {
            "id": rid,
            "query": src["query"],
            "expected_route": opus[row]["route"],
            "context": [{k: v for k, v in c.items() if k in CTX_KEYS} for c in src["context"]],
            "available_tools": src["available_tools"],
            "tags": [*src["tags"], "judged:unanimous"],
            "notes": f"{src['notes']} | opus={opus[row]['route']} sonnet={sonnet[row]['route']} "
                     f"| {opus[row]['why']}",
        }
        try:
            EvalCase.model_validate(case)
        except Exception as exc:
            bad.append((rid, str(exc)[:100]))
            continue
        fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        label_counts[case["expected_route"]] += 1
        strat_counts[stratum(rid)] += 1

json.dump([{"row": r, "id": id_map[r], "opus": opus[r]["route"], "opus_why": opus[r]["why"],
            "sonnet": sonnet[r]["route"], "sonnet_why": sonnet[r]["why"]} for r in contested],
          open(os.path.join(DIR, "contested.json"), "w"), ensure_ascii=False, indent=1)

print(f"\nagreed {len(agreed)} ({100*len(agreed)/len(both):.1f}%) | contested {len(contested)}")
print(f"EvalCase failures: {len(bad)}")
for b in bad[:3]:
    print("   ", b)
print(f"\nwrote {sum(label_counts.values())} -> {out_path}")
print("labels:", dict(label_counts))
print("\nper stratum:")
for k, c in sorted(strat_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:32} {c:>4}")
