"""Build golden.jsonl from the 10k dual-judged prefix (batches 0-399 of the judge order).

Same shape as build_golden.py, but keyed on id_map.json rather than smoke_rows.json, so it
works for the full judge order instead of a hand-picked smoke sample.

Rows where Opus and Sonnet agree become labelled EvalCases tagged `judged:unanimous`.
Rows where they disagree go to contested.json for Fable adjudication rather than being
silently resolved; finalize_golden_10k.py folds those back in.
"""
import glob
import json
import os
import re
from collections import Counter

from reliability_router.models import EvalCase

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = 25
ROWS = 10000
CTX_KEYS = {"id", "text", "source", "trust", "metadata"}

id_map = {int(k): v for k, v in json.load(open(os.path.join(DIR, "id_map.json"))).items()}
# Verdicts are positional within the judge order; id_map is keyed by judge_input line index.
# order[position] bridges the two -- indexing id_map by the raw position labels the wrong case.
order = json.load(open(os.path.join(DIR, "judge_order.json")))

cases = {}
for name in ("golden_real.jsonl", "golden_synth.jsonl"):
    for line in open(os.path.join(DIR, name), encoding="utf-8"):
        r = json.loads(line)
        cases[r["id"]] = r


def load(model):
    out, bad = {}, []
    for fp in glob.glob(os.path.join(DIR, "judge_out", model, "batch_*.json")):
        b = int(os.path.basename(fp).split("_")[1].split(".")[0])
        if b * BATCH >= ROWS:
            continue
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as exc:
            bad.append((os.path.basename(fp), str(exc)[:60]))
            continue
        for v in data:
            req = v.get("request")
            if isinstance(req, int) and 1 <= req <= BATCH:
                out[b * BATCH + req - 1] = {"route": v.get("route"), "why": v.get("why", "")}
    return out, bad


opus, bad_o = load("opus")
sonnet, bad_s = load("sonnet")
print(f"opus {len(opus)}/{ROWS} (unreadable {len(bad_o)}) | sonnet {len(sonnet)}/{ROWS} (unreadable {len(bad_s)})")
for b in (bad_o + bad_s)[:5]:
    print("   bad:", b)

both = sorted(set(opus) & set(sonnet))
missing = sorted(set(range(ROWS)) - set(both))
print(f"dual-judged {len(both)}/{ROWS} | missing {len(missing)}")
if missing:
    json.dump(missing, open(os.path.join(DIR, "missing_rows.json"), "w"))
    print(f"  missing rows -> missing_rows.json (batches: "
          f"{sorted({r // BATCH for r in missing})[:12]}{' ...' if len(set(r // BATCH for r in missing)) > 12 else ''})")


def stratum(rid):
    if rid.startswith("synth_"):
        return "synth:" + re.sub(r"_\d+$", "", rid[len("synth_"):])
    return "real:" + re.sub(r"_(train|test|validation|dev)_\d+$", "", rid)


agreed, contested = [], []
for row in both:
    (agreed if opus[row]["route"] == sonnet[row]["route"] else contested).append(row)

out_path = os.path.join(DIR, "golden.jsonl")
label_counts, strat_counts, bad = Counter(), Counter(), []
with open(out_path, "w", encoding="utf-8") as fh:
    for row in agreed:
        rid = id_map[order[row]]
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

json.dump([{"row": r, "id": id_map[order[r]], "opus": opus[r]["route"], "opus_why": opus[r]["why"],
            "sonnet": sonnet[r]["route"], "sonnet_why": sonnet[r]["why"]} for r in contested],
          open(os.path.join(DIR, "contested.json"), "w"), ensure_ascii=False, indent=1)

print(f"\nagreed {len(agreed)} ({100*len(agreed)/max(1,len(both)):.1f}%) | contested {len(contested)}")
print(f"EvalCase failures: {len(bad)}")
for b in bad[:3]:
    print("   ", b)
print(f"\nwrote {sum(label_counts.values())} -> {out_path}")
print("labels:", dict(label_counts))
print("\nper stratum:")
for k, c in sorted(strat_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:32} {c:>4}")
