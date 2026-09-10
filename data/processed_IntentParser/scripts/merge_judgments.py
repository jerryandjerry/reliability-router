"""Phase 4 merge: judge_out/batch_*.json -> judge_raw.jsonl + golden.jsonl (strict EvalCase).

golden.jsonl must load through reliability_router.eval.runner.load_jsonl, so ContextItem gets
stripped to its allowed fields and our bookkeeping keys (answer/meta/pair_id/source) are dropped.
"""
import glob
import json
import os
from collections import Counter

from reliability_router.models import EvalCase

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_RAW = os.path.join(DIR, "judge_raw.jsonl")
OUT_GOLD = os.path.join(DIR, "golden.jsonl")

CTX_KEYS = {"id", "text", "source", "trust", "metadata"}

# ---- collect judgments ----
verdicts = {}
dupes = 0
bad_files = []
for fp in sorted(glob.glob(os.path.join(DIR, "judge_out", "batch_*.json"))):
    try:
        data = json.load(open(fp, encoding="utf-8"))
    except Exception as exc:
        bad_files.append((os.path.basename(fp), str(exc)[:80]))
        continue
    for v in data:
        rid = v.get("id")
        if rid in verdicts:
            dupes += 1
            continue
        verdicts[rid] = v

print(f"batch files: {len(glob.glob(os.path.join(DIR, 'judge_out', 'batch_*.json')))}"
      f" | verdicts {len(verdicts)} | duplicate ids {dupes} | unreadable {len(bad_files)}")
for b in bad_files[:5]:
    print("   bad:", b)

# ---- join back onto the source rows ----
rows = []
for name in ("golden_real.jsonl", "golden_synth.jsonl"):
    for line in open(os.path.join(DIR, name), encoding="utf-8"):
        rows.append(json.loads(line))

missing = [r["id"] for r in rows if r["id"] not in verdicts]
print(f"rows {len(rows)} | missing verdicts {len(missing)}")
if missing:
    print("   first missing:", missing[:5])
    with open(os.path.join(DIR, "judge_missing.txt"), "w") as fh:
        fh.write("\n".join(missing))

route_counts, reason_counts, conf_bins = Counter(), Counter(), Counter()
by_origin = Counter()
n_written = 0
bad_cases = []

with open(OUT_RAW, "w", encoding="utf-8") as raw, open(OUT_GOLD, "w", encoding="utf-8") as gold:
    for r in rows:
        v = verdicts.get(r["id"])
        if not v:
            continue
        route = v["route"]
        reason = v.get("primary_reason", "unknown")
        conf = float(v.get("confidence", 0.5))

        raw.write(json.dumps({"id": r["id"], "route": route, "primary_reason": reason,
                              "confidence": conf, "origin": r.get("meta", {}).get("origin", "real"),
                              "family": r.get("meta", {}).get("family"),
                              "pair_id": r.get("pair_id")}, ensure_ascii=False) + "\n")

        case = {
            "id": r["id"],
            "query": r["query"],
            "expected_route": route,
            "context": [{k: v2 for k, v2 in c.items() if k in CTX_KEYS} for c in r["context"]],
            "available_tools": r["available_tools"],
            "tags": [*r["tags"], f"judge_reason:{reason}",
                     "confidence:low" if conf < 0.7 else "confidence:high"],
            "notes": f"{r['notes']} | judge={route} reason={reason} conf={conf:.2f}",
        }
        try:
            EvalCase.model_validate(case)
        except Exception as exc:
            if len(bad_cases) < 5:
                bad_cases.append((r["id"], str(exc)[:120]))
            continue
        gold.write(json.dumps(case, ensure_ascii=False) + "\n")
        n_written += 1
        route_counts[route] += 1
        reason_counts[reason] += 1
        conf_bins[round(conf, 1)] += 1
        by_origin[(r.get("meta", {}).get("origin", "real"), route)] += 1

print(f"\nEvalCase validation failures: {len(bad_cases)}")
for b in bad_cases:
    print("   ", b)

print(f"\nwrote {n_written} -> {OUT_GOLD} ({os.path.getsize(OUT_GOLD)/1e6:.1f} MB)")
print(f"wrote {len(verdicts)} -> {OUT_RAW}")
print("\nroute:", dict(route_counts))
print("by origin:", {f"{k[0]}/{k[1]}": v for k, v in sorted(by_origin.items())})
print("\nprimary_reason:")
for k, c in reason_counts.most_common():
    print(f"  {k:26} {c:>6}")
low = sum(v for k, v in conf_bins.items() if k < 0.7)
print(f"\nlow-confidence (<0.7): {low} ({100*low/max(1,n_written):.1f}%)  <- Pass-2 candidates")
