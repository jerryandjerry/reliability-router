"""Generator integrity: did each transform STRUCTURALLY apply? (independent of the router)"""
import json
import os
from collections import defaultdict

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

real = {}
for line in open(os.path.join(DIR, "golden_real.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    real[r["id"]] = r
synth = [json.loads(line) for line in open(os.path.join(DIR, "golden_synth.jsonl"), encoding="utf-8")]

checks = defaultdict(lambda: {"n": 0, "ok": 0, "fail": []})

TRANSFORM_VERBS = ["Translate to French", "Summarize", "Rewrite this more formally",
                   "Paraphrase", "Proofread", "翻译成中文", "总结"]

for s in synth:
    fam = s["meta"]["family"]
    twin = real[s["pair_id"]]
    c = checks[fam]
    c["n"] += 1
    ok = False

    if fam == "context_removal":
        ok = len(s["context"]) == 0 and len(twin["context"]) > 0
    elif fam == "context_shuffle":
        ok = (len(s["context"]) > 0
              and {i["text"] for i in s["context"]} != {i["text"] for i in twin["context"]})
    elif fam == "conflict":
        ok = (len(s["context"]) == len(twin["context"]) + 1
              and any(i["id"] == "ctx_conflict" for i in s["context"]))
    elif fam == "injection":
        if "placement:direct" in s["tags"]:
            ok = s["query"] != twin["query"] and len(s["query"]) > len(twin["query"])
        else:
            ok = any(i["id"] == "ctx_injected" for i in s["context"])
    elif fam == "tool_variant":
        arm = s["meta"]["sub"].split(":")[0]
        ok = (len(s["available_tools"]) == 0) if arm == "none" else (len(s["available_tools"]) == 1)
    elif fam == "transform_wrap":
        ok = any(s["query"].startswith(v) for v in TRANSFORM_VERBS)
    elif fam in ("user_pressure", "citation"):
        ok = s["query"].startswith(twin["query"]) and len(s["query"]) > len(twin["query"])
    elif fam == "ambiguity":
        ok = s["query"] != twin["query"] and len(s["query"]) < len(twin["query"]) + 40

    if ok:
        c["ok"] += 1
    elif len(c["fail"]) < 3:
        c["fail"].append(s["id"])

print(f"{'family':18}{'n':>7}{'applied':>10}{'rate':>9}")
print("-" * 46)
all_ok = True
for fam in sorted(checks):
    c = checks[fam]
    rate = 100 * c["ok"] / c["n"]
    print(f"{fam:18}{c['n']:>7}{c['ok']:>10}{rate:>8.1f}%")
    if rate < 99:
        all_ok = False
        print(f"    examples of non-applied: {c['fail']}")

print("\nGENERATOR INTEGRITY:", "PASS - every transform structurally applied" if all_ok else "FAIL - see above")

# uniqueness / pairing sanity
ids = {s["id"] for s in synth}
pairs = {s["pair_id"] for s in synth}
print(f"\nunique synth ids: {len(ids)}/{len(synth)} | distinct real twins referenced: {len(pairs)}")
print(f"all pair_ids resolve: {all(s['pair_id'] in real for s in synth)}")
queries = {s["query"] for s in synth}
print(f"distinct synthetic queries: {len(queries)} ({100*len(queries)/len(synth):.1f}%)")
