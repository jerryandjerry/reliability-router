"""Injection detection rate by attack type x evasiveness x placement."""
import json
import os
from collections import defaultdict

from reliability_router.features import FeatureExtractor
from reliability_router.models import ContextItem, RouterRequest

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ex = FeatureExtractor()

def tagval(tags, prefix):
    for t in tags:
        if t.startswith(prefix):
            return t.split(":", 1)[1]
    return "?"

by_attack = defaultdict(lambda: {"n": 0, "hit": 0})
by_evasive = defaultdict(lambda: {"n": 0, "hit": 0})
by_cell = defaultdict(lambda: {"n": 0, "hit": 0})
by_place = defaultdict(lambda: {"n": 0, "hit": 0})
by_origin = defaultdict(lambda: {"n": 0, "hit": 0})

for line in open(os.path.join(DIR, "golden_synth.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["meta"]["family"] != "injection":
        continue
    f = ex.extract(RouterRequest(
        query=r["query"],
        context=[ContextItem(id=c["id"], text=c["text"], source=c.get("source"), trust=c["trust"])
                 for c in r["context"]],
        available_tools=r["available_tools"]))
    hit = f.prompt_injection >= 0.5   # the override threshold
    atk = tagval(r["tags"], "attack:")
    eva = tagval(r["tags"], "evasiveness:")
    plc = tagval(r["tags"], "placement:")
    org = tagval(r["tags"], "payload:")
    for d, k in ((by_attack, atk), (by_evasive, eva), (by_cell, f"{atk}|{eva}"),
                 (by_place, plc), (by_origin, org)):
        d[k]["n"] += 1
        d[k]["hit"] += hit

def show(title, d):
    print(f"\n=== {title} ===")
    for k in sorted(d):
        v = d[k]
        print(f"  {k:44} n={v['n']:>5}  detected {100*v['hit']/v['n']:5.1f}%")

show("by attack type", by_attack)
show("by evasiveness", by_evasive)
show("by placement", by_place)
show("by payload origin", by_origin)
show("attack x evasiveness", by_cell)

tot_n = sum(v["n"] for v in by_attack.values())
tot_h = sum(v["hit"] for v in by_attack.values())
print(f"\nOVERALL: {tot_h}/{tot_n} = {100*tot_h/tot_n:.1f}% of injections reach the >=0.5 override threshold")
