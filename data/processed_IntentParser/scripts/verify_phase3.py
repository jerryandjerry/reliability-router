"""Verification gate: did each Phase-3 family actually move its target feature?

Compares each synthetic row against its real twin (pair_id) using the real FeatureExtractor.
"""
import json
import os
from collections import defaultdict

from reliability_router.features import FeatureExtractor
from reliability_router.models import ContextItem, RouterRequest
from reliability_router.service import ReliabilityRouter

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGET = {
    "context_removal": "evidence_gap", "context_shuffle": "evidence_gap", "conflict": "context_conflict",
    "injection": "prompt_injection", "tool_variant": "tool_gap", "transform_wrap": "freshness",
    "user_pressure": "user_pressure", "ambiguity": "ambiguity", "citation": "citation_need",
}

ex = FeatureExtractor()
router = ReliabilityRouter()


def to_request(rec):
    return RouterRequest(
        query=rec["query"],
        context=[ContextItem(id=c["id"], text=c["text"], source=c.get("source"), trust=c["trust"])
                 for c in rec["context"]],
        available_tools=rec["available_tools"],
    )


real = {json.loads(line)["id"]: json.loads(line)
        for line in open(os.path.join(DIR, "golden_real.jsonl"), encoding="utf-8")}
synth = [json.loads(line) for line in open(os.path.join(DIR, "golden_synth.jsonl"), encoding="utf-8")]
print(f"real {len(real)} | synth {len(synth)}\n")

stats = defaultdict(lambda: {"n": 0, "fired": 0, "rose": 0, "route_flip": 0, "sum_before": 0.0, "sum_after": 0.0})
sub_stats = defaultdict(lambda: {"n": 0, "fired": 0})

for s in synth:
    fam = s["meta"]["family"]
    tgt = TARGET[fam]
    fs = ex.extract(to_request(s))
    after = getattr(fs, tgt)
    twin = real.get(s["pair_id"])
    before = getattr(ex.extract(to_request(twin)), tgt) if twin else 0.0
    st = stats[fam]
    st["n"] += 1
    st["sum_before"] += before
    st["sum_after"] += after
    if after > 0:
        st["fired"] += 1
    if after > before:
        st["rose"] += 1
    if twin:
        if router.route(to_request(s)).route != router.route(to_request(twin)).route:
            st["route_flip"] += 1
    k = f"{fam}|{s['meta']['sub']}"
    sub_stats[k]["n"] += 1
    if after > 0:
        sub_stats[k]["fired"] += 1

print(f"{'family':18}{'target':18}{'n':>6}{'fires%':>8}{'rose%':>8}{'before':>8}{'after':>8}{'routeflip%':>11}")
print("-" * 95)
ok = True
for fam in sorted(stats):
    st = stats[fam]
    n = st["n"]
    fires = 100 * st["fired"] / n
    rose = 100 * st["rose"] / n
    flip = 100 * st["route_flip"] / n
    print(f"{fam:18}{TARGET[fam]:18}{n:>6}{fires:>7.1f}%{rose:>7.1f}%"
          f"{st['sum_before']/n:>8.3f}{st['sum_after']/n:>8.3f}{flip:>10.1f}%")
    if fires < 5.0:
        ok = False

print("\n=== per-sub-scenario activation (target feature > 0) ===")
for k in sorted(sub_stats):
    v = sub_stats[k]
    pct = 100 * v["fired"] / v["n"]
    flag = "   <-- never fires" if v["fired"] == 0 else ""
    print(f"  {k:52} n={v['n']:>5}  {pct:5.1f}%{flag}")

print("\nGATE:", "PASS" if ok else "REVIEW — some family never moved its target feature")
