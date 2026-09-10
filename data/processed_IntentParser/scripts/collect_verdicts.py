"""Phase 4 collect: fold Opus + Sonnet verdict files into one table and list the contested rows.

Verdicts are positional (request 1..25 within a batch), so row = batch*25 + (request-1).
judge_order.json turns that position into a judge_input line, and id_map.json turns that
line into the real case id — judges never saw any of it.

Outputs:
  verdicts.jsonl   one line per row: row, id, opus, sonnet, agree, whys
  contested.json   row numbers where the two judges disagree (input to Fable adjudication)
"""
import glob
import json
import os
from collections import Counter

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = 25

id_map = {int(k): v for k, v in json.load(open(os.path.join(DIR, "id_map.json"))).items()}
# id_map is keyed by judge_input line index, but verdict rows are positions in the judge
# order; order[position] bridges the two.
order = json.load(open(os.path.join(DIR, "judge_order.json")))


def load(model):
    out, bad = {}, []
    for fp in sorted(glob.glob(os.path.join(DIR, "judge_out", model, "batch_*.json"))):
        b = int(os.path.basename(fp).split("_")[1].split(".")[0])
        try:
            data = json.load(open(fp, encoding="utf-8"))
        except Exception as exc:
            bad.append((os.path.basename(fp), str(exc)[:60]))
            continue
        for v in data:
            req = v.get("request")
            if not isinstance(req, int):
                continue
            out[b * BATCH + req - 1] = {"route": v.get("route"), "why": v.get("why", "")}
    return out, bad


opus, bad_o = load("opus")
sonnet, bad_s = load("sonnet")
print(f"opus rows {len(opus)} (unreadable files {len(bad_o)}) | sonnet rows {len(sonnet)} (unreadable {len(bad_s)})")
for b in (bad_o + bad_s)[:5]:
    print("   bad:", b)

rows = sorted(set(opus) | set(sonnet))
missing_o = [r for r in rows if r not in opus]
missing_s = [r for r in rows if r not in sonnet]
print(f"total rows seen {len(rows)} / {len(id_map)} | missing opus {len(missing_o)} | missing sonnet {len(missing_s)}")

agree = 0
contested = []
counts = Counter()
with open(os.path.join(DIR, "verdicts.jsonl"), "w", encoding="utf-8") as fh:
    for r in rows:
        o, s = opus.get(r), sonnet.get(r)
        rec = {
            "row": r, "id": id_map.get(order[r]) if r < len(order) else None,
            "opus": o["route"] if o else None, "opus_why": o["why"] if o else "",
            "sonnet": s["route"] if s else None, "sonnet_why": s["why"] if s else "",
        }
        rec["agree"] = bool(o and s and o["route"] == s["route"])
        if rec["agree"]:
            agree += 1
            counts[o["route"]] += 1
        elif o and s:
            contested.append(r)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

json.dump(contested, open(os.path.join(DIR, "contested.json"), "w"))
both = agree + len(contested)
print(f"\nagree {agree}/{both} = {100*agree/max(1,both):.1f}%")
print(f"contested {len(contested)} -> contested.json (Fable adjudicates these)")
print(f"agreed labels: {dict(counts)}")
if missing_o or missing_s:
    json.dump({"opus": missing_o, "sonnet": missing_s},
              open(os.path.join(DIR, "missing_rows.json"), "w"))
    print("missing rows written to missing_rows.json — re-run those batches before merging")
