"""Build data/processed_IntentParser/fullset.jsonl from every context-bearing source.

One JSON object per line:
  {id, source, query, context:[{id,text,source,trust}], available_tools:[], tags:[], notes, answer, meta}

`context`/`available_tools` mirror RouterRequest so rows convert to EvalCase by adding
expected_route (and dropping answer/meta, which EvalCase forbids).
"""
import glob
import json
import os

import pandas as pd

OUTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(os.path.dirname(OUTDIR), "external")
OUT = os.path.join(OUTDIR, "fullset.jsonl")
MAX_ITEM_CHARS = 8000  # guard against pathological rows; recorded in meta when it bites

os.makedirs(OUTDIR, exist_ok=True)


def clean(text) -> str:
    return " ".join(str(text).split()).strip()


def citem(text, source, trust, cid, **meta):
    text = clean(text)
    truncated = len(text) > MAX_ITEM_CHARS
    return {
        "id": cid,
        "text": text[:MAX_ITEM_CHARS],
        "source": source,
        "trust": trust,
        **({"truncated": True} if truncated else {}),
        **meta,
    }


# ---------- per-source extractors: (row) -> (query, context_items, tags, answer, meta) ----------

def ex_squad(r):
    return (
        clean(r["question"]),
        [citem(r["context"], f"wikipedia:{r['title']}", 0.85, "ctx_0")],
        ["single-hop"],
        list(r["answers"]["text"])[:3],
        {},
    )


#: HotpotQA and MuSiQue bury the answer among decoy paragraphs to test *retrievers*. This router
#: receives already-retrieved context, so the decoys measure nothing here while costing 42% of the
#: token budget. Keep every supporting paragraph plus this many decoys, so "relevant evidence among
#: some noise" survives.
MAX_DECOYS = 3


def ex_hotpot(r):
    ctx = r["context"]
    supporting = set(r["supporting_facts"]["title"])
    keep, decoys = [], []
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        (keep if title in supporting else decoys).append((title, sents))
    keep += decoys[:MAX_DECOYS]
    items = [
        citem(" ".join(sents), f"wikipedia:{title}", 0.85, f"ctx_{i}",
              supporting=title in supporting)
        for i, (title, sents) in enumerate(keep)
    ]
    return clean(r["question"]), items, ["multi-hop", f"level:{r['level']}", f"type:{r['type']}"], r["answer"], {}


def ex_musique(r):
    paras = list(r["paragraphs"])
    keep = [p for p in paras if p["is_supporting"]]
    keep += [p for p in paras if not p["is_supporting"]][:MAX_DECOYS]
    items = []
    for i, p in enumerate(keep):
        items.append(
            citem(p["paragraph_text"], f"wikipedia:{p['title']}", 0.85, f"ctx_{i}",
                  supporting=bool(p["is_supporting"]))
        )
    hops = len(r["question_decomposition"])
    return clean(r["question"]), items, ["multi-hop", f"hops:{hops}"], r["answer"], {"answerable": bool(r["answerable"])}


def ex_pubmedqa(r):
    ctx = r["context"]
    items = [citem(t, "pubmed:abstract", 0.90, f"ctx_{i}") for i, t in enumerate(ctx["contexts"])]
    return (
        clean(r["question"]),
        items,
        ["high-stakes:medical", "citation-bearing"],
        r["final_decision"],
        {"long_answer": clean(r["long_answer"])[:500]},
    )


def ex_drop(r):
    return clean(r["question"]), [citem(r["passage"], "wikipedia:passage", 0.85, "ctx_0")], ["numerical"], \
        list(r["answers_spans"]["spans"])[:3], {}


def ex_finqa(r):
    items = []
    pre = " ".join(clean(s) for s in r["pre_text"])
    post = " ".join(clean(s) for s in r["post_text"])
    if pre:
        items.append(citem(pre, "financial-filing:narrative", 0.88, "ctx_pre"))
    tbl = " | ".join(" ; ".join(clean(c) for c in row) for row in r["table"])
    if tbl:
        items.append(citem(tbl, "financial-filing:table", 0.88, "ctx_table"))
    if post:
        items.append(citem(post, "financial-filing:narrative", 0.88, "ctx_post"))
    qa = r["qa"]
    return clean(qa["question"]), items, ["high-stakes:financial", "numerical", "table-context"], \
        qa.get("exe_ans", qa.get("answer")), {}


def ex_casehold(r):
    # lex_glue/case_hold is multiple-choice: pick which holding fills <HOLDING>.
    # Query is constructed from the first candidate so it varies per row and stays a natural legal question.
    endings = list(r["endings"])
    proposition = clean(endings[0]) if endings else "the cited holding applies"
    proposition = proposition.removeprefix("holding that ").removeprefix("holding ")
    query = f"Does the court hold that {proposition}?"
    items = [citem(r["context"], "court-opinion:excerpt", 0.88, "ctx_0")]
    items.append(citem(" | ".join(clean(e) for e in endings), "court-opinion:candidate-holdings", 0.88, "ctx_options"))
    return query, items, ["high-stakes:legal", "adapted:multiple-choice"], int(r["label"]), {}


def ex_streamingqa(r):
    return clean(r["question"]), [citem(r["context"], "news:article", 0.75, "ctx_0")], ["freshness"], \
        list(r["answers"])[:3], {}


def ex_asqa(r):
    items, i = [], 0
    for ann in r["annotations"]:
        for k in ann["knowledge"]:
            items.append(citem(k["content"], f"wikipedia:{k.get('wikipage')}", 0.85, f"ctx_{i}"))
            i += 1
    answers = []
    for qp in r["qa_pairs"]:
        answers.extend(list(qp["short_answers"])[:1])
    interpretations = len(r["qa_pairs"])
    return clean(r["ambiguous_question"]), items, ["ambiguity", f"interpretations:{interpretations}"], \
        answers[:4], {}


RQA_TAGS = {"realtimeqa": "freshness", "freshqa": "freshness", "popqa": "single-hop",
            "triviaqa": "single-hop", "toolqa": "tool-dependent"}
RQA_TRUST = {"realtimeqa": 0.75, "freshqa": 0.75, "popqa": 0.85, "triviaqa": 0.85, "toolqa": 0.70}


def ex_retrievalqa(r):
    src = str(r["data_source"])
    items = []
    for i, raw in enumerate(r["context"]):
        try:
            d = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        text = d.get("text") or ""
        if not clean(text):
            continue
        items.append(citem(text, f"{src}:{d.get('title')}", RQA_TRUST.get(src, 0.8), f"ctx_{i}"))
    tags = [RQA_TAGS.get(src, "single-hop"), f"origin:{src}", "retrieval-need"]
    try:
        answer = json.loads(str(r["ground_truth"]).replace("'", '"'))
    except (json.JSONDecodeError, TypeError, ValueError):
        answer = clean(r["ground_truth"])
    return clean(r["question"]), items, tags, answer, {}


def ex_crrag(r):
    # Real contradicting evidence: keep supporting ctxs plus the dataset's own conflict passage.
    items = []
    for i, c in enumerate(list(r["ctxs"])[:5]):
        items.append(citem(c["text"], f"news:{c.get('title')}", 0.75, f"ctx_{i}", nli=str(c.get("nli"))))
    if clean(r["conflict_passage"]):
        items.append(citem(r["conflict_passage"], "news:conflicting-report", 0.70, "ctx_conflict",
                           nli="contradiction"))
    return clean(r["question"]), items, ["conflict", "freshness", "real-contradiction"], \
        list(r["answers"])[:3], {"answerable": str(r["answerable"])}


SOURCES = [
    ("squad", ex_squad), ("hotpot_qa", ex_hotpot), ("musique", ex_musique), ("pubmedqa", ex_pubmedqa),
    ("drop", ex_drop), ("finqa", ex_finqa), ("casehold", ex_casehold), ("streamingqa", ex_streamingqa),
    ("asqa", ex_asqa), ("retrievalqa", ex_retrievalqa), ("crrag_conflict", ex_crrag),
]


def jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


written = {}
skipped = {}
with open(OUT, "w", encoding="utf-8") as out:
    for key, extractor in SOURCES:
        files = sorted(glob.glob(os.path.join(BASE, key, "*.parquet")))
        if not files:
            print(f"WARN {key}: no parquet files", flush=True)
            continue
        n = bad = 0
        for fp in files:
            split = os.path.basename(fp).replace(".parquet", "")
            df = pd.read_parquet(fp)
            for idx, row in df.iterrows():
                try:
                    query, items, tags, answer, meta = extractor(row)
                except Exception:
                    bad += 1
                    continue
                if not query or not items:
                    bad += 1
                    continue
                rec = {
                    "id": f"{key}_{split}_{idx}",
                    "source": key,
                    "query": query,
                    "context": items,
                    "available_tools": [],
                    "tags": [key, *tags],
                    "notes": f"real:{key}:{split}:{idx}",
                    "answer": jsonable(answer),
                    "meta": {"split": split, **jsonable(meta)},
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        written[key] = n
        skipped[key] = bad
        print(f"OK   {key}: wrote {n} (skipped {bad})", flush=True)

print("\n=== fullset summary ===")
for k, v in written.items():
    print(f"  {k:16} {v:>7}  (skipped {skipped[k]})")
print(f"  {'TOTAL':16} {sum(written.values()):>7}")
print(f"  size: {os.path.getsize(OUT) / 1e6:.1f} MB -> {OUT}")
