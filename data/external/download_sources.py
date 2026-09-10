"""Phase 0 — download every source into data/external/<name>/<split>.parquet.

Consolidated replacement for the original download_full.py + download_gapfillers.py +
download_legal_fin.py. Full fidelity: all splits, all columns, nothing sampled.

Usage:
    HF_TOKEN=<token> uv run --with datasets python download_sources.py

Notes on substitutions (the obvious paths are script-based repos that modern `datasets`
refuses to load, so these are the working equivalents):
  * casehold -> coastalcph/lex_glue cfg case_hold   (casehold/casehold is script-based)
  * finqa    -> wujian123/finqa                     (dreamerdeo/finqa is script-based;
                                                     this mirror keeps the ORIGINAL
                                                     pre_text/post_text/table/qa schema)
"""
import json
import os
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
from datasets import load_dataset

OUT = os.path.dirname(os.path.abspath(__file__))
TOKEN = os.environ.get("HF_TOKEN")

# key, repo, config, why-it-is-here
SOURCES = [
    # --- context-bearing (these 11 feed fullset.jsonl) ---
    ("squad",           "rajpurkar/squad",            None,           "single-hop, passage context"),
    ("hotpot_qa",       "hotpotqa/hotpot_qa",         "distractor",   "multi-hop, 10 paragraphs"),
    ("musique",         "dgslibisey/MuSiQue",         None,           "2-4 hop, 20 paragraphs"),
    ("pubmedqa",        "qiaojin/PubMedQA",           "pqa_labeled",  "medical high-stakes"),
    ("drop",            "ucinlp/drop",                None,           "numerical reasoning"),
    ("finqa",           "wujian123/finqa",            None,           "financial + table context"),
    ("casehold",        "coastalcph/lex_glue",        "case_hold",    "legal high-stakes"),
    ("streamingqa",     "bg51717/streamingqa",        None,           "freshness / news"),
    ("asqa",            "din0s/asqa",                 None,           "ambiguity"),
    ("retrievalqa",     "zihanz/RetrievalQA",         None,           "retrieval-need, incl. realtimeqa+freshqa"),
    ("crrag_conflict",  "SKIML-ICL/CRRAG_realtimeqa", None,           "REAL contradicting passages w/ NLI labels"),
    # --- no context: used as seeds/reference only, not in fullset ---
    ("prompt_injection", "deepset/prompt-injections", None,           "real injection strings (Phase 3 seeds)"),
    ("xstest",           "Paul/XSTest",               None,           "benign-but-scary (over-refusal probe)"),
    ("medqa",            "openlifescienceai/medqa",   None,           "clinical vignettes (no context field)"),
    ("ambig_qa",         "sewon/ambig_qa",            "light",        "ambiguity (superseded by asqa)"),
    ("trivia_qa",        "mandarjoshi/trivia_qa",     "rc.nocontext", "single-hop (context cols are empty)"),
]

manifest = {}
for key, repo, config, why in SOURCES:
    try:
        dd = load_dataset(repo, config, token=TOKEN)
        path = os.path.join(OUT, key)
        os.makedirs(path, exist_ok=True)
        rows, cols = {}, None
        for split in dd:
            fp = os.path.join(path, f"{split}.parquet")
            dd[split].to_parquet(fp)
            rows[split] = dd[split].num_rows
            cols = cols or dd[split].column_names
        manifest[key] = {"repo": repo, "config": config, "why": why, "rows": rows, "columns": cols}
        print(f"OK   {key}: {sum(rows.values())} rows {rows}", flush=True)
    except Exception as exc:
        manifest[key] = {"repo": repo, "config": config, "why": why,
                         "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        print(f"FAIL {key}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)

with open(os.path.join(OUT, "MANIFEST_sources.json"), "w") as fh:
    json.dump(manifest, fh, indent=2, ensure_ascii=False)

total = sum(sum(v.get("rows", {}).values()) for v in manifest.values())
print(f"\nTOTAL rows: {total}")
