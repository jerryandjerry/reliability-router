# Intent-Parser Golden Set — data preparation

A labeled dataset for the router's **QUICK vs DEEP** decision, replacing the 24-case smoke test in
`data/routing_eval.jsonl`.

**`golden.jsonl` — 10,000 rows, 4,406 QUICK / 5,594 DEEP.** Half real questions drawn from 11
public QA corpora, half adversarial variants built from them. Every row is labeled by two LLM
judges independently, with a third breaking ties. The runtime evaluation pipeline consumes the
published copy at `data/routing_eval_golden.jsonl`.

20,000 rows are prepared; 10,000 are labeled. Scoring a router against this file is a separate
concern — see the root `README.md`.

## What a row looks like

```json
{"id": "asqa_train_141",
 "query": "When did the ipad 1st gen come out?",
 "expected_route": "QUICK",
 "context": [{"id": "ctx_0",
              "text": "The first-generation iPad is a tablet computer designed and marketed by…",
              "source": "wikipedia:IPad (1st generation)",
              "trust": 0.85}],
 "available_tools": [],
 "tags": ["source:asqa", "risk:ambiguity", "meta:interpretations-5",
          "origin:real", "judged:unanimous"],
 "notes": "real:asqa:train:141 | opus=QUICK sonnet=QUICK | Static well-known fact…"}
```

Validates as a strict `EvalCase` (`reliability_router.models`) — extra fields are forbidden, so
`answer`/`meta`/`pair_id` are dropped on the way in.

**`context` is conversation history**, not a knowledge base: material already in the request, each
piece carrying a `trust` score. Below 0.8 means questionable provenance.

### Tags you can filter on

| prefix | values | on |
|---|---|--:|
| `source:` | the 11 corpora | all |
| `risk:` | `high-stakes-medical`, `-legal`, `-financial`, `freshness`, `ambiguity`, `conflict`, `numerical`, `citation-need`, `multi-hop`, `single-hop`, `retrieval-need`, `table-context`, `real-contradiction` | all |
| `origin:` | `real`, `synthetic` | all |
| `judged:` | `unanimous` (9,158), `boundary` (842, tie-broken) | all |
| `transform:` | the 9 adversarial families | synthetic |
| `attack:` `evasiveness:` `placement:` `payload:` | injection detail | injection rows |
| `pressure:` `intensity:` | phrasing detail | user_pressure rows |

`judged:boundary` exists so the 842 rows two judges disagreed on can be excluded.

## Where the labels come from

`judge_prompt.md` **is** the label definition — the policy both judges apply. Editing it changes
what this dataset means and invalidates every existing label.

Opus 5 and Sonnet 5 label all 10,000 rows independently. Fable 5 adjudicates the disagreements
**blind** — it sees the same rendering under the same policy and is never told what the other two
said, so it breaks ties rather than anchoring on one.

| | |
|---|--:|
| inter-judge agreement | **91.6%** (9,158 unanimous) |
| contested → Fable | 842 (422 DEEP / 420 QUICK) |
| Fable sided with | Opus 656 · Sonnet 186 |
| `EvalCase` validation failures | 0 |

Judges see **only what a caller would send** — user message, context text, context trust, tool
names — rendered by one shared script (`render_batch.py`). No ids, no corpus names, no counts.
This matters: an earlier run leaked corpus and adversarial-family names through row ids, which
told the judge the answer.

Labels describe **what a request is**, not what any particular implementation can detect. A
genuinely high-stakes clinical question stays DEEP whether or not a keyword matches.

## Pipeline

| phase | script | in | out | rows |
|---|---|---|---|--:|
| 0 | `data/external/download_sources.py` | 16 HF dataset ids | `<source>/<split>.parquet` | 618,345 |
| 1 | `build_fullset.py` | the 11 context-bearing sources | `fullset.jsonl` | 418,084 |
| 2 | `sample_phase2.py` | `fullset.jsonl` | `golden_real.jsonl` | 10,000 |
| 3a | `assign_families.py` | `golden_real.jsonl` | `assets/family_assignment.json` | 10,000 |
| 3b | `build_phase3.py` | `golden_real.jsonl` + `assets/*` | `golden_synth.jsonl` | 10,000 |
| 4a | `prepare_judge_input.py` | both golden files | `judge_input.jsonl`, `id_map.json`, `judge_order.json` | 20,000 |
| 4b | `render_batch.py` | `judge_input.jsonl` | one batch of 25 as text | 25/call |
| 4c | `judge_fill.js` | batch numbers | `judge_out/{opus,sonnet}/batch_NNN.json` | 10,000 |
| 4d | `build_golden_10k.py` | both judges | `golden.jsonl` + `contested.json` | 9,158 |
| 4e | `prepare_adjudication.py` | `contested.json` | `adjudication_chunks.json` | 842 |
| 4f | `judge_adjudicate.js` | those chunks | `judge_out/fable/chunk_NNN.json` | 842 |
| 4g | `finalize_golden_10k.py` | Fable's verdicts | `golden.jsonl` complete | **10,000** |

Real and synthetic stay in separate files so origin is distinguishable at the file level; every
synthetic row also carries `pair_id` back to its real twin.

### Phase 1 — the sources

| source | HF path | rows | context field | axis |
|---|---|--:|---|---|
| squad | `rajpurkar/squad` | 98,169 | `context` (~915 ch) | single-hop |
| hotpot_qa | `hotpotqa/hotpot_qa` (distractor) | 97,852 | `context`, 10 paras † | multi-hop |
| drop | `ucinlp/drop` | 86,935 | `passage` (~997 ch) | numerical |
| casehold | `coastalcph/lex_glue` (case_hold) | 52,500 | `context` (citing prompt) | legal |
| streamingqa | `bg51717/streamingqa` | 46,317 | `context`, news (~2.2k) | freshness |
| musique | `dgslibisey/MuSiQue` | 22,355 | `paragraphs`, 20 † | multi-hop 2–4 |
| finqa | `wujian123/finqa` | 7,134 | `pre_text`/`table`/`post_text` | financial + numeric |
| asqa | `din0s/asqa` | 5,301 | `annotations[].knowledge` | ambiguity |
| retrievalqa | `zihanz/RetrievalQA` | 1,271 | `context` (JSON strings) | retrieval-need |
| pubmedqa | `qiaojin/PubMedQA` | 1,000 | `context.contexts` | medical |
| crrag_conflict | `SKIML-ICL/CRRAG_realtimeqa` | 100 | `ctxs` + `conflict_passage` | **real conflict** |

Every row of every source, no sampling. `fullset.jsonl` is **418,084 rows / 930.6 MB**, context
1–25 items per row (mean 2.4), median 1,518 chars, max 25,084, 0.734 B characters total.

Skipped for genuinely lacking context: asqa 775 (empty `knowledge`), retrievalqa 75 (`toolqa`
rows). Trust by provenance: PubMed 0.90 · legal/financial filings 0.88 · Wikipedia 0.85 ·
news 0.75 · conflicting reports 0.70.

**Two substitutions are required** — the canonical repos are script-based and current `datasets`
refuses to load them: `casehold/casehold` → `coastalcph/lex_glue` cfg `case_hold`, and
`dreamerdeo/finqa` → `wujian123/finqa` (which keeps the original schema).

**† Distractor trimming, `MAX_DECOYS = 3`.** HotpotQA ships 10 paragraphs and MuSiQue 20, of which
only 2–4 support the answer; the rest are deliberate distractors, measured at 84–85% of those
sources' context and 42% of all context. `build_fullset.py` keeps every supporting paragraph plus
at most 3 distractors. No row is dropped, only shortened. This is why the sizes above are smaller
than what the sources ship — and changing `MAX_DECOYS` changes every size figure on this page.

### Phase 2 — sampling

Stratified by source × descriptive tags × context-size bucket, seed `20260726`, rare sources taken
whole. Deduplicated on `(source, lowercased query)`: 367,111 unique, 50,973 dropped.

### Phase 3 — adversarial variants

| family | n | targets | method |
|---|--:|---|---|
| injection | 2,500 | prompt injection | deterministic |
| context_removal | 1,500 | evidence gap | deterministic |
| context_shuffle | 1,200 | evidence relevance | deterministic |
| conflict | 1,200 | context conflict | deterministic |
| tool_variant | 800 | tool gap | deterministic |
| transform_wrap | 800 | transform guard-leak | deterministic |
| user_pressure | 700 | agreement pressure | **LLM bank** |
| ambiguity | 700 | ambiguity | **LLM per-row** |
| citation | 600 | citation need | **LLM bank** |

**`scripts/assets/` cannot be regenerated — preserve it.** These are LLM outputs: 220 pressure
clauses (12 subtypes), 160 citation clauses (9 subtypes), 746 injection payloads (5 types, each
split blatant/paraphrased/evasive), and 1,110 per-row ambiguity rewrites of which **700 are used**
— the surplus is a second generation pass, kept because rewrites are matched by row id and the
extras cover ids an earlier sampling produced. Injection pool = 746 generated + 263 real
`deepset/prompt-injections` = 1,009 distinct payloads. Both banks are part English, part Chinese.

The generating agents were never shown the router's regexes, so phrasings are natural rather than
engineered to match one.

Deleting these banks changes the output. The sharpest determinism check is `ambiguity: 700` — the
rewrites are keyed by row id, so any drift in sampling or family assignment leaves them unmatched
and silently yields fewer than 10,000 synthetic rows.

## Layout

Raw data and its downloader in `data/external/`; everything processed here. **Corpora and `.jsonl`
are gitignored; code, docs and the irreproducible LLM assets are tracked.**

```
data/external/                     # RAW
  download_sources.py              # tracked
  <source>/<split>.parquet         # ignored — 16 sources
  MANIFEST_sources.json            # ignored
  benchmark_catalog.csv            # ignored — 104-dataset catalog

data/processed_IntentParser/       # PROCESSED
  README_DataPrep.md               # tracked — this file
  judge_prompt.md                  # tracked — the label definition
  fullset.jsonl                    # ignored — 418,084
  golden_real.jsonl                # ignored — 10,000
  golden_synth.jsonl               # ignored — 10,000
  judge_input.jsonl                # ignored — content-only rows judges see
  id_map.json                      # ignored — row -> case id, never shown to a judge
  judge_order.json                 # ignored — stratified judging order
  judge_out/{opus,sonnet,fable}/   # ignored — raw verdicts
  contested.json                   # ignored — rows the two judges split on
  adjudication_chunks.json         # ignored — chunk -> rows
  golden.jsonl                     # ignored — 10,000 labeled EvalCase
  scripts/                         # tracked
    assets/                        # tracked — LLM-generated, NOT reproducible
```

## Build it

```bash
cd data/external
HF_TOKEN=<token> uv run --with datasets python download_sources.py   # Phase 0

cd ../processed_IntentParser/scripts
uv run --with pandas --with pyarrow python build_fullset.py           # Phase 1
uv run python sample_phase2.py                                       # Phase 2
uv run python assign_families.py                                     # Phase 3a
uv run --with pandas --with pyarrow python build_phase3.py            # Phase 3b
uv run python prepare_judge_input.py                                 # Phase 4a
```

Judging (4c, 4f) runs as Claude Code Workflows, one agent per batch of 25 — not a shell command:

```
Workflow({scriptPath: "scripts/judge_fill.js",
          args: {model: "opus",   batches: [0, 1, …, 399]}})
Workflow({scriptPath: "scripts/judge_fill.js",
          args: {model: "sonnet", batches: [0, 1, …, 399]}})
```

Every agent runs `python3 scripts/render_batch.py <batch>` — no agent formats anything itself,
which is what keeps the two judges seeing byte-identical input. Then:

```bash
uv run python build_golden_10k.py                                    # Phase 4d
uv run python prepare_adjudication.py                                # Phase 4e
#   Workflow({scriptPath: "scripts/judge_adjudicate.js",
#             args: {model: "fable", chunks: <adjudication_chunks.json>}})
uv run python finalize_golden_10k.py                                 # Phase 4g
```

`uv run` picks the project venv (3.11) and resolves `reliability_router` without `PYTHONPATH`.
`render_batch.py` is the exception — judges invoke it as bare `python3`, so it stays 3.9-safe.

**Verification:** `validate_fullset.py` (Phase 1) · `verify_generator.py` + `verify_phase3.py`
(Phase 3). A full wipe-and-rebuild reproduced identical output: fullset 418,084 / 930.6 MB, dedup
367,111 / 50,973, both golden files 10,000, identical family counts, generator integrity 100%.

### Resuming a partial judge pass

`build_golden_10k.py` writes any unlabeled rows to `missing_rows.json`; feed those batch numbers
back into `judge_fill.js`. Batch files are keyed by number, so re-running one overwrites rather
than duplicating.

If a batch fails with `400 Output blocked by content filtering policy`, **retry it** — the block
is non-deterministic and a plain retry usually succeeds. `judge_split.js` re-judges a single batch
in smaller groups if you need to isolate one row.

### Labeling the other 10,000

`judge_order.json` is stratified round-robin, so batches 400–799 are the same distribution as
0–399. Run Phase 4c over that range with both judges, then 4d–4g unchanged.

**Both judges have work left there** — Opus stopped at batch 662 when a session limit ended the
original run, and Sonnet never started the range. `build_golden_10k.py` also hardcodes
`ROWS = 10000`. Exact gaps, commands and the downstream artifacts that go stale are in
[`RESUME_20K.md`](RESUME_20K.md).

## Limitations of this data

* Labels encode `judge_prompt.md`. They validate a router against **that policy**, not the
  correctness of the policy itself.
* Labels are model-generated. Two independent judges plus a blind tie-breaker at 91.6% agreement
  bounds the noise; it does not remove it.
* CaseHOLD queries are **constructed** from candidate holdings (tagged
  `adapted:multiple-choice`) rather than asked by a person, and can be excluded.
* `user_pressure` and `citation` phrasings are synthetic — no context-bearing corpus supplies
  them.
* Two arms in `verify_phase3.py` are mis-specified and their gate numbers should be ignored:
  `transform_wrap` is scored on freshness *rising* when the guard is meant to suppress it, and
  `tool_variant` compares against a real twin whose `available_tools` is always empty.
