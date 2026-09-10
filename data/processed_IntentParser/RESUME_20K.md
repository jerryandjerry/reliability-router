# Resume: judging the second 10,000

The golden set is **20,000 rows prepared, 10,000 judged**. The judged half shipped as
`data/routing_eval_golden.jsonl`. This note is what you need to finish the other half.

State verified 2026-08-03.

## What is left

Judging order is `judge_order.json`; batch `b` covers positions `b*25 .. b*25+24`.
Batches 0–399 are the shipped 10k. Batches 400–799 are the remainder.

| judge | 0–399 | 400–799 | remaining | rough time |
|---|--:|--:|--:|--:|
| opus | 400/400 ✓ | 261/400 | **139 batches** (~3,475 rows) | ~35 min |
| sonnet | 400/400 ✓ | 0/400 | **400 batches** (10,000 rows) | ~8 h |
| fable | 34 chunks ✓ | — | after both | ~30 min |

Opus stopped at batch 662 when a session limit killed the original run, so its gap is
**658, 659, and 663–799**. It is *not* a Sonnet-only pass — an earlier note in
`README_DataPrep.md` said that and was wrong. If you skip Opus's 139 batches, those ~3,475 rows
are silently dropped at build time for lacking a second judge.

Recompute the real gap rather than trusting this table:

```bash
cd data/processed_IntentParser
uv run python - <<'PY'
import glob, os, json
for m in ("opus", "sonnet"):
    ok = set()
    for f in glob.glob(f"judge_out/{m}/batch_*.json"):
        b = int(os.path.basename(f).split("_")[1].split(".")[0])
        try:
            if len(json.load(open(f))) >= 1:
                ok.add(b)
        except Exception:
            pass
    missing = [b for b in range(800) if b not in ok]
    print(m, "missing", len(missing), missing[:20], "..." if len(missing) > 20 else "")
PY
```

## Steps

1. **Judge the gaps.** Both models, feeding the missing batch numbers from the snippet above:

   ```
   Workflow({scriptPath: "data/processed_IntentParser/scripts/judge_fill.js",
             args: {model: "opus",   batches: [658, 659, 663, ..., 799]}})
   Workflow({scriptPath: "data/processed_IntentParser/scripts/judge_fill.js",
             args: {model: "sonnet", batches: [400, ..., 799]}})
   ```

   Run them concurrently; they share the machine's agent slots. Sonnet sets the wall clock at
   roughly 4 batches / 5 min against Opus's 20. Poll `judge_out/*/` for progress; batch files are
   keyed by number, so re-running one overwrites rather than duplicating.

   On `400 Output blocked by content filtering policy`, **retry the batch first** — the block is
   non-deterministic. `judge_split.js` re-judges one batch in smaller groups if a retry keeps
   failing.

2. **Build.** `scripts/build_golden_10k.py` hardcodes `ROWS = 10000` at line 20. Set it to
   `20000` (or parameterise it) before running, or it will silently rebuild only the first half:

   ```bash
   cd data/processed_IntentParser/scripts
   uv run python build_golden_10k.py          # -> golden.jsonl + contested.json
   ```

   It writes any row still lacking two judges to `missing_rows.json`; feed those batch numbers
   back into step 1 until that file is empty.

3. **Adjudicate and finalise.**

   ```bash
   uv run python prepare_adjudication.py      # -> adjudication_chunks.json
   #   Workflow({scriptPath: "scripts/judge_adjudicate.js",
   #             args: {model: "fable", chunks: <adjudication_chunks.json>}})
   uv run python finalize_golden_10k.py       # folds Fable's verdicts into golden.jsonl
   ```

   At the 91.6% agreement seen on the first half, expect roughly 850 more contested rows.

4. **Publish.** Copy `golden.jsonl` to `data/routing_eval_golden.jsonl`, which is what the
   runtime eval consumes. Anything derived from it is now stale and must be rebuilt:
   `data/routing_eval_golden_1k.jsonl` (the held-out leaderboard set),
   `data/intent_parser_splits/` (train/dev, and its `split_manifest.json` hashes), and every
   leaderboard entry, since the dataset it was scored on has changed.

## Before you start

Two questions worth settling first, because both change whether this work pays off.

* **Does 20k buy anything?** The 10k already gives about ±1% on overall accuracy. The gain is in
  thin slices — `crrag_conflict` is 139 rows, `retrievalqa` 439 — where doubling roughly halves
  the interval. Worth it for per-category recall on rare scenarios; not needed to rank parsers.
* **Sequence it with the sampling fix.** The held-out 1k is drawn by
  `select_eval_cases(stratified=True)`, whose stratum key uses all 22 tag axes and so allocates
  one case per stratum (equal allocation) rather than proportionally. If that is going to be
  fixed, doing it *after* the 20k lands means the new held-out set draws from twice the pool and
  the LLM leaderboard entries only need re-running once.
