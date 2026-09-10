# Intent Parser Splits

Shared training data for supervised intent parsers.

| file | rows | purpose |
|---|--:|---|
| `../routing_eval_golden.jsonl` | 10,000 | the labelled golden set, the source of both splits |
| `../routing_eval_golden_1k.jsonl` | 1,000 | **held out.** Every parser is scored on this and nothing trains on it |
| `train.jsonl` | 9,000 | everything else |

**There is no dev split.** Models train on all 9,000 with hyperparameters fixed in their config,
so no checkpoint or threshold is chosen by looking at held-out data.

Rebuild both files:

```bash
uv run python data/intent_parser_splits/build_train_dev_split.py
```

The evaluation set is a miniature of the whole: rows are grouped by `(route, type-signature)` and
every group contributes 10% of itself. That keeps the eval set's label balance and type mix within
about half a percentage point of the full set. Grouping deliberately ignores `source:` — a router
never sees which corpus a question came from, so balancing on it would balance bookkeeping rather
than the request.

`train.jsonl` and the eval file are generated and gitignored. `split_manifest.json` is small,
records the recipe and the SHA-256 of each file, and is committed.

The split is deterministic: rows are ordered within a group by the SHA-256 of their id, so
rebuilding gives byte-identical files as long as the source is unchanged. If the source changes,
every trained model must be retrained — its old training rows may now be in the evaluation set.
