# BERT-Family Intent Parser

This folder owns training and export code for fine-tuned encoder classifiers.
One backbone is trained here: DistilBERT. Shared training data is built under
`data/intent_parser_splits/` so other model families reuse the same split.

There is no dev split — training runs a fixed number of epochs on all 9,000 rows,
because selecting a checkpoint needs held-out data and the only held-out data is
the evaluation set.

The runtime parser family is selected with `RR_INTENT_PARSER=bert`. The concrete
method name logged to the leaderboard is read from the exported manifest:

```json
{
  "name": "distilbert",
  "version": "v1"
}
```

For another backbone, keep the same folder and runtime family and point the
training config at a different `base_model`, writing whatever `name` identifies
it on the leaderboard.

## Build Shared Data

```bash
uv run python data/intent_parser_splits/build_train_dev_split.py
```

This creates two ignored JSONL files:

- `data/intent_parser_splits/train.jsonl` -- 9,000 rows, what models train on
- `data/routing_eval_golden_1k.jsonl` -- 1,000 rows, held out

Do not train, tune thresholds, or select checkpoints on the held-out file.

## Train

```bash
uv sync --extra bert
uv run --extra bert python models/intent_bert/scripts/train.py --config models/intent_bert/configs/v1.json
```

The trainer writes run metrics under `models/intent_bert/runs/v1/` and exports
the serving artifact under `models/intent_bert/exported/v1/`.

## Evaluate On Leaderboard

```bash
RR_INTENT_PARSER=bert \
RR_INTENT_PARSER_MODEL=models/intent_bert/exported/v1 \
uv run --extra bert reliability-router evaluate \
  --output artifacts/eval_report_bert_v1_1k.json \
  --html artifacts/eval_report_bert_v1_1k.html
```
