"""Sanity-check an exported BERT classifier against the training split.

There is no dev split, so this reports fit on data the model has already seen. It is a check that
export worked, never a result -- the only real score comes from
`reliability-router evaluate --dataset data/routing_eval_golden_1k.jsonl`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliability_router.config import RouterConfig  # noqa: E402
from reliability_router.eval.runner import evaluate_routes, load_jsonl, write_json_report  # noqa: E402
from reliability_router.intent.bert import BertFamilyIntentParser  # noqa: E402
from reliability_router.service import ReliabilityRouter  # noqa: E402


def main() -> int:
    args = _parse_args()
    config = RouterConfig(intent_parser="bert", intent_parser_model=args.model)
    router = ReliabilityRouter(config=config, intent_parser=BertFamilyIntentParser(config))
    summary = evaluate_routes(router, load_jsonl(args.dev), dataset_name=args.dev, workers=args.workers)
    write_json_report(summary, args.output)
    print(summary.model_dump_json(indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit check for an exported BERT intent parser (not a result).")
    parser.add_argument("--model", default="models/intent_bert/exported/v1")
    parser.add_argument("--dev", default="data/intent_parser_splits/train.jsonl")
    parser.add_argument("--output", default="models/intent_bert/runs/v1/fit_check.json")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
