"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import RouterConfig
from .eval.runner import (
    DEFAULT_EVAL_DATASET,
    DatasetMismatch,
    append_leaderboard_entry,
    evaluate_routes,
    load_jsonl,
    select_eval_cases,
    write_html_report,
    write_json_report,
)
from .models import ContextItem, RouterRequest
from .service import ReliabilityRouter


# load context from file to ContextItem, only cli loads context from file, api load context inline
def _context_from_file(path: str | None) -> list[ContextItem]:
    if not path:
        return []
    value: Any = json.loads(Path(path).read_text(encoding="utf-8")) # Any is a type hint
    if not isinstance(value, list): # context file has to be a list of values
        raise ValueError("Context file must contain a JSON array")
    
    return [ContextItem.model_validate(item) for item in value]

# turns parsed CLI arguments into one validated RouterRequest
def _request(args: argparse.Namespace) -> RouterRequest:
    tools = [item.strip() for item in (args.tools or "").split(",") if item.strip()]
    return RouterRequest(query=args.query, context=_context_from_file(args.context), available_tools=tools)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reliability-router", description="Risk-aware QUICK/DEEP LLM router"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("route", "answer"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--query", required=True)
        sub.add_argument("--context", help="Path to a JSON array of context items")
        sub.add_argument("--tools", default="", help="Comma-separated available tool names")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset", default=DEFAULT_EVAL_DATASET)
    evaluate.add_argument("--output", default="artifacts/eval_report.json")
    evaluate.add_argument("--html", default="artifacts/eval_report.html")
    evaluate.add_argument("--leaderboard", default="artifacts/leaderboard.json",
                          help="Append-only JSON record of every evaluation run.")
    evaluate.add_argument("--sample-size", type=int, help="Evaluate a deterministic subset of cases.")
    evaluate.add_argument("--stratified", action="store_true",
                          help="Sample by source/origin quotas, then route, tag axes, and context-size strata.")
    evaluate.add_argument("--workers", default=1, type=int,
                          help="Number of concurrent eval workers. Latency is still measured per sample.")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", default=8000, type=int)
    serve.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("reliability_router.api:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    router = ReliabilityRouter(config=RouterConfig.from_env())
    if args.command == "route":
        # intent parse only
        print(router.route(_request(args)).model_dump_json(indent=2))
        return 0
    if args.command == "answer":
        print(router.answer(_request(args)).model_dump_json(indent=2))
        return 0
    if args.command == "evaluate":
        cases = select_eval_cases(load_jsonl(args.dataset), sample_size=args.sample_size, stratified=args.stratified)
        dataset_name = str(args.dataset)
        if args.sample_size is not None:
            suffix = f":sample-{args.sample_size}"
            if args.stratified:
                suffix += "-stratified"
            dataset_name += suffix
        summary = evaluate_routes(router, cases, dataset_name=dataset_name, workers=args.workers)
        write_json_report(summary, args.output)
        write_html_report(summary, args.html)
        # "none" disables recording; without this the path is taken literally and a file
        # called "none" is written into the repository root.
        if str(args.leaderboard).strip().lower() != "none":
            try:
                append_leaderboard_entry(
                    summary,
                    name=router.intent_parser.name,
                    version=router.intent_parser.version,
                    path=Path(args.leaderboard),
                )
            except DatasetMismatch as exc:
                # the evaluation itself is valid -- only the leaderboard row is refused
                print(f"warning: not recorded on the leaderboard. {exc}", file=sys.stderr)
        print(summary.model_dump_json(indent=2))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
