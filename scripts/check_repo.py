"""Small repository integrity check used before release."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
DATASET = "data/routing_eval_golden_1k.jsonl"
REQUIRED = [
    "README.md",
    "pyproject.toml",
    "src/reliability_router/service.py",
    DATASET,
    "tests/test_service.py",
]


def main() -> int:
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    dataset_path = ROOT / DATASET
    lines = [
        line
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for line in lines:
        json.loads(line)
    print(f"Repository check passed: {len(REQUIRED)} required files, {len(lines)} eval cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
