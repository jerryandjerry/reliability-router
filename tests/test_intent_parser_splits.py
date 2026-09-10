"""The train/eval split: no leakage, and an eval set that looks like the whole set."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPLIT_SCRIPT = ROOT / "data" / "intent_parser_splits" / "build_train_dev_split.py"
MANIFEST = ROOT / "data" / "intent_parser_splits" / "split_manifest.json"
TRAIN = ROOT / "data" / "intent_parser_splits" / "train.jsonl"
EVAL = ROOT / "data" / "routing_eval_golden_1k.jsonl"


def _load_split_module():
    spec = importlib.util.spec_from_file_location("build_train_dev_split", SPLIT_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def test_grouping_ignores_source_and_keeps_route_and_types() -> None:
    module = _load_split_module()
    case = {
        "expected_route": "DEEP",
        "tags": ["source:pubmedqa", "type:high_stakes", "type:multi_step"],
    }
    key = module.group_key(case)

    assert key[0] == "DEEP"
    assert set(key[1:]) == {"type:high_stakes", "type:multi_step"}
    assert not any("source" in part for part in key), "a router never sees the corpus"


def test_group_key_is_order_insensitive() -> None:
    module = _load_split_module()
    a = {"expected_route": "QUICK", "tags": ["type:numerical", "type:multi_step"]}
    b = {"expected_route": "QUICK", "tags": ["type:multi_step", "type:numerical"]}
    assert module.group_key(a) == module.group_key(b)


def test_row_ordering_is_deterministic() -> None:
    module = _load_split_module()
    case = {"id": "squad_train_1"}
    assert module.stable_order(case) == module.stable_order(case)
    assert module.stable_order(case) != module.stable_order({"id": "squad_train_2"})


@pytest.mark.skipif(not MANIFEST.exists(), reason="split not built")
def test_manifest_records_no_dev_split_and_no_overlap() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["source_cases"] == manifest["train_cases"] + manifest["heldout_cases"]
    assert manifest["dev_cases"] == 0, "models train on everything that is not held out"
    assert manifest["overlap"]["train_heldout"] == 0


@pytest.mark.skipif(not (TRAIN.exists() and EVAL.exists()), reason="split not built")
def test_no_row_appears_in_both_splits() -> None:
    train_ids = {r["id"] for r in _rows(TRAIN)}
    eval_ids = {r["id"] for r in _rows(EVAL)}

    assert not (train_ids & eval_ids), "a model must never be scored on what it trained on"


@pytest.mark.skipif(not (TRAIN.exists() and EVAL.exists()), reason="split not built")
def test_eval_set_mirrors_the_population() -> None:
    """The eval set is a miniature of the whole, not a tour of its rare corners.

    An earlier sampler took one row per group regardless of group size, which left the eval set
    at 65.7% DEEP against a population of 55.9%.
    """
    train, evaluation = _rows(TRAIN), _rows(EVAL)
    everything = train + evaluation

    def deep_share(rows: list[dict]) -> float:
        return sum(1 for r in rows if r["expected_route"] == "DEEP") / len(rows)

    assert abs(deep_share(evaluation) - deep_share(everything)) < 0.02

    base = Counter(t for r in everything for t in r["tags"] if t.startswith("type:"))
    got = Counter(t for r in evaluation for t in r["tags"] if t.startswith("type:"))
    for tag, count in base.items():
        drift = abs(got[tag] / len(evaluation) - count / len(everything))
        assert drift < 0.02, f"{tag} drifts {100*drift:.1f} points in the eval set"
