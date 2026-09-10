from pathlib import Path

import pytest

from reliability_router.eval.runner import (
    DEFAULT_EVAL_DATASET,
    evaluate_routes,
    load_jsonl,
    select_eval_cases,
)
from reliability_router.models import EvalCase, Route
from reliability_router.service import ReliabilityRouter

SMOKE_DATASET = Path(__file__).parents[1] / "data" / "routing_eval.jsonl"


def test_default_eval_dataset_points_to_official_golden_set() -> None:
    assert DEFAULT_EVAL_DATASET == "data/routing_eval_golden_1k.jsonl"


def test_smoke_eval_dataset_quality() -> None:
    summary = evaluate_routes(ReliabilityRouter(), load_jsonl(SMOKE_DATASET), dataset_name=str(SMOKE_DATASET))
    assert summary.total >= 20
    assert summary.route_accuracy >= 0.95
    assert summary.deep_recall >= 0.95


def test_parallel_eval_matches_serial_predictions() -> None:
    cases = load_jsonl(SMOKE_DATASET)
    router = ReliabilityRouter()

    serial = evaluate_routes(router, cases, dataset_name=str(SMOKE_DATASET), workers=1)
    parallel = evaluate_routes(router, cases, dataset_name=str(SMOKE_DATASET), workers=4)

    assert parallel.total == serial.total
    assert parallel.correct == serial.correct
    assert parallel.route_accuracy == serial.route_accuracy
    assert [prediction.case_id for prediction in parallel.predictions] == [
        prediction.case_id for prediction in serial.predictions
    ]
    assert [prediction.predicted_route for prediction in parallel.predictions] == [
        prediction.predicted_route for prediction in serial.predictions
    ]


def test_stratified_sample_round_robins_tag_axes() -> None:
    cases = [
        EvalCase(id="a0", query="a", expected_route=Route.QUICK, tags=["source:squad", "origin:real", "risk:single-hop"]),
        EvalCase(id="a1", query="a", expected_route=Route.QUICK, tags=["source:squad", "origin:real", "risk:single-hop"]),
        EvalCase(id="b0", query="b", expected_route=Route.QUICK, tags=["source:finqa", "origin:real", "risk:numerical"]),
        EvalCase(id="b1", query="b", expected_route=Route.QUICK, tags=["source:finqa", "origin:real", "risk:numerical"]),
        EvalCase(
            id="c0",
            query="c",
            expected_route=Route.DEEP,
            tags=["source:pubmedqa", "origin:real", "risk:high-stakes-medical"],
        ),
        EvalCase(
            id="c1",
            query="c",
            expected_route=Route.DEEP,
            tags=["source:pubmedqa", "origin:real", "risk:high-stakes-medical"],
        ),
        EvalCase(
            id="d0",
            query="d",
            expected_route=Route.DEEP,
            tags=["source:squad", "origin:synthetic", "transform:injection", "attack:jailbreak"],
        ),
        EvalCase(
            id="d1",
            query="d",
            expected_route=Route.DEEP,
            tags=["source:squad", "origin:synthetic", "transform:injection", "attack:jailbreak"],
        ),
    ]

    sample = select_eval_cases(cases, sample_size=4, stratified=True)

    assert {case.id for case in sample} == {"a0", "b0", "c0", "d0"}


def test_eval_rejects_invalid_worker_and_sample_counts() -> None:
    cases = load_jsonl(SMOKE_DATASET)

    with pytest.raises(ValueError, match="workers"):
        evaluate_routes(ReliabilityRouter(), cases, dataset_name=str(SMOKE_DATASET), workers=0)
    with pytest.raises(ValueError, match="sample_size"):
        select_eval_cases(cases, sample_size=0)


def test_by_tag_breakdown_reconciles_with_totals() -> None:
    cases = load_jsonl(SMOKE_DATASET)
    summary = evaluate_routes(ReliabilityRouter(), cases, dataset_name=str(SMOKE_DATASET))

    assert summary.by_tag, "expected a per-tag breakdown"
    assert set(summary.by_tag) == {tag for case in cases for tag in case.tags}

    for tag, breakdown in summary.by_tag.items():
        tagged = [case for case in cases if tag in case.tags]
        assert breakdown.total == len(tagged)
        assert breakdown.expected_deep + breakdown.expected_quick == breakdown.total
        assert breakdown.caught_deep <= breakdown.expected_deep
        if breakdown.expected_deep:
            assert breakdown.deep_recall == round(breakdown.caught_deep / breakdown.expected_deep, 4)
        else:
            assert breakdown.deep_recall is None
