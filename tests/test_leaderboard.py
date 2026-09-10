import json
from pathlib import Path

import pytest

from reliability_router.eval.runner import DatasetMismatch, append_leaderboard_entry
from reliability_router.models import EvalSummary


def _summary(acc: float = 0.5, f1: float = 0.4, p95: float = 1.5) -> EvalSummary:
    return EvalSummary(
        dataset="data/routing_eval_golden_1k.jsonl",
        policy_version="v1",
        total=10,
        correct=5,
        route_accuracy=acc,
        deep_precision=0.6,
        deep_recall=0.3,
        deep_f1=f1,
        confusion_matrix={"deep_as_deep": 3, "quick_as_deep": 2, "quick_as_quick": 2, "deep_as_quick": 3},
        p50_latency_ms=0.5,
        p95_latency_ms=p95,
        reason_code_counts={},
        by_tag={},
        predictions=[],
    )


def test_leaderboard_json_accumulates_runs(tmp_path: Path) -> None:
    board = tmp_path / "leaderboard.json"

    first = append_leaderboard_entry(_summary(acc=0.52), name="regex", version="v1", path=board)
    second = append_leaderboard_entry(_summary(acc=0.81), name="bert", version="v1", path=board)

    assert [entry["name"] for entry in first] == ["regex"]
    assert [entry["name"] for entry in second] == ["regex", "bert"]
    assert json.loads(board.read_text(encoding="utf-8")) == second


def test_leaderboard_entry_contains_ranking_metrics(tmp_path: Path) -> None:
    board = tmp_path / "leaderboard.json"
    entry = append_leaderboard_entry(
        _summary(acc=0.52, f1=0.33, p95=0.49),
        name="regex",
        version="v1",
        path=board,
    )[0]

    assert entry["id"] == 1
    assert entry["name"] == "regex"
    assert entry["version"] == "v1"
    assert "policy_version" not in entry
    assert entry["accuracy"] == 0.52
    assert entry["deep_f1"] == 0.33
    assert entry["p95_latency_ms"] == 0.49
    assert entry["run_at"]


def test_legacy_entries_get_stable_ids(tmp_path: Path) -> None:
    board = tmp_path / "leaderboard.json"
    board.write_text(json.dumps([{"name": "old-run", "accuracy": 0.4}]), encoding="utf-8")

    entries = append_leaderboard_entry(_summary(), name="new-run", version="v1", path=board)

    assert [entry["id"] for entry in entries] == [1, 2]


def test_invalid_leaderboard_json_fails_loudly(tmp_path: Path) -> None:
    board = tmp_path / "leaderboard.json"
    board.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid leaderboard JSON"):
        append_leaderboard_entry(_summary(), name="run", version="v1", path=board)


def test_parser_identity_is_required_for_leaderboard(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="name"):
        append_leaderboard_entry(_summary(), name="", version="v1", path=tmp_path / "board.json")
    with pytest.raises(ValueError, match="version"):
        append_leaderboard_entry(_summary(), name="regex", version="", path=tmp_path / "board.json")


def test_board_refuses_a_run_scored_on_another_dataset(tmp_path: Path) -> None:
    """A ranking only means something when every row faced the same cases.

    Regression: the guard lived in eval/leaderboard.py and was dropped when that module was
    folded into runner.py, leaving the README documenting a guard the code no longer had.
    """
    board = tmp_path / "leaderboard.json"
    append_leaderboard_entry(_summary(), name="regex", version="v1", path=board)

    other = _summary()
    other.dataset = "data/smoke_24.jsonl"
    with pytest.raises(DatasetMismatch) as caught:
        append_leaderboard_entry(other, name="tfidf", version="v1", path=board)

    assert "smoke_24" in str(caught.value)
    assert len(json.loads(board.read_text(encoding="utf-8"))) == 1, "the row must not be appended"


def test_matching_dataset_still_appends(tmp_path: Path) -> None:
    """The complement: same dataset, so the rows are comparable and both are kept."""
    board = tmp_path / "leaderboard.json"
    append_leaderboard_entry(_summary(), name="regex", version="v1", path=board)
    entries = append_leaderboard_entry(_summary(acc=0.8), name="tfidf", version="v1", path=board)

    assert [entry["name"] for entry in entries] == ["regex", "tfidf"]


def test_static_leaderboard_viewer_loads_json() -> None:
    html = (Path(__file__).parents[1] / "artifacts" / "leaderboard.html").read_text(encoding="utf-8")

    assert 'fetch("leaderboard.json"' in html
    assert 'id="leaderboard-data"' in html
    assert 'id="entries"' not in html
    assert "<th>Run at</th>" not in html
    assert "DEEP precision" in html
    assert "DEEP recall" in html
    assert "p50 latency" in html


def test_leaderboard_json_syncs_static_html_snapshot(tmp_path: Path) -> None:
    board = tmp_path / "leaderboard.json"
    html = tmp_path / "leaderboard.html"
    html.write_text(
        '<script type="application/json" id="leaderboard-data">\n[]\n  </script>',
        encoding="utf-8",
    )

    append_leaderboard_entry(_summary(), name="regex", version="v1", path=board)

    assert '"name": "regex"' in html.read_text(encoding="utf-8")
