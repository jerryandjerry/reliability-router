import json

from reliability_router.cli import main


def test_cli_route(capsys) -> None:
    assert main(["route", "--query", "What is 2 + 2?"]) == 0
    assert '"route": "QUICK"' in capsys.readouterr().out


def _evaluate(tmp_path, leaderboard: str) -> None:
    assert main([
        "evaluate",
        "--dataset", "data/routing_eval_golden_1k.jsonl",
        "--sample-size", "3",
        "--output", str(tmp_path / "report.json"),
        "--html", str(tmp_path / "report.html"),
        "--leaderboard", leaderboard,
    ]) == 0


def test_leaderboard_none_records_nothing(tmp_path, monkeypatch, capsys) -> None:
    """`--leaderboard none` must disable recording, not write a file called "none".

    Regression: the guard was dropped in a CLI rewrite, so the sentinel was treated as a path and
    a file named "none" was written to the repository root -- and committed.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    # run from a scratch directory so a stray write is visible
    import os

    os.symlink(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        tmp_path / "data",
    )

    _evaluate(tmp_path, "none")
    capsys.readouterr()

    assert not (tmp_path / "none").exists(), "the sentinel was treated as a file path"
    assert not list(tmp_path.glob("*.jsonl")), "nothing else should have been written"


def test_leaderboard_path_records_one_entry(tmp_path, monkeypatch, capsys) -> None:
    """The complement: a real path still gets exactly one appended row."""
    monkeypatch.chdir(tmp_path)
    import os

    os.symlink(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        tmp_path / "data",
    )
    board = tmp_path / "board.json"

    _evaluate(tmp_path, str(board))
    capsys.readouterr()

    assert board.exists()
    entries = json.loads(board.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["name"] == "regex"
