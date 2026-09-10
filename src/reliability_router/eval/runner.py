"""Deterministic routing evaluation harness."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from ..models import EvalCase, EvalPrediction, EvalSummary, Route, RouterRequest, TagBreakdown
from ..service import ReliabilityRouter

DEFAULT_EVAL_DATASET = "data/routing_eval_golden_1k.jsonl"


def load_jsonl(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(EvalCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not cases:
        raise ValueError(f"Dataset {path} contains no cases")
    return cases


def select_eval_cases(cases: list[EvalCase], *, sample_size: int | None = None, stratified: bool = False) -> list[EvalCase]:
    """Select a deterministic sample.

    Stratified sampling follows the golden-set construction style: allocate
    proportional quotas by source and origin, then round-robin within each quota
    across route, the remaining tag axes, and a context-size bucket.
    """

    if sample_size is None:
        return cases
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not stratified:
        return cases[:sample_size]

    by_group: dict[tuple[str, str], list[EvalCase]] = defaultdict(list)
    for case in cases:
        by_group[_primary_stratification_axes(case)].append(case)

    quotas = _proportional_quotas({group: len(items) for group, items in by_group.items()}, sample_size)
    selections = {
        group: _round_robin_strata(items, quotas[group])
        for group, items in by_group.items()
        if quotas.get(group, 0) > 0
    }

    selected: list[EvalCase] = []
    groups = sorted(selections, key=lambda group: _stable_text_order("\x1f".join(group)))
    offsets = {group: 0 for group in groups}
    while len(selected) < min(sample_size, len(cases)):
        progressed = False
        for group in groups:
            index = offsets[group]
            if index < len(selections[group]):
                selected.append(selections[group][index])
                offsets[group] = index + 1
                progressed = True
                if len(selected) >= sample_size:
                    break
        if not progressed:
            break
    return selected


def _proportional_quotas(counts: dict[Any, int], sample_size: int) -> dict[Any, int]:
    total = sum(counts.values())
    if sample_size >= total:
        return counts.copy()

    raw = {key: sample_size * count / total for key, count in counts.items()}
    quotas = {key: min(counts[key], int(value)) for key, value in raw.items()}
    remaining = sample_size - sum(quotas.values())
    remainders = sorted(
        counts,
        key=lambda key: (raw[key] - int(raw[key]), _stable_text_order(str(key))),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in remainders:
            if quotas[key] >= counts[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def _round_robin_strata(cases: list[EvalCase], sample_size: int) -> list[EvalCase]:
    groups: dict[tuple[str, ...], list[EvalCase]] = defaultdict(list)
    for case in cases:
        groups[_eval_stratum_key(case)].append(case)

    selected: list[EvalCase] = []
    keys = sorted(groups, key=_stable_stratum_order)
    offsets = {key: 0 for key in keys}
    while len(selected) < min(sample_size, len(cases)):
        progressed = False
        for key in keys:
            index = offsets[key]
            if index < len(groups[key]):
                selected.append(groups[key][index])
                offsets[key] = index + 1
                progressed = True
                if len(selected) >= sample_size:
                    break
        if not progressed:
            break
    return selected


def _eval_stratum_key(case: EvalCase) -> tuple[str, ...]:
    return (
        f"route:{case.expected_route.value}",
        *[tag for tag in _normalized_tag_axes(case.tags) if not tag.startswith("source:")],
        f"ctx:{_context_size_bucket(case)}",
    )


def _primary_stratification_axes(case: EvalCase) -> tuple[str, str]:
    return (f"source:{_source_axis(case)}", f"origin:{_origin_axis(case)}")


def _source_axis(case: EvalCase) -> str:
    for tag in case.tags:
        if tag.startswith("source:"):
            return tag.partition(":")[2]
    return "unknown"


def _origin_axis(case: EvalCase) -> str:
    for tag in case.tags:
        if tag.startswith("origin:"):
            return tag.partition(":")[2]
    return "unknown"


def _normalized_tag_axes(tags: list[str]) -> tuple[str, ...]:
    return tuple(sorted(tags)) or ("tags:none",)


def _context_size_bucket(case: EvalCase) -> str:
    chars = sum(len(item.text) for item in case.context)
    for edge, name in ((500, "xs"), (2000, "s"), (5000, "m"), (12000, "l")):
        if chars < edge:
            return name
    return "xl"


def _stable_stratum_order(key: tuple[str, ...]) -> str:
    return _stable_text_order("\x1f".join(key))


def _stable_text_order(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate_routes(
    router: ReliabilityRouter,
    cases: list[EvalCase],
    dataset_name: str,
    *,
    workers: int = 1,
) -> EvalSummary:
    if workers <= 0:
        raise ValueError("workers must be positive")

    predictions = _evaluate_serial(router, cases) if workers == 1 else _evaluate_parallel(router, cases, workers)
    reason_counts: Counter[str] = Counter()
    latencies: list[float] = []
    for prediction in predictions:
        latencies.append(prediction.latency_ms)
        reason_counts.update(code.value for code in prediction.reason_codes)

    tp = sum(p.expected_route == Route.DEEP and p.predicted_route == Route.DEEP for p in predictions)
    fp = sum(p.expected_route == Route.QUICK and p.predicted_route == Route.DEEP for p in predictions)
    tn = sum(p.expected_route == Route.QUICK and p.predicted_route == Route.QUICK for p in predictions)
    fn = sum(p.expected_route == Route.DEEP and p.predicted_route == Route.QUICK for p in predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    correct = tp + tn

    return EvalSummary(
        dataset=dataset_name,
        policy_version=router.config.policy_version,
        total=len(predictions),
        correct=correct,
        route_accuracy=round(correct / len(predictions), 4),
        deep_precision=round(precision, 4),
        deep_recall=round(recall, 4),
        deep_f1=round(f1, 4),
        confusion_matrix={"deep_as_deep": tp, "quick_as_deep": fp, "quick_as_quick": tn, "deep_as_quick": fn},
        p50_latency_ms=round(_percentile(latencies, 50), 4),
        p95_latency_ms=round(_percentile(latencies, 95), 4),
        reason_code_counts=dict(sorted(reason_counts.items())),
        by_tag=_tag_breakdown(predictions),
        predictions=predictions,
    )


def _evaluate_one(router: ReliabilityRouter, case: EvalCase) -> EvalPrediction:
    request = RouterRequest(
        request_id=f"eval_{case.id}",
        query=case.query,
        context=case.context,
        available_tools=case.available_tools,
    )
    start = perf_counter()
    decision = router.route(request)
    latency_ms = (perf_counter() - start) * 1000
    return EvalPrediction(
        case_id=case.id,
        expected_route=case.expected_route,
        predicted_route=decision.route,
        correct=decision.route == case.expected_route,
        risk_score=decision.risk_score,
        reason_codes=decision.reason_codes,
        latency_ms=round(latency_ms, 4),
        tags=case.tags,
    )


def _evaluate_serial(router: ReliabilityRouter, cases: list[EvalCase]) -> list[EvalPrediction]:
    return [_evaluate_one(router, case) for case in cases]


def _evaluate_parallel(router: ReliabilityRouter, cases: list[EvalCase], workers: int) -> list[EvalPrediction]:
    predictions: list[EvalPrediction | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_one, router, case): index for index, case in enumerate(cases)}
        for future in as_completed(futures):
            predictions[futures[future]] = future.result()
    return [prediction for prediction in predictions if prediction is not None]


def _tag_breakdown(predictions: list[EvalPrediction]) -> dict[str, TagBreakdown]:
    """Slice the same metrics per tag, so a global score can be traced to its source.

    A case counts toward every tag it carries, so slices overlap by design.
    """

    buckets: dict[str, list[EvalPrediction]] = {}
    for prediction in predictions:
        for tag in prediction.tags:
            buckets.setdefault(tag, []).append(prediction)

    breakdown: dict[str, TagBreakdown] = {}
    for tag, items in sorted(buckets.items()):
        expected_deep = [p for p in items if p.expected_route == Route.DEEP]
        expected_quick = [p for p in items if p.expected_route == Route.QUICK]
        caught = sum(p.predicted_route == Route.DEEP for p in expected_deep)
        breakdown[tag] = TagBreakdown(
            total=len(items),
            correct=sum(p.correct for p in items),
            accuracy=round(sum(p.correct for p in items) / len(items), 4),
            expected_deep=len(expected_deep),
            caught_deep=caught,
            deep_recall=round(caught / len(expected_deep), 4) if expected_deep else None,
            expected_quick=len(expected_quick),
            false_deep=sum(p.predicted_route == Route.DEEP for p in expected_quick),
        )
    return breakdown


def write_json_report(summary: EvalSummary, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


class DatasetMismatch(Exception):
    """Raised when a run would be added to a board built on a different dataset."""

    def __init__(self, board_dataset: str, run_dataset: str, path: Path):
        self.board_dataset = board_dataset
        self.run_dataset = run_dataset
        super().__init__(
            f"leaderboard {path} scores '{board_dataset}'; this run scored '{run_dataset}'. "
            f"Rows are only comparable within one dataset. Re-run against "
            f"'{board_dataset}', or pass --leaderboard <other-path> to keep a separate board."
        )


def append_leaderboard_entry(
    summary: EvalSummary,
    *,
    name: str,
    version: str,
    path: str | Path,
) -> list[dict[str, Any]]:
    """Append one evaluation summary to the JSON leaderboard record.

    Raises DatasetMismatch if the board already holds runs on a different dataset.
    """

    if not name.strip():
        raise ValueError("name must be a non-empty string")
    if not version.strip():
        raise ValueError("version must be a non-empty string")

    output = Path(path)
    entries: list[dict[str, Any]] = []
    if output.exists():
        try:
            loaded = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid leaderboard JSON at {output}: {exc}") from exc
        if not isinstance(loaded, list):
            raise ValueError(f"Leaderboard JSON at {output} must contain an array")
        entries = loaded

    # One board, one dataset -- otherwise the ranking compares runs that never faced the same
    # cases. A 24-case smoke set scores 1.000 on everything and sits permanently at the top.
    existing_dataset = next((e["dataset"] for e in entries if isinstance(e, dict) and e.get("dataset")), None)
    if existing_dataset is not None and existing_dataset != summary.dataset:
        raise DatasetMismatch(existing_dataset, summary.dataset, output)

    for index, entry in enumerate(entries, start=1):
        if isinstance(entry, dict):
            entry.setdefault("id", index)

    next_id = max(
        (entry.get("id") for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), int)),
        default=0,
    ) + 1
    entries.append(
        {
            "id": next_id,
            "name": name,
            "version": version,
            "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": summary.dataset,
            "cases": summary.total,
            "accuracy": summary.route_accuracy,
            "deep_f1": summary.deep_f1,
            "deep_precision": summary.deep_precision,
            "deep_recall": summary.deep_recall,
            "p50_latency_ms": summary.p50_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    _sync_static_leaderboard_html(output, entries)
    return entries


def _sync_static_leaderboard_html(leaderboard_json: Path, entries: list[dict[str, Any]]) -> None:
    """Mirror the JSON board into sibling HTML so it works from file://."""

    html_path = leaderboard_json.with_suffix(".html")
    if not html_path.exists():
        return

    marker = '<script type="application/json" id="leaderboard-data">\n'
    html = html_path.read_text(encoding="utf-8")
    start = html.find(marker)
    if start == -1:
        return
    data_start = start + len(marker)
    data_end = html.find("\n  </script>", data_start)
    if data_end == -1:
        return

    payload = json.dumps(entries, indent=2).replace("</", "<\\/")
    updated = html[:data_start] + payload + html[data_end:]
    html_path.write_text(updated, encoding="utf-8")


#: dimension -> (heading, blurb). Order sets the page order; anything else lands in "More".
_DIMENSIONS: list[tuple[str, str, str]] = [
    ("type", "Risk feature", "The router feature the case is built to exercise."),
    ("source", "Source corpus", "Which dataset the request came from."),
]


def _bar_rows(items: list[tuple[str, object]]) -> str:
    """Horizontal part-to-whole bars: filled = DEEP caught, track = missed."""
    out = []
    for value, b in items:
        pct = 0.0 if b.deep_recall is None else b.deep_recall * 100
        missed = b.expected_deep - b.caught_deep
        tip = (f"{value} · {b.total} cases · accuracy {b.accuracy:.0%} · "
               f"{b.caught_deep}/{b.expected_deep} DEEP caught · {missed} missed · "
               f"{b.false_deep} false DEEP")
        out.append(
            f'<div class="row" tabindex="0" data-tip="{_esc(tip)}">'
            f'<div class="rl">{_esc(value)}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="rv"><b>{pct:.0f}%</b><span class="sub">{b.caught_deep}/{b.expected_deep}</span></div>'
            f"</div>"
        )
    return "".join(out)


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def write_html_report(summary: EvalSummary, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[tuple[str, object]]] = {}
    for tag, b in summary.by_tag.items():
        dim, _, value = tag.partition(":")
        grouped.setdefault(dim if value else "other", []).append((value or tag, b))

    def section(dim: str, heading: str, blurb: str) -> str:
        items = [(v, b) for v, b in grouped.get(dim, []) if b.expected_deep]
        if not items:
            return ""
        items.sort(key=lambda x: (x[1].deep_recall or 0, -x[1].total))
        return (f'<section class="panel"><h3>{_esc(heading)}</h3>'
                f'<p class="blurb">{_esc(blurb)}</p>'
                f'<div class="bars">{_bar_rows(items)}</div></section>')

    named = {d for d, _, _ in _DIMENSIONS}
    panels = "".join(section(d, h, s) for d, h, s in _DIMENSIONS)
    rest = sorted(d for d in grouped if d not in named)
    more = "".join(section(d, d.replace("-", " ").capitalize(), "") for d in rest)
    more_block = (f'<details class="more"><summary>{len(rest)} more dimensions</summary>'
                  f'<div class="grid">{more}</div></details>') if more else ""

    cm = summary.confusion_matrix
    biggest = max(cm.values()) or 1

    def cell(key: str, label: str) -> str:
        # one ramp for all four cells: a big error must not recede next to a big success
        n = cm.get(key, 0)
        return (f'<div class="cm" style="--w:{n / biggest:.3f}">'
                f'<div class="cmn">{n:,}</div><div class="cml">{label}</div></div>')

    pred_rows = "\n".join(
        "<tr>"
        f"<td class=mono>{_esc(p.case_id)}</td><td>{p.expected_route.value}</td>"
        f"<td>{p.predicted_route.value}</td>"
        f'<td class="{"ok" if p.correct else "bad"}">{"PASS" if p.correct else "FAIL"}</td>'
        f"<td class=num>{p.risk_score:.3f}</td>"
        f"<td class=mono>{_esc(', '.join(c.value for c in p.reason_codes))}</td>"
        "</tr>"
        for p in summary.predictions
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Routing evaluation · {_esc(summary.dataset.rsplit('/', 1)[-1])}</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--line:#c3c2b7;--ring:rgba(11,11,11,.10);--series:#2a78d6;--track:#eceae4}}
@media(prefers-color-scheme:dark){{:root:where(:not([data-theme=light])){{color-scheme:dark;
--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--line:#383835;--ring:rgba(255,255,255,.10);--series:#3987e5;--track:#2c2c2a}}}}
:root[data-theme=dark]{{color-scheme:dark;--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--line:#383835;--ring:rgba(255,255,255,.10);--series:#3987e5;--track:#2c2c2a}}
*{{box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--plane);color:var(--ink);
margin:0;padding:40px 20px 80px;line-height:1.5}}
.wrap{{max-width:1080px;margin:0 auto}}
h1{{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}}
h2{{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
margin:40px 0 14px;font-weight:600}}
h3{{font-size:15px;margin:0 0 2px}}
.meta{{color:var(--ink2);font-size:13px;margin:0 0 28px}}
.meta code{{background:var(--surface);border:1px solid var(--ring);border-radius:5px;padding:1px 6px;font-size:12px}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.tile{{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px 18px}}
.tile .k{{font-size:12px;color:var(--ink2);margin-bottom:6px}}
.tile .v{{font-size:32px;font-weight:650;letter-spacing:-.02em;line-height:1.1}}
.tile .n{{font-size:12px;color:var(--muted);margin-top:4px}}
.tile.lead{{border-color:var(--series);box-shadow:inset 0 0 0 1px var(--series)}}
.cmwrap{{display:grid;grid-template-columns:auto 1fr 1fr;gap:2px;align-items:stretch;
background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px}}
.cmh{{font-size:11px;color:var(--muted);display:flex;align-items:center;justify-content:center;padding:6px}}
.cm{{border-radius:8px;padding:14px 12px;background:
color-mix(in oklab,var(--series) calc(var(--w)*82%),var(--surface))}}
.cmn{{font-size:26px;font-weight:650;letter-spacing:-.02em}}
.cml{{font-size:11px;color:var(--ink2);margin-top:2px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.panel{{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px 18px 18px}}
.blurb{{font-size:12px;color:var(--muted);margin:0 0 14px}}
.bars{{display:flex;flex-direction:column;gap:6px}}
.row{{display:grid;grid-template-columns:150px 1fr 74px;gap:10px;align-items:center;
padding:2px 4px;border-radius:6px;outline:none;position:relative}}
.row:hover,.row:focus{{background:var(--plane)}}
.rl{{font-size:12.5px;color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.track{{height:11px;background:var(--track);border-radius:4px;overflow:hidden}}
.fill{{height:100%;background:var(--series);border-radius:0 4px 4px 0;min-width:2px}}
.rv{{font-size:12.5px;text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}}
.rv .sub{{display:block;font-size:11px;color:var(--muted)}}
.row[data-tip]:hover::after,.row[data-tip]:focus::after{{content:attr(data-tip);position:absolute;
left:150px;bottom:calc(100% + 6px);z-index:9;background:var(--ink);color:var(--surface);
font-size:11.5px;line-height:1.45;padding:7px 9px;border-radius:7px;max-width:420px;white-space:normal;
box-shadow:0 4px 14px rgba(0,0,0,.22)}}
details.more{{margin-top:12px}}
details.more>summary{{cursor:pointer;font-size:13px;color:var(--ink2);padding:8px 0}}
details.more[open]>summary{{margin-bottom:10px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;background:var(--surface);
border:1px solid var(--ring);border-radius:12px;overflow:hidden}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
padding:10px;border-bottom:1px solid var(--grid);font-weight:600}}
td{{padding:8px 10px;border-bottom:1px solid var(--grid);vertical-align:top}}
tr:last-child td{{border-bottom:0}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;color:var(--ink2)}}
.num{{font-variant-numeric:tabular-nums;text-align:right}}
.ok{{color:var(--ink2)}} .bad{{color:var(--ink);font-weight:600}}
@media(max-width:840px){{.tiles,.grid{{grid-template-columns:1fr 1fr}}.row{{grid-template-columns:110px 1fr 66px}}}}
@media(max-width:560px){{.tiles,.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">

<h1>Routing evaluation</h1>
<p class="meta"><code>{_esc(summary.dataset)}</code> · policy <code>{_esc(summary.policy_version)}</code>
 · {summary.total:,} cases</p>

<div class="tiles">
  <div class="tile"><div class="k">Route accuracy</div><div class="v">{summary.route_accuracy:.1%}</div>
    <div class="n">{summary.correct:,} of {summary.total:,} correct</div></div>
  <div class="tile lead"><div class="k">DEEP recall</div><div class="v">{summary.deep_recall:.1%}</div>
    <div class="n">{cm.get('deep_as_quick', 0):,} needed DEEP, got QUICK</div></div>
  <div class="tile"><div class="k">DEEP precision</div><div class="v">{summary.deep_precision:.1%}</div>
    <div class="n">{cm.get('quick_as_deep', 0):,} escalated unnecessarily</div></div>
  <div class="tile"><div class="k">P95 latency</div><div class="v">{summary.p95_latency_ms:.2f}<span
    style="font-size:16px"> ms</span></div><div class="n">p50 {summary.p50_latency_ms:.2f} ms</div></div>
</div>

<h2>Confusion</h2>
<div class="cmwrap">
  <div class="cmh"></div><div class="cmh">predicted DEEP</div><div class="cmh">predicted QUICK</div>
  <div class="cmh">should be&nbsp;DEEP</div>{cell('deep_as_deep', 'caught')}{cell('deep_as_quick', 'missed')}
  <div class="cmh">should be&nbsp;QUICK</div>{cell('quick_as_deep', 'over-escalated')}{cell('quick_as_quick', 'correct')}
</div>

<h2>DEEP recall by dimension</h2>
<p class="meta" style="margin:-6px 0 14px">Weakest first. A case carries one tag per dimension, so slices overlap between panels but never within one.</p>
<div class="grid">{panels}</div>
{more_block}

<h2>Cases</h2>
<details><summary style="cursor:pointer;font-size:13px;color:var(--ink2);padding:8px 0">
Show all {summary.total:,} predictions</summary>
<table><thead><tr><th>Case</th><th>Expected</th><th>Predicted</th><th>Result</th>
<th class=num>Risk</th><th>Reason codes</th></tr></thead><tbody>{pred_rows}</tbody></table>
</details>

</div></body></html>"""
    output.write_text(html, encoding="utf-8")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
