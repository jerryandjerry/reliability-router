"""TF-IDF parser: identity comes from the manifest, threshold decides the route."""

import json

import pytest

from reliability_router.config import RouterConfig
from reliability_router.intent import TfidfIntentParser, build_intent_parser
from reliability_router.models import ContextItem, Route, RouterRequest


def _manifest(tmp_path, **overrides):
    model_dir = tmp_path / "logreg_v1"
    model_dir.mkdir()
    payload = {"family": "tfidf", "name": "tfidf_logreg", "version": "v1", "threshold": 0.5}
    payload.update(overrides)
    (model_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return model_dir


def _request(query="What dose should I take?"):
    return RouterRequest(
        request_id="t1",
        query=query,
        context=[ContextItem(id="c1", text="Some supporting text.", source="wiki", trust=0.85)],
        available_tools=[],
    )


def test_identity_comes_from_the_manifest(tmp_path):
    model_dir = _manifest(tmp_path, name="tfidf_svm", version="v3")
    parser = TfidfIntentParser(
        RouterConfig(intent_parser="tfidf", intent_parser_model=str(model_dir)),
        classifier=lambda text: 0.9,
    )
    assert parser.name == "tfidf_svm"
    assert parser.version == "v3"


def test_manifest_requires_name_and_version(tmp_path):
    missing_version = _manifest(tmp_path, name="tfidf_logreg")
    (missing_version / "manifest.json").write_text(json.dumps({"name": "tfidf_logreg"}), encoding="utf-8")

    with pytest.raises(ValueError, match="version"):
        TfidfIntentParser(
            RouterConfig(intent_parser="tfidf", intent_parser_model=str(missing_version)),
            classifier=lambda text: 0.5,
        )


def test_threshold_maps_probability_to_route(tmp_path):
    model_dir = _manifest(tmp_path, threshold=0.6)
    config = RouterConfig(intent_parser="tfidf", intent_parser_model=str(model_dir))

    below = TfidfIntentParser(config, classifier=lambda text: 0.59).parse(_request())
    at = TfidfIntentParser(config, classifier=lambda text: 0.60).parse(_request())

    assert below.route is Route.QUICK
    assert at.route is Route.DEEP, "the threshold is inclusive, matching the other families"
    assert at.threshold == 0.6


def test_probability_is_reported_as_the_risk_score(tmp_path):
    model_dir = _manifest(tmp_path)
    parser = TfidfIntentParser(
        RouterConfig(intent_parser="tfidf", intent_parser_model=str(model_dir)),
        classifier=lambda text: 0.8123456,
    )
    decision = parser.parse(_request())
    assert decision.risk_score == pytest.approx(0.8123, abs=1e-4)
    assert "tfidf_logreg/v1" in decision.explanation


def test_traces_time_the_classifier_separately(tmp_path):
    model_dir = _manifest(tmp_path)
    parser = TfidfIntentParser(
        RouterConfig(intent_parser="tfidf", intent_parser_model=str(model_dir)),
        classifier=lambda text: 0.4,
    )
    _, traces = parser.parse_with_traces(_request())
    stages = [t.stage for t in traces]
    assert "intent.tfidf_logreg.classify" in stages
    assert "intent.tfidf_logreg.extract_features" in stages


def test_missing_artifact_says_how_to_build_it(tmp_path):
    model_dir = _manifest(tmp_path)  # manifest but no pipeline.joblib
    with pytest.raises(ValueError, match="Train/export a model"):
        TfidfIntentParser(RouterConfig(intent_parser="tfidf", intent_parser_model=str(model_dir)))


def test_factory_builds_the_family(tmp_path, monkeypatch):
    model_dir = _manifest(tmp_path)
    monkeypatch.setattr(
        "reliability_router.intent.factory.TfidfIntentParser",
        lambda config: TfidfIntentParser(config, classifier=lambda text: 0.1),
    )
    parser = build_intent_parser(
        RouterConfig(intent_parser="tfidf", intent_parser_model=str(model_dir))
    )
    assert parser.name == "tfidf_logreg"


def test_factory_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="tfidf"):
        build_intent_parser(RouterConfig(intent_parser="nonsense"))
