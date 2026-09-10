"""Embedding parser: one runtime, two variants, identity from the manifest."""

import json

import pytest

from reliability_router.config import RouterConfig
from reliability_router.intent import EmbeddingIntentParser, build_intent_parser
from reliability_router.models import ContextItem, Route, RouterRequest


def _manifest(tmp_path, **overrides):
    model_dir = tmp_path / "cls_v1"
    model_dir.mkdir()
    payload = {
        "family": "embedding",
        "name": "embedding_cls",
        "version": "v1",
        "variant": "cls",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "threshold": 0.5,
    }
    payload.update(overrides)
    (model_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return model_dir


def _request():
    return RouterRequest(
        request_id="e1",
        query="Should I stop this medication?",
        context=[ContextItem(id="c1", text="Clinical note.", source="pubmed", trust=0.9)],
        available_tools=[],
    )


def test_identity_and_variant_come_from_the_manifest(tmp_path):
    model_dir = _manifest(tmp_path, name="embedding_knn", variant="knn", version="v2")
    parser = EmbeddingIntentParser(
        RouterConfig(intent_parser="embedding", intent_parser_model=str(model_dir)),
        classifier=lambda text: 0.7,
    )
    assert parser.name == "embedding_knn"
    assert parser.variant == "knn"
    assert parser.version == "v2"


def test_manifest_requires_name_and_version(tmp_path):
    model_dir = _manifest(tmp_path)
    (model_dir / "manifest.json").write_text(json.dumps({"name": "embedding_cls"}), encoding="utf-8")

    with pytest.raises(ValueError, match="version"):
        EmbeddingIntentParser(
            RouterConfig(intent_parser="embedding", intent_parser_model=str(model_dir)),
            classifier=lambda t: 0.5,
        )


def test_threshold_maps_probability_to_route(tmp_path):
    model_dir = _manifest(tmp_path, threshold=0.4)
    config = RouterConfig(intent_parser="embedding", intent_parser_model=str(model_dir))

    below = EmbeddingIntentParser(config, classifier=lambda t: 0.39).parse(_request())
    at = EmbeddingIntentParser(config, classifier=lambda t: 0.40).parse(_request())

    assert below.route is Route.QUICK
    assert at.route is Route.DEEP


def test_variant_appears_in_the_trace(tmp_path):
    model_dir = _manifest(tmp_path, variant="knn", name="embedding_knn")
    parser = EmbeddingIntentParser(
        RouterConfig(intent_parser="embedding", intent_parser_model=str(model_dir)),
        classifier=lambda t: 0.6,
    )
    _, traces = parser.parse_with_traces(_request())
    classify = next(t for t in traces if t.stage.endswith(".classify"))
    assert classify.metadata["variant"] == "knn"
    assert classify.metadata["name"] == "embedding_knn"


def test_unknown_variant_is_rejected(tmp_path):
    model_dir = _manifest(tmp_path, variant="wormhole")
    # the guard lives in artifact loading, so build without an injected classifier
    with pytest.raises(ValueError, match="Unknown embedding variant"):
        EmbeddingIntentParser._load_classifier(
            model_dir, json.loads((model_dir / "manifest.json").read_text())
        )


def test_missing_model_path_says_how_to_build_it(tmp_path):
    with pytest.raises(ValueError, match="Train/export a model"):
        EmbeddingIntentParser._load_classifier(tmp_path / "nope", {"variant": "cls"})


def test_factory_builds_the_family(tmp_path, monkeypatch):
    model_dir = _manifest(tmp_path)
    monkeypatch.setattr(
        "reliability_router.intent.factory.EmbeddingIntentParser",
        lambda config: EmbeddingIntentParser(config, classifier=lambda t: 0.2),
    )
    parser = build_intent_parser(
        RouterConfig(intent_parser="embedding", intent_parser_model=str(model_dir))
    )
    assert parser.name == "embedding_cls"
