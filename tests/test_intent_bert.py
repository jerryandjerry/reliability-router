import json

import pytest

from reliability_router.config import RouterConfig
from reliability_router.intent.bert import (
    BertFamilyIntentParser,
    BertIntentParser,
)
from reliability_router.intent.text import request_to_classifier_text
from reliability_router.models import ContextItem, Route, RouterRequest


def test_bert_parser_maps_probability_to_routing_decision(tmp_path) -> None:
    model_path = tmp_path / "v1"
    model_path.mkdir()
    (model_path / "manifest.json").write_text(
        json.dumps({"name": "bert", "version": "v1"}),
        encoding="utf-8",
    )
    config = RouterConfig(intent_parser="bert", intent_parser_model=str(model_path))
    parser = BertFamilyIntentParser(config, classifier=lambda text: 0.84)

    decision = parser.parse(RouterRequest(query="Should I rely on this medical dosage advice?"))

    assert parser.name == "bert"
    assert parser.version == "v1"
    assert decision.route == Route.DEEP
    assert decision.risk_score == 0.84
    assert decision.threshold == 0.5
    assert "bert/v1" in decision.explanation


def test_bert_family_parser_uses_manifest_method_name(tmp_path) -> None:
    model_path = tmp_path / "exported" / "v1"
    model_path.mkdir(parents=True)
    (model_path / "manifest.json").write_text(
        json.dumps({"name": "deberta", "version": "v1", "threshold": 0.5}),
        encoding="utf-8",
    )
    config = RouterConfig(intent_parser="bert", intent_parser_model=str(model_path))
    parser = BertFamilyIntentParser(config, classifier=lambda text: 0.91)

    decision = parser.parse(RouterRequest(query="Does this require careful external validation?"))

    assert parser.name == "deberta"
    assert parser.version == "v1"
    assert decision.route == Route.DEEP
    assert "deberta/v1" in decision.explanation


def test_bert_parser_uses_config_threshold_without_manifest_threshold(tmp_path) -> None:
    model_path = tmp_path / "v1"
    model_path.mkdir()
    (model_path / "manifest.json").write_text(
        json.dumps({"name": "bert", "version": "v1"}),
        encoding="utf-8",
    )
    config = RouterConfig(
        intent_parser="bert",
        intent_parser_model=str(model_path),
        intent_parser_threshold=0.8,
    )
    parser = BertIntentParser(config, classifier=lambda text: 0.79)

    decision = parser.parse(RouterRequest(query="Simple rewrite"))

    assert parser.version == "v1"
    assert decision.route == Route.QUICK
    assert decision.threshold == 0.8


def test_bert_parser_requires_manifest_identity(tmp_path) -> None:
    model_path = tmp_path / "v1"
    model_path.mkdir()
    (model_path / "manifest.json").write_text(json.dumps({"name": "bert"}), encoding="utf-8")

    with pytest.raises(ValueError, match="version"):
        BertIntentParser(RouterConfig(intent_parser="bert", intent_parser_model=str(model_path)), classifier=lambda _: 0.4)


def test_bert_parser_requires_exported_model_without_test_classifier() -> None:
    config = RouterConfig(intent_parser="bert", intent_parser_model="missing-test-model")

    with pytest.raises(ValueError, match="Train/export"):
        BertIntentParser(config)


def test_request_to_classifier_text_includes_router_inputs() -> None:
    request = RouterRequest(
        query="What changed today?",
        available_tools=["web_search"],
        context=[ContextItem(id="ctx", text="Prior article", source="archive", trust=0.4)],
    )

    text = request_to_classifier_text(request)

    assert "What changed today?" in text
    assert "web_search" in text
    assert "Prior article" in text
    assert "source=archive" in text
