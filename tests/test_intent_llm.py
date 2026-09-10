import json
from pathlib import Path

import pytest

from reliability_router.config import RouterConfig
from reliability_router.intent import LLMIntentParser, RegexIntentParser, build_intent_parser
from reliability_router.models import ReasonCode, Route, RouterRequest


def test_intent_factory_defaults_to_regex() -> None:
    parser = build_intent_parser(RouterConfig())

    assert isinstance(parser, RegexIntentParser)


def test_intent_factory_builds_llm_parser_identity() -> None:
    parser = build_intent_parser(
        RouterConfig(
            intent_parser="llm",
            intent_parser_provider="openai",
            intent_parser_model="gpt-5.6-luna",
            intent_parser_thinking="medium",
        )
    )

    assert isinstance(parser, LLMIntentParser)
    assert parser.name == "openai"
    assert parser.version == "gpt-5.6-luna-medium"


def test_intent_factory_builds_claude_llm_parser_identity() -> None:
    parser = build_intent_parser(
        RouterConfig(
            intent_parser="llm",
            intent_parser_provider="anthropic",
            intent_parser_model="claude-sonnet-5",
            intent_parser_thinking="medium",
            intent_parser_transport="claude",
        )
    )

    assert isinstance(parser, LLMIntentParser)
    assert parser.name == "anthropic"
    assert parser.version == "claude-sonnet-5-medium"


def test_llm_parser_returns_routing_decision_from_codex_runner() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], prompt: str, timeout_seconds: int) -> str:
        commands.append(command)
        assert timeout_seconds == 120
        assert command[:2] == ["codex", "exec"]
        assert command[-1] == "-"
        assert command[command.index("--model") + 1] == "gpt-5.6-luna"
        assert 'model_reasoning_effort="medium"' in command
        assert "latest Bitcoin price" in prompt
        assert "--output-schema" in command
        assert json.loads(Path(command[command.index("--output-schema") + 1]).read_text(encoding="utf-8"))
        return json.dumps(
            {
                "route": "DEEP",
                "risk_score": 0.83,
                "reason_codes": ["FRESHNESS_REQUIRED", "not-a-real-code"],
                "forced": False,
                "explanation": "Needs current information.",
            }
        )

    parser = LLMIntentParser(RouterConfig(intent_parser="llm"), runner=runner)
    decision = parser.parse(RouterRequest(query="What is the latest Bitcoin price?"))

    assert decision.route == Route.DEEP
    assert decision.risk_score == 0.83
    assert decision.reason_codes == [ReasonCode.FRESHNESS_REQUIRED]
    assert decision.explanation == "Needs current information."
    assert commands


def test_llm_parser_can_use_claude_transport() -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], prompt: str, timeout_seconds: int) -> str:
        commands.append(command)
        assert timeout_seconds == 120
        assert command[:2] == ["claude", "--print"]
        assert command[command.index("--model") + 1] == "claude-opus-5"
        assert command[command.index("--effort") + 1] == "medium"
        assert "--json-schema" in command
        assert "medical dosage" in prompt
        return json.dumps(
            {
                "route": "DEEP",
                "risk_score": 0.91,
                "reason_codes": ["HIGH_STAKES"],
                "forced": False,
                "explanation": "Medical dosage is high stakes.",
            }
        )

    parser = LLMIntentParser(
        RouterConfig(
            intent_parser="llm",
            intent_parser_provider="anthropic",
            intent_parser_model="claude-opus-5",
            intent_parser_transport="claude",
        ),
        runner=runner,
    )
    decision = parser.parse(RouterRequest(query="Can I trust this medical dosage?"))

    assert decision.route == Route.DEEP
    assert decision.risk_score == 0.91
    assert commands


def test_llm_parser_rejects_invalid_route() -> None:
    def runner(command: list[str], prompt: str, timeout_seconds: int) -> str:
        return json.dumps(
            {
                "route": "MAYBE",
                "risk_score": 0.1,
                "reason_codes": [],
                "forced": False,
                "explanation": "bad",
            }
        )

    parser = LLMIntentParser(RouterConfig(intent_parser="llm"), runner=runner)

    with pytest.raises(ValueError, match="QUICK or DEEP"):
        parser.parse(RouterRequest(query="hello"))
