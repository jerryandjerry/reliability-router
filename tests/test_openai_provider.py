from reliability_router.providers.openai_provider import OpenAIProvider


def test_parse_json_from_fence() -> None:
    payload = OpenAIProvider._parse_json(
        'prefix```json\n{"answer":"ok","claims":[],"confidence":0.8,"abstained":false}\n```suffix'
    )
    assert payload is not None
    assert payload["answer"] == "ok"


def test_bounded_float() -> None:
    assert OpenAIProvider._bounded_float(2) == 1.0
    assert OpenAIProvider._bounded_float(-1) == 0.0
    assert OpenAIProvider._bounded_float("bad") == 0.5
