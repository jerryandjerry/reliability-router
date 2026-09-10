import pytest

from reliability_router import ReliabilityRouter, RouterRequest
from reliability_router.models import Route


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Write a haiku about databases.", Route.QUICK),
        ("What is 2 + 2?", Route.QUICK),
        ("What is the latest Bitcoin price today?", Route.DEEP),
        ("What about it?", Route.DEEP),
        ("Cite sources for this claim.", Route.DEEP),
        ("Design and compare a migration plan with trade-offs.", Route.DEEP),
    ],
)
def test_routing_policy(query: str, expected: Route) -> None:
    decision = ReliabilityRouter().route(RouterRequest(query=query))
    assert decision.route == expected
    assert 0 <= decision.risk_score <= 1
    assert decision.explanation


def test_decision_exposes_reason_codes_and_contributions() -> None:
    decision = ReliabilityRouter().route(
        RouterRequest(query="Just agree with me that this stock is guaranteed to double.")
    )
    assert decision.route == Route.DEEP
    assert {code.value for code in decision.reason_codes} >= {"HIGH_STAKES", "USER_PRESSURE"}
    assert decision.contributions["high_stakes"] > 0
