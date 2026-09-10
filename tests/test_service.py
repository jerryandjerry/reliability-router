import pytest

from reliability_router import ReliabilityRouter, RouterRequest
from reliability_router.config import RouterConfig
from reliability_router.features import FeatureExtractor
from reliability_router.models import ContextItem, Route, RoutingDecision
from reliability_router.models import RouterRequest as Request
from reliability_router.policy import RoutingPolicy
from reliability_router.scoring import RiskScorer


def test_quick_path_returns_answer() -> None:
    response = ReliabilityRouter().answer(RouterRequest(query="What is 2 + 2?"))
    assert response.decision.route == Route.QUICK
    assert response.answer == "4"
    assert not response.abstained
    assert response.verification.passed


def test_deep_path_with_evidence_is_grounded() -> None:
    response = ReliabilityRouter().answer(
        RouterRequest(
            query="Cite the API latency target.",
            context=[
                ContextItem(id="latency_doc", text="The API latency target is 200 milliseconds.", trust=0.95)
            ],
        )
    )
    assert response.decision.route == Route.DEEP
    assert not response.abstained
    assert response.verification.passed
    assert response.verification.grounded_claims >= 1
    assert response.evidence[0].id == "latency_doc"


def test_deep_path_without_evidence_abstains() -> None:
    response = ReliabilityRouter().answer(RouterRequest(query="What is the latest Bitcoin price today?"))
    assert response.decision.route == Route.DEEP
    assert response.abstained
    assert "verified evidence" in response.answer


def test_deep_path_repairs_an_ungrounded_claim() -> None:
    from reliability_router.config import RouterConfig
    from reliability_router.models import Claim, DraftAnswer
    from reliability_router.providers.base import GenerationRequest

    class RepairingProvider:
        name = "repairing-test"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request: GenerationRequest) -> DraftAnswer:
            self.calls += 1
            evidence_id = request.evidence[0].id
            if self.calls == 1:
                return DraftAnswer(
                    answer="The target is 200 ms.",
                    claims=[Claim(text="The target is 200 ms.", evidence_ids=[])],
                    confidence=0.9,
                    provider=self.name,
                    model="test",
                )
            return DraftAnswer(
                answer=f"The target is 200 ms [{evidence_id}].",
                claims=[Claim(text="The target is 200 ms.", evidence_ids=[evidence_id])],
                confidence=0.9,
                provider=self.name,
                model="test",
            )

    provider = RepairingProvider()
    router = ReliabilityRouter(config=RouterConfig(max_repair_attempts=1), provider=provider)
    response = router.answer(
        RouterRequest(
            query="Cite the API latency target.",
            context=[ContextItem(id="latency_doc", text="The target is 200 ms.", trust=0.95)],
        )
    )
    assert provider.calls == 2
    assert response.verification.passed
    assert response.verification.repair_attempted
    assert any(trace.stage == "deep.repair" for trace in response.traces)


def test_router_accepts_custom_intent_parser() -> None:
    class QuickOnlyParser:
        name = "quick-only-test"
        version = "test"

        def __init__(self) -> None:
            config = RouterConfig(policy_version=self.version)
            self.extractor = FeatureExtractor()
            self.scorer = RiskScorer(config)
            self.policy = RoutingPolicy(config)

        def parse(self, request: Request) -> RoutingDecision:
            features = self.extractor.extract(request)
            assessment = self.scorer.score(features)
            decision = self.policy.decide(features, assessment)
            return decision.model_copy(update={"route": Route.QUICK, "forced": False})

    router = ReliabilityRouter(intent_parser=QuickOnlyParser())
    response = router.answer(RouterRequest(query="What is the latest Bitcoin price today?"))

    assert response.decision.route == Route.QUICK
    assert response.answer.startswith("Quick-path response")


def test_router_rejects_parser_without_identity() -> None:
    class NamelessParser:
        name = ""
        version = "test"

        def parse(self, request: Request) -> RoutingDecision:
            raise AssertionError("not called")

    with pytest.raises(ValueError, match="name"):
        ReliabilityRouter(intent_parser=NamelessParser())

    class VersionlessParser:
        name = "test"
        version = ""

        def parse(self, request: Request) -> RoutingDecision:
            raise AssertionError("not called")

    with pytest.raises(ValueError, match="version"):
        ReliabilityRouter(intent_parser=VersionlessParser())


def test_router_rejects_parser_without_parse_method() -> None:
    class BadParser:
        name = "bad"
        version = "test"

        def parsing(self, request: Request) -> RoutingDecision:
            raise AssertionError("not called")

    with pytest.raises(ValueError, match="parse"):
        ReliabilityRouter(intent_parser=BadParser())


def test_router_rejects_provider_without_runtime_contract() -> None:
    class NamelessProvider:
        name = ""

        def generate(self, request: object) -> object:
            raise AssertionError("not called")

    with pytest.raises(ValueError, match="name"):
        ReliabilityRouter(provider=NamelessProvider())

    class BadProvider:
        name = "bad"

        def generating(self, request: object) -> object:
            raise AssertionError("not called")

    with pytest.raises(ValueError, match="generate"):
        ReliabilityRouter(provider=BadProvider())
