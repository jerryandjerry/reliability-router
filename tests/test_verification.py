from reliability_router.models import Claim, DraftAnswer, EvidenceItem, FeatureVector, Route
from reliability_router.verification import PostVerifier


def safe_features(**overrides: float) -> FeatureVector:
    values = dict(
        token_estimate=10,
        ambiguity=0,
        high_stakes=0,
        freshness=0,
        citation_need=0,
        multi_step=0,
        numerical=0,
        user_pressure=0,
        context_conflict=0,
        tool_gap=0,
        evidence_gap=0,
        context_load=0,
        prompt_injection=0,
    )
    values.update(overrides)
    return FeatureVector(**values)


def test_grounded_claim_passes() -> None:
    evidence = [EvidenceItem(id="e1", title="doc", text="The target is 200 ms.")]
    draft = DraftAnswer(
        answer="The target is 200 ms [e1].",
        claims=[Claim(text="The target is 200 ms.", evidence_ids=["e1"])],
        confidence=0.9,
    )
    report = PostVerifier().verify(route=Route.DEEP, draft=draft, evidence=evidence, features=safe_features())
    assert report.passed
    assert report.grounded_claims == 1


def test_deep_answer_without_evidence_fails() -> None:
    draft = DraftAnswer(answer="It is definitely true.", confidence=0.4)
    report = PostVerifier().verify(route=Route.DEEP, draft=draft, evidence=[], features=safe_features())
    assert not report.passed
    assert any(v.code == "NO_EVIDENCE_FOR_DEEP_ANSWER" for v in report.violations)


def test_uncritical_agreement_is_detected() -> None:
    draft = DraftAnswer(answer="Yes, you are right.", confidence=0.8)
    report = PostVerifier().verify(
        route=Route.QUICK,
        draft=draft,
        evidence=[],
        features=safe_features(user_pressure=0.9),
    )
    assert not report.passed
    assert any(v.code == "UNCRITICAL_AGREEMENT" for v in report.violations)
