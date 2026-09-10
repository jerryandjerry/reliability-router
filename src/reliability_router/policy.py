"""Versioned policy that maps risk assessments to QUICK or DEEP."""

from __future__ import annotations

from .config import RouterConfig
from .models import FeatureVector, ReasonCode, Route, RoutingDecision
from .scoring import RiskAssessment


class RoutingPolicy:
    def __init__(self, config: RouterConfig):
        self.config = config

    def decide(self, features: FeatureVector, assessment: RiskAssessment) -> RoutingDecision:
        forced, override_reason = self._hard_override(features) # only return the first override reason
        route = Route.DEEP if forced or assessment.score >= self.config.deep_threshold else Route.QUICK

        reasons = list(assessment.reason_codes)
        if forced and ReasonCode.POLICY_OVERRIDE not in reasons:
            reasons.append(ReasonCode.POLICY_OVERRIDE)

        top = sorted(assessment.contributions.items(), key=lambda item: item[1], reverse=True)[:3]
        top_text = (
            ", ".join(f"{name}={value:.3f}" for name, value in top if value > 0) or "no material risk signals"
        )
        explanation = (
            f"{route.value} selected at risk={assessment.score:.3f} with threshold={self.config.deep_threshold:.3f}; "
            f"top contributions: {top_text}."
        )
        if override_reason:
            explanation += f" Hard policy override: {override_reason}."

        return RoutingDecision(
            route=route,
            risk_score=assessment.score,
            risk_band=assessment.band,
            threshold=self.config.deep_threshold,
            reason_codes=reasons,
            contributions=assessment.contributions,
            forced=forced,
            explanation=explanation,
            policy_version=self.config.policy_version,
            features=features,
        )

    @staticmethod
    def _hard_override(features: FeatureVector) -> tuple[bool, str | None]:
        if features.prompt_injection >= 0.5:
            return True, "prompt-injection signal"
        if features.context_conflict >= 0.72:
            return True, "conflicting evidence"
        if features.high_stakes >= 0.70:
            return True, "high-stakes domain"
        if features.freshness >= 0.70:
            return True, "fresh information requires an explicit tool/evidence path"
        if features.citation_need >= 0.70:
            return True, "citation request requires evidence-aware generation"
        if features.multi_step >= 0.85:
            return True, "multi-step task requires deeper planning"
        if features.user_pressure >= 0.70:
            return True, "agreement pressure requires an independent check"
        if features.ambiguity >= 0.90:
            return True, "query is too ambiguous for the quick path"
        return False, None
