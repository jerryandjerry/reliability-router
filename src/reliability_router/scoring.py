"""Risk scoring with auditable feature contributions."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RouterConfig
from .models import FeatureVector, ReasonCode, RiskBand


@dataclass(frozen=True)
class RiskAssessment:
    score: float
    band: RiskBand
    reason_codes: list[ReasonCode]
    contributions: dict[str, float]


class RiskScorer:
    def __init__(self, config: RouterConfig):
        self.config = config

    def score(self, features: FeatureVector) -> RiskAssessment:
        raw = features.model_dump(exclude={"token_estimate"})
        contributions: dict[str, float] = {}
        total = 0.0
        for name, value in raw.items():
            weight = self.config.weights.get(name, 0.0)
            contribution = float(value) * weight # different contribution weight
            contributions[name] = round(contribution, 4)
            total += contribution

        # Interactions capture cases where two moderate signals create high risk.
        interactions = {
            "high_stakes_x_evidence_gap": features.high_stakes * features.evidence_gap * 0.16,
            "freshness_x_tool_gap": features.freshness * features.tool_gap * 0.14,
            "pressure_x_high_stakes": features.user_pressure * features.high_stakes * 0.12,
            "conflict_x_citation": features.context_conflict * max(features.citation_need, 0.5) * 0.10,
        }
        for name, value in interactions.items():
            contributions[name] = round(value, 4)
            total += value

        # Convert additive risk into a bounded score while preserving ranking.
        score = round(min(1.0, total), 4)
        band = self._band(score)
        reasons = self._reason_codes(features)
        return RiskAssessment(score=score, band=band, reason_codes=reasons, contributions=contributions)

    def _band(self, score: float) -> RiskBand:
        if score >= self.config.critical_threshold:
            return RiskBand.CRITICAL
        if score >= self.config.deep_threshold:
            return RiskBand.HIGH
        if score >= self.config.deep_threshold * 0.62:
            return RiskBand.MEDIUM
        return RiskBand.LOW

    @staticmethod
    def _reason_codes(features: FeatureVector) -> list[ReasonCode]:
        rules = [
            (features.high_stakes >= 0.5, ReasonCode.HIGH_STAKES),
            (features.freshness >= 0.5, ReasonCode.FRESHNESS_REQUIRED),
            (features.citation_need >= 0.5, ReasonCode.CITATIONS_REQUESTED),
            (features.multi_step >= 0.45, ReasonCode.MULTI_STEP_REASONING),
            (features.ambiguity >= 0.5, ReasonCode.AMBIGUOUS_QUERY),
            (features.context_conflict >= 0.5, ReasonCode.CONTEXT_CONFLICT),
            (features.user_pressure >= 0.5, ReasonCode.USER_PRESSURE),
            (features.numerical >= 0.5, ReasonCode.NUMERICAL_REASONING),
            (features.tool_gap >= 0.5, ReasonCode.TOOL_GAP),
            (features.evidence_gap >= 0.5, ReasonCode.EVIDENCE_GAP),
            (features.context_load >= 0.65, ReasonCode.LONG_CONTEXT),
            (features.prompt_injection >= 0.5, ReasonCode.PROMPT_INJECTION_RISK),
        ]
        return [code for enabled, code in rules if enabled]
