"""Deterministic post-generation checks for grounding and overconfidence."""

from __future__ import annotations

import re

from .models import (
    DraftAnswer,
    EvidenceItem,
    FeatureVector,
    Route,
    Severity,
    VerificationReport,
    VerificationViolation,
)


class PostVerifier:
    ABSOLUTE_RE = re.compile(
        r"\b(?:always|never|definitely|certainly|guaranteed|without doubt)\b|100\s*%|一定|绝对|百分之百|保证|毫无疑问",
        re.I,
    )
    AGREEMENT_RE = re.compile(
        r"^(?:yes|you are right|exactly|absolutely|correct)[,!.\s]|^(?:是的|你是对的|完全正确|没错)[，。！!\s]",
        re.I,
    )

    def verify(
        self,
        *,
        route: Route,
        draft: DraftAnswer,
        evidence: list[EvidenceItem],
        features: FeatureVector,
    ) -> VerificationReport:
        violations: list[VerificationViolation] = []
        valid_ids = {item.id for item in evidence}
        grounded = 0

        for index, claim in enumerate(draft.claims):
            unknown_ids = [evidence_id for evidence_id in claim.evidence_ids if evidence_id not in valid_ids]
            if unknown_ids:
                violations.append(
                    VerificationViolation(
                        code="UNKNOWN_EVIDENCE_ID",
                        severity=Severity.ERROR,
                        message=f"Claim references unknown evidence IDs: {unknown_ids}",
                        claim_index=index,
                    )
                )
            elif claim.evidence_ids:
                grounded += 1
            elif route == Route.DEEP and not draft.abstained:
                violations.append(
                    VerificationViolation(
                        code="UNGROUNDED_CLAIM",
                        severity=Severity.ERROR,
                        message="Deep-path claim has no evidence reference.",
                        claim_index=index,
                    )
                )

        if route == Route.DEEP and evidence and not draft.abstained and not draft.claims:
            violations.append(
                VerificationViolation(
                    code="MISSING_CLAIM_LEDGER",
                    severity=Severity.ERROR,
                    message="Deep-path answer did not return a machine-checkable claim ledger.",
                )
            )

        if route == Route.DEEP and not evidence and not draft.abstained:
            violations.append(
                VerificationViolation(
                    code="NO_EVIDENCE_FOR_DEEP_ANSWER",
                    severity=Severity.CRITICAL,
                    message="Deep path produced a factual answer without evidence.",
                )
            )

        if features.freshness >= 0.7 and features.tool_gap >= 0.7 and not draft.abstained:
            violations.append(
                VerificationViolation(
                    code="STALE_ANSWER_RISK",
                    severity=Severity.CRITICAL,
                    message="Current information was requested but no freshness-capable tool was available.",
                )
            )

        if features.user_pressure >= 0.5 and self.AGREEMENT_RE.search(draft.answer.strip()) and grounded == 0:
            violations.append(
                VerificationViolation(
                    code="UNCRITICAL_AGREEMENT",
                    severity=Severity.ERROR,
                    message="Answer appears to agree with user pressure without supporting evidence.",
                )
            )

        if draft.confidence < 0.65 and self.ABSOLUTE_RE.search(draft.answer):
            violations.append(
                VerificationViolation(
                    code="OVERCONFIDENT_LANGUAGE",
                    severity=Severity.WARNING,
                    message="Absolute language is inconsistent with the reported confidence.",
                )
            )

        if features.high_stakes >= 0.5 and draft.confidence < 0.7 and not draft.abstained:
            violations.append(
                VerificationViolation(
                    code="LOW_CONFIDENCE_HIGH_STAKES",
                    severity=Severity.ERROR,
                    message="High-stakes answer confidence is below the policy floor.",
                )
            )

        penalties = {
            Severity.INFO: 0.02,
            Severity.WARNING: 0.08,
            Severity.ERROR: 0.25,
            Severity.CRITICAL: 0.55,
        }
        score = max(0.0, 1.0 - sum(penalties[v.severity] for v in violations))
        passed = not any(v.severity in {Severity.ERROR, Severity.CRITICAL} for v in violations)
        return VerificationReport(
            passed=passed,
            score=round(score, 4),
            violations=violations,
            checked_claims=len(draft.claims),
            grounded_claims=grounded,
        )
