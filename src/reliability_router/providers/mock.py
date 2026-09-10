"""Offline deterministic provider used by tests and the zero-key demo."""

from __future__ import annotations

import re

from ..models import Claim, DraftAnswer, Route
from .base import GenerationRequest


class MockProvider:
    name = "mock"

    def __init__(self, model: str = "deterministic-demo-v1"):
        self.model = model

    def generate(self, request: GenerationRequest) -> DraftAnswer:
        if request.route == Route.QUICK:
            return DraftAnswer(
                answer=self._quick_answer(request.query),
                claims=[],
                confidence=0.82,
                abstained=False,
                provider=self.name,
                model=self.model,
            )

        if not request.evidence:
            return DraftAnswer(
                answer=(
                    "I do not have enough verified evidence to answer this reliably. "
                    "Provide a trusted source or connect a retrieval-capable tool."
                ),
                claims=[],
                confidence=0.18,
                abstained=True,
                provider=self.name,
                model=self.model,
            )

        snippets: list[str] = []
        claims: list[Claim] = []
        for evidence in request.evidence[:3]:
            sentence = self._first_sentence(evidence.text)
            snippets.append(f"[{evidence.id}] {sentence}")
            claims.append(Claim(text=sentence, evidence_ids=[evidence.id]))
        answer = "Based only on the supplied evidence:\n\n" + "\n".join(f"- {item}" for item in snippets)
        if request.repair_feedback:
            answer += "\n\nThe answer was regenerated after a grounding check."
        return DraftAnswer(
            answer=answer,
            claims=claims,
            confidence=min(0.94, 0.65 + 0.08 * len(claims)),
            abstained=False,
            provider=self.name,
            model=self.model,
        )

    @staticmethod
    def _quick_answer(query: str) -> str:
        stripped = query.strip()
        math_match = re.fullmatch(r"(?:what is |calculate )?(\d+)\s*\+\s*(\d+)\??", stripped, re.I)
        if math_match:
            return str(int(math_match.group(1)) + int(math_match.group(2)))
        if "haiku" in stripped.lower() or "俳句" in stripped:
            return "Quiet circuits hum\nQuestions choose the deeper path\nEvidence holds fast"
        return f"Quick-path response for: {stripped}"

    @staticmethod
    def _first_sentence(text: str) -> str:
        compact = " ".join(text.split())
        parts = re.split(r"(?<=[.!?。！？])\s+", compact, maxsplit=1)
        sentence = parts[0] if parts else compact
        return sentence[:320]
