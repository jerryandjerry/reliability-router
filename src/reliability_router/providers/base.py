"""Generation-provider contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..models import DraftAnswer, EvidenceItem, Route


@dataclass(frozen=True)
class GenerationRequest:
    query: str
    route: Route
    evidence: list[EvidenceItem] = field(default_factory=list)
    max_output_tokens: int = 500
    repair_feedback: list[str] = field(default_factory=list)


class GenerationProvider(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> DraftAnswer: ...
