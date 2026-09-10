"""Shared path result contract."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import DraftAnswer, EvidenceItem, ExecutionTrace, VerificationReport


@dataclass
class PathResult:
    draft: DraftAnswer
    verification: VerificationReport
    evidence: list[EvidenceItem] = field(default_factory=list)
    traces: list[ExecutionTrace] = field(default_factory=list)
