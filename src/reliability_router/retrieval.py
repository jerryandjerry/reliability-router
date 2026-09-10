"""Evidence retrieval abstractions and an offline context retriever."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from .models import ContextItem, EvidenceItem

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class Retriever(Protocol):
    def retrieve(self, query: str, context: Sequence[ContextItem], top_k: int) -> list[EvidenceItem]: ...

# keyword overlap
class InMemoryRetriever:
    """Ranks supplied context by token overlap; no network or vector DB required."""

    def retrieve(self, query: str, context: Sequence[ContextItem], top_k: int) -> list[EvidenceItem]:
        query_terms = self._terms(query)
        ranked: list[EvidenceItem] = []
        for item in context:
            terms = self._terms(item.text) 
            lexical = len(query_terms & terms) / max(1, len(query_terms | terms)) # jaccard similarity: intersection/union
            # Keep even zero-overlap items: a small context list may still be useful.
            score = min(1.0, 0.15 + lexical * 0.85)
            ranked.append(
                EvidenceItem(
                    id=item.id,
                    title=item.source or item.id,
                    text=item.text,
                    source=item.source,
                    score=round(score, 4),
                    trust=item.trust,
                    metadata=item.metadata,
                )
            )
        ranked.sort(key=lambda evidence: evidence.score * evidence.trust, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token.lower()
            for token in _TOKEN_RE.findall(text)
            if len(token) > 1 or "\u4e00" <= token <= "\u9fff"
        }
