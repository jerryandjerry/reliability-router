"""Intent parser construction."""

from __future__ import annotations

from ..config import RouterConfig
from .base import IntentParser
from .bert import BertFamilyIntentParser
from .embedding import EmbeddingIntentParser
from .llm import LLMIntentParser
from .regex import RegexIntentParser
from .tfidf import TfidfIntentParser


def build_intent_parser(config: RouterConfig) -> IntentParser:
    parser = config.intent_parser.lower().strip()
    if parser == "regex":
        return RegexIntentParser(config)
    if parser == "llm":
        return LLMIntentParser(config)
    if parser == "bert":
        return BertFamilyIntentParser(config)
    if parser == "tfidf":
        return TfidfIntentParser(config)
    if parser == "embedding":
        return EmbeddingIntentParser(config)
    raise ValueError(
        f"Unsupported RR_INTENT_PARSER={config.intent_parser!r}. "
        "Expected 'regex', 'llm', 'bert', 'tfidf', or 'embedding'."
    )
