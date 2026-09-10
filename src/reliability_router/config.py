"""Configuration and versioned routing policy defaults."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_WEIGHTS: dict[str, float] = {
    "ambiguity": 0.10,
    "high_stakes": 0.22,
    "freshness": 0.14,
    "citation_need": 0.08,
    "multi_step": 0.11,
    "numerical": 0.06,
    "user_pressure": 0.09,
    "context_conflict": 0.18,
    "tool_gap": 0.12,
    "evidence_gap": 0.13,
    "context_load": 0.05,
    "prompt_injection": 0.16,
}


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_version: str = "1.0"
    deep_threshold: float = Field(default=0.48, ge=0.0, le=1.0)
    critical_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    max_evidence_items: int = Field(default=5, ge=1, le=20)
    max_repair_attempts: int = Field(default=1, ge=0, le=3)
    fail_closed: bool = True
    quick_max_output_tokens: int = Field(default=350, ge=64, le=4000)
    deep_max_output_tokens: int = Field(default=900, ge=128, le=8000)
    intent_parser: str = "regex"
    intent_parser_provider: str = "openai"
    intent_parser_model: str = "gpt-5.6-luna"
    intent_parser_thinking: str = "medium"
    intent_parser_transport: str = "codex"
    intent_parser_timeout_seconds: int = Field(default=120, ge=1)
    intent_parser_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    provider: str = "mock"
    quick_model: str = "gpt-5.5"
    deep_model: str = "gpt-5.5"
    weights: dict[str, float] = Field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    @classmethod
    def from_env(cls, **overrides: Any) -> RouterConfig:
        values: dict[str, Any] = {
            "deep_threshold": float(os.getenv("RR_DEEP_THRESHOLD", "0.48")),
            "critical_threshold": float(os.getenv("RR_CRITICAL_THRESHOLD", "0.78")),
            "max_evidence_items": int(os.getenv("RR_MAX_EVIDENCE_ITEMS", "5")),
            "max_repair_attempts": int(os.getenv("RR_MAX_REPAIR_ATTEMPTS", "1")),
            "fail_closed": os.getenv("RR_FAIL_CLOSED", "true").lower() in {"1", "true", "yes"},
            "intent_parser": os.getenv("RR_INTENT_PARSER", "regex"),
            "intent_parser_provider": os.getenv("RR_INTENT_PARSER_PROVIDER", "openai"),
            "intent_parser_model": os.getenv("RR_INTENT_PARSER_MODEL", "gpt-5.6-luna"),
            "intent_parser_thinking": os.getenv("RR_INTENT_PARSER_THINKING", "medium"),
            "intent_parser_transport": os.getenv("RR_INTENT_PARSER_TRANSPORT", "codex"),
            "intent_parser_timeout_seconds": int(os.getenv("RR_INTENT_PARSER_TIMEOUT_SECONDS", "120")),
            "intent_parser_threshold": float(os.getenv("RR_INTENT_PARSER_THRESHOLD", "0.5")),
            "provider": os.getenv("RR_PROVIDER", "mock"),
            "quick_model": os.getenv("RR_QUICK_MODEL", os.getenv("RR_OPENAI_MODEL", "gpt-5.5")),
            "deep_model": os.getenv("RR_DEEP_MODEL", os.getenv("RR_OPENAI_MODEL", "gpt-5.5")),
        }
        values.update(overrides)
        return cls(**values)
