"""Public data contracts for Reliability Router."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Route(StrEnum):
    QUICK = "QUICK"
    DEEP = "DEEP"


class RiskBand(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ReasonCode(StrEnum):
    HIGH_STAKES = "HIGH_STAKES"
    FRESHNESS_REQUIRED = "FRESHNESS_REQUIRED"
    CITATIONS_REQUESTED = "CITATIONS_REQUESTED"
    MULTI_STEP_REASONING = "MULTI_STEP_REASONING"
    AMBIGUOUS_QUERY = "AMBIGUOUS_QUERY"
    CONTEXT_CONFLICT = "CONTEXT_CONFLICT"
    USER_PRESSURE = "USER_PRESSURE"
    NUMERICAL_REASONING = "NUMERICAL_REASONING"
    TOOL_GAP = "TOOL_GAP"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    LONG_CONTEXT = "LONG_CONTEXT"
    PROMPT_INJECTION_RISK = "PROMPT_INJECTION_RISK"
    POLICY_OVERRIDE = "POLICY_OVERRIDE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"


class ContextItem(BaseModel):
    """A user-supplied context item that may become evidence."""

    model_config = ConfigDict(extra="forbid") # any unknown field is rejected

    id: str = Field(default_factory=lambda: f"ctx_{uuid4().hex[:10]}")
    text: str = Field(min_length=1)
    source: str | None = None
    trust: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"req_{uuid4().hex[:12]}")
    query: str = Field(min_length=1, max_length=50_000)
    context: list[ContextItem] = Field(default_factory=list) # each context item is one object
    available_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_estimate: int = Field(ge=1)
    ambiguity: float = Field(ge=0.0, le=1.0)
    high_stakes: float = Field(ge=0.0, le=1.0)
    freshness: float = Field(ge=0.0, le=1.0)
    citation_need: float = Field(ge=0.0, le=1.0)
    multi_step: float = Field(ge=0.0, le=1.0)
    numerical: float = Field(ge=0.0, le=1.0)
    user_pressure: float = Field(ge=0.0, le=1.0)
    context_conflict: float = Field(ge=0.0, le=1.0)
    tool_gap: float = Field(ge=0.0, le=1.0)
    evidence_gap: float = Field(ge=0.0, le=1.0)
    context_load: float = Field(ge=0.0, le=1.0)
    prompt_injection: float = Field(ge=0.0, le=1.0)


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Route
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    threshold: float = Field(ge=0.0, le=1.0)
    reason_codes: list[ReasonCode]
    contributions: dict[str, float]
    forced: bool = False
    explanation: str
    policy_version: str
    features: FeatureVector


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    text: str
    source: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    trust: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    claims: list[Claim] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    abstained: bool = False
    provider: str = "unknown"
    model: str = "unknown"
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class VerificationViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    severity: Severity
    message: str
    claim_index: int | None = None


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    violations: list[VerificationViolation] = Field(default_factory=list)
    checked_claims: int = Field(default=0, ge=0)
    grounded_claims: int = Field(default=0, ge=0)
    repair_attempted: bool = False


class ExecutionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    duration_ms: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    decision: RoutingDecision
    answer: str
    abstained: bool
    verification: VerificationReport
    evidence: list[EvidenceItem] = Field(default_factory=list)
    provider: str
    model: str
    latency_ms: float = Field(ge=0.0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    traces: list[ExecutionTrace] = Field(default_factory=list)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    query: str
    expected_route: Route
    context: list[ContextItem] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class EvalPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    expected_route: Route
    predicted_route: Route
    correct: bool
    risk_score: float
    reason_codes: list[ReasonCode]
    latency_ms: float
    tags: list[str] = Field(default_factory=list)


class TagBreakdown(BaseModel):
    """Per-tag slice of an evaluation, so a score can be traced to where it came from."""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    expected_deep: int = Field(ge=0)
    caught_deep: int = Field(ge=0)
    deep_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_quick: int = Field(ge=0)
    false_deep: int = Field(ge=0)


class EvalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    policy_version: str
    total: int
    correct: int
    route_accuracy: float
    deep_precision: float
    deep_recall: float
    deep_f1: float
    confusion_matrix: dict[str, int]
    p50_latency_ms: float
    p95_latency_ms: float
    reason_code_counts: dict[str, int]
    by_tag: dict[str, TagBreakdown] = Field(default_factory=dict)
    predictions: list[EvalPrediction]
