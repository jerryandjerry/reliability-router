"""Default regex/heuristic intent parser."""

from __future__ import annotations

from time import perf_counter

from ..config import RouterConfig
from ..features import FeatureExtractor
from ..models import ExecutionTrace, RouterRequest, RoutingDecision
from ..policy import RoutingPolicy
from ..scoring import RiskScorer


class RegexIntentParser:
    """Current deterministic feature/scoring/policy parser, behind the parser interface."""

    name = "regex"

    def __init__(self, config: RouterConfig):
        self.config = config
        self.version = config.policy_version
        self.extractor = FeatureExtractor()
        self.scorer = RiskScorer(config)
        self.policy = RoutingPolicy(config)

    def parse(self, request: RouterRequest) -> RoutingDecision:
        decision, _ = self.parse_with_traces(request)
        return decision

    def parse_with_traces(self, request: RouterRequest) -> tuple[RoutingDecision, list[ExecutionTrace]]:
        traces: list[ExecutionTrace] = []

        start = perf_counter()
        features = self.extractor.extract(request)
        traces.append(
            ExecutionTrace(stage="intent.regex.extract_features", duration_ms=(perf_counter() - start) * 1000)
        )

        start = perf_counter()
        assessment = self.scorer.score(features)
        traces.append(ExecutionTrace(stage="intent.regex.score", duration_ms=(perf_counter() - start) * 1000))

        start = perf_counter()
        decision = self.policy.decide(features, assessment)
        traces.append(ExecutionTrace(stage="intent.regex.policy", duration_ms=(perf_counter() - start) * 1000))
        return decision, traces
