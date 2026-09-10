"""Low-latency path for low-risk requests."""

from __future__ import annotations

from time import perf_counter

from ..config import RouterConfig
from ..models import ExecutionTrace, Route, RouterRequest, RoutingDecision
from ..providers.base import GenerationProvider, GenerationRequest
from ..verification import PostVerifier
from .base import PathResult


class QuickPath:
    def __init__(self, *, config: RouterConfig, provider: GenerationProvider, verifier: PostVerifier):
        self.config = config
        self.provider = provider
        self.verifier = verifier

    def execute(self, request: RouterRequest, decision: RoutingDecision) -> PathResult:
        start = perf_counter()
        draft = self.provider.generate(
            GenerationRequest(
                query=request.query,
                route=Route.QUICK,
                evidence=[],
                max_output_tokens=self.config.quick_max_output_tokens,
            )
        )
        generation_ms = (perf_counter() - start) * 1000

        verify_start = perf_counter()
        report = self.verifier.verify(
            route=Route.QUICK,
            draft=draft,
            evidence=[],
            features=decision.features,
        )
        verify_ms = (perf_counter() - verify_start) * 1000
        return PathResult(
            draft=draft,
            verification=report,
            evidence=[],
            traces=[
                ExecutionTrace(stage="quick.generate", duration_ms=generation_ms),
                ExecutionTrace(stage="quick.verify", duration_ms=verify_ms),
            ],
        )
