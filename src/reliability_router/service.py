"""Application service that orchestrates routing and execution."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any

from .config import RouterConfig
from .intent import IntentParser, build_intent_parser
from .models import ExecutionTrace, Route, RouterRequest, RouterResponse, RoutingDecision
from .paths.deep import DeepPath
from .paths.quick import QuickPath
from .providers.base import GenerationProvider
from .providers.factory import build_provider
from .retrieval import InMemoryRetriever, Retriever
from .verification import PostVerifier


class ReliabilityRouter:
    def __init__(
        self,
        *,
        config: RouterConfig | None = None,
        intent_parser: IntentParser | None = None,
        provider: GenerationProvider | None = None,
        retriever: Retriever | None = None,
    ):
        self.config = config or RouterConfig.from_env()
        self.intent_parser = intent_parser or build_intent_parser(self.config)
        self._validate_intent_parser(self.intent_parser)
        self.provider = provider or build_provider(self.config) # if provider is None, provider = build_provider()
        self._validate_provider(self.provider)
        self.retriever = retriever or InMemoryRetriever()
        self.verifier = PostVerifier()
        self.quick_path = QuickPath(config=self.config, provider=self.provider, verifier=self.verifier)
        self.deep_path = DeepPath(
            config=self.config,
            provider=self.provider,
            retriever=self.retriever,
            verifier=self.verifier,
        )

    def route(self, request: RouterRequest) -> RoutingDecision:
        decision, _ = self._route_with_traces(request)
        return decision

    def answer(self, request: RouterRequest) -> RouterResponse:
        total_start = perf_counter()
        decision, traces = self._route_with_traces(request)
        result = (
            self.quick_path.execute(request, decision)
            if decision.route == Route.QUICK
            else self.deep_path.execute(request, decision)
        )
        traces.extend(result.traces)
        latency_ms = (perf_counter() - total_start) * 1000
        return RouterResponse(
            request_id=request.request_id,
            decision=decision,
            answer=result.draft.answer,
            abstained=result.draft.abstained,
            verification=result.verification,
            evidence=result.evidence,
            provider=result.draft.provider,
            model=result.draft.model,
            latency_ms=round(latency_ms, 4),
            estimated_cost_usd=result.draft.estimated_cost_usd,
            traces=traces,
        )

    def policy_snapshot(self) -> dict[str, Any]:
        return self.config.model_dump()

    @staticmethod
    def _validate_intent_parser(intent_parser: IntentParser) -> None:
        for field in ("name", "version"):
            value = getattr(intent_parser, field, None)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Intent parser must define a non-empty string {field}.")
        if not callable(getattr(intent_parser, "parse", None)):
            raise ValueError("Intent parser must define parse(request).")

    @staticmethod
    def _validate_provider(provider: GenerationProvider) -> None:
        value = getattr(provider, "name", None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Provider must define a non-empty string name.")
        if not callable(getattr(provider, "generate", None)):
            raise ValueError("Provider must define generate(request).")

    def _route_with_traces(self, request: RouterRequest) -> tuple[RoutingDecision, list[ExecutionTrace]]:
        parse_with_traces = getattr(self.intent_parser, "parse_with_traces", None)
        if callable(parse_with_traces):
            return parse_with_traces(request)

        start = perf_counter()
        decision = self.intent_parser.parse(request)
        return decision, [
            ExecutionTrace(
                stage=f"intent.{self.intent_parser.name}.parse",
                duration_ms=(perf_counter() - start) * 1000,
            )
        ]

@lru_cache(maxsize=1)
def get_default_router() -> ReliabilityRouter:
    return ReliabilityRouter()
