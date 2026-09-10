"""Evidence-first path with configurable repair passes and fail-closed behavior."""

from __future__ import annotations

from time import perf_counter

from ..config import RouterConfig
from ..models import DraftAnswer, ExecutionTrace, ReasonCode, Route, RouterRequest, RoutingDecision
from ..providers.base import GenerationProvider, GenerationRequest
from ..retrieval import Retriever
from ..verification import PostVerifier
from .base import PathResult


class DeepPath:
    def __init__(
        self,
        *,
        config: RouterConfig,
        provider: GenerationProvider,
        retriever: Retriever,
        verifier: PostVerifier,
    ):
        self.config = config
        self.provider = provider
        self.retriever = retriever
        self.verifier = verifier

    def execute(self, request: RouterRequest, decision: RoutingDecision) -> PathResult:
        traces: list[ExecutionTrace] = []
        retrieve_start = perf_counter()
        evidence = self.retriever.retrieve(
            query=request.query,
            context=request.context,
            top_k=self.config.max_evidence_items,
        )
        traces.append(
            ExecutionTrace(
                stage="deep.retrieve",
                duration_ms=(perf_counter() - retrieve_start) * 1000,
                metadata={"evidence_count": len(evidence)},
            )
        )

        draft, generation_ms = self._generate(request, evidence, repair_feedback=[])
        traces.append(ExecutionTrace(stage="deep.generate", duration_ms=generation_ms))

        verify_start = perf_counter()
        report = self.verifier.verify(
            route=Route.DEEP,
            draft=draft,
            evidence=evidence,
            features=decision.features,
        )
        traces.append(ExecutionTrace(stage="deep.verify", duration_ms=(perf_counter() - verify_start) * 1000))

        attempt = 0
        while not report.passed and attempt < self.config.max_repair_attempts and not draft.abstained:
            attempt += 1
            feedback = [f"{violation.code}: {violation.message}" for violation in report.violations]
            draft, repair_ms = self._generate(request, evidence, repair_feedback=feedback)
            traces.append(
                ExecutionTrace(stage="deep.repair", duration_ms=repair_ms, metadata={"attempt": attempt})
            )
            reverify_start = perf_counter()
            report = self.verifier.verify(
                route=Route.DEEP,
                draft=draft,
                evidence=evidence,
                features=decision.features,
            )
            report.repair_attempted = True
            traces.append(
                ExecutionTrace(
                    stage="deep.reverify",
                    duration_ms=(perf_counter() - reverify_start) * 1000,
                    metadata={"attempt": attempt},
                )
            )

        if not report.passed and self.config.fail_closed:
            decision.reason_codes = list(
                dict.fromkeys([*decision.reason_codes, ReasonCode.VERIFICATION_FAILED])
            )
            draft = DraftAnswer(
                answer=(
                    "The candidate answer failed reliability checks, so the router withheld it. "
                    "Add trusted evidence, a freshness-capable tool, or refine the question."
                ),
                claims=[],
                confidence=0.0,
                abstained=True,
                provider=draft.provider,
                model=draft.model,
                input_tokens=draft.input_tokens,
                output_tokens=draft.output_tokens,
                estimated_cost_usd=draft.estimated_cost_usd,
            )

        return PathResult(draft=draft, verification=report, evidence=evidence, traces=traces)

    def _generate(
        self, request: RouterRequest, evidence: list, repair_feedback: list[str]
    ) -> tuple[DraftAnswer, float]:
        start = perf_counter()
        draft = self.provider.generate(
            GenerationRequest(
                query=request.query,
                route=Route.DEEP,
                evidence=evidence,
                max_output_tokens=self.config.deep_max_output_tokens,
                repair_feedback=repair_feedback,
            )
        )
        return draft, (perf_counter() - start) * 1000
