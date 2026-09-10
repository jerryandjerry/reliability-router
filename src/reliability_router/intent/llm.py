"""LLM-backed intent parser.

The initial transport uses ``codex exec`` so local experiments can use the
user's Codex sign-in without this project reading or storing OAuth tokens.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from ..config import RouterConfig
from ..features import FeatureExtractor
from ..models import ExecutionTrace, ReasonCode, RiskBand, Route, RouterRequest, RoutingDecision
from ..scoring import RiskScorer

LLMRunner = Callable[[list[str], str, int], str]


class LLMIntentParser:
    """Use an LLM to map a request to QUICK/DEEP while preserving the router contract."""

    def __init__(self, config: RouterConfig, *, runner: LLMRunner | None = None):
        self.config = config
        self.name = config.intent_parser_provider
        self.model = config.intent_parser_model
        self.thinking = config.intent_parser_thinking
        self.version = f"{self.model}-{self.thinking}"
        self.transport = config.intent_parser_transport
        self.timeout_seconds = config.intent_parser_timeout_seconds
        self.extractor = FeatureExtractor()
        self.scorer = RiskScorer(config)
        if self.transport == "codex":
            self._runner = runner or self._run_codex
        elif self.transport == "claude":
            self._runner = runner or self._run_claude
        else:
            raise ValueError("LLM intent parser supports RR_INTENT_PARSER_TRANSPORT='codex' or 'claude'.")

    def parse(self, request: RouterRequest) -> RoutingDecision:
        decision, _ = self.parse_with_traces(request)
        return decision

    def parse_with_traces(self, request: RouterRequest) -> tuple[RoutingDecision, list[ExecutionTrace]]:
        traces: list[ExecutionTrace] = []

        start = perf_counter()
        features = self.extractor.extract(request)
        assessment = self.scorer.score(features)
        traces.append(ExecutionTrace(stage="intent.llm.extract_features", duration_ms=(perf_counter() - start) * 1000))

        start = perf_counter()
        payload = self._call_llm(request)
        traces.append(
            ExecutionTrace(
                stage=f"intent.llm.{self.name}.parse",
                duration_ms=(perf_counter() - start) * 1000,
                metadata={"model": self.model, "thinking": self.thinking, "transport": self.transport},
            )
        )

        route = Route(str(payload.get("route", "")).upper())
        risk_score = self._bounded_float(payload.get("risk_score", assessment.score))
        reason_codes = self._reason_codes(payload.get("reason_codes", []))
        explanation = str(payload.get("explanation", "")).strip() or (
            f"{route.value} selected by {self.name}/{self.version}."
        )
        return (
            RoutingDecision(
                route=route,
                risk_score=risk_score,
                risk_band=self._risk_band(risk_score),
                threshold=self.config.deep_threshold,
                reason_codes=reason_codes,
                contributions=assessment.contributions,
                forced=bool(payload.get("forced", False)),
                explanation=explanation,
                policy_version=self.config.policy_version,
                features=features,
            ),
            traces,
        )

    def _call_llm(self, request: RouterRequest) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="rr-intent-") as tmp_dir:
            schema_path = Path(tmp_dir) / "schema.json"
            output_path = Path(tmp_dir) / "output.json"
            schema_path.write_text(json.dumps(_OUTPUT_SCHEMA), encoding="utf-8")
            prompt = self._prompt(request)
            output = self._runner(
                self._transport_command(schema_path=schema_path, output_path=output_path),
                prompt,
                self.timeout_seconds,
            )
        payload = self._parse_json(output)
        if not isinstance(payload, dict):
            raise ValueError("LLM intent parser returned invalid JSON.")
        if str(payload.get("route", "")).upper() not in {Route.QUICK.value, Route.DEEP.value}:
            raise ValueError("LLM intent parser must return route QUICK or DEEP.")
        return payload

    def _transport_command(self, *, schema_path: Path, output_path: Path) -> list[str]:
        if self.transport == "claude":
            return [
                "claude",
                "--print",
                "--model",
                self.model,
                "--effort",
                self.thinking,
                "--json-schema",
                schema_path.read_text(encoding="utf-8"),
            ]
        return [
            "codex",
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.thinking}"',
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            "-",
        ]

    @staticmethod
    def _run_codex(command: list[str], prompt: str, timeout_seconds: int) -> str:
        output_path = Path(command[command.index("-o") + 1])
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"codex exec failed: {completed.stderr.strip() or completed.stdout.strip()}")
        if output_path.exists():
            return output_path.read_text(encoding="utf-8")
        return completed.stdout

    @staticmethod
    def _run_claude(command: list[str], prompt: str, timeout_seconds: int) -> str:
        completed = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"claude failed: {completed.stderr.strip() or completed.stdout.strip()}")
        return completed.stdout

    @staticmethod
    def _prompt(request: RouterRequest) -> str:
        context = [
            {"id": item.id, "text": item.text, "source": item.source, "trust": item.trust}
            for item in request.context
        ]
        payload = {
            "query": request.query,
            "context": context,
            "available_tools": request.available_tools,
        }
        reason_values = [code.value for code in ReasonCode if code != ReasonCode.VERIFICATION_FAILED]
        return (
            "You are an intent parser for a reliability router. Decide whether the request should use QUICK "
            "or DEEP. QUICK is for low-risk requests that can be answered without retrieval, citations, fresh "
            "data, high-stakes judgment, multi-step analysis, or prompt-injection handling. DEEP is for any "
            "request needing evidence, current information, citations, safety-critical judgment, multi-step "
            "reasoning, tool use, or injection resistance.\n\n"
            "Return JSON only with keys: route, risk_score, reason_codes, forced, explanation. "
            f"route must be QUICK or DEEP. risk_score must be 0..1. reason_codes must use only: {reason_values}.\n\n"
            f"Request JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        candidates = [stripped]
        if "```" in stripped:
            parts = stripped.split("```")
            candidates.extend(part.removeprefix("json").strip() for part in parts)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1])
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _bounded_float(value: Any) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 4)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _reason_codes(values: Any) -> list[ReasonCode]:
        if not isinstance(values, list):
            return []
        codes: list[ReasonCode] = []
        for value in values:
            try:
                code = ReasonCode(str(value))
            except ValueError:
                continue
            if code != ReasonCode.VERIFICATION_FAILED and code not in codes:
                codes.append(code)
        return codes

    def _risk_band(self, score: float) -> RiskBand:
        if score >= self.config.critical_threshold:
            return RiskBand.CRITICAL
        if score >= self.config.deep_threshold:
            return RiskBand.HIGH
        if score >= self.config.deep_threshold * 0.62:
            return RiskBand.MEDIUM
        return RiskBand.LOW


_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "route": {"type": "string", "enum": ["QUICK", "DEEP"]},
        "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
        "reason_codes": {"type": "array", "items": {"type": "string"}},
        "forced": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["route", "risk_score", "reason_codes", "forced", "explanation"],
}
