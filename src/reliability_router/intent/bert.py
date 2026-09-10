"""BERT-family intent parser for exported supervised classifiers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from ..config import RouterConfig
from ..features import FeatureExtractor
from ..models import ExecutionTrace, RiskBand, Route, RouterRequest, RoutingDecision
from ..scoring import RiskScorer
from .text import request_to_classifier_text

ProbabilityClassifier = Callable[[str], float]


class BertFamilyIntentParser:
    """Load an exported BERT-family classifier and map requests to QUICK/DEEP decisions."""

    def __init__(self, config: RouterConfig, *, classifier: ProbabilityClassifier | None = None):
        self.config = config
        self.model_path = Path(config.intent_parser_model)
        self.extractor = FeatureExtractor()
        self.scorer = RiskScorer(config)
        self.manifest = self._load_manifest(self.model_path)
        self.name = self._required_manifest_string("name")
        self.version = self._required_manifest_string("version")
        self.threshold = float(self.manifest.get("threshold", config.intent_parser_threshold))
        self.max_length = int(self.manifest.get("max_length", 512))
        self._classifier = classifier or self._load_classifier(self.model_path, self.max_length)

    def parse(self, request: RouterRequest) -> RoutingDecision:
        decision, _ = self.parse_with_traces(request)
        return decision

    def parse_with_traces(self, request: RouterRequest) -> tuple[RoutingDecision, list[ExecutionTrace]]:
        traces: list[ExecutionTrace] = []

        start = perf_counter()
        features = self.extractor.extract(request)
        assessment = self.scorer.score(features)
        traces.append(
            ExecutionTrace(stage=f"intent.{self.name}.extract_features", duration_ms=(perf_counter() - start) * 1000)
        )

        text = request_to_classifier_text(request)
        start = perf_counter()
        deep_probability = _bounded_probability(self._classifier(text))
        traces.append(
            ExecutionTrace(
                stage=f"intent.{self.name}.classify",
                duration_ms=(perf_counter() - start) * 1000,
                metadata={"model_path": str(self.model_path), "name": self.name, "version": self.version},
            )
        )

        route = Route.DEEP if deep_probability >= self.threshold else Route.QUICK
        return (
            RoutingDecision(
                route=route,
                risk_score=deep_probability,
                risk_band=self._risk_band(deep_probability),
                threshold=self.threshold,
                reason_codes=[],
                contributions=assessment.contributions,
                forced=False,
                explanation=f"{route.value} selected by {self.name}/{self.version} at p(DEEP)={deep_probability:.4f}.",
                policy_version=self.config.policy_version,
                features=features,
            ),
            traces,
        )

    @staticmethod
    def _load_manifest(model_path: Path) -> dict[str, Any]:
        if not model_path.exists():
            raise ValueError(
                f"BERT model path does not exist: {model_path}. "
                "Train/export a model under models/intent_bert/exported/<version> first."
            )
        manifest_path = model_path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"BERT manifest not found: {manifest_path}")
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"BERT manifest must be a JSON object: {manifest_path}")
        return loaded

    def _required_manifest_string(self, field: str) -> str:
        value = str(self.manifest.get(field) or "").strip()
        if not value:
            raise ValueError(f"BERT-family manifest field {field!r} must be a non-empty string.")
        return value

    @staticmethod
    def _load_classifier(model_path: Path, max_length: int) -> ProbabilityClassifier:
        if not model_path.exists():
            raise ValueError(
                f"BERT model path does not exist: {model_path}. "
                "Train/export a model under models/intent_bert/exported/<version> first."
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("BERT intent parser requires: uv sync --extra bert") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        def classify(text: str) -> float:
            encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            with torch.no_grad():
                logits = model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            return float(probabilities[1].item())

        return classify

    def _risk_band(self, score: float) -> RiskBand:
        if score >= self.config.critical_threshold:
            return RiskBand.CRITICAL
        if score >= self.threshold:
            return RiskBand.HIGH
        if score >= self.threshold * 0.62:
            return RiskBand.MEDIUM
        return RiskBand.LOW


def _bounded_probability(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


BertIntentParser = BertFamilyIntentParser
