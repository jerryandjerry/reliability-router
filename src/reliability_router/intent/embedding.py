"""Embedding intent parser for exported dense-vector models.

Two variants share this runtime, distinguished only by what sits on top of the frozen encoder:

  embedding_cls  a logistic regression fitted on training embeddings
  embedding_knn  a distance-weighted vote over stored training embeddings

Neither fine-tunes the encoder, so both are cheap to rebuild. The concrete method and variant come
from the exported manifest, so adding a variant needs no code here.
"""

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


class EmbeddingIntentParser:
    """Load an exported embedding model and map requests to QUICK/DEEP decisions."""

    def __init__(self, config: RouterConfig, *, classifier: ProbabilityClassifier | None = None):
        self.config = config
        self.model_path = Path(config.intent_parser_model)
        self.extractor = FeatureExtractor()
        self.scorer = RiskScorer(config)
        self.manifest = self._load_manifest(self.model_path)
        self.name = self._required_manifest_string("name")
        self.version = self._required_manifest_string("version")
        self.variant = str(self.manifest.get("variant", "cls"))
        self.threshold = float(self.manifest.get("threshold", config.intent_parser_threshold))
        self._classifier = classifier or self._load_classifier(self.model_path, self.manifest)

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
                metadata={
                    "model_path": str(self.model_path),
                    "name": self.name,
                    "version": self.version,
                    "variant": self.variant,
                },
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
        manifest_path = model_path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Embedding manifest not found: {manifest_path}")
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Embedding manifest must be a JSON object: {manifest_path}")
        return loaded

    def _required_manifest_string(self, field: str) -> str:
        value = str(self.manifest.get(field) or "").strip()
        if not value:
            raise ValueError(f"Embedding manifest field {field!r} must be a non-empty string.")
        return value

    @staticmethod
    def _load_classifier(model_path: Path, manifest: dict[str, Any]) -> ProbabilityClassifier:
        if not model_path.exists():
            raise ValueError(
                f"Embedding model path does not exist: {model_path}. "
                "Train/export a model under models/intent_embedding/exported/<version> first."
            )
        try:
            import joblib
            import numpy as np
        except ImportError as exc:
            raise ImportError("Embedding intent parser requires: uv sync --extra embedding") from exc

        def load_encoder(name: str):
            """Imported lazily and deliberately late -- see the ordering note below."""
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Embedding intent parser requires: uv sync --extra embedding") from exc
            return SentenceTransformer(name)

        encoder_name = str(manifest.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
        variant = str(manifest.get("variant", "cls"))

        if variant == "cls":
            path = model_path / "classifier.joblib"
            if not path.exists():
                raise ValueError(f"Embedding classifier not found: {path}")

            # Load and warm the classifier BEFORE the encoder. LightGBM and XGBoost initialise
            # their own OpenMP runtime, and on macOS doing that after torch has initialised its
            # own segfaults on the first predict. Touching the classifier first makes the order
            # deterministic and costs one dummy prediction.
            clf = joblib.load(path)
            try:
                clf.predict_proba(np.zeros((1, clf.n_features_in_), dtype="float32"))
            except Exception:  # noqa: BLE001 - a warm-up failure must not block loading
                pass

            encoder = load_encoder(encoder_name)

            def classify(text: str) -> float:
                vector = encoder.encode([text], normalize_embeddings=True)
                return float(clf.predict_proba(vector)[0][1])

            return classify

        encoder = load_encoder(encoder_name)

        if variant == "knn":
            path = model_path / "index.npz"
            if not path.exists():
                raise ValueError(f"Embedding index not found: {path}")
            store = np.load(path)
            vectors, labels = store["vectors"], store["labels"]
            k = int(manifest.get("k", 7))

            def classify(text: str) -> float:
                query = encoder.encode([text], normalize_embeddings=True)[0]
                # vectors are unit-normalised, so a dot product is cosine similarity
                similarity = vectors @ query
                top = np.argpartition(-similarity, min(k, len(similarity) - 1))[:k]
                weights = np.clip(similarity[top], 0.0, None)
                if weights.sum() <= 0:
                    return float(labels[top].mean())
                return float((weights * labels[top]).sum() / weights.sum())

            return classify

        raise ValueError(f"Unknown embedding variant {variant!r}; expected 'cls' or 'knn'.")

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
