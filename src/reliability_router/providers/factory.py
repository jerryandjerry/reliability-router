"""Provider construction."""

from __future__ import annotations

from ..config import RouterConfig
from .base import GenerationProvider
from .mock import MockProvider
from .openai_provider import OpenAIProvider


def build_provider(config: RouterConfig) -> GenerationProvider:
    provider = config.provider.lower().strip()
    if provider == "mock":
        return MockProvider()
    if provider == "openai":
        return OpenAIProvider(quick_model=config.quick_model, deep_model=config.deep_model)
    raise ValueError(f"Unsupported RR_PROVIDER={config.provider!r}. Expected 'mock' or 'openai'.")
