"""Backend registry: build adapters from ``"provider:model"`` strings."""

from __future__ import annotations

from typing import Any

from .base import EmbeddingBackend, LLMBackend, LLMError, LLMResponse
from .embeddings import HashingEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder

__all__ = [
    "EmbeddingBackend",
    "LLMBackend",
    "LLMError",
    "LLMResponse",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "create_llm",
    "create_embedder",
]


def _split(spec: str) -> tuple[str, str | None]:
    provider, _, model = spec.partition(":")
    return provider.strip().lower(), (model.strip() or None)


def create_llm(spec: str, **kwargs: Any) -> LLMBackend:
    """``"anthropic"``, ``"anthropic:claude-sonnet-5-5"``, ``"openai:<model>"``."""
    provider, model = _split(spec)
    if provider == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model, **kwargs) if model else AnthropicBackend(**kwargs)
    if provider == "openai":
        from .openai_backend import OpenAIBackend

        if not model:
            raise ValueError("openai backend needs a model: 'openai:<model>'")
        return OpenAIBackend(model=model, **kwargs)
    raise ValueError(f"unknown LLM provider {provider!r}")


def create_embedder(spec: str, **kwargs: Any) -> EmbeddingBackend:
    """``"hashing"``, ``"openai[:model]"``, ``"st[:hf-model-id]"``."""
    provider, model = _split(spec)
    if provider == "hashing":
        return HashingEmbedder(**kwargs)
    if provider == "openai":
        return OpenAIEmbedder(model=model, **kwargs) if model else OpenAIEmbedder(**kwargs)
    if provider in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder(model=model, **kwargs) if model else SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"unknown embedding provider {provider!r}")
