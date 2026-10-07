"""Provider-agnostic backend interfaces. Adapters live next to this file."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..types import Usage


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    data: dict[str, Any] | None
    usage: Usage = field(default_factory=Usage)
    # Set when the provider's own safety layer declined the request. For a guardrail
    # classifier that is itself a strong signal, so the pipeline turns it into a flag.
    refusal: str | None = None


@runtime_checkable
class LLMBackend(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str, schema: dict[str, Any], schema_name: str) -> LLMResponse:
        """Return JSON matching ``schema``. ``system`` is stable across calls (cacheable)."""
        ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str
    # Cosine similarity above which a nearest-neighbour match is trusted without an LLM.
    # Depends heavily on the model, so each backend ships its own default.
    default_accept_threshold: float

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised vectors."""
        ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)
