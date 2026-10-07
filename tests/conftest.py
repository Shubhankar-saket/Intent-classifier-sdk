from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from intent_classify import Taxonomy
from intent_classify.backends.base import LLMResponse
from intent_classify.types import Usage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def taxonomy() -> Taxonomy:
    return Taxonomy.from_yaml(ROOT / "projects" / "shopease.yaml")


class FakeLLM:
    """Records calls; answers via a function of (system, user) -> response dict / LLMResponse."""

    name = "fake"
    model = "fake-1"

    def __init__(self, respond: Callable[[str, str], Any]):
        self.respond = respond
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, schema: dict, schema_name: str) -> LLMResponse:
        self.calls.append((system, user))
        out = self.respond(system, user)
        if isinstance(out, LLMResponse):
            return out
        return LLMResponse(data=out, usage=Usage(input_tokens=100, output_tokens=20, llm_calls=1, estimated_cost_usd=0.001))


@pytest.fixture
def fake_llm() -> Callable[[Callable[[str, str], Any]], FakeLLM]:
    return FakeLLM


def llm_answer(intents=(), guardrails=(), language="en", rationale="r") -> dict:
    return {
        "rationale": rationale,
        "guardrails": [{"name": n, "score": s, "evidence": "e"} for n, s in guardrails],
        "intents": [{"name": n, "confidence": c} for n, c in intents],
        "language": language,
    }
