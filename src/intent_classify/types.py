"""Public data types: conversation input and the stable classification result contract."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]
Layer = Literal["guardrail", "universal", "project"]
Source = Literal["rules", "embedding", "llm"]

UNKNOWN = "unknown"


class Turn(BaseModel):
    role: Role
    content: str


class Action(str, Enum):
    """What the caller should do. Ordered by severity (see ``SEVERITY``)."""

    ALLOW = "allow"
    FLAG = "flag"  # proceed, but log / mask / monitor
    ESCALATE = "escalate"  # route to a human or a safe flow (e.g. self-harm)
    BLOCK = "block"  # refuse to process


SEVERITY = {Action.ALLOW: 0, Action.FLAG: 1, Action.ESCALATE: 2, Action.BLOCK: 3}


class GuardrailFlag(BaseModel):
    name: str
    score: float
    action: Action
    source: Source
    evidence: str | None = None


class GuardrailResult(BaseModel):
    action: Action = Action.ALLOW
    flags: list[GuardrailFlag] = Field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return self.action != Action.ALLOW


class IntentPrediction(BaseModel):
    name: str
    confidence: float
    layer: Layer | None = None  # None for the synthetic "unknown" label
    source: Source


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    llm_calls: int = 0
    estimated_cost_usd: float | None = None

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.llm_calls += other.llm_calls
        if other.estimated_cost_usd is not None:
            self.estimated_cost_usd = (self.estimated_cost_usd or 0.0) + other.estimated_cost_usd


class ClassificationResult(BaseModel):
    """Stable output contract. Add fields; never rename or remove them."""

    guardrail: GuardrailResult
    # intents[0] is the primary, task-level intent of the latest user message, resolved
    # with conversation context ("yes" -> the thing being confirmed). Dialogue acts such
    # as `affirm` follow as secondary intents. Below the abstain threshold intents[0] is
    # `unknown` and the low-confidence candidates follow it.
    intents: list[IntentPrediction]
    language: str | None = None
    history_turns_used: int = 0
    stages: list[Source] = Field(default_factory=list)
    taxonomy_version: str
    latency_ms: float
    usage: Usage = Field(default_factory=Usage)
    rationale: str | None = None
    # Non-empty when a stage failed and the result came from a cheaper fallback stage.
    errors: list[str] = Field(default_factory=list)

    @property
    def top_intent(self) -> IntentPrediction:
        return self.intents[0]

    @property
    def is_unknown(self) -> bool:
        return self.top_intent.name == UNKNOWN

    @property
    def multi_intent(self) -> bool:
        return sum(1 for i in self.intents if i.name != UNKNOWN and i.confidence >= 0.5) > 1
