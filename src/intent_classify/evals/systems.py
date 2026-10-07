"""Systems compared in an eval run.

``zero_shot`` is the status quo we need to beat: one prompt listing label names, latest
message only. The rest are ablations of the library pipeline, so the report shows what
each layer buys in accuracy, safety, cost and latency.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from ..backends.base import EmbeddingBackend, LLMBackend, LLMError
from ..pipeline import ClassifierConfig, IntentClassifier
from ..taxonomy import Taxonomy
from ..types import (
    UNKNOWN,
    Action,
    ClassificationResult,
    GuardrailFlag,
    GuardrailResult,
    IntentPrediction,
    Turn,
)

SYSTEM_DESCRIPTIONS = {
    "zero_shot": "Baseline: one LLM prompt with label names only, latest message only",
    "rules": "Rules only (regex + validated PII detectors + exact matches); no ML",
    "embeddings": "Rules + embedding nearest-neighbour; no LLM",
    "full_no_history": "Full cascade (rules -> embeddings -> LLM) without conversation history",
    "full_llm_only": "Rules + LLM on every message (no embedding fast path), with history",
    "full": "Full cascade with conversation history",
}
LLM_SYSTEMS = {"zero_shot", "full_no_history", "full_llm_only", "full"}


class Classifies(Protocol):
    def classify(self, message: str, history: list[Turn] | None = None) -> ClassificationResult: ...


class ZeroShotClassifier:
    """What a team typically ships first. Kept deliberately simple — it is the baseline."""

    def __init__(self, taxonomy: Taxonomy, llm: LLMBackend):
        self.taxonomy = taxonomy
        self.llm = llm
        labels = [*taxonomy.intent_names, *taxonomy.guardrail_names, UNKNOWN]
        self.system = (
            "Classify the user's message into exactly one of these labels: "
            + ", ".join(labels)
            + ". Respond with JSON."
        )
        self.schema = {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": labels}},
            "required": ["label"],
            "additionalProperties": False,
        }

    def classify(self, message: str, history: list[Turn] | None = None) -> ClassificationResult:
        start = time.perf_counter()
        errors: list[str] = []
        label, usage = UNKNOWN, None
        try:
            resp = self.llm.complete_json(self.system, message, self.schema, "label")
            usage = resp.usage
            if resp.data:
                label = resp.data.get("label", UNKNOWN)
        except (LLMError, ValueError) as e:
            errors.append(f"llm: {e}")
        flags, intents = [], [IntentPrediction(name=UNKNOWN, confidence=1.0, source="llm")]
        guard = self.taxonomy.guardrail(label)
        if guard:
            flags = [GuardrailFlag(name=label, score=1.0, action=guard.action, source="llm")]
        elif label != UNKNOWN:
            intents = [IntentPrediction(name=label, confidence=1.0, layer=self.taxonomy.layer_of(label), source="llm")]
        result = ClassificationResult(
            guardrail=GuardrailResult(action=flags[0].action if flags else Action.ALLOW, flags=flags),
            intents=intents,
            stages=["llm"],
            taxonomy_version=self.taxonomy.version,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            errors=errors,
        )
        if usage:
            result.usage = usage
        return result


def build_system(
    name: str,
    taxonomy: Taxonomy,
    llm_factory: Callable[[], LLMBackend] | None,
    embedder: EmbeddingBackend | None,
    config: ClassifierConfig | None = None,
) -> Classifies:
    base = config or ClassifierConfig()
    if name in LLM_SYSTEMS and llm_factory is None:
        raise ValueError(f"system {name!r} needs an LLM backend")
    if name == "zero_shot":
        return ZeroShotClassifier(taxonomy, llm_factory())
    if name == "rules":
        return IntentClassifier(taxonomy, None, None, base)
    if name == "embeddings":
        return IntentClassifier(taxonomy, None, embedder, base)
    if name == "full_no_history":
        return IntentClassifier(taxonomy, llm_factory(), embedder, base.model_copy(update={"use_history": False}))
    if name == "full_llm_only":
        return IntentClassifier(taxonomy, llm_factory(), None, base)
    if name == "full":
        return IntentClassifier(taxonomy, llm_factory(), embedder, base)
    raise ValueError(f"unknown system {name!r}; choose from {list(SYSTEM_DESCRIPTIONS)}")
