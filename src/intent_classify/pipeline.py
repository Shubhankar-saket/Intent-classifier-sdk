"""The cascade: normalise -> rules -> embeddings -> LLM (only when needed) -> policy."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from .backends import create_embedder, create_llm
from .backends.base import EmbeddingBackend, LLMBackend, LLMError
from .normalize import normalize_text, window_history
from .stages.embedding import EmbeddingOutput, EmbeddingStage
from .stages.llm import LLMStage
from .stages.rules import RulesStage
from .taxonomy import Taxonomy
from .types import (
    SEVERITY,
    UNKNOWN,
    Action,
    ClassificationResult,
    GuardrailFlag,
    GuardrailResult,
    IntentPrediction,
    Source,
    Turn,
    Usage,
)


class ClassifierConfig(BaseModel):
    use_history: bool = True
    # Embedding fast path: answer without the LLM when the nearest-neighbour match is clear.
    fast_path: bool = True
    accept_threshold: float | None = None  # None -> the embedder's own default
    margin_threshold: float = 0.05
    guard_risk_margin: float = 0.15  # fast path is refused if any guardrail example is within this of accept
    min_words_for_fast_path_with_history: int = 4  # short replies need context -> LLM
    # Rules stage
    short_circuit_rule_blocks: bool = False  # skip the LLM when a rule fires a `block` guardrail
    rules_trust_threshold: float = 0.95  # rule flags below this need LLM agreement when the LLM runs
    # Policy
    guardrail_threshold: float = 0.5  # a flag affects the action only at/above this score
    abstain_threshold: float = 0.4
    provider_refusal_action: Action = Action.BLOCK
    # Offline mode (no LLM configured)
    offline_min_similarity: float = 0.3
    # Errors: "fallback" degrades to embeddings/rules and records the error; "raise" propagates.
    on_llm_error: str = "fallback"
    max_examples_in_prompt: int = 5


class IntentClassifier:
    def __init__(
        self,
        taxonomy: Taxonomy,
        llm: LLMBackend | None = None,
        embedder: EmbeddingBackend | None = None,
        config: ClassifierConfig | None = None,
    ):
        self.taxonomy = taxonomy
        self.config = config or ClassifierConfig()
        self.rules = RulesStage(taxonomy)
        self.embedding = EmbeddingStage(taxonomy, embedder) if embedder else None
        self.llm = LLMStage(taxonomy, llm, self.config.max_examples_in_prompt) if llm else None
        self._accept = self.config.accept_threshold or (embedder.default_accept_threshold if embedder else 1.0)

    @classmethod
    def from_project(
        cls,
        project: str | Path | Taxonomy,
        llm: str | LLMBackend | None = "anthropic",
        embedder: str | EmbeddingBackend | None = "hashing",
        **config: object,
    ) -> "IntentClassifier":
        taxonomy = project if isinstance(project, Taxonomy) else Taxonomy.from_yaml(project)
        llm_backend = create_llm(llm) if isinstance(llm, str) else llm
        emb_backend = create_embedder(embedder) if isinstance(embedder, str) else embedder
        return cls(taxonomy, llm_backend, emb_backend, ClassifierConfig(**config))

    # ------------------------------------------------------------------ public API
    def classify(self, message: str, history: Iterable[Turn | dict] | None = None) -> ClassificationResult:
        start = time.perf_counter()
        cfg = self.config
        msg = normalize_text(message)
        turns = [t if isinstance(t, Turn) else Turn.model_validate(t) for t in (history or [])]
        hist = window_history(turns, self.taxonomy.history) if cfg.use_history else []
        stages: list[Source] = ["rules"]
        usage = Usage()
        errors: list[str] = []

        def done(flags, intents, **extra) -> ClassificationResult:
            return self._finish(flags, intents, hist, stages, usage, errors, start, **extra)

        rules = self.rules.run(msg, hist)
        if cfg.short_circuit_rule_blocks and any(f.action == Action.BLOCK for f in rules.flags):
            return done(rules.flags, [rules.intent] if rules.intent else [])
        if rules.intent and not rules.flags:
            return done([], [rules.intent])

        emb: EmbeddingOutput | None = None
        if self.embedding:
            stages.append("embedding")
            emb = self.embedding.run(msg)
            if self.llm and cfg.fast_path and not rules.flags and self._fast_path_ok(msg, hist, emb):
                return done([], [self._emb_prediction(*emb.top)])

        if self.llm:
            stages.append("llm")
            try:
                out = self.llm.run(msg, hist)
            except (LLMError, ValueError) as e:  # ValueError: malformed JSON
                if cfg.on_llm_error == "raise":
                    raise
                errors.append(f"llm: {e}")
            else:
                usage.add(out.usage)
                flags = self._merge_flags(rules.flags, out.flags)
                if out.refusal is not None:
                    flags.append(
                        GuardrailFlag(
                            name="provider_refusal", score=1.0, action=cfg.provider_refusal_action,
                            source="llm", evidence=f"model declined to classify ({out.refusal})",
                        )
                    )
                    return done(flags, self._offline_intents(rules.intent, emb))
                return done(flags, out.intents, language=out.language, rationale=out.rationale)

        # Offline mode, or the LLM failed and we degrade.
        flags = list(rules.flags)
        if emb:
            existing = {f.name for f in flags}
            for name, sim in emb.guardrail_ranking:
                if sim >= self._accept and name not in existing:
                    g = self.taxonomy.guardrail(name)
                    flags.append(GuardrailFlag(name=name, score=sim, action=g.action, source="embedding"))
        return done(flags, self._offline_intents(rules.intent, emb))

    def classify_many(
        self, items: list[tuple[str, list[Turn] | None]], max_workers: int = 8
    ) -> list[ClassificationResult]:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda it: self.classify(it[0], it[1]), items))

    # ------------------------------------------------------------------ internals
    def _fast_path_ok(self, msg: str, hist: list[Turn], emb: EmbeddingOutput) -> bool:
        cfg = self.config
        if hist and len(msg.split()) < cfg.min_words_for_fast_path_with_history:
            return False
        _, score = emb.top
        return (
            score >= self._accept
            and emb.margin >= cfg.margin_threshold
            and emb.max_guardrail_similarity < self._accept - cfg.guard_risk_margin
        )

    def _emb_prediction(self, name: str, score: float) -> IntentPrediction:
        return IntentPrediction(name=name, confidence=round(score, 4), layer=self.taxonomy.layer_of(name), source="embedding")

    def _offline_intents(self, rule_intent: IntentPrediction | None, emb: EmbeddingOutput | None) -> list[IntentPrediction]:
        if rule_intent:
            return [rule_intent]
        if emb and emb.ranking and emb.top[1] >= self.config.offline_min_similarity:
            return [self._emb_prediction(n, s) for n, s in emb.ranking[:3]]
        return []

    def _merge_flags(self, rule_flags: list[GuardrailFlag], llm_flags: list[GuardrailFlag]) -> list[GuardrailFlag]:
        """The LLM can veto heuristic rule hits; validated detectors always stand."""
        merged = {f.name: f for f in llm_flags}
        for rf in rule_flags:
            if rf.name in merged:
                if rf.score > merged[rf.name].score:
                    merged[rf.name] = rf
            elif rf.score >= self.config.rules_trust_threshold:
                merged[rf.name] = rf
        return list(merged.values())

    def _finish(
        self,
        flags: list[GuardrailFlag],
        intents: list[IntentPrediction],
        hist: list[Turn],
        stages: list[Source],
        usage: Usage,
        errors: list[str],
        start: float,
        language: str | None = None,
        rationale: str | None = None,
    ) -> ClassificationResult:
        cfg = self.config
        action = Action.ALLOW
        for f in flags:
            if f.score >= cfg.guardrail_threshold and SEVERITY[f.action] > SEVERITY[action]:
                action = f.action
        # Flags that drive the action first, then sub-threshold ones kept for monitoring.
        flags = sorted(flags, key=lambda f: (f.score < cfg.guardrail_threshold, -SEVERITY[f.action], -f.score))

        intents = [i for i in intents if i.name != UNKNOWN or i is intents[0]]
        if not intents:
            intents = [IntentPrediction(name=UNKNOWN, confidence=1.0, source=stages[-1])]
        elif intents[0].name != UNKNOWN and intents[0].confidence < cfg.abstain_threshold:
            top = intents[0]
            intents = [IntentPrediction(name=UNKNOWN, confidence=round(1 - top.confidence, 4), source=top.source), *intents]

        return ClassificationResult(
            guardrail=GuardrailResult(action=action, flags=flags),
            intents=intents,
            language=language,
            history_turns_used=len(hist),
            stages=stages,
            taxonomy_version=self.taxonomy.version,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            usage=usage,
            rationale=rationale,
            errors=errors,
        )
