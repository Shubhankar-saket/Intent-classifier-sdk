"""Embedding stage: nearest-neighbour match against the taxonomy's examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backends.base import EmbeddingBackend
from ..taxonomy import Taxonomy


@dataclass
class EmbeddingOutput:
    ranking: list[tuple[str, float]]  # intents only, best first
    guardrail_ranking: list[tuple[str, float]]  # guardrails only, best first

    @property
    def top(self) -> tuple[str, float]:
        return self.ranking[0]

    @property
    def margin(self) -> float:
        return self.ranking[0][1] - self.ranking[1][1] if len(self.ranking) > 1 else self.ranking[0][1]

    @property
    def max_guardrail_similarity(self) -> float:
        return self.guardrail_ranking[0][1] if self.guardrail_ranking else 0.0


class EmbeddingStage:
    def __init__(self, taxonomy: Taxonomy, embedder: EmbeddingBackend):
        self.embedder = embedder
        texts: list[str] = []
        labels: list[str] = []
        is_guard: list[bool] = []
        for label in [*taxonomy.intents, *taxonomy.guardrails]:
            for ex in label.examples or [label.description]:
                texts.append(ex)
                labels.append(label.name)
                is_guard.append(label.layer == "guardrail")
        self._labels = np.array(labels)
        self._is_guard = np.array(is_guard)
        self._matrix = embedder.embed(texts)

    def run(self, message: str) -> EmbeddingOutput:
        sims = self._matrix @ self.embedder.embed([message])[0]
        best: dict[str, float] = {}
        for label, sim in zip(self._labels, sims):
            if sim > best.get(label, -1.0):
                best[label] = float(sim)
        guard_names = set(self._labels[self._is_guard])
        intents = sorted(((k, v) for k, v in best.items() if k not in guard_names), key=lambda kv: -kv[1])
        guards = sorted(((k, v) for k, v in best.items() if k in guard_names), key=lambda kv: -kv[1])
        return EmbeddingOutput(ranking=intents, guardrail_ranking=guards)
