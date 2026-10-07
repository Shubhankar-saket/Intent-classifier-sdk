"""Flag eval messages that are near-copies of taxonomy examples (which would inflate scores)."""

from __future__ import annotations

from ..backends.embeddings import HashingEmbedder
from ..taxonomy import Taxonomy
from .dataset import EvalCase


def find_leaks(cases: list[EvalCase], taxonomy: Taxonomy, threshold: float = 0.85) -> list[tuple[str, str, str, float]]:
    """Return (case_id, eval message, taxonomy example, similarity) above ``threshold``."""
    examples = [ex for label in [*taxonomy.intents, *taxonomy.guardrails] for ex in label.examples]
    if not examples or not cases:
        return []
    emb = HashingEmbedder()
    sims = emb.embed([c.message for c in cases]) @ emb.embed(examples).T
    leaks = []
    for i, case in enumerate(cases):
        j = int(sims[i].argmax())
        if sims[i, j] >= threshold:
            leaks.append((case.id, case.message, examples[j], float(sims[i, j])))
    return sorted(leaks, key=lambda x: -x[3])
