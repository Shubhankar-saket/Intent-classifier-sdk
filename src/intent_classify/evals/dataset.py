from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..taxonomy import Taxonomy
from ..types import UNKNOWN, Turn


class EvalCase(BaseModel):
    id: str
    message: str
    history: list[Turn] = Field(default_factory=list)
    expected_intent: str | None = None
    acceptable_intents: list[str] = Field(default_factory=list)
    expected_guardrails: list[str] = Field(default_factory=list)  # any-of
    tags: list[str] = Field(default_factory=list)
    suite: str = ""


def load_cases(paths: list[str | Path]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for p in paths:
        p = Path(p)
        with open(p, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    case = EvalCase.model_validate({**json.loads(line), "suite": p.stem})
                except Exception as e:  # noqa: BLE001 - surface file/line for any bad row
                    raise ValueError(f"{p}:{lineno}: {e}") from e
                if case.id in seen:
                    raise ValueError(f"{p}:{lineno}: duplicate case id {case.id!r}")
                seen.add(case.id)
                cases.append(case)
    return cases


def validate_against(cases: list[EvalCase], taxonomy: Taxonomy) -> list[str]:
    """Return problems such as labels that don't exist in the taxonomy."""
    intents = set(taxonomy.intent_names) | {UNKNOWN}
    guards = set(taxonomy.guardrail_names)
    problems = []
    for c in cases:
        for name in filter(None, [c.expected_intent, *c.acceptable_intents]):
            if name not in intents:
                problems.append(f"{c.id}: unknown intent {name!r}")
        for name in c.expected_guardrails:
            if name not in guards:
                problems.append(f"{c.id}: unknown guardrail {name!r}")
    return problems
