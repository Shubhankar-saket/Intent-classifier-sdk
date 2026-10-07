"""Taxonomy = built-in guardrails + built-in universal intents + one project's intents.

Projects only write their own YAML; the shared layers are loaded from ``builtin/``
and can be selectively disabled or have their guardrail actions overridden.
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .types import UNKNOWN, Action, Layer

CORE_VERSION = "0.1.0"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$")


class LabelDef(BaseModel):
    name: str
    description: str
    examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    layer: Layer = "project"

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _NAME_RE.match(v) or v == UNKNOWN:
            raise ValueError(f"invalid label name {v!r}: use lowercase snake_case, dots for hierarchy")
        return v

    @property
    def category(self) -> str:
        return self.name.split(".", 1)[0]


class IntentDef(LabelDef):
    pass


class GuardrailDef(LabelDef):
    layer: Layer = "guardrail"
    action: Action = Action.BLOCK
    # High-precision regexes for the rules stage. Keep them strict: a match can
    # short-circuit the pipeline without an LLM call.
    patterns: list[str] = Field(default_factory=list)


class HistoryConfig(BaseModel):
    max_turns: int = 6
    max_chars: int = 4000


class CoreOverrides(BaseModel):
    disable_intents: list[str] = Field(default_factory=list)
    disable_guardrails: list[str] = Field(default_factory=list)
    guardrail_actions: dict[str, Action] = Field(default_factory=dict)


class ProjectSpec(BaseModel):
    """Schema of a project YAML file."""

    project: str
    version: str = "0.1.0"
    domain: str = Field(description="One or two sentences on what the assistant is for; drives off-topic detection.")
    languages: list[str] = Field(default_factory=lambda: ["en"])
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    core: CoreOverrides = Field(default_factory=CoreOverrides)
    intents: list[IntentDef]


class Taxonomy(BaseModel):
    project: str
    project_version: str
    domain: str
    languages: list[str]
    history: HistoryConfig
    guardrails: list[GuardrailDef]
    intents: list[IntentDef]  # universal + project

    @model_validator(mode="after")
    def _unique_names(self) -> "Taxonomy":
        seen: set[str] = set()
        for label in [*self.guardrails, *self.intents]:
            if label.name in seen:
                raise ValueError(f"duplicate label {label.name!r} (project intents may not reuse built-in names)")
            seen.add(label.name)
        return self

    # -- lookup -----------------------------------------------------------------
    @property
    def intent_names(self) -> list[str]:
        return [i.name for i in self.intents]

    @property
    def guardrail_names(self) -> list[str]:
        return [g.name for g in self.guardrails]

    def intent(self, name: str) -> IntentDef | None:
        return next((i for i in self.intents if i.name == name), None)

    def guardrail(self, name: str) -> GuardrailDef | None:
        return next((g for g in self.guardrails if g.name == name), None)

    def layer_of(self, name: str) -> Layer | None:
        label = self.intent(name) or self.guardrail(name)
        return label.layer if label else None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:10]

    @property
    def version(self) -> str:
        """e.g. ``shopease@0.1.0+core0.1.0#a1b2c3d4e5`` — the hash catches unversioned edits."""
        return f"{self.project}@{self.project_version}+core{CORE_VERSION}#{self.fingerprint}"

    # -- loading ----------------------------------------------------------------
    @classmethod
    def from_spec(cls, spec: ProjectSpec) -> "Taxonomy":
        guardrails = [
            g.model_copy(update={"action": spec.core.guardrail_actions.get(g.name, g.action)})
            for g in load_builtin_guardrails()
            if g.name not in spec.core.disable_guardrails
        ]
        universal = [i for i in load_builtin_universal() if i.name not in spec.core.disable_intents]
        _check_known(spec.core.disable_guardrails, [g.name for g in load_builtin_guardrails()], "guardrail")
        _check_known(list(spec.core.guardrail_actions), [g.name for g in load_builtin_guardrails()], "guardrail")
        _check_known(spec.core.disable_intents, [i.name for i in load_builtin_universal()], "universal intent")
        project_intents = [i.model_copy(update={"layer": "project"}) for i in spec.intents]
        return cls(
            project=spec.project,
            project_version=spec.version,
            domain=spec.domain,
            languages=spec.languages,
            history=spec.history,
            guardrails=guardrails,
            intents=[*universal, *project_intents],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Taxonomy":
        return cls.from_spec(ProjectSpec.model_validate(data))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Taxonomy":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


def _check_known(names: list[str], known: list[str], kind: str) -> None:
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"unknown built-in {kind}(s) in core overrides: {unknown}")


def _load_builtin(filename: str) -> list[dict[str, Any]]:
    text = resources.files("intent_classify.builtin").joinpath(filename).read_text(encoding="utf-8")
    return yaml.safe_load(text)["labels"]


def load_builtin_guardrails() -> list[GuardrailDef]:
    return [GuardrailDef.model_validate({**d, "layer": "guardrail"}) for d in _load_builtin("guardrails.yaml")]


def load_builtin_universal() -> list[IntentDef]:
    return [IntentDef.model_validate({**d, "layer": "universal"}) for d in _load_builtin("universal.yaml")]
