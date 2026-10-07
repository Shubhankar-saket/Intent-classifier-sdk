"""L0 normalisation and conversation windowing."""

from __future__ import annotations

import re
import unicodedata

from .taxonomy import HistoryConfig
from .types import Turn

# Zero-width and bidi control characters are a common way to hide injected text
# from regexes and human reviewers.
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿]")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # folds full-width / stylised letters
    text = _INVISIBLE.sub("", text)
    return _WS.sub(" ", text).strip()


def window_history(history: list[Turn] | None, cfg: HistoryConfig) -> list[Turn]:
    """Keep the last ``max_turns`` turns, then drop the oldest until under ``max_chars``."""
    if not history or cfg.max_turns <= 0:
        return []
    turns = [Turn(role=t.role, content=normalize_text(t.content)) for t in history[-cfg.max_turns :]]
    while turns and sum(len(t.content) for t in turns) > cfg.max_chars:
        turns.pop(0)
    return turns
