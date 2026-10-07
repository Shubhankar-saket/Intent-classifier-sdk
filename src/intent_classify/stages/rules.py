"""Rules stage: free, deterministic, runs on every message.

Two kinds of signal, distinguished by score:
- heuristic regexes from guardrail ``patterns`` (``PATTERN_SCORE``) — the LLM may veto these;
- validated detectors such as Luhn-checked card numbers (``VALIDATED_SCORE``) — always kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..taxonomy import Taxonomy
from ..types import GuardrailFlag, IntentPrediction, Turn

PATTERN_SCORE = 0.9
VALIDATED_SCORE = 0.98
EXACT_INTENT_SCORE = 0.97

_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_AADHAAR = re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
# "pin" is deliberately absent: Indian postal PIN codes are routine in support chats.
_SECRET_CUE = re.compile(r"\b(password|passcode|otp|cvv|cvc)\b\s*(is|:|=)\s*\S+", re.I)

# Exact-match fast path for unambiguous one-liners. Context-dependent words such as
# "yes"/"no" are only matched when there is no history (see ``RulesStage.run``).
_EXACT: dict[str, list[str]] = {
    "greeting": ["hi", "hello", "hey", "hii", "hiya", "good morning", "good afternoon", "good evening", "namaste"],
    "thanks": ["thanks", "thank you", "thx", "ty", "thanks a lot", "thank you so much", "cheers"],
    "goodbye": ["bye", "goodbye", "bye bye", "see you", "see ya", "cya"],
}
_EXACT_CONTEXT_FREE_ONLY: dict[str, list[str]] = {
    "affirm": ["yes", "yeah", "yep", "sure", "ok", "okay"],
    "deny": ["no", "nope", "nah"],
}


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def detect_pii(text: str) -> list[str]:
    """Return evidence strings for sensitive data found in ``text`` (masked)."""
    found: list[str] = []
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append(f"card number ****{digits[-4:]}")
    if _SSN.search(text):
        found.append("SSN")
    if _AADHAAR.search(text) and re.search(r"aadha?ar", text, re.I):
        found.append("Aadhaar number")
    if _PAN.search(text):
        found.append("PAN")
    m = _SECRET_CUE.search(text)
    if m:
        found.append(f"{m.group(1).lower()} disclosed")
    return found


@dataclass
class RulesOutput:
    flags: list[GuardrailFlag] = field(default_factory=list)
    intent: IntentPrediction | None = None


class RulesStage:
    def __init__(self, taxonomy: Taxonomy):
        self.taxonomy = taxonomy
        self._patterns = [
            (g, [re.compile(p, re.I | re.M) for p in g.patterns]) for g in taxonomy.guardrails if g.patterns
        ]
        enabled = set(taxonomy.intent_names)
        self._exact = {k: v for k, v in _EXACT.items() if k in enabled}
        self._exact_cf = {k: v for k, v in _EXACT_CONTEXT_FREE_ONLY.items() if k in enabled}

    def run(self, message: str, history: list[Turn]) -> RulesOutput:
        out = RulesOutput()
        for g, regexes in self._patterns:
            for rx in regexes:
                m = rx.search(message)
                if m:
                    out.flags.append(
                        GuardrailFlag(name=g.name, score=PATTERN_SCORE, action=g.action, source="rules", evidence=m.group()[:120])
                    )
                    break

        pii = self.taxonomy.guardrail("pii_disclosure")
        if pii:
            evidence = detect_pii(message)
            if evidence:
                out.flags.append(
                    GuardrailFlag(name=pii.name, score=VALIDATED_SCORE, action=pii.action, source="rules", evidence=", ".join(evidence))
                )

        key = re.sub(r"[^\w\s]", "", message.lower()).strip()
        tables = [self._exact] if history else [self._exact, self._exact_cf]
        for table in tables:
            for intent, phrases in table.items():
                if key in phrases:
                    out.intent = IntentPrediction(name=intent, confidence=EXACT_INTENT_SCORE, layer="universal", source="rules")
                    return out
        return out
