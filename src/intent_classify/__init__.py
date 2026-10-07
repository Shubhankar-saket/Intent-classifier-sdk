"""Organisation-level, conversation-aware intent classification.

    from intent_classify import IntentClassifier
    clf = IntentClassifier.from_project("projects/shopease.yaml", llm="anthropic", embedder="hashing")
    result = clf.classify("yes please", history=[
        {"role": "user", "content": "I want to cancel order 5512"},
        {"role": "assistant", "content": "Order 5512 hasn't shipped yet. Shall I cancel it?"},
    ])
    result.guardrail.action, result.top_intent.name   # -> allow, order.cancel
"""

from .pipeline import ClassifierConfig, IntentClassifier
from .taxonomy import GuardrailDef, IntentDef, ProjectSpec, Taxonomy
from .types import (
    UNKNOWN,
    Action,
    ClassificationResult,
    GuardrailFlag,
    GuardrailResult,
    IntentPrediction,
    Turn,
    Usage,
)

__all__ = [
    "UNKNOWN",
    "Action",
    "ClassificationResult",
    "ClassifierConfig",
    "GuardrailDef",
    "GuardrailFlag",
    "GuardrailResult",
    "IntentClassifier",
    "IntentDef",
    "IntentPrediction",
    "ProjectSpec",
    "Taxonomy",
    "Turn",
    "Usage",
]
