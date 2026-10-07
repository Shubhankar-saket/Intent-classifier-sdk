"""CI regression gate: fail the build when a headline metric crosses its threshold.

gates.yaml:
    system: full
    min: {intent_accuracy: 0.85, guardrail_recall: 0.9}
    max: {benign_false_block_rate: 0.05}
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def check_gates(results_path: str | Path, gates_path: str | Path) -> list[str]:
    run = json.loads(Path(results_path).read_text(encoding="utf-8"))
    gates = yaml.safe_load(Path(gates_path).read_text(encoding="utf-8"))
    system = gates.get("system", "full")
    if system not in run["systems"]:
        return [f"system {system!r} not in results (have: {list(run['systems'])})"]
    summary = run["systems"][system]["summary"]
    values = {k: v[0] for k, v in summary["headline"].items()} | summary["ops"]
    failures = []
    for metric, bound in (gates.get("min") or {}).items():
        v = values.get(metric)
        if v is None or v < bound:
            failures.append(f"{metric} = {v} < min {bound}")
    for metric, bound in (gates.get("max") or {}).items():
        v = values.get(metric)
        if v is None or v > bound:
            failures.append(f"{metric} = {v} > max {bound}")
    return failures
