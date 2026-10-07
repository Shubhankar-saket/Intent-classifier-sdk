"""Fetch public safety benchmarks and convert them to EvalCase JSONL.

These suites are project-independent: they test the shared guardrail layer, so every project
inherits them. Rows come from the Hugging Face datasets-server API (no extra dependencies).

| Suite | Source | License | Tests |
|---|---|---|---|
| public_deepset_injections | deepset/prompt-injections (test) | Apache-2.0 | injection / task hijacking vs benign |
| public_jbb_harmful | JailbreakBench/JBB-Behaviors (harmful) | MIT | direct harmful requests |
| public_jbb_benign | JailbreakBench/JBB-Behaviors (benign) | MIT | topic-matched benign twins (over-blocking) |
| public_xstest | Paul/XSTest | CC-BY-4.0 | scary-but-safe prompts vs unsafe contrasts |
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

API = "https://datasets-server.huggingface.co/rows"
PAGE = 100

STOPPING = ["harmful_request", "jailbreak", "prompt_injection", "data_exfiltration", "self_harm", "system_prompt_extraction"]
UNSAFE_ANY = [*STOPPING, "toxicity"]
# deepset labels "make the assistant do some other task" as injection. For a domain
# assistant, flagging that as off_topic is a correct outcome too.
DEEPSET_ATTACK_ANY = ["prompt_injection", "jailbreak", "system_prompt_extraction", "off_topic"]
JBB_CATEGORY_GUARDRAILS = {
    "Harassment/Discrimination": ["toxicity", "harmful_request"],
    "Privacy": ["data_exfiltration", "harmful_request"],
}


def _rows(dataset: str, config: str, split: str) -> Iterator[dict[str, Any]]:
    offset = 0
    while True:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": PAGE})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{API}?{q}", timeout=60) as resp:
                    data = json.load(resp)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        rows = data.get("rows", [])
        yield from (r["row"] for r in rows)
        offset += len(rows)
        if not rows or offset >= data.get("num_rows_total", 0):
            return


def _case(cid: str, message: str, guardrails: list[str], tags: list[str]) -> dict[str, Any]:
    return {
        "id": cid,
        "message": message.strip(),
        "history": [],
        "expected_intent": None,
        "expected_guardrails": guardrails,
        "tags": ["public", "single_turn", *tags],
    }


def deepset() -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(_rows("deepset/prompt-injections", "default", "test")):
        if r["label"] == 1:
            out.append(_case(f"pub-ds-{i:03d}", r["text"], DEEPSET_ATTACK_ANY, ["attack", "deepset"]))
        else:
            out.append(_case(f"pub-ds-{i:03d}", r["text"], [], ["benign_general", "deepset"]))
    return out


def jbb_harmful() -> list[dict[str, Any]]:
    return [
        _case(f"pub-jbbh-{r['Index']:03d}", r["Goal"], JBB_CATEGORY_GUARDRAILS.get(r["Category"], ["harmful_request"]),
              ["attack", "jbb", r["Category"].lower().replace("/", "_").replace(" ", "_")])
        for r in _rows("JailbreakBench/JBB-Behaviors", "behaviors", "harmful")
    ]


def jbb_benign() -> list[dict[str, Any]]:
    return [
        _case(f"pub-jbbb-{r['Index']:03d}", r["Goal"], [], ["over_refusal", "jbb"])
        for r in _rows("JailbreakBench/JBB-Behaviors", "behaviors", "benign")
    ]


def xstest() -> list[dict[str, Any]]:
    out = []
    for r in _rows("Paul/XSTest", "default", "train"):
        if r["label"] == "safe":
            out.append(_case(f"pub-xs-{r['id']:03d}", r["prompt"], [], ["over_refusal", "xstest", r["type"]]))
        else:
            out.append(_case(f"pub-xs-{r['id']:03d}", r["prompt"], UNSAFE_ANY, ["attack", "xstest", r["type"]]))
    return out


SUITES: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "public_deepset_injections": deepset,
    "public_jbb_harmful": jbb_harmful,
    "public_jbb_benign": jbb_benign,
    "public_xstest": xstest,
}


def fetch_all(out_dir: str | Path, progress: Callable[[str], None] = print) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, fn in SUITES.items():
        cases = fn()
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        counts[name] = len(cases)
        progress(f"{name}: {len(cases)} cases -> {path}")
    return counts
