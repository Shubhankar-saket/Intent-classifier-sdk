from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..backends.base import EmbeddingBackend, LLMBackend
from ..pipeline import ClassifierConfig
from ..taxonomy import Taxonomy
from .dataset import EvalCase, validate_against
from .metrics import score_case, summarize
from .systems import SYSTEM_DESCRIPTIONS, build_system


def run_eval(
    taxonomy: Taxonomy,
    cases: list[EvalCase],
    systems: list[str],
    llm_factory: Callable[[], LLMBackend] | None,
    embedder: EmbeddingBackend | None,
    config: ClassifierConfig | None = None,
    max_workers: int = 8,
    out_dir: str | Path | None = None,
    meta: dict[str, Any] | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    problems = validate_against(cases, taxonomy)
    if problems:
        raise ValueError("dataset does not match taxonomy:\n  " + "\n  ".join(problems))
    config = config or ClassifierConfig()
    run: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "taxonomy_version": taxonomy.version,
        "suites": sorted({c.suite for c in cases}),
        "config": config.model_dump(mode="json"),
        "meta": meta or {},
        "systems": {},
    }
    for name in systems:
        clf = build_system(name, taxonomy, llm_factory, embedder, config)
        progress(f"[{name}] {len(cases)} cases ...")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda c: clf.classify(c.message, c.history), cases))
        rows = [score_case(c, r, config.guardrail_threshold) for c, r in zip(cases, results)]
        summary = summarize(rows)
        run["systems"][name] = {"description": SYSTEM_DESCRIPTIONS[name], "summary": summary, "rows": rows}
        h = summary["headline"]
        progress(
            f"[{name}] done in {time.perf_counter() - t0:.1f}s  intent_acc={_fmt(h['intent_accuracy'][0])}  "
            f"guardrail_recall={_fmt(h['guardrail_recall'][0])}  false_block={_fmt(h['benign_false_block_rate'][0])}"
            + (f"  errors={summary['ops']['cases_with_errors']}" if summary["ops"]["cases_with_errors"] else "")
        )
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = out / f"{taxonomy.project}-{stamp}.json"
        path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
        run["path"] = str(path)
        progress(f"results -> {path}")
    return run


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"
