"""intent-classify CLI.

classify | eval | report | gate | check-leakage | fetch-benchmarks | generate | review | promote
Reads credentials from ./.env (never overriding variables already set).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

from ._env import load_dotenv
from .backends import create_embedder, create_llm
from .backends.base import LLMBackend
from .pipeline import ClassifierConfig, IntentClassifier
from .taxonomy import Taxonomy


def _expand(patterns: list[str]) -> list[str]:
    paths = sorted({p for pat in patterns for p in (glob.glob(pat) or [pat])})
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        sys.exit(f"dataset(s) not found: {missing}")
    return paths


def _llm_kwargs(a: argparse.Namespace) -> dict[str, Any]:
    kw: dict[str, Any] = {}
    provider = a.llm.split(":", 1)[0]
    if getattr(a, "effort", None):
        kw["reasoning_effort" if provider == "openai" else "effort"] = a.effort
    if getattr(a, "llm_price", None):
        if provider != "openai":
            sys.exit("--llm-price is only used for openai (anthropic prices are built in)")
        p_in, p_out = (float(x) for x in a.llm_price.split(","))
        kw["price_per_mtok"] = (p_in, p_out)
    return kw


def _make_llm(a: argparse.Namespace) -> LLMBackend | None:
    return None if a.llm == "none" else create_llm(a.llm, **_llm_kwargs(a))


def _add_llm_args(p: argparse.ArgumentParser, default: str = "anthropic") -> None:
    p.add_argument("--llm", default=default, help="'anthropic[:model]', 'openai:<model>' or 'none'")
    p.add_argument("--effort", help="anthropic effort / openai reasoning_effort, e.g. low")
    p.add_argument("--llm-price", help="openai only: 'input,output' USD per 1M tokens, for cost estimates")


def _cmd_classify(a: argparse.Namespace) -> None:
    clf = IntentClassifier.from_project(a.project, llm=_make_llm(a), embedder=a.embedder)
    history = json.loads(Path(a.history).read_text(encoding="utf-8")) if a.history else None
    print(clf.classify(a.message, history).model_dump_json(indent=2))


def _cmd_eval(a: argparse.Namespace) -> None:
    from .evals import load_cases, run_eval, write_report

    taxonomy = Taxonomy.from_yaml(a.project)
    cases = load_cases(_expand(a.datasets))
    if a.tags:
        wanted = set(a.tags.split(","))
        cases = [c for c in cases if wanted & set(c.tags)]
    if a.limit:
        cases = cases[: a.limit]
    llm_factory = None if a.llm == "none" else (lambda: create_llm(a.llm, **_llm_kwargs(a)))
    run = run_eval(
        taxonomy, cases, a.systems.split(","), llm_factory, create_embedder(a.embedder), ClassifierConfig(),
        a.workers, a.out,
        meta={"llm": a.llm, "effort": a.effort, "embedder": a.embedder, "cases": len(cases), "label": a.label},
    )
    report = Path(run["path"]).with_suffix(".html")
    write_report(run["path"], report)
    print(f"report  -> {report}")


def _cmd_report(a: argparse.Namespace) -> None:
    from .evals import write_report

    print(write_report(a.results, a.out or str(Path(a.results).with_suffix(".html"))))


def _cmd_gate(a: argparse.Namespace) -> None:
    from .evals import check_gates

    failures = check_gates(a.results, a.gates)
    if failures:
        print("GATE FAILED\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("gate passed")


def _cmd_leakage(a: argparse.Namespace) -> None:
    from .evals import find_leaks, load_cases

    leaks = find_leaks(load_cases(_expand(a.datasets)), Taxonomy.from_yaml(a.project), a.threshold)
    for cid, msg, ex, sim in leaks:
        print(f"{sim:.2f}  {cid}: {msg!r}\n      ~ taxonomy example: {ex!r}")
    print(f"{len(leaks)} near-duplicates at >= {a.threshold}")
    sys.exit(1 if leaks and a.strict else 0)


def _cmd_fetch(a: argparse.Namespace) -> None:
    from .evals.public import fetch_all

    fetch_all(a.out)


def _cmd_generate(a: argparse.Namespace) -> None:
    from datetime import datetime

    from .evals.dataset import load_cases
    from .evals.generate import generate_candidates, write_jsonl

    taxonomy = Taxonomy.from_yaml(a.project)
    existing = [c.message for c in load_cases(_expand(a.existing))] if a.existing else []
    llm = _make_llm(a)
    if llm is None:
        sys.exit("generate needs an LLM")
    rows = generate_candidates(
        taxonomy, llm, per_label=a.per_label, include_guardrails=a.guardrails,
        intents=a.intents.split(",") if a.intents else None, existing_messages=existing,
    )
    out = Path(a.out or f"evals/candidates/{taxonomy.project}_{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    write_jsonl(out, rows)
    print(f"{len(rows)} candidates (after de-duplication) -> {out}\nnext: intent-classify review {out}")


def _cmd_review(a: argparse.Namespace) -> None:
    from .evals.generate import read_jsonl, write_jsonl

    rows = read_jsonl(a.candidates)
    pending = [r for r in rows if r.get("review") == "pending"]
    print(f"{len(pending)} pending of {len(rows)}. Keys: [a]pprove [r]eject [i]ntent relabel "
          f"[g]uardrails relabel [s]kip [q]uit\n")
    for n, r in enumerate(pending, 1):
        print(f"--- {n}/{len(pending)}  {r['id']}  tags={r['tags']}")
        for t in r.get("history", []):
            print(f"    {t['role']:>9}: {t['content']}")
        print(f"  >>  user: {r['message']}")
        print(f"      expected_intent={r['expected_intent']}  expected_guardrails={r['expected_guardrails']}")
        while True:
            k = input("  decision> ").strip().lower()
            if k == "a":
                r["review"] = "approved"
            elif k == "r":
                r["review"] = "rejected"
                r["review_note"] = input("  why (optional)> ").strip()
            elif k == "i":
                r["expected_intent"] = input("  intent (blank = none)> ").strip() or None
                r["acceptable_intents"] = [x for x in input("  acceptable, comma-sep> ").split(",") if x.strip()]
                r["review"] = "approved"
            elif k == "g":
                r["expected_guardrails"] = [x.strip() for x in input("  guardrails, comma-sep> ").split(",") if x.strip()]
                r["review"] = "approved"
            elif k in ("s", "q"):
                pass
            else:
                continue
            break
        write_jsonl(a.candidates, rows)  # save after every decision
        if k == "q":
            break
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("review", "?")] = counts.get(r.get("review", "?"), 0) + 1
    print(f"\n{counts}\nnext: intent-classify promote {a.candidates} --to <dataset.jsonl>")


def _cmd_apply(a: argparse.Namespace) -> None:
    from .evals.generate import apply_review_export

    print(f"applied review-page decisions: {apply_review_export(a.candidates, a.export_dir)}")


def _cmd_promote(a: argparse.Namespace) -> None:
    from .evals.generate import promote

    print(f"promoted {promote(a.candidates, a.to)} approved cases -> {a.to}")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="intent-classify")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="classify one message")
    c.add_argument("message")
    c.add_argument("--project", required=True)
    c.add_argument("--history", help="JSON file: [{role, content}, ...]")
    c.add_argument("--embedder", default="hashing")
    _add_llm_args(c)
    c.set_defaults(fn=_cmd_classify)

    e = sub.add_parser("eval", help="run eval suites and write JSON + HTML report")
    e.add_argument("--project", required=True)
    e.add_argument("--datasets", nargs="+", required=True, help="JSONL files or globs")
    e.add_argument("--systems", default="zero_shot,rules,embeddings,full_no_history,full")
    e.add_argument("--embedder", default="hashing")
    e.add_argument("--tags", help="only cases with any of these comma-separated tags")
    e.add_argument("--limit", type=int)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--out", default="evals/results")
    e.add_argument("--label", help="free-text label stored with the run")
    _add_llm_args(e)
    e.set_defaults(fn=_cmd_eval)

    r = sub.add_parser("report", help="render HTML from a results JSON")
    r.add_argument("results")
    r.add_argument("--out")
    r.set_defaults(fn=_cmd_report)

    g = sub.add_parser("gate", help="CI regression gate")
    g.add_argument("results")
    g.add_argument("--gates", default="evals/gates.yaml")
    g.set_defaults(fn=_cmd_gate)

    lk = sub.add_parser("check-leakage", help="find eval cases copied from taxonomy examples")
    lk.add_argument("--project", required=True)
    lk.add_argument("--datasets", nargs="+", required=True)
    lk.add_argument("--threshold", type=float, default=0.85)
    lk.add_argument("--strict", action="store_true")
    lk.set_defaults(fn=_cmd_leakage)

    fb = sub.add_parser("fetch-benchmarks", help="download public guardrail benchmarks as eval suites")
    fb.add_argument("--out", default="evals/datasets/public")
    fb.set_defaults(fn=_cmd_fetch)

    gen = sub.add_parser("generate", help="LLM-generate candidate eval cases for human review")
    gen.add_argument("--project", required=True)
    gen.add_argument("--per-label", type=int, default=12)
    gen.add_argument("--intents", help="comma-separated intents (default: all project intents)")
    gen.add_argument("--guardrails", action="store_true", help="also generate attack cases per guardrail")
    gen.add_argument("--existing", nargs="*", default=["evals/datasets/*.jsonl"], help="datasets to de-duplicate against")
    gen.add_argument("--out")
    _add_llm_args(gen)
    gen.set_defaults(fn=_cmd_generate)

    rv = sub.add_parser("review", help="interactively approve / reject / relabel candidates")
    rv.add_argument("candidates")
    rv.set_defaults(fn=_cmd_review)

    ap = sub.add_parser("apply-review", help="apply decisions exported from the review page")
    ap.add_argument("candidates")
    ap.add_argument("export_dir", help="directory of exported JSON docs, one per case")
    ap.set_defaults(fn=_cmd_apply)

    pr = sub.add_parser("promote", help="append approved candidates to a dataset")
    pr.add_argument("candidates")
    pr.add_argument("--to", required=True)
    pr.set_defaults(fn=_cmd_promote)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
