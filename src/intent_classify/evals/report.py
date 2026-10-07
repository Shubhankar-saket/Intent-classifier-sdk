"""Self-contained HTML report comparing systems from one results JSON."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

# (key, label, higher_is_better, help)
HEADLINE = [
    ("intent_accuracy", "Intent accuracy", True, "Primary intent correct (incl. acceptable alternatives)"),
    ("intent_macro_f1", "Intent macro-F1", True, "Unweighted mean F1 over intents"),
    ("context_dependent_accuracy", "Context-dependent accuracy", True, "Intent accuracy on replies that need history"),
    ("multilingual_accuracy", "Multilingual accuracy", True, "Intent accuracy on non-English / code-mixed"),
    ("guardrail_recall", "Guardrail recall", True, "Attack / sensitive cases where an expected guardrail fired"),
    ("attack_stop_rate", "Attack stop rate", True, "Attacks that ended in block or escalate"),
    ("multi_turn_attack_recall", "Multi-turn attack recall", True, "Attacks only visible across turns"),
    ("benign_false_block_rate", "Benign false-block rate", False, "Benign messages blocked or escalated"),
    ("benign_false_flag_rate", "Benign false-flag rate", False, "Benign messages with any guardrail action"),
    ("over_refusal_false_block_rate", "Scary-but-benign false-block", False, "'How do I kill a process' style cases"),
    ("oos_handled_rate", "Out-of-scope handled", True, "off_topic fired or abstained with unknown"),
    ("intent_ece", "Calibration error (ECE)", False, "Gap between confidence and accuracy; 0 is perfect"),
]
OPS = [
    ("llm_call_rate", "LLM call rate", False, "pct"),
    ("cost_per_1k_usd", "Est. cost / 1k messages", False, "usd"),
    ("latency_p50_ms", "Latency p50", False, "ms"),
    ("latency_p95_ms", "Latency p95", False, "ms"),
    ("cache_read_share", "Prompt cache hit share", True, "pct"),
    ("cases_with_errors", "Cases with errors", False, "int"),
    ("avg_input_tokens", "Avg uncached input tokens / msg", False, "int"),
    ("avg_cached_tokens", "Avg cached input tokens / msg", False, "int"),
    ("avg_output_tokens", "Avg output tokens / msg", False, "int"),
]

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6a66;--line:#e4e2dc;--card:#fff;--good:#1f7a4d;--good-bg:#e5f3ec;--bad:#a33a2b;--accent:#3b5bdb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ecebe7;--muted:#a19f98;--line:#2e2d2a;--card:#1e1e1c;--good:#6fcf9a;--good-bg:#1d3328;--bad:#f08b7a;--accent:#8ea4ff}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe7;--muted:#a19f98;--line:#2e2d2a;--card:#1e1e1c;--good:#6fcf9a;--good-bg:#1d3328;--bad:#f08b7a;--accent:#8ea4ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:36px 0 10px}
.muted{color:var(--muted)}.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}thead th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
tr:last-child td{border-bottom:none}td.best{color:var(--good);font-weight:600;background:var(--good-bg)}.n{color:var(--muted);font-size:11px}
.help{display:block;color:var(--muted);font-size:12px}code{font-size:12px}details{margin:8px 0}summary{cursor:pointer;font-weight:600}
.fail td{text-align:left;font-size:13px}.tag{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 5px;margin:1px;font-size:11px;color:var(--muted)}
.bad{color:var(--bad)}
"""


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.1f}%"


def _fmt_ops(v: float | None, kind: str) -> str:
    if v is None:
        return "–"
    return {"pct": f"{v * 100:.0f}%", "usd": f"${v:.2f}", "ms": f"{v:.0f} ms", "int": f"{int(v)}"}[kind]


def _best(values: list[float | None], higher: bool) -> set[int]:
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(present) < 2:
        return set()
    target = max(v for _, v in present) if higher else min(v for _, v in present)
    return {i for i, v in present if abs(v - target) < 1e-9}


def render(run: dict[str, Any]) -> str:
    systems = list(run["systems"])
    e = html.escape
    head = "".join(f"<th>{e(s)}</th>" for s in systems)

    rows = []
    for key, label, higher, help_ in HEADLINE:
        vals = [run["systems"][s]["summary"]["headline"][key] for s in systems]
        best = _best([v[0] for v in vals], higher)
        if all(v[0] is None for v in vals):
            continue
        cells = "".join(
            f'<td class="{"best" if i in best else ""}">{"–" if v is None else (f"{v:.3f}" if key == "intent_ece" else _pct(v))}'
            f' <span class="n">n={n}</span></td>'
            for i, (v, n) in enumerate(vals)
        )
        arrow = "↑" if higher else "↓"
        rows.append(f"<tr><td>{e(label)} {arrow}<span class='help'>{e(help_)}</span></td>{cells}</tr>")
    for key, label, higher, kind in OPS:
        vals = [run["systems"][s]["summary"]["ops"].get(key) for s in systems]
        best = _best(vals, higher)
        cells = "".join(f'<td class="{"best" if i in best else ""}">{_fmt_ops(v, kind)}</td>' for i, v in enumerate(vals))
        rows.append(f"<tr><td>{e(label)} {'↑' if higher else '↓'}</td>{cells}</tr>")

    desc = "".join(f"<li><b>{e(s)}</b> — {e(run['systems'][s]['description'])}</li>" for s in systems)

    # Per-intent F1 and per-guardrail recall across systems
    intents = sorted({k for s in systems for k in run["systems"][s]["summary"]["per_intent"]})
    intent_rows = []
    for it in intents:
        vals = [run["systems"][s]["summary"]["per_intent"].get(it, {}).get("f1") for s in systems]
        support = next((run["systems"][s]["summary"]["per_intent"][it]["support"] for s in systems if it in run["systems"][s]["summary"]["per_intent"]), 0)
        best = _best(vals, True)
        cells = "".join(f'<td class="{"best" if i in best else ""}">{_pct(v)}</td>' for i, v in enumerate(vals))
        intent_rows.append(f"<tr><td><code>{e(it)}</code> <span class='n'>n={support}</span></td>{cells}</tr>")
    guards = sorted({k for s in systems for k in run["systems"][s]["summary"]["per_guardrail"]})
    guard_rows = []
    for g in guards:
        vals = [run["systems"][s]["summary"]["per_guardrail"].get(g, {}).get("recall") for s in systems]
        support = next((run["systems"][s]["summary"]["per_guardrail"][g]["support"] for s in systems if g in run["systems"][s]["summary"]["per_guardrail"]), 0)
        best = _best(vals, True)
        cells = "".join(f'<td class="{"best" if i in best else ""}">{_pct(v)}</td>' for i, v in enumerate(vals))
        guard_rows.append(f"<tr><td><code>{e(g)}</code> <span class='n'>n={support}</span></td>{cells}</tr>")

    failures = []
    for s in systems:
        bad = [
            r for r in run["systems"][s]["rows"]
            if r["intent_correct"] is False or r["guardrail_caught"] is False or r["false_block"] or r["errors"]
        ]
        items = []
        for r in bad:
            what = []
            if r["intent_correct"] is False:
                exp = r["expected_intent"] + (f" (or {', '.join(r['acceptable_intents'])})" if r["acceptable_intents"] else "")
                what.append(f"intent: expected <code>{e(exp)}</code>, got <code>{e(r['pred_intent'])}</code> ({r['pred_confidence']:.2f})")
            if r["guardrail_caught"] is False:
                what.append(f"missed guardrail <code>{e('/'.join(r['expected_guardrails']))}</code>; fired: <code>{e(', '.join(r['fired_guardrails']) or 'none')}</code>")
            if r["false_block"]:
                what.append(f"<span class='bad'>false {e(r['action'])}</span> via <code>{e(', '.join(r['fired_guardrails']))}</code>")
            if r["errors"]:
                what.append(f"<span class='bad'>error</span>: {e('; '.join(r['errors']))[:200]}")
            tags = "".join(f"<span class='tag'>{e(t)}</span>" for t in r["tags"])
            ctx = f" <span class='n'>(+{r['history_len']} turns)</span>" if r["history_len"] else ""
            why = f"<br><span class='n'>{e(r['rationale'] or '')}</span>" if r.get("rationale") else ""
            items.append(f"<tr class='fail'><td><code>{e(r['id'])}</code></td><td>{e(r['message'])}{ctx}<br>{tags}</td><td>{'<br>'.join(what)}{why}</td></tr>")
        failures.append(
            f"<details><summary>{e(s)} — {len(bad)} failing cases</summary><div class='wrap'><table>"
            f"<thead><tr><th>id</th><th style='text-align:left'>message</th><th style='text-align:left'>problem</th></tr></thead>"
            f"<tbody>{''.join(items) or '<tr><td>none</td></tr>'}</tbody></table></div></details>"
        )

    meta = run.get("meta", {})
    meta_line = " · ".join(f"{e(str(k))}: <code>{e(str(v))}</code>" for k, v in meta.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Intent Classifier Evals</title><style>{CSS}</style></head><body><main>
<h1>Intent classifier eval — {e(run['taxonomy_version'])}</h1>
<p class="muted">{e(run['created_at'])} · suites: {e(', '.join(run['suites']))}<br>{meta_line}</p>
<h2>Systems</h2><ul>{desc}</ul>
<h2>Headline</h2><p class="muted">Best value per row highlighted. ↑ higher is better, ↓ lower is better. n = cases scored.</p>
<div class="wrap"><table><thead><tr><th>metric</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Per-intent F1</h2><div class="wrap"><table><thead><tr><th>intent</th>{head}</tr></thead><tbody>{''.join(intent_rows)}</tbody></table></div>
<h2>Per-guardrail recall</h2><div class="wrap"><table><thead><tr><th>guardrail</th>{head}</tr></thead><tbody>{''.join(guard_rows)}</tbody></table></div>
<h2>Failures</h2>{''.join(failures)}
</main></body></html>"""


def write_report(results_path: str | Path, out_path: str | Path) -> Path:
    run = json.loads(Path(results_path).read_text(encoding="utf-8"))
    out = Path(out_path)
    out.write_text(render(run), encoding="utf-8")
    return out
