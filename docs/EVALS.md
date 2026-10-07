# Evals: how we show this works

## The claim we need to prove

> A shared, conversation-aware classifier beats what each team ships today (a single
> zero-shot prompt) on accuracy **and** safety, at an acceptable cost, and a new project gets
> that from a YAML file.

Every number in the report exists to support or refute part of that sentence.

## Systems compared (one run, same data)

| System | What it is | Why it's in the report |
|---|---|---|
| `zero_shot` | One LLM prompt listing label names; latest message only | **Status quo baseline** |
| `rules` | Regex + validated PII detectors + exact matches | Floor; shows what's free |
| `embeddings` | `rules` + nearest-neighbour on examples | What you get with no LLM |
| `full_no_history` | Full cascade, history switched off | Isolates the value of **context** |
| `full_llm_only` | Rules + LLM on every message (no fast path) | Isolates the cost/quality effect of the **fast path** |
| `full` | Full cascade with history | The product |

`full` vs `zero_shot` is the headline comparison. `full` vs `full_no_history` is the argument for
building conversation-awareness in from day one.

## Metrics

| Metric | Definition | Direction |
|---|---|---|
| Intent accuracy | Primary intent ∈ {expected} ∪ acceptable, over cases with an expected intent | ↑ |
| Intent macro-F1 | Mean F1 over intents (an acceptable prediction counts as the expected label) | ↑ |
| Context-dependent accuracy | Intent accuracy on `context_dependent` cases ("yes", "the second one") | ↑ |
| Multilingual accuracy | Intent accuracy on `multilingual` cases | ↑ |
| Guardrail recall | Cases with expected guardrails (excluding `oos`) where **any** listed guardrail fired at ≥ threshold | ↑ |
| Attack stop rate | Cases expecting a stopping guardrail (injection, jailbreak, extraction, harmful, exfiltration, self-harm) that ended in `block`/`escalate` | ↑ |
| Multi-turn attack recall | Guardrail recall on `multi_turn` + `attack` cases | ↑ |
| Benign false-block rate | Benign cases (no expected guardrails) that ended in `block`/`escalate` | ↓ |
| Benign false-flag rate | Benign cases with any non-`allow` action (includes PII flags) | ↓ |
| Scary-but-benign false-block | False-block rate on `over_refusal` cases ("how do I kill a process") | ↓ |
| OOS handled | `oos` cases where `off_topic` fired or the classifier abstained | ↑ |
| Intent ECE | Expected calibration error of primary-intent confidence (10 bins) | ↓ |
| LLM call rate / cost per 1k / p50 / p95 / cache hit share | Operational | ↓ / ↓ / ↓ / ↓ / ↑ |

**Guardrail recall and false-block rate are always reported together.** Either one alone is
meaningless: blocking everything gets 100% recall. Likewise **OOS handled must be read with intent
accuracy**: a system that abstains on everything (e.g. `rules`) scores 100% on OOS.

## Datasets

See [evals/datasets/README.md](../evals/datasets/README.md). The current seed set is 220
hand-written synthetic cases: 64 guardrail, 38 universal, 70 project, 30 multi-turn,
18 multilingual.

Known limits of the seed set, so nobody over-reads the numbers:
- It's small: a 1-case change moves some slices by 3–10 points. Treat differences under
  ~5 points as noise until the set grows.
- It's synthetic and written by the same people who wrote the taxonomy. Real traffic will be messier.
- The out-of-scope set is small (≈ 11 cases).

## Growth plan

1. **Public benchmarks for core guardrails:** add loaders for deepset/prompt-injections,
   JailbreakBench, WildJailbreak (attacks), and XSTest / OR-Bench (over-refusal). These suites are
   project-independent, so every project inherits them.
2. **LLM-generated synthetic expansion** per project: paraphrases, typos, code-mixing and
   multi-turn variants of each intent. Every generated case is reviewed by a human before entering the golden set.
3. **Real traffic** once a pilot logs (anonymised): sample, label, freeze a regression set.
4. **Calibration:** use the ECE + threshold sweep to set `abstain_threshold` and
   `guardrail_threshold` per project.

## Running

```bash
# offline (no API key): rules + embeddings only
intent-classify eval --project projects/shopease.yaml --datasets "evals/datasets/*.jsonl" \
    --systems rules,embeddings --llm none

# full comparison (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
intent-classify eval --project projects/shopease.yaml --datasets "evals/datasets/*.jsonl" \
    --systems zero_shot,rules,embeddings,full_no_history,full_llm_only,full --llm anthropic

# swap provider/model; same report
intent-classify eval ... --llm anthropic:claude-sonnet-5-5
intent-classify eval ... --llm openai:<model>

# CI
intent-classify gate evals/results/<run>.json --gates evals/gates.yaml
intent-classify check-leakage --project projects/shopease.yaml --datasets "evals/datasets/*.jsonl"
```

Each run writes `evals/results/<project>-<timestamp>.json` (every prediction, for diffing) and an
HTML report beside it.
