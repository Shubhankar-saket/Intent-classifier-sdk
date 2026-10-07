# Eval datasets

JSONL, one case per line:

```json
{"id": "g-001", "message": "...", "history": [{"role": "user", "content": "..."}],
 "expected_intent": "order.track", "acceptable_intents": [],
 "expected_guardrails": [], "tags": ["single_turn"]}
```

| Field | Meaning |
|---|---|
| `message` | latest user message (the thing being classified) |
| `history` | earlier turns, oldest first; `[]` for single-turn |
| `expected_intent` | primary intent; `null` = don't score intent (e.g. attacks); `"unknown"` = should abstain |
| `acceptable_intents` | other primary intents that also count as correct (genuinely ambiguous cases only) |
| `expected_guardrails` | **any-of**: the case is caught if at least one of these fires. `[]` = benign, nothing should block |
| `tags` | slices for reporting: `single_turn`, `multi_turn`, `context_dependent`, `attack`, `over_refusal`, `oos`, `multilingual`, `typo`, `multi_intent` |

## Suites

| File | What it measures | Project-specific? |
|---|---|---|
| `core_guardrails.jsonl` | Attacks caught + benign-but-scary messages not blocked | no — reusable for every project |
| `core_universal.jsonl` | Shared conversational intents | no |
| `shopease_intents.jsonl` | Project intents, out-of-scope, typos, multi-intent | yes |
| `shopease_multiturn.jsonl` | Context-resolved intents + multi-turn attacks | yes |
| `shopease_multilingual.jsonl` | Hindi / Hinglish / other languages | yes |

### Public suites (`public/`, from `intent-classify fetch-benchmarks`)

| File | Source | License |
|---|---|---|
| `public_deepset_injections.jsonl` | deepset/prompt-injections (test split) | Apache-2.0 |
| `public_jbb_harmful.jsonl` / `public_jbb_benign.jsonl` | JailbreakBench JBB-Behaviors | MIT |
| `public_xstest.jsonl` | XSTest (Paul/XSTest) | CC-BY-4.0 |

These test the shared guardrails, so every project can run them. deepset counts "make the assistant do
another task" as injection, so `off_topic` is also accepted as a catch there.

### Synthetic cases (generated, then human-reviewed)

```
intent-classify generate --project projects/shopease.yaml --guardrails --llm openai:gpt-5.4 --effort low
# review in the review page, or: intent-classify review evals/candidates/<file>.jsonl
intent-classify apply-review evals/candidates/<file>.jsonl <export-dir>   # only when using the review page
intent-classify promote evals/candidates/<file>.jsonl --to evals/datasets/shopease_synthetic.jsonl
```

## Rules

- **No leakage:** eval messages must not copy examples from the taxonomy YAML.
  `intent-classify check-leakage` flags near-duplicates.
- This seed set is hand-written and synthetic. Grow it with real (anonymised) traffic once
  a pilot starts logging, and keep a frozen copy as the regression set.
