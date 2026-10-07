# intent-classify

An SDK-level intent classifier. **Guardrails and common conversational intents are
shared by every project**. Each project adds only its own intents in a YAML file.
It understands conversation context, and evals are built in.

```
user message + last N turns
   │
   ├─ L0  normalise          unicode folding, invisible-char stripping, history window
   ├─ L1  guardrails (shared) injection · jailbreak · prompt extraction · harmful · exfiltration
   │                          self-harm · toxicity · PII · off-topic       → allow/flag/escalate/block
   ├─ L2  universal (shared)  greeting · goodbye · thanks · small talk · human handoff · complaint
   │                          feedback · affirm · deny · capabilities · clarification · gibberish
   └─ L3  project intents     from projects/<name>.yaml
   ▼
ClassificationResult { guardrail.action, flags[], intents[] (primary first), language, usage, ... }
```

## How a message is classified (cheapest first)

1. **Rules.** Free and deterministic:
   - Strict regexes catch obvious injection or jailbreak attempts. These are *heuristic*: the LLM can veto them.
   - Validated detectors catch PII, e.g. Luhn-checked card numbers, SSN, PAN, and disclosed passwords or OTPs. These are *deterministic* and always kept.
   - Exact matches handle one-liners like "hi" or "thanks".
2. **Embeddings.** Nearest neighbour against the intent examples. If the match is clear, nothing looks risky, and the message doesn't depend on context, we answer without an LLM.
3. **LLM.** Gets a structured-output prompt generated from the taxonomy (cached per project) plus the last N turns. It resolves context ("yes" → `order.cancel`) and catches multi-turn attacks.
4. **Policy.** The guardrail action is the most severe flag at or above the threshold. Below `abstain_threshold`, the primary intent becomes `unknown`.

Providers are swappable: `anthropic[:model]`, `openai:<model>` (which also covers Azure, vLLM and Ollama via `base_url`), and embedders `hashing` (offline), `openai`, or `st` (local multilingual sentence-transformers).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest
```

```python
from intent_classify import IntentClassifier

clf = IntentClassifier.from_project("projects/shopease.yaml", llm="anthropic", embedder="hashing")
r = clf.classify("yes", history=[
    {"role": "user", "content": "I'd like to cancel order 5512"},
    {"role": "assistant", "content": "It hasn't shipped yet. Shall I cancel it?"},
])
r.guardrail.action      # Action.ALLOW
r.top_intent.name       # "order.cancel"   (secondary: "affirm")
```

Credentials: `ANTHROPIC_API_KEY` (or an `ant auth login` profile) / `OPENAI_API_KEY`.

## Onboarding a new project

1. Copy `projects/shopease.yaml`. Write a one-paragraph `domain` (this drives off-topic detection) and your intents. For each intent, give a precise description, 5–8 examples, and **negative examples** for the intents it's most often confused with.
2. Optionally use `core:` to switch off universal intents you don't want, or change a guardrail's action (e.g. `toxicity: block`).
3. Write `evals/datasets/<project>_*.jsonl` (format in [evals/datasets/README.md](evals/datasets/README.md)). The core guardrail and universal suites run automatically.
4. Run `intent-classify eval` and commit gate thresholds.

## Evals

See **[docs/EVALS.md](docs/EVALS.md)**. One command compares six systems: the zero-shot baseline vs. each layer of the cascade. It reports accuracy, guardrail recall *with* false-block rate, context and multilingual slices, calibration, latency and cost, plus a failure browser.

## Layout

```
src/intent_classify/
  types.py            stable result contract
  taxonomy.py         YAML loading + merging of shared and project layers, versioning
  builtin/            shared guardrails.yaml + universal.yaml
  normalize.py        L0 + history window
  stages/             rules · embedding · llm (prompt + schema)
  pipeline.py         cascade + policy (IntentClassifier, ClassifierConfig)
  backends/           anthropic · openai · embeddings · registry
  evals/              dataset · systems · metrics · runner · report · gate · leakage
  cli.py              intent-classify classify | eval | report | gate | check-leakage
projects/shopease.yaml   synthetic pilot project
evals/datasets/          220 seed cases (core + shopease)
```

## Design decisions worth knowing

- **Primary intent = the task, resolved with context.** Dialogue acts like affirm, deny or complaint come second when a task is present.
- **Conversation content is data, not instructions.** It's escaped and wrapped in tags. An injected instruction becomes evidence for a guardrail, never a command.
- **Provider refusals become a `provider_refusal` flag** (default `block`). When the model's own safety layer declines to even classify a message, that is itself a strong signal. On Claude Opus 5.5 / Sonnet 5.5 the adapter enables server-side refusal fallbacks first.
- **LLM errors degrade instead of crash** (`on_llm_error="fallback"`). The result records `errors`, so callers can choose to fail closed.
- **Default model is `claude-opus-5-5` at `low` effort.** Cheaper options are one flag away (`--llm anthropic:claude-haiku-4-5`). Compare them in the eval report before choosing; don't guess.
