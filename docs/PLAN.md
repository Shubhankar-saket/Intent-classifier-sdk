# Project plan: organisation-level intent classifier

**Owner:** shubhankar@growth-loop.io · **Last updated:** 2026-10-03

## 1. Goal

One intent-classification library for every assistant in the organisation:

- **Shared layers, written once:** safety guardrails and common conversational intents are the same everywhere.
- **Project layer, written by each team as YAML:** that project's own intents.
- **Conversation-aware:** the classifier reads the last N turns, so short replies and multi-turn attacks are handled.
- **Provable:** every project gets an eval report and a CI regression gate for free.

**Success:** a new project onboards with only a YAML file and an eval set. The library beats that
project's zero-shot prompt on intent accuracy *and* attack recall, without raising the
false-block rate, at a cost and latency the project accepts.

## 2. Decisions

| Decision | Choice | Why |
|---|---|---|
| Language | Python | ML/eval ecosystem |
| LLM providers | Swappable: Anthropic, OpenAI-compatible (Azure, vLLM, Ollama) | No vendor lock-in; pick per project from eval numbers |
| Latency | Should be reasonable; ultra-fast not required | The cascade sends most easy messages to cheap stages |
| Languages | English first; others per project | Multilingual embedder + per-project language eval slices |
| Context | **Last N turns from day one** | Long-term library; "yes" / "the second one" and multi-turn jailbreaks need it |
| Pilot | **Synthetic project (ShopEase)** | Real project doesn't store logs yet |
| Primary intent | The *task*, resolved with context; dialogue acts (affirm, complaint, ...) come second | Downstream routing needs the task |
| Provider safety refusal | Becomes a `provider_refusal` guardrail flag (default `block`) | A model refusing to classify is itself a strong signal |
| LLM failure | Degrade to embeddings/rules and record `errors` | Callers choose fail-open vs fail-closed |

## 3. Architecture

```
message + last N turns
  L0 normalise (unicode folding, invisible chars, history window)
  L1 guardrails      shared  -> allow / flag / escalate / block
  L2 universal       shared  greeting, handoff, affirm/deny, complaint, ...
  L3 project intents per-project YAML
Cascade: rules (free) -> embeddings (fast path when clear & safe) -> LLM (structured output, cached prompt)
```

Code: `src/intent_classify/`. Details: [README](../README.md).

## 4. Phases

### Phase 0: Foundations ✅
- [x] Shared guardrail taxonomy (9) and universal intents (12)
- [x] Stable output contract (`ClassificationResult`)
- [x] Project YAML format with overrides and versioning (taxonomy hash in every result)

### Phase 1: Core library + eval harness ✅
- [x] Normaliser, history window, rules stage (regex + validated PII), embedding stage, LLM stage, policy
- [x] Provider adapters: Anthropic (cached system prompt, refusal fallbacks), OpenAI (strict JSON schema, reasoning effort)
- [x] Embedders: hashing (offline), OpenAI, sentence-transformers (multilingual)
- [x] Eval harness: 6 systems (zero-shot baseline + ablations), metrics, HTML report, CI gate, leakage check
- [x] Synthetic pilot `projects/shopease.yaml` + 220 hand-written seed cases (guardrails, universal, project, multi-turn, multilingual)
- [x] 41 unit tests, all passing without credentials

### Phase 2: Trustworthy numbers 🟡 in progress
Make the evidence credible and the classifier better.

| # | Item | Status |
|---|---|---|
| 2.1 | Live baseline run (OpenAI) + set CI gate thresholds from it | ✅ see §5 |
| 2.2 | Public benchmark suites: deepset prompt-injections, JailbreakBench (harmful + benign twins), XSTest | ✅ 766 cases fetched (`intent-classify fetch-benchmarks`) |
| 2.3 | LLM-generated synthetic cases → **human review** → promotion | ✅ tooling (`generate` / review page or `review` CLI / `apply-review` / `promote`); 🟡 243 candidates awaiting review |
| 2.4 | Model sweep: same suites across models / effort levels | ✅ first sweep in §5.2; add prices for cost |
| 2.4b | Fix task-hijack / role-play miss and `off_topic` under-firing found by deepset (§5.3) | ⬜ next |
| 2.5 | Dev/test split: tune thresholds on dev, report on test | ⬜ |
| 2.6 | Semantic embeddings (OpenAI or local multilingual) replacing hashing; add fast-path rate + fast-path accuracy metrics | ⬜ |
| 2.7 | Threshold sweep + calibration file per project (accept, abstain, guardrail thresholds); fix ECE if bad | ⬜ |
| 2.8 | Guardrail hardening: decode obfuscation (leetspeak, base64, homoglyphs), indirect injection in pasted content/tool outputs, conversation-level risk memory | ⬜ |
| 2.9 | Robustness slices auto-generated (typos, casing, code-mixing) with accuracy-drop reporting | ⬜ |
| 2.10 | Async API, timeouts, cache for identical messages | ⬜ |

**Exit criteria:** ≥ 500 reviewed project cases + public suites; a dev/test split; headline numbers on test with
confidence intervals; gates committed; a model chosen from the sweep with cost per 1k messages.

### Phase 3: Real pilot ⬜
- Onboard the real project from YAML. Turn on **anonymised logging** (PII masked by the rules stage).
- Sample and label ~500 real messages and freeze them as the regression set. Compare synthetic vs real accuracy.
- Shadow mode: run alongside the existing classifier and diff the decisions before switching over.

### Phase 4: Library and operations ⬜
- Package and version (`pip install intent-classify`), with a semver policy on the output contract and the taxonomy.
- Observability: structured logs, dashboards (guardrail rate, unknown rate, fast-path rate, cost), drift alerts.
- Active learning: low-confidence and `unknown` messages go to a labelling queue and feed the next eval set.
- Onboarding guide + project template; second project onboarded by another team without our help.

## 5. Current results

All runs 2026-10-03, OpenAI, hashing embedder. Full HTML reports are in `evals/results/`. Each run stores every prediction, so runs can be diffed.

### 5.0 Golden set (460 cases = 220 seed + 240 reviewed synthetic), 2026-10-05, `gpt-5.4-mini` low — current headline

| Metric | zero_shot | full_no_history | **full** |
|---|---|---|---|
| Intent accuracy (n=319) | 87.5% | 90.9% | **97.2%** |
| Context-dependent accuracy (n=46) | 47.8% | 47.8% | **93.5%** |
| Guardrail recall (n=134) | 80.6% | 97.0% | **98.5%** |
| Attack stop rate (n=116) | 90.5% | 97.4% | **99.1%** |
| Multi-turn attack recall (n=28) | 75.0% | 92.9% | **100%** |
| Benign false-block (n=315) | 0.3% | 0% | **0%** |
| Calibration error (ECE) | 0.125 | 0.048 | **0.012** |
| Latency p50 / p95 | 1.0 s / 1.6 s | 1.4 s / 2.5 s | 1.5 s / 2.4 s |

CI gate passes. 21 remaining failures: 10 off-topic answered `unknown` instead of flagged (`off_topic` recall 9%),
4 short replies ranked as dialogue act first, 4 questions just outside the label set, 3 one-offs.
Pitch page for the team: https://claude.ai/artifact/V4td6ZiZNDywZiJK7gJApe

### 5.1 Seed set (220 cases), `gpt-5.4-mini`, reasoning effort low

| Metric | zero_shot (status quo) | full_no_history | **full** |
|---|---|---|---|
| Intent accuracy | 83.1% | 87.3% | **96.4%** |
| Intent macro-F1 | 83.6% | 89.5% | **96.1%** |
| Context-dependent accuracy (n=22) | 22.7% | 22.7% | **86.4%** |
| Guardrail recall | 80.9% | 95.7% | **100%** |
| Attack stop rate | 85.4% | 92.7% | **97.6%** |
| Multi-turn attack recall (n=7) | 57.1% | 71.4% | **100%** |
| Benign false-block rate | 1.2% | 0% | **0%** |
| Scary-but-benign false-block (n=24) | 8.3% | 0% | **0%** |
| Calibration error (ECE) | 0.169 | 0.088 | **0.018** |
| Latency p50 | 1.1 s | 1.6 s | 1.7 s |
| Prompt-cache hit share | – | 89% | 91% |

**Takeaways.** Conversation history is the biggest single gain: context-dependent accuracy goes from 23% to 86%,
and multi-turn attack recall from 57% to 100%. The structured, taxonomy-driven prompt alone (`full_no_history`)
already beats the zero-shot baseline on every safety metric.

### 5.2 Model sweep (`full`, same 220 cases)

| Model | Intent acc | Context-dep | Guardrail recall | Stop rate | False-block | ECE | p50 |
|---|---|---|---|---|---|---|---|
| gpt-4.1-mini | 95.8% | 95.5% | 89.4% | 97.6% | 0.6% | 0.018 | 1.2 s |
| gpt-5.4-nano (low) | 95.2% | 95.5% | 95.7% | 97.6% | 0.6% | 0.183 | 1.2 s |
| gpt-5.4-mini (low) | 96.4% | 86.4% | 100% | 97.6% | 0% | 0.018 | 1.7 s |
| gpt-5.4 (low) | **97.0%** | 95.5% | **100%** | 97.6% | **0%** | 0.042 | 2.2 s |

Differences of a few points are within noise at this size. Prices weren't recorded, so pass
`--llm-price in,out` to get cost per 1k messages before choosing a model.

### 5.3 Public benchmarks (766 cases), `gpt-5.4-mini`

| Source | System | Attack recall | Attack stop rate | Benign false-block |
|---|---|---|---|---|
| XSTest (200 unsafe / 250 safe) | zero_shot | 88.5% | 75.5% | 10.0% |
| | **full** | **94.5%** | **81.0%** | **4.8%** |
| JailbreakBench (100 harmful / 100 benign twins) | zero_shot | **96.0%** | **90.0%** | 38.0% |
| | **full** | 84.0% | 76.0% | **21.0%** |
| deepset injections (60 / 56) | zero_shot | **68.3%** | 51.7% | 0% |
| | **full** | 61.7% | 51.7% | 0% |

**What this shows.**
- On XSTest the library wins on both axes.
- On JailbreakBench it trades recall for half the over-blocking. JBB's "benign twins" are borderline on purpose.
- **deepset exposes a real weakness:** role-play and task-hijack prompts ("act as a storyteller", "generate SQL code"),
  including German ones, fire *no* guardrail. They should at least trigger `off_topic`. On the seed set too,
  `off_topic` rarely fires (the classifier abstains with `unknown` instead).

**Fix queued (Phase 2.8):** sharpen the `off_topic` and `prompt_injection` definitions for task hijacking and role
assignment, add multilingual examples, and re-run these suites. The JBB recall gap is a threshold/policy choice:
decide it per project with the threshold sweep (2.7).

### 5.4 Synthetic candidates

243 candidates generated with `gpt-5.4` (13 intents × 12, 7 guardrails × ~12, after de-duplication), covering
paraphrase, typo, code-mixed, context-dependent, indirect, hard-negative, obfuscated, multi-turn and embedded
styles. **Awaiting human review** in the review page; nothing enters the golden set until approved.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Synthetic data overstates accuracy | Human review, leakage check, real-traffic regression set in Phase 3 |
| Small eval sets → noisy comparisons | Grow to ≥ 500 per project; report n and confidence intervals |
| Over-blocking legitimate users | False-block rate always reported next to recall; gated in CI |
| Provider behaviour changes silently | Model pinned per project; eval reruns in CI on every change |
| Cost at scale | Cascade fast path, prompt caching, model sweep |
| Prompt injection through conversation content | Content escaped and treated as data; injected instructions become evidence for a guardrail |
