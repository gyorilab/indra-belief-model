# indra-belief-model

LLM-based evidence quality scoring for [INDRA](https://github.com/gyorilab/indra) biomedical text-mining extractions.

## What this does

INDRA's NLP readers extract structured biological relationships from scientific papers. For example, from the sentence:

> *"The kinase-dead RSK1 mutant, however, was unable to phosphorylate YB-1 at S102."*

a reader might extract: **RPS6KA1 [Phosphorylation] YBX1 @S102**

This scorer judges whether such extractions are correct. Here, the extraction is *incorrect* — the sentence describes a negative result (the mutant was **unable** to phosphorylate).

### Input

Native INDRA Statement + Evidence objects, resolved through `ScoringRecord`:

| Field | Example | Source |
|-------|---------|--------|
| **Claim** | `RPS6KA1 [Phosphorylation] YBX1 @S102` | Statement type + agents + modification site |
| **Evidence** | *"The kinase-dead RSK1 mutant..."* | Source sentence from paper |
| **Entity aliases** | RSK1, YB-1, p90Rsk... | [Gilda](https://github.com/gyorilab/gilda) grounding via `GroundedEntity.resolve()` |

### Output

```json
{"verdict": "correct", "confidence": "high"}
```

Mapped to a continuous score: `{correct+high: 0.95, correct+medium: 0.80, ..., incorrect+high: 0.05}`.

## How it works

Model: gemma-4-26b (Ollama remote or local MLX 8-bit).

### Production scoring architecture

The CLI default is the monolithic scorer: a deterministic LLM call per
`(Statement, Evidence)` pair with type-adaptive contrastive examples (a second
call fires only for `[Complex]` claims — see Tier 2). The decomposed four-probe
scorer remains available for ablations with `--arch decomposed`.

### Two-tier monolithic path

**Tier 1: Deterministic grounding** (no LLM call)

| Status | Action | Example |
|--------|--------|---------|
| **MISMATCH** | Auto-reject | "RhoA" → RHOA != ARHGEF25 |
| **PSEUDOGENE + AMBIGUOUS** | Auto-reject | "DVL" → DVL1P1 (pseudogene) |
| **AMBIGUOUS** | Pass to Tier 2 | "9G8" → SRSF7/SLU7 (tied scores) |
| **MATCH** | Pass to Tier 2 | "FAK" → PTK2 (confirmed alias) |

**Tier 2: LLM text comprehension**

- Commit-first "disconfirm" system prompt: the model commits a defeating
  objection to a structured field before it rationalizes a verdict (negation,
  hedging, family/member equivalence, relation-direction reversal).
- Seven adaptive contrastive pairs (14 examples) selected by statement type.
- One deterministic LLM call per pair at low temperature — plus, for `[Complex]`
  claims, a focused relation-nature step (Gilda-grounded entity aliases) that
  rejects a non-binding relationship mistaken for a Complex.

The default variant is `disconfirm_relnature` (set `MONO_VARIANT=""` for the
plain six-rule baseline; `MONO_VARIANT=disconfirm` for disconfirm without the
relation-nature step).

### Adaptive few-shot selection

The example bank has type-specific contrastive pairs. For each record, 7 pairs are selected by priority:

1. **Own type** from bank (e.g., Activation pairs for an Activation claim)
2. **Adjacent types** from `TYPE_ADJACENCY` map (e.g., IncreaseAmount for Activation)
3. **Universal patterns** (logical inversion, hedging scope)
4. **Fill** from the base contrastive pair set

Types with bank examples: Activation (2 pairs), Inhibition (2), Phosphorylation, Complex, IncreaseAmount, DecreaseAmount, Dephosphorylation, Autophosphorylation, Translocation, Ubiquitination.

### Run the scorer

```bash
PYTHONPATH=src python -m indra_belief.scorers.scorer \
    --model gemma-remote \
    --arch monolithic
```

## Design decisions we already paid for

Earlier iterations measured the following approaches and rejected them. If you're considering a change that resembles one of these, check the data before re-proposing:

| Approach | Outcome | Why it fails |
|---|---|---|
| Decomposed multi-call scorer | Strictly dominated on holdout_cc (F1 0.657 vs 0.751 monolithic) | Natural-language extraction can't bridge INDRA's soft ontology boundaries — requires multiple LLM probes to agree on a fuzzy contract |
| Native tool-calling (agentic lookup) | 84.9%, below baseline | Model ignores tool results after committing to a verdict in its first pass |
| Structured provenance, full population | -6.7pp accuracy | Attention dilution on 26B model outweighs disambiguation benefit — selectively enabling provenance only for flagged-grounding records preserves the signal without the cost |
| Graduated warnings for every grounding quirk | 3 regressions per 1 fix | Redirects attention from sentence comprehension; now limited to PSEUDOGENE and LOW_CONFIDENCE |
| Indirect-evidence marker in the prompt | +5pp false negatives | Prejudices model toward rejection; removed |
| LOW_CONFIDENCE auto-reject (blanket) | 53.6% precision at scale (32 false rejections on 3,754 records) | The gilda score threshold is too noisy to gate on deterministically; the signal is still available to the LLM as context |

Headline baselines measured during iteration: gemma-4-26b + adaptive bank + voting reaches ~84% accuracy on the 501-record stratified sample. Small-holdout numbers (200 records) overstate by ~4-5pp relative to large-scale evaluation (3,000+ records) — check the larger set before celebrating.

## Setup

### Dependencies

```bash
pip install gilda indra openai anthropic   # anthropic only for the claude-* path

# Download the benchmark corpus (460MB, not included in repo)
# Place at data/benchmark/indra_benchmark_corpus.json.gz
# Source: https://doi.org/10.5281/zenodo.7559353
```

### Model configuration

The scorer calls an LLM via `ModelClient(model_name)`. Model names map to
entries in `model_client.py`'s `LOCAL_MODELS` dict, or to Anthropic model
IDs (any string starting with `claude-`).

**Local Ollama (recommended for getting started):**

```bash
# Install Ollama: https://ollama.com
ollama pull gemma3:27b          # or any model you prefer
ollama serve                    # starts on localhost:11434
```

Then add an entry to `LOCAL_MODELS` in `src/indra_belief/model_client.py`:

```python
"ollama-local": {
    "base_url": "http://localhost:11434/v1",
    "model_id": "gemma3:27b",
    "reasoning_in_content": False,
    "max_tokens": 1000,
    "timeout": 120,
},
```

Use it: `ModelClient("ollama-local")` or `--model ollama-local` from the CLI.

**Remote Ollama (e.g., a beefy server on your network):**

Same as above but point `base_url` at the remote host. The `gemma-remote`
entry in the registry shows this pattern — it targets an Ollama instance
over Tailscale.

**Anthropic API:**

```bash
export ANTHROPIC_API_KEY=sk-...
```

```python
client = ModelClient("claude-sonnet-4-20250514")
```

Any `claude-*` model name routes to the Anthropic backend automatically.

**Key `LOCAL_MODELS` fields:**

| Field | Purpose |
|-------|---------|
| `base_url` | OpenAI-compatible endpoint (Ollama serves this at `/v1`) |
| `model_id` | Model name as known to the server (`ollama list` to check) |
| `reasoning_in_content` | `True` if CoT appears in `content` (Qwen CRACK); `False` for models with a separate `reasoning_content` field (Gemma 4) or no reasoning |
| `max_tokens` | Completion token budget — reasoning models need more (8000+) |
| `num_ctx` | Ollama-specific: context window size (passed via `extra_body`) |
| `timeout` | Seconds before retry — increase for large models or slow hardware |

## Usage

### Score a Statement's evidence

An INDRA `Statement` bundles a list of `Evidence` objects. `score_statement`
mirrors that abstraction: one per-sentence verdict per evidence, returned
in order.

```python
from indra.statements import Phosphorylation, Agent, Evidence
from indra_belief import ModelClient, score_statement

stmt = Phosphorylation(
    Agent("RPS6KA1"), Agent("YBX1"),
    residue="S", position="102",
)
stmt.evidence = [
    Evidence(source_api="reach",
             text="RSK1 phosphorylates YB-1 at S102 in response to stress."),
    Evidence(source_api="sparser",
             text="The kinase-dead RSK1 mutant was unable to phosphorylate YB-1 at S102."),
]

client = ModelClient("gemma-remote")
verdicts = score_statement(stmt, client)
# verdicts is list[dict], one per evidence:
#   verdicts[i]["verdict"]    → "correct" | "incorrect" | None
#   verdicts[i]["score"]      → 0.95 (correct+high) … 0.05 (incorrect+high)
#   verdicts[i]["confidence"] → "high" | "medium" | "low"
#   verdicts[i]["tier"]       → which scoring path produced the verdict
```

The importable `score_statement` / `score_evidence` run the **monolithic**
scorer — the default arch (empirically dominant on holdout_cc, F1 0.751 vs the
decomposed 0.657). For the decomposed four-probe path, import the same names
from `indra_belief.scorers.decomposed`.

To score just one evidence of a Statement (skipping the rest of `stmt.evidence`), use `score_evidence(stmt, ev, client)`.

### Composition with INDRA belief

`score_statement` is the per-sentence comprehension layer. The edge-level
question — *given all evidence for a statement, what is the belief?* — is
answered by a calibrated score that operationally treats those verdicts as noisy
measurements of one latent fact: whether the statement is correct. The mixed-
evidence limitation of that assumption is stated below. The two layers chain directly:

```python
from indra_belief import score_statement
from indra_belief.statement_belief import statement_belief
from indra_belief.noise_model import RECALIBRATED_PRIORS
from indra_belief.calibration_constants import calibration_for_run

verdicts = score_statement(stmt, client)  # list[dict], one per stmt.evidence
rows = [
    {"source_api": ev.source_api, "verdict": v["verdict"],
     "confidence": v.get("confidence"), "tier": v.get("tier")}
    for ev, v in zip(stmt.evidence, verdicts)
]
# Canonical edge belief for verdicts loaded from a persisted scoring run.
# Resolve model + scorer-prompt identity from that same run's call logs; a
# model name alone is deliberately insufficient.
profile = calibration_for_run("data/results/my_run.jsonl")
sb = statement_belief(rows, RECALIBRATED_PRIORS, soft=profile)
# sb.belief             → hybrid log-odds score (fitted) / hard gate (unfitted)
# sb.parametric_only    → belief before any LLM gating (ablation)
# sb.verdict_statement  → tiered decision: correct | review | incorrect
```

For a ship-approved reader configuration, `calibration_for_run` returns the reader's measured
verdict-by-gold confusion matrix and quantities derived from it—never hand-set
weights. Its `log_lr_confirm` field is
`log(P(confirm|correct) / P(confirm|incorrect))`; `log_lr_reject` is the
analogous rejection log-likelihood ratio. A confirmed read contributes the
stronger of the reader's confirm log-LR and its INDRA source-reliability
log-odds, so the confirmation contribution cannot undercut an already stronger
curated-source contribution; a
rejected read contributes `log_lr_reject`, and an unscored direct input uses
source reliability alone. Correlated reads from the same source are averaged,
independent sources are summed with the explicit fit-set prior, and a sigmoid
converts the resulting log-odds to belief.

Production currently enables two exact configurations: remote Gemma with prompt
fingerprint `b44638216740…` (4/4 on the independent holdout) and reasoning-first
Bedrock Gemma with `07377e338ff2…` (4/4 on external curator gold). Remote MedPsy's
`b44638216740…` profile remains a measured diagnostic candidate but is disabled:
its matched holdout failed the ECE leg (3/4), while its external run used the
different `07377e338ff2…` prompt and cannot validate that fit. Missing, mixed, or
mismatched prompt provenance therefore returns `None` and retains the hard gate.

That source term is a posterior reliability estimate from a separate 9,342-row
source-prior fit, not another likelihood ratio. The fitted-reader scalar is
therefore an explicit hybrid calibration score, not a pure Bayesian posterior;
changing the prior anchor is a global score shift, not a clean deployment-
prevalence correction. The evaluation target is also conservative: evidence
labels roll up to statement gold with any-incorrect-wins. It is a useful review
proxy, not a literal observation of one latent statement truth when evidence is
mixed.

The `soft=` argument name is retained for API compatibility; it now accepts this
measurement profile, not survival weights.

For an unfitted reader, the hard-gate fallback retains confirmed/unscored
evidence and removes rejected evidence before applying the parametric noisy-OR.
The tiered `verdict_statement` is the production decision (deterministic
hard-flag → `incorrect`; else any LLM `incorrect` → `review`; else `correct`)
and is independent of the belief scalar. Source priors live in `noise_model.py`
(`INDRA_PRIORS`, `RECALIBRATED_PRIORS`).

### Score a corpus + browse the results

For corpora larger than a single Statement (e.g. an INDRA-native JSON dump
from rasmachine), the monolithic pipeline is the production path. It scores
each evidence and writes append-only per-evidence JSONL alongside a run
`.meta.json` and `.progress.ndjson`:

```bash
set -a; . ./.env; set +a   # GEMINI_API_KEY / AWS_BEARER_TOKEN_BEDROCK / HF_TOKEN
PYTHONPATH=src python scripts/run_rasmachine_monolithic.py \
    --model gemma-remote \
    --input data/corpora/latest_statements_rasmachine.json \
    --output data/results/rasmachine_run.jsonl
```

Estimate cost first: `from indra_belief.corpus import estimate_cost` returns
projected LLM-call counts and USD per model before you spend.

The `viewer/` SvelteKit app browses finished runs. It is a read-only,
in-memory projection over the per-run exports under `data/exports/<run>/`
(`per_statement.json` + `per_evidence.jsonl` + `export_meta.json` +
`metrics.json`), loaded by SvelteKit server load functions (`+page.server.ts`) —
no database. Current calibration comparisons fail closed unless both products
carry matching byte-level corpus and gold digests, the same exact evaluated
evidence- and statement-key sets, and a compatible metrics contract. Temporal deltas are
stricter still: they require the same exact reader configuration. Fit-set
results are labeled in-sample and are never presented as external validation.

```bash
cd viewer && npm install && npm run dev  # http://127.0.0.1:5173
```

### Observed LLM cost (per run)

Each run's export carries the real USD it cost to score, computed from the token
usage actually observed during the run — not an estimate. Prices live in exactly
one place (`src/indra_belief/corpus/cost.py`); the viewer only reads baked numbers.

At export time, every evidence row's `call_log` (one entry per LLM call, each
carrying `prompt_tokens`, `out_tokens`, and the real `model_id`) is priced via
`token_cost_usd` and summed. Per-row `cost_usd` is baked into `per_evidence.jsonl`;
a run total + input/output token totals + `usd_per_1k_evidence` go into
`export_meta.json`. The run feed (`/runs`) shows a compact per-run cost; the run
detail (`/runs/<id>`) shows total, cost per 1k LLM-scored evidence, tokens, and
the model(s) billed.

Three honest states — the viewer never invents a price:

- **known** — every scored row used a model with a verified price (local /
  self-hosted models are genuinely free → `$0.00`).
- **partial** — some rows used a priced model and some an unverified one; the
  total covers only the priced rows, with the unavailable-row count shown.
- **unavailable** — no row had a verified per-token price, or the export predates
  cost capture. Shows "cost unavailable" with token counts, never a fabricated `$0`.

AWS Bedrock Claude (`sonnet-4-6`, `haiku-4-5`) and Gemma 4 (`gemma-4-26b-a4b`,
`gemma-4-31b`, `gemma-4-e2b`) are priced at published AWS/Anthropic on-demand
list rates; local models are zero marginal cost. A model in neither table reads
"unavailable" rather than a fabricated $0. To price a model, add its per-1M-token
input/output rate to `MODEL_PRICES_PER_M_TOKENS` (or its id to
`ZERO_COST_MODEL_IDS` if free) in `cost.py`, then re-export the run.

### Benchmark evaluation against a holdout file

```bash
PYTHONPATH=src python -m indra_belief.scorers.scorer \
    --model gemma-remote \
    --arch monolithic \
    --holdout data/benchmark/holdout_large.jsonl \
    --output data/results/run.jsonl \
    --resume data/results/run.jsonl  # resume interrupted runs
```

## How we iterate

Contributor-facing rules to keep the repository legible:

- **`main` is the canonical state.** Every "ship" decision ends with `git push`. Local ship decisions don't count.
- **Version labels don't belong in source.** Version numbers appear in PR titles, CHANGELOG entries, and benchmark-run output filenames (`data/results/<run>.jsonl`). They do *not* appear in source comments, docstrings, or identifier names. `scripts/check_no_version_labels.py` enforces this.
- **Public API is `score_statement(statement, client)` + `score_evidence(statement, evidence, client)`.** `score_statement` mirrors INDRA's abstraction (a Statement owns a list of Evidence) and returns one dict per evidence. `score_evidence` is the atomic per-sentence call. `score(client, record, …)` is the benchmark-harness path used by `indra_belief.scorers.scorer.main`; treat it as internal.
- **Comments explain current constraints, not past versions.** If a reader needs history, `git log` is the source of truth. "Provenance is selectively enabled because full-population provenance dilutes attention" is legitimate. "Removed in v12" is not.

## Project structure

```
src/indra_belief/
  model_client.py          # Model transport (OpenAI-compat + Anthropic)
  noise_model.py           # INDRA SimpleScorer (parametric belief from source priors)
  statement_belief.py      # verdicts → hybrid log-odds score (hard-gate fallback)
  curation.py              # INDRA-curation gold rule + hash bridge + index
  metrics.py               # Binary confusion P/R/F1 + ECE calibration
  sampling.py              # Two-stage / priority sampling + Wilson half-width
  results.py               # Run-result loading + row shaping
  scorers/
    scorer.py              # Public score_statement / score_evidence + benchmark main
    _shared.py             # Verdict→score mapping shared across scorers
    context.py             # Per-record scoring context
    context_builder.py     # Grounding + alias context assembly
    commitments.py         # Claim-commitment extraction
    grounding.py           # Gilda-backed entity grounding
    kg_signal.py           # Knowledge-graph corroboration signal
    parse_claim.py         # Statement → typed claim parse
    relation_patterns.py   # Regex relation cues
    monolithic/            # Default scorer
      scorer.py            # MONO_VARIANT dispatch (default disconfirm_relnature)
      _prompts.py          # Baseline six-rule system prompt
      _prompts_disconfirm.py  # Commit-first disconfirm prompt + backstop
      _prompts_relation.py    # [Complex] relation-nature step (Gilda aliases)
    probes/                # Decomposed four-probe scorer (--arch decomposed)
      orchestrator.py      # Probe pipeline + router
      router.py            # Statement → probe set
      subject_role.py object_role.py relation_axis.py scope.py bind_check.py
      adjudicator.py       # Probe verdicts → final
      _llm.py types.py
    panel/                 # Objection-panel ablation
      orchestrator.py detectors.py adjudicator.py types.py
  corpus/
    cost.py                # estimate_cost + MODEL_PRICES_PER_M_TOKENS (only surviving surface)
  tools/
    gilda_tools.py         # Gilda lookup helpers
  data/
    entity.py              # GroundedEntity: single gilda resolution per entity
    scoring_record.py      # ScoringRecord: wraps INDRA Statement + Evidence
    corpus.py              # CorpusIndex: source_hash → Statement lookup
    example_bank.json      # Type-specific contrastive pairs

viewer/                    # SvelteKit dashboard — read-only projection over data/exports/<run>/
  src/lib/
    format.ts              # Cue extraction, verdict rendering, sentence formatting
    residuals.ts           # Residual histogram bucket logic
    index.ts               # Re-exports
    components/            # BeliefPrimitive, BeliefRuler, SiteNav, Validity
    data/                  # In-memory data layer over the JSONL exports
      runs.ts              # Run discovery (dirs with export_meta.json)
      queries.ts           # Per-run / per-statement / per-evidence selectors
      curation.ts          # INDRA-curation gold lane (twin of curation.py)
      adjudicate.ts review.ts store.ts types.ts
  src/routes/                # each route pairs a +page.svelte with a sibling
                             # +page.server.ts load (runs/[run_id]/ adds
                             # +layout.server.ts); the server loads run the
                             # $lib/data selectors over the per-run JSONL exports
    +page.svelte           # Dashboard: focus + findings + validity + runs feed
    +layout.svelte         # Shared nav shell
    +error.svelte          # Generic 4xx/5xx error page
    runs/+page.svelte                      # Runs index
    runs/[run_id]/+page.svelte             # Per-run detail (+layout.server.ts loads the run)
    statements/+page.svelte                # Matrix (paginated, URL-stated)
    statements/[stmt_hash]/+page.svelte    # Per-stmt deep-dive (evidence cards + rollup)
    compare/+page.svelte                   # Model-vs-model dig (L0–L3, optional gold mode)
    adjudicate/+page.svelte                # Blind human verdict (curation revealed as 3rd judge)
    review/+page.svelte                    # Faithfulness / correctness review queue

data/
  benchmark/
    holdout.jsonl          # 200-record balanced evaluation set
    holdout_large.jsonl    # 4,625-record half-corpus evaluation
    example_pairs.json     # Entity pairs excluded from holdouts
  exports/<run>/           # Per-run viewer exports (per_statement.json + per_evidence.jsonl + export_meta.json)
  corpora/                 # Sampled INDRA Statement dumps to score
  results/                 # Evaluation results

scripts/
  run_rasmachine_monolithic.py  # Production scoring runner
  check_contamination.py        # Pre-eval gate: examples must not overlap holdout
  check_no_version_labels.py    # CI guard: no v{n} labels in src, tests, scripts

.github/workflows/
  ci.yml                        # pytest + both guards on every push and PR
```

## References

- Gyori et al. (2023). "Automated assembly of molecular mechanisms at scale from text mining and curated databases." *Molecular Systems Biology*, e11325. [Benchmark corpus: Zenodo 7559353](https://doi.org/10.5281/zenodo.7559353)
- [Gilda](https://github.com/gyorilab/gilda) — Biomedical entity grounding
- [INDRA](https://github.com/gyorilab/indra) — Integrated Network and Dynamical Reasoning Assembler
