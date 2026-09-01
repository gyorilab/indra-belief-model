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

The parsed `(verdict, confidence)` pair is closed-set audit output and is not
converted into a probability (`src/indra_belief/verdict.py::Verdict`). At
`src/indra_belief/scorers/monolithic/scorer.py::score`,
`src/indra_belief/probes/calibration.py::replace_sentence_score` attaches
`score`: the calibrated sentence probability when the client can perform a
fitted direct-probe read, otherwise `None`.

## How it works

Model: gemma-4-26b (Ollama remote or local MLX 8-bit).

### Production scoring architecture

The CLI default is the monolithic scorer: an LLM call per
`(Statement, Evidence)` pair with type-adaptive contrastive examples (a second
call fires only for `[Complex]` claims — see Tier 2). A decomposed four-probe
scorer used to sit beside it for ablations; it lost on holdout_cc and has been
removed (git history holds it).

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
- One LLM call per pair at temperature 0.1 — low but nonzero, so replies are not
  bit-reproducible, and what is fixed per pair is the rendered prompt. For
  `[Complex]` claims, a focused relation-nature step (Gilda-grounded entity
  aliases) rejects a non-binding relationship mistaken for a Complex.

The default variant is `disconfirm_relnature_rf` (set `MONO_VARIANT=""` for
the plain six-rule baseline; `MONO_VARIANT=disconfirm` for disconfirm without
the relation-nature step; `MONO_VARIANT=disconfirm_relnature` for the
reasoning-SECOND variant, which is no longer the default). That
reasoning-first prompt is what the shipped `gemma_bedrock_rf` calibration
profile is keyed on, so changing the variant changes the `(model, prompt_sha)`
calibration key and the fitted profile stops applying — see `_FITTED_CONFIGS`
in `src/indra_belief/calibration_constants.py`.

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

Python 3.10+ (CI pins 3.12). From a clean checkout:

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"        # gilda + indra + openai/anthropic + test and script deps

python -m pytest -q                          # 15 tests skip without the untracked data tree
python scripts/check_contamination.py        # few-shot examples must not overlap an eval file
python scripts/check_import_boundary.py      # no core module may import the research harness
python scripts/smoke_end_to_end.py           # 5 checks, hermetic: no socket, no credential
```

CI runs `pytest` and the contamination guard (`.github/workflows/ci.yml`); the
boundary guard and the smoke check are local. `PYTHONPATH=src` substitutes for
the editable install when running the scorer and the scripts — the form used
throughout this README — but the test suite still needs the `[dev]` extra for its
analysis dependencies.

The benchmark corpus (460MB) is not in the repo: download it from
[Zenodo 7559353](https://doi.org/10.5281/zenodo.7559353) and place it at
`data/benchmark/indra_benchmark_corpus.json.gz`.

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

**Local MLX (Apple Silicon):**

`scripts/serve_mlx.sh` serves a reader through `mlx_lm.server`, which speaks
the same OpenAI-compatible protocol as Ollama. Setup, once:

```bash
uv venv ~/.venvs/mlx-serve --python 3.12
VIRTUAL_ENV=~/.venvs/mlx-serve uv pip install mlx-lm
```

```bash
scripts/serve_mlx.sh                    # gemma-4-26b-a4b 8-bit on :8085
MODEL=mlx-community/gemma-4-31b-it-8bit PORT=8084 scripts/serve_mlx.sh
```

The MLX stack lives in its own virtualenv at `~/.venvs/mlx-serve` **by choice,
not by necessity** — there is no dependency conflict, and an earlier revision of
this paragraph claiming one was wrong. Re-runnable evidence:
`VIRTUAL_ENV=$PWD/.venv uv pip install --dry-run mlx-lm` resolves 34 packages and
would install exactly three — `mlx==0.32.0`, `mlx-lm==0.31.3`,
`mlx-metal==0.32.0` — upgrading nothing and removing nothing. `mlx-lm` 0.31.3
declares no torch and no sympy at all; its core requirements are `mlx>=0.31.2`
(marker `platform_system == "Darwin"`), numpy, `transformers>=5.0.0`,
sentencepiece, protobuf, pyyaml and jinja2 — print them with
`~/.venvs/mlx-serve/bin/python -c "from importlib.metadata import distribution;
print(distribution('mlx-lm').requires)"`, whose remaining entries
(`datasets`, `lm-eval`, `tqdm`, the CUDA/CPU `mlx` variants) all sit behind
`extra ==` markers and so are never pulled by a bare install. Torch, sympy, pysb
and transformers already coexist in the project interpreter:
`.venv/bin/python -c "import torch,sympy,pysb,transformers as tf; print(torch.__version__, sympy.__version__, pysb.__version__, tf.__version__)"`
exits 0 and prints `2.11.0 1.11.1 1.17.0 5.9.0`. What is true is only a violated
*declaration* — `.venv/bin/python -m pip check` reports, among other metadata
complaints, `torch 2.11.0 has requirement sympy>=1.13.3, but you have sympy
1.11.1` — a complaint about declared bounds, not a failed import, and not what
forces the split.

What the split buys is a judgement, not a necessity. The cost it avoids is
measurable and modest: installing `mlx-lm` into `.venv` would add three
distributions totalling ~192 MB, ~188 MB of it the Apple-Silicon-only
`mlx-metal` binary.

```bash
~/.venvs/mlx-serve/bin/python -c 'from importlib.metadata import distribution as D; print([(n, D(n).version, round(sum(D(n).locate_file(f).stat().st_size for f in D(n).files)/1e6, 1)) for n in ("mlx", "mlx-lm", "mlx-metal")])'
# [('mlx', '0.32.0', 1.8), ('mlx-lm', '0.31.3', 1.6), ('mlx-metal', '0.32.0', 188.4)]
```

Do not quote `du -sm ~/.venvs/mlx-serve` (316 MiB) as the avoided cost: that is
the whole serving venv, and most of its non-MLX bulk — numpy, transformers —
`.venv` already carries. `mlx-metal`'s files also unpack *into* the `mlx/` import
directory (there is no `mlx_metal/` beside it), which is why `du -sm` on
`~/.venvs/mlx-serve/lib/python3.12/site-packages/mlx` reports 189 MiB of
allocated blocks rather than the ~2 MB `mlx`'s own files come to.

That weight stays out of the `uv.lock`-resolved environment, which has no `mlx`
entry at all (`grep -c '^name = "mlx' uv.lock` → 0) and is exercised on
`ubuntu-latest` CI (`.github/workflows/ci.yml:9`). The marker doing the work
there is `mlx-lm`'s own: it requires `mlx>=0.31.2; platform_system == "Darwin"`,
so off Darwin a resolver never *requests* `mlx` — it is not that `mlx` would
refuse to install. (`mlx` 0.32.0 carries no Darwin marker itself; its
Darwin-only piece is `mlx-metal==0.32.0; platform_system == "Darwin"`.)

No module in this repository imports `mlx` or `mlx_lm`; the `mlx-lm` dependency
is purely a `~/.venvs/mlx-serve` serving-venv concern. The scorer reaches the
MLX server over HTTP through `ModelClient`'s plain `openai_compat` backend, and
`scripts/probe_logprobs.py` is the dedicated logprob probe on that route. No
MLX-specific `httpx` caller remains; the generic
`scripts/run_vllm_processed_shards.py` server backend uses `httpx` and can be
pointed at `local-gemma-4-26b`. The 2.7 MB rows left by the retired probe remain
at `data/probe_battery/mlx_verdict_logit_rows.jsonl`, with no in-tree producer.

The script's `MODEL` and `PORT` defaults must stay equal to the
`local-gemma-4-26b` entry in `src/indra_belief/model_client.py` —
`mlx-community/gemma-4-26b-a4b-it-8bit` on port `8085`. Change one without the
other and every call 404s. This agreement used to be enforced by a test that
compared the script's defaults against the registry entry; that guard was removed
with the rest of the prose-checking apparatus, so the two are now kept in step by
hand.

Why serve locally at all: this is currently the **only** reader we can read
token logprobs from. Bedrock's gemma-4 routes accept `top_logprobs` and return
an empty array, so `p_raw` cannot be measured there at all. Two consequences
for callers:

- Stock `mlx_lm.server` caps `top_logprobs` at **11**. That is not enough at a
  forced verdict position: the losing label was measured at rank 42/83/168, and
  k=11 yielded no usable `p_raw` in three of four cases. The serving venv carries
  a local patch raising the validator ceiling;
  `src/indra_belief/model_client.py::LOCAL_MODELS` declares
  `max_top_logprobs: 1024` for `local-gemma-4-26b` on the assumption that patch
  is present. Rebuilding or upgrading the venv silently restores 11; the
  resulting registry/server mismatch records a per-row `score_error` rather
  than a calibrated score. See
  `scripts/serve_mlx.sh::LOCAL PATCH REQUIRED` for the patch commands.
- Serve at `temperature 0.0` — the script's default, and a correctness
  precondition for scoring, because the sampled verdict must be reproducible.
  The logprobs themselves are indifferent to it: `mlx_lm` computes
  `logprobs = logits - logsumexp(logits)` *before* the sampler runs
  (`generate.py:420-421` in mlx_lm 0.31.3), so temperature, top-p and top-k
  change which token is sampled but never the distribution we read.

Smoke-check a running server before trusting a scoring run:

```bash
.venv/bin/python scripts/probe_logprobs.py --model local-gemma-4-26b
```

It exits 0 only when logprobs came back and were non-degenerate. A route that
accepts `top_logprobs` and then returns nothing exits non-zero with status
`empty` rather than looking like a clean pass.

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
| `supports_logprobs` | `True` if the route actually returns `choices[].logprobs.content[]`. Absent/`False` means the field is accepted and ignored — the Bedrock gemma-4 failure mode |
| `max_top_logprobs` | Server-enforced ceiling on `top_logprobs`. Stock `mlx_lm.server` validates with `max_val=11` and 400s above it; the shipped `local-gemma-4-26b` entry declares 1024 because its serving venv is locally patched as documented in `scripts/serve_mlx.sh::LOCAL PATCH REQUIRED` |

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
#   verdicts[i]["score"]      → calibrated sentence probability; None without a fitted calibration/probe read
#   verdicts[i]["confidence"] → "high" | "medium" | "low"
#   verdicts[i]["tier"]       → which scoring path produced the verdict
```

The importable `score_statement` / `score_evidence` run the **monolithic**
scorer — the default arch (empirically dominant on holdout_cc, F1 0.751 vs the
decomposed 0.657). That decomposed path has since been removed, so these names
are the only ones.

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

Production registers seven exact configurations and enables six: remote Gemma
with prompt fingerprint `b44638216740…`, reasoning-first Bedrock Gemma with and
without verbalized confidence (`07377e338ff2…`, `bad4cb2d9f89…`), deliberated
local MLX Gemma (`07377e338ff2…`), and the verdict-only local MLX and vLLM
readers (`cd14d9e74d2e…`). Remote MedPsy's
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

To drive that chain from an existing INDRA pipeline instead of assembling it by
hand, `src/indra_belief/belief_scorer.py::LLMBeliefScorer` implements INDRA's own
`BeliefScorer` socket — `BeliefEngine(scorer=LLMBeliefScorer(client))` — and
resolves its calibration profile from the client and variant by itself. It reads
evidence text where INDRA's other scorers count sources, so it spends one
provider call per evidence: check `estimate_calls(statements)` first. It raises
`UnscorableStatement` rather than returning a float it did not measure, because a
missing belief reads back as `1.0`; `score_statements_detailed` returns the
tallies and a `float | None` belief.

### Representative INDRA curations

The `representative` lane starts from a **5,000-pair uniform evidence-row
reservoir** drawn without replacement with Algorithm R from exactly
**44,944,056 grounded/assembled evidence rows** in the CoGEx 2025-09-16 dump.
The 5,000 rows are the sampling frame, not the size of CoGEx, and the sampling
unit is an evidence row rather than a statement or every raw INDRA extraction.
The source dump and reproduced reservoir are pinned by SHA-256.

At serving time, the tracked manifest retains all 5,000 reservoir keys for
provenance but blocks two exact pairs that occur in older benchmarks, leaving
4,998 eligible keys. Every card shown to a curator is atomically reserved in a
persistent draw ledger; completed INDRA history and all prior reservations,
including skips, are removed before the next random draw. Production must set
`CURATION_DRAW_LEDGER_DIR` to storage shared by every consumer and
acknowledge it with `CURATION_DRAW_LEDGER_SHARED=1`; sampling fails closed when
that guarantee is not configured. Rows that no longer materialize through INDRA
or lack usable text are retried, so the served population is conditional on
materializability and text availability.

The June 29 snapshot was **not** drawn from this frame. It predates the July 3
reservoir lane and came from the older hand-selected, high-coverage agent-panel
sampler. The first qualifying reservoir curation is ID 19920 on July 6.

The tracked `mock7ee@gmail.com` artifact
`data/benchmark/representative_indra_curations_400.jsonl` is a
**first-write-wins unique-pair progress snapshot**; `_400` names the benchmark
target, not the current row count. It contains all 403 unique exact pairs from
415 qualifying submissions available through curation ID 20334 at the recorded
export cutoff. For each pair, the first
qualifying submission supplies the tag and derived binary label; the 12 later
repeat submissions are excluded from canonical rows and labels rather than
aggregated as votes, while their provenance remains in audit metadata. The
resulting snapshot is 199 correct / 204 incorrect. Each row contains one
curation event and embeds the judged INDRA statement structure for
clean-checkout inspection; no duplicate-event arrays or any-incorrect-wins
conflict rollups are present. The 400-pair target is `complete` and exceeded by
3 pairs. Benchmark status remains `pending` because the historical
completed-sequence randomness is unproven.

All 403 snapshot pairs are reservoir members and have zero prior-benchmark or
pre-reservoir-curation overlap. The latter is pinned against
`mock7ee_pre_reservoir_pair_manifest.jsonl` (124 genuine submissions from the retired viewer,
123 unique pairs; the unrelated API auth probe is excluded). That proves frame
membership, not that the historical completion sequence was a simple random
sample: the legacy UI retained no draw/skip log, retried unusable rows, and
allowed pairs to be drawn again. First-write deduplication removes those repeat
events from the artifact but cannot reconstruct a no-replacement draw history.
The artifact is therefore described as reservoir-sourced, not as a provable
simple-random sample of the reservoir.

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

A `viewer/` SvelteKit app used to browse finished runs as a read-only
projection over the per-run exports under `data/exports/<run>/`. **It has been
removed.** The exports it read are still produced and still tracked
(`per_statement.json` + `per_evidence.jsonl` + `export_meta.json` +
`metrics.json`), so the data layer is intact and every exporter script still
runs — there is simply no UI over it. Read the artifacts directly, or via the
Python helpers in `src/indra_belief/results.py`.

The invariants the viewer enforced at render time still hold as properties of
the artifacts: current calibration comparisons fail closed unless both products
carry matching byte-level corpus and gold digests, the same exact evaluated
evidence- and statement-key sets, and a compatible metrics contract; temporal
deltas require the same exact reader configuration; fit-set results are labeled
in-sample and are never external validation. The publication-grade statement
comparison has its own frozen artifact and status contract, recorded in the
committed artifacts under `data/results/` and in git history; the standalone
research memo that described it was retired with the comparison harness.

### Observed LLM cost (per run)

Each ordinary run export carries the real USD it cost to score, computed from
the token usage actually observed during the run — not an estimate. Pricing for
these exports lives in `src/indra_belief/corpus/cost.py`, which is where the
numbers are baked; consumers only read them.

At export time, every evidence row's `call_log` (one entry per LLM call, each
carrying `prompt_tokens`, `out_tokens`, and the real `model_id`) is priced via
`token_cost_usd` and summed. Per-row `cost_usd` is baked into `per_evidence.jsonl`;
a run total + input/output token totals + `usd_per_1k_evidence` go into
`export_meta.json`, alongside total input/output token counts and
`usd_per_1k_evidence`.

Three honest states — a price is never invented:

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

The statement-level INDRA comparison does not silently inherit that mutable
run-export table. Its LLM bundles bind `data/comparison/pricing.json`: structured
AWS Bedrock `us-east-1` on-demand pricing, requested tier `default`, resolved
tier Standard, exact provider model and token rates, retrieval date, and one
cost-comparability identity. All-source and five-reader costs are observed
projections of the same run and are explicitly non-additive.

### Benchmark evaluation against a holdout file

```bash
PYTHONPATH=src python -m indra_belief.scorers.scorer \
    --model gemma-remote \
    --holdout data/benchmark/holdout_large.jsonl \
    --output data/results/run.jsonl \
    --resume data/results/run.jsonl  # resume interrupted runs
```

## How we iterate

Contributor-facing rules to keep the repository legible:

- **`main` is the canonical state.** Every "ship" decision ends with `git push`. Local ship decisions don't count.
- **Immutable identities are explicit.** Dataset, schema, model, prompt, and decision-artifact identities retain their real names and hashes; prose describes the current contract rather than narrating refactor chronology.
- **Public API is `score_statement(statement, client)` + `score_evidence(statement, evidence, client)`.** `score_statement` mirrors INDRA's abstraction (a Statement owns a list of Evidence) and returns one dict per evidence. `score_evidence` is the atomic per-sentence call. `score(client, record, …)` is the benchmark-harness path used by `indra_belief.scorers.scorer.main`; treat it as internal.
- **Comments explain current constraints.** Historical implementation rationale belongs in `git log`; source comments state only the causal constraint that governs current behavior.

## Project structure

```
src/indra_belief/
  model_client.py          # Model transport: the ONE client. Seven backends
                           #   (OpenAI-compat, three Bedrock lanes, local
                           #   transformers, in-process vLLM, Anthropic)
  vllm_offline.py          # In-process vLLM engine behind model_client's
                           #   vllm_offline backend; batches N calls into one
  noise_model.py           # INDRA SimpleScorer (parametric belief from source priors)
  statement_belief.py      # verdicts → hybrid log-odds score (hard-gate fallback)
  belief_scorer.py         # LLMBeliefScorer: the above as an indra.belief.BeliefScorer
  curation.py              # INDRA-curation gold rule + hash bridge + index
  metrics.py               # Binary confusion P/R/F1 + ECE calibration
  results.py               # Run-result loading + row shaping
  scorers/
    scorer.py              # Public score_statement / score_evidence + benchmark main
    monolithic/            # The scorer (the only architecture)
    _shared.py             # JSON extraction from a model reply, shared
      scorer.py            # MONO_VARIANT dispatch (default disconfirm_relnature_rf)
      _prompts.py          # Baseline six-rule system prompt
      _prompts_disconfirm.py  # Commit-first disconfirm prompt + backstop
      _prompts_relation.py    # [Complex] relation-nature step (Gilda aliases)
  corpus/
    cost.py                # estimate_cost + MODEL_PRICES_PER_M_TOKENS (only surviving surface)
  tools/
    gilda_tools.py         # Gilda lookup helpers
  data/
    entity.py              # GroundedEntity: single gilda resolution per entity
    scoring_record.py      # ScoringRecord: wraps INDRA Statement + Evidence
    corpus.py              # CorpusIndex: source_hash → Statement lookup
    example_bank.json      # Type-specific contrastive pairs
data/
  benchmark/
    holdout.jsonl          # 200-record balanced evaluation set
    holdout_large.jsonl    # 4,625-record half-corpus evaluation
    example_pairs.json     # Entity pairs excluded from holdouts
  exports/<run>/           # Per-run exports (per_statement.json + per_evidence.jsonl + export_meta.json)
  corpora/                 # Sampled INDRA Statement dumps to score
  results/                 # Evaluation results

scripts/
  run_rasmachine_monolithic.py  # Production scoring runner
  check_contamination.py        # Pre-eval gate: examples must not overlap holdout
  serve_mlx.sh                  # Local MLX reader on Apple Silicon (the one logprob-capable route)

.github/workflows/
  ci.yml                        # pytest + guards on every push and PR
```

## References

- Gyori et al. (2023). "Automated assembly of molecular mechanisms at scale from text mining and curated databases." *Molecular Systems Biology*, e11325. [Benchmark corpus: Zenodo 7559353](https://doi.org/10.5281/zenodo.7559353)
- [Gilda](https://github.com/gyorilab/gilda) — Biomedical entity grounding
- [INDRA](https://github.com/gyorilab/indra) — Integrated Network and Dynamical Reasoning Assembler
