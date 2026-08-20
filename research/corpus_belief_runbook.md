# Corpus-scale belief with logits — runbook

> **Status: pipeline complete and rehearsed end to end 2026-08-18; the
> production calibration is NOT yet fitted.** Every stage below has been run
> against real production statements on a local MLX gemma-4-26b. No profile
> exists for `vllm-gemma-4-26b`, so a corpus run today produces valid verdicts and
> HARD-GATE beliefs. Fitting that profile is stage 3, and it is the one stage
> that must run on the target hardware.

Turns 60M INDRA evidences into a `{stmt_hash: belief}` table, using the LLM's
own verdict-token logits as a per-evidence weight of evidence.

Four stages, and they are deliberately separable:

| stage | script | needs a GPU |
|---|---|---|
| 1. prepare | `scripts/build_processed_grounding_shards.py` | no |
| 2. score | `scripts/run_vllm_processed_shards.py` | yes — this is the 60M pass |
| 3. calibrate | `scripts/fit_incall_calibration.py` | once, on a gold subset |
| 4. believe | `scripts/build_corpus_beliefs.py` | no |

## The property that makes the order flexible

**Stage 2 does not depend on stage 3.** The scoring pass persists a RAW
`probe_delta_logit` and computes no weight; both calibrations are applied in
stage 4. Verified by running stage 4 twice over byte-identical shard output and
getting two different belief tables.

So the expensive irreversible pass can run before any calibration exists, and
recalibrating 60M beliefs later costs minutes and zero GPU. This is also why the
runner must never be "improved" to bake a weight in: doing so would make every
recalibration a re-read of the corpus.

## Prerequisites

The serving flag is not optional:

```bash
vllm serve google/gemma-4-26B-A4B-it --port 8000 --max-logprobs 128
```

Every scoring request carries a 128-wide logprob window, because the verdict and
its margin come from the same call. vLLM's default cap is far below that and
rejects the whole request, not just the logprobs — so a server without this flag
fails every row. A preflight issues one real scoring request before any shard is
opened and exits naming the flag.

## 1. Prepare shards

From the production dump, or from a labelled corpus for stage 3:

```bash
PYTHONPATH=src python scripts/build_processed_grounding_shards.py \
    --from-corpus-json data/corpora/external_curator_gold_v2_statements.json \
    --all-evidence \
    --output-dir gold_shards
```

`--from-corpus-json` converts a JSON statement list into this script's own
`statement_hash<TAB>statement_json` input, keyed on `matches_hash` so shards
prepared from a labelled corpus key exactly as shards from the real dump.
The v2 file contains 748 statement objects but 1,084 labelled
statement/evidence pairs, so calibration preparation uses `--all-evidence`;
production preparation retains its historical first-evidence default.

The batch user message is byte-identical to the live scorer's — the record owns
the parts via `ScoringRecord.execution_body` and `ExecutionBody.render` owns the
join. That is asserted in `tests/test_build_processed_grounding_shards.py`, and
it was NOT true for a period in which this builder called a deleted method and
raised on every LLM-bound job.

## 2. Score

```bash
PYTHONPATH=src python scripts/run_vllm_processed_shards.py \
    --input-dir gold_shards --output-dir gold_results --workers 64
```

Defaults to `verdict_only`: no chain-of-thought, one request per evidence, the
label margin read free from the scoring response. Add
`--variant disconfirm_relnature_rf` for the deliberative path.

Do NOT pass `--probe`. It buys a second request per evidence for a strictly
worse reading — MEASURED n=80, in-call AUROC 0.8734 against the probe's 0.7237 —
and is suppressed automatically when the free margin is available.

Interruption is safe at shard granularity: completed result shards are skipped
on restart, an interrupted shard is rerun, and the gzip result is written to a
temp file then atomically renamed. Each failed job is retried three additional
times; an exhausted failure is retained with verdict "error" so one bad row
does not block publication of the shard.

## 3. Calibrate — the stage that must run on the target stack

### Which dataset

**Fit on `external_curator_gold_v2`** (n=1084, 542/542, 27 curators, built
2026-08-18). **Validate on `external_curator_gold_v1`** (n=578) and
`eval_curation_v1` (n=1606); all three are mutually disjoint under BOTH the
pa_hash and matches_hash identities.

The choice is about POPULATION, and the belief distribution of the underlying
statements is the reason:

| set | median INDRA belief | below 0.65 | at exactly 1.0 |
|---|---|---|---|
| `eval_curation_v1` | 0.9996 | 5.7% | 13.1% |
| `external_curator_gold_v2` | 0.5932 | 59.0% | 0.0% |

Neither set is thresholded on belief — `eval_curation_v1` reaches down to 0.3195
— but it is overwhelmingly statements INDRA already believes, which is not what
a uniform draw from the corpus looks like. The corpus is dominated by the
low-evidence tail, and that is the region `external_curator_gold_v2` occupies.

It was built with `scripts/build_multicurator_gold.py --cap 200 --name
external_curator_gold_v2`. Both its files are TRACKED -- the labels in
`data/benchmark/` and, unusually, the statements JSON force-added past the
`data/corpora/` ignore rule -- because the build inputs
(`curation_universe_all.jsonl` and the recovered-curation corpus) are not in the
repository, so a collaborator cannot regenerate it and stage 1 takes the
statements JSON as its input.

Validation on the other two golds runs on the authoring machine, which has their
statements; only the fit set has to travel. The cap is a real trade and 200 is where it was set:
the top curator contributes 20% of rows at cap 100, 24% at 200 and 32% at 400.
Above 200 the set starts belonging to one curator.

KNOWN LIMIT: every labelled set we hold is balanced 1:1 by construction, so
`fit_prevalence` will be about 0.5 whichever is used, and the corpus base rate
is not 0.5. That displaces every weight uniformly. It is correctable after the
fact -- from the corpus verdict rate and the fitted sensitivity and
false-positive rate -- and no dataset choice fixes it, because the two
corpus-representative samples we hold are unlabelled.



```bash
PYTHONPATH=src python scripts/fit_incall_calibration.py --from-shards \
    --input-dir gold_shards --results-dir gold_results \
    --gold data/benchmark/external_curator_gold_v2.jsonl \
    --model vllm-gemma-4-26b --served-model-id google/gemma-4-26B-A4B-it \
    --out incall_vllm.json --report incall_vllm_report.json
```

Pass `--report`. The isotonic carries the curve and nothing about how the curve
was gated — not the splits, not the medians, not the verdict — so without it
those numbers exist only in this process's stdout, and a profile ends up quoting
a Brier with nothing to cite. The report also persists the belief profile's
log-LRs, which are the other half of the fit.

Both artifacts belong in `data/probe_battery/`, **committed in the same commit as
the registry row**. A row pointing at a file that stayed on the fitting machine
does not fail loudly: `supports_sentence_calibration` starts answering True and
stage 4 dies on `FileNotFoundError` rather than its own refusal.

### Why this cannot be inherited

Two independent reasons, and neither is fixed by using the same prompt.

**Path.** MEASURED on 300 identical evidences: the gold-eval path and the shard
path disagree on the VERDICT for 10% of them, their margins correlate at r=0.874,
and the sign differs on 30/290. The control — the same shard path run twice over
the same shards — agrees 300/300 with r=1.0000 and 290/290 identical margins, so
the serving stack is deterministic and the divergence is the path.

**Population.** `fit_prevalence` is baked into the artifact and anchors every
weight: `weight_of_evidence = logit(p_hat) - logit(fit_prevalence)`. A curve
fitted on a balanced curated gold at 0.513 and applied to a corpus at 0.70
displaces every weight by +0.88 log-odds, and the isotonic's knots sit where the
FIT population's margins fell.

### It produces TWO artifacts and you need both

| artifact | registered in | keyed on | without it |
|---|---|---|---|
| belief profile | `calibration_constants._FITTED_CONFIGS` | (model, prompt sha256) | every belief is hard gate — ECE 0.237 against 0.045 |
| in-call isotonic | `probes/calibration._INCALL_CALIBRATIONS` | (model, served_model_id) | margins are carried but unused; every row keeps its verdict weight |

`_INCALL_CALIBRATIONS`, **not** `_SENTENCE_CALIBRATIONS` beside it. That one holds
separate-probe curves. The two tables have the same shape and the loader accepts
either route's artifact, so a row in the wrong table type-checks, loads, and
saturates every reading to 0 or 1 — probe knots span -1.70..+1.61 while in-call
margins run about 3x wider. Nothing downstream can tell.

Registering only the isotonic yields beliefs byte-identical to the hard gate.
The script registers NEITHER — it prints the two edits, because each is a ship
decision — so keep the printed output, not just the JSON.

### Reading the gate

Two conditions decide: ranking (paired-bootstrap CI lower bound above zero) and
scoring (Brier improves). Both are read from the MEDIAN over reseeded splits,
with the worst split printed beside them, because a single split is one draw and
this codebase has twice shipped a signal a different partition dissolved.

The reliability/resolution trade is PRINTED AND NOT GATED. See
`src/indra_belief/calibration_gate.py` for why: the quantity is binned and
noise-dominated at these sample sizes, and `scripts/calibration_ship_gate.py`
already declines to gate on it at four times the n.

A FAIL is a result. Send it.

## 4. Build beliefs

```bash
PYTHONPATH=src python scripts/build_corpus_beliefs.py \
    --input-dir shards --results-dir results \
    --model vllm-gemma-4-26b --out beliefs.json
```

Add `--served-model-id google/gemma-4-26B-A4B-it` once the isotonic is
registered. If the scoring run used `--limit`, the output filenames carry it;
the results path is resolved rather than reconstructed, and an ambiguous
directory holding two generations of a shard is refused rather than guessed.

Until stage 3 lands this prints `NONE — every row keeps its verdict weight`,
which is the honest state and still worth running: correct-but-uncalibrated
beats absent.

## Failure modes the pipeline refuses rather than absorbs

Each of these once produced a well-formed artifact and a zero exit code.

- **No shard results readable.** Usually a `--limit` mismatch. Now an error, not
  an empty table.
- **A statement spanning two shards.** Merging would replace a whole-statement
  belief with one computed from a fraction of its evidence. Refused.
- **A registered isotonic that weighted nothing.** Would publish a
  verdict-weighted table under a calibrated manifest. Refused, and the first
  underlying error is recorded beside the count.
- **A server returning 200 with no logprobs, or a tokenizer whose label token
  does not match.** Verdicts would land and every margin would be null. The
  preflight issues a real scoring request and requires an actual number from the
  same read the run uses.
- **Conflicting gold labels on one (statement, evidence) pair.** Dropped and
  counted, never resolved by file order.

## Known limits

- **No test covers the shipping configuration.** `vllm-gemma-4-26b` +
  `verdict_only` now has a belief profile, and a test asserts that pair gates
  green — but that checks the REGISTRY, not the reader. Every test of the logit
  path still runs against the local MLX reader, the only one with both artifacts
  registered. The first real exercise of the vLLM reader is the run itself.
- **The vLLM isotonic is not registered yet.** `incall_vllm.json` is fitted and
  gated but lives on cluster scratch, so the row in `_INCALL_CALIBRATIONS` is
  deliberately withheld until the file is committed beside it. Stage 4 with
  `--require-calibrated` therefore refuses — correctly, and by its own message
  rather than a `FileNotFoundError`. Scoring is unaffected: it gates on the
  profile alone, which IS registered.
- **The vLLM profile is the weakest-cited row in the table.** Its counts and
  gold digest are pinned, but its fit ran on `/scratch` with no `--report`
  artifact, so the Brier and ECE in its note cannot be re-derived by anyone
  else. Re-fit with `--report` to close it.
- **The offline backend fails a whole batch on one bad job.** `--backend
  offline` batches through `llm.chat()`, which is all-or-nothing, and retries
  re-batch with the same offender. The corpus path uses `--backend server`,
  where each job is its own request and the coupling does not exist.
- **Gold rows inside the corpus keep verdict weights.** The combiner refuses to
  score a record it was fitted on. Correct, counted, and negligible against 60M.
- **The labelled sets are curator-selected**, so `fit_prevalence` is the curated
  base rate rather than the corpus one. Closing that needs labels on a
  representative sample; the uniform CoGEx draw is the candidate and is
  unlabelled today.
- **Calibration degrades while discrimination improves.** On the local fit, ECE
  rose while Brier fell. The trade was favourable, but a consumer thresholding
  on belief feels reliability directly. More gold is the remedy.
