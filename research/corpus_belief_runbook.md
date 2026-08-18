# Corpus-scale belief with logits — runbook

> **Status: pipeline complete and rehearsed end to end 2026-08-18; the
> production calibration is NOT yet fitted.** Every stage below has been run
> against real production statements on a local MLX gemma-4-26b. No profile
> exists for `vllm-local`, so a corpus run today produces valid verdicts and
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
    --from-corpus-json data/corpora/eval_curation_v1_statements.json \
    --output-dir gold_shards
```

`--from-corpus-json` converts a JSON statement list into this script's own
`statement_hash<TAB>statement_json` input, keyed on `matches_hash` so shards
prepared from a labelled corpus key exactly as shards from the real dump.

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

Interruption is safe: finished jobs append to a partial file and are skipped on
restart, and the gzip result is written to a temp file then atomically renamed.

## 3. Calibrate — the stage that must run on the target stack

```bash
PYTHONPATH=src python scripts/fit_incall_calibration.py --from-shards \
    --input-dir gold_shards --results-dir gold_results \
    --gold data/benchmark/eval_curation_v1.jsonl \
    --model vllm-local --served-model-id google/gemma-4-26B-A4B-it \
    --out incall_vllm.json
```

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

| artifact | keyed on | without it |
|---|---|---|
| belief profile | (model, prompt sha256) | every belief is hard gate — ECE 0.237 against 0.045 |
| in-call isotonic | (model, served_model_id) | margins are carried but unused; every row keeps its verdict weight |

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
    --model vllm-local --out beliefs.json
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

- **No test covers the shipping configuration.** Every test of the logit path
  uses the local MLX reader, the only one with both artifacts registered. The
  first real exercise of `vllm-local` + `verdict_only` is the run itself.
- **Gold rows inside the corpus keep verdict weights.** The combiner refuses to
  score a record it was fitted on. Correct, counted, and negligible against 60M.
- **The labelled sets are curator-selected**, so `fit_prevalence` is the curated
  base rate rather than the corpus one. Closing that needs labels on a
  representative sample; the uniform CoGEx draw is the candidate and is
  unlabelled today.
- **Calibration degrades while discrimination improves.** On the local fit, ECE
  rose while Brier fell. The trade was favourable, but a consumer thresholding
  on belief feels reliability directly. More gold is the remedy.
