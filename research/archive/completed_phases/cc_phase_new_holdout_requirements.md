# CC-phase holdout requirements

## Problem

`eval_set_v4` (100 records, curator-audited) is now variance-limited
at 27B-Gemma scale. BB-phase reverted after BB1 hit its two targets
cleanly but cross-record LLM resampling drift moved F1 by −0.027
(0.827 → 0.800). The signal-to-noise on this holdout is too low to
discriminate small architectural wins.

The practical accuracy ceiling on `eval_set_v4` is 96-98% (per
`scripts/z_phase_curator_audit.py`: 4/100 records hedged or reader-
error-flagged). AA at 81.6% is approaching the achievable maximum
given the gold-tag distribution. ~12 of 18 residual errors are at the
L1/L2 grounding ceiling or contradicted-evidence ceiling — unreachable
from the probe layer.

## Goal

Commission a new holdout that:
1. Discriminates ≥0.01 F1 differences between architectures with
   statistical reliability (`p < 0.05` McNemar test on paired binary
   verdicts).
2. Has a high practical accuracy ceiling (≤5% hedged-or-flagged) so
   probe-level work is not bottlenecked by gold noise.
3. Stratifies across statement types, gold-tag categories, and
   source_api so per-class metrics carry meaning.
4. Has zero contamination with `eval_set_v4` and zero overlap with any
   training data used for substrate/LF tuning.

## Size requirement

McNemar discrimination at α=0.05 power=0.8 for a single-percentage-
point F1 lift requires roughly **n ≈ 384 paired observations** when
the discordant-pair rate is ~5%. Round up for stratification overhead:
**target n = 500 records minimum**.

Rationale: the 100-record `eval_set_v4` has a discordant-pair count
between AA and BB1-only of 9 (the 6 regressions + 3 recoveries) on a
total of 98 shared records. McNemar's exact test on 9 discordant pairs
gives `p ≈ 0.18` — not significant. At n=500 with the same proportional
discordance, p would clear 0.05.

## Stratification

Target distribution to reflect the corpus mix:
- **Statement type**: 30% Activation, 20% Phosphorylation, 20% Complex,
  10% IncreaseAmount, 10% DecreaseAmount, 5% Inhibition, 5% other
  (Methylation, Acetylation, Dephosphorylation, Translocation, etc.).
- **Gold tag**: 50% correct, 15% grounding, 10% no_relation, 8%
  wrong_relation, 7% entity_boundaries, 5% act_vs_amt, 3% polarity,
  2% other.
- **Source API**: ≥4 readers represented (reach, sparser, trips,
  medscan, rlimsp at minimum).
- **Belief prior bins**: spread across the belief-score distribution
  (don't oversample high-prior records — current eval has too many).

## Curator audit threshold

- **Hedged**: ≤5% of records (curator note contains "should be...
  allowing", "technically", "borderline", or similar).
- **Reader-error-flagged**: ≤2% of records (curator note has
  `[X] -> HGNC:N` or "wrong grounding").
- Run `scripts/z_phase_curator_audit.py` against the new holdout before
  it is sealed; reject the set if either threshold is exceeded and
  re-sample.

## Contamination guard

`scripts/check_contamination.py` exists already. Run against:
1. `eval_set_v4` (any record present in new holdout → drop)
2. `holdout_v5`, `holdout.jsonl` (older holdouts, may overlap with
   substrate-tuning data)
3. The V-phase training set (`data/training/` if it exists)
4. Few-shot examples in the prompt files
   (`src/indra_belief/scorers/probes/*.py`)

Reject the new holdout if ANY overlap is detected. Use INDRA pa_hash
+ source_hash + evidence_text hash for deduplication.

## Sourcing

Three viable paths:

### Path A — Re-curate from INDRA dump
- Pull a fresh 1000-record subset from the latest INDRA assembled
  statements, stratified per above.
- Hand-curate to 500 records (drop reader errors, ambiguous cases).
- Cost: ~2 person-days curator time. Cleanest provenance.

### Path B — Augment eval_set_v4 + holdout_v5
- Combine the unused records in `holdout_v5` (3000+) with `eval_set_v4`,
  then sub-sample to 500 stratified records.
- Cost: ~half-day. Some contamination risk with substrate training.
- Useful as a transitional holdout while Path A is being curated.

### Path C — RAS-machine corpus subset
- Use the `data/corpora/rasmachine_subset.json` already in repo.
- Cost: lowest. But this corpus has limited gold-tag annotation —
  needs curator pass.

**Recommendation**: Path B as immediate transitional (1 week);
commission Path A in parallel for the canonical CC+ phase holdout.

## Evaluation protocol

For each architecture (CC, CC+T1.A, etc.):
1. Run on the full 500-record set with the production model
   (gemma-remote at fixed temperature/seed).
2. Run TWICE more with re-sampled LLM outputs (different seed) to
   establish a variance band.
3. Report F1 ± 2σ across the three runs. An architecture is
   accepted only if the lower-bound exceeds the previous baseline's
   upper-bound (i.e., the bands don't overlap).
4. Standard error-profile output: per-stmt_type, per-gold-tag,
   per-source_api, ECE.

## When to commission

Trigger CC-phase holdout work IF AND ONLY IF the BB-T2.2 design (extract-
then-bind-check) is being implemented. There is no point in commissioning
a new holdout to evaluate further prompt-iteration phases — those will
be variance-limited at any size and the new holdout is wasted.

Order of operations:
1. Implement BB-T2.2 (extract-then-bind-check) against eval_set_v4 as
   a feasibility check — does it cleanly recover the 5 movable AA errors?
2. If yes: commission Path B transitional holdout.
3. Re-validate BB-T2.2 on Path B holdout with ≥3 runs.
4. If discrimination ≥ 0.02 F1 over AA baseline with non-overlapping
   variance bands: ship BB-T2.2 to origin/main.
5. Commission Path A canonical holdout for next phase's work.

## Open questions

1. Is the curator (collaborator, not in-house) available for the
   Path A curation effort, or do we need an external curator?
2. Should the new holdout include negative examples that are
   biologically *correct* but mis-curated? (Helps distinguish gold-
   ceiling from scorer error.)
3. Is the production model fixed for the next phase, or do we want
   to evaluate AA / BB-T2.2 on multiple model sizes (E4B + 27B + larger)?
