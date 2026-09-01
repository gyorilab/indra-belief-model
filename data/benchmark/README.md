# Gold sets in `data/benchmark/`

**These sets measure how well a judge separates correct from incorrect evidence.
The six evaluable ones cannot tell you what fraction of the corpus is wrong** —
selection into them depended on the label, so they carry no population rate. The
one exception is the reservoir-drawn snapshot described under *What a population
rate would need*.

## The balance is forced, not observed

| file | n | correct / incorrect |
| --- | --- | --- |
| `eval_curation_v1.jsonl` | 1606 | 803 / 803 |
| `external_curator_gold_v2.jsonl` | 1084 | 542 / 542 |
| `external_curator_gold_v1.jsonl` | 578 | 289 / 289 |
| `external_gold_v1.jsonl` | 154 | 77 / 77 |
| `rasmachine_v2_gold.jsonl` | 304 | 246 / 58 |
| `holdout_cc_rebuilt.jsonl` | 465 | 207 / 258 |

The first four are exactly 1:1 because a sampler made them so.
`scripts/build_curation_eval.py::stratified_balanced` takes
`min(#correct, #incorrect)` per `stmt_type` and draws that many of *each* class,
matching the `source_api` mix within type; `build_external_gold.py`,
`build_multicurator_gold.py` and `build_v2_balanced.py` all import it.
`eval_curation_v1.meta.json` and `external_gold_v1.meta.json` declare it:
`"balance": "1:1 forced"`. The older holdouts are exactly 1:1 too
(`holdout.jsonl`, `holdout_v7_original.jsonl` 100/100; `holdout_v5.jsonl`,
`eval_set_v4.jsonl` 50/50); no meta file records the mechanism for those.

That was the right call for the purpose: on a skewed set accuracy is dominated by
the majority class and a judge that flags nothing scores well, so error-detection
F1 is the headline metric here, and an identical `stmt_type` marginal across
classes keeps statement type from masquerading as a label effect.

## Why they cannot estimate a corpus error rate

Whether a row is in the set depends on its own label. Every estimator that
transports a labeled sample's accuracy onto an unlabeled population — the
prediction-powered family included — requires the labeled sample to be a
probability sample of that population with known inclusion probabilities. Forced
1:1 violates that at the first step, and no reweighting recovers it: the weights
would have to be inverse probabilities of a selection that ran on the outcome.

Post-stratifying on the belief score does not repair it. Belief is not the
variable selection depended on, and it is too coarse to stand in for one: at an
identical belief of 0.65 on single-evidence statements, measured accuracy runs
from 64.8% (trips) to 11.9% (sparser), χ² = 30.7, p = 3.6e-6, n = 315 — see
`research/indra_paper_literal_vs_llm_comparison.md`. The score distributions do not match
either. `eval_curation_v1.jsonl` carries a `belief` field, and its median is
0.9996 — the curated sets sit where the scorer is most confident, which is not
where the corpus sits. (Sharper band-by-band comparisons have been made against
a collaborator's 3.25M-statement gene run; that file is not in this repository,
so the figures are not reproducible here and are left out.)

Three defensible post-stratified estimates of the corpus error rate, run over
different pools, gave 61.2%, 67.3% and 68.2%. That spread is the
non-identification showing through, not sampling noise.

Independent of balance: evidence correctness clusters inside a statement —
intraclass correlation ρ = 0.828 on the curated pool. Sampling evidences directly
gives a design effect near 1; sampling statements and taking all their evidence
gives 1 + (4.67 − 1)(0.828) ≈ 4.0 at the corpus mean of 4.67 evidences per
statement, so 1000 evidences drawn that way carry the precision of 250.

## What they are good for

Judge selection, prompt work, threshold and calibration fitting, error-detection
F1 between arms — any question of the form "does A beat B at catching errors".

## What a population rate would need

A fresh probability sample drawn at **evidence** grain with recorded inclusion
probabilities; curators blinded to the LLM verdict; the judge frozen before the
draw. Roughly 1,050 evidences gives ±3pp at design effect ≈ 1. Half the
machinery is already here: `cogex_representative_pool_manifest.jsonl` is 5,000
rows drawn by uniform reservoir (Algorithm R, without replacement, seed
20260701) from 44,944,056 CoGEx evidence rows — a real probability sample. Its
curated subset `representative_indra_curations_400.jsonl` (n=403, 199/204, one
curator) is the closest thing to one that exists. Its completed subset is **204/403 = 50.6% incorrect**. Read that as the rate
among the pairs that were actually curated, not as a corpus rate: the draw of
5,000 is provably uniform, but which 403 of them got completed is not. Treat it as indicative, not settled: its
meta records `"simple_random_completed_subset_proven": false`, because the viewer
that served the rows kept no draw/skip log, so the 403 curated pairs cannot be
shown to be a random subset of the 5,000 drawn. Logging the draw order fixes
that, and finishing the remaining 4,597 removes the question entirely.

Method references: Tenenbein, *JASA* 65(331):1350–61 (1970), double sampling under
misclassification; Rogan & Gladen, *Am J Epidemiol* 107(1) (1978); Hui & Walter,
*Biometrics* 36(1):167 (1980); Angelopoulos et al., "Prediction-Powered
Inference", *Science* (2023); Rao & Scott, *JASA* (1981) for design effects.
PPI++ (arXiv:2311.01453), the variant `ppi_py` implements, has no peer-reviewed
venue.

## Other files here

- `belief_benchmark.jsonl` — 9,342 curations by the two expert curators, 58%
  correct; the pool `build_curation_eval.py` draws from. Not corpus-drawn either.
- `holdout*.jsonl`, `eval_set_v4.jsonl`, `fewshot_pool*.jsonl` — earlier splits
  off that pool, carrying the raw curation `tag` rather than a `gold` field.
- `rasmachine_v1_gold.jsonl` (60) and `rasmachine_v2_balanced_*.jsonl` (114, 64)
  — Rasmachine cuts, the latter two `stratified_balanced` carvings of
  `rasmachine_v2_gold.jsonl`. `probe_relation_logic.jsonl` (199, 100/99) is a
  relation-logic probe.
- `multi_evidence_statement_gold_*.jsonl` — statement-grain gold
  (`statement_gold`), for aggregation experiments, not per-evidence scoring.
