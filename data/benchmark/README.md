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
| `archive/holdout_cc_rebuilt.jsonl` | 465 | 207 / 258 |

The first four are exactly 1:1 because a sampler made them so.
`scripts/build_curation_eval.py::stratified_balanced` takes
`min(#correct, #incorrect)` per `stmt_type` and draws that many of *each* class,
matching the `source_api` mix within type; `build_external_gold.py`,
`build_multicurator_gold.py` and `build_v2_balanced.py` all import it.
`eval_curation_v1.meta.json` and `external_gold_v1.meta.json` declare it:
`"balance": "1:1 forced"`. The older holdouts (now under `archive/`) are
exactly 1:1 too (`holdout.jsonl` 100/100; `holdout_v5.jsonl`,
`eval_set_v4.jsonl` 50/50); no meta file records the mechanism for those.

**Known leak (waived, enumerated):** nine of `eval_curation_v1.jsonl`'s 1,606
rows overlap the monolithic prompt's v6/v7 contrastive examples — six share
the sentence itself, three more only the entity pair. Five of the six leaked
source_hashes sit verbatim in `build_holdout.py::V6_V7_EXAMPLE_HASHES`, the
rows the builders were meant to exclude — consistent with the guard's old
Source-1 silent-zero import bug. Gold and prompt are both sha-frozen, so
`check_contamination.py` waives exactly these eleven findings by key
(`KNOWN_LEAKS`) and fails on any new one. Surfaced 2026-09-01, when the guard
began deriving its eval set from the calibration profiles instead of a
hand-maintained list that never included the fit golds. Measured the same day
to be immaterial: refitting the three profiles fitted here with the nine rows
excluded moves every log-likelihood ratio by ≤ 0.009 nats.

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

## `gene_corpus_uniform_400_llm_curated.jsonl`

A uniform random draw of 400 statements from a 3,250,298-statement gene-gene
corpus run, one evidence sampled uniformly per statement, judged on whether the
evidence sentence ALONE backs the statement. Not class-balanced, no selection on
the label, so unlike the sets above it *does* carry a population rate:
**72.0% incorrect [67.4, 76.2]**. Belief ranks that outcome at AUROC 0.83 —
13% correct at belief <= 0.5, 58% above it; the INDRA DB belief on the same
statements ranks at 0.56.

**The verdicts are LLM judgements, not human curation.** Two independent judges
agreed on 376/390 (96.4%) with a third adjudicating the rest, against a written
standard calibrated on ten hand-curated items; every label was then re-examined
by a reviewer and every review by an assessor, which moved five labels (all
correct -> incorrect, four of them grounding errors). That is enough to estimate a rate;
it is not a human label set. Do not pool these rows with `external_curator_gold_*`
or `eval_curation_*`, and do not call them gold. `.meta.json` records the seed,
the frame, the standard, and the retrieval endpoint.

## `gene_corpus_uniform_400_all_evidence_llm_curated.jsonl`

The same 400 statements with **every** evidence sentence judged — 1,346 pairs,
not one sampled sentence each. Use this for anything at statement grain; the
one-sentence file above cannot answer a statement-level question when 37% of
statements have more than one evidence.

Rates: **57.8% of evidence sentences** fail to support their statement
[55.1, 60.4]. At statement grain, **68.2%** have no supporting sentence at all
[63.5, 72.6], and **79.8%** have at least one unsupported sentence [75.5, 83.4].
Two Opus judges agreed on 1,320/1,346 (98.1%); 26 went to a third.

Against the same statements, ranking "does any evidence support it": LLM belief
AUROC **0.842**, INDRA DB belief **0.634**.

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
- `archive/` — closed-finding splits of that pool (`holdout*.jsonl`,
  `eval_set_v4.jsonl`, `fewshot_pool*.jsonl`, `probe_relation_logic.jsonl`,
  `holdout_cc_rebuilt.jsonl`, the orphaned `example_pairs.json`): every row is
  contained in `belief_benchmark.jsonl`, nothing reads them at runtime, and
  they persist because result manifests under `data/results/` cite them by
  path. Gold-builder exclusion globs sweep `archive/` too. The live exceptions
  stay at top level: `holdout_large.jsonl` (default guard holdout) and
  `holdout_large_fit.jsonl` / `eval_curation_v1.jsonl` (profile fit golds).
- `rasmachine_v1_gold.jsonl` (60) and `rasmachine_v2_balanced_*.jsonl` (114, 64)
  — Rasmachine cuts, the latter two `stratified_balanced` carvings of
  `rasmachine_v2_gold.jsonl`. `archive/probe_relation_logic.jsonl`
  (199, 100/99) is a relation-logic probe.
- `multi_evidence_statement_gold_*.jsonl` — statement-grain gold
  (`statement_gold`), for aggregation experiments, not per-evidence scoring.
