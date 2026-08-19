# Probe-battery findings

GO, but **not for the battery**. Adding no-reasoning probe signal to the incumbent verdict passed the held-out ΔAUROC gate; the battery ALONE (the replacement pitch the GOAL funded) is NO-GO. The final integrated review then ran the ablation no node had run, and it moved the claim: **a SINGLE probe carries essentially the whole gain**, and the other fifteen add an amount whose CI spans zero. Read §1b before quoting §1 — §1's Arm C row is a true measurement of a 16-probe arm, but 16 probes are not what earned it.

## 1. Verdict

| arm (artifact) | verdict | ΔAUROC | 95% CI | bootstrap | paired n | AUROC, candidate / incumbent | AP, candidate / incumbent | distinct scores, candidate / incumbent | ECE, candidate / incumbent | seconds per record |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| C: incumbent + battery (`data/probe_battery/decision.json`) | GO (`verdict`) | +0.05106813907508323 (`paired_bootstrap.delta_auroc`) | [+0.028335815405272274, +0.07467979904212096] (`paired_bootstrap.ci95_low`, `paired_bootstrap.ci95_high`) | percentile paired bootstrap clustered over `source_hash`, 453 clusters, max multiplicity 8 (`paired_bootstrap.resampling_unit` = `cluster`, `paired_bootstrap.cluster_field` = `source_hash`); 2000 (`paired_bootstrap.n_bootstrap`), 2000 valid (`paired_bootstrap.n_valid_resamples`), seed 0 (`paired_bootstrap.seed`) | 500 (`n`) | 0.7924965038337272 (`candidate.auroc`) / 0.7414283647586439 (`incumbent.auroc`) | 0.714757342116621 (`candidate.auprc`) / 0.6428498566745323 (`incumbent.auprc`) | 30 (`candidate.distinct_scores`) / 3 (`incumbent.distinct_scores`) | 0.14400892731661882 (`candidate.ece`) / 0.21369999999999995 (`incumbent.ece`), partitioned by `indra_belief.metrics.BINS_8` | battery portion 7.169757397527632 (JSON Pointer `/cost/candidate_s_per_record`) + incumbent 29.300608 (JSON Pointer `/cost/incumbent_s_per_record`) = **36.47036539752763 corrected combined wall-clock s/record**; see §5 |
| B: battery alone (`data/probe_battery/decision_B_battery_alone.json`) | NO-GO (`verdict`) | -0.0018967706675667717 (`paired_bootstrap.delta_auroc`) | [-0.048973162601383916, +0.044405574108855456] (`paired_bootstrap.ci95_low`, `paired_bootstrap.ci95_high`) | percentile paired bootstrap clustered over `source_hash`, 453 clusters, max multiplicity 8 (`paired_bootstrap.resampling_unit` = `cluster`, `paired_bootstrap.cluster_field` = `source_hash`); 2000 (`paired_bootstrap.n_bootstrap`), 2000 valid (`paired_bootstrap.n_valid_resamples`), seed 0 (`paired_bootstrap.seed`) | 500 (`n`) | 0.7395315940910772 (`candidate.auroc`) / 0.7414283647586439 (`incumbent.auroc`) | 0.6833455362168646 (`candidate.auprc`) / 0.6428498566745323 (`incumbent.auprc`) | 41 (`candidate.distinct_scores`) / 3 (`incumbent.distinct_scores`) | 0.08018977025707279 (`candidate.ece`) / 0.21369999999999995 (`incumbent.ece`), partitioned by `indra_belief.metrics.BINS_8` | battery 7.169757397527632 (JSON Pointer `/cost/candidate_s_per_record`) versus incumbent 29.300608 (JSON Pointer `/cost/incumbent_s_per_record`) wall-clock s/record; see §5 |

`data/probe_battery/decision.json` was byte-identical to `data/probe_battery/decision_C_incumbent_plus_battery.json`; it is the Arm C alias, not a third arm. The corresponding Arm C score aliases were also byte-identical.

The independent scratchpad recompute loaded both score JSONLs, derived TEST gold with `indra_belief.curation.is_gold_correct`, and reused `indra_belief.metrics.auroc`, `indra_belief.metrics.auprc`, `indra_belief.metrics.ece`, `indra_belief.metrics.BINS_8`, `indra_belief.metrics.brier_murphy`, and `indra_belief.metrics.reliability_bins`. Every asserted candidate and incumbent metric, timing field, cost field, and recorded paired-bootstrap field agreed with both decisions; the run ended `ALL_ASSERTIONS_PASSED`. This was independent of the artifact self-check.

Separately, the artifact's own `--reproduce` mode in `scripts/eval_probe_battery.py` returned exit 0 and `compared_fields=129` for Arm C, then exit 0 and `compared_fields=129` for Arm B.

## 1b. Ablation — the battery is not the win

No ablation was run anywhere in this arc until the final integrated review. The kill gate
(§4) measured redundancy IN-SAMPLE on the 600 FIT rows and used it only to justify fitting;
the question "does each probe contribute HELD-OUT?" sat in the seam between the kill-gate
node and the held-out-evaluation node, and neither node's gate owned it. Held out, it does
not convert.

Same shipped recipe, same isotonic `.score()` estimator, clustered paired bootstrap over
the 453 `source_hash` clusters. Reproduced independently by the orchestrator to the digit:

| comparison | ΔAUROC | 95% CI (clustered) | P(Δ≤0) |
|---|---:|---:|---:|
| FULL16 + incumbent − incumbent | +0.050921 | [+0.028053, +0.074677] | 0.0000 |
| **BASE1 + incumbent − incumbent** | **+0.045872** | **[+0.020968, +0.069776]** | 0.0005 |
| FULL16 − BASE1 | +0.005049 | [−0.022254, +0.033164] | **0.3668** |

BASE1 is `pol.verdict_direct` alone — the probe `killgate.json` marks `is_base`. The
fifteen non-base probes add +0.005 with an interval spanning zero, for fifteen-sixteenths
of the arm's added wall clock. A single-probe sweep is blunter still: `perturb.evidence_first`
ALONE scores 0.804311, BEATING all sixteen together (0.792497); the median single probe
reaches 0.784407 against the incumbent's 0.741428.

**This inverts the cost conclusion in our favour.** Sixteen probes cost
7.169757 s/record (~0.448 s each), making the additive arm
36.470365 s/record =
1.2447x the incumbent's
29.300608. ONE probe costs ~0.45 s/record, i.e. roughly 1.015x
the incumbent. The defensible finding is therefore **~+0.046 AUROC for ~1.5% more compute**,
not +0.051 for +24%.

**Transferable lesson:** in-sample non-redundancy does not imply held-out marginal
contribution. This is the third time in this program that a signal survived a stratified or
in-sample screen and died (or, here, proved superfluous) under a held-out combination test.

## 1c. The effect size is a range, and +0.051 is its top

The shipped bootstrap freezes the fitted combiner and resamples TEST only, so it prices
ZERO fit uncertainty. Three designs that do price it:

| design | ΔAUROC |
|---|---:|
| shipped (TEST-only cluster bootstrap, combiner frozen) | +0.051068 |
| refit on 600-row FIT bootstraps, 300 draws | +0.042301 (sd 0.010153, [+0.022118, +0.059856]) |
| grouped 5-fold CV inside TEST, 20 reshuffles | +0.033078 |
| swapped direction (fit on the 500, evaluate on the 600) | +0.025709 |

The sign never flipped in 300 refits, 20 reshuffles, or the swap, so the GO is safe. But the
effect should be quoted as **+0.026 to +0.051**, and the shipped interval understates the
uncertainty a deployment would face.

## 1d. Probe selection, done on FIT only — one probe beats the battery

§1b showed the fifteen non-base probes add nothing. That left an obvious question the arc
never asked: is `pol.verdict_direct` even the right single probe? It is the one
`killgate.json` marks `is_base`, but that flag designates the ANCHOR the redundancy test is
measured against, not a performance ranking — so using it as the ablation baseline was
arbitrary.

Selection was therefore run INSIDE FIT ONLY: grouped 5-fold cross-validation over the 600
FIT rows, folds split on `pa_hash` (437 clusters), fitting `(one probe + incumbent)` per
fold and scoring the held-out fold. TEST was not touched during selection. The winner was
then scored on TEST exactly once.

FIT-CV ranking, top and bottom: `perturb.evidence_first` 0.887190, `tax.relation_present`
0.881994, `tax.relation_type` 0.881064 ... `pol.verdict_direct` 0.872760 (8th of 16) ...
`tax.mod_site_match` 0.848805.

Held out, incumbent AUROC 0.741428:

| arm | AUROC | ΔAUROC | 95% CI (clustered) | P(Δ≤0) |
|---|---:|---:|---:|---:|
| **1 probe, FIT-CV selected (`perturb.evidence_first`)** | **0.804311** | **+0.062621** | [+0.036308, +0.087746] | 0.0000 |
| BASE1 (`pol.verdict_direct`, pre-registered anchor) | 0.787554 | +0.045872 | [+0.020968, +0.069776] | 0.0005 |
| FULL16 battery | 0.792497 | +0.050921 | [+0.028053, +0.074677] | 0.0000 |

**A single FIT-selected probe beats the whole battery held out** (+0.0626 vs +0.0509) at
about one sixteenth of its cost — roughly 0.45 s/record, so the additive arm runs at about
1.015x the incumbent instead of 1.2447x. The strongest single signal is a PROMPT
PERTURBATION (evidence presented before the claim), not the direct verdict re-ask.

Two honesty notes. The FIT-CV pick and the TEST-best pick MATCH, so the choice was not made
by peeking. But selecting the maximum of sixteen on FIT-CV and then reporting that arm's
TEST delta still carries a mild winner's curse: +0.0626 is an optimistic point estimate for
what a fresh selection would replicate, and the fully pre-registered comparator remains
BASE1's +0.045872. Second, this inherits §1c's caveat — the interval prices no fit
uncertainty, so read the effect as a range with +0.063 at its top.

## 2. AP beside the count

AP must be read beside score cardinality. On this historical TEST artifact, the then-current six-cell projection realized 3 values, {0.05, 0.2, 0.95}; it did not realize 2 values. That projection has since been retired from the scoring path. The count is a split-specific measurement, not a property to carry between datasets.

A low-cardinality incumbent leaves many ties for any more continuous score to split. AP can therefore rise even when AUROC does not. The measured post-CoT `p_raw` instance scored AP 0.7511 against the incumbent's 0.6901 while its AUROC was worse, 0.7320 against 0.7476. An AP-only reading would have shipped a worse ranking score. `research/scoring_methods.md` §7 already defined AP; no alternative definition is introduced here.

Calibration also required a domain check. Both decisions recorded `candidate.in_unit_interval: true`, and the independent recompute passed probabilities—not logits—to ECE. A scratch check over 4 deliberately out-of-range values returned ECE 0.0, so raw logits can look perfectly calibrated to this estimator even though they are invalid inputs.

## 3. Split discipline and pairing grain

File totals and rows actually fitted were different:

- FIT file `data/benchmark/eval_curation_v1.jsonl`: 1606 rows, 803 gold-correct and 803 gold-incorrect under `indra_belief.curation.is_gold_correct`, with 1604 distinct `source_hash` values.
- Rows actually fitted, from `data/probe_battery/killgate.json` at JSON Pointers `/inputs/n_rows` and `/inputs/n_clusters`: 600 rows over 437 `pa_hash` clusters.
- TEST file `data/results/cc_holdout_cc/holdout_cc.jsonl`: 500 rows, 233 gold-correct and 267 gold-incorrect, with 453 distinct `source_hash` values. **TEST was not balanced.** FIT and TEST had 0 overlapping `source_hash` values.

I re-derived the FIT total, its two class counts and distinct-hash count, and the TEST total, its two class counts and distinct-hash count directly from those JSONLs. TEST was tag-sorted: its first 100 rows were all gold-correct, so the recompute used the complete holdout and never a prefix.

The stored incumbent scores came from `data/results/holdout_cc_gemma.jsonl`, not from the older S-phase composite stored in `data/results/cc_holdout_cc/holdout_cc.jsonl`. The decision artifact's incumbent used 3 stored grid cells and had AUROC 0.7414283647586439. The older source-hash-collapsed S-phase composite had AUROC 0.6485; substituting it would have flattered the apparent candidate delta by roughly 0.10.

There is a further provenance caveat. All 500 stored incumbent scores and timings matched a last-occurrence-per-`source_hash` mapping from `data/results/holdout_cc_gemma.jsonl`. Against strict row identity, 6 stored scores and 46 stored timings differed, and 5 repeated hashes had varying source scores before that collapse. This does not break reproduction of the frozen decisions—their own score JSONLs are internally reproducible—but it limits what their recorded incumbent provenance establishes.

At TEST pairing time, `record_id` was `row_index:source_hash` and all 500 values were distinct, so the paired decision comparison was at row grain. The 453 distinct hashes contained 34 hashes with more than one row, covering 81 rows and accounting for 47 rows beyond one per hash. Their multiplicity histogram was {2: 29, 3: 2, 4: 1, 5: 1, 8: 1}, with maximum multiplicity 8. Zero repeated-hash groups carried contradictory gold, leaving zero unwinnable opposite-label pairs. Incumbent score was identical within all 34 groups; Arm C battery score was identical within 18 of 34, and Arm B battery score within 17 of 34.

The shipped `paired_bootstrap_delta_auroc` estimator resamples the 453 sorted `source_hash` cluster keys with replacement and includes every row belonging to each drawn cluster. Each decision artifact declares `resampling_unit` = `cluster`, `cluster_field` = `source_hash`, `n_clusters` = 453, and `max_cluster_multiplicity` = 8, so its uncertainty unit is auditable without consulting the input rows.

Relative to the former row bootstrap, Arm C's interval widened from approximately [+0.0286, +0.0731] to [+0.028335815405272274, +0.07467979904212096] and remained GO. Arm B's interval widened from approximately [-0.0474, +0.0423] to [-0.048973162601383916, +0.044405574108855456] and remained NO-GO. Both point estimates were unchanged; the after-fix clustered intervals are the `paired_bootstrap` values shipped in the decision artifacts.

## 4. Kill gate

`data/probe_battery/killgate.json` recorded verdict GO, `powered: true`, empty `power_failure: []`, split `fit`, and `in_sample: true`. Its input was 600 rows in 437 `pa_hash` clusters (`cluster_field`), with class imbalance 0.05. Base AUROC was 0.7362684489000278 and its cluster-bootstrap standard error was 0.023899209601884096.

The preregistered rule was NO-GO = (K1 and K2) or K3. K1 used 0.9 as the probe-vs-base absolute-Spearman threshold; K2 used 0.02; K3 used pairwise absolute Spearman 0.9 together with PC1 explained-variance-ratio CI low 0.95. K1, K2, and K3 were all false. The recorded `rule_sha256`, `b23952b83b98f9cc8bf8ed7e438b4192f454df8ff4b349905a13bfd80a014650`, matched the preregistration.

The measured ranges were:

- per-probe AUROC 0.2796–0.7611;
- probe-versus-base absolute Spearman point estimate 0.5589–0.9754, with `ci_low_abs` 0.4901–0.9684;
- pairwise absolute rho 0.2962410689048916–0.9754 over 120 pairs;
- PC1 explained-variance ratio 0.7591496421970682, CI [0.7375554141318241, 0.7807066792310041], and `n_effective_dimensions` 7.

There were 9 non-redundant probes among the 15 non-base probes—16 probes only when the base anchor `pol.verdict_direct` is counted. Six were flagged under K1: `perturb.paraphrase`, `pol.relation_direct`, `pol.verdict_flipped`, `tax.direction_polarity`, `tax.relation_present`, and `tax.relation_type`.

`pol.relation_flipped` had AUROC 0.2796, below chance and therefore informative by inversion, with Spearman -0.8923 against the base. The scalar choice was also a disclosed deviation from C1: `scalar_source` was `delta_logit when present; otherwise logit(clip(p_raw, 1e-6, 1-1e-6))`. PC1 is not rank-invariant, so K3's PC1 term depends on that scalar choice.

GO at this kill gate meant the probe set retained enough non-redundant in-sample structure to justify fitting and held-out evaluation. It did not mean that the battery replaced the incumbent, passed the held-out gate, calibrated a belief model, or generalized beyond this run.

## 5. Cost correction

The `cost` blocks in both decisions were byte-identical. They recorded JSON Pointer `/cost/candidate_s_per_record` as 7.169757397527632 and JSON Pointer `/cost/speedup_x_if_replacement` as 4.0866944828710405 in both arms. That ratio was 29.300608 / 7.169757397527632: the battery-alone wall-clock ratio. It was mislabeled when copied onto Arm C, because Arm C consumes the incumbent verdict as a feature.

For Arm C, the true per-record wall clock recomputed from `incumbent_seconds` plus `candidate_seconds` in `data/probe_battery/holdout_scores.jsonl` was 29.300608 + 7.169757397527632 = 36.47036539752763 seconds per record, displayed as 36.470365. This was 1.2446965400010686 times the incumbent wall clock, displayed as 1.2447×: the winning arm cost more, not less.

The GOAL's “matching at 4× less compute is a real win” condition describes what Arm B achieved on wall clock, and Arm B was NO-GO. Placing `speedup_x_if_replacement` or 4.0866944828710405 beside Arm C's GO would fabricate a cost win.

On the current measurement host, `system_profiler` identified an Apple M5 Max; the decision artifacts themselves did not persist a hardware identity. These are **WALL CLOCK** observations—local MLX inference for `mlx-community/gemma-4-26b-a4b-it-8bit` versus the incumbent's remote API calls—not a compute-cost comparison.

## 6. Relation to `research/scoring_methods.md`

Section 2.2 already proposed the renormalised verdict-token probability and already cited Vashurin et al., TACL 2025. This document reports a measurement, not a new proposal.

- **§2.3 says** the gemma-4 logprobs parameter was accepted but an empty array was returned. **Measured here:** that was a route fact about the hosted responses endpoint, not a model fact. The local MLX substrate returned full-vocabulary logprobs for gemma-4-26b-a4b, which made these measurements possible. The constraint needs narrower scope, not deletion.
- **§2.2 says** “our output is fourteen tokens.” **Measured here:** that described the verdict-only arm and did not transfer to the deployed `disconfirm_relnature_rf` prompt. The approximately 437–502 median-completion-token range came from the operator's project and session measurements; the checked-in production run independently had a median of 463 tokens across 33,361 scored rows in `data/comparison/runs/gemma_26b_primary/attempts.jsonl`, whose manifest named `disconfirm_relnature_rf` in `data/comparison/grounding_replay/manifest.json`.
- **§2.2 proposes** the renormalised probability as a confidence signal. **Measured here:** taken as one scalar replacement, that proposal was refuted on this reader. The battery combined 16 such readings; it beat the incumbent only when the incumbent was added as a feature, while the 16-reading battery alone did not.

Recommended exact edit, without editing `research/scoring_methods.md`: annotate §2.2 immediately after its short-output rationale with: “The fourteen-token rationale applies only to the verdict-only arm; the deployed `disconfirm_relnature_rf` prompt had an approximately 437–502-token median completion length. On gemma-4-26b-a4b, one renormalised verdict-token probability did not improve held-out AUROC as a replacement. A fitted battery of 16 no-reasoning readings also failed as a replacement and improved held-out AUROC only when added to the incumbent verdict score. Treat this subsection as a proposal for a measurement family, not evidence for a validated drop-in score.” Also change §2.3's lead-in to “Measured by route and substrate” and replace the gemma cell with “hosted responses route: parameter accepted, empty array returned; local MLX substrate: full-vocabulary log-probabilities returned.”

## 7. Refuted and survived arms

| arm | measured | dataset and n | in-sample or held-out | result | why it died |
|---|---|---|---|---|---|
| Verbalized confidence (the historical six-cell grid's `confidence` field) | ΔAUROC of the retired six-cell projection over the bare binary verdict, across 15 stored arms | eval_curation_v1, n≈1600 | held-out gold, joined on `source_hash` | **+0.0013** AUROC on the production arm; exactly 0.0000 on `gpt-oss-120b`; most arms emit `high` on 99.4–100% of executions | a field that takes one value carries no information; matches Vashurin et al.'s "reflexive methods do not demonstrate good performance" |
| Post-CoT `p_raw` | renormalised verdict-token probability after ~502 median tokens of deliberation | holdout_v5, n=99 | held-out | AUROC 0.7320 vs incumbent 0.7476; ECE 0.2525 vs 0.2025; **within-verdict AUROC 0.4966 (correct) / 0.4241 (incorrect)**; 100% of values within 1e-6 of 0 or 1 | the model has already committed by the time it emits the label — the distribution measures commitment, not claim uncertainty |
| Single no-reasoning prefill `p_raw` | one forced-prefill token, `enable_thinking=False` | holdout_v5, n=99 | held-out | AUROC **0.678** vs incumbent 0.748; accuracy 61% vs 75%; ECE 0.393; unsaturates (4–9% interior) and restores within-verdict AUROC 0.675 | unsaturation is not signal — it recovered within-stratum ranking and still lost outright, and within the *deliberated* verdict it gave 0.519/0.526, i.e. chance, so it is not complementary either |
| Deliberation length | `tokens` already persisted in every run row; logistic on [verdict] vs [verdict, log tokens, interaction], 50/50 split, 200 reseeds | eval_curation_v1, n≈1550 per arm, 15 arms | held-out, 200 reseeds | within-verdict AUROC 0.56–0.72 across all 15 arms, but held-out **−0.0078** on the production arm, **7/200 reseeds won**; ECE degraded in every arm | strong within-stratum signal that is redundant once the verdict is in hand |
| Self-consistency K=5 | 5 sampled chains per record, in-process MLX with an explicit reseed (the HTTP server returns byte-identical output and cannot produce sampling diversity) | holdout_v5, n=100 | held-out | **+0.009 AUROC, not significant** at that n | same shape as the others, and it costs 5× inference for a delta inside the noise |
| Arm B: battery alone | fitted combination of 16 no-reasoning readings, without the incumbent feature | holdout_cc, n=500 | held-out TEST | NO-GO; ΔAUROC -0.0018967706675667717, 95% CI [-0.048973162601383916, +0.044405574108855456] | it replaced the committed decision, its point estimate was worse, and its interval spanned zero |
| Arm C: incumbent + battery | the same fitted no-reasoning readings combined with the incumbent verdict score | holdout_cc, n=500 | held-out TEST | GO; ΔAUROC +0.05106813907508323, 95% CI [+0.028335815405272274, +0.07467979904212096] | it did not die: adding the battery to the committed score passed, although the combined arm cost more and was not independent of the incumbent |

The holdout_v5 arms and the self-consistency run were measured in uncommitted session scratchpad files. Those rows are a historical record, not a reproducible artifact. Only the two probe-battery rows are backed by files under `data/probe_battery/`.

The repeated failure mode was one scalar trying to replace an already committed decision: promising stratified behavior did not survive the held-out comparison. Arm C changed the shape by adding a fitted battery to that decision rather than replacing it; Arm B shows that the battery did not survive alone.

## 8. What this does not establish

This was one reader, gemma-4-26b-a4b, one prompt family, one 500-row holdout, and a fit of 600 rows. It does not establish cross-reader, cross-prompt, or deployment generalization. Arm C's ranking gain rode on the incumbent verdict as a feature, so it was not an independent second opinion; the last-per-hash incumbent provenance noted in §3 narrows the claim further.

`indra_belief.noise_model` remained unchanged, and this score was not a belief input. Nothing here was wired into the belief mathematics.

The guard in `scripts/check_doc_anchors.py` was a referential check. Every path and symbol here can resolve while a sentence is false, so a green verification result is not a warrant for these claims.

### 8b. Residuals recorded by the final integrated review

- **Content-level overlap, 2 of 500 TEST rows (0.4%).** Hash-grain disjointness
  (`source_hash`, `record_id`, `pa_hash`, `matches_hash`) is genuine and was all that was
  ever checked, but TEST rows 214 (`INSR`/`PDK1`/`grounding`) and 327
  (`TJP2`/`Cyclin`/`entity_boundaries`) share an exact `evidence_text` and triple with a
  FIT row. Far too small to move a +0.05 delta; recorded because the split claim was
  phrased absolutely and only hashes were compared.
- **The incumbent baseline is join-order dependent across ~0.008 AUROC.**
  `holdout_cc_gemma.jsonl` carries 453 distinct `source_hash` over 500 rows and 5 hashes
  hold more than one distinct score, so the join resolves duplicates last-occurrence-wins.
  First-occurrence 0.7457202102522063 / last-occurrence 0.7414283647586439 (shipped) /
  positional 0.7374017456719872. The shipped value is the MIDDLE of the three, so the
  delta is not manufactured by the join.
- **38.87% of positive-negative pairs are TIED under the real 3-value incumbent**, and that
  tie mass is the mechanism of the reported gain. `scripts/eval_probe_battery.py`'s
  docstring argues its AP-flattery defence against a SIX-value incumbent measured on
  `data/results/cc_holdout_cc/holdout_cc.jsonl` — a file this run never used. The real
  incumbent is more degenerate than the one defended, which strengthens the caution rather
  than weakening it.
- **Two decorative constructs, both resolved.** A module-level
  `AP_REQUIRES_DISTINCT_COUNT = True` that nothing read has been deleted (the invariant is
  enforced by an AST-locked test, not by that symbol). The unreached-report prefix grouping
  in `scripts/check_import_boundary.py`, which no live run can exercise because every
  unreached module is currently a singleton, was extracted to a tested function — and the
  extraction exposed a latent ORDER-DEPENDENCE that would have printed a child module on
  two lines had the input not happened to be sorted upstream. Now order-independent.

### 8c. Selection caveat, stated plainly

§1d's single-probe selection was computed on FIT only — grouped 5-fold CV over the 600 FIT
rows with folds split on `pa_hash`, TEST scored exactly once afterwards — and FIT-CV
independently ranked `perturb.evidence_first` first (0.887190, clear of second at
0.881994), matching the TEST-best.

But the integrated review had ALREADY reported the TEST single-probe sweep before that
selection was run. The procedure is untainted and the agreement is real; the analyst was
not blind. A fully pre-registered selection would have been run before anyone looked at
TEST. The comparator carrying no such caveat is BASE1 at +0.045872 — treat §1d's +0.062621
as the optimistic end and BASE1 as the conservative one.

Two supporting facts from the leakage audit. No estimator hyperparameter was ever tuned:
`fit_combiner` defaults (n_splits=5, C=1.0, seed=0) are used at every call site in the arc,
and a repo-wide grep finds no site overriding `C` or `n_splits`. The 600 FIT rows were
drawn with `{"mode": "sample", "n": 600, "seed": 0}` and the runner has no prefix-selection
path at all, which matters because holdout_cc is tag-sorted and a prefix would have been
single-class.

Arm C fits 17 features on 600 rows (35.3 rows per feature) against the one-probe arm's 2 on
600 (300 per feature). Thin-data overfitting in the wide arm is consistent with, and helps
explain, why its extra fifteen probes do not convert held out.
