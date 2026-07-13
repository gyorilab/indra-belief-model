# Can the Gyori-lab deck re-base on the new multi-curator gold?

> [!IMPORTANT]
> **CURRENT CORRECTION — 2026-07-13.** The calibration reruns proposed in this
> document were completed, and their result supersedes the assessment below.
>
> - The live fitted scalar is a source-aware **hybrid log-odds calibration
>   score**. Reader verdict weights are genuine confusion-derived likelihood
>   ratios, but confirmations retain a separately fitted source-reliability
>   posterior-logit floor. The final scalar is therefore **not a pure Bayesian
>   posterior**, and its fit prior is a score anchor rather than a clean
>   deployment-prevalence knob.
> - Remote/Ollama Gemma fit: **704/157/97/646** on **1,604 unique pairs**.
>   Reasoning-first Bedrock Gemma fit: **662/81/139/722** on its own separately
>   scored, deduplicated **1,604 unique pairs**. The Bedrock configuration is
>   fitted; it does not inherit the remote profile.
> - Remote/Ollama Gemma on `holdout_cc` (**n=414**): raw count→hybrid
>   **ECE .430→.148 / AUROC .610→.745**; formal hard→hybrid
>   **.157→.148 / .734→.745**, error-F1 Δ **+.016 [.003,.032]**, 4/4.
> - Reasoning-first Bedrock Gemma on the independent external gold (464 scored
>   statements): raw count→hybrid **AUROC .688→.814 / ECE .129→.061**;
>   formal hard→hybrid **ECE .237→.061 / AUROC .794→.814**, error-F1 Δ
>   **−.001 [−.015,+.014]**, 4/4.
> - MedPsy's matched `holdout_cc` test (**n=414**) worsens ECE
>   **.129→.146**, so it is 3/4 and production-disabled. It has no valid
>   fitted-soft external result: the external monolithic
>   prompt SHA prefix `07377e338ff2` does not match its fitted/eval profile SHA
>   `b44638216740`. Treat that artifact only as an unmatched-profile stress test,
>   not as a formal gate/pass.
> - The validated Gemma checks use configuration-specific frozen profiles. Do not describe one Gemma
>   profile or one pair of numbers as having “held twice.” The 13-model frontier
>   remains a separate hard-verdict error-F1/cost analysis and does not validate
>   or invalidate the hybrid belief by itself.

## Current bottom line

The new multi-curator gold supports both strands, with different metrics: the
13-model fleet supports the reader-choice/error-detection sibling, while only the
configuration-matched reasoning-first Bedrock rerun supplies formal external
belief-calibration evidence. The MedPsy artifact is an unmatched-profile stress
test and supplies no fitted soft metric or gate result. Keep those strands and
provenance boundaries distinct.

## Historical assessment — preserved for decision provenance

> [!WARNING]
> The remainder is the pre-rerun assessment. Its statements that no calibration
> existed on the external gold, every Bedrock serving was unfitted, the fit had
> `n=1606`, or belief calibration was still a future run are intentionally
> historical. They explain why the reruns were commissioned; they are not current
> deck guidance.

*Historical verified assessment (4 examine-readers + adversarial verify). The
adversarial verdict held for the evidence available at that time.*

## Historical bottom line (superseded)

Partially — and the split is clean. The new multi-curator gold (external_curator_gold_v1, 578 rows / 289-289 balanced / 32 capped curators / live-refetch / de-contaminated vs eval_curation_v1) plus the 13-model ~$14.6 fleet re-bases ONLY the reader-choice and forward-validation beats (Slides 10 and 12) and strengthens the de-biased-gold methodology + "led with the losses" honesty ethos. It CANNOT re-base the deck's calibration core (Slides 4/7/8/9): the new runs are verdict error-detection (error-F1) grain, and no ECE/AUROC/Brier was recomputed on the new gold — the entire ECE 0.37->0.156 / AUROC 0.814/0.834 story stays anchored to eval_curation_v1 (n=913/1606) + holdout_cc (n=342). The single hardest tension: the frontier "pick" (gemma-4-31b @ 0.827) is a DIFFERENT model from the calibration-fitted gemma-26B and returns None from calibration_for() -> hard gate, so it carries none of the deck's belief calibration. Keep the calibration spine as-is; bring the frontier/cost result in as a sibling reader-selection section, not woven into the belief throughline.

## Historical crux — what the then-available fleet runs measured

The new fleet runs measure VERDICT ERROR-DETECTION F1 only: positive class = curator-flagged INCORRECT, prediction = verdict=="incorrect" (eval_curation_compare.py:281), computed by frontier_table.py via confusion_pr + cost. It imports NO ece/auroc/brier; there is no metrics.json / per_statement / calibration export for any external_curator run, and no ship_gate/belief_headtohead file references the new substrate. So the belief-calibration core (ECE/AUROC/Brier) was NOT re-run on the new gold. Note also the per-evidence `belief` field that travels in every run jsonl is the verdict-independent source-reliability parametric scalar (gemma-4-31b mean 0.691 on correct vs 0.595 on incorrect — barely separates), NOT a fitted soft statement-grain belief — and every external reader resolves to None -> hard gate, so no fitted soft belief was even produced. Grain note: headline "n=578" is gold rows (575 unique pairs, 3 dup rows); F1 is actually over ~587 joined evidence pairs (paired tests on n~572-573).

## Historical scope recommendation

SIBLING SECTION, not woven into the spine. The cost x error-F1 model frontier (gemma-4-31b vs glm-5 vs the big reasoners) is a verdict/error-detection + cost result orthogonal to the belief-formula/calibration throughline ("Second Reader = calibration gate"); importing it wholesale dilutes that spine and risks implying the new gold validated the belief numbers (it did not). Best placement: (1) it is the concrete evidence for Slide 10's existing "only 2 readers fitted; everything else (incl. 31b + bedrock) falls back to the hard gate" limit — now demonstrated by 13 uncalibrated fleet runs; (2) it advances Slide 12's "validate on our own gold + broaden the fitted-reader set" — the n=578 de-biased gold IS the new independent gold and the fleet IS the broadening exercise; (3) optionally a short standalone appendix/sibling deck for the cost-frontier audience. Do NOT promote gemma-4-31b to the calibration headline. If the lab wants the cost frontier as a first-class talk, it should be its own deck (different metric, different substrate, different question).

## Historical slide-by-slide plan

| action | deck target | detail | source |
|---|---|---|---|
| **KEEP** | Slide 1 (schematic 0.9 two-reader edge) | Pure schematic, self-labeled illustrative; new gold is irrelevant. No change. | ben_gyori_deck_outline.md:12-25 |
| **KEEP** | Slide 2 (additive second reader; 'ranks fine but sure of everything') | Seeds the Slide-4 calibration figures on eval_curation_v1; not touched by new gold. | ben_gyori_deck_outline.md:29-42; B1/B2 |
| **KEEP** | Slide 3 (formula stays) | INDRA SimpleScorer formula; substrate-independent. No change. | ben_gyori_deck_outline.md:46-59 |
| **FLAG** | Slide 4 (ranks ~0.74 / ECE ~0.37 calibration collapse) | DO NOT re-base and must not be implied as validated on new gold. ECE/AUROC were never recomputed on n=578 — they stay on eval_curation_v1 (n=913) + holdout_cc OOD. Add a one-line guard in the Say so no one infers the n=578 fleet confirmed these belief numbers. The 0.96->0.81 mirage is error-F1 grain and belongs on Slide 12 / the sibling, NOT here. | deck-fit report Q1 (frontier_table.py:23-25; no calib export); belief_headtohead_gemma.md:16,19,22,24 |
| **KEEP** | Slide 5 (same belief, opposite truth — relation-direction reversal) | Schematic; the rasmachine polarity-incorrect record citation is unaffected. No change. | ben_gyori_deck_outline.md:80-93 |
| **KEEP** | Slide 6 (membership gate mechanics) | Hard-gate / rand=1.0 mechanics are substrate-independent. No change. | ben_gyori_deck_outline.md:97-110 |
| **KEEP** | Slide 7 (soft 'clean' form; gemma-26B {0.183,0.869} / medpsy-4B {0.243,0.873}) | These two fitted readers (and only these two) remain the calibration instrument; fit is n=1606 eval_curation_v1, NOT the new gold. Keep verbatim. Optionally note that the fleet result on Slide 10/sibling is precisely why these are still the only two fitted. | calibration_constants.py:42-43; gold-integrity + errf1-mirage reports (calibration_for gates 31b/bedrock to None) |
| **KEEP** | Slide 8 (priors refit, reach 0.30->0.462, n=9342) | Prior refit is on the INDRA assembly benchmark; unrelated to the new gold. No change. | calibration_constants.py:48-62 |
| **KEEP** | Slide 9 (OOD ship gate, holdout_cc n=342, ECE 0.171->0.129 / AUROC 0.740->0.768) | Calibration ship-gate; new gold did not touch it. Existing Appendix-B TODO (re-run clean arm vs guard predecessor) still stands and is independent of this arc. Could be cited on Slide 12 as 'the next OOD set is our own n=578 gold' but only after calibration is actually run on it. | calibration_ship_gate.md:12,46; Appendix B item 3 |
| **REBASE** | Slide 10 ('only 2 readers fitted; rest incl. 31b + bedrock fall back to hard gate') | STRENGTHEN with concrete evidence: the 13-model bedrock fleet ran the error-detection eval and EVERY one resolved to None -> hard gate (no fitted soft belief). Turn the abstract limit into 'we just ran 13 models on fresh balanced gold; all 13 are hard-gate-only — here is exactly what unfitted looks like at scale.' This is the cleanest, most defensible re-base of the new arc into the existing deck. | deck-fit report Q4; calibration_constants.py:41-66; frontier_table.py (13 runs) |
| **KEEP** | Slide 11 (review tier) | Tier mechanics + grounding-reject precision are on eval_curation_v1 / holdout_cc; unaffected. No change. | ben_gyori_deck_outline.md:184-197 |
| **REBASE** | Slide 12 ('validate on EMMAA + our own gold; broaden the fitted-reader set') | STRENGTHEN directly: 'our own gold' now exists — n=578 de-biased multi-curator gold (32 curators, live-refetch, de-contaminated), and the 'broaden the readers' beat is concretely advanced by a 13-model fleet ranking. State it as forward progress made, not just intent. Pair with the to_rerun item: error-detection is validated, belief-calibration on this gold is the next step. | gold-integrity report (build_multicurator_gold.py); deck-fit report Q4 |
| **ADD** | Cost/frontier result (gemma-4-31b 0.827@$0.68; glm-5 0.839@$5.89; size doesn't buy error-detection; fleet ~$14.6) | Add as a SIBLING section/appendix, NOT in the spine. Frame: top ~8 tied 0.800-0.839 (overlapping CIs) so cost decides; gemma-4-31b is the cheapest of the tied tier (production cost-winner); glm-5 is Pareto-OPTIMAL (highest F1) not 'dominated' — its +0.012 is within CI for 8.7x cost; truly dominated = kimi/deepseek/qwen3-coder-480b/qwen3-235b/gpt-oss-120b. Use 'n=587 evidence pairs' for the F1 denominator and 'n=578 balanced gold' for the gold. Carry the explicit caveat that the cost-winner is NOT calibration-covered. | frontier-cost report (frontier_table.py output; Pareto set gemma-4-e2b/nemotron-nano-30b/gemma-4-31b/glm-5) |
| **ADD** | 0.96->0.81 mirage beat | Add as an honesty beat on Slide 12 / the sibling (NOT Slide 4, which is calibration ECE): gemma-26B error-F1 collapses 0.957 (n=60 rasmachine) -> 0.867 (v2-balanced n=117) -> 0.811 [0.777,0.844] (external n=587) as gold grows and balances. Fits the deck's 'led with the losses' ethos. Name the model (gemma-26B = the deck's fitted reader) so it doesn't get confused with the 31b pick. | analyze_external_gold.py:5; errf1-mirage report |
| **FLAG** | Any slide quoting fleet n or 'beats nemotron'/'dominated' | Three precision fixes the reports caught: (1) say '32 curators total (anchor is 1 of 32, capped)', not 'anchor + 32'; (2) 'gemma beats nemotron robustly' holds ONLY vs nemotron-nano-30b (p=0.0013) — it is a dead tie vs nemotron-super-120b (0.811 vs 0.810, p~0.58); always name the variant; (3) glm-5 is Pareto-optimal, not dominated. | all four reports (gemma beats nemotron / glm-5 caveats); build_multicurator_gold.py:73,126 |

## Historical tensions (do not present as current)

- FRONTIER PICK NOT CALIBRATED: gemma-4-31b (the cost-winner, 0.827) is a DIFFERENT model from the calibration-fitted gemma-26B and calibration_for() returns None for it (explicit '31b' not in m guard) AND for all bedrock-* serving -> hard gate. The deck's 0.814/0.156 belief numbers are gemma-26B on eval_curation_v1. Promoting 31b to headline reader requires a fresh soft-weight fit first; the frontier pick and the calibration instrument are currently different readers on different substrates. Do not conflate.
- MEDPSY-4B FATE: NOT dropped from the calibration story — still 1 of 2 fitted readers on Slide 7. BUT medpsy-4B is ABSENT from the bedrock frontier fleet, and the bedrock 4B-class proxy (gemma-4-e2b) is dead last at error-F1 0.700, which PRESSURES rather than supports any 'a small 4B is a fine second reader' beat. Slide 7 stays; just don't lean on the fleet to argue for a small second reader.
- BELIEF CORE NOT RE-RUN ON NEW GOLD: no ECE/AUROC/Brier was computed on external_curator_gold_v1; the per-evidence belief field in the runs is the verdict-independent source-reliability scalar, not fitted soft belief. Slides 4/7/8/9 cannot claim validation on the new gold.
- N BOOKKEEPING: headline 'n=578' is gold rows but there are only 575 unique (matches_hash,source_hash) pairs (3 exact-dup rows) and F1 is actually over ~587 joined evidence pairs (469 statements); paired tests on n~572-573. De-dup balance is 287/288, not 289/289. Cosmetic for ranking but the 'n=578' label is nominal.
- DE-CONTAMINATION SCOPE: gold is provably 0-pair-overlap vs eval_curation_v1/_clean and the data/benchmark holdouts, but holdout_cc (in data/results/, lacks matches_hash) was NOT in the exclusion globs — scope the claim to 'vs eval_curation_v1 and the benchmark holdouts', not 'vs holdout_cc'. Practical risk negligible.
- STALE COST ARTIFACT: cost_report.py prints $1.10 (only 4 hardcoded session runs) — the real fleet spend is $14.56 from summing call_logs; session ~$14.92 INCLUDES the fleet (do NOT add to ~$29).
- 'EXPENSIVE REASONERS DOMINATED' OVERSTATED: glm-5 (0.839, $5.89) is the single highest F1 and Pareto-optimal — not dominated; honest framing is '+0.012 within CI for 8.7x cost'. Truly dominated = kimi/deepseek/qwen3-coder-480b/qwen3-235b/gpt-oss-120b.

## Historical reruns proposed at the time — now completed in part

- To fully re-base the calibration core onto the new gold: run scripts/statement_belief (and belief_headtohead) on external_curator_gold_v1 for the TWO fitted readers (gemma-26B, medpsy-4B) to produce belief ECE/AUROC/Brier on the de-biased n=578/587 set — this is the only way the deck's headline metrics could claim the new gold as a validation substrate. Currently only error-F1 exists.
- Run scripts/calibration_ship_gate.py treating external_curator_gold_v1 as a new OOD holdout (in addition to holdout_cc) for gemma-26B/medpsy-4B — gives Slide 12 a real 'validated on our own gold' OOD result instead of intent.
- IF gemma-4-31b is to become the production/headline reader: fit fresh soft weights {w_correct, w_rejected} for gemma-4-31b on a balanced set (it currently inherits nothing), then re-run the 4-leg ship gate — without this it stays hard-gate-only and cannot carry the belief-calibration story.
- Pre-existing Appendix-B TODO (independent of this arc): re-run calibration_ship_gate.py with the current 'clean' arm to replace the on-disk 'guard' arm OOD figures (ECE 0.171->0.129 / AUROC 0.740->0.768 are guard-era at tau=0.5).
- If 'gemma beats nemotron' is presented: add a recomputed paired McNemar/perm test naming the nemotron variant (the +/-0.027 / p=0.005 in analyze_external_gold.py are hardcoded comments, not recomputed in that script).

## Historical adversarial verify — corrections recorded at the time

- cost_report.py does NOT print $1.10 (plan tension #6 is stale): the current scripts/cost_report.py prints a forward fleet-estimate table ('SUM to run ALL others ... $40.99'), no $1.10 line and no 4 hardcoded session runs. The ~$14.6 real fleet spend is NOT produced by cost_report.py either; it comes from summing the $/run column of scripts/frontier_table.py (verified sum = $14.58). Fix the plan's description of the artifact; the $14.6 headline itself is fine.
- Plan flag #2 cites p=0.0013 for 'gemma beats nemotron-nano'; the actual source value in scripts/analyze_external_gold.py is a hardcoded comment p=0.005 (line ~ power-forecast block: 'paired Δ(gemma-nemotron) +/-0.027 ... p=0.005'), not 0.0013, and it is NOT recomputed by the script. Directional claim holds (robust vs nemotron-nano-30b 0.767; dead tie vs nemotron-super-120b 0.810 vs gemma 0.811), but use the real p and recompute before quoting.
- frontier_table.py's computed 'value champion' (F1/cost ratio) is gemma-4-e2b (0.700 @ $0.18/1k), NOT gemma-4-31b. The plan's reframe of gemma-4-31b as 'cheapest of the tied tier / production cost-winner' is defensible (e2b sits below the tied tier at 0.700), but the deck must not cite gemma-4-31b as the script's literal 'value champion'.
- Minor labeling: the deck/plan 'n=578' is gold ROWS; the F1 denominator is n=587 joined evidence pairs (frontier_table EV=587, confirmed by live n_gold=587) and the v2-balanced point is gold-file 114 rows but joins to n=117 (matches the plan's '117'). Keep 'n=578 balanced gold' and 'n=587 evidence pairs' distinct as the plan already advises.
- Nuance on the 0.811 'gemma-26B' mirage point: the fleet run is model_id 'bedrock-gemma' (bedrock serving), which calibration_for() resolves to None just like 31b. It is the same gemma family as the deck's fitted gemma-4-26B but a different serving substrate; naming it 'gemma-26B' in the mirage beat is approximately right but technically it is the uncalibrated bedrock serving, not the fitted reader.

*Historical confidence at the time: High. Every then-deck-bound number in the
assessment was independently reproduced from the artifacts available then. The
central “no external calibration export exists” conclusion was later superseded
by the configuration-specific reruns summarized in the current-correction block;
the cost/error-F1 cautions and overlapping-CI warning remain useful.*
