# Real examples for the deck — sampled from the n=578 gold, verified against the live INDRA DB API

*Every row is a real human-curated external_curator_gold_v1 record joined to the 13-model bedrock fleet; every belief was confirmed via POST db.indra.bio/statements/from_hashes (HTTP 200). Sentences are verbatim.*

> [!IMPORTANT]
> **CURRENT INTERPRETATION — 2026-07-13.** The stored beliefs in these example
> records are INDRA's parametric/source-reputation values used to illustrate what
> the count cannot see; they are not the final fitted hybrid score. Likewise, the
> 13-model verdicts and frontier F1 are hard-verdict error-detection evidence, not
> belief-calibration metrics.
>
> The current calibration is configuration-specific: remote/Ollama Gemma is fit
> from **704/157/97/646**, while reasoning-first Bedrock Gemma is fit separately
> from **662/81/139/722**; each matrix totals **1,604 unique pairs** after
> deduplication. The Bedrock profile never inherits the remote profile. On the
> external gold, the Bedrock hybrid score reaches **ECE .061 / AUROC .814** from
> raw count **.129 / .688**; formal hard→hybrid is **.237→.061 /
> .794→.814**, error-F1 Δ **−.001 [−.015,+.014]**, 4/4. This Bedrock run is the
> only formal external validation. The external MedPsy run used monolithic prompt
> SHA `07377e338ff2`, while its fitted/eval profile used `b44638216740`; it is an
> unmatched-profile stress test with no valid fitted soft metric or gate/pass.
> Its valid matched `holdout_cc` test (**n=414**) worsens ECE **.129→.146**, so
> it is 3/4 and production-disabled. Remote/Ollama Gemma's matched holdout
> (**n=414**) is raw→hybrid **.430→.148 ECE / .610→.745 AUROC**; formal
> hard→hybrid is **.157→.148 / .734→.745**, Δ **+.016 [.003,.032]**, 4/4.
> Reader verdict LRs are
> Bayesian evidence components, but the final confirmation source-floor
> composition is a hybrid log-odds score, not a pure posterior.
>
> The example rows and verbatim sentences remain valid. Historical slide numbers
> and any “every Bedrock model is hard-gate-only” banner do not.

## Slide 1 — cold-open reassuring edge  — replace_schematic

**Example:**
CHOSEN — ATM phosphorylates H2AX (the reassuring real edge that replaces the schematic MEK1→ERK2 0.91 chip).
• Canonical statement: "ATM phosphorylates H2AX" (Phosphorylation; enz ATM UP:Q13315/HGNC:795, sub H2AX UP:P16104/HGNC:4739).
• VERBATIM sentence: "An early event in the DDR is the phosphorylation of H2AX by ATM."
• source_api: reach.
• parametric belief: 0.906 (INDRA stored 0.9059981; total evidence in DB = 1,777 across 6 source_apis: reach 853 / sparser 695 / rlimsp 184 / trips 6 / pc 1 / isi 1; the gold carries 2 evidence rows, reach + sparser, both gold=correct).
• gold: correct (external_curator_gold_v1.jsonl line 566, n_curations=2, curator klas_karis@hms.harvard.edu).
• model verdicts: 13/13 CORRECT, every model score 0.95 / confidence high — genuinely unanimous (gemma-4-31b, glm-5, kimi-k2.5, qwen3-235b, deepseek-v3.2, nemotron-super-120b, gpt-oss-120b, minimax-m2.5, qwen3-coder-480b, nemotron-nano-30b, gpt-oss-20b, gemma-4-e2b, gemma).
• INDRA matches_hash: -15631331293963241 (cite for the live db.indra.bio/statements/from_hashes lookup).

**Backup:** BACKUP — PINK1 phosphorylates MFN2 (Phosphorylation; enz PINK1, sub MFN2). VERBATIM sentence: "In other cell types, previous studies have reported that PINK1 mediates the phosphorylation of Mfn2, which is an important step in Parkin mediated ubiquitination of Mfn2 [XREF_BIBR, XREF_BIBR, XREF_BIBR, XREF_BIBR]." source_api reach; parametric belief 0.929 (INDRA stored 0.92867506; 302 total evidence, sparser 162 / reach 138 — genuinely multi-source, two readers); gold=correct (curator thomaslim6793@gmail.com); 13/13 unanimous CORRECT @0.95/high; INDRA matches_hash -32394471335521317. Higher belief and a cleaner two-reader story, but the sentence carries XREF_BIBR noise and a "previous studies have reported" hedge, so it is a less crisp textbook one-liner than ATM→H2AX.

**Slot-in text:**

ON SCREEN (chip: SEES lit):
- ATM phosphorylates H2AX — belief 0.906
- Two independent readers — reach + sparser — they agree
- 13 of 13 models we ran later: correct
- — so: does the sentence actually say it?

VISUAL: One INDRA edge as a graph — [ATM] —phosphorylates→ [H2AX] — a confident green belief chip "0.906", two lit source badges (reach, sparser). A thin rule splits the reassuring top from the raw sentence below — "An early event in the DDR is the phosphorylation of H2AX by ATM." — a magnifying glass and a single large "?" hovering over it: no one ever read the sentence. (Real scored edge, INDRA matches_hash -15631331293963241 — not a schematic.)

SAY: "Pull up a real edge in our assembly — ATM phosphorylates H2AX, sitting at belief 0.906. Two independent readers, reach and sparser, extracted it; they agree; our score says trust this — and thirteen of thirteen models we ran later all call it correct. No argument there. I want to ask a different question about the *same* edge, one I don't think the pipeline ever asks out loud: the sentence those readers were standing on — does it actually say this? **[WAKE-UP]** *Pivot mid-sentence from the confident top to the sentence below:* 'Two readers agree. So — does the sentence actually say that?' And here — read it — it does: 'the phosphorylation of H2AX by ATM.' Hold that question; the whole talk lives in the gap between 'the readers are reliable' and 'the sentence supports the claim.'"

**Caveat:** Belief 0.906 honestly grounds the slide's "≈0.9" framing because it is THIS edge's real INDRA stored belief over 1,777 multi-source evidences — the F4 caveat (a bare reach+sparser n=1 pair drops to ~0.71 under production RECALIBRATED_PRIORS) does NOT bite here, since the high belief comes from heavy multi-source aggregation, not a 2-reader pair. Keep the two GOLD readers framed as reach + sparser (the DB has 6 source_apis, but the curated rows are reach + sparser). Do NOT say "10 evidence" — 10 is only the API page sample; the true total is 1,777. The original schematic's belief 0.91 is preserved to the same rounding (0.906).

---

## Slide 4 — overconfidence (high parametric belief, gold=incorrect)  — add

**Example:**
CHOSEN — Ubiquitination of SREBF1 (the overconfidence headliner: max belief among gold=incorrect rows AND maximal automated agreement with the wrong belief).
• Canonical statement: Ubiquitination of SREBF1 — SUBSTRATE-ONLY, no enzyme/object (sub=SREBF1; db_refs UP:P36956, TEXT:SREBP-1, HGNC:11289, EGID:6720; type=Ubiquitination).
• VERBATIM sentence: "It has been reported that the ubiquitination of SREBP1 mediated by E3 ubiquitin ligase is a crucial step in activating SREBP1 [ 30 ]."
• source_api: sparser.
• parametric belief: 0.9887 (INDRA stored 0.988736; total_evidence 101 — sparser 99, reach 1, trips 1; the value is IDENTICAL across all 13 fleet run files, spread=0 — confirming it is the parametric INDRA source-reliability scalar, not a fitted soft belief).
• gold: incorrect (curator mock7ee@gmail.com, tag=grounding).
• model verdicts: 11 correct / 2 incorrect. CORRECT @0.95/high (every expensive model): gemma-4-31b, glm-5, kimi-k2.5, deepseek-v3.2, qwen3-235b, qwen3-coder-480b, nemotron-super-120b, minimax-m2.5, gpt-oss-120b, gpt-oss-20b, nemotron-nano-30b. INCORRECT: gemma-4-e2b (med/0.2), gemma (high/0.05). Gilda grounding all_match.
• INDRA matches_hash: -34342438092911441.

**Backup:** BACKUP — Inhibition(serotonin → HTR2A), "Serotonin inhibits HTR2A" (subj serotonin CHEBI:28790, obj HTR2A HGNC:5293/UP:P28223). VERBATIM sentence: "It is an antagonist of presynaptic alpha 2-adrenergic autoreceptors and heteroreceptors on both norepinephrine and serotonin (5-HT) presynaptic axons, plus is a potent antagonist of postsynaptic 5-HT2 and 5-HT3 receptors." source_api reach; parametric belief 0.9660 (INDRA stored 0.9660092; total_evidence 25; 2nd-highest of the gold=incorrect rows; identical across all 13 runs); gold=incorrect (the unnamed drug "It", not serotonin, is the HTR2A antagonist — wrong subject = entity boundary); model verdicts 0 correct / 13 incorrect @0.05/high; INDRA matches_hash 29908095842136599. Use this when the speaker wants the version where the PARAMETRIC belief is the lone overconfident signal (every model AND the human caught it) rather than the version where the models are also fooled.

**Slot-in text:**

ON SCREEN (chip: BLIND lit) — add one callout line beneath the metrics:
- e.g. SREBF1 ubiquitination — belief 0.989, curator says WRONG; 11 of 13 models say correct

VISUAL — keep the reliability/calibration plate and the two dials; add a small real-edge callout box anchored to the high-belief bin: "SREBF1 ubiquitination · belief 0.989 (highest among curator-incorrect) · 11/13 models correct @0.95 · grounding all_match · gold = INCORRECT · matches_hash -34342438092911441."

SAY — append after the existing calibration reveal: "Concretely — the single highest-belief edge a curator marked wrong in our gold is the ubiquitination of SREBF1, at belief 0.989. The sentence: 'It has been reported that the ubiquitination of SREBP1 mediated by E3 ubiquitin ligase is a crucial step in activating SREBP1.' It's a background hedge — 'it has been reported' — and the extraction is substrate-only, no enzyme actually bound. Our count belief is 0.989, near-certain; and eleven of thirteen models, every expensive one, agreed at 0.95, high confidence; Gilda grounding matched. The reputation is maxed and nearly everything automated nodded along — and the human said wrong. That's the calibration collapse in one edge."

**Caveat:** 0.989 is the MAX belief among gold=incorrect rows, NOT the entire gold set (a gold=correct row reaches 0.9979787) — phrase it as "highest among curator-incorrect," never "highest belief in the gold set." The gold=incorrect count is 289, not 286. Belief 0.988736 is the parametric source-reliability scalar (identical across all 13 runs, spread=0), NOT fitted soft belief — fully consistent with the slide's GUARD line. Do NOT say "10 evidence" — true total is 101 (10 is the API display cap).

---

## Slide 5 — same belief, opposite truth (the matched PAIR; headline use)  — replace_schematic

**Example:**
CHOSEN — PAIR 1, two real Phosphorylation rows at near-identical belief (0.7951 vs 0.7947), one faithful, one a relation reversal. Replaces the synthetic "A phosphorylates B" template.

CORRECT side (left card):
• Canonical statement: "DYRK1A phosphorylates MEF2D on S251" (Phosphorylation; enz DYRK1A UP:Q13627/HGNC:3091, sub MEF2D UP:Q14814/HGNC:6997, residue S251).
• VERBATIM sentence: "Here, we uncovered that dual-specificity tyrosine phosphorylation regulated kinase 1A (DYRK1A), a kinase critical in Down 's syndrome pathogenesis, directly bound to and phosphorylated MEF2D at Ser251 in vitro."
• source_api: reach. parametric belief: 0.7950922 (INDRA stored, identical across all 13 fleet rows; 5 evidences / 4 source APIs). gold: correct (curator marta.iannuccelli@gmail.com, tag=correct). model verdicts: 13/13 CORRECT @0.95/high — unanimous. INDRA matches_hash: -6401338401063282.

INCORRECT side (right card):
• Canonical statement: Phosphorylation enz=STAT1 (HGNC:11362, UP:P42224) → sub=IFNG (HGNC:5438, UP:P01579), renders "STAT1 phosphorylates IFNG" — relation reversed: the sentence makes STAT1 the phospho-substrate and IFN-gamma the stimulus.
• VERBATIM sentence: "Here we show that, after IFN-gamma stimulation, SOCS1 inhibited IFN-gamma receptor and STAT1 phosphorylation but maintained ERK1/2 activation."
• source_api: sparser. parametric belief: 0.79469573 (INDRA stored, identical across all 13 fleet rows; 47 stored evidences). gold: incorrect (curator anthonywu92@gmail.com, tag=polarity). model verdicts: 12/13 INCORRECT @0.05/high; only gpt-oss-20b fooled (correct, 0.95). INDRA matches_hash: -21956960619464458.

**Backup:** NO second fully-verified matched pair exists in the verified set — only PAIR 1 (both rows) was confirmed against the live INDRA API. Do not improvise a backup pair: substituting a higher-belief positive (e.g. PINK1→MFN2 at 0.929) for the correct side would break the "same belief" match (0.929 vs 0.795). If a backup is required, re-mine a second same-stmt_type/same-belief pair before the talk; treat PAIR 1 as the sole verified Slide-5 anchor.

**Slot-in text:**

ON SCREEN (chip: BLIND lit):
- Two Phosphorylation edges · belief 0.795 vs 0.795 · same readers, same count
- Left ✓ DYRK1A phosphorylates MEF2D — the sentence says exactly that
- Right ✗ STAT1 phosphorylates IFNG — the reader flipped it; IFN-gamma is the stimulus, STAT1 the substrate
- The meter can't tell them apart — 0.7951 vs 0.7947

VISUAL: Split-screen of two real evidence cards, belief meters near-identical (0.795 vs 0.795) across the top — that sameness is the punch. Left (DYRK1A→MEF2D, reach, gold ✓): "Here, we uncovered that... DYRK1A... directly bound to and phosphorylated MEF2D at Ser251 in vitro." Right (STAT1→IFNG, sparser, gold ✗): "Here we show that, after IFN-gamma stimulation, SOCS1 inhibited IFN-gamma receptor and STAT1 phosphorylation but maintained ERK1/2 activation." — extraction reads "STAT1 phosphorylates IFNG" but the sentence makes STAT1 the substrate. Labeled "real curated rows — matches_hash -6401338401063282 (left) / -21956960619464458 (right)," not a schematic.

SAY: "Concretely: here are two real edges our pipeline scores essentially identically — belief 0.795 against 0.795, same kind of source, same count, same meter. On the left, DYRK1A phosphorylates MEF2D — the sentence says it outright: the kinase 'directly bound to and phosphorylated MEF2D at Ser251 in vitro.' Thirteen of thirteen models agree it's correct. On the right, the extraction says STAT1 phosphorylates IFNG — but the sentence is 'after IFN-gamma stimulation... STAT1 phosphorylation' — IFN-gamma is the stimulus, STAT1 is the substrate; the reader inverted the relation. A curator tagged it polarity, incorrect. The reputation is intact on both; the read is wrong on one. **[WAKE-UP]** Identical belief, opposite truth — and the only difference lives in a sentence our formula never opened. This is what 'overconfident' looks like one edge at a time, and it's exactly where a second reader looks."

**Caveat:** The slide visual says the belief meters are "pixel-identical and maxed" — the real pair sits at ~0.795, NOT maxed; change "maxed" to "near-identical at 0.795" (the beliefs differ only at the 3rd decimal: 0.7951 vs 0.7947, so the same-belief axis holds tightly). Note 12/13 models DID catch the reversed STAT1→IFNG row (only gpt-oss-20b was fooled) — so this pair demonstrates the *parametric meter* being unable to tell them apart, which is precisely Slide 5's claim; keep the framing on the count belief, not on the LLM fleet, or the WAKE-UP weakens.

---

## Slide 11 — the review tier  — add

**Example:**
CHOSEN — INS activates transport (the review-tier RANK 1: clean grounding, solid belief, models split ~50/50, gold=correct → auto-drop would discard a curator-confirmed-correct statement).
• Canonical statement: INS (insulin) Activation of transport (subj INS UP:P01308/HGNC:6081/EGID:3630; obj transport GO:0006810), renders "INS activates transport."
• VERBATIM sentence: "This knowledge represents an essential building block for fully understanding the processes that control this key insulin signaling protein that is a crucial regulator of insulin stimulated glucose transport."
• source_api: reach.
• parametric belief: 0.852287 (INDRA stored 0.85228676, constant across all 13 model rows; total_evidence 4080 — reach 4023, sparser 57).
• gold: correct (external_curator_gold_v1.jsonl line 428, curator klas_karis@hms.harvard.edu).
• model verdicts: 7 correct / 6 incorrect. correct: glm-5(0.95/high), gemma-4-31b(0.95), kimi-k2.5(0.95), deepseek-v3.2(0.95), gpt-oss-120b(0.95), gemma(0.95), gpt-oss-20b(0.95). incorrect: qwen3-coder-480b(0.05/high), qwen3-235b(0.05/high), minimax-m2.5(0.05/high), nemotron-nano-30b(0.05/high), nemotron-super-120b(0.2/med), gemma-4-e2b(0.2/med). grounding_status = all_match for ALL 13 (NOT a deterministic grounding reject).
• INDRA matches_hash: -15491678236227482.

**Backup:** BACKUP — CCNE1 phosphorylates RB1 (RANK 2; the hedge-on-the-relationship variant, gold=incorrect). Canonical: Phosphorylation(enz=CCNE1 HGNC:1589/UP:P24864, sub=RB1 HGNC:9884/UP:P06400), "CCNE1 phosphorylates RB1." VERBATIM sentence: "This suggests that the increase of cyclin A2 after phase G1, that leads to the replacement of cyclin E1 by cyclin A2 in complexes formed of CDK2, might be triggered by the phosphorylation of RB on a third site by the complex cyclin E1-CDK2." source_api sparser; parametric belief 0.834676 (INDRA stored 0.8346759; total_evidence 17 — reach 10 + sparser 6); gold=incorrect; model verdicts 6 correct / 7 incorrect (all 13 grounding all_match); INDRA matches_hash 5249437691921303. The "This suggests... might be triggered by" hedge admits two readings (established mechanism vs inside-the-hypothesis) — use when the speaker wants the review-then-reject direction rather than review-then-keep.

**Slot-in text:**

ON SCREEN (chip: SEES lit) — add a callout beneath the funnel lanes:
- e.g. INS activates transport · belief 0.852 · 7 models correct / 6 incorrect · grounding clean · curator: CORRECT → review, don't drop

VISUAL — keep the three-lane funnel; on the glowing center "review" lane, drop a real card: "INS activates transport · belief 0.852 · grounding all_match (all 13) · models split 7/6 · gold=CORRECT · matches_hash -15491678236227482" — visibly NOT a clean reject, flowing to the curator-at-a-desk.

SAY — append into the existing "review is the deliverable" beat: "Concretely — take 'INS activates transport' at belief 0.852, clean grounding on both agents. The relation IS asserted, but it's buried in a background significance sentence: 'a crucial regulator of insulin stimulated glucose transport.' Our strong readers genuinely split — seven say correct, six say incorrect; glm-5, kimi, deepseek on one side, qwen, minimax, nemotron on the other. Nothing here is a clean reject; grounding matches for all thirteen. Auto-drop it and you've thrown away a statement a curator confirmed correct. **[WAKE-UP]** That's the review lane — not a hedge, the queue a human actually opens."

**Caveat:** Belief 0.852 is a hair above the slide's stated 0.70–0.85 band — change the band to "≈0.70–0.85" or just cite 0.852. Do NOT say "10 evidence each" — this statement has 4080 evidences (the 10 is the API page size); fix the slide's "10 evidence each" framing. This row is gold=CORRECT, so it makes the "don't auto-drop a correct statement" point; if the speaker wants a review-then-reject illustration instead, use the backup. Grounding all_match across all 13 is load-bearing — it proves this is a sentence-level judgement call, not a deterministic grounding reject.

---

## Sibling S13 — which reader, at what cost (model disagreement)  — add

**Example:**
CHOSEN — TNF phosphorylates INSR (the cheap-right / all-expensive-wrong headliner: gemma-4-31b, the production cost-winner, is the SOLE correct model of 13).
• Canonical statement: "TNF phosphorylates INSR" (Phosphorylation; enz TNF HGNC:11892/UP:P01375, sub INSR HGNC:6091/UP:P06213).
• VERBATIM sentence: "Furthermore, TNF-α induced serine phosphorylation of IR and IRS-1, and these effects were completely precluded by pretreatment with inhibitors of p38 MAPK."
• source_api: sparser.
• parametric belief: 0.8080459 (INDRA stored, matches the fleet rows' belief field exactly; API returned 10 evidences — sparser 4, reach 6).
• gold: incorrect (the sentence shows a p38-MAPK-mediated phosphorylation, not a direct TNF→INSR kinase event).
• model verdicts: gemma-4-31b = INCORRECT (RIGHT — the ONLY correct model of 13, score 0.05/high). WRONG (said correct @0.95/high, including all 4 named-expensive): glm-5, kimi-k2.5, qwen3-coder-480b, qwen3-235b, plus deepseek-v3.2, gemma, gemma-4-e2b, nemotron-super-120b, nemotron-nano-30b, gpt-oss-20b, gpt-oss-120b, minimax-m2.5.
• INDRA matches_hash: -16536742498079423.

**Backup:** BACKUP — PTPN1 inhibits MET (Inhibition; PTPN1/PTP1B UP:P18031 inhibits MET UP:P08581, obj_activity=activity). VERBATIM sentence: "It has been reported that the protein tyrosine phosphatase PTP1B could inactivate MET by direct dephosphorylation of Tyr 1234 and 1235 in its activation loop, and that this dephosphorylation takes place in peri-nuclear region of the cell [ xref ]." source_api sparser; parametric belief 0.7000229 (INDRA stored, matches fleet rows; total_evidence 15 — signor 7, reach 4, sparser 2); gold=correct (curator livia.perfetto@gmail.com). RIGHT: gemma-4-31b, gemma-4-e2b, gemma, qwen3-coder-480b, nemotron-nano-30b, gpt-oss-20b, minimax-m2.5. WRONG: glm-5, kimi-k2.5, qwen3-235b, deepseek-v3.2, nemotron-super-120b, gpt-oss-120b. INDRA matches_hash 16734535754802605. CAVEAT on backup: cost is non-monotonic here — the largest model qwen3-coder-480b is RIGHT while qwen3-235b is wrong — so use this row to make the "size doesn't buy error-detection" point rather than a strict "cheap beats expensive" point.

**Slot-in text:**

ON SCREEN (no chip — sibling) — add a callout under the frontier scatter:
- e.g. TNF phosphorylates INSR · belief 0.808 · curator says WRONG · 1 of 13 models caught it — gemma-4-31b, the cost-winner

VISUAL — keep the cost×error-F1 scatter; annotate the gemma-4-31b point with a real-edge flag: "TNF→INSR (matches_hash -16536742498079423): only model of 13 correct here; all 4 named-expensive wrong @0.95."

SAY — append after the cost-winner discussion, before the hard-gate red banner: "One edge makes the point. 'TNF phosphorylates INSR,' belief 0.808 — the sentence is 'TNF-α induced serine phosphorylation of IR... completely precluded by inhibitors of p38 MAPK,' so it's a p38-mediated event, not a direct TNF kinase action; the curator flagged it incorrect. Of all thirteen models, exactly one caught it — gemma-4-31b, our cost-winner. glm-5, kimi, the 480b coder, the 235b — every named expensive reader said correct at 0.95, high confidence. The cheapest reader on the tied tier was the only one that didn't get fooled. That's 'cost doesn't buy error-detection' in a single row."

**Caveat:** This is ONE vivid row, an illustration — not the aggregate claim. The slide's ranking rests on per-model F1 (glm-5 0.839 Pareto-optimal, gemma-4-31b 0.827 cheapest-of-tie); do NOT let the anecdote be heard as "31b beats glm-5 overall." Keep the sibling's separation banner: **every score here is hard-verdict F1; none of these numbers is a belief-calibration result.** The 31b cost-winner is not one of the three currently fitted configurations, while reasoning-first Bedrock Gemma 26B has its own separate profile. The candidate metadata's "20 across 4 readers" was an overstatement — the API returned 10 (sparser 4, reach 6); immaterial, but cite 0.808 belief, not an evidence count.

---

## Unfilled

- Sibling S14 — the 0.957→0.811 mirage: NO firm single real example. This slide is inherently a metric TRAJECTORY (error-detection F1: 0.957 [0.897,1.000] rasmachine n=60 → 0.867 v2-balanced n=117 → 0.811 [0.777,0.844] external balanced n=587), not a single statement. The verified set delivered the Hunt-A model-disagreement examples but NOT a Hunt-B 'trivially-easy gold=correct positive the whole 13-model fleet agreed on' drawn from the rasmachine n=60 set. The unanimous 13/13 positives that WERE verified (ATM→H2AX, PINK1→MFN2) come from external_curator_gold_v1, not rasmachine n=60, so presenting them as 'the kind of easy positive that inflated the 0.957' would be a category error. Recommendation: leave S14 as the three-stepping-stone trajectory it already is; if a concrete mirage positive is wanted, mine a unanimous-correct rasmachine-n=60 row and verify its belief/verdicts before slotting.

## Historical slotting notes — examples remain valid, slide numbering may not

All five filled slides use ONLY verified rows from external_curator_gold_v1 joined to the 13-model bedrock fleet, with beliefs confirmed against the live db.indra.bio API; every sentence is verbatim from the verified set, no embellishment. Deck source: /Users/noot/Documents/indra-belief-model/research/ben_gyori_deck_outline.md (Slide 1 lines 12-26, Slide 4 lines 63-76, Slide 5 lines 80-93, Slide 11 lines 184-197, S13 lines 224-238, S14 lines 242-256).\n\nRecommendation rationale: Slides 1 and 5 are explicitly self-labeled schematic/illustrative in the outline (E1 line 25; E2 line 93), so 'replace_schematic' swaps the synthetic edge/card for a real scored one. Slides 4, 11, and S13 carry statistical/funnel/scatter visuals with NO concrete example today, so 'add' drops a real anchor into the Say + visual without disturbing the existing numbers.\n\nNumbers preserved: Slide 1 keeps belief ≈0.9 (real 0.906). Slide 4 keeps the AUROC 0.71–0.74 / ECE 0.37 framing and the GUARD line; the SREBF1 belief 0.989 is the parametric source-reliability scalar (spread=0 across runs), consistent with the GUARD. Slide 5's 'same belief' axis holds tightly (0.7951 vs 0.7947). S13 keeps the per-model F1 ranking; the TNF→INSR row is an anecdote, not a re-ranking.\n\nTwo numeric fixes the real examples force (flag to deck owner): Slide 5 visual 'maxed' meters → 'near-identical at 0.795'; Slide 11 'belief 0.70–0.85' band is grazed by 0.852 and the '10 evidence each' phrasing is wrong (INS→transport has 4080 evidences; 10 is the API page size — same paging artifact already corrected in the outline's own Slide 11 source note).\n\nCarry-forward risk: Slide 4's SREBF1 (11/13 models fooled) and Slide 5's STAT1→IFNG (only 1/13 fooled) tell OPPOSITE model-behaviour stories — both are correct for their slides (Slide 4 = parametric belief + models overconfident; Slide 5 = parametric meter can't separate while the LLM fleet mostly can). Keep each slide's framing on the COUNT belief, not the fleet, so the two don't read as contradictory."
