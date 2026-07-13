# Reading the Evidence — deck research handoff
### historical working title: *The Second Reader*

> [!IMPORTANT]
> **CURRENT DECK CONTRACT — 2026-07-13.** This block supersedes the 12-slide
> storyboard and pre-rerun calibration claims preserved below.
>
> - A fitted reader **configuration** rebuilds statement belief as a hybrid
>   log-odds score. Verdicts contribute confusion-derived log-likelihood ratios;
>   repeated reads within one source are averaged and independent sources add.
>   A confirmation keeps the stronger of its reader log-LR and the separately
>   fitted source-reliability logit. Because that source floor is not itself a
>   likelihood ratio, the final scalar is **not a pure Bayesian posterior**.
> - Profiles are exact-configuration instruments and never inherit by model-family
>   resemblance. Remote/Ollama Gemma is fit on `eval_curation_v1_gemma.jsonl`
>   with confusion **704/157/97/646**. Reasoning-first Bedrock Gemma is fit
>   separately on `eval_curation_v1_gemma_rf_bedrock.jsonl` with confusion
>   **662/81/139/722**. Each fit uses **1,604 unique evidence pairs** after
>   duplicate-pair removal (801 correct / 803 incorrect).
> - Remote/Ollama Gemma on `holdout_cc` (**n=414**): raw count→hybrid
>   **ECE .430→.148 / AUROC .610→.745**; formal hard→hybrid
>   **.157→.148 / .734→.745**, error-F1 Δ **+.016 [.003,.032]**, 4/4.
> - Reasoning-first Bedrock Gemma on the independent external gold (464 scored
>   statements): raw count→hybrid **AUROC .688→.814 / ECE .129→.061**;
>   formal hard→hybrid **ECE .237→.061 / AUROC .794→.814**, error-F1 Δ **−.001**
>   CI **[−.015,+.014]**, 4/4.
> - MedPsy's matched `holdout_cc` test (**n=414**) worsens ECE
>   **.129→.146**, so it passes 3/4 and production is disabled. Its external
>   artifact is **not** a formal calibration gate: the monolithic prompt SHA
>   prefix `07377e338ff2` does not match the fitted/eval
>   MedPsy profile's `b44638216740`. It is an unmatched-profile stress test; a
>   fitted soft result and gate/pass determination are invalid/unavailable.
> - The remote and reasoning-first Bedrock Gemma results use separate frozen
>   profiles on separate serving configurations—not one pair of Gemma numbers
>   reused on two golds. Formal external validation is Bedrock only. The 13-model frontier remains
>   a sibling **hard-verdict error-F1** analysis; it does not stand in for belief
>   calibration or imply that every Bedrock configuration is unfitted.

**Current throughline:** INDRA's count belief reads the *masthead*—who extracted
the edge and how reliable those sources are. The fitted path reads the sentence
and rebuilds the statement number from configuration-matched verdict evidence,
while retaining source reliability as a conservative confirmation floor. The
deliverable is a calibrated scalar plus a review route, always accompanied by
its evidence tally.

**Audience:** Our group — the Gyori lab, INDRA / EMMAA authors. Co-owners in the room; peer-to-peer. **Current artifact:** 19 slides, including the clearly separated reader-choice/error-F1 sibling.

---

## Historical 12-slide storyboard — superseded, retained for design provenance

> [!WARNING]
> Everything below this heading records the earlier gate/survival-weight deck
> and its then-current evidence ledger. It remains useful for examples, visual
> reasoning, and decision history, but its slide numbering, `n=1606` fit claims,
> Bedrock fallback claims, guard-era metrics, and pure-Bayes language must not be
> quoted as the current deck contract. Use the correction block above and the
> rendered `presentations/gyori-belief/slides.md` instead.

*Historical navigation device:* a persistent `SEES · BLIND · TRUST` chip was
specified for the 12-slide storyboard so anyone tuning back in could re-anchor.

### Slide 1 — 0.9, two readers agree — does the sentence actually say that?
**Big idea:** Land on an edge we fully trust, then ask the orthogonal question the pipeline never voices out loud — not "are the readers reliable?" but "does the underlying sentence support the claim?"

**On screen** *(chip: SEES lit)*
- Any edge in our DB at belief ≈ 0.9
- Two independent readers — they agree
- Reliable · confirmed · trusted
- — so: does the sentence actually say it?

**Visual:** One INDRA edge as a graph — [MEK1] —phosphorylates→ [ERK2] — a confident green belief chip "0.91", two lit source badges (reach, sparser). A thin rule splits the reassuring top from a grayed/blurred raw sentence below, a magnifying glass and a single large "?" hovering over it: no one ever read the sentence.

**Say:** "Pull up any edge in our assembly sitting at belief 0.9 — two independent readers extracted it, they agree, our score says trust this. No argument there. I want to ask a different question about the *same* edge, one I don't think the pipeline ever asks out loud: the sentence those readers were standing on — does it actually say this? **[WAKE-UP]** *Pivot mid-sentence from the confident top to the blurred bottom:* 'Two readers agree. So — does the sentence actually say that?' Hold that question; the whole talk lives in the gap between 'the readers are reliable' and 'the sentence supports the claim.'"

**Sources:** Edge/chip is a *schematic illustration*, not a scored screenshot — ben_gyori_deck_outline.md:21 (visual spec); real MEK→MAPK1 Phosphorylation edges sit at belief ~1.0, never 0.91 (E1). The two-reader "≈0.9" holds under upstream INDRA default priors only (compute = 0.8775 for reach+sparser at n=1); the production `statement_belief` path defaults to RECALIBRATED_PRIORS, which gives ~0.71 for the same pair (statement_belief.py:145; F4).

---

### Slide 2 — A second reader, standing beside our score
**Big idea:** Plain collegial setup — this is additive, an orthogonal axis. Our belief answers source reliability with rare precision; we add a second axis that reads the sentence. Seed the held gap as *calibration*, not discrimination.

**On screen** *(chip: SEES lit)*
- Not editing our axis — adding one beside it
- Our score asks: *are the readers reliable?*
- We add: *does the sentence support the claim?*
- (held for later: it ranks fine — but it's sure of everything)

**Visual:** Two parallel lanes joined by a "+" (never a replacing arrow). Left lane "THE MASTHEAD" — a newspaper masthead of readers/DBs, green check, "who pulled it, how many agree." Right lane "THE SENTENCE" — faint silhouette of a second reader leaning over a line of text with a "?", "does it actually say this?" Two columns that coexist, never one on top of the other.

**Say:** "Our belief score nails source reliability — it reads the masthead of who pulled an edge and how many independently agree, with a precision I haven't seen elsewhere. **[WAKE-UP]** Here's the orthogonal axis I've been working on: a second reader that checks the sentence. It sits *beside* the score, not on top. And I'll hold one thing over the room — our score ranks good extractions from bad just fine, but it's badly overconfident about any one sentence. A second reader fixes that. The frame, for the rest of the talk, is one noun: a second reader."

**Sources:** "ranks fine but sure of everything" seeds the slide-4 figures — text-miner belief at statement grain ranks correctness AUROC 0.714–0.741 yet ECE ~0.37 (reports/belief_headtohead_gemma.md:16,19,22; B1/B2).

---

### Slide 3 — Our formula stays — and the two questions inside every edge
**Big idea:** Honor the instrument, anchor "this stays" exactly once, and name the second axis — without re-deriving our own math. One edge carries two orthogonal questions; counting reliable witnesses answers the first and never reaches the second.

**On screen** *(chip: BLIND lit)*
- `belief = 1 − ∏ₛ (systₛ + randₛⁿ)` — **this stays**
- 1. Who says so? → our belief owns this
- 2. Does the sentence actually say it? → a different axis
- one edge, two claims

**Visual:** The formula in a boxed frame with a wax seal "this stays" — pointed at, not narrated. The edge A→B forks into two bubbles: left "Who says so?" roped to the masthead (settled color); right "Does this sentence support it?" on a perpendicular axis, anchored to a magnifying glass over a raw sentence (open, unanswered color).

**Say:** "This is ours. It answers a genuinely hard question — how much do we trust the witnesses, and how many agree — and we don't touch a term of it. *(Point, don't re-derive — we wrote it.)* But slow down on one high-belief edge: there are two questions tangled inside it. One, how reliable are these readers — our belief answers that cleanly. Two, does this exact sentence say what we extracted — a different question, on a different axis. **[WAKE-UP]** Two claims are stacked in every edge — and the count has only ever been able to see one of them. Next slide is simply what happens when you measure the second axis directly."

**Sources:** Formula = INDRA's additive SimpleScorer form, verbatim — noise_model.py:5-6 (`P(incorrect)=∏ₛ[systₛ + ∏ⱼ randₛⱼ]`), :108 (`p_incorrect *= syst + rand**n`), :110 (clamp `1 − p_incorrect`) (F1).

---

### Slide 4 — It ranks the sentence at ~0.74 — then trusts a misread as much as a read
**Big idea:** The central reveal, told straight. Point the reliability instrument at "is THIS extraction faithful to its sentence?" and on balanced human-curated gold it *ranks* correctness at AUROC ~0.71–0.74 — real signal — but it's badly **miscalibrated** (ECE ~0.37). It's just as sure when it's wrong, because reputation barely moves on one sentence.

**On screen** *(chip: BLIND lit)*
- Same edge. New question: is THIS sentence correct?
- Ranks it: AUROC ~0.71–0.74 — real signal
- Trusts it: **ECE ~0.37** — sure of everything
- The gap isn't ranking. It's overconfidence.

**Visual:** A reliability (calibration) plate — predicted-belief vs actual-correct — with the count-belief curve bowed off the diagonal. ⚠ *The "overconfident high-belief bin sitting at ~50% actual correct" annotation is illustrative — no data artifact pins a per-bin value for the count belief (only scalar AUROC/ECE are exported); draw the bowed curve qualitatively, do not label a specific bin percentage.* A second small panel: two dials — left "How reliable are these sources?" pinned sharp; right "Does this sentence support the claim?" reading confident-but-wrong. *(No 45° ROC line — at 0.74 that would be a fabricated figure.)*

**Say:** "Here's the turn. We took our belief and asked it the question it wasn't built for — not 'are these sources reliable,' but 'did THIS sentence say what we extracted.' It *ranks* correct from incorrect at about 0.74 — genuine signal, I won't undersell it. But its calibration error is around 0.37: it backs a misread with nearly the same confidence as a true read. **[WAKE-UP]** A top-tier reader earns a near-perfect reputation — and a reputation is not a reading of *this* sentence. So the score is sure of everything. That's the axis our count was blind to: not whether it can rank, but whether it should be trusted on a given sentence. *(Aside for us nerds: under the strictest any-incorrect gold rule its ranking degrades out-of-distribution — AUROC 0.717 in-distribution, n=862, drops to 0.571 on the holdout_cc OOD set, n=342 — but the clean, consistent story here is the calibration collapse.)* **[GUARD — say it so no one mishears me later:** these calibration numbers, ECE and AUROC, are anchored to eval_curation_v1 and holdout_cc. Our newer n=578 multi-curator gold and the fleet that ran on it measure *error-detection*, not this — no ECE, AUROC, or Brier was recomputed on it. The fleet did **not** confirm these belief numbers; don't let the later sibling slides be heard as if it did.**]**"

**Sources:** AUROC band — text-miner belief at statement grain, n=913 (eval_curation_v1, balanced 803/803, gold=majority-vote, positive=correct): stored 0.714 / recal 0.738 / INDRA-priors 0.741 (belief_headtohead_gemma.md:16,19,22; B1). ECE ~0.37 = stored 0.372 / recal 0.370 (8-bin, statement-grain); INDRA-priors variant is higher at 0.401 (B2). OOD-degradation aside: indra_prior_reference AUROC 0.717 in-dist (n=862) → 0.571 OOD on holdout_cc (n=342), any-incorrect-wins gold (calibration_stage0.md:44; calibration_ship_gate.md:53; B4). High-belief bin %: ⚠ not found in any file (B3). Guard line: no ECE/AUROC/Brier was recomputed on the new gold — frontier_table.py exports error-F1 + cost only, no calibration; the per-evidence `belief` in fleet runs is the source-reliability parametric scalar (the hard-gate product), not fitted soft belief (X16; calibration_constants.py:41-66).

---

### Slide 5 — Same belief, opposite truth
**Big idea:** Make the overconfidence concrete. Two edges with identical maxed belief — same readers, same count — one faithful, one a relation-direction reversal. The formula scores them the same because it never read the sentence.

**On screen** *(chip: BLIND lit)*
- Two edges · same belief · same readers · same count
- Left: the sentence supports it ✓
- Right: the reader flipped the direction ✗
- The meter can't tell them apart

**Visual:** Split-screen of two evidence cards, belief meters pixel-identical and maxed across the top — that sameness is the punch. Below: left card's extraction matches its sentence (green ✓); right card extracts "A phosphorylates B" from a sentence reading "A is phosphorylated by B" (red ✗). *Labeled "illustrative — schematic of a real curated-incorrect case," not a screenshot of a scored edge.*

**Say:** "Concretely: here are two edges our pipeline scores identically — same source, same count, same meter. On the left the sentence genuinely supports the statement. On the right, the reader flipped the relation direction. The reputation is intact; the read is wrong. **[WAKE-UP]** Identical belief, opposite truth — and the only difference lives in a sentence our formula never opened. This is what 'overconfident' looks like one edge at a time, and it's exactly where a second reader looks."

**Sources:** Schematic, self-labeled illustrative — ben_gyori_deck_outline.md:81 (E2). The relation-direction-reversal failure is real and curated-incorrect: README disconfirm block names "relation-direction reversal"; rasmachine_v1_gold.jsonl:59 / rasmachine_v2_gold.jsonl:294 carry a MAPK3→MAPK1 Phosphorylation record tagged polarity, gold=incorrect. The specific A/B sentence pair is the deck's synthetic template (E2).

---

### Slide 6 — We changed who gets counted — not how we count
**Big idea:** The move, purely additive: a second reader as a per-evidence membership gate in front of the count. Hard-gate mechanics included as the membership rule — including why a fully-rejected source *leaves* the product rather than getting rand=1.0.

**On screen** *(chip: TRUST lit)*
- Second reader = a turnstile in front of our count
- correct → keep the read · incorrect → drop it
- all of a source's reads rejected → the source **leaves** the product
- we refuse rand = 1.0 (factor > 1, breaks the additive form)

**Visual:** The belief formula on a greyed pedestal tagged "stays"; a turnstile in *front* of it. A queue of evidence sentences checks against the actual sentence; passes flow into a source's count n behind the gate. When every read of a source is X'd, the whole parenthesized factor slides clean out of the ∏ — next to a struck-out "rand = 1.0 ✗ · leaves [0,1]."

**Say:** "The move: we stand a second reader as a turnstile in front of our count. It reads the sentence and decides one thing — does this read get to be a member of the product? Same equation, different membership. The obvious alternative — zeroing a discredited source by setting rand to 1.0 — we already know pushes its factor above one and shoves belief out of [0,1]. So we don't touch the parameter; a fully-rejected source just leaves the room. **[WAKE-UP]** We didn't touch the equation. We changed what counts as a member of the product."

**Sources:** Membership gate — statement_belief.py:10-12 ("per-sentence MEMBERSHIP GATE on INDRA's additive noisy-OR"), :199 (`included = verdict == "correct"`) (F3). rand=1.0 invalid / source leaves the product — noise_model.py:354-357 ("setting rand_j=1.0 … is INVALID (syst + 1.0 > 1.0) … source is removed entirely"), :427/:432 (surviving==0 → neutral factor 1.0, not multiplied in) (F2).

---

### Slide 7 — The soft form fits on a napkin
**Big idea:** The adopted "clean" scalar re-rates the read instead of deleting it, self-calibrating at n=1; the whole calibration is two numbers per reader; and the confidence axis we tried, watched collapse, and cut.

**On screen** *(chip: TRUST lit)*
- soft "clean" form: re-rate the read, don't delete it
- gemma-26B {0.183, 0.869} · medpsy-4B {0.243, 0.873}
- = P(read wrong | confirmed), P(read wrong | rejected)
- 1 confirmed gemma read → 1 − 0.183 = **0.817** = P(correct | confirmed)
- confidence axis: tried → collapsed → ~~cut~~

**Visual:** A literal index card / napkin: a 2-row table, one row per reader, two number-cells each ("wrong-rate | confirmed", "wrong-rate | rejected"); a third "confidence" column drawn in and struck through with one pen line. A small inset: within a source, geometric mean of verdict-conditioned per-read wrong-rates; across independent sources, they multiply — our product structure.

**Say:** "The hard gate throws information away; the soft 'clean' form — the one we ship — keeps the read and just re-rates how wrong it's likely to be, given the verdict. Within a source we take the geometric mean of those per-read wrong-rates; independent sources multiply, exactly as our product does. And the whole instrument is this card — two numbers per reader, nothing hidden. **[WAKE-UP]** At a single read it's self-calibrating: one confirmed gemma read gives 1 minus 0.183 — 0.817. That's not a tuned knob; it's the measured probability a confirmed read is actually right. We also tried a third axis — the model's own reported confidence — watched it fail to separate right from wrong, and killed it. I'm showing you the strike-through on purpose."

**Sources:** gemma {0.183, 0.869=1−0.131} / medpsy {0.243, 0.873=1−0.127} — calibration_constants.py:42-43 (W1/W2), the canonical clean/production weights, fit n=1606 balanced. Framing P(read wrong|confirmed/rejected) — constants.py:5-6 (W3; note 0.869 is the noisy-OR n=1 per-read failure rate, not the curation-sense rejected-read-wrong rate 0.131). 0.817 = 1−0.183 self-calibrating at n=1 — constants.py:16-17, noise_model.py:285 (W4). Within-source geomean / across-source product — noise_model.py:321,326 (W5). Confidence axis collapsed/cut (gemma 1596/1606 high) — calibration_task_hypergraph.md:11,26 (W6).

---

### Slide 8 — The surprise isn't in the model — it's in our priors
**Big idea:** A shared discovery: recalibrating on our own curations surfaced something about our priors. We refit only the random-error term on 9,342 curations, and the findings are about our own instrument.

**On screen** *(chip: TRUST lit)*
- refit only `rand` on **9,342** curations (our INDRA assembly benchmark; syst held at defaults)
- reach: 0.30 → **0.462** (≈50% noisier)
- sparser: ~0.52 · medscan: ~0.48 *(also refit, n=1161)*
- trips ~0.08 · rlimsp ~0.06 (far *cleaner* than defaults)

**Visual:** Before/after bar chart of the random-error term for the 5 high-n sources, default vs refit. reach jumps up (red up-arrow); trips and rlimsp drop well below default (green down-arrows); sparser and medscan nudge up. The syst row shown greyed/locked.

**Say:** "We held our systematic term exactly at our defaults and refit only the random term — rand = 1 minus accuracy minus syst — across the five high-n sources, straight off 9,342 curations in our INDRA assembly benchmark. **[WAKE-UP]** The interesting part isn't the model, it's our priors: reach comes out at 0.462, roughly fifty percent noisier than the 0.30 default — but trips and rlimsp are far *cleaner* than we set them, about 0.08 and 0.06 (rlimsp was 0.20, not 0.30). Two more refit the same way — sparser nudges up to ~0.52, medscan to ~0.48. One caveat: these live as standalone literals in the file today — there's no programmatic tie back to our defaults, so they'll drift if we update them ⚠ — a wiring job worth doing."

**Sources:** Fit set "INDRA assembly curation benchmark (n=9,342)", refit `rand` only, syst held — calibration_constants.py:48-50 (P1/P7). rand=1−acc−syst — :49,:12 (P2). reach 0.30→0.462 (n=3802, acc 0.488) :54 (P3); sparser 0.516 :55 (P4); trips 0.077 :56 (P5); medscan 0.481 (n=1161) :57 (P7); rlimsp 0.20→0.056 (n=995, acc 0.894) :59 (P6). ⚠ "hand-copied / will drift" is a code-structure inference (RECALIBRATED_PRIORS is a standalone literal dict with no reference to INDRA_PRIORS), not a documented warning in any file (P8).

---

### Slide 9 — We built the gate to fail. It didn't.
**Big idea:** We didn't grade our own homework. Fit two numbers per reader on the balanced set, then validated out-of-distribution through a 4-leg ship gate built to fail — and report the OOD numbers, distinct from the in-distribution headline.

**On screen** *(chip: TRUST lit)*
- Fit: n=1606 balanced (pairs) · Tested: separate OOD holdout (holdout_cc, n=342)
- 4-leg gate: ECE strict-improve · AUROC no-regress · error-F1 non-inferior · None-byte-identity
- **OOD result (gemma-26B):** ECE 0.171 → 0.129 · AUROC 0.740 → 0.768 — all legs pass ⚠ *guard-arm predecessor*
- In-distribution headline (n=913 statements, hard gate): AUROC 0.814 / ECE 0.156 (beats text-miner 0.71–0.74; clean soft on same set: 0.834 / 0.049)

**Visual:** A hurdle race — the calibrated scalar clearing four labeled hurdles in sequence. Beside it, two clearly separated fact-columns: "OOD gate (holdout_cc n=342, gemma-26B): 0.171→0.129, 0.740→0.768" and "In-distribution (n=913): hard 0.814/0.156 · soft 0.834/0.049." The dominant graphic is the ECE collapse (0.37 → 0.156), with AUROC a modest secondary bar.

**Say:** "We fit those two numbers per reader on the balanced n=1606 pair set, then refused to score ourselves there — we ran a ship gate on a separate out-of-distribution holdout, holdout_cc, built to fail. Four legs, all had to pass: ECE strict-improvement, AUROC no-regression, error-F1 non-inferiority — bootstrap CI against a 0.154 margin, which is the medpsy-4B identical-run error-F1 spread (0.871−0.717), not a measured floor — and a byte-identity safety check. **[WAKE-UP]** We built the gate to fail. It didn't. Out-of-distribution, for gemma-26B on holdout_cc (n=342), ECE went 0.171 to 0.129 and AUROC 0.740 to 0.768, and both readers pass all four legs. *(Provenance for us: the persisted gate figures are the 'guard' arm at fixed τ=0.5; per the soft-belief promotion the current 'clean' arm re-passes the same gate at per-arm-optimal τ — re-run the gate to cite fresh clean numbers.)* In-distribution the headline is 0.814 / 0.156 — that's the hard gate on n=913 statements; the clean soft scalar on the same set is actually a touch better, 0.834 / 0.049. The real win, across the board, is calibration: ECE roughly 0.37 down to 0.156 (hard) / 0.049 (soft)."

**Sources:** Fit n=1606 balanced 803/803, holdout_cc disjoint (454 leakage pairs excluded) — eval_curation_v1.meta.json; calibration_ship_gate.py:302 (G1). 4-leg gate — calibration_ship_gate.py:26-32 (G2). OOD gemma ECE 0.171→0.129, AUROC 0.740→0.768, holdout_cc n=342 — calibration_ship_gate.md:12,46; .json:207-214 (G3/G4); ⚠ on-disk soft arm keyed 'guard' at τ=0.5, not current 'clean' arm. Noise floor 0.154 = 0.871−0.717 — calibration_ship_gate.py:67-68 (G5). Both readers PASS — calibration_ship_gate.json (G6). In-dist hard 0.814/0.156 (belief_llm, n=913, gemma-26b, eval_curation_v1, variant=guard run) — belief_headtohead_gemma.md:10; .json:23-24 (H1/H2/H3); soft 0.834/0.049 = belief_llm_soft same line; text-miner band 0.714–0.741 (B1/B5).

---

### Slide 10 — Everything still wrong with it — up front
**Big idea:** Lay every limitation out front; the wins are believable because we led with the losses. Glanceable on screen, nuance in the voice — peers will scrutinize, so give them the list first. The "only 2 readers fitted" limit is no longer abstract — we just demonstrated it at scale.

**On screen** *(chip: TRUST lit)*
- only 2 readers fitted (gemma-26B, medpsy-4B) — we just ran **13** models on fresh balanced gold; **all 13** fall back to the hard gate (incl. 31b + bedrock)
- calibration-only — doesn't move the leaderboard yet
- LLM "incorrect" → maxes at *review*
- belief=None ≠ belief=0
- belief-math reimplemented (no `import indra`) — won't auto-track our priors ⚠ *(repo still depends on indra elsewhere)*

**Visual:** A plain "KNOWN LIMITS" datasheet card — five line items, each an unchecked box, monospaced and flat. Deliberately not a hype slide. The first line item carries a tiny inset of the 13-row fleet list, every row stamped "hard gate."

**Say:** "**[WAKE-UP]** Here they are — all five, on one slide; we led with the losses on purpose. One: only gemma-26B and medpsy-4B have soft weights — and this isn't hand-waving anymore. We just ran *thirteen* models on fresh, balanced, de-biased gold, and every single one — including the 31b and the bedrock servings — returned None from calibration_for and fell back to the hard gate. That's exactly what 'unfitted' looks like at scale: thirteen capable readers, zero fitted soft belief. Not general yet. Two: the calibrated belief is statement-grain and calibration-only — it sharpens the scalar, it does not yet move our accuracy, error-F1, or model ranking, which all stay evidence-pair-grain. Three: only a *deterministic* grounding-reject auto-condemns; an LLM saying 'incorrect' earns at most a 'review,' because its error rate is too high to be a judge. Four: belief=None — nothing readable — stays distinct from belief=0 — genuinely contradicted; the scalar always travels with its evidence tally. Five: the belief-math modules reimplement INDRA's SimpleScorer formula with no `import indra`, so they won't auto-track upstream prior changes — though to be precise, the repo as a whole still depends on indra for data prep. That's the whole list."

**Sources:** Exactly 2 fitted readers, else None→hard gate — calibration_constants.py:41-44,58-66 (L1). "13 models, all hard-gate" demonstrated on n=587 balanced evidence pairs from the n=578 multi-curator gold — data/results/external_curator_v1_bedrock-*.jsonl + .meta.json; scripts/frontier_table.py (13-run table); calibration_for() returns None for '31b' and all 'bedrock-*' serving (calibration_constants.py:41-66) (X6/X14). Calibration-only, headline/ranking stay pair-grain — project_statement_belief.md WIRED-UPDATE note (L2). LLM incorrect→review — statement_belief.py:221-222 (T2). belief=None≠0.0 (contradicted) — statement_belief.py:74,201-203 (T4). ⚠ "clean-room / no runtime dependency": the phrase is in no file (editorial); the no-`import indra` property holds only for noise_model/calibration_constants/statement_belief — repo declares indra>=1.22.0 (pyproject.toml:14) and data-prep scripts import indra.statements (L3).

---

### Slide 11 — "review" is the whole point
**Big idea:** The middle tier isn't a hedge — it's the deliverable: a sentence-level signal orthogonal to source reliability that routes scarce curator hours to exactly the edges worth a look.

**On screen** *(chip: SEES lit)*
- correct → trust it, no human
- incorrect → deterministic grounding-reject (precision 1.0 in-distribution; 0.750 OOD), auto-drop
- **review → the queue a curator actually opens**
- orthogonal to source reliability — a new axis, not a re-ranking

**Visual:** A three-lane funnel: a wide stream of edges enters at top; "correct" peels left (green, no human), "incorrect" peels right (auto-drop), the narrow center "review" lane flows into a single curator at a desk. The middle lane glows.

**Say:** "Here's why that 'review' limit from the last slide is actually the feature. None of us has the curator hours to read everything, so sort. Confirmed reads, you trust. Deterministic grounding-rejects — precision 1.0 in-distribution, and I'll be straight that it's 0.750 out-of-distribution, six of eight — you auto-drop. And everything the LLM is suspicious of but can't condemn lands in 'review' — the queue a human actually opens. The key move: this tier is *orthogonal* to source reliability — a high-belief edge from our most trusted readers can still land in review because the *sentence* is the problem. **[WAKE-UP]** The 'review' limit from the last slide? That's the deliverable — the model tells our curators which 100 statements to read first."

**Sources:** Three-tier {correct|review|incorrect} mapping — statement_belief.py:75,218-224 (T3). Deterministic grounding-reject precision 1.000 in-distribution (eval_curation_v1, n=913) — belief_headtohead_gemma.md:29; 0.750 OOD on holdout_cc (n=393, 6/8) — belief_headtohead_holdout_gemma.md:29; .json:118 (T1). Only deterministic tiers auto-condemn — statement_belief.py:51,219-220.

---

### Slide 12 — Where this goes: validate (started), where it plugs in, broaden the readers (started)
**Big idea:** Collaborative next-steps among co-owners — but two of the three are no longer just intent. Our own gold now *exists* and the reader set is *already* being broadened; say so as progress made. Make the EMMAA seam visible on screen, name the one open to-do, and end on a beat.

**On screen** *(chip: SEES · BLIND · TRUST all lit)*
- 1. Validate on our own gold — **started**: an **n=578** de-biased multi-curator gold now EXISTS (32 curators, live-refetch, de-contaminated). Error-detection ranked; **belief-calibration on it is the open to-do.** (EMMAA corpora still ahead)
- 2. Where it plugs in: *review-tier flags → EMMAA curation queue* **or** *per-evidence support signal inside assembly*
- 3. Broaden the fitted-reader set beyond two — **started**: a 13-model error-detection ranking is the broadening already underway (next: fit soft weights past gemma-26B / medpsy-4B)
- *This isn't a finished thing — we have a second reader now, our own gold to test it on, and a fleet ranked. The question for the room is where we point it.*

**Visual:** Three numbered milestone pins on a short roadmap line; pins 1 and 3 wear a small "✓ started" flag, pin 1 also carries an open "□ belief-calibration" sub-pin. The two seam options drawn as two concrete plug-points into an EMMAA/assembly diagram. Closing line set large beneath.

**Say:** "So, here's where I think this goes — and some of it we've already started. First — validate on our own gold, not just the balanced set. That gold now exists: an n=578 de-biased multi-curator set — 32 curators, built by live evidence re-fetch off /statements/from_hashes, de-contaminated against eval_curation_v1 and our benchmark holdouts. We've already ranked thirteen readers on it for error-detection; what's still genuinely open — and I want to be clean about this — is running the belief-calibration, the ECE and AUROC, on it. That hasn't happened yet. EMMAA corpora are the gold after that. Second — where it plugs in: two natural seams, review-tier flags feeding an EMMAA curation queue, or a per-evidence support signal inside assembly — tell me where you think the gate belongs. Third — broaden the fitted-reader set; two readers was a start, and the thirteen-model ranking is that broadening already underway — the next step is fitting soft weights beyond gemma-26B and medpsy-4B. **[WAKE-UP]** This isn't a finished thing being pitched. We have a second reader now — our own gold to test it on, a fleet ranked — and the open question for the room is where in our pipeline we point it."

**Sources:** "Our own gold now exists" — external_curator_gold_v1.jsonl: 578 rows / 575 unique (matches_hash,source_hash) pairs / 587 joined evidence pairs across 469 statements; 289/289 balanced (de-dup 287/288); 32 curators total, anchor (mock7ee@gmail.com) is 1 of 32, capped 61 rows, cap=40 pairs/curator/class (X1/X2); built by live evidence re-fetch (POST /statements/from_hashes; recovered 3,846 statements / 196,023 evidence) — scripts/recover_curation_evidence.py, build_multicurator_gold.py (X3); de-contaminated 0-pair-overlap vs eval_curation_v1 (n=1606) + eval_curation_v1_clean + data/benchmark holdouts; NOT formally checked vs holdout_cc (lives in data/results/, lacks matches_hash) — scope claim to "vs eval_curation_v1 + the benchmark holdouts" (X4). "Readers broadened" = the 13-model error-detection ranking — scripts/frontier_table.py (X6). Open to-do: NO ECE/AUROC/Brier recomputed on this gold — only error-F1 exists (X16). Plug-in seams + "only 2 fitted readers → broaden" — calibration_constants.py:41-44 (L1); project_statement_belief.md (L2).

---

## Sibling — the model frontier (different question, different metric)

*These two slides answer a **different question** from the 12-slide spine — not "how well-calibrated is the second reader's belief?" but "**which reader, at what cost?**" Different metric: verdict error-detection F1, not belief ECE/AUROC. Different substrate: the n=578 multi-curator gold (n=587 evidence pairs), not eval_curation_v1. Different readers: a 13-model bedrock fleet, not the two fitted calibration readers. The `SEES · BLIND · TRUST` chip is **OFF** on these slides on purpose — the spine stays 12. Nothing here validates the belief-calibration numbers in Slides 4/7/8/9.*

---

### Slide S13 — Which reader, at what cost (no chip)
**Big idea:** A cost × error-detection frontier across 13 readers on the new balanced gold. The top tier is a *tie* — so cost decides. And the cost-winner is explicitly NOT the calibrated reader.

**On screen** *(no chip — sibling)*
- Question: **which reader, at what cost?** — verdict error-detection F1 (positive = curator-flagged incorrect), n=587 evidence pairs / n=578 balanced gold
- **glm-5 0.839** @ $5.89/1k — single highest F1, **Pareto-optimal** (not "dominated")
- **gemma-4-31b 0.827** @ $0.68/1k — **cheapest of the tied tier**, production cost-winner (Pareto knee)
- top 8 tied **0.800–0.839** = **overlapping CIs**, NOT "statistically indistinguishable"; size doesn't buy error-detection — tied incl. **nemotron-super-120b 0.810**; trailing incl. **nemotron-nano-30b 0.767** (always name the variant), qwen3-coder-480b 0.774, gpt-oss-120b 0.786; gemma-4-e2b dead last 0.700
- ⚠ **CAVEAT:** all 13 are **hard-gate-only** — gemma-4-31b ≠ the calibrated reader (calibration_for → None for '31b' + all 'bedrock-*'); carries NO fitted soft belief

**Visual:** A cost (log $/1k, x) vs error-F1 (y) scatter with the Pareto frontier traced. The Pareto frontier (4 points) = gemma-4-e2b, nemotron-nano-30b, gemma-4-31b, glm-5; glm-5 highest on F1 (top, Pareto-optimal); gemma-4-31b sitting at the cheap knee (near-top F1, far-left cost); gemma-4-e2b at the bottom (0.700). The nemotron points are labeled by variant on the plate — nemotron-super-120b (0.810, near the top, in the tie) and nemotron-nano-30b (0.767, trailing) — so the nano/super distinction is visible, not just spoken. The 9 non-Pareto points are greyed: the strictly-dominated set {kimi, deepseek, qwen3-coder-480b, qwen3-235b, gpt-oss-120b} tagged "higher cost, not higher F1," plus 4 that are also dominated by gemma-4-31b but sit inside the tied tier (bedrock-gemma 0.811 @ $0.83, nemotron-super-120b 0.810 @ $0.93, gpt-oss-20b 0.804 @ $0.88, minimax-m2.5 0.800 @ $1.65) tagged "dominated, but inside the tie." A red banner across the whole plate: "EVERY model here = hard gate. None is the calibrated reader."

**Say:** "Different question now, off the spine — not how well-calibrated the belief is, but which reader to run and at what cost. We ran thirteen on the new balanced gold and scored error-detection F1 — catching the curator-flagged incorrects. glm-5 is the single highest at 0.839, and it's Pareto-optimal — I won't call it dominated. But the top eight are bunched 0.800 to 0.839 with *overlapping* confidence intervals — that's an overlapping-CI tie, not a formal 'statistically indistinguishable' result, so I'll only say cost decides at the top. The cheapest of that tied tier is gemma-4-31b at 0.827 for sixty-eight cents per thousand — the production cost-winner, the knee of the curve. Not the absolute cheapest on the plate, mind you — that's the e2b at eighteen cents, but it's dead last on F1 at 0.700, so cost-winner means cheapest *of the tie*. Two caveats. One, size does not buy error-detection here: the 480b coder is 0.774, the 120b oss is 0.786, and the little 4B-class e2b is dead last at 0.700. And if anyone asks whether gemma beats nemotron — it beats nemotron-*nano*-30b at 0.767, but it's a dead tie against nemotron-*super*-120b, 0.811 to 0.810, so I always name the variant. Two — and this is the one that matters for the spine: every one of these thirteen is hard-gate-only. The cost-winner, gemma-4-31b, returns None from calibration_for, same as all the bedrock servings. It is *not* the calibrated reader. Our fitted instrument is still gemma-26B and medpsy-4B on eval_curation_v1. Frontier pick and calibration instrument are different readers on different substrates — don't conflate them."

**Sources:** Fleet error-detection F1 over n=587 joined evidence pairs (n=578 balanced gold rows / 575 unique pairs) — scripts/frontier_table.py; data/results/external_curator_v1_bedrock-*.jsonl + .meta.json (X6–X12). glm-5 0.839 @ $5.89/1k Pareto-optimal (X7); gemma-4-31b 0.827 @ $0.68/1k cheapest of tied tier — NOT the literal value champion, which is gemma-4-e2b 0.700 @ $0.18/1k (X8/X11). Tied tier (overlapping CIs, not formal NS): kimi-k2.5 0.816, bedrock-gemma 0.811 [0.777,0.844], nemotron-super-120b 0.810, deepseek-v3.2 0.809, gpt-oss-20b 0.804, minimax-m2.5 0.800 (X9). Trailing: qwen3-235b 0.788, gpt-oss-120b 0.786, qwen3-coder-480b 0.774, gemma-4-e2b 0.700 (X10). Pareto frontier = {gemma-4-e2b, nemotron-nano-30b, gemma-4-31b, glm-5}; strictly dominated (higher cost, not higher F1) = kimi/deepseek/qwen3-coder-480b/qwen3-235b/gpt-oss-120b; also dominated by gemma-4-31b but inside the tie = bedrock-gemma/nemotron-super-120b/gpt-oss-20b/minimax-m2.5 (X11). Fleet spend $14.56 (sum per-run $) (X12). "gemma beats nemotron" holds vs nemotron-NANO-30b (0.767) only; dead tie vs nemotron-SUPER-120b (0.811 vs 0.810); p-values in scripts/analyze_external_gold.py are hardcoded comments, not recomputed — present the directional claim (X13). calibration_for() → None for '31b' + all 'bedrock-*' → hard gate, all 13 uncalibrated — calibration_constants.py:41-66 (X14). Source mix of the gold: reach 361 / sparser 155 (=89%) / eidos 23 / rlimsp 19 / trips 9 / signor 6 / hprd 4 / biogrid 1 (X5).

---

### Slide S14 — The 0.96 that became 0.81 (no chip)
**Big idea:** An error-detection loss-first beat that fits the deck's "led with the losses" ethos: the gemma family's error-F1 *shrinks* as the gold grows AND balances. It's a comment on the gold, not a regression in the model — and it is error-F1, NOT belief ECE.

**On screen** *(no chip — sibling)*
- gemma family error-detection F1, as the gold grows AND balances:
- **0.957** [0.897, 1.000] — rasmachine, n=60 (tiny, positive-skewed)
- **0.867** — v2-balanced, n=117
- **0.811** [0.777, 0.844] — external balanced, n=587
- the number shrank because the gold got **bigger and fairer** — an error-F1 loss-first beat (NOT belief ECE; bedrock serving = uncalibrated)

**Visual:** Three stepping-stones descending left-to-right — 0.957 → 0.867 → 0.811 — each captioned with its n and balance. CI whiskers on the first (wide, [0.897,1.000]) and last (tight, [0.777,0.844]) to show the small-n inflation collapsing. A side label: "same gemma family, bedrock serving — uncalibrated; this is error-F1, not the belief ECE on Slide 4."

**Say:** "One more loss-first beat, still off the spine. Watch the gemma family's error-detection F1 as we grew and balanced the gold. On sixty rasmachine examples — small and positive-skewed — it was 0.957, confidence interval all the way to 1.0. On the v2-balanced set, n=117, it dropped to 0.867. On the external balanced gold, n=587, it's 0.811, interval 0.777 to 0.844. The number didn't fall because the model got worse — it fell because the gold got bigger and fairer; the early 0.957 was small-sample, skew-inflated. Two things to keep straight: this is the same gemma family as our fitted reader but the bedrock *serving*, which is uncalibrated — and this is error-detection F1, not the belief ECE from Slide 4. I'm putting it up because it's the same discipline as that 'all five limits up front' slide: when the test gets harder and fairer, the number comes down, and we say so."

**Sources:** Mirage trajectory 0.957 [0.897,1.000] (rasmachine n=60) → 0.867 (v2-balanced n=117) → 0.811 [0.777,0.844] (external balanced n=587) — scripts/analyze_external_gold.py (run output, B-section). Error-F1 grain, gemma family / bedrock serving (model_id 'bedrock-gemma', calibration_for → None, uncalibrated) — NOT belief ECE; do not confuse with the deck's fitted gemma-26B (calibration_constants.py:41-66; X14/X16).

---

## Speaker cheat-sheet

**The wake-up beats (the spine — say these verbatim if nothing else):**
1. **S1:** "Two readers agree. So — does the sentence actually say that?" *(mid-sentence pivot, confident top → blurred sentence)*
2. **S2:** "Our score nails source reliability — here's the orthogonal axis it doesn't see: does the sentence support the claim?"
3. **S3:** "Two claims are stacked in every edge — and the count has only ever been able to see one of them."
4. **S4:** "A reputation is not a reading of *this* sentence. So the score is sure of everything." *(the BLIND reveal — calibration, not coin-flip)* — then the **GUARD:** "these ECE/AUROC numbers are eval_curation_v1 + holdout_cc; the new n=578 gold measures error-detection and did NOT confirm them."
5. **S5:** "Identical belief. Opposite truth. The difference is in a sentence our formula never opened."
6. **S6:** "We didn't touch the equation. We changed what counts as a member of the product." *(throughline anchor)*
7. **S7:** "At n=1 the belief reads itself off the data — 0.817 is no knob, it's the measured P(correct | confirmed)."
8. **S8:** "The surprise isn't in the model — it's in our priors: reach is about 50% noisier than our own default."
9. **S9:** "We built the gate to fail. It didn't."
10. **S10:** "Here they are — all five, on one slide. We led with the losses on purpose — and we just proved limit one at scale: thirteen models, all hard-gate."
11. **S11:** "The 'review' limit from the last slide? That's the deliverable."
12. **S12 (close):** "We have a second reader now — our own gold to test it on, a fleet ranked — and the open question for the room is where we point it."

**Sibling beats (different question — say only if the room turns to cost / model choice; NO chip, the spine stays 12):**
- **S13:** "Different question now — which reader, at what cost. glm-5 is the highest at 0.839 and Pareto-optimal; gemma-4-31b is the cheapest of the tied tier at 0.827 for sixty-eight cents per thousand — not the absolute cheapest (that's e2b at eighteen cents, dead last on F1). And all thirteen are hard-gate-only — the cost-winner is *not* the calibrated reader."
- **S14:** "0.957 on sixty skewed rasmachine examples became 0.811 on five-hundred-eighty-seven balanced ones. The number shrank because the gold got bigger and fairer — that's an error-detection loss-first beat, not the belief ECE."

**The numbers to never get wrong:**
1. **0.817 = 1 − 0.183** — one confirmed gemma read; the *clean*-form, n=1 self-calibration property (= measured P(correct | confirmed)). Two numbers per reader: gemma {0.183, 0.869}, medpsy {0.243, 0.873}.
2. **The gap is calibration, not coin-flip.** Count belief *ranks* correctness at AUROC ~0.71–0.74 but ECE ~0.37. Win = ECE collapse. **In-distribution headline 0.814 / 0.156** (hard gate, n=913 statements; the clean soft scalar on the same set is 0.834 / 0.049) is SEPARATE from the **OOD ship-gate (gemma-26B, holdout_cc n=342): ECE 0.171→0.129, AUROC 0.740→0.768** (these are the guard-arm predecessor figures; clean re-passes). Never say "AUROC 0.50" as the headline; never present 0.814/0.156 as the OOD result.
3. **reach 0.30 → 0.462** (≈50% noisier), refit on **9,342** curations (INDRA assembly benchmark); syst held at defaults; trips ~0.08 / rlimsp ~0.06 are *cleaner* (rlimsp was 0.20).
4. **The frontier is a DIFFERENT question from the belief spine.** gemma-4-31b (0.827, the cost-winner) **≠ the calibrated reader** — calibration_for() returns None for '31b' and all 'bedrock-*', so **ALL 13 fleet models are hard-gate-only**; the fitted calibration readers stay gemma-26B + medpsy-4B on eval_curation_v1. The top-8 "tie" (0.800–0.839) is an **overlapping-CI tie, NOT a formal pairwise-NS result** — do not say "statistically indistinguishable." Mirage = error-F1 (0.957→0.867→0.811), not belief ECE. "gemma beats nemotron" holds only vs nemotron-**NANO**-30b (0.767); dead tie vs nemotron-**SUPER**-120b (0.810) — always name the variant. gemma-4-31b is "cheapest of the tied tier," NOT the literal value champion (that's gemma-4-e2b, 0.700 @ $0.18/1k). n bookkeeping: "n=578 balanced gold" (575 unique pairs) and "n=587 evidence pairs" (the F1 denominator, 469 statements) are distinct — keep both.

**Things to never claim:**
1. That count belief is a "coin flip" (it ranks at ~0.74 — the failure is calibration).
2. That 0.814/0.156 is a "clean" OOD result (it's the in-distribution hard-gate / guard predecessor).
3. **That the n=578 multi-curator gold *validated* the belief ECE/AUROC.** It did not — no ECE/AUROC/Brier was recomputed on it; it measures verdict error-detection only. The belief-calibration numbers stay anchored to eval_curation_v1 (n=913/1606) + holdout_cc (n=342). The per-evidence `belief` field in the fleet runs is the source-reliability parametric scalar — the hard-gate product, NOT fitted soft belief (and it is *not* truly verdict-independent for a hard-gate reader: the product drops verdict==incorrect reads). Grouped by the model's own verdict it barely separates: gemma-4-31b 0.691 (confirmed) vs 0.594 (rejected) — or by gold label 0.688 vs 0.593, a ~0.10 gap either way.
---

## Appendix A — Provenance ledger

Every asserted value, traced to a real source. Status: exact = verbatim in the file; approx = rounded/banded; ⚠ = assumed/guard-era/context-corrected (see Appendix B).

| id | slide | value | source | status |
|----|-------|-------|--------|--------|
| E1 | 1 | edge MEK1→ERK2 belief 0.91, reach+sparser (schematic) | ben_gyori_deck_outline.md:21; real MEK→MAPK1 edges ~1.0 | exact (illustrative) |
| F4 | 1 | two-reader belief ≈0.9 (INDRA defaults 0.8775; recal 0.710) | noise_model.py:24-25; statement_belief.py:145 | approx (context: prod uses recal priors → ~0.71) |
| B1 | 2,4,9 | text-miner AUROC 0.714/0.738/0.741 (n=913) | belief_headtohead_gemma.md:16,19,22 | exact |
| B2 | 2,4 | text-miner ECE ~0.37 (stored 0.372/recal 0.370; INDRA 0.401) | belief_headtohead_gemma.json:79,119; .md:24 | approx (statement-grain, 8-bin) |
| F1 | 3 | belief = 1 − ∏ₛ(systₛ + randₛⁿ) | noise_model.py:5-6,108,110 | exact |
| B3 | 4 | high-belief bin at ~50% actual correct | none — visual only; no per-bin export for count belief | not_found ⚠ |
| B4 | 4 | any-incorrect gold AUROC 0.717 in-dist (n=862) → 0.571 OOD (n=342) | calibration_stage0.md:44; calibration_ship_gate.md:53 | exact |
| E2 | 5 | A/B relation-direction reversal card (schematic) | ben_gyori_deck_outline.md:81; rasmachine_v1_gold.jsonl:59 / v2:294 polarity incorrect | exact (illustrative) |
| F3 | 6 | second reader = per-sentence membership gate | statement_belief.py:10-12,199 | exact |
| F2 | 6 | rand=1.0 invalid (syst+1>1); fully-rejected source leaves product | noise_model.py:354-357,427,432 | exact |
| W1 | 7 | gemma {0.183, 0.869} | calibration_constants.py:42 | exact |
| W2 | 7 | medpsy {0.243, 0.873} | calibration_constants.py:43 | exact |
| W3 | 7 | = P(read wrong\|confirmed), P(read wrong\|rejected) | calibration_constants.py:5-6 (0.869 = noisy-OR n=1 failure rate) | exact (nuance) |
| W4 | 7 | 1 − 0.183 = 0.817 = P(correct\|confirmed) at n=1 | calibration_constants.py:16-17; noise_model.py:285 | exact |
| W5 | 7 | within-source geomean; across-source product | noise_model.py:321,326 | exact |
| W6 | 7 | confidence axis tried→collapsed→cut (gemma 1596/1606 high) | calibration_task_hypergraph.md:11,26 | exact |
| P1 | 8 | refit rand on n=9,342 (INDRA assembly benchmark), syst held | calibration_constants.py:48-50 | exact (attribution: upstream benchmark, not fresh curation) |
| P2 | 8 | rand = 1 − accuracy − syst | calibration_constants.py:49,12 | exact |
| P3 | 8 | reach 0.30 → 0.462 (n=3802, acc 0.488) | calibration_constants.py:54 | exact (file=0.462; deck rounded ~0.46) |
| P4 | 8 | sparser 0.516 (~0.52) | calibration_constants.py:55 | exact |
| P5 | 8 | trips 0.077 (~0.08) | calibration_constants.py:56 | exact |
| P6 | 8 | rlimsp 0.20 → 0.056 (~0.06) | calibration_constants.py:59 | exact (default was 0.20 not 0.30) |
| P7 | 8 | 5 sources refit; medscan 0.481 (n=1161) is the unnamed 5th | calibration_constants.py:50-62 | exact |
| P8 | 8 | hand-typed literals → will drift if defaults change | code-structure inference (standalone dict, no ref); no file text | approx ⚠ (inference, not documented) |
| G1 | 9 | fit n=1606 balanced 803/803; test holdout_cc disjoint | eval_curation_v1.meta.json; calibration_ship_gate.py:302 | exact |
| G2 | 9 | 4-leg gate (ECE strict / AUROC no-regress / errF1 non-inf / None-byte-id) | calibration_ship_gate.py:26-32 | exact |
| G3 | 9 | OOD ECE 0.171→0.129 (gemma-26B, holdout_cc n=342) | calibration_ship_gate.md:12,46; .json:207-209 | exact ⚠ (guard arm, τ=0.5) |
| G4 | 9 | OOD AUROC 0.740→0.768 (gemma-26B, n=342) | calibration_ship_gate.md:12,46; .json:211-214 | exact ⚠ (guard arm, τ=0.5) |
| G5 | 9 | noise floor 0.154 = 0.871−0.717 medpsy identical-run spread | calibration_ship_gate.py:67-68 | exact |
| G6 | 9 | both readers PASS all 4 legs | calibration_ship_gate.json:115,231 | exact |
| H1 | 9 | in-dist AUROC 0.814 (hard gate belief_llm, n=913) | belief_headtohead_gemma.md:10; .json:23 | exact |
| H2 | 9 | in-dist ECE 0.156 (hard gate, n=913); soft = 0.834/0.049 | belief_headtohead_gemma.json:24,41; .md:13 | exact |
| H3 | 9 | in-dist gold=eval_curation_v1, variant=guard run | belief_headtohead_gemma.json:3,9 | exact |
| H4 | 4,9 | ECE 0.37 → 0.156 collapse (hard) | belief_headtohead_gemma.json:58,92,24 | exact |
| B5 | 9 | text-miner baseline band 0.714/0.738/0.741; calibrated soft beats it | belief_headtohead_gemma.md:13,16,19,22 | exact (nuance: OOD 0.759@n=393 ≠ 0.768@n=342) |
| L1 | 10,12 | only gemma+medpsy fitted; else hard-gate fallback | calibration_constants.py:41-44,58-66 | exact |
| L2 | 10,12 | calibration-only; headline/ranking stay pair-grain | project_statement_belief.md WIRED-UPDATE note | exact (memory note, not results JSON) |
| T2 | 10 | LLM incorrect → review (never auto-incorrect) | statement_belief.py:221-222 | exact |
| T4 | 10 | belief=None (unread) ≠ belief=0 (contradicted) | statement_belief.py:74,201-203 | exact |
| L3 | 10 | belief-math reimplemented, no import indra → won't auto-track priors | noise_model.py:3-9 vs pyproject.toml:14 (indra>=1.22.0) | approx ⚠ (no-runtime-dep false at repo scope) |
| T3 | 11 | 3-tier {correct\|review\|incorrect} mapping | statement_belief.py:75,218-224 | exact |
| T1 | 11 | deterministic grounding-reject precision 1.000 in-dist; 0.750 OOD | belief_headtohead_gemma.md:29; belief_headtohead_holdout_gemma.md:29; .json:118 | exact (context corrected) |
| X1 | 10,12,S13 | external_curator_gold_v1: 578 rows / 575 unique (matches_hash,source_hash) pairs / 587 joined evidence pairs / 469 statements; balanced 289/289 (de-dup 287/288) | data/benchmark/external_curator_gold_v1.jsonl; scripts/build_multicurator_gold.py | exact (n=578 rows is the nominal label; 587 is the F1 denominator) |
| X2 | 10,12,S13 | 32 curators total; anchor (mock7ee@gmail.com) is 1 of 32, capped 61 rows; cap = 40 pairs/curator/class | scripts/build_multicurator_gold.py | exact |
| X3 | 12,S13 | built by LIVE evidence re-fetch (POST /statements/from_hashes; recovered 3,846 statements / 196,023 evidence) → broke the old ~218 balanced ceiling | scripts/recover_curation_evidence.py; build_multicurator_gold.py | exact |
| X4 | 12 | de-contaminated 0-pair-overlap vs eval_curation_v1 (n=1606) + eval_curation_v1_clean + data/benchmark holdouts; NOT formally checked vs holdout_cc | scripts/build_multicurator_gold.py; deck_rebasing_on_external_gold.md:43 | exact (scope claim to "vs eval_curation_v1 + benchmark holdouts") |
| X5 | S13 | source mix: reach 361 / sparser 155 (=89%) / eidos 23 / rlimsp 19 / trips 9 / signor 6 / hprd 4 / biogrid 1 | data/benchmark/external_curator_gold_v1.jsonl | exact |
| X6 | 10,S13 | 13-model fleet error-detection F1 (positive=incorrect; verdict==incorrect) over n=587 evidence pairs; all resolve None→hard gate | data/results/external_curator_v1_bedrock-*.jsonl + .meta.json; scripts/frontier_table.py; calibration_constants.py:41-66 | exact |
| X7 | S13 | glm-5 0.839 @ $5.89/1k — single highest F1, Pareto-optimal | scripts/frontier_table.py | exact |
| X8 | S13 | gemma-4-31b 0.827 @ $0.68/1k — cheapest of the tied tier (Pareto knee, production cost-winner); NOT the literal value champion | scripts/frontier_table.py | exact (context: see X11) |
| X9 | S13 | tied tier 0.800–0.839 overlapping CIs: kimi-k2.5 0.816, bedrock-gemma 0.811 [0.777,0.844], nemotron-super-120b 0.810, deepseek-v3.2 0.809, gpt-oss-20b 0.804, minimax-m2.5 0.800 | scripts/frontier_table.py | exact (overlapping-CI tie, NOT formal pairwise-NS) |
| X10 | S13 | trailing: qwen3-235b 0.788, gpt-oss-120b 0.786, qwen3-coder-480b 0.774, nemotron-nano-30b 0.767, gemma-4-e2b 0.700 (dead last, 4B-class) — size does not buy error-detection | scripts/frontier_table.py | exact |
| X11 | S13 | Pareto frontier = {gemma-4-e2b, nemotron-nano-30b, gemma-4-31b, glm-5} → 9 non-Pareto. Strictly dominated (higher cost, not higher F1) = kimi/deepseek/qwen3-coder-480b/qwen3-235b/gpt-oss-120b; ALSO dominated by gemma-4-31b but inside the tie = bedrock-gemma (0.811 @ $0.83) / nemotron-super-120b (0.810 @ $0.93) / gpt-oss-20b (0.804 @ $0.88) / minimax-m2.5 (0.800 @ $1.65). Literal F1/cost value champion = gemma-4-e2b (0.700 @ $0.18/1k) | scripts/frontier_table.py | exact (dominated set is 9, not 5 — see Appendix B #12) |
| X12 | S13 | fleet spend $14.56 (sum of per-run $) | scripts/frontier_table.py $/run column | exact ($14.5557; session ~$14.92 figure DROPPED — not computable from cited source) |
| X13 | S13 | "gemma beats nemotron" holds vs nemotron-NANO-30b (0.767) only; dead tie vs nemotron-SUPER-120b (0.811 vs 0.810); p-values are hardcoded comments, not recomputed | scripts/analyze_external_gold.py | approx ⚠ (directional only; p not recomputed) |
| X14 | 10,S13,S14 | calibration_for() returns None for '31b' AND all 'bedrock-*' serving → hard gate; all 13 fleet models uncalibrated; fitted readers stay gemma-26B + medpsy-4B | calibration_constants.py:41-66 | exact |
| X15 | S14 | mirage 0.957 [0.897,1.000] (rasmachine n=60) → 0.867 (v2-balanced n=117) → 0.811 [0.777,0.844] (external balanced n=587); shrinks as gold grows AND balances | scripts/analyze_external_gold.py (run output, B-section) | exact (error-F1 grain; gemma family, bedrock serving = uncalibrated) |
| X16 | 4,12,S14 | NO ECE/AUROC/Brier recomputed on new gold; per-evidence `belief` = source-reliability parametric scalar (the hard-gate product), NOT fitted soft belief. Grouped by the model's own verdict it barely separates: gemma-4-31b 0.691 (confirmed) vs 0.594 (rejected); by gold label 0.688 vs 0.593 (~0.10 gap either way) | scripts/frontier_table.py (no calib export); analyze_external_gold.py; calibration_constants.py:41-66 | exact (number is by-verdict / by-gold grouping, NOT regenerable from the two cited scripts directly) |

## Appendix B — Flags & actions needed

Items that are NOT yet backed by a committed data artifact, or whose context was corrected. Resolve before presenting numbers as established.

**1. Slide 4 visual: 'overconfident high-belief bin sitting at ~50% actual correct'**

- *Issue:* not_found — no data artifact pins a per-bin value for the count/parametric belief; text_miner_baselines.json and calibration_stage0.md export only scalar AUROC/ECE, no per-bin reliability array. The ~50% figure lives only in the deck's described visual.
- *Fix:* Either drop the specific '~50%' bin annotation and draw the bowed curve qualitatively (done in-slide via ⚠), or generate a per-bin reliability table for the count belief (no current script exports it — would need to extend text_miner_baselines.py to emit per-bin arrays). Do not present ~50% as a measured value.

**2. Slide 9: '0.814 / 0.156' presented as the calibrated-scalar in-distribution headline**

- *Issue:* conflated grain + arm. 0.814/0.156 is the HARD gate (belief_llm) at STATEMENT grain n=913; the soft weights were fit on n=1606 PAIRS; and the clean SOFT scalar on the same n=913 is actually better at 0.834/0.049. Presenting 0.814/0.156 as 'the headline the clean scalar inherits' understates the soft form and mixes hard-gate with clean.
- *Fix:* Separate explicitly (done in-slide): say 'in-distribution hard-gate 0.814/0.156 on n=913 statements; clean soft on the same set 0.834/0.049; soft weights fit on n=1606 pairs.' Never imply 0.814/0.156 is the calibrated scalar's number.

**3. Slide 9: OOD 'ECE 0.171→0.129, AUROC 0.740→0.768' attributed to the shipped clean scalar**

- *Issue:* guard-vs-clean conflation. The persisted calibration_ship_gate artifact's soft arm is keyed 'guard' at fixed τ=0.5, not the current 'clean' arm at per-arm-optimal τ. These are the guard-era predecessor figures.
- *Fix:* Re-run scripts/calibration_ship_gate.py (current 'clean' arm) and cite the fresh clean OOD numbers, OR label them 'guard predecessor' (done in-slide via ⚠ and the Say provenance aside). Per MEMORY clean re-passes G2 OOD, but the on-disk numbers are guard.

**4. Slide 10/cheat-sheet: 'clean-room — no runtime dependency — won't auto-track our engine'**

- *Issue:* approx/partly false. The phrase 'clean-room / no runtime dependency' appears in NO file (editorial). 'No runtime dependency' is false at repo scope: pyproject.toml:14 declares indra>=1.22.0 and build_rasmachine_eval.py / run_rasmachine_monolithic.py import indra.statements. The property holds only for the belief-MATH modules (noise_model/calibration_constants/statement_belief have no import indra).
- *Fix:* Scope the claim to 'the belief-math modules reimplement INDRA's SimpleScorer and don't import indra, so won't auto-track upstream prior changes' (done in-slide); drop the repo-wide 'no runtime dependency'.

**5. Slide 11/cheat-sheet: deterministic grounding-reject 'precision ~1.0'**

- *Issue:* context_mismatch. Precision 1.000 is IN-DISTRIBUTION only (eval_curation_v1, n=913); on the OOD holdout_cc (n=393) it is 0.750 (6/8). Presenting ~1.0 without scope overstates it.
- *Fix:* State 'precision 1.0 in-distribution; 0.750 OOD' (done in-slide and in the Say).

**6. Slide 8: 'refit on 9,342 of OUR curations / straight off 9,342 of our own curations'**

- *Issue:* attribution drift. calibration_constants.py:48 labels the fit set the 'INDRA assembly curation benchmark (n=9,342)' — the upstream benchmark, not fresh curation by us; it is distinct from eval_curation_v1 (n=1606).
- *Fix:* Phrase as '9,342 curations from our INDRA assembly benchmark' (done in-slide). The lab-collective 'our' is defensible, but don't imply newly-curated data.

**7. Slide 8: 'these are hand-copied literals today, so they'll drift'**

- *Issue:* approx/undocumented. No file contains 'hand-copied'/'will drift' text; it is a valid code-structure inference (RECALIBRATED_PRIORS is a standalone literal dict with no programmatic reference to INDRA_PRIORS).
- *Fix:* Keep as an explicit engineering inference, not a quoted warning (marked ⚠ in-slide); optionally add a real code comment / wiring so the claim becomes documented.

**8. Slide 8: 'reach 0.30 → ~0.46'**

- *Issue:* rounded. File value is exactly 0.462 (n=3802, acc 0.488).
- *Fix:* Cite 0.462 (done in-slide); ~0.46 is acceptable shorthand and ≈50% noisier holds (0.462/0.30=1.54×).

**9. Slide 1 / Slide 5: belief 'maxed' / '≈0.9' two-reader edge**

- *Issue:* approx context. The ≈0.9 two-reader illustration holds under upstream INDRA default priors (compute 0.8775); the production statement_belief path defaults to RECALIBRATED_PRIORS, which gives ~0.710 for the same reach+sparser pair.
- *Fix:* Keep ≈0.9 as the INDRA-native schematic (slides are self-labeled illustrative), but be aware production recalibrated belief for the same two-reader case is ~0.71 (F4).

**10. Sibling section (S13/S14) + Slide 4 guard + Slide 12: the new gold/fleet is error-detection, NOT belief calibration**

- *Issue:* the single most dangerous mishearing. The n=578 multi-curator gold (n=587 evidence pairs) and the 13-model fleet measure VERDICT error-detection F1 only. NO ECE/AUROC/Brier was recomputed on it — there is no metrics.json / per_statement / calibration / ship_gate export for any external_curator run (frontier_table.py emits error-F1 + cost only). The per-evidence `belief` field in those runs is the source-reliability parametric scalar — the hard-gate product, NOT fitted soft belief; and it is NOT truly verdict-independent (a hard-gate reader's product drops verdict==incorrect reads). Grouped by the model's own verdict it barely separates: gemma-4-31b 0.691 (confirmed) vs 0.594 (rejected); by gold label 0.688 vs 0.593. calibration_for() returns None for '31b' and all 'bedrock-*' serving, so all 13 fleet models are hard-gate-only. The frontier PICK (gemma-4-31b) and the calibration INSTRUMENT (gemma-26B + medpsy-4B on eval_curation_v1) are different readers on different substrates.
- *Fix:* keep the sibling clearly labeled as a DIFFERENT question (which reader, at what cost) with the SEES · BLIND · TRUST chip OFF and the spine fixed at 12 (done). Slide 4 carries an explicit guard line; the cheat-sheet adds "never say the n=578 gold validated the belief ECE/AUROC." Belief-calibration core (Slides 4/7/8/9) stays anchored to eval_curation_v1 (n=913/1606) + holdout_cc (n=342). Frame the top-8 as an OVERLAPPING-CI tie, NOT "statistically indistinguishable" — paired tests were not run; the p-values in analyze_external_gold.py are hardcoded comments. glm-5 is Pareto-OPTIMAL (single highest F1), NOT dominated. gemma-4-31b is "cheapest of the tied tier," NOT the literal value champion (gemma-4-e2b). "gemma beats nemotron" holds only vs nemotron-NANO-30b (0.767), dead tie vs nemotron-SUPER-120b (0.810) — name the variant. De-contamination scope = 'vs eval_curation_v1 + benchmark holdouts' (NOT formally checked vs holdout_cc). Keep 'n=578 balanced gold' and 'n=587 evidence pairs' distinct. State the 0.691/0.594 split by its grouping (by-verdict) — do NOT call it 'verdict-independent' in the same breath.

**11. Sibling S14: '0.96→0.81 mirage' is the gemma FAMILY, bedrock serving (uncalibrated)**

- *Issue:* naming nuance. The fleet run in the mirage is model_id 'bedrock-gemma' (bedrock serving), which calibration_for() resolves to None — same gemma family as the deck's fitted gemma-4-26B but a DIFFERENT serving substrate, not the fitted reader.
- *Fix:* name it 'the gemma family / bedrock serving (uncalibrated)' so it is not confused with the fitted gemma-26B from Slide 7, nor with the 31b pick from S13 (done in-slide). The beat is error-F1, explicitly NOT the belief ECE of Slide 4.

**12. Slide S13 visual / Appendix X11: the 'truly dominated' set is incomplete (9 non-Pareto, not 5)**

- *Issue:* under-statement (does not inflate the deck's case). frontier_table.py's Pareto set is {gemma-4-e2b, nemotron-nano-30b, gemma-4-31b, glm-5}, so 9 models are non-Pareto. The deck originally labeled only 5 as 'truly dominated' (kimi/deepseek/qwen3-coder-480b/qwen3-235b/gpt-oss-120b) and omitted bedrock-gemma (0.811 @ $0.83), nemotron-super-120b (0.810 @ $0.93), gpt-oss-20b (0.804 @ $0.88), minimax-m2.5 (0.800 @ $1.65) — all also dominated by gemma-4-31b (0.827 @ $0.68) but sitting in the tied tier.
- *Fix:* Relabel (done in-slide and X11): the greyed set now reads as the strictly-dominated 5 PLUS 4 that are 'dominated, but inside the tie.' No headline number affected.

**13. Appendix X12: 'session ~$14.92' dropped**

- *Issue:* not computable from the cited source. Fleet spend $14.56 is verified exact ($14.5557, sum of 13 per-run costs in frontier_table.py); the 'session ~$14.92' figure has no stated provenance in these scripts.
- *Fix:* Dropped the $14.92 session figure (done in S13 Sources + X12). Keep only the verified $14.56 fleet spend; re-cite a billing/usage log if the session total is ever needed.

**14. Appendix X15: citation pointed at analyze_external_gold.py:5 (a docstring)**

- *Issue:* citation precision. Line 5 is a docstring; the 0.957/0.867/0.811 mirage figures are produced by RUNNING the script's B-section (all confirmed exact). The script's internal GEN label is a stale hardcoded 'v2-balanced-114' while the actual computed n is 117 — the deck correctly uses n=117, so this is a stale code label, not a deck error.
- *Fix:* Cite the script as 'run output (B-section)' rather than line 5 (done in S14 Sources + X15); optionally fix the stale 'v2-balanced-114' label in analyze_external_gold.py to 117. No deck number changes.
