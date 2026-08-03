# Deck arc reconciliation — task hypergraph (map → do → review)

> [!IMPORTANT]
> **CURRENT CORRECTION — 2026-07-13.** The fitted-reader scalar is a
> configuration-specific **hybrid log-odds score**, not the old membership gate
> and not a pure Bayesian posterior. Each verdict contributes a measured
> log-likelihood ratio; reads are averaged within a source and sources add. A
> confirmed read retains the stronger separately fitted source-reliability
> logit, which is the non-Bayesian hybrid term. Remote/Ollama Gemma is fit from
> **704/157/97/646** and reasoning-first Bedrock Gemma from its own
> **662/81/139/722** matrix; both are deduplicated **n=1,604 unique-pair** fits.
> Bedrock never borrows the remote profile. Current independent checks are remote
> `holdout_cc` (**n=414**) raw→hybrid **ECE .430→.148 / AUROC .610→.745**
> (formal hard→hybrid **.157→.148 / .734→.745**, error-F1 Δ
> **+.016 [.003,.032]**, 4/4) and Bedrock external
> raw→hybrid **AUROC .688→.814 / ECE .129→.061** (formal hard→hybrid **.237→.061 /
> .794→.814**, error-F1 Δ **−.001 [−.015,+.014]**, 4/4). Bedrock is the only
> formal external validation. MedPsy's matched `holdout_cc` test (**n=414**)
> worsens ECE **.129→.146**, so it is 3/4 and production-disabled. Its external
> run used monolithic prompt SHA
> `07377e338ff2`, not its fitted/eval profile SHA `b44638216740`; it is therefore
> an unmatched-profile stress test with no valid fitted soft or gate/pass result.
> The two Gemma checks are separate frozen instruments, not the same numbers
> “holding twice.” The final-arm
> labels are **hybrid log-odds**, not “likelihood-ratio belief”: likelihood ratios
> are the verdict evidence components, not the full source-aware scalar.

## Current resolution

The coherence decision still holds at the product level: fitted configurations
replace INDRA's statement-level count number with a score rebuilt from the
reading. The mechanism is now more precise than the July 7 wording: it combines
configuration-matched verdict likelihood ratios with a conservative source
reliability floor. Unfitted configurations retain the named hard-gate fallback.

## Historical coherence task — 2026-07-07

**Historical problem.** The "Reading the Evidence" deck told two conflicting stories about what we built:
- **Story A (gate):** slide 2 note "a gate in front of the belief, *not a replacement for it*"; slide 6 "*not the formula* … the gate changes only `n` … **The formula is untouched.** Same equation. Different membership." (ATM kept 0.906, AQP1 dropped.)
- **Story B (recalibrate, then-current wording):** slides 8–11 described a **new Bayes belief** (base rate + Σ reading) that **replaces** INDRA's count number. The replacement decision survived; the “pure Bayes” characterization did not. The live score is the hybrid described above.

Story A describes a **hard-gate design that the fitted path supersedes**. Production belief for a fitted configuration is the source-aware hybrid log-odds score; the "membership / which reads get in" survives only as a *reporting* count (`n_surviving`), NOT as the fitted belief mechanism (a rejected read is negative evidence, not dropped). The hard gate remains the explicit fallback for an unfitted configuration.

## THE INVARIANT (canonical throughline every slide must serve)

> INDRA scores an extraction by **reputation** (who read it, how many agree, how reliable the source). We **read the sentence itself** and turn configuration-matched verdicts into a **calibrated hybrid belief that replaces INDRA's statement-level reputation number**. Reading catches misreads the count cannot and its verdicts rebuild the scalar; source reliability remains a conservative confirmation floor, not a second likelihood ratio. The result is not a pure Bayesian posterior.

## Invariants (do-not-violate)

1. **One throughline** (above). Every arc slide serves it; none contradicts it.
2. **No "we keep INDRA's formula/number"** claim for the shipped (fitted-reader) belief. The number is rebuilt.
3. **Real examples only** — ATM→H2AX (0.906) and AQP1–EFEMP2 (0.923) are real gold; never fabricate or silently change a number without recomputing.
4. **Honesty / no overstatement** — the gate/membership is a *reporting view* of reading; the recalibration is the number. Don't oversell either.
5. **Minimal scope** — reconcile via targeted edits; preserve the deck's structure, tone, and the concrete ATM/AQP1 moment. This is a coherence fix, not a teardown.

## Historical nodes

### MAP (understand)
- **M1** (parallel, one per arc slide 2–11): extract the slide's *claim about what we built*, its stance (gate / recalibrate / neutral), the exact phrases (if any) that violate the invariant, and a proposed minimal edit. → coherence map.
- **M2** (synthesize, human): fold M1 into the ordered do-list; confirm the throughline + the 6→7→8 bridge.

### DO (edit) — depend on M2
- **D-s2**: scope slide 2's "not a replacement" — true at the *system* level (we add reading to INDRA, don't rip it out) but the belief *number* is rebuilt; make that explicit or drop the absolute claim.
- **D-s6**: the load-bearing fix. Drop "The formula is untouched. Same equation." Reframe slide 6 as *reading catches the misread* (keep ATM kept / AQP1 dropped) and set up "the number itself we rebuild next." Do NOT present INDRA's noisy-OR as "our approach."
- **D-s78**: 6→7→8 bridge — 7/8 own the pivot: even with clean inputs the number is overconfident, so we rebuild it from the verdicts (the recalibration = the new number).
- **D-resid**: any residual violations M1 surfaces on slides 3/4/5/9/10/11 (e.g., stray "gate not replacement" language).

### REVIEW (verify) — depend on all DO
- **R1** (adversarial coherence, workflow): read the full revised arc 2→12; try to find ANY residual Story-A/Story-B contradiction or a slide that still implies we keep INDRA's number. Report survivors.
- **R2** (render, human): export edited slides; verify no overflow/clipping and the edits read as intended.

## Historical edges
M1 → M2 → {D-s2, D-s6, D-s78, D-resid} → R1 → (fix survivors) → R2.

## Historical status — COMPLETE (2026-07-07)
- [x] M1 (workflow, 10 slides): 9/10 coherent; sole violator = slide 6. Also surfaced the "gate in front, not a replacement" line lives on the COVER note (L31), not slide 2.
- [x] M2: do-list = slide 6 + cover note (D-s2 retargeted to the cover note; D-s78/D-resid: 7-11 already coherent, no change).
- [x] D-s6: title→"We rebuild the belief from the reading"; dropped the noisy-OR + "formula untouched"; verdicts→supports/evidence-against; punchline→"The verdict — not the reputation — sets the belief." Kept the real ATM/AQP1 moment.
- [x] D-cover: L31 note "gate in front … not a replacement" → "rebuilds the belief … replacing the reputation number."
- [x] R1 (workflow, 3 lenses): CAUGHT WHAT M1 STRUCTURALLY COULDN'T — the **closing slide (19) reverted to Story A verbatim** ("Same equation, different membership", "gate in front of the formula, only the reads it confirms get counted", note "we didn't touch the equation"), the **cover metadata info block** still said "gate in front of the belief", and a **slide-6 side-effect**: ATM's un-struck 0.906 read as "confirm → keep INDRA's number", colliding with slide-8's →0.82.
- [x] R1-fixes: closing slide (body + note) → Story B; cover info block → replace framing; slide-6 ATM 0.906 → "holds" (no colliding number, keeps AQP1's struck 0.923); slide-15 stray "where the gate belongs" → "where the reading belongs". Legit "hard gate"/"ship gate"/"4-leg gate" terms left (fallback belief + validation gate, not Story A).
- [x] R2: slide 6 + closing (19) export-verified Story B; residual grep clean (only slide-11 note "same equation, transformed" = Beat1↔Beat3, a false positive).

LESSON: a per-node MAP can't catch a node it isn't assigned — the closing slide + cover metadata were outside the "slides 2–11" map scope; only the whole-arc adversarial REVIEW found them. Always scope review to the FULL artifact, not just the changed nodes.
