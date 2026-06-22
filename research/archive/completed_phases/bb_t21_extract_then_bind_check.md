# BB-T2.1 design spike — extract-then-bind-check probe redesign

## Question

Should the `relation_axis` LLM probe be replaced with a two-stage architecture
that (1) extracts the asserted relation tuple from evidence, then (2)
deterministically bind-checks against the typed Statement?

## Current architecture (post-AA, F1 = 0.827)

```
relation_axis(claim, evidence) → answer ∈ {
    direct_sign_match,
    direct_sign_mismatch,
    direct_axis_mismatch,
    direct_partner_mismatch,
    via_mediator,
    via_mediator_partial,
    no_relation,
    abstain,
}
```

The probe simultaneously (a) extracts what the evidence asserts AND (b)
classifies the relationship-to-claim into one of 8 buckets. These are
conflated. The closed answer set carries an implicit taxonomy that
struggles to represent valid biological patterns (transphosphorylation,
LOF-attenuates, multi-member complex, named-effector chains).

The AA-phase prompt iteration has clarified the cost of this
conflation:

- The COUNTER-EXAMPLES clause (AA-T1.C) added 3 carve-outs for
  legitimate `direct_sign_match` patterns (LOF-mediated, complex-
  disruption, named-effector) but is UNSTABLE on FBXW7 / E3-ligase
  semantics — bleeding in both directions (false-positives on E3-ligase
  substrate-degradation framing AND false-negatives on the LOF-attenuates
  pattern the clause names).
- Class-C residual errors (transphosphorylation, regulates_amount_via_
  degradation, blockade-readout polarity) are answer-set gaps that
  cannot be expressed within the 8-bucket vocabulary.
- Polarity confusion (FP-5 AR→E2F1) requires comparing the asserted
  sign in evidence against the typed Statement sign — explicit
  comparison the closed-set probe doesn't perform.

## Proposed architecture

```
# Stage 1 — extraction (single LLM call)
extract_relation(claim_aliases, evidence_text) → ExtractedRelation = {
    subject:        str | None,   # asserted subject of relation
    object:         str | None,   # asserted object
    axis:           str,          # the asserted verb taxonomy
    sign:           str,          # positive / negative / neutral
    perturbation:   str | None,   # LOF / GOF asserted in evidence
    intermediate:   str | None,   # mediator if a chain
    scope:          str,          # asserted / hedged / negated
    mechanism:      str | None,   # "proteasomal" / "transcriptional"
    rationale:      str,
}

# Stage 2 — bind-check (deterministic Python)
bind_check(claim, extracted, aliases) → BindCheckResult = {
    verdict:    str,    # exact_match / sign_mismatch / axis_mismatch /
                        # partner_mismatch / role_swap / via_mediator /
                        # no_match / abstain
    confidence: str,
    rationale:  str,
}
```

Stage 1 is the only LLM call (replaces `relation_axis`). Stage 2 is
pure Python with three concerns:

1. **Alias resolution**: claim subject/object aliases (Gilda map) vs
   extracted subject/object surface forms.
2. **Axis taxonomy**: a structured map from the extracted verb to the
   claim's typed axis. E.g., `transphosphorylation IS-A phosphorylation
   IS-A modification`. Also encodes the partner-type-aware binding
   distinction (Complex requires protein-protein; binding to DNA element
   is `partner_mismatch`).
3. **Sign reconciliation**: the asserted sign in evidence is compared
   against the typed Statement sign, with explicit LOF/GOF inversion
   logic moved from the adjudicator into bind-check (where it's the
   first thing examined, not an after-the-fact swap on a closed-set
   answer).

## Why this addresses the AA-phase residuals

For each of the 5 movable AA errors, walk through what extract-then-
bind-check would do:

### Test 1 — HGF→RAC1 (#17 FN, named-effector)
Evidence: "SF caused the activation of Src and the Rac1 effector Pak1."

- Extract: subject="SF", object="Pak1", axis="activity", sign="positive",
  intermediate=None, mechanism="Rac1 effector".
- Bind-check: SF aliases to HGF (Gilda) ✓; Pak1 ≠ RAC1, but mechanism
  field carries "Rac1 effector" → re-interpret as via_mediator with
  Pak1 as named intermediate, RAC1 as the actual relation object.
- Verdict: `via_mediator` → causal claim accepts indirect chain → correct.

The current architecture loses this because the 8-bucket probe is
forced to choose between `no_relation` (literal: Pak1 ≠ RAC1) and
`via_mediator` (implicit: Pak1 is a known RAC1 effector); without the
mechanism field, the choice is arbitrary.

### Test 2 — FBXW7→NFkB (#13 FN, LOF-attenuates)
Evidence: "Fbxw7 silencing attenuated the NF-κB response"

- Extract: subject="Fbxw7", object="NF-κB response", axis="activity",
  sign="negative" (attenuated), perturbation="LOF" (silencing).
- Bind-check: subject ✓, object NF-κB response → NFkB ✓, axis ✓.
  Sign reconciliation: LOF inverts asserted sign → effective positive.
  Claim sign = positive → match.
- Verdict: `direct_sign_match` (with LOF inversion explicit).

Current architecture fires `no_relation` because the closed-set probe
sees "Fbxw7 ... NF-κB" and can't unambiguously call it match given the
silencing framing.

### Test 3 — LEP→KDR (#15 FN, transphosphorylation)
Evidence: "leptin induced the transphosphorylation of Y1175, Y951, Y996 of VEGFR-2"

- Extract: subject="leptin", object="VEGFR-2", axis="transphosphorylation",
  sign="positive", site=["Y1175", "Y951", "Y996"].
- Bind-check: subject leptin → LEP ✓; object VEGFR-2 → KDR ✓; axis
  taxonomy: `transphosphorylation IS-A phosphorylation` ✓; sign ✓.
- Verdict: `direct_sign_match`.

The closed-set probe fires `direct_axis_mismatch` because
"transphosphorylation" isn't an answer token. Bind-check's axis
taxonomy handles it.

### Test 4 — AR→E2F1 (#4 FP, polarity)
Evidence: "Both c-Myc RNAi and AR RNAi reduced expression of the c-Myc-activated gene E2F1"

- Extract: subject="AR", object="E2F1", axis="amount" (expression),
  sign="positive" (after LOF inversion: RNAi reduces → AR normally
  increases E2F1), perturbation="LOF" (RNAi).
- Bind-check: claim is `DecreaseAmount(AR, E2F1)` → sign=negative.
  Extracted effective sign=positive. Sign mismatch.
- Verdict: `direct_sign_mismatch` → incorrect.

Current architecture: the LOF inversion lives at the adjudicator and
swaps `direct_sign_match` ↔ `direct_sign_mismatch` AFTER the closed-set
probe answers. The probe sees "AR RNAi reduced E2F1" and reports
`direct_sign_match` for a negative claim sign — the swap then yields
the WRONG `direct_sign_match` (positive) for a `DecreaseAmount` claim.
Bind-check makes the asserted sign and claim sign comparable explicitly.

### Test 5 — FBXW7→Proteasome (#8 FP, act_vs_amt + machinery-object)
Evidence: "Ectopic expression of Fbw7 destabilized GATA2 and promoted its proteasomal degradation"

- Extract: subject="Fbw7", object="GATA2", axis="amount", sign="negative",
  mechanism="proteasomal".
- Bind-check: claim is `DecreaseAmount(FBXW7, Proteasome)`. Extracted
  object is GATA2 ≠ Proteasome. Mechanism is proteasomal but the
  RELATION is on GATA2, not on Proteasome.
- Verdict: `no_match` or `direct_partner_mismatch` → incorrect.

Current architecture: the closed-set probe sees "Fbw7 destabilized" +
"Proteasome" tokens and fires `direct_sign_match`; the BB2 narrowing
catches this specific pattern but is a brittle text-anchored rule.
Bind-check makes the object identity explicit.

## Cost-benefit

### Cost
- One additional LLM call per claim (vs the existing 4 probes — 25% cost
  increase). Mitigable by replacing relation_axis entirely rather than
  adding a 5th probe.
- Implementation: new `extract_relation.py` probe, new bind-check
  module with axis taxonomy + alias resolution, adjudicator
  refactoring to consume the new probe.
- Calibration: the current `RELATION_AXIS_BASE` factor table (8 values)
  gets replaced with a bind-check verdict mapping. Less calibration
  surface, but the bind-check verdicts need their own factor map.
- Migration: holdout re-run + factor recalibration; risk of regression
  on the 41 current TPs.

### Benefit
Movable errors after extract-then-bind-check (estimated):
- #17 HGF→RAC1 (via_mediator) — recoverable
- #13 FBXW7→NFkB (LOF-attenuates) — recoverable
- #15 LEP→KDR (transphosphorylation) — recoverable (Class C closed)
- #4 AR→E2F1 (polarity) — recoverable
- #8 FBXW7→Proteasome (act_vs_amt + machinery-object) — recoverable
  (and the BB2 narrowing becomes redundant)
- #12 KISS1→GNRH1 (act_vs_amt) — possibly recoverable

Expected F1 delta: +0.04 to +0.06 (5-6 errors flipped from 12 total
residual budget).

### Tier-2 ship decision

**Recommendation: SHIP as BB-T2.2 in a focused next phase.** The 5
movable AA errors all derive their structural difficulty from the
closed-set vocabulary of `relation_axis`. Iterating prompts on the
current probe will continue to shuffle errors (Z1b confirmed this);
the redesign collapses the failure mode by changing the question.

### Phase-2 design constraints

1. **Single LLM call**: extract_relation must replace `relation_axis`,
   not augment it. Bundling logic in the orchestrator becomes simpler
   (3 probes + extract instead of 4 probes).
2. **Axis taxonomy as data, not prompt**: ship a per-statement-type
   axis lookup table (Phosphorylation accepts transphosphorylation,
   autophosphorylation, dephosphorylation-inverse; Complex requires
   protein-protein; etc.). This is structured knowledge that doesn't
   need to live in the LLM prompt.
3. **Bind-check unit-tested per row of the AA-phase residual**:
   before holdout re-run, every one of the 5 movable error patterns
   has a deterministic unit test against bind-check. If the test
   passes, the LLM extraction is the only remaining variability.
4. **Factor-table calibration**: the existing 8-bucket `RELATION_AXIS_BASE`
   gets replaced. The new verdict set is smaller (~5-6 buckets) and
   maps more cleanly to confidence (exact_match=0.92, axis_mismatch=0.15,
   sign_mismatch=0.10, partner_mismatch=0.15, via_mediator=0.65,
   no_match=0.08).

## Falsifier for BB-T2.2 ship decision

If implementation reveals that:
- Extraction prompt drift is worse than closed-set classifier drift
  (extracted subject/object/axis are NULL or unreliable more often
  than the closed-set probe abstains), OR
- Bind-check axis taxonomy turns into a deep ontology project rather
  than ~30 lines of mapping, OR
- The 41 current TPs regress by ≥3 records on holdout re-run

then BB-T2.2 is wrong and we revert to AA + BB-T1 + a fresh holdout.

## Recommendation summary

- **BB-T1 (in flight)**: ship; small +0.012–0.020 F1.
- **BB-T2.1 (this doc)**: design spike → DECIDED ship-as-BB-T2.2.
- **BB-T2.2 (proposed next phase)**: implement extract-then-bind-check
  redesign. Expected F1 +0.04 to +0.06 over post-BB1+BB2 baseline.
- **BB-T3 (after BB-T2.2)**: commission a new curator-audited holdout
  (≥150 records, ≤5% reader-error-flagged). Current eval_set_v4 has
  4/100 hedged-or-reader-flagged; the ceiling has been so well-mapped
  that further work needs more signal.

## Open questions

1. **Where does Gilda alias resolution live in bind-check?** Currently
   `context_builder._expand_synonyms` builds the alias set at probe-
   bundle construction. Bind-check should re-use this rather than
   re-resolving.
2. **What's the canonical axis vocabulary for extraction?** Probably
   the INDRA Statement type vocabulary (Phosphorylation, Activation,
   ...) extended with mechanism qualifiers (proteasomal,
   transcriptional, transphosphorylation).
3. **Backwards compatibility**: do we maintain the closed-set
   relation_axis as a fallback when extraction fails, or rip it out
   entirely? Recommend rip-out — partial migrations are debt.
