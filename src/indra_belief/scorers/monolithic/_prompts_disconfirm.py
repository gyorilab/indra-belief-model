"""Disconfirm-first scoring prompt.

A commit-first variant that counters lenient acceptance — the model reaching a
disqualifying observation and then rationalizing it away. The disposition is
INPUT-side (prompt + few-shots); the model's verdict is final — code never
overrides it:
 1. OUTPUT STRUCTURE — the model first states `support` (the exact evidence span
    stating THIS relation) and `objection` (the single strongest reason to doubt it),
    THEN the verdict. Surfacing the objection before deciding is a reasoning aid; it
    does not, by itself, force a verdict — the model judges whether the objection
    actually stands.
 2. SKEPTICISM PRIOR — default 'incorrect unless the evidence explicitly and directly
    states the claim'; background knowledge is not support.

Earlier revisions added a code "decision backstop" that forced 'incorrect' whenever the
parsed `objection` field was non-null. That was output-side determinism second-guessing
the model: it cannot tell a STANDING defeater from an apparent objection the model
surfaced and then legitimately resolved (e.g. miRNA inverse-inference — "inhibiting the
miRNA raised the target" reads like wrong-direction but supports DecreaseAmount), so it
threw away correct verdicts. Removed. Determinism's role here is INPUT (grounding,
disposition), never OUTPUT. `support`/`objection` remain as a reasoning scaffold and
telemetry, not as inputs to a verdict override. Do not re-add the override; sharpen the
prompt/few-shots instead.

This module is prompt + few-shot RENDERING only. Reading the four-field answer
back is `indra_belief.verdict`'s job — one parser for every profile, which is
where `support`/`objection` are recovered and their nullish spellings
normalized, and where the model's committed verdict passes through unchanged.

Selected via env MONO_VARIANT=disconfirm in the scorer.
"""
from __future__ import annotations

import json

from indra_belief.scorers.monolithic._prompts import (
    CONTRASTIVE_EXAMPLES,  # noqa: F401 — kept importable for parity
)

# Same rule body as the baseline, but a structured, commit-first output contract.
DISCONFIRM_SYSTEM_PROMPT = """\
You judge whether a biomedical text-mining extraction is correct.

You are given:
- A CLAIM: SUBJECT [TYPE] OBJECT, optionally with @residue+position
- Optionally an "Entities:" line (canonical names + aliases, family membership)
- Optionally an "Extraction provenance:" block when grounding is uncertain — treat
  MISMATCH / LOW_CONFIDENCE as strong signals the claim is probably incorrect
- EVIDENCE text from the source paper

Key rules:
1. @residue+position must match the evidence. "S51A" is a mutation, not a phospho-site.
2. Family-level claims: if the claim names a protein family (JUN, AKT, ERK, CK2, MAPK,
   PKA), evidence about any specific member SUPPORTS it — this is NOT an objection.
3. "Activation"/"Inhibition" = a change in the target's activity STATE (its function
   switched on/off — phosphorylation, catalysis, binding that gates activity), NOT a
   change in its AMOUNT. The following are AMOUNT (Increase/DecreaseAmount), so an
   Activation/Inhibition claim whose evidence is only about them is INCORRECT (objection:
   amount-not-activity) EVEN when the verb reads "represses"/"suppresses"/"activates":
   transcription or expression, mRNA/protein levels or abundance, promoter/luciferase/
   reporter activity, up-/down-regulation, stabilization, accumulation, or proteasomal
   degradation. It is activity ONLY when the target's OWN function/enzymatic activity is
   turned on/off. A miRNA reducing a target = Inhibition OR DecreaseAmount (accept either;
   don't reject a miRNA DecreaseAmount claim). Inverse: if inhibiting/knocking-down a
   miRNA RAISES a target, the miRNA DECREASES it.
4. "may/could/might" on the RELATIONSHIP ITSELF = hypothesis (incorrect). Hedging on a
   consequence while the relationship is stated = correct. METHODS/AIM framing ("to
   examine/test whether X Ys Z", "we asked whether") poses the experiment, not support.
5. Negation = incorrect even when the relation words appear: "was unable to (confirm)",
   "failed to detect/show", "did not <verb>", "no evidence that", and epithets in
   negative contexts ("kinase-dead mutant was unable to...").
6. Grounding: a flagged MISMATCH means the reader's text does not map to the claim
   entity — a strong incorrect signal unless the evidence independently names it.
7. The two entities must interact with EACH OTHER. If both are co-objects of one verb
   whose real partner is a THIRD entity ("A and B bind C"; "A and B bound to a SITE/
   promoter"), they bind C, not each other — that is no relation.

HOW TO DECIDE (surface the objection, then judge whether it actually stands):
- First state `support`: the EXACT span from the EVIDENCE that states THIS relation
  between THESE two entities. If the evidence does not explicitly and directly state it,
  support is null. Background/world knowledge is NOT support and may NOT be invented.
- Then state `objection`: the single strongest concrete reason to DOUBT the extraction
  (grounding mismatch, wrong direction, amount-not-activity, no-direct-relation,
  hypothesis-only, methods/aim framing, negation, different/ungrounded entities). Surface
  it even if a rule resolves it; null only if there is genuinely none worth considering.
  The family-level case (rule 2) is never a real objection.
- Then give the `verdict`, judging whether that objection STANDS:
  * correct — the evidence explicitly and directly supports the exact claim AND no
    objection stands. An apparent objection that an explicit rule resolves (e.g. the
    miRNA inverse-inference rule on "inhibiting the miRNA raised the target") does NOT
    stand and does NOT make the claim incorrect.
  * incorrect — support is null, OR an objection stands that defeats the claim.
  Default to incorrect unless the evidence explicitly and directly states the claim;
  background knowledge does not substitute for an explicit statement in THIS evidence.

Output JSON ONLY, in this order:
{"support": <exact evidence quote or null>, "objection": <string or null>, "verdict": "correct" | "incorrect", "confidence": "high" | "medium" | "low"}\
"""

# Reasoning-FIRST variant: identical rule body, but the model emits ONE terse,
# acceptance-routed `relation_check` clause BEFORE committing. A single-input
# STRUCTURE lever for the 4B-active model targeting over-rejection — reason toward a
# rule-licensed acceptance first, then judge. Zero extra LLM calls; user-side input
# held byte-identical to the disconfirm variant. MONO_VARIANT=disconfirm_relnature_rf.
REASONFIRST_SYSTEM_PROMPT = DISCONFIRM_SYSTEM_PROMPT.split("HOW TO DECIDE")[0] + """\
HOW TO DECIDE (reason FIRST in one short clause toward acceptance, then judge):
- First state `relation_check` — ONE short clause (<= ~15 words). It is an ACCEPTANCE check: look for a reason the relation HOLDS before doubting it.
  * [Activation]/[Inhibition]: is there a change in the target's activity STATE, OR a form an explicit rule LICENSES — an AMOUNT form (rule 3), a miRNA inverse (rule 3), a family member (rule 2), an indirect/inverse statement that still asserts the direction, or a biological PROCESS/phenotype object (e.g. proliferation, migration)? Name the licensing form, or say "no licensing form".
  * [Complex]: is a DIRECT physical bind between the two entities stated? Defer to any Extraction-provenance / relation-nature note present.
  * else: name the exact span stating this relation, or "no direct span".
  Keep it terse — one clause, not a paragraph; do not restate the rules.
- Then `support`: the EXACT evidence span stating THIS relation between THESE entities (null if none; background knowledge is NOT support and may not be invented).
- Then `objection`: the single strongest concrete reason to doubt (null if none worth considering). The family-level case (rule 2) is never a real objection; a rule-licensed form you named in relation_check is not a standing objection.
- Then the `verdict`, judging whether any objection STANDS:
  * correct — the evidence explicitly and directly supports the claim (including a rule-licensed activity / amount / family / inverse / process form) AND no objection stands.
  * incorrect — support is null, OR a standing objection defeats it (negation, hypothesis-only, methods/aim, wrong direction, amount-vs-activity with no licensing form, no-direct-relation, grounding mismatch).
  Default to incorrect ONLY when there is no licensing form AND no direct span; a licensing form named in relation_check is grounds to ACCEPT.

Output JSON ONLY, in this order:
{"relation_check": <one short clause>, "support": <exact evidence quote or null>, "objection": <string or null>, "verdict": "correct" | "incorrect", "confidence": "high" | "medium" | "low"}\
"""

def render_example(ex: dict) -> tuple[str, str]:
    """Render a base contrastive example in the variant's 4-field format, so the
    few-shots TEACH the structured output. Derives support/objection from the example.

    A correct example may carry a `considered` field — an apparent objection that an
    explicit rule resolves. Rendering it as objection=<considered>, verdict=correct
    teaches that surfacing an objection does NOT force 'incorrect' when it does not
    stand (e.g. the miRNA inverse-inference case). Without it, every correct example
    would show objection=null, training the model to equate any objection with rejection
    — the conflation we removed from the backstop."""
    user = (
        f"CLAIM: {ex['claim']}\n"
        f"EVIDENCE: {ex['evidence']}\n\n"
        f'Output JSON: {{"support": ..., "objection": ..., "verdict": ..., "confidence": ...}}'
    )
    reason = ex.get("reason") or ""
    if ex["verdict"] == "correct":
        support, objection = ex["evidence"], (ex.get("considered") or None)
    else:
        support, objection = None, (reason or "evidence does not support the exact claim")
    assistant = json.dumps(
        {"support": support, "objection": objection,
         "verdict": ex["verdict"], "confidence": ex.get("confidence", "high")},
        ensure_ascii=True,
    )
    return user, assistant


def render_example_reasonfirst(ex: dict) -> tuple[str, str]:
    """Render a base example in the reasoning-first format: a terse, acceptance-routed
    `relation_check` clause BEFORE support/objection/verdict, so the few-shots teach the
    reason-first structure. For correct examples the relation_check NAMES the licensing
    form (the `considered` rule-resolution if present); for incorrect it states the
    missing license. Derived from existing fields — no new authoring per example."""
    user = (
        f"CLAIM: {ex['claim']}\n"
        f"EVIDENCE: {ex['evidence']}\n\n"
        f'Output JSON: {{"relation_check": ..., "support": ..., "objection": ..., '
        f'"verdict": ..., "confidence": ...}}'
    )
    reason = ex.get("reason") or ""
    if ex["verdict"] == "correct":
        considered = ex.get("considered")
        relation_check = considered or "direct statement of the relation licenses acceptance"
        support, objection = ex["evidence"], (considered or None)
    else:
        relation_check = f"no licensing form: {reason}" if reason else "no direct span or licensing form"
        support, objection = None, (reason or "evidence does not support the exact claim")
    assistant = json.dumps(
        {"relation_check": relation_check, "support": support, "objection": objection,
         "verdict": ex["verdict"], "confidence": ex.get("confidence", "high")},
        ensure_ascii=True,
    )
    return user, assistant


# --- W2b: the reasoning-first prompt with verbalized confidence removed -------
#
# MEASURED: the confidence field adds ~+0.001 AUROC over the bare verdict and is
# 100% "high" on some arms. It is dead weight in the belief math, surviving only
# as a de-dup tiebreaker.
#
# This is a NEW variant, not an edit of the shipped prompt. The calibration
# profile is keyed on (model, sha256(system_prompt)), so mutating
# REASONFIRST_SYSTEM_PROMPT in place would silently orphan the fitted profile
# and drop production to an uncalibrated path. Adding a variant gives the
# no-confidence prompt its own key, leaves the shipped one byte-identical, and
# makes adopting it a separate, reversible decision.
#
# Both artifacts are DERIVED from the originals rather than copied, so they
# cannot drift; the asserts fail loudly at import if the source shape changes.
_CONFIDENCE_CLAUSE = ', "confidence": "high" | "medium" | "low"'
assert _CONFIDENCE_CLAUSE in REASONFIRST_SYSTEM_PROMPT, (
    "the reasoning-first prompt no longer contains the confidence clause verbatim; "
    "re-derive REASONFIRST_NOCONF_SYSTEM_PROMPT against its current wording"
)
REASONFIRST_NOCONF_SYSTEM_PROMPT = REASONFIRST_SYSTEM_PROMPT.replace(
    _CONFIDENCE_CLAUSE, ""
)

_EXAMPLE_CONFIDENCE_HINT = ', "confidence": ...'


def render_example_reasonfirst_noconf(ex: dict) -> tuple[str, str]:
    """`render_example_reasonfirst` with the confidence field dropped.

    Few-shots teach the output shape, so leaving `"confidence"` in the examples
    while removing it from the instruction would ask for one format and
    demonstrate another.
    """
    user, assistant = render_example_reasonfirst(ex)
    assert _EXAMPLE_CONFIDENCE_HINT in user, (
        "the reasoning-first example hint changed shape; re-derive the noconf renderer"
    )
    user = user.replace(_EXAMPLE_CONFIDENCE_HINT, "")
    payload = json.loads(assistant)
    payload.pop("confidence", None)
    return user, json.dumps(payload, ensure_ascii=True)
