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

Selected via env MONO_VARIANT=disconfirm in the scorer.
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

from indra_belief.scorers.monolithic._prompts import (
    CONTRASTIVE_EXAMPLES,  # noqa: F401 — kept importable for parity
    extract_verdict as _base_extract_verdict,
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
3. "Activation"/"Inhibition" = activity-state change, NOT expression/production/
   degradation (those are Increase/DecreaseAmount), even if the text says "activate"
   about a promoter/transcription/reporter. A miRNA reducing a target = Inhibition OR
   DecreaseAmount (accept either; don't reject a miRNA DecreaseAmount claim). Inverse:
   if inhibiting/knocking-down a miRNA RAISES a target, the miRNA DECREASES it.
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

_NULLISH = {"", "none", "null", "n/a", "na", "no objection", "no support", "-"}


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


def _norm_field(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in _NULLISH else s


def parse_structured(text: str) -> dict:
    """Pull {support, objection, verdict, confidence} from the model output.
    Falls back to the base verdict parser when the structured JSON is absent."""
    out = {"support": None, "objection": None, "verdict": None, "confidence": None}
    if not text:
        return out
    # last balanced object containing "verdict"
    for m in reversed(list(re.finditer(r"\{[^{}]*\"verdict\"[^{}]*\}", text, re.DOTALL))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            log.debug("parse_structured: skipping unparseable verdict-like span: %r", m.group(0), exc_info=True)
            continue
        out["support"] = _norm_field(obj.get("support"))
        out["objection"] = _norm_field(obj.get("objection"))
        v = obj.get("verdict")
        out["verdict"] = v.lower() if isinstance(v, str) else None
        c = obj.get("confidence")
        out["confidence"] = c.lower() if isinstance(c, str) else None
        if out["verdict"] in ("correct", "incorrect"):
            return out
    # fallback: base parser (phrase-level)
    v, c = _base_extract_verdict(text)
    out["verdict"], out["confidence"] = v, c
    return out


def derive_verdict(parsed: dict) -> tuple[str | None, str | None, str]:
    """Pass the model's committed verdict through unchanged. Returns
    (verdict, confidence, rule_applied).

    We deliberately do NOT re-derive the verdict from the parsed `support`/`objection`
    fields. Output-side determinism — code reading the objection field and flipping the
    verdict — is what made the scorer reject correct claims whose objection the model had
    already resolved (e.g. miRNA inverse-inference). The disconfirm disposition lives in
    the prompt; the verdict the model commits is final. Kept as a thin seam for telemetry
    and so callers don't change."""
    v, c = parsed.get("verdict"), parsed.get("confidence")
    if v is None:
        return None, None, "parse_null"
    return v, (c or "medium"), "model"
