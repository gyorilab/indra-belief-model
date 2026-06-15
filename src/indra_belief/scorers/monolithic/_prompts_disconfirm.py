"""Disconfirm-first scoring prompt.

A commit-first variant that counters lenient acceptance — the model reaching a
disqualifying observation and then rationalizing it away. It attacks that
structurally:
 1. OUTPUT STRUCTURE — the model must first emit `support` (the exact evidence span
    stating THIS relation) and `objection` (the single strongest reason it's wrong),
    THEN the verdict. The disconfirming finding is committed to a field before any
    free-form rationalization can bury it.
 2. SKEPTICISM PRIOR — default 'incorrect unless the evidence explicitly and directly
    states the claim'; background knowledge is not support.
 3. DECISION BACKSTOP (code) — if the model itself states a substantive `objection`
    (not the family-level carve-out) it MAY NOT return 'correct'; the verdict is
    derived, not freely chosen — it cannot raise a defeating objection and still
    accept.

Selected via env MONO_VARIANT=disconfirm in the scorer.
"""
from __future__ import annotations

import json
import re

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
   about a promoter/transcription/reporter. miRNA-subject reducing target = Inhibition.
4. "may/could/might" on the RELATIONSHIP ITSELF = hypothesis (incorrect). Hedging on a
   consequence while the relationship is stated = correct.
5. Epithets in negative contexts are background, not evidence ("kinase-dead mutant was
   unable to..." = negative result).
6. Grounding: a flagged MISMATCH means the reader's text does not map to the claim
   entity — a strong incorrect signal unless the evidence independently names it.
7. The two entities must interact with EACH OTHER. If both are co-objects of one verb
   whose real partner is a THIRD entity ("A and B bind C"; "A and B bound to a SITE/
   promoter"), they bind C, not each other — that is no relation.

HOW TO DECIDE (commit before you rationalize):
- First determine `support`: the EXACT span from the EVIDENCE that states THIS relation
  between THESE two entities. If the evidence does not explicitly and directly state it,
  support is null. Background/world knowledge is NOT support and may NOT be invented.
- Then determine `objection`: the single strongest concrete reason the extraction is
  wrong (grounding mismatch, wrong direction, amount-not-activity, no-direct-relation,
  hypothesis-only, different/ungrounded entities). null only if there is genuinely none.
  The family-level case (rule 2) is NOT an objection.
- VERDICT RULE: incorrect if support is null OR objection is non-null. correct only when
  the evidence explicitly supports the exact claim AND there is no objection. Do not let
  background knowledge substitute for an explicit statement in THIS evidence.

Output JSON ONLY, in this order:
{"support": <exact evidence quote or null>, "objection": <string or null>, "verdict": "correct" | "incorrect", "confidence": "high" | "medium" | "low"}\
"""

_FAMILY_RE = re.compile(r"family|member|specific isoform|paralog", re.IGNORECASE)
_NULLISH = {"", "none", "null", "n/a", "na", "no objection", "no support", "-"}


def render_example(ex: dict) -> tuple[str, str]:
    """Render a base contrastive example in the variant's 4-field format, so the
    few-shots TEACH the structured output. Derives support/objection from the
    example's verdict + reason (the base examples carry an optional 'reason')."""
    user = (
        f"CLAIM: {ex['claim']}\n"
        f"EVIDENCE: {ex['evidence']}\n\n"
        f'Output JSON: {{"support": ..., "objection": ..., "verdict": ..., "confidence": ...}}'
    )
    reason = ex.get("reason") or ""
    if ex["verdict"] == "correct":
        support, objection = ex["evidence"], None
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
        except Exception:
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
    """Decision backstop. Returns (verdict, confidence, rule_applied).

    Surgical override-kill: a stated, substantive objection (not the family carve-out)
    forces 'incorrect' regardless of the model's chosen verdict — you cannot raise a
    defeating objection and still accept. Null support also forces 'incorrect' (the
    skepticism prior). Otherwise the model's verdict stands."""
    v, c = parsed.get("verdict"), parsed.get("confidence")
    obj, sup = parsed.get("objection"), parsed.get("support")
    if v is None:
        return None, None, "parse_null"
    substantive_objection = bool(obj) and not _FAMILY_RE.search(obj)
    if substantive_objection and v == "correct":
        return "incorrect", (c or "medium"), "override_killed"  # committed a defeating objection but tried to accept
    if sup is None and v == "correct":
        return "incorrect", (c or "medium"), "no_support_skeptic"
    return v, (c or "medium"), "model"
