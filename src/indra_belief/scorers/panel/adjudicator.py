"""Accept-unless-objected adjudicator — the commit-first disposition, factored.

verdict = incorrect  iff  >= 1 detector commits an objection at >= min_confidence
verdict = correct    otherwise

This is the inverse of the old decomposed adjudicator. The old one multiplied
per-probe factors and rejected on score < 0.5, so uncertainty COMPOUNDED toward
rejection. Here uncertainty (a detector returning None, or a sub-threshold
objection) contributes NOTHING — the default is accept, and only a committed,
confident defect rejects. Binary verdict, no abstain. min_confidence is the one
calibration knob (raise it to demand stronger objections; tune on a held-out
split for precision)."""
from __future__ import annotations

from indra_belief.scorers.panel.types import CONFIDENCE_RANK, Objection


def adjudicate(objections: list[Objection | None], min_confidence: str = "medium") -> dict:
    """Combine detector objections into a binary verdict. NEVER abstains."""
    floor = CONFIDENCE_RANK[min_confidence]
    found = [o for o in objections if o is not None]
    committed = [o for o in found if CONFIDENCE_RANK.get(o.confidence, 0) >= floor]

    if committed:
        top = max(committed, key=lambda o: CONFIDENCE_RANK[o.confidence])
        return {
            "verdict": "incorrect",
            "confidence": top.confidence,
            "score": None,
            "reasons": [o.defect for o in committed],
            "rationale": top.rationale or f"{top.kind}: {top.defect}",
            "objections": committed,
        }

    # No committed objection -> CORRECT. Sub-threshold objections only soften
    # confidence; they never flip the verdict (and are never abstention).
    conf = "medium" if found else "high"
    return {
        "verdict": "correct",
        "confidence": conf,
        "score": None,
        "reasons": [],
        "rationale": ("no committed objection (weak signals: "
                      + ", ".join(o.defect for o in found) + ")") if found
                     else "no objection from any detector",
        "objections": found,
    }
