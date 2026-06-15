"""Panel orchestrator — runs the objection detectors and adjudicates
accept-unless-objected. Binary verdict, no abstention, ever.

score_via_panel(statement, evidence, client) -> dict  (same return shape as the
monolithic + decomposed scorers, so run_rasmachine_monolithic --arch panel can
consume it unchanged)."""
from __future__ import annotations

import logging

from indra_belief.scorers.panel import detectors as D
from indra_belief.scorers.panel.adjudicator import adjudicate

log = logging.getLogger(__name__)


def _entities(statement) -> tuple[str | None, str | None]:
    ags = [a for a in statement.agent_list() if a is not None]
    if not ags:
        return None, None
    subj = ags[0].name
    obj = ags[1].name if len(ags) > 1 else ags[0].name  # self-complex degenerates to subj
    return subj, obj


def _out(verdict, score, confidence, reasons, rationale, call_log) -> dict:
    return {
        "verdict": verdict,
        "score": score,
        "confidence": confidence,
        "tier": "panel",
        "grounding_status": "panel",
        "provenance_triggered": False,
        "reasons": list(reasons),
        "rationale": rationale,
        "raw_text": "[PANEL] %s (%s) — %s" % (verdict, confidence, rationale),
        "tokens": None,
        "call_log": call_log,
    }


def score_via_panel(statement, evidence, client) -> dict:
    pop = getattr(client, "pop_call_log", lambda: [])
    pop()
    text = (getattr(evidence, "text", "") or "")
    stmt_type = type(statement).__name__
    subj, obj = _entities(statement)

    # Empty-text: defer to INDRA belief, but resolve BINARY (never abstain).
    if not text.strip():
        prior = float(getattr(statement, "belief", 0.5) or 0.5)
        verdict = "correct" if prior >= 0.5 else "incorrect"
        conf = "low" if 0.4 < prior < 0.6 else "medium"
        return _out(verdict, prior, conf, [],
                    "no evidence text; deferred to INDRA belief %.2f" % prior, pop())

    # Run the active detector panel. Each returns Objection | None (None = no
    # defect found, NOT abstention). New detectors are appended here as they
    # pass held-out validation.
    objections = [
        D.relation_exist(subj, obj, stmt_type, text, client),
    ]
    adj = adjudicate(objections)
    return _out(adj["verdict"], adj["score"], adj["confidence"],
                adj["reasons"], adj["rationale"], pop())
