"""Relation-nature characterization for [Complex] extraction scoring.

A focused step that classifies, for a [Complex] claim, the relationship the
evidence sentence actually asserts between the two entities. When that nature is
not direct physical binding — a gene fusion, a signaling cascade, co-binding to a
shared third entity, or a title/aim mention — it returns a short note that the
disconfirm verdict reads as a grounding mismatch and rejects on. Known aliases
for each claim entity are grounded via Gilda and supplied to the model so a
complex stated under a synonym or descriptive name is recognized rather than
mistaken for a non-binding relationship.

The note is emitted only to reject: a stated physical bind, an unparseable
answer, or a non-Complex claim yield no note and leave the holistic verdict
untouched.
"""
from __future__ import annotations

import logging
import re

from indra_belief.scorers.probes._llm import _extract_json

log = logging.getLogger(__name__)

_RELATION_SYSTEM = (
    "You characterize the RELATIONSHIP a biomedical EVIDENCE sentence actually ASSERTS "
    "between two named entities — nothing else. Pick the single BEST-fitting nature:\n"
    "- physical_binding: the sentence states, as a finding, that the two entities directly "
    "bind / form a complex / physically associate WITH EACH OTHER. Choose this EVEN IF the "
    "sentence also names a downstream consequence (causing/activating/inducing/regulating) "
    "or a shared third partner — a trailing effect or third-party clause does NOT override a "
    "stated bind between the two claim entities.\n"
    "- fusion_construct: the two are named as a gene FUSION or chimeric protein "
    '("X-Y fusion", "X-Y") — ONE molecule, not two binding partners.\n'
    "- signaling_cascade: a functional/regulatory relationship (pathway, axis, activates, "
    "induces, downstream of). Choose ONLY when the sentence states NO direct bind between the "
    "two claim entities.\n"
    "- co_binding_third: each entity binds or acts on a shared THIRD entity, not each other. "
    "Choose ONLY when the sentence states NO direct bind between the two claim entities "
    "themselves.\n"
    "- topic_or_aim: the pairing appears only in a title/topic phrase, or inside an aim/"
    "methods clause ('to detect binding of...'), not an asserted result.\n"
    "- other: none of the above. An entity may appear under any listed alias, synonym, or "
    "descriptive name; do NOT answer other merely because a claim's literal gene symbol is "
    "not the exact token used — if an alias of either entity is present and a bind verb is "
    "stated, that is physical_binding.\n"
    "Judge ONLY what THIS sentence asserts (textual) — NEVER background knowledge of what the "
    'proteins do. Output JSON ONLY: {"nature": <one>, "span": <exact words that decide it>}.'
)

_NATURE_LABEL = {
    "fusionconstruct": "a gene FUSION / chimeric construct (one molecule)",
    "signalingcascade": "a signaling/regulatory cascade (functional, not physical binding)",
    "cobindingthird": "co-binding to a shared THIRD entity (not each other)",
    "topicoraim": "only a title/topic phrase or an aim/methods clause (not an asserted result)",
    "other": "not a direct physical interaction",
}


def _gilda():
    """Optional Gilda accessor; returns None where Gilda is unavailable."""
    try:
        from indra_belief.tools import gilda_tools as gt
        return gt
    except Exception:
        log.debug("gilda unavailable; relation aliases skipped", exc_info=True)
        return None


def _user_message(subj: str, obj: str, text: str,
                  gs: dict | None = None, go: dict | None = None) -> str:
    """Question for the characterizer, with grounded aliases for each entity when available."""
    def _ent(name, g):
        if not g:
            return name
        al = [a for a in g["aliases"] if a.lower() != name.lower()][:6]
        return f"{name} (also known as: {', '.join(al)})" if al else name
    return (f'Entities: {_ent(subj, gs)}, {_ent(obj, go)}\nSentence: "{text}"\n'
            f"What relationship does the sentence assert between {subj} and {obj}?")


def _norm_nature(n) -> str | None:
    if not isinstance(n, str):
        return None
    return re.sub(r"[^a-z]", "", n.lower()) or None  # punctuation/whitespace-only -> None


def resolve_relation_nature(subj: str, obj: str, stmt_type: str, text: str, client,
                            max_tokens: int = 3000) -> str:
    """Return a rejection note when the asserted relationship between the two claim
    entities is not direct physical binding, else "" (a stated bind, an unparseable
    answer, or a non-Complex claim leave the holistic verdict untouched). Claim-entity
    aliases are grounded via Gilda and supplied to the model so a complex stated under a
    synonym or descriptive name is recognized."""
    if (stmt_type != "Complex" or not subj or not obj
            or subj == "?" or obj == "?" or not (text or "").strip()):
        return ""
    gt = _gilda()
    gs = gt.entity_grounding(subj) if gt else None
    go = gt.entity_grounding(obj) if gt else None
    try:
        resp = client.call(
            system=_RELATION_SYSTEM,
            messages=[{"role": "user", "content": _user_message(subj, obj, text, gs, go)}],
            max_tokens=max_tokens, temperature=0.1,
            response_format={"type": "json_object"}, reasoning_effort="none",
            kind="relation_nature",
        )
    except Exception:
        log.warning(
            "relation_nature: client.call failed for (%r, %r); leaving holistic verdict untouched",
            subj, obj, exc_info=True,
        )
        return ""
    content = (getattr(resp, "content", None) or getattr(resp, "raw_text", None) or "").strip()
    o = _extract_json(content)
    if not isinstance(o, dict):
        return ""
    nat = _norm_nature(o.get("nature"))
    if nat is None or nat == "physicalbinding":
        return ""
    if nat not in _NATURE_LABEL:
        log.warning("relation_nature: unrecognized nature %r; treating as non-binding", nat)
    span = str(o.get("span", "") or "")[:160]
    label = _NATURE_LABEL.get(nat, "not direct physical binding")
    return ("Relation nature (resolved): the evidence asserts %s%s. A [Complex] claim requires a "
            "stated DIRECT PHYSICAL BIND between %s and %s — that is a grounding MISMATCH here, so "
            "the [Complex] extraction is unsupported." % (
                label, (' — "%s"' % span if span else ""), subj, obj))
