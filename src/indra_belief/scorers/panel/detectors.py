"""Objection detectors — one per root failure category. Each returns an
Objection (committed defect) or None (no defect found — NOT abstention).

Surface-detectable defects go to deterministic substrate (high precision, no LLM
leniency). Only irreducibly-semantic calls use a focused LLM, asked ONE binary
question with ONLY its relevant context (non-dilutive, disconfirming-by-
construction). No detector ever abstains: the LLM answer set is binary and an
LLM failure resolves to None (no objection), which the adjudicator treats as
accept — never a third state.

Vertical-slice status: relation_exist is implemented (largest error bucket); the
other four are stubs, added + calibrated one at a time, each gated on held-out.
"""
from __future__ import annotations

import logging
import os
import re

from indra_belief.scorers.panel.types import Objection
from indra_belief.scorers.probes._llm import llm_classify

log = logging.getLogger(__name__)

# Ablation gate: PANEL_SUBSTRATE=0 disables the deterministic regex pre-checks so we
# can measure whether the (fusion-aware, co-recruitment-aware) LLM catches the same
# cases on its own. Default on.
_SUBSTRATE_ON = os.environ.get("PANEL_SUBSTRATE", "1") != "0"

_FUSION = re.compile(r"\bfusion\b|\bchimeric\b|\bfused\b", re.I)
_CONJ_THIRD = re.compile(
    r"binding of (?:both )?\w[\w-]* (?:and|or) \w[\w-]* to\b"
    r"|\bbind both\b"
    r"|\w[\w-]* (?:and|or) \w[\w-]* (?:bound|binds|associated) to\b",
    re.I,
)

# Skeptical-default + support-first — a skeptical commit-first disposition applied to the
# relation-existence question. An affirm framing ("do they interact?") re-triggers
# the model's leniency (it says yes); defaulting to comention unless an explicit
# span exists inverts that.
_REL_SYS = (
    'Decide whether a sentence EXPLICITLY states that two NAMED entities directly '
    'interact WITH EACH OTHER. Be SKEPTICAL: the answer is "comention" UNLESS the '
    'sentence contains explicit words stating one of the two entities binds, '
    'complexes with, or directly acts on the OTHER. FIRST locate that exact span. '
    'If the only connection is — both in a list, both in a title/topic phrase, both '
    'binding or acting on a THIRD entity, a gene fusion ("X-Y fusion"), or mere '
    'co-occurrence — answer "comention". Do NOT infer interaction from biological '
    'plausibility. Reply JSON {"answer": "interact" | "comention", "rationale": '
    '"the exact supporting span, or why none exists"}.'
)


def _rel_user(subj: str, obj: str, stmt_type: str, text: str) -> str:
    return (f"Entities: {subj}, {obj}\nClaimed relation: {subj} [{stmt_type}] {obj}\n"
            f'Sentence: "{text}"\nDo {subj} and {obj} interact WITH EACH OTHER here?')


def relation_exist(subj, obj, stmt_type, text, client) -> Objection | None:
    """Objection: the two claim entities do NOT interact with each other
    (co-mention / shared-third-party / gene-fusion). Targets the no_relation
    + wrong_relation(Complex) buckets — the largest shared failure."""
    t = text or ""
    if not subj or not obj:
        return None
    # substrate, high precision: a gene/chimeric fusion read as a Complex.
    if _SUBSTRATE_ON and stmt_type == "Complex":
        pair = re.compile(
            r"%s\s*[-/]\s*%s|%s\s*[-/]\s*%s"
            % tuple(re.escape(x) for x in (subj, obj, obj, subj)), re.I)
        if pair.search(t) and _FUSION.search(t):
            return Objection("relation_exist", "fusion_not_complex", "high", "substrate",
                             "gene/chimeric fusion (chromosomal), not a protein-protein complex")
    # substrate, medium: A and B both bind a shared third party (co-recruitment).
    if _SUBSTRATE_ON and _CONJ_THIRD.search(t):
        return Objection("relation_exist", "no_relation_shared_partner", "medium", "substrate",
                         "entities co-bind a shared third party, not each other")
    # LLM, focused binary (no abstain): interact vs co-mention.
    try:
        ans, rat, ok = llm_classify(
            system_prompt=_REL_SYS, few_shots=[],
            user_message=_rel_user(subj, obj, stmt_type, t),
            answer_set=frozenset({"interact", "comention"}),
            kind="relation_axis", client=client,
            max_tokens=6000)  # thinking models burn tokens before the JSON; 200 truncates
    except Exception:
        log.warning(
            "relation_exist: llm_classify failed for %r/%r [%s]; "
            "no objection -> accept-default (NOT abstain)",
            subj, obj, stmt_type, exc_info=True)
        return None  # LLM failure -> no objection -> accept-default (NOT abstain)
    if ok and ans == "comention":
        return Objection("relation_exist", "no_relation", "medium", "llm",
                         rat or "co-mentioned; no stated interaction between the two entities")
    return None
