"""Decomposed four-probe scorer — sibling to the monolithic single-call arch.

Exposes the same public shape as `indra_belief.scorers.monolithic`:

  score_evidence(statement, evidence, client) -> dict
  score_statement(statement, client) -> list[dict]

This is the explicit home for the decomposed path. The package-level
`indra_belief.score_evidence` / `score_statement` default to the **monolithic**
scorer (empirically dominant on holdout_cc, F1 0.751 vs 0.657); import from here
when you specifically want the four-probe pipeline:

    from indra_belief.scorers.decomposed import score_statement

Pipeline: parse_claim → substrate_route → four probes (subject_role, object_role,
relation_axis, scope) → ProbeBundle → adjudicate. Implemented in
`indra_belief.scorers.probes.*`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from indra_belief.scorers.probes.orchestrator import score_via_probes

if TYPE_CHECKING:
    from indra_belief.model_client import ModelClient


def score_evidence(statement, evidence, client: "ModelClient") -> dict:
    """Score one (Statement, Evidence) pair via the four-probe pipeline.

    Returns a dict with keys:
        score                float in [0, 1]
        verdict              "correct" | "incorrect" | "abstain"
        confidence           "high" | "medium" | "low"
        tier                 "decomposed"
        grounding_status     "all_match" | "flagged"
        provenance_triggered bool
        tokens               completion tokens consumed
        raw_text             decision trace
        reasons              list[ReasonCode]
        rationale            informational human-readable note
        call_log             per-LLM-call telemetry
    """
    return score_via_probes(statement, evidence, client)


def score_statement(statement, client: "ModelClient") -> list[dict]:
    """Score every evidence in a Statement via the four-probe pipeline.

    Returns one dict per evidence, in `statement.evidence` order; `[]` if the
    statement has no evidence.
    """
    evidences = list(statement.evidence or [])
    return [score_evidence(statement, ev, client) for ev in evidences]


__all__ = ["score_evidence", "score_statement"]
