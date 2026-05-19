"""Monolithic single-call scorer — sibling to the decomposed four-probe arch.

Exposes the same public shape as `indra_belief.scorers`:

  score_evidence(statement, evidence, client) -> dict
  score_statement(statement, client) -> list[dict]

The monolithic module's internal `score_statement(stmt, ev, client)` is
the per-evidence atomic; we re-export it here under the canonical
`score_evidence` name so the dispatch in the top-level scorer.py can
swap architectures via a single import alias.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from indra_belief.scorers.monolithic.scorer import (
    score_statement as _score_evidence_monolithic,
)

if TYPE_CHECKING:
    from indra_belief.model_client import ModelClient


def score_evidence(statement, evidence, client: "ModelClient") -> dict:
    """Score one (Statement, Evidence) pair via the monolithic pipeline."""
    return _score_evidence_monolithic(statement, evidence, client)


def score_statement(statement, client: "ModelClient") -> list[dict]:
    """Score every evidence in a Statement via the monolithic pipeline."""
    evidences = list(statement.evidence or [])
    return [score_evidence(statement, ev, client) for ev in evidences]


__all__ = ["score_evidence", "score_statement"]
