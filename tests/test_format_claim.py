"""Golden coverage for ScoringRecord.format_claim — the canonical claim renderer
the LLM reads. A claim must NEVER contain a '?' placeholder: a missing object,
endpoint, or site is omitted, not rendered as '?' (a bare '?' reads as a
malformed/template claim and has caused real extractions to be rejected).
"""
from __future__ import annotations

import pytest
from indra.statements import (
    Activation,
    ActiveForm,
    Agent,
    Autophosphorylation,
    Complex,
    Evidence,
    Phosphorylation,
    Translocation,
)

from indra_belief.data.scoring_record import ScoringRecord

A = Agent("RHOA", db_refs={"HGNC": "667"})
B = Agent("MAP2K1", db_refs={"HGNC": "6840"})
C = Agent("PRKACA", db_refs={"HGNC": "9380"})


def _rec(stmt):
    return ScoringRecord(statement=stmt, evidence=Evidence(source_api="reach", text="s."))


@pytest.mark.parametrize(
    "stmt, expected",
    [
        # Binary — unchanged shape.
        (Phosphorylation(A, B), "RHOA [Phosphorylation] MAP2K1"),
        (Phosphorylation(A, B, residue="S", position="217"),
         "RHOA [Phosphorylation] MAP2K1 @S217"),
        (Activation(A, B), "RHOA [Activation] MAP2K1"),
        # Complex — members joined with '+'.
        (Complex([A, B]), "RHOA + MAP2K1 [Complex]"),
        (Complex([A, B, C]), "RHOA + MAP2K1 + PRKACA [Complex]"),
        # SelfModification — same entity both sides.
        (Autophosphorylation(A), "RHOA [Autophosphorylation] RHOA"),
        # Translocation — endpoint clause only when present (the RhoA fix).
        (Translocation(A, "nucleus", "cytoplasm"),
         "RHOA [Translocation] from nucleus to cytoplasm"),
        (Translocation(A, "nucleus", None), "RHOA [Translocation] from nucleus"),
        (Translocation(A, None, "membrane"), "RHOA [Translocation] to membrane"),
        (Translocation(A, None, None), "RHOA [Translocation]"),
        # Unary — no object, never '?' (the ActiveForm fix).
        (ActiveForm(A, "activity", True), "RHOA [ActiveForm]"),
    ],
)
def test_format_claim_shapes(stmt, expected):
    assert _rec(stmt).format_claim() == expected


@pytest.mark.parametrize(
    "stmt",
    [
        ActiveForm(A, "activity", True),
        Translocation(A, "nucleus", None),
        Translocation(A, None, None),
        Phosphorylation(A, B),
        Complex([A, B]),
        Autophosphorylation(A),
    ],
)
def test_claim_never_contains_question_mark(stmt):
    assert "?" not in _rec(stmt).format_claim()


def test_unary_statements_have_no_object_entity():
    # Unary types carry object == "?" — that sentinel must NOT be grounded into a
    # spurious object_entity (it would leak into the prompt's entity context).
    for stmt in (ActiveForm(A, "activity", True), Translocation(A, "nucleus", None)):
        rec = _rec(stmt)
        assert rec.object == "?"
        assert rec.object_entity is None
        assert rec.subject_entity is not None


def test_binary_and_selfmod_keep_object_entity():
    assert _rec(Phosphorylation(A, B)).object_entity is not None
    # SelfModification: object == subject, still a real grounded entity.
    selfmod = _rec(Autophosphorylation(A))
    assert selfmod.object == "RHOA"
    assert selfmod.object_entity is not None
