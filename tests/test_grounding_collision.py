"""Grounding-collision substrate fix (entity.py).

The bug: ``GroundedEntity._verify_raw_text`` was claim-anchored — a token
overlap with the claim's (dirty) HGNC alias list, or the claim merely being
gilda's top hit, asserted MATCH + has_grounding_signal=False even when raw_text
DENOTES A DIFFERENT GENE. The fix resolves raw_text INDEPENDENTLY and asks
whether it carries its OWN competing entity grounding:

  - a different specific gene wins      -> MISMATCH  (collision)
  - a rival ties the claim within band  -> AMBIGUOUS (ambiguous abbreviation)
  - a single clean candidate == claim   -> MATCH     (legit alias)

These are DETERMINISTIC substrate assertions verified via GroundedEntity.resolve
on labeled cases — NOT a gold-metric eval (the strict gold rewards the wrong
behavior on legit aliases). The substrate SIGNALS + surfaces candidates; it must
never make the verdict.

Requires gilda + INDRA bio_ontology; skipped if unavailable.
"""
from __future__ import annotations

import pytest

pytest.importorskip("gilda")

from indra_belief.data.entity import GroundedEntity, _competing_candidates


# raw_text DENOTES a different specific gene than the claim. Must SIGNAL.
COLLISIONS = [
    ("SH3BGRL3", "SH3BP1"),   # SH3BP1 = HGNC:10824, its own gene
    ("SRC", "SRC1"),          # SRC1 = NCOA1
    ("SLU7", "9G8"),          # 9G8 = SRSF7
    ("IL17B", "NIRF"),        # NIRF = UHRF2
    ("TM7SF2", "Ang-1"),      # Ang-1 = ANGPT1
    ("SH2D1A", "SAP"),        # SAP tops at a non-claim entity
    ("ARR3", "CAR"),          # CAR tops at SPG7; ARR3 not the band-top
]

# raw_text is an ambiguous abbreviation: claim is the top hit but a rival ties.
# Must SIGNAL.
AMBIGUOUS = [
    ("MBTPS1", "S1P"),        # ties sphingosine-1-phosphate (CHEBI)
    ("CSH1", "PL"),           # ties CSH2/HOXA10/PNLIP
]

# Legitimate aliases: raw_text independently grounds to the claim's own (db,id)
# with no competing specific gene. Must STAY MATCH with NO signal — flagging
# these would feed the verifier's over-rejection disposition.
LEGIT_MATCH = [
    ("YAP1", "YAP"),
    ("PTK2", "FAK"),
    ("APOA1", "apoA-I"),
    ("MMP9", "MMP-9"),
    ("F2", "thrombin"),
    ("JUN", "c-Jun"),
    ("RB1", "pRb"),
    ("INSR", "insulin receptor"),
    ("RRAS2", "TC21"),
]


@pytest.mark.parametrize("claim,raw_text", COLLISIONS)
def test_collision_signals_as_mismatch(claim, raw_text):
    e = GroundedEntity.resolve(claim, raw_text)
    assert e.verification_status == "MISMATCH", (
        f"{claim}<-{raw_text}: expected MISMATCH, got {e.verification_status}"
    )
    assert e.has_grounding_signal is True
    # the competing (correct) grounding is surfaced for disambiguation
    assert e.competing_candidates, "collision must surface the competing candidate(s)"


@pytest.mark.parametrize("claim,raw_text", AMBIGUOUS)
def test_ambiguous_abbreviation_signals(claim, raw_text):
    e = GroundedEntity.resolve(claim, raw_text)
    assert e.verification_status == "AMBIGUOUS", (
        f"{claim}<-{raw_text}: expected AMBIGUOUS, got {e.verification_status}"
    )
    assert e.has_grounding_signal is True
    assert e.competing_candidates, "ambiguous abbreviation must surface the rival(s)"


@pytest.mark.parametrize("claim,raw_text", LEGIT_MATCH)
def test_legit_alias_stays_clean_match(claim, raw_text):
    e = GroundedEntity.resolve(claim, raw_text)
    assert e.verification_status == "MATCH", (
        f"{claim}<-{raw_text}: legit alias must stay MATCH, got "
        f"{e.verification_status} (over-flagging regression)"
    )
    # no signal — the LLM must not be nudged toward rejecting a real alias
    assert e.has_grounding_signal is False, (
        f"{claim}<-{raw_text}: legit alias must NOT raise a grounding signal"
    )


def test_competing_candidates_excludes_claim_and_dedups():
    """The separator helper: only DISTINCT non-claim entities within band."""

    class _Term:
        def __init__(self, db, id, name):
            self.db, self.id, self.entry_name = db, id, name

    class _Match:
        def __init__(self, db, id, name, score):
            self.term = _Term(db, id, name)
            self.score = score

    results = [
        _Match("HGNC", "1", "CLAIM", 0.556),       # the claim itself -> excluded
        _Match("CHEBI", "X", "rival-lipid", 0.556),  # tied rival -> kept
        _Match("HGNC", "1", "CLAIM", 0.555),       # dup of claim -> excluded
        _Match("HGNC", "9", "far", 0.30),          # outside band -> dropped
    ]
    out = _competing_candidates(results, "HGNC", "1")
    assert [r.term.entry_name for r in out] == ["rival-lipid"]


def test_competing_candidates_empty_for_sole_claim_hit():
    """Legit sole-hit (TC21->RRAS2) yields no competitors -> stays MATCH."""

    class _Term:
        def __init__(self, db, id, name):
            self.db, self.id, self.entry_name = db, id, name

    class _Match:
        def __init__(self, db, id, name, score):
            self.term = _Term(db, id, name)
            self.score = score

    results = [_Match("HGNC", "17271", "RRAS2", 0.556)]
    assert _competing_candidates(results, "HGNC", "17271") == []
