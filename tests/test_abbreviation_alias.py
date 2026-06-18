"""Grounding cross-check for in-text abbreviation aliases (INPUT-only).

Exercises `scoped_abbreviation_aliases` with an injected fake grounder, so the
same-grounding / ungroundable / scope / collision / degrade branches are tested
without live Gilda or a ScoringRecord.
"""
from indra_belief.data.abbreviations import scoped_abbreviation_aliases

TEXT = "forming the 5S ribonucleoprotein complex (5S RNP), which binds MDM2."
RNP = ("FPLX", "5S_RNP", "5S ribonucleoprotein complex", "object")
MDM2 = ("HGNC", "6973", "MDM2", "subject")


def _grounder(table):
    """table: dict surface(lower) -> list[(db, db_id, score)]."""
    def ground(s):
        return table.get(s.strip().lower(), [])
    return ground


def test_asserts_when_long_and_short_same_grounding():
    g = _grounder({
        "5s ribonucleoprotein complex": [("FPLX", "5S_RNP", 0.9)],
        "5s rnp": [("FPLX", "5S_RNP", 0.8)],
    })
    out = scoped_abbreviation_aliases(TEXT, [RNP, MDM2], g)
    assert out == [("5S RNP", "5S ribonucleoprotein complex", "5S ribonucleoprotein complex", "object")]


def test_asserts_when_short_is_ungroundable():
    g = _grounder({"5s ribonucleoprotein complex": [("FPLX", "5S_RNP", 0.9)]})  # short → []
    out = scoped_abbreviation_aliases(TEXT, [RNP, MDM2], g)
    assert len(out) == 1 and out[0][0] == "5S RNP"


def test_scope_guard_long_not_a_claim_entity():
    # long grounds to FPLX:5S_RNP, but the only claim entity is MDM2 → withheld
    g = _grounder({
        "5s ribonucleoprotein complex": [("FPLX", "5S_RNP", 0.9)],
        "5s rnp": [("FPLX", "5S_RNP", 0.8)],
    })
    assert scoped_abbreviation_aliases(TEXT, [MDM2], g) == []


def test_collision_short_confident_different_entity_withheld():
    g = _grounder({
        "5s ribonucleoprotein complex": [("FPLX", "5S_RNP", 0.9)],
        "5s rnp": [("HGNC", "9999", 0.95)],   # confident, DIFFERENT id → collision
    })
    assert scoped_abbreviation_aliases(TEXT, [RNP, MDM2], g) == []


def test_collision_below_threshold_is_asserted():
    g = _grounder({
        "5s ribonucleoprotein complex": [("FPLX", "5S_RNP", 0.9)],
        "5s rnp": [("HGNC", "9999", 0.40)],   # different id but NOT confident → ok
    })
    out = scoped_abbreviation_aliases(TEXT, [RNP, MDM2], g)
    assert len(out) == 1 and out[0][0] == "5S RNP"


def test_degrades_when_grounder_raises():
    def boom(_s):
        raise RuntimeError("gilda down")
    assert scoped_abbreviation_aliases(TEXT, [RNP, MDM2], boom) == []


def test_empty_when_no_abbreviation():
    g = _grounder({"mdm2": [("HGNC", "6973", 0.9)]})
    assert scoped_abbreviation_aliases("RPL5 binds MDM2.", [MDM2], g) == []
