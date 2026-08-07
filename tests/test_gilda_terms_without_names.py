"""Gilda terms with no `entry_name`, and what grounding does with them.

Originally found by haohangyan on the scale_up branch while running the full
60,405,451-statement INDRA DB dump; the fix is commit "Use db:id when entry_name
from gilda is None". This file is the regression evidence that branch did not
carry.

THE DEFECT. `gilda.term.Term.__init__` validates `text` and `norm_text` and
assigns `entry_name` unchecked, so None is constructible — and occurs. Measured
against the installed gilda 1.6.1 index: exactly 2 of 2,045,852 terms carry an
empty `entry_name`, both `curated`, and both the TOP hit at score 1.0 for their
own surface string:

    DDR       -> GO:0042769   (DNA damage response)
    necrosis  -> GO:0070265   (necrotic process)

Before the fix, four call sites reached `entry_name.lower()` or joined over it,
so `GroundedEntity.resolve` raised AttributeError or TypeError straight out of
`ScoringRecord.__post_init__` on ordinary biomedical text.

WHY IT WAS NEVER SEEN HERE. Every corpus in this repository was parsed —
57,969 agent objects across 49 files — and ZERO have the triggering shape. The
three real `DDR`/`necrosis` agents all carry `db_refs["TEXT"]` equal to their
own name, so raw-text verification short-circuits before touching `entry_name`.
The bug is real and live; it is simply unreached at our scale and certain to be
reached at theirs.

These tests are written against the two terms BY NAME. If a future gilda release
gives them names, the guards below skip rather than fail, and say so — a test
that silently stops testing is worse than one that says it cannot run.
"""
from __future__ import annotations

import pytest

from indra_belief.data.entity import GroundedEntity, _term_display_name


def _nameless_terms() -> dict[str, object]:
    """The surface strings whose top gilda hit currently has no entry_name."""
    import gilda

    found = {}
    for text in ("DDR", "necrosis"):
        matches = gilda.ground(text)
        if matches and matches[0].term.entry_name is None:
            found[text] = matches[0].term
    return found


requires_nameless_terms = pytest.mark.skipif(
    not _nameless_terms(),
    reason=(
        "the installed gilda index no longer has a nameless top hit for DDR or "
        "necrosis — the defect's trigger is gone from this index, so these "
        "cases cannot be posed here"
    ),
)


@requires_nameless_terms
def test_the_index_still_contains_the_terms_these_tests_are_about():
    """Name what is being relied on, so a change in gilda is visible here."""
    terms = _nameless_terms()
    assert terms, "no nameless term found"
    for text, term in terms.items():
        assert term.entry_name is None, text
        assert term.db == "GO", (text, term.db)


@requires_nameless_terms
@pytest.mark.parametrize("claim,raw_text", [
    ("discoidin domain receptor 1", "DDR"),
    ("tissue necrosis", "necrosis"),
    ("DDR1", "DDR"),
    ("necroptosis", "necrosis"),
])
def test_grounding_does_not_raise_on_a_term_with_no_name(claim, raw_text):
    """The crash itself. Each of these raised before the fix.

    Three raised AttributeError from `entry_name.lower()`; `DDR1`/`DDR` raised
    TypeError from a join over a None name in the competing-candidates block.
    """
    entity = GroundedEntity.resolve(name=claim, raw_text=raw_text)
    assert entity.verification_status in {"MATCH", "MISMATCH", "AMBIGUOUS", None}


@requires_nameless_terms
def test_the_fallback_does_not_double_prefix_a_namespaced_id():
    """`GO:GO:0042769` would reach the model, not just a log.

    Gilda ids in some namespaces already carry their prefix. This string lands
    in the provenance block and the verification note, so the duplication is
    visible to the reader being asked to judge the claim.
    """
    for term in _nameless_terms().values():
        display = _term_display_name(term)
        assert display == str(term.id), display
        assert not display.startswith(f"{term.db}:{term.db}:"), display


@requires_nameless_terms
@pytest.mark.parametrize("claim,raw_text", [
    ("necrosis", "tissue necrosis"),
    ("DDR", "discoidin domain receptors"),
])
def test_the_claim_side_fix_reaches_the_ambiguity_check(claim, raw_text):
    """The behaviour change the fix carries, pinned deliberately.

    When the CLAIM's own top term has no name, `canonical` used to be None,
    which short-circuited equivalence check 3 at its first condition. The check
    exists for exactly this case: gilda ground the claim to something whose
    canonical name differs from the claim, but the reader's span independently
    grounds to a term whose display name IS the claim name — so the span is the
    better match.

    Both cases here are that. `DDR` grounds to the DNA-damage GO term while
    "discoidin domain receptors" grounds to FPLX:DDR, whose entry_name is
    literally "DDR"; `necrosis` grounds to the necrotic-process GO term while
    "tissue necrosis" grounds to MESH:D009336 "Necrosis". Both are MATCH, and
    the auto-rejection they used to trigger was a FALSE rejection produced by
    the same defect this file is about.
    """
    entity = GroundedEntity.resolve(name=claim, raw_text=raw_text)
    assert entity.verification_status == "MATCH", entity.verification_note
    should_reject, reason = entity.should_auto_reject(f"the {raw_text} were assayed")
    assert should_reject is False, reason
