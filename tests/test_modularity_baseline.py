"""The modularity instrument, tested for the two things it was never checked for.

`research/kernel_unification_findings.md` §7.2 item 1 records that
`scripts/modularity_baseline.py` compares hard-coded literals against themselves,
that its anchors had rotted, that `sys.settrace(None)` clobbers any outer tracer,
and that the test file meant to hold it does not exist. This is that file.

It does NOT try to make the instrument un-gameable — three of its five measures
are gameable by construction, and the script now says which and why. What it
gates is the part that CAN be defended: the duplicate-site table's citations
must name live code, and the tracer must give back what it took.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "modularity_baseline", ROOT / "scripts" / "modularity_baseline.py"
)
modularity = importlib.util.module_from_spec(_spec)
sys.modules["modularity_baseline"] = modularity
_spec.loader.exec_module(modularity)


# ---------------------------------------------------------------------------
# The anchors are claims about the code
# ---------------------------------------------------------------------------

def test_every_duplicate_site_anchor_resolves_today():
    """The rot this closes, measured: 4 of 5 line anchors pointed at nothing.

    Re-derived at the commit that replaced them: `scoring_record.py:369` landed
    on a comment about `ExecutionBody.render`, `replay.py:383` on
    `_entity_in_text`, `replay.py:394` on a bare `return False`. Only
    `scoring_record.py:392` still named what it claimed. A table whose citations
    are wrong cannot report that a duplication was removed, because it can no
    longer say where the duplication was.
    """
    assert modularity.unresolved_anchors() == []


def test_a_rotted_anchor_is_caught_rather_than_carried(monkeypatch):
    """The guard fails when it should — the half a green run cannot show.

    A test that only asserts today's table is clean would pass just as happily
    against an `unresolved_anchors` that returned [] unconditionally.
    """
    monkeypatch.setattr(modularity, "DUPLICATE_SITES", [
        {"concept": "a concept whose code was deleted",
         "live": "indra_belief.verdict::AFunctionThatWasRemoved",
         "batch": "indra_belief.comparison.replay::score_execution",
         "note": ""},
    ])
    dead = modularity.unresolved_anchors()
    assert len(dead) == 1
    assert "a concept whose code was deleted [live]" in dead[0]
    assert "AFunctionThatWasRemoved" in dead[0]


@pytest.mark.parametrize("anchor,fragment", [
    ("indra_belief.not_a_module::Thing", "no module"),
    ("indra_belief.verdict::NotDefinedHere", "has no"),
    ("indra_belief.verdict", "is not module::Symbol"),
    ("::Orphan", "is not module::Symbol"),
])
def test_each_way_an_anchor_can_be_wrong_is_reported_distinctly(anchor, fragment):
    """A dead module, a dead symbol and a malformed anchor are different repairs."""
    with pytest.raises(modularity.AnchorError) as caught:
        modularity.resolve_anchor(anchor)
    assert fragment in str(caught.value)


def test_a_dotted_attribute_chain_resolves_through_the_class():
    """Rows cite methods, not only module-level names, so the walk must recurse."""
    resolved = modularity.resolve_anchor(
        "indra_belief.comparison.replay::ReplayIndex.deterministic_result"
    )
    assert callable(resolved)


# ---------------------------------------------------------------------------
# The tracer gives back what it took
# ---------------------------------------------------------------------------

def test_the_tracer_restores_an_outer_tracer_instead_of_deleting_it():
    """`sys.settrace(None)` on exit silently disabled coverage, debuggers, profilers.

    A removed tracer produces no error — only no data — so nothing reported it.
    Anything running this instrument in-process got its own tracing turned off
    for the remainder of the run.
    """
    def outer(frame, event, arg):
        return outer

    previous = sys.gettrace()
    sys.settrace(outer)
    try:
        with modularity._Trace():
            pass
        assert sys.gettrace() is outer
    finally:
        sys.settrace(previous)


def test_the_tracer_restores_absence_too():
    """No outer tracer must stay no outer tracer — not a leaked hook."""
    previous = sys.gettrace()
    sys.settrace(None)
    try:
        with modularity._Trace():
            pass
        assert sys.gettrace() is None
    finally:
        sys.settrace(previous)


def test_the_tracer_restores_on_the_exception_path():
    """A raising body must not leave the instrument's hook installed."""
    def outer(frame, event, arg):
        return outer

    previous = sys.gettrace()
    sys.settrace(outer)
    try:
        with pytest.raises(RuntimeError):
            with modularity._Trace():
                raise RuntimeError("boom")
        assert sys.gettrace() is outer
    finally:
        sys.settrace(previous)


# ---------------------------------------------------------------------------
# The report says what it is
# ---------------------------------------------------------------------------

def test_the_duplicate_site_rows_carry_both_sides_and_a_note():
    """A row with one anchor is not a duplication claim; it is a location."""
    assert modularity.DUPLICATE_SITES, "the table is empty — say so deliberately"
    for row in modularity.DUPLICATE_SITES:
        assert set(row) == {"concept", "live", "batch", "note"}, row
        assert row["live"] != row["batch"], row["concept"]
        assert row["note"].strip(), row["concept"]
