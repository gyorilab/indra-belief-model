"""The behavioural freeze on the statement-belief aggregator.

``statement_belief.py`` produced 13460 published scores across four comparison
arms and two panels. Historically it was frozen by *byte* digest, which is a
strong claim about the file and a weak one about the numbers: it forbids a
comment and permits nothing, so the file could not be corrected even where its
output was provably unaffected.

This test makes the freeze behavioural instead. It re-runs the live aggregator
over the frozen observations each bundle's own manifest names — verifying every
input against the sha256 and byte count the manifest declares — and requires
EXACT float equality with the shipped prediction files. What is protected is
the property the published numbers actually depend on.

WHAT RUNS WHERE, and why it is not one rule. It reads multi-GB local, gitignored
attempts logs and takes ~35s, so on `ubuntu-latest` there is nothing to read.
The old resolution was to carry no skip guard at all, following
``tests/test_historical_e2b_bundle.py``; that made a fresh checkout die inside
``manifest_path.read_bytes()`` with a bare ``FileNotFoundError`` naming one file
and explaining nothing. The opposite resolution — skip whenever anything is
missing — lets the freeze silently stop running on the one machine that can run
it. Both concerns are real, so ``published_data_state`` is three-valued and this
file treats each value differently:

  * data present -> the freeze runs, and a difference is a FAILURE.
  * data wholly absent -> skipped, with a reason naming what is needed. This is
    CI and any fresh checkout.
  * data partly absent -> ``PartialPublishedData``, loudly, because that is a
    deletion or a half-finished copy and never a clean skip.

``test_the_freeze_declares_what_it_covers`` closes the remaining gap. It runs
with no data at all, on every machine, and asserts the CENSUS the freeze is
measured against — four arms, two panels, 1,689 + 1,676 statements, 13,460
scores. Deleting the corpus can stop the reproduction from running; it cannot
quietly shrink what the reproduction claims to cover.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reproduce_published_statement_beliefs import (
    EXPECTED_EXECUTIONS,
    EXPECTED_READER_STATEMENTS,
    EXPECTED_SCORES,
    EXPECTED_STATEMENTS,
    PANELS,
    PUBLISHED_ARMS,
    WHOLLY_ABSENT,
    published_data_state,
    reproduce,
)

# NO module-level `pytestmark` here, deliberately, although every other
# corpus-reading module carries one. Two of the three tests below are written to
# run WITHOUT the corpus — that is their whole point — and a module-level mark
# would skip them on exactly the machine they exist to cover. The guard goes on
# the one test that reads data.
#
# `published_data_state` is also narrower than `_local_artifacts.requires`: it
# walks each published bundle's manifest and every input path that manifest
# NAMES, so a manifest present without its attempts log is caught as PARTIAL
# rather than read as available.
requires_published_data = pytest.mark.skipif(
    published_data_state() == WHOLLY_ABSENT,
    reason=(
        "data/comparison/models/*/manifest.json and the attempts logs they name "
        "are absent (gitignored published artifacts); run locally with "
        "PYTHONPATH=src .venv/bin/python -m pytest "
        "tests/test_published_statement_belief_reproduction.py"
    ),
)


@requires_published_data
def test_every_published_statement_belief_rederives_exactly() -> None:
    """All 8 published prediction files, at delta exactly 0.0 — never approx."""
    report = reproduce(PUBLISHED_ARMS)
    assert [mismatch.describe() for mismatch in report.mismatches] == []
    assert report.max_delta == 0.0
    assert report.scores == EXPECTED_SCORES == 13_460
    assert report.files == 8
    assert report.ok


def test_the_freeze_declares_what_it_covers() -> None:
    """The census, asserted with no data on disk — so a skip cannot shrink it.

    A skip guard's failure mode is that the freeze stops running and nothing
    says so. This cannot detect that on a machine with no corpus, and does not
    pretend to. What it CAN do is hold the declaration still: if the arms, the
    panels or the counts are edited down to match a smaller run, the edit fails
    here on every machine including the ones that skip the reproduction. The
    numbers are the bundles' own census, and a drift in them is a substrate
    change rather than a rounding difference.
    """
    assert PUBLISHED_ARMS == ("gemma_4_26b", "glm_5", "gemma_4_31b", "gemma_4_e2b")
    assert PANELS == ("all_source", "reader")
    assert (EXPECTED_STATEMENTS, EXPECTED_EXECUTIONS) == (1689, 33361)
    assert EXPECTED_READER_STATEMENTS == 1676
    assert EXPECTED_SCORES == len(PUBLISHED_ARMS) * (
        EXPECTED_STATEMENTS + EXPECTED_READER_STATEMENTS
    ) == 13_460


def test_a_partial_checkout_is_never_reported_as_absent() -> None:
    """The three-valued predicate, driven at its boundary.

    `published_data_state` is the only thing standing between "CI has no data"
    and "somebody deleted an arm", and those must not resolve the same way. The
    probe touches no disk: it re-runs the predicate over an arm tuple carrying
    one name that was never published, which presents the predicate with exactly
    the shape of an arm having gone missing.
    """
    from scripts import reproduce_published_statement_beliefs as module

    if published_data_state() == WHOLLY_ABSENT:
        pytest.skip("no published data here; the partial state cannot be posed")
    with pytest.raises(module.PartialPublishedData) as caught:
        published_data_state((*PUBLISHED_ARMS, "an_arm_that_was_never_published"))
    assert "an_arm_that_was_never_published" in str(caught.value)
