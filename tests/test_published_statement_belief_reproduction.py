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

It reads multi-GB local, gitignored attempts logs and takes ~35s. It carries no
skip guard, following the precedent of its sibling
``tests/test_historical_e2b_bundle.py``, which hard-requires the same data.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reproduce_published_statement_beliefs import (
    EXPECTED_SCORES,
    PUBLISHED_ARMS,
    reproduce,
)


def test_every_published_statement_belief_rederives_exactly() -> None:
    """All 8 published prediction files, at delta exactly 0.0 — never approx."""
    report = reproduce(PUBLISHED_ARMS)
    assert [mismatch.describe() for mismatch in report.mismatches] == []
    assert report.max_delta == 0.0
    assert report.scores == EXPECTED_SCORES == 13_460
    assert report.files == 8
    assert report.ok
