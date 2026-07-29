"""Put the extended-Table-6 TS contract inside the declared pytest gate.

``viewer/src/lib/data/paper-table6-extended.ts`` feeds the one figure on /paper
that places our arms INSIDE the 2023 INDRA paper's own "all sources, specific"
table -- their estimator, their folds, their labels, one ranked list from 1 to
20 rather than two banded blocks. Three claims carry it, and the Node runner
pins each one against the shipped artifact:

* the ORDER (ranks 1-3 ours, their best fitted model at 4, our E2B at 11, Belief
  Orig 16, SVC 20), pinned independently of the artifact's own
  ``checks.expected_ranks`` and then required to agree with it both ways;
* the LICENCE for putting our rows in their table -- the <=0.0016 agreement
  between the ten rows we re-ran from their released code and the values they
  printed -- asserted as a partition, because ten of the twenty rows were never
  re-run and an earlier draft claimed the bound over rows it could not cover;
* the OBJECTION we raise against ourselves -- their trapezoidal estimator hands
  coarse-scored arms area no threshold reaches, worth +0.0097..+0.0143 to our
  reader arms against -0.0008..+0.0006 to their models, with our own INDRA CoGEx
  hybrid (1,176 distinct scores, +0.0006) sitting on THEIR side as the control
  that the effect tracks tie density rather than authorship.

Every cross-group gift comparison in the runner takes an absolute value on the
paper side explicitly and asserts the reader side is positive explicitly: their
gifts straddle zero, so a comparison of bare extremes would be reading a
separation out of a number whose sign it never checked. Sign-blindness has
shipped four times on this page.

The runner also exercises broken payloads -- a missing row, a duplicate rank, a
gap in the sequence, an origin outside the enum, a printed-only row handed a
complete tie-robust block, a deviation pushed past the bound, a fold SD removed,
the density cut slid to move the control -- each of which must gate the figure to
``unavailable`` rather than draw a wrong ranked list, and it prints the case
count so a suite that silently stopped mutating cannot still exit 0.

Shelled through Node's native type-stripping, mirroring
``tests/test_viewer_tie_inflation_contract.py``. Separate runner and separate
module on purpose: concurrently-developed panels share the paper-literal runner,
and two nodes appending to one tail collide.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-table6-extended-contract.mjs"


def test_table6_extended_runner_exists() -> None:
    """A skipped Node test must not be able to hide a deleted runner."""
    assert TS_RUNNER.is_file(), f"{TS_RUNNER} is missing"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_table6_extended_contract() -> None:
    completed = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        cwd=ROOT / "viewer",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "viewer extended-Table-6 contract assertions failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "fail-closed mutation cases exercised" in completed.stdout
