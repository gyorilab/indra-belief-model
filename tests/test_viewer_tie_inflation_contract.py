"""Put the tie-inflation TS contract inside the declared pytest gate.

``viewer/src/lib/data/paper-tie-inflation.ts`` feeds the one figure on /paper
whose job is to argue AGAINST our best-looking number: it shows that the 2023
paper's own trapezoidal estimator inflates our reading arms by roughly half their
apparent margin, because a gate's rejected evidence collapses onto a single score
and the trapezoid interpolates across the tie block.

That makes a silently-wrong number here worse than elsewhere on the page -- it
would understate a correction we are volunteering. The Node runner asserts the
inflation and distinct-score fields are READ from the shipped artifact rather
than recomputed, that the reader-vs-RF separation the figure claims actually
holds arithmetically, and that a list of broken payloads each gate to
``unavailable`` instead of drawing a wrong mark.

Shelled through Node's native type-stripping, mirroring
``tests/test_viewer_belief_ladder_contract.py``. Separate runner and separate
module on purpose: concurrently-developed panels share the paper-literal runner,
and two nodes appending to one tail collide.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-tie-inflation-contract.mjs"


def test_tie_inflation_runner_exists() -> None:
    """A skipped Node test must not be able to hide a deleted runner."""
    assert TS_RUNNER.is_file(), f"{TS_RUNNER} is missing"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_tie_inflation_contract() -> None:
    completed = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        cwd=ROOT / "viewer",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "viewer tie-inflation contract assertions failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "fail-closed mutation cases exercised" in completed.stdout
