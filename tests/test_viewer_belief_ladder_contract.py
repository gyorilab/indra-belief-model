"""Put the belief-model-ladder TS contract inside the declared pytest gate.

The ladder's viewer surface (``viewer/src/lib/data/paper-belief-ladder.ts`` +
``viewer/src/lib/server/paper-belief-ladder.ts`` +
``viewer/src/lib/components/BeliefModelLadder.svelte``) draws every number through
a fail-closed validator. That validator is exercised by its own Node runner,
``viewer/scripts/test-belief-ladder-contract.mjs``, which asserts the shipped
bytes validate, that their sha256 is the one the run manifest signed, that the
display sort is monotone in average precision, that the kind -> hue mapping
returns the page-wide tokens, that the gate's referents are derived (never typed)
and land inside the artifact's own shipped range, and that a long list of broken
arithmetic identities each THROW instead of drawing a wrong bar.

This module shells that runner through Node's native type-stripping, exactly as
``tests/test_viewer_paper_literal_contract.py`` does for its own runner, so the
TypeScript contract fails ``pytest -q`` rather than only ``npm run``. It is a
separate file (and a separate runner) on purpose: the paper-literal contract
files are shared with concurrently-developed panels, and two nodes appending to
one tail collide.

The artifact-side arithmetic (deltas re-derived from the named prediction files,
the guardrail range, the noisy-OR formula string) is already gated by
``tests/test_viewer_paper_literal_contract.py``; this file deliberately does not
duplicate it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-belief-ladder-contract.mjs"


def test_belief_ladder_runner_exists() -> None:
    """A skipped Node test must not be able to hide a deleted runner."""
    assert TS_RUNNER.is_file(), f"{TS_RUNNER} is missing"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_belief_ladder_contract() -> None:
    completed = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        cwd=ROOT / "viewer",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "viewer belief-model-ladder contract assertions failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    # The runner reports how many fail-closed mutations it exercised; a runner
    # that silently stopped mutating would still exit 0.
    assert "fail-closed mutation cases exercised" in completed.stdout
