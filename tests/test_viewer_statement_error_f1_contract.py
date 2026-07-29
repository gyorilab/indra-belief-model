"""Put the statement-grain error-F1 TS contract inside the declared pytest gate.

``viewer/src/lib/data/paper-error-f1.ts`` feeds the /paper figure that NAMES the
margin the 2023 INDRA paper's own panel actually supports: on their 1,689
statements, their released labels and their own folds, the three larger reader
gates beat their RF+promoter by +0.126 to +0.142 error-class F1 with intervals
that exclude zero simultaneously, while the smallest gate does not and says so.
Pooled average precision -- the number the page used to lead with -- clears
multiplicity by a hair at best on this saturated panel; this is the margin that
survives, so a silently-wrong number here would corrupt the one claim the whole
surface exists to make honestly.

The Node runner asserts the shipped bytes validate and carry the sha256 the run
manifest signed, that the point values and margins are the verified ones, that a
disclosed LOSS is never classified as a win, that all three threshold rules reach
the figure with a non-empty oracle disclosure, that the frozen join keys stay
decoupled from the on-screen names, and that a long list of broken payloads each
gate to ``unavailable`` instead of drawing a wrong mark.

"Every rendered scalar is READ off the artifact rather than recomputed in the
loader" is asserted in the one form that can FAIL. Comparing a rendered field
against the shipped float cannot: the loader's identities hold at residual
0.000e+00, so a derived value IS the shipped double and the comparison passes
either way -- a reviewer deleted loader gates one at a time and the runner stayed
green. So each field is instead perturbed on its own against an expectation
declared in advance: an UNGATED field must render exactly what was shipped (a
loader that derived it would render its own value), and a GATED field must take
the figure down, for the stated reason. For a gated field "read" and "recomputed"
are observationally identical by construction -- the gate is the statement that
the two agree -- so the gate firing is the property that is both checkable and
stronger. Deleting any gate the probes name turns this test red; the runner also
names, in its own comment block, the loader gates it deliberately leaves to
``test-paper-render-invariants.mjs`` or to the live-value pins, and the ones that
are genuinely uncovered.

It also runs two mutations the other way round -- artifacts that stay VALID and
whose sign classification must move. ``excludesZero`` is ``ciLow > 0 || ciHigh <
0``, blind to sign by construction, and four separate clauses on this page have
branched on it two-way and read a loss as a tie; a runner that only checked
"broken things gate" would pass while every loss rendered as a win.

Shelled through Node's native type-stripping, mirroring
``tests/test_viewer_tie_inflation_contract.py`` and
``tests/test_viewer_belief_ladder_contract.py``. Separate runner and separate
module on purpose: concurrently-developed panels share the paper-literal runner,
and two nodes appending to one tail collide. The label gutters and page-wide
render invariants for this component belong to
``viewer/scripts/test-paper-render-invariants.mjs`` and are not duplicated here.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-statement-error-f1-contract.mjs"


def test_statement_error_f1_runner_exists() -> None:
    """A skipped Node test must not be able to hide a deleted runner."""
    assert TS_RUNNER.is_file(), f"{TS_RUNNER} is missing"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_statement_error_f1_contract() -> None:
    completed = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        cwd=ROOT / "viewer",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "viewer statement-error-F1 contract assertions failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    # All three counters are reported so a runner that silently stopped mutating,
    # stopped exercising the sign cases, or stopped probing read-vs-gated fields
    # would still exit 0.
    assert "fail-closed mutation cases exercised" in completed.stdout
    assert "sign-reclassification cases exercised" in completed.stdout
    assert "read-vs-gated field probes exercised" in completed.stdout
