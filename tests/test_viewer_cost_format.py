"""Gate the viewer's run-cost formatters from the pytest suite.

The cost-rendering surface (fmtCost / fmtCostFull in viewer/src/lib/format.ts)
is otherwise typecheck-only — `npm run check` proves the types align but never
that a KNOWN-$0 run (status="known", total_usd=null) reads "$0.00" rather than
"cost n/a" / the "price unverified" message. The behavioral assertions live in
viewer/scripts/test-cost-format.mjs (run via Node native type-stripping); this
wrapper just makes pytest fail when they fail. Skipped if node is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-cost-format.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_cost_format():
    proc = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "viewer cost-format assertions failed:\n"
        f"  stdout: {proc.stdout}\n  stderr: {proc.stderr}"
    )
