"""Gate the viewer's assembled-statement comparison contract from pytest."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-belief-comparison-contract.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_belief_comparison_contract():
    proc = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "viewer statement-belief contract assertions failed:\n"
        f"  stdout: {proc.stdout}\n  stderr: {proc.stderr}"
    )
