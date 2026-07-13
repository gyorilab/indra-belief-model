"""Gate the viewer's calibration-contract semantics from the pytest suite.

The behavioral assertions live in
viewer/scripts/test-calibration-contract.mjs and run through Node's native TypeScript
stripping. This wrapper makes schema-v2/v3 arm separation, provenance compatibility,
and deterministic predecessor selection part of the repository test gate.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-calibration-contract.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_calibration_contract():
    proc = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "viewer calibration-contract assertions failed:\n"
        f"  stdout: {proc.stdout}\n  stderr: {proc.stderr}"
    )
