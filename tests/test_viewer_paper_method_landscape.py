"""Gate the viewer's strictly unpaired 2023 paper-method reference."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-paper-method-landscape.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_paper_method_landscape_contract() -> None:
    completed = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        cwd=ROOT / "viewer",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "viewer paper-method landscape assertions failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
