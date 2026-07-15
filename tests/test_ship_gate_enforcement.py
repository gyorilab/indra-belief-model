"""Regression guard for the ship-gate exit-code contract.

``scripts/calibration_ship_gate.py`` is named a *gate*: it must fail the process
on missing or failed evidence. A gate that exits 0 on a non-pass cannot stop a
ship (this was the pre-2026-07-15 behaviour — main() always returned 0). This
test pins the missing-evidence case, the worst one, because "no evidence" must
never read as a green ship.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ship_gate_exits_nonzero_when_evidence_pending(tmp_path):
    # A nonexistent --test-run is PENDING. Point --out/--json at a scratch dir so
    # the tracked decision-of-record (data/results/calibration_ship_gate.md) is
    # never clobbered by the test.
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "calibration_ship_gate.py"),
            "--train-run", "unused.jsonl",
            "--test-run", str(tmp_path / "does_not_exist.jsonl"),
            "--name", "Foo",
            "--out", str(tmp_path / "gate.md"),
            "--json", str(tmp_path / "gate.json"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "ship gate exited 0 on PENDING evidence — missing evidence must not read "
        "as a green ship:\n" + result.stdout + result.stderr
    )
    # Reached the report+summary cleanly (guards the pending tuple-unpack path),
    # and still wrote the report before failing the process.
    assert "PENDING" in result.stdout
    assert (tmp_path / "gate.md").exists()
