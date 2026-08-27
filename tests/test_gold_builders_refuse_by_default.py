"""A gold builder must not rebuild the benchmark because someone asked for help.

MEASURED, not hypothetical: these four scripts had no argument parsing at all,
so `python scripts/build_curation_eval.py --help` printed no help and instead
redrew `eval_curation_v1.jsonl` and its sidecar in place. `--help` is the safest
thing anyone types at an unfamiliar script.

Rebuilding is legitimate — the METHOD is what reproduces, and a smaller fresh
pool on a rerun is the leakage rule correctly excluding gold drawn since. What
must not happen is a rebuilt draw quietly standing in for the one a published
number was measured on. So: help is inert, a bare run refuses, and replacing an
artifact takes `--rebuild`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDERS = [
    "build_curation_eval.py",
    "build_holdout.py",
    "build_external_gold.py",
    "build_v2_balanced.py",
]


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args],
        capture_output=True, text=True, timeout=300, cwd=ROOT, env=env,
    )


@pytest.mark.parametrize("script", BUILDERS)
def test_help_prints_usage_and_writes_nothing(script):
    """`--help` reaches argparse rather than the builder."""
    before = {p: p.stat().st_mtime_ns for p in (ROOT / "data" / "benchmark").glob("*")}
    proc = _run(script, "--help")
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "usage:" in proc.stdout, f"--help produced no usage line:\n{proc.stdout[:400]}"
    assert "--rebuild" in proc.stdout, "the destructive flag is undocumented in --help"
    after = {p: p.stat().st_mtime_ns for p in (ROOT / "data" / "benchmark").glob("*")}
    touched = [p.name for p in before if p in after and before[p] != after[p]]
    assert not touched, f"--help modified gold artifacts: {touched}"


@pytest.mark.parametrize("script", BUILDERS)
def test_bare_run_refuses_and_names_the_artifact(script):
    """A run with no flag leaves an existing artifact alone and says why."""
    proc = _run(script)
    if proc.returncode == 0:
        pytest.fail(
            f"{script} rebuilt gold with no explicit flag "
            f"(stdout tail: {proc.stdout[-200:]})"
        )
    combined = proc.stdout + proc.stderr
    assert "refusing to overwrite" in combined, combined[-400:]
    assert "--rebuild" in combined, "the refusal does not say how to proceed"
