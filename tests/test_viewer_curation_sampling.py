"""Gate lossless, without-replacement /curate sampling from pytest."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-curation-sampling.mjs"
ROUTE = ROOT / "viewer" / "src" / "routes" / "curate" / "+page.server.ts"
PAGE = ROOT / "viewer" / "src" / "routes" / "curate" / "+page.svelte"
TRANSPORT = ROOT / "viewer" / "src" / "lib" / "server" / "indra.ts"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_curation_sampling_contract():
    proc = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        capture_output=True,
        text=True,
        cwd=ROOT / "viewer",
    )
    assert proc.returncode == 0, (
        "viewer curation-sampling assertions failed:\n"
        f"  stdout: {proc.stdout}\n  stderr: {proc.stderr}"
    )


def test_curate_route_wires_reservation_and_fresh_history_guard():
    """Pin the trust-boundary wiring around the behavior-tested ledger helpers."""
    route = ROUTE.read_text()
    page = PAGE.read_text()
    transport = TRANSPORT.read_text()

    assert 'name="draw_token" value={current.drawToken}' in page
    assert "String(fd.get('draw_token') ?? '')" in route
    assert "claimDrawReservation(" in route
    assert "getCuratorContext(locals.session.jwt, locals.session.email, true)" in route
    assert "fresh.curatedKeys.has(pairKey)" in route
    assert route.index("claimDrawReservation(") < route.index("submitCuration(args)")
    assert route.index("fresh.curatedKeys.has(pairKey)") < route.index("submitCuration(args)")
    assert "releaseDrawClaim(" in route
    assert "commitDrawReservation(" in route
    assert "submissionFailureIsDefinitive(res.status)" in route
    assert "INDRA may have accepted this curation" in route
    assert "getCuratorContext(locals.session.jwt, locals.session.email, true)" in route
    assert "sampleEvidence(datasetId, context.curatedKeys, locals.session.email)" in route
    assert "indra-belief viewer/${dataset.id}" in transport
