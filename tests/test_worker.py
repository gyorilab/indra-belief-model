"""Tests for the viewer-spawned worker (src/indra_belief/worker.py).

The U-phase pipeline-in-viewer architecture (see
research/pipeline_in_viewer_task_graph.md) spawns this worker via
node:child_process. The SvelteKit endpoints rely on its output
contract: newline-JSON events on stdout, structured `done` /
`error` final events, exit code 0 on success.

These tests exercise three of the four verbs (ingest,
estimate-cost, register-truth-set) without LLM credentials. The
score verb is exercised only structurally — it requires a live
ModelClient + API key; the existing end-to-end smoke path
(curl /api/runs/score) is the integration test.
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import duckdb
import pytest

from indra.statements import (
    Activation,
    Agent,
    Evidence,
    Phosphorylation,
)

from indra_belief import worker as W
from indra_belief.corpus import apply_schema, ingest_statements, score_corpus


# ---------- shared fixtures ---------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    db_path = tmp_path / "corpus.duckdb"
    con = duckdb.connect(str(db_path))
    apply_schema(con)
    con.close()
    return str(db_path)


@pytest.fixture()
def tiny_indra_json(tmp_path: Path) -> str:
    """Three INDRA Statements serialized to JSON — the same shape
    `stmts_from_json_file` consumes."""
    stmts = [
        Phosphorylation(
            Agent("MAP2K1", db_refs={"HGNC": "6840"}),
            Agent("MAPK1", db_refs={"HGNC": "6871"}),
            residue="T",
            position="185",
            evidence=[Evidence(source_api="reach",
                               text="MAP2K1 phosphorylates MAPK1.",
                               pmid="11111")],
        ),
        Activation(
            Agent("RAF1", db_refs={"HGNC": "9829"}),
            Agent("MAP2K1", db_refs={"HGNC": "6840"}),
            evidence=[Evidence(source_api="reach",
                               text="RAF1 activates MAP2K1.",
                               pmid="22222")],
        ),
        Activation(
            Agent("KRAS", db_refs={"HGNC": "6407"}),
            Agent("RAF1", db_refs={"HGNC": "9829"}),
            evidence=[Evidence(source_api="reach",
                               text="KRAS activates RAF1.",
                               pmid="33333")],
        ),
    ]
    for s in stmts:
        s.belief = 0.85
    path = tmp_path / "tiny_indra.json"
    path.write_text(json.dumps([s.to_json() for s in stmts]))
    return str(path)


@pytest.fixture()
def tiny_indra_json_gz(tiny_indra_json: str, tmp_path: Path) -> str:
    """Same statements as tiny_indra_json but written as gzip(json).

    The dashboard's [ingest from .gz] affordance only works if the worker
    handles .gz transparently — this exercises that path on a small file
    instead of the real 460MB benchmark corpus.
    """
    import gzip

    src = Path(tiny_indra_json).read_bytes()
    gz_path = tmp_path / "tiny_indra.json.gz"
    with gzip.open(gz_path, "wb") as fh:
        fh.write(src)
    return str(gz_path)


@pytest.fixture()
def tiny_jsonl(tmp_path: Path) -> str:
    """Three benchmark records with `tag` field and `source_hash` for
    evidence-kind truth-set registration."""
    records = [
        {
            "matches_hash": "111111111111111111",
            "source_hash": "-1000000000000000001",
            "stmt_type": "Phosphorylation",
            "subject": "MAP2K1",
            "object": "MAPK1",
            "evidence_text": "MAP2K1 phosphorylates MAPK1.",
            "tag": "correct",
        },
        {
            "matches_hash": "222222222222222222",
            "source_hash": "-1000000000000000002",
            "stmt_type": "Activation",
            "subject": "RAF1",
            "object": "MAP2K1",
            "evidence_text": "RAF1 activates MAP2K1.",
            "tag": "correct",
        },
        {
            "matches_hash": "333333333333333333",
            "source_hash": "-1000000000000000003",
            "stmt_type": "Activation",
            "subject": "KRAS",
            "object": "RAF1",
            "evidence_text": "KRAS activates RAF1.",
            "tag": "negative_result",
        },
    ]
    path = tmp_path / "tiny_bench.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(path)


def _parse_events(captured_stdout: str) -> list[dict]:
    """Parse newline-JSON events from the worker's stdout capture."""
    events: list[dict] = []
    for line in captured_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # Non-JSON content (shouldn't happen in passing tests, but be safe)
            pass
    return events


def _find_event(events: list[dict], kind: str) -> dict:
    matches = [e for e in events if e.get("event") == kind]
    assert matches, f"no event of kind {kind!r}; saw: {[e.get('event') for e in events]}"
    return matches[-1]


def _writer_lock_path(db_path: str) -> Path:
    return Path(db_path).resolve().parent / "viewer_state" / "writer_lock.json"


def _write_writer_lock(
    db_path: str,
    *,
    token: str = "held-token",
    kind: str = "single_score",
    pid: int | None = None,
) -> Path:
    path = _writer_lock_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "kind": kind,
        "token": token,
        "pid": os.getpid() if pid is None else pid,
        "label": "test lock",
        "source_dump_id": None,
        "dataset_path": None,
        "pair_id": None,
        "architecture": None,
        "model": None,
        "started_at": "2026-01-01T00:00:00.000Z",
        "updated_at": "2026-01-01T00:00:00.000Z",
    }))
    return path


def test_score_parser_accepts_arch_and_pair_id(monkeypatch):
    """The viewer endpoint threads architecture through argv; argparse must
    preserve it for do_score before any LLM-bearing work starts."""
    seen: dict[str, object] = {}

    def fake_do_score(args):
        seen["arch"] = args.arch
        seen["paired_run_group_id"] = args.paired_run_group_id
        seen["parent_run_id"] = args.parent_run_id
        seen["skip_ingest"] = args.skip_ingest
        seen["probe_step_filter"] = args.probe_step_filter
        seen["probe_only"] = args.probe_only
        return 0

    monkeypatch.setattr(W, "do_score", fake_do_score)
    rc = W.main([
        "score",
        "--db", "corpus.duckdb",
        "--path", "data/corpora/synthetic_demo.json",
        "--source-dump-id", "synthetic_demo",
        "--model", "mock-model",
        "--scorer-version", "test",
        "--arch", "monolithic",
        "--paired-run-group-id", "pair_smoke",
        "--parent-run-id", "0123456789abcdef0123456789abcdef",
        "--skip-ingest",
        "--probe-step-filter", "object_role_probe,scope_probe",
        "--probe-only",
    ])
    assert rc == 0
    assert seen == {
        "arch": "monolithic",
        "paired_run_group_id": "pair_smoke",
        "parent_run_id": "0123456789abcdef0123456789abcdef",
        "skip_ingest": True,
        "probe_step_filter": "object_role_probe,scope_probe",
        "probe_only": True,
    }


def test_score_progress_emits_actual_cost_state(
    tmp_db: str,
    tiny_indra_json: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """The live-run UI depends on worker progress carrying observed spend,
    not only final score_run accounting."""
    import indra_belief.corpus as corpus
    import indra_belief.model_client as model_client

    def fake_score_corpus(_con, _stmts, *, on_evidence=None, **kwargs):
        assert kwargs["parent_run_id"] == "parent123"
        assert on_evidence is not None
        on_evidence(
            "stmt123456789abc",
            "ev123456789abcde",
            {
                "_cost_so_far_usd": 0.0123,
                "_cost_cap_usd": 0.02,
                "_cost_actual_increment_usd": 0.0123,
            },
        )
        return "run123"

    monkeypatch.setattr(corpus, "score_corpus", fake_score_corpus)
    monkeypatch.setattr(model_client, "ModelClient", lambda _model: object())

    args = argparse.Namespace(
        db=tmp_db,
        path=tiny_indra_json,
        source_dump_id="test_score_cost_progress",
        model="claude-sonnet-4-6",
        scorer_version="test",
        arch="monolithic",
        paired_run_group_id=None,
        parent_run_id="parent123",
        skip_ingest=True,
        cost_threshold_usd=0.02,
    )
    assert W.do_score(args) == 0
    events = _parse_events(capsys.readouterr().out)
    started = _find_event(events, "started")
    assert isinstance(started["run_id"], str)
    assert len(started["run_id"]) == 32
    assert _find_event(events, "ingest_skipped")["reason"] == "statements already exist in corpus"
    done = _find_event(events, "done")
    assert done["parent_run_id"] == "parent123"
    progress = _find_event(events, "progress")
    assert progress["cost_so_far_usd"] == 0.0123
    assert progress["cost_cap_usd"] == 0.02
    assert progress["cost_increment_usd"] == 0.0123


def test_direct_worker_ingest_respects_existing_writer_lock(
    tmp_db: str,
    tiny_indra_json: str,
    capsys: pytest.CaptureFixture[str],
):
    """Direct CLI workers must not write around the viewer's DuckDB lock."""
    lock_path = _write_writer_lock(tmp_db, kind="single_score")
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="blocked_ingest"
    )

    with pytest.raises(RuntimeError, match="DuckDB writer is busy"):
        W.do_ingest(args)

    events = _parse_events(capsys.readouterr().out)
    blocked = _find_event(events, "blocked")
    assert blocked["code"] == "writer_lock_busy"
    assert lock_path.exists()

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM statement WHERE source_dump_id='blocked_ingest'"
        ).fetchone()[0]
        assert n == 0
    finally:
        con.close()


def test_direct_worker_ingest_fails_closed_on_malformed_writer_lock(
    tmp_db: str,
    tiny_indra_json: str,
    capsys: pytest.CaptureFixture[str],
):
    """A corrupt sidecar is unsafe writer state, not permission to write."""
    lock_path = _writer_lock_path(tmp_db)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{not-json")
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="malformed_lock_ingest"
    )

    with pytest.raises(RuntimeError, match="writer_lock_malformed"):
        W.do_ingest(args)

    events = _parse_events(capsys.readouterr().out)
    blocked = _find_event(events, "blocked")
    assert blocked["code"] == "writer_lock_malformed"
    assert "writer_lock_malformed" in blocked["message"]
    assert lock_path.exists()

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM statement WHERE source_dump_id='malformed_lock_ingest'"
        ).fetchone()[0]
        assert n == 0
    finally:
        con.close()


def test_direct_worker_ingest_fails_closed_on_invalid_writer_lock_shape(
    tmp_db: str,
    tiny_indra_json: str,
    capsys: pytest.CaptureFixture[str],
):
    """Valid JSON with invalid lock fields must not be stale-deleted."""
    lock_path = _writer_lock_path(tmp_db)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({
        "kind": "single_score",
        "token": "invalid-time",
        "started_at": "2026-01-01T00:00:00.000Z",
        "updated_at": "not-a-time",
    }))
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="invalid_lock_shape_ingest"
    )

    with pytest.raises(RuntimeError, match="writer_lock_malformed"):
        W.do_ingest(args)

    events = _parse_events(capsys.readouterr().out)
    blocked = _find_event(events, "blocked")
    assert blocked["code"] == "writer_lock_malformed"
    assert "writer_lock_malformed" in blocked["message"]
    assert lock_path.exists()

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM statement WHERE source_dump_id='invalid_lock_shape_ingest'"
        ).fetchone()[0]
        assert n == 0
    finally:
        con.close()


def test_direct_worker_ingest_fails_closed_on_non_string_writer_lock_fields(
    tmp_db: str,
    tiny_indra_json: str,
    capsys: pytest.CaptureFixture[str],
):
    """Invalid JSON field types must become malformed state, not exceptions."""
    lock_path = _writer_lock_path(tmp_db)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({
        "kind": ["single_score"],
        "token": "bad-kind",
        "started_at": "2026-01-01T00:00:00.000Z",
        "updated_at": ["not-a-time"],
    }))
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="bad_field_lock_ingest"
    )

    with pytest.raises(RuntimeError, match="writer_lock_malformed"):
        W.do_ingest(args)

    events = _parse_events(capsys.readouterr().out)
    blocked = _find_event(events, "blocked")
    assert blocked["code"] == "writer_lock_malformed"
    assert "writer_lock_malformed" in blocked["message"]
    assert lock_path.exists()

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM statement WHERE source_dump_id='bad_field_lock_ingest'"
        ).fetchone()[0]
        assert n == 0
    finally:
        con.close()


def test_viewer_spawned_worker_rejects_malformed_sentinel_token(
    tmp_db: str,
    tiny_indra_json: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """The malformed sentinel token must not authorize inherited writers."""
    lock_path = _writer_lock_path(tmp_db)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{not-json")
    monkeypatch.setenv("INDRA_VIEWER_WRITER_LOCK_TOKEN", "__malformed_writer_lock__")
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="sentinel_bypass_ingest"
    )

    with pytest.raises(RuntimeError, match="writer_lock_malformed"):
        W.do_ingest(args)

    events = _parse_events(capsys.readouterr().out)
    blocked = _find_event(events, "blocked")
    assert blocked["code"] == "writer_lock_malformed"
    assert lock_path.exists()

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM statement WHERE source_dump_id='sentinel_bypass_ingest'"
        ).fetchone()[0]
        assert n == 0
    finally:
        con.close()


def test_viewer_spawned_worker_accepts_lock_token_and_preserves_parent_lock(
    tmp_db: str,
    tiny_indra_json: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """Viewer-owned locks are validated by token; the parent endpoint clears them."""
    token = "viewer-token"
    lock_path = _write_writer_lock(tmp_db, token=token, kind="ingest")
    monkeypatch.setenv("INDRA_VIEWER_WRITER_LOCK_TOKEN", token)
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="viewer_owned_ingest"
    )

    assert W.do_ingest(args) == 0
    events = _parse_events(capsys.readouterr().out)
    assert _find_event(events, "done")["n_statements"] == 3
    assert lock_path.exists()
    assert json.loads(lock_path.read_text())["token"] == token


def test_estimate_cost_does_not_require_writer_lock(
    tmp_db: str,
    tiny_indra_json: str,
    capsys: pytest.CaptureFixture[str],
):
    """Spend preflight is read-only and must remain available during writes."""
    _write_writer_lock(tmp_db, kind="ingest")

    assert W.do_estimate_cost(argparse.Namespace(path=tiny_indra_json)) == 0
    events = _parse_events(capsys.readouterr().out)
    assert _find_event(events, "done")["n_statements"] == 3


# ---------- estimate-cost -----------------------------------------------------


def test_estimate_cost_returns_per_model_estimates(
    tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """estimate-cost emits one `done` event with a per-model estimates list.

    Required: every supported model in MODEL_PRICES_PER_M_TOKENS appears
    in the result with a non-negative cost_usd. Without this, the cost
    preflight panel on the dashboard renders empty / no rows.
    """
    args = argparse.Namespace(path=tiny_indra_json)
    rc = W.do_estimate_cost(args)
    assert rc == 0

    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_statements"] == 3

    estimates = done["estimates"]
    assert isinstance(estimates, list)
    model_ids = {e["model_id"] for e in estimates}
    # The dashboard expects these to render in the preflight table.
    assert "claude-sonnet-4-6" in model_ids
    assert "claude-opus-4-7" in model_ids
    assert "gemini-2.5-flash" in model_ids
    for e in estimates:
        assert e["cost_usd"] >= 0
        assert e["n_stmts"] == 3
        assert e["n_evidences_est"] >= 3  # one evidence per stmt minimum
        assert e["n_llm_calls_est"] > 0


def test_estimate_cost_per_model_ordering_makes_sense(
    tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """Opus should never be cheaper than Sonnet on the same workload —
    if this inverts, MODEL_PRICES_PER_M_TOKENS has drifted from
    actual provider pricing."""
    rc = W.do_estimate_cost(argparse.Namespace(path=tiny_indra_json))
    assert rc == 0
    events = _parse_events(capsys.readouterr().out)
    by_model = {e["model_id"]: e for e in _find_event(events, "done")["estimates"]}
    assert by_model["claude-opus-4-7"]["cost_usd"] > by_model["claude-sonnet-4-6"]["cost_usd"]
    assert by_model["claude-sonnet-4-6"]["cost_usd"] > by_model["claude-haiku-4-5"]["cost_usd"]
    assert by_model["gemini-2.5-pro"]["cost_usd"] > by_model["gemini-2.5-flash"]["cost_usd"]


def test_estimate_cost_accepts_architecture(
    tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """The dashboard preflight must estimate the architecture it will run."""
    rc = W.do_estimate_cost(
        argparse.Namespace(path=tiny_indra_json, arch="monolithic")
    )
    assert rc == 0

    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["architecture"] == "monolithic"
    for e in done["estimates"]:
        assert e["architecture"] == "monolithic"
        assert e["n_evidences_est"] == 3
        assert e["n_llm_calls_est"] == 3


def test_estimate_cost_accepts_probe_only_filter(
    tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    rc = W.do_estimate_cost(
        argparse.Namespace(
            path=tiny_indra_json,
            arch="decomposed",
            probe_only=True,
            probe_step_filter="object_role_probe,scope_probe",
        )
    )
    assert rc == 0
    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["scoring_mode"] == "probe_only"
    assert done["probe_step_filter"] == ["object_role_probe", "scope_probe"]
    for e in done["estimates"]:
        assert e["scoring_mode"] == "probe_only"
        assert e["n_evidences_est"] == 3
        assert e["n_llm_calls_est"] == 6


# ---------- ingest ------------------------------------------------------------


def test_ingest_writes_statements_to_corpus(
    tmp_db: str, tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """ingest verb emits `loaded` then `done` events and the DB ends up
    with the expected statement count."""
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="test_ingest"
    )
    rc = W.do_ingest(args)
    assert rc == 0

    events = _parse_events(capsys.readouterr().out)
    _find_event(events, "started")
    loaded = _find_event(events, "loaded")
    done = _find_event(events, "done")
    assert loaded["n_statements"] == 3
    assert done["n_statements"] == 3
    assert done["duration_s"] >= 0

    # Verify the DB rows
    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM statement WHERE source_dump_id='test_ingest'"
        ).fetchone()[0]
        assert n == 3
    finally:
        con.close()


def test_ingest_is_idempotent(
    tmp_db: str, tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """Re-ingesting the same JSON should NOT duplicate rows. The U-phase
    score verb relies on this — it calls ingest+score on the same input
    even when [ingest] was already clicked."""
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="test_idem"
    )
    assert W.do_ingest(args) == 0
    capsys.readouterr()  # drain
    assert W.do_ingest(args) == 0  # second run

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute("SELECT COUNT(*) FROM statement").fetchone()[0]
        # 3 statements, not 6.
        assert n == 3
    finally:
        con.close()


def test_ingest_emits_progress_events(
    tmp_db: str, tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """The U7.1 SSE flow relies on `progress` events: do_ingest must wire an
    on_progress callback into ingest_statements that emits at least one event
    when n_statements % 100 == 0 (or, on a tiny corpus, at least a `loaded`
    that the dashboard renders as a denominator while waiting for progress).

    With only 3 stmts we don't hit a progress threshold (first ping is at 100),
    so this test asserts the *loaded* event carries the total — which the
    dashboard uses as the denominator before any progress arrives.
    """
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json, source_dump_id="test_progress"
    )
    assert W.do_ingest(args) == 0
    events = _parse_events(capsys.readouterr().out)
    loaded = _find_event(events, "loaded")
    assert loaded["n_statements"] == 3


def test_ingest_progress_threshold_fires_at_100(
    tmp_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Build a 100-stmt synthetic corpus and verify the first progress event
    actually fires. Without this we'd silently drop progress wiring and never
    notice — the small-corpus tests above can't catch that."""
    stmts_json = []
    for i in range(100):
        # Distinct subj/obj pairs so stmt_hashes don't collide.
        a = f"AGENT{i:03d}A"
        b = f"AGENT{i:03d}B"
        stmts_json.append(
            Phosphorylation(
                Agent(a, db_refs={"HGNC": f"{i:04d}"}),
                Agent(b, db_refs={"HGNC": f"{(i + 1):04d}"}),
                evidence=[Evidence(
                    source_api="reach",
                    text=f"{a} phosphorylates {b}.",
                    pmid=str(10000 + i),
                )],
            )
        )
    path = tmp_path / "hundred.json"
    path.write_text(json.dumps([s.to_json() for s in stmts_json]))

    args = argparse.Namespace(
        db=tmp_db, path=str(path), source_dump_id="test_thresh"
    )
    assert W.do_ingest(args) == 0
    events = _parse_events(capsys.readouterr().out)
    progress_events = [e for e in events if e.get("event") == "progress"]
    assert len(progress_events) >= 1, (
        f"expected at least one progress event; saw events: "
        f"{[e.get('event') for e in events]}"
    )
    last = progress_events[-1]
    assert last["n_statements_done"] == 100
    assert last["n_statements_total"] == 100


def test_ingest_from_gzipped_json(
    tmp_db: str, tiny_indra_json_gz: str, capsys: pytest.CaptureFixture[str]
):
    """Worker must handle `.json.gz` transparently — the dashboard's
    `[ingest from .gz]` affordance routes the gzipped path to this same
    verb without any client-side decompression."""
    args = argparse.Namespace(
        db=tmp_db, path=tiny_indra_json_gz, source_dump_id="test_gz"
    )
    rc = W.do_ingest(args)
    assert rc == 0

    events = _parse_events(capsys.readouterr().out)
    loaded = _find_event(events, "loaded")
    done = _find_event(events, "done")
    assert loaded["n_statements"] == 3
    assert done["n_statements"] == 3

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM statement WHERE source_dump_id='test_gz'"
        ).fetchone()[0]
        assert n == 3
    finally:
        con.close()


def test_estimate_cost_from_gzipped_json(
    tiny_indra_json_gz: str, capsys: pytest.CaptureFixture[str]
):
    """Cost preflight on a .gz dataset should produce the same per-model
    estimates as on an uncompressed copy — proves the gz loader is wired
    through to estimate-cost too, not just ingest."""
    rc = W.do_estimate_cost(argparse.Namespace(path=tiny_indra_json_gz))
    assert rc == 0
    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_statements"] == 3
    by_model = {e["model_id"]: e for e in done["estimates"]}
    assert by_model["claude-sonnet-4-6"]["cost_usd"] > 0


# ---------- register-truth-set -----------------------------------------------


def test_register_truth_set_writes_truth_labels(
    tmp_db: str, tiny_jsonl: str, capsys: pytest.CaptureFixture[str]
):
    args = argparse.Namespace(
        db=tmp_db,
        path=tiny_jsonl,
        truth_set_id="test_bench",
        truth_set_name="test benchmark",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    rc = W.do_register_truth_set(args)
    assert rc == 0

    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_loaded"] == 3
    assert done["n_unique_targets"] == 3
    assert done["n_missing_target"] == 0
    assert done["n_missing_field"] == 0

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        rows = con.execute(
            "SELECT target_kind, target_id, field, value_text "
            "FROM truth_label WHERE truth_set_id='test_bench' "
            "ORDER BY target_id"
        ).fetchall()
        assert len(rows) == 3
        for target_kind, _target_id, field, value_text in rows:
            assert target_kind == "evidence"
            assert field == "tag"
            assert value_text in {"correct", "negative_result"}
    finally:
        con.close()


def test_register_truth_set_distinct_targets_separate_from_n_loaded(
    tmp_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """When two JSONL rows share a source_hash (a real condition in
    eval_set_v4), they may still be distinct labels if they have different
    matches_hash statement contexts. The done event must surface target count
    and active label count separately."""
    records = [
        {"matches_hash": "1", "source_hash": "100", "tag": "correct"},
        {"matches_hash": "2", "source_hash": "200", "tag": "incorrect"},
        # Same source_hash as record 1 → collapses on natural key
        {"matches_hash": "3", "source_hash": "100", "tag": "correct"},
    ]
    path = tmp_path / "dup.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    args = argparse.Namespace(
        db=tmp_db,
        path=str(path),
        truth_set_id="test_dup",
        truth_set_name="dup",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    assert W.do_register_truth_set(args) == 0
    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_loaded"] == 3
    assert done["n_unique_targets"] == 2
    assert done["n_unique_labels"] == 3

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM truth_label WHERE truth_set_id='test_dup'"
        ).fetchone()[0]
        # DB count tracks contextual active labels. Distinct evidence targets
        # can be lower when one source_hash appears under multiple statements.
        assert n == done["n_unique_labels"]
    finally:
        con.close()


def test_register_truth_set_is_idempotent(
    tmp_db: str, tiny_jsonl: str, capsys: pytest.CaptureFixture[str]
):
    """The viewer's [register tag as truth_set] button can be clicked
    twice; the second click must NOT duplicate rows or fail."""
    base = dict(
        db=tmp_db,
        path=tiny_jsonl,
        truth_set_id="test_idem",
        truth_set_name="test idem",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    assert W.do_register_truth_set(argparse.Namespace(**base)) == 0
    capsys.readouterr()
    assert W.do_register_truth_set(argparse.Namespace(**base)) == 0

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM truth_label WHERE truth_set_id='test_idem'"
        ).fetchone()[0]
        assert n == 3
    finally:
        con.close()


def test_register_truth_set_replaces_previous_label_set(
    tmp_db: str, tiny_jsonl: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Re-registering one truth_set id means "this file is the label set".

    Without deleting stale labels for the truth_set first, a narrowed
    benchmark file leaves old target ids behind and validity overstates gold
    coverage.
    """
    base = dict(
        db=tmp_db,
        truth_set_id="test_replace",
        truth_set_name="test replace",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    assert W.do_register_truth_set(argparse.Namespace(**base, path=tiny_jsonl)) == 0
    capsys.readouterr()

    replacement = tmp_path / "single_bench.jsonl"
    replacement.write_text(json.dumps({
        "matches_hash": "111111111111111111",
        "source_hash": "-1000000000000000001",
        "tag": "correct",
    }) + "\n")
    assert W.do_register_truth_set(argparse.Namespace(**base, path=str(replacement))) == 0

    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_loaded"] == 1
    assert done["n_unique_targets"] == 1

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        rows = con.execute(
            "SELECT target_id, value_text FROM truth_label "
            "WHERE truth_set_id='test_replace'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "correct"
    finally:
        con.close()


def test_register_truth_set_tag_labels_feed_truth_present_metrics(
    tmp_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Worker JSONL registration and validity agree on benchmark tag semantics."""
    a = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    b = Agent("MAPK1", db_refs={"HGNC": "6871"})
    s = Phosphorylation(
        a,
        b,
        evidence=[
            Evidence(source_api="reach", text="positive evidence"),
            Evidence(source_api="reach", text="negative evidence"),
        ],
    )
    s.belief = 0.8

    con = duckdb.connect(tmp_db)
    try:
        ingest_statements(con, [s])
        run_id = score_corpus(
            con,
            [s],
            scorer_version="worker-tag-validity",
            score_evidence=lambda _statement, _evidence, _client: {
                "score": 0.9,
                "verdict": "correct",
                "confidence": "high",
                "reasons": [],
                "call_log": [],
            },
            with_validity=False,
        )
        evidence_by_text = dict(con.execute(
            "SELECT text, evidence_hash FROM evidence"
        ).fetchall())
        stmt_hash = con.execute(
            "SELECT stmt_hash FROM statement LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()

    labels_path = tmp_path / "tag_gold.jsonl"
    labels_path.write_text("\n".join(json.dumps(r) for r in [
        {
            "matches_hash": str(int(stmt_hash, 16)),
            "source_hash": str(int(evidence_by_text["positive evidence"], 16)),
            "tag": "correct",
        },
        {
            "matches_hash": str(int(stmt_hash, 16)),
            "source_hash": str(int(evidence_by_text["negative evidence"], 16)),
            "tag": "wrong_relation",
        },
    ]) + "\n")

    args = argparse.Namespace(
        db=tmp_db,
        path=str(labels_path),
        truth_set_id="tag_roundtrip",
        truth_set_name="tag roundtrip",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=True,
    )
    assert W.do_register_truth_set(args) == 0

    events = _parse_events(capsys.readouterr().out)
    recomputed = _find_event(events, "validity_recomputed")
    assert recomputed["run_id"] == run_id

    def read_f1_slice() -> tuple[float, dict]:
        con = duckdb.connect(tmp_db, read_only=True)
        try:
            rows = con.execute(
                "SELECT metric_name, value, slice_json::VARCHAR FROM metric "
                "WHERE run_id=? AND truth_set_id='tag_roundtrip' "
                "ORDER BY metric_name",
                [run_id],
            ).fetchall()
        finally:
            con.close()
        by_name = {name: (value, json.loads(slice_json)) for name, value, slice_json in rows}
        return by_name["truth_present.aggregate.f1"]

    f1, slice_ = read_f1_slice()
    assert f1 == pytest.approx(2 / 3)
    assert slice_["gold_fields"] == ["tag"]
    assert slice_["n_compared"] == 2
    assert slice_["tp"] == 1
    assert slice_["fp"] == 1
    assert slice_["fn"] == 0

    replacement_path = tmp_path / "tag_gold_replacement.jsonl"
    replacement_path.write_text(json.dumps({
        "matches_hash": str(int(stmt_hash, 16)),
        "source_hash": str(int(evidence_by_text["positive evidence"], 16)),
        "tag": "correct",
    }) + "\n")
    args.path = str(replacement_path)
    assert W.do_register_truth_set(args) == 0
    events = _parse_events(capsys.readouterr().out)
    replacement_done = _find_event(events, "done")
    assert replacement_done["n_loaded"] == 1
    assert replacement_done["n_unique_labels"] == 1
    assert replacement_done["n_replaced_labels"] == 2

    f1, slice_ = read_f1_slice()
    assert f1 == pytest.approx(1.0)
    assert slice_["n_compared"] == 1
    assert slice_["tp"] == 1
    assert slice_["fp"] == 0
    assert slice_["fn"] == 0

    con = duckdb.connect(tmp_db, read_only=True)
    try:
        db_count = con.execute(
            "SELECT COUNT(*) FROM truth_label WHERE truth_set_id='tag_roundtrip'"
        ).fetchone()[0]
        assert db_count == 1
        rows = con.execute(
            "SELECT metric_name, value, slice_json::VARCHAR FROM metric "
            "WHERE run_id=? AND truth_set_id='tag_roundtrip' "
            "ORDER BY metric_name",
            [run_id],
        ).fetchall()
        assert len(rows) == 3
    finally:
        con.close()


def test_register_truth_set_handles_missing_field(
    tmp_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Records without the configured field are counted as
    n_missing_field, not silently dropped. The dashboard surfaces this
    in the response's `done` event."""
    path = tmp_path / "missing_field.jsonl"
    path.write_text(json.dumps({
        "matches_hash": "1",
        "source_hash": "1",
        # no `tag` field
        "evidence_text": "no tag here",
    }) + "\n" + json.dumps({
        "matches_hash": "2",
        "source_hash": "2",
        "tag": "correct",
    }) + "\n")

    args = argparse.Namespace(
        db=tmp_db,
        path=str(path),
        truth_set_id="test_partial",
        truth_set_name="partial",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    assert W.do_register_truth_set(args) == 0
    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_loaded"] == 1
    assert done["n_missing_field"] == 1


def test_register_truth_set_handles_missing_target_hash(
    tmp_db: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Records without the configured target-hash field are counted as
    n_missing_target. Curator can re-run with --target-hash-field to
    redirect."""
    path = tmp_path / "missing_target.jsonl"
    path.write_text(json.dumps({
        "matches_hash": "1",
        # no `source_hash` field — required for evidence-kind
        "tag": "correct",
    }) + "\n")

    args = argparse.Namespace(
        db=tmp_db,
        path=str(path),
        truth_set_id="test_missing_target",
        truth_set_name="missing target",
        target_kind="evidence",
        field="tag",
        target_hash_field=None,
        recompute_latest_validity=False,
    )
    assert W.do_register_truth_set(args) == 0
    events = _parse_events(capsys.readouterr().out)
    done = _find_event(events, "done")
    assert done["n_loaded"] == 0
    assert done["n_missing_target"] == 1


# ---------- CLI dispatcher ----------------------------------------------------


def test_main_unknown_verb_returns_nonzero(capsys: pytest.CaptureFixture[str]):
    """Unknown / missing subcommands print help + return non-zero exit
    so the SvelteKit endpoint's spawn-error path triggers."""
    assert W.main([]) == 2
    out = capsys.readouterr()
    # argparse prints to stdout when --help is invoked, stderr when args invalid.
    # We just check we didn't silently succeed.


def test_main_dispatches_estimate_cost(
    tiny_indra_json: str, capsys: pytest.CaptureFixture[str]
):
    """End-to-end through the argv dispatcher — what the SvelteKit
    endpoint actually invokes when it spawns the worker."""
    rc = W.main(["estimate-cost", "--path", tiny_indra_json])
    assert rc == 0
    events = _parse_events(capsys.readouterr().out)
    _find_event(events, "done")
