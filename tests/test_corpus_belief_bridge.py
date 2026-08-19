"""The corpus path computes beliefs, and logits reach them.

WHY THIS EXISTS
---------------
The shard runner wrote {stmt_hash: {source_hash: {verdict, confidence,
probe_delta_logit}}} and NOTHING read it. A grep for those fields across src/
and scripts/ returned writes and nothing else, so the 60M path produced verdicts
and stopped -- no belief was computed at corpus scale at all, with or without
logits. build_corpus_beliefs.py is the missing half, and these tests pin the
three things that would each be invisible in the output file itself:

  1. an UNSCORED evidence must be dropped, never credited. It is not a "wrong"
     evidence and it is not a "right" one; defaulting it either way silently
     moves every belief that contains one.
  2. TIER must come from the input job. The published cell does not carry it,
     and statement_belief gates on it -- no_text rows are excluded and the two
     deterministic tiers are credited differently from an LLM read.
  3. a margin becomes a weight ONLY through an isotonic registered for THIS
     serving stack. Borrowing another stack's curve produced saturated 0/1
     output when it was tried, and it looked entirely ordinary.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bridge = _load("_bridge_under_test", "scripts/build_corpus_beliefs.py")


JOBS = [
    {"job_id": "0:0", "input_row_index": 0, "stmt_hash": 100, "source_hash": 1,
     "source_api": "reach", "needs_llm": True, "tier1_result": None},
    {"job_id": "0:1", "input_row_index": 0, "stmt_hash": 100, "source_hash": 2,
     "source_api": "sparser", "needs_llm": True, "tier1_result": None},
    {"job_id": "1:0", "input_row_index": 1, "stmt_hash": 200, "source_hash": 3,
     "source_api": "reach", "needs_llm": False,
     "tier1_result": {"tier": "deterministic_mismatch", "verdict": "incorrect",
                      "confidence": "high"}},
    {"job_id": "2:0", "input_row_index": 2, "stmt_hash": 300, "source_hash": 4,
     "source_api": "reach", "needs_llm": True, "tier1_result": None},
    {"job_id": "2:1", "input_row_index": 2, "stmt_hash": 300, "source_hash": 5,
     "source_api": "reach", "needs_llm": True, "tier1_result": None},
]

VERDICTS = {
    "100": {"1": {"verdict": "correct", "probe_delta_logit": 7.6},
            "2": {"verdict": "correct", "probe_delta_logit": 0.4}},
    "200": {"3": {"verdict": "incorrect"}},
    # source_hash 5 is deliberately absent: scored-nothing, not scored-wrong.
    "300": {"4": {"verdict": "correct", "probe_delta_logit": -2.0}},
}


@pytest.fixture
def shards(tmp_path):
    ind, outd = tmp_path / "in", tmp_path / "out"
    ind.mkdir(); outd.mkdir()
    with gzip.open(ind / "grounded-000000.jsonl.gz", "wt") as fh:
        for job in JOBS:
            fh.write(json.dumps(job) + "\n")
    with gzip.open(outd / "verdicts-000000.json.gz", "wt") as fh:
        json.dump(VERDICTS, fh)
    return ind, outd


def _served_id():
    from indra_belief.probes.calibration import CALIBRATION_MODEL_ID
    return CALIBRATION_MODEL_ID


def _run(shards, out, *extra):
    ind, outd = shards
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_corpus_beliefs.py"),
         "--input-dir", str(ind), "--results-dir", str(outd),
         "--model", "local-gemma-4-26b", "--variant", "disconfirm_relnature_rf",
         "--out", str(out), *extra],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )


# ── row construction ──────────────────────────────────────────────────────────

def test_an_unscored_evidence_is_dropped_not_credited():
    assert bridge.evidence_rows(JOBS[4], None) is None


def test_an_unreadable_verdict_is_dropped_not_credited():
    """A parse failure is an absence of evidence, not evidence of absence."""
    assert bridge.evidence_rows(JOBS[0], {"verdict": None}) is None
    assert bridge.evidence_rows(JOBS[0], {"verdict": "garbage"}) is None


def test_tier_comes_from_the_job_because_the_cell_has_none():
    deterministic = bridge.evidence_rows(JOBS[2], {"verdict": "incorrect"})
    assert deterministic["tier"] == "deterministic_mismatch", (
        "the deterministic tier was lost; statement_belief credits it "
        "differently from an LLM read"
    )
    llm = bridge.evidence_rows(JOBS[0], {"verdict": "correct"})
    assert llm["tier"] == "llm"


def test_the_row_carries_what_the_belief_gate_reads():
    row = bridge.evidence_rows(JOBS[0], {"verdict": "correct"})
    for field in ("source_api", "verdict", "tier", "evidence_hash"):
        assert row.get(field), f"{field} missing; statement_belief reads it"


# ── the weight path ───────────────────────────────────────────────────────────

def test_without_an_isotonic_no_row_gets_a_weight():
    """The fail-safe. A margin must never be mapped through a curve fitted on a
    different serving stack -- that returns saturated 0/1 and looks ordinary."""
    stats = {"n_weighted": 0, "n_weight_failed": 0}
    rows = [bridge.evidence_rows(JOBS[0], VERDICTS["100"]["1"])]
    out = bridge.apply_weights(rows, None, stats)
    assert stats["n_weighted"] == 0
    assert "weight_of_evidence" not in out[0]


def test_with_an_isotonic_a_measured_margin_becomes_a_weight():
    from indra_belief.probes.calibration import _calibration_at, DEFAULT_CALIBRATION_PATH

    stats = {"n_weighted": 0, "n_weight_failed": 0}
    rows = [bridge.evidence_rows(JOBS[0], VERDICTS["100"]["1"])]
    out = bridge.apply_weights(rows, _calibration_at(DEFAULT_CALIBRATION_PATH), stats)
    assert stats["n_weighted"] == 1
    assert isinstance(out[0]["weight_of_evidence"], float)


def test_a_row_with_no_margin_is_left_alone_even_with_an_isotonic():
    """tier1 rows never went through the model, so there is nothing to weight."""
    from indra_belief.probes.calibration import _calibration_at, DEFAULT_CALIBRATION_PATH

    stats = {"n_weighted": 0, "n_weight_failed": 0}
    rows = [bridge.evidence_rows(JOBS[2], {"verdict": "incorrect"})]
    out = bridge.apply_weights(rows, _calibration_at(DEFAULT_CALIBRATION_PATH), stats)
    assert "weight_of_evidence" not in out[0]
    assert stats["n_weighted"] == 0


# ── end to end ────────────────────────────────────────────────────────────────

def test_end_to_end_produces_beliefs_and_drops_the_unscored_row(shards, tmp_path):
    out = tmp_path / "beliefs.json"
    proc = _run(shards, out)
    assert proc.returncode == 0, proc.stderr[-900:]
    table = json.loads(out.read_text())
    assert set(table) == {"100", "200", "300"}
    assert all(0.0 <= v <= 1.0 for v in table.values())
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["n_unscored"] == 1, "the unscored evidence was not counted"
    assert manifest["n_evidence"] == 4
    assert manifest["weighting"] == {"verdict_weight": 3}


def test_the_logit_signal_actually_moves_the_belief(shards, tmp_path):
    """The whole point. Statement 300 is verdict=correct with a NEGATIVE margin,
    so a run that consumes the margin must disagree with one that ignores it."""
    plain, weighted = tmp_path / "a.json", tmp_path / "b.json"
    assert _run(shards, plain).returncode == 0
    proc = _run(shards, weighted, "--served-model-id", _served_id())
    assert proc.returncode == 0, proc.stderr[-900:]
    a, b = json.loads(plain.read_text()), json.loads(weighted.read_text())
    assert b["300"] < a["300"] - 0.1, (
        "a correct verdict carrying a negative margin did not lose belief; "
        "the logit signal is not reaching statement_belief"
    )
    assert b["200"] == a["200"], (
        "a row with no margin moved; only measured readings may change a belief"
    )
    manifest = json.loads(weighted.with_suffix(".manifest.json").read_text())
    assert manifest["weighting"].get("probe_weight"), manifest["weighting"]
    assert manifest["n_weighted"] == 3


def test_require_calibrated_refuses_when_the_isotonic_is_missing(shards, tmp_path):
    proc = _run(shards, tmp_path / "x.json", "--require-calibrated")
    assert proc.returncode != 0
    assert "require-calibrated" in (proc.stdout + proc.stderr)


def test_the_manifest_names_the_configuration_that_produced_the_numbers(shards, tmp_path):
    """A belief is meaningless without the reader and profile behind it, and the
    manifest is the only place a consumer can read that."""
    out = tmp_path / "beliefs.json"
    assert _run(shards, out).returncode == 0
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["model"] == "local-gemma-4-26b"
    assert manifest["variant"] == "disconfirm_relnature_rf"
    assert len(manifest["prompt_sha256"]) == 64
    assert manifest["belief_profile_fitted"] is True
    assert manifest["in_call_isotonic"] is None
    assert "stmt_hash" in manifest["key"], (
        "the table's key must be stated; it is NOT re-derived as a matches_hash here"
    )


def test_no_readable_results_is_an_error_not_an_empty_table(shards, tmp_path):
    """CONTRACT CORRECTED. This test previously asserted the opposite -- exit 0
    with an empty table -- which encoded a silent failure as the requirement.

    The realistic trigger is not a deleted file: output names carry the scoring
    run's --limit (verdicts-NNNNNN.limit-K.json.gz), so a bridge run that is not
    told the same --limit misses EVERY shard. That produced a well-formed empty
    table, a confident manifest, and exit 0 -- a configuration error wearing a
    success.
    """
    ind, outd = shards
    (outd / "verdicts-000000.json.gz").unlink()
    out = tmp_path / "beliefs.json"
    proc = _run((ind, outd), out)
    assert proc.returncode != 0, "an empty table was published as success"
    assert "no shard results" in (proc.stdout + proc.stderr)
    assert not out.exists(), "a refused run still wrote a table"


def test_a_limited_scoring_run_is_joinable(shards, tmp_path):
    """The naming convention that made every lookup miss."""
    ind, outd = shards
    (outd / "verdicts-000000.json.gz").rename(outd / "verdicts-000000.limit-2.json.gz")
    out = tmp_path / "beliefs.json"
    assert _run((ind, outd), out, "--limit", "2").returncode == 0
    assert json.loads(out.read_text())


def test_a_statement_spanning_two_shards_is_refused_not_merged(shards, tmp_path):
    """dict.update() would replace a whole-statement belief with one computed
    from a fraction of its evidence, silently. The invariant holds in today's
    writer; this script cannot enforce it, so it checks it."""
    ind, outd = shards
    import gzip as _gz
    with _gz.open(ind / "grounded-000001.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(JOBS[0]) + "\n")          # stmt_hash 100 again
    with _gz.open(outd / "verdicts-000001.json.gz", "wt") as fh:
        json.dump({"100": {"1": {"verdict": "correct"}}}, fh)
    proc = _run((ind, outd), tmp_path / "beliefs.json")
    assert proc.returncode != 0
    assert "more than one shard" in (proc.stdout + proc.stderr)


def test_a_registered_isotonic_that_weighted_nothing_is_refused(shards, tmp_path):
    """A table published under a calibrated manifest whose curve touched no row
    misdescribes every belief in it."""
    ind, outd = shards
    with __import__("gzip").open(outd / "verdicts-000000.json.gz", "wt") as fh:
        json.dump({k: {kk: {"verdict": vv["verdict"]}      # strip every margin
                       for kk, vv in v.items()}
                   for k, v in VERDICTS.items()}, fh)
    proc = _run((ind, outd), tmp_path / "beliefs.json",
                "--served-model-id", _served_id())
    assert proc.returncode != 0
    assert "NOT ONE row was weighted" in (proc.stdout + proc.stderr)
