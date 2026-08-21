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
    assert manifest["isotonic"] is None
    assert "in_call_isotonic" not in manifest, (
        "an `in_call` key naming a separate-probe curve contradicts margin_route"
    )
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


def test_the_reason_nothing_was_weighted_reaches_the_refusal(shards, tmp_path,
                                                             monkeypatch):
    """The recorded reason is discarded exactly where it was needed.

    `apply_weights` keeps the FIRST weighting exception because a bare counter
    hid a total failure behind a number -- an in-call artifact that loaded fine
    and then raised "X column order does not match probe_ids" on every row, with
    n_weight_failed climbing as the only symptom. But the reason was only ever
    written to the manifest, and the n_weighted==0 refusal aborts before the
    manifest exists, so the one run that most needs the diagnosis is the one run
    that never gets it: the operator is told a count and left to guess.
    """
    import sys

    from indra_belief.probes import calibration as calib

    def explode(*_args, **_kwargs):
        raise ValueError("X column order does not match probe_ids")

    monkeypatch.setattr(calib, "calibrate_probe", explode)

    ind, outd = shards
    out = tmp_path / "beliefs.json"
    monkeypatch.setattr(sys, "argv", [
        "build_corpus_beliefs.py",
        "--input-dir", str(ind), "--results-dir", str(outd),
        "--model", "local-gemma-4-26b", "--variant", "disconfirm_relnature_rf",
        "--served-model-id", _served_id(), "--out", str(out),
    ])

    with pytest.raises(SystemExit) as excinfo:
        bridge.main()

    message = str(excinfo.value)
    assert "NOT ONE row was weighted" in message
    assert "X column order does not match probe_ids" in message, (
        f"the recorded reason was dropped where it was needed: {message}"
    )
    assert not out.exists()


# ── which curve, and for which run ────────────────────────────────────────────


def test_an_in_call_margin_is_weighted_through_the_in_call_isotonic(shards, tmp_path):
    """Two registries, two quantities, and the loader accepts either artifact.

    `probe_delta_logit` holds an IN-CALL margin when the variant reads the label
    from the scoring response and a separate-PROBE margin when it does not. This
    script resolved BOTH through `sentence_calibration_path_for` -- the probe
    registry -- so an in-call margin was mapped through a curve whose knots span
    -1.70..+1.61 while in-call margins run ~3x wider. MEASURED at the documented
    median |13.22|: the probe curve returns p_hat 0.0000/1.0000 where the
    correct curve returns 0.164/0.758. Every row is weighted, n_weighted climbs,
    nothing raises, and every statement is pushed to 0 or 1. The live path
    refuses that swap by name; this one committed it.
    """
    out = tmp_path / "beliefs.json"
    proc = _run(shards, out, "--variant", "verdict_only",
                "--served-model-id", _served_id())
    assert proc.returncode == 0, proc.stderr[-900:]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["margin_route"] == "pol.verdict_incall"
    assert manifest["isotonic"] == "incall_calibration_local_mlx.json", (
        "an in-call margin was weighted through the separate-probe curve"
    )
    assert manifest["n_weighted"] == 3


def test_the_probe_route_still_resolves_the_probe_curve(shards, tmp_path):
    """The other half: a variant that reads no in-call margin keeps the probe
    registry, because under --probe that IS where its deltas came from."""
    out = tmp_path / "beliefs.json"
    proc = _run(shards, out, "--served-model-id", _served_id())
    assert proc.returncode == 0, proc.stderr[-900:]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["margin_route"] == "pol.verdict_direct"
    # ROUTE-NEUTRAL KEY. `in_call_isotonic` named a curve from the separate-probe
    # registry beside a margin_route saying the margins were not in-call: the two
    # fields cannot both be right, and this route is reachable through the
    # runbook's own `--variant disconfirm_relnature_rf`.
    assert manifest["isotonic"] == "sentence_probe_calibration.json"
    assert "in_call_isotonic" not in manifest
    assert "[weights] pol.verdict_direct isotonic: registered" in proc.stdout, (
        f"the console still names the other route: {proc.stdout[-400:]}"
    )


def test_a_join_that_kept_nothing_is_refused(shards, tmp_path):
    """The "empty table is never a legitimate answer" guard counted FILES.

    A shard whose results file resolves but joins to nothing -- a scoring window
    where every cell finalized as "error" -- still incremented n_shards, so the
    guard passed and `{}` was published with exit 0. That is not a harmless
    empty file: build_table omits unscored statements precisely because INDRA's
    Statement.from_json defaults a missing belief to 1.0, so an empty table
    makes the whole corpus read as certainly true.
    """
    ind, outd = shards
    with gzip.open(outd / "verdicts-000000.json.gz", "wt") as fh:
        json.dump({stmt: {source: {"verdict": "error", "confidence": None,
                                   "error": "ConnectError", "attempts": 4}
                          for source in cells}
                   for stmt, cells in VERDICTS.items()}, fh)
    out = tmp_path / "beliefs.json"
    proc = _run((ind, outd), out)
    assert proc.returncode != 0, "an empty table was published as success"
    assert "NOT ONE statement" in (proc.stdout + proc.stderr)
    assert not out.exists()


def test_the_bridge_refuses_a_variant_the_shard_was_not_scored_with(shards, tmp_path):
    """--model/--variant are assertions about a run that already happened.

    Scored under `verdict_only` and believed under `disconfirm_relnature_rf`,
    every belief is computed from log-LRs fitted for a prompt the run never
    sent, and the manifest then publishes that prompt's sha as the provenance of
    the numbers. The scoring run records what it did beside each shard, so the
    assertion is checkable rather than trusted.
    """
    import hashlib

    from indra_belief.scorers.monolithic import scorer as mono

    ind, outd = shards
    sha = hashlib.sha256(
        mono.VARIANTS["verdict_only"].system_prompt.encode("utf-8")
    ).hexdigest()
    (outd / "verdicts-000000.meta.json").write_text(json.dumps({
        "model": "local-gemma-4-26b",
        "served_model_id": None,
        "variant": "verdict_only",
        "prompt_sha256": sha,
        "margin_route": "pol.verdict_incall",
        "limit": None,
        "stmt_hash_filter": None,
    }))
    proc = _run((ind, outd), tmp_path / "beliefs.json")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "a table was computed under a variant the shards were not scored with"
    )
    # `"variant" in output` also passed on a model or a served-id mismatch, and
    # on any message that happens to use the word. The refusal has to name the
    # field that disagreed AND both sides of the disagreement.
    assert "was scored with variant='verdict_only'" in output, output[-900:]
    assert "this run asserts 'disconfirm_relnature_rf'" in output, output[-900:]


def test_the_manifest_separates_a_failed_read_from_a_missing_one(shards, tmp_path):
    """`n_unscored` conflated a join problem with a scored-and-failed evidence.

    A cell that says verdict "error" is a shard the runner published with that
    evidence lost, and every one of them lowers its statement's belief by
    removing a read -- a wrong number, not an absent one. A missing cell is a
    join problem instead. One counter for both meant neither was actionable.
    """
    ind, outd = shards
    verdicts = json.loads(json.dumps(VERDICTS))
    verdicts["100"]["2"] = {"verdict": "error", "confidence": None,
                            "error": "ConnectError", "attempts": 4}
    with gzip.open(outd / "verdicts-000000.json.gz", "wt") as fh:
        json.dump(verdicts, fh)

    out = tmp_path / "beliefs.json"
    assert _run((ind, outd), out).returncode == 0
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["n_unscored"] == 2
    assert manifest["n_error_cell"] == 1, "a failed read looked like a missing one"
    assert manifest["n_missing_cell"] == 1


# ── the assertions stage 4 makes about a run that already happened ────────────


def _sidecar(outd, **overrides):
    """The sidecar the runner writes beside a shard, for THIS test module's run."""
    import hashlib as _hashlib

    from indra_belief.scorers.monolithic import scorer as mono

    meta = {
        "model": "local-gemma-4-26b",
        "served_model_id": None,
        "variant": "disconfirm_relnature_rf",
        "prompt_sha256": _hashlib.sha256(
            mono.VARIANTS["disconfirm_relnature_rf"].system_prompt.encode("utf-8")
        ).hexdigest(),
        "margin_route": None,
        "limit": None,
        "stmt_hash_filter": None,
    }
    meta.update(overrides)
    (outd / "verdicts-000000.meta.json").write_text(json.dumps(meta))
    return meta


def test_a_sidecar_less_shard_joins_and_is_counted_as_unverified(shards, tmp_path):
    """The 60M run in flight was launched before the sidecar existed.

    Its published shards carry none, so the bridge has to join them -- they are
    valid -- while saying plainly that their configuration was NOT confirmed.
    The one thing absence must never do is read as agreement, which is why the
    assertion loop skips a key that is not recorded and this counter reports the
    size of the unverified part of the table.
    """
    out = tmp_path / "beliefs.json"
    assert _run(shards, out).returncode == 0
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["n_shards_without_provenance"] == 1, (
        "a shard scored before the sidecar existed was reported as verified"
    )

    ind, outd = shards
    _sidecar(outd)
    second = tmp_path / "verified.json"
    assert _run((ind, outd), second).returncode == 0
    verified = json.loads(second.with_suffix(".manifest.json").read_text())
    assert verified["n_shards_without_provenance"] == 0, (
        "the counter does not distinguish a recorded run from an unrecorded one"
    )


def test_a_sidecar_the_bridge_could_not_READ_is_refused_not_joined(shards, tmp_path):
    """The unverified path is for shards that recorded nothing, not for faults.

    `read_shard_provenance` answered None for both, so a sidecar truncated by a
    crash or unreadable on shared scratch joined as "scored before sidecars
    existed" and every assertion this build makes about it compared nothing --
    the fault read as agreement. It is a refusal now, and a refusal is a stated
    one: a traceback out of the join loop leaves the operator to guess which of
    ~1,200 shards it was and whether anything was published.
    """
    ind, outd = shards
    _sidecar(outd)
    meta = outd / "verdicts-000000.meta.json"
    meta.write_text(meta.read_text()[: len(meta.read_text()) // 2])

    out = tmp_path / "beliefs.json"
    result = _run((ind, outd), out)

    assert result.returncode != 0, "an unreadable sidecar joined as unverified"
    assert "Traceback" not in result.stderr, (
        f"the fault crashed the build instead of refusing it: {result.stderr}"
    )
    assert "sidecar" in result.stderr and "verdicts-000000" in result.stderr
    assert not out.exists(), "a table was published over an unread configuration"


def test_an_unverifiable_filter_is_announced_where_the_operator_is_looking(
        shards, tmp_path):
    """The guarantee the --stmt-hash-filter help states does not cover these.

    A sidecar-less shard records no digest, so the assertion loop compares
    nothing and a gene-filtered results directory joins as the whole corpus --
    every non-gene evidence in n_missing_cell, `stmt_hash_filter: null` in the
    manifest. That is every shard the 60M run has published. Nothing is
    refusable there without refusing the live corpus, so the one thing that must
    happen is that the run SAYS SO on the console, not only in a manifest field
    nobody reads until later.
    """
    out = tmp_path / "beliefs.json"
    proc = _run(shards, out)
    assert proc.returncode == 0
    assert "[provenance] 1 of 1 joined shards recorded no sidecar" in proc.stdout, (
        f"the unverified part of the table was silent: {proc.stdout[-400:]}"
    )
    assert "--stmt-hash-filter" in proc.stdout

    ind, outd = shards
    _sidecar(outd)
    quiet = _run((ind, outd), tmp_path / "verified.json")
    assert quiet.returncode == 0
    assert "[provenance]" not in quiet.stdout, (
        "a fully-recorded run warns about provenance it does have"
    )


def test_a_mis_filed_isotonic_is_refused_by_the_RUN_not_only_by_the_helper(
        shards, tmp_path, monkeypatch):
    """The guard was extracted and unit-tested; its call site was not.

    Deleting `check_margin_route(...)` from main() left all 25 bridge tests
    green -- the same "a guard nothing exercises" defect the extraction was
    supposed to close, moved one level up. What it protects against is silent:
    an in-call margin (median |13.22|) read through the probe isotonic (knots
    -1.70..+1.61) returns p_hat 0.0000/1.0000 for every row, n_weighted climbs,
    nothing raises, and every statement in a 60M-row table is pushed to 0 or 1
    under a manifest naming the curve.
    """
    import types as _types

    from indra_belief.probes import calibration as _calibration
    from indra_belief.probes.reader import DIRECT_PROBE_ID

    monkeypatch.setattr(
        _calibration, "_calibration_at",
        lambda _path: _types.SimpleNamespace(probe_ids=(DIRECT_PROBE_ID,)),
    )
    ind, outd = shards
    monkeypatch.setattr(sys, "argv", [
        "build_corpus_beliefs.py",
        "--input-dir", str(ind), "--results-dir", str(outd),
        "--model", "local-gemma-4-26b", "--variant", "verdict_only",
        "--served-model-id", _served_id(),
        "--out", str(tmp_path / "beliefs.json"),
    ])

    with pytest.raises(SystemExit) as excinfo:
        bridge.main()
    assert "saturate" in str(excinfo.value), excinfo.value
    assert not (tmp_path / "beliefs.json").exists()


def test_a_filtered_results_directory_cannot_pass_as_the_whole_corpus(shards,
                                                                      tmp_path):
    """The output NAME carries --limit but cannot carry --gene-stmt-hashes.

    A --gene-stmt-hashes results directory joined against an unfiltered build
    with every non-gene evidence falling to n_missing_cell, under a manifest
    that never said the table covered a subset. The runner records the digest
    and lists it in PROVENANCE_KEYS; stage 4 checked the other four keys.
    """
    ind, outd = shards
    digest = "3:0a1b2c3d4e5f6071"
    _sidecar(outd, stmt_hash_filter=digest)

    refused = _run((ind, outd), tmp_path / "beliefs.json")
    output = refused.stdout + refused.stderr
    assert refused.returncode != 0, (
        "a gene-filtered results directory was published as the whole corpus"
    )
    assert "stmt_hash_filter" in output, output[-900:]

    out = tmp_path / "acknowledged.json"
    proc = _run((ind, outd), out, "--stmt-hash-filter", digest)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-900:]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["stmt_hash_filter"] == digest, (
        "the manifest still does not say the table covers a subset"
    )


def test_a_table_covering_a_fraction_of_the_corpus_is_refused(shards, tmp_path):
    """999 of 1000 shards failing to resolve published 0.1% of the corpus, exit 0.

    By the argument the empty-table guard already makes -- INDRA's
    Statement.from_json defaults a missing belief to 1.0 -- the other 99.9% do
    not read as unscored downstream, they read as certainly true.
    """
    ind, outd = shards
    with gzip.open(ind / "grounded-000001.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({**JOBS[0], "stmt_hash": 999, "source_hash": 9}) + "\n")

    refused = _run((ind, outd), tmp_path / "beliefs.json")
    output = refused.stdout + refused.stderr
    assert refused.returncode != 0, "half the corpus was published as all of it"
    assert "min-shard-coverage" in output, output[-900:]

    out = tmp_path / "partial.json"
    proc = _run((ind, outd), out, "--min-shard-coverage", "0.5")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-900:]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["shard_coverage"] == 0.5


def test_a_mis_filed_isotonic_is_refused_rather_than_saturating_every_row():
    """The guard lived inside main(), below the print it protects.

    Nothing could reach it: neutering it to `if False:` left all 20 bridge tests
    green while an in-call margin (median |13.22|) read through the probe curve
    (knots -1.70..+1.61) returns 0.0000/1.0000 for every row -- weighted,
    counted, never an error.
    """
    import types as _types

    from indra_belief.probes.reader import DIRECT_PROBE_ID, IN_CALL_PROBE_ID

    mis_filed = _types.SimpleNamespace(probe_ids=(DIRECT_PROBE_ID,))
    with pytest.raises(SystemExit) as excinfo:
        bridge.check_margin_route(mis_filed, IN_CALL_PROBE_ID, "verdict_only",
                                  Path("incall_calibration_local_mlx.json"))
    assert "saturate" in str(excinfo.value)

    matching = _types.SimpleNamespace(probe_ids=(IN_CALL_PROBE_ID,))
    assert bridge.check_margin_route(
        matching, IN_CALL_PROBE_ID, "verdict_only",
        Path("incall_calibration_local_mlx.json")) is None
    assert bridge.check_margin_route(
        None, IN_CALL_PROBE_ID, "verdict_only", None) is None


def test_scoring_and_believing_through_one_alias_join(tmp_path, monkeypatch):
    """A correct join exited with the provenance refusal.

    `model_client._MODEL_ALIASES` keeps `vllm-local` -> `vllm-gemma-4-26b` alive
    deliberately, and the runner canonicalises before writing the sidecar. Stage
    4 asserted the raw --model, so scoring and believing with the SAME live alias
    recorded "vllm-gemma-4-26b", asserted "vllm-local", and refused a join that
    was right in every respect.
    """
    import sys as _sys

    runner = _load("_alias_runner", "scripts/run_vllm_processed_shards.py")

    ind, outd = tmp_path / "in", tmp_path / "out"
    ind.mkdir()
    jobs = [
        {"job_id": f"{i}:0", "input_row_index": i, "stmt_hash": 400 + i,
         "source_hash": 40 + i, "source_api": "reach", "needs_llm": True,
         "tier1_result": None, "stmt_type": "Activation",
         "user_message": f"CLAIM: A{i} [Activation] B\nEVIDENCE: A{i} activates B."}
        for i in range(2)
    ]
    with gzip.open(ind / "grounded-000000.jsonl.gz", "wt") as fh:
        for job in jobs:
            fh.write(json.dumps(job) + "\n")

    class _Reply:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {
                "content": "verdict correct\nconfidence high"},
                "finish_reason": "stop"}],
                "usage": {"completion_tokens": 4}}

    class _Engine:
        transport_description = runner.OfflineVllmClient.transport_description

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def post(self, *_args, **_kwargs):
            return _Reply()

    monkeypatch.setattr(runner, "OfflineVllmClient", _Engine)
    monkeypatch.setattr(runner, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(_sys, "argv", [
        "run_vllm_processed_shards.py",
        "--input-dir", str(ind), "--output-dir", str(outd),
        "--model", "vllm-local", "--backend", "offline",
        "--workers", "2", "--retries", "0",
    ])
    assert runner.main() == 0

    meta = json.loads((outd / "verdicts-000000.meta.json").read_text())
    assert meta["model"] == "vllm-gemma-4-26b", (
        "premise is stale: the runner no longer canonicalises before recording"
    )

    out = tmp_path / "beliefs.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_corpus_beliefs.py"),
         "--input-dir", str(ind), "--results-dir", str(outd),
         "--model", "vllm-local", "--variant", runner.DEFAULT_VARIANT,
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-900:]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["model"] == "vllm-gemma-4-26b", (
        "the manifest records the alias, so two names for one run produce two "
        "differently-labelled tables"
    )
    assert manifest["n_shards_without_provenance"] == 0
