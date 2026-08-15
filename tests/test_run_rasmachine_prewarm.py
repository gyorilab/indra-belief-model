from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_rasmachine_monolithic import (  # noqa: E402
    _build_parser,
    _load_done_keys,
    _prewarm_grounding,
    _scored_row,
    _sentence_probe_available,
    _require_compatible_resume_score_contract,
    _sentence_score_contract,
    _sentence_probe_record_id,
    _truncated_calls,
    _unmetered_client,
)
import run_rasmachine_monolithic as runner  # noqa: E402
from indra_belief.data.entity import GroundedEntity  # noqa: E402
from indra_belief.probes import calibration as sentence_calibration  # noqa: E402
from indra_belief.probes.calibration import (  # noqa: E402
    CALIBRATION_MODEL_ID,
    CalibratedProbeReading,
    calibrated_sentence_reading,
    replace_sentence_score,
)
from indra_belief.probes.reader import ProbeReading  # noqa: E402
from indra.ontology.bio import bio_ontology  # noqa: E402


def _fake_evidence(source_hash: int = 11):
    return SimpleNamespace(
        get_source_hash=lambda: source_hash,
        to_json=lambda: {"source_hash": source_hash},
        source_api="reach",
        pmid="1",
        text="t",
    )


_STATEMENT = {
    "stmt_hash": "a", "paper_statement_hash": "1", "subject": "EGFR",
    "object": "AKT1", "stmt_type": "Complex", "belief": 0.65,
}


def test_grounding_prewarm_resolves_only_first_real_agent(monkeypatch):
    seen: list[str] = []
    ontology_queries: list[tuple[str, str]] = []

    def fake_resolve(cls, name, raw_text=None):
        seen.append(name)
        return cls(name=name, raw_text=raw_text)

    monkeypatch.setattr(GroundedEntity, "resolve", classmethod(fake_resolve))
    monkeypatch.setattr(
        bio_ontology,
        "get_children",
        lambda namespace, identifier: ontology_queries.append((namespace, identifier)) or [],
    )
    stmts = [
        SimpleNamespace(agent_list=lambda: [None, SimpleNamespace(name="?")]),
        SimpleNamespace(
            agent_list=lambda: [SimpleNamespace(name="EGFR"), SimpleNamespace(name="AKT1")]
        ),
    ]

    assert _prewarm_grounding(stmts) is True
    assert seen == ["EGFR"]
    assert ontology_queries == [("FPLX", "ERK")]


def test_grounding_prewarm_skips_corpus_without_real_agents(monkeypatch):
    monkeypatch.setattr(
        GroundedEntity,
        "resolve",
        classmethod(lambda cls, name, raw_text=None: (_ for _ in ()).throw(AssertionError())),
    )

    assert _prewarm_grounding([SimpleNamespace(agent_list=lambda: [None])]) is False


def test_resume_uses_latest_row_and_retries_errors_and_abstentions(tmp_path):
    output = tmp_path / "run.jsonl"
    rows = [
        {"stmt_i": 0, "evidence_i": 0, "row_status": "scored", "verdict": "correct"},
        {"stmt_i": 1, "evidence_i": 0, "row_status": "error", "verdict": None},
        {"stmt_i": 2, "evidence_i": 0, "row_status": "scored", "verdict": None},
        {"stmt_i": 0, "evidence_i": 0, "row_status": "scored", "verdict": "incorrect"},
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    state = _load_done_keys(output)
    assert state.done == {(0, 0)}
    assert state.verdicts == {"incorrect": 1}
    assert (state.row_errors, state.parser_nulls, state.retryable_errors) == (1, 1, 1)
    assert (state.truncated_total, state.truncated_terminal) == (0, 0)


def test_resume_counts_a_legacy_truncated_row_without_the_flag(tmp_path):
    """A row written before ``truncated`` existed still counts, via its call_log.

    ``_load_done_keys`` ORs the flag with ``_truncated_calls(call_log)``; only the
    flag branch is covered above, so a resumed run over an older output file
    would silently report zero truncations if this OR were dropped.
    """
    output = tmp_path / "legacy.jsonl"
    output.write_text(
        json.dumps(
            {
                "stmt_i": 0,
                "evidence_i": 0,
                "row_status": "scored",
                "verdict": None,
                "call_log": [{"kind": "monolithic", "out_tokens": 4096, "finish_reason": "length"}],
            }
        )
        + "\n"
    )
    state = _load_done_keys(output)
    assert state.truncated_total == 1
    # the same row is verdict-None and non-error, so it ALSO counts as a parser
    # null and is retried — the two counters overlap and must never be summed.
    assert (state.done, state.row_errors, state.parser_nulls) == (set(), 0, 1)
    # and because it WILL be rewritten, nothing carries into the live counter.
    assert state.truncated_terminal == 0


def test_final_truncated_count_equals_a_fresh_read_of_the_finished_file(tmp_path):
    """The composed final count must agree with re-reading the finished file.

    ``main`` seeds its live ``truncated_rows`` from the resume state and then adds
    one per row it writes.  A row that is truncated at start is RETRIED, so it is
    written again — seeding from the file-level total double-counts it, and the
    number the run reports at the end is larger than the file it just wrote.
    """
    output = tmp_path / "run.jsonl"
    truncated_row = {
        "stmt_i": 0,
        "evidence_i": 0,
        "row_status": "scored",
        "verdict": None,
        "truncated": True,
    }
    output.write_text(json.dumps(truncated_row) + "\n")

    before = _load_done_keys(output)

    # the retry re-truncates (deterministic at temperature 0) and is appended.
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(truncated_row) + "\n")
    written_this_invocation = 1

    # THE CLAIM, asserted first because it is the one the run reports: what the
    # invocation composes equals what a fresh read of the file it wrote returns.
    # Seeding from the file-level total composes to 2 against a file holding 1.
    after = _load_done_keys(output)
    assert before.truncated_terminal + written_this_invocation == after.truncated_total == 1

    # the mechanism underneath it: the file held one truncation at start, but
    # this invocation was going to REWRITE that row, so nothing carried over.
    assert (before.truncated_total, before.truncated_terminal) == (1, 0)


def test_no_retry_parser_nulls_carries_a_truncated_row_as_terminal(tmp_path):
    """With retries off the carry-over is the whole count: terminal, written once."""
    output = tmp_path / "no_retry.jsonl"
    output.write_text(
        json.dumps(
            {
                "stmt_i": 0,
                "evidence_i": 0,
                "row_status": "scored",
                "verdict": None,
                "truncated": True,
            }
        )
        + "\n"
    )

    state = _load_done_keys(output, retry_parser_nulls=False)
    assert (state.truncated_total, state.truncated_terminal) == (1, 1)
    # the row is terminal, so this invocation never rewrites it and adds nothing.
    assert state.done == {(0, 0)}
    assert state.parser_nulls == 0


def test_truncated_read_withholds_the_verdict_and_keeps_the_audit_trail():
    result = {
        "verdict": "incorrect",
        "score": 0.714,
        "score_error": None,
        "confidence": "high",
        "tier": "llm_comprehension",
        "call_log": [
            {"kind": "monolithic", "out_tokens": 4096, "finish_reason": "length"},
            {"kind": "relation_nature", "out_tokens": 120, "finish_reason": "stop"},
        ],
    }
    row = _scored_row("r", 14, 1, _fake_evidence(), _STATEMENT, result, 134.356)

    assert (row["verdict"], row["score"], row["confidence"]) == (None, 0.714, None)
    assert row["score_error"] is None
    assert row["truncated"] is True
    assert row["truncated_verdict"] == "incorrect"
    assert row["truncated_call_kind"] == "monolithic"
    assert row["truncated_out_tokens"] == 4096
    # withheld, not errored: resume retries it rather than recording a failure
    assert row["row_status"] == "scored"


def test_untruncated_read_passes_the_verdict_through_unchanged():
    result = {
        "verdict": "correct",
        "score": 0.643,
        "score_error": None,
        "confidence": "high",
        "call_log": [{"kind": "monolithic", "out_tokens": 3695, "finish_reason": "stop"}],
    }
    row = _scored_row("r", 0, 0, _fake_evidence(), _STATEMENT, result, 22.28)

    assert (row["verdict"], row["score"], row["confidence"]) == ("correct", 0.643, "high")
    assert row["score_error"] is None
    assert row["truncated"] is False
    assert row["truncated_verdict"] is None
    assert (row["truncated_call_kind"], row["truncated_out_tokens"]) == (None, None)
    assert _truncated_calls(result["call_log"]) == []


def test_sentence_probe_is_enabled_only_for_its_fitted_reader():
    capable = SimpleNamespace(
        model_name="local-gemma-4-26b",
        backend="openai_compat",
        config={
            "model_id": CALIBRATION_MODEL_ID,
            "max_top_logprobs": 1024,
        },
    )
    assert _sentence_probe_available(capable)
    assert not _sentence_probe_available(
        SimpleNamespace(**{**vars(capable), "model_name": "local-gemma-4-31b"})
    )
    assert not _sentence_probe_available(
        SimpleNamespace(**{**vars(capable), "config": {"max_top_logprobs": 1}})
    )


def test_sentence_probe_serving_id_uses_fit_domain_to_activate_leakage_guard():
    evidence = _fake_evidence(source_hash=-123)
    record_id = _sentence_probe_record_id("run-7", 4, 2, evidence)

    assert record_id == "f-123"


def test_sentence_probe_reads_delta_and_applies_persisted_calibration(monkeypatch):
    seen: dict[str, object] = {}

    def fake_read(record, client):
        seen["record"] = record
        seen["client"] = client
        return ProbeReading(p_raw=0.75, delta_logit=1.0)

    monkeypatch.setattr(sentence_calibration, "read_probe", fake_read)
    client = object()
    calibrated = calibrated_sentence_reading(
        {
            "subject": _STATEMENT["subject"],
            "object": _STATEMENT["object"],
            "stmt_type": _STATEMENT["stmt_type"],
            "evidence_text": "t",
        },
        client,
        record_id="fresh:test:0:0:e",
    )

    assert isinstance(calibrated, CalibratedProbeReading)
    assert 0.0 <= calibrated.p_hat <= 1.0
    assert seen == {
        "record": {
            "subject": "EGFR",
            "object": "AKT1",
            "stmt_type": "Complex",
            "evidence_text": "t",
        },
        "client": client,
    }


def test_sentence_probe_failure_makes_the_only_score_explicitly_unavailable(monkeypatch):
    original = {"verdict": "incorrect", "score": 0.05}
    monkeypatch.setattr(
        sentence_calibration,
        "calibrated_sentence_reading",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("probe down")),
    )

    enriched = replace_sentence_score(
        original,
        {"evidence_text": "t"},
        object(),
        record_id="fresh:test",
        enabled=True,
    )

    assert enriched["score"] is None
    assert enriched["score_error"] == "RuntimeError: probe down"
    assert enriched["verdict"] == "incorrect"
    assert original == {"verdict": "incorrect", "score": 0.05}


def test_disabled_sentence_probe_never_preserves_an_architecture_score():
    enriched = replace_sentence_score(
        {"verdict": "correct", "score": 0.95},
        {"evidence_text": "t"},
        object(),
        record_id="fresh:test",
        enabled=False,
    )

    assert enriched["score"] is None
    assert enriched["score_error"] is None


def test_resume_rejects_historical_rows_without_calibrated_score_contract():
    current = _sentence_score_contract(enabled=True)

    with pytest.raises(ValueError, match="fresh output and regenerate"):
        _require_compatible_resume_score_contract({}, current, completed_rows=1)

    # An empty output has no historical numeric rows to contaminate.
    _require_compatible_resume_score_contract({}, current, completed_rows=0)


def test_empty_sentence_has_no_calibrated_read(monkeypatch):
    monkeypatch.setattr(
        sentence_calibration,
        "read_probe",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call")),
    )

    assert calibrated_sentence_reading(
        {
            "subject": _STATEMENT["subject"],
            "object": _STATEMENT["object"],
            "stmt_type": _STATEMENT["stmt_type"],
            "evidence_text": "",
        },
        object(),
        record_id="fresh:empty",
    ) is None


def test_generic_runner_rejects_provider_backed_models():
    with pytest.raises(ValueError, match="comparison run"):
        _unmetered_client("bedrock-gemma-4-e2b")


def test_generic_parser_has_no_spend_or_authorization_flags():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--spend-ledger", "ledger.ndjson"])
