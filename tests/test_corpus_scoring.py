"""Tests for score_corpus orchestration (Phase 3.1)."""

from __future__ import annotations

import json

import duckdb
import pytest
from indra.statements import Activation, Agent, Evidence, Phosphorylation

from indra_belief.corpus import (
    apply_schema,
    ingest_statements,
    score_corpus,
)


def _con():
    con = duckdb.connect(":memory:")
    apply_schema(con)
    return con


def _stmt():
    mek1 = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    erk1 = Agent("MAPK1", db_refs={"HGNC": "6871"})
    ev = Evidence(
        source_api="reach",
        text="MEK1 phosphorylates ERK at T202.",
        epistemics={"direct": True},
    )
    s = Phosphorylation(mek1, erk1, residue="T", position="202", evidence=[ev])
    s.belief = 0.82
    return s


def _mock_score(verdict="correct", confidence="high", score=0.95):
    """Returns a `score_evidence`-shaped dict, matching scorer.py docstring."""
    def _fn(statement, evidence, client):
        return {
            "score": score,
            "verdict": verdict,
            "confidence": confidence,
            "tier": "decomposed",
            "grounding_status": "all_match",
            "provenance_triggered": False,
            "tokens": 84,
            "raw_text": "mock trace",
            "reasons": ["match"],
            "rationale": "mock rationale",
            "call_log": [
                {"kind": "probe_subject_role", "duration_s": 0.4,
                 "prompt_tokens": 200, "out_tokens": 20, "finish_reason": "stop"},
                {"kind": "probe_relation_axis", "duration_s": 0.5,
                 "prompt_tokens": 250, "out_tokens": 25, "finish_reason": "stop"},
            ],
        }
    return _fn


def test_score_corpus_writes_run_and_step_rows():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    run_id = score_corpus(
        con, [stmt],
        scorer_version="test-v1",
        model_id_default="mock-model",
        score_evidence=_mock_score(),
    )

    runs = con.execute(
        "SELECT run_id, scorer_version, architecture, status, n_stmts FROM score_run"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0][0] == run_id
    assert runs[0][1] == "test-v1"
    assert runs[0][2] == "decomposed"
    assert runs[0][3] == "succeeded"
    assert runs[0][4] == 1

    steps = con.execute(
        "SELECT step_kind, scorer_version, architecture, model_id, latency_ms IS NOT NULL,"
        "       prompt_tokens, out_tokens FROM scorer_step"
    ).fetchall()
    assert len(steps) == 1
    assert steps[0][0] == "aggregate"
    assert steps[0][1] == "test-v1"
    assert steps[0][2] == "decomposed"
    assert steps[0][3] == "mock-model"
    assert steps[0][4] is True  # latency_ms recorded
    assert steps[0][5] == 450   # 200 + 250 from call_log
    assert steps[0][6] == 45    # 20 + 25


def test_score_corpus_architecture_separates_step_hashes():
    """Same evidence/scorer/model can be scored by both architectures
    without INSERT OR REPLACE collisions."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    decomp_run = score_corpus(
        con, [stmt],
        scorer_version="same-v",
        model_id_default="mock-model",
        architecture="decomposed",
        score_evidence=_mock_score(score=0.8),
        with_validity=False,
    )
    mono_run = score_corpus(
        con, [stmt],
        scorer_version="same-v",
        model_id_default="mock-model",
        architecture="monolithic",
        score_evidence=_mock_score(score=0.2),
        with_validity=False,
    )

    rows = con.execute(
        """SELECT run_id, architecture, step_hash,
                  CAST(json_extract(output_json, '$.score') AS DOUBLE)
           FROM scorer_step
           WHERE step_kind='aggregate'
           ORDER BY architecture"""
    ).fetchall()
    assert len(rows) == 2
    assert {r[0] for r in rows} == {decomp_run, mono_run}
    assert {r[1] for r in rows} == {"decomposed", "monolithic"}
    assert len({r[2] for r in rows}) == 2
    assert {r[3] for r in rows} == {0.8, 0.2}


def test_score_corpus_same_version_rerun_is_append_only():
    """Repeated same-version runs must not steal scorer_step rows."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    run1 = score_corpus(
        con, [stmt],
        scorer_version="same-v",
        model_id_default="mock-model",
        architecture="decomposed",
        score_evidence=_mock_score(score=0.8),
        with_validity=False,
    )
    run2 = score_corpus(
        con, [stmt],
        scorer_version="same-v",
        model_id_default="mock-model",
        architecture="decomposed",
        score_evidence=_mock_score(score=0.2),
        with_validity=False,
    )

    rows = con.execute(
        """SELECT run_id, step_hash,
                  CAST(json_extract(output_json, '$.score') AS DOUBLE)
           FROM scorer_step
           WHERE step_kind='aggregate'
           ORDER BY run_id"""
    ).fetchall()
    assert len(rows) == 2
    assert {r[0] for r in rows} == {run1, run2}
    assert len({r[1] for r in rows}) == 2
    assert {r[2] for r in rows} == {0.8, 0.2}


def test_score_corpus_persists_pair_and_parent_ids():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    parent = score_corpus(
        con, [stmt], scorer_version="parent",
        score_evidence=_mock_score(), with_validity=False,
    )
    child = score_corpus(
        con, [stmt],
        scorer_version="child",
        architecture="monolithic",
        paired_run_group_id="pair_test",
        parent_run_id=parent,
        score_evidence=_mock_score(score=0.2),
        with_validity=False,
    )

    row = con.execute(
        """SELECT architecture, paired_run_group_id, parent_run_id
           FROM score_run WHERE run_id=?""",
        [child],
    ).fetchone()
    assert row == ("monolithic", "pair_test", parent)


def test_score_corpus_dispatches_monolithic(monkeypatch):
    """architecture='monolithic' must select the monolithic scorer path,
    not merely label a decomposed run differently."""
    import indra_belief.scorers.monolithic as mono

    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def fake_mono(statement, evidence, client):
        return {
            "score": 0.2,
            "verdict": "incorrect",
            "confidence": "medium",
            "tier": "monolithic_test",
            "grounding_status": "all_match",
            "provenance_triggered": False,
            "tokens": 12,
            "raw_text": "mono trace",
            "call_log": [],
        }

    monkeypatch.setattr(mono, "score_evidence", fake_mono)
    score_corpus(
        con, [stmt],
        client=object(),
        scorer_version="mono-dispatch",
        architecture="monolithic",
        with_validity=False,
    )
    out = json.loads(con.execute(
        "SELECT output_json FROM scorer_step WHERE step_kind='aggregate'"
    ).fetchone()[0])
    assert out["tier"] == "monolithic_test"
    assert out["score"] == 0.2


def test_score_corpus_monolithic_default_persists_call_log_tokens():
    """Native monolithic dispatch must carry prompt tokens for cost_actual."""
    from scripts.run_paired_smoke import SmokeModelClient

    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(
        con, [stmt],
        client=SmokeModelClient("claude-sonnet-4-6"),
        scorer_version="mono-call-log",
        architecture="monolithic",
        model_id_default="claude-sonnet-4-6",
        with_validity=False,
    )
    row = con.execute(
        """SELECT prompt_tokens, out_tokens, output_json
           FROM scorer_step WHERE step_kind='aggregate'"""
    ).fetchone()
    assert row[0] > 0
    assert row[1] > 0
    out = json.loads(row[2])
    assert out["call_log"][0]["kind"] == "monolithic"
    assert len(out["selected_example_ids"]) > 0
    assert len(out["selected_example_ids"]) == len(out["selected_examples"])
    assert all(isinstance(example_id, str) and len(example_id) == 12 for example_id in out["selected_example_ids"])
    assert {"id", "claim", "verdict", "confidence"} <= set(out["selected_examples"][0])
    assert out["selected_examples"][0]["id"] == out["selected_example_ids"][0]


def test_score_corpus_rejects_decomposed_trace_for_monolithic():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    with pytest.raises(ValueError, match="decompose=True"):
        score_corpus(
            con, [stmt],
            architecture="monolithic",
            decompose=True,
            score_evidence=_mock_score(),
        )


def test_score_corpus_preserves_full_dict_in_output_json():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    score_corpus(con, [stmt], scorer_version="test-v1", score_evidence=_mock_score())

    raw = con.execute("SELECT output_json FROM scorer_step").fetchone()[0]
    out = json.loads(raw)
    assert out["score"] == 0.95
    assert out["verdict"] == "correct"
    assert out["confidence"] == "high"
    assert out["reasons"] == ["match"]
    assert len(out["call_log"]) == 2


def test_score_corpus_handles_evidence_failure_as_abstain():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def boom(statement, evidence, client):
        raise RuntimeError("LLM transport down")

    run_id = score_corpus(con, [stmt], scorer_version="test-v1", score_evidence=boom)
    runs = con.execute("SELECT status, n_stmts FROM score_run").fetchall()
    assert runs[0] == ("succeeded", 1)  # graceful degradation, not failure
    out = json.loads(con.execute("SELECT output_json FROM scorer_step").fetchone()[0])
    assert out["verdict"] == "abstain"
    assert "LLM transport down" in out["error"]


def test_score_corpus_append_only_across_versions():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="v1", score_evidence=_mock_score(score=0.75))
    score_corpus(con, [stmt], scorer_version="v2", score_evidence=_mock_score(score=0.95))

    rows = con.execute(
        "SELECT scorer_version, json_extract(output_json, '$.score') "
        "FROM scorer_step ORDER BY scorer_version"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "v1"
    assert rows[1][0] == "v2"


def test_score_corpus_on_evidence_callback():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    seen: list[tuple[str, str, str]] = []
    def cb(stmt_hash, evidence_hash, result):
        seen.append((stmt_hash, evidence_hash, result["verdict"]))

    score_corpus(
        con, [stmt],
        scorer_version="test-v1",
        score_evidence=_mock_score(),
        on_evidence=cb,
    )
    assert len(seen) == 1
    assert seen[0][2] == "correct"


def test_score_corpus_decompose_writes_per_step_rows():
    """Phase 3.4 partial: decompose=True writes rows for parse_claim,
    build_context, substrate_route + per-probe rows for substrate-answered."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="t",
                 score_evidence=_mock_score(), decompose=True)

    kinds = {row[0] for row in con.execute(
        "SELECT DISTINCT step_kind FROM scorer_step"
    ).fetchall()}
    # Always emitted
    assert "aggregate" in kinds
    assert "parse_claim" in kinds
    assert "build_context" in kinds
    assert "substrate_route" in kinds
    # Probe rows emitted iff substrate resolved them; we don't assert on
    # the specific probes since substrate behavior depends on the stmt+ev
    # content. We just assert no exceptions were swallowed.
    err_rows = con.execute(
        "SELECT step_kind, error FROM scorer_step WHERE error IS NOT NULL"
    ).fetchall()
    assert err_rows == []


def test_score_corpus_decompose_persists_structured_probe_trace_rows():
    """Decomposed aggregate output materializes named probe rows, including
    LLM/abstain answers that are not available from substrate pre-routing."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def scored_with_probe_trace(_statement, _evidence, _client):
        return {
            "score": 0.72,
            "verdict": "correct",
            "confidence": "medium",
            "tier": "decomposed",
            "grounding_status": "all_match",
            "provenance_triggered": False,
            "tokens": 11,
            "raw_text": "structured probe trace",
            "reasons": ["match"],
            "rationale": "probe bundle",
            "probe_trace": {
                "subject_role": {
                    "kind": "subject_role",
                    "answer": "present_as_subject",
                    "source": "substrate",
                    "confidence": "high",
                    "perturbation": None,
                    "span": "MEK1",
                    "rationale": "substrate route",
                },
                "object_role": {
                    "kind": "object_role",
                    "answer": "present_as_object",
                    "source": "llm",
                    "confidence": "medium",
                    "perturbation": None,
                    "span": "ERK",
                    "rationale": "llm classified object role",
                },
                "relation_axis": {
                    "kind": "relation_axis",
                    "answer": "abstain",
                    "source": "abstain",
                    "confidence": "low",
                    "perturbation": None,
                    "span": None,
                    "rationale": "underdetermined",
                },
                "scope": {
                    "kind": "scope",
                    "answer": "asserted",
                    "source": "llm",
                    "confidence": "medium",
                    "perturbation": None,
                    "span": "phosphorylates",
                    "rationale": "asserted relation",
                },
            },
            "call_log": [
                {"kind": "probe_object_role", "duration_s": 0.2,
                 "prompt_tokens": 101, "out_tokens": 9, "finish_reason": "stop"},
                {"kind": "probe_scope", "duration_s": 0.3,
                 "prompt_tokens": 103, "out_tokens": 7, "finish_reason": "stop"},
            ],
        }

    score_corpus(
        con,
        [stmt],
        scorer_version="trace-v1",
        model_id_default="mock-model",
        score_evidence=scored_with_probe_trace,
        decompose=True,
        with_validity=False,
    )

    probe_rows = con.execute(
        """SELECT step_kind,
                  json_extract_string(output_json, '$.source') AS source,
                  prompt_tokens,
                  out_tokens
           FROM scorer_step
           WHERE step_kind IN (
             'subject_role_probe', 'object_role_probe',
             'relation_axis_probe', 'scope_probe'
           )
           ORDER BY step_kind"""
    ).fetchall()
    assert {row[0] for row in probe_rows} == {
        "subject_role_probe",
        "object_role_probe",
        "relation_axis_probe",
        "scope_probe",
    }
    by_kind = {row[0]: row for row in probe_rows}
    assert by_kind["object_role_probe"][1:] == ("llm", 101, 9)
    assert by_kind["scope_probe"][1:] == ("llm", 103, 7)
    assert by_kind["relation_axis_probe"][1] == "abstain"


def test_score_corpus_probe_step_filter_materializes_only_selected_slots():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def scored_with_probe_trace(_statement, _evidence, _client):
        return {
            "score": 0.72,
            "verdict": "correct",
            "confidence": "medium",
            "probe_trace": {
                "subject_role": {"kind": "subject_role", "answer": "present_as_subject", "source": "substrate", "confidence": "high"},
                "object_role": {"kind": "object_role", "answer": "present_as_object", "source": "llm", "confidence": "medium"},
                "relation_axis": {"kind": "relation_axis", "answer": "direct_sign_match", "source": "llm", "confidence": "medium"},
                "scope": {"kind": "scope", "answer": "asserted", "source": "llm", "confidence": "medium"},
            },
            "call_log": [],
        }

    score_corpus(
        con,
        [stmt],
        scorer_version="trace-v1",
        model_id_default="mock-model",
        score_evidence=scored_with_probe_trace,
        decompose=True,
        with_validity=False,
        probe_step_filter=["object_role_probe", "scope_probe"],
    )

    probe_kinds = {
        row[0] for row in con.execute(
            """SELECT DISTINCT step_kind
               FROM scorer_step
               WHERE step_kind IN (
                 'subject_role_probe', 'object_role_probe',
                 'relation_axis_probe', 'scope_probe'
               )"""
        ).fetchall()
    }
    assert probe_kinds == {"object_role_probe", "scope_probe"}
    assert con.execute(
        "SELECT COUNT(*) FROM scorer_step WHERE step_kind='aggregate'"
    ).fetchone()[0] == 1


def test_score_corpus_probe_only_materializes_selected_slots_without_aggregate():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def scored_with_probe_trace(_statement, _evidence, _client):
        return {
            "score": None,
            "verdict": "abstain",
            "confidence": "low",
            "probe_trace": {
                "object_role": {
                    "kind": "object_role",
                    "answer": "present_as_object",
                    "source": "llm",
                    "confidence": "medium",
                },
                "scope": {
                    "kind": "scope",
                    "answer": "asserted",
                    "source": "llm",
                    "confidence": "medium",
                },
            },
            "call_log": [
                {"kind": "probe_object_role", "prompt_tokens": 100, "out_tokens": 10},
                {"kind": "probe_scope", "prompt_tokens": 80, "out_tokens": 8},
            ],
        }

    run_id = score_corpus(
        con,
        [stmt],
        scorer_version="trace-v1",
        model_id_default="claude-sonnet-4-6",
        score_evidence=scored_with_probe_trace,
        decompose=True,
        probe_step_filter=["object_role_probe", "scope_probe"],
        probe_only=True,
    )

    rows = con.execute(
        """SELECT step_kind, prompt_tokens, out_tokens
           FROM scorer_step
          WHERE run_id=?
          ORDER BY step_kind""",
        [run_id],
    ).fetchall()
    assert {row[0] for row in rows}.issuperset({
        "parse_claim",
        "build_context",
        "substrate_route",
        "object_role_probe",
        "scope_probe",
    })
    assert "aggregate" not in {row[0] for row in rows}
    probe_tokens = {
        row[0]: (row[1], row[2])
        for row in rows
        if row[0] in {"object_role_probe", "scope_probe"}
    }
    assert probe_tokens == {
        "object_role_probe": (100, 10),
        "scope_probe": (80, 8),
    }
    status, actual = con.execute(
        "SELECT status, cost_actual_usd FROM score_run WHERE run_id=?",
        [run_id],
    ).fetchone()
    assert status == "succeeded"
    assert actual > 0


def test_score_corpus_probe_only_merges_parent_trace_into_aggregate():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def parent_score(_statement, _evidence, _client):
        return {
            "score": 0.92,
            "verdict": "correct",
            "confidence": "high",
            "tier": "decomposed",
            "grounding_status": "all_match",
            "provenance_triggered": False,
            "tokens": 12,
            "raw_text": "parent trace",
            "reasons": ["match"],
            "rationale": "parent aggregate",
            "probe_trace": {
                "subject_role": {
                    "kind": "subject_role",
                    "answer": "present_as_subject",
                    "source": "llm",
                    "confidence": "medium",
                    "perturbation": None,
                    "span": "MEK1",
                    "rationale": "subject",
                },
                "object_role": {
                    "kind": "object_role",
                    "answer": "present_as_object",
                    "source": "llm",
                    "confidence": "medium",
                    "perturbation": None,
                    "span": "ERK",
                    "rationale": "object",
                },
                "relation_axis": {
                    "kind": "relation_axis",
                    "answer": "direct_sign_match",
                    "source": "llm",
                    "confidence": "medium",
                    "perturbation": None,
                    "span": "phosphorylates",
                    "rationale": "match",
                },
                "scope": {
                    "kind": "scope",
                    "answer": "asserted",
                    "source": "llm",
                    "confidence": "medium",
                    "perturbation": None,
                    "span": "phosphorylates",
                    "rationale": "asserted",
                },
            },
            "call_log": [],
        }

    parent_run = score_corpus(
        con,
        [stmt],
        scorer_version="parent-v1",
        model_id_default="mock-model",
        score_evidence=parent_score,
        decompose=True,
        with_validity=False,
    )

    def repaired_relation(_statement, _evidence, _client):
        return {
            "score": None,
            "verdict": "abstain",
            "confidence": "low",
            "tier": "decomposed_probe_only",
            "grounding_status": "not_run",
            "provenance_triggered": False,
            "tokens": 4,
            "raw_text": "child relation probe",
            "reasons": ["probe_only_rescore"],
            "rationale": "relation repaired",
            "probe_trace": {
                "relation_axis": {
                    "kind": "relation_axis",
                    "answer": "no_relation",
                    "source": "llm",
                    "confidence": "medium",
                    "perturbation": None,
                    "span": "phosphorylates",
                    "rationale": "repaired as no relation",
                },
            },
            "call_log": [
                {"kind": "probe_relation_axis", "prompt_tokens": 70, "out_tokens": 4},
            ],
        }

    child_run = score_corpus(
        con,
        [stmt],
        scorer_version="child-v1",
        model_id_default="mock-model",
        parent_run_id=parent_run,
        score_evidence=repaired_relation,
        decompose=True,
        probe_step_filter=["relation_axis_probe"],
        probe_only=True,
        with_validity=False,
    )

    aggregate_json = con.execute(
        """SELECT output_json
             FROM scorer_step
            WHERE run_id=? AND step_kind='aggregate'""",
        [child_run],
    ).fetchone()[0]
    aggregate = json.loads(aggregate_json)
    assert aggregate["tier"] == "decomposed_probe_repair_merge"
    assert aggregate["probe_trace"]["relation_axis"]["answer"] == "no_relation"
    assert aggregate["probe_trace"]["object_role"]["answer"] == "present_as_object"
    assert aggregate["repair_merge"]["aggregate_llm_call"] is False
    assert aggregate["repair_merge"]["source_by_probe"] == {
        "subject_role": "parent_trace",
        "object_role": "parent_trace",
        "relation_axis": "child_probe_only",
        "scope": "parent_trace",
    }
    assert con.execute(
        """SELECT prompt_tokens, out_tokens
             FROM scorer_step
            WHERE run_id=? AND step_kind='aggregate'""",
        [child_run],
    ).fetchone() == (None, None)


def test_score_corpus_probe_only_requires_filter_and_decomposition():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    with pytest.raises(ValueError, match="probe_step_filter"):
        score_corpus(
            con,
            [stmt],
            score_evidence=_mock_score(),
            decompose=True,
            probe_only=True,
        )
    with pytest.raises(ValueError, match="decompose=True"):
        score_corpus(
            con,
            [stmt],
            score_evidence=_mock_score(),
            probe_step_filter=["scope_probe"],
            probe_only=True,
        )


def test_score_corpus_cost_threshold_aborts_above():
    """G3b stop-the-line: cost_threshold_usd raises before scoring starts."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    with pytest.raises(ValueError, match=r"exceeds threshold"):
        score_corpus(con, [stmt], scorer_version="t",
                     model_id_default="claude-opus-4-7",  # expensive
                     score_evidence=_mock_score(),
                     cost_threshold_usd=0.000001)  # impossibly tight

    # No score_run row should have been written
    assert con.execute("SELECT COUNT(*) FROM score_run").fetchone()[0] == 0


def test_score_corpus_cost_threshold_passes_below():
    """A generous threshold lets the run proceed normally."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    run_id = score_corpus(con, [stmt], scorer_version="t",
                          model_id_default="claude-haiku-4-5",
                          score_evidence=_mock_score(),
                          cost_threshold_usd=1000.0)
    assert run_id
    assert con.execute("SELECT COUNT(*) FROM scorer_step").fetchone()[0] >= 1


def test_score_corpus_actual_cost_threshold_aborts_mid_run():
    """The cap is not just an upfront estimate gate; observed token spend
    aborts the run after the evidence that crosses the cap."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def costly_score(_statement, _evidence, _client):
        return {
            "score": 0.9,
            "verdict": "correct",
            "confidence": "high",
            "call_log": [
                {
                    "kind": "expensive_call",
                    "prompt_tokens": 10_000,
                    "out_tokens": 100,
                    "duration_s": 0.1,
                    "finish_reason": "stop",
                }
            ],
        }

    with pytest.raises(ValueError, match=r"actual cost .* exceeded cap"):
        score_corpus(
            con,
            [stmt],
            scorer_version="t",
            model_id_default="claude-sonnet-4-6",
            score_evidence=costly_score,
            cost_threshold_usd=0.02,
            with_validity=False,
        )

    status, terminated_by, reason, actual = con.execute(
        "SELECT status, terminated_by, termination_reason, cost_actual_usd "
        "FROM score_run"
    ).fetchone()
    assert status == "failed"
    assert terminated_by == "system"
    assert "exceeded cap" in reason
    assert actual > 0.02
    assert con.execute("SELECT COUNT(*) FROM scorer_step").fetchone()[0] == 1


def test_score_corpus_progress_callback_gets_actual_cost_state():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    seen: list[dict] = []

    def on_evidence(_stmt_hash, _ev_hash, result):
        seen.append({
            "cost_so_far": result.get("_cost_so_far_usd"),
            "cost_cap": result.get("_cost_cap_usd"),
            "cost_increment": result.get("_cost_actual_increment_usd"),
        })

    score_corpus(
        con,
        [stmt],
        scorer_version="t",
        model_id_default="claude-sonnet-4-6",
        score_evidence=_mock_score(),
        cost_threshold_usd=1.0,
        on_evidence=on_evidence,
        with_validity=False,
    )

    assert len(seen) == 1
    assert seen[0]["cost_cap"] == 1.0
    assert seen[0]["cost_so_far"] == pytest.approx(0.002025)
    assert seen[0]["cost_increment"] == pytest.approx(0.002025)


def test_score_corpus_accepts_preallocated_run_id():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    run_id = "0123456789abcdef0123456789abcdef"

    got = score_corpus(
        con,
        [stmt],
        run_id=run_id,
        scorer_version="t",
        score_evidence=_mock_score(),
        with_validity=False,
    )

    assert got == run_id
    assert con.execute(
        "SELECT status FROM score_run WHERE run_id=?",
        [run_id],
    ).fetchone()[0] == "succeeded"


def test_score_corpus_finalizer_preserves_external_canceled_status():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    run_id = "22222222222222222222222222222222"

    def cancel_after_first_evidence(_stmt_hash, _evhash, _result):
        con.execute(
            """UPDATE score_run
               SET status='canceled',
                   finished_at=CURRENT_TIMESTAMP,
                   terminated_by='user',
                   termination_reason='external tombstone'
               WHERE run_id=?""",
            [run_id],
        )

    got = score_corpus(
        con,
        [stmt],
        run_id=run_id,
        scorer_version="t",
        score_evidence=_mock_score(),
        on_evidence=cancel_after_first_evidence,
        with_validity=False,
    )

    assert got == run_id
    assert con.execute(
        "SELECT status, terminated_by, termination_reason FROM score_run WHERE run_id=?",
        [run_id],
    ).fetchone() == ("canceled", "user", "external tombstone")


def test_score_corpus_stops_when_run_is_externally_canceled():
    con = _con()
    stmt = _stmt()
    stmt.evidence.append(
        Evidence(
            source_api="reach",
            text="Second evidence should not be scored after cancel.",
        )
    )
    ingest_statements(con, [stmt])
    run_id = "33333333333333333333333333333333"

    def cancel_after_first_evidence(_stmt_hash, _evhash, _result):
        con.execute(
            """UPDATE score_run
               SET status='canceled',
                   finished_at=CURRENT_TIMESTAMP,
                   terminated_by='user',
                   termination_reason='external tombstone'
               WHERE run_id=?""",
            [run_id],
        )

    with pytest.raises(RuntimeError, match="score_corpus canceled"):
        score_corpus(
            con,
            [stmt],
            run_id=run_id,
            scorer_version="t",
            score_evidence=_mock_score(),
            on_evidence=cancel_after_first_evidence,
            with_validity=False,
        )

    assert con.execute(
        "SELECT status, terminated_by, termination_reason FROM score_run WHERE run_id=?",
        [run_id],
    ).fetchone() == ("canceled", "user", "external tombstone")
    assert con.execute(
        "SELECT COUNT(*) FROM scorer_step WHERE run_id=? AND step_kind='aggregate'",
        [run_id],
    ).fetchone()[0] == 1


def test_score_corpus_ignores_negative_token_sentinels_for_actual_cost():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def sentinel_score(_statement, _evidence, _client):
        return {
            "score": 0.7,
            "verdict": "correct",
            "confidence": "medium",
            "call_log": [
                {
                    "kind": "nonreporting_backend",
                    "prompt_tokens": -1,
                    "out_tokens": 10,
                    "duration_s": 0.1,
                }
            ],
        }

    run_id = score_corpus(
        con,
        [stmt],
        scorer_version="t",
        model_id_default="claude-sonnet-4-6",
        score_evidence=sentinel_score,
        with_validity=False,
    )
    prompt_tokens, out_tokens = con.execute(
        "SELECT prompt_tokens, out_tokens FROM scorer_step WHERE run_id=?",
        [run_id],
    ).fetchone()
    assert prompt_tokens is None
    assert out_tokens == 10
    actual = con.execute(
        "SELECT cost_actual_usd FROM score_run WHERE run_id=?",
        [run_id],
    ).fetchone()[0]
    assert actual == pytest.approx(10 * 15.0 / 1_000_000)


def test_score_corpus_rejects_unknown_priced_model_when_cap_set():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    with pytest.raises(ValueError, match=r"missing from MODEL_PRICES"):
        score_corpus(
            con,
            [stmt],
            scorer_version="t",
            model_id_default="new-provider-model",
            score_evidence=_mock_score(),
            cost_threshold_usd=1.0,
            with_validity=False,
        )

    assert con.execute("SELECT COUNT(*) FROM score_run").fetchone()[0] == 0


def test_score_corpus_actual_cost_uses_row_model_id_not_default():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def override_model_score(_statement, _evidence, _client):
        return {
            "model_id": "claude-opus-4-7",
            "score": 0.8,
            "verdict": "correct",
            "confidence": "high",
            "call_log": [
                {
                    "kind": "override",
                    "prompt_tokens": 1_000,
                    "out_tokens": 0,
                    "duration_s": 0.1,
                }
            ],
        }

    run_id = score_corpus(
        con,
        [stmt],
        scorer_version="t",
        model_id_default="claude-haiku-4-5",
        score_evidence=override_model_score,
        with_validity=False,
    )
    actual = con.execute(
        "SELECT cost_actual_usd FROM score_run WHERE run_id=?",
        [run_id],
    ).fetchone()[0]
    assert actual == pytest.approx(1_000 * 15.0 / 1_000_000)


def test_score_corpus_cost_threshold_none_skips_check():
    """Default behavior: no threshold check; expensive runs proceed."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    # No threshold passed → no upfront cost check, no error raised
    run_id = score_corpus(con, [stmt], scorer_version="t",
                          model_id_default="claude-opus-4-7",
                          score_evidence=_mock_score())
    assert run_id


def test_score_corpus_with_validity_default_true():
    """Auto-validity ON by default — score_corpus runs compute_validity at end."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="t", score_evidence=_mock_score())
    # metric rows should exist for the auto-computed validity
    n_metrics = con.execute("SELECT COUNT(*) FROM metric").fetchone()[0]
    assert n_metrics > 0


def test_score_corpus_with_validity_opt_out():
    """`with_validity=False` skips auto-compute (for tests / cost-sensitive runs)."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="t", score_evidence=_mock_score(),
                 with_validity=False)
    n_metrics = con.execute("SELECT COUNT(*) FROM metric").fetchone()[0]
    assert n_metrics == 0


def test_score_corpus_decompose_default_false():
    """Backwards-compatible: decompose defaults to False, only aggregate written."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="t",
                 score_evidence=_mock_score())  # no decompose

    kinds = [row[0] for row in con.execute(
        "SELECT step_kind FROM scorer_step"
    ).fetchall()]
    assert kinds == ["aggregate"]


def test_viewer_step_kinds_match_python_emit():
    """Lock the deep-dive's 9-step rail to Python's emitted step_kinds.

    `viewer/src/routes/statements/[stmt_hash]/+page.svelte` hardcodes a
    9-element STEP_KINDS list to render the rail. If Python renames a
    step_kind in `_decompose_steps`, the rail silently fails to light
    that step. This grep-asserts that every step_kind Python actually
    emits (excluding 'aggregate', which is special-cased to light the
    adjudicate tick) is present in the viewer's STEP_KINDS array.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "viewer" / "src" / "routes" / "statements"
           / "[stmt_hash]" / "+page.svelte").read_text(encoding="utf-8")

    # Extract the keys (first element of each tuple) from STEP_KINDS
    block_match = re.search(
        r"const STEP_KINDS:\s*Array<\[string,\s*string\]>\s*=\s*\[(.+?)\];",
        src, re.S
    )
    assert block_match, "STEP_KINDS array not found in deep-dive svelte"
    viewer_kinds = set(re.findall(r"\['([a-z_]+)'", block_match.group(1)))

    # Run a synthetic score with decompose=True; collect kinds Python emits
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    score_corpus(con, [stmt], scorer_version="t", decompose=True,
                 score_evidence=_mock_score())
    python_kinds = {row[0] for row in con.execute(
        "SELECT DISTINCT step_kind FROM scorer_step"
    ).fetchall()}
    # 'aggregate' is special — rendered via aggForAdjudicate, not in STEP_KINDS
    python_kinds.discard("aggregate")

    missing = python_kinds - viewer_kinds
    assert not missing, (
        f"Python emits step_kinds {missing} that the viewer's STEP_KINDS "
        f"won't render — silent rail bug. Update viewer + this test."
    )


def test_score_corpus_stmt_with_no_evidence_skipped_gracefully():
    con = _con()
    a = Agent("RAF1", db_refs={"HGNC": "9829"})
    b = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    stmt = Activation(a, b, evidence=[])
    ingest_statements(con, [stmt])
    run_id = score_corpus(con, [stmt], scorer_version="test-v1", score_evidence=_mock_score())
    n_steps = con.execute("SELECT COUNT(*) FROM scorer_step").fetchone()[0]
    assert n_steps == 0  # no evidence → no scorer_step rows
    n_runs = con.execute("SELECT n_stmts FROM score_run WHERE run_id = ?", [run_id]).fetchone()[0]
    assert n_runs == 1   # but the statement is counted


def test_score_corpus_raises_when_both_client_and_score_evidence_none():
    """Fail fast: without a ModelClient or a custom scorer, the default
    scorer crashes per-evidence with `client.call(None)` and silently
    yields all-abstain rows. Surface the misuse at the call site instead.
    """
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    with pytest.raises(ValueError, match=r"client=.*score_evidence="):
        score_corpus(con, [stmt], scorer_version="t")  # no client, no score_evidence


def test_score_corpus_persists_cost_estimate():
    """cost_estimate_usd was a forever-NULL schema column until iter-90.
    The estimate is computed upfront for the threshold gate; persisting
    it gives the audit trail (visible in model_card hand-off)."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    score_corpus(con, [stmt], scorer_version="t",
                 model_id_default="claude-sonnet-4-6",
                 score_evidence=_mock_score())
    cost_est = con.execute(
        "SELECT cost_estimate_usd FROM score_run"
    ).fetchone()[0]
    assert cost_est is not None
    assert cost_est > 0  # Sonnet is non-trivial


def test_score_corpus_persists_cost_actual():
    """cost_actual_usd was the second forever-NULL column; iter-92 wires
    it to sum(prompt_tokens × in_rate + out_tokens × out_rate) across
    aggregate scorer_step rows."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    score_corpus(con, [stmt], scorer_version="t",
                 model_id_default="claude-sonnet-4-6",
                 score_evidence=_mock_score())
    cost_actual = con.execute(
        "SELECT cost_actual_usd FROM score_run"
    ).fetchone()[0]
    assert cost_actual is not None
    # Mock returns 450 prompt + 45 out tokens. Sonnet: $3/M in, $15/M out.
    # 1 evidence: 450*3/1M + 45*15/1M = 0.00135 + 0.000675 = 0.002025
    expected = 450 * 3.0 / 1_000_000 + 45 * 15.0 / 1_000_000
    assert abs(cost_actual - expected) < 1e-6, f"got {cost_actual}, expected {expected}"


def test_score_corpus_mid_run_cancel_persists_partial_rows():
    """Phase 3 gate: simulating a mid-run cancel against the UPDATE branch
    must leave a queryable terminal score_run row AND persisted partial
    scorer_step rows. The cancel is delivered via the same DB status flip
    the viewer's markScoreRunCanceled endpoint uses; the worker detects it
    on the next `_raise_if_run_canceled` checkpoint inside the per-evidence
    loop.
    """
    con = _con()
    mek1 = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    erk1 = Agent("MAPK1", db_refs={"HGNC": "6871"})
    s1 = Phosphorylation(
        mek1, erk1, residue="T", position="202",
        evidence=[
            Evidence(source_api="reach", text="ev1"),
            Evidence(source_api="reach", text="ev2"),
            Evidence(source_api="reach", text="ev3"),
        ],
    )
    s1.belief = 0.5
    ingest_statements(con, [s1])

    call_count = {"n": 0}

    def _cancel_on_third_call(statement, evidence, client):
        call_count["n"] += 1
        # On the third evidence, simulate the viewer flipping status to
        # 'canceled' before the worker's next checkpoint. The persisted
        # rows from the first two evidences must survive.
        if call_count["n"] == 3:
            con.execute(
                "UPDATE score_run SET status='canceled', "
                "terminated_by='user', termination_reason='test cancel'"
            )
        return {
            "score": 0.7,
            "verdict": "correct",
            "confidence": "high",
            "tier": "decomposed",
            "grounding_status": "all_match",
            "provenance_triggered": False,
            "tokens": 100,
            "raw_text": "trace",
            "reasons": ["match"],
            "rationale": "ok",
            "call_log": [
                {
                    "kind": "aggregate",
                    "duration_s": 0.01,
                    "prompt_tokens": 50,
                    "out_tokens": 10,
                    "finish_reason": "stop",
                }
            ],
        }

    with pytest.raises(RuntimeError, match="canceled"):
        score_corpus(
            con, [s1],
            scorer_version="cancel-test",
            model_id_default="mock-model",
            score_evidence=_cancel_on_third_call,
        )

    # Terminal score_run row: status must remain 'canceled' (not overwritten
    # to 'failed' by the finalize block — that's the COALESCE(NOT IN cancel
    # statuses) guard in scoring.py:929).
    row = con.execute(
        "SELECT status, terminated_by, termination_reason, finished_at "
        "FROM score_run"
    ).fetchone()
    assert row[0] == "canceled"
    assert row[1] == "user"
    assert row[2] == "test cancel"
    assert row[3] is not None, "finalize must set finished_at even on cancel"

    # Partial rows persisted: the first two evidences each wrote their
    # aggregate row before the cancel landed (third call triggered cancel,
    # next checkpoint raised before any aggregate-row write).
    n_aggregate = con.execute(
        "SELECT COUNT(*) FROM scorer_step WHERE step_kind='aggregate'"
    ).fetchone()[0]
    assert n_aggregate >= 2, (
        f"expected partial scorer_step rows pre-cancel; got {n_aggregate}"
    )


def test_pre_started_cancelled_is_terminal_in_cancel_set():
    """Phase 3 gate: pre_started_cancelled is recognized as a cancel terminal
    status so score_corpus's finalize-UPDATE branch does not overwrite a row
    the paired-cancel pre-spawn path wrote. The synthesized row also has
    finished_at set at write time — operators can query it like any other
    terminal row.
    """
    from indra_belief.corpus.scoring import CANCEL_TERMINAL_STATUSES

    assert "pre_started_cancelled" in CANCEL_TERMINAL_STATUSES, (
        "pre_started_cancelled must be a recognized cancel terminal status"
    )

    con = _con()
    run_id = "deadbeef" * 4
    con.execute(
        """INSERT INTO score_run
           (run_id, scorer_version, indra_version, architecture,
            started_at, finished_at, n_stmts, status,
            cost_actual_usd, terminated_by, termination_reason)
         VALUES (?, 'pre-started-test', 'unknown', 'monolithic',
                 CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0,
                 'pre_started_cancelled', 0, 'user', 'client_disconnected')""",
        [run_id],
    )

    row = con.execute(
        "SELECT status, terminated_by, finished_at "
        "FROM score_run WHERE run_id=?",
        [run_id],
    ).fetchone()
    assert row[0] == "pre_started_cancelled"
    assert row[1] == "user"
    assert row[2] is not None, (
        "pre_started_cancelled row must have finished_at set so it is "
        "queryable as terminal alongside canceled/succeeded rows"
    )


def test_pre_started_cancelled_has_null_cost_estimate(tmp_path):
    """A4 of deferred hypergraph: pre_started_cancelled rows are written
    before the worker spawns, so no cost estimate is computed. The column
    is left NULL as the honest signal — audit queries aggregating
    cost_estimate_usd must handle NULL (COALESCE or filter by status).
    """
    con = _con()
    con.execute(
        """INSERT INTO score_run
           (run_id, scorer_version, indra_version, architecture,
            started_at, finished_at, n_stmts, status,
            cost_actual_usd, terminated_by, termination_reason)
         VALUES (?, 'a4-test', 'unknown', 'monolithic',
                 CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0,
                 'pre_started_cancelled', 0, 'user', 'client_disconnected')""",
        ['deadbeef' * 4],
    )
    cost_estimate, cost_actual = con.execute(
        "SELECT cost_estimate_usd, cost_actual_usd FROM score_run "
        "WHERE status='pre_started_cancelled'"
    ).fetchone()
    assert cost_estimate is None, (
        "pre_started_cancelled rows must keep cost_estimate_usd=NULL — "
        "the worker never ran, so any number would be misleading"
    )
    assert cost_actual == 0


def test_score_corpus_mid_run_cancel_cross_connection_visibility(tmp_path):
    """C5 of deferred hypergraph: mid-run cancel must be visible across
    DuckDB connections — the worker reads its own connection but the
    canceler writes from a separate connection (different process in
    production). DuckDB allows multiple connections to the same file;
    a commit on one is visible to other connections on the next read.

    This pins the cross-connection visibility property the Phase 3
    single-connection test could not exercise.
    """
    import threading
    import time

    db_path = str(tmp_path / "corpus.duckdb")
    worker_con = duckdb.connect(db_path)
    canceler_con = duckdb.connect(db_path)

    apply_schema(worker_con)
    # Build the corpus via the worker connection.
    mek1 = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    erk1 = Agent("MAPK1", db_refs={"HGNC": "6871"})
    stmt = Phosphorylation(
        mek1, erk1, residue="T", position="202",
        evidence=[
            Evidence(source_api="reach", text=f"ev{i}")
            for i in range(8)
        ],
    )
    stmt.belief = 0.5
    ingest_statements(worker_con, [stmt])

    # Mock scorer that paces itself: each call sleeps 50ms so the
    # canceler thread has time to inject the UPDATE before the worker
    # finishes its loop.
    call_count = {"n": 0}

    def _slow_scorer(statement, evidence, client):
        call_count["n"] += 1
        time.sleep(0.05)
        return {
            "score": 0.7,
            "verdict": "correct",
            "confidence": "high",
            "tier": "decomposed",
            "grounding_status": "all_match",
            "provenance_triggered": False,
            "tokens": 100,
            "raw_text": "trace",
            "reasons": ["match"],
            "rationale": "ok",
            "call_log": [
                {
                    "kind": "aggregate",
                    "duration_s": 0.05,
                    "prompt_tokens": 50,
                    "out_tokens": 10,
                    "finish_reason": "stop",
                }
            ],
        }

    # Run the worker in a thread; the main thread acts as canceler.
    worker_error = {"err": None}

    def _worker_thread():
        try:
            score_corpus(
                worker_con, [stmt],
                scorer_version="cross-conn-cancel",
                model_id_default="mock-model",
                score_evidence=_slow_scorer,
            )
        except Exception as e:
            worker_error["err"] = e

    t = threading.Thread(target=_worker_thread)
    t.start()

    # Wait until the worker has processed at least 2 evidences, then
    # write the cancel from the *other* connection.
    deadline = time.time() + 5
    while call_count["n"] < 2 and time.time() < deadline:
        time.sleep(0.01)
    canceler_con.execute(
        "UPDATE score_run SET status='canceled', "
        "terminated_by='user', termination_reason='cross-connection cancel'"
    )
    t.join(timeout=10)
    assert not t.is_alive(), "worker thread did not exit within timeout"

    # Worker should have raised RuntimeError carrying the cancel.
    assert worker_error["err"] is not None
    assert "canceled" in str(worker_error["err"])

    # Final state: the score_run row is 'canceled', the canceler's
    # terminated_by/termination_reason survived, and partial aggregate
    # rows are persisted.
    status, terminated_by, reason, finished_at = worker_con.execute(
        "SELECT status, terminated_by, termination_reason, finished_at "
        "FROM score_run"
    ).fetchone()
    assert status == "canceled"
    assert terminated_by == "user"
    assert reason == "cross-connection cancel"
    assert finished_at is not None, (
        "finalize must set finished_at on cancel even when cross-connection"
    )

    n_aggregate = worker_con.execute(
        "SELECT COUNT(*) FROM scorer_step WHERE step_kind='aggregate'"
    ).fetchone()[0]
    # At least one partial aggregate row must have committed before the
    # cancel landed. The exact count depends on timing.
    assert n_aggregate >= 1, (
        f"expected partial aggregate rows pre-cancel; got {n_aggregate}"
    )
    assert n_aggregate < 8, (
        f"all 8 evidences scored — cancel did not take effect; got {n_aggregate}"
    )

    worker_con.close()
    canceler_con.close()
