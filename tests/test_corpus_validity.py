"""Tests for validity-layer metric computation (Phase 4)."""

from __future__ import annotations

import math

import duckdb
import pytest
from indra.statements import Activation, Agent, Evidence, Phosphorylation

from indra_belief.corpus import (
    apply_schema,
    compute_validity,
    ingest_statements,
    score_corpus,
)


def _con():
    con = duckdb.connect(":memory:")
    apply_schema(con)
    return con


def _stmt(belief: float = 0.82, n_ev: int = 1):
    a = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    b = Agent("MAPK1", db_refs={"HGNC": "6871"})
    evs = [
        Evidence(source_api=f"reach_{i}", text=f"sentence {i}",
                 epistemics={"direct": True})
        for i in range(n_ev)
    ]
    s = Phosphorylation(a, b, residue="T", position="202", evidence=evs)
    s.belief = belief
    return s


def _scorer(score: float, verdict: str = "correct", confidence: str = "high"):
    def _fn(statement, evidence, client):
        return {
            "score": score,
            "verdict": verdict,
            "confidence": confidence,
            "reasons": ["match"],
            "call_log": [],
        }
    return _fn


def _evidence_hashes_by_text(con, texts: list[str]) -> list[str]:
    rows = dict(con.execute(
        "SELECT text, evidence_hash FROM evidence"
    ).fetchall())
    return [rows[text] for text in texts]


def test_compute_validity_raises_on_unknown_run_id():
    """Aligned with export_beliefs + model_card: silently returning a
    hollow summary dict on a typo'd run_id was a foot-gun.
    """
    con = _con()
    with pytest.raises(ValueError, match="not found"):
        compute_validity(con, "nonexistent-run-id")


def test_compute_validity_requires_succeeded_run():
    con = _con()
    run_id = "canceled-run"
    con.execute(
        """
        INSERT INTO score_run
          (run_id, scorer_version, indra_version, architecture, model_id_default,
           started_at, finished_at, n_stmts, status, terminated_by, termination_reason)
        VALUES (?, 't', 'unknown', 'monolithic', 'mock',
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, 'canceled', 'user', 'client_disconnected')
        """,
        [run_id],
    )
    with pytest.raises(ValueError, match="require status=succeeded"):
        compute_validity(con, run_id)


def test_compute_validity_is_idempotent():
    """Re-running compute_validity does not double-write metric rows."""
    con = _con()
    s = _stmt(belief=0.82, n_ev=1)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=_scorer(0.95), with_validity=False)

    compute_validity(con, run_id)
    n1 = con.execute("SELECT COUNT(*) FROM metric WHERE run_id = ?",
                     [run_id]).fetchone()[0]

    compute_validity(con, run_id)
    n2 = con.execute("SELECT COUNT(*) FROM metric WHERE run_id = ?",
                     [run_id]).fetchone()[0]

    assert n1 == n2, f"compute_validity not idempotent: {n1} → {n2}"


def test_calibration_writes_mae_rmse_bias():
    con = _con()
    s = _stmt(belief=0.82, n_ev=1)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t", score_evidence=_scorer(0.95), with_validity=False)

    summary = compute_validity(con, run_id)
    cal = summary["calibration"]
    assert cal["n_stmts"] == 1
    assert cal["mae"] == pytest.approx(0.13)
    assert cal["bias"] == pytest.approx(0.13)
    assert cal["rmse"] == pytest.approx(0.13)

    rows = con.execute(
        "SELECT metric_name, value FROM metric "
        "WHERE truth_set_id = 'indra_published_belief' ORDER BY metric_name"
    ).fetchall()
    names = [r[0] for r in rows]
    assert "indra_belief_calibration.mae" in names
    assert "indra_belief_calibration.rmse" in names
    assert "indra_belief_calibration.bias" in names


def test_inter_evidence_consistency_stdev_for_multi_ev():
    con = _con()
    s = _stmt(belief=0.7, n_ev=3)
    ingest_statements(con, [s])

    # Vary scores per evidence so stdev is non-zero
    scores_iter = iter([0.9, 0.5, 0.7])

    def varying(statement, evidence, client):
        return {"score": next(scores_iter), "verdict": "correct",
                "confidence": "high", "reasons": [], "call_log": []}

    run_id = score_corpus(con, [s], scorer_version="t", score_evidence=varying, with_validity=False)
    summary = compute_validity(con, run_id)
    cons = summary["inter_evidence_consistency"]
    assert cons["n_multi_evidence_stmts"] == 1
    assert cons["mean_stdev"] > 0


def test_singleton_only_writes_unavailable_reason():
    con = _con()
    s = _stmt(belief=0.9, n_ev=1)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t", score_evidence=_scorer(0.85), with_validity=False)

    summary = compute_validity(con, run_id)
    assert summary["inter_evidence_consistency"]["n_multi_evidence_stmts"] == 0
    assert "unavailable_reason" in summary["inter_evidence_consistency"]

    # G4 honest-failure: NaN row written with unavailable_reason in slice_json
    row = con.execute(
        "SELECT value, slice_json::VARCHAR FROM metric "
        "WHERE metric_name = 'inter_evidence_consistency.mean_stdev'"
    ).fetchone()
    assert row is not None
    assert math.isnan(row[0])
    assert "unavailable_reason" in row[1]


def test_truth_present_metrics_compute_pr_f1_against_gold(tmp_path):
    """4a — when gold verdict labels exist, P/R/F1 lands in metric table."""
    from indra_belief.corpus import register_truth_set, load_truth_labels

    con = _con()
    s = _stmt(belief=0.5, n_ev=3)
    ingest_statements(con, [s])

    # Score the 3 evidences as: correct, correct, abstain (mock)
    verdicts = iter(["correct", "correct", "abstain"])
    def mock_v(statement, evidence, client):
        v = next(verdicts)
        return {"score": 0.8 if v == "correct" else 0.5,
                "verdict": v, "confidence": "high",
                "reasons": [], "call_log": []}
    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=mock_v, with_validity=False)

    # Register gold pool. Gold says: correct, incorrect, correct
    # Comparing scorer ↔ gold: correct=correct (TP), correct≠incorrect (FP),
    # abstain≠correct (FN). Expect: precision 1/2=0.5, recall 1/2=0.5, f1=0.5
    register_truth_set(con, id="gold_test", name="test gold")

    ev_hashes = _evidence_hashes_by_text(
        con, ["sentence 0", "sentence 1", "sentence 2"]
    )
    gold_verdicts = ["correct", "incorrect", "correct"]
    labels = [
        {"target_kind": "evidence", "target_id": eh,
         "field": "verdict", "value_text": gv, "provenance": "test"}
        for eh, gv in zip(ev_hashes, gold_verdicts)
    ]
    load_truth_labels(con, "gold_test", labels)

    # Compute validity (which now picks up 4a metrics)
    summary = compute_validity(con, run_id)

    assert "gold_test" in summary["truth_present_metrics"]
    agg_metrics = summary["truth_present_metrics"]["gold_test"]["aggregate"]
    assert agg_metrics["n_compared"] == 3
    assert agg_metrics["tp"] == 1
    assert agg_metrics["fp"] == 1
    assert agg_metrics["fn"] == 1
    assert agg_metrics["precision"] == pytest.approx(0.5)
    assert agg_metrics["recall"] == pytest.approx(0.5)
    assert agg_metrics["f1"] == pytest.approx(0.5)

    # metric rows persisted
    metric_names = {r[0] for r in con.execute(
        "SELECT metric_name FROM metric WHERE truth_set_id = 'gold_test'"
    ).fetchall()}
    assert "truth_present.aggregate.precision" in metric_names
    assert "truth_present.aggregate.recall" in metric_names
    assert "truth_present.aggregate.f1" in metric_names


def test_truth_present_metrics_accept_benchmark_tag_labels():
    """Benchmark JSONL uses field='tag'. For P/R/F1, tag=correct is the
    positive class and every other tag is a not-correct gold label.
    """
    from indra_belief.corpus import register_truth_set, load_truth_labels

    con = _con()
    s = _stmt(belief=0.5, n_ev=3)
    ingest_statements(con, [s])

    verdicts = iter(["correct", "correct", "abstain"])

    def mock_v(statement, evidence, client):
        v = next(verdicts)
        return {"score": 0.8 if v == "correct" else 0.5,
                "verdict": v, "confidence": "high",
                "reasons": [], "call_log": []}

    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=mock_v, with_validity=False)

    register_truth_set(con, id="tag_gold", name="tag gold")
    ev_hashes = _evidence_hashes_by_text(
        con, ["sentence 0", "sentence 1", "sentence 2"]
    )
    tags = ["correct", "wrong_relation", "correct"]
    labels = [
        {"target_kind": "evidence", "target_id": eh,
         "field": "tag", "value_text": tag, "provenance": "test"}
        for eh, tag in zip(ev_hashes, tags)
    ]
    load_truth_labels(con, "tag_gold", labels)

    summary = compute_validity(con, run_id)

    assert "tag_gold" in summary["truth_present_metrics"]
    agg_metrics = summary["truth_present_metrics"]["tag_gold"]["aggregate"]
    assert agg_metrics["n_compared"] == 3
    assert agg_metrics["tp"] == 1
    assert agg_metrics["fp"] == 1
    assert agg_metrics["fn"] == 1
    assert agg_metrics["gold_fields"] == ["tag"]
    assert agg_metrics["precision"] == pytest.approx(0.5)
    assert agg_metrics["recall"] == pytest.approx(0.5)
    assert agg_metrics["f1"] == pytest.approx(0.5)

    row = con.execute(
        "SELECT slice_json::VARCHAR FROM metric "
        "WHERE truth_set_id = 'tag_gold' "
        "  AND metric_name = 'truth_present.aggregate.f1'"
    ).fetchone()
    assert row is not None
    assert '"gold_fields": ["tag"]' in row[0]
    assert '"tn": 0' in row[0]
    assert "\"negative_gold_rule\": \"any value != 'correct'\"" in row[0]


def test_truth_present_metrics_prefer_explicit_verdict_over_tag():
    """If both fields exist for one evidence, the explicit verdict label is
    the better mental model than the benchmark tag and must win deterministically.
    """
    from indra_belief.corpus import register_truth_set, load_truth_labels

    con = _con()
    s = _stmt(belief=0.5, n_ev=2)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=_scorer(0.9, verdict="correct"),
                          with_validity=False)
    ev0, ev1 = _evidence_hashes_by_text(con, ["sentence 0", "sentence 1"])

    register_truth_set(con, id="mixed_gold", name="mixed gold")
    load_truth_labels(con, "mixed_gold", [
        {"target_kind": "evidence", "target_id": ev0,
         "field": "tag", "value_text": "correct", "provenance": "test"},
        {"target_kind": "evidence", "target_id": ev0,
         "field": "verdict", "value_text": "incorrect", "provenance": "test"},
        {"target_kind": "evidence", "target_id": ev1,
         "field": "tag", "value_text": "wrong_relation", "provenance": "test"},
        {"target_kind": "evidence", "target_id": ev1,
         "field": "verdict", "value_text": "correct", "provenance": "test"},
    ])

    summary = compute_validity(con, run_id)

    agg_metrics = summary["truth_present_metrics"]["mixed_gold"]["aggregate"]
    assert agg_metrics["n_compared"] == 2
    assert agg_metrics["tp"] == 1
    assert agg_metrics["fp"] == 1
    assert agg_metrics["fn"] == 0
    assert agg_metrics["gold_fields"] == ["verdict"]


def test_truth_present_metrics_use_statement_context_for_shared_evidence_tags():
    """The same evidence sentence can be curated differently for two statements.

    Benchmark JSONL carries source_hash plus matches_hash; validity must not
    collapse those labels to evidence_hash alone.
    """
    from indra_belief.corpus import register_truth_set, load_truth_labels

    con = _con()
    shared_text = "shared evidence supports different statements"
    s1 = Phosphorylation(
        Agent("MAP2K1", db_refs={"HGNC": "6840"}),
        Agent("MAPK1", db_refs={"HGNC": "6871"}),
        evidence=[Evidence(source_api="reach", text=shared_text)],
    )
    s2 = Activation(
        Agent("RAF1", db_refs={"HGNC": "9829"}),
        Agent("MAP2K1", db_refs={"HGNC": "6840"}),
        evidence=[Evidence(source_api="reach", text=shared_text)],
    )
    s1.belief = 0.7
    s2.belief = 0.7
    ingest_statements(con, [s1, s2])
    run_id = score_corpus(con, [s1, s2], scorer_version="t",
                          score_evidence=_scorer(0.9, verdict="correct"),
                          with_validity=False)
    scored = con.execute(
        "SELECT stmt_hash, evidence_hash FROM scorer_step "
        "WHERE run_id=? AND step_kind='aggregate' ORDER BY stmt_hash",
        [run_id],
    ).fetchall()
    assert len(scored) == 2
    assert scored[0][1] == scored[1][1]
    ev_hash = scored[0][1]

    register_truth_set(con, id="context_gold", name="context gold")
    load_truth_labels(con, "context_gold", [
        {"target_kind": "evidence", "target_id": ev_hash,
         "relation_target_id": scored[0][0],
         "field": "tag", "value_text": "correct", "provenance": "test"},
        {"target_kind": "evidence", "target_id": ev_hash,
         "relation_target_id": scored[1][0],
         "field": "tag", "value_text": "wrong_relation", "provenance": "test"},
    ])

    summary = compute_validity(con, run_id)

    agg_metrics = summary["truth_present_metrics"]["context_gold"]["aggregate"]
    assert agg_metrics["n_compared"] == 2
    assert agg_metrics["tp"] == 1
    assert agg_metrics["fp"] == 1
    assert agg_metrics["fn"] == 0
    assert agg_metrics["gold_fields"] == ["tag"]


def test_truth_present_returns_empty_when_no_overlap():
    """4a remains empty when no truth_set carries verdict/tag gold labels."""
    con = _con()
    s = _stmt(belief=0.5, n_ev=1)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=_scorer(0.8), with_validity=False)

    summary = compute_validity(con, run_id)
    # No gold registered → empty truth_present_metrics
    assert summary["truth_present_metrics"] == {}


def test_truth_present_writes_unavailable_row_when_truth_set_has_zero_overlap():
    """Registered truth sets should not disappear when they miss the run."""
    from indra_belief.corpus import register_truth_set, load_truth_labels

    con = _con()
    s = _stmt(belief=0.5, n_ev=1)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=_scorer(0.8), with_validity=False)

    register_truth_set(con, id="miss_gold", name="missing overlap gold")
    load_truth_labels(con, "miss_gold", [
        {"target_kind": "evidence", "target_id": "not-in-this-run",
         "field": "tag", "value_text": "correct", "provenance": "test"},
    ])

    summary = compute_validity(con, run_id)

    unavailable = summary["truth_present_metrics"]["miss_gold"]["aggregate"]
    assert unavailable["n_compared"] == 0
    assert unavailable["n_gold_labels"] == 1
    assert unavailable["n_applicable_gold_labels"] == 0
    assert unavailable["n_scored_evidences"] == 1
    assert unavailable["gold_fields"] == ["tag"]
    assert unavailable["unavailable_reason"] == (
        "no scored aggregate evidence rows overlap this truth_set"
    )

    row = con.execute(
        "SELECT value, slice_json::VARCHAR FROM metric "
        "WHERE truth_set_id = 'miss_gold' "
        "  AND metric_name = 'truth_present.aggregate.precision'"
    ).fetchone()
    assert row is not None
    assert math.isnan(row[0])
    assert '"n_gold_labels": 1' in row[1]
    assert '"n_applicable_gold_labels": 0' in row[1]
    assert "no scored aggregate evidence rows overlap this truth_set" in row[1]

    names = {r[0] for r in con.execute(
        "SELECT metric_name FROM metric WHERE truth_set_id = 'miss_gold'"
    ).fetchall()}
    assert "truth_present.aggregate.precision" in names
    assert "truth_present.aggregate.recall" in names
    assert "truth_present.aggregate.f1" in names
    assert "truth_present.aggregate.unavailable" not in names


def test_truth_present_measured_rows_carry_gold_coverage_denominators():
    from indra_belief.corpus import register_truth_set, load_truth_labels

    con = _con()
    s = _stmt(belief=0.5, n_ev=2)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=_scorer(0.8, verdict="correct"),
                          with_validity=False)
    ev0 = _evidence_hashes_by_text(con, ["sentence 0"])[0]

    register_truth_set(con, id="partial_gold", name="partial overlap gold")
    load_truth_labels(con, "partial_gold", [
        {"target_kind": "evidence", "target_id": ev0,
         "field": "tag", "value_text": "correct", "provenance": "test"},
        {"target_kind": "evidence", "target_id": "not-in-this-run",
         "field": "tag", "value_text": "correct", "provenance": "test"},
    ])

    summary = compute_validity(con, run_id)

    aggregate = summary["truth_present_metrics"]["partial_gold"]["aggregate"]
    assert aggregate["n_compared"] == 1
    assert aggregate["n_gold_labels"] == 2
    assert aggregate["n_applicable_gold_labels"] == 1
    assert aggregate["n_scored_evidences"] == 2

    row = con.execute(
        "SELECT slice_json::VARCHAR FROM metric "
        "WHERE truth_set_id = 'partial_gold' "
        "  AND metric_name = 'truth_present.aggregate.precision'"
    ).fetchone()
    assert row is not None
    assert '"n_gold_labels": 2' in row[0]
    assert '"n_applicable_gold_labels": 1' in row[0]
    assert '"n_scored_evidences": 2' in row[0]

    names = {r[0] for r in con.execute(
        "SELECT metric_name FROM metric WHERE truth_set_id = 'partial_gold'"
    ).fetchall()}
    assert "truth_present.aggregate.precision" in names
    assert "truth_present.aggregate.unavailable" not in names


def test_truth_present_unavailable_when_run_has_no_aggregate_verdicts():
    from indra_belief.corpus import register_truth_set, load_truth_labels

    con = _con()
    s = _stmt(belief=0.5, n_ev=1)
    ingest_statements(con, [s])

    def no_verdict(statement, evidence, client):
        return {"score": 0.8, "confidence": "high", "reasons": [], "call_log": []}

    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=no_verdict, with_validity=False)
    ev0 = _evidence_hashes_by_text(con, ["sentence 0"])[0]
    register_truth_set(con, id="no_verdict_gold", name="no verdict gold")
    load_truth_labels(con, "no_verdict_gold", [
        {"target_kind": "evidence", "target_id": ev0,
         "field": "tag", "value_text": "correct", "provenance": "test"},
    ])

    summary = compute_validity(con, run_id)

    unavailable = summary["truth_present_metrics"]["no_verdict_gold"]["aggregate"]
    assert unavailable["n_compared"] == 0
    assert unavailable["n_gold_labels"] == 1
    assert unavailable["n_applicable_gold_labels"] == 0
    assert unavailable["n_scored_evidences"] == 0
    assert unavailable["unavailable_reason"] == (
        "no scored aggregate evidence rows with verdicts in this run"
    )


def test_verdict_share_metrics_written():
    con = _con()
    s1 = _stmt(belief=0.8, n_ev=1)
    s2 = Activation(
        Agent("RAF1", db_refs={"HGNC": "9829"}),
        Agent("MAP2K1", db_refs={"HGNC": "6840"}),
        evidence=[Evidence(source_api="reach", text="x")],
    )
    s2.belief = 0.6
    ingest_statements(con, [s1, s2])

    verdicts_iter = iter(["correct", "abstain"])
    def per_v(statement, evidence, client):
        return {"score": 0.8, "verdict": next(verdicts_iter), "confidence": "high",
                "reasons": [], "call_log": []}

    run_id = score_corpus(con, [s1, s2], scorer_version="t", score_evidence=per_v, with_validity=False)
    summary = compute_validity(con, run_id)
    assert summary["verdicts"] == {"correct": 1, "abstain": 1}

    rows = con.execute(
        "SELECT metric_name, value FROM metric "
        "WHERE metric_name LIKE 'verdict_share.%' ORDER BY metric_name"
    ).fetchall()
    assert len(rows) == 2
    for _name, value in rows:
        assert value == pytest.approx(0.5)
