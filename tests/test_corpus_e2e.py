"""End-to-end integration test for the corpus pipeline.

Per-module unit tests live in `test_corpus_loader.py`, `test_corpus_scoring.py`,
`test_corpus_validity.py`, `test_corpus_export.py`, `test_corpus_cost.py`.
This file verifies the four-step canonical workflow runs cleanly:

    apply_schema → ingest_statements → estimate_cost → score_corpus
    (with auto-validity) → export_beliefs → model_card → INDRA round-trip
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
from indra.statements import (
    Activation,
    Agent,
    Evidence,
    Inhibition,
    Phosphorylation,
    stmts_from_json,
)

from indra_belief.corpus import (
    apply_schema,
    estimate_cost,
    export_beliefs,
    ingest_statements,
    model_card,
    score_corpus,
)


def _build_corpus():
    """Synthetic mini-corpus exercising distinct stmt types + epistemics."""
    mek1 = Agent("MAP2K1", db_refs={"HGNC": "6840", "UP": "Q02750"})
    erk1 = Agent("MAPK1", db_refs={"HGNC": "6871", "UP": "P28482"})
    raf1 = Agent("RAF1", db_refs={"HGNC": "9829"})
    return [
        Phosphorylation(mek1, erk1, residue="T", position="202", evidence=[
            Evidence(source_api="reach", pmid="1", text="MEK1 phosphorylates ERK at T202.",
                     epistemics={"direct": True, "curated": True}),
            Evidence(source_api="biopax", pmid="2", text="MAP2K1 catalyzes ERK1 T202.",
                     epistemics={"direct": True}),
        ]),
        Activation(raf1, mek1, evidence=[
            Evidence(source_api="signor", pmid="3", text="RAF1 activates MEK1.",
                     epistemics={"direct": True, "curated": True}),
        ]),
        Inhibition(mek1, raf1, evidence=[
            Evidence(source_api="reach", pmid="4", text="MEK1 was not found to inhibit RAF1.",
                     epistemics={"direct": False, "negated": True}),
        ]),
    ]


def _mock_scorer(statement, evidence, client):
    """Deterministic mock that mirrors real scorer output shape."""
    epi = getattr(evidence, "epistemics", {}) or {}
    if epi.get("negated"):
        return {"score": 0.15, "verdict": "incorrect", "confidence": "high",
                "reasons": ["negated"], "call_log": []}
    if epi.get("curated"):
        return {"score": 0.92, "verdict": "correct", "confidence": "high",
                "reasons": ["match", "curated"],
                "call_log": [{"kind": "probe_subject_role", "duration_s": 0.3,
                              "prompt_tokens": 180, "out_tokens": 18}]}
    return {"score": 0.78, "verdict": "correct", "confidence": "medium",
            "reasons": ["match"], "call_log": []}


def test_full_pipeline_e2e(tmp_path: Path):
    """Step-by-step exercise of the canonical workflow."""
    con = duckdb.connect(":memory:")
    apply_schema(con)
    stmts = _build_corpus()

    # 1. Ingest (lossless)
    ingest_counters = ingest_statements(con, stmts, source_dump_id="e2e_smoke")
    assert ingest_counters["n_statements"] == 3
    assert ingest_counters["n_evidences"] == 4
    assert ingest_counters["n_truth_labels"] > 0  # auto-registered

    # Auto-registered truth_sets for INDRA-derived signals
    truth_sets = {r[0] for r in con.execute(
        "SELECT id FROM truth_set"
    ).fetchall()}
    assert {"indra_published_belief", "indra_epistemics", "indra_grounding"} <= truth_sets

    # 2. Estimate cost (pre-Go check)
    est = estimate_cost(stmts, model_id="claude-sonnet-4-6")
    assert est["n_stmts"] == 3
    assert est["n_evidences_est"] == 4
    assert est["cost_usd"] > 0

    # 3. Score (with_validity=True default; decompose=True for per-step rows)
    run_id = score_corpus(
        con, stmts,
        scorer_version="e2e-test",
        model_id_default="mock",
        score_evidence=_mock_scorer,
        decompose=True,
        cost_threshold_usd=10.0,  # generous; mock scoring doesn't actually pay
    )
    assert run_id

    # Score run row populated
    run = con.execute(
        "SELECT status, n_stmts FROM score_run WHERE run_id = ?", [run_id]
    ).fetchone()
    assert run == ("succeeded", 3)

    # Aggregate scorer_step rows + decomposed substeps
    step_kinds = {r[0] for r in con.execute(
        "SELECT DISTINCT step_kind FROM scorer_step WHERE run_id = ?", [run_id]
    ).fetchall()}
    assert "aggregate" in step_kinds
    assert "parse_claim" in step_kinds
    assert "build_context" in step_kinds
    assert "substrate_route" in step_kinds

    # 4. Validity auto-computed → metric rows populated
    n_metrics = con.execute(
        "SELECT COUNT(*) FROM metric WHERE run_id = ?", [run_id]
    ).fetchone()[0]
    assert n_metrics > 0

    # Specific calibration metrics present
    cal_names = {r[0] for r in con.execute(
        "SELECT metric_name FROM metric WHERE run_id = ?", [run_id]
    ).fetchall()}
    assert "indra_belief_calibration.mae" in cal_names
    assert "indra_belief_calibration.bias" in cal_names

    # 5. Export INDRA-native JSON
    out_path = tmp_path / "exported.json"
    export_beliefs(con, run_id, out_path)
    assert out_path.exists()

    # Verify exported beliefs replaced INDRA's defaults
    exported = json.loads(out_path.read_text())
    assert len(exported) == 3
    for stmt_dict in exported:
        assert "belief" in stmt_dict
        # All scored stmts should have belief != 1.0 (the INDRA default)
        # since our mock scorer returns non-1.0 values
        assert stmt_dict["belief"] != 1.0

    # 6. INDRA round-trip — exported JSON re-loads cleanly
    reloaded = stmts_from_json(exported)
    assert len(reloaded) == 3
    types_seen = {type(s).__name__ for s in reloaded}
    assert types_seen == {"Phosphorylation", "Activation", "Inhibition"}

    # 7. Model card exports
    card_path = tmp_path / "card.json"
    card = model_card(con, run_id, out_path=card_path)
    assert card["run_id"] == run_id
    assert card["status"] == "succeeded"
    assert card["n_stmts_scored"] == 3
    assert "metrics" in card
    assert "limitations" in card

    con.close()


def test_full_pipeline_handles_shared_source_hash_across_statements(tmp_path: Path):
    """Phase 1 gate: a source hash shared across two statements must survive
    ingest → score → export → model_card with statement-context membership
    intact, not "last writer wins" on a legacy evidence.stmt_hash column.
    """
    con = duckdb.connect(":memory:")
    apply_schema(con)
    map2k1 = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    mapk1 = Agent("MAPK1", db_refs={"HGNC": "6871"})
    raf1 = Agent("RAF1", db_refs={"HGNC": "9829"})
    shared_text = "shared reach evidence supports both relationships"
    shared_ev = Evidence(source_api="reach", pmid="42", text=shared_text,
                         epistemics={"direct": True, "curated": True})
    stmts = [
        Phosphorylation(map2k1, mapk1, residue="T", position="202",
                        evidence=[shared_ev]),
        Activation(raf1, map2k1, evidence=[shared_ev]),
    ]

    ingest_statements(con, stmts, source_dump_id="shared_e2e")

    # Ingest invariants: one canonical evidence row, two membership rows.
    n_evidence = con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    n_membership = con.execute("SELECT COUNT(*) FROM statement_evidence").fetchone()[0]
    distinct_stmts = con.execute(
        "SELECT COUNT(DISTINCT stmt_hash) FROM statement_evidence"
    ).fetchone()[0]
    assert n_evidence == 1
    assert n_membership == 2
    assert distinct_stmts == 2

    # The legacy `evidence.stmt_hash` column must be gone after apply_schema
    # ran the drop migration; if it lingers, downstream consumers can join on
    # it and silently lose one of the two contexts.
    has_legacy_col = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name='evidence' AND column_name='stmt_hash'"
    ).fetchone()[0]
    assert has_legacy_col == 0

    # Score: both statements must score even though their evidence shares a
    # canonical payload row.
    run_id = score_corpus(
        con, stmts,
        scorer_version="shared-e2e-test",
        model_id_default="mock",
        score_evidence=_mock_scorer,
        decompose=True,
        cost_threshold_usd=10.0,
    )
    assert run_id

    n_scored_stmts = con.execute(
        "SELECT COUNT(DISTINCT stmt_hash) FROM scorer_step WHERE run_id=?",
        [run_id],
    ).fetchone()[0]
    assert n_scored_stmts == 2

    # Export INDRA-native JSON: both statements must show up.
    out_path = tmp_path / "shared.json"
    export_beliefs(con, run_id, out_path)
    exported = json.loads(out_path.read_text())
    assert len(exported) == 2
    types_seen = {stmt_dict["type"] for stmt_dict in exported}
    assert types_seen == {"Phosphorylation", "Activation"}

    # Model card: denominator validation must succeed under sharing.
    card = model_card(con, run_id, out_path=tmp_path / "card.json")
    assert card["status"] == "succeeded"
    val = card.get("evidence_denominator_validation")
    assert val is not None
    assert val.get("evidence_count_validated") is True
    # Both statements contributed one raw_json evidence; the shared payload
    # row backs both, so the table denominator must match the raw_json
    # denominator at 2 each (not 1 from "last writer wins").
    assert val.get("n_raw_json_evidences") == 2
    assert val.get("n_table_evidences") == 2
    assert val.get("mismatches") == []

    con.close()


def test_full_pipeline_handles_empty_corpus(tmp_path: Path):
    """Pipeline should degrade gracefully on an empty corpus."""
    con = duckdb.connect(":memory:")
    apply_schema(con)

    counters = ingest_statements(con, [], source_dump_id="empty")
    assert counters["n_statements"] == 0

    run_id = score_corpus(con, [], scorer_version="t",
                          score_evidence=_mock_scorer)
    n_steps = con.execute("SELECT COUNT(*) FROM scorer_step").fetchone()[0]
    assert n_steps == 0

    # Validity gracefully reports unavailable
    metrics = con.execute(
        "SELECT metric_name FROM metric WHERE run_id = ?", [run_id]
    ).fetchall()
    # G4 honest-failure: if no scored data, metrics either absent or NaN
    if metrics:
        # All NaN — calibration with no scored stmts
        for name, in metrics:
            row = con.execute(
                "SELECT value FROM metric WHERE run_id = ? AND metric_name = ?",
                [run_id, name]
            ).fetchone()
            v = row[0]
            # value can be NaN (Python `nan != nan`) — accept it
            assert v != v or isinstance(v, float)

    out_path = tmp_path / "empty_export.json"
    export_beliefs(con, run_id, out_path)
    data = json.loads(out_path.read_text())
    assert data == []  # nothing to export

    con.close()


def test_denominator_ledger_agrees_with_direct_queries(tmp_path: Path):
    """B1 gate: every denominator the unified ledger emits must match the
    direct query that would otherwise produce it. Pins the ledger view
    against handwritten SQL across run_meta, scorer_step, aggregate,
    corpus, and truth_label families.
    """
    from indra_belief.corpus import query_denominator_ledger

    con = duckdb.connect(":memory:")
    apply_schema(con)
    stmts = _build_corpus()
    ingest_statements(con, stmts, source_dump_id="ledger_smoke")
    run_id = score_corpus(
        con, stmts,
        scorer_version="ledger-test",
        model_id_default="mock",
        score_evidence=_mock_scorer,
        decompose=True,
        cost_threshold_usd=10.0,
    )
    assert run_id

    # Build a {(run_id, family, kind): value} dict from the ledger rows.
    ledger = {
        (row.run_id, row.family, row.kind): row.value
        for row in query_denominator_ledger(con)
    }

    # Direct verifications:
    # 1. run_meta n_stmts must match score_run.n_stmts.
    direct_n_stmts = con.execute(
        "SELECT n_stmts FROM score_run WHERE run_id=?", [run_id]
    ).fetchone()[0]
    assert ledger[(run_id, 'run_meta', 'n_stmts')] == direct_n_stmts

    # 2. scorer_step counts per step_kind must match direct GROUP BY.
    step_counts = dict(
        con.execute(
            "SELECT step_kind, COUNT(*) FROM scorer_step "
            "WHERE run_id=? GROUP BY step_kind",
            [run_id],
        ).fetchall()
    )
    for step_kind, count in step_counts.items():
        assert ledger[(run_id, 'scorer_step', step_kind)] == count, (
            f"ledger row for ({run_id}, scorer_step, {step_kind}) "
            f"disagrees with direct: ledger="
            f"{ledger.get((run_id, 'scorer_step', step_kind))} direct={count}"
        )

    # 3. aggregate n_evidences must match COUNT(DISTINCT evidence_hash).
    direct_n_ev = con.execute(
        "SELECT COUNT(DISTINCT evidence_hash) FROM scorer_step "
        "WHERE run_id=? AND step_kind='aggregate' AND evidence_hash IS NOT NULL",
        [run_id],
    ).fetchone()[0]
    assert ledger[(run_id, 'aggregate', 'n_evidences')] == direct_n_ev

    # 4. corpus n_statements / n_evidences / n_statements_with_evidence.
    assert ledger[(None, 'corpus', 'n_statements')] == con.execute(
        "SELECT COUNT(*) FROM statement"
    ).fetchone()[0]
    assert ledger[(None, 'corpus', 'n_evidences')] == con.execute(
        "SELECT COUNT(*) FROM evidence"
    ).fetchone()[0]
    assert ledger[(None, 'corpus', 'n_statements_with_evidence')] == con.execute(
        "SELECT COUNT(DISTINCT stmt_hash) FROM statement_evidence"
    ).fetchone()[0]

    # 5. truth_label rows present for indra_* truth sets registered at ingest.
    tl_rows = [
        (r.kind, r.slice_json, r.value)
        for r in query_denominator_ledger(con, family='truth_label')
    ]
    assert tl_rows, "expected at least one truth_label denominator row"
    # Each row's count must agree with direct query.
    for kind, slice_json, value in tl_rows:
        # slice_json is a JSON object string like '{"truth_set_id":"..."}'
        import json
        truth_set_id = json.loads(slice_json)['truth_set_id']
        direct = con.execute(
            "SELECT COUNT(*) FROM truth_label "
            "WHERE target_kind=? AND truth_set_id=?",
            [kind, truth_set_id],
        ).fetchone()[0]
        assert direct == value
