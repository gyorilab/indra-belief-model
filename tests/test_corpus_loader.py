"""Smoke tests for the corpus DuckDB loader.

Phase 2.3 (`from_indra_json`) + Phase 2.4 (truth-set ingestion).
"""

from __future__ import annotations

import json

import duckdb
import pytest
from indra.statements import (
    Activation,
    Agent,
    Evidence,
    Phosphorylation,
)

from indra_belief.corpus import (
    apply_schema,
    ingest_statements,
    load_truth_labels,
    register_truth_set,
    validate_statement_evidence_denominators,
)


def _con():
    con = duckdb.connect(":memory:")
    apply_schema(con)
    return con


def _phospho_stmt():
    mek1 = Agent("MAP2K1", db_refs={"HGNC": "6840", "UP": "Q02750"})
    erk = Agent("MAPK1", db_refs={"HGNC": "6871", "UP": "P28482"})
    ev = Evidence(
        source_api="reach",
        pmid="18599499",
        text="MEK1 phosphorylates ERK at T202.",
        epistemics={"direct": True, "negated": False, "curated": False},
    )
    stmt = Phosphorylation(mek1, erk, residue="T", position="202", evidence=[ev])
    stmt.belief = 0.82
    return stmt


def test_apply_schema_idempotent():
    con = _con()
    # Second application must not error.
    apply_schema(con)
    rows = con.execute("SELECT COUNT(*) FROM schema_meta WHERE key='schema_version'").fetchone()
    assert rows[0] == 1
    correction_table = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name='scorer_step_correction'"
    ).fetchone()[0]
    correction_arch = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name='scorer_step_correction' AND column_name='architecture'"
    ).fetchone()[0]
    correction_lineage = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name='scorer_step_correction' "
        "AND column_name IN ('parent_correction_id', 'child_run_id', 'repair_source_dump_id')"
    ).fetchone()[0]
    statement_evidence = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name='statement_evidence'"
    ).fetchone()[0]
    assert correction_table == 1
    assert correction_arch == 1
    assert correction_lineage == 3
    assert statement_evidence == 1


def test_apply_schema_backfills_statement_evidence_context_table():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE statement (
            stmt_hash VARCHAR PRIMARY KEY,
            extraction_hash VARCHAR,
            indra_uuid VARCHAR,
            indra_type VARCHAR NOT NULL,
            indra_belief DOUBLE,
            supports_count INTEGER NOT NULL DEFAULT 0,
            supported_by_count INTEGER NOT NULL DEFAULT 0,
            raw_json JSON NOT NULL,
            source_dump_id VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE evidence (
            evidence_hash VARCHAR PRIMARY KEY,
            stmt_hash VARCHAR NOT NULL,
            source_api VARCHAR,
            source_id VARCHAR,
            pmid VARCHAR,
            text TEXT,
            is_direct BOOLEAN,
            is_negated BOOLEAN,
            is_curated BOOLEAN,
            epistemics_json JSON,
            annotations_json JSON,
            raw_json JSON NOT NULL
        )
        """
    )
    con.execute(
        """
        INSERT INTO statement
            (stmt_hash, indra_type, indra_belief, supports_count,
             supported_by_count, raw_json, source_dump_id)
        VALUES
            ('stmt_old', 'Activation', 0.4, 0, 0,
             '{"type":"Activation","evidence":[{"source_hash":"ev_old"}]}'::JSON,
             'old_dump')
        """
    )
    con.execute(
        """
        INSERT INTO evidence
            (evidence_hash, stmt_hash, source_api, text, raw_json)
        VALUES
            ('ev_old', 'stmt_old', 'reach', 'old evidence',
             '{"source_api":"reach","text":"old evidence"}'::JSON)
        """
    )

    apply_schema(con)

    row = con.execute(
        "SELECT stmt_hash, evidence_hash, evidence_index, source_dump_id "
        "FROM statement_evidence"
    ).fetchone()
    flag = con.execute(
        "SELECT value FROM schema_meta WHERE key='statement_evidence_backfill_v1'"
    ).fetchone()
    assert row == ("stmt_old", "ev_old", 0, "old_dump")
    assert flag == ("done",)

    drop_flag = con.execute(
        "SELECT value FROM schema_meta WHERE key='evidence_stmt_hash_drop_v1'"
    ).fetchone()
    assert drop_flag == ("done",), (
        "evidence.stmt_hash drop migration must run after backfill completes"
    )
    has_stmt_hash = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name='evidence' AND column_name='stmt_hash'"
    ).fetchone()
    assert has_stmt_hash == (0,), (
        "evidence.stmt_hash column must be dropped after backfill"
    )


def test_apply_schema_migrates_v1_score_tables_before_indexing():
    """Reopening a v1 DuckDB must add architecture columns before indexes."""
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE score_run (
            run_id VARCHAR PRIMARY KEY,
            scorer_version VARCHAR NOT NULL,
            indra_version VARCHAR NOT NULL,
            model_id_default VARCHAR,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            n_stmts INTEGER,
            status VARCHAR NOT NULL,
            cost_estimate_usd DOUBLE,
            cost_actual_usd DOUBLE,
            notes TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE scorer_step (
            step_hash VARCHAR PRIMARY KEY,
            stmt_hash VARCHAR NOT NULL,
            evidence_hash VARCHAR,
            run_id VARCHAR,
            scorer_version VARCHAR NOT NULL,
            model_id VARCHAR,
            step_kind VARCHAR NOT NULL,
            is_substrate_answered BOOLEAN,
            input_payload_json TEXT,
            output_json TEXT,
            latency_ms DOUBLE,
            prompt_tokens INTEGER,
            out_tokens INTEGER,
            finish_reason VARCHAR,
            error TEXT
        )
        """
    )
    con.execute(
        """
        INSERT INTO score_run
            (run_id, scorer_version, indra_version, model_id_default,
             started_at, finished_at, n_stmts, status,
             cost_estimate_usd, cost_actual_usd, notes)
        VALUES ('run_v1', 'prod-v1', 'indra-x', 'mock',
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1, 'succeeded',
                0.1, 0.1, NULL)
        """
    )
    con.execute(
        """
        INSERT INTO scorer_step
            (step_hash, stmt_hash, evidence_hash, run_id, scorer_version,
             model_id, step_kind, is_substrate_answered, input_payload_json,
             output_json, latency_ms, prompt_tokens, out_tokens,
             finish_reason, error)
        VALUES ('step_v1', 'stmt', 'ev', 'run_v1', 'prod-v1',
                'mock', 'aggregate', NULL, NULL, '{}', 1, 1, 1,
                'stop', NULL)
        """
    )

    apply_schema(con)

    run_arch = con.execute(
        "SELECT architecture, paired_run_group_id, review_status FROM score_run"
    ).fetchone()
    step_arch = con.execute(
        "SELECT architecture FROM scorer_step WHERE step_hash='step_v1'"
    ).fetchone()
    correction_table = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name='scorer_step_correction'"
    ).fetchone()[0]
    correction_arch = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name='scorer_step_correction' AND column_name='architecture'"
    ).fetchone()[0]
    correction_lineage = con.execute(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name='scorer_step_correction' "
        "AND column_name IN ('parent_correction_id', 'child_run_id', 'repair_source_dump_id')"
    ).fetchone()[0]
    assert run_arch == ("decomposed", None, None)
    assert step_arch == ("decomposed",)
    assert correction_table == 1
    assert correction_arch == 1
    assert correction_lineage == 3


def test_apply_schema_architecture_backfill_is_one_shot():
    """Future unknown rows should not be relabeled by scorer_version text."""
    con = _con()
    con.execute(
        """
        INSERT INTO score_run
            (run_id, scorer_version, indra_version, architecture,
             model_id_default, started_at, status, n_stmts)
        VALUES ('future_unknown', 'v17-mono-fix', 'indra-x', 'unknown',
                'mock', CURRENT_TIMESTAMP, 'succeeded', 0)
        """
    )

    apply_schema(con)

    arch = con.execute(
        "SELECT architecture FROM score_run WHERE run_id='future_unknown'"
    ).fetchone()[0]
    assert arch == "unknown"


def test_apply_schema_backfills_repair_lineage_projection():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE scorer_step_correction (
            correction_id BIGINT,
            step_hash VARCHAR,
            run_id VARCHAR,
            stmt_hash VARCHAR,
            evidence_hash VARCHAR,
            correction_kind VARCHAR,
            status VARCHAR,
            reviewer VARCHAR,
            note TEXT,
            value_json JSON,
            source_route VARCHAR,
            source_filters_json JSON,
            created_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        INSERT INTO scorer_step_correction
            (correction_id, step_hash, run_id, stmt_hash, evidence_hash,
             correction_kind, status, value_json, created_at)
        VALUES (
            41, 'step', 'parent_run', 'stmt', 'ev', 'rerun_intent', 'open',
            '{"parent_correction_id":17,"child_run_id":"child_run","source_dump_id":"repair_src"}'::JSON,
            CURRENT_TIMESTAMP
        )
        """
    )

    apply_schema(con)

    row = con.execute(
        "SELECT parent_correction_id, child_run_id, repair_source_dump_id "
        "FROM scorer_step_correction WHERE correction_id=41"
    ).fetchone()
    flag = con.execute(
        "SELECT value FROM schema_meta WHERE key='repair_lineage_backfill_v1'"
    ).fetchone()
    assert row == (17, "child_run", "repair_src")
    assert flag == ("done",)


def test_ingest_phosphorylation_writes_all_grains():
    con = _con()
    counters = ingest_statements(con, [_phospho_stmt()], source_dump_id="test_dump")
    assert counters["n_statements"] == 1
    assert counters["n_evidences"] == 1
    assert counters["n_agents"] == 2  # enz + sub
    # 1 belief label + 3 epistemics labels + 2 grounding labels
    assert counters["n_truth_labels"] == 6


def test_statement_row_lossless():
    con = _con()
    stmt = _phospho_stmt()
    ingest_statements(con, [stmt], source_dump_id="test_dump")
    row = con.execute(
        "SELECT indra_type, indra_belief, raw_json FROM statement"
    ).fetchone()
    assert row[0] == "Phosphorylation"
    assert row[1] == pytest.approx(0.82)
    raw = json.loads(row[2])
    # Lossless: original to_json fields survive
    assert raw["type"] == "Phosphorylation"
    assert raw["residue"] == "T"
    assert raw["position"] == "202"


def test_ingest_validates_raw_json_table_evidence_denominator():
    con = _con()
    stmt = _phospho_stmt()
    ingest_statements(con, [stmt], source_dump_id="test_dump")
    stmt_hash = con.execute("SELECT stmt_hash FROM statement").fetchone()[0]

    validation = validate_statement_evidence_denominators(con, [stmt_hash])

    assert validation.validated is True
    assert validation.n_statements == 1
    assert validation.n_raw_json_evidences == 1
    assert validation.n_table_evidences == 1
    assert validation.to_dict()["evidence_count_validated"] is True


def test_ingest_rejects_same_statement_with_different_evidence_set():
    con = _con()
    subj = Agent("RAF1", db_refs={"HGNC": "9829"})
    obj = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    first = Activation(
        subj,
        obj,
        evidence=[Evidence(source_api="reach", text="RAF1 activates MAP2K1.")],
    )
    second = Activation(
        subj,
        obj,
        evidence=[
            Evidence(source_api="reach", text="RAF1 activates MAP2K1."),
            Evidence(source_api="reach", text="Additional evidence for the same activation."),
        ],
    )
    ingest_statements(con, [first], source_dump_id="first_dump")

    with pytest.raises(ValueError, match="evidence denominator conflict"):
        ingest_statements(con, [second], source_dump_id="second_dump")

    validation = validate_statement_evidence_denominators(con)
    assert validation.n_raw_json_evidences == 1
    assert validation.n_table_evidences == 1


def test_ingest_preserves_shared_evidence_hash_statement_contexts():
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

    ingest_statements(con, [s1, s2], source_dump_id="shared_dump")

    evidence_rows = con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    context_rows = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT evidence_hash), COUNT(DISTINCT stmt_hash) "
        "FROM statement_evidence"
    ).fetchone()
    validation = validate_statement_evidence_denominators(con)
    assert evidence_rows == 1, "canonical evidence payload remains source-hash keyed"
    assert context_rows == (2, 1, 2)
    assert validation.validated is True
    assert validation.n_raw_json_evidences == 2
    assert validation.n_table_evidences == 2


def test_epistemics_split_to_columns_and_truth_labels():
    con = _con()
    ingest_statements(con, [_phospho_stmt()], source_dump_id="test_dump")
    ev = con.execute(
        "SELECT is_direct, is_negated, is_curated FROM evidence"
    ).fetchone()
    assert ev == (True, False, False)
    # Epistemics also surface as truth_label rows
    epi_rows = con.execute(
        "SELECT field, value_text FROM truth_label "
        "WHERE truth_set_id='indra_epistemics' ORDER BY field"
    ).fetchall()
    fields = {f for f, _ in epi_rows}
    assert {"is_direct", "is_negated", "is_curated"} <= fields


def test_indra_truth_sets_registered():
    con = _con()
    ingest_statements(con, [_phospho_stmt()])
    sets = {r[0] for r in con.execute("SELECT id FROM truth_set").fetchall()}
    assert {"indra_published_belief", "indra_epistemics", "indra_grounding"} <= sets


def test_supports_edge_persisted():
    con = _con()
    stmt = _phospho_stmt()
    stmt.supports = ["uuid-x", "uuid-y"]
    stmt.supported_by = ["uuid-z"]
    counters = ingest_statements(con, [stmt])
    assert counters["n_edges"] == 3
    edges = con.execute(
        "SELECT kind, COUNT(*) FROM supports_edge GROUP BY kind ORDER BY kind"
    ).fetchall()
    assert dict(edges) == {"supports": 2, "supported_by": 1}


def test_re_ingest_idempotent():
    con = _con()
    stmt = _phospho_stmt()
    ingest_statements(con, [stmt])
    n1 = con.execute(
        "SELECT (SELECT COUNT(*) FROM statement), "
        "(SELECT COUNT(*) FROM evidence), "
        "(SELECT COUNT(*) FROM agent), "
        "(SELECT COUNT(*) FROM truth_label)"
    ).fetchone()
    # Same statement, second time → all writers idempotent on natural key
    ingest_statements(con, [stmt])
    n2 = con.execute(
        "SELECT (SELECT COUNT(*) FROM statement), "
        "(SELECT COUNT(*) FROM evidence), "
        "(SELECT COUNT(*) FROM agent), "
        "(SELECT COUNT(*) FROM truth_label)"
    ).fetchone()
    # 6 truth_labels: 1 published_belief + 2 grounding (one per agent)
    # + 3 epistemics (direct + negated + curated)
    assert n1 == (1, 1, 2, 6), f"first-ingest counts: {n1}"
    assert n2 == n1, (
        f"re-ingest changed counts: {n1} → {n2}. "
        f"Auto-registered INDRA truth_labels likely re-inserted; check "
        f"that loader uses _upsert_truth_label, not plain INSERT."
    )


def test_register_and_load_custom_truth_set():
    con = _con()
    stmt = _phospho_stmt()
    ingest_statements(con, [stmt])
    stmt_hash = con.execute("SELECT stmt_hash FROM statement").fetchone()[0]

    register_truth_set(
        con,
        id="gold_pool_v15",
        name="Project gold pool v15",
        source="project_annotators",
        version="v15",
    )
    n = load_truth_labels(con, "gold_pool_v15", [
        {"target_kind": "stmt", "target_id": stmt_hash,
         "field": "verdict", "value_text": "correct",
         "provenance": "annotator_eric"},
        {"target_kind": "stmt", "target_id": stmt_hash,
         "field": "confidence", "value_text": "high",
         "provenance": "annotator_eric"},
    ])
    assert n == 2
    rows = con.execute(
        "SELECT field, value_text FROM truth_label "
        "WHERE truth_set_id='gold_pool_v15' ORDER BY field"
    ).fetchall()
    assert rows == [("confidence", "high"), ("verdict", "correct")]


def test_load_truth_labels_is_idempotent():
    """Re-loading the same truth labels does not duplicate rows.

    Schema docstring says the writer is INSERT OR REPLACE on the natural key
    (truth_set_id, target_kind, target_id, field). Without idempotency, a
    user re-running register_truth_set + load_truth_labels (e.g. correcting
    a typo and re-loading) doubles their label counts — inflating P/R/F1
    n_compared and the deep-dive's `truth_count` display.
    """
    con = _con()
    stmt = _phospho_stmt()
    ingest_statements(con, [stmt])
    stmt_hash = con.execute("SELECT stmt_hash FROM statement").fetchone()[0]

    register_truth_set(con, id="gold_test", name="Test gold")
    labels = [
        {"target_kind": "stmt", "target_id": stmt_hash,
         "field": "verdict", "value_text": "correct",
         "provenance": "test"},
    ]
    load_truth_labels(con, "gold_test", labels)
    load_truth_labels(con, "gold_test", labels)  # second load — same rows
    rows = con.execute(
        "SELECT COUNT(*) FROM truth_label WHERE truth_set_id = 'gold_test'"
    ).fetchone()[0]
    assert rows == 1, f"expected 1 row after re-load, got {rows} (duplicate)"

    # Updating the value_text should replace, not append
    load_truth_labels(con, "gold_test", [
        {"target_kind": "stmt", "target_id": stmt_hash,
         "field": "verdict", "value_text": "incorrect",
         "provenance": "test"},
    ])
    final = con.execute(
        "SELECT value_text FROM truth_label "
        "WHERE truth_set_id = 'gold_test' AND field = 'verdict'"
    ).fetchall()
    assert final == [("incorrect",)], (
        f"expected re-load to replace value, got {final}"
    )


def test_activation_writes_subj_obj_roles():
    con = _con()
    a = Agent("RAF1", db_refs={"HGNC": "9829"})
    b = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    stmt = Activation(a, b, evidence=[Evidence(source_api="reach", text="RAF1 activates MAP2K1.")])
    ingest_statements(con, [stmt])
    roles = {r[0] for r in con.execute("SELECT DISTINCT role FROM agent").fetchall()}
    assert roles == {"subj", "obj"}


def test_supports_edge_resolves_uuid_to_stmt_hash():
    """INDRA's Statement.to_json() writes UUIDs (not stmt_hashes) into
    supports/supported_by. The loader must resolve UUID→stmt_hash for
    in-corpus refs so deep-dive's `shortHash(edge.to_stmt_hash)` actually
    points at a known statement.
    """
    con = _con()
    a = _phospho_stmt()
    b = Activation(
        Agent("RAF1", db_refs={"HGNC": "9829"}),
        Agent("MAP2K1", db_refs={"HGNC": "6840"}),
        evidence=[Evidence(source_api="reach", text="x")],
    )
    a.supports = [b.uuid]  # JSON-loaded form: list of UUID strings
    ingest_statements(con, [a, b])

    # to_stmt_hash should be b's stmt_hash (16 hex chars), not b's UUID (36 chars)
    edge_to = con.execute(
        "SELECT to_stmt_hash FROM supports_edge WHERE kind = 'supports'"
    ).fetchone()[0]
    b_hash = con.execute(
        f"SELECT stmt_hash FROM statement WHERE indra_uuid = '{b.uuid}'"
    ).fetchone()[0]
    assert edge_to == b_hash, (
        f"Expected supports_edge.to_stmt_hash to resolve to b's stmt_hash "
        f"({b_hash}), got {edge_to!r} (likely an unresolved UUID)"
    )
    assert len(edge_to) == 16, f"stmt_hash should be 16 hex chars, got {len(edge_to)}"


def test_supports_edge_keeps_uuid_when_target_outside_corpus():
    """An edge referencing an out-of-corpus stmt UUID stays as the UUID —
    honest signal that the supports graph is incomplete relative to this dump.
    """
    con = _con()
    a = _phospho_stmt()
    a.supports = ["external-uuid-not-in-corpus-1234"]  # raw UUID, not a Statement
    ingest_statements(con, [a])

    edge_to = con.execute(
        "SELECT to_stmt_hash FROM supports_edge WHERE kind = 'supports'"
    ).fetchone()[0]
    assert edge_to == "external-uuid-not-in-corpus-1234"


def test_load_truth_labels_raises_on_unregistered_truth_set():
    """Foot-gun: schema documents truth_label.truth_set_id as FK truth_set.id,
    but DuckDB doesn't enforce. App-level check prevents orphaned labels
    from a typo'd truth_set_id."""
    con = _con()
    stmt = _phospho_stmt()
    ingest_statements(con, [stmt])
    stmt_hash = con.execute("SELECT stmt_hash FROM statement").fetchone()[0]

    with pytest.raises(ValueError, match="not registered"):
        load_truth_labels(con, "demo_old", [
            {"target_kind": "stmt", "target_id": stmt_hash,
             "field": "verdict", "value_text": "correct"}
        ])
