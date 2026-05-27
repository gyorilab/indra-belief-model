"""Regression coverage for the paired architecture smoke artifact."""
from __future__ import annotations

from pathlib import Path

import duckdb

from scripts.run_paired_smoke import run_exact_support_divergence_fixture, run_paired_smoke


def test_paired_smoke_creates_overlap_without_probe_retrofit(tmp_path: Path):
    db_path = tmp_path / "paired_smoke.duckdb"
    summary = run_paired_smoke(db_path=db_path, pair_id="pair_test")

    assert summary["overlap_evidences"] == 10
    assert set(summary["runs"]) == {"decomposed", "monolithic"}

    con = duckdb.connect(str(db_path))
    try:
        runs = con.execute(
            """
            SELECT architecture, paired_run_group_id, status, n_stmts
              FROM score_run
             ORDER BY architecture
            """
        ).fetchall()
        assert runs == [
            ("decomposed", "pair_test", "succeeded", 10),
            ("monolithic", "pair_test", "succeeded", 10),
        ]

        aggregate_counts = dict(con.execute(
            """
            SELECT architecture, COUNT(*)
              FROM scorer_step
             WHERE step_kind = 'aggregate'
             GROUP BY architecture
            """
        ).fetchall())
        assert aggregate_counts == {"decomposed": 10, "monolithic": 10}

        mono_nonaggregate = con.execute(
            """
            SELECT COUNT(*)
              FROM scorer_step
             WHERE architecture = 'monolithic'
               AND step_kind != 'aggregate'
            """
        ).fetchone()[0]
        decomp_nonaggregate = con.execute(
            """
            SELECT COUNT(*)
              FROM scorer_step
             WHERE architecture = 'decomposed'
               AND step_kind != 'aggregate'
            """
        ).fetchone()[0]
        assert mono_nonaggregate == 0
        assert decomp_nonaggregate > 0

        total_hashes, distinct_hashes = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT step_hash) FROM scorer_step"
        ).fetchone()
        assert total_hashes == distinct_hashes
    finally:
        con.close()


def test_exact_label_support_state_divergence_fixture(tmp_path: Path):
    db_path = tmp_path / "paired_smoke.duckdb"
    summary = run_exact_support_divergence_fixture(db_path=db_path, pair_id="pair_divergence")
    fixture = summary["fixture"]

    assert fixture["n_overlap"] == 10
    assert fixture["n_divergence_rows"] == 2
    assert fixture["exact_label_agreement_n"] == 7
    assert fixture["support_state_agreement_n"] == 9
    assert fixture["exact_label_agreement_n"] != fixture["support_state_agreement_n"]
    assert fixture["support_state_agreement_n"] - fixture["exact_label_agreement_n"] == 2
    assert fixture["both_supported_n"] == 7
    assert fixture["monolithic_only_supported_n"] == 1
    assert fixture["decomposed_only_supported_n"] == 0
    assert fixture["neither_supported_n"] == 2
    assert {
        (r["monolithic_verdict"], r["decomposed_verdict"])
        for r in fixture["divergence_rows"]
    } == {("abstain", "incorrect"), ("incorrect", "abstain")}
    summary_step_counts = {
        (r["architecture"], r["step_kind"]): r["n"]
        for r in summary["step_counts"]
    }
    assert summary_step_counts[("monolithic", "aggregate")] == 12
    assert summary_step_counts[("decomposed", "aggregate")] == 12

    con = duckdb.connect(str(db_path))
    try:
        appended_pairs = con.execute(
            """
            SELECT stmt_hash,
                   evidence_hash,
                   string_agg(
                     architecture || ':' || json_extract_string(output_json, '$.verdict'),
                     ',' ORDER BY architecture
                   ) AS verdict_pair
              FROM scorer_step
             WHERE json_extract_string(output_json, '$.fixture') = 'exact_support_divergence'
             GROUP BY stmt_hash, evidence_hash
             ORDER BY stmt_hash, evidence_hash
            """
        ).fetchall()
        assert [r[2] for r in appended_pairs] == [
            "decomposed:incorrect,monolithic:abstain",
            "decomposed:abstain,monolithic:incorrect",
        ]

        aggregate_counts = dict(con.execute(
            """
            SELECT architecture, COUNT(*)
              FROM scorer_step
             WHERE step_kind = 'aggregate'
             GROUP BY architecture
            """
        ).fetchall())
        assert aggregate_counts == {"decomposed": 12, "monolithic": 12}

        total_hashes, distinct_hashes = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT step_hash) FROM scorer_step"
        ).fetchone()
        assert total_hashes == distinct_hashes
    finally:
        con.close()
