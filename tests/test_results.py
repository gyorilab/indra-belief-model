"""Tests for indra_belief.results — the run-output enrichment + bucket taxonomy.

The byte-exact parity vs the old export_collaborator_data.py was verified against
the real 47k-row run; these lock the taxonomy partition and the build contract.
"""
import json

from indra_belief.results import (
    ORDER,
    build_run_export,
    classify,
    split_preview,
    write_run_export,
)


def test_bucket_partition_precedence():
    # telemetry > schema-shape > verdict-based
    assert classify({"verdict": "correct", "text_len": 50}, "ev", "") == "semantic_correct"
    assert classify({"verdict": None, "error": "boom", "text_len": 50}, "", "") == "row_error"
    assert classify({"verdict": "incorrect", "text_len": 0}, "", "") == "no_evidence"
    assert classify({"verdict": "incorrect", "text_len": 10}, "", "") == "placeholder_text"
    assert (
        classify({"verdict": "incorrect", "text_len": 50, "stmt_type": "ActiveForm", "object": "?"}, "", "")
        == "incomplete_claim"
    )
    assert (
        classify({"verdict": "incorrect", "text_len": 50}, "", "evidence does not mention X")
        == "reader_hallucination"
    )
    assert classify({"verdict": "incorrect", "text_len": 50}, "X may bind Y", "reasoning") == "hedged_evidence"
    assert classify({"verdict": "incorrect", "text_len": 50}, "X binds Y", "ok") == "semantic_incorrect"


def test_split_preview_validates_against_text_len():
    p = 'The evidence is: "MEK phosphorylates ERK."\nReasoning: clearly stated.'
    ev, reasoning = split_preview(p, text_len=24)
    assert ev == "MEK phosphorylates ERK."
    assert "Reasoning" in reasoning
    # text_len 0 → no evidence captured (empty marker); reasoning is the whole preview
    assert split_preview("just reasoning, no quote", text_len=0) == ("", "just reasoning, no quote")


def _write_run(tmp_path, rows):
    p = tmp_path / "run.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _write_corpus(tmp_path, stmts):
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(stmts))
    return p


def test_build_run_export_joins_text_and_rolls_up(tmp_path):
    corpus = [
        {
            "matches_hash": "mh1", "id": "id1", "belief": 0.9,
            "evidence": [
                {"text": "MEK phosphorylates ERK in cells.", "source_hash": 1},
                {"text": "MEK does not bind DNA.", "source_hash": 2},
            ],
        }
    ]
    rows = [
        {"stmt_i": 0, "evidence_i": 0, "stmt_hash": "h1", "evidence_hash": "e0", "source_hash": 1,
         "subject": "MAP2K1", "stmt_type": "Phosphorylation", "object": "MAPK1", "source_api": "reach",
         "pmid": "111", "text_len": 32, "belief": 0.9, "score": 0.95, "verdict": "correct",
         "confidence": "high", "raw_text_preview": '[TIER 2 LLM]\nThe evidence is: "MEK phosphorylates ERK in cells."\nYes.',
         "grounding_status": "all_match", "tier": "llm_comprehension", "provenance_triggered": False,
         "error": None, "latency_s": 1.2, "tokens": 50, "call_log": [{"finish_reason": "stop", "out_tokens": 50}]},
        {"stmt_i": 0, "evidence_i": 1, "stmt_hash": "h1", "evidence_hash": "e1", "source_hash": 2,
         "subject": "MAP2K1", "stmt_type": "Phosphorylation", "object": "MAPK1", "source_api": "reach",
         "pmid": "222", "text_len": 22, "belief": 0.9, "score": 0.05, "verdict": "incorrect",
         "confidence": "high", "raw_text_preview": '[TIER 2 LLM]\nThe evidence is: "MEK does not bind DNA."\nNo.',
         "grounding_status": "all_match", "tier": "llm_comprehension", "provenance_triggered": False,
         "error": None, "latency_s": 1.0, "tokens": 40, "call_log": [{"finish_reason": "stop", "out_tokens": 40}]},
    ]
    run = _write_run(tmp_path, rows)
    corp = _write_corpus(tmp_path, corpus)

    per_ev, per_stmt, meta = build_run_export(str(run), str(corp), run_id="r1", model="test-model")

    assert len(per_ev) == 2
    # evidence text was joined from the corpus, not the run
    assert per_ev[0]["evidence_text"] == "MEK phosphorylates ERK in cells."
    assert per_ev[0]["indra_matches_hash"] == "mh1" and per_ev[0]["indra_id"] == "id1"
    assert per_ev[0]["bucket"] == "semantic_correct"

    assert len(per_stmt) == 1
    s = per_stmt[0]
    assert s["stmt_hash"] == "h1"
    assert s["n_evidence"] == 2 and s["n_correct"] == 1 and s["n_incorrect"] == 1
    assert s["our_mean_score"] == 0.5
    assert s["rasmachine_belief"] == 0.9

    assert meta["run_id"] == "r1" and meta["model"] == "test-model"
    assert sum(meta["bucket_counts"].values()) == 2
    assert set(meta["bucket_counts"]) == set(ORDER)


def test_write_run_export_emits_three_files(tmp_path):
    corpus = [{"matches_hash": "mh", "id": "id", "belief": 1.0,
               "evidence": [{"text": "A binds B.", "source_hash": 9}]}]
    rows = [{"stmt_i": 0, "evidence_i": 0, "stmt_hash": "h", "evidence_hash": "e", "source_hash": 9,
             "subject": "A", "stmt_type": "Complex", "object": "B", "source_api": "bel", "pmid": "1",
             "text_len": 10, "belief": 1.0, "score": 0.8, "verdict": "correct", "confidence": "medium",
             "raw_text_preview": "ok", "grounding_status": "all_match", "tier": "llm_comprehension",
             "provenance_triggered": False, "error": None, "latency_s": 1.0, "tokens": 5, "call_log": []}]
    run = _write_run(tmp_path, rows)
    corp = _write_corpus(tmp_path, corpus)
    out = tmp_path / "export"
    meta = write_run_export(str(run), str(corp), str(out), run_id="rr", model="m")
    assert (out / "per_evidence.jsonl").exists()
    assert (out / "per_statement.json").exists()
    assert (out / "export_meta.json").exists()
    assert meta["counts"]["unique_evidence_rows"] == 1
