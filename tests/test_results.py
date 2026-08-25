"""Tests for indra_belief.results — the run-output enrichment + bucket taxonomy.

The byte-exact parity vs the old export_collaborator_data.py was verified against
the real 47k-row run; these lock the taxonomy partition and the build contract.
"""
import hashlib
import json

import pytest

from indra_belief.calibration_constants import (
    BASELINE_PROMPT_SHA256,
    REASONING_FIRST_PROMPT_SHA256,
    calibration_for,
    fitted_calibration_for,
    reader_configuration_for_run,
)
from indra_belief.results import (
    ORDER,
    _soft_calibration_block,
    build_run_export,
    call_log_cost,
    classify,
    split_preview,
    write_run_export,
)
from indra_belief.probes.calibration import (
    CALIBRATION_FILENAME,
    CALIBRATION_MODEL,
    CALIBRATION_MODEL_ID,
    CALIBRATION_PROBE_DIGEST,
    DEFAULT_CALIBRATION_PATH,
    SENTENCE_SCORE_CONTRACT_VERSION,
    SENTENCE_SCORE_KIND,
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
    p.with_suffix(".meta.json").write_text(json.dumps({
        "sentence_score": {
            "status": "enabled",
            "contract_version": SENTENCE_SCORE_CONTRACT_VERSION,
            "grain": "sentence",
            "kind": SENTENCE_SCORE_KIND,
            "calibration_model": CALIBRATION_MODEL,
            "calibration_model_id": CALIBRATION_MODEL_ID,
            "probe_id": "pol.verdict_direct",
            "probe_digest": CALIBRATION_PROBE_DIGEST,
            "calibration_artifact": CALIBRATION_FILENAME,
            "calibration_artifact_sha256": hashlib.sha256(
                DEFAULT_CALIBRATION_PATH.read_bytes()
            ).hexdigest(),
            "raw_field": "score",
            "export_field": "our_score",
            "unavailable_value": None,
        }
    }))
    return p


def _write_corpus(tmp_path, stmts):
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(stmts))
    return p


def _minimal_scored_row(*, stmt_hash="7b", evidence_i=0, source_hash=1):
    return {
        "stmt_i": 0, "evidence_i": evidence_i, "stmt_hash": stmt_hash,
        "evidence_hash": "e", "source_hash": source_hash,
        "subject": "MEK", "stmt_type": "Phosphorylation", "object": "ERK",
        "source_api": "reach", "pmid": "1", "text_len": 20, "belief": 0.9,
        "score": 0.9, "verdict": "correct", "confidence": "high",
        "raw_text_preview": "ok", "grounding_status": "all_match",
        "tier": "llm_comprehension", "provenance_triggered": False,
        "error": None, "latency_s": 0.1, "tokens": 1, "call_log": [],
    }


def test_corpus_join_recovers_unique_source_hash_and_fails_closed_on_statement_drift(tmp_path):
    corpus = [{
        "matches_hash": "123", "id": "id1", "belief": 0.9,
        "evidence": [
            {"text": "first evidence text", "source_hash": 1},
            {"text": "second evidence text", "source_hash": 2},
        ],
    }]
    corp = _write_corpus(tmp_path, corpus)

    # Stale evidence_i, but the source hash uniquely identifies the correct row.
    run = _write_run(tmp_path, [_minimal_scored_row(evidence_i=0, source_hash=2)])
    per_ev, _ps, meta, _m = build_run_export(str(run), str(corp), run_id="recover")
    assert per_ev[0]["evidence_text"] == "second evidence text"
    assert meta["join_quality"]["source_hash_recoveries"] == 1

    # A different statement hash must never consume positional corpus content.
    run = _write_run(tmp_path, [_minimal_scored_row(stmt_hash="7c")])
    per_ev, _ps, meta, _m = build_run_export(str(run), str(corp), run_id="closed")
    assert per_ev[0]["evidence_text"] == ""
    assert per_ev[0]["indra_matches_hash"] is None
    assert meta["join_quality"]["statement_hash_mismatches"] == 1


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

    per_ev, per_stmt, meta, _m = build_run_export(str(run), str(corp), run_id="r1", model="test-model")

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

    # Cost: both rows have a NON-empty call_log with NO model_id → unavailable
    # (never a fabricated $0), but observed output tokens are still reported.
    assert per_ev[0]["cost_status"] == "unavailable" and per_ev[0]["cost_usd"] is None
    assert per_ev[0]["output_tokens"] == 50 and per_ev[0]["input_tokens"] == 0
    assert per_ev[0]["n_calls"] == 1
    assert per_ev[1]["cost_status"] == "unavailable" and per_ev[1]["cost_usd"] is None
    assert per_ev[1]["output_tokens"] == 40
    assert meta["cost"]["status"] == "unavailable"
    assert meta["cost"]["total_usd"] is None
    assert meta["cost"]["n_evidence_unavailable"] == 2
    assert meta["cost"]["n_evidence_costed"] == 0
    assert meta["schema_version"] == 8


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

    # The single row has an EMPTY call_log — a no-LLM row, a genuine $0 (NOT
    # unavailable). Read it back from the written export to lock the on-disk schema.
    ev_rows = [json.loads(l) for l in (out / "per_evidence.jsonl").read_text().splitlines() if l]
    assert ev_rows[0]["cost_status"] == "known" and ev_rows[0]["cost_usd"] == 0.0
    assert ev_rows[0]["n_calls"] == 0
    assert ev_rows[0]["input_tokens"] == 0 and ev_rows[0]["output_tokens"] == 0
    assert meta["cost"]["status"] == "known"
    # The ONLY row is a no-LLM ($0) row; no call contributed to the total, so per
    # the locked aggregation (total_usd is None iff n_evidence_costed == 0) the
    # run total is None — there is genuinely no priced spend to sum. The $0 nature
    # is carried by status=="known" + n_evidence_no_llm, NOT a fabricated total.
    assert meta["cost"]["total_usd"] is None
    assert meta["cost"]["n_evidence_no_llm"] == 1
    assert meta["cost"]["n_evidence_costed"] == 0
    assert meta["cost"]["n_evidence_unavailable"] == 0
    assert meta["schema_version"] == 8
    # E5: soft_calibration block baked per run. model "m" is unfitted → named-
    # unavailable with a reason, never an imputed zero.
    assert meta["soft_calibration"]["status"] == "unavailable"
    assert meta["soft_calibration"]["soft_weights"] is None
    assert meta["soft_calibration"]["reason"]


# ── call_log_cost helper (single-pass observed USD + token totals) ────────────


def test_call_log_cost_empty_is_known_zero():
    c = call_log_cost([])
    assert c["cost_status"] == "known"
    assert c["cost_usd"] == 0.0
    assert c["n_calls"] == 0
    assert c["input_tokens"] == 0 and c["output_tokens"] == 0
    assert c["models"] == []


def test_call_log_cost_single_estimated_local_call():
    # gemma-4-26b is a local model → Bedrock-grounded ESTIMATE ($0.13/$0.40),
    # cost_status "estimated", never a fabricated $0.
    c = call_log_cost([{"model_id": "gemma-4-26b", "prompt_tokens": 2549, "out_tokens": 296}])
    assert c["cost_status"] == "estimated"
    assert c["cost_usd"] == round(2549 * 0.13 / 1e6 + 296 * 0.40 / 1e6, 6)
    assert c["cost_usd"] > 0.0
    assert c["input_tokens"] == 2549 and c["output_tokens"] == 296
    assert c["models"] == ["gemma-4-26b"]


def test_call_log_cost_single_priced_bedrock_call():
    c = call_log_cost(
        [{"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 2000, "out_tokens": 500}]
    )
    assert c["cost_status"] == "known"
    assert c["cost_usd"] == round(2000 * 3 / 1e6 + 500 * 15 / 1e6, 6) == 0.0135


def test_call_log_cost_multi_call_sum():
    c = call_log_cost([
        {"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 2000, "out_tokens": 500},
        {"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 1000, "out_tokens": 200},
    ])
    assert c["input_tokens"] == 3000 and c["output_tokens"] == 700
    assert c["n_calls"] == 2
    assert c["cost_usd"] == round((2000 + 1000) * 3 / 1e6 + (500 + 200) * 15 / 1e6, 6)


def test_call_log_cost_mixed_priced_and_unverified():
    # one priced + one unverified → unavailable, but observed token totals across
    # BOTH calls are still reported (they are facts), USD withheld.
    c = call_log_cost([
        {"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 2000, "out_tokens": 500},
        {"model_id": "vendor.unlisted-7b", "prompt_tokens": 1500, "out_tokens": 300},
    ])
    assert c["cost_status"] == "unavailable" and c["cost_usd"] is None
    assert c["input_tokens"] == 3500 and c["output_tokens"] == 800
    assert c["n_calls"] == 2
    assert c["models"] == ["anthropic.claude-sonnet-4-6", "vendor.unlisted-7b"]


def test_call_log_cost_missing_model_id_is_unavailable():
    c = call_log_cost([{"prompt_tokens": 100, "out_tokens": 20}])
    assert c["cost_status"] == "unavailable" and c["cost_usd"] is None
    assert c["input_tokens"] == 100 and c["output_tokens"] == 20


def test_call_log_cost_unknown_sentinel_is_unavailable():
    c = call_log_cost([{"model_id": "unknown", "prompt_tokens": 10, "out_tokens": 5}])
    assert c["cost_status"] == "unavailable" and c["cost_usd"] is None


def test_call_log_cost_error_call_shape_clamps_to_zero():
    c = call_log_cost([{"model_id": "gemma-4-26b", "prompt_tokens": -1, "out_tokens": 0}])
    assert c["input_tokens"] == 0 and c["output_tokens"] == 0
    # gemma-4-26b is an estimated local model → status "estimated" even though the
    # clamped (-1/0) tokens make the computed cost exactly $0.00.
    assert c["cost_status"] == "estimated" and c["cost_usd"] == 0.0


def test_call_log_cost_priced_unreported_prompt_tokens_is_a_floor():
    # FLOOR caveat (see call_log_cost docstring): a PRICED call whose prompt_tokens
    # is the unreported sentinel (-1) clamps input to 0 — input_tokens and the input
    # portion of cost are a LOWER BOUND, cost_status stays "known". Output cost is
    # still exact. This locks the documented behavior so the understatement is a
    # known, tested limitation rather than a silent surprise.
    c = call_log_cost(
        [{"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": -1, "out_tokens": 500}]
    )
    assert c["cost_status"] == "known"
    assert c["input_tokens"] == 0  # the -1 input is dropped (a floor, not the truth)
    assert c["output_tokens"] == 500
    # cost = output only (500 * 15/1M); the real input cost is unknowably missing
    assert c["cost_usd"] == round(500 * 15 / 1e6, 6) == 0.0075


def test_call_log_cost_multi_priced_models_sum_and_sort():
    # Two DISTINCT priced models in one row → cross-model USD sum, models sorted.
    c = call_log_cost([
        {"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 2000, "out_tokens": 500},
        {"model_id": "anthropic.claude-haiku-4-5", "prompt_tokens": 1000, "out_tokens": 200},
    ])
    assert c["cost_status"] == "known"
    sonnet = 2000 * 3 / 1e6 + 500 * 15 / 1e6
    haiku = 1000 * 0.8 / 1e6 + 200 * 4 / 1e6
    assert c["cost_usd"] == round(sonnet + haiku, 6)
    assert c["models"] == ["anthropic.claude-haiku-4-5", "anthropic.claude-sonnet-4-6"]
    assert c["input_tokens"] == 3000 and c["output_tokens"] == 700


# ── build_run_export cost integration (synthetic call_logs, no LLM calls) ─────


def _cost_corpus():
    return [{
        "matches_hash": "mh", "id": "id", "belief": 0.9,
        "evidence": [
            {"text": "Aaa binds Bbb in cells normally.", "source_hash": 1},
            {"text": "Ccc does not bind Ddd in cells.", "source_hash": 2},
            {"text": "Eee phosphorylates Fff strongly.", "source_hash": 3},
        ],
    }]


def _cost_row(evidence_i, source_hash, verdict, call_log):
    return {
        "stmt_i": 0, "evidence_i": evidence_i, "stmt_hash": "h", "evidence_hash": f"e{evidence_i}",
        "source_hash": source_hash, "subject": "Aaa", "stmt_type": "Phosphorylation", "object": "Bbb",
        "source_api": "reach", "pmid": "1", "text_len": 32, "belief": 0.9, "score": 0.5,
        "verdict": verdict, "confidence": "high", "raw_text_preview": "[TIER 2 LLM]\nYes.",
        "grounding_status": "all_match", "tier": "llm_comprehension", "provenance_triggered": False,
        "error": None, "latency_s": 1.0, "tokens": 10, "call_log": call_log,
    }


def test_export_cost_local_run_is_estimated(tmp_path):
    # A run scored entirely on a local model (gemma-4-26b-ollama) is no longer
    # "free" — it is GROUNDED in the Bedrock gemma-4-26b-a4b list rate, so the run
    # cost is a positive ESTIMATE and cost_status is "estimated".
    rows = [
        _cost_row(0, 1, "correct",
                  [{"model_id": "gemma-4-26b-ollama", "prompt_tokens": 2549, "out_tokens": 296, "finish_reason": "stop"}]),
        _cost_row(1, 2, "incorrect",
                  [{"model_id": "gemma-4-26b-ollama", "prompt_tokens": 1000, "out_tokens": 100, "finish_reason": "stop"}]),
    ]
    run = _write_run(tmp_path, rows)
    corp = _write_corpus(tmp_path, _cost_corpus())
    per_ev, _, meta, _m = build_run_export(str(run), str(corp), run_id="z", model="gemma")

    row0 = round(2549 * 0.13 / 1e6 + 296 * 0.40 / 1e6, 6)
    row1 = round(1000 * 0.13 / 1e6 + 100 * 0.40 / 1e6, 6)
    assert per_ev[0]["cost_usd"] == row0 and per_ev[0]["cost_status"] == "estimated"
    assert per_ev[0]["input_tokens"] == 2549 and per_ev[0]["output_tokens"] == 296
    assert per_ev[0]["n_calls"] == 1
    assert meta["cost"]["status"] == "estimated"
    assert meta["cost"]["total_usd"] == round(row0 + row1, 4)
    assert meta["cost"]["total_usd"] > 0.0
    assert meta["cost"]["models"] == ["gemma-4-26b-ollama"]
    assert meta["cost"]["n_evidence_costed"] == 2
    assert meta["cost"]["n_evidence_unavailable"] == 0


def test_export_cost_priced_bedrock_run_sums(tmp_path):
    # one 1-call row + one 2-call row (Complex relnature second call), all priced.
    rows = [
        _cost_row(0, 1, "correct",
                  [{"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 2000, "out_tokens": 500}]),
        _cost_row(1, 2, "incorrect", [
            {"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 1000, "out_tokens": 200},
            {"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 800, "out_tokens": 150, "finish_reason": "stop"},
        ]),
    ]
    run = _write_run(tmp_path, rows)
    corp = _write_corpus(tmp_path, _cost_corpus())
    per_ev, _, meta, _m = build_run_export(str(run), str(corp), run_id="b", model="sonnet")

    by_i = {r["evidence_i"]: r for r in per_ev}
    # 2-call row: output_tokens is the SUM (350) but gen_out_tokens is LAST-call (150)
    assert by_i[1]["output_tokens"] == 350
    assert by_i[1]["gen_out_tokens"] == 150
    assert by_i[1]["output_tokens"] > by_i[1]["gen_out_tokens"]

    total = (2000 + 1000 + 800) * 3 / 1e6 + (500 + 200 + 150) * 15 / 1e6
    assert meta["cost"]["status"] == "known"
    assert meta["cost"]["total_usd"] == round(total, 4)
    assert meta["cost"]["usd_per_1k_evidence"] == round(total / 2 * 1000, 4)
    assert meta["cost"]["n_evidence_costed"] == 2


def test_export_cost_mixed_priced_and_unverified_is_partial(tmp_path):
    # Synthetic run covering ALL THREE classes the spec asks for:
    #   row 0: Bedrock-PRICED list (anthropic.claude-sonnet-4-6) → costed
    #   row 1: local ESTIMATED (gemma-4-26b) → costed at its Bedrock-grounded estimate
    #   row 2: UNVERIFIED-price (vendor.unlisted-7b) → unavailable, excluded
    rows = [
        _cost_row(0, 1, "correct",
                  [{"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 2000, "out_tokens": 500}]),
        _cost_row(1, 2, "incorrect",
                  [{"model_id": "gemma-4-26b", "prompt_tokens": 1000, "out_tokens": 100}]),
        _cost_row(2, 3, "correct",
                  [{"model_id": "vendor.unlisted-7b", "prompt_tokens": 1500, "out_tokens": 300}]),
    ]
    run = _write_run(tmp_path, rows)
    corp = _write_corpus(tmp_path, _cost_corpus())
    per_ev, _, meta, _m = build_run_export(str(run), str(corp), run_id="m", model="mixed")

    by_i = {r["evidence_i"]: r for r in per_ev}
    assert by_i[0]["cost_status"] == "known"
    assert by_i[0]["cost_usd"] == round(2000 * 3 / 1e6 + 500 * 15 / 1e6, 6)
    by1 = round(1000 * 0.13 / 1e6 + 100 * 0.40 / 1e6, 6)
    assert by_i[1]["cost_status"] == "estimated" and by_i[1]["cost_usd"] == by1
    assert by_i[2]["cost_status"] == "unavailable" and by_i[2]["cost_usd"] is None
    # the unverified row still reports its observed tokens
    assert by_i[2]["input_tokens"] == 1500 and by_i[2]["output_tokens"] == 300

    # row 2 is unavailable → the run is "partial" (dominates "estimated").
    assert meta["cost"]["status"] == "partial"
    assert meta["cost"]["n_evidence_unavailable"] == 1
    assert meta["cost"]["n_evidence_costed"] == 2  # priced + estimated both count
    sonnet = round(2000 * 3 / 1e6 + 500 * 15 / 1e6, 6)  # + the gemma estimate
    assert meta["cost"]["total_usd"] == round(sonnet + by1, 4)
    assert "vendor.unlisted-7b" in meta["cost"]["models"]
    assert meta["schema_version"] == 8


def test_export_cost_multi_priced_models_run_sums_and_sorts(tmp_path):
    # A run mixing TWO distinct PRICED models (sonnet + haiku). Locks the
    # cross-model run total AND the sorted `models` list for multi-priced runs.
    rows = [
        _cost_row(0, 1, "correct",
                  [{"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 2000, "out_tokens": 500}]),
        _cost_row(1, 2, "incorrect",
                  [{"model_id": "anthropic.claude-haiku-4-5", "prompt_tokens": 1000, "out_tokens": 200}]),
    ]
    run = _write_run(tmp_path, rows)
    corp = _write_corpus(tmp_path, _cost_corpus())
    _, _, meta, _m = build_run_export(str(run), str(corp), run_id="mp", model="multi-priced")

    sonnet = 2000 * 3 / 1e6 + 500 * 15 / 1e6
    haiku = 1000 * 0.8 / 1e6 + 200 * 4 / 1e6
    assert meta["cost"]["status"] == "known"
    assert meta["cost"]["total_usd"] == round(sonnet + haiku, 4)
    assert meta["cost"]["n_evidence_costed"] == 2
    assert meta["cost"]["models"] == [
        "anthropic.claude-haiku-4-5", "anthropic.claude-sonnet-4-6",
    ]


def test_export_cost_empty_run_is_known_zero(tmp_path):
    # An empty run (0 rows) routes to status=="known", total_usd==None — the same
    # KNOWN-$0 shape as an all-no-LLM run (no priced spend to sum). The $0 is
    # carried by status + the (all-zero) row counts, never a fabricated total.
    run = tmp_path / "empty.jsonl"
    run.write_text("")
    corp = _write_corpus(tmp_path, _cost_corpus())
    per_ev, per_stmt, meta, _m = build_run_export(str(run), str(corp), run_id="e", model="empty")

    assert per_ev == [] and per_stmt == []
    assert meta["cost"]["status"] == "known"
    assert meta["cost"]["total_usd"] is None
    assert meta["cost"]["usd_per_1k_evidence"] is None
    assert meta["cost"]["n_evidence_costed"] == 0
    assert meta["cost"]["n_evidence_no_llm"] == 0
    assert meta["cost"]["n_evidence_unavailable"] == 0
    assert meta["cost"]["models"] == []


def _one_row(tmp_path, call_log):
    corpus = [{"matches_hash": "mh1", "id": "id1", "belief": 0.9,
               "evidence": [{"text": "MEK phosphorylates ERK in cells.", "source_hash": 1}]}]
    rows = [{"stmt_i": 0, "evidence_i": 0, "stmt_hash": "h1", "evidence_hash": "e0", "source_hash": 1,
             "subject": "MAP2K1", "stmt_type": "Phosphorylation", "object": "MAPK1", "source_api": "reach",
             "pmid": "111", "text_len": 32, "belief": 0.9, "score": 0.05, "verdict": "incorrect",
             "confidence": "high", "raw_text_preview": "...", "grounding_status": "all_match",
             "tier": "llm_comprehension", "provenance_triggered": False, "error": None,
             "latency_s": 1.2, "tokens": 50, "call_log": call_log}]
    return build_run_export(str(_write_run(tmp_path, rows)),
                            str(_write_corpus(tmp_path, corpus)), run_id="r1", model="test-model")


def test_build_run_export_carries_reasoning_trace(tmp_path):
    rt = {"free_cot": "x" * 50, "status": "encrypted", "reasoning_tokens": 172,
          "provider_source": "bedrock_responses.output[].reasoning", "backend": "bedrock_responses",
          "model_id": "openai.gpt-5.5", "finish_reason": "stop",
          "committed_justification": {"support": "MEK phosphorylates ERK", "objection": None, "source": "answer_json"}}
    per_ev, _ps, meta, _m = _one_row(tmp_path, [{"finish_reason": "stop", "out_tokens": 50, "reasoning_trace": rt}])
    tr = per_ev[0]["reasoning_trace"]
    assert tr is not None
    assert tr["status"] == "encrypted" and tr["reasoning_tokens"] == 172
    assert tr["committed_justification"]["support"] == "MEK phosphorylates ERK"
    assert tr["free_cot_chars"] == 50
    assert meta["reasoning_quality"]["trace_status"]["encrypted"] == 1
    assert meta["schema_version"] == 8


def test_build_run_export_legacy_row_has_null_reasoning_trace(tmp_path):
    # call_log without reasoning_trace (run scored before the trace existed)
    per_ev, _ps, meta, _m = _one_row(tmp_path, [{"finish_reason": "stop", "out_tokens": 50}])
    assert per_ev[0]["reasoning_trace"] is None
    assert meta["reasoning_quality"]["trace_status"].get("no_trace") == 1


def _confirmed_read_export(tmp_path, model):
    # A statement with one CONFIRMED read (verdict='correct'): on a confirmed
    # read the calibrated hybrid score shifts the belief scale vs the hard gate,
    # so belief != belief_hard is observable for a fitted configuration.
    corpus = [{"matches_hash": "mh1", "id": "id1", "belief": 0.9,
               "evidence": [{"text": "MEK phosphorylates ERK in cells.", "source_hash": 1}]}]
    rows = [{"stmt_i": 0, "evidence_i": 0, "stmt_hash": "hc", "evidence_hash": "e0", "source_hash": 1,
             "subject": "MAP2K1", "stmt_type": "Phosphorylation", "object": "MAPK1", "source_api": "reach",
             "pmid": "111", "text_len": 32, "belief": 0.9, "score": 0.95, "verdict": "correct",
             "confidence": "high", "raw_text_preview": '[TIER 2 LLM]\nYes.',
             "grounding_status": "all_match", "tier": "llm_comprehension", "provenance_triggered": False,
             "error": None, "latency_s": 1.0, "tokens": 50, "call_log": [{"finish_reason": "stop", "out_tokens": 50}]}]
    prompt_sha256 = (
        REASONING_FIRST_PROMPT_SHA256
        if model == "local-gemma-4-26b" else BASELINE_PROMPT_SHA256
    )
    _pe, per_stmt, _meta, _m = build_run_export(
        str(_write_run(tmp_path, rows)), str(_write_corpus(tmp_path, corpus)),
        run_id="k1", model=model, prompt_sha256=prompt_sha256)
    return per_stmt


def test_canonical_belief_is_calibrated_for_remote_fitted_reader(tmp_path):
    # K1: a FITTED reader configuration → canonical belief is the calibrated
    # arm and shifts off the hard gate on a confirmed read.
    per_stmt = _confirmed_read_export(tmp_path, model="remote-gemma-4-26b")
    assert len(per_stmt) == 1
    s = per_stmt[0]
    assert s["belief_soft"] is not None
    assert s["belief"] == s["belief_soft"]
    assert s["belief"] != s["belief_hard"]


def test_canonical_belief_is_calibrated_for_a_fitted_reader(tmp_path):
    # Reasoning-first Bedrock Gemma has its own measured configuration profile;
    # it must not fall back to hard or inherit the remote profile implicitly.
    remote = _confirmed_read_export(tmp_path, model="remote-gemma-4-26b")[0]
    per_stmt = _confirmed_read_export(tmp_path, model="local-gemma-4-26b")
    assert len(per_stmt) == 1
    s = per_stmt[0]
    assert s["belief_soft"] is not None
    assert s["belief"] == s["belief_soft"]
    assert s["belief"] != s["belief_hard"]
    assert s["belief"] != remote["belief"]


@pytest.mark.parametrize("model", ["local-gemma-4-31b", "some-unrecognized-reader"])
def test_canonical_belief_falls_back_to_hard_for_unfitted_configuration(tmp_path, model):
    # Same weights on an unvalidated host and wholly unknown readers are both
    # named-empty: canonical belief remains the hard-gate fallback.
    per_stmt = _confirmed_read_export(tmp_path, model=model)
    assert len(per_stmt) == 1
    s = per_stmt[0]
    assert s["belief_soft"] is None
    assert s["belief"] == s["belief_hard"]


# ── unavailability prose names the half that actually disagrees ──────────────
#
# reader_configuration_for_run collapses two independent cross-checks (monolithic
# prompt digest, served model id) into one status, so the sentence the viewer
# renders has to read the evidence back out. The three PROMPT-side sentences are
# FROZEN prose — they are quoted verbatim in already-generated data/exports
# artifacts — so these are byte-literal pins, not "a reason exists" smoke tests.
# The two MODEL-side sentences are the only new strings.

_REASON_SYSTEM_A = "You judge whether a biomedical text-mining extraction is correct.\n"
_REASON_SYSTEM_B = _REASON_SYSTEM_A + "Answer strictly in JSON.\n"
_REASON_SYSTEM_A_SHA256 = hashlib.sha256(_REASON_SYSTEM_A.encode("utf-8")).hexdigest()


def _reason_call(system=None, model_id=None, kind="monolithic"):
    call = {"kind": kind}
    if system is not None:
        call["system"] = system
    if model_id is not None:
        call["model_id"] = model_id
    return call


def _status_and_reason(tmp_path, call_logs, meta):
    """Resolve a synthesized run exactly as build_run_metrics does, then bake
    the soft-calibration block and hand back (status, reason)."""
    run = tmp_path / "reason_run.jsonl"
    run.write_text("".join(
        json.dumps({"stmt_i": i, "evidence_i": 0, "call_log": cl}) + "\n"
        for i, cl in enumerate(call_logs)
    ))
    run.with_suffix(".meta.json").write_text(json.dumps(meta))
    config = reader_configuration_for_run(run, meta.get("model"))
    block = _soft_calibration_block(
        config["model"], config,
        calibration_for(config["model"], prompt_sha256=config["prompt_sha256"]),
        fitted_calibration_for(config["model"], prompt_sha256=config["prompt_sha256"]),
    )
    assert block["status"] == "unavailable"
    return config["status"], block["reason"]


def test_reason_prompt_mixed_is_byte_identical(tmp_path):
    status, reason = _status_and_reason(
        tmp_path,
        [[_reason_call(_REASON_SYSTEM_A, "gemma-4-26b-ollama")],
         [_reason_call(_REASON_SYSTEM_B, "gemma-4-26b-ollama")]],
        {"model": "gemma-remote"},
    )
    assert status == "mixed"
    assert reason == "run contains more than one monolithic prompt fingerprint"


def test_reason_prompt_mismatch_is_byte_identical(tmp_path):
    status, reason = _status_and_reason(
        tmp_path,
        [[_reason_call(_REASON_SYSTEM_A, "gemma-4-26b-ollama")]],
        {"model": "gemma-remote", "prompt_sha256": BASELINE_PROMPT_SHA256},
    )
    assert status == "mismatch"
    assert reason == "declared prompt fingerprint disagrees with persisted call logs"


def test_reason_missing_prompt_is_byte_identical(tmp_path):
    status, reason = _status_and_reason(
        tmp_path,
        [[_reason_call(model_id="gemma-4-26b-ollama", kind="relation_nature")]],
        {"model": "gemma-remote"},
    )
    assert status == "missing_prompt"
    assert reason == "run has no persisted monolithic system prompt fingerprint"


def test_reason_model_mixed_names_the_model_half(tmp_path):
    # ONE prompt digest, TWO served ids: the prompt half agrees with itself, so
    # the sentence must not blame it.
    status, reason = _status_and_reason(
        tmp_path,
        [[_reason_call(_REASON_SYSTEM_A, "gemma-4-26b-ollama")],
         [_reason_call(_REASON_SYSTEM_A, "medpsy-4b")]],
        {"model": "gemma-remote", "prompt_sha256": _REASON_SYSTEM_A_SHA256},
    )
    assert status == "mixed"
    assert reason == "run call logs record more than one served model id"


def test_reason_model_mismatch_names_the_model_half(tmp_path):
    # The declared prompt digest matches the persisted one exactly; only the
    # served id contradicts the declared model.
    status, reason = _status_and_reason(
        tmp_path,
        [[_reason_call(_REASON_SYSTEM_A, "medpsy-4b")]],
        {"model": "gemma-remote", "prompt_sha256": _REASON_SYSTEM_A_SHA256},
    )
    assert status == "mismatch"
    assert reason == (
        "declared model disagrees with the served model id in persisted call logs"
    )
