"""Tests for indra_belief.results — the run-output enrichment + bucket taxonomy.

The byte-exact parity vs the old export_collaborator_data.py was verified against
the real 47k-row run; these lock the taxonomy partition and the build contract.
"""
import json

from indra_belief.results import (
    ORDER,
    build_run_export,
    call_log_cost,
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
    assert meta["schema_version"] == 4


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
    assert meta["schema_version"] == 4
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


def test_call_log_cost_single_zero_cost_call():
    c = call_log_cost([{"model_id": "gemma-4-26b", "prompt_tokens": 2549, "out_tokens": 296}])
    assert c["cost_usd"] == 0.0 and c["cost_status"] == "known"
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
        {"model_id": "gemma-4-26b-a4b-it", "prompt_tokens": 1500, "out_tokens": 300},
    ])
    assert c["cost_status"] == "unavailable" and c["cost_usd"] is None
    assert c["input_tokens"] == 3500 and c["output_tokens"] == 800
    assert c["n_calls"] == 2
    assert c["models"] == ["anthropic.claude-sonnet-4-6", "gemma-4-26b-a4b-it"]


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
    assert c["cost_status"] == "known" and c["cost_usd"] == 0.0


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


def test_export_cost_zero_cost_run_is_known(tmp_path):
    rows = [
        _cost_row(0, 1, "correct",
                  [{"model_id": "gemma-4-26b-ollama", "prompt_tokens": 2549, "out_tokens": 296, "finish_reason": "stop"}]),
        _cost_row(1, 2, "incorrect",
                  [{"model_id": "gemma-4-26b-ollama", "prompt_tokens": 1000, "out_tokens": 100, "finish_reason": "stop"}]),
    ]
    run = _write_run(tmp_path, rows)
    corp = _write_corpus(tmp_path, _cost_corpus())
    per_ev, _, meta = build_run_export(str(run), str(corp), run_id="z", model="gemma")

    assert per_ev[0]["cost_usd"] == 0.0 and per_ev[0]["cost_status"] == "known"
    assert per_ev[0]["input_tokens"] == 2549 and per_ev[0]["output_tokens"] == 296
    assert per_ev[0]["n_calls"] == 1
    assert meta["cost"]["status"] == "known"
    assert meta["cost"]["total_usd"] == 0.0
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
    per_ev, _, meta = build_run_export(str(run), str(corp), run_id="b", model="sonnet")

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
    #   row 0: Bedrock-PRICED (anthropic.claude-sonnet-4-6) → costed
    #   row 1: local ZERO-COST (gemma-4-26b) → costed at $0
    #   row 2: UNVERIFIED-price (gemma-4-26b-a4b-it) → unavailable, excluded
    rows = [
        _cost_row(0, 1, "correct",
                  [{"model_id": "anthropic.claude-sonnet-4-6", "prompt_tokens": 2000, "out_tokens": 500}]),
        _cost_row(1, 2, "incorrect",
                  [{"model_id": "gemma-4-26b", "prompt_tokens": 1000, "out_tokens": 100}]),
        _cost_row(2, 3, "correct",
                  [{"model_id": "gemma-4-26b-a4b-it", "prompt_tokens": 1500, "out_tokens": 300}]),
    ]
    run = _write_run(tmp_path, rows)
    corp = _write_corpus(tmp_path, _cost_corpus())
    per_ev, _, meta = build_run_export(str(run), str(corp), run_id="m", model="mixed")

    by_i = {r["evidence_i"]: r for r in per_ev}
    assert by_i[0]["cost_status"] == "known"
    assert by_i[0]["cost_usd"] == round(2000 * 3 / 1e6 + 500 * 15 / 1e6, 6)
    assert by_i[1]["cost_status"] == "known" and by_i[1]["cost_usd"] == 0.0
    assert by_i[2]["cost_status"] == "unavailable" and by_i[2]["cost_usd"] is None
    # the unverified row still reports its observed tokens
    assert by_i[2]["input_tokens"] == 1500 and by_i[2]["output_tokens"] == 300

    assert meta["cost"]["status"] == "partial"
    assert meta["cost"]["n_evidence_unavailable"] == 1
    assert meta["cost"]["n_evidence_costed"] == 2  # priced + zero-cost both count
    only_priced = 2000 * 3 / 1e6 + 500 * 15 / 1e6  # zero-cost row adds $0
    assert meta["cost"]["total_usd"] == round(only_priced, 4)
    assert "gemma-4-26b-a4b-it" in meta["cost"]["models"]
    assert meta["schema_version"] == 4


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
    _, _, meta = build_run_export(str(run), str(corp), run_id="mp", model="multi-priced")

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
    per_ev, per_stmt, meta = build_run_export(str(run), str(corp), run_id="e", model="empty")

    assert per_ev == [] and per_stmt == []
    assert meta["cost"]["status"] == "known"
    assert meta["cost"]["total_usd"] is None
    assert meta["cost"]["usd_per_1k_evidence"] is None
    assert meta["cost"]["n_evidence_costed"] == 0
    assert meta["cost"]["n_evidence_no_llm"] == 0
    assert meta["cost"]["n_evidence_unavailable"] == 0
    assert meta["cost"]["models"] == []
