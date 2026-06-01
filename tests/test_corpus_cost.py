"""Tests for cost estimation (Phase 0.1 helper)."""

from __future__ import annotations

import pytest
from indra.statements import Agent, Evidence, Phosphorylation

from indra_belief.corpus import estimate_cost


def _stmt(n_ev: int):
    a = Agent("A", db_refs={"HGNC": "1"})
    b = Agent("B", db_refs={"HGNC": "2"})
    evs = [Evidence(source_api="reach", text=f"sentence {i}") for i in range(n_ev)]
    return Phosphorylation(a, b, evidence=evs)


def test_estimate_zero_stmts():
    out = estimate_cost([])
    assert out["n_stmts"] == 0
    assert out["n_evidences_est"] == 0
    assert out["n_llm_calls_est"] == 0
    assert out["cost_usd"] == 0.0


def test_estimate_uses_actual_evidence_count():
    stmts = [_stmt(2), _stmt(3), _stmt(1)]  # total 6 evidences
    out = estimate_cost(stmts, model_id="claude-sonnet-4-6",
                       avg_llm_calls_per_evidence=5.0)
    assert out["n_stmts"] == 3
    assert out["n_evidences_est"] == 6
    assert out["n_llm_calls_est"] == 30  # 6 * 5
    # Sonnet: $3/M in + $15/M out, 330 in / 70 out per call
    # 30 * 330 = 9900 in tokens; 30 * 70 = 2100 out tokens
    expected = 9900 * (3.00 / 1_000_000) + 2100 * (15.00 / 1_000_000)
    assert out["cost_usd"] == pytest.approx(round(expected, 4))


def test_estimate_monolithic_defaults_to_one_call_per_evidence():
    stmts = [_stmt(2), _stmt(3), _stmt(1)]
    out = estimate_cost(stmts, model_id="claude-sonnet-4-6",
                        architecture="monolithic")
    assert out["n_evidences_est"] == 6
    assert out["n_llm_calls_est"] == 6
    assert out["assumptions"]["architecture"] == "monolithic"
    assert out["assumptions"]["avg_llm_calls_per_evidence"] == 1.0


def test_estimate_probe_only_uses_selected_probe_count():
    stmts = [_stmt(2), _stmt(1)]
    out = estimate_cost(
        stmts,
        model_id="claude-sonnet-4-6",
        architecture="decomposed",
        probe_only=True,
        probe_step_filter=["object_role_probe", "scope_probe"],
    )
    assert out["n_evidences_est"] == 3
    assert out["n_llm_calls_est"] == 6
    assert out["assumptions"]["scoring_mode"] == "probe_only"
    assert out["assumptions"]["probe_step_filter"] == [
        "object_role_probe",
        "scope_probe",
    ]


def test_estimate_probe_only_requires_decomposed_probe_filter():
    with pytest.raises(ValueError, match="probe_step_filter"):
        estimate_cost([_stmt(1)], architecture="decomposed", probe_only=True)
    with pytest.raises(ValueError, match="decomposed"):
        estimate_cost(
            [_stmt(1)],
            architecture="monolithic",
            probe_only=True,
            probe_step_filter=["scope_probe"],
        )


def test_estimate_unknown_model_warns_and_zeros():
    stmts = [_stmt(1)]
    out = estimate_cost(stmts, model_id="nonexistent-model-9000")
    # Unknown model defaults to 0 prices → 0 cost
    assert out["cost_usd"] == 0.0


def test_estimate_override_prices():
    stmts = [_stmt(1)]  # 1 evidence
    # 5 LLM calls × 330 in × $1/M = $0.00165
    # 5 LLM calls × 70 out × $5/M = $0.00175
    out = estimate_cost(stmts, model_id="custom",
                       in_price_per_m=1.0, out_price_per_m=5.0)
    expected = 5 * 330 * (1.0 / 1_000_000) + 5 * 70 * (5.0 / 1_000_000)
    assert out["cost_usd"] == pytest.approx(round(expected, 4))


def test_estimate_rasmachine_scale_sonnet():
    """Smoke-test against the cost-estimate range in the task graph (0.1)."""
    # 8,724 stmts with avg ~3 evidences, default LLM calls
    out = estimate_cost([_stmt(3) for _ in range(8724)],
                       model_id="claude-sonnet-4-6")
    # Per task graph 0.1 estimate: ~$264 for Sonnet on full rasmachine
    # Allow ±20% for assumption drift
    assert 200 < out["cost_usd"] < 350
    assert out["n_evidences_est"] == 8724 * 3
    assert out["n_llm_calls_est"] == 8724 * 3 * 5


def test_estimate_includes_assumptions():
    out = estimate_cost([_stmt(2)], model_id="claude-sonnet-4-6")
    assert "assumptions" in out
    assert out["assumptions"]["avg_evidences_per_stmt"] == 2.0
    assert out["assumptions"]["avg_llm_calls_per_evidence"] == 5.0
    assert out["assumptions"]["in_price_per_m_tokens_usd"] == 3.00
