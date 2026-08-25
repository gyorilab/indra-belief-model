"""Tests for cost estimation (Phase 0.1 helper)."""

from __future__ import annotations

import pytest
from indra.statements import Agent, Evidence, Phosphorylation

from indra_belief.corpus import estimate_cost
from indra_belief.corpus.cost import model_has_known_cost, price_basis, token_cost_usd


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


def test_estimate_defaults_to_one_call_per_evidence():
    """The default must be 1.0. It was 5.0 while an `architecture` argument
    defaulted to the decomposed scorer; that arch is gone, and a default of 5
    would silently quintuple every estimate taken without an explicit call
    count."""
    stmts = [_stmt(2), _stmt(3), _stmt(1)]
    out = estimate_cost(stmts, model_id="claude-sonnet-4-6")
    assert out["n_evidences_est"] == 6
    assert out["n_llm_calls_est"] == 6
    assert out["assumptions"]["architecture"] == "monolithic"
    assert out["assumptions"]["avg_llm_calls_per_evidence"] == 1.0


def test_estimate_unknown_model_warns_and_zeros():
    stmts = [_stmt(1)]
    out = estimate_cost(stmts, model_id="nonexistent-model-9000")
    # Unknown model defaults to 0 prices → 0 cost
    assert out["cost_usd"] == 0.0


def test_estimate_override_prices():
    stmts = [_stmt(1)]  # 1 evidence
    # 1 LLM call × 330 in × $1/M + 1 LLM call × 70 out × $5/M
    out = estimate_cost(stmts, model_id="custom",
                       in_price_per_m=1.0, out_price_per_m=5.0)
    expected = 1 * 330 * (1.0 / 1_000_000) + 1 * 70 * (5.0 / 1_000_000)
    assert out["cost_usd"] == pytest.approx(round(expected, 4))


def test_estimate_rasmachine_scale_sonnet():
    """Full-rasmachine projection at ONE call per evidence.

    The task-graph 0.1 figure this used to assert (~$264, bounded 200-350) was
    computed at the decomposed scorer's five calls per evidence. That
    architecture is gone; the monolithic scorer issues one call, so the honest
    projection is a fifth of it. The number moved because the thing being
    estimated moved, not because an assumption drifted.
    """
    out = estimate_cost([_stmt(3) for _ in range(8724)],
                       model_id="claude-sonnet-4-6")
    assert 40 < out["cost_usd"] < 70
    assert out["n_evidences_est"] == 8724 * 3
    assert out["n_llm_calls_est"] == 8724 * 3


def test_estimate_includes_assumptions():
    out = estimate_cost([_stmt(2)], model_id="claude-sonnet-4-6")
    assert "assumptions" in out
    assert out["assumptions"]["avg_evidences_per_stmt"] == 2.0
    assert out["assumptions"]["avg_llm_calls_per_evidence"] == 1.0
    assert out["assumptions"]["in_price_per_m_tokens_usd"] == 3.00


# ── observed-cost price-table contract (token_cost_usd / model_has_known_cost) ──


def test_bedrock_prefixed_sonnet_is_priced():
    # 1M in @ $3.00 + 1M out @ $15.00 = $18.00
    assert token_cost_usd("anthropic.claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.0
    assert model_has_known_cost("anthropic.claude-sonnet-4-6") is True


def test_bedrock_prefixed_haiku_is_priced():
    assert token_cost_usd("anthropic.claude-haiku-4-5", 1_000_000, 0) == 0.80
    assert model_has_known_cost("anthropic.claude-haiku-4-5") is True


def test_flagship_gemma_is_bedrock_estimated():
    # gemma-remote (bare + ollama call_log ids) is NO LONGER free — it is GROUNDED
    # in the Bedrock gemma-4-26b-a4b list rate ($0.13/$0.40), basis 'estimate'.
    for mid in ("gemma-4-26b", "gemma-4-26b-ollama"):
        assert model_has_known_cost(mid) is True
        assert price_basis(mid) == "estimate"
        assert token_cost_usd(mid, 5000, 5000) == pytest.approx(5000 * 0.13 / 1e6 + 5000 * 0.40 / 1e6)


def test_medpsy_4b_estimated_from_gemma_e2b():
    # No Bedrock 4B twin → proxy 2x gemma-4-e2b ($0.04/$0.08 → $0.08/$0.16).
    assert price_basis("medpsy-4b") == "estimate"
    assert token_cost_usd("medpsy-4b", 1_000_000, 1_000_000) == pytest.approx(0.08 + 0.16)


@pytest.mark.parametrize(
    "model_id",
    [
        "medpsy-4b",
        "qvac/MedPsy-4B",
        "mlx-community/gemma-4-26b-a4b-it-8bit",
        "mlx-community/gemma-4-31b-it-8bit",
        "minimax-m2.7-jangtq-crack",
        "dealignai/Qwen3.5-VL-122B-A10B-4bit-MLX-CRACK",
    ],
)
def test_local_self_hosted_models_are_estimated_not_free(model_id):
    # No model is free to run: each local model resolves to a Bedrock-grounded
    # estimate (>$0), never $0.
    assert model_has_known_cost(model_id) is True
    assert price_basis(model_id) == "estimate"
    assert token_cost_usd(model_id, 9999, 9999) > 0.0


def test_bedrock_gemma_list_google_aistudio_gemma_estimated_from_twin():
    # Bedrock-served Gemma 4 carries the published AWS on-demand rate (list basis).
    assert price_basis("google.gemma-4-26b-a4b") == "list"
    assert token_cost_usd("google.gemma-4-26b-a4b", 1_000_000, 1_000_000) == pytest.approx(0.53)
    # Google AI Studio Gemma (gemma-4-*-it) is the SAME model on a different host
    # with no published Google rate → grounded in the Bedrock twin as an ESTIMATE
    # (no model is free), at the twin's rate. Overridable by adding a real Google
    # list price to MODEL_PRICES_PER_M_TOKENS.
    assert price_basis("gemma-4-26b-a4b-it") == "estimate"
    assert price_basis("gemma-4-31b-it") == "estimate"
    assert token_cost_usd("gemma-4-26b-a4b-it", 1_000_000, 1_000_000) == pytest.approx(0.53)
    assert token_cost_usd("gemma-4-31b-it", 1_000_000, 1_000_000) == pytest.approx(0.14 + 0.40)


def test_unknown_sentinel_was_removed_from_zero_cost():
    # The pre-existing fabricated-zero "unknown" id must NOT resolve to $0 — it is
    # exactly the genuinely-unverified case and must degrade to "unavailable".
    assert model_has_known_cost("unknown") is False


def test_null_negative_tokens_clamp_to_zero():
    assert token_cost_usd("anthropic.claude-sonnet-4-6", -1, None) == 0.0


def test_every_bedrock_model_has_known_cost():
    """Cost-coverage guard: every wired Bedrock model must resolve to a known cost
    (priced or zero), so wiring a model can never silently skip cost tracking — the
    registry->price-table link the abstraction was missing. If this fails, add the
    model's per-1M Bedrock price to MODEL_PRICES_PER_M_TOKENS in corpus/cost.py."""
    from indra_belief.model_client import LOCAL_MODELS

    missing = sorted(
        name
        for name, cfg in LOCAL_MODELS.items()
        if name.startswith("bedrock-") and not model_has_known_cost(cfg.get("model_id", ""))
    )
    assert not missing, f"bedrock models missing a price (add to corpus/cost.py): {missing}"
