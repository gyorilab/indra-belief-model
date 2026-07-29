"""LLM-call cost estimation helper.

`estimate_cost(stmts, model_id)` projects how many LLM calls + tokens +
USD a `score_corpus` run will consume. The auditor's natural pre-run
"what will this cost?" check before clicking Go.

Empirical anchor:
  - The deterministic substrate resolves only ~1.2% of records to zero
    LLM calls; ~68.5% use all four LLM probes per evidence.
  - Plus ~1 LLM call per evidence for grounding verification.
  - Avg ~400 tokens per LLM call (~330 in + ~70 out → 5:1 ratio).

Defaults bake the conservative assumption (substrate ≤2%, ~5 LLM calls
per evidence). Override per project.
"""

from __future__ import annotations

import logging
from typing import Iterable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from indra.statements import Statement

log = logging.getLogger(__name__)


# Cost per million tokens (USD), public list pricing.
# Verify pricing is current before deployment; adjust for your contracted rates.
# This hard-coded table is the single source of truth — the viewer does not
# currently compute or display cost (no client-side mirror).
def _load_pricing_table() -> tuple[dict[str, tuple[float, float]], set[str]]:
    """Return the code-authenticated USD-per-million token price table.

    Paid reservations must not depend on an optional repository file read at
    module import: that would let a create/import/remove race change prices
    while every authenticated Python pathname still matched.  This frozen
    table is the sole paid-run source of truth.
    """
    return (
        {
            # Anthropic-API-spelled ids (estimate_cost defaults / direct calls).
            "claude-haiku-4-5": (0.80, 4.00),
            "claude-sonnet-4-6": (3.00, 15.00),
            "claude-opus-4-7": (15.00, 75.00),
            # Bedrock-PREFIXED ids — the model_id the call_log actually emits for
            # Bedrock Claude. Same Anthropic list rates as the non-prefixed keys.
            "anthropic.claude-sonnet-4-6": (3.00, 15.00),
            "anthropic.claude-haiku-4-5": (0.80, 4.00),
            # Bedrock-served Google Gemma 4 — AWS on-demand list rates, us-east-1
            # (https://aws.amazon.com/bedrock/pricing, 2026-06).
            "google.gemma-4-26b-a4b": (0.13, 0.40),
            "google.gemma-4-31b": (0.14, 0.40),
            "google.gemma-4-e2b": (0.04, 0.08),
            # ── Bedrock mantle open-weight reasoners (AWS on-demand, per 1M tokens,
            # us-east-1, fetched 2026-06-20 from aws.amazon.com/bedrock/pricing
            # unless flagged). Keyed on the exact call_log model_id. ──
            "deepseek.v3.2": (0.62, 1.85),
            "moonshotai.kimi-k2.5": (0.60, 3.00),
            "zai.glm-5": (1.00, 3.20),
            "minimax.minimax-m2.5": (0.30, 1.20),
            "qwen.qwen3-235b-a22b-2507": (0.2266, 0.9064),
            "qwen.qwen3-coder-480b-a35b-instruct": (0.22, 1.80),  # 2nd-party (Bifrost/pricepertoken); AWS page didn't surface a per-1M figure
            "nvidia.nemotron-nano-3-30b": (0.06, 0.24),
            "nvidia.nemotron-super-3-120b": (0.15, 0.65),
            "openai.gpt-oss-20b": (0.07, 0.30),    # 2nd-party (Bedrock cost directories; AWS page surfaced only Sydney)
            "openai.gpt-oss-120b": (0.15, 0.60),   # 2nd-party (Bedrock cost directories)
            "openai.gpt-5.5": (5.50, 33.00),       # Bedrock-specific, ~10% over OpenAI-direct $5/$30
            # Bedrock Claude — the 'us.' REGIONAL inference-profile ids we actually
            # call: base $1/$5 (Haiku 4.5) & $5/$25 (Opus 4.8) + the 10% regional-
            # endpoint premium Anthropic documents = the effective rate below.
            "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.10, 5.50),
            "us.anthropic.claude-opus-4-8": (5.50, 27.50),
            "gemini-2.5-flash": (0.075, 0.30),
            "gemini-2.5-pro": (1.25, 5.00),
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
        },
        {
            # Test sentinels ONLY — genuinely $0 because they are not real models.
            # Real local / self-hosted models are NO LONGER treated as free: each
            # carries a Bedrock-GROUNDED estimate (ESTIMATED_PRICE_REFS below).
            # "No model is free to run." "unknown" stays OUT (it degrades to
            # unavailable, never a fabricated $0).
            "mock", "mock-model", "smoke-local",
        },
    )


MODEL_PRICES_PER_M_TOKENS, ZERO_COST_MODEL_IDS = _load_pricing_table()

# ── Estimated prices for self-hosted / local models ─────────────────────────
# No model is free to run. A local model's per-token cost is GROUNDED in a
# provider list price (Bedrock first) for the closest comparable model, scaled by
# an active-parameter factor where there is no exact twin. Expressed as a
# REFERENCE (not a frozen number) so the estimate tracks the reference's list
# price. Keyed on the REAL call_log model_id; basis surfaces as 'estimate' so the
# UI can mark it (~) distinctly from observed ('list') spend.
#
# EXTENSIBLE TO ANY PROVIDER: to price a new provider's served model, add its list
# rate to MODEL_PRICES_PER_M_TOKENS (keyed by whatever model_id its call_log
# emits). To price a self-hosted model, add a (reference_model_id, factor, note)
# row here — the cost then follows that reference's list price automatically.
ESTIMATED_PRICE_REFS: dict[str, tuple[str, float, str]] = {
    # Exact Bedrock twins (the same model is also served on Bedrock) — factor 1.0.
    "gemma-4-26b": ("google.gemma-4-26b-a4b", 1.0, "Bedrock gemma-4-26b-a4b list (exact twin)"),
    "gemma-4-26b-ollama": ("google.gemma-4-26b-a4b", 1.0, "Bedrock gemma-4-26b-a4b list (exact twin)"),
    "mlx-community/gemma-4-26b-a4b-it-8bit": ("google.gemma-4-26b-a4b", 1.0, "Bedrock gemma-4-26b-a4b list (same model, local MLX)"),
    "mlx-community/gemma-4-31b-it-8bit": ("google.gemma-4-31b", 1.0, "Bedrock gemma-4-31b list (same model, local MLX)"),
    # Google AI Studio (Gemini API) Gemma — the SAME gemma-4 model on a different
    # host with no published Google rate; grounded in the Bedrock twin. If a real
    # Google list price is verified, add it to MODEL_PRICES_PER_M_TOKENS above
    # (keyed on these ids) — that takes precedence over the estimate.
    "gemma-4-26b-a4b-it": ("google.gemma-4-26b-a4b", 1.0, "Google AI Studio Gemma; ~ Bedrock gemma-4-26b-a4b list (same model)"),
    "gemma-4-31b-it": ("google.gemma-4-31b", 1.0, "Google AI Studio Gemma; ~ Bedrock gemma-4-31b list (same model)"),
    "minimax-m2.7-jangtq-crack": ("minimax.minimax-m2.5", 1.0, "Bedrock minimax-m2.5 list (closest MiniMax)"),
    "dealignai/Qwen3.5-VL-122B-A10B-4bit-MLX-CRACK": ("qwen.qwen3-235b-a22b-2507", 1.0, "Bedrock Qwen3 MoE list (closest Qwen)"),
    # No Bedrock 4B twin — proxy to gemma-4-e2b (2B-effective) scaled 2x by the
    # active-param ratio (medpsy-4b is 4B dense). Tune the factor to taste.
    "medpsy-4b": ("google.gemma-4-e2b", 2.0, "~2x Bedrock gemma-4-e2b (4B dense vs 2B-effective)"),
    "qvac/MedPsy-4B": ("google.gemma-4-e2b", 2.0, "~2x Bedrock gemma-4-e2b (4B dense vs 2B-effective)"),
}


def price_for(model_id: str) -> tuple[float, float, str] | None:
    """Resolve ``(in_per_M, out_per_M, basis)`` for a model, or ``None`` if it
    cannot be priced. The single resolver all cost math goes through.

    basis: ``'list'`` (a provider's observed list rate) | ``'estimate'`` (grounded
    in a reference list price) | ``'test'`` (a $0 test sentinel).
    """
    p = MODEL_PRICES_PER_M_TOKENS.get(model_id)
    if p is not None:
        return (p[0], p[1], "list")
    ref = ESTIMATED_PRICE_REFS.get(model_id)
    if ref is not None:
        ref_id, factor, _note = ref
        rp = MODEL_PRICES_PER_M_TOKENS.get(ref_id)
        if rp is not None:
            return (rp[0] * factor, rp[1] * factor, "estimate")
    if model_id in ZERO_COST_MODEL_IDS:
        return (0.0, 0.0, "test")
    return None


def price_basis(model_id: str) -> str | None:
    """``'list'`` | ``'estimate'`` | ``'test'`` | ``None`` — how a price is known."""
    p = price_for(model_id)
    return p[2] if p is not None else None


PROBE_STEP_KINDS = frozenset({
    "subject_role_probe",
    "object_role_probe",
    "relation_axis_probe",
    "scope_probe",
})


def _normalize_probe_step_filter(
    probe_step_filter: Iterable[str] | None,
) -> tuple[str, ...]:
    if probe_step_filter is None:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in probe_step_filter:
        step_kind = str(raw).strip()
        if not step_kind:
            continue
        if step_kind not in PROBE_STEP_KINDS:
            raise ValueError(
                "probe_step_filter only accepts decomposed probe step kinds; "
                f"got {step_kind!r}"
            )
        if step_kind not in seen:
            seen.add(step_kind)
            out.append(step_kind)
    return tuple(out)


def model_has_known_cost(model_id: str) -> bool:
    return price_for(model_id) is not None


def _nonnegative_tokens(value: int | float | None) -> int | float:
    if value is None:
        return 0
    return value if value > 0 else 0


def token_cost_usd(
    model_id: str,
    prompt_tokens: int | float | None,
    out_tokens: int | float | None,
    *,
    on_unknown: Literal["zero", "raise"] = "zero",
) -> float:
    """Compute USD from token counts and the resolved price (list or estimate)."""
    prices = price_for(model_id)
    if prices is None:
        if on_unknown == "raise":
            raise ValueError(
                f"model_id {model_id!r} has no price (list or estimate); "
                "cannot enforce observed-spend cap"
            )
        log.warning(
            "model_id %r has no list price or estimate; cost recorded as 0",
            model_id,
        )
        return 0.0
    in_price, out_price, _basis = prices
    return (
        _nonnegative_tokens(prompt_tokens) * in_price / 1_000_000
        + _nonnegative_tokens(out_tokens) * out_price / 1_000_000
    )


def estimate_cost(
    stmts: Iterable["Statement"],
    *,
    model_id: str = "claude-sonnet-4-6",
    architecture: str = "decomposed",
    probe_step_filter: Iterable[str] | None = None,
    probe_only: bool = False,
    avg_evidences_per_stmt: float | None = None,
    avg_llm_calls_per_evidence: float | None = None,
    avg_input_tokens_per_call: int = 330,
    avg_output_tokens_per_call: int = 70,
    in_price_per_m: float | None = None,
    out_price_per_m: float | None = None,
) -> dict:
    """Project LLM-call counts + token volume + USD for a `score_corpus` run.

    Args:
        stmts: list/iterable of INDRA Statements (consumed once for counts).
        model_id: looked up in `MODEL_PRICES_PER_M_TOKENS` unless overridden.
        avg_evidences_per_stmt: if None, computed from the actual stmts.
        architecture: scoring architecture. `decomposed` defaults to
            about 5 LLM calls/evidence; `monolithic` defaults to 1.
        probe_step_filter: selected decomposed probe rows for probe-only
            repair runs.
        probe_only: estimate only the selected decomposed probes, excluding
            grounding and aggregate adjudication.
        avg_llm_calls_per_evidence: if None, chosen from architecture.
        avg_input_tokens_per_call / avg_output_tokens_per_call: typical
            decomposed-probe call shape.
        in_price_per_m / out_price_per_m: override model's rate (e.g. for
            negotiated rates or unlisted models).

    Returns:
        dict with `n_stmts`, `n_evidences_est`, `n_llm_calls_est`,
        `input_tokens_est`, `output_tokens_est`, `cost_usd`,
        `model_id`, `assumptions`.
    """
    stmts = list(stmts)
    n_stmts = len(stmts)

    if architecture not in {"decomposed", "monolithic"}:
        raise ValueError(
            "architecture must be 'decomposed' or 'monolithic', "
            f"got {architecture!r}"
        )
    normalized_probe_filter = _normalize_probe_step_filter(probe_step_filter)
    if probe_only:
        if architecture != "decomposed":
            raise ValueError("probe_only estimates are only valid for decomposed runs")
        if not normalized_probe_filter:
            raise ValueError("probe_only estimates require probe_step_filter")
    if avg_llm_calls_per_evidence is None:
        if probe_only:
            avg_llm_calls_per_evidence = float(len(normalized_probe_filter))
        else:
            avg_llm_calls_per_evidence = 1.0 if architecture == "monolithic" else 5.0

    if avg_evidences_per_stmt is None:
        total_evidences = sum(len(getattr(s, "evidence", []) or []) for s in stmts)
        avg_evidences_per_stmt = (total_evidences / n_stmts) if n_stmts else 0.0
        n_evidences = total_evidences
    else:
        n_evidences = round(n_stmts * avg_evidences_per_stmt)

    n_llm_calls = round(n_evidences * avg_llm_calls_per_evidence)
    input_tokens = n_llm_calls * avg_input_tokens_per_call
    output_tokens = n_llm_calls * avg_output_tokens_per_call

    if in_price_per_m is None or out_price_per_m is None:
        prices = price_for(model_id)
        if prices is None:
            log.warning(
                "model_id %r has no list price or estimate; "
                "pass in_price_per_m + out_price_per_m to override",
                model_id,
            )
            in_price_per_m = in_price_per_m or 0.0
            out_price_per_m = out_price_per_m or 0.0
        else:
            in_price_per_m = in_price_per_m or prices[0]
            out_price_per_m = out_price_per_m or prices[1]

    cost_usd = (
        input_tokens * (in_price_per_m / 1_000_000)
        + output_tokens * (out_price_per_m / 1_000_000)
    )

    return {
        "n_stmts": n_stmts,
        "n_evidences_est": n_evidences,
        "n_llm_calls_est": n_llm_calls,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "cost_usd": round(cost_usd, 4),
        "model_id": model_id,
        "assumptions": {
            "avg_evidences_per_stmt": round(avg_evidences_per_stmt, 2),
            "architecture": architecture,
            "scoring_mode": "probe_only" if probe_only else "aggregate",
            "probe_step_filter": list(normalized_probe_filter),
            "avg_llm_calls_per_evidence": avg_llm_calls_per_evidence,
            "avg_input_tokens_per_call": avg_input_tokens_per_call,
            "avg_output_tokens_per_call": avg_output_tokens_per_call,
            "in_price_per_m_tokens_usd": in_price_per_m,
            "out_price_per_m_tokens_usd": out_price_per_m,
        },
    }
