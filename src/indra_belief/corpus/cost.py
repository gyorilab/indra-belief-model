"""LLM-call cost estimation — Phase 0.1 made into a helper.

`estimate_cost(stmts, model_id)` projects how many LLM calls + tokens +
USD a `score_corpus` run will consume. The auditor's natural pre-run
"what will this cost?" check before clicking Go.

Empirical anchor (per memory: substrate-vs-LLM lever):
  - S-phase substrate-resolves only ~1.2% of records to zero LLM calls
    (target was 50%). 68.5% use all 4 LLM probes per evidence.
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


# Cost per million tokens (USD), as of 2026-05-09 published rates.
# Public list pricing — adjust for your contracted rates.
#
# NOTE: viewer/src/routes/+page.svelte mirrors this table client-side for
# the dashboard cost panel. When rates change, update both — there is no
# build-time check that they match.
def _load_pricing_table() -> tuple[dict[str, tuple[float, float]], set[str]]:
    """Single source of truth for LLM input/output pricing.

    Reads `viewer/src/lib/modelPrices.json` (the same file the SvelteKit
    viewer imports), so a rate change lands in Python, the DuckDB tombstone
    CASE expression, and the UI display from one edit. Falls back to a
    minimal hard-coded table if the file is unreadable so unit tests can
    still run without the viewer tree on disk.
    """
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.normpath(os.path.join(here, "..", "..", "..", "viewer", "src", "lib", "modelPrices.json")),
    ]
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_prices = data.get("prices_per_million_tokens", {}) or {}
            prices: dict[str, tuple[float, float]] = {}
            for model_id, pair in raw_prices.items():
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) == 2
                    and all(isinstance(v, (int, float)) for v in pair)
                ):
                    prices[str(model_id)] = (float(pair[0]), float(pair[1]))
            zero_raw = data.get("zero_cost_model_ids", []) or []
            zero = {str(m) for m in zero_raw if isinstance(m, str)}
            if prices:
                return prices, zero
        except (OSError, json.JSONDecodeError) as e:
            log.warning(
                "could not read pricing table from %s: %s; using fallback", path, e
            )
    # Hard-coded fallback so unit tests in isolated environments still work.
    return (
        {
            "claude-haiku-4-5": (0.80, 4.00),
            "claude-sonnet-4-6": (3.00, 15.00),
            "claude-opus-4-7": (15.00, 75.00),
            "gemini-2.5-flash": (0.075, 0.30),
            "gemini-2.5-pro": (1.25, 5.00),
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
        },
        {"mock", "mock-model", "smoke-local", "unknown"},
    )


MODEL_PRICES_PER_M_TOKENS, ZERO_COST_MODEL_IDS = _load_pricing_table()
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
    return model_id in MODEL_PRICES_PER_M_TOKENS or model_id in ZERO_COST_MODEL_IDS


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
    """Compute observed USD from token counts and the local price table."""
    if model_id in ZERO_COST_MODEL_IDS:
        return 0.0
    prices = MODEL_PRICES_PER_M_TOKENS.get(model_id)
    if prices is None:
        if on_unknown == "raise":
            raise ValueError(
                f"model_id {model_id!r} is missing from MODEL_PRICES_PER_M_TOKENS; "
                "cannot enforce observed-spend cap"
            )
        log.warning(
            "model_id %r unknown to MODEL_PRICES_PER_M_TOKENS; observed cost recorded as 0",
            model_id,
        )
        return 0.0
    in_price, out_price = prices
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
        prices = MODEL_PRICES_PER_M_TOKENS.get(model_id)
        if prices is None:
            log.warning(
                "model_id %r unknown to MODEL_PRICES_PER_M_TOKENS; "
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
