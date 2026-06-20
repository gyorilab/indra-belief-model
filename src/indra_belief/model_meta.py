"""Ground-truth model metadata — parameter counts, keyed by the run's `model`.

The single source of truth for **model size**, the static capability axis that
sits beside cost. Like the cost price table, this is curated, sourced, and baked
into each run's export (`export_meta.model_meta`) so it TRAVELS with the run —
the viewer holds no model-size table of its own.

Two honesty rules, mirroring the cost contract:
  - Closed-weight models (gpt-5.5, claude-*) have UNDISCLOSED parameter counts.
    Their size is `status="unknown"`, never a guessed number — the viewer drops
    them to an "unknown size" rail rather than plotting a fabricated point.
  - MoE models carry BOTH `total_b` (headline size, the plotted axis) and
    `active_b` (params per forward pass); dense models have `active_b=None`.

Sizes are best-effort from public model cards; `source` records provenance and
flags estimates. Correct here — this file is the ground truth.
"""

from __future__ import annotations

# Normalized key -> (total_params_B, active_params_B | None, source).
# OPEN-weight models only; size is published ground truth.
MODEL_PARAMS: dict[str, tuple[float, float | None, str]] = {
    # Google Gemma 4 family (MoE where an "aNb"/"eNb" suffix names active/effective).
    "gemma": (26.0, 4.0, "Gemma-4-26B-a4b — 26B total, 4B active (MoE)"),
    "gemma-4-26b": (26.0, 4.0, "Gemma-4-26B-a4b — 26B total, 4B active (MoE)"),
    "gemma-4-31b": (31.0, None, "Gemma-4-31B — dense"),
    "gemma-4-e2b": (5.0, 2.0, "Gemma-4-E2B — ~5B raw, 2B effective (selective activation)"),
    # MedPsy — 4B clinical fine-tune.
    "medpsy": (4.0, None, "MedPsy-4B — dense fine-tune"),
    "medpsy-4b": (4.0, None, "MedPsy-4B — dense fine-tune"),
    # Large open reasoners (MoE: total / active).
    "deepseek-v3.2": (671.0, 37.0, "DeepSeek-V3 class — 671B total, 37B active"),
    "kimi-k2.5": (1000.0, 32.0, "Kimi K2 class — ~1T total, 32B active"),
    "glm-5": (355.0, 32.0, "GLM-4.5 family — 355B/32B active (glm-5 exact undisclosed; ESTIMATE)"),
    "minimax-m2.5": (230.0, 10.0, "MiniMax-M2 family — ~230B/10B active (ESTIMATE)"),
    "qwen3-235b-a22b": (235.0, 22.0, "Qwen3-235B-A22B — 235B total, 22B active"),
    "qwen3-coder-480b-a35b": (480.0, 35.0, "Qwen3-Coder-480B-A35B — 480B total, 35B active"),
    "nemotron-nano-3-30b": (30.0, None, "Nemotron Nano 3 — ~30B"),
    "nemotron-super-3-120b": (120.0, None, "Nemotron Super 3 — ~120B"),
    "gpt-oss-20b": (21.0, 3.6, "gpt-oss-20B — 21B total, 3.6B active (MoE)"),
    "gpt-oss-120b": (117.0, 5.1, "gpt-oss-120B — 117B total, 5.1B active (MoE)"),
    # Local-only deployments (no recorded run yet); sizes PROVISIONAL — confirm
    # the local build's version/size before relying on these.
    "minimax-m2.7": (230.0, 10.0, "MiniMax-M2.7 family — ~230B/10B active (ESTIMATE; distinct from m2.5)"),
    "qwen3.5-vl-122b-a10b": (122.0, 10.0, "Qwen3.5-VL-122B-A10B — 122B total, 10B active (MoE, vision-language)"),
}

# Param keys whose size is an ESTIMATE/inference, not a confirmed published spec
# for that exact model — exact count undisclosed (glm-5), inferred from the base
# version (deepseek-v3.2←V3, kimi-k2.5←K2), a family approximation (minimax), or
# provisional (no recorded run). Surfaced so the plot can render these as hollow
# dots (estimated x) the same way estimated COST renders hollow — one rule, both
# axes. Published-exact sizes (gemma, qwen3, nemotron, gpt-oss, medpsy) are NOT here.
ESTIMATED_SIZE_KEYS: frozenset[str] = frozenset({
    "deepseek-v3.2",
    "kimi-k2.5",
    "glm-5",
    "minimax-m2.5",
    "minimax-m2.7",
    "qwen3.5-vl-122b-a10b",
})

# Models KNOWN to be closed-weight — size is undisclosed, recorded as unknown
# (NOT guessed). Distinct from an unrecognized model (also unknown, but is_open
# is undetermined rather than known-false).
CLOSED_MODELS: frozenset[str] = frozenset({
    "gpt-5.5",
    "claude-opus",
    "claude-haiku",
    "claude-sonnet",
    "claude-opus-4-8",
    "claude-haiku-4-5",
})


def _normalize(model: str) -> str:
    """Reduce a run's canonical `model` field to a param-table key.

    Strips the host PREFIX (bedrock-/remote-/local-/google-) that names where the
    model ran but not which model it is: `bedrock-gemma-4-26b` -> `gemma-4-26b`,
    `remote-medpsy-4b` -> `medpsy-4b`. Also handles a raw `qvac/MedPsy-4B` id.
    """
    m = (model or "").strip().lower()
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    for pre in ("bedrock-", "remote-", "local-", "google-"):
        if m.startswith(pre):
            m = m[len(pre):]
            break
    return m


def model_size(model: str) -> dict:
    """Ground-truth size block for a model. Always returns a dict with `status`.

    `status="known"` -> total_b/active_b populated, is_open=True.
    `status="unknown"` -> total_b/active_b None; is_open is False for a model we
    KNOW is closed, None for an unrecognized model.
    """
    key = _normalize(model)
    entry = MODEL_PARAMS.get(key)
    if entry is not None:
        total_b, active_b, source = entry
        return {
            "status": "known",
            "total_b": total_b,
            "active_b": active_b,
            "is_open": True,
            # estimated: the size is inferred/undisclosed for this exact model
            # (vs a confirmed published spec) — the size-axis analogue of an
            # estimated cost. Drives the hollow-dot rendering.
            "estimated": key in ESTIMATED_SIZE_KEYS,
            "source": source,
        }
    if key in CLOSED_MODELS:
        return {
            "status": "unknown",
            "total_b": None,
            "active_b": None,
            "is_open": False,
            "estimated": False,
            "source": "closed-weight; parameter count undisclosed",
        }
    return {
        "status": "unknown",
        "total_b": None,
        "active_b": None,
        "is_open": None,
        "estimated": False,
        "source": "unrecognized model; size not in registry",
    }
