"""Per-reader soft-survival-weight constants (calibration C2 / E5).

Two numbers per reader, each a plain conditional wrong-rate (intuitive by design):

    w_correct   = P(read wrong | verdict=correct)    # confirmed-but-wrong rate
    w_incorrect = P(read wrong | verdict=incorrect)   # rejected-and-indeed-wrong rate
    variant     = clean

Fitted on eval_curation_v1 (n=1606, balanced); see
``research/calibration_task_hypergraph.md``. ``w_incorrect`` is stored as
``1 - rand_rej`` only because the fit reports ``rand_rej = P(correct|incorrect)``;
the value IS the per-read wrong-rate.

The ``clean`` variant (adopted, ``noise_model._soft_gated_belief``) uses these
directly as per-read wrong-rates in INDRA's noisy-OR: a source's factor is the
geometric mean of its reads' rates (no additive ``syst``, no clamp). This is
self-calibrating at n=1 — a single confirmed gemma read gives belief
``1 - 0.183 = 0.817`` = the measured P(correct | confirmed gemma read). The older
``guard`` form added ``syst`` on top, leaving belief ~``syst`` under-confident
(0.767); ``clean`` removes that double-count and de-conflates syst/rand.

A source's repeated reads are correlated (same reader), so they don't compound —
the geometric mean is the source's single aggregate per-read rate (the κ=0
finding, confirmed on the held-out ``holdout_cc``); there is NO correlation
exponent. Because ``clean`` shifts the belief scale, the error-detection
threshold must be derived per reader (see ``calibration_ship_gate``), never
hardcoded to 0.5.

These are the ONLY hardcoded calibration numbers; they are resolved per run by
the reader's model name and baked into that run's export (they travel with the
run — no global belief-math hardcode). A reader with no fitted calibration
resolves to None, and the caller stays on the hard gate.
"""
from __future__ import annotations

# reader family -> fitted soft-weight pair. rand_corr / rand_rej anchors:
#   gemma-26B : w_correct (rand_corr) 0.183, rand_rej 0.131
#   medpsy-4B : w_correct (rand_corr) 0.243, rand_rej 0.127
_CALIBRATION: dict[str, dict] = {
    "gemma": {"w_correct": 0.183, "w_incorrect": 1 - 0.131, "variant": "clean"},
    "medpsy": {"w_correct": 0.243, "w_incorrect": 1 - 0.127, "variant": "clean"},
}


def calibration_for(model: str | None) -> dict | None:
    """Resolve the soft-weight pair for a run's reader model name.

    The gemma fit covers **gemma-4-26B** (served as ``remote-gemma-4-26b`` /
    ``local-gemma-4-26b`` / ``google-gemma-4-26b`` — same weights, which inherit
    the fit). The **gemma-4-31B** variants (``local-gemma-4-31b`` /
    ``google-gemma-4-31b``) are a DIFFERENT model and must NOT inherit the 26B
    fit; ``bedrock-*`` serving is
    uncertain/unprovisioned — all return None (→ hard gate) until separately fit
    (open question Q3). ``medpsy-4B`` is the only medpsy model. None ⇒ hard gate.
    """
    if not model:
        return None
    m = model.strip().lower()
    if "medpsy" in m:
        return dict(_CALIBRATION["medpsy"])
    # 31B is a different model; bedrock serving is uncertain — neither inherits.
    if "gemma" in m and "31b" not in m and "bedrock" not in m:
        return dict(_CALIBRATION["gemma"])
    return None
