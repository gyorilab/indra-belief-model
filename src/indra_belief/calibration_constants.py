"""Per-reader soft-survival-weight constants (calibration C2 / E5).

Fitted on eval_curation_v1 (n=1606, balanced), verified in
``research/calibration_task_hypergraph.md``:

    w_correct   = rand_corr        = P(read wrong | verdict=correct)
    w_incorrect = 1 - rand_rej     = P(read wrong | verdict=incorrect)
    variant     = guard            (confirmation can only lower a read's error,
                                    rejection only raise it — the C1 form that
                                    beat the hard gate without flattening)

A source's repeated reads are correlated (same reader), so they don't compound —
a source contributes one aggregate per-read wrong-rate (the geometric mean of its
reads' ``w``). This was confirmed on the held-out ``holdout_cc`` and is baked into
the model directly; there is NO correlation-exponent parameter.

These are the ONLY hardcoded calibration numbers; they are resolved per run by
the reader's model name and baked into that run's export (they travel with the
run — no global belief-math hardcode). A reader with no fitted calibration
resolves to None, and the caller stays on the hard gate.
"""
from __future__ import annotations

# reader family -> fitted soft-weight pair. rand_corr / rand_rej anchors:
#   gemma-26B : rand_corr 0.183, rand_rej 0.131
#   medpsy-4B : rand_corr 0.243, rand_rej 0.127
_CALIBRATION: dict[str, dict] = {
    "gemma": {"w_correct": 0.183, "w_incorrect": 1 - 0.131, "variant": "guard"},
    "medpsy": {"w_correct": 0.243, "w_incorrect": 1 - 0.127, "variant": "guard"},
}


def calibration_for(model: str | None) -> dict | None:
    """Resolve the soft-weight pair for a run's reader model name.

    The gemma fit covers **gemma-4-26B** (served as ``gemma-remote`` /
    ``gemma-moe`` / ``gemma-google-moe`` — same weights, which inherit the fit).
    The **gemma-4-31B** variants (``gemma-31b`` / ``gemma-google-31b``) are a
    DIFFERENT model and must NOT inherit the 26B fit; ``bedrock-*`` serving is
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
