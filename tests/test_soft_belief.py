"""E4 — byte-identity + correctness guard for the soft survival weight.

This is the gate that lets the calibration soft weight (C2.1) coexist with the
hard gate that backs shipped F1: the default-off path MUST be byte-identical to
today, and the soft path MUST reproduce the validated `calibration_stage1.soft_belief`
form and forward through the contradiction recursion (the `noise_model.py:387` gap).
"""
import math
import sys
from pathlib import Path

import pytest

from indra_belief.noise_model import (
    RECALIBRATED_PRIORS,
    compute_gated_belief,
    compute_gated_belief_with_contradiction,
)

# gemma anchors (research/calibration_task_hypergraph.md): w_correct = rand_corr,
# w_incorrect = 1 - rand_rej.
W_CORRECT = 0.183
W_INCORRECT = 1 - 0.131

# A mixed multi-source fixture (carries both 'included' and 'verdict').
MIXED = [
    {"source_api": "reach", "verdict": "correct", "included": True},
    {"source_api": "reach", "verdict": "incorrect", "included": False},
    {"source_api": "trips", "verdict": "correct", "included": True},
    {"source_api": "signor", "verdict": None, "included": True},
]
def _soft(ev, **kw):
    return compute_gated_belief(
        ev, RECALIBRATED_PRIORS, soft_weights=True,
        w_correct=W_CORRECT, w_incorrect=W_INCORRECT, **kw,
    )


# ── the real invariant: default-off == today, exactly (D7, backs shipped F1) ──

def test_flag_off_is_byte_identical():
    a = compute_gated_belief(MIXED, RECALIBRATED_PRIORS)
    b = compute_gated_belief(MIXED, RECALIBRATED_PRIORS, soft_weights=False)
    assert a.belief == b.belief
    assert a.parametric_only == b.parametric_only
    assert [s.incorrectness_factor for s in a.per_source] == [s.incorrectness_factor for s in b.per_source]


def test_default_is_hard_gate():
    """The new keyword params must default off — calling as before is unchanged."""
    assert compute_gated_belief(MIXED, RECALIBRATED_PRIORS).belief == \
        compute_gated_belief(MIXED, RECALIBRATED_PRIORS, soft_weights=False).belief


# ── soft path reproduces the validated reference (scripts/calibration_stage1) ──

@pytest.mark.parametrize("variant", ["guard", "replace"])
def test_soft_matches_stage1_reference(variant):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import calibration_stage1 as c1  # noqa: E402

    ref = c1.soft_belief(MIXED, W_CORRECT, W_INCORRECT, variant, RECALIBRATED_PRIORS)
    got = _soft(MIXED, variant=variant).belief
    assert math.isclose(got, ref, abs_tol=1e-12)


# ── the soft path is actually live (differs from the hard gate on real verdicts) ──

def test_soft_differs_from_hard_on_mixed():
    hard = compute_gated_belief(MIXED, RECALIBRATED_PRIORS).belief
    soft = _soft(MIXED).belief
    assert abs(soft - hard) > 1e-6, "soft path produced the hard-gate value — not wired"


def test_soft_requires_weights():
    with pytest.raises(ValueError):
        compute_gated_belief(MIXED, RECALIBRATED_PRIORS, soft_weights=True)  # no w_*


def test_soft_and_hard_report_identical_counts():
    """Only belief may differ across paths — n_surviving/n_gated are the gate
    decision ('included') on BOTH paths, so a consumer reads them consistently."""
    hard = compute_gated_belief(MIXED, RECALIBRATED_PRIORS)
    soft = _soft(MIXED)
    assert soft.n_total_evidence == hard.n_total_evidence
    assert soft.n_surviving_evidence == hard.n_surviving_evidence
    assert soft.n_gated == hard.n_gated
    assert [s.n_surviving for s in soft.per_source] == [s.n_surviving for s in hard.per_source]
    assert soft.belief != hard.belief  # the one thing that SHOULD differ


# ── contradiction recursion must forward the soft kwargs (the :387 gap) ──

CONTRA = [
    {"source_api": "reach", "verdict": "correct", "regulation_type": "up", "included": True},
    {"source_api": "reach", "verdict": "incorrect", "regulation_type": "down", "included": False},
    {"source_api": "trips", "verdict": "correct", "regulation_type": "up", "included": True},
]
def test_contradiction_forwards_soft_kwargs():
    hard, _, _ = compute_gated_belief_with_contradiction(CONTRA, RECALIBRATED_PRIORS)
    soft, _, _ = compute_gated_belief_with_contradiction(
        CONTRA, RECALIBRATED_PRIORS, soft_weights=True,
        w_correct=W_CORRECT, w_incorrect=W_INCORRECT,
    )
    # If the kwargs were not forwarded at the per-direction recursion, soft == hard.
    assert abs(soft.belief - hard.belief) > 1e-6


# ── fitted-constants resolver (calibration_constants) ──────────────────────────

def test_calibration_resolver():
    from indra_belief.calibration_constants import calibration_for
    assert calibration_for("gemma-remote")["w_correct"] == 0.183
    assert calibration_for("gemma-26B")["variant"] == "guard"
    assert calibration_for("gemma-moe")["w_correct"] == 0.183          # 26B weights inherit
    assert calibration_for("gemma-google-moe")["w_correct"] == 0.183   # 26B weights inherit
    assert calibration_for("medpsy-remote")["w_correct"] == 0.243
    assert math.isclose(calibration_for("medpsy-4b")["w_incorrect"], 1 - 0.127)
    assert "kappa" not in calibration_for("medpsy-remote")  # κ removed from the model
    # gemma-4-31B is a DIFFERENT model — must NOT inherit the 26B fit
    assert calibration_for("gemma-31b") is None
    assert calibration_for("gemma-google-31b") is None
    # bedrock serving uncertain / unfitted
    assert calibration_for("bedrock-gemma") is None
    assert calibration_for(None) is None
    assert calibration_for("some-unfitted-model") is None


# ── statement_belief soft integration (end-to-end through the production roll-up) ──

def _ev(source_api, verdict, **kw):
    return {"source_api": source_api, "verdict": verdict, "confidence": "high",
            "tier": "llm_comprehension", **kw}


def test_statement_belief_soft_lifts_confirmed_single_read():
    """A gemma-confirmed single reach read: hard gives 1-(0.05+0.462)=0.488; the
    calibrated soft weight gives 1-(0.05+min(0.183,0.462))=0.767, near its true
    ~82% reliability — the single-evidence calibration fix."""
    from indra_belief.statement_belief import statement_belief
    from indra_belief.calibration_constants import calibration_for
    rows = [_ev("reach", "correct", evidence_text="a")]
    hard = statement_belief(rows, RECALIBRATED_PRIORS).belief
    soft = statement_belief(rows, RECALIBRATED_PRIORS, soft=calibration_for("gemma-remote")).belief
    assert math.isclose(hard, 0.488, abs_tol=1e-3)
    assert math.isclose(soft, 0.767, abs_tol=1e-3)
    assert soft > hard


def test_statement_belief_soft_preserves_undefined_contract():
    """all-no_text → belief is None (UNDEFINED) on BOTH hard and soft paths."""
    from indra_belief.statement_belief import statement_belief
    from indra_belief.calibration_constants import calibration_for
    rows = [_ev("signor", "correct", tier="no_text"), _ev("biogrid", "correct", tier="no_text")]
    assert statement_belief(rows, RECALIBRATED_PRIORS).belief is None
    assert statement_belief(rows, RECALIBRATED_PRIORS, soft=calibration_for("gemma-remote")).belief is None
