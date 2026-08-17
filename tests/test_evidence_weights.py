"""The per-evidence weight aggregation reduces EXACTLY to the frozen model.

`indra_belief.evidence_weights` re-states an aggregation that already exists
inside `noise_model._soft_gated_belief`. Duplicated arithmetic is how two
implementations drift apart while both look right, and `noise_model` is
byte-frozen so the duplication cannot be avoided by refactoring.

This file is the thing that makes the duplication safe: fed VERDICT-DERIVED
weights, the generalization must reproduce `compute_gated_belief(...,
soft_weights=True)` to the bit. Not approximately — the same float. If that
holds, the new function is the old formula with the weight lifted into an
argument, and any difference in a future measurement is attributable to the
WEIGHTS rather than to a second, subtly different belief model.

The configurations below deliberately cover what the frozen branch distinguishes:
confirmations where the reader LR wins, confirmations where the SOURCE FLOOR
wins (signor's reliability exceeds the reader's confirm LR), rejections,
unscored reads, repeated reads within one source (averaged), and several sources
(summed).
"""
from __future__ import annotations

import pytest

from indra_belief.calibration_constants import (
    REASONING_FIRST_PROMPT_SHA256,
    calibration_for,
)
from indra_belief.evidence_weights import (
    belief_from_weights,
    probe_weight,
    source_logodds_for,
    verdict_weight,
)
from indra_belief.noise_model import RECALIBRATED_PRIORS, compute_gated_belief

_P = calibration_for("bedrock-gemma-4-26b", prompt_sha256=REASONING_FIRST_PROMPT_SHA256)
LC, LR, PRIOR = _P["log_lr_confirm"], _P["log_lr_reject"], _P["prior_logodds"]

CONFIGURATIONS = {
    "single confirm, reader LR wins": [
        {"source_api": "reach", "verdict": "correct", "included": True},
    ],
    "single confirm, SOURCE FLOOR wins": [
        # signor's reliability logit exceeds the reader's confirm LR
        {"source_api": "signor", "verdict": "correct", "included": True},
    ],
    "single reject": [
        {"source_api": "reach", "verdict": "incorrect", "included": False},
    ],
    "unscored read": [
        {"source_api": "reach", "verdict": None, "included": True},
    ],
    "repeats within one source (averaged)": [
        {"source_api": "reach", "verdict": "correct", "included": True},
        {"source_api": "reach", "verdict": "correct", "included": True},
        {"source_api": "reach", "verdict": "incorrect", "included": False},
    ],
    "several sources (summed)": [
        {"source_api": "reach", "verdict": "correct", "included": True},
        {"source_api": "sparser", "verdict": "incorrect", "included": False},
        {"source_api": "signor", "verdict": "correct", "included": True},
    ],
    "mixed, with an unscored row": [
        {"source_api": "reach", "verdict": "correct", "included": True},
        {"source_api": "trips", "verdict": None, "included": True},
        {"source_api": "medscan", "verdict": "incorrect", "included": False},
    ],
}


def _as_weights(evidence):
    """Re-express verdict-driven evidence as explicit per-row weights."""
    out = []
    for ev in evidence:
        s_logodds = source_logodds_for(ev["source_api"], RECALIBRATED_PRIORS)
        out.append({
            "source_api": ev["source_api"],
            "weight_of_evidence": verdict_weight(ev["verdict"], s_logodds,
                                  log_lr_confirm=LC, log_lr_reject=LR),
        })
    return out


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_verdict_derived_weights_reproduce_the_frozen_model_exactly(name):
    evidence = CONFIGURATIONS[name]
    frozen = compute_gated_belief(
        evidence, RECALIBRATED_PRIORS, soft_weights=True,
        log_lr_confirm=LC, log_lr_reject=LR, prior_logodds=PRIOR,
    ).belief
    lifted = belief_from_weights(
        _as_weights(evidence), RECALIBRATED_PRIORS, prior_logodds=PRIOR
    )
    assert lifted == frozen, (
        f"{name}: the lifted aggregation disagrees with the frozen model "
        f"({lifted!r} vs {frozen!r}); it is not the same formula"
    )


def test_the_source_floor_case_is_actually_exercised():
    """Guards the test above from passing vacuously.

    If no configuration ever hit the floor branch, the reduction test would not
    cover the one place the frozen model prefers the source over the reader.
    """
    s_logodds = source_logodds_for("signor", RECALIBRATED_PRIORS)
    assert s_logodds > LC, (
        "signor's reliability no longer exceeds the reader confirm LR, so the "
        "floor branch is untested — pick a source whose prior still does"
    )
    assert verdict_weight("correct", s_logodds,
                          log_lr_confirm=LC, log_lr_reject=LR) == s_logodds


def test_a_missing_weight_is_refused_not_treated_as_zero():
    """A zero weight is a measurement: 'this read is exactly uninformative'."""
    with pytest.raises(ValueError, match="no numeric 'weight_of_evidence'"):
        belief_from_weights([{"source_api": "reach"}], RECALIBRATED_PRIORS)
    with pytest.raises(ValueError, match="no numeric 'weight_of_evidence'"):
        belief_from_weights([{"source_api": "reach", "weight_of_evidence": None}], RECALIBRATED_PRIORS)


def test_probe_weight_preserves_the_floor_only_for_supporting_readings():
    s = source_logodds_for("signor", RECALIBRATED_PRIORS)
    # a weakly supporting probe reading must not drag a curated source down
    assert probe_weight(0.2, s) == s
    # a disconfirming reading stands on its own, exactly as a rejection does
    assert probe_weight(-1.3, s) == -1.3


def test_continuous_weights_move_the_belief():
    """The point of the module: magnitude has to matter.

    Two readings that a verdict would collapse to the same value must not
    produce the same belief once the margin is carried.
    """
    s = source_logodds_for("reach", RECALIBRATED_PRIORS)
    weak = belief_from_weights(
        [{"source_api": "reach", "weight_of_evidence": probe_weight(0.3, s)}],
        RECALIBRATED_PRIORS, prior_logodds=PRIOR)
    strong = belief_from_weights(
        [{"source_api": "reach", "weight_of_evidence": probe_weight(3.0, s)}],
        RECALIBRATED_PRIORS, prior_logodds=PRIOR)
    assert strong > weak
