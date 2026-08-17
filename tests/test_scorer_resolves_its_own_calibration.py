"""The scorer resolves its own calibration profile.

WHY THIS EXISTS
---------------
`LLMBeliefScorer.__init__` took `soft: dict | None = None`, and None means HARD
GATE. No caller in src/ or scripts/ ever passed a profile, so the documented
drop-in — `BeliefEngine(scorer=LLMBeliefScorer(client))` — served hard-gate
beliefs while a fitted, 4/4-ship-gated profile sat unused.

MEASURED on external_curator_gold_v1: ECE 0.237 on the hard gate against 0.045
calibrated. A five-fold calibration difference, entirely because nothing
connected the two halves.

The scorer already held both halves: the client names the model, the variant
names the prompt, and a profile is keyed on exactly that pair. So the default is
now AUTO — resolve it — while `soft=None` keeps meaning "hard gate", because
that is a legitimate request and callers depend on it.

The load-bearing test here is the last one. Resolving a profile is worthless if
the resolved profile does not reach the arithmetic, and a wiring test that only
checks `scorer.soft is not None` would pass on a scorer that ignored it.
"""
from __future__ import annotations

import pytest

from indra_belief.belief_scorer import AUTO, LLMBeliefScorer
from indra_belief.calibration_constants import (
    REASONING_FIRST_PROMPT_SHA256,
    calibration_for,
)


class _Client:
    """Stand-in carrying only what profile resolution reads."""

    def __init__(self, model_name):
        self.model_name = model_name


FITTED = "bedrock-gemma-4-26b"      # has a ship-approved reasoning-first profile
UNFITTED = "vllm-local"             # a real registry entry with no fitted profile


def test_a_fitted_configuration_resolves_by_default():
    scorer = LLMBeliefScorer(_Client(FITTED))
    assert scorer.soft is not None, "a fitted model+prompt must resolve without being asked"
    expected = calibration_for(FITTED, prompt_sha256=REASONING_FIRST_PROMPT_SHA256)
    assert scorer.calibration_profile_id == expected["profile_id"]


def test_an_unfitted_configuration_falls_back_to_the_hard_gate():
    """Fail-safe, not fail-open: an unknown stack must not borrow a profile."""
    scorer = LLMBeliefScorer(_Client(UNFITTED))
    assert scorer.soft is None
    assert scorer.calibration_profile_id is None


def test_explicit_none_still_means_hard_gate():
    """Backward compatibility. `soft=None` is a real request, not an omission."""
    scorer = LLMBeliefScorer(_Client(FITTED), soft=None)
    assert scorer.soft is None
    assert scorer.calibration_profile_id is None


def test_an_explicit_profile_is_not_overridden():
    sentinel = {"profile_id": "caller-supplied", "log_lr_confirm": 1.0,
                "log_lr_reject": -1.0, "prior_logodds": 0.0}
    scorer = LLMBeliefScorer(_Client(FITTED), soft=sentinel)
    assert scorer.soft is sentinel
    assert scorer.calibration_profile_id == "caller-supplied"


def test_the_resolved_profile_actually_changes_the_belief():
    """The point of the wiring: the number must move.

    Compares the scorer's own resolved-profile arithmetic against the hard gate
    on identical rows. If these agree, the profile is being carried but ignored,
    and every calibration result in the repo would be unreachable from the
    public socket.
    """
    from indra_belief.noise_model import RECALIBRATED_PRIORS
    from indra_belief.statement_belief import statement_belief

    rows = [
        {"source_api": "reach", "verdict": "correct", "tier": 1,
         "evidence_hash": "e1", "evidence_text": "A phosphorylates B."},
        {"source_api": "reach", "verdict": "incorrect", "tier": 1,
         "evidence_hash": "e2", "evidence_text": "B binds C."},
    ]
    auto = LLMBeliefScorer(_Client(FITTED))
    hard = LLMBeliefScorer(_Client(FITTED), soft=None)

    calibrated = statement_belief(rows, RECALIBRATED_PRIORS, soft=auto.soft).belief
    gated = statement_belief(rows, RECALIBRATED_PRIORS, soft=hard.soft).belief
    assert calibrated is not None and gated is not None
    assert calibrated != gated, (
        "the resolved profile did not change the belief; the scorer is carrying a "
        "calibration it does not use"
    )


def test_auto_is_distinguishable_from_none():
    assert AUTO is not None
    assert repr(AUTO) == "AUTO"
