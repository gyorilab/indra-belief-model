"""The self-hosted reader reaches the calibrated belief, not the fallback.

A self-hosted run resolves the registered calibration profile rather than
falling back to the source-counting hard gate.

Before it, `calibration_for("local-gemma-4-26b", ...)` returned None for every
prompt, so a self-hosted run silently used the source-counting hard gate — the
behaviour the reading model exists to replace. There was no error and no
warning; a run produced plausible numbers from the wrong formula.

The registered profile clears the four-leg G2 gate for lower ECE, non-inferior
AUROC, non-inferior err-F1, and default-off/hard-path byte identity; measured
2026-08-13:

    ECE    0.231 -> 0.052        AUROC  0.793 -> 0.808
    err-F1 0.796 -> 0.800        delta +0.004, CI [-0.009, +0.019] non-inferior
    default-off and explicit hard paths byte-identical  tests/test_soft_belief.py, 17 passed

Every assertion below fails if the profile is removed from `_FITTED_CONFIGS`, is
flipped to disabled, or is keyed on a different prompt. A test that passed either
way would prove nothing, so `test_the_calibrated_belief_differs_from_the_fallback`
pins the actual numeric gap rather than the mere presence of a profile.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.calibration_constants import (  # noqa: E402
    BASELINE_PROMPT_SHA256,
    REASONING_FIRST_PROMPT_SHA256,
    _CONFUSION,
    calibration_for,
    calibration_for_run,
    fitted_calibration_for,
    profile_from_confusion,
    reader_configuration_for_run,
)
from indra_belief.noise_model import RECALIBRATED_PRIORS  # noqa: E402
from indra_belief.statement_belief import statement_belief  # noqa: E402

LOCAL = "local-gemma-4-26b"
# Gitignored under data/results/, so CI sees neither. Tests that need them skip.
FIT_RUN = ROOT / "data/results/eval_curation_v1_local-gemma-4-26b.jsonl"
VALIDATION_RUN = ROOT / "data/results/external_curator_v1_local-gemma-4-26b.jsonl"


def _rows(verdicts_by_source):
    return [
        {"source_api": src, "verdict": v, "confidence": "high", "tier": "llm_comprehension",
         "evidence_text": f"{src}-{i}-{v}"}
        for i, (src, v) in enumerate(verdicts_by_source)
    ]


def test_the_local_reader_resolves_a_profile_at_the_production_prompt():
    profile = calibration_for(LOCAL, prompt_sha256=REASONING_FIRST_PROMPT_SHA256)
    assert profile is not None, (
        "the self-hosted reader has no enabled profile at the production prompt; "
        "a local run would silently fall back to the source-counting hard gate"
    )
    assert profile["deployment_status"] == "enabled"
    assert profile["profile_id"] == "local-gemma-4-26b@prompt-07377e338ff2@eval_curation_v1"


def test_a_profile_is_keyed_on_the_prompt_not_the_model_alone():
    """Changing the scorer prompt invalidates the fit, so the baseline prompt
    must NOT inherit the reasoning-first profile."""
    assert calibration_for(LOCAL, prompt_sha256=BASELINE_PROMPT_SHA256) is None
    assert calibration_for(LOCAL, prompt_sha256=None) is None
    assert fitted_calibration_for(LOCAL, prompt_sha256=BASELINE_PROMPT_SHA256) is None


def test_the_registered_counts_reproduce_the_shipped_weights():
    """The four counts ARE the calibration; the weights are arithmetic on them.

    Transcribing counts from a gate report is the step where a digit can slip,
    so derive the weights here rather than trusting the stored values.
    """
    counts = _CONFUSION["local_gemma_mlx"]
    assert counts == {"cc": 651, "ci": 91, "ic": 148, "ii": 710}
    derived = profile_from_confusion(counts)
    profile = calibration_for(LOCAL, prompt_sha256=REASONING_FIRST_PROMPT_SHA256)
    for field in ("log_lr_confirm", "log_lr_reject", "prior_correct", "prior_logodds"):
        assert derived[field] == pytest.approx(profile[field], abs=1e-12), field
    # Sanity on direction: a confirmation must raise the odds, a rejection lower them.
    assert derived["log_lr_confirm"] > 0 > derived["log_lr_reject"]
    assert derived["log_lr_confirm"] == pytest.approx(1.9701501369938037, abs=1e-12)
    assert derived["log_lr_reject"] == pytest.approx(-1.5655526949691612, abs=1e-12)


def test_the_calibrated_belief_differs_from_the_fallback():
    """The load-bearing assertion: the same evidence must produce a DIFFERENT number.

    If these two agreed, registering the profile would have changed nothing and
    every other assertion here would be decoration.
    """
    profile = calibration_for(LOCAL, prompt_sha256=REASONING_FIRST_PROMPT_SHA256)
    rows = _rows([("reach", "correct"), ("sparser", "correct")])

    calibrated = statement_belief(rows, RECALIBRATED_PRIORS, soft=profile).belief
    fallback = statement_belief(rows, RECALIBRATED_PRIORS, soft=None).belief

    assert calibrated is not None and fallback is not None
    assert abs(calibrated - fallback) > 0.01, (
        f"calibrated {calibrated} and hard-gate {fallback} agree to within 0.01; "
        "the profile is not changing the belief"
    )
    # And the calibrated path must actually use the weights: a single confirmation
    # from one source reproduces sigmoid(prior + log_lr_confirm) when no source
    # floor binds. reach's recalibrated floor is negative, so the LR governs.
    one = statement_belief(_rows([("reach", "correct")]), RECALIBRATED_PRIORS, soft=profile).belief
    expected = 1.0 / (1.0 + math.exp(-(profile["prior_logodds"] + profile["log_lr_confirm"])))
    assert one == pytest.approx(expected, abs=1e-9)


def test_a_rejection_moves_the_belief_below_a_confirmation():
    profile = calibration_for(LOCAL, prompt_sha256=REASONING_FIRST_PROMPT_SHA256)
    up = statement_belief(_rows([("reach", "correct")]), RECALIBRATED_PRIORS, soft=profile).belief
    down = statement_belief(_rows([("reach", "incorrect")]), RECALIBRATED_PRIORS, soft=profile).belief
    assert down < 0.5 < up


@pytest.mark.parametrize("run_path", [FIT_RUN, VALIDATION_RUN], ids=["fit_run", "validation_run"])
def test_a_real_self_hosted_run_resolves_to_the_calibrated_path(run_path: Path):
    """End to end on an actual run, when the (gitignored) artifact is present.

    `reader_configuration_for_run` fingerprints the run's own call logs, so this
    exercises the identity plumbing rather than a declared label.
    """
    if not run_path.exists():
        pytest.skip(f"{run_path.name} is gitignored and absent")
    config = reader_configuration_for_run(str(run_path))
    assert config["status"] == "identified", config
    assert config["model"] == LOCAL
    assert config["prompt_sha256"] == REASONING_FIRST_PROMPT_SHA256
    assert config["prompt_fingerprint_source"] == "call_log"
    resolved = calibration_for_run(str(run_path))
    assert resolved is not None, (
        "a completed self-hosted run still resolves no profile — the defect this "
        "arc closed has regressed"
    )
    assert resolved["profile_id"].startswith(LOCAL)


def test_the_fit_run_reproduces_the_registered_counts():
    """The counts in the registry must be the counts the fit run actually yields."""
    if not FIT_RUN.exists():
        pytest.skip("fit run is gitignored and absent")
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_ship_gate as gate
    import calibration_stage1 as stage1

    statements, _ = gate.statements_for_run(
        str(FIT_RUN), str(ROOT / "data/benchmark/eval_curation_v1.jsonl")
    )
    assert stage1.fit_reader_profile(statements)["confusion"] == _CONFUSION["local_gemma_mlx"]
