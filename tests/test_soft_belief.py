"""E4 — byte-identity + correctness guard for calibrated log-odds belief.

The default-off hard path must remain byte-identical. The calibrated path must
match the deterministic stage-1 reference, preserve its source-reliability floor,
remain numerically stable, and forward through contradiction recursion.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

from indra_belief.noise_model import (
    RECALIBRATED_PRIORS,
    compute_gated_belief,
    compute_gated_belief_with_contradiction,
)

# gemma reader profile — DERIVED from its confusion matrix (calibration_constants),
# not hand-set. The belief model consumes per-verdict log-LIKELIHOOD-RATIOS.
from indra_belief.calibration_constants import (  # noqa: E402
    BASELINE_PROMPT_SHA256,
    EXTERNAL_GOLD_SHA256,
    FIT_GOLD_SHA256,
    HOLDOUT_GOLD_SHA256,
    REASONING_FIRST_PROMPT_SHA256,
    calibration_for,
    fitted_calibration_for,
)
_GEMMA = calibration_for(
    "remote-gemma-4-26b", prompt_sha256=BASELINE_PROMPT_SHA256
)
LOG_LR_CONFIRM = _GEMMA["log_lr_confirm"]
LOG_LR_REJECT = _GEMMA["log_lr_reject"]
PRIOR_LOGODDS = _GEMMA["prior_logodds"]
CC, CI, IC, II = 704, 157, 97, 646  # unique-pair remote-gemma confusion cells

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
        log_lr_confirm=LOG_LR_CONFIRM, log_lr_reject=LOG_LR_REJECT,
        prior_logodds=PRIOR_LOGODDS, **kw,
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

def test_soft_matches_stage1_reference():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import calibration_stage1 as c1  # noqa: E402

    ref = c1.soft_belief(MIXED, LOG_LR_CONFIRM, LOG_LR_REJECT, RECALIBRATED_PRIORS, PRIOR_LOGODDS)
    got = _soft(MIXED).belief
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
        log_lr_confirm=LOG_LR_CONFIRM, log_lr_reject=LOG_LR_REJECT, prior_logodds=PRIOR_LOGODDS,
    )
    # If the kwargs were not forwarded at the per-direction recursion, soft == hard.
    assert abs(soft.belief - hard.belief) > 1e-6


# ── fitted-constants resolver (calibration_constants) ──────────────────────────

def test_calibration_resolver():
    # parameters are LIKELIHOODS (conditioned on the truth), derived from the
    # confusion matrix, not hand-set: duplicate evidence pairs are removed.
    gp = calibration_for(
        "remote-gemma-4-26b", prompt_sha256=BASELINE_PROMPT_SHA256
    )
    assert gp["confusion"] == {"cc": 704, "ci": 157, "ic": 97, "ii": 646}
    assert math.isclose(gp["sensitivity"], 704 / 801)              # P(confirm | correct)
    assert math.isclose(gp["false_positive_rate"], 157 / 803)      # P(confirm | incorrect)
    assert math.isclose(gp["log_lr_confirm"], math.log((704 / 801) / (157 / 803)))
    assert math.isclose(gp["prior_correct"], 801 / 1604)           # unique-pair base rate
    assert gp["profile_id"].startswith("remote-gemma-4-26b@")
    assert gp["fit_gold_sha256"] == FIT_GOLD_SHA256
    # A model name alone is never enough to identify the scorer configuration.
    assert calibration_for("remote-gemma-4-26b") is None
    assert calibration_for("gemma") is None
    # Same weights on an unvalidated host/configuration do not inherit a fit.
    assert calibration_for(
        "local-gemma-4-26b", prompt_sha256=BASELINE_PROMPT_SHA256
    ) is None
    assert calibration_for(
        "google-gemma-4-26b", prompt_sha256=BASELINE_PROMPT_SHA256
    ) is None
    # MedPsy has a measured candidate, but its matched holdout failed the ECE
    # leg, so production resolution remains disabled.
    mp = fitted_calibration_for(
        "remote-medpsy-4b", prompt_sha256=BASELINE_PROMPT_SHA256
    )
    assert math.isclose(mp["sensitivity"], 718 / (718 + 83))
    assert mp["deployment_status"] == "disabled"
    assert calibration_for(
        "remote-medpsy-4b", prompt_sha256=BASELINE_PROMPT_SHA256
    ) is None
    assert "kappa" not in mp  # no correlation exponent in the model
    # gemma-4-31B is a DIFFERENT model — must NOT inherit the 26B fit
    assert calibration_for("local-gemma-4-31b") is None
    assert calibration_for("google-gemma-4-31b") is None
    # The local MLX reader at the reasoning-first prompt has its OWN measured
    # serving/configuration profile. This read the bedrock-gemma-4-26b profile
    # (confusion cc 1995 / ci 336 / ic 467 / ii 1505, refitted onto
    # holdout_large_fit) until the paid Bedrock lane was removed.
    bp = calibration_for(
        "local-gemma-4-26b", prompt_sha256=REASONING_FIRST_PROMPT_SHA256
    )
    # Fitted on eval_curation_v1; see calibration_constants._CONFUSION.
    assert bp["confusion"] == {"cc": 651, "ci": 91, "ic": 148, "ii": 710}
    assert bp["confusion"] != gp["confusion"]
    assert calibration_for(
        "local-gemma-4-26b", prompt_sha256=BASELINE_PROMPT_SHA256
    ) is None
    assert calibration_for("local-gemma-4-31b") is None
    assert calibration_for(None) is None
    assert calibration_for("some-unfitted-model") is None
    assert calibration_for("not-gemma") is None
    assert calibration_for("gemma-2-9b") is None
    assert calibration_for("medpsy-8b") is None


def test_profile_gold_digests_match_pinned_artifacts():
    root = Path(__file__).resolve().parents[1]
    # Tracked golds are byte-checked on every checkout.
    pinned = {
        "data/benchmark/eval_curation_v1.jsonl": FIT_GOLD_SHA256,
        "data/benchmark/external_curator_gold_v1.jsonl": EXTERNAL_GOLD_SHA256,
    }
    for relative, expected in pinned.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected

    # The old holdout is a large ignored validation input. Check its bytes when
    # restored locally, while always pinning its digest through the tracked gate
    # decision so a clean clone remains testable and auditable.
    holdout = root / "data/results/cc_holdout_cc/holdout_cc.jsonl"
    if holdout.exists():
        assert hashlib.sha256(holdout.read_bytes()).hexdigest() == HOLDOUT_GOLD_SHA256
    gate = json.loads((root / "data/results/calibration_ship_gate.json").read_text())
    assert {row["provenance"]["test_gold_sha256"] for row in gate} == {
        HOLDOUT_GOLD_SHA256
    }


# ── statement_belief soft integration (end-to-end through the production roll-up) ──

def _ev(source_api, verdict, **kw):
    return {"source_api": source_api, "verdict": verdict, "confidence": "high",
            "tier": "llm_comprehension", **kw}


def test_statement_belief_soft_lifts_confirmed_single_read():
    """A gemma-confirmed single reach read: hard gives 1-(0.05+0.462)=0.488; the
    reader confirm log-LR gives 0.817 at the fit anchor, matching measured
    P(correct | confirmed Gemma read). Reach's source floor does not bind."""
    from indra_belief.statement_belief import statement_belief
    from indra_belief.calibration_constants import calibration_for
    rows = [_ev("reach", "correct", evidence_text="a")]
    hard = statement_belief(rows, RECALIBRATED_PRIORS).belief
    soft = statement_belief(rows, RECALIBRATED_PRIORS, soft=_GEMMA).belief
    assert math.isclose(hard, 0.488, abs_tol=1e-3)
    assert math.isclose(soft, 0.817, abs_tol=1e-3)
    assert soft > hard


def test_conflicting_duplicate_resolution_is_order_invariant_and_conservative():
    """A retry conflict must not let JSONL order choose the canonical belief."""
    from indra_belief.statement_belief import statement_belief

    correct = _ev("reach", "correct", evidence_text="same evidence")
    incorrect = _ev(
        "reach", "incorrect", evidence_text="  SAME   EVIDENCE  ",
        tier="deterministic_mismatch",
    )
    forward = statement_belief([correct, incorrect], soft=_GEMMA)
    reverse = statement_belief([incorrect, correct], soft=_GEMMA)
    assert forward.as_dict() == reverse.as_dict()
    assert forward.n_evidence == 2 and forward.n_dedup_groups == 1
    assert forward.n_correct == 0 and forward.n_incorrect == 1
    assert forward.verdict_statement == "incorrect"


def test_clean_is_self_calibrating_at_n1():
    """At the fit prior, an ordinary single read equals measured verdict accuracy."""
    confirmed = _soft([{"source_api": "reach", "verdict": "correct"}]).belief
    rejected = _soft([{"source_api": "reach", "verdict": "incorrect"}]).belief
    assert math.isclose(confirmed, CC / (CC + CI), abs_tol=1e-9)   # P(correct | confirmed)
    assert math.isclose(rejected, IC / (IC + II), abs_tol=1e-9)    # P(correct | rejected)


def test_confirm_uses_reader_lr_for_an_ordinary_text_miner():
    """Reach is weaker than the reader confirm LR, so the reader controls."""
    ev = [{"source_api": "reach", "verdict": "correct"}]
    clean = _soft(ev).belief
    assert math.isclose(clean, CC / (CC + CI), abs_tol=1e-9)


def test_confirm_preserves_stronger_curated_source_floor():
    """A confirmed curated source is never weakened by the generic reader profile."""
    signor = _soft([{"source_api": "signor", "verdict": "correct"}]).belief
    source_reliability = 1.0 - sum(RECALIBRATED_PRIORS["signor"])
    expected = 1.0 / (
        1.0 + math.exp(-(PRIOR_LOGODDS + math.log(source_reliability / (1.0 - source_reliability))))
    )
    assert math.isclose(signor, expected, abs_tol=1e-9)
    assert signor > CC / (CC + CI)


def test_many_independent_sources_are_numerically_stable():
    rows = [
        {"source_api": f"source-{i}", "verdict": "correct", "included": True}
        for i in range(1000)
    ]
    assert _soft(rows).belief == 1.0


def test_statement_belief_soft_preserves_undefined_contract():
    """all-no_text → undefined belief + review on both paths."""
    from indra_belief.statement_belief import statement_belief
    from indra_belief.calibration_constants import calibration_for
    rows = [_ev("signor", "correct", tier="no_text"), _ev("biogrid", "correct", tier="no_text")]
    hard = statement_belief(rows, RECALIBRATED_PRIORS)
    soft = statement_belief(rows, RECALIBRATED_PRIORS, soft=_GEMMA)
    assert hard.belief is None and hard.verdict_statement == "review"
    assert soft.belief is None and soft.verdict_statement == "review"


def test_empty_soft_core_validates_profile_and_uses_anchor():
    with pytest.raises(ValueError):
        compute_gated_belief([], RECALIBRATED_PRIORS, soft_weights=True)
    result = compute_gated_belief(
        [], RECALIBRATED_PRIORS, soft_weights=True,
        log_lr_confirm=LOG_LR_CONFIRM, log_lr_reject=LOG_LR_REJECT,
        prior_logodds=math.log(3),
    )
    assert math.isclose(result.belief, 0.75)
