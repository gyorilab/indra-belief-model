"""Regression guards for the three measured calibration ship-gate legs.

The all-four-green case and the requirement to assert E4 identity explicitly
are owned by
``tests/test_metrics_export.py::test_ship_gate_requires_explicit_e4_identity_assertion``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibration_ship_gate as ship_gate  # noqa: E402


def _ev(**overrides):
    values = {
        "hard_ece": 0.2,
        "clean_ece": 0.1,
        "hard_auroc": 0.8,
        "clean_auroc": 0.81,
        "f1_hard": 0.7,
        "f1_soft": 0.71,
        "delta": 0.01,
        "ci_low": -ship_gate.NI_MARGIN + 0.01,
        "ci_high": 0.03,
    }
    values.update(overrides)
    return {
        "metrics": {
            "hard": {"ece": values["hard_ece"], "auroc": values["hard_auroc"]},
            "clean": {
                "ece": values["clean_ece"],
                "auroc": values["clean_auroc"],
            },
        },
        "errf1_boot": {
            "f1_hard": values["f1_hard"],
            "f1_soft": values["f1_soft"],
            "delta": values["delta"],
            "ci_delta": [values["ci_low"], values["ci_high"]],
        },
    }


def test_ece_leg_passes_when_calibrated_ece_is_lower():
    result = ship_gate.gate(_ev(clean_ece=0.1), e4_identity_pass=True)

    assert result["ece"]["pass"] is True
    assert result["auroc"]["pass"] is True
    assert result["errf1"]["pass"] is True
    assert result["overall"] is True


def test_ece_leg_rejects_higher_calibrated_ece():
    result = ship_gate.gate(_ev(clean_ece=0.3), e4_identity_pass=True)

    assert result["ece"]["pass"] is False
    assert result["auroc"]["pass"] is True
    assert result["errf1"]["pass"] is True
    assert result["overall"] is False


def test_ece_leg_rejects_equal_ece_boundary():
    result = ship_gate.gate(_ev(clean_ece=0.2), e4_identity_pass=True)

    assert result["ece"]["pass"] is False
    assert result["auroc"]["pass"] is True
    assert result["errf1"]["pass"] is True
    assert result["overall"] is False


def test_auroc_leg_passes_when_calibrated_auroc_is_higher():
    result = ship_gate.gate(_ev(clean_auroc=0.81), e4_identity_pass=True)

    assert result["auroc"]["pass"] is True
    assert result["ece"]["pass"] is True
    assert result["errf1"]["pass"] is True
    assert result["overall"] is True


def test_auroc_leg_rejects_drop_above_float_noise():
    result = ship_gate.gate(_ev(clean_auroc=0.8 - 1e-6), e4_identity_pass=True)

    assert result["auroc"]["pass"] is False
    assert result["ece"]["pass"] is True
    assert result["errf1"]["pass"] is True
    assert result["overall"] is False


def test_auroc_leg_accepts_exact_equality_boundary():
    assert ship_gate.EPS == 0.0
    result = ship_gate.gate(
        _ev(clean_auroc=0.8 - ship_gate.EPS), e4_identity_pass=True
    )

    assert result["auroc"]["pass"] is True
    assert result["ece"]["pass"] is True
    assert result["errf1"]["pass"] is True
    assert result["overall"] is True


def test_errf1_leg_passes_when_ci_low_is_above_margin():
    result = ship_gate.gate(
        _ev(ci_low=-ship_gate.NI_MARGIN + 0.01), e4_identity_pass=True
    )

    assert result["errf1"]["pass"] is True
    assert result["ece"]["pass"] is True
    assert result["auroc"]["pass"] is True
    assert result["overall"] is True


def test_errf1_leg_rejects_ci_low_below_margin():
    result = ship_gate.gate(
        _ev(ci_low=-ship_gate.NI_MARGIN - 0.01), e4_identity_pass=True
    )

    assert result["errf1"]["pass"] is False
    assert result["ece"]["pass"] is True
    assert result["auroc"]["pass"] is True
    assert result["overall"] is False


def test_errf1_leg_accepts_exact_noninferiority_boundary():
    # This is the pre-specified, deliberately disclosed medpsy-4B identical-run
    # spread printed in the generated report; moving it re-baselines the gate.
    assert ship_gate.NI_MARGIN == 0.154
    result = ship_gate.gate(
        _ev(ci_low=-ship_gate.NI_MARGIN), e4_identity_pass=True
    )

    assert result["errf1"]["pass"] is True
    assert result["ece"]["pass"] is True
    assert result["auroc"]["pass"] is True
    assert result["overall"] is True


def test_errf1_leg_rejects_just_below_noninferiority_boundary():
    result = ship_gate.gate(
        _ev(ci_low=-ship_gate.NI_MARGIN - 1e-6), e4_identity_pass=True
    )

    assert result["errf1"]["pass"] is False
    assert result["ece"]["pass"] is True
    assert result["auroc"]["pass"] is True
    assert result["overall"] is False


def test_ece_leg_auroc_leg_errf1_leg_conjunction_rejects_all_failures():
    result = ship_gate.gate(
        _ev(
            clean_ece=0.3,
            clean_auroc=0.8 - 1e-6,
            ci_low=-ship_gate.NI_MARGIN - 1e-6,
        ),
        e4_identity_pass=True,
    )

    assert result["ece"]["pass"] is False
    assert result["auroc"]["pass"] is False
    assert result["errf1"]["pass"] is False
    assert result["overall"] is False
