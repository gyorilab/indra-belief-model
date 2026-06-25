"""Unit tests for the paired err-F1 inference helpers in
scripts/eval_curation_compare.py (A4 back-port).

Covers the paired bootstrap ΔerrF1 CI and the permutation p-value on err-F1:
determinism under a fixed seed, the CI bracketing the point estimate, and the
permutation p staying in [0, 1].
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_curation_compare as ec  # noqa: E402


# Small deterministic fixture: 12 statements. positive class = error (True).
# Model A misses two real errors (under-flags); model B catches them — so B's
# err-F1 should exceed A's, giving a positive Δ.
GOLD_ERR = [True, True, True, True, True, True, False, False, False, False, False, False]
PRED_A = [True, True, True, True, False, False, False, False, False, False, False, False]
PRED_B = [True, True, True, True, True, True, False, False, False, True, False, False]


def test_point_estimate_matches_confusion_pr():
    """The bootstrap's f1_a/f1_b POINT estimates equal a direct confusion_pr.f1."""
    boot = ec.bootstrap_errf1(GOLD_ERR, PRED_A, PRED_B, n_boot=200, seed=0)
    f1_a_direct = ec._errf1_from_pairs(GOLD_ERR, PRED_A)
    f1_b_direct = ec._errf1_from_pairs(GOLD_ERR, PRED_B)
    assert boot["f1_a"] == f1_a_direct
    assert boot["f1_b"] == f1_b_direct
    assert boot["delta"] == f1_b_direct - f1_a_direct


def test_bootstrap_deterministic_under_seed():
    """Same seed -> identical CIs; different seed -> may differ (sanity, not asserted)."""
    a = ec.bootstrap_errf1(GOLD_ERR, PRED_A, PRED_B, n_boot=300, seed=0)
    b = ec.bootstrap_errf1(GOLD_ERR, PRED_A, PRED_B, n_boot=300, seed=0)
    assert a == b
    assert a["n"] == len(GOLD_ERR)


def test_ci_brackets_point_estimate():
    """Each model's 95% bootstrap CI brackets its own point err-F1, and the
    ΔerrF1 CI brackets the observed Δ."""
    boot = ec.bootstrap_errf1(GOLD_ERR, PRED_A, PRED_B, n_boot=1000, seed=0)
    lo_a, hi_a = boot["ci_a"]
    lo_b, hi_b = boot["ci_b"]
    lo_d, hi_d = boot["ci_delta"]
    assert lo_a <= boot["f1_a"] <= hi_a
    assert lo_b <= boot["f1_b"] <= hi_b
    assert lo_d <= boot["delta"] <= hi_d
    # B catches more real errors than A here.
    assert boot["delta"] > 0


def test_permutation_p_in_unit_interval_and_deterministic():
    p1 = ec.permutation_errf1(GOLD_ERR, PRED_A, PRED_B, n_perm=500, seed=0)
    p2 = ec.permutation_errf1(GOLD_ERR, PRED_A, PRED_B, n_perm=500, seed=0)
    assert p1 == p2  # fixed seed -> deterministic
    assert 0.0 <= p1 <= 1.0


def test_permutation_p_high_when_models_identical():
    """If A and B predict identically, every swap leaves |Δ|=0 == observed, so
    p == 1.0 (all permuted |Δ| >= observed)."""
    p = ec.permutation_errf1(GOLD_ERR, PRED_A, PRED_A, n_perm=200, seed=0)
    assert p == 1.0
