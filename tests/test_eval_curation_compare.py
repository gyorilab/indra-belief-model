"""Unit tests for the paired err-F1 inference helpers in
scripts/eval_curation_compare.py (A4 back-port).

Covers the paired bootstrap ΔerrF1 CI and the permutation p-value on err-F1:
determinism under a fixed seed, the CI bracketing the point estimate, and the
permutation p staying in [1/(n_perm+1), 1] — the (hits+1)/(n_perm+1) correction
shared with frontier_paired_stats.paired_permutation_errf1, which keeps a finite
Monte Carlo run from reporting an impossible p of exactly zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_curation_compare as ec  # noqa: E402


# Small deterministic fixture: 12 statements. positive class = error (True).
# Model A misses two real errors (under-flags); model B catches them — so B's
# err-F1 should exceed A's, giving a positive Δ.
GOLD_ERR = [True, True, True, True, True, True, False, False, False, False, False, False]
PRED_A = [True, True, True, True, False, False, False, False, False, False, False, False]
PRED_B = [True, True, True, True, True, True, False, False, False, True, False, False]

# Perfectly separated pair at n=40: A reproduces gold exactly (err-F1 1.0), B is
# gold inverted (err-F1 0.0), so the observed |ΔerrF1| is the maximal 1.0. The only
# permutations that tie it are the two all-same swaps, probability 2/2**40 — no
# 2000-perm run hits one, so `hits` is 0 and p sits exactly on the floor.
SEP_GOLD = [True] * 20 + [False] * 20
SEP_PRED_A = list(SEP_GOLD)
SEP_PRED_B = [not x for x in SEP_GOLD]


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
    # (hits+1)/(n_perm+1) floors p strictly above zero, for every input.
    assert p1 >= 1 / (500 + 1)


def test_permutation_p_never_zero_for_separated_inputs():
    """A perfectly separated pair returns the Monte Carlo FLOOR, never 0.0.

    hits == 0 here, so the uncorrected hits/n_perm would report p = 0.0000 — an
    impossible p-value for a finite permutation run. (hits+1)/(n_perm+1) reports
    1/2001 instead: "no permutation beat the observed split", not "p is zero".
    """
    for n_perm in (500, 2000):
        p = ec.permutation_errf1(SEP_GOLD, SEP_PRED_A, SEP_PRED_B,
                                 n_perm=n_perm, seed=0)
        assert p > 0.0
        assert p == pytest.approx(1 / (n_perm + 1))
    # The floor the report discloses as `minimum attainable p` at the shipped N_PERM.
    assert ec.permutation_errf1(SEP_GOLD, SEP_PRED_A, SEP_PRED_B,
                                n_perm=ec.N_PERM, seed=0) == 1 / (ec.N_PERM + 1)


def test_permutation_p_high_when_models_identical():
    """If A and B predict identically, every swap leaves |Δ|=0 == observed, so
    p == 1.0 (all permuted |Δ| >= observed)."""
    p = ec.permutation_errf1(GOLD_ERR, PRED_A, PRED_A, n_perm=200, seed=0)
    assert p == 1.0
