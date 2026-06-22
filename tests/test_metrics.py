"""Tests for indra_belief.metrics — the binary confusion math lifted out of the
eval scripts. Locks the counts, the P/R/F1/accuracy formulas, the empty/edge
cases, and the two dict shapes (long-key from y/z_phase binary_confusion;
short-key from aa/cc/three_way confusion). The behavioral guarantee that the
five eval scripts reproduce their committed .md reports byte-exactly is the
real extraction proof; these lock the unit contract.
"""
import math

from indra_belief.metrics import (
    BINS_8,
    auprc,
    auroc,
    brier_murphy,
    confusion_counts,
    confusion_metrics,
    confusion_pr,
    ece,
    reliability_bins,
)


def test_confusion_counts_quadrants():
    # (gold, pred): TP=both, FP=pred only, FN=gold only, TN=neither
    pairs = [(True, True), (False, True), (True, False), (False, False), (True, True)]
    assert confusion_counts(pairs) == (2, 1, 1, 1)


def test_confusion_metrics_long_key_shape_and_math():
    # 3 TP, 1 FP, 1 FN, 1 TN
    pairs = [(True, True)] * 3 + [(False, True), (True, False), (False, False)]
    m = confusion_metrics(pairs)
    assert set(m) == {"n", "tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1"}
    assert (m["tp"], m["fp"], m["fn"], m["tn"], m["n"]) == (3, 1, 1, 1, 6)
    assert m["precision"] == 3 / 4
    assert m["recall"] == 3 / 4
    assert m["f1"] == 0.75
    assert m["accuracy"] == 4 / 6


def test_confusion_pr_short_key_shape():
    pairs = [(True, True)] * 3 + [(False, True), (True, False), (False, False)]
    m = confusion_pr(pairs)
    assert set(m) == {"tp", "fp", "fn", "tn", "p", "r", "f1", "acc"}
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 1)
    assert m["p"] == 0.75 and m["r"] == 0.75 and m["f1"] == 0.75
    assert m["acc"] == 4 / 6


def test_empty_is_all_zero_not_nan():
    m = confusion_metrics([])
    assert m["n"] == 0
    assert m["precision"] == 0 and m["recall"] == 0 and m["f1"] == 0 and m["accuracy"] == 0
    p = confusion_pr([])
    assert p["p"] == 0 and p["r"] == 0 and p["f1"] == 0 and p["acc"] == 0


def test_no_positives_predicted_zero_precision_recall():
    # all gold-positive, none predicted positive → P undefined→0, R=0, all FN
    pairs = [(True, False), (True, False)]
    m = confusion_metrics(pairs)
    assert (m["tp"], m["fn"]) == (0, 2)
    assert m["precision"] == 0 and m["recall"] == 0 and m["f1"] == 0
    assert m["accuracy"] == 0


def test_perfect_classifier():
    pairs = [(True, True), (False, False), (True, True)]
    m = confusion_metrics(pairs)
    assert m["precision"] == 1 and m["recall"] == 1 and m["f1"] == 1 and m["accuracy"] == 1


# ---- ECE ------------------------------------------------------------------
# `ece` takes (score, is_correct) pairs the caller has already resolved (the
# five profile scripts each kept a one-line adapter that applies the
# `(x.get("score") or 0.5)` fallback + the gold predicate). These lock the
# 8-bin math; the real proof is the five scripts' .md reports reproducing
# byte-exactly through this function.


def test_ece_empty_is_zero_not_nan():
    assert ece([]) == 0.0


def test_ece_hand_computed_two_bins():
    # bin (0.05,0.20): two 0.1 scores, 1 correct → |0.1 - 0.5| * 2/4 = 0.2
    # bin (0.80,0.95): two 0.9 scores, 1 correct → |0.9 - 0.5| * 2/4 = 0.2
    items = [(0.9, True), (0.9, False), (0.1, False), (0.1, True)]
    assert ece(items) == 0.4


def test_ece_perfect_calibration_is_zero():
    # all scores 0.9, empirical correct-rate also 0.9 → bin gap 0
    items = [(0.9, True)] * 9 + [(0.9, False)]
    assert ece(items) == 0.0


def test_ece_bins_are_half_open_low_inclusive():
    # 0.05 lands in (0.05,0.20) NOT (0,0.05); 0.95 and 1.0 both in (0.95,1.001).
    # one item per occupied bin → each bin gap weighted by 1/3, summed.
    items = [(0.05, True), (0.95, False), (1.0, False)]
    # (0.05,0.20): mean 0.05, emp 1.0, gap 0.95, w 1/3
    # (0.95,1.001): two items, mean 0.975, emp 0.0, gap 0.975, w 2/3
    expected = 0.95 * (1 / 3) + 0.975 * (2 / 3)
    assert abs(ece(items) - expected) < 1e-12


def test_ece_default_bins_are_the_8_bin_scheme():
    assert BINS_8 == [
        (0.0, 0.05), (0.05, 0.20), (0.20, 0.35), (0.35, 0.50),
        (0.50, 0.65), (0.65, 0.80), (0.80, 0.95), (0.95, 1.001),
    ]
    assert len(BINS_8) == 8


# ---- AUROC / AUPRC / Brier / reliability_bins -----------------------------
# Lifted VERBATIM from scripts/calibration_stage0.py into src so results.py and
# the calibration scripts share ONE definition. The real proof is that
# calibration_stage0.md/.json reproduce their COMMITTED bytes through these (the
# metric-extraction discipline; verified at lift time and on every regen). These
# lock the unit contract + the re-export identity (the structural-change guard).


def test_lifted_metrics_are_reexported_by_calibration_stage0():
    # calibration_stage0 must resolve to the SAME function objects (it re-imports
    # them from src) — so a future divergence can't silently desync the served
    # metrics.json numbers from the ship-gate numbers.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import calibration_stage0 as c0
    assert c0.auroc is auroc
    assert c0.auprc is auprc
    assert c0.brier_murphy is brier_murphy
    assert c0.reliability_bins is reliability_bins


def test_auroc_perfect_and_reversed():
    # perfect separation → 1.0; fully reversed → 0.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    # single-class → NaN
    assert math.isnan(auroc([0.1, 0.2], [True, True]))


def test_auroc_ties_averaged():
    # two tied scores straddling the boundary → 0.5 (chance)
    assert auroc([0.5, 0.5], [True, False]) == 0.5


def test_auprc_perfect_is_one():
    assert auprc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert math.isnan(auprc([0.1, 0.2], [False, False]))  # no positives


def test_brier_murphy_decomposition_identity():
    # Brier == reliability − resolution + uncertainty (Murphy)
    scores = [0.9, 0.9, 0.1, 0.1, 0.6, 0.4]
    labels = [True, False, False, True, True, False]
    b = brier_murphy(scores, labels)
    assert b["brier"] == pytest_approx(
        b["reliability"] - b["resolution"] + b["uncertainty"])
    assert b["n"] == 6
    # empty → all NaN, n 0
    e = brier_murphy([], [])
    assert e["n"] == 0 and math.isnan(e["brier"])


def test_reliability_bins_are_eight_with_null_empties():
    bins = reliability_bins([0.97, 0.97], [True, False])
    assert len(bins) == 8
    occupied = [b for b in bins if b["n"]]
    assert len(occupied) == 1  # both in (0.95, 1.001)
    assert occupied[0]["n"] == 2
    assert occupied[0]["mean_pred"] == pytest_approx(0.97)
    assert occupied[0]["empirical"] == pytest_approx(0.5)
    empties = [b for b in bins if b["n"] == 0]
    assert all(b["mean_pred"] is None and b["empirical"] is None for b in empties)


def pytest_approx(x):
    import pytest
    return pytest.approx(x)
