"""Tests for indra_belief.metrics — the binary confusion math lifted out of the
eval scripts. Locks the counts, the P/R/F1/accuracy formulas, the empty/edge
cases, and the two dict shapes (long-key from y/z_phase binary_confusion;
short-key from aa/cc/three_way confusion). The behavioral guarantee that the
five eval scripts reproduce their committed .md reports byte-exactly is the
real extraction proof; these lock the unit contract.
"""
from indra_belief.metrics import confusion_counts, confusion_metrics, confusion_pr


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
