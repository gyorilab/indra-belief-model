"""Binary classification metrics — the confusion math the eval scripts each
re-implemented. Lifted here so the SHIPPED F1/accuracy numbers have one
definition. The five consumers (y/z/aa phase profiles, cc + three_way holdout
compares) reproduce their committed .md reports byte-exactly through this module;
that golden-output equality is the extraction's safety net.

Design: the duplicated functions differed ONLY in how they reached the per-item
`(gold, pred)` booleans — `binary_confusion(rows)` read tag/verdict off each row,
`confusion(run, shared, src)` joined two dicts by shared keys. The math after
that point was identical. So the reusable core operates on a stream of
`(gold: bool, pred: bool)` pairs; each caller keeps its own one-line adapter.

The dict shape is preserved EXACTLY as the callers emitted it (two slightly
different key sets existed: tp/fp/fn/tn/accuracy/precision/recall/f1 for the
row form, and tp/fp/fn/tn/p/r/f1/acc for the join form). `confusion_metrics`
returns the long-key form; `confusion_pr` the short-key form. Neither invents a
key the callers didn't already produce.
"""
from __future__ import annotations

import statistics
from typing import Iterable


def confusion_counts(pairs: Iterable[tuple[bool, bool]]) -> tuple[int, int, int, int]:
    """Tally (tp, fp, fn, tn) over `(gold, pred)` boolean pairs. `correct`/
    supported is the positive class: tp = pred&gold, fp = pred&~gold,
    fn = ~pred&gold, tn = ~pred&~gold."""
    tp = fp = fn = tn = 0
    for gold, pred in pairs:
        if pred and gold:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _pr_f1_acc(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float, float]:
    n = tp + fp + fn + tn
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    acc = (tp + tn) / n if n else 0
    return p, r, f1, acc


def confusion_metrics(pairs: Iterable[tuple[bool, bool]]) -> dict:
    """Long-key confusion dict — the y/z_phase `binary_confusion` shape:
    {n, tp, fp, fn, tn, accuracy, precision, recall, f1}."""
    tp, fp, fn, tn = confusion_counts(pairs)
    p, r, f1, acc = _pr_f1_acc(tp, fp, fn, tn)
    n = tp + fp + fn + tn
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": acc, "precision": p, "recall": r, "f1": f1,
    }


def confusion_pr(pairs: Iterable[tuple[bool, bool]]) -> dict:
    """Short-key confusion dict — the aa/cc/three_way `confusion` shape:
    {tp, fp, fn, tn, p, r, f1, acc}."""
    tp, fp, fn, tn = confusion_counts(pairs)
    p, r, f1, acc = _pr_f1_acc(tp, fp, fn, tn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "p": p, "r": r, "f1": f1, "acc": acc}


# ---- Calibration ----------------------------------------------------------

# The standard 8-bin reliability scheme the five profile/holdout scripts each
# hardcoded identically: half-open [lo, hi) bins, the last one's 1.001 ceiling
# admitting score==1.0. Named here so "the 8-bin ECE" has one definition the
# way confusion_* gave the shipped F1/accuracy one. (The two benchmark scripts
# use a different 10-bin numpy scheme and stay separate — a different scheme
# yields a different ECE on the same data, and they carry no committed anchor.)
BINS_8 = [
    (0.0, 0.05), (0.05, 0.20), (0.20, 0.35), (0.35, 0.50),
    (0.50, 0.65), (0.65, 0.80), (0.80, 0.95), (0.95, 1.001),
]


def ece(items: Iterable[tuple[float, bool]], *, bins=BINS_8) -> float:
    """Expected Calibration Error over `(score, is_correct)` pairs.

    The five duplicated bodies differed ONLY in how each reached the per-row
    score and gold label — a row-carried `tag`, a `src[source_hash]` lookup, or
    a nested eval-subset lookup — and in an `n_all` that was always the
    count of rows actually iterated. The math after that was identical. So the
    reusable core takes pairs the caller has already resolved (`score` with its
    `(x.get("score") or 0.5)` fallback applied, `is_correct` the gold predicate)
    and each of the five keeps a one-line adapter. Per bin: the gap between mean
    predicted score and empirical correct-rate, weighted by the bin's share.

    Summation order matches the originals (the caller streams its rows in order
    → identical float accumulation → byte-identical `.3f` in the committed
    reports), which is this extraction's golden-output safety net.
    """
    pairs = list(items)
    n_all = len(pairs)
    if not n_all:
        return 0.0
    tot = 0.0
    for lo, hi in bins:
        bin_pairs = [(s, c) for s, c in pairs if lo <= s < hi]
        if not bin_pairs:
            continue
        mean_pred = statistics.mean(s for s, _ in bin_pairs)
        empirical = sum(1 for _, c in bin_pairs if c) / len(bin_pairs)
        tot += abs(mean_pred - empirical) * len(bin_pairs) / n_all
    return tot
