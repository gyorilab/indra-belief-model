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

import numpy as np


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
    reusable core takes pairs the caller has already resolved (`score`, and
    `is_correct` the gold predicate) and each of the five keeps a one-line
    adapter. A row with NO score is EXCLUDED by its adapter, never defaulted to
    0.5: an absent measurement is not a neutral one, and calibrating against an
    invented midpoint is exactly the error `indra_belief.verdict` exists to make
    unrepresentable. Per bin: the gap between mean
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


# ---- Discrimination + Brier (lifted verbatim from scripts/calibration_stage0.py
#      so results.py and the calibration scripts share ONE definition — the same
#      metrics-extraction discipline as confusion_*/ece. The scripts re-import
#      these; their committed .md/.json reproduce byte-exact through this module).


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (no scipy)."""
    order = a.argsort(kind="mergesort")
    sa = a[order]
    n = len(a)
    rnk = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        rnk[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    out = np.empty(n, float)
    out[order] = rnk
    return out


def auroc(scores, labels) -> float:
    """AUROC via Mann-Whitney U. positive class = label True."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(s)
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auprc(scores, labels) -> float:
    """Average precision (AUPRC), positive class = label True.

    AP = sum over THRESHOLDS of (R_k - R_{k-1}) * P_k. A threshold is a distinct
    score, not a row: rows carrying the same score are indistinguishable to any
    decision rule, so they collapse into ONE point on the PR curve. Walking the
    curve row-by-row instead would let the arbitrary within-tie ordering decide
    the answer and would inflate it (each tied positive would be credited at a
    precision that no reachable threshold actually delivers). The result here
    equals `sklearn.metrics.average_precision_score` and is invariant to the
    order of the input rows.

    Tie grouping was mirrored from the since-removed comparison harness's
    `_weighted_pr_summaries`
    (the weighted sibling; same reduceat algorithm, weights all 1 here). Mirrored
    rather than imported: that module imports sklearn at top level and its
    `_unit_interval` raises on out-of-range input.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_labels = y[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    cumulative_total = np.cumsum(np.add.reduceat(np.ones(len(sorted_scores)), starts))
    cumulative_positive = np.cumsum(np.add.reduceat(sorted_labels.astype(float), starts))
    precision = cumulative_positive / cumulative_total
    recall = cumulative_positive / n_pos
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def brier_murphy(scores, labels, bins=BINS_8) -> dict:
    """Brier score + Murphy decomposition (reliability - resolution + uncertainty)."""
    p = np.asarray(scores, float)
    y = np.asarray(labels, float)
    n = len(p)
    if n == 0:
        return {"brier": float("nan"), "reliability": float("nan"),
                "resolution": float("nan"), "uncertainty": float("nan"), "n": 0}
    brier = float(np.mean((p - y) ** 2))
    ybar = float(np.mean(y))
    uncertainty = ybar * (1.0 - ybar)
    reliability = 0.0
    resolution = 0.0
    for lo, hi in bins:
        m = (p >= lo) & (p < hi)
        nk = int(m.sum())
        if nk == 0:
            continue
        pbar_k = float(np.mean(p[m]))
        ybar_k = float(np.mean(y[m]))
        reliability += nk / n * (pbar_k - ybar_k) ** 2
        resolution += nk / n * (ybar_k - ybar) ** 2
    return {"brier": brier, "reliability": reliability,
            "resolution": resolution, "uncertainty": uncertainty, "n": n}


def reliability_bins(scores, labels, bins=BINS_8) -> list[dict]:
    p = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    out = []
    for lo, hi in bins:
        m = (p >= lo) & (p < hi)
        nk = int(m.sum())
        out.append({
            "lo": lo, "hi": hi, "n": nk,
            "mean_pred": float(np.mean(p[m])) if nk else None,
            "empirical": float(np.mean(y[m])) if nk else None,
        })
    return out
