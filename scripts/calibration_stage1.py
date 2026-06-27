"""Calibration Stage C1 — soft-survival-weight library surface.

The reusable library functions behind the soft survival weight: fit the per-read
weights from train cells, split statements, and score belief on the hard gate or
the adopted clean soft form. ``scripts/calibration_ship_gate.py`` imports these
(``statements_from_joined``, ``split_stratified``, ``fit_weights``,
``hard_belief``, ``soft_belief``, ``metric_block``) for the G2 ship gate.

The soft model:

    per source s, per read j with verdict v_j:
        w_j = w_correct     if v_j == "correct"     (= P(read does NOT support | confirmed))
            = w_incorrect    if v_j == "incorrect"   (= P(read does NOT support | rejected))
            = base_s         if v_j is None          (prior fallback, syst+rand)
        f_s = geomean_j w_j              (a source's reads are correlated, so they
                                          do NOT compound — one aggregate per source)
        belief = 1 - prod_s f_s

CRITICAL: w_j is P(read does not support) = P(gold == incorrect | verdict). For a
REJECTED read that is (1 - rand_rej) ~ 0.87 (high w -> low belief), NOT rand_rej.
We fit both conditionals directly from the train cells to avoid the sign trap:
    w_correct   = P(gold incorrect | verdict correct)   [== rand_corr]
    w_incorrect = P(gold incorrect | verdict incorrect)  [== 1 - rand_rej]
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import calibration_stage0 as c0  # noqa: E402  (reuse join + metrics)
from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402
from indra_belief.metrics import ece  # noqa: E402
from indra_belief.noise_model import (  # noqa: E402
    _DEFAULT_PRIOR,
    RECALIBRATED_PRIORS,
    compute_gated_belief,
)

PRIORS = RECALIBRATED_PRIORS


def statements_from_joined(joined) -> list[dict]:
    """Group joined (gold, scored) pairs into statements keyed by pa_hash."""
    by_stmt: dict[int, dict] = defaultdict(lambda: {"ev": [], "tags": []})
    for g, s in joined:
        rec = by_stmt[g["pa_hash"]]
        rec["ev"].append({
            "source_api": s.get("source_api") or g.get("source_api"),
            "verdict": s.get("verdict"),
            "ev_gold_correct": is_gold_correct(g["tag"]),
        })
        rec["tags"].append(g["tag"])
    out = []
    for pa, rec in by_stmt.items():
        gold = aggregate_gold(rec["tags"])
        if gold is None:
            continue
        out.append({"pa_hash": pa, "ev": rec["ev"],
                    "n_ev": len(rec["ev"]), "gold_correct": is_gold_correct(gold)})
    return out


def split_stratified(stmts, seed=0, frac_train=0.5):
    """Stratified train/test split by per-statement gold (deterministic)."""
    rng = random.Random(seed)
    pos = [s for s in stmts if s["gold_correct"]]
    neg = [s for s in stmts if not s["gold_correct"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    tr, te = [], []
    for group in (pos, neg):
        k = int(round(len(group) * frac_train))
        tr += group[:k]
        te += group[k:]
    return tr, te


def fit_weights(train_stmts) -> dict:
    """Fit the two per-read conditional wrong-rates from train EVIDENCES."""
    cc = ci = ic = ii = 0  # (verdict, ev_gold)
    for st in train_stmts:
        for ev in st["ev"]:
            gc = ev["ev_gold_correct"]
            if ev["verdict"] == "correct":
                cc += gc
                ci += (not gc)
            elif ev["verdict"] == "incorrect":
                ic += gc
                ii += (not gc)
    w_correct = ci / (cc + ci) if (cc + ci) else 0.5        # P(gold incorrect | correct) == rand_corr
    w_incorrect = ii / (ic + ii) if (ic + ii) else 0.5      # P(gold incorrect | incorrect) == 1 - rand_rej
    rand_corr = w_correct
    rand_rej = ic / (ic + ii) if (ic + ii) else float("nan")  # P(gold correct | incorrect)
    return {"w_correct": w_correct, "w_incorrect": w_incorrect,
            "rand_corr": rand_corr, "rand_rej": rand_rej,
            "cells": {"cc": cc, "ci": ci, "ic": ic, "ii": ii}}


def soft_belief(evidence, w_correct, w_incorrect, priors=PRIORS) -> float:
    by_source: dict[str, list[float]] = defaultdict(list)
    for ev in evidence:
        src = (ev["source_api"] or "").lower()
        rand_s, syst_s = priors.get(src, _DEFAULT_PRIOR)
        base_s = min(1.0, syst_s + rand_s)  # INDRA single-read per-read wrong rate
        v = ev["verdict"]
        # per-read-rate space: syst folded into base, no additive syst, no clamp
        if v == "correct":
            w = min(w_correct, base_s)
        elif v == "incorrect":
            w = w_incorrect
        else:
            w = base_s
        by_source[src].append(min(max(w, 1e-9), 1.0))
    p_inc = 1.0
    for src, ws in by_source.items():
        n = len(ws)
        geomean = math.exp(sum(math.log(w) for w in ws) / n)
        p_inc *= geomean
    return max(0.0, min(1.0, 1.0 - p_inc))


def hard_belief(evidence) -> tuple[float, float]:
    ev = [{"source_api": e["source_api"], "included": e["verdict"] != "incorrect"} for e in evidence]
    res = compute_gated_belief(ev, priors=PRIORS)
    return res.belief, res.parametric_only


def metric_block(scores, labels) -> dict:
    b = c0.brier_murphy(scores, labels)
    return {
        "n": len(scores),
        "ece": ece(list(zip(scores, [bool(x) for x in labels]))),
        "brier": b["brier"], "resolution": b["resolution"], "reliability": b["reliability"],
        "auroc": c0.auroc(scores, labels), "auprc": c0.auprc(scores, labels),
        "mean_correct": float(np.mean([s for s, y in zip(scores, labels) if y])) if any(labels) else None,
        "mean_incorrect": float(np.mean([s for s, y in zip(scores, labels) if not y])) if not all(labels) else None,
    }
