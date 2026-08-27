"""Confusion-profile measurement reference for the G2 ship gate.

``scripts/calibration_ship_gate.py`` imports this module as ``c1`` and calls
``c1.fit_reader_profile`` when fitting each reader at line 585 and
``c1.metric_block`` when evaluating each belief arm at line 388.
``tests/test_soft_belief.py`` pins ``soft_belief`` as the reference that the
production ``statement_belief`` path must reproduce.

The confusion cells are authoritative for the reader profile. From them,
``profile_from_confusion`` derives sensitivity, false-positive rate, the
confirm/reject log-likelihood ratios, and the fit-set prior. ``soft_belief``
then mirrors production: it averages weights within each correlated source,
sums independent-source evidence with the prior log-odds, and applies a stable
sigmoid. Confirmations retain the explicit, separately fitted source-reliability
floor used by production, making the final value a hybrid calibration score
rather than a pure posterior.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import calibration_stage0 as c0  # noqa: E402  (reuse metrics)
from indra_belief.calibration_constants import profile_from_confusion  # noqa: E402
from indra_belief.metrics import ece  # noqa: E402
from indra_belief.noise_model import (  # noqa: E402
    _DEFAULT_PRIOR,
    RECALIBRATED_PRIORS,
)

PRIORS = RECALIBRATED_PRIORS


def fit_reader_profile(train_stmts) -> dict:
    """Derive a reader profile from unique evidence-grain verdict×gold pairs.

    Exact duplicate run rows (same statement/source/evidence hash) are one reader
    measurement, not independent calibration observations. Multi-curator gold is
    already collapsed by the caller's any-incorrect-wins join.
    """
    cc = ci = ic = ii = 0  # (verdict, ev_gold)
    seen: set[tuple] = set()
    for statement_index, st in enumerate(train_stmts):
        statement_key = st.get("statement_key", st.get("pa_hash", statement_index))
        for evidence_index, ev in enumerate(st["ev"]):
            evidence_hash = ev.get("evidence_hash")
            key = (
                statement_key,
                (ev.get("source_api") or "").lower(),
                evidence_hash if evidence_hash is not None else ("row", evidence_index),
            )
            if key in seen:
                continue
            seen.add(key)
            gc = ev["ev_gold_correct"]
            if ev["verdict"] == "correct":
                cc += gc
                ci += (not gc)
            elif ev["verdict"] == "incorrect":
                ic += gc
                ii += (not gc)
    return profile_from_confusion({"cc": cc, "ci": ci, "ic": ic, "ii": ii})


def soft_belief(evidence, log_lr_confirm, log_lr_reject, priors=PRIORS, prior_logodds=0.0) -> float:
    """Reference implementation mirroring production's source-aware log-odds model."""
    by_source: dict[str, list[float]] = defaultdict(list)
    for ev in evidence:
        src = (ev["source_api"] or "").lower()
        rand_s, syst_s = priors.get(src, _DEFAULT_PRIOR)
        base_s = min(1.0 - 1e-9, max(1e-9, syst_s + rand_s))
        v = ev["verdict"]
        source_logodds = math.log((1.0 - base_s) / base_s)
        if v == "correct":
            ell = max(log_lr_confirm, source_logodds)
        elif v == "incorrect":
            ell = log_lr_reject
        else:
            ell = source_logodds
        by_source[src].append(ell)
    total = prior_logodds
    for src, ls in by_source.items():
        total += sum(ls) / len(ls)
    if total >= 0:
        z = math.exp(-total)
        return 1.0 / (1.0 + z)
    z = math.exp(total)
    return z / (1.0 + z)


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
