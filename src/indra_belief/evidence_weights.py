"""Belief from PER-EVIDENCE weights of evidence, rather than per-verdict constants.

WHAT THIS IS FOR
----------------
``noise_model._soft_gated_belief`` aggregates in log-odds:

    logit(belief) = prior_logodds + Σ_sources ( mean of that source's weights )

and computes each read's weight ``weight_of_evidence`` from its VERDICT — two constants, one per
verdict, taken from the reader's confusion matrix:

    correct    weight = max(log_lr_confirm, source_logodds)
    incorrect  weight = log_lr_reject
    unscored   weight = source_logodds

That discards the margin. A verdict given at 0.99 and one given at 0.51 carry
identical weight. The direct verdict probe measures that margin and already
returns it in exactly this currency — ``CalibratedProbeReading.weight_of_evidence``, the
calibrated log-odds relative to the fit base rate — and nothing consumes it.

(The published source spells this weight ``ell`` — the letter used for it in
that formula. It is the same quantity under a name that says what it is.)

``noise_model`` is a PUBLISHED implementation: its bytes are recorded under
``implementation.notes.implementation_components.noise_model`` in the four
``data/comparison/models/*/manifest.json`` bundles, and the module still matches
that record. Threading a per-row weight through it would break that
correspondence, so this module is the generalization instead: the same
aggregation, taking a weight per evidence rather than deriving it from a
verdict. (That record is provenance, not a lock. Nothing in this tree enforces
it -- the function that computed it, ``comparison/llm.py::_implementation_digest``,
was retired with the paid benchmark harness. What actually holds ``noise_model``
still is behavioural: tests/test_noise_model.py, tests/test_soft_belief.py,
tests/test_statement_belief_truth_table.py, and the exact-reduction property
below.)

WHY THIS IS NOT A SECOND BELIEF MODEL
-------------------------------------
Duplicating an aggregation is how two implementations drift. The guard is an
exact-reduction property, asserted in tests/test_evidence_weights.py: feed this
function VERDICT-DERIVED weights and it must reproduce
``compute_gated_belief(..., soft_weights=True)`` to the bit. It is the same
formula with the weight lifted into an argument, and the test is what keeps that
true rather than the comment claiming it.

STATUS: NOT WIRED INTO PRODUCTION, DELIBERATELY
-----------------------------------------------
Carrying the probe into statement belief was MEASURED as a loss once already:
per sentence +0.020 AUROC (95% CI [+0.007, +0.033]), but per statement +0.004
with the interval spanning zero and ECE worsening 0.0199 -> 0.0388. That test
replaced the per-evidence SCORE; it never supplied a measured weight
additively, which is the
form the aggregation actually consumes. This module exists so that the
better-posed version can be measured — not because the question is settled.

Until a measurement says otherwise, production keeps the verdict path.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .noise_model import _DEFAULT_PRIOR, _sigmoid

__all__ = [
    "source_logodds_for",
    "verdict_weight",
    "belief_from_weights",
]


def source_logodds_for(
    source_api: str | None, priors: Mapping[str, tuple[float, float]]
) -> float:
    """The source's reliability in log-odds — INDRA's prior, not a likelihood ratio.

    Mirrors ``_soft_gated_belief`` exactly, including the clamp, so the two
    cannot disagree about what a source is worth.
    """
    src = (source_api or "").lower()
    rand_s, syst_s = priors.get(src, _DEFAULT_PRIOR)
    base_s = min(1.0 - 1e-9, max(1e-9, syst_s + rand_s))
    return math.log((1.0 - base_s) / base_s)


def verdict_weight(
    verdict: str | None,
    source_logodds: float,
    *,
    log_lr_confirm: float,
    log_lr_reject: float,
) -> float:
    """The weight the published model gives a read. Extracted so it is testable.

    This is a transcription of the branch inside ``_soft_gated_belief``; the
    reduction test pins it against the real thing.
    """
    if verdict == "correct":
        return max(log_lr_confirm, source_logodds)
    if verdict == "incorrect":
        return log_lr_reject
    return source_logodds


def probe_weight(weight: float, source_logodds: float) -> float:
    """A continuous reading's weight, with the source floor preserved.

    The floor exists so a generic reader cannot drag a well-curated source below
    its own track record. That intent is verdict-shaped in the published model
    (applied to confirmations only), and the faithful continuous analogue is to
    apply it when the reading FAVOURS correct — ``weight > 0`` — and to let a
    disconfirming reading stand on its own, exactly as a rejection does.
    """
    return max(weight, source_logodds) if weight > 0.0 else weight


def belief_from_weights(
    evidence: Sequence[Mapping],
    priors: Mapping[str, tuple[float, float]],
    *,
    prior_logodds: float = 0.0,
) -> float:
    """Aggregate per-evidence weights into a belief.

    Each evidence mapping must carry ``source_api`` and ``weight_of_evidence`` (its weight of
    evidence, in log-odds, already floored by the caller if that is wanted).

    Within a source the reads are correlated — same reader — so their weights are
    AVERAGED and the source counts once; across sources they SUM. That is the
    published model's rule and this does not relitigate it.
    """
    if not evidence:
        raise ValueError("belief_from_weights requires at least one evidence row")
    by_source: dict[str, list[float]] = {}
    for i, ev in enumerate(evidence):
        src_raw = ev.get("source_api")
        if src_raw is None:
            raise ValueError(
                f"Evidence at index {i} is missing required 'source_api' key: {ev!r}"
            )
        weight = ev.get("weight_of_evidence")
        if weight is None or not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError(
                f"Evidence at index {i} carries no numeric 'weight_of_evidence': {ev!r}. "
                "An absent weight is not a zero weight — a zero says 'this read "
                "is exactly uninformative', which is a measurement, not a gap."
            )
        by_source.setdefault(src_raw.lower(), []).append(float(weight))

    total = prior_logodds
    for _src, weights in sorted(by_source.items()):
        total += sum(weights) / len(weights)
    return _sigmoid(total)
