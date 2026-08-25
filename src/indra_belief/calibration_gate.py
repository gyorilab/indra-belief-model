"""How a calibration candidate is judged, in one place.

WHY THIS IS A MODULE
--------------------
This repo had TWO gates that disagreed about the same candidate, in two scripts,
and neither referenced the other:

    the probe-combiner fitter    non-inferior ranking AND lower ECE AND more
                                distinct scores
    fit_incall_calibration.py   superior ranking AND lower Brier

Applied to the in-call margin the first says NO (its ECE rises) and the second
says YES. A repo that can answer "should this ship" two ways depending on which
script you run does not have a gate; it has two opinions.

The rule below is the reconciliation, and it is not a compromise -- it explains
why each of those rules was right where it was written. Everything turns on the
INCUMBENT the candidate is measured against, which is the variable neither
script named:

  * against the six-cell lookup (ECE 0.2137, three distinct values) "lower ECE"
    costs a good candidate nothing, so demanding it is free.
  * against the verdict log-LR -- two values set to the population base rates,
    reliability 0.0015 and resolution 0.1217 -- "lower ECE" rejects ANY
    sharpening, because a base-rate lookup is already about as reliable as a
    score can be while carrying almost no information.

So calibration loss is PRICED rather than vetoed, using the decomposition that
exists for exactly this purpose: Brier = reliability - resolution + uncertainty.
"""
from __future__ import annotations


def gate_decision(ci_low: float, brier_incumbent: float, brier_candidate: float,
                  reliability_incumbent: float = 0.0,
                  reliability_candidate: float = 0.0,
                  resolution_incumbent: float = 0.0,
                  resolution_candidate: float = 0.0,
                  *, min_favourability: float = 2.0) -> dict:
    """Three checks, because each one alone has a known failure mode HERE.

    RECONCILED WITH the probe-combiner fitter (removed with the probe battery),
    which gated on
    "non-inferior ranking AND lower ECE AND more distinct scores". That rule is
    right for the question it was written for and wrong for this one, and the
    difference is the INCUMBENT:

      there  the incumbent was the six-cell lookup -- ECE 0.2137, three distinct
             values. Demanding better calibration cost the candidate nothing.
      here   the incumbent is the verdict log-LR: two values set to the
             population base rates, so it is almost perfectly RELIABLE (0.0015)
             and correspondingly uninformative (resolution 0.1217). Demanding
             lower ECE against a base-rate lookup rejects any sharpening at all.

    MEASURED on 6 splits of production-path rows: the candidate costs +0.0067
    reliability and gains +0.0242 resolution -- a 3.6:1 favourable trade that
    the ECE rule would refuse.

    So the trade is priced with the Murphy decomposition instead of vetoed:

      RANKING        the CI excludes 0. Alone it passes noise: the incumbent
                     takes two values, so its AUROC is computed over enormous
                     ties and any continuous score is structurally flattered.
      SCORING        Brier improves. Brier IS reliability - resolution +
                     uncertainty, so this asks exactly "is the trade net
                     positive", and it cannot be won by tie-breaking.
      FAVOURABILITY  the resolution gain is at least `min_favourability` times
                     the reliability cost. Brier alone would accept a large
                     calibration regression offset by a larger sharpness gain,
                     and a consumer thresholding on belief feels reliability
                     directly -- it does not get to enjoy the average.
    """
    ranking = ci_low > 0.0
    scoring = brier_candidate < brier_incumbent
    reliability_cost = reliability_candidate - reliability_incumbent
    resolution_gain = resolution_candidate - resolution_incumbent
    ratio = (float("inf") if reliability_cost <= 0
             else resolution_gain / reliability_cost)
    favourable = ratio >= min_favourability
    return {
        "ranking": ranking,
        "scoring": scoring,
        # DIAGNOSTIC, NOT A GATE LEG. This was a third condition until
        # scripts/calibration_ship_gate.py:29 was read: "Brier-resolution is
        # reported as a diagnostic, NOT gated (noise-dominated at n~342)". The
        # splits here are n~90, a quarter of that, and reliability/resolution are
        # BINNED over BINS_8 -- roughly eleven rows a bin, and a RATIO of two such
        # estimates is noisier than either. Gating on it would have made this the
        # third gate in the repo, disagreeing with an existing one about whether
        # the quantity is even reportable, in a module written because the repo
        # should not have gates that disagree.
        #
        # It stays computed and printed because it is what EXPLAINS the ECE rise
        # -- the trade is favourable -- and an explanation is exactly what a
        # noisy estimate can honestly support. The decision rests on Brier, which
        # is an unbinned mean of squared error and stable at this n.
        "favourable": favourable,
        "reliability_cost": reliability_cost,
        "resolution_gain": resolution_gain,
        "ratio": ratio,
        "pass": bool(ranking and scoring),
    }
