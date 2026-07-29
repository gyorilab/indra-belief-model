"""Exact average-precision decomposition of the LLM-vs-paper deltas, banded by
an EXOGENOUS variable: how many evidence entries each statement carries.

Average precision decomposes with no residual: for a positive statement i,

    contribution(i) = precision at the cut-point that includes i's TIE GROUP
                      / n_positives

and the n_positives contributions sum to sklearn's ``average_precision_score``
exactly (the group-wise form is what sklearn computes: sum_n (R_n - R_{n-1})P_n
over the distinct score values).  Differencing an arm's per-statement
contributions against the paper's literal RF+promoter model and accumulating
them across bands therefore produces a waterfall whose right-hand endpoint IS
that arm's observed point delta-AP -- reached, not asserted.  That part is
banding-independent and is unchanged.

WHY THE BANDING VARIABLE CHANGED (this is the point of this script's revision).
--------------------------------------------------------------------------
The first version of this artifact banded by the REFERENCE ARM'S OWN out-of-fold
score and reported a striking shape: every reader ran below the RF through the
bands the RF ranked most confidently and took it all back in the RF's least
confident bands.  That shape is a conditioning artifact -- regression to the mean
on the reference's own estimation noise -- and the artifact now SHIPS THE PROOF
rather than dropping the claim quietly.  Same decomposition, same arm
(Gemma 4 26B), four banding variables, summarised as head (D1-D5) and tail
(D9+D10) net contribution in AP points:

    banded by the reference's own score      head -0.52   tail +0.95
    banded by Gemma 4 26B's own score        head +1.01   tail -0.97
    banded by the unfitted noisy-OR belief   head +0.35   tail +0.09
    banded by evidence count                 head +0.51   tail +0.09

Band by the arm instead of the reference and the sign flips exactly; band by
anything exogenous and the structure disappears.  The paper's own +prom/avglen
variant could never have detected this: it is Spearman 0.977 with the banding
variable, so it is pinned flat either way.  ``banding_sensitivity`` in the
payload carries this table for all four reader arms, under BOTH extreme
tie-break orderings, so the reversal can be checked rather than believed.

THE BANDING VARIABLE.  Evidence entries per statement -- an integer census of
the corpus fixed before any model ran.  It is not a score, it is not fitted to
any label, it carries no arm's estimation noise, and there is no "mirror" of it
to reverse.  It is also the noisy-OR's own saturating input
(belief = 1 - PROD_s (syst_s + rand_s^{n_s})), so the bands are the mechanism
rather than a proxy for it.  Verified three independent ways on all 1689
statements (assertion (d)): the execution map's multiplicity-weighted pair
count, the shared gold's ``evidence_review.corpus_evidence_entries``, and the
sum of the paper's own released per-source counts
(``paper_eligibility.historical_all_source_counts``) are equal statement for
statement.

The unfitted noisy-OR SimpleScorer belief was the other exogenous candidate and
is deliberately NOT used, for a reason this script verifies rather than asserts
(assertion (e)): the reader gate is purely subtractive, so no reader belief ever
exceeds the ungated noisy-OR -- it is each reader arm's own ceiling.  Banding by
it would condition on part of the reader's own score.  It is kept as a mirror
diagnostic, where that is exactly what makes it informative.

BANDS.  A power-of-two ladder on the evidence count -- 1, 2, 3-4, 5-8, 9-16,
17-32, 33+ -- rather than equal-count deciles.  Band membership is a pure
function of the count, so every statement carrying the same amount of evidence
lands in the same band and NO statement is assigned by a tie-break.  Equal-count
deciles could not say that: 328 statements carry exactly one evidence entry, so
a decile edge would have to split that block on ``sorted(stmt_hash)``, which is
arbitrary, and (unlike the single all-false tie the previous version pinned)
that block carries real average-precision mass.  The ladder is geometric because
the noisy-OR saturates exponentially in the count.

The join (which statements, in which order, with which labels) is exactly the
one used by scripts/compare_paper_literal_vs_llms.py; its module-level constants
and loader are imported rather than re-derived, so the two artifacts cannot
drift apart.  Ordering is sorted(stmt_hash).

Hard assertions (the script fails loudly rather than emitting a plausible file):
  (a) every arm's decomposition sum equals sklearn.metrics.average_precision_score
      to within 1e-12;
  (b) every arm's summed decomposition reproduces the SHIPPED point delta from
      paper_literal_vs_llms.json::point_metrics to within 1e-12, and the shipped
      ``delta`` field (which is the MEAN OF THE BOOTSTRAP DRAWS, not the observed
      difference) to within 1e-4;
  (c) the bands partition the panel with no tie-break: every band is non-empty,
      the count ranges are contiguous and strictly increasing, and the largest
      count in a band is strictly smaller than the smallest count in the next;
  (d) the banding variable agrees, statement for statement, with the shared
      gold's own evidence census AND with the sum of the paper's released
      per-source counts;
  (e) no reader belief exceeds the unfitted noisy-OR (the fact that disqualifies
      the noisy-OR as a banding variable);
  (f) every mirror-diagnostic summary is stable under the two extreme tie-break
      orderings to within 0.15 AP points, so the reversal it documents cannot be
      an artifact of how the decile edges split tied scores.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compute_paper_ap_decomposition.py \
      --literal data/results/indra_paper_literal_models_20260724/paper_literal_table6_and_oof.json \
      --comparison data/results/indra_paper_literal_models_20260724/paper_literal_vs_llms.json \
      --out-json data/results/indra_paper_literal_models_20260724/ap_decomposition_by_paper_band.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the head-to-head script's join contract verbatim: same gold file, same
# model bundles, same arm names, same headline/variant keys.
from compare_paper_literal_vs_llms import (  # noqa: E402
    BEST,
    GOLD,
    HEADLINE,
    MODELS_DIR,
    load_jsonl,
)

# Single home for the two corpus-side paths, so the banding variable and the
# rejected-candidate belief cannot drift from the control artifact that prices
# them.
from compute_non_reading_control import (  # noqa: E402
    EXECUTION_MAP,
    SIMPLE_SCORER_PREDICTIONS,
)

ROOT = Path(__file__).resolve().parents[1]

REFERENCE = "Paper literal RF+promoter"
PAPER_VARIANT = "Paper literal RF+prom/avglen"

# Fixed presentation order = descending point delta-AP (the terminus-fan order).
# Not sortable, not configurable.
ARMS = [
    ("Gemma 4 26B", "gemma_4_26b"),
    ("GLM-5", "glm_5"),
    ("Gemma 4 31B", "gemma_4_31b"),
    (PAPER_VARIANT, None),          # served from the literal run's own OOF table
    ("Gemma 4 E2B", "gemma_4_e2b"),
]

# The four reader arms, in the same fixed order, for the mirror diagnostic.
READER_ARMS = ["Gemma 4 26B", "GLM-5", "Gemma 4 31B", "Gemma 4 E2B"]

# The arm the viewer draws in the mirror strip: the headline arm.
MIRROR_DRAWN_ARM = "Gemma 4 26B"

# Power-of-two ladder on the evidence count.  ``None`` = open upper end.
# Contiguity and strict increase are asserted, not assumed (check (c)).
BAND_EDGES: list[tuple[int, int | None]] = [
    (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, None),
]
N_BANDS = len(BAND_EDGES)

# Mirror diagnostic: equal-count deciles of each candidate banding variable, so
# all four variables are summarised on identical head/tail definitions.  Deciles
# (not the drawn ladder) precisely because the point is to reproduce the shape
# the first version of this figure reported.
MIRROR_N_BANDS = 10
MIRROR_HEAD_BANDS = 5      # D1-D5
MIRROR_TAIL_BANDS = 2      # D9+D10

TOL_SKLEARN = 1e-12
TOL_POINT_DELTA = 1e-12
TOL_BOOTSTRAP_MEAN = 1e-4
# A decile edge on a tied variable has to split the tie block somewhere.  The
# mirror summaries are recomputed under both extreme orderings and must agree to
# this, or the diagnostic is not reporting structure (check (f)).
TOL_MIRROR_TIE_SPREAD_PTS = 0.15


def ap_contributions(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-statement average-precision contributions, indexed like ``y``.

    contribution(i) = 0 for a negative statement; for a positive statement it is
    (precision at the cut-point that admits i's whole tie group) / n_positives.
    The vector sums to ``average_precision_score(y, p)``.
    """
    n = len(y)
    order = np.argsort(-p, kind="mergesort")
    ys = y[order].astype(np.float64)
    ps = p[order]

    tp = np.cumsum(ys)
    seen = np.arange(1, n + 1, dtype=np.float64)
    precision = tp / seen

    # Last index of each tie group in the descending-score ordering.
    ends = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
    starts = np.r_[0, ends[:-1] + 1]
    group_precision = np.repeat(precision[ends], ends - starts + 1)

    n_pos = float(ys.sum())
    contrib_sorted = ys * group_precision / n_pos

    contrib = np.empty(n, dtype=np.float64)
    contrib[order] = contrib_sorted
    return contrib


def band_label(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo}+"
    return str(lo) if lo == hi else f"{lo}–{hi}"   # en dash


def decile_bands(values: np.ndarray, *, reversed_pre_sort: bool) -> np.ndarray:
    """Equal-count deciles of ``values``, descending (D1 = highest).

    A decile edge inside a block of tied values has to fall somewhere; which
    statements land either side is then decided by the pre-sort order, which is
    arbitrary.  ``reversed_pre_sort`` flips that order, giving the two extreme
    tie-breaks the caller compares (check (f)).
    """
    n = len(values)
    idx = np.arange(n)[::-1] if reversed_pre_sort else np.arange(n)
    order = idx[np.argsort(-values[idx], kind="mergesort")]
    sizes = [len(s) for s in np.array_split(np.arange(n), MIRROR_N_BANDS)]
    out = np.empty(n, dtype=int)
    start = 0
    for b, end in enumerate(np.cumsum(sizes)):
        out[order[start:end]] = b
        start = end
    return out


def head_tail_pts(diff: np.ndarray, bands: np.ndarray) -> tuple[float, float]:
    """Net contribution difference, in AP points, over the head and tail bands."""
    head = float(diff[bands < MIRROR_HEAD_BANDS].sum()) * 100.0
    tail = float(diff[bands >= MIRROR_N_BANDS - MIRROR_TAIL_BANDS].sum()) * 100.0
    return head, tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--literal", required=True,
                    help="paper_literal_table6_and_oof.json from "
                         "run_indra_paper_literal_models.py")
    ap.add_argument("--comparison", required=True,
                    help="paper_literal_vs_llms.json from "
                         "compare_paper_literal_vs_llms.py")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--manifest", default=None,
                    help="run manifest.json to record the output path + sha256 in")
    args = ap.parse_args()

    lit = json.load(open(args.literal))
    cmp_art = json.load(open(args.comparison))

    oof = {r["stmt_hash"]: r for r in lit["oof_predictions"][HEADLINE]}
    oof_best = {r["stmt_hash"]: r for r in lit["oof_predictions"][BEST]}

    gold = {}
    for r in load_jsonl(GOLD):
        h = int(r["paper_statement_hash"])
        gold[h] = {
            "sid": r["canonical_corpus"]["statement_id"],
            "matches_hash": str(r["canonical_corpus"]["matches_hash"]),
            "label": r["paper_replication_policy"]["released_paper_correct"],
            # The gold's own evidence census, and the paper's own released
            # per-source counts.  Both are cross-checked in (d).
            "corpus_evidence_entries": int(r["evidence_review"]["corpus_evidence_entries"]),
            "paper_source_counts": int(
                sum(r["paper_eligibility"]["historical_all_source_counts"])),
        }

    hashes = sorted(oof)                       # deterministic order, 1689 statements
    sids = [gold[h]["sid"] for h in hashes]
    y = np.array([oof[h]["y_true"] for h in hashes])
    assert all(oof[h]["y_true"] == gold[h]["label"] for h in hashes), "label mismatch"

    probs = {REFERENCE: np.array([oof[h]["prob_correct"] for h in hashes]),
             PAPER_VARIANT: np.array([oof_best[h]["prob_correct"] for h in hashes])}
    for name, arm in ARMS:
        if arm is None:
            continue
        pj = {r["statement_id"]: r["probability_correct"]
              for r in load_jsonl(f"{MODELS_DIR}/{arm}/all_source_predictions.jsonl")}
        probs[name] = np.array([pj[s] for s in sids])

    n = len(hashes)
    paper_score = probs[REFERENCE]

    # ---- the banding variable, and its three-way provenance check (d) -------
    # The execution map is keyed on the paper matches_hash, which the shared gold
    # carries alongside the statement UUID.
    unique_pairs: collections.Counter = collections.Counter()
    entries: collections.Counter = collections.Counter()
    for r in load_jsonl(ROOT / EXECUTION_MAP):
        h = str(r["paper_statement_hash"])
        unique_pairs[h] += 1
        entries[h] += int(r["pair_multiplicity"])

    mhash = [gold[h]["matches_hash"] for h in hashes]
    missing = [m for m in mhash if m not in entries]
    assert not missing, (
        f"{len(missing)} panel statements are absent from {EXECUTION_MAP}, "
        f"e.g. {missing[:3]}")

    n_evidence = np.array([entries[m] for m in mhash])
    n_unique_pairs = np.array([unique_pairs[m] for m in mhash])
    gold_census = np.array([gold[h]["corpus_evidence_entries"] for h in hashes])
    paper_counts = np.array([gold[h]["paper_source_counts"] for h in hashes])

    agree_gold = int((n_evidence == gold_census).sum())
    agree_paper = int((n_evidence == paper_counts).sum())
    assert agree_gold == n, (
        f"the execution map's evidence census matches the shared gold's "
        f"evidence_review.corpus_evidence_entries on only {agree_gold} of {n} "
        "statements; the banding variable is not the panel's evidence count")
    assert agree_paper == n, (
        f"the execution map's evidence census matches the sum of the paper's own "
        f"released per-source counts on only {agree_paper} of {n} statements; the "
        "banding variable is not the PAPER's evidence count")
    print(f"[check d] evidence census agrees with the shared gold and with the "
          f"paper's own released per-source counts on {n}/{n} statements")

    # ---- banding: a power-of-two ladder on the evidence count --------------
    band_of = np.full(n, -1, dtype=int)
    for b, (lo, hi) in enumerate(BAND_EDGES):
        m = (n_evidence >= lo) if hi is None else ((n_evidence >= lo) & (n_evidence <= hi))
        band_of[m] = b

    # (c) the bands partition the panel and NOTHING is assigned by a tie-break.
    unbanded = int((band_of < 0).sum())
    assert unbanded == 0, (
        f"{unbanded} statements fall outside the evidence-count ladder "
        f"{BAND_EDGES} (observed counts {int(n_evidence.min())}..{int(n_evidence.max())})")
    for b, (lo, hi) in enumerate(BAND_EDGES):
        assert int((band_of == b).sum()) > 0, f"band {band_label(lo, hi)} is empty"
        if b + 1 < N_BANDS:
            upper = int(n_evidence[band_of == b].max())
            lower = int(n_evidence[band_of == b + 1].min())
            assert upper < lower, (
                f"bands {b + 1} and {b + 2} overlap on evidence count "
                f"({upper} >= {lower}); a statement would be assigned by "
                "something other than its own evidence count")
            assert BAND_EDGES[b][1] is not None and BAND_EDGES[b + 1][0] == BAND_EDGES[b][1] + 1, (
                f"the ladder is not contiguous at band {b + 1}: {BAND_EDGES[b]} -> "
                f"{BAND_EDGES[b + 1]}")
    n_distinct_counts = int(len(np.unique(n_evidence)))
    print(f"[check c] {N_BANDS} bands partition {n} statements over "
          f"{n_distinct_counts} distinct evidence counts; every statement with the "
          "same count is in the same band, so no boundary breaks a tie")

    # Scope note: the reader panel is built from UNIQUE (statement, evidence)
    # pairs while the census above counts every entry.  Measured, not assumed.
    band_of_unique = np.full(n, -1, dtype=int)
    for b, (lo, hi) in enumerate(BAND_EDGES):
        m = (n_unique_pairs >= lo) if hi is None else (
            (n_unique_pairs >= lo) & (n_unique_pairs <= hi))
        band_of_unique[m] = b
    n_band_moves_under_unique_pairs = int((band_of_unique != band_of).sum())

    # ---- exact decomposition ------------------------------------------------
    contribs, aps = {}, {}
    for name, p in probs.items():
        c = ap_contributions(y, p)
        sk = float(average_precision_score(y, p))
        # (a) the decomposition is EXACT, not approximate.
        assert abs(float(c.sum()) - sk) <= TOL_SKLEARN, (
            f"{name}: decomposition sum {c.sum()!r} != sklearn average_precision "
            f"{sk!r} (|diff| {abs(float(c.sum()) - sk):.3e} > {TOL_SKLEARN:g})")
        contribs[name], aps[name] = c, sk
        print(f"[check a] {name:<30} decomposition {c.sum():.12f} vs sklearn AP "
              f"{sk:.12f}  |diff| {abs(float(c.sum()) - sk):.3e}")

    base_contrib = contribs[REFERENCE]

    bands = []
    for b, (lo, hi) in enumerate(BAND_EDGES):
        m = band_of == b
        counts = n_evidence[m]
        bands.append({
            "index": b + 1,
            "label": band_label(lo, hi),
            "evidence_low": lo,
            "evidence_high": hi,
            "n": int(m.sum()),
            "n_true": int(y[m].sum()),
            "n_false": int((1 - y[m]).sum()),
            "error_rate": float((1 - y[m]).sum() / m.sum()),
            "evidence_entries": int(counts.sum()),
            "evidence_min": int(counts.min()),
            "evidence_max": int(counts.max()),
            # The reference's own average-precision mass in this band: what the
            # per-band nets below are a difference AGAINST.
            "reference_contribution_pts": float(base_contrib[m].sum()) * 100.0,
        })

    arms_out = []
    for name, _arm in ARMS:
        diff = contribs[name] - base_contrib
        nets = np.array([diff[band_of == b].sum() for b in range(N_BANDS)])
        total = float(nets.sum())
        cumulative = np.cumsum(nets)

        # (b) the summed decomposition reproduces the SHIPPED delta.  The
        # artifact's `delta` field is the MEAN OF THE 10,000 BOOTSTRAP DRAWS
        # (compare_paper_literal_vs_llms.py::_ci), not the observed difference,
        # so the exact comparison is against the POINT metrics, and the
        # bootstrap mean is only checked for agreement to 1e-4.
        shipped = cmp_art["paired_delta_vs_paper_literal"][name]["pooled_average_precision"]
        point_delta = (cmp_art["point_metrics"][name]["pooled_average_precision"]
                       - cmp_art["point_metrics"][REFERENCE]["pooled_average_precision"])
        assert abs(total - point_delta) <= TOL_POINT_DELTA, (
            f"{name}: decomposition total {total!r} != shipped POINT delta "
            f"{point_delta!r} (|diff| {abs(total - point_delta):.3e})")
        assert abs(total - shipped["delta"]) <= TOL_BOOTSTRAP_MEAN, (
            f"{name}: decomposition total {total!r} disagrees with the shipped "
            f"bootstrap-mean delta {shipped['delta']!r} by "
            f"{abs(total - shipped['delta']):.3e} > {TOL_BOOTSTRAP_MEAN:g}")
        # Also confirm the point AP we recompute matches the shipped point AP.
        assert abs(aps[name] - cmp_art["point_metrics"][name]["pooled_average_precision"]) <= 1e-12

        print(f"[check b] {name:<30} total {total:+.12f} vs point delta "
              f"{point_delta:+.12f} |diff| {abs(total - point_delta):.3e}; vs shipped "
              f"bootstrap-mean {shipped['delta']:+.12f} |diff| "
              f"{abs(total - shipped['delta']):.3e}")

        lo_ci, hi_ci = float(shipped["ci95_low"]), float(shipped["ci95_high"])
        signs = np.sign(nets)
        arms_out.append({
            "name": name,
            "model_key": _arm,
            "average_precision": aps[name],
            "total_delta_ap": total,
            "total_pts": total * 100.0,
            "per_band_net_pts": [float(v) * 100.0 for v in nets],
            "cumulative_pts": [float(v) * 100.0 for v in cumulative],
            # How diffuse the arm's delta is across the exogenous bands.  A delta
            # concentrated in one band and a delta spread over all of them are
            # different findings and the figure has to be able to tell them apart.
            "n_bands_agreeing_with_total_sign": int((signs == np.sign(total)).sum()),
            "largest_band_share_of_total": float(
                np.max(np.abs(nets)) / abs(total)) if total != 0 else 0.0,
            "largest_band_index": int(np.argmax(np.abs(nets))) + 1,
            "ci95_low_pts": lo_ci * 100.0,
            "ci95_high_pts": hi_ci * 100.0,
            "p_arm_greater": float(shipped["p_arm_greater"]),
            "clears_zero": bool(lo_ci > 0 or hi_ci < 0),
            "shipped_bootstrap_mean_delta_ap": float(shipped["delta"]),
        })

    # ---- (e) why the noisy-OR is a mirror and not the banding variable ------
    simple_by_sid = {r["statement_id"]: float(r["probability_correct"])
                     for r in load_jsonl(ROOT / SIMPLE_SCORER_PREDICTIONS)}
    assert set(simple_by_sid) >= set(sids), (
        f"{SIMPLE_SCORER_PREDICTIONS} does not cover the panel")
    simple = np.array([simple_by_sid[s] for s in sids])
    n_exceeding = 0
    for name in READER_ARMS:
        n_exceeding += int((probs[name] > simple).sum())
    assert n_exceeding == 0, (
        f"{n_exceeding} of {n * len(READER_ARMS)} reader beliefs EXCEED the "
        "unfitted noisy-OR; the premise that disqualifies it as a banding "
        "variable (it is each reader's own ungated ceiling) no longer holds")
    print(f"[check e] 0 of {n * len(READER_ARMS)} reader beliefs exceed the unfitted "
          "noisy-OR, so banding by it would condition on part of the reader's own score")

    # ---- the mirror diagnostic ---------------------------------------------
    # Four candidate banding variables, same decile machinery, same head/tail
    # definitions, both extreme tie-break orderings.
    mirror_specs = [
        ("reference_own_score", REFERENCE, "endogenous", paper_score,
         "the reference arm's own out-of-fold score — the banding this figure "
         "used to use"),
        ("drawn_arm_own_score", MIRROR_DRAWN_ARM, "endogenous", probs[MIRROR_DRAWN_ARM],
         "the compared arm's own score — the mirror of the row above"),
        ("unfitted_noisy_or", "noisy-OR SimpleScorer (direct)", "exogenous", simple,
         "the unfitted noisy-OR belief — no labels, no fit, but it is every "
         "reader's own ceiling"),
        ("evidence_count", "evidence entries per statement", "exogenous",
         n_evidence.astype(np.float64),
         "the corpus census this figure bands on, cut into deciles so it lands on "
         "the same head/tail definition as the rows above"),
    ]
    mirror_variants = []
    max_tie_spread = 0.0
    for key, display, kind, values, note in mirror_specs:
        canonical = decile_bands(values, reversed_pre_sort=False)
        alternate = decile_bands(values, reversed_pre_sort=True)
        arm_rows = []
        for name in READER_ARMS:
            diff = contribs[name] - base_contrib
            head, tail = head_tail_pts(diff, canonical)
            head_alt, tail_alt = head_tail_pts(diff, alternate)
            spread = max(abs(head - head_alt), abs(tail - tail_alt))
            max_tie_spread = max(max_tie_spread, spread)
            # (f) the summary must not be an artifact of how a decile edge splits
            # a block of tied scores.
            assert spread <= TOL_MIRROR_TIE_SPREAD_PTS, (
                f"mirror banding {key!r} / arm {name!r}: the head/tail summary moves "
                f"{spread:.3f} AP points between the two extreme tie-break orderings "
                f"(> {TOL_MIRROR_TIE_SPREAD_PTS} pts); it is reporting the tie-break, "
                "not the banding variable")
            arm_rows.append({
                "arm": name,
                "head_pts": head,
                "tail_pts": tail,
                # head - tail: one signed number for "which end of the banding
                # variable does this arm appear to win at".
                "tilt_pts": head - tail,
                "head_pts_reversed_tie_break": head_alt,
                "tail_pts_reversed_tie_break": tail_alt,
                "max_tie_break_spread_pts": spread,
            })
        mirror_variants.append({
            "key": key,
            "banding_arm": display,
            "kind": kind,
            "note": note,
            "arms": arm_rows,
        })

    tilt = {v["key"]: {a["arm"]: a["tilt_pts"] for a in v["arms"]}
            for v in mirror_variants}
    reference_tilts = tilt["reference_own_score"]
    mirror_tilts = tilt["drawn_arm_own_score"]
    reverses = [name for name in READER_ARMS
                if np.sign(reference_tilts[name]) != np.sign(mirror_tilts[name])]
    exogenous_max_abs_tilt = max(
        abs(tilt[k][name]) for k in ("unfitted_noisy_or", "evidence_count")
        for name in READER_ARMS)
    endogenous_max_abs_tilt = max(
        abs(tilt[k][name]) for k in ("reference_own_score", "drawn_arm_own_score")
        for name in READER_ARMS)

    print(f"[check f] worst mirror tie-break spread {max_tie_spread:.3f} pts "
          f"(tolerance {TOL_MIRROR_TIE_SPREAD_PTS})")

    payload = {
        "artifact_kind": "paper_ap_decomposition_by_evidence_count",
        "schema_version": 2,
        "metric": "pooled_average_precision",
        "unit": "AP points (1 pt = 0.01 AP)",
        "reference_arm": REFERENCE,
        "reference_average_precision": aps[REFERENCE],
        "n_statements": n,
        "n_true": int(y.sum()),
        "n_false": int((1 - y).sum()),
        "banding": {
            "kind": "power_of_two_ladder_on_evidence_count",
            "variable": "evidence entries per statement",
            "variable_is_exogenous": True,
            "why_exogenous": "An integer census of the corpus, fixed before any "
                             "model ran. Not a score, not fitted to any label, "
                             "carrying no arm's estimation noise — so it has no "
                             "mirror to reverse under. It is also the noisy-OR's "
                             "own saturating input, "
                             "belief = 1 - PROD_s (syst_s + rand_s^{n_s}).",
            "why_not_the_noisy_or": "The unfitted noisy-OR belief is the other "
                                    "exogenous candidate and is rejected: the "
                                    "reader gate is purely subtractive, so no "
                                    "reader belief ever exceeds it (verified: 0 of "
                                    f"{n * len(READER_ARMS)}). It is each reader "
                                    "arm's own ceiling, so banding by it would "
                                    "condition on part of the reader's own score. "
                                    "It is kept as a mirror diagnostic instead.",
            "n_bands": N_BANDS,
            "edges": [[lo, hi] for lo, hi in BAND_EDGES],
            "direction": "left = 1 evidence entry (the noisy-OR at its weakest); "
                         "right = 33 or more (saturated)",
            "ordering": "sorted(stmt_hash); band membership is a pure function of "
                        "the evidence count, so no statement is assigned by a "
                        "tie-break",
            "n_distinct_evidence_counts": n_distinct_counts,
            "evidence_min": int(n_evidence.min()),
            "evidence_max": int(n_evidence.max()),
            "verified_against": {
                "execution_map_multiplicity_weighted_pairs": EXECUTION_MAP,
                "shared_gold_field": "evidence_review.corpus_evidence_entries",
                "paper_released_field": "sum(paper_eligibility."
                                        "historical_all_source_counts)",
                "n_statements_agreeing": n,
            },
            "unique_pair_scope": {
                "note": "The reader panel is built from UNIQUE (statement, "
                        "evidence) pairs; this census counts every entry, which is "
                        "what the paper's own released per-source counts do.",
                "n_unique_pairs": int(n_unique_pairs.sum()),
                "n_evidence_entries": int(n_evidence.sum()),
                "n_statements_changing_band_under_unique_pairs":
                    n_band_moves_under_unique_pairs,
            },
        },
        "bands": bands,
        "band_true_counts": [b["n_true"] for b in bands],
        "band_false_counts": [b["n_false"] for b in bands],
        "arms": arms_out,
        "banding_sensitivity": {
            "question": "Does the shape this decomposition shows depend on WHICH "
                        "variable the bands are cut on?",
            "finding": "Yes, when the banding variable is one of the two scores "
                       "being differenced. Banded by the reference's own score "
                       "every reader appears to lose at the head and win at the "
                       "tail; banded by the compared arm's own score the same "
                       "decomposition of the same arm says the opposite. Both are "
                       "regression to the mean on the banding score's own noise. "
                       "Banded on either exogenous variable the tilt collapses.",
            "summary_metric": "tilt = (net AP points over D1-D5) - (net over "
                              "D9+D10), in AP points, from equal-count deciles of "
                              "each candidate banding variable",
            "n_bands": MIRROR_N_BANDS,
            "head_bands": MIRROR_HEAD_BANDS,
            "tail_bands": MIRROR_TAIL_BANDS,
            "drawn_arm": MIRROR_DRAWN_ARM,
            "variants": mirror_variants,
            "arms_whose_tilt_sign_reverses_under_mirroring": reverses,
            "n_arms_compared": len(READER_ARMS),
            "max_abs_tilt_endogenous_banding_pts": endogenous_max_abs_tilt,
            "max_abs_tilt_exogenous_banding_pts": exogenous_max_abs_tilt,
            "max_tie_break_spread_pts": max_tie_spread,
            "tie_break_spread_tolerance_pts": TOL_MIRROR_TIE_SPREAD_PTS,
            "note": "Reported rather than dropped. The reversal is the reason the "
                    "drawn bands are an exogenous count; volunteering it is the "
                    "evidence for that choice.",
        },
        "checks": {
            "decomposition_vs_sklearn_average_precision_tol": TOL_SKLEARN,
            "decomposition_vs_shipped_point_delta_tol": TOL_POINT_DELTA,
            "decomposition_vs_shipped_bootstrap_mean_delta_tol": TOL_BOOTSTRAP_MEAN,
            "bands_partition_the_panel": True,
            "band_membership_is_a_function_of_the_banding_variable_alone": True,
            "n_statements_assigned_by_a_tie_break": 0,
            "evidence_census_agrees_with_shared_gold": agree_gold,
            "evidence_census_agrees_with_paper_released_counts": agree_paper,
            "n_reader_beliefs_exceeding_the_unfitted_noisy_or": n_exceeding,
            "n_reader_belief_comparisons": n * len(READER_ARMS),
            "mirror_max_tie_break_spread_pts": max_tie_spread,
            "mirror_tie_break_spread_tolerance_pts": TOL_MIRROR_TIE_SPREAD_PTS,
            "note": "Assertions are enforced in code; a violation fails the build "
                    "rather than being reported here as False.",
        },
        "provenance": {
            "literal": str(Path(args.literal)),
            "comparison": str(Path(args.comparison)),
            "gold": GOLD,
            "execution_map": EXECUTION_MAP,
            "unfitted_noisy_or_predictions": SIMPLE_SCORER_PREDICTIONS,
            "llm_bundles": f"{MODELS_DIR}/{{arm}}/all_source_predictions.jsonl",
            "generated_by": "scripts/compute_paper_ap_decomposition.py",
            "output_filename_note": "The file name ap_decomposition_by_paper_band"
                                    ".json is a frozen join key (viewer loader + "
                                    "run manifest) and is kept across the banding "
                                    "change; artifact_kind and banding.kind carry "
                                    "the banding variable.",
            "bootstrap": {
                "n_bootstrap": cmp_art.get("n_bootstrap"),
                "seed": cmp_art.get("seed"),
                "note": "ci95_low_pts/ci95_high_pts/p_arm_greater are copied verbatim "
                        "from paper_literal_vs_llms.json; total_pts is the exact "
                        "decomposition sum (the observed point delta), NOT the "
                        "bootstrap-mean `delta` field.",
            },
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    if args.manifest:
        mpath = Path(args.manifest)
        man = json.loads(mpath.read_text())
        man.setdefault("outputs", {})["ap_decomposition"] = out.name
        man.setdefault("output_sha256", {})[out.name] = sha
        mpath.write_text(json.dumps(man, indent=2) + "\n")
        print(f"recorded sha256 in {mpath}")

    print(f"wrote {out} ({out.stat().st_size} bytes)\nsha256 {sha}\n")
    print("evidence   " + " ".join(f"{b['label']:>7}" for b in bands))
    print("n          " + " ".join(f"{b['n']:>7}" for b in bands))
    print("n_true     " + " ".join(f"{b['n_true']:>7}" for b in bands))
    print("n_false    " + " ".join(f"{b['n_false']:>7}" for b in bands))
    for a in arms_out:
        print(f"{a['name'][:10]:<10} " +
              " ".join(f"{v:+7.2f}" for v in a["cumulative_pts"]) +
              f"   total {a['total_pts']:+.4f} pts  "
              f"[{a['ci95_low_pts']:+.2f}, {a['ci95_high_pts']:+.2f}] "
              f"{'FILLED' if a['clears_zero'] else 'HOLLOW'}  "
              f"{a['n_bands_agreeing_with_total_sign']}/{N_BANDS} bands agree")
    print()
    print("mirror (tilt = D1-D5 net minus D9+D10 net, AP points)")
    print(f"{'banded by':<34}" + " ".join(f"{a[:11]:>12}" for a in READER_ARMS))
    for v in mirror_variants:
        print(f"{v['banding_arm'][:33]:<34}" +
              " ".join(f"{a['tilt_pts']:>+12.2f}" for a in v["arms"]) +
              f"   [{v['kind']}]")
    print(f"tilt sign reverses under mirroring for {len(reverses)} of "
          f"{len(READER_ARMS)} reader arms: {', '.join(reverses) or 'none'}")


if __name__ == "__main__":
    main()
