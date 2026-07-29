"""Statement-grain ERROR-class F1: the margin /paper already computes but never states.

The /paper page names pooled average precision "the verdict metric".  On this
panel that is the WEAKEST number we have.  The panel is 73.2% positive (1237
correct, 452 errors) and the reference AP is already .9412, so average precision
is measured against a ceiling almost all of which the base rate supplies for
free: the paper's own RF beats every reader arm's AP delta into the noise
(+0.0098, t 2.21, fails a max-t band over the four arms).

The decision the page is actually about is the other one.  A curator does not
consume a ranking; they consume a queue of statements someone claims are WRONG.
Scoring that job means making ERROR the positive class, and on that metric the
same arms, the same panel, the same labels and the same folds give +0.1416 at
t 8.49 — a margin that survives the same simultaneous correction AP fails.

    positive class     ERROR (the paper's released label says the statement is wrong)
    decision rule      flag a statement as an ERROR iff belief < tau
    threshold rule     tau = the arm's OWN full-panel best-error-F1 cut, chosen
                       over its own distinct scores; ties broken to the SMALLEST
                       such cut so the rule is a function of the scores alone

THE THRESHOLD IS AN ORACLE AND IT IS NOT OURS TO BENEFIT FROM.  Every tau here
is fitted and evaluated on the same 1689 statements.  That is disclosed in
``oracle_disclosure`` rather than hidden, and the direction of the advantage is
against us: the paper's RF emits 1546 distinct scores, so the oracle gets to
optimise it over 1546 candidate cuts, where the three reader arms that win have
475-498 and Gemma 4 E2B has 420.  It buys those three arms nothing at all —
their best cut lands on the block of statements whose evidence the reader
rejected outright, which is not a threshold anyone chose (see
``modal_threshold_note``).  The side handed the finest-grained oracle still
loses.

THE SECOND CUT IS AN ORACLE TOO.  ``matched_recall`` scores the same arms at a
different cut — the cheapest one catching at least 60% of the panel's errors —
and that cut is fitted on this panel by a different rule, so it is a second
oracle and is disclosed as one in ``matched_recall_rule``.  Its delta scores the
reference AT EACH ARM'S OWN ACHIEVED ERROR RECALL, because the arms' score grids
are coarse and their achieved recalls at a 60% target spread across 26 points:
subtracting the reference's F1 at the REFERENCE's own 60% cut would compare two
different recalls and flatter whichever arm overshot furthest.  Both deltas are
emitted, under names that say which is which.

NOTHING HERE IS NEW DATA.  These are the same prediction vectors the AP
head-to-head and the review queue already score.  The error-F1 numbers are
already latent in ``statement_review_queue.json`` — its ``operating_point``
block publishes the precision and recall they are computed from, and has since
that artifact shipped — but they are rendered only as queue sizes and never
named.  ``reconciliation`` recomputes them from that artifact's own published
counts through the same ``confusion_pr`` and records the residual, so the two
derivations are pinned to each other rather than merely believed to agree.  They
are two RULES, not two independent measurements: on the three winning arms both
rules select the same statements, so their agreement there is a cross-check of
two code paths over one flag set (see ``reconciliation.note``).

WHAT THIS IS NOT.  F1 at a chosen cut is a DECISION metric at ONE operating
point.  It does not supersede AP or AUROC and it does not retract them: it
answers a different question, and both live on the page.  Every AP number and
every AP qualification stays exactly where it was.

Usage:
    uv run python scripts/compute_statement_error_f1.py            # write
    uv run python scripts/compute_statement_error_f1.py --check    # gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from itertools import chain, repeat
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The join contract, the panel, the resampling design, the seed and the
# resample count come from the AP head-to-head verbatim.  Importing them is what
# makes "the same panel, scored differently" a property of one implementation.
from compare_paper_literal_vs_llms import (  # noqa: E402
    GOLD,
    HEADLINE,
    MODELS_DIR,
    N_BOOT,
    SEED,
    load_jsonl,
)

# The multiplicity family, the band and the fold-stratified draw are the AP
# robustness script's, unchanged.  Same four arms, same alpha, same max-t.
from compute_paper_robustness import (  # noqa: E402
    FAMILY_ALPHA,
    POINTWISE_NORMAL_Z,
    READER_ARMS,
    REFERENCE_ID,
    REFERENCE_LABEL,
    RUN_PLAN,
    excludes_zero,
    fold_stratified_indices,
    sha256_of,
    simultaneous_band,
)

# P/R/F1 has ONE definition in this repo and it is not re-implemented here.
from indra_belief.metrics import confusion_pr  # noqa: E402

RUN_DIR = "data/results/indra_paper_literal_models_20260724"
OUT_DEFAULT = f"{RUN_DIR}/statement_error_f1.json"
MANIFEST_DEFAULT = f"{RUN_DIR}/manifest.json"
PAPER_LITERAL = f"{RUN_DIR}/paper_literal_table6_and_oof.json"
# The second derivation this artifact reconciles against.  Its `operating_point`
# block already publishes the precision and recall these F1s are made of; it just
# never names the F1.  It reaches its cut by a different RULE, not from different
# data — see `reconciliation.note` for how far that is and is not independence.
REVIEW_QUEUE = f"{RUN_DIR}/statement_review_queue.json"

MANIFEST_OUTPUT_KEY = "statement_error_f1"

# On-screen names, keyed by arm key.  DECOUPLED from `label`, which is the
# FROZEN `point_metrics` join key into the already-shipped AP artifact, and from
# `review_queue_model_key`, which is the join key into the review queue.  A
# display string has become a join key on this page seven times; keeping the
# three in separate fields is the only reason it cannot happen again here.
DISPLAY = {
    REFERENCE_ID: "RF 2k-d13 + Type/#PMIDs/promoter",
    "gemma-4-26b": "Gemma 4 26B gate",
    "glm-5": "GLM-5 gate",
    "gemma-4-31b": "Gemma 4 31B gate",
    "gemma-4-e2b": "Gemma 4 E2B gate",
    "indra-cogex-hybrid": "INDRA CoGEx hybrid",
}

# The review queue's own arm keys, so the reconciliation joins on a stable key
# rather than on a rendered name.
REVIEW_QUEUE_KEY = {
    REFERENCE_ID: "paper_literal_rf_promoter",
    "gemma-4-26b": "gemma_4_26b",
    "glm-5": "glm_5",
    "gemma-4-31b": "gemma_4_31b",
    "gemma-4-e2b": "gemma_4_e2b",
}
REVIEW_QUEUE_TARGET_RECALL = 0.7
# The review queue reaches its operating point by a DIFFERENT rule (smallest cut
# hitting a target recall, `belief <= tau`); this one maximises error-F1 with
# `belief < tau`.  Two rules, two derivations, one quantity — they agree this
# closely or the artifact does not ship.
RECONCILIATION_TOLERANCE = 0.008

# Carried for completeness, NOT in the confirmatory family: the CoGEx bundle's
# own manifest declares `analysis_role: descriptive_nonconfirmatory`, and the
# frozen run plan stages the four reader arms and nothing else.
CONTEXT_ARMS = [
    {"id": "indra-cogex-hybrid", "label": "INDRA CoGEx hybrid", "bundle": "indra_cogex_hybrid"},
]

# The second operating point, so the headline is not the only cut on record: the
# cheapest cut that catches at least this share of the panel's errors.  The cut
# is per arm, so the recall each arm ACHIEVES overshoots the target by its own
# grid's granularity; the reference is therefore re-cut at each arm's achieved
# recall rather than quoted once at its own (see `matched_recall_block`).
MATCHED_ERROR_RECALL = 0.60

# ---------------------------------------------------------------------------
# The numbers this script exists to reproduce.  Measured and verified before the
# script was written; a mismatch is a defect HERE and must fail rather than
# quietly ship a different value.
# ---------------------------------------------------------------------------
TOL_POINT = 1e-6
TOL_INTERVAL = 5e-3
EXPECTED_REFERENCE_ERROR_F1 = 0.6439
EXPECTED_MAX_T_CRITICAL = 2.3684
EXPECTED_ARMS = {
    "glm-5": {"error_f1": 0.7855, "delta": +0.1416, "ci95_low": +0.1090,
              "ci95_high": +0.1744, "t_statistic": 8.49, "excludes_zero_simultaneous": True},
    "gemma-4-26b": {"error_f1": 0.7746, "delta": +0.1307, "ci95_low": +0.0955,
                    "ci95_high": +0.1655, "excludes_zero_simultaneous": True},
    "gemma-4-31b": {"error_f1": 0.7699, "delta": +0.1260, "ci95_low": +0.0912,
                    "ci95_high": +0.1609, "excludes_zero_simultaneous": True},
    "gemma-4-e2b": {"error_f1": 0.6345, "delta": -0.0094, "ci95_low": -0.0381,
                    "ci95_high": +0.0194, "excludes_zero_simultaneous": False},
}
# Rounded reference values from the review queue's `operating_point`, restated so
# the reconciliation block is checked against a written-down number and not only
# against whatever that artifact happens to contain today.
EXPECTED_REVIEW_QUEUE_ERROR_F1 = {
    "glm-5": 0.785, "gemma-4-26b": 0.775, "gemma-4-31b": 0.770, REFERENCE_ID: 0.636,
}
# The SECOND operating point's deltas, pinned so the recall-matching cannot
# silently regress.  These are the arm's F1 at its own 60%-target cut MINUS the
# reference's F1 at THAT ARM'S ACHIEVED RECALL.  Quoting the reference at its own
# 60% cut instead gives +0.0350 / +0.1753 / +0.1706 / +0.1862 — larger for every
# arm and a SIGN FLIP for E2B, which is exactly the defect these values guard.
EXPECTED_MATCHED_RECALL_DELTA = {
    "glm-5": +0.1451, "gemma-4-26b": +0.1391, "gemma-4-31b": +0.1368,
    "gemma-4-e2b": -0.0059,
}

# Prose that a structural guard governs, when that guard no longer holds.  The
# guards do not raise: they are collected and reported AFTER the numeric checks,
# so a regression reads as the number it broke rather than as an assertion in the
# middle of a disclosure string.  A payload carrying this string is never written
# (`main` refuses on any failure) — it exists so the run reaches the number check.
GUARD_UNAVAILABLE = (
    "UNAVAILABLE — a structural guard on this text no longer holds; see the "
    "failure list. This payload was not written."
)

THRESHOLD_RULE = (
    "tau is the arm's OWN full-panel best-error-F1 cut: of the arm's own "
    "distinct scores, the one whose flag set {belief < tau} maximises "
    "error-class F1 on all 1689 statements. Ties are broken to the SMALLEST "
    "such cut, so the rule is a function of the score vector alone and does not "
    "depend on the order the rows arrived in. Each arm is cut at its own tau; "
    "no cut is shared and none is transferred between arms."
)

BOOTSTRAP_DESIGN = (
    "paired fold-stratified bootstrap over the paper's own out-of-fold fold "
    "assignment (fold_ix); ONE index vector per resample, SHARED across every "
    "arm, so the arms' deltas are drawn jointly and the max-t band carries their "
    "correlation rather than assuming it away. The taus are FIXED at their "
    "full-panel values and are not refit per resample: the interval is on the "
    "margin at a stated operating point, not on the threshold search. Same seed, "
    "same resample count and same design as scripts/compute_paper_robustness.py "
    "and scripts/compute_statement_review_queue.py."
)

POSITIVE_CLASS_NOTE = (
    "ERROR is the positive class: the pair scored per statement is "
    "(gold_error, pred_error) = (released_paper_correct == 0, belief < tau). "
    "The panel is 73.2% positive in the CORRECT class, so correct-class F1 is "
    "majority-dominated — a constant 'everything is correct' classifier scores "
    "0.845 correct-class F1 and 0.000 error-class F1 — and must not be the "
    "headline. Correct-class precision/recall/F1 at the SAME cut are emitted "
    "beside every error-class row, and every confusion count is emitted, so the "
    "majority-class view is recoverable rather than deleted."
)


def load_panel() -> dict:
    """The 1689 assembled statements carrying a released paper label.

    The head-to-head's join, unchanged: the paper's own out-of-fold block keyed
    on stmt_hash, joined to the frozen gold's statement_id, ordered by
    sorted(stmt_hash) exactly as the AP comparison orders it.
    """
    lit = json.loads((ROOT / PAPER_LITERAL).read_text())
    oof = {int(r["stmt_hash"]): r for r in lit["oof_predictions"][HEADLINE]}

    gold = {}
    for row in load_jsonl(ROOT / GOLD):
        policy = row["paper_replication_policy"]
        if policy.get("released_paper_correct") is None:
            continue
        gold[int(row["paper_statement_hash"])] = {
            "sid": row["canonical_corpus"]["statement_id"],
            "label": int(policy["released_paper_correct"]),
        }

    hashes = sorted(oof)
    assert set(hashes) == set(gold), (
        "the paper's out-of-fold run does not cover exactly the labelled panel")
    sids = [gold[h]["sid"] for h in hashes]
    assert len(set(sids)) == len(sids), "duplicate statement_id in the paper gold"
    y = np.array([int(oof[h]["y_true"]) for h in hashes], dtype=int)
    assert all(int(oof[h]["y_true"]) == gold[h]["label"] for h in hashes), (
        "the paper's out-of-fold labels disagree with the frozen gold")

    return {
        "hashes": hashes,
        "sids": sids,
        "y": y,
        "is_error": y == 0,
        "folds": np.array([int(oof[h]["fold_ix"]) for h in hashes], dtype=int),
        "reference_scores": np.array([float(oof[h]["prob_correct"]) for h in hashes]),
    }


def error_confusion(is_error: np.ndarray, flagged: np.ndarray) -> dict:
    """Error-class confusion, through the repo's single P/R/F1 definition."""
    return confusion_pr(zip(is_error.tolist(), flagged.tolist()))


def correct_confusion(is_error: np.ndarray, flagged: np.ndarray) -> dict:
    """The SAME cut, read from the majority class. Emitted so nothing is hidden."""
    return confusion_pr(zip((~is_error).tolist(), (~flagged).tolist()))


def best_error_f1_threshold(scores: np.ndarray, is_error: np.ndarray) -> float:
    """The arm's own best-error-F1 cut; ties broken to the SMALLEST such cut.

    The flag set {belief < tau} only changes at the arm's observed scores, so the
    observed scores are the complete candidate set. Scanning them in ascending
    order and keeping a cut only on a STRICT improvement makes the tie-break
    explicit rather than an accident of max().
    """
    best_tau, best_f1 = None, -1.0
    for tau in np.unique(scores):
        f1 = error_confusion(is_error, scores < tau)["f1"]
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    assert best_tau is not None and best_f1 > 0.0, "no cut separates the panel at all"
    return best_tau


def threshold_catching_at_least(scores: np.ndarray, is_error: np.ndarray,
                                min_errors: int) -> float:
    """Smallest cut over these scores catching at least ``min_errors`` errors.

    Caught errors are monotone non-decreasing in tau, so the first cut that
    reaches the count is the cheapest one that does. Everything recall-shaped in
    this script goes through here, so "the arm's cut at 60%" and "the reference's
    cut at the arm's achieved recall" are the same rule applied to two vectors.
    """
    for tau in np.unique(scores):
        if int(is_error[scores < tau].sum()) >= min_errors:
            return float(tau)
    raise AssertionError(
        f"no cut over this arm's own scores catches {min_errors} of the panel's "
        f"{int(is_error.sum())} errors")


def threshold_for_error_recall(scores: np.ndarray, is_error: np.ndarray,
                               target: float) -> float:
    """Smallest cut catching at least ``target`` of the panel's errors."""
    n_errors = int(is_error.sum())
    # ceil of target*n_errors, nudged so a target that lands exactly on an
    # integer count is not pushed one statement further by binary float error.
    need = int(np.ceil(target * n_errors - 1e-9))
    return threshold_catching_at_least(scores, is_error, need)


def operating_point(scores: np.ndarray, is_error: np.ndarray, tau: float) -> dict:
    """One cut, fully described: both classes, every count, the queue it implies."""
    flagged = scores < tau
    err = error_confusion(is_error, flagged)
    cor = correct_confusion(is_error, flagged)
    n_errors = int(is_error.sum())
    assert err["tp"] + err["fn"] == n_errors, "error-class counts do not close on the panel"
    assert err["tp"] + err["fp"] == int(flagged.sum()), "flag set is not tp + fp"
    return {
        "tau": tau,
        "flagged": int(flagged.sum()),
        "error_precision": err["p"],
        "error_recall": err["r"],
        "error_f1": err["f1"],
        "tp": err["tp"],
        "fp": err["fp"],
        "fn": err["fn"],
        "tn": err["tn"],
        "accuracy": err["acc"],
        "correct_precision": cor["p"],
        "correct_recall": cor["r"],
        "correct_f1": cor["f1"],
        "flag_set_is_the_arms_zero_pile": bool(
            flagged.sum() > 0 and np.array_equal(flagged, scores <= 0.0)),
    }


def review_queue_error_f1(block: dict, n_errors: int, n: int) -> dict:
    """The review queue's own published counts, run through the SAME F1.

    ``operating_point`` there publishes ``true_errors_caught`` (tp) and
    ``false_alarms`` (fp) at a target recall of 0.70 under a DIFFERENT rule
    (``belief <= tau``). Rebuilding the pair stream from those counts and
    scoring it with ``confusion_pr`` is a second derivation of the same
    quantity: no score vector is re-read and no threshold is re-searched.
    """
    tp = int(block["true_errors_caught"])
    fp = int(block["false_alarms"])
    fn = n_errors - tp
    tn = n - tp - fp - fn
    assert min(tp, fp, fn, tn) >= 0, "the review queue's counts do not close on this panel"
    pairs = chain(repeat((True, True), tp), repeat((False, True), fp),
                  repeat((True, False), fn), repeat((False, False), tn))
    got = confusion_pr(pairs)
    assert (got["tp"], got["fp"], got["fn"], got["tn"]) == (tp, fp, fn, tn)
    return got


def reconciliation_row(model_key: str, block: dict, got: dict, point: dict) -> dict:
    """ONE row shape for the reference and for every arm.

    The reference used to be emitted under `f1`/`p`/`r` while the arms used the
    `review_queue_*` names, so a consumer had to special-case which side of the
    comparison it was reading. Both go through here now; the reference is a row
    like any other.
    """
    return {
        "review_queue_model_key": model_key,
        "review_queue_tau": block["tau"],
        "review_queue_error_precision": block["precision"],
        "review_queue_error_recall": block["recall_achieved"],
        "review_queue_error_f1": got["f1"],
        "review_queue_tp": got["tp"],
        "review_queue_fp": got["fp"],
        "this_artifact_tau": point["tau"],
        "this_artifact_error_f1": point["error_f1"],
        "residual": abs(got["f1"] - point["error_f1"]),
        "same_flag_set": (got["tp"], got["fp"]) == (point["tp"], point["fp"]),
    }


def build_payload() -> tuple[dict, list[str]]:
    """The payload, and the list of PROSE guards that no longer hold.

    A non-empty guard list means some disclosure string is describing a state of
    the world that stopped being true. It is returned rather than raised so the
    caller can report the numeric checks first; `main` refuses to write on it.
    """
    panel = load_panel()
    sids, is_error = panel["sids"], panel["is_error"]
    n = len(sids)
    n_errors = int(is_error.sum())
    n_correct = n - n_errors
    assert 0 < n_errors < n

    # ---- arms, in a fixed order: reference, the four staged reader arms, then
    # the descriptive one.  Order is a presentation decision, never a data one.
    scores = {REFERENCE_ID: panel["reference_scores"]}
    specs = []
    for arm, in_family in [(a, True) for a in READER_ARMS] + [(a, False) for a in CONTEXT_ARMS]:
        bundle = f"{MODELS_DIR}/{arm['bundle']}/all_source_predictions.jsonl"
        rows = load_jsonl(ROOT / bundle)
        table = {r["statement_id"]: float(r["probability_correct"]) for r in rows}
        assert len(table) == len(rows), f"{arm['id']}: duplicate statement_id"
        assert set(table) == set(sids), (
            f"{arm['id']}: prediction bundle does not cover the panel exactly")
        scores[arm["id"]] = np.array([table[s] for s in sids])
        specs.append({**arm, "bundle_path": bundle, "in_family": in_family})

    # ---- the cuts, and the point estimates at them --------------------------
    taus = {key: best_error_f1_threshold(vector, is_error) for key, vector in scores.items()}
    points = {key: operating_point(scores[key], is_error, taus[key]) for key in scores}
    # ---- the SECOND cut, genuinely recall-matched ---------------------------
    # Each arm gets the cheapest cut catching >= 60% of the panel's errors; how
    # far past 60% that lands is a property of the arm's score grid, not of its
    # quality.  The reference is then RE-CUT at the recall this arm achieved, so
    # the delta compares two rows at the same recall.  The unmatched subtraction
    # (both sides at their own 60% cut) is a DIFFERENT quantity — larger for
    # every arm here, and sign-flipping for E2B — so it is kept, under a name
    # that says which one it is.
    matched = {}
    for key, vector in scores.items():
        tau = threshold_for_error_recall(vector, is_error, MATCHED_ERROR_RECALL)
        row = operating_point(vector, is_error, tau)
        assert row["error_recall"] >= MATCHED_ERROR_RECALL - 1e-12, (
            f"{key}: matched-recall cut delivers {row['error_recall']:.4f}")
        matched[key] = row

    reference_f1 = points[REFERENCE_ID]["error_f1"]
    reference_matched_f1 = matched[REFERENCE_ID]["error_f1"]

    def matched_recall_block(key: str) -> dict:
        row = matched[key]
        ref_tau = threshold_catching_at_least(scores[REFERENCE_ID], is_error, row["tp"])
        ref_row = operating_point(scores[REFERENCE_ID], is_error, ref_tau)
        overshoot = ref_row["error_recall"] - row["error_recall"]
        assert ref_row["tp"] >= row["tp"] and overshoot >= -1e-12, (
            f"{key}: the reference's matched cut catches {ref_row['tp']} errors, "
            f"fewer than this row's {row['tp']}")
        assert overshoot <= 1 / n_errors + 1e-12, (
            f"{key}: matching cost {overshoot:.4f} of recall — more than the "
            f"reference's own one-statement grid step, so the rows are not matched")
        return {
            "target_error_recall": MATCHED_ERROR_RECALL,
            **row,
            "reference_at_this_rows_recall": ref_row,
            "reference_error_recall_at_this_row": ref_row["error_recall"],
            "reference_error_f1_at_this_row": ref_row["error_f1"],
            "reference_recall_overshoot": overshoot,
            "delta_error_f1_at_matched_recall": row["error_f1"] - ref_row["error_f1"],
            "delta_error_f1_each_side_at_its_own_target_cut":
                row["error_f1"] - reference_matched_f1,
        }

    matched_blocks = {key: matched_recall_block(key) for key in scores}
    achieved = sorted(matched[key]["error_recall"] for key in scores)
    matched_recall_spread = achieved[-1] - achieved[0]

    # ---- the paired bootstrap, at those FIXED cuts ---------------------------
    flags = {key: scores[key] < taus[key] for key in scores}
    boot_idx = fold_stratified_indices(panel["folds"], N_BOOT, SEED)
    keys = [spec["id"] for spec in specs]
    draws: dict[str, list[float]] = {key: [] for key in keys}
    for take in boot_idx:
        errors_b = is_error[take]
        # A resample with no error at all has no error-class recall to speak of;
        # it is dropped for EVERY arm at once so the columns stay aligned.
        if not errors_b.any():
            continue
        base_b = error_confusion(errors_b, flags[REFERENCE_ID][take])["f1"]
        for key in keys:
            draws[key].append(error_confusion(errors_b, flags[key][take])["f1"] - base_b)

    results = {}
    for key in keys:
        column = np.array(draws[key])
        delta = points[key]["error_f1"] - reference_f1
        se = float(column.std(ddof=1))
        assert se > 0, f"{key}: degenerate bootstrap standard error"
        results[key] = {
            "delta": delta,
            "bootstrap_se": se,
            "delta_bootstrap_mean": float(column.mean()),
            "ci95_low": float(np.percentile(column, 2.5)),
            "ci95_high": float(np.percentile(column, 97.5)),
            "p_greater_than_zero": float((column > 0).mean()),
            "n_valid_resamples": int(len(column)),
            "t_statistic": delta / se,
            "_draws": column,
        }

    # ---- the simultaneous band, over the staged family and nothing else ------
    family_keys = [spec["id"] for spec in specs if spec["in_family"]]
    plan = json.loads((ROOT / RUN_PLAN).read_text())
    staged = {stage["id"] for stage in plan["stages"]}
    assert staged == {arm["stage"] for arm in READER_ARMS}, (
        "the frozen run plan no longer stages exactly this family")
    critical, bands = simultaneous_band({k: results[k] for k in family_keys})
    bonferroni = statistics.NormalDist().inv_cdf(1 - FAMILY_ALPHA / (2 * len(family_keys)))
    assert POINTWISE_NORMAL_Z < critical <= bonferroni, (
        f"max-t critical value {critical:.4f} is outside "
        f"({POINTWISE_NORMAL_Z:.4f}, {bonferroni:.4f}] — a simultaneous band cannot be "
        f"narrower than pointwise nor wider than Bonferroni")

    arms = []
    for spec in specs:
        key = spec["id"]
        res = results[key]
        band = bands.get(key)
        point = points[key]
        arms.append({
            "key": key,
            "display": DISPLAY[key],
            "label": spec["label"],
            "review_queue_model_key": REVIEW_QUEUE_KEY.get(key),
            "role": "reader-gate" if spec["in_family"] else "descriptive-nonconfirmatory",
            "in_max_t_family": spec["in_family"],
            "prediction_bundle": spec["bundle_path"],
            "distinct_scores": int(len(np.unique(scores[key]))),
            "operating_point": point,
            "matched_recall": matched_blocks[key],
            "delta_error_f1": res["delta"],
            "delta_bootstrap_mean": res["delta_bootstrap_mean"],
            "ci95_low": res["ci95_low"],
            "ci95_high": res["ci95_high"],
            "bootstrap_se": res["bootstrap_se"],
            "t_statistic": res["t_statistic"],
            "p_greater_than_zero": res["p_greater_than_zero"],
            "n_valid_resamples": res["n_valid_resamples"],
            "simultaneous_low": band[0] if band else None,
            "simultaneous_high": band[1] if band else None,
            "excludes_zero_pointwise": excludes_zero(res["ci95_low"], res["ci95_high"]),
            "excludes_zero_simultaneous": excludes_zero(*band) if band else None,
        })

    # ---- disclosures, with every number derived ------------------------------
    # The three checks below govern PROSE, not numbers. They are collected, not
    # raised: raising here would abort the run before verify_expected got to say
    # which number moved, and the number is the thing a reader needs. `main`
    # refuses to write on any collected failure, so nothing ships unguarded.
    guards: list[str] = []
    winners = [a for a in arms if a["in_max_t_family"] and a["excludes_zero_simultaneous"]]
    losers = [a for a in arms if a["in_max_t_family"] and not a["excludes_zero_simultaneous"]]
    family_is_split = bool(winners) and bool(losers)
    if not family_is_split:
        guards.append(
            "oracle_disclosure describes a SPLIT family; the family is no longer "
            f"split (excludes zero simultaneously: {[a['key'] for a in winners]}; "
            f"does not: {[a['key'] for a in losers]})")
    spelled = {1: "one", 2: "two", 3: "three", 4: "four"}
    reference_cuts = int(len(np.unique(scores[REFERENCE_ID])))

    if family_is_split:
        win_cuts = sorted(a["distinct_scores"] for a in winners)
        best_arm = max(winners, key=lambda a: a["delta_error_f1"])
        oracle_disclosure = (
            f"Every tau here is chosen ON THIS PANEL, with these labels already in "
            f"hand, to maximise the arm's OWN error-class F1. It is fitted AND "
            f"evaluated on the same {n} statements, it would not be available before "
            f"curation, and no cut here is validated out of sample. That is an "
            f"ORACLE, and it FAVOURS the paper's RF rather than us: the RF's "
            f"near-continuous scores give the search {reference_cuts} candidate cuts "
            f"to optimise over, where the {spelled[len(winners)]} reader arms that win have "
            f"{win_cuts[0]}-{win_cuts[-1]} and "
            + ", ".join(f"{a['display']} has {a['distinct_scores']}" for a in losers)
            + f". The side handed the finest-grained oracle still loses by "
            f"{best_arm['delta_error_f1']:+.4f} to {best_arm['display']}. Read this as "
            f"an operating-point comparison, not as a held-out result. It covers the "
            f"headline cut ONLY; the matched_recall block's second cut is fitted on "
            f"this panel too, by a different rule, and is disclosed separately in "
            f"matched_recall_rule."
        )
    else:
        oracle_disclosure = GUARD_UNAVAILABLE

    modal_cut = sorted({a["operating_point"]["tau"] for a in winners})
    zero_pile_winners = [a for a in winners
                         if a["operating_point"]["flag_set_is_the_arms_zero_pile"]]
    if len(modal_cut) != 1:
        guards.append(
            f"modal_threshold_note says the winning arms share one cut; they now "
            f"hold {modal_cut}")
    if len(zero_pile_winners) != len(winners):
        guards.append(
            "modal_threshold_note says every winning arm's best cut is its own zero "
            "pile; that stopped being true for "
            + ", ".join(a["key"] for a in winners if a not in zero_pile_winners))
    if family_is_split and len(modal_cut) == 1 and len(zero_pile_winners) == len(winners):
        modal_threshold_note = (
            f"{', '.join(a['display'] for a in winners)} all land on the SAME cut, "
            f"tau = {modal_cut[0]:.4f} — the modal NON-ZERO belief INDRA's noisy-OR "
            f"returns on this panel, and the smallest it can return from a single "
            f"surviving piece of evidence. No reader score falls between 0 and that "
            f"value, so at that cut their flagged set is exactly the block of "
            f"statements whose evidence the reader rejected outright (belief 0.0). "
            f"Their 'best' threshold is therefore not a tuned one at all: it is the "
            f"arm's own untuned rejection block, and the oracle bought them nothing. "
            + "; ".join(f"{a['display']} lands at {a['operating_point']['tau']:.4f}"
                        for a in losers)
            + f"; the paper's RF at {points[REFERENCE_ID]['tau']:.4f}. Disclosed "
            f"because it is the strongest thing anyone could say against these "
            f"numbers, and it turns out to cut the other way."
        )
    else:
        modal_threshold_note = GUARD_UNAVAILABLE

    matched_recall_rule = (
        f"matched_recall is a SECOND cut, and a second panel-fitted threshold: "
        f"each arm's cheapest cut catching at least "
        f"{MATCHED_ERROR_RECALL:.0%} of the panel's {n_errors} errors. It is "
        f"chosen with these labels in hand, exactly like the headline tau, so the "
        f"oracle_disclosure applies to it as well — by a DIFFERENT rule (target "
        f"recall, not best F1), which is why it is stated separately. Because each "
        f"arm's scores are discrete, the recall an arm ACHIEVES overshoots the "
        f"target by its own grid's granularity, and across these arms the achieved "
        f"error recalls span {achieved[0]:.4f} to {achieved[-1]:.4f} — "
        f"{matched_recall_spread * 100:.1f} points, across "
        f"{len(set(achieved))} distinct values in {len(matched)} rows. A single "
        f"reference row would therefore be subtracted from rows sitting at "
        f"{len(set(achieved))} different recalls, which is not a matched "
        f"comparison and is not what this block reports. "
        f"delta_error_f1_at_matched_recall is the one to read: for each row the "
        f"reference is RE-CUT at that row's own achieved recall (the cheapest "
        f"reference cut catching at least as many errors), and "
        f"reference_error_recall_at_this_row and the full "
        f"reference_at_this_rows_recall row are emitted beside it so the match is "
        f"checkable rather than asserted. "
        f"delta_error_f1_each_side_at_its_own_target_cut is the unmatched "
        f"subtraction — both sides at their own {MATCHED_ERROR_RECALL:.0%} cut — "
        f"kept because it is a real quantity, and named for what it is because it "
        f"is NOT recall-matched and flatters whichever arm overshot furthest."
    )

    # ---- reconciliation against the independent derivation -------------------
    queue = json.loads((ROOT / REVIEW_QUEUE).read_text())
    queue_by_key = {a["model_key"]: a for a in queue["arms"]}
    assert queue["headline_target_recall"] == REVIEW_QUEUE_TARGET_RECALL, (
        "the review queue's headline target moved; the reconciliation would be "
        "comparing a different operating point")
    # The queue publishes tp and fp; fn and tn are rebuilt from THIS panel's
    # totals. That is only a reconciliation if the queue was built on the same
    # panel — if it is ever regenerated on the 1578-row adjudication-safe panel
    # the rebuilt counts would still close and the residual would still look
    # small while comparing two populations. Same guard as the target-recall one.
    assert queue["panel"]["n"] == n and queue["panel"]["n_errors"] == n_errors, (
        f"the review queue was built on panel n={queue['panel']['n']} / "
        f"n_errors={queue['panel']['n_errors']}, this artifact on n={n} / "
        f"n_errors={n_errors}; the reconciliation would compare two populations")

    def reconcile(key: str, point: dict) -> dict:
        model_key = REVIEW_QUEUE_KEY[key]
        block = queue_by_key[model_key]["operating_point"]
        row = reconciliation_row(
            model_key, block, review_queue_error_f1(block, n_errors, n), point)
        assert row["residual"] <= RECONCILIATION_TOLERANCE, (
            f"{key}: this artifact's error-F1 {point['error_f1']:.6f} and the "
            f"review queue's {row['review_queue_error_f1']:.6f} differ by "
            f"{row['residual']:.6f} > {RECONCILIATION_TOLERANCE}")
        return row

    reconciliation_arms = {
        arm["key"]: reconcile(arm["key"], arm["operating_point"])
        for arm in arms if arm["review_queue_model_key"] is not None
    }
    reference_reconciliation = reconcile(REFERENCE_ID, points[REFERENCE_ID])
    worst_residual = max([reference_reconciliation["residual"]]
                         + [r["residual"] for r in reconciliation_arms.values()])

    caveats = [
        "The 2023 paper NEVER published a decision metric, a threshold, or a "
        "test of any kind — only trapezoidal PR-AUC as a 10-fold mean with a "
        "population SD, over 59 rows. Error-class F1 is OUR derivation on their "
        "panel, their released labels and their own folds. They did not report "
        "it and this is not a number they lost on; it is a number nobody ran.",

        f"EVERY threshold here is fitted and evaluated on the same {n} "
        "statements — the headline best-F1 cut AND the matched_recall block's "
        "target-recall cut, which is a second panel-fitted threshold chosen by a "
        "second rule. See oracle_disclosure for the headline cut and "
        "matched_recall_rule for the other. On the headline cut the advantage "
        "that buys runs toward the paper's RF, which gets the finest-grained "
        "search, and away from the three reader arms, whose best cut is their "
        "own untuned rejection block.",

        "F1 at a chosen cut is a DECISION metric at ONE operating point. It does "
        "not supersede pooled average precision or AUROC and it retracts none of "
        "them: it answers a different question — how good is the queue this arm "
        "hands a curator — and both frames stay on the page. The matched_recall "
        "block gives a second cut on the same arms so the headline is not the "
        "only operating point on record. Read its "
        "delta_error_f1_at_matched_recall, in which the reference is re-cut at "
        "each row's OWN achieved error recall: the arms' achieved recalls at a "
        f"{MATCHED_ERROR_RECALL:.0%} target span "
        f"{matched_recall_spread * 100:.1f} points, so the unmatched subtraction "
        "kept beside it as delta_error_f1_each_side_at_its_own_target_cut is "
        "larger for every arm and flips Gemma 4 E2B's sign. The matched deltas "
        "are "
        + ", ".join(
            f"{a['display']} {a['matched_recall']['delta_error_f1_at_matched_recall']:+.4f}"
            for a in arms if a["in_max_t_family"])
        + ". They are POINT values: no interval, no bootstrap and no "
        "multiplicity correction is computed at this second cut, so the tested "
        "claim remains the headline family's.",

        "ERROR is the positive class here BECAUSE the panel is "
        f"{n_correct / n:.1%} correct. Correct-class F1 on this panel is "
        "majority-dominated and would flatter every arm; correct-class "
        "precision, recall and F1 at the same cut are emitted beside every "
        "error-class row so that view is recoverable, not deleted.",

        "Gemma 4 E2B LOSES: its delta is negative and its pointwise interval "
        "includes zero. 'Reader gates beat the paper's RF on error-F1' is "
        "therefore false as a universal, and is true of the three larger arms "
        "only. The family is corrected over all four, not over the winners.",

        "The reader arms are INDRA's own noisy-OR applied to the evidence the "
        "reader KEPT, so this compares belief models on a shared aggregation and "
        "the reader's contribution is the filtering. They are also not "
        "zero-shot: each call carries 14 hand-authored demonstration pairs.",

        "The paper's RF is scored on the paper's own 2023 feature matrix and its "
        "own folds, because that is what re-running their released code "
        "produces; the reader arms are scored on current INDRA evidence. The "
        "arms compare cleanly to each other on the same statements and the same "
        "labels, but only loosely to the paper's published 2023 table.",

        "INDRA CoGEx hybrid is carried for completeness and is NOT in the "
        "simultaneous family: its own bundle manifest declares it "
        "descriptive_nonconfirmatory, and the frozen run plan stages the four "
        "reader arms and nothing else. Its interval is pointwise only.",

        "This artifact adds no data. It scores the same prediction vectors the "
        "AP head-to-head and the review queue already score; the reconciliation "
        "block pins it to the review queue's separately RULED operating point to "
        f"within {worst_residual:.4f}. Separately ruled, not independent: same "
        "panel, same labels, same scores, and on the three winning arms the two "
        "rules select the same statements — see reconciliation.note.",
    ]

    payload = {
        "artifact_kind": "statement_error_f1",
        "schema_version": 1,
        "metric": "statement-grain error-class F1 at each arm's own best-F1 cut",
        "positive_class": "error (released_paper_correct == 0)",
        "positive_class_note": POSITIVE_CLASS_NOTE,
        "decision_rule": "flag a statement as an ERROR iff belief < tau",
        "threshold_rule": THRESHOLD_RULE,
        "oracle_disclosure": oracle_disclosure,
        "matched_recall_rule": matched_recall_rule,
        "modal_threshold_note": modal_threshold_note,
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "bootstrap_design": BOOTSTRAP_DESIGN,
        "panel": {
            "id": "paper_labels_1689",
            "n": n,
            "n_errors": n_errors,
            "n_correct": n_correct,
            "error_base_rate": n_errors / n,
            "correct_base_rate": n_correct / n,
            "label_field": "paper_replication_policy.released_paper_correct",
            "label_provenance": "the 2023 paper's released labels, unmodified",
            "ordering": "sorted(paper_statement_hash), as the AP head-to-head orders it",
            "n_folds": int(len(set(panel["folds"].tolist()))),
        },
        "reference": {
            "key": REFERENCE_ID,
            "display": DISPLAY[REFERENCE_ID],
            "label": REFERENCE_LABEL,
            "review_queue_model_key": REVIEW_QUEUE_KEY[REFERENCE_ID],
            "description": "our re-run of the paper's own released RF code "
                           "(sorgerlab/indra_assembly_paper), scored out of fold on "
                           "the paper's own folds",
            "method_string": HEADLINE,
            "distinct_scores": reference_cuts,
            "operating_point": points[REFERENCE_ID],
            "matched_recall": matched_blocks[REFERENCE_ID],
        },
        "multiplicity": {
            "family": family_keys,
            "family_size": len(family_keys),
            "family_alpha": FAMILY_ALPHA,
            "method": "studentized max-t over the shared paired-bootstrap draws",
            "max_t_critical_value": critical,
            "pointwise_normal_critical_value": POINTWISE_NORMAL_Z,
            "bonferroni_critical_value": bonferroni,
            "n_excluding_zero_simultaneous": len(winners),
            "run_plan": {
                "path": RUN_PLAN,
                "sha256": sha256_of(ROOT / RUN_PLAN),
                "stages": sorted(staged),
            },
            "note": "The frozen run plan stages all four reader arms and "
                    "designates none of them as the confirmatory one, so a "
                    "simultaneous band over the family is the fair ask. The arms "
                    "move together from resample to resample, so max-t costs far "
                    "less than Bonferroni would. INDRA CoGEx hybrid is outside "
                    "the family and is quoted pointwise only.",
        },
        "arms": arms,
        "reconciliation": {
            "source": REVIEW_QUEUE,
            "sha256": sha256_of(ROOT / REVIEW_QUEUE),
            "source_target_recall": REVIEW_QUEUE_TARGET_RECALL,
            "tolerance": RECONCILIATION_TOLERANCE,
            "worst_residual": worst_residual,
            "panel_matches": {
                "n": queue["panel"]["n"],
                "n_errors": queue["panel"]["n_errors"],
                "asserted": True,
            },
            "reference": reference_reconciliation,
            "arms": reconciliation_arms,
            "note": "Two THRESHOLD RULES over one quantity, recorded together "
                    "rather than one being trusted. The review queue reaches its "
                    "cut by target recall under 'belief <= tau'; this artifact "
                    "maximises error-F1 under 'belief < tau'. They are NOT "
                    "independent derivations: the panel, the labels and the score "
                    "vectors are the same, and for the three larger reader arms "
                    "the two rules select the SAME statements (both land on the "
                    "arm's zero pile), so `same_flag_set` is true and the residual "
                    "is exactly 0. That is a cross-check of two code paths over "
                    "one flag set — worth recording, but it is not a second "
                    "measurement and it cannot corroborate the flag set itself. "
                    "For the paper's RF the two rules land on genuinely different "
                    "cuts, which is the only row here where the residual carries "
                    "information; it is the largest one, and it is why the "
                    "tolerance is stated rather than assumed to be zero.",
        },
        "caveats": caveats,
        "checks": {
            "every_arm_covers_the_panel_exactly": True,
            "error_counts_close_on_the_panel": True,
            "flag_set_equals_tp_plus_fp": True,
            "matched_recall_cut_delivers_the_target": True,
            "matched_recall_reference_is_recut_at_each_rows_achieved_recall": True,
            "family_is_exactly_the_frozen_run_plan_stages": True,
            "max_t_band_is_between_pointwise_and_bonferroni": True,
            "review_queue_panel_is_this_panel": True,
            "reconciles_with_statement_review_queue": True,
            "family_is_split_into_winners_and_losers": family_is_split,
            "winning_arms_share_one_cut_and_it_is_their_zero_pile": bool(
                len(modal_cut) == 1 and len(zero_pile_winners) == len(winners)),
            "note": "Assertions are enforced in code; a violation fails the build "
                    "rather than being reported here as False. The last two are "
                    "the exception: they govern PROSE (oracle_disclosure, "
                    "modal_threshold_note), so they are reported here as real "
                    "booleans and raised as build failures AFTER the numeric "
                    "checks, letting a regression read as the number that moved.",
        },
        "provenance": {
            "inputs": {
                "paper_literal_oof": {"path": PAPER_LITERAL,
                                      "sha256": sha256_of(ROOT / PAPER_LITERAL)},
                "gold": {"path": GOLD, "sha256": sha256_of(ROOT / GOLD)},
                "review_queue": {"path": REVIEW_QUEUE,
                                 "sha256": sha256_of(ROOT / REVIEW_QUEUE)},
                "run_plan": {"path": RUN_PLAN, "sha256": sha256_of(ROOT / RUN_PLAN)},
                **{
                    spec["id"]: {"path": spec["bundle_path"],
                                 "sha256": sha256_of(ROOT / spec["bundle_path"])}
                    for spec in specs
                },
            },
            "join": "paper_statement_hash -> canonical_corpus.statement_id; the "
                    "paper's own out-of-fold block is joined on stmt_hash",
            "metric_implementation": "src/indra_belief/metrics.py::confusion_pr",
            "generated_by": "scripts/compute_statement_error_f1.py",
        },
    }
    return payload, guards


def verify_expected(payload: dict) -> list[str]:
    """The spec's verified numbers, checked against what this run produced.

    Returns the list of failures; an empty list is the only passing result. The
    numbers were measured before this script existed, so a mismatch is a defect
    here and must fail loudly rather than ship a different value.
    """
    failures: list[str] = []

    def check_point(name: str, actual: float, expected: float, places: int) -> None:
        """A POINT value: the spec quotes it to `places`, so the check is that the
        re-derived value rounds to exactly that, to within TOL_POINT."""
        rounded = round(actual, places)
        if not abs(rounded - expected) <= TOL_POINT:
            failures.append(f"{name}: {actual!r} rounds to {rounded} != expected "
                            f"{expected} at {places} places")

    def check_interval(name: str, actual: float, expected: float) -> None:
        """A BOOTSTRAP endpoint: compared at TOL_INTERVAL, not to the last digit."""
        if not abs(actual - expected) <= TOL_INTERVAL:
            failures.append(f"{name}: {actual:.6f} != expected {expected:.6f} "
                            f"(|delta| {abs(actual - expected):.6f} > {TOL_INTERVAL})")

    check_point("reference error_f1", payload["reference"]["operating_point"]["error_f1"],
                EXPECTED_REFERENCE_ERROR_F1, 4)
    check_point("max_t_critical_value", payload["multiplicity"]["max_t_critical_value"],
                EXPECTED_MAX_T_CRITICAL, 4)

    by_key = {a["key"]: a for a in payload["arms"]}
    for key, want in EXPECTED_ARMS.items():
        if key not in by_key:
            failures.append(f"{key}: absent from the artifact")
            continue
        arm = by_key[key]
        check_point(f"{key} error_f1", arm["operating_point"]["error_f1"],
                    want["error_f1"], 4)
        check_point(f"{key} delta", arm["delta_error_f1"], want["delta"], 4)
        check_interval(f"{key} ci95_low", arm["ci95_low"], want["ci95_low"])
        check_interval(f"{key} ci95_high", arm["ci95_high"], want["ci95_high"])
        if "t_statistic" in want:
            check_point(f"{key} t_statistic", arm["t_statistic"], want["t_statistic"], 2)
        if arm["excludes_zero_simultaneous"] != want["excludes_zero_simultaneous"]:
            failures.append(
                f"{key} excludes_zero_simultaneous: {arm['excludes_zero_simultaneous']} "
                f"!= expected {want['excludes_zero_simultaneous']}")
        matched = arm["matched_recall"]
        check_point(f"{key} matched-recall delta",
                    matched["delta_error_f1_at_matched_recall"],
                    EXPECTED_MATCHED_RECALL_DELTA[key], 4)
        # The whole point of that number: the two rows being differenced are at
        # the SAME error recall. Checked here, not only asserted at build time,
        # so re-quoting the reference at its own cut fails the gate by name.
        if abs(matched["reference_error_recall_at_this_row"]
               - matched["error_recall"]) > 1 / payload["panel"]["n_errors"] + TOL_POINT:
            failures.append(
                f"{key} matched_recall is not recall-matched: this row is at error "
                f"recall {matched['error_recall']:.4f}, the reference row it is "
                f"differenced against at "
                f"{matched['reference_error_recall_at_this_row']:.4f}")

    simultaneous = [a["key"] for a in payload["arms"]
                    if a["in_max_t_family"] and a["excludes_zero_simultaneous"]]
    if sorted(simultaneous) != sorted(k for k, v in EXPECTED_ARMS.items()
                                      if v["excludes_zero_simultaneous"]):
        failures.append(
            f"the set of arms excluding zero simultaneously is {sorted(simultaneous)}, "
            f"not the three larger arms")

    reference_reconciliation = payload["reconciliation"]["reference"]
    for key, want in EXPECTED_REVIEW_QUEUE_ERROR_F1.items():
        block = (reference_reconciliation if key == REFERENCE_ID
                 else payload["reconciliation"]["arms"].get(key))
        if block is None:
            failures.append(f"{key}: absent from the reconciliation block")
            continue
        # ONE key schema for the reference and the arms; reading them the same
        # way here is what keeps it that way.
        got = block.get("review_queue_error_f1")
        if got is None:
            failures.append(
                f"{key}: reconciliation row has no review_queue_error_f1 "
                f"(keys: {sorted(block)})")
            continue
        check_point(f"{key} review-queue error_f1", got, want, 3)
    arm_schema = {k for r in payload["reconciliation"]["arms"].values() for k in r}
    if arm_schema and set(reference_reconciliation) != arm_schema:
        failures.append(
            f"reconciliation.reference and reconciliation.arms use different key "
            f"schemas: {sorted(set(reference_reconciliation) ^ arm_schema)}")
    if payload["reconciliation"]["worst_residual"] > RECONCILIATION_TOLERANCE:
        failures.append(
            f"reconciliation worst residual "
            f"{payload['reconciliation']['worst_residual']:.6f} > {RECONCILIATION_TOLERANCE}")
    return failures


def render(payload: dict) -> str:
    return json.dumps(payload, indent=1) + "\n"


def report(payload: dict) -> None:
    panel = payload["panel"]
    ref = payload["reference"]["operating_point"]
    print(f"panel n={panel['n']}  {panel['n_correct']} correct / {panel['n_errors']} errors  "
          f"({panel['correct_base_rate']:.1%} positive in the CORRECT class)")
    print(f"reference {payload['reference']['display']}: tau={ref['tau']:.6g} "
          f"err-P {ref['error_precision']:.4f} err-R {ref['error_recall']:.4f} "
          f"err-F1 {ref['error_f1']:.4f}  ({payload['reference']['distinct_scores']} cuts)")
    print(f"max-t critical {payload['multiplicity']['max_t_critical_value']:.4f} "
          f"(pointwise {payload['multiplicity']['pointwise_normal_critical_value']:.4f}, "
          f"Bonferroni {payload['multiplicity']['bonferroni_critical_value']:.4f})\n")
    print(f"{'arm':<20}{'cuts':>6}{'tau':>9}{'err-F1':>9}{'delta':>9}{'t':>7}"
          f"{'pointwise 95%':>22}{'simultaneous':>22}")
    for arm in payload["arms"]:
        point = arm["operating_point"]
        pointwise = "[{:+.4f}, {:+.4f}]".format(arm["ci95_low"], arm["ci95_high"])
        band = ("[{:+.4f}, {:+.4f}]".format(arm["simultaneous_low"], arm["simultaneous_high"])
                if arm["simultaneous_low"] is not None else "— (not in family)")
        print(f"{arm['display']:<20}{arm['distinct_scores']:>6}{point['tau']:>9.4f}"
              f"{point['error_f1']:>9.4f}{arm['delta_error_f1']:>+9.4f}"
              f"{arm['t_statistic']:>7.2f}{pointwise:>22}{band:>22}"
              + ("  EXCLUDES ZERO" if arm["excludes_zero_simultaneous"] else ""))
    print(f"\nsecond cut — error recall >= "
          f"{payload['arms'][0]['matched_recall']['target_error_recall']:.0%}, "
          f"reference RE-CUT at each row's achieved recall:")
    print(f"{'arm':<20}{'tau':>9}{'r achieved':>12}{'err-F1':>9}"
          f"{'ref tau':>9}{'ref r':>8}{'ref F1':>9}{'delta':>9}"
          f"{'(unmatched)':>13}")
    for arm in payload["arms"]:
        m = arm["matched_recall"]
        ref = m["reference_at_this_rows_recall"]
        print(f"{arm['display']:<20}{m['tau']:>9.4f}{m['error_recall']:>12.4f}"
              f"{m['error_f1']:>9.4f}{ref['tau']:>9.4f}"
              f"{m['reference_error_recall_at_this_row']:>8.4f}"
              f"{ref['error_f1']:>9.4f}"
              f"{m['delta_error_f1_at_matched_recall']:>+9.4f}"
              f"{m['delta_error_f1_each_side_at_its_own_target_cut']:>+13.4f}")
    print(f"\nreconciliation vs {payload['reconciliation']['source']}: worst residual "
          f"{payload['reconciliation']['worst_residual']:.6f} "
          f"(tolerance {payload['reconciliation']['tolerance']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-json", default=OUT_DEFAULT)
    parser.add_argument("--manifest", default=MANIFEST_DEFAULT,
                        help="run manifest.json to record the output path + sha256 in")
    parser.add_argument("--check", action="store_true",
                        help="re-derive from the inputs and exit non-zero on any "
                             "mismatch against the committed artifact, its manifest "
                             "sha256, or the verified numbers")
    args = parser.parse_args()

    payload, guards = build_payload()
    text = render(payload)
    digest = hashlib.sha256(text.encode()).hexdigest()
    out = Path(args.out_json)

    # Numbers first, prose guards after: a regression should read as the number
    # that moved, not as a disclosure sentence that stopped applying to it.
    failures = verify_expected(payload) + guards

    if args.check:
        report(payload)
        if not out.exists():
            failures.append(f"{out} does not exist; run the script without --check first")
        else:
            committed = out.read_text()
            if committed != text:
                failures.append(
                    f"{out} does not reproduce byte-for-byte from its inputs "
                    f"(committed sha256 {hashlib.sha256(committed.encode()).hexdigest()}, "
                    f"re-derived {digest})")
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            recorded = manifest.get("output_sha256", {}).get(out.name)
            if recorded != digest:
                failures.append(
                    f"{manifest_path} records sha256 {recorded} for {out.name}, "
                    f"re-derived {digest}")
        else:
            failures.append(f"{manifest_path} does not exist")

        if failures:
            print(f"\nCHECK FAILED ({len(failures)}):", file=sys.stderr)
            for line in failures:
                print(f"  - {line}", file=sys.stderr)
            return 1
        print(f"\nCHECK OK — {out} reproduces from its inputs and every verified "
              f"number matches\nsha256 {digest}")
        return 0

    if failures:
        print(f"\nREFUSING TO WRITE ({len(failures)} verified numbers do not reproduce):",
              file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("outputs", {})[MANIFEST_OUTPUT_KEY] = out.name
    manifest.setdefault("output_sha256", {})[out.name] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report(payload)
    print(f"\nwrote {out} ({out.stat().st_size} bytes)\nsha256 {digest}")
    print(f"recorded sha256 in {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
