#!/usr/bin/env python3
"""Extended Table 6: the 2023 INDRA paper's own "all sources, specific" block
with OUR arms interleaved into the SAME ranked list, computed by the paper's own
estimator and reported in the paper's own convention.

WHY THIS ARTIFACT EXISTS.  The viewer already plots the published rows and our
arms, but in SEPARATE BANDS (PaperOwnMetric.svelte), which is exactly what hides
the fact that our arms occupy ranks 1-3 of the paper's own table.  One ordered
list, one estimator, one convention -- and every caveat that comes with it
carried as a field rather than as prose.

THE PAPER'S ESTIMATOR, verbatim from their released notebook
(sorgerlab/indra_assembly_paper @63abdf1, "Training Belief ML Models.ipynb",
sha256 3bd1a684...):

    precision_recall_curve(y_fold, p_fold) -> auc(recall, precision)   # per fold
    auc_mean = np.mean(pr_aucs);  auc_sd = np.std(pr_aucs)             # population SD

``auc_sd`` is a DISPERSION statistic over ten folds, NOT a confidence interval,
and the paper reports no test of any kind for any of its 59 rows.  Both facts
are carried in the payload (``metric_contract``, ``paper_reports_no_tests``)
because the ranked list is otherwise easy to read as if it were adjudicated.

THREE KINDS OF ROW, and they are not interchangeable:
  * ``paper_rerun``           -- the paper's own code re-run by us on the paper's
                                 own panel and folds; 10 rows.  Eight of them
                                 also carry out-of-fold score vectors, so they
                                 get a tie-robust pooled average precision; the
                                 two unregularised baselines (plain Log LR, plain
                                 RF) were not re-run with released OOF vectors,
                                 so they get the paper metric and nothing else.
  * ``paper_published_only``  -- Belief Orig, KNN(+Type/#PMIDs), SVC(+Type/#PMIDs).
                                 We never re-ran these, so there is no score
                                 vector and there can never be an AP or a tie
                                 gift for them.  Enforced, not just documented.
  * ``ours``                  -- the four LLM reader arms plus the deployed INDRA
                                 CoGEx hybrid, scored on the identical panel and
                                 assigned to the paper's identical folds.

THE TIE DISCLOSURE.  The paper's estimator is trapezoidal, which over-credits
heavily-tied score distributions.  The gift (paper metric minus pooled AP) is
+0.0097..+0.0143 for our four reader arms and -0.0008..+0.0006 for every paper
model -- which reads like an authorship effect until you notice that OUR OWN
INDRA CoGEx hybrid, at 1176 distinct scores, collects only +0.0006.  The gift
tracks TIE DENSITY, not who wrote the model, and ``tie_disclosure`` records that
reconciliation as first-class fields.  Tie-corrected, our best arm still leads
their best arm, by +0.0090 instead of +0.0231.

Reuse, not reimplementation: the join contract (gold file, panel order, fold
assignment) and the fold-mean estimator are imported from
scripts/compare_paper_literal_vs_llms.py, the mid-rank ranker behind the tie
correlation from indra_belief.metrics, and the pinned notebook digest from
scripts/extract_indra_paper_method_metrics.py, so this artifact cannot drift
from the head-to-head it extends.  Every arm is addressed here by a STABLE key
(see ``OUR_ARMS``): the bundle directory is declared, the head-to-head's own
display-shaped key is pinned as data, and both are cross-checked against
``LLM_ARMS`` rather than being inferred from anything printed on screen.

Usage:
    uv run python scripts/compute_paper_table6_extended.py            # write
    uv run python scripts/compute_paper_table6_extended.py --check    # gate
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The head-to-head's join contract, verbatim: same gold, same panel order, same
# fold assignment, same arm bundles, same estimator.
from compare_paper_literal_vs_llms import (  # noqa: E402
    GOLD,
    HEADLINE,
    LLM_ARMS,
    MODELS_DIR,
    fold_mean_pr_auc,
    load_jsonl,
)

# The pinned notebook digest, from the extractor that froze the published table.
from extract_indra_paper_method_metrics import (  # noqa: E402
    EXPECTED_NOTEBOOK_SHA256,
    NOTEBOOK_PATH,
)

# The repo's tie-averaging ranker, shared with auroc(); ranks with ties averaged
# are what makes the tie-density correlation below an actual Spearman rho.
from indra_belief.metrics import _rankdata_avg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data/results/indra_paper_literal_models_20260724"
DEFAULT_LITERAL = RUN_DIR / "paper_literal_table6_and_oof.json"
DEFAULT_COMPARISON = RUN_DIR / "paper_literal_vs_llms.json"
DEFAULT_OUT = RUN_DIR / "paper_table6_extended.json"
PUBLISHED = ROOT / "data/benchmark/indra_paper_2023_published_method_metrics.json"
BENCHMARK_MANIFEST = ROOT / "data/benchmark/indra_paper_2023.manifest.json"

# The one Table 6 block every arm here is scored on.
CONFIG = "all sources, specific"

# Stable join keys.  ``label`` is the key, ``display`` is the screen string, and
# they are decoupled on purpose: a display string must never become a join key.
# Keyed on the paper's OWN method identity (the string their table prints), which
# is how compare_paper_literal_vs_llms.py already joins to this run.
PAPER_METHODS: dict[str, tuple[str, str]] = {
    # paper method (minus the " - <config>" suffix): (label, family)
    "Belief Orig": ("paper_belief_orig", "paper_unfitted_belief"),
    "RF 2k-d13": ("paper_rf", "paper_fitted_ml"),
    "RF 2k-d13 + Type/#PMIDs": ("paper_rf_type_pmids", "paper_fitted_ml"),
    "RF 2k-d13 + Type/#PMIDs/avglen": ("paper_rf_avglen", "paper_fitted_ml"),
    "RF 2k-d13 + Type/#PMIDs/promoter": ("paper_rf_promoter", "paper_fitted_ml"),
    "RF 2k-d13 + Type/#PMIDs/prom/avglen": ("paper_rf_prom_avglen", "paper_fitted_ml"),
    "Log LR": ("paper_loglr", "paper_fitted_ml"),
    "Log LR + Type/#PMIDs": ("paper_loglr_type_pmids", "paper_fitted_ml"),
    "Log LR + Type/#PMIDs/avglen": ("paper_loglr_avglen", "paper_fitted_ml"),
    "Log LR + Type/#PMIDs/promoter": ("paper_loglr_promoter", "paper_fitted_ml"),
    "Log LR + Type/#PMIDs/prom/avglen": ("paper_loglr_prom_avglen", "paper_fitted_ml"),
    "KNN": ("paper_knn", "paper_fitted_ml"),
    "KNN + Type/#PMIDs": ("paper_knn_type_pmids", "paper_fitted_ml"),
    "SVC": ("paper_svc", "paper_fitted_ml"),
    "SVC + Type/#PMIDs": ("paper_svc_type_pmids", "paper_fitted_ml"),
}

# Our arms.  The dict KEY is the stable label -- the one this artifact emits and
# the one every gate below looks up -- and NOTHING here is joined on a screen
# string:
#   * ``display``         is the name we print.  Change it freely; no lookup moves.
#   * ``model_key``       is the prediction-bundle directory under MODELS_DIR, i.e.
#                         the join into the score files.  It is ours and stable.
#   * ``head_to_head_key`` is the one token that still has a display SHAPE, and it
#                         is declared as data for exactly that reason: the already
#                         shipped paper_literal_vs_llms.json keys its
#                         ``point_metrics`` on the head-to-head's own display
#                         names (compare_paper_literal_vs_llms.py::LLM_ARMS), so
#                         reading that frozen artifact needs its key, not ours.
#                         Pinning it here keeps the shipped artifact's key space
#                         separate from what this file prints; the guards in the
#                         row loop fail closed if either side moves.
class OurArm(NamedTuple):
    display: str
    family: str
    model_key: str
    head_to_head_key: str


OUR_ARMS: dict[str, OurArm] = {
    "ours_glm_5": OurArm("GLM-5", "llm_reader", "glm_5", "GLM-5"),
    "ours_gemma_4_26b": OurArm("Gemma 4 26B", "llm_reader", "gemma_4_26b", "Gemma 4 26B"),
    "ours_gemma_4_31b": OurArm("Gemma 4 31B", "llm_reader", "gemma_4_31b", "Gemma 4 31B"),
    "ours_gemma_4_e2b": OurArm("Gemma 4 E2B", "llm_reader", "gemma_4_e2b", "Gemma 4 E2B"),
    "ours_indra_cogex_hybrid": OurArm(
        "INDRA CoGEx hybrid", "indra_hybrid", "indra_cogex_hybrid", "INDRA CoGEx hybrid"),
}

# The arm the AP head-to-head differences against.
REFERENCE_LABEL = "paper_rf_promoter"

# ---------------------------------------------------------------------------
# The gate.  These were measured before this script existed; the script's job is
# to reproduce them, and to fail loudly rather than emit a plausible file.
# (label, rank, fold-mean rounded to 4dp, fold SD rounded to 4dp or None)
EXPECTED_RANKS: list[tuple[str, int, float, float | None]] = [
    ("ours_glm_5", 1, 0.9649, 0.0103),
    ("ours_gemma_4_26b", 2, 0.9610, 0.0111),
    ("ours_gemma_4_31b", 3, 0.9601, 0.0111),
    ("paper_rf_prom_avglen", 4, 0.9418, 0.0149),
    ("paper_rf_promoter", 5, 0.9413, 0.0140),
    ("ours_gemma_4_e2b", 11, 0.9349, 0.0174),
    ("paper_belief_orig", 16, 0.9230, None),
    ("paper_svc", 20, 0.8950, None),
]
EXPECTED_N_ROWS = 20
# "Every re-run row matches its published value to <=0.0016."
TOL_ABS_DEV_VS_PUBLISHED = 0.0016
# Our four LLM reader arms collect this much trapezoidal gift; every paper model
# collects essentially none.  Bounds are on the 4dp-rounded gift.
EXPECTED_READER_GIFT = (0.0097, 0.0143)
EXPECTED_PAPER_GIFT = (-0.0008, 0.0006)
EXPECTED_READER_DISTINCT = (420, 498)
EXPECTED_PAPER_DISTINCT = (1521, 1681)
EXPECTED_HYBRID_DISTINCT = 1176
EXPECTED_HYBRID_GIFT = 0.0006
# Tie-corrected best against best.
EXPECTED_OUR_BEST_AP = 0.9510
EXPECTED_THEIR_BEST_AP = 0.9420
EXPECTED_TIE_CORRECTED_MARGIN = 0.0090
# The paper's own headline gain, in its own convention.
EXPECTED_PAPER_HEADLINE_GAIN = 0.0190
EXPECTED_PAPER_HEADLINE_GAIN_SD = 1.36
EXPECTED_PUBLISHED_ROW_COUNT = 59
EXPECTED_SHA256_AGREEMENTS = 3
# Our re-run reproduces the paper's estimator from the released OOF vectors.
TOL_ESTIMATOR = 1e-12
# Our arms must equal the shipped head-to-head exactly.
TOL_VS_SHIPPED = 1e-12


def rank_key(row: dict) -> tuple:
    """Deterministic total order: fold mean descending, then the tighter fold SD,
    then the stable label.  Same rule the viewer's ``rankPublished`` already uses
    (viewer/src/lib/data/paper-method-landscape.ts), so the artifact and the
    existing published-row ranking cannot disagree about ties."""
    return (-row["fold_mean_trapezoidal_pr_auc"], row["fold_population_sd"],
            row["label"])


def r4(x: float) -> float:
    return round(x, 4)


def build() -> tuple[dict, list[str]]:
    """Return (payload, printable table lines).  Every gate assertion fires here,
    so --check and the writing path are gated identically."""
    notes: list[str] = []

    lit = json.loads(DEFAULT_LITERAL.read_text())
    pub = json.loads(PUBLISHED.read_text())
    bench = json.loads(BENCHMARK_MANIFEST.read_text())
    shipped = json.loads(DEFAULT_COMPARISON.read_text())

    # ---- the panel, in the head-to-head's own order ------------------------
    oof = {r["stmt_hash"]: r for r in lit["oof_predictions"][HEADLINE]}
    gold = {}
    for r in load_jsonl(ROOT / GOLD):
        h = int(r["paper_statement_hash"])
        gold[h] = {"sid": r["canonical_corpus"]["statement_id"],
                   "label": r["paper_replication_policy"]["released_paper_correct"]}
    hashes = sorted(oof)
    sids = [gold[h]["sid"] for h in hashes]
    y = np.array([oof[h]["y_true"] for h in hashes])
    folds = np.array([oof[h]["fold_ix"] for h in hashes])
    assert all(oof[h]["y_true"] == gold[h]["label"] for h in hashes), "label mismatch"
    n = len(hashes)
    n_folds = len(set(folds.tolist()))

    # ---- the published block ------------------------------------------------
    suffix = f" - {CONFIG}"
    published = {}
    for m in pub["methods"]:
        if not m["method"].endswith(suffix):
            continue
        stem = m["method"][: -len(suffix)]
        assert stem in PAPER_METHODS, (
            f"published method {m['method']!r} has no stable label; the extended "
            "table would silently drop or mis-key it")
        published[stem] = m
    missing = sorted(set(PAPER_METHODS) - set(published))
    assert not missing, (
        f"{len(missing)} labelled paper methods are absent from {PUBLISHED.name} "
        f"for config {CONFIG!r}: {missing}")

    rows: list[dict] = []

    def paper_row(stem: str, *, origin: str, mean: float, sd: float,
                  fold_list: list[float] | None) -> dict:
        label, family = PAPER_METHODS[stem]
        p = published[stem]
        return {
            "label": label,
            "display": stem,
            "origin": origin,
            "family": family,
            "paper_method": p["method"],
            "method_id": p["method_id"],
            "paper_table_id": p["table_id"],
            "model_key": None,
            "fold_mean_trapezoidal_pr_auc": mean,
            "fold_population_sd": sd,
            "fold_count": p["fold_count"],
            "folds": fold_list,
            "published_mean": p["fold_mean_trapezoidal_pr_auc"],
            "published_fold_population_sd": p["fold_population_sd"],
            "abs_dev_vs_published": (
                None if origin == "paper_published_only"
                else abs(mean - p["fold_mean_trapezoidal_pr_auc"])),
            "pooled_average_precision": None,
            "distinct_scores": None,
            "tie_gift": None,
            "has_out_of_fold_scores": False,
            "is_reference_arm": label == REFERENCE_LABEL,
        }

    # ---- rows 1: the paper's methods re-run by us ---------------------------
    rerun_stems: list[str] = []
    n_rerun_with_oof = 0
    for t in lit["table6"]:
        if not t["method"].endswith(suffix):
            continue
        stem = t["method"][: -len(suffix)]
        assert stem in PAPER_METHODS, f"re-run method {t['method']!r} has no label"
        rerun_stems.append(stem)
        row = paper_row(stem, origin="paper_rerun",
                        mean=t["fold_mean_trapezoidal_pr_auc"],
                        sd=t["fold_population_sd"],
                        fold_list=[float(v) for v in t["folds"]])
        # The reported summary IS the paper's estimator applied to the reported
        # folds: mean and POPULATION sd (np.std, ddof=0).
        assert abs(float(np.mean(row["folds"])) - row["fold_mean_trapezoidal_pr_auc"]) <= TOL_ESTIMATOR
        assert abs(float(np.std(row["folds"])) - row["fold_population_sd"]) <= TOL_ESTIMATOR

        if t["method"] in lit["oof_predictions"]:
            o = {r["stmt_hash"]: r for r in lit["oof_predictions"][t["method"]]}
            p_vec = np.array([o[h]["prob_correct"] for h in hashes])
            assert all(o[h]["y_true"] == oof[h]["y_true"] for h in hashes), (
                f"{t['method']}: out-of-fold labels disagree with the panel")
            assert all(o[h]["fold_ix"] == oof[h]["fold_ix"] for h in hashes), (
                f"{t['method']}: out-of-fold fold assignment differs from the panel")
            fm, fold_aucs = fold_mean_pr_auc(y, p_vec, folds)
            # The paper's estimator, re-derived from the released score vectors,
            # must reproduce the table this run published for itself.
            assert abs(fm - row["fold_mean_trapezoidal_pr_auc"]) <= TOL_ESTIMATOR, (
                f"{t['method']}: re-derived fold mean {fm!r} != table6 "
                f"{row['fold_mean_trapezoidal_pr_auc']!r}")
            assert max(abs(a - b) for a, b in zip(fold_aucs, row["folds"])) <= TOL_ESTIMATOR
            ap = float(average_precision_score(y, p_vec))
            row["pooled_average_precision"] = ap
            row["distinct_scores"] = int(len(np.unique(p_vec)))
            row["tie_gift"] = row["fold_mean_trapezoidal_pr_auc"] - ap
            row["has_out_of_fold_scores"] = True
            n_rerun_with_oof += 1
        rows.append(row)

    assert len(rerun_stems) == 10, (
        f"expected 10 re-run {CONFIG!r} rows, got {len(rerun_stems)}")
    notes.append(
        f"{len(rerun_stems)} re-run rows, {n_rerun_with_oof} of them with released "
        "out-of-fold score vectors (the two unregularised baselines, plain Log LR "
        "and plain RF, have none, so they carry the paper metric and no AP)")

    # ---- rows 2: published only, and they can never acquire an AP -----------
    for stem in PAPER_METHODS:
        if stem in rerun_stems:
            continue
        p = published[stem]
        rows.append(paper_row(stem, origin="paper_published_only",
                              mean=p["fold_mean_trapezoidal_pr_auc"],
                              sd=p["fold_population_sd"],
                              fold_list=None))

    # ---- rows 3: our arms ---------------------------------------------------
    # Coverage guard: every arm the head-to-head scores gets a row here, or the
    # extended table silently drops one.  Compared on the DECLARED head-to-head
    # keys, never on anything this file prints.
    assert {a.head_to_head_key for a in OUR_ARMS.values()} == set(LLM_ARMS), (
        "OUR_ARMS and compare_paper_literal_vs_llms.py::LLM_ARMS disagree about "
        f"which arms exist: ours {sorted(a.head_to_head_key for a in OUR_ARMS.values())} "
        f"vs head-to-head {sorted(LLM_ARMS)}")
    for label, spec in OUR_ARMS.items():
        # The bundle path is our own stable model_key; LLM_ARMS is consulted only
        # to prove the two maps still name the same directory for this arm.
        arm = spec.model_key
        assert LLM_ARMS[spec.head_to_head_key] == arm, (
            f"{label}: head-to-head arm directory {LLM_ARMS[spec.head_to_head_key]!r} "
            f"!= our model_key {arm!r}; one of the two maps moved")
        pj = {r["statement_id"]: r["probability_correct"]
              for r in load_jsonl(ROOT / f"{MODELS_DIR}/{arm}/all_source_predictions.jsonl")}
        p_vec = np.array([pj[s] for s in sids])
        fm, fold_aucs = fold_mean_pr_auc(y, p_vec, folds)
        sd = float(np.std(fold_aucs))
        ap = float(average_precision_score(y, p_vec))
        distinct = int(len(np.unique(p_vec)))
        # Drift guard: identical to the shipped head-to-head, or this artifact is
        # telling a different story about the same arm.  The shipped file's own
        # key space is display-shaped; we read it through the pinned
        # head_to_head_key and fail closed rather than KeyError-ing on a rename.
        assert spec.head_to_head_key in shipped["point_metrics"], (
            f"{label}: {DEFAULT_COMPARISON.name} has no point_metrics entry keyed "
            f"{spec.head_to_head_key!r}; the shipped head-to-head's key space "
            f"moved, so OUR_ARMS[{label!r}].head_to_head_key must be re-pinned")
        s = shipped["point_metrics"][spec.head_to_head_key]
        for field, got in (("fold_mean_trapezoidal_pr_auc", fm),
                           ("fold_population_sd", sd),
                           ("pooled_average_precision", ap)):
            assert abs(got - s[field]) <= TOL_VS_SHIPPED, (
                f"{label}: {field} {got!r} != shipped {s[field]!r} in "
                f"{DEFAULT_COMPARISON.name}")
        assert distinct == s["distinct_scores"], (
            f"{label}: distinct_scores {distinct} != shipped {s['distinct_scores']}")
        rows.append({
            "label": label,
            "display": spec.display,
            "origin": "ours",
            "family": spec.family,
            "paper_method": None,
            "method_id": None,
            "paper_table_id": None,
            "model_key": arm,
            "fold_mean_trapezoidal_pr_auc": fm,
            "fold_population_sd": sd,
            "fold_count": n_folds,
            "folds": [float(a) for a in fold_aucs],
            "published_mean": None,
            "published_fold_population_sd": None,
            "abs_dev_vs_published": None,
            "pooled_average_precision": ap,
            "distinct_scores": distinct,
            "tie_gift": fm - ap,
            "has_out_of_fold_scores": True,
            "is_reference_arm": False,
        })

    # ---- rank, interleaved, one list ----------------------------------------
    rows.sort(key=rank_key)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    with_ap = sorted([r for r in rows if r["pooled_average_precision"] is not None],
                     key=lambda r: (-r["pooled_average_precision"], r["label"]))
    ap_rank = {r["label"]: i for i, r in enumerate(with_ap, 1)}
    for row in rows:
        row["pooled_ap_rank"] = ap_rank.get(row["label"])

    labels = [r["label"] for r in rows]
    assert len(set(labels)) == len(labels) == EXPECTED_N_ROWS, (
        f"expected {EXPECTED_N_ROWS} uniquely labelled rows, got {len(labels)} "
        f"({len(set(labels))} unique)")

    # ---- acceptance: published-only rows can NEVER acquire an AP ------------
    for row in rows:
        if row["origin"] != "paper_published_only":
            continue
        for field in ("pooled_average_precision", "distinct_scores", "tie_gift",
                      "folds", "abs_dev_vs_published", "pooled_ap_rank"):
            assert row[field] is None, (
                f"{row['label']}: published-only row carries {field}="
                f"{row[field]!r}; we never re-ran this method, so it has no score "
                "vector and must not be given a tie-robust number")
        assert row["has_out_of_fold_scores"] is False

    by_label = {r["label"]: r for r in rows}

    # ---- gate: the ranked numbers -------------------------------------------
    for label, rank, mean, sd in EXPECTED_RANKS:
        row = by_label[label]
        assert row["rank"] == rank, (
            f"{label}: rank {row['rank']} != expected {rank}")
        assert r4(row["fold_mean_trapezoidal_pr_auc"]) == mean, (
            f"{label}: fold mean {row['fold_mean_trapezoidal_pr_auc']!r} rounds to "
            f"{r4(row['fold_mean_trapezoidal_pr_auc'])} != expected {mean}")
        if sd is not None:
            assert r4(row["fold_population_sd"]) == sd, (
                f"{label}: fold SD {row['fold_population_sd']!r} rounds to "
                f"{r4(row['fold_population_sd'])} != expected {sd}")

    devs = {r["label"]: r["abs_dev_vs_published"] for r in rows
            if r["abs_dev_vs_published"] is not None}
    max_dev_label = max(devs, key=lambda k: devs[k])
    max_dev = devs[max_dev_label]
    assert len(devs) == 10, f"expected 10 re-run deviations, got {len(devs)}"
    assert max_dev <= TOL_ABS_DEV_VS_PUBLISHED, (
        f"{max_dev_label} deviates from its published value by {max_dev:.5f} > "
        f"{TOL_ABS_DEV_VS_PUBLISHED}")

    # ---- the tie disclosure --------------------------------------------------
    reader = [r for r in rows if r["family"] == "llm_reader"]
    paper_scored = [r for r in rows if r["origin"] == "paper_rerun"
                    and r["tie_gift"] is not None]
    hybrid = by_label["ours_indra_cogex_hybrid"]

    reader_gift = (min(r["tie_gift"] for r in reader),
                   max(r["tie_gift"] for r in reader))
    paper_gift = (min(r["tie_gift"] for r in paper_scored),
                  max(r["tie_gift"] for r in paper_scored))
    reader_distinct = (min(r["distinct_scores"] for r in reader),
                       max(r["distinct_scores"] for r in reader))
    paper_distinct = (min(r["distinct_scores"] for r in paper_scored),
                      max(r["distinct_scores"] for r in paper_scored))
    assert (r4(reader_gift[0]), r4(reader_gift[1])) == EXPECTED_READER_GIFT, (
        f"reader tie gift {r4(reader_gift[0])}..{r4(reader_gift[1])} != "
        f"{EXPECTED_READER_GIFT}")
    assert (r4(paper_gift[0]), r4(paper_gift[1])) == EXPECTED_PAPER_GIFT, (
        f"paper tie gift {r4(paper_gift[0])}..{r4(paper_gift[1])} != "
        f"{EXPECTED_PAPER_GIFT}")
    assert reader_distinct == EXPECTED_READER_DISTINCT, reader_distinct
    assert paper_distinct == EXPECTED_PAPER_DISTINCT, paper_distinct
    assert hybrid["distinct_scores"] == EXPECTED_HYBRID_DISTINCT
    assert r4(hybrid["tie_gift"]) == EXPECTED_HYBRID_GIFT, r4(hybrid["tie_gift"])

    # The reconciliation, measured rather than asserted: across every row that
    # has a score vector, the gift falls as the score distribution gets finer.
    # Spearman needs MID-RANKS: two pairs of rows tie on distinct_scores here
    # (1546 twice, 1681 twice), and rank-by-argsort breaks those ties arbitrarily,
    # which is not Spearman's rho -- it is Pearson over an arbitrary order.  The
    # repo's own tie-averaging ranker is reused rather than re-derived.
    gifts = np.array([r["tie_gift"] for r in with_ap])
    dcount = np.array([float(r["distinct_scores"]) for r in with_ap])
    rho = float(np.corrcoef(_rankdata_avg(dcount), _rankdata_avg(gifts))[0, 1])
    coarse = [r for r in with_ap if r["distinct_scores"] <= EXPECTED_READER_DISTINCT[1]]
    fine = [r for r in with_ap if r["distinct_scores"] > EXPECTED_READER_DISTINCT[1]]
    coarse_min_gift = min(r["tie_gift"] for r in coarse)
    fine_max_abs_gift = max(abs(r["tie_gift"]) for r in fine)
    assert coarse_min_gift > fine_max_abs_gift, (
        "the tie gift no longer separates coarse from fine score distributions; "
        "the tie-density reading of the gift is no longer supported")
    assert hybrid["origin"] == "ours" and hybrid["label"] in {r["label"] for r in fine}, (
        "the INDRA CoGEx hybrid is the row that separates tie density from "
        "authorship; it must be ours AND fine-grained")

    our_best = max([r for r in with_ap if r["origin"] == "ours"],
                   key=lambda r: r["pooled_average_precision"])
    their_best = max([r for r in with_ap if r["origin"] != "ours"],
                     key=lambda r: r["pooled_average_precision"])
    tie_corrected_margin = (our_best["pooled_average_precision"]
                            - their_best["pooled_average_precision"])
    our_best_paper = min([r for r in rows if r["origin"] == "ours"], key=rank_key)
    their_best_paper = min([r for r in rows if r["origin"] != "ours"], key=rank_key)
    paper_metric_margin = (our_best_paper["fold_mean_trapezoidal_pr_auc"]
                           - their_best_paper["fold_mean_trapezoidal_pr_auc"])
    assert r4(our_best["pooled_average_precision"]) == EXPECTED_OUR_BEST_AP
    assert r4(their_best["pooled_average_precision"]) == EXPECTED_THEIR_BEST_AP
    assert r4(tie_corrected_margin) == EXPECTED_TIE_CORRECTED_MARGIN, (
        f"tie-corrected margin {r4(tie_corrected_margin)} != "
        f"{EXPECTED_TIE_CORRECTED_MARGIN}")
    reference = by_label[REFERENCE_LABEL]
    head_to_head_ap_delta = (by_label["ours_gemma_4_26b"]["pooled_average_precision"]
                             - reference["pooled_average_precision"])
    # The prose in tie_corrected_best_vs_best claims these two land within 0.001
    # of each other; the claim is gated, not asserted.
    assert abs(tie_corrected_margin - head_to_head_ap_delta) <= 0.001, (
        f"tie-corrected best-vs-best margin {tie_corrected_margin:+.6f} and the "
        f"head-to-head AP delta {head_to_head_ap_delta:+.6f} are further apart "
        "than the payload's own note says")

    # ---- the paper's reporting convention -----------------------------------
    digests = {
        str(PUBLISHED.relative_to(ROOT)): pub["source"]["notebook_sha256"],
        str(BENCHMARK_MANIFEST.relative_to(ROOT)):
            bench["paper_code"]["critical_file_sha256"][NOTEBOOK_PATH],
        "scripts/extract_indra_paper_method_metrics.py": EXPECTED_NOTEBOOK_SHA256,
    }
    agreeing = sorted(k for k, v in digests.items() if v == EXPECTED_NOTEBOOK_SHA256)
    assert len(agreeing) == EXPECTED_SHA256_AGREEMENTS == len(digests), (
        f"only {len(agreeing)} of {len(digests)} pinned locations agree on the "
        f"notebook digest: {digests}")
    assert pub["method_count"] == EXPECTED_PUBLISHED_ROW_COUNT == len(pub["methods"])
    assert pub["metric_contract"]["uncertainty_is_confidence_interval"] is False
    # No published row carries a test statistic of any kind -- verified against
    # the frozen extraction, not asserted from memory.
    test_fields = {"p_value", "p", "ci95_low", "ci95_high", "ci_low", "ci_high",
                   "confidence_interval", "stderr", "se", "t", "z", "q_value"}
    stray = sorted({k for m in pub["methods"] for k in m} & test_fields)
    assert not stray, f"published rows carry test-like fields: {stray}"

    fitted = sorted((m for stem, m in published.items()
                     if PAPER_METHODS[stem][1] == "paper_fitted_ml"),
                    key=lambda m: (-m["fold_mean_trapezoidal_pr_auc"],
                                   m["fold_population_sd"], m["method_id"]))
    best_fitted = fitted[0]
    belief_orig = published["Belief Orig"]
    headline_gain = (best_fitted["fold_mean_trapezoidal_pr_auc"]
                     - belief_orig["fold_mean_trapezoidal_pr_auc"])
    headline_gain_sd = headline_gain / best_fitted["fold_population_sd"]
    assert r4(headline_gain) == EXPECTED_PAPER_HEADLINE_GAIN, r4(headline_gain)
    assert round(headline_gain_sd, 2) == EXPECTED_PAPER_HEADLINE_GAIN_SD, (
        f"paper headline gain {headline_gain_sd:.4f} SD != "
        f"{EXPECTED_PAPER_HEADLINE_GAIN_SD}")

    payload = {
        "artifact_kind": "paper_table6_extended_all_sources_specific",
        "schema_version": 1,
        "config": CONFIG,
        "what_this_is": (
            "The 2023 INDRA paper's own Table 6 block for the "
            f"{CONFIG!r} configuration, with our arms added as rows and the whole "
            "list ranked together by the paper's own estimator. Our arms occupy "
            "ranks 1-3; the published table's own best fitted model is rank 4."),
        "n_statements": n,
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_folds": n_folds,
        "n_rows": len(rows),
        "ranking_rule": (
            "fold_mean_trapezoidal_pr_auc descending, then the tighter "
            "fold_population_sd, then the stable label; a total order, so rank "
            "never depends on row order (same rule as the viewer's rankPublished)"),
        "metric_contract": {
            "per_fold_metric": pub["metric_contract"]["per_fold_metric"],
            "summary": pub["metric_contract"]["summary"],
            "uncertainty_field": pub["metric_contract"]["uncertainty_field"],
            "uncertainty_is_confidence_interval": False,
            "uncertainty_is_dispersion_not_a_confidence_interval": True,
            "uncertainty_note": (
                "fold_population_sd is np.std over the ten fold-wise areas: a "
                "DISPERSION statistic describing how much the estimate moves "
                "between folds. It is not a confidence interval, it does not "
                "shrink with the number of folds, and no interval in this file "
                "may be read as a test."),
            "metric_is_pooled_average_precision": False,
            "estimator_source": (
                "sorgerlab/indra_assembly_paper@"
                f"{bench['paper_code']['commit']} :: {NOTEBOOK_PATH} :: "
                "precision_recall_curve -> auc(recall, precision) per fold, "
                "np.mean / np.std across folds"),
            "trapezoidal_note": (
                "auc(recall, precision) interpolates linearly between adjacent "
                "operating points, which over-credits heavily-tied score "
                "distributions. See tie_disclosure: it is worth up to +0.0143 to "
                "a coarse-scored arm and essentially nothing to a fine-scored "
                "one, whoever wrote it."),
        },
        "origins": {
            "paper_rerun": (
                "the paper's own released code, re-run by us on the paper's own "
                "panel, labels and folds"),
            "paper_published_only": (
                "printed in the paper's own tables; never re-run here, so there "
                "is no score vector and no tie-robust number can exist for it"),
            "ours": (
                "our arms, scored on the identical panel and assigned to the "
                "paper's identical folds"),
        },
        "rows": rows,
        "reproduction_fidelity": {
            "n_rerun_rows": len(rerun_stems),
            "n_rerun_rows_with_out_of_fold_scores": n_rerun_with_oof,
            "max_abs_dev_vs_published": max_dev,
            "max_abs_dev_row": max_dev_label,
            "tolerance": TOL_ABS_DEV_VS_PUBLISHED,
            "published_values_are_rounded_to_3dp": True,
            "note": (
                "Published means and SDs are printed to three decimals in the "
                "paper's own table, so abs_dev_vs_published is bounded below by "
                "that rounding; every re-run row is within "
                f"{TOL_ABS_DEV_VS_PUBLISHED} of its published value."),
        },
        "tie_disclosure": {
            "definition": ("tie_gift = fold_mean_trapezoidal_pr_auc (the paper's "
                           "own estimator) - pooled_average_precision (tie-robust)"),
            "llm_reader_arms": {
                "min": reader_gift[0], "max": reader_gift[1],
                "labels": [r["label"] for r in reader],
                "distinct_scores_min": reader_distinct[0],
                "distinct_scores_max": reader_distinct[1],
            },
            "paper_rerun_rows": {
                "min": paper_gift[0], "max": paper_gift[1],
                "labels": [r["label"] for r in paper_scored],
                "distinct_scores_min": paper_distinct[0],
                "distinct_scores_max": paper_distinct[1],
            },
            "indra_cogex_hybrid": {
                "label": hybrid["label"],
                "origin": hybrid["origin"],
                "tie_gift": hybrid["tie_gift"],
                "distinct_scores": hybrid["distinct_scores"],
            },
            "tracks_tie_density_not_authorship": True,
            "reconciliation": (
                "The gift looks like an authorship effect until our own INDRA "
                f"CoGEx hybrid is read: it is ours, it emits "
                f"{hybrid['distinct_scores']} distinct scores over {n} statements, "
                f"and it collects only {hybrid['tie_gift']:+.4f} -- inside the "
                "paper models' range. Every arm with a coarse score distribution "
                "collects the gift and every arm with a fine one does not, "
                "regardless of who built it."),
            "spearman_gift_vs_distinct_scores": rho,
            "spearman_method": (
                "Spearman's rho over MID-RANKS (tied values share their average "
                "rank), computed with indra_belief.metrics._rankdata_avg over the "
                f"{len(with_ap)} rows that have a score vector. distinct_scores "
                "carries ties here, so ranking by argsort-of-argsort would break "
                "them arbitrarily and would not be Spearman."),
            "separation": {
                "coarse_max_distinct_scores": EXPECTED_READER_DISTINCT[1],
                "min_gift_among_coarse_scored_arms": coarse_min_gift,
                "max_abs_gift_among_fine_scored_arms": fine_max_abs_gift,
                "n_coarse": len(coarse),
                "n_fine": len(fine),
                "note": ("No fine-scored arm's |gift| reaches any coarse-scored "
                         "arm's gift; the two groups do not overlap."),
            },
            "tie_corrected_best_vs_best": {
                "our_best_label": our_best["label"],
                "our_best_pooled_average_precision": our_best["pooled_average_precision"],
                "their_best_label": their_best["label"],
                "their_best_pooled_average_precision": their_best["pooled_average_precision"],
                "margin": tie_corrected_margin,
                "our_best_paper_metric_label": our_best_paper["label"],
                "their_best_paper_metric_label": their_best_paper["label"],
                "paper_metric_margin": paper_metric_margin,
                "share_of_paper_metric_margin_that_is_tie_gift": (
                    (paper_metric_margin - tie_corrected_margin) / paper_metric_margin),
                "head_to_head_ap_delta_gemma_4_26b_vs_reference": head_to_head_ap_delta,
                "reference_label": REFERENCE_LABEL,
                "note": (
                    "On the paper's own estimator our best-ranked arm "
                    f"({our_best_paper['label']}) leads their best-ranked "
                    f"({their_best_paper['label']}) by {paper_metric_margin:+.4f}. "
                    "Tie-corrected, our best arm by pooled AP "
                    f"({our_best['label']}) leads theirs ({their_best['label']}) by "
                    f"{tie_corrected_margin:+.4f} -- within 0.001 of the AP delta "
                    "the head-to-head reports for Gemma 4 26B against the paper's "
                    f"RF+promoter reference arm ({head_to_head_ap_delta:+.4f}). The "
                    "lead survives the correction; most of its apparent size does "
                    "not."),
            },
        },
        "paper_reports_no_tests": True,
        "paper_reporting_convention": {
            "n_published_rows": pub["method_count"],
            "reported_statistics": ["fold mean", "population SD over the folds"],
            "n_p_values": 0,
            "n_confidence_intervals": 0,
            "n_multiplicity_corrections": 0,
            "verified": (
                "The frozen extraction of the paper's own tables carries exactly "
                "a mean and a population fold SD per row for all "
                f"{pub['method_count']} rows and no field of any test-like kind."),
            "pinned_source": {
                "repository": bench["paper_code"]["repository"],
                "commit": bench["paper_code"]["commit"],
                "notebook_path": NOTEBOOK_PATH,
                "notebook_sha256": EXPECTED_NOTEBOOK_SHA256,
                "n_sha256_agreements": len(agreeing),
                "agreeing_locations": agreeing,
            },
            "their_own_headline_gain": {
                "best_fitted_method": best_fitted["method"],
                "best_fitted_method_id": best_fitted["method_id"],
                "best_fitted_mean": best_fitted["fold_mean_trapezoidal_pr_auc"],
                "best_fitted_fold_population_sd": best_fitted["fold_population_sd"],
                "belief_orig_mean": belief_orig["fold_mean_trapezoidal_pr_auc"],
                "belief_orig_fold_population_sd": belief_orig["fold_population_sd"],
                "gain": headline_gain,
                "gain_in_fold_sd": headline_gain_sd,
                "tie_break": (
                    "two published rows tie at the best value; the tighter fold "
                    "SD wins, matching this file's ranking rule"),
                "note": (
                    "The paper's own headline improvement over its unfitted "
                    f"Belief Orig baseline, same configuration, is {headline_gain:+.4f} "
                    f"= {headline_gain_sd:.2f} fold SD, reported with no test. It "
                    "is the scale against which every row in this table should be "
                    "read."),
            },
        },
        "checks": {
            "expected_ranks": [
                {"label": label, "rank": rank, "fold_mean_4dp": mean,
                 "fold_population_sd_4dp": sd}
                for label, rank, mean, sd in EXPECTED_RANKS],
            "n_rows": EXPECTED_N_ROWS,
            "max_abs_dev_vs_published_tolerance": TOL_ABS_DEV_VS_PUBLISHED,
            "estimator_tolerance": TOL_ESTIMATOR,
            "vs_shipped_head_to_head_tolerance": TOL_VS_SHIPPED,
            "published_only_rows_carry_no_tie_robust_number": True,
            "note": ("Assertions are enforced in code and gate --check; a "
                     "violation fails the run rather than being reported here."),
        },
        "provenance": {
            "literal": str(DEFAULT_LITERAL.relative_to(ROOT)),
            "comparison": str(DEFAULT_COMPARISON.relative_to(ROOT)),
            "published_method_metrics": str(PUBLISHED.relative_to(ROOT)),
            "benchmark_manifest": str(BENCHMARK_MANIFEST.relative_to(ROOT)),
            "gold": GOLD,
            "llm_bundles": f"{MODELS_DIR}/{{arm}}/all_source_predictions.jsonl",
            "generated_by": "scripts/compute_paper_table6_extended.py",
            "join": ("panel = sorted(stmt_hash) over the paper's own out-of-fold "
                     "table; arms joined stmt_hash <-> statement_id through the "
                     "frozen paper gold; fold assignment is the paper's own"),
            "reused": ["scripts/compare_paper_literal_vs_llms.py::fold_mean_pr_auc",
                       "scripts/compare_paper_literal_vs_llms.py::LLM_ARMS",
                       "scripts/extract_indra_paper_method_metrics.py::"
                       "EXPECTED_NOTEBOOK_SHA256"],
        },
    }

    lines = [
        f"{'rank':>4}  {'label':<26} {'origin':<21} {'paper metric':>14} "
        f"{'AP':>8} {'distinct':>9} {'gift':>8} {'dev':>8}",
    ]
    for r in rows:
        ap_s = ("     -   " if r["pooled_average_precision"] is None
                else f"{r['pooled_average_precision']:8.4f}")
        d_s = "        -" if r["distinct_scores"] is None else f"{r['distinct_scores']:9d}"
        g_s = "       -" if r["tie_gift"] is None else f"{r['tie_gift']:+8.4f}"
        v_s = "       -" if r["abs_dev_vs_published"] is None else f"{r['abs_dev_vs_published']:8.5f}"
        lines.append(
            f"{r['rank']:>4}  {r['label']:<26} {r['origin']:<21} "
            f"{r['fold_mean_trapezoidal_pr_auc']:.4f} +/- "
            f"{r['fold_population_sd']:.4f} {ap_s} {d_s} {g_s} {v_s}")
    lines += [
        "",
        f"re-run fidelity: max |dev vs published| {max_dev:.5f} ({max_dev_label}), "
        f"tolerance {TOL_ABS_DEV_VS_PUBLISHED}",
        f"tie gift: readers {reader_gift[0]:+.4f}..{reader_gift[1]:+.4f} "
        f"(distinct {reader_distinct[0]}-{reader_distinct[1]}), paper "
        f"{paper_gift[0]:+.4f}..{paper_gift[1]:+.4f} (distinct "
        f"{paper_distinct[0]}-{paper_distinct[1]}), INDRA CoGEx hybrid "
        f"{hybrid['tie_gift']:+.4f} (distinct {hybrid['distinct_scores']}, ours)",
        f"tie-corrected best vs best: {our_best['label']} "
        f"{our_best['pooled_average_precision']:.4f} vs {their_best['label']} "
        f"{their_best['pooled_average_precision']:.4f} = {tie_corrected_margin:+.4f} "
        f"(paper metric {paper_metric_margin:+.4f})",
        f"paper's own headline gain: {best_fitted['method']} - Belief Orig = "
        f"{headline_gain:+.4f} = {headline_gain_sd:.2f} fold SD, no test "
        f"({pub['method_count']} published rows, {len(agreeing)} sha256 agreements)",
    ]
    return payload, lines + notes


def serialize(payload: dict) -> str:
    return json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="re-derive and compare against the artifact on disk; "
                         "exit non-zero on any mismatch. Writes nothing.")
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="run manifest.json to record the output path + sha256 in")
    args = ap.parse_args()

    payload, lines = build()
    text = serialize(payload)
    print("\n".join(lines))
    print()

    if args.check:
        if not args.out_json.exists():
            print(f"FAIL: {args.out_json} does not exist; nothing to check against")
            return 1
        on_disk = args.out_json.read_text(encoding="utf-8")
        if on_disk != text:
            diff = list(difflib.unified_diff(
                on_disk.splitlines(), text.splitlines(),
                fromfile=f"{args.out_json} (on disk)", tofile="re-derived",
                lineterm="", n=1))
            print(f"FAIL: {args.out_json} does not match the re-derivation "
                  f"({len(diff)} diff lines; first 40 shown)")
            print("\n".join(diff[:40]))
            return 1
        print(f"OK: {args.out_json} reproduces byte for byte "
              f"({len(text.encode())} bytes, sha256 "
              f"{hashlib.sha256(text.encode()).hexdigest()})")
        return 0

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(args.out_json.read_bytes()).hexdigest()
    if args.manifest:
        man = json.loads(args.manifest.read_text())
        man.setdefault("outputs", {})["table6_extended"] = args.out_json.name
        man.setdefault("output_sha256", {})[args.out_json.name] = sha
        args.manifest.write_text(json.dumps(man, indent=2) + "\n")
        print(f"recorded sha256 in {args.manifest}")
    print(f"wrote {args.out_json} ({args.out_json.stat().st_size} bytes)\nsha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
