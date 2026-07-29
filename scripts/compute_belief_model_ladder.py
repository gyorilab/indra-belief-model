"""The belief-model ladder: every family in the 2023 paper on one shared panel.

The head-to-head reads as a two-arm duel, which is the wrong shape.  The 2023
paper produced a LINEAGE of belief models — an unfitted noisy-OR, a hierarchy
propagation, Bayesian refits of the source reliabilities, and a supervised RF —
and the interesting question is where each of those landed and what the reading
gate adds *relative to the paper's own feature engineering*, not relative to the
weakest member of the family.

Twelve arms, one metric (pooled average precision), one panel: the same 1689
all-source assembled statements carrying a released paper label. NOTE: the
paper's ", specific" suffix names an INPUT CONFIGURATION (include_more_specific),
not a statement set, and it is ON for only some rungs here — never stamp it on
the panel as a whole.

    paper-family    seven re-implementations on indra 1.24.0, each cross-checked
                    against its OWN run manifest's diagnostic_metrics block
    paper-literal   the 2023 paper's released code, run as-is on the released
                    corpus (paper_literal_table6_and_oof.json)
    reader-gate     the paper's noisy-OR applied to the evidence a reader kept

Every average precision here is RE-DERIVED with sklearn's tie-aware
``average_precision_score`` on the imported panel and then compared to the value
the shipping artifact recorded.  Where the two disagree the artifact carries
OURS, records the recorded value alongside it, and the script prints a loud
banner to stderr — a disagreement is never silently reconciled to the shipped
number.  (metrics.py::auprc() is deliberately NOT used: it is order-dependent
and inflates values, per the 2026-07-25 finding.)

The join (which statements, with which labels, in which order) is imported from
scripts/compare_paper_literal_vs_llms.py and scripts/compute_statement_review_queue.py
rather than re-derived, so this artifact cannot drift from
statement_review_queue.json or ap_decomposition_by_paper_band.json.  Ordering is
sorted(statement_id); the paper-gold ``matches_hash`` is cross-checked against the
current-INDRA prediction provenance so the panel identity is pinned on both keys.

The baseline is the unfitted noisy-OR ``SimpleScorer``, because it is the thing
BOTH the paper's engineered features and the reading gate modify.  Its belief is
``1 - PROD_s (syst_s + rand_s^{n_s})``.

Hard assertions (the script fails loudly rather than emitting a plausible file):
  (a) every entry scores all 1689 panel statements, with no duplicates and no
      extras;
  (b) the paper-literal arm's own embedded ``y_true`` agrees with the gold label
      on all 1689 rows, and it joins on ``matches_hash``, not position;
  (c) the baseline row's delta against itself is exactly 0.0;
  (d) ``current_indra_simple_default_predictions.jsonl`` and
      ``current_simple_direct_all_sources_predictions.jsonl`` agree per statement
      bit-exactly, so the ladder's baseline is provably the same arm the review
      queue calls "noisy-OR (paper original)";
  (e) the CountsScorer RF (full features) and the hybrid-local (full features)
      rows agree to 1e-12 — they are the same fitted model reported twice, not
      two independent results.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compute_belief_model_ladder.py \
      --out-json data/results/indra_paper_literal_models_20260724/belief_model_ladder.json \
      --manifest data/results/indra_paper_literal_models_20260724/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the head-to-head script's join contract verbatim: same gold file, same
# model-bundle root, same loader, same headline key.
from compare_paper_literal_vs_llms import (  # noqa: E402
    GOLD,
    HEADLINE,
    MODELS_DIR,
    load_jsonl,
)

# Reuse the review queue's panel loader verbatim: same 1689 statements, same
# sorted(statement_id) ordering, same matches_hash cross-check target.
from compute_statement_review_queue import (  # noqa: E402
    PROVENANCE,
    load_panel,
)

ROOT = Path(__file__).resolve().parents[1]

COMPARISON = ("data/results/indra_paper_literal_models_20260724/"
              "paper_literal_vs_llms.json")
LITERAL = ("data/results/indra_paper_literal_models_20260724/"
           "paper_literal_table6_and_oof.json")

# The SimpleScorer run's own predictions.  Cross-checked bit-exactly against the
# ladder's baseline scores file so the baseline is provably the same arm the
# review-queue panel calls "noisy-OR (paper original)".
SIMPLE_SCORER_PREDICTIONS = ("data/results/current_indra_simple_paper_20260717/"
                             "current_indra_simple_default_predictions.jsonl")

BAYESIAN_MANIFEST = ("data/results/current_indra_bayesian_paper_20260717/"
                     "current_bayesian_paper_manifest.json")
COUNTS_MANIFEST = ("data/results/current_indra_counts_hybrid_paper_20260718/"
                   "current_counts_hybrid_paper_manifest.json")
HIERARCHY_MANIFEST = ("data/results/current_indra_hierarchy_paper_20260717/"
                      "current_simple_hierarchy_paper_manifest.json")

TOL_RECORDED = 1e-12
TOL_SAME_MODEL = 1e-12

BASELINE_LABEL = "noisy-OR SimpleScorer (direct)"
NOISY_OR_FORMULA = "belief = 1 - PROD_s (syst_s + rand_s^{n_s})"

# Fixed presentation order: the paper's own family in ascending pooled average
# precision, then the paper's LITERAL released model, then the reading gates.
# Not sortable, not configurable.
#
# ``recorded`` names the artifact each entry's shipped value lives in.  The
# paper-family runs each carry a diagnostic_metrics block keyed by arm_id; the
# paper-literal and reader-gate values live in the head-to-head artifact's
# point_metrics, because the reader bundles carry no metrics block of their own.
LADDER = [
    {
        "label": "CountsScorer RF, source counts",
        "kind": "paper-family",
        "scores": ("data/results/current_indra_counts_hybrid_paper_20260718/"
                   "current_counts_source_only_oof_predictions.jsonl"),
        "recorded": {
            "kind": "run_manifest_diagnostic_metrics",
            "path": COUNTS_MANIFEST,
            "key": "indra_1.24.0_counts_rf_2kd13_source_only_oof_all_sources",
        },
        "note": "A supervised RF given ONLY the per-source evidence "
                "counts the noisy-OR already sees.",
    },
    {
        "label": "Hierarchy propagation",
        "kind": "paper-family",
        "scores": ("data/results/current_indra_hierarchy_paper_20260717/"
                   "current_simple_hierarchy_all_sources_predictions.jsonl"),
        "recorded": {
            "kind": "run_manifest_diagnostic_metrics",
            "path": HIERARCHY_MANIFEST,
            "key": "indra_1.24.0_simple_hierarchy_all_sources",
        },
        "note": "Noisy-OR with belief propagated over the statement hierarchy.",
    },
    {
        "label": BASELINE_LABEL,
        "kind": "paper-family",
        "scores": ("data/results/current_indra_bayesian_paper_20260717/"
                   "current_simple_direct_all_sources_predictions.jsonl"),
        "recorded": {
            "kind": "run_manifest_diagnostic_metrics",
            "path": BAYESIAN_MANIFEST,
            "key": "indra_1.24.0_simple_direct_all_sources",
        },
        "note": "INDRA's published source-prior noisy-OR, unfitted, on current "
                f"evidence: {NOISY_OR_FORMULA}.",
    },
    {
        "label": "BayesianScorer, source refit",
        "kind": "paper-family",
        "scores": ("data/results/current_indra_bayesian_paper_20260717/"
                   "current_bayesian_source_oof_all_sources_predictions.jsonl"),
        "recorded": {
            "kind": "run_manifest_diagnostic_metrics",
            "path": BAYESIAN_MANIFEST,
            "key": "indra_1.24.0_bayesian_source_oof_all_sources",
        },
        "note": "Out-of-fold refit of the per-source reliabilities.",
    },
    {
        "label": "BayesianScorer, source+subtype refit",
        "kind": "paper-family",
        "scores": ("data/results/current_indra_bayesian_paper_20260717/"
                   "current_bayesian_source_subtype_oof_all_sources_predictions.jsonl"),
        "recorded": {
            "kind": "run_manifest_diagnostic_metrics",
            "path": BAYESIAN_MANIFEST,
            "key": "indra_1.24.0_bayesian_source_subtype_oof_all_sources",
        },
        "note": "Out-of-fold refit of the source AND source-subtype reliabilities "
                "— the best noisy-OR variant on this ladder, which is "
                "OURS: the 2023 paper publishes no subtype-resolved belief arm.",
    },
    {
        "label": "CountsScorer RF, full features",
        "kind": "paper-family",
        "scores": ("data/results/current_indra_counts_hybrid_paper_20260718/"
                   "current_counts_full_features_oof_predictions.jsonl"),
        "recorded": {
            "kind": "run_manifest_diagnostic_metrics",
            "path": COUNTS_MANIFEST,
            "key": "indra_1.24.0_counts_rf_2kd13_full_features_oof_all_sources",
        },
        "note": "A supervised RF over INDRA's full CountsScorer feature "
                "set, a SUPERSET of the paper's engineered "
                "statement-type / #PMIDs / promoter features.",
    },
    {
        "label": "HybridScorer, full features",
        "kind": "paper-family",
        "scores": ("data/results/current_indra_counts_hybrid_paper_20260718/"
                   "current_hybrid_local_full_features_oof_predictions.jsonl"),
        "recorded": {
            "kind": "run_manifest_diagnostic_metrics",
            "path": COUNTS_MANIFEST,
            "key": "indra_1.24.0_hybrid_local_rf_2kd13_full_features_oof_all_sources",
        },
        "note": "The same fitted RF served through the hybrid wrapper — two INDRA "
                "scorer classes over one model, not two independent results.",
    },
    {
        "label": "RF 2k-d13 + Type/#PMIDs/promoter",
        "kind": "paper-literal",
        "scores": LITERAL,
        "literal_key": HEADLINE,
        "recorded": {
            "kind": "head_to_head_point_metrics",
            "path": COMPARISON,
            "key": "Paper literal RF+promoter",
        },
        "note": "The 2023 paper's OWN released code, run as-is on the released "
                "corpus — a different codebase and a different evidence snapshot.",
    },
    {
        "label": "Gemma 4 26B gate",
        "kind": "reader-gate",
        "scores": f"{MODELS_DIR}/gemma_4_26b/all_source_predictions.jsonl",
        "recorded": {
            "kind": "head_to_head_point_metrics",
            "path": COMPARISON,
            "key": "Gemma 4 26B",
        },
        "note": "INDRA's own unfitted noisy-OR applied to the evidence the "
                "reader KEPT — not the paper's MCMC-refit Belief Orig.",
    },
    {
        "label": "GLM-5 gate",
        "kind": "reader-gate",
        "scores": f"{MODELS_DIR}/glm_5/all_source_predictions.jsonl",
        "recorded": {
            "kind": "head_to_head_point_metrics",
            "path": COMPARISON,
            "key": "GLM-5",
        },
        "note": "INDRA's own unfitted noisy-OR applied to the evidence the "
                "reader KEPT — not the paper's MCMC-refit Belief Orig.",
    },
    {
        "label": "Gemma 4 31B gate",
        "kind": "reader-gate",
        "scores": f"{MODELS_DIR}/gemma_4_31b/all_source_predictions.jsonl",
        "recorded": {
            "kind": "head_to_head_point_metrics",
            "path": COMPARISON,
            "key": "Gemma 4 31B",
        },
        "note": "INDRA's own unfitted noisy-OR applied to the evidence the "
                "reader KEPT — not the paper's MCMC-refit Belief Orig.",
    },
    {
        "label": "Gemma 4 E2B gate",
        "kind": "reader-gate",
        "scores": f"{MODELS_DIR}/gemma_4_e2b/all_source_predictions.jsonl",
        "recorded": {
            "kind": "head_to_head_point_metrics",
            "path": COMPARISON,
            "key": "Gemma 4 E2B",
        },
        "note": "INDRA's own unfitted noisy-OR applied to the evidence the "
                "reader KEPT — not the paper's MCMC-refit Belief Orig.",
    },
]

# The two rows that are the same fitted model served two ways.
SAME_MODEL_PAIR = ("CountsScorer RF, full features",
                   "HybridScorer, full features")

# The paper's best model overall, measured two ways: our re-implementation on
# current INDRA, and the paper's literal released code on the released corpus.
BEST_PAPER_MODEL_LABELS = ("CountsScorer RF, full features",
                           "RF 2k-d13 + Type/#PMIDs/promoter")
BEST_NOISY_OR_VARIANT_LABEL = "BayesianScorer, source+subtype refit"
ENGINEERED_FEATURES_LABEL = "CountsScorer RF, full features"
FLAT_LABELS = ("Hierarchy propagation",
               "CountsScorer RF, source counts")


def _read_recorded(spec: dict, cache: dict[str, dict]) -> float:
    """The value the shipping artifact already recorded for this entry."""
    rec = spec["recorded"]
    path = rec["path"]
    if path not in cache:
        cache[path] = json.loads((ROOT / path).read_text())
    doc = cache[path]
    if rec["kind"] == "run_manifest_diagnostic_metrics":
        return float(doc["diagnostic_metrics"][rec["key"]]["pooled_average_precision"])
    if rec["kind"] == "head_to_head_point_metrics":
        return float(doc["point_metrics"][rec["key"]]["pooled_average_precision"])
    raise ValueError(f"unknown recorded-value kind {rec['kind']!r}")


def _scores_for(spec: dict, sids: list[str], mhash: dict[str, str],
                y: np.ndarray) -> tuple[np.ndarray, int]:
    """This entry's probability vector over the panel, in panel order."""
    if spec["kind"] == "paper-literal":
        lit = json.loads((ROOT / spec["scores"]).read_text())
        # stmt_hash MUST be compared as a string: 1429/1689 paper hashes exceed
        # 2^53, so a float round-trip would collide them.
        oof = {str(r["stmt_hash"]): r for r in lit["oof_predictions"][spec["literal_key"]]}
        assert len(oof) == len(lit["oof_predictions"][spec["literal_key"]]), (
            f"{spec['label']}: duplicate stmt_hash in the literal OOF table")
        wanted = {mhash[s] for s in sids}
        # (a) exact panel coverage, no extras, joined on matches_hash.
        assert set(oof) == wanted, (
            f"{spec['label']}: literal OOF table does not cover the panel exactly "
            f"({len(wanted - set(oof))} missing, {len(set(oof) - wanted)} extra)")
        p = np.array([float(oof[mhash[s]]["prob_correct"]) for s in sids])
        # (b) the literal arm's own embedded labels agree with the gold.
        embedded = np.array([int(oof[mhash[s]]["y_true"]) for s in sids])
        n_agree = int((embedded == y).sum())
        assert n_agree == len(sids), (
            f"{spec['label']}: embedded y_true disagrees with the gold label on "
            f"{len(sids) - n_agree} of {len(sids)} rows")
        return p, n_agree

    rows = load_jsonl(ROOT / spec["scores"])
    table = {r["statement_id"]: float(r["probability_correct"]) for r in rows}
    assert len(table) == len(rows), f"{spec['label']}: duplicate statement_id"
    # (a) exact panel coverage, no extras.
    assert set(table) == set(sids), (
        f"{spec['label']}: score file does not cover the panel exactly "
        f"({len(set(sids) - set(table))} missing, {len(set(table) - set(sids))} extra)")
    return np.array([table[s] for s in sids]), 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--manifest", default=None,
                    help="run manifest.json to record the output path + sha256 in")
    args = ap.parse_args()

    # `load_panel` returns ONE panel object (sids, labels, matches_hash, folds,
    # adjudication-safe flags) so every /paper artifact reads the same join.
    # Take the three fields this script needs; a missing key must raise here
    # rather than silently re-deriving a different panel downstream.
    panel = load_panel()
    sids, y, mhash = panel["sids"], panel["y"], panel["matches_hash"]
    n = len(sids)
    is_error = y == 0
    n_errors = int(is_error.sum())

    # Panel identity pinned on matches_hash too, not just the UUID (same check
    # the review queue makes, against the same provenance file).
    prov = {r["statement_id"]: str(r["matches_hash"]) for r in load_jsonl(PROVENANCE)}
    missing_prov = [s for s in sids if s not in prov]
    assert not missing_prov, (
        f"{len(missing_prov)} panel statements absent from {PROVENANCE}")
    mismatched = [s for s in sids if prov[s] != mhash[s]]
    assert not mismatched, (
        f"{len(mismatched)} statement_id -> matches_hash disagreements between the "
        f"paper gold and {PROVENANCE}, e.g. {mismatched[:3]}")

    # The 452 = 341 + 111 breakdown of the label convention, derived here.
    safe_negatives = flagged_negatives = 0
    for r in load_jsonl(GOLD):
        policy = r.get("paper_replication_policy") or {}
        if policy.get("released_paper_correct") != 0:
            continue
        if policy.get("label_is_adjudication_safe"):
            safe_negatives += 1
        else:
            flagged_negatives += 1
    assert safe_negatives + flagged_negatives == n_errors, (
        f"negative breakdown {safe_negatives} + {flagged_negatives} != {n_errors}")

    print(f"panel n={n}  errors={n_errors} "
          f"({safe_negatives} adjudication-safe + {flagged_negatives} flagged)  "
          f"base rate={n_errors / n:.4f}")

    # (d) the baseline scores file IS the SimpleScorer run, per statement.
    simple = {r["statement_id"]: float(r["probability_correct"])
              for r in load_jsonl(ROOT / SIMPLE_SCORER_PREDICTIONS)}
    baseline_spec = next(s for s in LADDER if s["label"] == BASELINE_LABEL)
    baseline_scores = {r["statement_id"]: float(r["probability_correct"])
                       for r in load_jsonl(ROOT / baseline_spec["scores"])}
    assert set(simple) == set(sids), (
        f"{SIMPLE_SCORER_PREDICTIONS} does not cover the panel exactly")
    n_bit_exact = sum(1 for s in sids if simple[s] == baseline_scores[s])
    assert n_bit_exact == n, (
        f"the ladder baseline disagrees with {SIMPLE_SCORER_PREDICTIONS} on "
        f"{n - n_bit_exact} of {n} statements; it is not the same arm the review "
        "queue calls 'noisy-OR (paper original)'")

    cache: dict[str, dict] = {}
    entries = []
    n_disagreements = 0
    literal_label_agreements = 0
    for spec in LADDER:
        p, n_agree = _scores_for(spec, sids, mhash, y)
        if spec["kind"] == "paper-literal":
            literal_label_agreements = n_agree
        ours = float(average_precision_score(y, p))
        recorded = _read_recorded(spec, cache)
        disagreement = ours - recorded
        agrees = abs(disagreement) <= TOL_RECORDED
        if not agrees:
            n_disagreements += 1
            print("=" * 78, file=sys.stderr)
            print(f"!! RE-DERIVED VALUE DISAGREES WITH THE RECORDED VALUE !!",
                  file=sys.stderr)
            print(f"   entry     {spec['label']}", file=sys.stderr)
            print(f"   ours      {ours!r}  (sklearn.metrics.average_precision_score "
                  f"over {spec['scores']})", file=sys.stderr)
            print(f"   recorded  {recorded!r}  ({spec['recorded']['path']} "
                  f"-> {spec['recorded']['key']})", file=sys.stderr)
            print(f"   delta     {disagreement:+.3e}  (tolerance {TOL_RECORDED:g})",
                  file=sys.stderr)
            print("   The artifact carries OURS. Reconcile the shipped value or "
                  "explain the difference.", file=sys.stderr)
            print("=" * 78, file=sys.stderr)

        entries.append({
            "label": spec["label"],
            "kind": spec["kind"],
            "average_precision": ours,
            "recorded_average_precision": recorded,
            "disagreement_vs_recorded": disagreement,
            "agrees_with_recorded": bool(agrees),
            "distinct_scores": int(len(np.unique(p))),
            "scores_path": spec["scores"],
            # null for the per-statement JSONL prediction files (keyed by
            # statement_id); the OOF table name for the paper-literal row, whose
            # scores live inside a JSON document keyed by stmt_hash.
            "scores_key": spec.get("literal_key"),
            "recorded_in": spec["recorded"]["path"],
            "recorded_key": spec["recorded"]["key"],
            "note": spec["note"],
        })

    by_label = {e["label"]: e for e in entries}
    baseline_ap = by_label[BASELINE_LABEL]["average_precision"]
    for e in entries:
        e["delta_vs_noisy_or_baseline"] = e["average_precision"] - baseline_ap
    # (c) the baseline's delta against itself is exactly 0.0.
    assert by_label[BASELINE_LABEL]["delta_vs_noisy_or_baseline"] == 0.0, (
        "the baseline row's own delta is not exactly 0.0")

    # (e) the two full-feature rows are the same fitted model reported twice.
    same_model_gap = abs(by_label[SAME_MODEL_PAIR[0]]["average_precision"]
                         - by_label[SAME_MODEL_PAIR[1]]["average_precision"])
    assert same_model_gap <= TOL_SAME_MODEL, (
        f"{SAME_MODEL_PAIR[0]} and {SAME_MODEL_PAIR[1]} differ by "
        f"{same_model_gap:.3e} > {TOL_SAME_MODEL:g}; they are supposed to be the "
        "same fitted model served two ways")

    # ---- the guardrail numbers, derived here so nothing downstream can type ----
    readers = [e for e in entries if e["kind"] == "reader-gate"]
    best_reader = max(readers, key=lambda e: e["average_precision"])
    engineered = by_label[ENGINEERED_FEATURES_LABEL]
    best_variant = by_label[BEST_NOISY_OR_VARIANT_LABEL]

    against_best = {
        label: best_reader["average_precision"] - by_label[label]["average_precision"]
        for label in BEST_PAPER_MODEL_LABELS
    }
    against_best_low = min(against_best.values())
    against_best_high = max(against_best.values())

    reimplemented_rf = by_label["CountsScorer RF, full features"]["average_precision"]
    literal_rf = by_label["RF 2k-d13 + Type/#PMIDs/promoter"]["average_precision"]
    proximity = abs(reimplemented_rf - literal_rf)
    fidelity = json.loads((ROOT / COMPARISON).read_text())["faithfulness_literal_vs_port"]
    fidelity_r = float(fidelity["pearson_r"])

    guardrails = {
        "baseline_label": BASELINE_LABEL,
        "baseline_average_precision": baseline_ap,
        "engineered_features": {
            "label": ENGINEERED_FEATURES_LABEL,
            "average_precision": engineered["average_precision"],
            "delta_vs_noisy_or_baseline": engineered["delta_vs_noisy_or_baseline"],
        },
        "reading_gate": {
            "label": best_reader["label"],
            "average_precision": best_reader["average_precision"],
            "delta_vs_noisy_or_baseline": best_reader["delta_vs_noisy_or_baseline"],
            "delta_vs_best_noisy_or_variant": {
                "label": BEST_NOISY_OR_VARIANT_LABEL,
                "delta": best_reader["average_precision"] - best_variant["average_precision"],
            },
            "delta_vs_best_paper_model": {
                label: against_best[label] for label in BEST_PAPER_MODEL_LABELS
            },
            "delta_vs_best_paper_model_range": [against_best_low, against_best_high],
        },
        "flat_against_baseline": {
            label: by_label[label]["delta_vs_noisy_or_baseline"] for label in FLAT_LABELS
        },
        "reimplementation_proximity": {
            "reimplemented_rf_full_features": reimplemented_rf,
            "paper_literal_rf_promoter": literal_rf,
            "absolute_gap": proximity,
            "status": "consistency check ACROSS DIFFERENT CORPORA — not fidelity "
                      "evidence",
            "fidelity_evidence": {
                "statistic": "per-statement Pearson r, literal vs semantic port",
                "value": fidelity_r,
                "source": COMPARISON,
            },
        },
    }

    caveats = [
        "Every value here is pooled average precision on the SAME 1689 "
        "all-source curated statements with the SAME released paper labels, "
        "re-derived with sklearn's tie-aware average_precision_score and then "
        "cross-checked against the value each run's own artifact recorded.",
        "The baseline is the unfitted noisy-OR SimpleScorer, "
        f"{NOISY_OR_FORMULA}, at {baseline_ap:.4f} pooled average precision. It is "
        "the baseline because it is what BOTH the engineered-feature RF and the "
        "reading gate modify — not because it is the paper's best model. That RF is "
        "OURS: INDRA's full CountsScorer feature set is a superset of the paper's "
        "Type/#PMIDs/promoter panel.",
        "From that baseline INDRA's full CountsScorer feature set — a superset of the "
        "paper's statement-type / #PMIDs / promoter panel — is worth "
        f"{engineered['delta_vs_noisy_or_baseline']:+.4f} and the "
        f"reading gate is worth {best_reader['delta_vs_noisy_or_baseline']:+.4f} "
        f"({best_reader['label']}) — but that gate figure is measured against the "
        "WEAKEST member of the paper's family. Against the paper's best noisy-OR "
        f"variant ({BEST_NOISY_OR_VARIANT_LABEL}) the same gate is "
        f"{guardrails['reading_gate']['delta_vs_best_noisy_or_variant']['delta']:+.4f}; "
        "against the paper's best model overall (RF with full features, "
        f"{reimplemented_rf:.4f} re-implemented / {literal_rf:.4f} literal) it is "
        f"{against_best_low:+.4f} to {against_best_high:+.4f}. Quote the range; "
        f"never quote the {best_reader['delta_vs_noisy_or_baseline']:+.4f} alone.",
        "Hierarchy propagation "
        f"({guardrails['flat_against_baseline']['Hierarchy propagation']:+.4f}) and an "
        "RF trained on source counts alone "
        f"({guardrails['flat_against_baseline']['CountsScorer RF, source counts']:+.4f}) "
        "are flat against plain noisy-OR on this panel: the supervised lift comes "
        "from the non-count features, not from learning a better source-count "
        "function.",
        f"Our re-implemented RF lands at {reimplemented_rf:.4f} and the paper's "
        f"literal released model at {literal_rf:.4f}, {proximity:.4f} apart. That is "
        "a CONSISTENCY CHECK ACROSS DIFFERENT CORPORA — two codebases, two evidence "
        "snapshots, agreeing in one scalar — and it is NOT evidence of "
        "implementation fidelity. The fidelity evidence is the per-statement "
        f"Pearson r = {fidelity_r:.4f} between the literal model and our semantic "
        "port, already reported on this page.",
        "The seven non-reader rows are re-implementations on indra 1.24.0 and current "
        "INDRA evidence, and only some are models the paper published: it has no "
        "Bayesian, subtype-resolved, hierarchy-propagated or HybridScorer arm. The "
        "paper-literal row is the 2023 released code on "
        "the released corpus. The families compare cleanly to each other, and the "
        "literal row compares only loosely to them.",
        f"{SAME_MODEL_PAIR[0]} and {SAME_MODEL_PAIR[1]} are the SAME fitted model "
        f"reported twice (agreeing to {same_model_gap:.3e}), not two independent "
        "results. They appear as two rows because INDRA's scorer classes name them "
        "separately; the paper names neither.",
        "The reader-gate rows are the paper's own noisy-OR applied to the evidence "
        "the reader kept, so the ladder compares belief models under a shared "
        "aggregation; the reader's contribution is the filtering.",
    ]

    payload = {
        "artifact_kind": "belief_model_ladder",
        "schema_version": 1,
        "metric": "pooled_average_precision",
        "metric_source": "sklearn.metrics.average_precision_score (tie-aware)",
        "noisy_or_formula": NOISY_OR_FORMULA,
        "panel": {
            "n": n,
            "n_errors": n_errors,
            "n_correct": int((~is_error).sum()),
            "error_base_rate": n_errors / n,
            "label": "paper_replication_policy.released_paper_correct",
            "label_convention": "\"error\" means the paper's own released label says "
                                "incorrect",
            "negative_breakdown": {
                "n_errors": n_errors,
                "adjudication_safe_negatives": safe_negatives,
                "flagged_label_is_adjudication_safe_false": flagged_negatives,
            },
            "ordering": "sorted(statement_id)",
        },
        "baseline": {
            "label": BASELINE_LABEL,
            "average_precision": baseline_ap,
            "why": "the unfitted noisy-OR is what BOTH the paper's engineered "
                   "features and the reading gate modify",
            "formula": NOISY_OR_FORMULA,
        },
        "entries": entries,
        "delta_guardrails": guardrails,
        "caveats": caveats,
        "checks": {
            "every_entry_covers_the_panel_exactly": True,
            "gold_matches_hash_agrees_with_prediction_provenance": True,
            "literal_arm_joins_on_matches_hash": True,
            "literal_arm_embedded_labels_agree_with_gold": literal_label_agreements,
            "baseline_matches_simple_scorer_predictions_bit_exactly": n_bit_exact,
            "baseline_delta_is_exactly_zero": True,
            "recorded_value_agreement_tol": TOL_RECORDED,
            "n_entries": len(entries),
            "n_entries_agreeing_with_recorded_value": len(entries) - n_disagreements,
            "n_entries_disagreeing_with_recorded_value": n_disagreements,
            "same_fitted_model_pair": list(SAME_MODEL_PAIR),
            "same_fitted_model_absolute_gap": same_model_gap,
            "same_fitted_model_tol": TOL_SAME_MODEL,
            "note": "Assertions are enforced in code; a violation fails the build "
                    "rather than being reported here as False. A recorded-value "
                    "disagreement is NOT an assertion: the artifact carries our "
                    "re-derived value, records the shipped one beside it, and the "
                    "script prints a stderr banner.",
        },
        "provenance": {
            "gold": GOLD,
            "matches_hash_crosscheck": PROVENANCE,
            "simple_scorer_predictions_crosscheck": SIMPLE_SCORER_PREDICTIONS,
            "scores": {e["label"]: e["scores_path"] for e in entries},
            "recorded_values": {e["label"]: {"path": e["recorded_in"],
                                             "key": e["recorded_key"]}
                                for e in entries},
            "join": "statement_id (paper gold canonical_corpus.statement_id) for "
                    "every arm except the paper-literal row, which joins on the "
                    "paper matches_hash; matches_hash is cross-checked for all",
            "generated_by": "scripts/compute_belief_model_ladder.py",
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    if args.manifest:
        mpath = Path(args.manifest)
        man = json.loads(mpath.read_text())
        man.setdefault("outputs", {})["belief_model_ladder"] = out.name
        man.setdefault("output_sha256", {})[out.name] = sha
        mpath.write_text(json.dumps(man, indent=2) + "\n")
        print(f"recorded sha256 in {mpath}")

    print(f"\nwrote {out} ({out.stat().st_size} bytes)\nsha256 {sha}\n")
    print(f"{'arm':<44}{'kind':<15}{'AP':>10}{'delta':>12}")
    for e in entries:
        print(f"{e['label']:<44}{e['kind']:<15}{e['average_precision']:>10.4f}"
              f"{e['delta_vs_noisy_or_baseline']:>+12.4f}")
    print(f"\nengineered features {engineered['delta_vs_noisy_or_baseline']:+.4f}  |  "
          f"reading gate {best_reader['delta_vs_noisy_or_baseline']:+.4f} vs baseline, "
          f"{guardrails['reading_gate']['delta_vs_best_noisy_or_variant']['delta']:+.4f} vs "
          f"{BEST_NOISY_OR_VARIANT_LABEL}, "
          f"{against_best_low:+.4f} to {against_best_high:+.4f} vs the paper's best model")
    if n_disagreements:
        print(f"\n!! {n_disagreements} entr{'y' if n_disagreements == 1 else 'ies'} "
              "disagree with the recorded value; the artifact carries OURS",
              file=sys.stderr)


if __name__ == "__main__":
    main()
