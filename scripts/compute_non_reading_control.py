"""The no-LLM control: what the non-reading subtractions alone are worth.

Three subtractions sit inside the phrase "the evidence the reader kept", and
none of them is a model judgement:

  * de-duplication      the reader panel is built from UNIQUE (statement,
                        evidence) pairs, while the paper's noisy-OR counts every
                        evidence entry, multiplicity and all;
  * ``no_text`` rows    the execution map's route name for evidence that carries
                        no sentence text at all — there is nothing to read, so
                        these are skipped before the gate ever runs.  The route
                        NAME is a join key and stays; every reader-facing string
                        this script emits says "no sentence" in words instead;
  * deterministic rejects  ``deterministic_mismatch`` and
                        ``deterministic_pseudogene`` come from our own grounding
                        rules, not from the model.

Before any reading gain can be claimed, those three have to be priced. This
script applies exactly them with NO model verdicts at all — every readable pair
accepted — and scores the result with the paper's own aggregation:

    belief = 1 - PROD_s (syst_s + rand_s^{n_s})

over ``sorted(sources)``, using ``src/indra_belief/noise_model.py::compute_gated_belief``
and the priors in ``data/comparison/aggregation.json``. Nothing about the formula
or the priors is written down here; both are read from the artifacts the reader
arms themselves declare.

The finding IS an assertion: the full control lands BELOW the ungated noisy-OR,
so the gain is not an artifact of what was removed before reading. The script
exits non-zero if it does not.

The raw row is the control's own correctness proof. Scored with weight =
``pair_multiplicity`` and nothing dropped, it must reproduce
``current_indra_simple_default_predictions.jsonl`` BIT-EXACTLY on all 1689
statements — same numbers, not the same ballpark — which is what licenses reading
the other rows as subtractions from `SimpleScorer` rather than as a different
model.

The join (which statements, with which labels, in which order) is imported from
scripts/compare_paper_literal_vs_llms.py and scripts/compute_statement_review_queue.py
rather than re-derived. Ordering is sorted(statement_id); identity is pinned on
the paper ``matches_hash`` as well as the UUID.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compute_non_reading_control.py \
      --out-json data/results/indra_paper_literal_models_20260724/non_reading_control.json \
      --manifest data/results/indra_paper_literal_models_20260724/manifest.json
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
# model-bundle root, same loader.
from compare_paper_literal_vs_llms import (  # noqa: E402
    MODELS_DIR,
    load_jsonl,
)

# Reuse the review queue's panel loader verbatim: same 1689 statements, same
# sorted(statement_id) ordering, same matches_hash cross-check target.
from compute_statement_review_queue import (  # noqa: E402
    PROVENANCE,
    load_panel,
)

from indra_belief.noise_model import compute_gated_belief  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

EXECUTION_MAP = "data/benchmark/indra_paper_unique_pairs_20260717_execution_map.jsonl"
AGGREGATION = "data/comparison/aggregation.json"

# The SimpleScorer run the raw row must reproduce bit-exactly, and the manifest
# that records its pooled average precision.  (The SimpleScorer run's own
# manifest carries no metrics block; the shared three-family diagnostic block
# lives in the Bayesian run's manifest, keyed by the arm below.)
SIMPLE_SCORER_PREDICTIONS = ("data/results/current_indra_simple_paper_20260717/"
                             "current_indra_simple_default_predictions.jsonl")
SIMPLE_SCORER_METRIC_MANIFEST = ("data/results/current_indra_bayesian_paper_20260717/"
                                 "current_bayesian_paper_manifest.json")
SIMPLE_SCORER_METRIC_KEY = "indra_1.24.0_simple_direct_all_sources"

CONTRAST_ARM = "gemma_4_26b"
CONTRAST_LABEL = "Gemma 4 26B (same subtractions PLUS reading)"

# The four reader arms, for the subtractive check: dropping evidence removes
# factors from a product of numbers < 1, so a reader can only ever LOWER belief.
READER_ARMS = ["gemma_4_26b", "glm_5", "gemma_4_31b", "gemma_4_e2b"]

REQUIRED_AGGREGATION = "indra_default_hard_gate"

NOISY_OR_FORMULA = "belief = 1 - PROD_s (syst_s + rand_s^{n_s})"

TOL_RECORDED = 1e-12

NO_TEXT = "no_text"
DETERMINISTIC_ROUTES = ("deterministic_mismatch", "deterministic_pseudogene")

# Fixed presentation order: the ungated baseline, then one subtraction added per
# row, then the reading contrast.  Not sortable, not configurable.
#
# ``key`` is the join key the viewer indexes on and is FROZEN.  ``label`` is the
# display string the control strip draws, and it is drawn right-aligned into
# STRIP_LEFT - 10 = 340 user units at 8.5px monospace (measured advance 0.602 em,
# i.e. 5.118 units per character) -> 66 characters.  A longer label is silently
# clipped at the left edge of the viewBox, so keep every label at or under 66
# characters, or widen STRIP_LEFT in FramingCorrection.svelte to match.
#
# No label may carry a raw field name: ``no_text`` is the execution map's route
# key, not something a reader of the figure can look up, so the labels say "no
# sentence" in words.  ``drop_routes`` still carries the real route names.
ROWS = [
    {
        "key": "raw",
        "label": "noisy-OR SimpleScorer (direct), every evidence entry",
        "weight": "pair_multiplicity",
        "drop_routes": [],
        "note": "The ungated baseline: all evidence, multiplicity retained, no "
                "subtraction of any kind.",
    },
    {
        "key": "dedup_only",
        "label": "de-dup only",
        "weight": "unique_pair",
        "drop_routes": [],
        "note": "One vote per UNIQUE (statement, evidence) pair — the panel the "
                "reader arms were built on.",
    },
    {
        "key": "dedup_plus_no_text",
        "label": "de-dup + drop evidence with no sentence",
        "weight": "unique_pair",
        "drop_routes": [NO_TEXT],
        "note": "Also drops evidence with no sentence to read; the gate never "
                "sees these.",
    },
    {
        "key": "full_control",
        "label": "de-dup + no-sentence + deterministic rejects, no LLM (control)",
        "weight": "unique_pair",
        "drop_routes": [NO_TEXT, *DETERMINISTIC_ROUTES],
        "note": "All three non-reading subtractions, every readable pair "
                "accepted, no model verdict anywhere.",
    },
]

BASELINE_ROW = "raw"
CONTROL_ROW = "full_control"

# Reported in research/indra_paper_literal_vs_llm_comparison.md Result 3d from
# the PRODUCTION statement_belief de-dup pass, which collapses roughly 40 further
# within-source text-normalized near-duplicates the execution map does not.  Not
# re-derived here (it needs the production pass, not the execution map), and
# carried only so the scope difference is visible rather than hidden.
MEMO_PRODUCTION_DEDUP = {
    "status": "memo-reported; NOT re-derived by this script",
    "source": "research/indra_paper_literal_vs_llm_comparison.md, Result 3d",
    "why_not_rederived": "reproducing it requires the production statement_belief "
                         "de-dup pass, not the execution map this script reads",
    "excess_pairs": 714,
    "statements_with_excess": 339,
    "average_precision": {
        "dedup_only": 0.9025,
        "dedup_plus_no_text": 0.8981,
        "full_control": 0.9015,
    },
    "conclusion_unchanged": "the control sits BELOW the ungated noisy-OR under "
                            "either de-dup scope",
}


def _belief(rows: list[dict], priors: dict[str, tuple[float, float]],
            *, weighted: bool, drop: frozenset[str]) -> tuple[float, int]:
    """Noisy-OR belief over the surviving pairs, and how many pairs survived."""
    evidence = []
    for r in rows:
        if r["route"] in drop:
            continue
        n = int(r["pair_multiplicity"]) if weighted else 1
        evidence.extend({"source_api": r["source_api"], "included": True}
                        for _ in range(n))
    return compute_gated_belief(evidence, priors).belief, len(evidence)


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

    # Panel identity pinned on matches_hash too, not just the UUID.
    prov = {r["statement_id"]: str(r["matches_hash"]) for r in load_jsonl(PROVENANCE)}
    mismatched = [s for s in sids if prov.get(s) != mhash[s]]
    assert not mismatched, (
        f"{len(mismatched)} statement_id -> matches_hash disagreements between the "
        f"paper gold and {PROVENANCE}, e.g. {mismatched[:3]}")

    # ---- priors: read, never written down --------------------------------
    agg = json.loads((ROOT / AGGREGATION).read_text())
    assert agg["aggregation"] == REQUIRED_AGGREGATION, (
        f"{AGGREGATION} declares aggregation {agg['aggregation']!r}, not "
        f"{REQUIRED_AGGREGATION!r}; the control would not be the reader arms' "
        "own aggregation")
    assert agg["reader_profile"] is None, (
        f"{AGGREGATION} carries a fitted reader profile ({agg['reader_profile']!r}); "
        "the control must run the unfitted path the reader bundles declare")
    priors = {src: (float(pair[0]), float(pair[1]))
              for src, pair in agg["priors"].items()}

    # ---- execution map ----------------------------------------------------
    by_hash: dict[str, list[dict]] = collections.defaultdict(list)
    route_pairs: collections.Counter = collections.Counter()
    route_multiplicity: collections.Counter = collections.Counter()
    excess_by_hash: collections.Counter = collections.Counter()
    n_map_rows = 0
    unknown_sources = set()
    for r in load_jsonl(ROOT / EXECUTION_MAP):
        n_map_rows += 1
        h = str(r["paper_statement_hash"])
        by_hash[h].append(r)
        route_pairs[r["route"]] += 1
        multiplicity = int(r["pair_multiplicity"])
        route_multiplicity[r["route"]] += multiplicity
        excess_by_hash[h] += multiplicity - 1
        if r["source_api"] not in priors:
            unknown_sources.add(r["source_api"])
    assert not unknown_sources, (
        f"{EXECUTION_MAP} carries sources absent from {AGGREGATION}: "
        f"{sorted(unknown_sources)}; they would silently take a fallback prior")

    h2sid = {mhash[s]: s for s in sids}
    assert len(h2sid) == n, "duplicate matches_hash in the panel"
    assert set(by_hash) == set(h2sid), (
        f"execution map covers {len(by_hash)} statements, panel has {n} "
        f"({len(set(h2sid) - set(by_hash))} missing, "
        f"{len(set(by_hash) - set(h2sid))} extra)")

    n_unique_pairs = n_map_rows
    n_summed_multiplicity = int(sum(route_multiplicity.values()))
    excess_pairs = int(sum(excess_by_hash.values()))
    statements_with_excess = int(sum(1 for v in excess_by_hash.values() if v > 0))
    assert n_summed_multiplicity - n_unique_pairs == excess_pairs, (
        "de-dup arithmetic does not close: "
        f"{n_summed_multiplicity} - {n_unique_pairs} != {excess_pairs}")

    print(f"panel n={n}  errors={n_errors}  base rate={n_errors / n:.4f}")
    print(f"execution map: {n_unique_pairs} unique pairs, "
          f"{n_summed_multiplicity} summed multiplicity, {excess_pairs} excess "
          f"across {statements_with_excess} statements")
    print("routes: " + "  ".join(f"{k}={v}" for k, v in sorted(route_pairs.items())))

    # ---- the four control rows -------------------------------------------
    scores: dict[str, np.ndarray] = {}
    rows_out = []
    for spec in ROWS:
        weighted = spec["weight"] == "pair_multiplicity"
        drop = frozenset(spec["drop_routes"])
        beliefs, surviving = {}, 0
        for h, rs in by_hash.items():
            b, k = _belief(rs, priors, weighted=weighted, drop=drop)
            beliefs[h2sid[h]] = b
            surviving += k
        p = np.array([beliefs[s] for s in sids])
        scores[spec["key"]] = p
        rows_out.append({
            "key": spec["key"],
            "label": spec["label"],
            "weight": spec["weight"],
            "dropped_routes": list(spec["drop_routes"]),
            "n_evidence_scored": surviving,
            "n_statements_with_no_surviving_evidence": int((p == 0.0).sum()),
            "average_precision": float(average_precision_score(y, p)),
            "distinct_scores": int(len(np.unique(p))),
            "note": spec["note"],
        })

    raw_row = next(r for r in rows_out if r["key"] == BASELINE_ROW)
    control_row = next(r for r in rows_out if r["key"] == CONTROL_ROW)
    assert raw_row["n_evidence_scored"] == n_summed_multiplicity
    assert next(r for r in rows_out if r["key"] == "dedup_only")["n_evidence_scored"] \
        == n_unique_pairs

    # ---- the raw row must BE SimpleScorer, bit for bit --------------------
    simple = {r["statement_id"]: float(r["probability_correct"])
              for r in load_jsonl(ROOT / SIMPLE_SCORER_PREDICTIONS)}
    assert set(simple) == set(sids), (
        f"{SIMPLE_SCORER_PREDICTIONS} does not cover the panel exactly")
    raw = scores[BASELINE_ROW]
    shipped = np.array([simple[s] for s in sids])
    n_bit_exact = int(sum(1 for a, b in zip(raw, shipped) if a == b))
    max_abs_delta = float(np.max(np.abs(raw - shipped)))
    assert n_bit_exact == n, (
        f"the raw row reproduces {SIMPLE_SCORER_PREDICTIONS} on only "
        f"{n_bit_exact} of {n} statements (max |delta| {max_abs_delta:.3e}); the "
        "control is not a subtraction from SimpleScorer")
    assert max_abs_delta == 0.0

    recorded_ap = float(
        json.loads((ROOT / SIMPLE_SCORER_METRIC_MANIFEST).read_text())
        ["diagnostic_metrics"][SIMPLE_SCORER_METRIC_KEY]["pooled_average_precision"])
    raw_disagreement = raw_row["average_precision"] - recorded_ap
    if abs(raw_disagreement) > TOL_RECORDED:
        print("=" * 78, file=sys.stderr)
        print("!! RE-DERIVED RAW AP DISAGREES WITH THE RECORDED SimpleScorer AP !!",
              file=sys.stderr)
        print(f"   ours      {raw_row['average_precision']!r}", file=sys.stderr)
        print(f"   recorded  {recorded_ap!r}  "
              f"({SIMPLE_SCORER_METRIC_MANIFEST} -> {SIMPLE_SCORER_METRIC_KEY})",
              file=sys.stderr)
        print(f"   delta     {raw_disagreement:+.3e}  (tolerance {TOL_RECORDED:g})",
              file=sys.stderr)
        print("   The artifact carries OURS.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
    assert abs(raw_disagreement) <= TOL_RECORDED, (
        f"raw AP {raw_row['average_precision']!r} != recorded SimpleScorer AP "
        f"{recorded_ap!r}")

    # ---- the finding -------------------------------------------------------
    control_minus_raw = control_row["average_precision"] - raw_row["average_precision"]
    assert control_row["average_precision"] < raw_row["average_precision"], (
        f"the full no-LLM control ({control_row['average_precision']!r}) does NOT "
        f"land below the ungated noisy-OR ({raw_row['average_precision']!r}); the "
        "non-reading subtractions would then be part of the reading gain and this "
        "artifact's whole claim is void")
    for row in rows_out:
        row["delta_vs_raw_noisy_or"] = (row["average_precision"]
                                        - raw_row["average_precision"])

    # ---- the reading contrast ---------------------------------------------
    contrast_scores = {r["statement_id"]: float(r["probability_correct"])
                       for r in load_jsonl(f"{MODELS_DIR}/{CONTRAST_ARM}/"
                                           "all_source_predictions.jsonl")}
    assert set(contrast_scores) == set(sids), (
        f"{CONTRAST_ARM}: score file does not cover the panel exactly")
    contrast_p = np.array([contrast_scores[s] for s in sids])
    contrast = {
        "key": CONTRAST_ARM,
        "label": CONTRAST_LABEL,
        "weight": "unique_pair",
        "dropped_routes": [NO_TEXT, *DETERMINISTIC_ROUTES],
        "average_precision": float(average_precision_score(y, contrast_p)),
        "delta_vs_raw_noisy_or": float(average_precision_score(y, contrast_p))
                                 - raw_row["average_precision"],
        "delta_vs_full_control": float(average_precision_score(y, contrast_p))
                                 - control_row["average_precision"],
        "distinct_scores": int(len(np.unique(contrast_p))),
        "scores_path": f"{MODELS_DIR}/{CONTRAST_ARM}/all_source_predictions.jsonl",
        "note": "The same three subtractions PLUS the reader's per-evidence "
                "verdicts, taken from the shipped bundle rather than recomputed.",
    }

    # ---- subtractive check: a reader can only remove belief ----------------
    # Dropping evidence removes factors from a product of numbers < 1, so the
    # gated belief can never exceed the ungated one.  Checked, not assumed.
    subtractive = {"baseline": "raw noisy-OR (all evidence, multiplicity retained)",
                   "arms": {}}
    total_exceed = 0
    for arm in READER_ARMS:
        table = {r["statement_id"]: float(r["probability_correct"])
                 for r in load_jsonl(f"{MODELS_DIR}/{arm}/all_source_predictions.jsonl")}
        assert set(table) == set(sids), (
            f"{arm}: score file does not cover the panel exactly")
        p = np.array([table[s] for s in sids])
        exceed = int((p > raw).sum())
        total_exceed += exceed
        subtractive["arms"][arm] = {
            "n_statements": n,
            "n_exceeding_noisy_or": exceed,
            "n_at_exactly_zero": int((p == 0.0).sum()),
            "max_belief_above_noisy_or": float(np.max(p - raw)),
        }
    subtractive["n_comparisons"] = n * len(READER_ARMS)
    subtractive["n_exceeding_noisy_or"] = total_exceed
    assert total_exceed == 0, (
        f"{total_exceed} of {n * len(READER_ARMS)} reader beliefs EXCEED the "
        "ungated noisy-OR; the gate is not purely subtractive and every claim "
        "resting on that is void")

    caveats = [
        "This is the EXECUTION-MAP pass. It de-duplicates to unique "
        "(statement, evidence) pairs as recorded in "
        f"{EXECUTION_MAP}. Production statement_belief collapses roughly 40 further "
        "within-source text-normalized near-duplicates, which needs the production "
        "de-dup pass to reproduce and is therefore NOT re-derived here; the "
        "memo-reported figures for that scope are carried in "
        "production_dedup_scope_difference, clearly marked. The conclusion is the "
        "same either way: the control sits below the ungated noisy-OR.",
        "Every row uses the paper's OWN aggregation and the priors the reader "
        f"bundles declare ({AGGREGATION}): {NOISY_OR_FORMULA} over sorted(sources). "
        "No model verdict enters rows 1-4 at all.",
        "The de-dup row is not a defect being corrected — it is a scope difference "
        "between counting every evidence entry and counting each distinct "
        "(statement, evidence) pair once. It is priced here so it cannot be "
        "mistaken for a reading gain.",
        f"{control_row['n_statements_with_no_surviving_evidence']} statements lose "
        "ALL their evidence to the deterministic rejects and score belief 0 in the "
        "full control. That is a property of our grounding rules, not of any "
        "reader.",
        "Average precision is sklearn's tie-aware average_precision_score "
        "throughout; the paper's own trapezoidal PR-AUC is not used here, because "
        "it over-credits tied score distributions.",
    ]

    payload = {
        "artifact_kind": "non_reading_control",
        "schema_version": 1,
        "metric": "pooled_average_precision",
        "metric_source": "sklearn.metrics.average_precision_score (tie-aware)",
        "noisy_or_formula": NOISY_OR_FORMULA,
        "aggregation": agg["aggregation"],
        "question": "Do the three NON-reading subtractions — de-duplication, "
                    "skipping evidence that carries no sentence text, and "
                    "deterministic grounding rejects — account for the reader "
                    "arms' gain?",
        "finding": "No. Applied with no model verdicts at all, they land BELOW the "
                   "noisy-OR over every evidence entry, with no gate applied.",
        "panel": {
            "n": n,
            "n_errors": n_errors,
            "n_correct": int((~is_error).sum()),
            "error_base_rate": n_errors / n,
            "label": "paper_replication_policy.released_paper_correct",
            "ordering": "sorted(statement_id)",
        },
        "rows": rows_out,
        "contrast": contrast,
        "baseline_row": BASELINE_ROW,
        "control_row": CONTROL_ROW,
        "control_minus_raw_average_precision": control_minus_raw,
        "control_lands_below_raw": True,
        "route_census": {
            "basis": "unique (statement, evidence) pairs in the execution map",
            "n_unique_pairs": n_unique_pairs,
            "routes": dict(sorted(route_pairs.items())),
            "routes_multiplicity_weighted": dict(sorted(route_multiplicity.items())),
            "readable_routes": sorted(set(route_pairs) - {NO_TEXT, *DETERMINISTIC_ROUTES}),
            "no_text": route_pairs[NO_TEXT],
            "deterministic_mismatch": route_pairs["deterministic_mismatch"],
            "deterministic_pseudogene": route_pairs["deterministic_pseudogene"],
        },
        "dedup": {
            "n_unique_pairs": n_unique_pairs,
            "n_summed_multiplicity": n_summed_multiplicity,
            "excess_pairs": excess_pairs,
            "statements_with_excess": statements_with_excess,
        },
        "production_dedup_scope_difference": MEMO_PRODUCTION_DEDUP,
        "subtractive_check": subtractive,
        "caveats": caveats,
        "checks": {
            "raw_row_reproduces_simple_scorer_bit_exactly": n_bit_exact,
            "raw_row_max_abs_delta_vs_simple_scorer": max_abs_delta,
            "raw_row_average_precision_vs_recorded": {
                "ours": raw_row["average_precision"],
                "recorded": recorded_ap,
                "disagreement": raw_disagreement,
                "tol": TOL_RECORDED,
                "recorded_in": SIMPLE_SCORER_METRIC_MANIFEST,
                "recorded_key": SIMPLE_SCORER_METRIC_KEY,
            },
            "full_control_below_raw": True,
            "dedup_arithmetic_closes": True,
            "every_execution_map_source_has_a_declared_prior": True,
            "execution_map_covers_the_panel_exactly": True,
            "gold_matches_hash_agrees_with_prediction_provenance": True,
            "readers_never_exceed_the_noisy_or": subtractive["n_exceeding_noisy_or"] == 0,
            "note": "Assertions are enforced in code; a violation fails the build "
                    "rather than being reported here as False.",
        },
        "provenance": {
            "execution_map": EXECUTION_MAP,
            "priors": AGGREGATION,
            "belief_function": "src/indra_belief/noise_model.py::compute_gated_belief",
            "simple_scorer_predictions": SIMPLE_SCORER_PREDICTIONS,
            "simple_scorer_recorded_metric": {
                "path": SIMPLE_SCORER_METRIC_MANIFEST,
                "key": SIMPLE_SCORER_METRIC_KEY,
            },
            "matches_hash_crosscheck": PROVENANCE,
            "reader_bundles": f"{MODELS_DIR}/{{arm}}/all_source_predictions.jsonl",
            "join": "paper_statement_hash == canonical_corpus.matches_hash -> "
                    "statement_id",
            "generated_by": "scripts/compute_non_reading_control.py",
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    if args.manifest:
        mpath = Path(args.manifest)
        man = json.loads(mpath.read_text())
        man.setdefault("outputs", {})["non_reading_control"] = out.name
        man.setdefault("output_sha256", {})[out.name] = sha
        mpath.write_text(json.dumps(man, indent=2) + "\n")
        print(f"recorded sha256 in {mpath}")

    print(f"\nwrote {out} ({out.stat().st_size} bytes)\nsha256 {sha}\n")
    print(f"{'row':<62}{'AP':>9}{'vs raw':>10}{'evidence':>10}")
    for row in rows_out:
        print(f"{row['label']:<62}{row['average_precision']:>9.4f}"
              f"{row['delta_vs_raw_noisy_or']:>+10.4f}{row['n_evidence_scored']:>10d}")
    print(f"{contrast['label']:<62}{contrast['average_precision']:>9.4f}"
          f"{contrast['delta_vs_raw_noisy_or']:>+10.4f}")
    print(f"\nfull control - raw noisy-OR = {control_minus_raw:+.4f} "
          "(the control lands BELOW the baseline)")
    print(f"subtractive check: {subtractive['n_exceeding_noisy_or']} of "
          f"{subtractive['n_comparisons']} reader beliefs exceed the noisy-OR")


if __name__ == "__main__":
    main()
