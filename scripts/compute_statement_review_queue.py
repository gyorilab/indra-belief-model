"""Curator review queue: what each belief model would put in front of a human.

Every arm on the /paper panel emits a belief in [0, 1] for the same 1689
"all sources, specific" assembled statements.  Turning that scalar into WORK
requires one more decision the 2023 paper never published: a threshold.  This
script makes that decision explicit and identical across arms.

    decision rule      flag a statement as WRONG iff  belief <= tau
    threshold rule     tau = the SMALLEST of the arm's own distinct scores whose
                       flag set catches at least TARGET_RECALL of the 452 known
                       errors

Matching recall is what makes the columns comparable: every arm is asked to
catch the same fraction of the errors, and the question becomes how much
correct work a curator wades through to get there (queue size, false alarms,
precision).

WHICH BELIEF MODELS ARE HERE, AND WHOSE THEY ARE.  Five, and they are NOT
interchangeable:

  * ``indra_simple_default`` and ``indra_simple_hierarchy`` are INDRA's SHIPPED
    SimpleScorer at its default per-source priors — the unfitted noisy-OR the
    library computes for every statement today, without and with hierarchy
    propagation.  They are NOT the paper's "Belief Orig", which refits its
    source reliabilities per fold by MCMC and is not what INDRA serves.
  * ``paper_literal_rf_promoter`` is the 2023 paper's OWN released random
    forest, re-run from ``sorgerlab/indra_assembly_paper`` on the paper's own
    feature matrix and its own 10 folds, scored out of fold.  Its key here is
    the paper's published method string, asserted verbatim against
    ``data/benchmark/indra_paper_2023_published_method_metrics.json``.  It is a
    RESEARCH model: it was never deployed.
  * ``our_bayesian_source_subtype`` and ``our_counts_full_features`` are OUR
    reimplementations over INDRA's feature superset.  Neither is published by
    the paper and neither is served.  The paper publishes no Bayesian arm, no
    subtype arm and no hierarchy arm at all.

The paper's own RF is the comparator the operational block below reports
against, because it is the only arm on this panel the paper actually published.
The other four are carried so the claim can be checked against the served
incumbent and against our own ports, not only against the one model that
flatters it.

The second half of the artifact turns the comparison around and fixes the YIELD
instead of the threshold.  A reader gate has an operating point nobody chose:
the block of statements whose evidence it rejected outright (belief exactly 0).
Asking the belief models to catch the SAME number of errors that block catches
gives a number a curator can act on directly — how many statements each one has
to read to get there — and asking them at the gate's own budget gives the other
half.  Both are derived by walking the arm's scores in ascending order, and the
threshold that lands a belief model on the gate's yield is chosen ON THIS PANEL,
with the labels in hand.  That is an ORACLE and it FAVOURS the belief model; it
is used deliberately, because the claim is stronger when the comparator is
handed the advantage and the gate is handed no threshold at all.  A budget sweep
beside it shows the whole curve, so the budget-dependence of the result is
visible rather than asserted.

The third block answers the question the sweep invites: is the gap real?  Error
recall at a fixed review budget (25% of the panel) is bootstrapped against the
paper's own RF with the SAME fold-stratified paired design, seed and resample
count as the ranking-margin robustness in
``scripts/compute_paper_robustness.py``, and reported with a studentized max-t
band over the family of four reader gates plus the same adjudication-safe
label-completeness sensitivity panel.  Nothing about the interval is
hard-coded downstream: the viewer reads these bytes.

The join (which statements, with which labels, in which order) is imported from
scripts/compare_paper_literal_vs_llms.py rather than re-derived, so this
artifact cannot drift from the AP/AUROC comparison or from
scripts/compute_paper_ap_decomposition.py.  Ordering is sorted(statement_id).
The paper-gold ``matches_hash`` is additionally cross-checked against the
current-INDRA prediction provenance so the panel identity is pinned on both
keys, not just the UUID.

Hard assertions (the script fails loudly rather than emitting a plausible file):
  (a) queue == true_errors_caught + false_alarms, and precision == real/queue,
      exactly, for every arm at every target;
  (b) the achieved recall is >= the target for every arm at every target (the
      threshold search is a >= search, so a miss means the arm cannot reach that
      recall at all, which must fail rather than silently under-deliver);
  (c) every arm scores all 1689 panel statements, with no extras;
  (d) for the LLM arms, the "zero pile" (belief exactly 0.0) is the arm's whole
      minimum-score tie block, and the bundle manifest declares the unfitted
      hard gate with no reader profile.  Under that aggregation
      (noise_model.compute_gated_belief: sources with zero surviving evidence
      leave the product entirely, and every prior factor syst + rand**k is < 1),
      belief == 0.0 holds if and only if the reader rejected EVERY piece of
      evidence it read.  llm.py hard-fails on a None belief, so "read nothing"
      cannot masquerade as a zero;
  (e) the ``promotion_ceiling`` count is a subset count of the panel's true
      statements, and NO reader arm scores any of those statements above the
      unfiltered noisy-OR (the subtractive-only premise the ceiling rests on);
  (f) every ``equal_yield`` budget is MINIMAL (one fewer review misses the
      target), every reference budget reproduces the arm's own zero pile
      exactly, and the budget sweep is monotone, starts at zero and ends at the
      full error count — so neither derivation can drift from the other;
  (g) the swept catch and the bootstrap's own catch are computed by two
      INDEPENDENT implementations (block walk vs sorted-prefix) and must agree
      at every swept budget for every arm, so the interval and the curve cannot
      describe different quantities;
  (h) the paper-model arm's key is a method string the paper actually published,
      matched verbatim (U+002D hyphen) against the published-metrics artifact.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compute_statement_review_queue.py \
      --out-json data/results/indra_paper_literal_models_20260724/statement_review_queue.json \
      --manifest data/results/indra_paper_literal_models_20260724/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the head-to-head script's join contract verbatim: same gold file, same
# model-bundle root, same loader, same bootstrap design.
from compare_paper_literal_vs_llms import (  # noqa: E402
    GOLD,
    HEADLINE,
    MODELS_DIR,
    N_BOOT,
    SEED,
    load_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]

# statement_id -> matches_hash, from the current-INDRA simple run.  Used only to
# cross-check the gold's own matches_hash; it is not the scoring join.
PROVENANCE = ("data/results/current_indra_simple_paper_20260717/"
              "current_indra_simple_default_prediction_provenance.jsonl")

# The paper's own released model, re-run by us: out-of-fold predictions keyed on
# the paper statement hash, plus the fold assignment the bootstrap below reuses.
PAPER_LITERAL = ("data/results/indra_paper_literal_models_20260724/"
                 "paper_literal_table6_and_oof.json")
# Every method string the 2023 paper actually published.  HEADLINE is asserted
# against it so the one arm we attribute to the paper cannot drift into a name
# they never used.
PUBLISHED_METHODS = "data/benchmark/indra_paper_2023_published_method_metrics.json"

TARGET_RECALLS = [0.5, 0.6, 0.7, 0.8]
HEADLINE_TARGET = 0.7

# The other half of what this gate design costs.  The reader arms ARE INDRA's
# own noisy-OR applied to a SUBSET of the evidence (the hard gate can only drop
# evidence rows), so a reader can only ever REMOVE belief.  Every true statement
# the unfiltered noisy-OR already scores below a promotion bar is therefore
# unpromotable by every reader arm, however well it reads.  The bar below is a
# stated illustrative one, not a fitted one; the count is monotone in it.
PROMOTION_THRESHOLD = 0.90

# Budget sweep resolution, in statements.  25 over a 1689-statement panel is 68
# grid points: fine enough that the piecewise-linear curve between them is
# invisible at figure scale, coarse enough that the artifact stays small.  The
# gates' own operating points and the robustness budget are added to the grid
# exactly, so the drawn curve passes through them rather than near them.
BUDGET_SWEEP_STEP = 25

# --- error recall at a fixed review budget --------------------------------
# The operational metric, and the one the robustness block bootstraps.  A budget
# is a share of the panel rather than a count so the primary and sensitivity
# panels are asked the same question; floor(), not round(), so the number is one
# operation with no tie-to-even surprise (0.25 x 1689 = 422.25 -> 422).
BUDGET_SHARE = 0.25
# Family-wise level for the simultaneous band, and the two reference critical
# values it is quoted against.  DERIVED from stdlib rather than transcribed: a
# typed normal quantile is exactly the kind of constant that is wrong in the
# fourth decimal and never noticed.
FAMILY_ALPHA = 0.05
POINTWISE_NORMAL_Z = statistics.NormalDist().inv_cdf(1 - FAMILY_ALPHA / 2)

# The equal-yield block's whole rhetorical weight rests on this being said out
# loud, so it travels with the numbers rather than living in a component.
ORACLE_DISCLOSURE = (
    "The budget each comparator needs is found by choosing its threshold ON "
    "THIS PANEL, with these labels already in hand, to land exactly on the "
    "reference arm's error count. That is an ORACLE threshold: it is fitted "
    "and evaluated on the same 1689 statements, it would not be available "
    "before curation, and it FAVOURS the comparator. The reference arm is "
    "given no such help — it has NO threshold at all. Its operating point is "
    "the block of statements whose evidence the reader rejected outright, "
    "which is not a threshold anyone chose, could not have been tuned, and is "
    "the same block on any panel. The comparison is stated in this direction "
    "deliberately: the side that is handed the advantage still loses."
)

BUDGET_RULE = (
    "A curator with a budget of B reviews reads the B lowest-scoring "
    "statements. Where B falls inside a block of tied scores the arm cannot "
    "say which of the tied statements to read first, so the block contributes "
    "its errors pro rata and the count is the EXPECTED number of errors found "
    "reading a uniformly random B-sized prefix. No arbitrary tie order is ever "
    "imposed: the same panel scored in a different row order gives the same "
    "curve. Reporting a single arbitrary tie order instead would move the "
    "gate's count by several errors and would be a property of the sort, not "
    "of the model."
)

BOOTSTRAP_DESIGN = (
    "paired fold-stratified bootstrap over the paper's own out-of-fold fold "
    "assignment; ONE index vector per resample, SHARED across every arm, so "
    "the arms' deltas are drawn jointly and the max-t band carries their "
    "correlation rather than assuming it away. Same seed, same resample count "
    "and same design as scripts/compute_paper_robustness.py, which bootstraps "
    "the ranking margin on this panel."
)

# Fixed presentation order.  The two forms of what INDRA SERVES, then the one
# model the 2023 paper actually published, then the reader gates, then our own
# two reimplementations — which are neither served nor published and are carried
# for completeness, not as anybody's baseline.  Not sortable, not configurable;
# the figure re-orders by queue size for layout, which is not a data decision.
#
# `kind` is a FROZEN contract tag with exactly two values: a belief model that
# scores by formula or fit ("paper-model", the string the shipped viewer
# contract keys on) versus a reader gate. It is NOT an attribution — `provenance`
# is, and it is the field any sentence about whose model this is must read.
ARMS = [
    {
        "name": "INDRA SimpleScorer, default priors",
        "kind": "paper-model",
        "provenance": "indra-served",
        "model_key": "current_indra_simple_default",
        "source": "jsonl",
        "scores": ("data/results/current_indra_simple_paper_20260717/"
                   "current_indra_simple_default_predictions.jsonl"),
        "note": "INDRA's shipped unfitted noisy-OR at its default per-source "
                "priors, over the statement's own evidence. This is what INDRA "
                "serves today. It is NOT the paper's MCMC-refit Belief Orig.",
    },
    {
        "name": "INDRA SimpleScorer + hierarchy",
        "kind": "paper-model",
        "provenance": "indra-served",
        "model_key": "current_simple_hierarchy_all_sources",
        "source": "jsonl",
        "scores": ("data/results/current_indra_hierarchy_paper_20260717/"
                   "current_simple_hierarchy_all_sources_predictions.jsonl"),
        "note": "The same shipped scorer with belief propagated over the "
                "statement hierarchy (direct plus non-negated descendant "
                "evidence), which is the stronger of the two served forms on "
                "this panel. The paper publishes no hierarchy arm.",
    },
    {
        # The one arm on this panel the 2023 paper actually published, and the
        # comparator the operational block reports against.  Deliberately NOT at
        # index 0: it is the strongest belief model here, and the shipped
        # contract's negative controls need index 0 to be a weaker one.
        "name": "Paper RF 2k-d13 + Type/#PMIDs/promoter",
        "kind": "paper-model",
        "provenance": "paper-published",
        "model_key": "paper_literal_rf_promoter",
        "source": "paper-literal-oof",
        "scores": PAPER_LITERAL,
        "literal_key": HEADLINE,
        "note": "The 2023 paper's own released random forest, re-run from "
                "sorgerlab/indra_assembly_paper on the paper's feature matrix "
                "and its own 10 folds, scored out of fold. A research model: "
                "it is not in indra and has never been served.",
    },
    {
        "name": "Gemma 4 26B gate",
        "kind": "llm-gate",
        "provenance": "reader-gate",
        "model_key": "gemma_4_26b",
        "source": "jsonl",
        "scores": f"{MODELS_DIR}/gemma_4_26b/all_source_predictions.jsonl",
        "manifest": f"{MODELS_DIR}/gemma_4_26b/manifest.json",
        "note": "INDRA's own noisy-OR applied to the evidence the reader KEPT.",
    },
    {
        "name": "GLM-5 gate",
        "kind": "llm-gate",
        "provenance": "reader-gate",
        "model_key": "glm_5",
        "source": "jsonl",
        "scores": f"{MODELS_DIR}/glm_5/all_source_predictions.jsonl",
        "manifest": f"{MODELS_DIR}/glm_5/manifest.json",
        "note": "INDRA's own noisy-OR applied to the evidence the reader KEPT.",
    },
    {
        "name": "Gemma 4 31B gate",
        "kind": "llm-gate",
        "provenance": "reader-gate",
        "model_key": "gemma_4_31b",
        "source": "jsonl",
        "scores": f"{MODELS_DIR}/gemma_4_31b/all_source_predictions.jsonl",
        "manifest": f"{MODELS_DIR}/gemma_4_31b/manifest.json",
        "note": "INDRA's own noisy-OR applied to the evidence the reader KEPT.",
    },
    # The weakest reader arm, and the one that LOSES this comparison: at the
    # headline target it needs a LONGER queue than the paper's RF at LOWER
    # precision.  It is here precisely because omitting it would let the panel
    # assert a universal over "every reader gate" that is false.
    {
        "name": "Gemma 4 E2B gate",
        "kind": "llm-gate",
        "provenance": "reader-gate",
        "model_key": "gemma_4_e2b",
        "source": "jsonl",
        "scores": f"{MODELS_DIR}/gemma_4_e2b/all_source_predictions.jsonl",
        "manifest": f"{MODELS_DIR}/gemma_4_e2b/manifest.json",
        "note": "INDRA's own noisy-OR applied to the evidence the reader KEPT.",
    },
    {
        "name": "Our BayesianScorer, source+subtype refit",
        "kind": "paper-model",
        "provenance": "our-reimplementation",
        "model_key": "current_bayesian_source_subtype_oof",
        "source": "jsonl",
        "scores": ("data/results/current_indra_bayesian_paper_20260717/"
                   "current_bayesian_source_subtype_oof_all_sources_predictions.jsonl"),
        "note": "OUR out-of-fold refit of the source (and subtype) "
                "reliabilities over INDRA's feature superset. The paper "
                "publishes no Bayesian or subtype arm; this is not theirs.",
    },
    {
        "name": "Our CountsScorer RF, full features",
        "kind": "paper-model",
        "provenance": "our-reimplementation",
        "model_key": "current_counts_full_features_oof",
        "source": "jsonl",
        "scores": ("data/results/current_indra_counts_hybrid_paper_20260718/"
                   "current_counts_full_features_oof_predictions.jsonl"),
        "note": "OUR port of INDRA's CountsScorer over its 77-feature "
                "superset, out of fold. It is not the paper's engineered "
                "feature panel and must never be called 'the paper's RF'.",
    },
]

# The four reader gates, in the frozen run plan's family order — the same family
# scripts/compute_paper_robustness.py corrects over for the ranking margin.
ROBUSTNESS_FAMILY = [
    ("gemma-4-e2b", "Gemma 4 E2B gate"),
    ("gemma-4-26b", "Gemma 4 26B gate"),
    ("gemma-4-31b", "Gemma 4 31B gate"),
    ("glm-5", "GLM-5 gate"),
]
ROBUSTNESS_REFERENCE = "Paper RF 2k-d13 + Type/#PMIDs/promoter"

# The LLM zero pile is only interpretable as "rejected every piece of evidence"
# under the unfitted hard gate.  Pinned, not assumed.
REQUIRED_LLM_AGGREGATION = "indra_default_hard_gate"


def load_panel() -> dict:
    """The 1689 assembled statements carrying a released paper label.

    Returns the panel in ONE object so every downstream block reads the same
    ordering, the same labels, the same folds and the same completeness flags.
    """
    sids, labels, mhash, phash, safe = [], {}, {}, {}, {}
    for r in load_jsonl(GOLD):
        policy = r.get("paper_replication_policy") or {}
        if policy.get("released_paper_correct") is None:
            continue
        sid = r["canonical_corpus"]["statement_id"]
        assert sid not in labels, f"duplicate statement_id in gold: {sid}"
        sids.append(sid)
        labels[sid] = int(policy["released_paper_correct"])
        # matches_hash MUST stay a string: the paper hashes exceed 2^53.
        mhash[sid] = str(r["canonical_corpus"]["matches_hash"])
        phash[sid] = int(r["paper_statement_hash"])
        safe[sid] = bool(policy["label_is_adjudication_safe"])
    sids.sort()
    y = np.array([labels[s] for s in sids], dtype=int)

    # The paper's own out-of-fold run supplies the fold assignment the bootstrap
    # resamples within, and its own copy of the labels — which must agree.
    lit = json.loads((ROOT / PAPER_LITERAL).read_text())
    oof = {int(r["stmt_hash"]): r for r in lit["oof_predictions"][HEADLINE]}
    assert set(oof) == {phash[s] for s in sids}, (
        "the paper's out-of-fold run does not cover exactly the panel")
    assert all(int(oof[phash[s]]["y_true"]) == labels[s] for s in sids), (
        "the paper's out-of-fold labels disagree with the frozen gold")
    folds = np.array([int(oof[phash[s]]["fold_ix"]) for s in sids], dtype=int)

    return {
        "sids": sids,
        "y": y,
        "is_error": y == 0,
        "matches_hash": mhash,
        "paper_hash": phash,
        "adjudication_safe": np.array([safe[s] for s in sids], dtype=bool),
        "folds": folds,
        "literal_oof": {phash[s]: oof[phash[s]] for s in sids},
    }


def load_scores(spec: dict, panel: dict) -> np.ndarray:
    """One arm's belief for every panel statement, in panel order.

    Two shapes, one contract: a prediction jsonl keyed on ``statement_id``, or
    the paper's own out-of-fold block keyed on ``stmt_hash``.  Either way the
    coverage is asserted to be the panel exactly — no missing rows, no extras.
    """
    sids = panel["sids"]
    if spec["source"] == "paper-literal-oof":
        table = {sid: float(panel["literal_oof"][panel["paper_hash"][sid]]["prob_correct"])
                 for sid in sids}
    else:
        rows = load_jsonl(ROOT / spec["scores"])
        table = {r["statement_id"]: float(r["probability_correct"]) for r in rows}
        assert len(table) == len(rows), f"{spec['name']}: duplicate statement_id"
    # (c) exact panel coverage, no extras.
    assert set(table) == set(sids), (
        f"{spec['name']}: score file does not cover the panel exactly "
        f"({len(set(sids) - set(table))} missing, {len(set(table) - set(sids))} extra)")
    return np.array([table[s] for s in sids])


def threshold_for_target(scores: np.ndarray, is_error: np.ndarray,
                         n_errors: int, target: float) -> float:
    """Smallest distinct score tau with recall(belief <= tau) >= target.

    Intermediate values of tau are redundant: the flag set only changes at the
    arm's observed scores, so the observed scores are the complete candidate set.
    """
    taus = np.unique(scores)
    caught = np.array([int(is_error[scores <= t].sum()) for t in taus])
    need = target * n_errors
    reach = np.flatnonzero(caught >= need)
    assert len(reach), (
        f"no threshold reaches target recall {target}: the arm's maximum "
        f"reachable recall is {caught.max() / n_errors:.4f}")
    return float(taus[reach[0]])


def tie_blocks(scores: np.ndarray, is_error: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Ascending distinct-score blocks: (cum_queue, cum_errors, size, errors).

    The blocks ARE the arm's reachable operating points; everything below reads
    off them, so no derivation here can depend on the order the rows arrived in.
    """
    out: list[tuple[int, int, int, int]] = []
    cum_q = cum_e = 0
    for tau in np.unique(scores):
        block = scores == tau
        size = int(block.sum())
        errs = int(is_error[block].sum())
        cum_q += size
        cum_e += errs
        out.append((cum_q, cum_e, size, errs))
    return out


def caught_at_budget(blocks: list[tuple[int, int, int, int]], budget: int) -> float:
    """Expected errors found reading the ``budget`` lowest-scoring statements.

    A budget landing inside a tie block cannot pick which of the tied statements
    to read, so the block contributes its errors pro rata.  See BUDGET_RULE.
    """
    prev_q = prev_e = 0
    for cum_q, cum_e, size, errs in blocks:
        if cum_q <= budget:
            prev_q, prev_e = cum_q, cum_e
            continue
        return prev_e + errs * ((budget - prev_q) / size)
    return float(prev_e)


def prefix_catch(scores: np.ndarray, is_error: np.ndarray, budget: int) -> float:
    """The SAME quantity as ``caught_at_budget``, derived a different way.

    ``caught_at_budget`` walks precomputed distinct-score blocks; this sorts the
    scores and cuts the prefix directly.  The bootstrap needs the second form
    (its resamples have their own tie structure), and (g) asserts the two agree
    on the panel at every swept budget — so the interval and the drawn curve
    cannot silently be measuring different things.
    """
    if budget <= 0:
        return 0.0
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    cum = np.cumsum(is_error[order].astype(float))
    if budget >= len(ordered):
        return float(cum[-1])
    boundary = ordered[budget - 1]
    start = int(np.searchsorted(ordered, boundary, "left"))
    end = int(np.searchsorted(ordered, boundary, "right")) - 1
    before = float(cum[start - 1]) if start > 0 else 0.0
    return before + (float(cum[end]) - before) * (budget - start) / (end - start + 1)


def budget_for_yield(blocks: list[tuple[int, int, int, int]], target: int) -> int:
    """Smallest budget whose expected catch reaches ``target`` errors.

    Minimality is asserted by the caller against caught_at_budget, so this is a
    derivation and a check, not a formula anyone has to trust.
    """
    assert target > 0
    prev_q = prev_e = 0
    for cum_q, cum_e, size, errs in blocks:
        if cum_e >= target:
            # errs > 0 necessarily: prev_e < target <= cum_e = prev_e + errs.
            take = math.ceil((target - prev_e) * size / errs - 1e-9)
            return prev_q + max(1, min(size, take))
        prev_q, prev_e = cum_q, cum_e
    raise AssertionError(f"no budget reaches {target} errors; the arm tops out at {prev_e}")


def build_equal_yield(arms_out: list[dict], scores_by_arm: dict[str, np.ndarray],
                      is_error: np.ndarray, n: int, n_errors: int,
                      extra_budgets: set[int]) -> dict:
    """Fix the YIELD instead of the threshold, and sweep the whole budget curve.

    Every reader gate has an operating point nobody chose — the block of
    statements whose evidence it rejected outright.  For each of them, ask each
    belief model how many reviews it needs to catch the SAME number of errors
    (an oracle threshold, fitted here, in the comparator's favour), and what it
    catches at the gate's own budget instead.
    """
    blocks = {name: tie_blocks(p, is_error) for name, p in scores_by_arm.items()}
    by_name = {a["name"]: a for a in arms_out}
    gates = [a for a in arms_out if a["kind"] == "llm-gate"]
    models = [a for a in arms_out if a["kind"] == "paper-model"]
    assert gates and models, "equal-yield needs at least one gate and one belief model"

    references = []
    for gate in gates:
        pile = gate["zero_pile"]
        assert pile is not None, f"{gate['name']}: gate without a zero pile"
        budget = pile["size"]
        target = pile["true_errors"]
        # The gate's untuned point IS a block boundary, so the tie-fair count
        # lands on it exactly.  This ties the new block to the bar already drawn.
        exact = caught_at_budget(blocks[gate["name"]], budget)
        assert exact == float(target), (
            f"{gate['name']}: zero pile of {budget} catches {exact}, not the "
            f"{target} the drawn bar reports")
        assert target > 0

        comparators = []
        for model in models:
            blk = blocks[model["name"]]
            need = budget_for_yield(blk, target)
            # Minimality, asserted rather than assumed: one fewer review misses.
            assert caught_at_budget(blk, need) + 1e-9 >= target, (
                f"{model['name']}: {need} reviews do not reach {target} errors")
            assert need == 0 or caught_at_budget(blk, need - 1) < target - 1e-9, (
                f"{model['name']}: {need} reviews is not the SMALLEST budget "
                f"reaching {target} errors")
            at_budget = caught_at_budget(blk, budget)
            assert 0.0 <= at_budget <= min(budget, n_errors) + 1e-9
            comparators.append({
                "arm": model["name"],
                "budget_for_equal_yield": need,
                "extra_reviews": need - budget,
                "precision_at_equal_yield": target / need,
                "errors_caught_at_reference_budget": at_budget,
                "shortfall_at_reference_budget": target - at_budget,
                "threshold_fitted_on_this_panel": True,
            })

        references.append({
            "arm": gate["name"],
            "budget": budget,
            "true_errors_caught": target,
            "false_alarms": pile["false_alarms"],
            "precision": pile["precision"],
            "recall": pile["share_of_all_errors"],
            "threshold_fitted_on_this_panel": False,
            "origin": "the arm's zero pile — every statement whose evidence the "
                      "reader rejected outright. No threshold was chosen.",
            "is_whole_flag_set_at_headline_target":
                pile["is_whole_flag_set_at_headline_target"],
            "comparators": comparators,
        })

    # The arm the figure draws: the shortest queue among the gates whose zero
    # pile IS their whole flag set at the headline target — the same rule the
    # zero-pile callout already uses, so the reader is looking at that bar.
    whole = [r for r in references if r["is_whole_flag_set_at_headline_target"]]
    assert whole, "no gate operates at its own zero pile; the callout would have nothing to name"
    shortest = min(r["budget"] for r in whole)
    tied = [r for r in whole if r["budget"] == shortest]
    assert len(tied) == 1, f"ambiguous reference arm: {[r['arm'] for r in tied]}"
    reference = tied[0]

    # The strongest comparator, derived: the one needing the FEWEST extra reviews.
    best = min(reference["comparators"], key=lambda c: c["budget_for_equal_yield"])
    cheapest = [c for c in reference["comparators"]
                if c["budget_for_equal_yield"] == best["budget_for_equal_yield"]]
    assert len(cheapest) == 1, f"ambiguous comparator: {[c['arm'] for c in cheapest]}"
    assert by_name[best["arm"]]["kind"] == "paper-model"

    # --- the whole curve, so budget-dependence is visible, not asserted -------
    grid = sorted({0, n} | set(range(0, n + 1, BUDGET_SWEEP_STEP))
                  | {r["budget"] for r in references} | extra_budgets)
    caught = {}
    for name, blk in blocks.items():
        row = [caught_at_budget(blk, b) for b in grid]
        assert row[0] == 0.0 and row[-1] == float(n_errors), (
            f"{name}: sweep must run from 0 errors at budget 0 to all {n_errors} "
            f"at budget {n}, got {row[0]} .. {row[-1]}")
        for i in range(1, len(row)):
            assert row[i] + 1e-9 >= row[i - 1], f"{name}: sweep is not monotone at budget {grid[i]}"
            assert row[i] <= min(grid[i], n_errors) + 1e-9, (
                f"{name}: sweep claims {row[i]} errors from {grid[i]} reviews")
            # (g) the block walk and the sorted-prefix cut are two implementations
            # of one quantity; the bootstrap uses the second, the curve the first.
            assert abs(row[i] - prefix_catch(scores_by_arm[name], is_error, grid[i])) <= 1e-9, (
                f"{name}: the two catch derivations disagree at budget {grid[i]}")
        caught[name] = row

    advantage = [a - b for a, b in zip(caught[reference["arm"]], caught[best["arm"]])]
    peak_index = max(range(len(grid)), key=lambda i: advantage[i])
    positive = [i for i, v in enumerate(advantage) if v > 0]
    assert positive, "the reference arm never leads its strongest comparator at any budget"
    half = advantage[peak_index] / 2.0
    decayed = [i for i in range(peak_index + 1, len(grid)) if advantage[i] <= half]

    return {
        "operating_rule": BUDGET_RULE,
        "oracle_disclosure": ORACLE_DISCLOSURE,
        "reference_arm": reference["arm"],
        "references": references,
        "budget_sweep": {
            "step": BUDGET_SWEEP_STEP,
            "budgets": grid,
            "errors_caught": caught,
            "comparator_arm": best["arm"],
            "advantage": advantage,
            "first_positive_budget": grid[positive[0]],
            "peak_budget": grid[peak_index],
            "peak_advantage": advantage[peak_index],
            "half_peak_decay_budget": grid[decayed[0]] if decayed else None,
            "note": "advantage = the reference arm's expected catch minus the "
                    "comparator's, at the same budget. It is a property of the "
                    "budget, not of the arms: it is negative at small budgets, "
                    "peaks at the reference arm's own operating point, and "
                    "decays back to zero once the budget covers the panel.",
        },
    }


# ---------------------------------------------------------------------------
# Robustness of the OPERATIONAL result: error recall at a fixed review budget.
# ---------------------------------------------------------------------------

def fold_stratified_indices(folds: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """The head-to-head's resampling design, unchanged.

    ONE index vector per resample, SHARED across every arm — that pairing is what
    makes the deltas comparable, and it is also what makes the max-t band valid:
    the arms' bootstrap deltas are drawn jointly, so their correlation is carried
    rather than assumed away.
    """
    rng = np.random.default_rng(seed)
    fold_ids = sorted(set(folds.tolist()))
    idx_by_fold = {f: np.where(folds == f)[0] for f in fold_ids}
    return [
        np.concatenate(
            [rng.choice(idx_by_fold[f], size=len(idx_by_fold[f]), replace=True) for f in fold_ids]
        )
        for _ in range(n_boot)
    ]


def error_recall_panel(mask: np.ndarray, panel: dict, scores_by_arm: dict[str, np.ndarray],
                       panel_id: str, role: str) -> dict:
    """One panel's error-recall deltas, pointwise interval and max-t band.

    No model is refit and no score is recomputed between panels: the sensitivity
    panel scores the SAME prediction vectors on fewer statements.  The only thing
    that changes is which labels are admitted.
    """
    is_error = panel["is_error"][mask]
    folds = panel["folds"][mask]
    probs = {name: vector[mask] for name, vector in scores_by_arm.items()}
    n = int(mask.sum())
    n_errors = int(is_error.sum())
    budget = int(BUDGET_SHARE * n)
    assert 0 < budget < n, f"{panel_id}: degenerate budget {budget}"
    base = probs[ROBUSTNESS_REFERENCE]

    boot_idx = fold_stratified_indices(folds, N_BOOT, SEED)

    def recall(vector: np.ndarray, errors: np.ndarray, take: int) -> float:
        return prefix_catch(vector, errors, take) / int(errors.sum())

    reference_recall = recall(base, is_error, budget)
    draws: dict[str, list[float]] = {label: [] for _, label in ROBUSTNESS_FAMILY}
    for take in boot_idx:
        errors_b = is_error[take]
        if not errors_b.any():
            continue
        base_b = recall(base[take], errors_b, budget)
        for _, label in ROBUSTNESS_FAMILY:
            draws[label].append(recall(probs[label][take], errors_b, budget) - base_b)

    widths = {label: len(rows) for label, rows in draws.items()}
    assert len(set(widths.values())) == 1, (
        f"{panel_id}: arms were measured on different numbers of resamples ({widths}); "
        f"the max-t band requires one shared draw per arm per resample")

    labels = [label for _, label in ROBUSTNESS_FAMILY]
    matrix = np.column_stack([np.array(draws[label]) for label in labels])
    centers = np.array([recall(probs[label], is_error, budget) - reference_recall
                        for label in labels])
    ses = matrix.std(ddof=1, axis=0)
    assert (ses > 0).all(), f"{panel_id}: a degenerate bootstrap standard error"
    t = np.abs((matrix - centers) / ses)
    critical = float(np.percentile(t.max(axis=1), 100 * (1 - FAMILY_ALPHA)))
    bonferroni = statistics.NormalDist().inv_cdf(1 - FAMILY_ALPHA / (2 * len(labels)))
    assert POINTWISE_NORMAL_Z < critical <= bonferroni, (
        f"{panel_id}: max-t critical value {critical:.4f} is outside "
        f"({POINTWISE_NORMAL_Z:.4f}, {bonferroni:.4f}] — a simultaneous band cannot be "
        f"narrower than pointwise nor wider than Bonferroni")

    arms = {}
    for index, label in enumerate(labels):
        column = matrix[:, index]
        low, high = float(np.percentile(column, 2.5)), float(np.percentile(column, 97.5))
        band = (float(centers[index] - critical * ses[index]),
                float(centers[index] + critical * ses[index]))
        arms[label] = {
            "error_recall": float(recall(probs[label], is_error, budget)),
            "delta": float(centers[index]),
            "delta_bootstrap_mean": float(column.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "bootstrap_se": float(ses[index]),
            "p_greater_than_zero": float((column > 0).mean()),
            "simultaneous_low": band[0],
            "simultaneous_high": band[1],
            "excludes_zero_pointwise": bool(low > 0 or high < 0),
            "excludes_zero_simultaneous": bool(band[0] > 0 or band[1] < 0),
            "n_valid_resamples": int(len(column)),
        }
        # The delta the interval is centred on IS recall(arm) - recall(reference).
        assert abs(arms[label]["error_recall"] - reference_recall
                   - arms[label]["delta"]) <= 1e-12, f"{panel_id}/{label}: delta does not reconstruct"

    return {
        "id": panel_id,
        "role": role,
        "n_statements": n,
        "n_errors": n_errors,
        "budget": budget,
        "budget_share_of_panel": budget / n,
        "reference_error_recall": reference_recall,
        "max_t_critical_value": critical,
        "bonferroni_critical_value": bonferroni,
        "arms": arms,
    }


def build_error_recall_robustness(panel: dict, scores_by_arm: dict[str, np.ndarray],
                                  headline_arm: str) -> dict:
    """Both panels, the multiplicity correction, and the label-completeness split."""
    n = len(panel["sids"])
    safe = panel["adjudication_safe"]
    dropped = ~safe
    n_dropped = int(dropped.sum())
    # The sensitivity check is only meaningful if the dropped rows are what the
    # field says they are.  Assert it rather than describe it.
    assert n_dropped > 0, "no statement is flagged adjudication-unsafe"
    assert bool((panel["y"][dropped] == 0).all()), \
        "an adjudication-unsafe statement is not a negative in the paper's labels"

    primary = error_recall_panel(np.ones(n, dtype=bool), panel, scores_by_arm,
                                "paper_labels_1689", "primary")
    sensitivity = error_recall_panel(safe, panel, scores_by_arm,
                                     "adjudication_safe_1578", "sensitivity")
    assert sensitivity["n_errors"] == primary["n_errors"] - n_dropped, \
        "the label-completeness check must drop negatives only"
    assert headline_arm in primary["arms"], f"{headline_arm} is not in the robustness family"

    return {
        "metric": "error recall at a fixed review budget — the expected share of "
                  "the panel's known errors found by reviewing the B "
                  "lowest-scoring statements, with tied scores contributing pro "
                  "rata (the same rule the budget sweep uses)",
        "budget_share": BUDGET_SHARE,
        "budget_rule": f"B = floor({BUDGET_SHARE} x panel n), so both panels are "
                       f"asked the same question at the same share of their own size",
        "reference_arm": ROBUSTNESS_REFERENCE,
        "headline_arm": headline_arm,
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "bootstrap_design": BOOTSTRAP_DESIGN,
        "multiplicity": {
            "family": [label for _, label in ROBUSTNESS_FAMILY],
            "family_ids": [arm_id for arm_id, _ in ROBUSTNESS_FAMILY],
            "family_size": len(ROBUSTNESS_FAMILY),
            "family_alpha": FAMILY_ALPHA,
            "method": "studentized max-t over the shared paired-bootstrap draws",
            "pointwise_normal_critical_value": POINTWISE_NORMAL_Z,
            "note": "The frozen run plan stages all four reader arms and "
                    "designates none of them as the confirmatory one, so a "
                    "simultaneous band over the family is the fair ask. The arms "
                    "move together from resample to resample, so max-t costs far "
                    "less than Bonferroni would.",
        },
        "label_completeness": {
            "field": "paper_replication_policy.label_is_adjudication_safe",
            "n_dropped": n_dropped,
            "all_dropped_are_negative": True,
            "dropped_share_of_all_negatives": n_dropped / primary["n_errors"],
            "dropped_share_of_panel": n_dropped / n,
            "no_model_is_refit": True,
            "note": "These statements are NEGATIVE in the paper's released "
                    "labels and their evidence review is incomplete. Dropping "
                    "them is OUR revision of THEIR labels, so this is a "
                    "sensitivity panel and never the primary result. It is not "
                    "free either: it removes a quarter of the negatives and "
                    "changes what the panel is a sample of.",
        },
        "panels": {"primary": primary, "sensitivity": sensitivity},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--manifest", default=None,
                    help="run manifest.json to record the output path + sha256 in")
    args = ap.parse_args()

    panel = load_panel()
    sids, y, is_error = panel["sids"], panel["y"], panel["is_error"]
    n = len(sids)
    n_errors = int(is_error.sum())
    assert n_errors > 0 and n_errors < n

    # (h) the one arm we attribute to the 2023 paper is keyed on a method string
    # the paper actually published, matched verbatim.
    published = {m["method"] for m in
                 json.loads((ROOT / PUBLISHED_METHODS).read_text())["methods"]}
    assert HEADLINE in published, (
        f"{HEADLINE!r} is not one of the {len(published)} method strings the paper "
        f"published; the paper-model arm would be attributing a name they never used")
    assert "‐" not in HEADLINE and "–" not in HEADLINE, \
        "the published method string must carry a U+002D hyphen"

    # (c') panel identity is pinned on matches_hash too, not just the UUID.
    prov = {r["statement_id"]: str(r["matches_hash"]) for r in load_jsonl(PROVENANCE)}
    missing_prov = [s for s in sids if s not in prov]
    assert not missing_prov, (
        f"{len(missing_prov)} panel statements absent from {PROVENANCE}")
    mismatched = [s for s in sids if prov[s] != panel["matches_hash"][s]]
    assert not mismatched, (
        f"{len(mismatched)} statement_id -> matches_hash disagreements between the "
        f"paper gold and {PROVENANCE}, e.g. {mismatched[:3]}")

    print(f"panel n={n}  errors={n_errors}  base rate={n_errors / n:.4f}  "
          f"adjudication-safe={int(panel['adjudication_safe'].sum())}")

    arms_out = []
    scores_by_arm: dict[str, np.ndarray] = {}
    for spec in ARMS:
        p = load_scores(spec, panel)
        scores_by_arm[spec["name"]] = p

        grid = []
        for target in TARGET_RECALLS:
            tau = threshold_for_target(p, is_error, n_errors, target)
            flagged = p <= tau
            queue = int(flagged.sum())
            real = int(is_error[flagged].sum())
            wasted = queue - real
            precision = real / queue
            recall = real / n_errors
            # (a) the arithmetic identities, exactly.
            assert queue == real + wasted, (
                f"{spec['name']}@{target}: queue {queue} != real {real} + wasted {wasted}")
            assert precision == real / queue
            # real/queue is a single correctly-rounded division, so the inverse
            # identity holds only to within one ulp; assert it at that scale
            # rather than exactly (an exact test fails on e.g. 325/676).
            assert abs(precision * queue - real) <= 1e-9 * max(1, real), (
                f"{spec['name']}@{target}: precision {precision!r} does not "
                f"reconstruct {real}/{queue}")
            # (b) the threshold search delivered the target.
            assert recall >= target, (
                f"{spec['name']}@{target}: achieved recall {recall:.6f} < target {target}")
            grid.append({
                "target_recall": target,
                "tau": tau,
                "queue": queue,
                "true_errors_caught": real,
                "false_alarms": wasted,
                "precision": precision,
                "recall_achieved": recall,
                "recall_overshoot": recall - target,
                "queue_share_of_panel": queue / n,
            })

        headline = next(g for g in grid if g["target_recall"] == HEADLINE_TARGET)

        zero_pile = None
        if spec["kind"] == "llm-gate":
            man = json.loads((ROOT / spec["manifest"]).read_text())
            notes = man["implementation"]["notes"]
            # (d) the zero pile only means "rejected everything" under the
            # unfitted hard gate.
            assert notes["aggregation"] == REQUIRED_LLM_AGGREGATION, (
                f"{spec['name']}: bundle aggregation is {notes['aggregation']!r}, not "
                f"{REQUIRED_LLM_AGGREGATION!r}; belief == 0.0 no longer implies "
                "'the reader rejected every piece of evidence'")
            assert notes["reader_profile"] is None, (
                f"{spec['name']}: bundle carries a fitted reader profile "
                f"({notes['reader_profile']!r}); the soft path floors belief at "
                "sigmoid(prior_logodds), not 0.0")
            z = p == 0.0
            assert z.any(), f"{spec['name']}: no statement scores exactly 0.0"
            assert float(p.min()) == 0.0, (
                f"{spec['name']}: minimum score is {p.min()!r}, not 0.0")
            size = int(z.sum())
            errs = int(is_error[z].sum())
            zero_pile = {
                "definition": "belief exactly 0.0 == the reader rejected every "
                              "piece of evidence it read (hard gate: a source with "
                              "no surviving evidence leaves the noisy-OR product "
                              "entirely, and an empty product is belief 0)",
                "size": size,
                "true_errors": errs,
                "false_alarms": size - errs,
                "precision": errs / size,
                "share_of_all_errors": errs / n_errors,
                "is_whole_flag_set_at_headline_target":
                    bool(headline["tau"] == 0.0 and headline["queue"] == size),
                "requires_no_threshold_tuning": True,
            }
            assert zero_pile["size"] == zero_pile["true_errors"] + zero_pile["false_alarms"]

        # Coarseness: how many DISTINCT operating points the grid actually
        # resolves.  1 means the whole grid is a single tie block.
        distinct_ops = sorted({g["queue"] for g in grid})

        arms_out.append({
            "name": spec["name"],
            "kind": spec["kind"],
            "provenance": spec["provenance"],
            "model_key": spec["model_key"],
            "note": spec["note"],
            "distinct_scores": int(len(np.unique(p))),
            "operating_point": {
                "target_recall": HEADLINE_TARGET,
                "tau": headline["tau"],
                "queue": headline["queue"],
                "true_errors_caught": headline["true_errors_caught"],
                "false_alarms": headline["false_alarms"],
                "precision": headline["precision"],
                "recall_achieved": headline["recall_achieved"],
            },
            "precision_at_matched_recall": grid,
            "n_distinct_queue_sizes_across_targets": len(distinct_ops),
            "zero_pile": zero_pile,
            "scores_path": spec["scores"],
        })

        print(f"{spec['name']:<40} tau={headline['tau']:<20.12g} "
              f"queue={headline['queue']:4d} real={headline['true_errors_caught']:4d} "
              f"wasted={headline['false_alarms']:4d} "
              f"precision={headline['precision'] * 100:5.1f}%  "
              f"recall={headline['recall_achieved'] * 100:5.1f}%"
              + (f"  zero-pile {zero_pile['size']}/{zero_pile['true_errors']} "
                 f"({zero_pile['precision'] * 100:.1f}% prec, "
                 f"{zero_pile['share_of_all_errors'] * 100:.1f}% of errors)"
                 if zero_pile else ""))

    # --- the OTHER half of what the gate design costs -----------------------
    # The zero pile above is the reader's own loss (true statements it zeroes
    # into an unrankable block).  This is the loss the FORMULA has already
    # taken before any reader runs: true statements the unfiltered noisy-OR
    # already scores below the promotion bar.  Because a reader can only remove
    # evidence, no reader arm can lift any of them back over the bar — asserted
    # below on the arms themselves, not assumed.  The reference is the DIRECT
    # served scorer, named by model_key rather than by position: the gates are
    # that exact scorer over kept evidence, so it is the only arm the
    # subtractive premise holds against.
    ceiling_spec = next(s for s in ARMS if s["model_key"] == "current_indra_simple_default")
    ceiling_arm = ceiling_spec["name"]
    base = scores_by_arm[ceiling_arm]
    is_true = ~is_error
    n_true = int(is_true.sum())
    below = is_true & (base < PROMOTION_THRESHOLD)
    n_below = int(below.sum())
    assert 0 < n_below <= n_true, (
        f"promotion ceiling {n_below} is not a subset count of {n_true} true statements")
    for spec in ARMS:
        if spec["kind"] != "llm-gate":
            continue
        reader = scores_by_arm[spec["name"]]
        assert bool((reader[below] <= base[below]).all()), (
            f"{spec['name']} scores a below-threshold true statement ABOVE the "
            f"unfiltered noisy-OR; the subtractive-only premise of the promotion "
            f"ceiling no longer holds")
    # The ceiling and a reader's zero pile OVERLAP heavily and predictably: a true
    # statement carried by one weak-source evidence is both already below the bar
    # AND exactly what a reader zeroes when it rejects that evidence.  Derive the
    # overlap and union per arm so no prose has to assert the relation.
    ceiling_overlap = {}
    for spec in ARMS:
        if spec["kind"] != "llm-gate":
            continue
        reader = scores_by_arm[spec["name"]]
        zeroed = is_true & (reader <= 1e-12)
        both = below & zeroed
        union = below | zeroed
        ceiling_overlap[spec["name"]] = {
            "n_true_zeroed_by_arm": int(zeroed.sum()),
            "n_also_already_below_threshold": int(both.sum()),
            "n_newly_lost_by_arm": int((zeroed & ~below).sum()),
            "n_true_affected_union": int(union.sum()),
            "share_of_true_affected": float(union.sum() / n_true),
        }
        assert (ceiling_overlap[spec["name"]]["n_also_already_below_threshold"]
                + ceiling_overlap[spec["name"]]["n_newly_lost_by_arm"]
                == int(zeroed.sum())), "overlap split does not close"
        assert int(union.sum()) == n_below + int((zeroed & ~below).sum()), (
            "union is not the ceiling plus the newly-lost")

    print(f"\npromotion ceiling: {n_below} of {n_true} true statements are already "
          f"below {PROMOTION_THRESHOLD} under {ceiling_arm}")
    for name, blk in ceiling_overlap.items():
        print(f"  {name}: zeroes {blk['n_true_zeroed_by_arm']} true "
              f"({blk['n_also_already_below_threshold']} already below the bar, "
              f"{blk['n_newly_lost_by_arm']} newly lost) -> union "
              f"{blk['n_true_affected_union']} ({blk['share_of_true_affected']:.1%})")

    # --- fix the yield instead of the threshold -----------------------------
    # The robustness budget joins the sweep grid so the drawn curve passes
    # THROUGH the point the interval is quoted at, and the cross-check below is
    # an equality rather than an interpolation.
    robustness_budget = int(BUDGET_SHARE * n)
    equal_yield = build_equal_yield(arms_out, scores_by_arm, is_error, n, n_errors,
                                    {robustness_budget})
    ref = next(r for r in equal_yield["references"]
               if r["arm"] == equal_yield["reference_arm"])
    sweep = equal_yield["budget_sweep"]
    print(f"\nequal yield vs {ref['arm']} at its own untuned point "
          f"({ref['budget']} reviews, {ref['true_errors_caught']} errors):")
    for c in ref["comparators"]:
        print(f"  {c['arm']:<40} needs {c['budget_for_equal_yield']:4d} reviews "
              f"({c['extra_reviews']:+d}) for the same {ref['true_errors_caught']}; "
              f"at {ref['budget']} it catches {c['errors_caught_at_reference_budget']:.1f} "
              f"({c['shortfall_at_reference_budget']:.1f} short)")
    print(f"  sweep vs {sweep['comparator_arm']}: first positive at "
          f"{sweep['first_positive_budget']}, peak {sweep['peak_advantage']:.1f} at "
          f"{sweep['peak_budget']}, halved by {sweep['half_peak_decay_budget']}")

    # --- is the operational gap real? ---------------------------------------
    robustness = build_error_recall_robustness(panel, scores_by_arm, ref["arm"])
    # The interval and the curve must describe the SAME number: the swept catch
    # at the robustness budget, over the panel errors, IS each arm's error recall.
    at = sweep["budgets"].index(robustness_budget)
    primary = robustness["panels"]["primary"]
    assert abs(sweep["errors_caught"][ROBUSTNESS_REFERENCE][at] / n_errors
               - primary["reference_error_recall"]) <= 1e-12, \
        "the swept reference catch and the bootstrapped reference recall disagree"
    for label, block in primary["arms"].items():
        assert abs(sweep["errors_caught"][label][at] / n_errors
                   - block["error_recall"]) <= 1e-12, \
            f"{label}: the swept catch and the bootstrapped recall disagree"

    print(f"\nerror recall at a {BUDGET_SHARE:.0%} budget "
          f"({primary['budget']} reviews), vs {ROBUSTNESS_REFERENCE} "
          f"({primary['reference_error_recall']:.4f}):")
    for panel_key in ("primary", "sensitivity"):
        blk = robustness["panels"][panel_key]
        print(f"  {panel_key} (n={blk['n_statements']}, B={blk['budget']}, "
              f"max-t {blk['max_t_critical_value']:.3f})")
        for label, row in blk["arms"].items():
            print(f"    {label:<20} recall {row['error_recall']:.4f} "
                  f"delta {row['delta']:+.4f} "
                  f"pointwise [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] "
                  f"simultaneous [{row['simultaneous_low']:+.4f}, "
                  f"{row['simultaneous_high']:+.4f}]"
                  + ("  EXCLUDES ZERO" if row["excludes_zero_simultaneous"] else ""))

    # --- caveats, with every number derived ---------------------------------
    by_name = {a["name"]: a for a in arms_out}
    gate = by_name[ref["arm"]]
    gate_grid = {g["target_recall"]: g for g in gate["precision_at_matched_recall"]}
    repeated = sorted({t for t, g in gate_grid.items() if g["queue"] == ref["budget"]})
    next_point = min((g for g in gate_grid.values() if g["queue"] > ref["budget"]),
                     key=lambda g: g["queue"], default=None)
    assert next_point is not None, f"{gate['name']} has a single reachable operating point"
    flat = [a for a in arms_out
            if a["kind"] == "llm-gate" and a["n_distinct_queue_sizes_across_targets"] == 1]
    top_target = max(TARGET_RECALLS)
    top_gate = gate_grid[top_target]
    top_cmp = next(g for g in by_name[sweep["comparator_arm"]]["precision_at_matched_recall"]
                   if g["target_recall"] == top_target)
    spelled = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

    def series(items: list[str]) -> str:
        """`a, b and c` — an Oxford-free list, so a derived count reads as prose."""
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    caveats = [
        "The 2023 paper NEVER published a decision or threshold metric — only "
        "trapezoidal PR-AUC as a 10-fold mean with a population SD. This figure "
        "is OUR derivation. Exactly one belief model here is theirs: "
        f"{sweep['comparator_arm']}, their own released code re-run out of fold. "
        "Two are INDRA's shipped SimpleScorer, which is not their MCMC-refit "
        "Belief Orig, and two are our own reimplementations over INDRA's feature "
        "superset, which they never published. Nothing here was reported by the "
        "paper.",

        "Each arm's threshold is chosen on THIS SAME PANEL to catch at least the "
        "target share of errors. That is an operating-point choice, not a held-out "
        "result, and no arm's threshold is validated out of sample. The arms are "
        "asked for the same catch rate but do not all achieve it — an arm that "
        "cannot land on the target overshoots it — so read the bars as a "
        "both-axes-at-once comparison (shorter queue AND more errors caught), not "
        "as a like-for-like held at equal recall.",

        "The LLM arms can only be operated at a few discrete points. Their lowest "
        "score is an exact tie shared by hundreds of statements — every statement "
        "whose evidence the reader rejected outright — and that one block is the "
        f"whole flagged set here, so the threshold cannot be tuned finely. "
        f"{gate['name']} returns the same {ref['budget']} statements at the "
        + series([f"{int(t * 100)}%" for t in repeated])
        + " targets; "
        + "; ".join(f"{a['name']} returns the same "
                    f"{a['operating_point']['queue']} at all "
                    f"{spelled[len(TARGET_RECALLS)]}" for a in flat)
        + f"; the next reachable point for {gate['name']} jumps the queue to "
        f"{next_point['queue']} and drops precision to "
        f"{next_point['precision'] * 100:.1f}%, with nothing in between. That is "
        "the cost of the score ties this page examines in the score-distribution "
        "panel below.",

        f"The headline target of {int(HEADLINE_TARGET * 100)}% is a choice, and "
        f"the queue ordering depends on it: at an {int(top_target * 100)}% target "
        f"{gate['name']}'s queue grows to {top_gate['queue']}, within "
        f"{abs(top_gate['queue'] - top_cmp['queue'])} of {sweep['comparator_arm']}'s "
        f"{top_cmp['queue']} — the margin has closed. What the figure shows is a "
        "property of this operating point, not of every operating point.",

        "The four reader arms and the four non-paper belief models are scored on "
        "CURRENT INDRA evidence. The paper's own RF is scored on the paper's own "
        "2023 feature matrix and its own folds, because that is what re-running "
        "their released code produces. The arms therefore compare cleanly to each "
        "other on the same statements and the same labels, but only loosely to "
        "the paper's published 2023 table.",

        "The reader arms are INDRA's own noisy-OR applied to the evidence the "
        "reader kept, so this compares belief models on a shared aggregation; the "
        "reader's contribution is the filtering. The reader arms are also not "
        "zero-shot: each call carries 14 hand-authored demonstration pairs.",

        "The interval on the operational result is a review-budget metric, not a "
        "ranking metric. It is the one place on this page where a reader gate is "
        "measured on what it actually does — decide — rather than on the ranking "
        "the 2023 paper reported, and the two do not move together: the ranking "
        "margin shrinks on the adjudication-safe panel while this one grows.",
    ]

    payload = {
        "artifact_kind": "statement_review_queue",
        # 3 is DELIBERATELY SKIPPED and must stay unknown forever: the shipped
        # viewer contract runner uses the literal schema_version 3 as its
        # negative control for "an unknown version fails closed", so emitting a
        # 3 would silently turn that assertion into a pass. No 3 was ever
        # written. 4 carries the paper's own released RF and INDRA's served
        # hierarchy variant as arms, per-arm `provenance`, and
        # `error_recall_robustness`.
        "schema_version": 4,
        "decision_rule": "flag a statement as WRONG iff belief <= tau",
        "threshold_rule": "tau = the smallest of the arm's own distinct scores whose "
                          "flag set reaches the target recall of known errors",
        "headline_target_recall": HEADLINE_TARGET,
        "target_recalls": TARGET_RECALLS,
        "panel": {
            "n": n,
            "n_errors": n_errors,
            "n_correct": int((~is_error).sum()),
            "error_base_rate": n_errors / n,
            "label": "paper_replication_policy.released_paper_correct",
            "ordering": "sorted(statement_id)",
        },
        "promotion_ceiling": {
            "threshold": PROMOTION_THRESHOLD,
            "reference_arm": ceiling_arm,
            "n_true": n_true,
            "n_true_below_threshold": n_below,
            "per_arm_overlap": ceiling_overlap,
            "why": "The reader arms are INDRA's own noisy-OR run over the "
                   "evidence the reader KEPT, so a reader can only remove "
                   "belief, never add it. Every true statement the unfiltered "
                   f"noisy-OR already scores below {PROMOTION_THRESHOLD} is "
                   "therefore unpromotable by every reader arm here, however "
                   "well it reads. Verified in code: no reader arm scores any "
                   "of these statements above the unfiltered noisy-OR. The "
                   "threshold is a stated illustrative promotion bar, not a "
                   "fitted one, and the count is monotone in it. NOTE: this "
                   "count and a reader's zero pile OVERLAP — see "
                   "per_arm_overlap; they are NOT additive.",
        },
        "arms": arms_out,
        "equal_yield": equal_yield,
        "error_recall_robustness": robustness,
        "caveats": caveats,
        "checks": {
            "queue_equals_real_plus_wasted": True,
            "equal_yield_budget_is_minimal": True,
            "equal_yield_reference_is_the_arms_own_zero_pile": True,
            "budget_sweep_is_monotone_and_closes_at_every_error": True,
            "precision_equals_real_over_queue": True,
            "recall_achieved_at_least_target": True,
            "every_arm_covers_the_panel_exactly": True,
            "gold_matches_hash_agrees_with_prediction_provenance": True,
            "llm_zero_pile_is_the_minimum_score_tie_block": True,
            "two_catch_derivations_agree_at_every_swept_budget": True,
            "swept_catch_agrees_with_bootstrapped_error_recall": True,
            "paper_model_key_is_a_published_method_string": True,
            "llm_bundles_use_unfitted_hard_gate": REQUIRED_LLM_AGGREGATION,
            "note": "Assertions are enforced in code; a violation fails the build "
                    "rather than being reported here as False.",
        },
        "provenance": {
            "gold": GOLD,
            "matches_hash_crosscheck": PROVENANCE,
            "published_methods": PUBLISHED_METHODS,
            "scores": {a["name"]: a["scores_path"] for a in arms_out},
            "join": "statement_id (paper gold canonical_corpus.statement_id); the "
                    "paper matches_hash is cross-checked but not used to score, and "
                    "the paper's own out-of-fold block is joined on "
                    "paper_statement_hash",
            "generated_by": "scripts/compute_statement_review_queue.py",
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    if args.manifest:
        mpath = Path(args.manifest)
        man = json.loads(mpath.read_text())
        man.setdefault("outputs", {})["review_queue"] = out.name
        man.setdefault("output_sha256", {})[out.name] = sha
        mpath.write_text(json.dumps(man, indent=2) + "\n")
        print(f"recorded sha256 in {mpath}")

    print(f"\nwrote {out} ({out.stat().st_size} bytes)\nsha256 {sha}")
    print("\nprecision at matched recall (%)")
    print(f"{'arm':<40}" + "".join(f"{int(t * 100):>8}%" for t in TARGET_RECALLS))
    for a in arms_out:
        print(f"{a['name']:<40}" +
              "".join(f"{g['precision'] * 100:>8.1f} " for g in a["precision_at_matched_recall"]))


if __name__ == "__main__":
    main()
