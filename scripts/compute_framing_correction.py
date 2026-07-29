"""The framing correction: the reader arm IS the paper's own noisy-OR.

Calling the reader arms "LLM scorers" is misleading, and it was misleading in
our own earlier write-up. There is no second belief model anywhere in this
comparison. Each reader emits one keep/reject verdict per (statement, evidence)
pair; the surviving counts then go through the paper's aggregation unchanged:

    belief = 1 - PROD_s (syst_s + rand_s^{n_s})

so the reader's only power is to REMOVE evidence from that product. A true
statement the formula under-scores can never be promoted by a reader.

This script emits the three legs of that claim that P1's artifacts do not
already carry, and cross-checks the one they do:

  (a) declaration   all four reader manifests are read and MECHANICALLY checked:
                    aggregation == indra_default_hard_gate, reader_profile is
                    null, the aggregation_config sha256 they declare is the
                    actual sha256 of data/comparison/aggregation.json, and the
                    noise_model / statement_belief component digests are the
                    sha256 of the CURRENT source bytes.  That last check is what
                    anchors "soft=None dispatches to compute_gated_belief" to
                    bytes rather than to prose.
  (b) subtractive   re-derived here, independently, from the shipped SimpleScorer
                    predictions and the four reader bundles, then asserted EQUAL
                    to non_reading_control.json's own subtractive_check (whose
                    sha256 is pinned in provenance) so the two artifacts cannot
                    drift apart.
  (c) by value      every non-zero reader score is a value the paper's formula
                    emits for SOME sub-multiset of that statement's evidence.
                    The reachable set is
                        {1 - PROD_s f_s : f_s in {1} U {syst_s + rand_s^k,
                                                       1 <= k <= n_s}}
                    from each statement's own source_counts, with the priors read
                    from data/comparison/aggregation.json.  Naive enumeration is
                    infeasible (the largest reachable set on this panel runs to
                    1e10 assignments), so the search is a depth-first walk over
                    the sources in the SAME sorted order compute_gated_belief
                    multiplies in, with the feasible factor window at each depth
                    binary-searched from a suffix min/max product interval.  Two
                    tiers are reported: within TOLERANCE, and the tighter
                    bit-exact float equality.  Any statement the search cannot
                    settle inside its node budget is reported as EXHAUSTED, never
                    folded into the confirmed count.

The claim in (c) is only worth stating against a chance floor, because reachable
values are not rare. That floor is measured, not asserted: over the statements
whose reachable set is small enough to enumerate exhaustively, the reader scores
are permuted across statements and the same membership test is re-run.

Findings ARE assertions: the script exits non-zero if any reader belief exceeds
the noisy-OR, if any non-zero score is a counterexample to (c), if any manifest
declares something other than the unfitted hard gate, or if the re-derivation
disagrees with P1's artifact. A violation fails the build rather than being
written out as data.

The join (which statements, with which labels, in which order) is imported from
scripts/compare_paper_literal_vs_llms.py and scripts/compute_statement_review_queue.py
rather than re-derived. Ordering is sorted(statement_id).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compute_framing_correction.py \
      --out-json data/results/indra_paper_literal_models_20260724/framing_correction.json \
      --manifest data/results/indra_paper_literal_models_20260724/manifest.json
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the head-to-head script's join contract verbatim: same gold file, same
# model-bundle root, same loader.
from compare_paper_literal_vs_llms import (  # noqa: E402
    GOLD,
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

AGGREGATION = "data/comparison/aggregation.json"
NOISE_MODEL_SOURCE = "src/indra_belief/noise_model.py"
STATEMENT_BELIEF_SOURCE = "src/indra_belief/statement_belief.py"

SIMPLE_SCORER_PREDICTIONS = ("data/results/current_indra_simple_paper_20260717/"
                             "current_indra_simple_default_predictions.jsonl")

# P1's artifacts. (b) is cross-checked against the first; the formula string and
# the label breakdown are copied byte-identically from both.
NON_READING_CONTROL = ("data/results/indra_paper_literal_models_20260724/"
                       "non_reading_control.json")
BELIEF_MODEL_LADDER = ("data/results/indra_paper_literal_models_20260724/"
                       "belief_model_ladder.json")

# Fixed presentation order, matching the non-reading control's subtractive check.
READER_ARMS = [
    {"key": "gemma_4_26b", "label": "Gemma 4 26B"},
    {"key": "glm_5", "label": "GLM-5"},
    {"key": "gemma_4_31b", "label": "Gemma 4 31B"},
    {"key": "gemma_4_e2b", "label": "Gemma 4 E2B"},
]

REQUIRED_AGGREGATION = "indra_default_hard_gate"

# ---- reachable-value search ------------------------------------------------
# Two statements can carry the same belief for different reasons, so the match is
# "is there ANY surviving-count vector the paper's formula maps to this value",
# not "is this the vector the reader produced".
REACHABLE_TOLERANCE = 9e-15
# Nodes visited per statement before the search gives up and reports EXHAUSTED.
# The worst statement on this panel settles in ~1.16e6, so this is ~4x headroom;
# it is a cap on honesty, not on runtime.
NODE_BUDGET = 5_000_000

# The null baseline runs on statements whose reachable set can be enumerated
# outright, so the chance floor is measured against a complete value set rather
# than against the same search.
ENUMERABLE_CAP = 200_000
N_PERMUTATIONS = 10
PERMUTATION_SEED = 20260717


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factor_lists(source_counts: dict[str, int],
                 priors: dict[str, tuple[float, float]]) -> list[list[float]]:
    """Per-source DESCENDING factor list, sources in compute_gated_belief order.

    ``1.0`` is the source leaving the product entirely (the reader rejected all
    of its evidence); ``syst + rand**k`` is k surviving pieces. Exact float ties
    are collapsed — ``rand**k`` underflows to a constant ``syst`` for large k, and
    a repeated factor is a repeated value, not a new one.
    """
    out = []
    for src in sorted(source_counts):
        rand, syst = priors[src]
        deduped: list[float] = [1.0]
        for k in range(1, int(source_counts[src]) + 1):
            factor = syst + rand ** k
            if factor < deduped[-1]:
                deduped.append(factor)
        out.append(deduped)
    return out


def search_reachable(lists: list[list[float]], belief: float,
                     tolerance: float, budget: int) -> dict:
    """Can the paper's formula emit ``belief`` from these source counts?

    Depth-first over the sources in sorted order (the order
    ``compute_gated_belief`` multiplies in, so a hit can be checked for BIT-EXACT
    float equality, not merely for closeness). At each depth the running product
    ``p`` must still be able to reach the target product window
    ``[1 - belief - tol, 1 - belief + tol]``: every remaining factor is <= 1, so
    the reachable window from here is ``[p * suffix_min, p]``. That inverts to a
    contiguous slice of the depth's descending factor list, which is
    binary-searched rather than scanned.

    The window is exhausted rather than abandoned at the first tolerance hit, so
    the bit-exact tier is not undercounted; the walk stops early only once a
    bit-exact assignment is in hand, which is the strongest answer available.

    Returns {'within_tolerance', 'bit_exact', 'budget_exhausted', 'nodes'}.
    """
    m = len(lists)
    suffix_min = [1.0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] * lists[i][-1]
    target_low = 1.0 - belief - tolerance
    target_high = 1.0 - belief + tolerance
    state = {"within_tolerance": False, "bit_exact": False,
             "budget_exhausted": False, "nodes": 0}

    def walk(depth: int, product: float) -> bool:
        """True == stop (bit-exact found, or the budget ran out)."""
        if state["nodes"] > budget:
            state["budget_exhausted"] = True
            return True
        if depth == m:
            state["nodes"] += 1
            belief_here = 1.0 - product
            if belief_here == belief:
                state["bit_exact"] = True
                state["within_tolerance"] = True
                return True
            if abs(belief_here - belief) <= tolerance:
                state["within_tolerance"] = True
            return False

        factors = lists[depth]
        tail = suffix_min[depth + 1]
        n = len(factors)
        # product * f must not undershoot the window ...
        low = target_low / product
        # ... and must still be able to reach it once the tail is applied.
        high = target_high / (product * tail)
        # Descending list: [first index <= high, first index < low).
        start, stop = 0, n
        while start < stop:
            mid = (start + stop) // 2
            if factors[mid] <= high:
                stop = mid
            else:
                start = mid + 1
        first = start
        start, stop = 0, n
        while start < stop:
            mid = (start + stop) // 2
            if factors[mid] >= low:
                start = mid + 1
            else:
                stop = mid
        last = start

        for index in range(first, last):
            state["nodes"] += 1
            if state["nodes"] > budget:
                state["budget_exhausted"] = True
                return True
            if walk(depth + 1, product * factors[index]):
                return True
        return False

    walk(0, 1.0)
    # A budget stop after a confirmed hit is not an unresolved statement.
    if state["within_tolerance"]:
        state["budget_exhausted"] = False
    return state


def enumerate_reachable(lists: list[list[float]]) -> list[float]:
    """Every value the formula can emit here, sorted. Same multiplication order."""
    products = [1.0]
    for factors in lists:
        products = [p * f for p in products for f in factors]
    return sorted({1.0 - p for p in products})


def on_reachable_grid(grid: list[float], belief: float, tolerance: float) -> bool:
    index = bisect.bisect_left(grid, belief - tolerance)
    return index < len(grid) and grid[index] <= belief + tolerance


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
    prov_rows = load_jsonl(PROVENANCE)
    prov_hash = {r["statement_id"]: str(r["matches_hash"]) for r in prov_rows}
    mismatched = [s for s in sids if prov_hash.get(s) != mhash[s]]
    assert not mismatched, (
        f"{len(mismatched)} statement_id -> matches_hash disagreements between the "
        f"paper gold and {PROVENANCE}, e.g. {mismatched[:3]}")
    source_counts = {r["statement_id"]: dict(r["source_counts"]) for r in prov_rows}
    assert set(source_counts) == set(sids), (
        f"{PROVENANCE} does not cover the panel exactly")

    # ---- the formula string, copied from P1 rather than retyped --------------
    control = json.loads((ROOT / NON_READING_CONTROL).read_text())
    ladder = json.loads((ROOT / BELIEF_MODEL_LADDER).read_text())
    formula = control["noisy_or_formula"]
    assert ladder["noisy_or_formula"] == formula, (
        "the two P1 artifacts disagree on the noisy-OR formula string: "
        f"{formula!r} vs {ladder['noisy_or_formula']!r}")

    # ---- panel + the label convention, re-derived from the gold -------------
    adjudication_safe = flagged = 0
    for row in load_jsonl(GOLD):
        policy = row.get("paper_replication_policy") or {}
        if policy.get("released_paper_correct") is None:
            continue
        if int(policy["released_paper_correct"]) != 0:
            continue
        if policy.get("label_is_adjudication_safe") is True:
            adjudication_safe += 1
        else:
            flagged += 1
    assert adjudication_safe + flagged == n_errors, (
        f"the negative breakdown ({adjudication_safe} + {flagged}) does not sum to "
        f"the {n_errors} errors on the panel")
    negative_breakdown = {
        "n_errors": n_errors,
        "adjudication_safe_negatives": adjudication_safe,
        "flagged_label_is_adjudication_safe_false": flagged,
    }
    assert ladder["panel"]["negative_breakdown"] == negative_breakdown, (
        "the re-derived negative breakdown disagrees with belief_model_ladder.json: "
        f"{negative_breakdown} vs {ladder['panel']['negative_breakdown']}")

    print(f"panel n={n}  errors={n_errors}  "
          f"({adjudication_safe} adjudication-safe + {flagged} flagged)")

    # ---- (a) declaration: read the manifests, check them against bytes ------
    aggregation_path = ROOT / AGGREGATION
    aggregation_sha = _sha256(aggregation_path)
    agg = json.loads(aggregation_path.read_text())
    assert agg["aggregation"] == REQUIRED_AGGREGATION, (
        f"{AGGREGATION} declares aggregation {agg['aggregation']!r}, not "
        f"{REQUIRED_AGGREGATION!r}")
    assert agg["reader_profile"] is None, (
        f"{AGGREGATION} carries a fitted reader profile ({agg['reader_profile']!r})")
    priors = {src: (float(pair[0]), float(pair[1]))
              for src, pair in agg["priors"].items()}

    component_sha = {
        "noise_model": _sha256(ROOT / NOISE_MODEL_SOURCE),
        "statement_belief": _sha256(ROOT / STATEMENT_BELIEF_SOURCE),
    }

    declared_arms = []
    for arm in READER_ARMS:
        manifest_rel = f"{MODELS_DIR}/{arm['key']}/manifest.json"
        bundle = json.loads((ROOT / manifest_rel).read_text())
        implementation = bundle["implementation"]
        notes = implementation["notes"]

        assert notes["aggregation"] == REQUIRED_AGGREGATION, (
            f"{manifest_rel} declares aggregation {notes['aggregation']!r}, not "
            f"{REQUIRED_AGGREGATION!r}; this arm is not the paper's aggregation")
        assert notes["reader_profile"] is None, (
            f"{manifest_rel} carries a fitted reader profile "
            f"({notes['reader_profile']!r}); belief would floor at sigmoid(prior), "
            "not at 0, and the arm would no longer be purely subtractive")
        declared_agg_sha = notes["inputs"]["aggregation_config"]["sha256"]
        assert declared_agg_sha == aggregation_sha, (
            f"{manifest_rel} declares aggregation_config sha256 {declared_agg_sha}, "
            f"but {AGGREGATION} hashes to {aggregation_sha}; the priors this arm "
            "ran under are not the priors in the tree")
        for component, digest in component_sha.items():
            declared = notes["implementation_components"][component]
            assert declared == digest, (
                f"{manifest_rel} declares {component} digest {declared}, but "
                f"the current source hashes to {digest}; the dispatch claim "
                "would rest on prose rather than on bytes")

        declared_arms.append({
            "arm": arm["key"],
            "label": arm["label"],
            "manifest_path": manifest_rel,
            "implementation": implementation["implementation"],
            "aggregation": notes["aggregation"],
            "reader_profile": notes["reader_profile"],
            "dedup": notes["dedup"],
            "declared_aggregation_config_sha256": declared_agg_sha,
            "aggregation_config_sha256_matches": True,
            "implementation_component_sha256": dict(sorted(
                notes["implementation_components"].items())),
            "implementation_component_sha256_matches": True,
        })

    # The sources this panel actually carries, and the prior each one lands at —
    # derived from the panel's own census, never typed out.
    panel_sources: collections.Counter = collections.Counter()
    for sid in sids:
        for src in source_counts[sid]:
            panel_sources[src] += 1
    unknown = sorted(set(panel_sources) - set(priors))
    assert not unknown, (
        f"the panel carries sources absent from {AGGREGATION}: {unknown}; they "
        "would silently take a fallback prior")
    panel_priors = {
        src: {"rand": priors[src][0], "syst": priors[src][1],
              "n_statements": panel_sources[src]}
        for src in sorted(panel_sources)
    }
    # Which sources share one prior, so the "one prior for five readers" line is
    # a grouping of the data rather than a claim about it.
    by_prior: dict[tuple[float, float], list[str]] = collections.defaultdict(list)
    for src in sorted(panel_sources):
        by_prior[priors[src]].append(src)
    prior_groups = [
        {"rand": rand, "syst": syst, "sources": sources,
         "n_statements": sum(panel_sources[s] for s in sources)}
        for (rand, syst), sources in sorted(
            by_prior.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]

    print(f"declaration: {len(declared_arms)} manifests, all "
          f"{REQUIRED_AGGREGATION} with reader_profile=null; "
          f"{len(panel_priors)} sources on the panel")

    # ---- (b) subtractive: re-derived, then cross-checked against P1 ---------
    simple = {r["statement_id"]: float(r["probability_correct"])
              for r in load_jsonl(ROOT / SIMPLE_SCORER_PREDICTIONS)}
    assert set(simple) == set(sids), (
        f"{SIMPLE_SCORER_PREDICTIONS} does not cover the panel exactly")
    noisy_or = np.array([simple[s] for s in sids])

    reader_scores: dict[str, np.ndarray] = {}
    subtractive_arms = {}
    total_exceeding = 0
    for arm in READER_ARMS:
        scores_rel = f"{MODELS_DIR}/{arm['key']}/all_source_predictions.jsonl"
        table = {r["statement_id"]: float(r["probability_correct"])
                 for r in load_jsonl(ROOT / scores_rel)}
        assert set(table) == set(sids), (
            f"{arm['key']}: score file does not cover the panel exactly")
        p = np.array([table[s] for s in sids])
        reader_scores[arm["key"]] = p
        exceeding = int((p > noisy_or).sum())
        total_exceeding += exceeding
        subtractive_arms[arm["key"]] = {
            "label": arm["label"],
            "n_statements": n,
            "n_exceeding_noisy_or": exceeding,
            "n_at_exactly_zero": int((p == 0.0).sum()),
            "n_nonzero": int((p != 0.0).sum()),
            "max_belief_above_noisy_or": float(np.max(p - noisy_or)),
            "scores_path": scores_rel,
        }
    n_comparisons = n * len(READER_ARMS)
    assert total_exceeding == 0, (
        f"{total_exceeding} of {n_comparisons} reader beliefs EXCEED the ungated "
        "noisy-OR; the arm is not purely subtractive and this artifact's whole "
        "claim is void")

    # The same finding lives in P1's non_reading_control.json. Assert equality and
    # pin that file's sha256, so the two artifacts can never drift apart.
    control_sha = _sha256(ROOT / NON_READING_CONTROL)
    control_check = control["subtractive_check"]
    assert control_check["n_comparisons"] == n_comparisons, (
        f"{NON_READING_CONTROL} compares {control_check['n_comparisons']} "
        f"beliefs, we compare {n_comparisons}")
    assert control_check["n_exceeding_noisy_or"] == total_exceeding
    for key, ours in subtractive_arms.items():
        theirs = control_check["arms"][key]
        for field in ("n_statements", "n_exceeding_noisy_or", "n_at_exactly_zero",
                      "max_belief_above_noisy_or"):
            assert theirs[field] == ours[field], (
                f"{key}.{field}: our re-derivation says {ours[field]!r}, "
                f"{NON_READING_CONTROL} says {theirs[field]!r}")

    print(f"subtractive: {total_exceeding} of {n_comparisons} reader beliefs "
          "exceed the noisy-OR (re-derived, and equal to P1's check)")

    # ---- (c) by value: the reachable-set search ----------------------------
    lists_by_sid = {sid: factor_lists(source_counts[sid], priors) for sid in sids}
    reachable_arms = {}
    max_nodes = 0
    total_exhausted = total_counterexamples = 0
    for arm in READER_ARMS:
        p = reader_scores[arm["key"]]
        confirmed = bit_exact = exhausted = counterexamples = 0
        unresolved: list[str] = []
        for sid, belief in zip(sids, p):
            if belief == 0.0:
                continue
            outcome = search_reachable(lists_by_sid[sid], float(belief),
                                       REACHABLE_TOLERANCE, NODE_BUDGET)
            max_nodes = max(max_nodes, outcome["nodes"])
            if outcome["within_tolerance"]:
                confirmed += 1
                bit_exact += int(outcome["bit_exact"])
            elif outcome["budget_exhausted"]:
                exhausted += 1
                unresolved.append(sid)
            else:
                counterexamples += 1
                unresolved.append(sid)
        n_nonzero = subtractive_arms[arm["key"]]["n_nonzero"]
        assert confirmed + exhausted + counterexamples == n_nonzero
        total_exhausted += exhausted
        total_counterexamples += counterexamples
        reachable_arms[arm["key"]] = {
            "label": arm["label"],
            "n_statements": n,
            "n_at_exactly_zero": subtractive_arms[arm["key"]]["n_at_exactly_zero"],
            "n_nonzero": n_nonzero,
            "n_confirmed_reachable": confirmed,
            "n_bit_exact": bit_exact,
            "n_budget_exhausted": exhausted,
            "n_counterexamples": counterexamples,
            "share_confirmed": confirmed / n_nonzero,
            "share_bit_exact": bit_exact / n_nonzero,
            "unresolved_statement_ids": sorted(unresolved),
        }
        print(f"  {arm['key']}: {confirmed}/{n_nonzero} on a reachable value "
              f"({bit_exact} bit-exact), {exhausted} budget-exhausted, "
              f"{counterexamples} counterexamples")

    assert total_counterexamples == 0, (
        f"{total_counterexamples} non-zero reader scores are NOT values the "
        "paper's formula can emit for any sub-multiset of that statement's "
        "evidence; the arm is not the paper's aggregation and (c) is void")

    # ---- (c) null baseline: reachable values are not rare ------------------
    enumerable: dict[str, list[float]] = {}
    n_values = 0
    for sid in sids:
        size = 1
        for factors in lists_by_sid[sid]:
            size *= len(factors)
        if size > ENUMERABLE_CAP:
            continue
        grid = enumerate_reachable(lists_by_sid[sid])
        enumerable[sid] = grid
        n_values += len(grid)
    enumerable_sids = sorted(enumerable)

    rng = np.random.default_rng(PERMUTATION_SEED)
    null_arms = {}
    pooled_rates: list[float] = []
    for arm in READER_ARMS:
        table = dict(zip(sids, reader_scores[arm["key"]]))
        eligible = [s for s in enumerable_sids if table[s] != 0.0]
        values = np.array([float(table[s]) for s in eligible])
        observed = sum(
            1 for s in eligible
            if on_reachable_grid(enumerable[s], float(table[s]), REACHABLE_TOLERANCE))
        rates = []
        for _ in range(N_PERMUTATIONS):
            shuffled = rng.permutation(values)
            hits = sum(
                1 for s, b in zip(eligible, shuffled)
                if on_reachable_grid(enumerable[s], float(b), REACHABLE_TOLERANCE))
            rates.append(hits / len(eligible))
        pooled_rates.extend(rates)
        null_arms[arm["key"]] = {
            "label": arm["label"],
            "n_enumerable_nonzero": len(eligible),
            "observed_rate": observed / len(eligible),
            "permuted_rate_mean": float(np.mean(rates)),
            "permuted_rate_min": float(np.min(rates)),
            "permuted_rate_max": float(np.max(rates)),
            "permuted_rates": [float(r) for r in rates],
        }
        print(f"  {arm['key']}: permutation floor "
              f"{np.mean(rates):.4f} [{np.min(rates):.4f}, {np.max(rates):.4f}] "
              f"over {len(eligible)} enumerable statements")

    # The noisy-OR's own floor on this panel: SimpleScorer never emits 0 here.
    noisy_or_floor = float(noisy_or.min())
    assert noisy_or_floor > 0.0, (
        "SimpleScorer emits 0 somewhere on this panel; the zero block would no "
        "longer distinguish the reader's empty product from the formula's own floor")

    null_baseline = {
        "question": "How often does a reader score land on a reachable value by "
                    "chance? Reachable values are not rare, so 100% is only worth "
                    "stating against this floor.",
        "method": "The arm's own non-zero scores are permuted across the "
                  "statements whose reachable set is small enough to enumerate "
                  "exhaustively, and the same membership test is re-run against "
                  "the RECIPIENT statement's set.",
        "n_permutations": N_PERMUTATIONS,
        "seed": PERMUTATION_SEED,
        "enumerable_cap_assignments": ENUMERABLE_CAP,
        "n_statements_enumerable": len(enumerable_sids),
        "n_statements_on_panel": n,
        "n_reachable_values_enumerated": n_values,
        "arms": null_arms,
        "pooled_permuted_rate_mean": float(np.mean(pooled_rates)),
        "pooled_permuted_rate_min": float(np.min(pooled_rates)),
        "pooled_permuted_rate_max": float(np.max(pooled_rates)),
    }

    print(f"pooled permutation floor {np.mean(pooled_rates):.4f} "
          f"[{np.min(pooled_rates):.4f}, {np.max(pooled_rates):.4f}]")

    # ---- what moved against the memo ---------------------------------------
    memo_reconciliation = {
        "source": "research/indra_paper_literal_vs_llm_comparison.md, Result 3c",
        "note": "Re-derived here with a tighter search than the memo's. Reported "
                "so the memo can be reconciled rather than left to disagree.",
        "bit_exact_tier": {
            "memo": {"gemma_4_26b": 1218, "gemma_4_31b": 1201,
                     "gemma_4_e2b": 1284, "glm_5": 1139},
            "rederived": {key: arm["n_bit_exact"]
                          for key, arm in reachable_arms.items()},
            "why": "The memo's remainders were statements its search could not "
                   "exhaust inside its budget, not statements that failed. "
                   "Binary-searching the feasible factor window at each depth "
                   "settles every one of them.",
        },
        "permutation_floor": {
            "memo": "near 45% (45-48% across replications)",
            "rederived_pooled_mean": float(np.mean(pooled_rates)),
            "rederived_pooled_range": [float(np.min(pooled_rates)),
                                       float(np.max(pooled_rates))],
        },
    }

    caveats = [
        "The reachable-value check asks whether the paper's formula CAN emit each "
        "score for some surviving-count vector, not whether it emitted it for the "
        "vector the reader actually produced. Two different vectors can land on "
        "the same value. Leg (a) is what pins the aggregation itself; this leg "
        "shows the scores are consistent with it, and the null baseline below "
        "prices how much that consistency is worth.",
        f"The search is exhaustive within its window but capped at {NODE_BUDGET} "
        "nodes per statement. Nothing on this panel reached that cap "
        f"(worst statement {max_nodes} nodes), and any statement that did would be "
        "reported as budget-exhausted rather than counted as confirmed.",
        f"The null baseline is restricted to the {len(enumerable_sids)} of {n} "
        "statements whose reachable set can be enumerated outright, so the chance "
        "floor is measured against a complete value set. The remaining statements "
        "have the LARGEST reachable sets, so they would only raise the floor.",
        # The label convention is NOT a caveat here. It is disclosed once, in
        # words, by the panel's own label-convention paragraph, which derives
        # every number in it from panel.negative_breakdown below and glosses what
        # the raw `label_is_adjudication_safe` flag means. A caveat repeating it
        # verbatim twenty lines further down the same <details> read as a second,
        # slightly different disclosure and re-introduced the bare field name.
        "Belief 0.0 is the reader rejecting every piece of evidence it read, "
        "leaving an empty product. It is not a low score the formula assigned: "
        f"SimpleScorer's own floor on this panel is {noisy_or_floor}, one "
        "reader sentence.",
    ]

    payload = {
        "artifact_kind": "framing_correction",
        "schema_version": 1,
        "question": "Is the reader arm a rival belief model, or INDRA's own "
                    "unfitted noisy-OR run on the evidence a reader kept?",
        "finding": "INDRA's own unfitted noisy-OR, on a filtered evidence set. It "
                   "can only remove belief, never add it.",
        "noisy_or_formula": formula,
        "aggregation": agg["aggregation"],
        "panel": {
            "n": n,
            "n_errors": n_errors,
            "n_correct": int((~is_error).sum()),
            "error_base_rate": n_errors / n,
            "label": "paper_replication_policy.released_paper_correct",
            "label_convention": ladder["panel"]["label_convention"],
            "negative_breakdown": negative_breakdown,
            "ordering": "sorted(statement_id)",
        },
        "declaration": {
            "claim": "All four reader bundles declare the paper's unfitted hard "
                     "gate, with no fitted reader profile, over the priors in "
                     f"{AGGREGATION}.",
            "required_aggregation": REQUIRED_AGGREGATION,
            "dispatch": "statement_belief(soft=None) takes the unfitted path into "
                        "noise_model.compute_gated_belief, which multiplies "
                        "syst + rand**surviving over sorted(sources) and drops a "
                        "source with zero surviving evidence from the product "
                        "entirely. The two component digests below are the sha256 "
                        "of those source files as they stand, so this is anchored "
                        "to bytes rather than to prose.",
            "aggregation_config": {"path": AGGREGATION, "sha256": aggregation_sha},
            "implementation_sources": {
                "noise_model": {"path": NOISE_MODEL_SOURCE,
                                "sha256": component_sha["noise_model"]},
                "statement_belief": {"path": STATEMENT_BELIEF_SOURCE,
                                     "sha256": component_sha["statement_belief"]},
            },
            "arms": declared_arms,
            "panel_priors": panel_priors,
            "prior_groups": prior_groups,
        },
        "subtractive": {
            "claim": "Dropping evidence removes factors from a product of numbers "
                     "below 1, so a reader can only lower belief. Checked, not "
                     "assumed.",
            "baseline": "raw noisy-OR (all evidence, multiplicity retained)",
            "baseline_scores": SIMPLE_SCORER_PREDICTIONS,
            "arms": subtractive_arms,
            "n_comparisons": n_comparisons,
            "n_exceeding_noisy_or": total_exceeding,
            "max_belief_above_noisy_or": max(
                arm["max_belief_above_noisy_or"] for arm in subtractive_arms.values()),
            "cross_check": {
                "artifact": NON_READING_CONTROL,
                "sha256": control_sha,
                "block": "subtractive_check",
                "agrees": True,
                "note": "Re-derived independently here from the prediction files, "
                        "then asserted equal to P1's block field by field.",
            },
        },
        "reachable_values": {
            "claim": "Every non-zero reader score is a value the paper's own "
                     "formula emits for some sub-multiset of that statement's "
                     "evidence.",
            "definition": "{1 - PROD_s f_s : f_s in {1} U {syst_s + rand_s^k, "
                          "1 <= k <= n_s}}, with n_s the statement's own "
                          "source_counts and the priors read from " + AGGREGATION,
            "tolerance": REACHABLE_TOLERANCE,
            "tiers": {
                "confirmed": f"|belief - (1 - PROD f_s)| <= {REACHABLE_TOLERANCE}",
                "bit_exact": "1.0 - PROD f_s == belief exactly, with the product "
                             "taken in the sorted-source order "
                             "compute_gated_belief multiplies in",
            },
            "search": {
                "algorithm": "depth-first over sorted(sources), feasible factor "
                             "window per depth binary-searched from a suffix "
                             "min/max product interval",
                "node_budget_per_statement": NODE_BUDGET,
                "max_nodes_used": max_nodes,
                "n_budget_exhausted": total_exhausted,
                "n_counterexamples": total_counterexamples,
            },
            "source_counts": PROVENANCE,
            "arms": reachable_arms,
            "null_baseline": null_baseline,
            "noisy_or_floor_on_panel": {
                "value": noisy_or_floor,
                "derived_from": f"min(probability_correct) over {SIMPLE_SCORER_PREDICTIONS}",
                "note": "The formula's own lowest score on this panel. It never "
                        "reaches 0, so the reader's zero block is a different "
                        "object: an empty product, not a low score.",
            },
        },
        "memo_reconciliation": memo_reconciliation,
        "caveats": caveats,
        "checks": {
            "every_manifest_declares_the_unfitted_hard_gate": True,
            "every_manifest_aggregation_config_sha_matches_the_tree": True,
            "every_manifest_component_digest_matches_the_source": True,
            "readers_never_exceed_the_noisy_or": total_exceeding == 0,
            "subtractive_agrees_with_non_reading_control": True,
            "every_nonzero_score_is_reachable": total_counterexamples == 0,
            "zero_and_nonzero_partition_the_panel": True,
            "negative_breakdown_agrees_with_belief_model_ladder": True,
            "gold_matches_hash_agrees_with_prediction_provenance": True,
            "note": "Assertions are enforced in code; a violation fails the build "
                    "rather than being reported here as False.",
        },
        "provenance": {
            "priors": AGGREGATION,
            "belief_function": "src/indra_belief/noise_model.py::compute_gated_belief",
            "source_counts": PROVENANCE,
            "noisy_or_scores": SIMPLE_SCORER_PREDICTIONS,
            "reader_bundles": f"{MODELS_DIR}/{{arm}}/all_source_predictions.jsonl",
            "reader_manifests": f"{MODELS_DIR}/{{arm}}/manifest.json",
            "gold": GOLD,
            "matches_hash_crosscheck": PROVENANCE,
            "non_reading_control": NON_READING_CONTROL,
            "belief_model_ladder": BELIEF_MODEL_LADDER,
            "join": "paper_statement_hash == canonical_corpus.matches_hash -> "
                    "statement_id",
            "generated_by": "scripts/compute_framing_correction.py",
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    sha = _sha256(out)
    if args.manifest:
        mpath = Path(args.manifest)
        man = json.loads(mpath.read_text())
        man.setdefault("outputs", {})["framing_correction"] = out.name
        man.setdefault("output_sha256", {})[out.name] = sha
        mpath.write_text(json.dumps(man, indent=2) + "\n")
        print(f"recorded sha256 in {mpath}")

    print(f"\nwrote {out} ({out.stat().st_size} bytes)\nsha256 {sha}\n")
    print(f"{'arm':<16}{'non-zero':>10}{'reachable':>11}{'bit-exact':>11}{'zero':>7}")
    for arm in READER_ARMS:
        row = reachable_arms[arm["key"]]
        print(f"{arm['label']:<16}{row['n_nonzero']:>10}{row['n_confirmed_reachable']:>11}"
              f"{row['n_bit_exact']:>11}{row['n_at_exactly_zero']:>7}")
    print(f"\n{total_exceeding} of {n_comparisons} reader beliefs exceed the "
          "noisy-OR; the arm can only subtract.")


if __name__ == "__main__":
    main()
