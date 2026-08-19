"""The reader gate against INDRA's own belief, on four independently-sourced panels.

WHAT CHANGED IN SCHEMA 2, and why it had to.
--------------------------------------------
Schema 1 drew four rows and let them read as "the same comparison, four times".
They were not.  The comparator differed per row, and — worse — two of the rows
were measured against a number this file DESCRIBED as "SimpleScorer at default
priors, with the hierarchy propagation INDRA applies" and which demonstrably is
no such thing.  The description was wrong; the data was fine.  Schema 2 fixes
the description, names the comparator ON every row, and computes the panel
heterogeneity instead of asserting it.

THE PROOF THAT THE OLD DESCRIPTION WAS WRONG (shipped as `served_belief_identity`,
recomputed on every build so it can never rot):

  INDRA's SimpleScorer at the priors bundled in
  ``indra/resources/default_belief_probs.json`` computes
  ``belief = 1 - PROD_s (syst_s + rand_s^{n_s})``.  Every factor in that product
  is at most ``syst_s + rand_s``, so the score has a HARD FLOOR: no statement,
  with any evidence whatsoever, can score below
  ``1 - max_s (syst_s + rand_s)`` = 0.60 (the worst source in the shipped file is
  `gnbr`).  Hierarchy propagation only ever RAISES a belief, so it cannot go
  below the floor either.

  48.2% of `external_curator_v1`'s stored beliefs are BELOW that floor.  So are
  11.4% of `holdout_cc`'s and 9.4% of `eval_curation_v1`'s.  A second, sharper
  check: on `eval_curation_v1` the 29 statements whose INDRA source counts are
  exactly ``{'reach': 1}`` carry SIX distinct stored beliefs, and none of them is
  the 0.65 that SimpleScorer at default priors is forced to produce.

  The stored belief is therefore the number INDRA's own export/assembly pipeline
  WROTE on the statement — the fitted ``HybridScorer`` path the CoGEx export uses
  (see ``scripts/score_current_indra_counts_hybrid_paper.py`` and
  ``data/comparison/models/indra_cogex_hybrid/``), not the library default.

THAT SPLITS THE INCUMBENT INTO TWO HONEST FAMILIES, both of which are INDRA's:

  `indra_library_default`     ``indra.belief.SimpleScorer`` at the bundled
                              default priors — what anyone who pip-installs
                              `indra` and calls ``BeliefEngine`` gets.  Unfitted.
  `indra_production_served`   the belief INDRA's own pipeline computed and stored
                              — what a db.indra.bio / CoGEx user reads off the
                              statement.  Fitted, and its training provenance is
                              not ours to establish.

AND IT FORCES A CONSEQUENCE we did not take before.  The strongest-incumbent rule
says the headline comparator is the STRONGEST sourceable form of INDRA's own
belief on each panel.  The production-served family IS sourceable on the paper's
own panel — ``data/comparison/models/indra_cogex_hybrid/`` is a replay of that
very artifact on those very 1689 statements, and it is already carried in the
shipped ``paper_literal_vs_llms.json`` point metrics.  It scores 0.8272, above
SimpleScorer+hierarchy's 0.7823.  Leaving it out would have been exactly the
"convenient baseline" the rule exists to forbid, so it is now drawn, and the
paper panel's headline delta is +0.0738 rather than +0.1187.  Both numbers ship:
every variant's own delta and paired interval are in the artifact, so the larger
figure against the library default is one field away and is never hidden.

THE ARM CONTRAST IS STILL NARROW WHERE IT MATTERS.  The `gate` arm is INDRA's own
library-default noisy-OR over only the evidence a reader kept.  It fits nothing,
tunes no threshold, and is purely SUBTRACTIVE (``research/`` calls this the hard
gate): it can only ever LOWER a statement's belief.  Whatever separation appears
is separation that reading bought.

METRIC.  AUROC (positive class = gold-correct), because it is the one metric
comparable ACROSS panels whose base rates run 0.45 to 0.73.  Average precision is
carried per panel and is explicitly NOT cross-panel comparable.  Every delta gets
a paired statement-level bootstrap sharing one resample between the two arms, so
the interval is on the difference.  EVERY variant gets one, not just the argmax,
so "the gate beats every sourceable form of INDRA's belief on every panel" is a
claim the bytes can carry.

THE HETEROGENEITY IS COMPUTED, NOT ASSERTED (``panels[].heterogeneity``).  These
are not identical comparisons and the figure now says so on its face:

  * evidence reads per statement — 19.75 on the paper's panel, 1.21 to 1.76 on
    the curation panels, a ~16-fold span that the artifact COUNTS rather than
    asserts (``replication.evidence_regime_fold_span``).  A mean cannot carry
    that difference on its own, so every census also ships ``share_single``: the
    fraction of the panel's statements holding exactly ONE evidence.  Where that
    share is high the gate has no aggregation left to do and its decision is a
    bare keep-or-drop on one sentence — a materially different task from the
    paper panel's 19.75 reads, and the figure prints both.  On the curation
    panels INDRA's incumbent is additionally scored over the statement's FULL
    database evidence while the gate sees only the curated subset, so the
    comparison there is conservative;
  * the evidence-matched control prices exactly that asymmetry: INDRA's own
    library-default noisy-OR restricted to the reader's evidence set.  It lands
    at or below chance on all three curation panels, which is why the incumbent's
    signal there comes from evidence the gate never saw.  It is NOT an incumbent
    (INDRA serves no such number) so it never enters the argmax;
  * curator count, class balance, in/out-of-sample, and the join mode each panel
    actually used — `holdout_cc` joins 100% by the source-hash fallback because
    its gold carries no matches_hash, and that travels with the row.

PANELS.  Four, sourced independently:
  * `indra_paper_2023`      the paper's own 1689 assembled statements with the
                            paper's own released labels — the largest panel, and
                            the only one whose labels we did not produce.
  * `eval_curation_v1`      913 statements, curated by two of the paper's own
                            authors.  This is the corpus our SOFT calibration was
                            fitted on; the arm drawn here has no fitted parameter,
                            so it is still an honest read, but the flag travels.
  * `external_curator_v1`   464 statements, 32 curators, none of them ours.
  * `holdout_cc`            414 statements, the out-of-distribution holdout the
                            calibration ship gate tests against.

Every join is IMPORTED rather than re-derived: the paper panel comes from
``compute_statement_review_queue.load_panel`` and the curation panels from
``calibration_ship_gate.statements_for_run``, so this artifact cannot drift from
``statement_review_queue.json`` or ``calibration_ship_gate.json``.

THE HEADLINE SENTENCE IS COUNTED, NOT COMPOSED (``claim``, ``claim_is_not``, and
two of the ``caveats``).  "Four panels, the gate wins" is a sentence this figure
cannot carry, because the comparator differs by row and so does the evidence
regime.  The sentence it CAN carry is stronger and every quantity in it is a
count taken from the rows: the panel count, the number of distinct forms of
INDRA belief scored, how many of their paired intervals exclude zero, the
fold-span of the evidence density, and the AUROC the argmax forfeits.  Two
caveats that used to carry hand-typed numbers ("1.2 to 1.8 evidence per
statement", "0.50 balanced on two of them") are filled from the panels for the
same reason: a hand-typed number in a caveat is the number that rots first.

Hard assertions (the script fails loudly rather than emitting a plausible file):
  (a) the paper panel's SimpleScorer scores agree BIT-EXACTLY with the second
      copy of the same run, so the library-default incumbent is provably the arm
      the ladder calls "noisy-OR SimpleScorer (direct)";
  (b) every panel's gate score is <= the matched no-read score on every statement
      where both are defined — the gate is subtractive, and if it ever exceeded
      the noisy-OR it would not be the same aggregation;
  (c) each panel's headline incumbent is the argmax over its own variants;
  (d) every delta equals gate AUROC minus the compared AUROC exactly;
  (e) the re-derived AUROCs for the paper panel agree with the shipped
      ``paper_literal_vs_llms.json`` values to ``TOL_RECORDED`` — for the gate,
      for the paper's RF, AND for the CoGEx hybrid;
  (f) the served-belief identity check actually finds served beliefs below the
      SimpleScorer floor, i.e. the family split is earned rather than asserted;
  (g) every caveat slot declared in ``CAVEATS`` is filled from the built panels,
      so a renamed slot fails the build rather than printing its own placeholder
      into an author-facing figure.

Note on the curation panels' censuses: a statement whose belief is undefined is
EXCLUDED rather than imputed, and every census counts the surviving statements.
`external_curator_v1` joins 587 evidence rows over 469 statements; 5 of those
statements have no defined belief, so the panel is 464 statements over 582
reads and `n_undefined_excluded` carries the difference. Quoting 587 reads
beside 464 statements would mix the two.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compute_deployed_baseline_replication.py \
      --out-json data/results/deployed_baseline_replication_20260727/deployed_baseline_replication.json \
      --manifest data/results/deployed_baseline_replication_20260727/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from indra_belief.noise_model import (  # noqa: E402
    RECALIBRATED_PRIORS,
    compute_edge_reliability_from_counts,
)
from indra_belief.indra_priors import (  # noqa: E402
    INDRA_DEFAULT_PRIOR_RESOURCE,
    INDRA_DEFAULT_PRIORS,
    INDRA_DEFAULT_PRIORS_SHA256,
    with_benchmark_recalibration,
)
from indra_belief.statement_belief import statement_belief  # noqa: E402

import calibration_ship_gate as sg  # noqa: E402
import calibration_stage0 as c0  # noqa: E402
from indra_belief.results import load_gold_map  # noqa: E402

# Reuse the paper head-to-head's join contract verbatim.
from compare_paper_literal_vs_llms import (  # noqa: E402
    GOLD as PAPER_GOLD,
    HEADLINE as PAPER_RF_KEY,
    MODELS_DIR,
    load_jsonl,
)

# Reuse the review queue's panel loader verbatim: same 1689 statements, same
# sorted(statement_id) ordering.
from compute_statement_review_queue import load_panel  # noqa: E402

ARTIFACT_KIND = "deployed_baseline_replication"
# 3: every per-statement census also carries `n_statements`, `n_single` and
# `share_single`; `replication` carries the evidence-regime fold span; and the
# headline claim, the anti-claim and two caveats are COUNTED from the panels
# instead of typed. Consumers require those fields, so this is a new contract.
SCHEMA_VERSION = 3

N_BOOT = 10000
# The SAME seed the paper head-to-head uses, on purpose: both artifacts draw
# paired statement-level bootstraps on the paper's panel, and a reviewer who
# checks one against the other should not have to reason about two RNG streams.
SEED = 20260717
TOL_RECORDED = 1e-12
TOL_IDENTITY = 1e-12

METRIC = "auroc"
METRIC_SOURCE = "rank-sum AUROC with mid-ranks for ties (verified against sklearn.metrics.roc_auc_score)"
POSITIVE_CLASS = "gold-correct"
NOISY_OR_FORMULA = "belief = 1 - PROD_s (syst_s + rand_s^{n_s})"

# INDRA's own installed resource is loaded once by the shared package module.
# Recalibration overrides only measured sources; everything else retains the
# corresponding installed default rather than the generic unknown-source prior.
RECALIBRATED_WITH_INDRA_DEFAULTS = with_benchmark_recalibration(RECALIBRATED_PRIORS)

# The hard floor of SimpleScorer at those priors. Each source contributes a
# factor of at least syst_s + rand_s^1 to P(incorrect), so belief can never
# exceed 1 - min_s(syst_s + rand_s) ... and can never fall BELOW
# 1 - max_s(syst_s + rand_s), which is the number that matters here: it is the
# score of a statement with exactly one evidence from the least reliable source.
_WORST_SOURCE = max(
    INDRA_DEFAULT_PRIORS.complete_priors.items(),
    key=lambda kv: kv[1][0] + kv[1][1],
)
SIMPLE_SCORER_FLOOR = 1.0 - (_WORST_SOURCE[1][0] + _WORST_SOURCE[1][1])
SIMPLE_SCORER_FLOOR_SOURCE = _WORST_SOURCE[0]

# ---------------------------------------------------------------------------
# Arm and family identities. `key` is the frozen join key; `display` is what the
# figure renders. They are deliberately decoupled — renaming the on-screen text
# must never move a join.
# ---------------------------------------------------------------------------
FAMILY_LIBRARY = "indra_library_default"
FAMILY_SERVED = "indra_production_served"

INCUMBENT_FAMILIES = [
    {
        "key": FAMILY_LIBRARY,
        "display": "INDRA's library default",
        "deployed": True,
        "fitted": False,
        "what_it_computes": (
            "indra.belief.SimpleScorer at the per-source priors bundled in "
            "indra/resources/default_belief_probs.json: "
            f"{NOISY_OR_FORMULA}. Nothing about it is fitted."
        ),
        "where_it_runs": (
            "what anyone who installs indra and calls BeliefEngine gets, and what "
            "the 2023 paper calls the unfitted noisy-OR"
        ),
        "ships_in": "indra.belief.SimpleScorer + indra/resources/default_belief_probs.json",
        "resource_sha256": INDRA_DEFAULT_PRIORS_SHA256,
    },
    {
        "key": FAMILY_SERVED,
        "display": "the belief INDRA stored",
        "deployed": True,
        "fitted": True,
        "what_it_computes": (
            "the number INDRA's own export/assembly pipeline computed and wrote on "
            "the statement — the fitted HybridScorer path the CoGEx export uses, "
            "NOT the library default. We read it; we do not re-derive it."
        ),
        "where_it_runs": (
            "what a db.indra.bio or CoGEx user reads off the statement today"
        ),
        "ships_in": "indra_db readonly_dumping/export_assembly.py via indra.belief.skl.HybridScorer",
        "resource_sha256": None,
    },
]

GATE = {
    "key": "gemma_4_26b_gate",
    "display": "Gemma 4 26B reader gate",
    "deployed": False,
    "fitted": False,
    "what_it_is": (
        "INDRA's own library-default noisy-OR over only the evidence the reader "
        "kept; purely subtractive, no threshold fitted, nothing trained"
    ),
    "not_zero_shot": "14 hand-authored demonstration pairs per call; no calibration",
}
RESEARCH_MODEL = {
    "key": "paper_rf_promoter",
    "display": "paper RF + Type/#PMIDs/promoter",
    "deployed": False,
    "fitted": True,
    "what_it_is": (
        "the 2023 paper's own released supervised random forest, out-of-fold. A "
        "research model: it is not in indra and has never been served."
    ),
}

# The paper panel's library-default incumbent, and the second copy of the same
# run the ladder cross-checks it against. Assertion (a) requires them identical.
PAPER_SIMPLE = ("data/results/current_indra_simple_paper_20260717/"
                "current_indra_simple_default_predictions.jsonl")
PAPER_SIMPLE_MIRROR = ("data/results/current_indra_bayesian_paper_20260717/"
                       "current_simple_direct_all_sources_predictions.jsonl")
# INDRA also propagates belief up the statement hierarchy; that variant is part
# of the library-default family too, so it competes for the strongest slot.
PAPER_HIERARCHY = ("data/results/current_indra_hierarchy_paper_20260717/"
                   "current_simple_hierarchy_all_sources_predictions.jsonl")
# The production-served family, replayed on the paper's own 1689 statements.
PAPER_COGEX = "data/comparison/models/indra_cogex_hybrid/all_source_predictions.jsonl"
PAPER_COGEX_MANIFEST = "data/comparison/models/indra_cogex_hybrid/manifest.json"
PAPER_GATE = f"{MODELS_DIR}/gemma_4_26b/all_source_predictions.jsonl"
PAPER_GATE_MANIFEST = f"{MODELS_DIR}/gemma_4_26b/manifest.json"
PAPER_EXECUTION_MAP = "data/benchmark/indra_paper_unique_pairs_20260717_execution_map.jsonl"
PAPER_LITERAL = ("data/results/indra_paper_literal_models_20260724/"
                 "paper_literal_table6_and_oof.json")
PAPER_VS_LLMS = ("data/results/indra_paper_literal_models_20260724/"
                 "paper_literal_vs_llms.json")
PAPER_COGEX_RECORDED_KEY = "INDRA CoGEx hybrid"
# The reader arm's aggregation must be INDRA's default hard gate, or the two
# arms are not the same scorer and the whole contrast is void.
REQUIRED_GATE_AGGREGATION = "indra_default_hard_gate"

# ---------------------------------------------------------------------------
# Curation panels. Each names the gold, the scored reader run, and the reader
# configuration those two produce — exactly the (gold, run) pairs the shipped
# head-to-head and ship-gate artifacts already use.
# ---------------------------------------------------------------------------
CURATION_PANELS = [
    {
        "key": "eval_curation_v1",
        "display": "two of the paper’s authors",
        "gold": "data/benchmark/eval_curation_v1.jsonl",
        "run": "data/results/eval_curation_v1_gemma.jsonl",
        "model": "remote-gemma-4-26b",
        "n_curators": 2,
        "curator_note": "curated by two of the 2023 paper’s own authors",
        "out_of_sample": False,
        "in_sample_note": (
            "This corpus is what our SOFT calibration profile was fitted on. The "
            "arm drawn here uses INDRA's default priors and no fitted parameter, "
            "so it is still out-of-sample for what is plotted — but the flag "
            "travels with the panel."
        ),
        "balanced_by_construction": True,
    },
    {
        "key": "external_curator_v1",
        "display": "32 external curators",
        "gold": "data/benchmark/external_curator_gold_v1.jsonl",
        "run": "data/results/external_curator_v1_bedrock-gemma.jsonl",
        "model": "bedrock-gemma-4-26b",
        "n_curators": 32,
        "curator_note": "32 curators from the public INDRA curation database, none of them ours",
        "out_of_sample": True,
        "in_sample_note": None,
        "balanced_by_construction": True,
    },
    {
        "key": "holdout_cc",
        "display": "out-of-distribution holdout",
        "gold": "data/results/cc_holdout_cc/holdout_cc.jsonl",
        "run": "data/results/holdout_cc_gemma.jsonl",
        "model": "remote-gemma-4-26b",
        "n_curators": None,
        "curator_note": "the out-of-distribution holdout the calibration ship gate tests against",
        "out_of_sample": True,
        "in_sample_note": None,
        "balanced_by_construction": False,
    },
]

PAPER_PANEL_KEY = "indra_paper_2023"
PAPER_PANEL_DISPLAY = "the 2023 paper’s own panel"

# The sentence the figure can actually carry, and the one it cannot.
FIGURE_TITLE = "Against INDRA’s own belief"


def build_claim(panels: list[dict], replication: dict) -> str:
    """The headline sentence, COUNTED from the panels rather than composed.

    An earlier draft read "four panels, the same comparison, the reader gate
    wins", which is the sentence the figure cannot carry: the comparator differs
    by panel and so does the evidence regime. The sentence it CAN carry is
    stronger, and every quantity in it is a count taken from the rows above —
    the panel count, the number of distinct forms of INDRA belief scored, how
    many of their paired intervals exclude zero, the fold-span of the evidence
    density, and the AUROC the argmax rule forfeits. If any of those move, the
    sentence moves with them.
    """
    n_panels = len(panels)
    n_forms = replication["n_incumbent_variants_total"]
    n_ci = replication["n_incumbent_variants_ci_excludes_zero"]
    lo = replication["reads_per_statement_mean_min"]
    hi = replication["reads_per_statement_mean_max"]
    fold = replication["evidence_regime_fold_span"]
    cost = replication["selection_cost_auroc_max"]
    return (
        f"The reader gate beats every sourceable form of INDRA's own belief on "
        f"every one of {n_panels} independently-sourced panels — {n_forms} forms "
        f"in all, {n_ci} of {n_forms} with a paired 95% interval excluding zero — "
        f"across evidence regimes that differ {fold:.0f}-fold ({lo:.1f} to "
        f"{hi:.1f} evidence read per statement), under a rule that always draws "
        f"the STRONGEST form each panel can source and forfeits up to "
        f"{cost:.4f} AUROC of margin to do it."
    )
def build_claim_is_not(panels: list[dict]) -> str:
    """The anti-claim, COUNTED from the panels rather than asserted.

    An earlier draft of this sentence said one panel could source only the
    library default. None can. Writing it by hand is how a figure ends up
    describing a composition it does not have, so it is counted here instead.
    """
    both = only_library = only_served = 0
    for panel in panels:
        families = {v["family"] for v in panel["incumbent_variants"]}
        if families == {FAMILY_LIBRARY, FAMILY_SERVED}:
            both += 1
        elif families == {FAMILY_LIBRARY}:
            only_library += 1
        else:
            only_served += 1

    def plural(n: int) -> str:
        return "panel" if n == 1 else "panels"

    parts = []
    if both:
        parts.append(f"{both} {plural(both)} can source both forms of INDRA's own belief")
    if only_served:
        parts.append(f"{only_served} can source only the belief INDRA stored")
    if only_library:
        parts.append(f"{only_library} can source only the library default")
    # The sharpest single heterogeneity fact, counted rather than described: on
    # some panels most statements carry exactly one evidence, so the gate's
    # decision there is a bare keep/drop with no aggregation to perform.
    shares = sorted(
        (
            p["heterogeneity"]["evidence_reads_per_statement"]["share_single"],
            p["display"],
        )
        for p in panels
    )
    lo_share, lo_panel = shares[0]
    hi_share, hi_panel = shares[-1]
    return (
        f"This is not the same comparison {len(panels)} times. The comparator "
        f"differs by panel — {', '.join(parts)} — and so do the evidence per "
        "statement, the curators, the class balance and the join. Single-evidence "
        f"statements run from {lo_share:.0%} of {lo_panel} to {hi_share:.0%} of "
        f"{hi_panel}, so on some rows the gate aggregates and on others it is a "
        "bare keep-or-drop on one sentence. Every row names its own comparator "
        "and prints its own composition."
    )


# Two caveats carry numbers that are properties of the PANELS, so they cannot be
# module constants without going stale the first time a panel moves. They ship as
# named slots in `CAVEATS` and are filled from the built rows; `build_caveats`
# asserts every slot was consumed, so a renamed slot fails the build instead of
# printing its own placeholder into an author-facing figure.
BASE_RATE_CAVEAT_SLOT = "BASE_RATE_CAVEAT"
SCOPE_ASYMMETRY_CAVEAT_SLOT = "SCOPE_ASYMMETRY_CAVEAT"


def build_base_rate_caveat(panels: list[dict]) -> str:
    """Why AUROC and not average precision, with the panels' own base rates."""
    rates = ", ".join(
        f"{p['base_rate_correct']:.2f} on {p['display']}"
        + (" (balanced by construction)" if p["balanced_by_construction"] else "")
        for p in sorted(panels, key=lambda p: p["base_rate_correct"])
    )
    return (
        "AUROC is the cross-panel metric because the panels' base rates differ — "
        f"{rates}. Average precision moves with the base rate, so it is reported "
        "per panel and must not be compared across panels."
    )


def build_scope_asymmetry_caveat(panels: list[dict], replication: dict) -> str:
    """The scope-asymmetry caveat with its numbers COUNTED, not transcribed.

    An earlier draft typed "1.2 to 1.8 evidence per statement against the paper
    panel's 19.75" by hand. Hand-typed numbers in a caveat are the numbers that
    rot first, and this one carries the concession the whole comparison rests
    on, so it is assembled from the same census the rows print.
    """
    paper = next(p for p in panels if p["is_paper_panel"])
    others = [p for p in panels if not p["is_paper_panel"]]
    means = [p["heterogeneity"]["evidence_reads_per_statement"]["mean"] for p in others]
    singles = [
        p["heterogeneity"]["evidence_reads_per_statement"]["share_single"] for p in others
    ]
    paper_reads = paper["heterogeneity"]["evidence_reads_per_statement"]
    n_control_at_chance = sum(
        1
        for p in others
        if p["evidence_matched_control"] is not None
        and p["evidence_matched_control"]["at_or_below_chance"]
    )
    n_control = sum(1 for p in others if p["evidence_matched_control"] is not None)
    return (
        f"On the {len(others)} curation panels the incumbent is scored over the "
        "statement's FULL database evidence while the gate sees only the curated "
        f"subset — {min(means):.2f} to {max(means):.2f} evidence per statement, "
        f"and {min(singles):.0%} to {max(singles):.0%} of those statements carry "
        f"exactly one, against the paper panel's {paper_reads['mean']:.2f} and "
        f"{paper_reads['share_single']:.0%}. That is a "
        f"{replication['evidence_regime_fold_span']:.1f}-fold difference in "
        "evidence density between the extreme panels. INDRA's own scorer "
        "restricted to the reader's evidence set lands at or below chance on "
        f"{n_control_at_chance} of those {n_control} panels "
        "(`evidence_matched_control`), so that asymmetry runs against the gate, "
        "not for it."
    )


CAVEATS = [
    "There are TWO incumbents and both are INDRA's. The library default is the "
    "unfitted noisy-OR SimpleScorer at the priors indra bundles. The stored "
    "belief is what INDRA's export pipeline wrote on the statement, and it is "
    "NOT that scorer: it falls below SimpleScorer's hard floor on thousands of "
    "statements here, which is recomputed on every build as "
    "`served_belief_identity`.",
    "The CoGEx hybrid drawn on the paper's panel is a compatibility-recovered "
    "offline replay of the artifact the CoGEx export path references — not a "
    "live production capture, and its training provenance is not established. "
    "If it was fitted on curations overlapping this panel it is optimistic "
    "here, which makes it a HARDER incumbent, not an easier one.",
    "Our unfitted SimpleScorer is NOT the paper's \"Belief Orig\" row. Theirs "
    "refits the source reliabilities per fold by MCMC; ours is the shipped "
    "default-prior scorer, unfitted anywhere. The paper publishes no Bayesian, "
    "subtype or hierarchy arm at all.",
    "The reader arms are NOT zero-shot: each call carries 14 hand-authored "
    "demonstration pairs. They are also not calibrated — this is the hard gate, "
    "which can only subtract evidence.",
    "BASE_RATE_CAVEAT",
    "SCOPE_ASYMMETRY_CAVEAT",
    "Four panels is replication of a DIRECTION, not a meta-analysis: the panels "
    "differ in curator, corpus, sampling and reader deployment, and the "
    "intervals are per-panel, not pooled.",
    "The gate is a DETECTOR, not a ranker. Conditional on the statements it does "
    "not zero, it ranks worse than the paper's fitted RF. Its advantage over the "
    "incumbent comes from the statements it zeroes, which is what the "
    "operational review-queue result measures directly.",
]


def build_caveats(panels: list[dict], replication: dict) -> list[str]:
    """`CAVEATS` with its two computed slots filled. Fails if a slot survives."""
    filled = {
        BASE_RATE_CAVEAT_SLOT: build_base_rate_caveat(panels),
        SCOPE_ASYMMETRY_CAVEAT_SLOT: build_scope_asymmetry_caveat(panels, replication),
    }
    out = [filled.get(c, c) for c in CAVEATS]
    unconsumed = [slot for slot in filled if slot in out]
    assert not unconsumed, f"caveat slots left unfilled: {unconsumed}"
    consumed = sum(1 for c in CAVEATS if c in filled)
    assert consumed == len(filled), (
        f"expected {len(filled)} caveat slots in CAVEATS, found {consumed}"
    )
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-sum AUROC with mid-ranks for ties, positive class = 1.

    Written out rather than imported because the bootstrap calls it ~200k times
    and because `metrics.py::auprc` is order-dependent (2026-07-25 finding) —
    nothing in this file may reuse that estimator. Verified against
    ``sklearn.metrics.roc_auc_score`` in ``main``.
    """
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC is undefined when a class is absent")
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    i = 0
    while i < ordered.size:
        j = i
        while j + 1 < ordered.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = float(ranks[y[order] == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, scores))


def paired_bootstrap_delta(
    y: np.ndarray, high: np.ndarray, low: np.ndarray, seed: int, n_boot: int
) -> dict:
    """Percentile CI for AUROC(high) - AUROC(low) under a SHARED resample.

    Statements are resampled with replacement; both arms are scored on the very
    same resample, so this is an interval on the difference. Resamples that lose
    a class are skipped and counted, never silently imputed.
    """
    rng = np.random.default_rng(seed)
    n = y.size
    deltas = np.empty(n_boot, dtype=float)
    n_valid = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        deltas[n_valid] = auroc(yb, high[idx]) - auroc(yb, low[idx])
        n_valid += 1
    if n_valid < n_boot // 2:
        raise AssertionError(f"only {n_valid}/{n_boot} bootstrap resamples were valid")
    valid = deltas[:n_valid]
    return {
        "ci95_low": float(np.percentile(valid, 2.5)),
        "ci95_high": float(np.percentile(valid, 97.5)),
        "p_gate_greater": float(np.mean(valid > 0.0)),
        "n_valid_resamples": int(n_valid),
        "n_bootstrap": int(n_boot),
        "seed": int(seed),
    }


def spread(values: list[int]) -> dict:
    """The census of a per-statement count, so heterogeneity is computed.

    `n_single` / `share_single` are the census a MEAN cannot carry. On
    `external_curator_v1` the mean is 1.25 evidence per statement and the median
    is 1 — but the number that says what the gate was actually asked to do there
    is that 84% of its statements carry exactly ONE evidence, so the "gate" is a
    bare keep/drop on one sentence with no aggregation left to perform. On the
    paper's panel the same share is a small minority. That contrast is the
    figure's heterogeneity disclosure, so it is counted here rather than
    described.
    """
    n_single = sum(1 for v in values if v == 1)
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": int(max(values)),
        "min": int(min(values)),
        "total": int(sum(values)),
        "n_statements": int(len(values)),
        "n_single": int(n_single),
        "share_single": n_single / len(values),
    }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(ROOT / path).read_bytes()).hexdigest()


def score_vector(path: str, statement_ids: list[str]) -> np.ndarray:
    by_id = {r["statement_id"]: r["probability_correct"] for r in load_jsonl(path)}
    missing = [s for s in statement_ids if s not in by_id]
    if missing:
        raise AssertionError(f"{path} is missing {len(missing)} panel statements")
    return np.array([by_id[s] for s in statement_ids], dtype=float)


def source_counts_by_statement(run_path: str, gold_path: str) -> dict[str, dict]:
    """Statement key -> INDRA source_counts, where the gold carries them.

    Uses the SAME candidate lookup ``calibration_ship_gate.statements_for_run``
    uses (exact canonical pair first, source-hash otherwise) so a statement can
    never pick up counts from a row the authoritative join rejected. Statements
    whose rows disagree on their counts are dropped rather than guessed at:
    ``source_counts`` is a statement-level property, so a disagreement means the
    grouping is wrong and the variant must not be reported.
    """
    gold_rows = c0.load_jsonl(ROOT / gold_path)
    gold_map = load_gold_map(str(ROOT / gold_path))
    by_pair: dict[tuple[int, int], list[dict]] = {}
    by_source: dict[int, list[dict]] = {}
    for g in gold_rows:
        sh = sg._ukey(g.get("source_hash"))
        if sh is None:
            continue
        by_source.setdefault(sh, []).append(g)
        mh = sg._ukey(g.get("matches_hash"))
        if mh is not None:
            by_pair.setdefault((mh, sh), []).append(g)

    seen: dict[str, list[dict]] = {}
    run_rows_by_position: dict[tuple, dict] = {}
    for row in c0.load_jsonl(ROOT / run_path):
        run_rows_by_position[(row.get("stmt_i"), row.get("evidence_i"))] = row
    for scored in run_rows_by_position.values():
        stmt_key, mh = sg._run_statement_key(scored.get("stmt_hash"))
        sh = sg._ukey(scored.get("source_hash"))
        if gold_map.for_row(mh, sh) is None:
            continue
        candidates = by_pair.get((mh, sh)) or by_source.get(sh) or []
        if not candidates:
            continue
        counts = candidates[0].get("source_counts")
        if isinstance(counts, dict) and counts:
            seen.setdefault(stmt_key, []).append(counts)
    out: dict[str, dict] = {}
    for key, observed in seen.items():
        first = observed[0]
        if all(c == first for c in observed):
            out[key] = first
    return out


def paper_reads_per_statement() -> dict:
    """Unique (statement, evidence) pairs the reader was given, per statement.

    Streamed from the frozen execution map — the same 33,361 rows the reader
    bundle's own cost denominator counts — rather than asserted from the
    manifest, so the number in the figure is a census of the bytes.
    """
    counts: dict[str, int] = {}
    with Path(ROOT / PAPER_EXECUTION_MAP).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["paper_statement_hash"])
            counts[key] = counts.get(key, 0) + 1
    return spread(list(counts.values()))


def load_paper_panel() -> dict:
    """The paper's own 1689 statements, its own labels, its own released model."""
    # `compute_statement_review_queue.load_panel` is the authority on which
    # statements, in which order, with which labels. It returns one object so
    # every consumer reads the same panel; take the two fields we need and fail
    # loudly if either stops being there rather than re-deriving a join.
    panel = load_panel()
    assert isinstance(panel, dict) and {"sids", "y"} <= set(panel), (
        "compute_statement_review_queue.load_panel no longer returns a panel "
        f"object carrying sids and y (got {type(panel).__name__}); this artifact "
        "imports that join rather than re-deriving it and must not guess"
    )
    sids = list(panel["sids"])
    y = np.asarray(panel["y"], dtype=int)

    simple = score_vector(PAPER_SIMPLE, sids)
    mirror = score_vector(PAPER_SIMPLE_MIRROR, sids)
    # (a) the library-default incumbent is provably the ladder's baseline arm.
    assert np.array_equal(simple, mirror), (
        "the SimpleScorer run and its mirror copy disagree; the incumbent is not "
        "provably the ladder's noisy-OR baseline"
    )
    hierarchy = score_vector(PAPER_HIERARCHY, sids)
    cogex = score_vector(PAPER_COGEX, sids)
    gate = score_vector(PAPER_GATE, sids)

    manifest = json.loads(Path(ROOT / PAPER_GATE_MANIFEST).read_text())
    # Fail-closed on the bundle's own declaration: if the reader arm did not
    # aggregate with INDRA's default hard gate, the two arms drawn on this panel
    # are not the same scorer and the contrast is void.
    aggregation = manifest["implementation"]["notes"]["aggregation"]
    assert aggregation == REQUIRED_GATE_AGGREGATION, (
        f"the reader bundle aggregates as {aggregation!r}, not "
        f"{REQUIRED_GATE_AGGREGATION!r}; the two arms would not be the same scorer"
    )
    cogex_manifest = json.loads(Path(ROOT / PAPER_COGEX_MANIFEST).read_text())
    cogex_arm = cogex_manifest["arm"]

    literal = json.loads(Path(ROOT / PAPER_LITERAL).read_text())
    oof = {r["stmt_hash"]: r for r in literal["oof_predictions"][PAPER_RF_KEY]}
    sid_by_hash = {}
    for r in load_jsonl(PAPER_GOLD):
        sid_by_hash[int(r["paper_statement_hash"])] = r["canonical_corpus"]["statement_id"]
    rf_by_sid = {sid_by_hash[h]: oof[h]["prob_correct"] for h in oof}
    research = np.array([rf_by_sid[s] for s in sids], dtype=float)

    reads = paper_reads_per_statement()
    assert reads["total"] == 33361, (
        f"the execution map carries {reads['total']} pairs, not the 33,361 the "
        "reader bundle's cost denominator counts"
    )

    return {
        "key": PAPER_PANEL_KEY,
        "display": PAPER_PANEL_DISPLAY,
        "is_paper_panel": True,
        "label_field": "paper_replication_policy.released_paper_correct",
        "label_note": "the paper's own released curation labels; we produced none of them",
        "n_curators": None,
        "curator_note": "the paper's own curators",
        "out_of_sample": True,
        "in_sample_note": None,
        "balanced_by_construction": False,
        "y": y,
        "gate": gate,
        "matched_no_read": None,
        "reads_per_statement": reads,
        # The reader was handed the statement's OWN assembled evidence here, so
        # the corpus census and the read census are the same census.
        "corpus_evidence_per_statement": reads,
        "corpus_evidence_absent_because": None,
        "reader_saw_full_evidence": True,
        "reader_evidence_share_of_corpus": 1.0,
        "join_mode": (
            "sorted(statement_id) over the released-label panel; imported from "
            "compute_statement_review_queue.load_panel"
        ),
        "join_diagnostics": {
            "n_statements": int(y.size),
            "n_evidence_pairs": reads["total"],
            "join_is_exact": True,
        },
        "n_undefined_excluded": 0,
        "join_summary": "exact join",
        "variants": [
            {
                "key": "simple_scorer_direct",
                "display": "INDRA SimpleScorer, direct",
                "family": FAMILY_LIBRARY,
                "fitted": False,
                "evidence_scope": "statement_full",
                "source": PAPER_SIMPLE,
                "scores": simple,
                "what_it_computes": (
                    "indra.belief.SimpleScorer 1.24.0 at the bundled default priors, "
                    "run on the paper's own object graph over every direct evidence "
                    "entry (34,035 across the panel). Arm id "
                    "indra_1.24.0_simple_default_direct."
                ),
            },
            {
                "key": "simple_scorer_hierarchy",
                "display": "INDRA SimpleScorer + hierarchy",
                "family": FAMILY_LIBRARY,
                "fitted": False,
                "evidence_scope": "statement_full_plus_hierarchy",
                "source": PAPER_HIERARCHY,
                "scores": hierarchy,
                "what_it_computes": (
                    "the same scorer through indra.belief.BeliefEngine.get_hierarchy_probs "
                    "over indra.belief.build_refinements_graph, so a statement also "
                    "inherits evidence from its more specific descendants. 477 of the "
                    "1,689 statements change. Arm id "
                    "indra_1.24.0_simple_hierarchy_all_sources."
                ),
            },
            {
                "key": "cogex_fitted_hybrid",
                "display": "INDRA’s stored belief (fitted Hybrid)",
                "family": FAMILY_SERVED,
                "fitted": True,
                "evidence_scope": "statement_full",
                "source": PAPER_COGEX,
                "scores": cogex,
                "what_it_computes": (
                    f"{cogex_arm['implementation']} — {cogex_arm['environment']}. "
                    "This is the SAME family as the belief stored on the curation "
                    "panels' statements, replayed on the paper's own 1,689."
                ),
                "provenance_caveat": cogex_arm["label"],
                "analysis_role": cogex_arm["analysis_role"],
            },
        ],
        "research": {"scores": research, "source": PAPER_LITERAL, "key_in_source": PAPER_RF_KEY},
        "gate_sensitivity": None,
        "evidence_matched_control": None,
        "evidence_matched_control_absent_because": (
            "On this panel the incumbent and the gate read the SAME corpus — the "
            "paper's own assembled evidence, 19.75 unique pairs per statement — so "
            "the scope asymmetry the control exists to price does not arise. What "
            "the gate additionally drops without reading (duplicates, evidence with "
            "no sentence, deterministic grounding rejects) is priced separately by "
            "data/results/indra_paper_literal_models_20260724/non_reading_control.json."
        ),
        "provenance": {
            "gold": PAPER_GOLD,
            "gold_sha256": file_sha256(PAPER_GOLD),
            "run": PAPER_GATE,
            "run_sha256": file_sha256(PAPER_GATE),
            "reader_model": "bedrock-gemma-4-26b",
            "join": (
                "sorted(statement_id) over the released-label panel; imported from "
                "compute_statement_review_queue.load_panel"
            ),
            "aggregation": aggregation,
        },
    }


def load_curation_panel(spec: dict) -> dict:
    """One curation panel, through the ship gate's own join."""
    statements, diagnostics = sg.statements_for_run(spec["run"], spec["gold"])
    counts_by_key = source_counts_by_statement(spec["run"], spec["gold"])

    y, gate, matched, served, recomputed, shipped_gate = [], [], [], [], [], []
    reads: list[int] = []
    corpus_evidence: list[int | None] = []
    n_undefined = 0
    for stmt in statements:
        indra_belief = statement_belief(stmt["ev"], INDRA_DEFAULT_PRIORS)
        recal = statement_belief(stmt["ev"], RECALIBRATED_WITH_INDRA_DEFAULTS)
        if (
            indra_belief.belief is None
            or indra_belief.parametric_only is None
            or recal.belief is None
        ):
            n_undefined += 1
            continue
        y.append(1 if stmt["gold_correct"] else 0)
        gate.append(indra_belief.belief)
        matched.append(indra_belief.parametric_only)
        shipped_gate.append(recal.belief)
        reads.append(len(stmt["ev"]))
        stored = stmt["stored_belief"]
        served.append(float(stored) if isinstance(stored, (int, float)) else None)
        counts = counts_by_key.get(stmt["statement_key"])
        corpus_evidence.append(sum(int(v) for v in counts.values()) if counts else None)
        recomputed.append(
            float(compute_edge_reliability_from_counts(counts, INDRA_DEFAULT_PRIORS))
            if counts else None
        )

    y = np.array(y, dtype=int)
    gate = np.array(gate, dtype=float)
    matched = np.array(matched, dtype=float)

    variants = []
    if all(v is not None for v in recomputed):
        variant = {
            "key": "simple_scorer_recomputed",
            "display": "INDRA SimpleScorer, full evidence",
            "family": FAMILY_LIBRARY,
            "fitted": False,
            "evidence_scope": "statement_full",
            "source": spec["gold"],
            "scores": np.array(recomputed, dtype=float),
            "what_it_computes": (
                "the bundled-prior noisy-OR recomputed from the statement's own "
                "INDRA source counts — the statement's FULL database evidence, not "
                "just the curated rows. Priors read verbatim from "
                f"indra/resources/default_belief_probs.json (sha256 "
                f"{INDRA_DEFAULT_PRIORS_SHA256[:12]})."
            ),
        }
        variants.append(variant)
    if all(v is not None for v in served):
        variants.append({
            "key": "indra_served_belief",
            "display": "INDRA’s stored belief",
            "family": FAMILY_SERVED,
            "fitted": True,
            "evidence_scope": "statement_full",
            "source": spec["run"],
            "scores": np.array(served, dtype=float),
            "what_it_computes": (
                "the belief INDRA's own pipeline wrote on the statement, read off "
                "the statement and not re-derived. It is NOT the library-default "
                "SimpleScorer — see served_belief_identity."
            ),
        })
    if not variants:
        raise AssertionError(
            f"{spec['key']}: no deployed-incumbent variant could be sourced; the "
            "panel must be dropped rather than quoted"
        )

    served_arr = np.array([v for v in served if v is not None], dtype=float)
    below_floor = int(np.sum(served_arr < SIMPLE_SCORER_FLOOR - 1e-12))

    # The scope asymmetry as a ratio. The reader was shown only the curated
    # evidence; the incumbent is scored over the statement's whole corpus
    # evidence. Where the gold does not carry source counts we say so rather
    # than estimate it.
    if all(v is not None for v in corpus_evidence):
        corpus_census = spread([int(v) for v in corpus_evidence])
        corpus_absent = None
        share = spread(reads)["total"] / corpus_census["total"]
    else:
        corpus_census = None
        corpus_absent = (
            f"{spec['key']}'s gold carries no per-source evidence counts, so the "
            "statement's corpus evidence total cannot be censused here. The "
            "incumbent drawn on this panel is the belief INDRA stored, which was "
            "computed over that full evidence set whatever its size."
        )
        share = None

    return {
        "key": spec["key"],
        "display": spec["display"],
        "is_paper_panel": False,
        "label_field": "curation gold, any-incorrect-wins",
        "label_note": "statement is correct only when every curated evidence pair is correct",
        "n_curators": spec["n_curators"],
        "curator_note": spec["curator_note"],
        "out_of_sample": spec["out_of_sample"],
        "in_sample_note": spec["in_sample_note"],
        "balanced_by_construction": spec["balanced_by_construction"],
        "y": y,
        "gate": gate,
        "matched_no_read": matched,
        "reads_per_statement": spread(reads),
        "corpus_evidence_per_statement": corpus_census,
        "corpus_evidence_absent_because": corpus_absent,
        "reader_saw_full_evidence": False,
        "reader_evidence_share_of_corpus": share,
        "join_mode": diagnostics["join_mode"],
        # What the join ACTUALLY did, not what it was willing to do. `join_mode`
        # describes a strategy that always mentions the source-hash fallback;
        # only `holdout_cc` ever uses it (its gold carries no matches_hash), and
        # the figure must be able to say which without parsing a sentence.
        "join_summary": (
            "exact join"
            if diagnostics["n_source_fallback_rows"] == 0
            else "source-hash join"
            if diagnostics["n_exact_joined_rows"] == 0
            else "mixed join"
        ),
        "join_diagnostics": diagnostics,
        "n_undefined_excluded": n_undefined,
        "served_below_simple_scorer_floor": below_floor,
        "n_served": int(served_arr.size),
        "variants": variants,
        "research": None,
        "gate_sensitivity": {
            "key": "shipped_production_gate",
            "display": "shipped gate, recalibrated priors",
            "scores": np.array(shipped_gate, dtype=float),
            "note": "Our production hard gate, which also swaps INDRA's priors for the recalibrated ones. Drawn nowhere; carried so the prior swap can be priced.",
        },
        # INDRA's own library-default scorer restricted to EXACTLY the evidence
        # the reader saw. Not an incumbent — INDRA serves no such number — so it
        # never enters the argmax. It exists to price the scope asymmetry.
        "evidence_matched_control": {
            "key": "simple_scorer_reader_evidence",
            "display": "INDRA SimpleScorer, reader’s evidence only",
            "scores": matched,
            "note": (
                "INDRA's library-default noisy-OR over exactly the evidence the "
                "reader was shown, with no gate applied. Not an incumbent: INDRA "
                "never serves this number. It prices the scope asymmetry — the "
                "incumbent above is scored over the statement's full database "
                "evidence, the gate over the curated subset only."
            ),
        },
        "evidence_matched_control_absent_because": None,
        "provenance": {
            "gold": spec["gold"],
            "gold_sha256": file_sha256(spec["gold"]),
            "run": spec["run"],
            "run_sha256": file_sha256(spec["run"]),
            "reader_model": spec["model"],
            "join": diagnostics["join_mode"],
            "join_diagnostics": diagnostics,
            "n_undefined_excluded": n_undefined,
        },
    }


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
EVIDENCE_SCOPES = {
    "statement_full": "the statement's full evidence in the source corpus",
    "statement_full_plus_hierarchy": (
        "the statement's full evidence plus evidence inherited from its more "
        "specific descendants"
    ),
    "reader_evidence_only": "only the evidence the reader was shown",
}


def build_panel_row(panel: dict) -> dict:
    y = panel["y"]
    gate = panel["gate"]
    n = int(y.size)
    n_correct = int(y.sum())
    n_errors = n - n_correct

    gate_auroc = auroc(y, gate)

    # (b) the gate is subtractive: it may never exceed the matched no-read score.
    matched = panel.get("matched_no_read")
    if matched is not None:
        exceed = int(np.sum(gate > matched + 1e-12))
        assert exceed == 0, (
            f"{panel['key']}: the gate exceeds the ungated noisy-OR on {exceed} "
            "statements; it is not the same aggregation"
        )

    variants = []
    for variant in panel["variants"]:
        v_auroc = auroc(y, variant["scores"])
        row = {
            "key": variant["key"],
            "display": variant["display"],
            "family": variant["family"],
            "fitted": variant["fitted"],
            "evidence_scope": variant["evidence_scope"],
            "evidence_scope_note": EVIDENCE_SCOPES[variant["evidence_scope"]],
            "source": variant["source"],
            "source_sha256": file_sha256(variant["source"]),
            "auroc": v_auroc,
            "average_precision": average_precision(y, variant["scores"]),
            "delta_auroc": gate_auroc - v_auroc,
            # EVERY variant gets its own paired interval, so "the gate beats every
            # sourceable form of INDRA's belief" is a fact in the bytes.
            "bootstrap": paired_bootstrap_delta(y, gate, variant["scores"], SEED, N_BOOT),
            "what_it_computes": variant["what_it_computes"],
        }
        for optional in ("provenance_caveat", "analysis_role"):
            if optional in variant:
                row[optional] = variant[optional]
        if "cross_check" in variant:
            cross = dict(variant["cross_check"])
            cross["this_auroc"] = v_auroc
            cross["delta_vs_sibling"] = v_auroc - cross["sibling_auroc"]
            # The whole point of recording it: reading INDRA's own resource must
            # not have handed us an EASIER comparator than the shipped sibling.
            assert cross["delta_vs_sibling"] >= -TOL_IDENTITY, (
                f"{variant['key']}: reading INDRA's own prior resource produced a "
                f"WEAKER incumbent than {cross['sibling']}; the argmax rule "
                "requires the stronger form"
            )
            row["cross_check"] = cross
        variants.append(row)

    # (c) headline incumbent = the STRONGEST variant this panel can source.
    strongest = max(variants, key=lambda v: v["auroc"])
    weakest = min(variants, key=lambda v: v["auroc"])
    strongest_scores = next(
        v["scores"] for v in panel["variants"] if v["key"] == strongest["key"]
    )
    boot = paired_bootstrap_delta(y, gate, strongest_scores, SEED, N_BOOT)
    delta = gate_auroc - strongest["auroc"]
    # (d) the drawn delta is exactly the difference of the two drawn levels, and
    # it is exactly the strongest variant's own recorded delta.
    assert abs(delta - strongest["delta_auroc"]) <= TOL_IDENTITY
    assert boot == strongest["bootstrap"], (
        f"{panel['key']}: the headline interval is not the strongest variant's own"
    )

    research = None
    if panel["research"] is not None:
        r_scores = panel["research"]["scores"]
        r_auroc = auroc(y, r_scores)
        research = {
            "key": RESEARCH_MODEL["key"],
            "display": RESEARCH_MODEL["display"],
            "source": panel["research"]["source"],
            "key_in_source": panel["research"]["key_in_source"],
            "auroc": r_auroc,
            "average_precision": average_precision(y, r_scores),
            "delta_auroc_gate_minus_research": gate_auroc - r_auroc,
            "delta_auroc_research_minus_incumbent": r_auroc - strongest["auroc"],
            "bootstrap": paired_bootstrap_delta(y, gate, r_scores, SEED, N_BOOT),
        }

    sensitivity = None
    if panel["gate_sensitivity"] is not None:
        s = panel["gate_sensitivity"]
        s_auroc = auroc(y, s["scores"])
        sensitivity = {
            "key": s["key"],
            "display": s["display"],
            "auroc": s_auroc,
            "delta_auroc": s_auroc - strongest["auroc"],
            "note": s["note"],
        }

    control = None
    if panel["evidence_matched_control"] is not None:
        c = panel["evidence_matched_control"]
        c_auroc = auroc(y, c["scores"])
        control = {
            "key": c["key"],
            "display": c["display"],
            "evidence_scope": "reader_evidence_only",
            "evidence_scope_note": EVIDENCE_SCOPES["reader_evidence_only"],
            "auroc": c_auroc,
            "average_precision": average_precision(y, c["scores"]),
            "delta_auroc_incumbent_minus_control": strongest["auroc"] - c_auroc,
            "at_or_below_chance": c_auroc <= 0.5,
            "is_an_incumbent": False,
            "note": c["note"],
        }

    heterogeneity = {
        "evidence_reads_per_statement": panel["reads_per_statement"],
        "n_curators": panel["n_curators"],
        "curator_note": panel["curator_note"],
        "base_rate_correct": n_correct / n,
        "balanced_by_construction": panel["balanced_by_construction"],
        "out_of_sample": panel["out_of_sample"],
        "in_sample_note": panel["in_sample_note"],
        "label_field": panel["label_field"],
        "label_note": panel["label_note"],
        "join_mode": panel["join_mode"],
        "join_summary": panel["join_summary"],
        "join_diagnostics": panel["join_diagnostics"],
        "n_undefined_excluded": panel["n_undefined_excluded"],
        "incumbent_evidence_scope": strongest["evidence_scope"],
        "incumbent_evidence_scope_note": strongest["evidence_scope_note"],
        "gate_evidence_scope": "reader_evidence_only",
        "gate_evidence_scope_note": EVIDENCE_SCOPES["reader_evidence_only"],
        # The evidence the statement carries in the SOURCE corpus, where the
        # panel can source it. Beside `evidence_reads_per_statement` it is the
        # scope asymmetry as a ratio rather than an adjective.
        "corpus_evidence_per_statement": panel["corpus_evidence_per_statement"],
        "corpus_evidence_absent_because": panel["corpus_evidence_absent_because"],
        "reader_saw_full_evidence": panel["reader_saw_full_evidence"],
        "reader_evidence_share_of_corpus": panel["reader_evidence_share_of_corpus"],
    }

    return {
        "key": panel["key"],
        "display": panel["display"],
        "is_paper_panel": panel["is_paper_panel"],
        "n_statements": n,
        "n_correct": n_correct,
        "n_errors": n_errors,
        "positive_class": POSITIVE_CLASS,
        "base_rate_correct": n_correct / n,
        "balanced_by_construction": panel["balanced_by_construction"],
        "n_curators": panel["n_curators"],
        "curator_note": panel["curator_note"],
        "out_of_sample": panel["out_of_sample"],
        "in_sample_note": panel["in_sample_note"],
        "label_field": panel["label_field"],
        "label_note": panel["label_note"],
        "heterogeneity": heterogeneity,
        "gate": {
            "key": GATE["key"],
            "display": GATE["display"],
            "auroc": gate_auroc,
            "average_precision": average_precision(y, gate),
        },
        "incumbent": {
            "key": strongest["key"],
            "display": strongest["display"],
            "family": strongest["family"],
            "fitted": strongest["fitted"],
            "auroc": strongest["auroc"],
            "average_precision": strongest["average_precision"],
            "selected_by": "argmax auroc over incumbent_variants",
        },
        "incumbent_variants": variants,
        "n_incumbent_variants": len(variants),
        # What the strongest-incumbent rule costs us on this panel: the margin we
        # forfeit by refusing the weakest sourceable form of INDRA's belief.
        "selection_cost_auroc": strongest["auroc"] - weakest["auroc"],
        "weakest_variant_key": weakest["key"],
        "weakest_variant_auroc": weakest["auroc"],
        "gate_beats_every_variant": all(v["delta_auroc"] > 0.0 for v in variants),
        "n_variants_ci_excludes_zero": sum(
            1 for v in variants if v["bootstrap"]["ci95_low"] > 0.0
        ),
        "delta_auroc": delta,
        "delta_favors_gate": delta > 0.0,
        "bootstrap": boot,
        "research_model": research,
        "gate_sensitivity": sensitivity,
        "evidence_matched_control": control,
        "evidence_matched_control_absent_because": panel["evidence_matched_control_absent_because"],
        "provenance": panel["provenance"],
    }


def build_served_identity(panels: list[dict], raw_panels: list[dict]) -> dict:
    """The recomputed proof that the stored belief is NOT SimpleScorer.

    Shipped as data rather than prose: if INDRA ever changes its bundled priors
    or its export scorer so that the stored beliefs DO land inside SimpleScorer's
    reachable range, `n_panels_with_served_below_floor` drops to zero and the
    viewer's contract fails closed on a family split it can no longer justify.
    """
    per_panel = []
    for panel, raw in zip(panels, raw_panels):
        if "served_below_simple_scorer_floor" not in raw:
            continue
        per_panel.append({
            "panel_key": panel["key"],
            "n_served": raw["n_served"],
            "n_below_floor": raw["served_below_simple_scorer_floor"],
            "fraction_below_floor": raw["served_below_simple_scorer_floor"] / raw["n_served"],
        })
    total_below = sum(p["n_below_floor"] for p in per_panel)
    # (f) the split has to be earned by the data, not asserted by the author.
    assert total_below > 0, (
        "no stored belief falls below the SimpleScorer floor; the two-family "
        "split is unsupported and the artifact must not claim it"
    )
    return {
        "question": (
            "Is the belief INDRA stored on the statement the same thing as "
            "indra.belief.SimpleScorer at the bundled default priors?"
        ),
        "finding": (
            "No. SimpleScorer at those priors cannot score below "
            f"{SIMPLE_SCORER_FLOOR:.2f}, and hierarchy propagation only raises a "
            f"belief, yet {total_below} stored beliefs across the curation panels "
            "fall below it."
        ),
        "simple_scorer_floor": SIMPLE_SCORER_FLOOR,
        "floor_derivation": (
            "belief = 1 - PROD_s (syst_s + rand_s^{n_s}); every factor is at most "
            "syst_s + rand_s, so belief >= 1 - max_s (syst_s + rand_s). The worst "
            f"source in the bundled file is {SIMPLE_SCORER_FLOOR_SOURCE}."
        ),
        # The canonical package path, never the local site-packages path: the
        # digest identifies the bytes, and the bytes are the library's, not ours.
        "floor_source": "indra/resources/default_belief_probs.json",
        "floor_source_sha256": INDRA_DEFAULT_PRIORS_SHA256,
        "per_panel": per_panel,
        "n_served_below_floor": total_below,
        "n_panels_with_served_below_floor": sum(1 for p in per_panel if p["n_below_floor"] > 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    # AUROC estimator parity: this file's rank-sum implementation must agree with
    # sklearn's on real data before any of it is trusted.
    from sklearn.metrics import roc_auc_score

    paper = load_paper_panel()
    check = abs(auroc(paper["y"], paper["gate"]) - float(roc_auc_score(paper["y"], paper["gate"])))
    assert check <= 1e-12, f"the local AUROC disagrees with sklearn by {check}"

    raw_panels = [paper] + [load_curation_panel(spec) for spec in CURATION_PANELS]
    panels = [build_panel_row(raw) for raw in raw_panels]

    # (e) the paper panel's re-derived levels must equal the shipped ones.
    shipped = json.loads(Path(ROOT / PAPER_VS_LLMS).read_text())["point_metrics"]
    paper_row = panels[0]
    cogex_row = next(
        v for v in paper_row["incumbent_variants"] if v["key"] == "cogex_fitted_hybrid"
    )
    recorded_checks = {
        "gate_auroc_matches_paper_literal_vs_llms": abs(
            paper_row["gate"]["auroc"] - shipped["Gemma 4 26B"]["auroc"]
        ) <= TOL_RECORDED,
        "research_auroc_matches_paper_literal_vs_llms": abs(
            paper_row["research_model"]["auroc"]
            - shipped["Paper literal RF+promoter"]["auroc"]
        ) <= TOL_RECORDED,
        "cogex_auroc_matches_paper_literal_vs_llms": abs(
            cogex_row["auroc"] - shipped[PAPER_COGEX_RECORDED_KEY]["auroc"]
        ) <= TOL_RECORDED,
        "cogex_ap_matches_paper_literal_vs_llms": abs(
            cogex_row["average_precision"]
            - shipped[PAPER_COGEX_RECORDED_KEY]["pooled_average_precision"]
        ) <= TOL_RECORDED,
    }
    for name, passed in recorded_checks.items():
        assert passed, f"{name} FAILED"

    served_identity = build_served_identity(panels, raw_panels)

    deltas = [p["delta_auroc"] for p in panels]
    largest = max(panels, key=lambda p: p["n_statements"])
    # The evidence regime, as a span rather than an adjective. The gate does real
    # aggregation on the paper's panel (19.75 reads/statement) and, on the
    # curation panels, is mostly a bare keep/drop on ONE sentence. That the claim
    # survives both regimes is the point; the fold-span is how the figure says so
    # without an author asserting "16x".
    read_means = [
        p["heterogeneity"]["evidence_reads_per_statement"]["mean"] for p in panels
    ]
    single_shares = [
        p["heterogeneity"]["evidence_reads_per_statement"]["share_single"] for p in panels
    ]
    replication = {
        "n_panels": len(panels),
        "n_panels_favoring_gate": sum(1 for d in deltas if d > 0.0),
        "n_panels_ci_excludes_zero": sum(
            1 for p in panels if p["bootstrap"]["ci95_low"] > 0.0
        ),
        "n_panels_gate_beats_every_variant": sum(
            1 for p in panels if p["gate_beats_every_variant"]
        ),
        "n_incumbent_variants_total": sum(p["n_incumbent_variants"] for p in panels),
        "n_incumbent_variants_ci_excludes_zero": sum(
            p["n_variants_ci_excludes_zero"] for p in panels
        ),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "largest_panel_key": largest["key"],
        "largest_panel_delta": largest["delta_auroc"],
        "largest_panel_is_at_top_of_range": largest["delta_auroc"] == max(deltas),
        "selection_cost_auroc_max": max(p["selection_cost_auroc"] for p in panels),
        "panel_keys_by_delta_descending": [
            p["key"] for p in sorted(panels, key=lambda p: -p["delta_auroc"])
        ],
        "reads_per_statement_mean_min": min(read_means),
        "reads_per_statement_mean_max": max(read_means),
        "evidence_regime_fold_span": max(read_means) / min(read_means),
        "share_single_evidence_min": min(single_shares),
        "share_single_evidence_max": max(single_shares),
    }

    artifact = {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "metric": METRIC,
        "metric_source": METRIC_SOURCE,
        "positive_class": POSITIVE_CLASS,
        "noisy_or_formula": NOISY_OR_FORMULA,
        "question": (
            "Does a reader gate beat INDRA's own belief — both the unfitted "
            "SimpleScorer the library ships and the fitted score INDRA's pipeline "
            "actually stores — and does it replicate?"
        ),
        "figure_title": FIGURE_TITLE,
        "claim": build_claim(panels, replication),
        "claim_is_not": build_claim_is_not(panels),
        "incumbent_families": INCUMBENT_FAMILIES,
        "arms": {
            "gate": GATE,
            "research_model": RESEARCH_MODEL,
        },
        "incumbent_selection_rule": (
            "Every sourceable form of INDRA's own belief is scored on each panel — "
            "library default and stored production belief alike — and the headline "
            "incumbent is the one with the HIGHEST AUROC on that panel. The gate is "
            "therefore compared against the strongest version of the incumbent each "
            "panel can produce, never the most convenient one."
        ),
        "incumbent_selection_rule_cost": (
            "Taking the argmax costs us margin, and the figure prints how much per "
            "row. The largest forfeit is "
            f"{replication['selection_cost_auroc_max']:.4f} AUROC."
        ),
        "served_belief_identity": served_identity,
        "evidence_scopes": EVIDENCE_SCOPES,
        "panels": panels,
        "replication": replication,
        "checks": {
            "simple_scorer_mirror_is_bit_identical": True,
            "gate_never_exceeds_ungated_noisy_or": True,
            "headline_incumbent_is_panel_argmax": True,
            "delta_equals_gate_minus_incumbent": True,
            "headline_interval_is_the_strongest_variants_own": True,
            "served_belief_is_not_simple_scorer": True,
            "recorded_value_tol": TOL_RECORDED,
            **recorded_checks,
            "note": (
                "Assertions are enforced in code; a violation fails the build "
                "rather than being reported here as False."
            ),
        },
        "caveats": build_caveats(panels, replication),
        "generated_by": "scripts/compute_deployed_baseline_replication.py",
    }

    out = Path(ROOT / args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=1, sort_keys=False, default=float) + "\n"
    out.write_text(payload)

    manifest_path = Path(ROOT / args.manifest)
    manifest = {
        "artifact_kind": ARTIFACT_KIND,
        "schema_version": SCHEMA_VERSION,
        "generated_by": "scripts/compute_deployed_baseline_replication.py",
        "n_boot": N_BOOT,
        "seed": SEED,
        "indra_default_prior_resource": INDRA_DEFAULT_PRIOR_RESOURCE,
        "indra_default_prior_resource_sha256": INDRA_DEFAULT_PRIORS_SHA256,
        "inputs": {
            p["key"]: {
                "gold": p["provenance"]["gold"],
                "gold_sha256": p["provenance"]["gold_sha256"],
                "run": p["provenance"]["run"],
                "run_sha256": p["provenance"]["run_sha256"],
                "incumbent_variants": {
                    v["key"]: {"source": v["source"], "source_sha256": v["source_sha256"]}
                    for v in p["incumbent_variants"]
                },
            }
            for p in panels
        },
        "output_sha256": {
            out.name: hashlib.sha256(payload.encode("utf8")).hexdigest(),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"wrote {args.out_json}")
    print(
        f"  SimpleScorer floor {SIMPLE_SCORER_FLOOR:.2f} ({SIMPLE_SCORER_FLOOR_SOURCE}); "
        f"{served_identity['n_served_below_floor']} stored beliefs fall below it"
    )
    for p in panels:
        print(
            f"  {p['key']:22s} n={p['n_statements']:5d}  "
            f"reads/stmt {p['heterogeneity']['evidence_reads_per_statement']['mean']:5.2f}  "
            f"single-ev {p['heterogeneity']['evidence_reads_per_statement']['share_single']:5.1%}  "
            f"incumbent {p['incumbent']['auroc']:.4f} ({p['incumbent']['key']})  "
            f"gate {p['gate']['auroc']:.4f}  "
            f"delta {p['delta_auroc']:+.4f} "
            f"[{p['bootstrap']['ci95_low']:+.4f}, {p['bootstrap']['ci95_high']:+.4f}]  "
            f"rule cost {p['selection_cost_auroc']:.4f}"
        )
        for v in p["incumbent_variants"]:
            print(
                f"      {v['key']:26s} {v['auroc']:.4f}  delta {v['delta_auroc']:+.4f} "
                f"[{v['bootstrap']['ci95_low']:+.4f}, {v['bootstrap']['ci95_high']:+.4f}]  "
                f"{v['family']}"
            )
        if p["evidence_matched_control"]:
            c = p["evidence_matched_control"]
            print(f"      {'(control) ' + c['key']:26s} {c['auroc']:.4f}  reader-evidence only")


if __name__ == "__main__":
    main()
