"""Gate the deployed-baseline replication artifact the /paper figure draws.

THE CLAIM UNDER TEST. There are TWO forms of INDRA's own belief, and both are
INDRA's. ``indra.belief.SimpleScorer`` at the priors ``indra`` bundles is the
unfitted noisy-OR anyone who installs the library gets. The belief INDRA's own
export pipeline WROTE on the statement is a different, FITTED thing — the
HybridScorer path the CoGEx export uses — and it is what a db.indra.bio user
reads. The figure says a reader gate beats the STRONGEST form each panel can
source, on four independently-sourced panels. If that stops being true, or stops
being derivable from the files the artifact names, the figure must go dark rather
than keep asserting it.

THE TWO-FAMILY SPLIT IS RE-DERIVED HERE, not taken on trust. INDRA's shipped
prior file is read from the installed package and the SimpleScorer floor is
recomputed from it: no statement, with any evidence, can score below
``1 - max_s (syst_s + rand_s)``, and hierarchy propagation only raises a belief.
The artifact's floor must equal that recomputation, and its per-panel
below-the-floor counts must be arithmetically consistent and non-zero — because a
split the data no longer supports must not be drawn.

WHAT ELSE IS RE-DERIVED, and what is not.

The paper's own 1689-statement panel is re-derived END TO END in this file: the
labels are read from the frozen gold, every arm's scores are read from the
prediction file the artifact names, and every AUROC is recomputed with
``sklearn.metrics.roc_auc_score`` — a different estimator implementation from the
rank-sum one the compute script uses. Its evidence census is recounted from the
frozen execution map. That is the headline panel, so pytest owns it outright.

The three curation panels reach their statements through the ship gate's join,
which the compute script asserts against for itself (same precedent as the 19 MB
execution map in ``test_viewer_paper_literal_contract.py``: the compute script
owns its own heavy join). What pytest owns for those panels is INDEPENDENT
CROSS-CHECK: every level that also appears in an already-shipped sibling artifact
must agree with the sibling to the precision the sibling prints, and where a
level deliberately DIVERGES from its sibling the divergence itself is pinned on
both ends. Two of the three panels are covered that way; the third is not, and
this file records exactly why rather than pretending otherwise.

The .mjs runner asserts the same properties on the TypeScript side, so the figure
cannot drift from the shipped numbers in either language.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


import _local_artifacts as _artifacts

ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-deployed-baseline-contract.mjs"

_MODEL_DIR = ROOT / "data" / "results" / "deployed_baseline_replication_20260727"
_ARTIFACT_NAME = "deployed_baseline_replication.json"
_ARTIFACT_PATH = _MODEL_DIR / _ARTIFACT_NAME
_MANIFEST_PATH = _MODEL_DIR / "manifest.json"

_PAPER_GOLD = ROOT / "data/results/indra_paper_statement_gold_20260717/paper_statement_gold.jsonl"
_PAPER_LITERAL = ROOT / (
    "data/results/indra_paper_literal_models_20260724/paper_literal_table6_and_oof.json"
)
_PAPER_VS_LLMS = ROOT / (
    "data/results/indra_paper_literal_models_20260724/paper_literal_vs_llms.json"
)
_PAPER_GATE_SCORES = ROOT / "data/comparison/models/gemma_4_26b/all_source_predictions.jsonl"
_PAPER_EXECUTION_MAP = ROOT / (
    "data/benchmark/indra_paper_unique_pairs_20260717_execution_map.jsonl"
)

_PAPER_PANEL_KEY = "indra_paper_2023"
_PANEL_KEYS = ["indra_paper_2023", "eval_curation_v1", "external_curator_v1", "holdout_cc"]
_FAMILY_LIBRARY = "indra_library_default"
_FAMILY_SERVED = "indra_production_served"

# The artifact lives under `data/results/`, which .gitignore excludes: like every
# other /paper artifact it is REPRODUCED by its compute script rather than
# checked in. On a tree where it has not been generated there is nothing to gate,
# so this module skips rather than failing — a missing artifact is "not
# applicable here", not "the claim is false". An artifact that IS present and has
# drifted still fails loudly, which is the case this file exists for.
# TWO marks, as a list. An assignment would REPLACE the `_artifacts.requires()`
# above rather than add to it — which it silently did when that guard was first
# added here, leaving eight tests running on a tree with no comparison corpus.
# The two conditions are different: this one is about the module's own artifact
# having been generated, that one about the gitignored corpus being present at
# all, and the tests below read both.
pytestmark = [_artifacts.requires(), pytest.mark.skipif(
    not _ARTIFACT_PATH.exists(),
    reason=(
        "data/results/deployed_baseline_replication_20260727/ has not been "
        "generated — run scripts/compute_deployed_baseline_replication.py"
    ),
)]

# The artifact is float64 throughout; these are float-noise tolerances, not a
# licence for disagreement.
_PARITY_TOL = 1e-9
# Sibling head-to-head artifacts print AUROC rounded to four decimals, so a
# cross-check against them can only be held to half a unit in the last place.
_SIBLING_TOL = 5e-5


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return _load(_ARTIFACT_PATH)


@pytest.fixture(scope="module")
def paper_panel() -> tuple[list[str], np.ndarray]:
    """The 1689 statements with a released paper label, sorted by statement_id.

    Re-derived here rather than imported, so the panel this file scores is not
    the panel the compute script scored by construction.
    """
    labels: dict[str, int] = {}
    for row in _load_jsonl(_PAPER_GOLD):
        policy = row.get("paper_replication_policy") or {}
        if policy.get("released_paper_correct") is None:
            continue
        sid = row["canonical_corpus"]["statement_id"]
        assert sid not in labels, f"duplicate statement_id in gold: {sid}"
        labels[sid] = int(policy["released_paper_correct"])
    sids = sorted(labels)
    return sids, np.array([labels[s] for s in sids], dtype=int)


def _scores(path: Path, sids: list[str]) -> np.ndarray:
    by_id = {r["statement_id"]: r["probability_correct"] for r in _load_jsonl(path)}
    return np.array([by_id[s] for s in sids], dtype=float)


def _panel(artifact: dict, key: str) -> dict:
    match = [p for p in artifact["panels"] if p["key"] == key]
    assert len(match) == 1, f"expected exactly one {key} panel"
    return match[0]


def _family(artifact: dict, key: str) -> dict:
    match = [f for f in artifact["incumbent_families"] if f["key"] == key]
    assert len(match) == 1, f"expected exactly one {key} family"
    return match[0]


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_artifact_matches_its_manifest_sha256() -> None:
    """The bytes the viewer draws must be the bytes the run signed."""
    digest = hashlib.sha256(_ARTIFACT_PATH.read_bytes()).hexdigest()
    recorded = _load(_MANIFEST_PATH)["output_sha256"][_ARTIFACT_NAME]
    assert recorded == digest


def test_every_panel_pins_the_files_it_was_computed_from(artifact: dict) -> None:
    """A panel that cannot be re-run from the bytes it names is not sourced."""
    for panel in artifact["panels"]:
        prov = panel["provenance"]
        for role in ("gold", "run"):
            path = ROOT / prov[role]
            assert path.exists(), f"{panel['key']} names a missing {role}: {prov[role]}"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == prov[f"{role}_sha256"], (
                f"{panel['key']} {role} has drifted from the digest the artifact pins"
            )


def test_every_incumbent_variant_pins_its_own_source(artifact: dict) -> None:
    """Each form of INDRA belief names, and matches, the bytes it was read from.

    The panel-level gold/run digests do not cover these: three of the seven forms
    are scored from prediction files no panel provenance block mentions.
    """
    for panel in artifact["panels"]:
        for variant in panel["incumbent_variants"]:
            path = ROOT / variant["source"]
            assert path.exists(), f"{panel['key']}/{variant['key']} names a missing source"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == variant["source_sha256"], (
                f"{panel['key']}/{variant['key']} source has drifted from its pinned digest"
            )


def test_panel_order_and_census_are_frozen(artifact: dict) -> None:
    assert [p["key"] for p in artifact["panels"]] == _PANEL_KEYS
    assert artifact["artifact_kind"] == "deployed_baseline_replication"
    assert artifact["schema_version"] == 3
    assert artifact["metric"] == "auroc"
    assert artifact["positive_class"] == "gold-correct"


# ---------------------------------------------------------------------------
# the two families, and the proof they are two
# ---------------------------------------------------------------------------
def test_both_incumbent_families_are_indras_own_and_only_one_is_fitted(artifact: dict) -> None:
    """The distinction the whole figure now rests on, asserted not described."""
    library = _family(artifact, _FAMILY_LIBRARY)
    assert library["deployed"] is True
    assert library["fitted"] is False, "the library default is the UNFITTED noisy-OR"
    assert "SimpleScorer" in library["ships_in"]
    assert "default_belief_probs" in library["ships_in"]

    served = _family(artifact, _FAMILY_SERVED)
    assert served["deployed"] is True
    assert served["fitted"] is True, "the belief INDRA stored comes from a FITTED scorer"

    research = artifact["arms"]["research_model"]
    assert research["deployed"] is False, "the paper's RF was never deployed"
    assert research["fitted"] is True

    gate = artifact["arms"]["gate"]
    assert gate["fitted"] is False, "the hard gate fits nothing"
    assert "14" in gate["not_zero_shot"], "the demonstration-pair count must travel"


def test_the_simple_scorer_floor_is_rederived_from_indras_own_resource(artifact: dict) -> None:
    """The floor is a property of INDRA's bundled priors, recomputed here.

    ``belief = 1 - PROD_s (syst_s + rand_s^{n_s})`` and every factor is at most
    ``syst_s + rand_s``, so no SimpleScorer belief can fall below
    ``1 - max_s (syst_s + rand_s)``. Recomputed from the installed package rather
    than read off the artifact, so the artifact cannot define its own floor.
    """
    import indra

    resource = Path(indra.__file__).resolve().parent / "resources" / "default_belief_probs.json"
    raw = resource.read_bytes()
    probs = json.loads(raw)
    worst = max(probs["syst"][s] + probs["rand"][s] for s in probs["syst"])
    floor = 1.0 - worst

    identity = artifact["served_belief_identity"]
    assert abs(identity["simple_scorer_floor"] - floor) <= _PARITY_TOL, (
        "the artifact's SimpleScorer floor is not the floor INDRA's own priors imply"
    )
    assert identity["floor_source_sha256"] == hashlib.sha256(raw).hexdigest(), (
        "the artifact pins a different prior resource than the installed indra ships"
    )


def test_the_two_family_split_is_earned_by_the_data(artifact: dict) -> None:
    """Stored beliefs must actually fall below the floor, or the split is invented."""
    identity = artifact["served_belief_identity"]
    panel_keys = {p["key"] for p in artifact["panels"]}
    total = 0
    for row in identity["per_panel"]:
        assert row["panel_key"] in panel_keys
        assert 0 <= row["n_below_floor"] <= row["n_served"]
        assert abs(
            row["fraction_below_floor"] - row["n_below_floor"] / row["n_served"]
        ) <= _PARITY_TOL
        total += row["n_below_floor"]
    assert identity["n_served_below_floor"] == total
    assert total > 0, (
        "no stored belief falls below the SimpleScorer floor, so the served belief "
        "may BE the library default and the figure must not draw two families"
    )
    assert identity["n_panels_with_served_below_floor"] == sum(
        1 for row in identity["per_panel"] if row["n_below_floor"] > 0
    )


def test_a_variant_that_is_the_served_belief_is_never_called_unfitted(artifact: dict) -> None:
    """Family and fitted-ness travel together, on every variant and every row."""
    by_family = {
        _FAMILY_LIBRARY: False,
        _FAMILY_SERVED: True,
    }
    for panel in artifact["panels"]:
        for variant in panel["incumbent_variants"]:
            assert variant["family"] in by_family, f"unknown family {variant['family']}"
            assert variant["fitted"] is by_family[variant["family"]], (
                f"{panel['key']}/{variant['key']} claims fitted={variant['fitted']} "
                f"for family {variant['family']}"
            )
        assert panel["incumbent"]["family"] == next(
            v["family"] for v in panel["incumbent_variants"] if v["key"] == panel["incumbent"]["key"]
        )


def test_both_families_are_actually_used(artifact: dict) -> None:
    """A family nobody sources is a legend entry the reader cannot find."""
    drawn = {v["family"] for p in artifact["panels"] for v in p["incumbent_variants"]}
    assert drawn == {_FAMILY_LIBRARY, _FAMILY_SERVED}
    paper = _panel(artifact, _PAPER_PANEL_KEY)
    assert {v["family"] for v in paper["incumbent_variants"]} == {
        _FAMILY_LIBRARY,
        _FAMILY_SERVED,
    }, (
        "the paper's own panel must be compared against BOTH forms of INDRA belief; "
        "it is the one panel where both are sourceable, and skipping the stronger "
        "would be the convenient baseline the argmax rule exists to forbid"
    )


# ---------------------------------------------------------------------------
# the claim
# ---------------------------------------------------------------------------
def test_every_panel_favours_the_gate_and_excludes_zero(artifact: dict) -> None:
    for panel in artifact["panels"]:
        assert panel["delta_auroc"] > 0, f"{panel['key']} does not favour the gate"
        assert panel["delta_favors_gate"] is True
        assert panel["bootstrap"]["ci95_low"] > 0, (
            f"{panel['key']}'s paired interval does not exclude zero"
        )
    rep = artifact["replication"]
    assert rep["n_panels_favoring_gate"] == rep["n_panels"] == len(_PANEL_KEYS)
    assert rep["n_panels_ci_excludes_zero"] == rep["n_panels"]


def test_the_gate_beats_every_sourceable_form_not_just_the_one_drawn(artifact: dict) -> None:
    """The strong form of the claim, and the one the figure's ticks assert."""
    for panel in artifact["panels"]:
        for variant in panel["incumbent_variants"]:
            assert variant["delta_auroc"] > 0, (
                f"{panel['key']}/{variant['key']} is not beaten by the gate"
            )
            assert variant["bootstrap"]["ci95_low"] > 0, (
                f"{panel['key']}/{variant['key']}'s paired interval does not exclude zero"
            )
        assert panel["gate_beats_every_variant"] is True
    rep = artifact["replication"]
    assert rep["n_panels_gate_beats_every_variant"] == rep["n_panels"]
    assert rep["n_incumbent_variants_ci_excludes_zero"] == rep["n_incumbent_variants_total"]


def test_the_largest_panel_is_the_papers_own(artifact: dict) -> None:
    """The result must not be riding on the smallest panel.

    NOTE what this does NOT assert. Under schema 1 the paper's panel also sat at
    the TOP of the delta range, because its comparator was the unfitted
    SimpleScorer. It no longer does: the paper panel can source INDRA's fitted
    stored belief too, the argmax rule takes it, and the delta there drops from
    +0.119 to +0.074 — below the two curator panels. That is the rule working,
    not a regression, so the assertion is that the biggest panel still clears
    zero with room, not that it leads.
    """
    largest = max(artifact["panels"], key=lambda p: p["n_statements"])
    assert largest["key"] == _PAPER_PANEL_KEY
    assert artifact["replication"]["largest_panel_key"] == _PAPER_PANEL_KEY
    deltas = [p["delta_auroc"] for p in artifact["panels"]]
    assert artifact["replication"]["largest_panel_is_at_top_of_range"] == (
        largest["delta_auroc"] == max(deltas)
    ), "the flag must be the comparison it reports, not a claim"
    assert largest["bootstrap"]["ci95_low"] > 0
    assert largest["delta_auroc"] > 0


def test_each_panel_draws_its_strongest_sourceable_incumbent(artifact: dict) -> None:
    """The comparison is against the best version of the incumbent, not a handy one."""
    for panel in artifact["panels"]:
        variants = panel["incumbent_variants"]
        assert variants, f"{panel['key']} has no sourceable incumbent and must be dropped"
        strongest = max(variants, key=lambda v: v["auroc"])
        weakest = min(variants, key=lambda v: v["auroc"])
        assert panel["incumbent"]["key"] == strongest["key"], (
            f"{panel['key']} draws {panel['incumbent']['key']} while "
            f"{strongest['key']} scores higher"
        )
        assert abs(panel["incumbent"]["auroc"] - strongest["auroc"]) <= _PARITY_TOL
        # The rule's price is arithmetic, not a sentence.
        assert abs(
            panel["selection_cost_auroc"] - (strongest["auroc"] - weakest["auroc"])
        ) <= _PARITY_TOL
        assert panel["weakest_variant_key"] == weakest["key"]
        # And the headline interval is the DRAWN comparator's own, never a
        # different pairing that happens to be narrower.
        assert panel["bootstrap"] == strongest["bootstrap"]


def test_the_argmax_rule_actually_costs_us_something(artifact: dict) -> None:
    """A conservative rule that never gives anything up is not being exercised."""
    costs = {p["key"]: p["selection_cost_auroc"] for p in artifact["panels"]}
    assert all(c >= 0 for c in costs.values())
    assert max(costs.values()) > 0, (
        "the strongest-incumbent rule forfeits nothing anywhere, which means no "
        "panel sources more than one form of INDRA belief and the rule is inert"
    )
    assert abs(
        artifact["replication"]["selection_cost_auroc_max"] - max(costs.values())
    ) <= _PARITY_TOL
    # It must cost the most exactly where the most is sourceable — the paper's
    # own panel, which is also where an author will look hardest.
    assert costs[_PAPER_PANEL_KEY] == max(costs.values())


def test_the_papers_rf_is_drawn_on_the_papers_panel_alone(artifact: dict) -> None:
    """Anywhere else it would be invented: its released predictions exist only there."""
    for panel in artifact["panels"]:
        has_rf = panel["research_model"] is not None
        assert has_rf == (panel["key"] == _PAPER_PANEL_KEY)


def test_every_drawn_delta_is_the_difference_of_two_drawn_levels(artifact: dict) -> None:
    for panel in artifact["panels"]:
        gate = panel["gate"]["auroc"]
        assert abs(panel["delta_auroc"] - (gate - panel["incumbent"]["auroc"])) <= _PARITY_TOL
        for variant in panel["incumbent_variants"]:
            assert abs(variant["delta_auroc"] - (gate - variant["auroc"])) <= _PARITY_TOL
        research = panel["research_model"]
        if research is not None:
            assert abs(
                research["delta_auroc_gate_minus_research"] - (gate - research["auroc"])
            ) <= _PARITY_TOL
            assert abs(
                research["delta_auroc_research_minus_incumbent"]
                - (research["auroc"] - panel["incumbent"]["auroc"])
            ) <= _PARITY_TOL
            # The figure cuts the row's advance into these two segments, so they
            # must sum to it exactly or the drawing is not the arithmetic.
            assert abs(
                research["delta_auroc_gate_minus_research"]
                + research["delta_auroc_research_minus_incumbent"]
                - panel["delta_auroc"]
            ) <= _PARITY_TOL
            # And both segments must point the same way, or the figure would draw
            # a bar through a comparator that is not between its two ends.
            assert research["delta_auroc_research_minus_incumbent"] > 0
            assert research["delta_auroc_gate_minus_research"] > 0


def test_panel_counts_and_base_rates_are_internally_consistent(artifact: dict) -> None:
    for panel in artifact["panels"]:
        assert panel["n_correct"] + panel["n_errors"] == panel["n_statements"]
        assert abs(
            panel["base_rate_correct"] - panel["n_correct"] / panel["n_statements"]
        ) <= _PARITY_TOL
        boot = panel["bootstrap"]
        assert boot["ci95_low"] <= boot["ci95_high"]
        assert 0 < boot["n_valid_resamples"] <= boot["n_bootstrap"]


# ---------------------------------------------------------------------------
# the heterogeneity the figure discloses
# ---------------------------------------------------------------------------
def test_every_panel_discloses_its_own_composition(artifact: dict) -> None:
    """The fields that make "not the same comparison four times" checkable."""
    for panel in artifact["panels"]:
        h = panel["heterogeneity"]
        reads = h["evidence_reads_per_statement"]
        assert reads["min"] <= reads["mean"] <= reads["max"]
        assert reads["min"] <= reads["median"] <= reads["max"]
        assert reads["total"] >= panel["n_statements"]
        # The census counts THIS panel, and its mean is that census's own
        # arithmetic rather than a number carried alongside it.
        assert reads["n_statements"] == panel["n_statements"]
        assert abs(reads["mean"] - reads["total"] / reads["n_statements"]) <= _PARITY_TOL
        # The single-evidence share: the disclosure a mean cannot make. Where it
        # is high the gate has nothing to aggregate and its decision is a bare
        # keep-or-drop on one sentence, which is a materially different task from
        # the paper panel's ~20 reads per statement.
        assert 0 <= reads["n_single"] <= reads["n_statements"]
        assert abs(reads["share_single"] - reads["n_single"] / reads["n_statements"]) <= _PARITY_TOL
        if reads["n_single"] > 0:
            assert reads["min"] == 1
        assert abs(h["base_rate_correct"] - panel["base_rate_correct"]) <= _PARITY_TOL
        assert h["join_summary"] in {"exact join", "source-hash join", "mixed join"}
        # Either the corpus census or the stated reason there is none.
        assert (h["corpus_evidence_per_statement"] is None) != (
            h["corpus_evidence_absent_because"] is None
        )
        if h["corpus_evidence_per_statement"] is not None:
            corpus = h["corpus_evidence_per_statement"]
            assert abs(
                h["reader_evidence_share_of_corpus"] - reads["total"] / corpus["total"]
            ) <= _PARITY_TOL
            assert corpus["n_statements"] == panel["n_statements"]
            assert abs(corpus["mean"] - corpus["total"] / corpus["n_statements"]) <= _PARITY_TOL
            assert abs(
                corpus["share_single"] - corpus["n_single"] / corpus["n_statements"]
            ) <= _PARITY_TOL
        assert h["reader_saw_full_evidence"] is (h["reader_evidence_share_of_corpus"] == 1.0)


def test_the_panels_really_are_heterogeneous(artifact: dict) -> None:
    """If they were interchangeable the disclosure would be theatre. They are not."""
    panels = artifact["panels"]
    comparators = {p["incumbent"]["key"] for p in panels}
    assert len(comparators) > 1, "every row would then share one comparator"
    reads = [p["heterogeneity"]["evidence_reads_per_statement"]["mean"] for p in panels]
    assert max(reads) / min(reads) > 5, (
        "the evidence-per-statement spread is what makes these not-identical "
        "comparisons; if it collapses, re-check the censuses"
    )
    # The span is SHIPPED, so the figure can say "16-fold" without an author
    # asserting it, and it must be the ratio of the panels' own extremes.
    rep = artifact["replication"]
    assert abs(rep["reads_per_statement_mean_min"] - min(reads)) <= _PARITY_TOL
    assert abs(rep["reads_per_statement_mean_max"] - max(reads)) <= _PARITY_TOL
    assert abs(rep["evidence_regime_fold_span"] - max(reads) / min(reads)) <= _PARITY_TOL
    # And the single-evidence share genuinely separates the panels: on one the
    # gate aggregates, on another it almost never does.
    singles = [
        p["heterogeneity"]["evidence_reads_per_statement"]["share_single"] for p in panels
    ]
    assert abs(rep["share_single_evidence_min"] - min(singles)) <= _PARITY_TOL
    assert abs(rep["share_single_evidence_max"] - max(singles)) <= _PARITY_TOL
    assert max(singles) - min(singles) > 0.3, (
        "if every panel had the same single-evidence share, the four rows would "
        "be four copies of one comparison after all"
    )
    assert any(p["heterogeneity"]["join_summary"] != "exact join" for p in panels)
    assert any(p["heterogeneity"]["in_sample_note"] is not None for p in panels)


def test_the_claim_the_figure_draws_is_counted_not_composed(artifact: dict) -> None:
    """The headline sentence has to carry the panels' own numbers.

    An earlier draft read "four panels, the same comparison, the gate wins" —
    the sentence the figure cannot carry. The sentence it CAN carry names the
    number of panels, the number of distinct forms of INDRA belief beaten, the
    span of the evidence regime, and the margin the argmax rule gives away. If
    any of those stop appearing, the claim has drifted off the data.
    """
    rep = artifact["replication"]
    claim = artifact["claim"]
    for fragment in (
        str(rep["n_panels"]),
        str(rep["n_incumbent_variants_total"]),
        f"{rep['evidence_regime_fold_span']:.0f}-fold",
        f"{rep['selection_cost_auroc_max']:.4f}",
    ):
        assert fragment in claim, f"the claim no longer carries {fragment!r}"
    # The anti-claim must keep refusing the reading that made the old figure
    # indefensible, and must name the comparator split it is refusing.
    anti = artifact["claim_is_not"]
    assert "not the same comparison" in anti.lower()
    assert "single-evidence" in anti.lower()
    # Every caveat is a filled sentence, never an unconsumed slot.
    for caveat in artifact["caveats"]:
        assert caveat.upper() != caveat, f"caveat slot never filled: {caveat!r}"


def test_the_evidence_matched_control_is_never_a_comparator(artifact: dict) -> None:
    """It is below chance on the curation panels; treating it as one would flatter us."""
    for panel in artifact["panels"]:
        control = panel["evidence_matched_control"]
        # Either the control or the stated reason it is absent — never silence.
        assert (control is None) != (panel["evidence_matched_control_absent_because"] is None)
        if control is None:
            continue
        assert control["is_an_incumbent"] is False
        assert control["key"] not in {v["key"] for v in panel["incumbent_variants"]}
        assert abs(
            control["delta_auroc_incumbent_minus_control"]
            - (panel["incumbent"]["auroc"] - control["auroc"])
        ) <= _PARITY_TOL
        assert control["at_or_below_chance"] is (control["auroc"] <= 0.5)
        # And the asymmetry it prices runs AGAINST the gate: the comparator is
        # scored over evidence the reader never saw, and that extra evidence is
        # where the comparator's signal is.
        assert control["auroc"] < panel["incumbent"]["auroc"], (
            f"{panel['key']}: restricting INDRA's scorer to the reader's evidence "
            "does not weaken it, so the scope asymmetry is not conservative"
        )


def test_paper_panel_evidence_census_is_recounted_from_the_execution_map(
    artifact: dict,
) -> None:
    """19.75 reads per statement is a census of the frozen map, not a manifest claim."""
    counts: Counter = Counter()
    with _PAPER_EXECUTION_MAP.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counts[str(json.loads(line)["paper_statement_hash"])] += 1
    values = sorted(counts.values())
    panel = _panel(artifact, _PAPER_PANEL_KEY)
    reads = panel["heterogeneity"]["evidence_reads_per_statement"]
    assert len(counts) == panel["n_statements"]
    assert reads["total"] == sum(values)
    assert abs(reads["mean"] - sum(values) / len(values)) <= _PARITY_TOL
    assert reads["max"] == values[-1]
    assert reads["min"] == values[0]
    # And on this panel the reader read the statement's whole evidence set, which
    # is why it carries no evidence-matched control.
    assert panel["heterogeneity"]["reader_saw_full_evidence"] is True
    assert panel["evidence_matched_control"] is None
    assert panel["evidence_matched_control_absent_because"]


# ---------------------------------------------------------------------------
# the paper panel, re-derived end to end
# ---------------------------------------------------------------------------
def test_paper_panel_levels_are_rederived_from_the_named_files(
    artifact: dict, paper_panel: tuple[list[str], np.ndarray]
) -> None:
    """Every level on the headline panel, recomputed with a different estimator."""
    sids, y = paper_panel
    panel = _panel(artifact, _PAPER_PANEL_KEY)
    assert panel["n_statements"] == len(sids)
    assert panel["n_correct"] == int(y.sum())

    gate = _scores(_PAPER_GATE_SCORES, sids)
    assert abs(panel["gate"]["auroc"] - float(roc_auc_score(y, gate))) <= _PARITY_TOL

    for variant in panel["incumbent_variants"]:
        scores = _scores(ROOT / variant["source"], sids)
        assert abs(variant["auroc"] - float(roc_auc_score(y, scores))) <= _PARITY_TOL, (
            f"{variant['key']} disagrees with {variant['source']}"
        )

    literal = _load(_PAPER_LITERAL)
    research = panel["research_model"]
    oof = {r["stmt_hash"]: r for r in literal["oof_predictions"][research["key_in_source"]]}
    sid_by_hash = {
        int(r["paper_statement_hash"]): r["canonical_corpus"]["statement_id"]
        for r in _load_jsonl(_PAPER_GOLD)
    }
    rf_by_sid = {sid_by_hash[h]: oof[h]["prob_correct"] for h in oof}
    rf = np.array([rf_by_sid[s] for s in sids], dtype=float)
    assert abs(research["auroc"] - float(roc_auc_score(y, rf))) <= _PARITY_TOL


def test_paper_panel_gate_never_exceeds_the_deployed_scorer(
    artifact: dict, paper_panel: tuple[list[str], np.ndarray]
) -> None:
    """The reader is SUBTRACTIVE, so it cannot promote a statement.

    Re-derived rather than read off the artifact, because this single property is
    what licenses calling the gate and the library default the same INDRA scorer.
    If the gate ever exceeded the ungated noisy-OR on any statement, the two arms
    would differ in their aggregation and the comparison would not isolate
    reading.
    """
    sids, _ = paper_panel
    panel = _panel(artifact, _PAPER_PANEL_KEY)
    direct = [v for v in panel["incumbent_variants"] if v["key"] == "simple_scorer_direct"]
    assert len(direct) == 1, "the paper panel must carry the direct SimpleScorer variant"
    assert direct[0]["family"] == _FAMILY_LIBRARY
    ungated = _scores(ROOT / direct[0]["source"], sids)
    gate = _scores(_PAPER_GATE_SCORES, sids)
    exceeding = int(np.sum(gate > ungated + 1e-12))
    assert exceeding == 0, f"the gate exceeds the ungated noisy-OR on {exceeding} statements"


def test_paper_panel_agrees_with_the_shipped_head_to_head(artifact: dict) -> None:
    """The gate, the paper's RF and the CoGEx hybrid already ship values.

    The CoGEx entry matters most: it is the paper panel's DRAWN comparator now,
    and it is the same artifact `paper_literal_vs_llms.json` already carries as
    "INDRA CoGEx hybrid". If the two ever disagree, one of the two pages is
    quoting a number the other cannot reproduce.
    """
    shipped = _load(_PAPER_VS_LLMS)["point_metrics"]
    panel = _panel(artifact, _PAPER_PANEL_KEY)
    assert abs(panel["gate"]["auroc"] - shipped["Gemma 4 26B"]["auroc"]) <= _PARITY_TOL
    assert abs(
        panel["research_model"]["auroc"] - shipped["Paper literal RF+promoter"]["auroc"]
    ) <= _PARITY_TOL
    assert abs(
        panel["research_model"]["average_precision"]
        - shipped["Paper literal RF+promoter"]["pooled_average_precision"]
    ) <= _PARITY_TOL

    cogex = [v for v in panel["incumbent_variants"] if v["key"] == "cogex_fitted_hybrid"]
    assert len(cogex) == 1, "the paper panel must carry INDRA's stored production belief"
    assert cogex[0]["family"] == _FAMILY_SERVED
    assert abs(cogex[0]["auroc"] - shipped["INDRA CoGEx hybrid"]["auroc"]) <= _PARITY_TOL
    assert abs(
        cogex[0]["average_precision"] - shipped["INDRA CoGEx hybrid"]["pooled_average_precision"]
    ) <= _PARITY_TOL
    # It is a recovered replay, not a live capture, and that has to travel with it.
    assert cogex[0].get("provenance_caveat"), (
        "the CoGEx replay must carry its own admissibility caveat"
    )
    assert cogex[0].get("analysis_role") == "descriptive_nonconfirmatory"
    # And it must be the drawn comparator: it is the strongest form here, so
    # anything else would be the convenient baseline.
    assert panel["incumbent"]["key"] == "cogex_fitted_hybrid"


# ---------------------------------------------------------------------------
# curation panels: independent cross-check against already-shipped siblings
# ---------------------------------------------------------------------------
# (panel key, sibling artifact, {artifact level -> sibling belief_discrimination key})
# `belief_stored` is the belief INDRA served; `belief_llm` is the production hard
# gate at the recalibrated priors, which this artifact carries as a sensitivity
# rather than drawing. `belief_indra` is NOT matched exactly — see
# `test_the_recomputed_library_default_pins_its_divergence_from_the_sibling`.
_SIBLING_CROSS_CHECKS = [
    (
        "eval_curation_v1",
        "data/results/belief_headtohead_gemma.json",
        {"indra_served_belief": "belief_stored"},
        "belief_llm",
    ),
    (
        "external_curator_v1",
        "data/results/belief_headtohead_external_gemma.json",
        {"indra_served_belief": "belief_stored"},
        "belief_llm",
    ),
]


@pytest.mark.parametrize(
    "panel_key,sibling_path,variant_map,gate_sensitivity_key", _SIBLING_CROSS_CHECKS
)
def test_curation_levels_agree_with_shipped_siblings(
    artifact: dict,
    panel_key: str,
    sibling_path: str,
    variant_map: dict,
    gate_sensitivity_key: str,
) -> None:
    """Every level with a shipped sibling must equal it, to the sibling's precision."""
    sibling = _load(ROOT / sibling_path)["belief_discrimination"]
    panel = _panel(artifact, panel_key)
    assert panel["n_statements"] == sibling[gate_sensitivity_key]["all"]["n"]

    by_key = {v["key"]: v for v in panel["incumbent_variants"]}
    for variant_key, sibling_key in variant_map.items():
        assert variant_key in by_key, f"{panel_key} no longer carries {variant_key}"
        assert abs(by_key[variant_key]["auroc"] - sibling[sibling_key]["all"]["auroc"]) <= _SIBLING_TOL

    sensitivity = panel["gate_sensitivity"]
    assert sensitivity is not None, f"{panel_key} must carry the production-gate sensitivity"
    assert abs(
        sensitivity["auroc"] - sibling[gate_sensitivity_key]["all"]["auroc"]
    ) <= _SIBLING_TOL


def test_the_recomputed_library_default_pins_its_divergence_from_the_sibling(
    artifact: dict,
) -> None:
    """A DELIBERATE divergence, pinned on both ends rather than smoothed over.

    ``belief_headtohead_gemma.json`` scores the same statements with
    ``noise_model.INDRA_PRIORS`` — an 18-source transcription of INDRA's priors
    with a (0.30, 0.10) fallback. That module is byte-frozen under the reader
    bundle's implementation digest, so this artifact reads
    ``indra/resources/default_belief_probs.json`` itself instead, which is what
    makes the variant literally "INDRA's bundled default priors". The two land
    0.0009 apart. Both numbers are pinned here, and the drawn one must be the
    STRONGER — otherwise reading INDRA's own resource would have handed us an
    easier comparator, which is the argmax rule running backwards.
    """
    panel = _panel(artifact, "eval_curation_v1")
    variant = [v for v in panel["incumbent_variants"] if v["key"] == "simple_scorer_recomputed"]
    assert len(variant) == 1
    cross = variant[0].get("cross_check")
    assert cross is not None, "the divergence must be recorded, not discovered later"

    sibling = _load(ROOT / cross["sibling"])["belief_discrimination"]["belief_indra"]["all"]
    assert abs(cross["sibling_auroc"] - sibling["auroc"]) <= _PARITY_TOL, (
        "the recorded sibling value is not the value the sibling ships"
    )
    assert cross["sibling_n"] == sibling["n"] == panel["n_statements"], (
        "the sibling scores a different panel, so the two numbers are not comparable"
    )
    assert abs(cross["this_auroc"] - variant[0]["auroc"]) <= _PARITY_TOL
    assert abs(
        cross["delta_vs_sibling"] - (cross["this_auroc"] - cross["sibling_auroc"])
    ) <= _PARITY_TOL
    assert cross["delta_vs_sibling"] >= 0, (
        "reading INDRA's own prior resource produced a WEAKER comparator than the "
        "shipped sibling; the strongest-incumbent rule requires the stronger form"
    )
    assert cross["n_statements_scored_differently"] > 0


def test_holdout_cc_has_no_shipped_sibling_to_cross_check(artifact: dict) -> None:
    """Recorded, not hidden: this panel's older head-to-head is unreproducible.

    ``data/results/belief_headtohead_holdout_gemma.json`` reports n=393 against a
    gold at ``data/benchmark/holdout_cc.jsonl`` that no longer exists in the tree.
    This artifact re-joins the SAME scored run against the gold the calibration
    ship gate uses (``data/results/cc_holdout_cc/holdout_cc.jsonl``, whose digest
    the ship gate also records) and lands on n=414. The panel is therefore sourced
    from files that exist, and the stale sibling is deliberately NOT used as a
    cross-check. This test exists so that fact stays visible if the stale file is
    ever mistaken for a reference.
    """
    stale = ROOT / "data/results/belief_headtohead_holdout_gemma.json"
    panel = _panel(artifact, "holdout_cc")
    assert Path(ROOT / panel["provenance"]["gold"]).exists()
    if stale.exists():
        stale_gold = ROOT / _load(stale)["gold_source"]
        assert not stale_gold.exists(), (
            "the stale holdout head-to-head's gold has reappeared; reconcile n=393 "
            "against this artifact's n=414 before treating either as a reference"
        )
    ship_gate = _load(ROOT / "data/results/calibration_ship_gate.json")
    gemma = [row for row in ship_gate if row["name"] == "gemma-26B"]
    assert len(gemma) == 1
    assert gemma[0]["provenance"]["test_gold_sha256"] == panel["provenance"]["gold_sha256"], (
        "this panel's gold is no longer the gold the calibration ship gate tests against"
    )


# ---------------------------------------------------------------------------
# the TypeScript end of the same parity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_typescript_contract_runner_passes() -> None:
    """The .mjs asserts the same properties on the data the viewer imports."""
    result = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        cwd=TS_RUNNER.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
