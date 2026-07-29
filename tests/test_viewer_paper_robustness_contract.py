"""Gate the margin-robustness artifact the /paper robustness figure draws.

The figure makes two claims that are easy to state and easy to get subtly wrong,
so both are re-derived here rather than read off the artifact:

  * the PRIMARY result is the 1689-statement panel with the paper's own released
    labels, unmodified, and the artifact's pointwise half must be BYTE-IDENTICAL
    to the shipped head-to-head (``paper_literal_vs_llms.json``). If the two
    disagreed, the simultaneous band on this figure would be widening an interval
    the rest of the page reports differently;

  * the SENSITIVITY panel is OUR revision of the paper's labels. Its census — 111
    dropped statements, every one a negative whose evidence review is incomplete,
    a quarter of all negatives, class balance 26.8% -> 21.6% — is recomputed from
    ``paper_statement_gold.jsonl`` here, and every point delta on both panels is
    recomputed with ``sklearn.average_precision_score`` from the prediction files.

What is NOT re-run here is the 10,000-resample paired bootstrap: the compute
script owns that and asserts its own reproduction of the shipped numbers, the same
precedent the execution map and the reachable-set enumeration already set. What IS
checked is that the simultaneous band is exactly ``delta +/- critical_value * se``
over those draws, that the critical value sits strictly between the pointwise
normal quantile and Bonferroni (both re-derived from ``statistics.NormalDist``),
and that the frozen run plan the multiplicity argument rests on is the file on
disk.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
TS_RUNNER = ROOT / "viewer" / "scripts" / "test-paper-robustness-contract.mjs"

_MODEL_DIR = ROOT / "data" / "results" / "indra_paper_literal_models_20260724"
_ROBUSTNESS_PATH = _MODEL_DIR / "paper_margin_robustness.json"
_VS_LLMS_PATH = _MODEL_DIR / "paper_literal_vs_llms.json"
_LITERAL_PATH = _MODEL_DIR / "paper_literal_table6_and_oof.json"
_RUN_MANIFEST_PATH = _MODEL_DIR / "manifest.json"
_GOLD_PATH = (
    ROOT / "data" / "results" / "indra_paper_statement_gold_20260717" / "paper_statement_gold.jsonl"
)
_MODELS_DIR = ROOT / "data" / "comparison" / "models"
_RUN_PLAN_PATH = ROOT / "data" / "comparison" / "run_plan.json"

# The paper's own headline out-of-fold vector — the paired-delta reference.
_HEADLINE = "RF 2k-d13 + Type/#PMIDs/promoter - all sources, specific"

# arm id -> (frozen point_metrics join key, prediction bundle directory).
_ARMS = {
    "gemma-4-e2b": ("Gemma 4 E2B", "gemma_4_e2b"),
    "gemma-4-26b": ("Gemma 4 26B", "gemma_4_26b"),
    "gemma-4-31b": ("Gemma 4 31B", "gemma_4_31b"),
    "glm-5": ("GLM-5", "glm_5"),
}

# Deltas are differences of doubles at ~0.95, so an independent recomputation of
# the same quantity lands within a few ulps; 1e-12 is decisive without being
# brittle across BLAS builds.
_TOL = 1e-12


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _panel() -> tuple[list[str], np.ndarray, np.ndarray]:
    """The 1689 statements, their released labels, and the adjudication-safe mask.

    Re-derived from the gold and the released out-of-fold vector rather than
    imported, because the artifact under test is a claim ABOUT that derivation.
    Ordering is sorted(paper_statement_hash), matching the head-to-head.
    """
    oof = {row["stmt_hash"]: row for row in _load(_LITERAL_PATH)["oof_predictions"][_HEADLINE]}
    gold = {}
    for row in _jsonl(_GOLD_PATH):
        policy = row["paper_replication_policy"]
        gold[int(row["paper_statement_hash"])] = (
            row["canonical_corpus"]["statement_id"],
            int(policy["released_paper_correct"]),
            bool(policy["label_is_adjudication_safe"]),
        )
    hashes = sorted(oof)
    sids = [gold[h][0] for h in hashes]
    labels = np.array([gold[h][1] for h in hashes])
    safe = np.array([gold[h][2] for h in hashes])
    assert [int(oof[h]["y_true"]) for h in hashes] == list(labels), "released label drift"
    return sids, labels, safe


def _reference_scores() -> np.ndarray:
    oof = {row["stmt_hash"]: row for row in _load(_LITERAL_PATH)["oof_predictions"][_HEADLINE]}
    return np.array([oof[h]["prob_correct"] for h in sorted(oof)])


def _arm_scores(bundle: str, sids: list[str]) -> np.ndarray:
    scores = {
        row["statement_id"]: row["probability_correct"]
        for row in _jsonl(_MODELS_DIR / bundle / "all_source_predictions.jsonl")
    }
    return np.array([scores[sid] for sid in sids])


def _arm_by_id(artifact: dict) -> dict[str, dict]:
    return {arm["id"]: arm for arm in artifact["arms"]}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_viewer_paper_robustness_contract() -> None:
    """The TS end: the validator gates on every framing and geometry invariant."""
    completed = subprocess.run(
        ["node", "--experimental-strip-types", str(TS_RUNNER)],
        cwd=ROOT / "viewer",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "viewer paper-robustness contract assertions failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_robustness_is_the_artifact_the_manifest_records() -> None:
    """The viewer draws the exact bytes the run manifest signed."""
    manifest = _load(_RUN_MANIFEST_PATH)
    assert manifest["outputs"]["margin_robustness"] == _ROBUSTNESS_PATH.name
    recorded = manifest["output_sha256"][_ROBUSTNESS_PATH.name]
    assert hashlib.sha256(_ROBUSTNESS_PATH.read_bytes()).hexdigest() == recorded
    # The outputs this artifact was added beside are untouched.
    for name in (
        "ap_decomposition_by_paper_band.json",
        "statement_review_queue.json",
        "belief_model_ladder.json",
        "non_reading_control.json",
        "framing_correction.json",
    ):
        assert hashlib.sha256((_MODEL_DIR / name).read_bytes()).hexdigest() == (
            manifest["output_sha256"][name]
        ), name


def test_panel_census_is_the_golds_own_adjudication_safety_field() -> None:
    """The 1578 panel is 1689 minus 111 label-incomplete NEGATIVES — recomputed."""
    artifact = _load(_ROBUSTNESS_PATH)
    _sids, labels, safe = _panel()

    primary = artifact["panels"]["primary"]
    sensitivity = artifact["panels"]["sensitivity"]
    completeness = artifact["label_completeness"]

    assert primary["n_statements"] == len(labels) == 1689
    assert primary["n_positive"] == int((labels == 1).sum())
    assert primary["n_negative"] == int((labels == 0).sum())
    assert sensitivity["n_statements"] == int(safe.sum())
    assert sensitivity["n_positive"] == int((labels[safe] == 1).sum())
    assert sensitivity["n_negative"] == int((labels[safe] == 0).sum())

    dropped = ~safe
    assert completeness["n_dropped"] == int(dropped.sum())
    # Every dropped statement is a NEGATIVE in the paper's released labels: that is
    # what makes removing them our revision rather than a repair.
    assert (labels[dropped] == 0).all()
    assert completeness["all_dropped_are_negative"] is True
    # ...and every one of them has an incomplete evidence review, per the gold's
    # own policy field, not per this artifact.
    unsafe_rows = [
        row
        for row in _jsonl(_GOLD_PATH)
        if not row["paper_replication_policy"]["label_is_adjudication_safe"]
    ]
    assert len(unsafe_rows) == completeness["n_dropped"]
    assert all(
        row["paper_replication_policy"]["negative_has_complete_evidence"] is False
        for row in unsafe_rows
    )
    assert completeness["all_dropped_have_incomplete_evidence_review"] is True

    # The check is not free: it removes a quarter of all negatives and moves the
    # class balance, which the artifact must say in numbers.
    assert completeness["dropped_share_of_all_negatives"] == pytest.approx(
        int(dropped.sum()) / int((labels == 0).sum()), abs=_TOL
    )
    assert completeness["negative_fraction_before"] == pytest.approx(
        int((labels == 0).sum()) / len(labels), abs=_TOL
    )
    assert completeness["negative_fraction_after"] == pytest.approx(
        int((labels[safe] == 0).sum()) / int(safe.sum()), abs=_TOL
    )
    assert completeness["negative_fraction_after"] < completeness["negative_fraction_before"]

    # Framing, gated on both sides: the paper-label panel is never our revision,
    # and our revision is never presented as the paper's data.
    assert primary["is_our_label_revision"] is False
    assert sensitivity["is_our_label_revision"] is True
    assert primary["role"] == "primary" and sensitivity["role"] == "sensitivity"


def test_point_deltas_rederive_on_both_panels() -> None:
    """Every ΔAP the figure draws, recomputed from the prediction files."""
    artifact = _load(_ROBUSTNESS_PATH)
    sids, labels, safe = _panel()
    reference = _reference_scores()
    arms = _arm_by_id(artifact)

    for panel_key, mask in (("primary", np.ones(len(labels), dtype=bool)), ("sensitivity", safe)):
        y = labels[mask]
        reference_ap = float(average_precision_score(y, reference[mask]))
        assert artifact["panels"][panel_key]["reference_average_precision"] == pytest.approx(
            reference_ap, abs=_TOL
        ), panel_key

        for arm_id, (label, bundle) in _ARMS.items():
            scores = _arm_scores(bundle, sids)[mask]
            arm_ap = float(average_precision_score(y, scores))
            side = arms[arm_id][panel_key]
            assert side["average_precision"] == pytest.approx(arm_ap, abs=_TOL), (arm_id, panel_key)
            assert side["delta"] == pytest.approx(arm_ap - reference_ap, abs=_TOL), (
                arm_id,
                panel_key,
            )
            # The frozen point_metrics join key travels with the arm and is never
            # the on-screen name (the viewer owns display; see paper-robustness.ts).
            assert arms[arm_id]["label"] == label


def test_pointwise_half_is_identical_to_the_shipped_head_to_head() -> None:
    """The primary interval here IS the shipped one — not a second estimate of it."""
    artifact = _load(_ROBUSTNESS_PATH)
    shipped = _load(_VS_LLMS_PATH)
    arms = _arm_by_id(artifact)

    assert artifact["seed"] == shipped["seed"]
    assert artifact["n_bootstrap"] == shipped["n_bootstrap"]
    assert artifact["panels"]["primary"]["n_statements"] == shipped["n_statements"]
    assert artifact["panels"]["primary"]["reference_average_precision"] == (
        shipped["point_metrics"]["Paper literal RF+promoter"]["pooled_average_precision"]
    )

    for arm_id, (label, _bundle) in _ARMS.items():
        want = shipped["paired_delta_vs_paper_literal"][label]["pooled_average_precision"]
        got = arms[arm_id]["primary"]
        assert got["delta_bootstrap_mean"] == want["delta"], arm_id
        assert got["ci95_low"] == want["ci95_low"], arm_id
        assert got["ci95_high"] == want["ci95_high"], arm_id
        assert got["p_greater_than_zero"] == want["p_arm_greater"], arm_id
        assert got["n_valid_resamples"] == want["n_valid_resamples"], arm_id
        assert got["average_precision"] == (
            shipped["point_metrics"][label]["pooled_average_precision"]
        ), arm_id

    # And the artifact's own reconciliation says the same thing.
    reconciliation = artifact["shipped_reconciliation"]
    assert reconciliation["worst_residual_vs_shipped"] <= reconciliation["tolerance"]


def test_simultaneous_band_is_the_critical_value_times_the_standard_error() -> None:
    """The band is an identity on shipped scalars, so it is checked as one."""
    artifact = _load(_ROBUSTNESS_PATH)
    multiplicity = artifact["multiplicity"]

    for panel_key, critical in (
        ("primary", multiplicity["critical_value"]),
        ("sensitivity", multiplicity["sensitivity_critical_value"]),
    ):
        for arm in artifact["arms"]:
            side = arm[panel_key]
            assert side["simultaneous_low"] == pytest.approx(
                side["delta"] - critical * side["bootstrap_se"], abs=_TOL
            ), (arm["id"], panel_key)
            assert side["simultaneous_high"] == pytest.approx(
                side["delta"] + critical * side["bootstrap_se"], abs=_TOL
            ), (arm["id"], panel_key)
            # The drawing nests the pointwise interval inside the band; if that ever
            # stopped holding, the figure would draw a correction that shrinks.
            assert side["simultaneous_low"] <= side["ci95_low"], (arm["id"], panel_key)
            assert side["simultaneous_high"] >= side["ci95_high"], (arm["id"], panel_key)
            assert side["excludes_zero_pointwise"] == (
                side["ci95_low"] > 0 or side["ci95_high"] < 0
            ), (arm["id"], panel_key)
            assert side["excludes_zero_simultaneous"] == (
                side["simultaneous_low"] > 0 or side["simultaneous_high"] < 0
            ), (arm["id"], panel_key)


def test_critical_value_sits_between_pointwise_and_bonferroni() -> None:
    """max-t is milder than Bonferroni because the arms move together."""
    artifact = _load(_ROBUSTNESS_PATH)
    multiplicity = artifact["multiplicity"]
    k = multiplicity["family_size"]
    alpha = multiplicity["family_alpha"]
    assert k == len(artifact["arms"]) == 4

    normal = statistics.NormalDist()
    assert multiplicity["pointwise_normal_critical_value"] == pytest.approx(
        normal.inv_cdf(1 - alpha / 2), abs=1e-12
    )
    assert multiplicity["bonferroni_critical_value"] == pytest.approx(
        normal.inv_cdf(1 - alpha / (2 * k)), abs=1e-12
    )
    assert (
        multiplicity["pointwise_normal_critical_value"]
        < multiplicity["critical_value"]
        <= multiplicity["bonferroni_critical_value"]
    )
    # The correlation that buys the mildness is reported, and it is high.
    assert 0 < multiplicity["score_spearman_min"] <= multiplicity["score_spearman_max"] <= 1
    assert multiplicity["score_spearman_min"] > 0.5
    assert len(multiplicity["pairwise_correlation"]) == k * (k - 1) // 2


def test_run_plan_provenance_is_the_frozen_file_on_disk() -> None:
    """The multiplicity argument rests on the plan, so the plan is pinned."""
    artifact = _load(_ROBUSTNESS_PATH)
    recorded = artifact["multiplicity"]["run_plan"]
    assert recorded["path"] == "data/comparison/run_plan.json"
    assert recorded["sha256"] == hashlib.sha256(_RUN_PLAN_PATH.read_bytes()).hexdigest()

    plan = _load(_RUN_PLAN_PATH)
    assert recorded["frozen_at"] == plan["amendment"]["frozen_at"]
    assert recorded["stages"] == [stage["id"] for stage in plan["stages"]]
    # Every reader arm on the figure is a staged arm, and the plan singles none out.
    staged = {stage["id"] for stage in plan["stages"]}
    assert {arm["run_plan_stage"] for arm in artifact["arms"]} == staged
    assert artifact["multiplicity"]["no_designated_primary_arm"] is True
    designating = [
        key
        for stage in plan["stages"]
        for key in stage
        if any(token in key.lower() for token in ("primary", "confirmatory", "preregister"))
    ]
    assert designating == []


def test_dose_response_sign_pattern_is_recomputed_not_asserted() -> None:
    """Exactly one arm is negative, and it is the smallest one."""
    artifact = _load(_ROBUSTNESS_PATH)
    dose = artifact["dose_response"]
    negative = [arm for arm in artifact["arms"] if arm["primary"]["delta"] < 0]
    positive = [arm for arm in artifact["arms"] if arm["primary"]["delta"] > 0]

    assert len(negative) == dose["n_arms_with_negative_delta"] == 1
    assert len(positive) == dose["n_arms_with_positive_delta"]
    assert negative[0]["id"] == dose["smallest_arm_id"] == "gemma-4-e2b"
    assert dose["only_negative_arm_is_the_smallest"] is True
    # The negative result is the robust one: it excludes zero on both views AND on
    # the label-completeness panel. Nothing on this figure is more robust.
    assert negative[0]["primary"]["excludes_zero_pointwise"] is True
    assert negative[0]["primary"]["excludes_zero_simultaneous"] is True
    assert negative[0]["sensitivity"]["excludes_zero_pointwise"] is True


def test_presentation_order_puts_the_best_arm_first() -> None:
    """The page's headline sentence names arms[0]; the order is therefore a contract."""
    artifact = _load(_ROBUSTNESS_PATH)
    deltas = [arm["primary"]["delta"] for arm in artifact["arms"]]
    assert deltas == sorted(deltas, reverse=True)
    assert artifact["arms"][0]["id"] == "gemma-4-26b"
    assert artifact["arms"][0]["primary"]["excludes_zero_pointwise"] is True


def test_power_half_widths_are_the_order_of_the_effect() -> None:
    """The stated power limitation is arithmetic, not a hedge."""
    artifact = _load(_ROBUSTNESS_PATH)
    half_widths = artifact["power"]["pointwise_ci_half_width"]
    best = max(arm["primary"]["delta"] for arm in artifact["arms"])

    for arm in artifact["arms"]:
        side = arm["primary"]
        assert half_widths[arm["id"]] == pytest.approx(
            (side["ci95_high"] - side["ci95_low"]) / 2, abs=_TOL
        ), arm["id"]
    worst = max(half_widths.values())
    # The half-width and the effect are the same size — the borderline claim.
    assert math.isclose(worst, best, rel_tol=0.5)
