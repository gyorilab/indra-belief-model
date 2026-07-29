from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data/results/current_indra_counts_hybrid_paper_20260718"
GOLD_PATH = (
    ROOT
    / "data/results/indra_paper_comparison_gold_20260717/paper_released_gold.jsonl"
)
sys.path.insert(0, str(ROOT / "scripts"))

import score_current_indra_counts_hybrid_paper as adapter  # noqa: E402


EXPECTED_OUTPUT_SHA256 = {
    "current_counts_full_features_oof_predictions.jsonl": (
        "fd4bb65bdd69e7981778bf88caa3acfa59db62c36ae007deadb1c9c37ad52bc2"
    ),
    "current_counts_hybrid_diagnostic_fold_metrics.jsonl": (
        "5b30c8ec2f7d770d65f4ca9f09104571ad1e0b43cde0ea947c006a6ec37711c9"
    ),
    "current_counts_hybrid_exclusions.jsonl": (
        "9e78e7320ada7e213f1919c59cb4d319a2f512c7f921550a66ae07f3143a71a5"
    ),
    "current_counts_hybrid_prediction_provenance.jsonl": (
        "d5bce4ecd44c94c5cb5171b68a4b5a889c8409e125edb5588dd46a7bfe793af8"
    ),
    "current_counts_oof_fit_provenance.jsonl": (
        "00ffa16097ce16bfa306c30263466027a7396958c2fd8e69b79b454203f339f0"
    ),
    "current_counts_source_only_oof_predictions.jsonl": (
        "71e05c859dea8b9a467182766d657d5291f054c7e37ba11ffa7111286a5fd4b4"
    ),
    "current_hybrid_local_full_features_oof_predictions.jsonl": (
        "71e2118637bb0f774c40e12a39728c9837d05be8a8acc68a54b5910430d75fe1"
    ),
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_counts_hybrid_bundle_is_digest_pinned_and_complete() -> None:
    manifest = json.loads((ARTIFACT / adapter.MANIFEST_FILENAME).read_text())
    assert manifest["artifact_kind"] == adapter.ARTIFACT_KIND
    assert manifest["implementation"]["adapter"]["sha256"] == _sha(
        ROOT / "scripts/score_current_indra_counts_hybrid_paper.py"
    )
    assert set(manifest["outputs"]) == set(EXPECTED_OUTPUT_SHA256)
    for filename, expected_sha in EXPECTED_OUTPUT_SHA256.items():
        path = ARTIFACT / filename
        descriptor = manifest["outputs"][filename]
        rows = _rows(path)
        assert _sha(path) == expected_sha == descriptor["sha256"]
        assert len(rows) == descriptor["rows"]

    gold = _rows(GOLD_PATH)
    expected_ids = [row["statement_id"] for row in gold]
    assert len(expected_ids) == 1689 == len(set(expected_ids))
    for filename in adapter.PREDICTION_FILENAMES.values():
        rows = _rows(ARTIFACT / filename)
        assert [row["statement_id"] for row in rows] == expected_ids
        assert all(
            set(row) == {"probability_correct", "statement_id"} for row in rows
        )
        assert all(
            math.isfinite(row["probability_correct"])
            and 0.0 <= row["probability_correct"] <= 1.0
            for row in rows
        )

    assert manifest["coverage"] == {
        "direct_evidence_entries": 34035,
        "graph_edges": 637573,
        "graph_nodes": 895459,
        "invalid_predictions": 0,
        "missing_predictions": 0,
        "more_specific_unique_nonnegated_evidence_entries": 11466,
        "predictions_per_arm": 1689,
        "statements_with_more_specific_evidence": 491,
        "target_statements": 1689,
    }


def test_fit_provenance_proves_frozen_oof_partitions_and_fitted_state() -> None:
    gold = _rows(GOLD_PATH)
    fits = _rows(ARTIFACT / adapter.FIT_FILENAME)
    assert len(fits) == 20
    by_key = {(row["arm_id"], row["fit_fold_id"]): row for row in fits}
    assert set(by_key) == {
        (arm_id, fold_id)
        for arm_id in (adapter.ARM_COUNTS_SOURCE, adapter.ARM_COUNTS_FULL)
        for fold_id in range(10)
    }
    assert len({row["fitted_state_sha256"] for row in fits}) == 20

    for arm_id, columns in (
        (adapter.ARM_COUNTS_SOURCE, 11),
        (adapter.ARM_COUNTS_FULL, 77),
    ):
        for fold_id in range(10):
            fit = by_key[(arm_id, fold_id)]
            train = [row for row in gold if row["fold_id"] != fold_id]
            test = [row for row in gold if row["fold_id"] == fold_id]
            assert fit["train_statement_id_sha256"] == adapter._ordered_id_sha(train)
            assert fit["test_statement_id_sha256"] == adapter._ordered_id_sha(test)
            assert fit["train_label_sha256"] == adapter._label_sha(train)
            assert fit["train_statement_count"] + fit["test_statement_count"] == 1689
            assert fit["train_feature_shape"] == [len(train), columns]
            assert fit["test_feature_shape"] == [len(test), columns]
            assert fit["train_test_statement_intersection"] == 0
            assert fit["test_label_count_not_passed_to_fit"] == len(test)
            assert fit["estimator_parameters"]["n_estimators"] == 2000
            assert fit["estimator_parameters"]["max_depth"] == 13
            assert fit["estimator_parameters"]["random_state"] == 4
            assert fit["estimator_parameters"]["n_jobs"] == 1
            assert len(fit["fitted_state_sha256"]) == 64
            assert fit["fitted_state_pickle_bytes"] > 0


def test_local_hybrid_is_verified_redundant_and_not_the_production_point() -> None:
    counts = _rows(
        ARTIFACT / adapter.PREDICTION_FILENAMES[adapter.ARM_COUNTS_FULL]
    )
    hybrid = _rows(
        ARTIFACT / adapter.PREDICTION_FILENAMES[adapter.ARM_HYBRID_FULL]
    )
    differences = [
        abs(left["probability_correct"] - right["probability_correct"])
        for left, right in zip(counts, hybrid, strict=True)
    ]
    manifest = json.loads((ARTIFACT / adapter.MANIFEST_FILENAME).read_text())
    audit = manifest["current_class_semantics"]["hybrid_audit"]
    assert max(differences) == audit["max_abs_hybrid_minus_counts_probability"]
    assert max(differences) == np.finfo(float).eps / 4
    assert audit["hybrid_simple_fallback_evidence_entries"] == 0
    assert manifest["exclusion_contract"]["production_point_materialized"] is False

    provenance = _rows(ARTIFACT / adapter.PROVENANCE_FILENAME)
    by_arm: defaultdict[str, list[dict]] = defaultdict(list)
    for row in provenance:
        by_arm[row["arm_id"]].append(row)
    assert set(by_arm) == set(adapter.PREDICTION_FILENAMES)
    assert all(len(rows) == 1689 for rows in by_arm.values())
    assert all(
        row["simple_fallback_evidence_count"] == 0
        and row["simple_fallback_source_counts"] == {}
        for row in by_arm[adapter.ARM_HYBRID_FULL]
    )

    exclusions = _rows(ARTIFACT / adapter.EXCLUSIONS_FILENAME)
    assert {row["excluded_scorer_id"] for row in exclusions} == {
        "indra_1.24.0_counts_unfitted",
    }


def test_diagnostic_values_are_named_and_not_promoted_to_formal_comparison() -> None:
    manifest = json.loads((ARTIFACT / adapter.MANIFEST_FILENAME).read_text())
    metrics = manifest["diagnostic_metrics"]
    assert metrics[adapter.ARM_COUNTS_SOURCE][
        "fold_mean_trapezoidal_pr_auc"
    ] == 0.9048957891330721
    assert metrics[adapter.ARM_COUNTS_FULL][
        "fold_mean_trapezoidal_pr_auc"
    ] == 0.9416598398747833
    assert metrics[adapter.ARM_HYBRID_FULL] == {
        **metrics[adapter.ARM_COUNTS_FULL],
        "arm_id": adapter.ARM_HYBRID_FULL,
    }
    assert all(
        row["status"] == "diagnostic_only_until_shared_three_family_metrics_artifact"
        for row in metrics.values()
    )
