from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data/results/current_indra_hierarchy_paper_20260717"
DIRECT = ROOT / "data/results/current_indra_bayesian_paper_20260717"
GOLD = ROOT / "data/results/indra_paper_comparison_gold_20260717"
sys.path.insert(0, str(ROOT / "scripts"))

import score_current_indra_hierarchy_paper as adapter  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hierarchy_bundle_digests_graph_contract_and_exact_prediction_schema() -> None:
    manifest = json.loads((ARTIFACT / adapter.MANIFEST_FILENAME).read_text())
    assert _sha(ROOT / "scripts/score_current_indra_hierarchy_paper.py") == manifest[
        "implementation"
    ]["adapter"]["sha256"]
    for filename, descriptor in manifest["outputs"].items():
        path = ARTIFACT / filename
        assert _sha(path) == descriptor["sha256"]
        assert len(_rows(path)) == descriptor["rows"]

    graph = manifest["graph_execution"]
    assert graph["root_statements"] == adapter.EXPECTED_PICKLE_ROOTS
    assert graph["nodes"] == adapter.EXPECTED_GRAPH_NODES
    assert graph["edges"] == adapter.EXPECTED_GRAPH_EDGES
    assert graph["acyclic"] is True
    assert graph["current_ontology_used"] is False
    assert graph["engine_helper_byte_equality_check"] == "pass"
    assert manifest["coverage"]["statements_with_descendants"] == 491
    assert manifest["coverage"]["target_descendant_statement_links"] == 2035

    all_gold = _rows(GOLD / "paper_released_gold.jsonl")
    reader_gold = _rows(GOLD / "paper_reader_eligible_released_gold.jsonl")
    for arm_id, filename in adapter.PREDICTION_FILENAMES.items():
        rows = _rows(ARTIFACT / filename)
        assert all(
            set(row) == {"statement_id", "probability_correct"} for row in rows
        )
        expected = reader_gold if "reader" in arm_id else all_gold
        assert [row["statement_id"] for row in rows] == [
            row["statement_id"] for row in expected
        ]


def test_hierarchy_reader_primary_is_five_source_and_sensitivity_is_not() -> None:
    provenance = _rows(ARTIFACT / adapter.PROVENANCE_FILENAME)
    by_arm: defaultdict[str, list[dict]] = defaultdict(list)
    for row in provenance:
        by_arm[row["arm_id"]].append(row)
    readers = set(adapter.bayes.READER_SOURCES)
    assert len(by_arm[adapter.ARM_READERS]) == 1676
    assert all(
        set(row["projected_combined_source_counts"]) <= readers
        for row in by_arm[adapter.ARM_READERS]
    )
    assert any(
        set(row["projected_combined_source_counts"]) - readers
        for row in by_arm[adapter.ARM_READER_SENSITIVITY]
    )
    manifest = json.loads((ARTIFACT / adapter.MANIFEST_FILENAME).read_text())
    sensitivity_arm = next(
        row for row in manifest["arms"] if row["arm_id"] == adapter.ARM_READER_SENSITIVITY
    )
    assert sensitivity_arm["direct_reader_parity"] is False
    assert sensitivity_arm["sensitivity_only"] is True


def test_hierarchy_effect_occurs_only_where_frozen_graph_has_descendants() -> None:
    hierarchy = {
        row["statement_id"]: row["probability_correct"]
        for row in _rows(ARTIFACT / adapter.PREDICTION_FILENAMES[adapter.ARM_ALL])
    }
    direct = {
        row["statement_id"]: row["probability_correct"]
        for row in _rows(DIRECT / "current_simple_direct_all_sources_predictions.jsonl")
    }
    provenance = [
        row
        for row in _rows(ARTIFACT / adapter.PROVENANCE_FILENAME)
        if row["arm_id"] == adapter.ARM_ALL
    ]
    changed = {
        statement_id
        for statement_id in hierarchy
        if hierarchy[statement_id] != direct[statement_id]
    }
    descendant_ids = {
        row["statement_id"]
        for row in provenance
        if row["descendant_statement_count"] > 0
    }

    assert len(changed) == 477
    assert len(descendant_ids) == 491
    assert changed <= descendant_ids
    assert all(hierarchy[key] == direct[key] for key in set(hierarchy) - descendant_ids)


def test_hierarchy_all_evidence_reader_sensitivity_is_exact_row_subset() -> None:
    all_predictions = {
        row["statement_id"]: row["probability_correct"]
        for row in _rows(ARTIFACT / adapter.PREDICTION_FILENAMES[adapter.ARM_ALL])
    }
    sensitivity = _rows(
        ARTIFACT / adapter.PREDICTION_FILENAMES[adapter.ARM_READER_SENSITIVITY]
    )
    assert all(
        row["probability_correct"] == all_predictions[row["statement_id"]]
        for row in sensitivity
    )


@pytest.mark.parametrize(
    ("arm_id", "expected"),
    [
        (adapter.ARM_ALL, 0.9084234724272212),
        (adapter.ARM_READERS, 0.8974683581998469),
        (adapter.ARM_READER_SENSITIVITY, 0.9081345518045394),
    ],
)
def test_hierarchy_diagnostic_metrics_are_recomputable(
    arm_id: str, expected: float
) -> None:
    predictions = _rows(ARTIFACT / adapter.PREDICTION_FILENAMES[arm_id])
    gold = _rows(
        GOLD
        / (
            "paper_reader_eligible_released_gold.jsonl"
            if "reader" in arm_id
            else "paper_released_gold.jsonl"
        )
    )
    _, summary = adapter.bayes._diagnostics(arm_id, predictions, gold)
    assert summary["fold_mean_trapezoidal_pr_auc"] == pytest.approx(expected, abs=1e-15)

