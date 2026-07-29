from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from indra.belief import SimpleScorer
from indra.belief.skl import HybridScorer
from indra.statements import Activation, Agent, Evidence
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_current_indra_counts_hybrid_paper as adapter  # noqa: E402


def _panel() -> tuple[list[Activation], list[list[Evidence]], list[dict]]:
    statements: list[Activation] = []
    extras: list[list[Evidence]] = []
    gold: list[dict] = []
    for index in range(20):
        direct = Evidence(
            source_api="reach",
            pmid=str(1000 + index),
            text=f"Promoter A{index} activates B{index}.",
        )
        statement = Activation(
            Agent(f"A{index}"), Agent(f"B{index}"), evidence=[direct]
        )
        statements.append(statement)
        extras.append(
            [
                Evidence(
                    source_api="signor",
                    pmid=str(2000 + index),
                    text=f"A{index} induces B{index}.",
                )
            ]
        )
        gold.append(
            {
                "fold_id": index % 10,
                "label": index // 10,
                "statement_id": statement.uuid,
            }
        )
    return statements, extras, gold


def _tiny_estimator(fold_id: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=8,
        max_depth=3,
        max_features="sqrt",
        random_state=400 + fold_id,
        n_jobs=1,
    )


def test_feature_contract_is_exact_current_counts_scorer_order() -> None:
    source = adapter._feature_contract(adapter.SOURCE_CONFIGURATION)
    full = adapter._feature_contract(adapter.FULL_CONFIGURATION)
    assert source["feature_count"] == 11
    assert source["feature_names"] == [
        f"direct_source_count:{name}" for name in adapter.ALL_SOURCES
    ]
    assert full["feature_count"] == 77
    assert full["feature_names"][:11] == source["feature_names"]
    assert full["feature_names"][11:22] == [
        f"more_specific_source_count:{name}" for name in adapter.ALL_SOURCES
    ]
    assert full["feature_names"][-6:] == [
        "has_residue_and_position",
        "statement_member_count",
        "direct_unique_pmid_count",
        "more_specific_unique_pmid_count",
        "direct_promoter_sentence_fraction",
        "average_direct_evidence_sentence_length",
    ]
    assert full["feature_names_sha256"] == hashlib.sha256(
        adapter._canonical_bytes(full["feature_names"])
    ).hexdigest()


def test_cross_fit_has_complete_coverage_and_test_label_isolation() -> None:
    statements, extras, gold = _panel()
    predictions, provenance, fits, hybrid_audit = adapter._cross_fit(
        statements, extras, gold, estimator_factory=_tiny_estimator
    )
    assert set(predictions) == set(adapter.PREDICTION_FILENAMES)
    assert all(len(rows) == 20 for rows in predictions.values())
    assert len(provenance) == 60
    assert len(fits) == 20
    assert all(row["train_statement_count"] == 18 for row in fits)
    assert all(row["test_statement_count"] == 2 for row in fits)
    assert all(row["train_test_statement_intersection"] == 0 for row in fits)
    assert all(row["test_label_count_not_passed_to_fit"] == 2 for row in fits)
    assert all(len(row["fitted_state_sha256"]) == 64 for row in fits)
    assert hybrid_audit["hybrid_simple_fallback_evidence_entries"] == 0
    assert hybrid_audit["max_abs_hybrid_minus_counts_probability"] <= (
        4 * np.finfo(float).eps
    )
    assert all(
        set(row) == {"statement_id", "probability_correct"}
        for rows in predictions.values()
        for row in rows
    )

    # Changing only fold-0 test labels cannot change fold-0 predictions.  Those
    # labels can affect other folds, where the same rows legitimately enter training.
    changed_gold = copy.deepcopy(gold)
    for row in changed_gold:
        if row["fold_id"] == 0:
            row["label"] = 1 - row["label"]
    changed, _, _, _ = adapter._cross_fit(
        statements, extras, changed_gold, estimator_factory=_tiny_estimator
    )
    fold_zero_ids = {row["statement_id"] for row in gold if row["fold_id"] == 0}
    for arm_id in adapter.PREDICTION_FILENAMES:
        original_map = {
            row["statement_id"]: row["probability_correct"]
            for row in predictions[arm_id]
        }
        changed_map = {
            row["statement_id"]: row["probability_correct"]
            for row in changed[arm_id]
        }
        assert {item: original_map[item] for item in fold_zero_ids} == {
            item: changed_map[item] for item in fold_zero_ids
        }


def test_hybrid_fallback_semantics_are_current_class_not_production_artifact() -> None:
    statements, _, gold = _panel()
    labels = np.asarray([row["label"] for row in gold], dtype=int)
    counts = adapter._counts_scorer(
        adapter.SOURCE_CONFIGURATION,
        RandomForestClassifier(n_estimators=16, random_state=7),
    )
    counts.fit(statements, labels)
    hybrid = HybridScorer(counts, SimpleScorer())

    known = statements[0]
    known_counts = float(counts.score_statements([known])[0])
    known_hybrid = float(hybrid.score_statements([known])[0])
    assert known_hybrid == pytest.approx(known_counts, abs=2e-16)

    mixed = copy.copy(known)
    mixed.evidence = [
        *known.evidence,
        Evidence(source_api="eidos", text="A activates B."),
    ]
    mixed_counts = float(counts.score_statements([mixed])[0])
    mixed_hybrid = float(hybrid.score_statements([mixed])[0])
    assert mixed_hybrid > mixed_counts


def test_exclusions_only_cover_the_unidentified_official_counts_template() -> None:
    rows = adapter._exclusions()
    assert [row["excluded_scorer_id"] for row in rows] == [
        "indra_1.24.0_counts_unfitted",
    ]
    assert rows[0]["admission_requirement"]["currently_satisfied_by"] == [
        adapter.ARM_COUNTS_SOURCE,
        adapter.ARM_COUNTS_FULL,
    ]


def test_bundle_publication_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    rows = {"rows.jsonl": [{"value": 1}]}
    manifest = {"schema_version": 1}
    adapter._publish_bundle(output, rows, manifest)
    assert json.loads((output / "rows.jsonl").read_text()) == {"value": 1}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        adapter._publish_bundle(output, rows, manifest)
