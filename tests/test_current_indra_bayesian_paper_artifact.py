from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data/results/current_indra_bayesian_paper_20260717"
GOLD = ROOT / "data/results/indra_paper_comparison_gold_20260717"
EVIDENCE = (
    ROOT
    / "data/results/indra_paper_statement_gold_20260717/paper_evidence_adjudication.jsonl"
)
sys.path.insert(0, str(ROOT / "scripts"))

import score_current_indra_bayesian_paper as adapter  # noqa: E402



# Reads the gitignored local artifact trees; skipped only when they are WHOLLY
# absent (CI, a fresh checkout). A PARTIAL tree is a failure in
# tests/test_local_artifacts.py, never a skip here.
import _local_artifacts as _artifacts

pytestmark = _artifacts.requires()

def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bayesian_bundle_digests_schema_coverage_and_input_projections() -> None:
    manifest = json.loads((ARTIFACT / adapter.MANIFEST_FILENAME).read_text())
    assert _sha(ROOT / "scripts/score_current_indra_bayesian_paper.py") == manifest[
        "implementation"
    ]["adapter"]["sha256"]
    for filename, descriptor in manifest["outputs"].items():
        path = ARTIFACT / filename
        rows = _rows(path)
        assert _sha(path) == descriptor["sha256"]
        assert len(rows) == descriptor["rows"]

    all_gold = _rows(GOLD / "paper_released_gold.jsonl")
    reader_gold = _rows(GOLD / "paper_reader_eligible_released_gold.jsonl")
    expected_order = {
        "all": [row["statement_id"] for row in all_gold],
        "readers": [row["statement_id"] for row in reader_gold],
    }
    for arm_id, filename in adapter.PREDICTION_FILENAMES.items():
        rows = _rows(ARTIFACT / filename)
        assert all(
            set(row) == {"statement_id", "probability_correct"} for row in rows
        )
        panel = "readers" if "reader" in arm_id else "all"
        assert [row["statement_id"] for row in rows] == expected_order[panel]
        assert all(0.0 <= row["probability_correct"] <= 1.0 for row in rows)

    provenance = _rows(ARTIFACT / adapter.PROVENANCE_FILENAME)
    by_arm: defaultdict[str, list[dict]] = defaultdict(list)
    for row in provenance:
        by_arm[row["arm_id"]].append(row)
    reader_sources = set(adapter.READER_SOURCES)
    primary_reader_arms = {
        adapter.ARM_SIMPLE_READERS,
        adapter.ARM_BAYES_SOURCE_READERS,
        adapter.ARM_BAYES_SUBTYPE_READERS,
    }
    for arm_id in primary_reader_arms:
        assert len(by_arm[arm_id]) == 1676
        assert all(
            set(row["projected_source_counts"]) <= reader_sources
            for row in by_arm[arm_id]
        )
        assert all(row["input_projection"] == list(adapter.READER_SOURCES) for row in by_arm[arm_id])
    sensitivity = by_arm[adapter.ARM_SIMPLE_READER_SENSITIVITY]
    assert any(
        set(row["projected_source_counts"]) - reader_sources for row in sensitivity
    )
    assert manifest["panels"][adapter.PANEL_READER_SENSITIVITY][
        "direct_reader_parity"
    ] is False


def test_bayesian_fit_ledgers_exactly_exclude_each_test_folds_curated_pairs() -> None:
    all_gold = _rows(GOLD / "paper_released_gold.jsonl")
    reader_gold = _rows(GOLD / "paper_reader_eligible_released_gold.jsonl")
    all_fold_by_position = {index: row["fold_id"] for index, row in enumerate(all_gold)}
    targets = _rows(GOLD / "paper_prediction_targets.jsonl")
    reader_positions = [index for index, row in enumerate(targets) if row["reader_eligible"]]
    reader_fold_by_position = {
        position: row["fold_id"]
        for position, row in zip(reader_positions, reader_gold, strict=True)
    }
    reviewed = [
        row for row in _rows(EVIDENCE) if row["evidence_gold_label"] is not None
    ]
    fits = _rows(ARTIFACT / adapter.FIT_FILENAME)
    fit_by_key = {(row["arm_id"], row["fit_fold_id"]): row for row in fits}

    for arm_id, fold_by_position, sources, expected_pairs in (
        (
            adapter.ARM_BAYES_SOURCE_ALL,
            all_fold_by_position,
            set(adapter.ALL_SOURCES),
            5379,
        ),
        (
            adapter.ARM_BAYES_SUBTYPE_ALL,
            all_fold_by_position,
            set(adapter.ALL_SOURCES),
            5379,
        ),
        (
            adapter.ARM_BAYES_SOURCE_READERS,
            reader_fold_by_position,
            set(adapter.READER_SOURCES),
            5339,
        ),
        (
            adapter.ARM_BAYES_SUBTYPE_READERS,
            reader_fold_by_position,
            set(adapter.READER_SOURCES),
            5339,
        ),
    ):
        eligible = [
            row
            for row in reviewed
            if row["eligible_position"] in fold_by_position
            and row["source_apis"][0] in sources
        ]
        assert len(eligible) == expected_pairs
        for fold_id in range(10):
            fit = fit_by_key[(arm_id, fold_id)]
            expected_train = [
                row
                for row in eligible
                if fold_by_position[row["eligible_position"]] != fold_id
            ]
            expected_test = [
                row
                for row in eligible
                if fold_by_position[row["eligible_position"]] == fold_id
            ]
            source_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
            for row in expected_train:
                label = row["evidence_gold_label"]
                source_counts[row["source_apis"][0]][0 if label else 1] += 1
            assert fit["train_reviewed_pairs"] == len(expected_train)
            assert fit["excluded_test_reviewed_pairs"] == len(expected_test)
            assert fit["train_reviewed_pairs"] + fit["excluded_test_reviewed_pairs"] == expected_pairs
            assert fit["train_source_counts"] == dict(sorted(source_counts.items()))
            assert fit["no_pseudocounts"] is True

    for fold_id in range(10):
        source_all = fit_by_key[(adapter.ARM_BAYES_SOURCE_ALL, fold_id)]
        subtype_all = fit_by_key[(adapter.ARM_BAYES_SUBTYPE_ALL, fold_id)]
        assert source_all["train_source_counts"] == subtype_all["train_source_counts"]
        assert source_all["train_subtype_counts"] == {}
        assert set(subtype_all["train_subtype_counts"]) <= {"reach"}
        assert source_all["bundled_default_fallback_sources"] == sorted(
            set(adapter.ALL_SOURCES) - set(source_all["train_source_counts"])
        )
        # All nine reviewed SIGNOR pairs happen to occupy fold 4, so current
        # BayesianScorer correctly uses its bundled SIGNOR prior in that fold.
        assert ("signor" in source_all["bundled_default_fallback_sources"]) == (
            fold_id == 4
        )
        source_readers = fit_by_key[(adapter.ARM_BAYES_SOURCE_READERS, fold_id)]
        assert source_readers["bundled_default_fallback_sources"] == []


@pytest.mark.parametrize(
    ("arm_id", "expected"),
    [
        (adapter.ARM_SIMPLE_ALL, 0.9081126722127291),
        (adapter.ARM_SIMPLE_READERS, 0.9005174389031696),
        (adapter.ARM_SIMPLE_READER_SENSITIVITY, 0.9081095482651447),
        (adapter.ARM_BAYES_SOURCE_ALL, 0.9157259512379661),
        (adapter.ARM_BAYES_SUBTYPE_ALL, 0.9183220351471307),
        (adapter.ARM_BAYES_SOURCE_READERS, 0.908438955550575),
        (adapter.ARM_BAYES_SUBTYPE_READERS, 0.9105651525971895),
    ],
)
def test_bayesian_diagnostic_metrics_are_recomputable(
    arm_id: str, expected: float
) -> None:
    rows = _rows(ARTIFACT / adapter.PREDICTION_FILENAMES[arm_id])
    gold_name = (
        "paper_reader_eligible_released_gold.jsonl"
        if "reader" in arm_id
        else "paper_released_gold.jsonl"
    )
    gold = _rows(GOLD / gold_name)
    _, summary = adapter._diagnostics(arm_id, rows, gold)
    assert summary["fold_mean_trapezoidal_pr_auc"] == pytest.approx(expected, abs=1e-15)


def test_simple_all_evidence_outputs_reconcile_existing_direct_artifact() -> None:
    previous = ROOT / "data/results/current_indra_simple_paper_20260717"
    assert (
        ARTIFACT / adapter.PREDICTION_FILENAMES[adapter.ARM_SIMPLE_ALL]
    ).read_bytes() == (previous / "current_indra_simple_default_predictions.jsonl").read_bytes()
    assert (
        ARTIFACT
        / adapter.PREDICTION_FILENAMES[adapter.ARM_SIMPLE_READER_SENSITIVITY]
    ).read_bytes() == (
        previous / "current_indra_simple_default_reader_predictions.jsonl"
    ).read_bytes()
