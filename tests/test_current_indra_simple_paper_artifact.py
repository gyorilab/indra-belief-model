from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data/results/current_indra_simple_paper_20260717"
GOLD_DIR = ROOT / "data/results/indra_paper_comparison_gold_20260717"
REGISTRY = ROOT / "data/comparison/scorers.json"
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.comparison import metrics  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_current_simple_artifact_reconciles_registry_manifest_and_exact_coverage() -> None:
    manifest = json.loads(
        (ARTIFACT_DIR / "current_indra_simple_default_manifest.json").read_text()
    )
    registry = json.loads(REGISTRY.read_text())
    registry_arm = next(
        arm
        for arm in registry["scorers"]
        if arm["scorer_id"] == "indra_1.24.0_simple_default"
    )
    execution = registry_arm["benchmark_execution"]
    outputs = manifest["outputs"]

    for key, filename in (
        ("predictions", "current_indra_simple_default_predictions.jsonl"),
        (
            "reader_eligible_predictions",
            "current_indra_simple_default_reader_predictions.jsonl",
        ),
        (
            "prediction_provenance",
            "current_indra_simple_default_prediction_provenance.jsonl",
        ),
    ):
        path = ARTIFACT_DIR / filename
        assert _sha(path) == outputs[key]["sha256"]
        assert len(_rows(path)) == outputs[key]["rows"]

    assert execution["all_source_predictions_sha256"] == outputs["predictions"]["sha256"]
    assert (
        execution["reader_eligible_predictions_sha256"]
        == outputs["reader_eligible_predictions"]["sha256"]
    )
    assert manifest["coverage"]["target_statements"] == 1689
    assert manifest["coverage"]["predicted_statements"] == 1689
    assert manifest["coverage"]["reader_eligible_statements"] == 1676
    assert manifest["coverage"]["evidence_entries"] == 34035
    assert manifest["coverage"]["missing"] == 0
    assert manifest["coverage"]["invalid"] == 0

    targets = _rows(GOLD_DIR / "paper_prediction_targets.jsonl")
    all_predictions = _rows(
        ARTIFACT_DIR / "current_indra_simple_default_predictions.jsonl"
    )
    reader_predictions = _rows(
        ARTIFACT_DIR / "current_indra_simple_default_reader_predictions.jsonl"
    )
    assert [row["statement_id"] for row in all_predictions] == [
        row["statement_id"] for row in targets
    ]
    assert [row["statement_id"] for row in reader_predictions] == [
        row["statement_id"] for row in targets if row["reader_eligible"]
    ]
    assert all(
        set(row) == {"statement_id", "probability_correct"}
        for row in all_predictions + reader_predictions
    )


@pytest.mark.parametrize(
    ("gold_filename", "prediction_filename", "expected_fold_pr"),
    [
        (
            "paper_released_gold.jsonl",
            "current_indra_simple_default_predictions.jsonl",
            0.9081126722127291,
        ),
        (
            "paper_reader_eligible_released_gold.jsonl",
            "current_indra_simple_default_reader_predictions.jsonl",
            0.9081095482651447,
        ),
    ],
)
def test_current_simple_diagnostic_fold_metric_is_recomputable(
    gold_filename: str, prediction_filename: str, expected_fold_pr: float
) -> None:
    gold = _rows(GOLD_DIR / gold_filename)
    predictions = {
        row["statement_id"]: row["probability_correct"]
        for row in _rows(ARTIFACT_DIR / prediction_filename)
    }
    labels = np.asarray([bool(row["label"]) for row in gold])
    folds = np.asarray([row["fold_id"] for row in gold], dtype=int)
    scores = np.asarray([predictions[row["statement_id"]] for row in gold])
    computer = metrics._MetricComputer(
        labels,
        scores,
        folds,
        log_loss_epsilon=1e-6,
        calibration_edges=np.linspace(0.0, 1.0, 11),
        threshold=None,
    )
    point, fold_rows, fold_sd, _ = computer.point()

    assert len(fold_rows) == 10
    assert fold_sd > 0
    assert point["fold_mean_trapezoidal_pr_auc"] == pytest.approx(
        expected_fold_pr, abs=1e-15
    )
