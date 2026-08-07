from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import reproduce_indra_paper_headlines as reproduction  # noqa: E402



# Reads the gitignored local artifact trees; skipped only when they are WHOLLY
# absent (CI, a fresh checkout). A PARTIAL tree is a failure in
# tests/test_local_artifacts.py, never a skip here.
import _local_artifacts as _artifacts

pytestmark = _artifacts.requires()

ARTIFACT_DIR = ROOT / "data/results/indra_paper_reproduction_20260717"


def _statement(
    stmt_hash: int,
    stmt_uuid: str,
    stmt_type: str,
    evidence: list[dict],
    supports: list[str] | None = None,
) -> dict:
    return {
        "id": stmt_uuid,
        "matches_hash": stmt_hash,
        "type": stmt_type,
        "supports": supports or [],
        "evidence": evidence,
    }


def _evidence(
    source: str,
    pmid: str | None,
    text: str | None,
    *,
    negated: bool = False,
) -> dict:
    return {"source_api": source, "pmid": pmid, "text": text, "negated": negated}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_metric_is_fold_trapezoidal_pr_not_average_precision() -> None:
    labels = [1, 0, 1, 0]
    probabilities = [0.9, 0.8, 0.7, 0.6]

    trapezoid = reproduction.trapezoidal_pr_auc(labels, probabilities)
    average_precision = average_precision_score(labels, probabilities)

    assert trapezoid == pytest.approx(0.7916666666666666)
    assert average_precision == pytest.approx(0.8333333333333333)
    assert trapezoid != pytest.approx(average_precision)


def test_orig_belief_formula_and_source_combination() -> None:
    counts = np.array([0, 1, 2, 3])
    source_a = reproduction.orig_belief_source_probability(counts, pr=0.2, ps=0.1)
    source_b = reproduction.orig_belief_source_probability(counts, pr=0.5, ps=0.2)

    assert source_a.tolist() == pytest.approx([0.0, 0.72, 0.864, 0.8928])
    combined = 1.0 - (1.0 - source_a) * (1.0 - source_b)
    assert combined[0] == 0.0
    assert np.all(combined[1:] >= source_a[1:])
    assert np.all(combined <= 1.0)


def test_orig_belief_mode_fit_is_deterministic_and_improves_likelihood() -> None:
    counts = [1, 1, 1, 2, 2, 3, 4, 4]
    labels = [0, 1, 1, 0, 1, 1, 0, 1]

    first = reproduction.fit_orig_belief_map(counts, labels)
    second = reproduction.fit_orig_belief_map(counts, labels)
    baseline = reproduction._orig_negative_log_likelihood(
        [0.5, 0.5], np.asarray(counts, dtype=float), np.asarray(labels, dtype=int)
    )

    assert first["pr"] == pytest.approx(second["pr"], abs=1e-14)
    assert first["ps"] == pytest.approx(second["ps"], abs=1e-14)
    assert 0.0 <= first["pr"] <= 1.0
    assert 0.0 <= first["ps"] <= 1.0
    assert -first["log_likelihood"] < baseline


def test_more_specific_evidence_is_partition_local_and_filters_negation() -> None:
    root_a = _statement(
        1,
        "root-a",
        "Activation",
        [_evidence("reach", None, "PROMOTER evidence")],
        ["support-two"],
    )
    root_b = _statement(
        2,
        "root-b",
        "Inhibition",
        [_evidence("sparser", "2", "root b evidence")],
        ["support-three"],
    )
    selected = {1: root_a, 2: root_b}
    supports = {
        "support-two": _statement(
            2,
            "support-two",
            "Inhibition",
            [_evidence("medscan", "support-two", "external support")],
        ),
        "support-three": _statement(
            3,
            "support-three",
            "Complex",
            [
                _evidence("hprd", "negated", "negative", negated=True),
                _evidence("bel", "positive", "positive"),
            ],
        ),
    }
    rows = [{"stmt_hash": "1"}, {"stmt_hash": "2"}]

    joint = reproduction._partition_extra_evidence(rows, selected, supports)
    isolated = reproduction._partition_extra_evidence(rows[:1], selected, supports)

    # In the joint partition, hash 2 is a root and gains an outgoing edge to
    # hash 3; its root statement also overwrites the external node attribute.
    assert set(ev["source_api"] for ev in joint[0]) == {"sparser", "bel"}
    assert [ev["source_api"] for ev in joint[1]] == ["bel"]
    # In the isolated partition hash 2 is only a leaf, so no transitive edge is
    # available and its external support evidence is used.
    assert [ev["source_api"] for ev in isolated[0]] == ["medscan"]
    assert all(ev["source_api"] != "hprd" for fold in joint for ev in fold)


def test_counts_scorer_feature_widths_and_material_columns() -> None:
    selected = {
        1: _statement(
            1,
            "root-a",
            "Activation",
            [
                _evidence("reach", None, "a promoter sentence"),
                _evidence("reach", None, None),
            ],
            ["support-two"],
        )
    }
    supports = {
        "support-two": _statement(
            2,
            "support-two",
            "Complex",
            [_evidence("bel", None, "indirect evidence")],
        )
    }
    rows = [{"stmt_hash": "1"}]

    promoter = reproduction.counts_scorer_matrix(
        rows, selected, supports, use_avg_evidence_len=False
    )
    avglen = reproduction.counts_scorer_matrix(
        rows, selected, supports, use_avg_evidence_len=True
    )

    assert promoter.shape == (1, 74)
    assert avglen.shape == (1, 75)
    assert promoter[0, reproduction.ALL_SOURCES.index("reach")] == 2
    assert promoter[0, 11 + reproduction.ALL_SOURCES.index("bel")] == 1
    activation_column = 22 + reproduction.STATEMENT_TYPES.index("Activation")
    assert promoter[0, activation_column] == 1
    assert promoter[0, 71] == 1  # Direct unique PMID set contains None.
    assert promoter[0, 72] == 1  # Indirect unique PMID set contains None.
    assert promoter[0, 73] == pytest.approx(0.5)
    assert avglen[0, 74] == pytest.approx(3.0)


def test_minimal_prediction_projection_has_exact_schema_and_gold_order() -> None:
    predictions = [
        {
            "arm_id": reproduction.ORIG_ARM.arm_id,
            "statement_id": "statement-b",
            "probability_correct": 0.25,
            "label": 0,
        },
        {
            "arm_id": reproduction.ORIG_ARM.arm_id,
            "statement_id": "statement-a",
            "probability_correct": 0.75,
            "label": 1,
        },
    ]
    gold = [
        {"statement_id": "statement-a", "fold_id": 0, "label": 1},
        {"statement_id": "statement-b", "fold_id": 1, "label": 0},
    ]

    projected = reproduction.minimal_prediction_projection(
        predictions, reproduction.ORIG_ARM, gold
    )

    assert projected == [
        {"statement_id": "statement-a", "probability_correct": 0.75},
        {"statement_id": "statement-b", "probability_correct": 0.25},
    ]
    assert all(
        set(row) == {"statement_id", "probability_correct"} for row in projected
    )


def test_released_reproduction_artifact_is_digest_pinned_and_recomputable() -> None:
    manifest_path = ARTIFACT_DIR / "paper_reproduction_manifest.json"
    digest_path = ARTIFACT_DIR / "paper_reproduction_manifest.sha256"
    predictions_path = ARTIFACT_DIR / "paper_reproduction_oof_predictions.jsonl"
    metrics_path = ARTIFACT_DIR / "paper_reproduction_fold_metrics.jsonl"
    fits_path = ARTIFACT_DIR / "paper_reproduction_fit_provenance.jsonl"
    minimal_paths = {
        "orig_belief_readers_minimal_predictions": (
            ARTIFACT_DIR / "orig_belief_readers_predictions.jsonl",
            ROOT
            / "data/results/indra_paper_comparison_gold_20260717/"
            "paper_reader_eligible_released_gold.jsonl",
            reproduction.ORIG_ARM.arm_id,
        ),
        "rf_promoter_minimal_predictions": (
            ARTIFACT_DIR / "rf_promoter_all_sources_specific_predictions.jsonl",
            ROOT
            / "data/results/indra_paper_comparison_gold_20260717/"
            "paper_released_gold.jsonl",
            reproduction.RF_PROMOTER_ARM.arm_id,
        ),
        "rf_promoter_avglen_minimal_predictions": (
            ARTIFACT_DIR / "rf_promoter_avglen_all_sources_specific_predictions.jsonl",
            ROOT
            / "data/results/indra_paper_comparison_gold_20260717/"
            "paper_released_gold.jsonl",
            reproduction.RF_PROMOTER_AVGLEN_ARM.arm_id,
        ),
    }

    manifest = json.loads(manifest_path.read_text())
    expected_manifest_digest = digest_path.read_text().split()[0]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected_manifest_digest
    for key, path in (
        ("oof_predictions", predictions_path),
        ("fold_metrics", metrics_path),
        ("fit_provenance", fits_path),
        *[(key, value[0]) for key, value in minimal_paths.items()],
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["outputs"][key]["sha256"]

    predictions = _read_jsonl(predictions_path)
    metrics = _read_jsonl(metrics_path)
    assert len(predictions) == 1676 + 1689 + 1689
    assert len(metrics) == 30
    assert len({(row["arm_id"], row["statement_id"]) for row in predictions}) == len(
        predictions
    )

    for output_key, (minimal_path, gold_path, arm_id) in minimal_paths.items():
        minimal = _read_jsonl(minimal_path)
        gold = _read_jsonl(gold_path)
        combined_by_id = {
            row["statement_id"]: row["probability_correct"]
            for row in predictions
            if row["arm_id"] == arm_id
        }
        assert [row["statement_id"] for row in minimal] == [
            row["statement_id"] for row in gold
        ]
        assert all(
            set(row) == {"statement_id", "probability_correct"} for row in minimal
        )
        assert all(
            row["probability_correct"] == combined_by_id[row["statement_id"]]
            for row in minimal
        )
        assert manifest["outputs"][output_key]["schema"] == [
            "statement_id",
            "probability_correct",
        ]

    for arm in reproduction.ARMS:
        arm_predictions = [row for row in predictions if row["arm_id"] == arm.arm_id]
        arm_metrics = [row for row in metrics if row["arm_id"] == arm.arm_id]
        assert len(arm_metrics) == 10
        recomputed = []
        for fold_id in range(10):
            fold_predictions = [
                row for row in arm_predictions if row["fold_id"] == fold_id
            ]
            value = reproduction.trapezoidal_pr_auc(
                [row["label"] for row in fold_predictions],
                [row["probability_correct"] for row in fold_predictions],
            )
            stored = next(row for row in arm_metrics if row["fold_id"] == fold_id)
            assert value == pytest.approx(stored["trapezoidal_pr_auc"], abs=1e-15)
            recomputed.append(value)
        result = manifest["results"][arm.arm_id]
        assert float(np.mean(recomputed)) == pytest.approx(
            result["fold_mean_trapezoidal_pr_auc"], abs=1e-15
        )
        assert float(np.std(recomputed, ddof=0)) == pytest.approx(
            result["fold_population_std"], abs=1e-15
        )
        assert round(float(np.mean(recomputed)), 3) == arm.published_mean
        assert round(float(np.std(recomputed, ddof=0)), 3) == arm.published_std
        assert result["rounded_headline_match"] is True

    assert (
        manifest["results"][reproduction.ORIG_ARM.arm_id]["eligible_set"]
        != manifest["results"][reproduction.RF_PROMOTER_ARM.arm_id]["eligible_set"]
    )
    suspect = manifest["published_row_execution_state"][
        "orig_belief_all_sources_specific_counts_only"
    ]
    assert suspect["visible_completed_iterations"] == 2
    assert suspect["status"] == "forensic_stale_state_suspect_not_a_parity_gate"
