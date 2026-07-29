from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.comparison import metrics as comparison  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call(call_id: str, cost: str, *, conservative: bool = False) -> dict:
    return {
        "call_id": call_id,
        "call_ordinal": 1,
        "kind": "statement-belief",
        "model_id": "fixture-model",
        "accounting_basis": (
            "conservative_reserved_maximum" if conservative else "provider_reported_usage"
        ),
        "input_tokens": None if conservative else 100,
        "output_tokens": None if conservative else 20,
        "settled_cost_usd_exact": cost,
    }


def _attempt(
    attempt_id: str,
    ordinal: int,
    *,
    final: bool,
    calls: list[dict],
) -> dict:
    return {
        "attempt_id": attempt_id,
        "attempt_ordinal": ordinal,
        "status": "completed" if final else "error",
        "selected_final": final,
        "error_type": None if final else "transport",
        "calls": calls,
    }


def _ledger(
    statement_ids: list[str],
    *,
    unit_cost: str,
    retry_with_reservation: bool,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    provider = Decimal("0")
    conservative = Decimal("0")
    provider_calls = 0
    conservative_calls = 0
    for index, statement_id in enumerate(statement_ids):
        attempts: list[dict]
        if retry_with_reservation and index == 0:
            first = _call(f"call-{index}-1", "0.004000")
            second = _call(f"call-{index}-2", "0.006000", conservative=True)
            attempts = [
                _attempt(f"attempt-{index}-1", 1, final=False, calls=[first]),
                _attempt(f"attempt-{index}-2", 2, final=True, calls=[second]),
            ]
            provider += Decimal(first["settled_cost_usd_exact"])
            conservative += Decimal(second["settled_cost_usd_exact"])
            provider_calls += 1
            conservative_calls += 1
        else:
            call = _call(f"call-{index}-1", unit_cost)
            attempts = [_attempt(f"attempt-{index}-1", 1, final=True, calls=[call])]
            provider += Decimal(unit_cost)
            provider_calls += 1
        rows.append(
            {
                "record_type": "evidence_execution",
                "statement_id": statement_id,
                "execution_identity": f"execution-{index}",
                "call_eligible": True,
                "attempts": attempts,
            }
        )
    upper = provider + conservative
    return rows, {
        "provider_measured_cost_usd_exact": format(provider, "f"),
        "conservative_reserved_cost_usd_exact": format(conservative, "f"),
        "accounted_cost_lower_usd_exact": format(provider, "f"),
        "accounted_cost_upper_usd_exact": format(upper, "f"),
        "provider_measured_call_count": provider_calls,
        "conservative_call_count": conservative_calls,
        "includes_retries": True,
        "includes_relation_subcalls": True,
        "denominator": {
            "statements": len(statement_ids),
            "evidence_executions": len(statement_ids),
        },
        "excluded_cost_categories": list(comparison.COST_EXCLUDED_CATEGORIES),
    }


def _fixture(tmp_path: Path, *, resamples: int = 80) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    n = 100
    statement_ids = [f"stmt-{index:03d}" for index in range(n)]
    labels = np.asarray([int(index % 4 != 0) for index in range(n)], dtype=int)
    folds = np.asarray([index // 10 for index in range(n)], dtype=int)
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(
        gold_path,
        [
            {"statement_id": statement_id, "label": int(label), "fold_id": int(fold)}
            for statement_id, label, fold in zip(statement_ids, labels, folds, strict=True)
        ],
    )
    unresolved_positions = [
        next(
            index
            for index, (label, fold) in enumerate(zip(labels, folds, strict=True))
            if label == 0 and fold == fold_id
        )
        for fold_id in range(10)
    ]
    unresolved_set = set(unresolved_positions)
    strict_statement_ids = [
        statement_id
        for index, statement_id in enumerate(statement_ids)
        if index not in unresolved_set
    ]
    strict_gold_path = tmp_path / "strict_gold.jsonl"
    _write_jsonl(
        strict_gold_path,
        [
            {
                "statement_id": statement_id,
                "label": int(labels[index]),
                "fold_id": int(folds[index]),
            }
            for index, statement_id in enumerate(statement_ids)
            if index not in unresolved_set
        ],
    )
    substrate_path = tmp_path / "substrate.json"
    _write_json(substrate_path, {"n": n, "purpose": "shared-gold fixture"})
    threshold_path = tmp_path / "threshold.json"
    _write_json(threshold_path, {"threshold": 0.5, "selected_without_gold": True})

    x = np.arange(n, dtype=float)
    score_sets = {
        "paper-arm": np.clip(0.28 + 0.22 * labels + 0.30 * np.sin(x * 0.73), 0.01, 0.99),
        "current-arm": np.clip(0.26 + 0.30 * labels + 0.28 * np.sin(x * 0.61 + 0.2), 0.01, 0.99),
        "llm-arm": np.clip(0.24 + 0.38 * labels + 0.28 * np.sin(x * 0.49 + 0.4), 0.01, 0.99),
    }
    families = {"paper-arm": "paper", "current-arm": "current", "llm-arm": "llm"}
    costs = {"paper-arm": "0.001000", "current-arm": "0.002000", "llm-arm": "0.001500"}
    arms = []
    for arm_id, scores in score_sets.items():
        prediction_path = tmp_path / f"{arm_id}.predictions.jsonl"
        _write_jsonl(
            prediction_path,
            [
                {"statement_id": statement_id, "probability_correct": float(score)}
                for statement_id, score in zip(statement_ids, scores, strict=True)
            ],
        )
        cost_rows, accounting = _ledger(
            statement_ids,
            unit_cost=costs[arm_id],
            retry_with_reservation=arm_id == "llm-arm",
        )
        cost_path = tmp_path / f"{arm_id}.cost.jsonl"
        _write_jsonl(cost_path, cost_rows)
        arms.append(
            {
                "arm_id": arm_id,
                "label": arm_id.replace("-", " "),
                "family": families[arm_id],
                "predictions": {"path": prediction_path.name, "sha256": _sha(prediction_path)},
                "implementation": {
                    "implementation": f"fixture:{arm_id}",
                    "implementation_digest": hashlib.sha256(arm_id.encode()).hexdigest(),
                    "training_data_sha256": None,
                    "environment": "pytest",
                    "notes": None,
                },
                "threshold": {
                    "status": "available",
                    "value": 0.5,
                    "operator": "greater_than_or_equal",
                    "source_path": threshold_path.name,
                    "source_sha256": _sha(threshold_path),
                    "frozen_at": "2026-07-01T00:00:00Z",
                },
                "cost": {
                    "status": "ledger",
                    "record_type": comparison.EVIDENCE_EXECUTION_RECORD,
                    "path": cost_path.name,
                    "sha256": _sha(cost_path),
                    "basis": (
                        "mixed_conservative_upper_bound"
                        if arm_id == "llm-arm"
                        else "provider_measured_observed"
                    ),
                    "view_id": comparison.COST_VIEW_ID,
                    "price_source": "https://example.test/pricing",
                    "price_date": "2026-07-01",
                    "cost_comparability_id": "fixture_provider_token_cost",
                    "pricing": {
                        "cost_comparability_id": "fixture_provider_token_cost",
                        "currency": "USD",
                        "provider": "Fixture Provider",
                        "provider_model_id": arm_id,
                        "pricing_mode": "on_demand",
                        "region": "fixture-region",
                        "resolved_service_tier": "standard",
                        "retrieved_on": "2026-07-01",
                        "service_tier_request": "default",
                        "source_url": "https://example.test/pricing",
                        "tariff": {
                            "input_usd_per_million": "0.5",
                            "output_usd_per_million": "1",
                            "pricing_basis": "deterministic_test_tariff",
                        },
                        "unit": "per_million_tokens",
                    },
                    "projection": "all_executions",
                    "counterfactual_run_cost": False,
                    "shared_run_id": f"run.fixture.{arm_id}",
                    "additive_across_panels": False,
                    "accounting": accounting,
                },
            }
        )

    contract = {
        "prediction_unit": comparison.PREDICTION_UNIT,
        "gold_rule": comparison.GOLD_RULE,
        "substrate_sha256": _sha(substrate_path),
        "gold_sha256": _sha(gold_path),
        "evaluation_set_sha256": comparison.ordered_statement_id_sha256(statement_ids),
    }
    panel = {
        "substrate_id": comparison.CANONICAL_PANEL_IDS[0],
        "lane": "paper",
        "label": "Paper all-source fixture",
        "analysis_scope": "primary",
        "released_label_audit": {
            "released_label_rule": "positive if reviewed positive; released negative otherwise",
            "strict_e0_rule": "positive if any pair positive; negative only if all pairs negative",
            "released": {"statements": n, "positive": 75, "negative": 25},
            "strict_e0": {
                "resolved": 90,
                "positive": 75,
                "negative": 15,
                "unresolved": 10,
                "ordered_statement_id_sha256": comparison.ordered_statement_id_sha256(
                    strict_statement_ids
                ),
            },
            "released_negative_assumption": {
                "statements": 10,
                "share_of_released_negatives": 0.4,
                "ordered_statement_id_sha256": comparison.ordered_statement_id_sha256(
                    [statement_ids[index] for index in unresolved_positions]
                ),
            },
        },
        "contract": contract,
        "substrate_manifest": {"path": substrate_path.name, "sha256": _sha(substrate_path)},
        "gold": {"path": gold_path.name, "sha256": _sha(gold_path)},
        "strict_e0_resolved_gold": {
            "path": strict_gold_path.name,
            "sha256": _sha(strict_gold_path),
        },
        "arms": arms,
        "excluded_arms": [
            {
                "arm_id": "missing-model",
                "label": "Missing model",
                "family": "llm",
                "status": "excluded",
                "reason": "no complete prediction bundle",
                "required_artifact": "exact-panel predictions",
                "provenance": "fixture run plan",
            }
        ],
    }
    reader_panel = copy.deepcopy(panel)
    reader_panel.update(
        {
            "substrate_id": comparison.CANONICAL_PANEL_IDS[1],
            "label": "Paper readers fixture",
        }
    )
    for arm in reader_panel["arms"]:
        arm["cost"]["projection"] = "observed_execution_subset"
    scorer_registry_path = tmp_path / "scorers.json"
    _write_json(scorer_registry_path, {"fixture": "scorer registry"})
    spec = {
        "artifact_kind": comparison.SPEC_KIND,
        "frozen_at": "2026-07-17T00:00:00Z",
        "bootstrap": {"seed": 20260717, "resamples": resamples, "ci_level": 0.95},
        "scorer_registry": {
            "path": scorer_registry_path.name,
            "bytes": scorer_registry_path.stat().st_size,
            "sha256": _sha(scorer_registry_path),
        },
        "metrics": {
            "log_loss_epsilon": 1e-6,
            "calibration_bin_edges": [0, 0.2, 0.4, 0.6, 0.8, 1],
            "minimum_valid_bootstrap_fraction": 0.8,
            "pareto_metric": comparison.PARETO_METRIC,
        },
        "substrates": [panel, reader_panel],
    }
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, spec)
    return spec_path, spec


def test_canonical_artifact_is_deterministic_and_complete(tmp_path: Path) -> None:
    spec_path, _ = _fixture(tmp_path)
    artifact = comparison.build_artifact(spec_path)
    assert artifact == comparison.build_artifact(spec_path)
    assert set(artifact) == {"artifact_kind", "frozen_at", "provenance", "substrates"}
    assert [panel["substrate_id"] for panel in artifact["substrates"]] == list(
        comparison.CANONICAL_PANEL_IDS
    )
    panel = artifact["substrates"][0]
    assert (panel["n_evaluable"], panel["n_positive"], panel["n_negative"]) == (100, 75, 25)
    assert {arm["family"] for arm in panel["arms"]} == {"paper", "current", "llm"}
    assert panel["excluded_arms"][0]["arm_id"] == "missing-model"
    assert len(panel["comparisons"]) == 36
    assert all(row["contract"] == panel["contract"] for row in panel["comparisons"])

    metric = panel["arms"][0]["metrics"]
    assert "auprc" not in metric
    assert not math.isclose(
        metric["fold_mean_trapezoidal_pr_auc"]["estimate"],
        metric["pooled_average_precision"]["estimate"],
    )
    assert len(metric["fold_mean_trapezoidal_pr_auc"]["fold_estimates"]) == 10
    assert sum(row["n"] for row in metric["calibration"]["reliability_bins"]) == 100
    assert sum(metric["threshold"]["confusion"].values()) == 100
    assert metric["calibration"]["intercept_abs_error"]["estimate"] == pytest.approx(
        abs(metric["calibration"]["intercept"]["estimate"])
    )
    assert metric["calibration"]["slope_abs_error"]["estimate"] == pytest.approx(
        abs(metric["calibration"]["slope"]["estimate"] - 1)
    )
    pareto = panel["pareto"]
    assert pareto["objective_metric"] == "fold_mean_trapezoidal_pr_auc"
    assert pareto["cost_axis"] == "usd_per_1k_statements_upper"
    assert pareto["views"][0]["basis"] == "mixed"
    assert set(pareto["views"][0]["eligible_arm_ids"]) == {
        "paper-arm",
        "current-arm",
        "llm-arm",
    }
    sensitivity = panel["strict_e0_resolved_sensitivity"]
    assert sensitivity["analysis_scope"] == "fixed_resolved_only_sensitivity"
    assert (sensitivity["n_evaluable"], sensitivity["n_positive"], sensitivity["n_negative"]) == (
        90,
        75,
        15,
    )
    assert sensitivity["excluded_unresolved"] == 10
    assert len(sensitivity["comparisons"]) == 36
    assert all("cost" not in arm and "pareto" not in arm for arm in sensitivity["arms"])
    assert panel["arms"][0]["released_label_error_strata"][
        "released_negative_assumption"
    ]["statements"] == 10


def test_metrics_revalidates_bound_scorer_registry(tmp_path: Path) -> None:
    spec_path, spec = _fixture(tmp_path)
    registry_path = tmp_path / spec["scorer_registry"]["path"]
    payload = registry_path.read_bytes()
    registry_path.write_bytes(payload.replace(b"scorer", b"broken", 1))

    with pytest.raises(comparison.ContractError, match="digest mismatch"):
        comparison.build_artifact(spec_path)


def test_point_metrics_match_sklearn_on_shared_gold(tmp_path: Path) -> None:
    spec_path, spec = _fixture(tmp_path, resamples=30)
    panel = comparison.build_artifact(spec_path)["substrates"][0]
    gold = [json.loads(line) for line in (tmp_path / "gold.jsonl").read_text().splitlines()]
    labels = np.asarray([row["label"] for row in gold], dtype=int)
    folds = np.asarray([row["fold_id"] for row in gold], dtype=int)
    for arm_out, arm_in in zip(panel["arms"], spec["substrates"][0]["arms"], strict=True):
        prediction_path = tmp_path / arm_in["predictions"]["path"]
        scores = np.asarray(
            [json.loads(line)["probability_correct"] for line in prediction_path.read_text().splitlines()]
        )
        expected_fold = []
        for fold in range(10):
            precision, recall, _ = precision_recall_curve(labels[folds == fold], scores[folds == fold])
            expected_fold.append(auc(recall, precision))
        metrics = arm_out["metrics"]
        assert metrics["fold_mean_trapezoidal_pr_auc"]["estimate"] == pytest.approx(np.mean(expected_fold))
        assert metrics["pooled_average_precision"]["estimate"] == pytest.approx(
            average_precision_score(labels, scores)
        )
        assert metrics["auroc"]["estimate"] == pytest.approx(roc_auc_score(labels, scores))


def test_retry_inclusive_exact_decimal_cost(tmp_path: Path) -> None:
    spec_path, _ = _fixture(tmp_path, resamples=30)
    panel = comparison.build_artifact(spec_path)["substrates"][0]
    llm = next(arm for arm in panel["arms"] if arm["arm_id"] == "llm-arm")
    cost = llm["cost"]
    assert cost["record_type"] == "evidence_execution"
    assert cost["basis"] == "mixed_conservative_upper_bound"
    assert cost["inference_usd_lower_exact"] == "0.152500"
    assert cost["inference_usd_upper_exact"] == "0.158500"
    assert cost["provider_measured_call_count"] == 100
    assert cost["conservative_call_count"] == 1
    assert cost["execution_count"] == 100
    assert cost["attempt_count"] == 101
    assert cost["retry_attempt_count"] == 1
    assert cost["token_accounting_complete"] is False
    assert cost["input_tokens"] is None and cost["output_tokens"] is None
    assert cost["denominator"] == {"statements": 100, "evidence_executions": 100}
    assert cost["includes_retries"] is True
    assert cost["includes_relation_subcalls"] is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda spec: spec.update({"non_comparison_audits": []}), "unexpected fields"),
        (lambda spec: spec.update({"schema_version": 1}), "unexpected fields"),
        (lambda spec: spec.update({"notes": None}), "unexpected fields"),
        (
            lambda spec: spec["substrates"][0]["arms"][0]["cost"].update(
                {"record_type": "statement_attempt"}
            ),
            "expected evidence_execution",
        ),
        (
            lambda spec: spec["substrates"][0]["arms"][0]["implementation"].update(
                {"system_visible_inputs": {}}
            ),
            "unexpected fields",
        ),
    ],
)
def test_removed_branches_fail_closed(tmp_path: Path, mutate, match: str) -> None:
    spec_path, spec = _fixture(tmp_path, resamples=20)
    mutate(spec)
    _write_json(spec_path, spec)
    with pytest.raises(comparison.ContractError, match=match):
        comparison.build_artifact(spec_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda spec: spec["substrates"].pop(), "expected exactly the canonical panels"),
        (lambda spec: spec["substrates"].reverse(), "expected canonical panel"),
        (
            lambda spec: spec["substrates"][0].update({"lane": "representative"}),
            "only accepts the paper lane",
        ),
        (
            lambda spec: spec["substrates"][0].update(
                {"analysis_scope": "resolved_only_sensitivity"}
            ),
            "only accepts primary scope",
        ),
        (
            lambda spec: spec["substrates"][0]["released_label_audit"][
                "strict_e0"
            ].update(
                {"unresolved": 1}
            ),
            "counts or identity differ",
        ),
    ],
)
def test_only_two_complete_paper_panels_are_accepted(tmp_path: Path, mutate, match: str) -> None:
    spec_path, spec = _fixture(tmp_path, resamples=20)
    mutate(spec)
    _write_json(spec_path, spec)
    with pytest.raises(comparison.ContractError, match=match):
        comparison.build_artifact(spec_path)


def test_cost_ledger_identity_and_accounting_are_authenticated(tmp_path: Path) -> None:
    spec_path, spec = _fixture(tmp_path, resamples=20)
    arm = spec["substrates"][0]["arms"][2]
    cost_path = tmp_path / arm["cost"]["path"]
    rows = [json.loads(line) for line in cost_path.read_text().splitlines()]
    rows[1]["execution_identity"] = rows[0]["execution_identity"]
    rows[1]["statement_id"] = rows[0]["statement_id"]
    _write_jsonl(cost_path, rows)
    arm["cost"]["sha256"] = _sha(cost_path)
    _write_json(spec_path, spec)
    with pytest.raises(comparison.ContractError, match="duplicate statement/execution identity"):
        comparison.build_artifact(spec_path)

    spec_path, spec = _fixture(tmp_path / "accounting", resamples=20)
    spec["substrates"][0]["arms"][2]["cost"]["accounting"][
        "accounted_cost_upper_usd_exact"
    ] = "0.158501"
    _write_json(spec_path, spec)
    with pytest.raises(comparison.ContractError, match="ledger recomputes"):
        comparison.build_artifact(spec_path)

    spec_path, spec = _fixture(tmp_path / "obsolete-row-version", resamples=20)
    arm = spec["substrates"][0]["arms"][2]
    cost_path = tmp_path / "obsolete-row-version" / arm["cost"]["path"]
    rows = [json.loads(line) for line in cost_path.read_text().splitlines()]
    rows[0]["schema_version"] = 1
    _write_jsonl(cost_path, rows)
    arm["cost"]["sha256"] = _sha(cost_path)
    _write_json(spec_path, spec)
    with pytest.raises(comparison.ContractError, match="unexpected fields.*schema_version"):
        comparison.build_artifact(spec_path)


def test_cost_pricing_and_panel_binding_fail_closed(tmp_path: Path) -> None:
    spec_path, spec = _fixture(tmp_path / "rate-spelling", resamples=20)
    spec["substrates"][0]["arms"][0]["cost"]["pricing"]["tariff"][
        "input_usd_per_million"
    ] = "0.50"
    _write_json(spec_path, spec)
    with pytest.raises(comparison.ContractError, match="canonical decimal spelling"):
        comparison.build_artifact(spec_path)

    spec_path, spec = _fixture(tmp_path / "mixed-comparability", resamples=20)
    cost = spec["substrates"][0]["arms"][0]["cost"]
    cost["cost_comparability_id"] = "incomparable_fixture_cost"
    cost["pricing"]["cost_comparability_id"] = "incomparable_fixture_cost"
    _write_json(spec_path, spec)
    with pytest.raises(comparison.ContractError, match="one cost-comparability basis"):
        comparison.build_artifact(spec_path)

    spec_path, spec = _fixture(tmp_path / "shared-run", resamples=20)
    llm = next(
        arm
        for arm in spec["substrates"][1]["arms"]
        if arm["family"] == "llm"
    )
    llm["cost"]["shared_run_id"] = "run.different"
    _write_json(spec_path, spec)
    with pytest.raises(comparison.ContractError, match="disagree on pricing or shared run"):
        comparison.build_artifact(spec_path)


def test_prediction_coverage_and_atomic_output_fail_closed(tmp_path: Path) -> None:
    spec_path, spec = _fixture(tmp_path, resamples=20)
    prediction = spec["substrates"][0]["arms"][0]["predictions"]
    path = tmp_path / prediction["path"]
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
    prediction["sha256"] = _sha(path)
    _write_json(spec_path, spec)
    with pytest.raises(comparison.ContractError, match="incomplete coverage"):
        comparison.build_artifact(spec_path)

    valid_path, _ = _fixture(tmp_path / "valid", resamples=20)
    artifact = comparison.build_artifact(valid_path)
    output = tmp_path / "artifact.json"
    comparison.write_artifact(artifact, output)
    assert json.loads(output.read_text()) == artifact
    with pytest.raises(FileExistsError):
        comparison.write_artifact(artifact, output)
