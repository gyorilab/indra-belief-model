from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from indra_belief.comparison.error_review import CANONICAL_PROTOCOL_SHA256
from indra_belief.comparison.report import ReportError, build_report, render_reports


def _estimate(value: float) -> dict:
    return {"estimate": value, "ci95": [value - 0.01, value + 0.01]}


def _count_summary(count: int, denominator: int) -> dict:
    return {
        "count": count,
        "denominator": denominator,
        "proportion": None if denominator == 0 else count / denominator,
    }


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _descriptor(label: str, *, sha256: str | None = None) -> dict:
    return {"sha256": sha256 or _digest(label), "bytes": len(label)}


def _error_review(panel_id: str) -> dict:
    decisions = [
        ("false_positive", "supports_claim", "defensible", "human_matches_system"),
        ("false_positive", "supports_claim", "defensible", "human_matches_system"),
        ("false_positive", "indeterminate", "defensible", "indeterminate_ambiguity"),
        ("false_positive", "rejects_claim", "non_defensible", "human_matches_reference"),
        ("false_negative", "rejects_claim", "defensible", "human_matches_system"),
        ("false_negative", "rejects_claim", "defensible", "human_matches_system"),
        ("false_negative", "supports_claim", "non_defensible", "human_matches_reference"),
        ("false_negative", "supports_claim", "non_defensible", "human_matches_reference"),
    ]
    adjudications = [
        {
            "case_id": f"case_{index}_{panel_id}",
            "error_type": error_type,
            "human_classification": classification,
            "judgment": judgment,
            "defensibility_basis": basis,
            "dimensions": ["explicit_support"],
            "comment": None,
            "decision_source": "resolver" if index == 0 else "reviewer_agreement",
        }
        for index, (error_type, classification, judgment, basis) in enumerate(decisions)
    ]
    projection = (
        "all_executions" if panel_id == "paper_all_source" else "observed_execution_subset"
    )
    comparison_files = {
        "spec": _descriptor("spec", sha256=_digest("comparison-spec")),
        "bundle_manifest": _descriptor(
            "bundle", sha256=_digest("llm_gemma:bundle")
        ),
        "gold": _descriptor("gold", sha256=_digest(f"{panel_id}:gold")),
        "predictions": _descriptor(
            "predictions",
            sha256=_digest(f"llm_gemma:{projection}:predictions"),
        ),
        "execution_ledger": _descriptor(
            "execution-ledger",
            sha256=_digest(f"llm_gemma:{projection}:ledger"),
        ),
    }
    return {
        "artifact_kind": "indra_belief_error_review_report",
        "status": "complete",
        "panel_id": panel_id,
        "arm_id": "llm_gemma",
        "model_id": "llm_gemma",
        "packet_id": f"packet_{'a' * 64}",
        "evaluated_statements": 10,
        "threshold_errors": _count_summary(8, 10),
        "error_types": {
            "false_positive": {
                **_count_summary(4, 8),
                "defensible": _count_summary(3, 4),
                "non_defensible": _count_summary(1, 4),
            },
            "false_negative": {
                **_count_summary(4, 8),
                "defensible": _count_summary(2, 4),
                "non_defensible": _count_summary(2, 4),
            },
        },
        "human_classifications": {
            "supports_claim": _count_summary(4, 8),
            "rejects_claim": _count_summary(3, 8),
            "indeterminate": _count_summary(1, 8),
        },
        "review": {
            "reviewer_pseudonyms": ["reviewer.alpha", "reviewer.beta"],
            "resolver_pseudonym": "resolver.gamma",
            "exact_agreement": _count_summary(7, 8),
            "disagreement_count": 1,
            "resolved_by_resolver_count": 1,
            "classification_reliability": {},
            "human_attestation": (
                "I attest that I personally reviewed every assigned case without "
                "model-generated adjudication and that this ledger accurately records my decisions."
            ),
        },
        "defensibility": {
            "denominator": "all_threshold_errors",
            "defensible": _count_summary(5, 8),
            "non_defensible": _count_summary(3, 8),
            "system_supported_defensible": _count_summary(4, 8),
            "indeterminate_ambiguity_defensible": _count_summary(1, 8),
            "unresolved": _count_summary(0, 8),
        },
        "dimensions": {
            "multiple_dimensions_per_case": True,
            "denominator": "all_threshold_errors",
            "rows": [
                {
                    "dimension": "explicit_support",
                    **_count_summary(8, 8),
                    "by_judgment": {"defensible": 5, "non_defensible": 3},
                    "by_error_type": {"false_positive": 4, "false_negative": 4},
                }
            ],
        },
        "taxonomy_refinements": [],
        "adjudications": adjudications,
        "provenance": {
            "protocol": {"sha256": CANONICAL_PROTOCOL_SHA256, "bytes": 1},
            "comparison_inputs": {
                "panel_id": panel_id,
                "arm_id": "llm_gemma",
                "model_id": "llm_gemma",
                "files": comparison_files,
            },
        },
    }


def _arm(
    arm_id: str,
    value: float,
    *,
    cost: float | None = None,
    projection: str = "all_executions",
) -> dict:
    return {
        "arm_id": arm_id,
        "label": arm_id.replace("_", " "),
        "family": "llm" if arm_id.startswith("llm") else "current",
        "provenance": {
            "implementation_digest": _digest(f"{arm_id}:bundle"),
            "predictions_sha256": _digest(f"{arm_id}:{projection}:predictions"),
        },
        "metrics": {
            "fold_mean_trapezoidal_pr_auc": _estimate(value),
            "pooled_average_precision": _estimate(value - 0.01),
            "auroc": _estimate(value - 0.02),
            "brier": _estimate(1.0 - value),
            "log_loss": _estimate(1.1 - value),
            "calibration": {
                "ece": _estimate(0.08),
                "intercept": _estimate(0.02),
                "slope": _estimate(0.95),
            },
            "threshold": {"metrics": {"f1": _estimate(value - 0.03)}},
        },
        "cost": (
            {
                "status": "available",
                "usd_per_1k_statements": cost,
                "usd_per_1k_statements_lower": cost,
                "usd_per_1k_statements_upper": cost,
                "inference_usd_lower": cost,
                "inference_usd_upper": cost,
                "view_id": "provider-runtime-retry-inclusive",
                "basis": "provider_measured_observed",
                "provider_measured_call_count": 2,
                "conservative_call_count": 0,
                "token_accounting_complete": True,
                "input_tokens": 10,
                "output_tokens": 5,
                "price_date": "2026-07-17",
                "price_source": "https://example.test/pricing",
                "cost_comparability_id": "fixture_provider_token_cost",
                "pricing": {
                    "cost_comparability_id": "fixture_provider_token_cost",
                    "currency": "USD",
                    "provider": "AWS Bedrock",
                    "provider_model_id": "google.gemma-test",
                    "pricing_mode": "on_demand",
                    "region": "us-east-1",
                    "resolved_service_tier": "standard",
                    "retrieved_on": "2026-07-17",
                    "service_tier_request": "default",
                    "source_url": "https://example.test/pricing",
                    "tariff": {
                        "input_usd_per_million": "0.14",
                        "output_usd_per_million": "0.4",
                        "pricing_basis": "fixture",
                    },
                    "unit": "per_million_tokens",
                },
                "projection": projection,
                "counterfactual_run_cost": False,
                "shared_run_id": "run.fixture",
                "ledger_sha256": _digest(f"{arm_id}:{projection}:ledger"),
                "additive_across_panels": False,
                "includes_retries": True,
                "includes_relation_subcalls": True,
                "scope": {"excluded_cost_categories": ["training"]},
            }
            if cost is not None
            else {"status": "unavailable"}
        ),
        "pareto": {
            "status": "available" if cost is not None else "unavailable",
            "point_pareto": cost is not None,
            "uncertainty_pareto": False,
        },
    }


def _metrics() -> dict:
    def comparison() -> dict:
        return {
            "a_arm_id": "current_simple",
            "b_arm_id": "llm_gemma",
            "metric": "fold_mean_trapezoidal_pr_auc",
            "better_when": "higher",
            "delta": _estimate(0.02),
        }

    def audit(n: int, positive: int, negative: int) -> dict:
        return {
            "released_label_rule": "positive if reviewed positive; otherwise released negative",
            "strict_e0_rule": "negative only when every exact evidence pair is reviewed negative",
            "released": {"statements": n, "positive": positive, "negative": negative},
            "strict_e0": {
                "resolved": n - 111,
                "positive": positive,
                "negative": negative - 111,
                "unresolved": 111,
            },
            "released_negative_assumption": {
                "statements": 111,
                "share_of_released_negatives": 111 / negative,
            },
        }

    def sensitivity(arms: list[dict], n: int, positive: int, negative: int) -> dict:
        sensitivity_arms = []
        for arm in arms:
            row = dict(arm)
            row.pop("cost")
            row.pop("pareto")
            sensitivity_arms.append(row)
        return {
            "n_evaluable": n - 111,
            "n_positive": positive,
            "n_negative": negative - 111,
            "excluded_unresolved": 111,
            "arms": sensitivity_arms,
            "comparisons": [comparison()],
        }

    all_arms = [_arm("current_simple", 0.90), _arm("llm_gemma", 0.92, cost=0.42)]
    reader_arms = [
        _arm("current_simple", 0.89),
        _arm(
            "llm_gemma",
            0.91,
            cost=0.39,
            projection="observed_execution_subset",
        ),
    ]
    return {
        "artifact_kind": "indra_statement_belief_comparison",
        "provenance": {
            "bootstrap_resamples": 200,
            "source_manifest_sha256": _digest("comparison-spec"),
        },
        "substrates": [
            {
                "substrate_id": "paper_all_source",
                "contract": {"gold_sha256": _digest("paper_all_source:gold")},
                "label": "Paper all-source",
                "n_evaluable": 1689,
                "n_positive": 1237,
                "n_negative": 452,
                "released_label_audit": audit(1689, 1237, 452),
                "arms": all_arms,
                "comparisons": [comparison()],
                "strict_e0_resolved_sensitivity": sensitivity(all_arms, 1689, 1237, 452),
                "excluded_arms": [],
            },
            {
                "substrate_id": "paper_readers",
                "contract": {"gold_sha256": _digest("paper_readers:gold")},
                "label": "Paper readers",
                "n_evaluable": 1676,
                "n_positive": 1236,
                "n_negative": 440,
                "released_label_audit": audit(1676, 1236, 440),
                "arms": reader_arms,
                "comparisons": [comparison()],
                "strict_e0_resolved_sensitivity": sensitivity(reader_arms, 1676, 1236, 440),
                "excluded_arms": [],
            },
        ],
    }


def test_report_keeps_panels_separate_and_labels_literature_context() -> None:
    literature = {
        "artifact_kind": "indra_assembly_paper_published_method_metrics",
        "methods": [
            {
                "method": "Belief Orig - readers",
                "fold_mean_trapezoidal_pr_auc": 0.917,
                "fold_population_sd": 0.019,
            }
        ],
    }
    markdown, html = build_report(_metrics(), literature=literature)
    assert markdown.count("## Paper") == 2
    assert "Pareto sets never mix panels" in markdown
    assert "2023 published landscape" in markdown
    assert "Belief Orig - readers" in markdown
    assert "AWS Bedrock / google.gemma-test" in markdown
    assert "$0.14 input / $0.4 output per 1M tokens" in markdown
    assert "observed_execution_subset" in markdown
    assert "additive across panels=False" in markdown
    assert "<table>" in html


def test_report_names_the_resolved_error_denominator() -> None:
    error_reviews = [
        _error_review("paper_all_source"),
        _error_review("paper_readers"),
    ]
    markdown, html = build_report(_metrics(), error_reviews=error_reviews)

    expected = "5 were defensible (62.5%): 4 system-supported and 1 ambiguity-defensible."
    assert expected in markdown
    assert expected in html
    assert "must not be interpreted as confirmed model correctness" in markdown


def test_report_rejects_obsolete_or_partial_error_review_shape() -> None:
    legacy = {
        "status": "complete",
        "defensibility": {"proportion_defensible_resolved_errors": 0.625},
    }
    with pytest.raises(ReportError, match="fields differ"):
        build_report(_metrics(), error_reviews=[legacy, legacy])


@pytest.mark.parametrize(
    ("file_name", "message"),
    [
        ("spec", "comparison spec digest differs"),
        ("bundle_manifest", "bundle digest differs"),
        ("gold", "gold digest differs"),
        ("predictions", "prediction digest differs"),
        ("execution_ledger", "execution-ledger digest differs"),
    ],
)
def test_report_rejects_error_review_not_bound_to_supplied_metrics(
    file_name: str, message: str
) -> None:
    reviews = [_error_review("paper_all_source"), _error_review("paper_readers")]
    reviews[0]["provenance"]["comparison_inputs"]["files"][file_name][
        "sha256"
    ] = _digest(f"tampered:{file_name}")

    with pytest.raises(ReportError, match=message):
        build_report(_metrics(), error_reviews=reviews)


def test_report_rejects_reviewed_arm_absent_from_metrics_panel() -> None:
    reviews = [_error_review("paper_all_source"), _error_review("paper_readers")]
    reviews[0]["arm_id"] = "llm_absent"
    reviews[0]["model_id"] = "llm_absent"
    reviews[0]["provenance"]["comparison_inputs"]["arm_id"] = "llm_absent"
    reviews[0]["provenance"]["comparison_inputs"]["model_id"] = "llm_absent"

    with pytest.raises(ReportError, match="reviewed arm is absent"):
        build_report(_metrics(), error_reviews=reviews)


def test_render_reports_publishes_manifest_last(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    literature_path = tmp_path / "literature.json"
    metrics_path.write_text(json.dumps(_metrics()), encoding="utf-8")
    literature_path.write_text(
        json.dumps(
            {
                "artifact_kind": "indra_assembly_paper_published_method_metrics",
                "methods": [
                    {
                        "method": "Belief Orig - readers",
                        "fold_mean_trapezoidal_pr_auc": 0.917,
                        "fold_population_sd": 0.019,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    markdown = tmp_path / "report.md"
    html = tmp_path / "report.html"
    manifest = tmp_path / "manifest.json"
    value = render_reports(
        metrics_path,
        markdown_path=markdown,
        html_path=html,
        manifest_path=manifest,
        literature_path=literature_path,
    )
    assert markdown.exists() and html.exists() and manifest.exists()
    assert json.loads(manifest.read_text()) == value
    assert value["outputs"]["markdown"]["sha256"]
