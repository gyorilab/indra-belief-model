from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from indra_belief.comparison.assemble import (
    ALL_SOURCE_PANEL,
    READER_PANEL,
    AssemblyError,
    AssemblyInputs,
    LlmModelInput,
    assemble_spec,
    write_spec,
)
from indra_belief.comparison.metrics import build_artifact


SIMPLE_IMPLEMENTATION_SHA = "1" * 64
SKLEARN_IMPLEMENTATION_SHA = "2" * 64
DEFAULT_PRIOR_SHA = "3" * 64


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _canonical_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    return path


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    return path


def _desc(path: Path, rows: int | None = None, *, relative_to: Path | None = None):
    value = {
        "path": path.name if relative_to is not None else str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _prediction_rows(ids: list[str], shift: int = 0):
    return [
        {
            "statement_id": statement_id,
            "probability_correct": 0.05 + 0.9 * (((index + shift) % 17) / 16),
        }
        for index, statement_id in enumerate(ids)
    ]


def _production_bundle(
    tmp_path: Path,
    *,
    targets: Path,
    reader_gold: Path,
    all_predictions: Path,
    reader_predictions: Path,
    all_ids: list[str],
    reader_ids: list[str],
) -> Path:
    model = tmp_path / "production_fitted_counts.pkl"
    model.write_bytes(b"opaque fitted CountsScorer fixture\n")
    model_desc = _desc(model, relative_to=tmp_path)

    acquisition = _json(
        tmp_path / "production_acquisition.json",
        {"artifact": {"bytes": model_desc["bytes"], "sha256": model_desc["sha256"]}},
    )
    static_audit = _json(
        tmp_path / "production_static_audit.json",
        {
            "dangerous_globals": [],
            "model": {"sha256": model_desc["sha256"]},
            "status": "passed",
        },
    )

    retained: dict[str, Path] = {}
    for name in (
        "export_assembly",
        "license",
        "adapter",
        "indra_belief",
        "indra_belief_skl",
        "default_prior",
    ):
        path = tmp_path / f"production_{name}.txt"
        path.write_text(f"retained {name} fixture\n")
        retained[name] = path
    commit = "1" * 40
    git_blob = "2" * 40
    export_desc = _desc(retained["export_assembly"], relative_to=tmp_path)
    authority = _json(
        tmp_path / "production_export_authority.json",
        {"commit": commit, "git_blob": git_blob, "sha256": export_desc["sha256"]},
    )

    wheels = [
        {
            "filename": f"fixture-package-{index}.whl",
            "retained": False,
            "sha256": hashlib.sha256(f"wheel-{index}".encode()).hexdigest(),
        }
        for index in range(19)
    ]
    all_sources = [
        "hprd",
        "rlimsp",
        "bel",
        "trips",
        "reach",
        "biopax",
        "sparser",
        "medscan",
        "isi",
        "trrust",
        "signor",
    ]
    reader_sources = ["reach", "sparser", "medscan", "rlimsp", "trips"]
    limitations = {
        key: f"synthetic fixture limitation: {key}"
        for key in (
            "deployment_parity",
            "training_provenance",
            "training_overlap",
            "reader_counterfactual",
            "runtime_replay",
            "simple_fallback",
        )
    }
    return _canonical_json(
        tmp_path / "production.json",
        {
            "arm": {
                "analysis_role": "descriptive_nonconfirmatory",
                "arm_id": "production.hybrid.all",
                "environment": "synthetic production replay fixture",
                "family": "current",
                "implementation": "fitted Hybrid artifact fixture",
                "label": "Production Hybrid all source",
            },
            "artifact_kind": "indra_cogex_hybrid_comparison_bundle",
            "excluded_diagnostics": {
                "arm_id": "production.hybrid.reader_row_subset",
                "reason": "all-source row filtering is not a reader-specific re-score",
            },
            "implementation": {
                "combined_replay_sha256": "3" * 64,
                "export_assembly": {
                    "authority": _desc(authority, relative_to=tmp_path),
                    "commit": commit,
                    "git_blob": git_blob,
                    "license": _desc(retained["license"], relative_to=tmp_path),
                    "repository": "https://example.test/indra_db",
                    "repository_path": "export_assembly.py",
                    "source": export_desc,
                },
                "scorer": {
                    "adapter": _desc(retained["adapter"], relative_to=tmp_path),
                    "default_prior": _desc(
                        retained["default_prior"], relative_to=tmp_path
                    ),
                    "indra_belief_module": _desc(
                        retained["indra_belief"], relative_to=tmp_path
                    ),
                    "indra_belief_skl_module": _desc(
                        retained["indra_belief_skl"], relative_to=tmp_path
                    ),
                },
            },
            "inputs": {
                "prediction_targets": _desc(
                    targets, len(all_ids), relative_to=tmp_path
                ),
                "reader_gold": _desc(
                    reader_gold, len(reader_ids), relative_to=tmp_path
                ),
            },
            "limitations": limitations,
            "model": {
                "artifact": model_desc,
                "artifact_role": "fitted_counts_component",
                "classifier": {
                    "class": "sklearn.ensemble._forest.RandomForestClassifier",
                    "classifier_state_sha256": "4" * 64,
                    "fitted_contract_sha256": "5" * 64,
                    "max_depth": 13,
                    "n_estimators": 2_000,
                    "n_features": 87,
                    "tree_topology_sha256": "6" * 64,
                },
                "evaluated_class": "indra.belief.skl.HybridScorer",
                "fallback_class": "indra.belief.SimpleScorer",
                "fitted_component_class": "indra.belief.skl.CountsScorer",
                "source_list": [f"fitted-source-{index}" for index in range(17)],
                "static_audit": _desc(static_audit, relative_to=tmp_path),
            },
            "panels": {
                "all_source": {
                    "arm_id": "production.hybrid.all",
                    "execution_route": "synthetic all-source replay",
                    "input_sources": all_sources,
                    "label": "Production Hybrid all source",
                    "literal_export_route_and_input_semantics": True,
                    "panel_id": "all_source_1689",
                    "panel_specific_rescore": True,
                    "predictions": _desc(
                        all_predictions, len(all_ids), relative_to=tmp_path
                    ),
                    "raw_statement_ids_sha256": _raw_id_digest(all_ids),
                    "route_census": {
                        "counts_only": len(all_ids),
                        "counts_plus_simple": 0,
                        "simple_only": 0,
                    },
                    "rows": len(all_ids),
                    "statement_ids_sha256": _ordered_id_digest(all_ids),
                },
                "readers": {
                    "arm_id": "production.hybrid.readers",
                    "execution_route": "synthetic five-reader replay",
                    "input_sources": reader_sources,
                    "label": "Production Hybrid readers",
                    "literal_export_route_and_input_semantics": False,
                    "panel_id": "readers_only_1676",
                    "panel_specific_rescore": True,
                    "predictions": _desc(
                        reader_predictions, len(reader_ids), relative_to=tmp_path
                    ),
                    "raw_statement_ids_sha256": _raw_id_digest(reader_ids),
                    "route_census": {
                        "counts_only": len(reader_ids),
                        "counts_plus_simple": 0,
                        "simple_only": 0,
                    },
                    "rows": len(reader_ids),
                    "statement_ids_sha256": _ordered_id_digest(reader_ids),
                },
            },
            "provenance": {
                "acquisition": _desc(acquisition, relative_to=tmp_path),
                "historical_runtime_established": False,
                "literal_live_deployment_output": False,
                "live_deployment_parity_established": False,
                "predictions_are_exact_evidence": True,
                "private_training_provenance_available": False,
                "training_overlap_status": "unknown",
            },
            "runtime": {
                "package_versions": {
                    "indra": "1.24.0",
                    "scikit_learn": "1.4.1.post1",
                },
                "python": "3.12.10",
                "serialization": {
                    "embedded_sklearn": "1.3.2",
                    "learned_state_mutated": False,
                    "native_control_exact_prediction_parity": True,
                    "replay_sklearn": "1.4.1.post1",
                    "unmodified_replay_sklearn_accepted": False,
                },
                "wheel_identities": wheels,
            },
            "status": "complete",
        },
    )


def _scorer_registry(tmp_path: Path, production_path: Path) -> Path:
    production = json.loads(production_path.read_bytes())
    model = production["model"]
    return _json(
        tmp_path / "scorers.json",
        {
            "frozen_at": "2026-07-21",
            "scope": "synthetic scorer registry fixture",
            "source_snapshots": {},
            "installed_runtime": {
                "belief_init_sha256": SIMPLE_IMPLEMENTATION_SHA,
                "belief_skl_sha256": SKLEARN_IMPLEMENTATION_SHA,
                "default_prior_resource_sha256": DEFAULT_PRIOR_SHA,
                "matches_current_release_implementation": True,
            },
            "taxonomy": {
                "ast_exhaustive_class_count": 6,
                "classes": [
                    "BeliefScorer",
                    "SimpleScorer",
                    "BayesianScorer",
                    "SklearnScorer",
                    "CountsScorer",
                    "HybridScorer",
                ],
                "deprecated_classes": [],
            },
            "scorers": [
                {
                    "scorer_id": "indra_1.24.0_simple_default",
                    "class": "indra.belief.SimpleScorer",
                    "implementation_sha256": SIMPLE_IMPLEMENTATION_SHA,
                    "resource_sha256": DEFAULT_PRIOR_SHA,
                },
                {
                    "scorer_id": "indra_1.24.0_bayesian_unfitted",
                    "class": "indra.belief.BayesianScorer",
                    "implementation_sha256": SIMPLE_IMPLEMENTATION_SHA,
                },
                {
                    "scorer_id": "indra_1.24.0_counts_unfitted",
                    "class": "indra.belief.skl.CountsScorer",
                    "implementation_sha256": SKLEARN_IMPLEMENTATION_SHA,
                },
                {
                    "scorer_id": "indra_1.24.0_hybrid_unfitted",
                    "class": "indra.belief.skl.HybridScorer",
                    "implementation_sha256": SKLEARN_IMPLEMENTATION_SHA,
                },
                {
                    "scorer_id": "indra_db_7dc8bf5_cogex_hybrid_production",
                    "kind": "recovered_fitted_scorer",
                    "class": "indra.belief.skl.HybridScorer",
                    "artifact_access": "recovered_authenticated_download",
                    "official_status": [
                        "artifact_referenced_by_current_cogex_export_pipeline",
                        "recovered_storage_artifact",
                    ],
                    "benchmark_decision": (
                        "include_recovered_replay_as_descriptive_nonconfirmatory_only"
                    ),
                    "artifact_sha256": model["artifact"]["sha256"],
                    "artifact_bytes": model["artifact"]["bytes"],
                    "artifact_provenance_manifest": _desc(
                        production_path, relative_to=tmp_path
                    ),
                    "fitted_state": {
                        "classifier": "sklearn.ensemble._forest.RandomForestClassifier",
                        "n_estimators": 2_000,
                        "max_depth": 13,
                        "n_features": 87,
                        "source_list": model["source_list"],
                    },
                    "comparison_status": {
                        "analysis_role": "descriptive_nonconfirmatory",
                        "all_source_route": "counts_only",
                        "reader_route": "counts_only_counterfactual_source_projection",
                        "literal_live_deployment_output": False,
                        "live_deployment_parity_established": False,
                        "historical_runtime_established": False,
                        "private_training_provenance_available": False,
                        "training_overlap_status": "unknown",
                        "simple_fallback_exercised": False,
                    },
                },
            ],
            "comparison_axes": {},
            "audit_conclusions": [],
        },
    )


def _cost_rows(ids: list[str]):
    return [
        {
            "record_type": "evidence_execution",
            "statement_id": statement_id,
            "execution_identity": f"execution.{index}",
            "call_eligible": True,
            "attempts": [
                {
                    "attempt_id": f"attempt.{index}",
                    "attempt_ordinal": 1,
                    "status": "completed",
                    "selected_final": True,
                    "error_type": None,
                    "calls": [
                        {
                            "call_id": f"call.{index}",
                            "call_ordinal": 1,
                            "kind": "monolithic",
                            "model_id": "llm-test",
                            "accounting_basis": "provider_reported_usage",
                            "input_tokens": 2,
                            "output_tokens": 1,
                            "settled_cost_usd_exact": "0.001",
                        }
                    ],
                }
            ],
        }
        for index, statement_id in enumerate(ids)
    ]


def _cost_desc(
    path: Path,
    *,
    rows: int,
    statements: int,
    projection: str,
    relative_to: Path,
):
    total = f"{rows / 1000:.3f}"
    pricing = {
        "cost_comparability_id": "fixture_provider_token_cost",
        "currency": "USD",
        "provider": "Fixture Provider",
        "provider_model_id": "llm-test",
        "pricing_mode": "on_demand",
        "region": "fixture-region",
        "resolved_service_tier": "standard",
        "retrieved_on": "2026-07-20",
        "service_tier_request": "default",
        "source_url": "https://example.test/pricing",
        "tariff": {
            "input_usd_per_million": "0.5",
            "output_usd_per_million": "1",
            "pricing_basis": "deterministic_test_tariff",
        },
        "unit": "per_million_tokens",
    }
    return {
        **_desc(path, rows, relative_to=relative_to),
        "record_type": "evidence_execution",
        "basis": "provider_measured_observed",
        "view_id": "provider-runtime-retry-inclusive",
        "price_source": "https://example.test/pricing",
        "price_date": "2026-07-20",
        "cost_comparability_id": "fixture_provider_token_cost",
        "pricing": pricing,
        "projection": projection,
        "counterfactual_run_cost": False,
        "shared_run_id": "run.test",
        "additive_across_panels": False,
        "accounting": {
            "provider_measured_cost_usd_exact": total,
            "conservative_reserved_cost_usd_exact": "0",
            "accounted_cost_lower_usd_exact": total,
            "accounted_cost_upper_usd_exact": total,
            "provider_measured_call_count": rows,
            "conservative_call_count": 0,
            "includes_retries": True,
            "includes_relation_subcalls": True,
            "denominator": {
                "statements": statements,
                "evidence_executions": rows,
            },
            "excluded_cost_categories": [
                "training",
                "local_aggregation",
                "feature_materialization",
                "upstream_reading",
            ],
        },
    }


def _fixture(tmp_path: Path) -> tuple[AssemblyInputs, Path, Path]:
    all_ids = [f"statement.{index:04d}" for index in range(1689)]
    # Drop thirteen rows across the stream so the reader panel is a genuine
    # order-preserving subset, not merely a prefix.
    omitted = {17 + 127 * index for index in range(13)}
    reader_ids = [value for index, value in enumerate(all_ids) if index not in omitted]
    assert len(reader_ids) == 1676
    all_gold_rows = [
        {"statement_id": value, "label": (index // 10) % 2, "fold_id": index % 10}
        for index, value in enumerate(all_ids)
    ]
    reader_gold_rows = [
        {
            "statement_id": value,
            "label": all_gold_rows[all_ids.index(value)]["label"],
            "fold_id": index % 10,
        }
        for index, value in enumerate(reader_ids)
    ]
    unresolved_candidates = [
        row["statement_id"]
        for row in all_gold_rows
        if row["label"] == 0 and row["statement_id"] in set(reader_ids)
    ]
    unresolved_ids = set(unresolved_candidates[:111])
    assert len(unresolved_ids) == 111
    all_strict_rows = [
        row for row in all_gold_rows if row["statement_id"] not in unresolved_ids
    ]
    reader_strict_rows = [
        row for row in reader_gold_rows if row["statement_id"] not in unresolved_ids
    ]
    assert len(all_strict_rows) == 1578
    assert len(reader_strict_rows) == 1565
    all_gold = _jsonl(tmp_path / "all_gold.jsonl", all_gold_rows)
    reader_gold = _jsonl(tmp_path / "reader_gold.jsonl", reader_gold_rows)
    all_strict_gold = _jsonl(tmp_path / "all_strict_gold.jsonl", all_strict_rows)
    reader_strict_gold = _jsonl(
        tmp_path / "reader_strict_gold.jsonl", reader_strict_rows
    )
    targets = _jsonl(
        tmp_path / "targets.jsonl",
        [
            {
                "statement_id": value,
                "reader_eligible": value in set(reader_ids),
            }
            for value in all_ids
        ],
    )
    all_gold_desc = _desc(all_gold, len(all_ids))
    all_gold_desc["ordered_statement_id_sha256"] = _ordered_id_digest(all_ids)
    reader_gold_desc = _desc(reader_gold, len(reader_ids))
    reader_gold_desc["ordered_statement_id_sha256"] = _ordered_id_digest(reader_ids)
    all_strict_desc = _desc(all_strict_gold, len(all_strict_rows))
    all_strict_desc["ordered_statement_id_sha256"] = _ordered_id_digest(
        [str(row["statement_id"]) for row in all_strict_rows]
    )
    reader_strict_desc = _desc(reader_strict_gold, len(reader_strict_rows))
    reader_strict_desc["ordered_statement_id_sha256"] = _ordered_id_digest(
        [str(row["statement_id"]) for row in reader_strict_rows]
    )
    target_desc = _desc(targets, len(all_ids))
    gold_manifest_path = tmp_path / "gold_manifest.json"
    _json(
        gold_manifest_path,
        {
            "artifact_kind": "indra_paper_comparison_gold",
            "outputs": {
                "paper_released_gold": all_gold_desc,
                "paper_reader_eligible_released_gold": reader_gold_desc,
                "paper_strict_e0_resolved_gold": all_strict_desc,
                "paper_reader_eligible_strict_e0_resolved_gold": reader_strict_desc,
                "paper_prediction_targets": target_desc,
            },
        },
    )
    gold_manifest_sha = hashlib.sha256(gold_manifest_path.read_bytes()).hexdigest()
    all_sha = all_gold_desc["sha256"]
    reader_sha = reader_gold_desc["sha256"]
    all_predictions = _jsonl(
        tmp_path / "all_predictions.jsonl", _prediction_rows(all_ids)
    )
    reader_predictions = _jsonl(
        tmp_path / "reader_predictions.jsonl", _prediction_rows(reader_ids, 3)
    )
    llm_all_predictions = _jsonl(
        tmp_path / "llm_all_predictions.jsonl", _prediction_rows(all_ids, 5)
    )
    llm_reader_predictions = _jsonl(
        tmp_path / "llm_reader_predictions.jsonl", _prediction_rows(reader_ids, 7)
    )
    all_prediction_desc = _desc(all_predictions, len(all_ids))
    reader_prediction_desc = _desc(reader_predictions, len(reader_ids))

    published_path = _json(
        tmp_path / "published.json",
        {
            "artifact_kind": "indra_assembly_paper_published_method_metrics",
            "methods": [
                _published("paper:promoter", "RF promoter", 0.942, 0.014),
                _published("paper:avglen", "RF promoter avglen", 0.942, 0.015),
                _published("paper:orig", "Orig belief readers", 0.917, 0.019),
            ],
        },
    )
    reproduction_path = _json(
        tmp_path / "paper_reproduction.json",
        {
            "artifact_kind": "indra_paper_headline_deterministic_reproduction",
            "inputs": {
                "comparison_gold_manifest": {"sha256": gold_manifest_sha},
                "released_gold": {"sha256": all_sha},
                "reader_gold": {"sha256": reader_sha},
            },
            "runtime": {"python": "test"},
            "outputs": {
                "rf_promoter_minimal_predictions": all_prediction_desc,
                "rf_promoter_avglen_minimal_predictions": all_prediction_desc,
                "orig_belief_readers_minimal_predictions": reader_prediction_desc,
            },
            "results": {
                "rf_2k_d13_type_pmids_promoter_all_sources_specific": _reproduced(
                    "RF promoter", 0.942, 0.014, 0.9415, 0.0138
                ),
                "rf_2k_d13_type_pmids_promoter_avglen_all_sources_specific": _reproduced(
                    "RF promoter avglen", 0.942, 0.015, 0.9416, 0.0146
                ),
                "orig_belief_readers": _reproduced(
                    "Orig belief readers", 0.917, 0.019, 0.9169, 0.0190
                ),
            },
        },
    )

    simple_path = _json(
        tmp_path / "simple.json",
        {
            "artifact_kind": "current_indra_simple_paper_predictions",
            "inputs": {"prediction_targets_manifest": {"sha256": gold_manifest_sha}},
            "arm": {"arm_id": "current.simple", "class": "SimpleScorer"},
            "implementation": {
                "runtime": {"python": "test"},
                "indra_belief_module": {"sha256": SIMPLE_IMPLEMENTATION_SHA},
            },
            "outputs": {"predictions": all_prediction_desc},
        },
    )

    bayesian_arms = [
        _bayes_arm("current.simple.readers", "indra.belief.SimpleScorer", False, False),
        _bayes_arm("current.bayes.source.all", "indra.belief.BayesianScorer", True, False),
        _bayes_arm("current.bayes.subtype.all", "indra.belief.BayesianScorer", True, True),
        _bayes_arm("current.bayes.source.readers", "indra.belief.BayesianScorer", False, False),
        _bayes_arm("current.bayes.subtype.readers", "indra.belief.BayesianScorer", False, True),
    ]
    bayes_path = _json(
        tmp_path / "bayes.json",
        {
            "artifact_kind": "current_indra_bayesian_paper_predictions",
            "inputs": {
                "comparison_manifest": {"sha256": gold_manifest_sha},
                "all_source_gold_and_folds": {"sha256": all_sha},
                "reader_gold_and_folds": {"sha256": reader_sha},
            },
            "implementation": {
                "runtime": {"python": "test"},
                "indra_belief_module": {"sha256": SIMPLE_IMPLEMENTATION_SHA},
            },
            "arms": bayesian_arms,
            "outputs": {
                "current_bayesian_source_oof_all_sources_predictions.jsonl": all_prediction_desc,
                "current_bayesian_source_subtype_oof_all_sources_predictions.jsonl": all_prediction_desc,
                "current_simple_direct_readers_only_predictions.jsonl": reader_prediction_desc,
                "current_bayesian_source_oof_readers_only_predictions.jsonl": reader_prediction_desc,
                "current_bayesian_source_subtype_oof_readers_only_predictions.jsonl": reader_prediction_desc,
            },
        },
    )

    hierarchy_path = _json(
        tmp_path / "hierarchy.json",
        {
            "artifact_kind": "current_indra_simple_hierarchy_paper_predictions",
            "inputs": {
                "comparison_manifest": {"sha256": gold_manifest_sha},
                "all_source_gold_and_folds": {"sha256": all_sha},
                "reader_gold_and_folds": {"sha256": reader_sha},
            },
            "implementation": {
                "runtime": {"python": "test"},
                "indra_belief_module": {"sha256": SIMPLE_IMPLEMENTATION_SHA},
            },
            "arms": [
                {
                    "arm_id": "current.hierarchy.all",
                    "class": "HierarchySimpleScorer",
                    "rows": len(all_ids),
                    "hierarchy_propagation": True,
                },
                {
                    "arm_id": "current.hierarchy.readers",
                    "class": "HierarchySimpleScorer",
                    "rows": len(reader_ids),
                    "hierarchy_propagation": True,
                    "input_sources": ["reach", "sparser", "medscan", "rlimsp", "trips"],
                },
            ],
            "outputs": {
                "current_simple_hierarchy_all_sources_predictions.jsonl": all_prediction_desc,
                "current_simple_hierarchy_readers_only_predictions.jsonl": reader_prediction_desc,
            },
        },
    )

    counts_all_path = _counts_manifest(
        tmp_path / "counts_all.json",
        kind="current_indra_counts_hybrid_paper_predictions",
        panel_id="all_sources_1689",
        gold_key="all_source_gold_and_folds",
        gold_sha=all_sha,
        comparison_sha=gold_manifest_sha,
        prediction=all_prediction_desc,
        reader=False,
    )
    counts_reader_path = _counts_manifest(
        tmp_path / "counts_reader.json",
        kind="current_indra_counts_hybrid_reader_predictions",
        panel_id="readers_only_1676",
        gold_key="reader_gold_and_folds",
        gold_sha=reader_sha,
        comparison_sha=gold_manifest_sha,
        prediction=reader_prediction_desc,
        reader=True,
    )

    production_path = _production_bundle(
        tmp_path,
        targets=targets,
        reader_gold=reader_gold,
        all_predictions=all_predictions,
        reader_predictions=reader_predictions,
        all_ids=all_ids,
        reader_ids=reader_ids,
    )
    scorer_registry_path = _scorer_registry(tmp_path, production_path)

    all_cost_rows = _cost_rows(all_ids)
    # An all-source-only execution can belong to a reader-eligible statement;
    # true-reader cost projection is defined by execution identity, not just by
    # statement eligibility.
    extra_nonreader = copy.deepcopy(all_cost_rows[0])
    extra_nonreader["execution_identity"] = "execution.extra_nonreader"
    extra_nonreader["attempts"][0]["attempt_id"] = "attempt.extra_nonreader"
    extra_nonreader["attempts"][0]["calls"][0]["call_id"] = "call.extra_nonreader"
    all_cost_rows.insert(1, extra_nonreader)
    reader_set = set(reader_ids)
    reader_execution_ids = {
        f"execution.{index}"
        for index, statement_id in enumerate(all_ids)
        if statement_id in reader_set
    }
    reader_cost_rows = [
        row
        for row in all_cost_rows
        if row["execution_identity"] in reader_execution_ids
    ]
    all_cost_path = _jsonl(tmp_path / "llm_all_cost.jsonl", all_cost_rows)
    reader_cost_path = _jsonl(tmp_path / "llm_reader_cost.jsonl", reader_cost_rows)
    all_cost_desc = _cost_desc(
        all_cost_path,
        rows=len(all_cost_rows),
        statements=len(all_ids),
        projection="all_executions",
        relative_to=tmp_path,
    )
    reader_cost_desc = _cost_desc(
        reader_cost_path,
        rows=len(reader_cost_rows),
        statements=len(reader_ids),
        projection="observed_execution_subset",
        relative_to=tmp_path,
    )
    bundle_path = _json(
        tmp_path / "llm_bundle.json",
        {
            "kind": "llm_model_bundle",
            "model_id": "llm.test",
            "run_id": "run.test",
            "implementation": {
                "implementation": "canonical LLM statement aggregation",
                "implementation_digest": "a" * 64,
                "training_data_sha256": None,
                "environment": {"python": "test", "package": "test"},
                "notes": {
                    "dedup": True,
                    "reader_profile": None,
                    "served_model": "llm-test",
                    "provider_model_id": "llm-test",
                },
            },
            "panels": {
                ALL_SOURCE_PANEL: {
                    "prediction_unit": "assembled_statement",
                    "substrate_id": ALL_SOURCE_PANEL,
                    "predictions": _desc(
                        llm_all_predictions, len(all_ids), relative_to=tmp_path
                    ),
                    "cost": all_cost_desc,
                },
                READER_PANEL: {
                    "prediction_unit": "assembled_statement",
                    "substrate_id": READER_PANEL,
                    "predictions": _desc(
                        llm_reader_predictions, len(reader_ids), relative_to=tmp_path
                    ),
                    "cost": reader_cost_desc,
                },
            },
        },
    )
    protocol_path = _json(
        tmp_path / "error_review.json",
        {
            "frozen_at": "2026-07-20T00:00:00Z",
            "error_definition": {
                "primary_threshold": 0.5,
                "operator": "greater_than_or_equal",
            },
        },
    )
    inputs = AssemblyInputs(
        workspace_root=tmp_path,
        gold_manifest=gold_manifest_path,
        paper_reproduction_manifest=reproduction_path,
        published_metrics=published_path,
        current_simple_manifest=simple_path,
        current_bayesian_manifest=bayes_path,
        current_hierarchy_manifest=hierarchy_path,
        current_counts_all_source_manifest=counts_all_path,
        current_counts_reader_manifest=counts_reader_path,
        scorer_registry=scorer_registry_path,
        production_hybrid_manifest=production_path,
        error_review_protocol=protocol_path,
        llm_models=(
            LlmModelInput(
                model_id="llm.test",
                action_id=None,
                run_id="run.test",
                served_model="llm-test",
                provider_model_id="llm-test",
                bundle_manifest=bundle_path,
            ),
            LlmModelInput(
                model_id="llm_gemma_4_26b",
                action_id="gemma_26b_primary",
                run_id="run.gemma26",
                served_model="bedrock-gemma",
                provider_model_id="google.gemma-4-26b-a4b",
                bundle_manifest=tmp_path / "missing_gemma26_manifest.json",
            ),
            LlmModelInput(
                model_id="llm_gemma_4_31b",
                action_id="gemma_31b_primary",
                run_id="run.gemma31",
                served_model="bedrock-gemma-4-31b",
                provider_model_id="google.gemma-4-31b",
                bundle_manifest=tmp_path / "missing_gemma31_manifest.json",
            ),
            LlmModelInput(
                model_id="llm_glm_5",
                action_id="glm_5_primary",
                run_id="run.glm5",
                served_model="bedrock-glm-5",
                provider_model_id="zai.glm-5",
                bundle_manifest=tmp_path / "missing_glm5_manifest.json",
            ),
        ),
        frozen_at="2026-07-20T00:00:00Z",
        bootstrap_resamples=2,
    )
    return inputs, tmp_path / "comparison_spec.json", bundle_path


def _ordered_id_digest(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for statement_id in ids:
        digest.update(
            json.dumps(
                {"statement_id": statement_id}, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _raw_id_digest(ids: list[str]) -> str:
    values = "".join(f"{statement_id}\n" for statement_id in ids).encode()
    return hashlib.sha256(values).hexdigest()


def _published(method_id: str, method: str, mean: float, sd: float):
    return {
        "method_id": method_id,
        "method": method,
        "fold_mean_trapezoidal_pr_auc": mean,
        "fold_population_sd": sd,
    }


def _reproduced(name: str, published_mean: float, published_sd: float, mean: float, sd: float):
    return {
        "display_name": name,
        "published_rounded_mean": published_mean,
        "published_rounded_population_std": published_sd,
        "fold_mean_trapezoidal_pr_auc": mean,
        "fold_population_std": sd,
        "rounded_headline_match": True,
    }


def _bayes_arm(arm_id: str, class_name: str, all_source: bool, subtype: bool):
    return {
        "arm_id": arm_id,
        "class": class_name,
        "panel_id": "all_sources_1689" if all_source else "readers_only_1676",
        "input_sources": (
            ["all"]
            if all_source
            else ["reach", "sparser", "medscan", "rlimsp", "trips"]
        ),
        "subtype_counts": subtype,
        "training_required": class_name != "indra.belief.SimpleScorer",
    }


def _counts_manifest(
    path: Path,
    *,
    kind: str,
    panel_id: str,
    gold_key: str,
    gold_sha: str,
    comparison_sha: str,
    prediction: dict[str, object],
    reader: bool,
) -> Path:
    suffix = "readers_only" if reader else "all_sources"
    outputs = (
        {
            "current_counts_source_only_oof_readers_only_predictions.jsonl": prediction,
            "current_counts_full_features_oof_readers_only_predictions.jsonl": prediction,
        }
        if reader
        else {
            "current_counts_source_only_oof_predictions.jsonl": prediction,
            "current_counts_full_features_oof_predictions.jsonl": prediction,
        }
    )
    panel = {"panel_id": panel_id}
    if reader:
        panel["input_sources"] = ["reach", "sparser", "medscan", "rlimsp", "trips"]
    implementation = (
        {
            "runtime": {"python": "test"},
            "runtime_scorer_crosschecks": {
                "belief_skl": {"sha256": SKLEARN_IMPLEMENTATION_SHA}
            },
        }
        if reader
        else {
            "runtime": {"python": "test"},
            "indra_belief_skl_module": {"sha256": SKLEARN_IMPLEMENTATION_SHA},
        }
    )
    return _json(
        path,
        {
            "artifact_kind": kind,
            "inputs": {
                "comparison_manifest": {"sha256": comparison_sha},
                gold_key: {"sha256": gold_sha},
            },
            "implementation": implementation,
            "panel": panel,
            "arms": [
                {
                    "arm_id": f"current.counts.source_only.{suffix}",
                    "class": "indra.belief.skl.CountsScorer",
                    "panel_id": panel_id,
                    "configuration": "source counts",
                },
                {
                    "arm_id": f"current.counts.full_features.{suffix}",
                    "class": "indra.belief.skl.CountsScorer",
                    "panel_id": panel_id,
                    "configuration": "full features",
                },
                {
                    "arm_id": f"current.hybrid.alias.{suffix}",
                    "class": "indra.belief.skl.HybridScorer",
                    "panel_id": panel_id,
                },
            ],
            "outputs": outputs,
        },
    )


def test_assembles_exact_metrics_spec_and_metrics_accepts_it(tmp_path: Path):
    inputs, output, _bundle = _fixture(tmp_path)
    spec = assemble_spec(inputs, output)
    write_spec(spec, output)
    assert str(tmp_path) not in json.dumps(spec)

    assert set(spec) == {
        "artifact_kind",
        "frozen_at",
        "bootstrap",
        "scorer_registry",
        "metrics",
        "substrates",
    }
    assert spec["scorer_registry"] == {
        "path": "scorers.json",
        "bytes": inputs.scorer_registry.stat().st_size,
        "sha256": hashlib.sha256(inputs.scorer_registry.read_bytes()).hexdigest(),
    }
    assert [panel["substrate_id"] for panel in spec["substrates"]] == [
        ALL_SOURCE_PANEL,
        READER_PANEL,
    ]
    assert [len(panel["arms"]) for panel in spec["substrates"]] == [10, 9]
    assert spec["metrics"]["pareto_metric"] == "fold_mean_trapezoidal_pr_auc"
    for panel in spec["substrates"]:
        assert "gold_resolution" not in panel
        assert panel["released_label_audit"]["released_label_rule"] == (
            "positive if any reviewed evidence was tagged correct; otherwise the "
            "released binary target is negative"
        )
        assert panel["released_label_audit"]["strict_e0_rule"] == (
            "positive if any exact evidence pair is reviewed positive; negative only "
            "if every exact evidence pair is reviewed negative; unresolved otherwise"
        )
        assert panel["released_label_audit"]["strict_e0"]["unresolved"] == 111
        assert panel["released_label_audit"]["released_negative_assumption"][
            "statements"
        ] == 111
        strict_gold_path = panel["strict_e0_resolved_gold"]["path"]
        assert strict_gold_path.endswith(".jsonl")
        assert "strict" in strict_gold_path
        assert {arm["family"] for arm in panel["arms"]} == {
            "paper",
            "current",
            "llm",
        }
        paper_arms = [arm for arm in panel["arms"] if arm["family"] == "paper"]
        assert all(
            arm["label"].startswith("2023 paper semantic reconstruction")
            for arm in paper_arms
        )
        recovered_hybrid = next(
            arm
            for arm in panel["arms"]
            if arm["label"].startswith("Recovered fitted Hybrid artifact")
        )
        assert recovered_hybrid["label"].startswith(
            "Recovered fitted Hybrid artifact"
        )
        assert recovered_hybrid["implementation"]["training_data_sha256"] is None
        assert "training provenance" in recovered_hybrid["implementation"]["notes"]
        assert all(item["status"] == "excluded" for item in panel["excluded_arms"])
        assert {
            "llm_gemma_4_26b",
            "llm_gemma_4_31b",
            "llm_glm_5",
        } <= {item["arm_id"] for item in panel["excluded_arms"]}
    reader_llm = next(
        arm for arm in spec["substrates"][1]["arms"] if arm["family"] == "llm"
    )
    assert reader_llm["cost"]["projection"] == "observed_execution_subset"
    assert reader_llm["cost"]["counterfactual_run_cost"] is False
    assert reader_llm["cost"]["additive_across_panels"] is False
    assert reader_llm["cost"]["pricing"]["resolved_service_tier"] == "standard"
    assert "panel_totals_never_additive=true" in reader_llm["implementation"]["notes"]

    artifact = build_artifact(output)
    assert artifact["provenance"]["source_manifest_path"] == output.name
    assert str(tmp_path) not in json.dumps(artifact)
    assert [panel["substrate_id"] for panel in artifact["substrates"]] == [
        ALL_SOURCE_PANEL,
        READER_PANEL,
    ]
    assert [
        panel["strict_e0_resolved_sensitivity"]["n_evaluable"]
        for panel in artifact["substrates"]
    ] == [1578, 1565]
    for panel in artifact["substrates"]:
        strict = panel["strict_e0_resolved_sensitivity"]
        assert strict["excluded_unresolved"] == 111
        assert all("cost" not in arm and "pareto" not in arm for arm in strict["arms"])


def test_llm_model_expectations_come_from_inputs(tmp_path: Path) -> None:
    inputs, output, _bundle = _fixture(tmp_path)
    wrong_identity = replace(inputs.llm_models[0], model_id="llm_gemma_4_26b")
    with pytest.raises(AssemblyError, match="model_id differs"):
        assemble_spec(
            replace(inputs, llm_models=(wrong_identity,)),
            output,
        )
    duplicate = replace(inputs.llm_models[1], model_id="llm.test")
    with pytest.raises(AssemblyError, match="nonempty ordered set"):
        assemble_spec(
            replace(inputs, llm_models=(inputs.llm_models[0], duplicate)),
            output,
        )


def test_llm_bundle_runtime_identity_is_bound_to_its_declaration(tmp_path: Path) -> None:
    inputs, output, bundle = _fixture(tmp_path)
    value = json.loads(bundle.read_bytes())
    value["implementation"]["notes"]["provider_model_id"] = "google.gemma-4-31b"
    _json(bundle, value)
    with pytest.raises(AssemblyError, match="runtime model identity differs"):
        assemble_spec(inputs, output)


def test_rejects_scorer_registry_that_diverges_from_recovered_artifact(
    tmp_path: Path,
) -> None:
    inputs, output, _bundle = _fixture(tmp_path)
    registry = json.loads(inputs.scorer_registry.read_bytes())
    production = next(
        row
        for row in registry["scorers"]
        if row["scorer_id"] == "indra_db_7dc8bf5_cogex_hybrid_production"
    )
    production["artifact_sha256"] = "0" * 64
    _json(inputs.scorer_registry, registry)

    with pytest.raises(AssemblyError, match="recovered artifact binding differs"):
        assemble_spec(inputs, output)


def test_rejects_current_scorer_manifest_outside_registry(
    tmp_path: Path,
) -> None:
    inputs, output, _bundle = _fixture(tmp_path)
    manifest = json.loads(inputs.current_simple_manifest.read_bytes())
    manifest["implementation"]["indra_belief_module"]["sha256"] = "0" * 64
    _json(inputs.current_simple_manifest, manifest)

    with pytest.raises(AssemblyError, match="does not bind the canonical registry"):
        assemble_spec(inputs, output)


def test_rejects_cross_panel_llm_prediction(tmp_path: Path):
    inputs, output, bundle_path = _fixture(tmp_path)
    bundle = json.loads(bundle_path.read_text())
    bundle["panels"][READER_PANEL]["substrate_id"] = ALL_SOURCE_PANEL
    _json(bundle_path, bundle)
    with pytest.raises(AssemblyError, match="prediction unit or substrate differs"):
        assemble_spec(inputs, output)


@pytest.mark.parametrize("obsolete_field", ["source_run_id", "inputs", "publication", "cost"])
def test_rejects_obsolete_llm_bundle_shapes(tmp_path: Path, obsolete_field: str):
    inputs, output, bundle_path = _fixture(tmp_path)
    bundle = json.loads(bundle_path.read_text())
    bundle[obsolete_field] = {}
    _json(bundle_path, bundle)
    with pytest.raises(AssemblyError, match="unexpected canonical bundle fields"):
        assemble_spec(inputs, output)


def test_rejects_reader_cost_that_is_not_exact_observed_subset(tmp_path: Path):
    inputs, output, bundle_path = _fixture(tmp_path)
    bundle = json.loads(bundle_path.read_text())
    reader_cost_path = tmp_path / bundle["panels"][READER_PANEL]["cost"]["path"]
    rows = [json.loads(line) for line in reader_cost_path.read_text().splitlines()]
    rows[0]["attempts"][0]["calls"][0]["settled_cost_usd_exact"] = "0.002"
    _jsonl(reader_cost_path, rows)
    descriptor = bundle["panels"][READER_PANEL]["cost"]
    descriptor["bytes"] = reader_cost_path.stat().st_size
    descriptor["sha256"] = hashlib.sha256(reader_cost_path.read_bytes()).hexdigest()
    _json(bundle_path, bundle)
    with pytest.raises(AssemblyError, match="exact observed all-source execution subset"):
        assemble_spec(inputs, output)


def test_rejects_llm_panels_with_different_frozen_pricing(tmp_path: Path):
    inputs, output, bundle_path = _fixture(tmp_path)
    bundle = json.loads(bundle_path.read_text())
    bundle["panels"][READER_PANEL]["cost"]["pricing"]["tariff"][
        "output_usd_per_million"
    ] = "2"
    _json(bundle_path, bundle)

    with pytest.raises(AssemblyError, match="disagree on frozen pricing"):
        assemble_spec(inputs, output)


def test_rejects_incomplete_or_out_of_range_predictions(tmp_path: Path):
    inputs, output, bundle_path = _fixture(tmp_path)
    bundle = json.loads(bundle_path.read_text())
    prediction_path = tmp_path / bundle["panels"][READER_PANEL]["predictions"]["path"]
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    rows[0]["probability_correct"] = 1.01
    _jsonl(prediction_path, rows)
    descriptor = bundle["panels"][READER_PANEL]["predictions"]
    descriptor["bytes"] = prediction_path.stat().st_size
    descriptor["sha256"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    _json(bundle_path, bundle)
    with pytest.raises(AssemblyError, match=r"outside \[0, 1\]"):
        assemble_spec(inputs, output)
