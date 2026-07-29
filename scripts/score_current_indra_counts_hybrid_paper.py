#!/usr/bin/env python3
"""Cross-fit current INDRA CountsScorer and a local HybridScorer on paper gold.

This adapter instantiates the *public classes*, not an official fitted INDRA
model.  INDRA 1.24.0 ships ``CountsScorer`` and ``HybridScorer`` without fitted
state.  We therefore freeze two project-local CountsScorer arms under the
released ten-fold statement ledger and wrap the full-feature arm in the current
HybridScorer.  Every test-fold prediction is produced by a model fit without
that fold's statement labels.

The random-forest architecture (2,000 trees, depth 13, seed 4) is fixed from
the paper protocol rather than selected on these predictions.  The source-only
arm uses the current CountsScorer default feature configuration.  The
full-feature arm enables every feature family exposed by INDRA 1.24.0,
including current refinement-ancestor evidence obtained through the verified
release object graph.

The local Hybrid is intentionally separate from the recovered fitted CoGEx
artifact evaluated by the canonical comparison bundle.  On this frozen
eleven-source universe its SimpleScorer fallback has no evidence to score, so
it is a class-path audit rather than a new source of discrimination.
Machine-readable exclusion requirements cover only unfitted official templates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import platform
import struct
import tempfile
import time
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np
from indra.belief import (
    SimpleScorer,
    build_refinements_graph,
    get_ev_for_stmts_from_supports,
)
from indra.belief.skl import CountsScorer, HybridScorer
from indra.statements import Evidence, Statement
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

import score_current_indra_bayesian_paper as bayes
import score_current_indra_hierarchy_paper as hierarchy
import score_current_indra_simple_paper as base


SCHEMA_VERSION = 1
ARTIFACT_KIND = "current_indra_counts_hybrid_paper_predictions"

ARM_COUNTS_SOURCE = "indra_1.24.0_counts_rf_2kd13_source_only_oof_all_sources"
ARM_COUNTS_FULL = "indra_1.24.0_counts_rf_2kd13_full_features_oof_all_sources"
ARM_HYBRID_FULL = "indra_1.24.0_hybrid_local_rf_2kd13_full_features_oof_all_sources"

PREDICTION_FILENAMES = {
    ARM_COUNTS_SOURCE: "current_counts_source_only_oof_predictions.jsonl",
    ARM_COUNTS_FULL: "current_counts_full_features_oof_predictions.jsonl",
    ARM_HYBRID_FULL: "current_hybrid_local_full_features_oof_predictions.jsonl",
}
FIT_FILENAME = "current_counts_oof_fit_provenance.jsonl"
PROVENANCE_FILENAME = "current_counts_hybrid_prediction_provenance.jsonl"
FOLD_METRICS_FILENAME = "current_counts_hybrid_diagnostic_fold_metrics.jsonl"
EXCLUSIONS_FILENAME = "current_counts_hybrid_exclusions.jsonl"
MANIFEST_FILENAME = "current_counts_hybrid_paper_manifest.json"

ALL_SOURCES = bayes.ALL_SOURCES
EXPECTED_ROWS = 1689
EXPECTED_FOLDS = tuple(range(10))
EXPECTED_DIRECT_EVIDENCE = 34_035
EXPECTED_GRAPH_NODES = hierarchy.EXPECTED_GRAPH_NODES
EXPECTED_GRAPH_EDGES = hierarchy.EXPECTED_GRAPH_EDGES

RF_N_ESTIMATORS = 2000
RF_MAX_DEPTH = 13
RF_RANDOM_STATE = 4


class ContractError(ValueError):
    """Raised when an identity, cross-fit, feature, or coverage gate fails."""


@dataclass(frozen=True)
class CountsConfiguration:
    """A frozen local instantiation of the current CountsScorer class."""

    arm_id: str
    config_id: str
    include_more_specific: bool
    use_stmt_type: bool
    use_num_members: bool
    use_num_pmids: bool
    use_promoter: bool
    use_avg_evidence_len: bool
    use_residue_position: bool

    def kwargs(self) -> dict[str, bool]:
        return {
            "include_more_specific": self.include_more_specific,
            "use_stmt_type": self.use_stmt_type,
            "use_num_members": self.use_num_members,
            "use_num_pmids": self.use_num_pmids,
            "use_promoter": self.use_promoter,
            "use_avg_evidence_len": self.use_avg_evidence_len,
            "use_residue_position": self.use_residue_position,
        }


SOURCE_CONFIGURATION = CountsConfiguration(
    arm_id=ARM_COUNTS_SOURCE,
    config_id="current_default_direct_source_counts_only",
    include_more_specific=False,
    use_stmt_type=False,
    use_num_members=False,
    use_num_pmids=False,
    use_promoter=False,
    use_avg_evidence_len=False,
    use_residue_position=False,
)
FULL_CONFIGURATION = CountsConfiguration(
    arm_id=ARM_COUNTS_FULL,
    config_id="all_indra_1.24.0_counts_feature_families",
    include_more_specific=True,
    use_stmt_type=True,
    use_num_members=True,
    use_num_pmids=True,
    use_promoter=True,
    use_avg_evidence_len=True,
    use_residue_position=True,
)
CONFIGURATIONS = (SOURCE_CONFIGURATION, FULL_CONFIGURATION)


def _sha256(path: Path) -> str:
    return base._sha256(path)


def _canonical_bytes(value: Any) -> bytes:
    return base._canonical_bytes(value)


def _ordered_id_sha(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_bytes({"statement_id": row["statement_id"]}) + b"\n")
    return digest.hexdigest()


def _label_sha(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            _canonical_bytes(
                {"label": int(row["label"]), "statement_id": row["statement_id"]}
            )
            + b"\n"
        )
    return digest.hexdigest()


def _matrix_sha(matrix: np.ndarray) -> str:
    value = np.asarray(matrix, dtype="<f8", order="C")
    header = struct.pack("<QQ", *value.shape)
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _row_sha(row: np.ndarray) -> str:
    value = np.asarray(row, dtype="<f8", order="C")
    return hashlib.sha256(struct.pack("<Q", value.size) + value.tobytes()).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert estimator parameters to strict, deterministic JSON values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractError("non-finite estimator parameter")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    return repr(value)


def _new_rf(_fold_id: int, *, n_jobs: int) -> RandomForestClassifier:
    """Return the externally fixed paper RF architecture under a fixed seed."""
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        criterion="gini",
        max_depth=RF_MAX_DEPTH,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=n_jobs,
        random_state=RF_RANDOM_STATE,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    )


def _counts_scorer(
    configuration: CountsConfiguration, estimator: BaseEstimator
) -> CountsScorer:
    return CountsScorer(estimator, list(ALL_SOURCES), **configuration.kwargs())


def _feature_names(scorer: CountsScorer) -> list[str]:
    names = [f"direct_source_count:{source}" for source in scorer.source_list]
    if scorer.include_more_specific:
        names.extend(
            f"more_specific_source_count:{source}" for source in scorer.source_list
        )
    if scorer.use_stmt_type:
        type_names = [
            name for name, _ in sorted(scorer.stmt_type_map.items(), key=lambda item: item[1])
        ]
        names.extend(f"statement_type:{name}" for name in type_names)
    if scorer.use_residue_position:
        names.append("has_residue_and_position")
    if scorer.use_num_members:
        names.append("statement_member_count")
    if scorer.use_num_pmids:
        names.append("direct_unique_pmid_count")
        if scorer.include_more_specific:
            names.append("more_specific_unique_pmid_count")
    if scorer.use_promoter:
        names.append("direct_promoter_sentence_fraction")
    if scorer.use_avg_evidence_len:
        names.append("average_direct_evidence_sentence_length")
    return names


def _feature_contract(configuration: CountsConfiguration) -> dict[str, Any]:
    scorer = _counts_scorer(configuration, RandomForestClassifier(n_estimators=1))
    names = _feature_names(scorer)
    expected = 11 if configuration is SOURCE_CONFIGURATION else 77
    if len(names) != expected:
        raise ContractError(
            f"{configuration.config_id}: feature count {len(names)} != {expected}"
        )
    return {
        "config_id": configuration.config_id,
        "flags": configuration.kwargs(),
        "feature_count": len(names),
        "feature_names": names,
        "feature_names_sha256": hashlib.sha256(_canonical_bytes(names)).hexdigest(),
        "source_list_order": list(ALL_SOURCES),
        "direct_evidence_multiplicity": "raw entries retained by CountsScorer",
        "more_specific_evidence_semantics": (
            "current get_ev_for_stmts_from_supports: transitive graph descendants, "
            "negated evidence excluded, duplicate Evidence objects collapsed by set"
            if configuration.include_more_specific
            else "not used"
        ),
    }


def _evidence_counts(evidence: Sequence[Evidence]) -> dict[str, int]:
    return dict(sorted(Counter(str(item.source_api) for item in evidence).items()))


def _prediction(statement_id: str, probability: float) -> dict[str, Any]:
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ContractError(f"{statement_id}: invalid probability {value!r}")
    return {"probability_correct": value, "statement_id": statement_id}


def _fitted_state_sha(scorer: CountsScorer) -> tuple[str, int]:
    payload = pickle.dumps(scorer, protocol=5)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _cross_fit(
    statements: Sequence[Statement],
    extra_evidence: Sequence[Sequence[Evidence]],
    gold_rows: Sequence[dict[str, Any]],
    *,
    estimator_factory: Callable[[int], BaseEstimator],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Produce OOF predictions; no test label is passed to a fit call."""
    if not (len(statements) == len(extra_evidence) == len(gold_rows)):
        raise ContractError("statement, extra-evidence, and gold lengths differ")
    if [statement.uuid for statement in statements] != [
        row["statement_id"] for row in gold_rows
    ]:
        raise ContractError("statement/gold order differs")
    folds = tuple(sorted({int(row["fold_id"]) for row in gold_rows}))
    if not folds or folds != tuple(range(len(folds))):
        raise ContractError("cross-fit folds must be contiguous from zero")

    prediction_maps: dict[str, dict[str, dict[str, Any]]] = {
        arm_id: {} for arm_id in PREDICTION_FILENAMES
    }
    provenance_maps: dict[str, dict[str, dict[str, Any]]] = {
        arm_id: {} for arm_id in PREDICTION_FILENAMES
    }
    fits: list[dict[str, Any]] = []
    max_hybrid_difference = 0.0
    hybrid_fallback_entries = 0

    for configuration in CONFIGURATIONS:
        contract = _feature_contract(configuration)
        for fold_id in folds:
            train_positions = [
                index for index, row in enumerate(gold_rows) if row["fold_id"] != fold_id
            ]
            test_positions = [
                index for index, row in enumerate(gold_rows) if row["fold_id"] == fold_id
            ]
            if not train_positions or not test_positions:
                raise ContractError(f"fold {fold_id}: empty train or test partition")
            train_gold = [gold_rows[index] for index in train_positions]
            test_gold = [gold_rows[index] for index in test_positions]
            train_ids = {row["statement_id"] for row in train_gold}
            test_ids = {row["statement_id"] for row in test_gold}
            if train_ids.intersection(test_ids):
                raise ContractError(f"fold {fold_id}: train/test statement leakage")

            train_statements = [statements[index] for index in train_positions]
            test_statements = [statements[index] for index in test_positions]
            if configuration.include_more_specific:
                train_extra: list[list[Evidence]] | None = [
                    list(extra_evidence[index]) for index in train_positions
                ]
                test_extra: list[list[Evidence]] | None = [
                    list(extra_evidence[index]) for index in test_positions
                ]
            else:
                train_extra = None
                test_extra = None

            estimator = estimator_factory(fold_id)
            scorer = _counts_scorer(configuration, estimator)
            feature_names = _feature_names(scorer)
            if feature_names != contract["feature_names"]:
                raise ContractError(f"{configuration.config_id}: feature order drift")
            train_matrix = scorer.stmts_to_matrix(train_statements, train_extra)
            test_matrix = scorer.stmts_to_matrix(test_statements, test_extra)
            expected_columns = contract["feature_count"]
            if train_matrix.shape != (len(train_positions), expected_columns):
                raise ContractError(f"fold {fold_id}: invalid train feature shape")
            if test_matrix.shape != (len(test_positions), expected_columns):
                raise ContractError(f"fold {fold_id}: invalid test feature shape")
            train_labels = np.asarray([row["label"] for row in train_gold], dtype=int)
            if set(train_labels.tolist()) != {0, 1}:
                raise ContractError(f"fold {fold_id}: training fold lacks a class")

            # CountsScorer.fit accepts an already constructed matrix.  The matrix was
            # produced immediately above by the current CountsScorer implementation;
            # passing it avoids a second, potentially expensive feature traversal.
            scorer.fit(train_matrix, train_labels)
            classes = [int(value) for value in scorer.model.classes_]
            if classes != [0, 1]:
                raise ContractError(f"fold {fold_id}: estimator class order is {classes}")
            matrix_scores = np.asarray(scorer.predict_proba(test_matrix)[:, 1], dtype=float)
            class_scores = np.asarray(
                scorer.score_statements(test_statements, test_extra), dtype=float
            )
            if not np.array_equal(matrix_scores, class_scores):
                raise ContractError("CountsScorer matrix and class scoring paths diverge")

            hybrid_scores: np.ndarray | None = None
            unknown_by_test: list[list[Evidence]] = []
            if configuration is FULL_CONFIGURATION:
                simple = SimpleScorer()
                hybrid = HybridScorer(scorer, simple)
                hybrid.check_prior_probs(test_statements)
                hybrid_scores = np.asarray(
                    hybrid.score_statements(test_statements, test_extra), dtype=float
                )
                for statement, extras in zip(
                    test_statements, test_extra or [[] for _ in test_statements], strict=True
                ):
                    unknown = [
                        item
                        for item in [*statement.evidence, *extras]
                        if item.source_api not in ALL_SOURCES
                    ]
                    unknown_by_test.append(unknown)
                    hybrid_fallback_entries += len(unknown)
                fold_difference = float(np.max(np.abs(hybrid_scores - class_scores)))
                max_hybrid_difference = max(max_hybrid_difference, fold_difference)
                if fold_difference > 4 * np.finfo(float).eps:
                    raise ContractError(
                        f"fold {fold_id}: no-fallback Hybrid differs from Counts by "
                        f"{fold_difference}"
                    )

            fitted_sha, fitted_bytes = _fitted_state_sha(scorer)
            estimator_parameters = {
                key: _json_value(value)
                for key, value in sorted(scorer.model.get_params(deep=False).items())
            }
            fits.append(
                {
                    "arm_id": configuration.arm_id,
                    "config_id": configuration.config_id,
                    "estimator_class": (
                        f"{type(scorer.model).__module__}.{type(scorer.model).__qualname__}"
                    ),
                    "estimator_parameters": estimator_parameters,
                    "feature_names_sha256": contract["feature_names_sha256"],
                    "fit_api": "indra.belief.skl.CountsScorer.fit",
                    "fit_fold_id": fold_id,
                    "fitted_state_pickle_bytes": fitted_bytes,
                    "fitted_state_pickle_protocol": 5,
                    "fitted_state_sha256": fitted_sha,
                    "source_list_order": list(ALL_SOURCES),
                    "test_feature_matrix_sha256": _matrix_sha(test_matrix),
                    "test_feature_shape": list(test_matrix.shape),
                    "test_label_count_not_passed_to_fit": len(test_gold),
                    "test_statement_count": len(test_gold),
                    "test_statement_id_sha256": _ordered_id_sha(test_gold),
                    "train_feature_matrix_sha256": _matrix_sha(train_matrix),
                    "train_feature_shape": list(train_matrix.shape),
                    "train_label_sha256": _label_sha(train_gold),
                    "train_negative": int(len(train_labels) - train_labels.sum()),
                    "train_positive": int(train_labels.sum()),
                    "train_statement_count": len(train_gold),
                    "train_statement_id_sha256": _ordered_id_sha(train_gold),
                    "train_test_statement_intersection": 0,
                }
            )

            for local_index, (position, gold, counts_score) in enumerate(
                zip(test_positions, test_gold, class_scores, strict=True)
            ):
                statement_id = gold["statement_id"]
                if statement_id in prediction_maps[configuration.arm_id]:
                    raise ContractError(f"duplicate OOF prediction {statement_id}")
                counts_prediction = _prediction(statement_id, counts_score)
                prediction_maps[configuration.arm_id][statement_id] = counts_prediction
                direct = statements[position].evidence
                inherited = (
                    list(extra_evidence[position])
                    if configuration.include_more_specific
                    else []
                )
                provenance_maps[configuration.arm_id][statement_id] = {
                    "arm_id": configuration.arm_id,
                    "config_id": configuration.config_id,
                    "direct_evidence_count": len(direct),
                    "direct_source_counts": _evidence_counts(direct),
                    "feature_vector_sha256": _row_sha(test_matrix[local_index]),
                    "fit_fold_id": fold_id,
                    "more_specific_evidence_count": len(inherited),
                    "more_specific_source_counts": _evidence_counts(inherited),
                    "probability_correct": counts_prediction["probability_correct"],
                    "statement_id": statement_id,
                }

                if configuration is FULL_CONFIGURATION:
                    assert hybrid_scores is not None
                    hybrid_prediction = _prediction(
                        statement_id, hybrid_scores[local_index]
                    )
                    prediction_maps[ARM_HYBRID_FULL][statement_id] = hybrid_prediction
                    unknown = unknown_by_test[local_index]
                    provenance_maps[ARM_HYBRID_FULL][statement_id] = {
                        "arm_id": ARM_HYBRID_FULL,
                        "config_id": configuration.config_id,
                        "counts_probability_correct": counts_prediction[
                            "probability_correct"
                        ],
                        "direct_evidence_count": len(direct),
                        "direct_source_counts": _evidence_counts(direct),
                        "feature_vector_sha256": _row_sha(test_matrix[local_index]),
                        "fit_fold_id": fold_id,
                        "more_specific_evidence_count": len(inherited),
                        "more_specific_source_counts": _evidence_counts(inherited),
                        "probability_correct": hybrid_prediction[
                            "probability_correct"
                        ],
                        "simple_fallback_evidence_count": len(unknown),
                        "simple_fallback_source_counts": _evidence_counts(unknown),
                        "statement_id": statement_id,
                    }

    predictions: dict[str, list[dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    expected_ids = [row["statement_id"] for row in gold_rows]
    for arm_id in PREDICTION_FILENAMES:
        if set(prediction_maps[arm_id]) != set(expected_ids):
            raise ContractError(f"{arm_id}: incomplete OOF coverage")
        predictions[arm_id] = [prediction_maps[arm_id][item] for item in expected_ids]
        provenance.extend(provenance_maps[arm_id][item] for item in expected_ids)
    return predictions, provenance, fits, {
        "hybrid_simple_fallback_evidence_entries": hybrid_fallback_entries,
        "max_abs_hybrid_minus_counts_probability": max_hybrid_difference,
    }


def _module_descriptor(obj: Any) -> dict[str, Any]:
    path = Path(inspect.getsourcefile(obj) or "").resolve()
    if not path.is_file():
        raise ContractError(f"cannot locate source for {obj!r}")
    return {
        "bytes": path.stat().st_size,
        "path": base._display_path(path),
        "sha256": _sha256(path),
    }


def _registry_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = base._read_json(path)
    scorers = registry.get("scorers")
    if not isinstance(scorers, list):
        raise ContractError("scorer registry has no scorer rows")
    by_id = {
        row.get("scorer_id"): row for row in scorers if isinstance(row, dict)
    }
    required = {
        "indra_1.24.0_counts_unfitted",
        "indra_1.24.0_hybrid_unfitted",
        "indra_db_7dc8bf5_cogex_hybrid_production",
    }
    if not required.issubset(by_id):
        raise ContractError("scorer registry is missing Counts/Hybrid identities")
    runtime_skl = _module_descriptor(CountsScorer)
    if by_id["indra_1.24.0_counts_unfitted"].get("implementation_sha256") != runtime_skl[
        "sha256"
    ]:
        raise ContractError("runtime CountsScorer differs from frozen registry")
    if by_id["indra_1.24.0_hybrid_unfitted"].get("implementation_sha256") != runtime_skl[
        "sha256"
    ]:
        raise ContractError("runtime HybridScorer differs from frozen registry")
    return (
        {
            "bytes": path.stat().st_size,
            "path": base._display_path(path),
            "sha256": _sha256(path),
            "verification": "pass",
        },
        by_id,
    )


def _exclusions() -> list[dict[str, Any]]:
    return [
        {
            "admission_requirement": {
                "acceptable_resolution": (
                    "freeze an estimator class and parameters, source-list order, feature "
                    "flags, fitted state, sklearn runtime, training identities, labels, and "
                    "split protocol; use OOF predictions on this panel"
                ),
                "currently_satisfied_by": [ARM_COUNTS_SOURCE, ARM_COUNTS_FULL],
            },
            "blockers": [
                "INDRA ships the CountsScorer class without a bundled estimator or fitted state",
                "the class alone does not identify a benchmark point",
            ],
            "excluded_scorer_id": "indra_1.24.0_counts_unfitted",
            "status": "excluded_as_unidentified_official_fitted_model",
        }
    ]


def _descriptor(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": base._display_path(path),
        "rows": len(rows),
        "sha256": _sha256(path),
    }


def _publish_bundle(
    output_dir: Path,
    row_outputs: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = {name: output_dir / name for name in row_outputs}
    final[MANIFEST_FILENAME] = output_dir / MANIFEST_FILENAME
    existing = [str(path) for path in final.values() if os.path.lexists(path)]
    if existing:
        raise FileExistsError("refusing to overwrite existing outputs: " + ", ".join(existing))
    with tempfile.TemporaryDirectory(prefix=".current-counts-", dir=output_dir) as tmp:
        stage = Path(tmp)
        staged: dict[str, Path] = {}
        for name, rows in row_outputs.items():
            path = stage / name
            base._write_jsonl(path, rows)
            staged[name] = path
        manifest_path = stage / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staged[MANIFEST_FILENAME] = manifest_path
        published: list[Path] = []
        try:
            for name in [*row_outputs, MANIFEST_FILENAME]:
                os.link(staged[name], final[name])
                published.append(final[name])
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise


def materialize(
    *,
    pickle_path: Path,
    paper_manifest_path: Path,
    targets_path: Path,
    comparison_manifest_path: Path,
    all_gold_path: Path,
    scorer_registry_path: Path,
    output_dir: Path,
    n_jobs: int = 1,
) -> dict[str, Any]:
    started = time.perf_counter()
    (
        pickle_path,
        paper_manifest_path,
        targets_path,
        comparison_manifest_path,
        all_gold_path,
        scorer_registry_path,
    ) = [
        path.resolve()
        for path in (
            pickle_path,
            paper_manifest_path,
            targets_path,
            comparison_manifest_path,
            all_gold_path,
            scorer_registry_path,
        )
    ]
    output_dir = output_dir.resolve()
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ContractError("n_jobs must be a non-zero integer")

    paper_manifest = base._read_json(paper_manifest_path)
    comparison_manifest = base._read_json(comparison_manifest_path)
    targets = base._read_targets(targets_path)
    gold = bayes._read_gold(all_gold_path)
    if len(targets) != EXPECTED_ROWS or len(gold) != EXPECTED_ROWS:
        raise ContractError("frozen all-source panel must contain exactly 1,689 rows")
    if [row["statement_id"] for row in targets] != [
        row["statement_id"] for row in gold
    ]:
        raise ContractError("target and all-source gold order differs")
    if tuple(sorted({row["fold_id"] for row in gold})) != EXPECTED_FOLDS:
        raise ContractError("frozen gold does not contain exactly ten folds")

    pickle_input = hierarchy._pickle_descriptor(pickle_path, paper_manifest)
    target_input = bayes._verify_output_descriptor(
        targets_path, comparison_manifest, "paper_prediction_targets", targets
    )
    gold_input = bayes._verify_output_descriptor(
        all_gold_path, comparison_manifest, "paper_released_gold", gold
    )
    registry_input, _registry_rows = _registry_contract(scorer_registry_path)

    roots = hierarchy._load_verified_pickle(pickle_path)
    statements = hierarchy._select_targets(roots, targets)
    if sum(len(statement.evidence) for statement in statements) != EXPECTED_DIRECT_EVIDENCE:
        raise ContractError("target direct-evidence count differs from frozen contract")
    direct_sources = {
        evidence.source_api for statement in statements for evidence in statement.evidence
    }
    if direct_sources != set(ALL_SOURCES):
        raise ContractError(f"direct source universe differs: {sorted(direct_sources)}")

    graph = build_refinements_graph(roots)
    graph_shape = (graph.number_of_nodes(), graph.number_of_edges())
    if graph_shape != (EXPECTED_GRAPH_NODES, EXPECTED_GRAPH_EDGES):
        raise ContractError(f"current refinement graph shape differs: {graph_shape}")
    if not nx.is_directed_acyclic_graph(graph):
        raise ContractError("current refinement graph contains a cycle")
    extra_evidence = get_ev_for_stmts_from_supports(statements, graph)
    inherited_sources = {
        evidence.source_api for items in extra_evidence for evidence in items
    }
    if inherited_sources.difference(ALL_SOURCES):
        raise ContractError(
            f"inherited source universe differs: {sorted(inherited_sources)}"
        )

    predictions, provenance, fit_rows, hybrid_audit = _cross_fit(
        statements,
        extra_evidence,
        gold,
        estimator_factory=lambda fold_id: _new_rf(fold_id, n_jobs=n_jobs),
    )
    if hybrid_audit["hybrid_simple_fallback_evidence_entries"] != 0:
        raise ContractError("local Hybrid unexpectedly activated SimpleScorer fallback")

    fold_metrics: list[dict[str, Any]] = []
    diagnostic_summaries: dict[str, Any] = {}
    for arm_id, rows in predictions.items():
        arm_fold, summary = bayes._diagnostics(arm_id, rows, gold)
        fold_metrics.extend(arm_fold)
        diagnostic_summaries[arm_id] = summary

    exclusions = _exclusions()
    row_outputs = {
        PREDICTION_FILENAMES[arm_id]: rows for arm_id, rows in predictions.items()
    }
    row_outputs[FIT_FILENAME] = fit_rows
    row_outputs[PROVENANCE_FILENAME] = provenance
    row_outputs[FOLD_METRICS_FILENAME] = fold_metrics
    row_outputs[EXCLUSIONS_FILENAME] = exclusions

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".current-counts-descriptors-", dir=output_dir) as tmp:
        stage = Path(tmp)
        outputs: dict[str, Any] = {}
        for name, rows in row_outputs.items():
            path = stage / name
            base._write_jsonl(path, rows)
            outputs[name] = _descriptor(path, rows)
            outputs[name]["path"] = base._display_path(output_dir / name)

    resource_path = bayes._resource_path()
    forest_module = Path(inspect.getsourcefile(RandomForestClassifier) or "").resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "claim_scope": {
            "lane": "P",
            "output_semantics": "probability_statement_correct",
            "paid_inference_calls": 0,
            "prediction_unit": "assembled_statement",
            "status": "project_local_current_class_instantiations",
            "not_claimed": [
                "a bundled or default fitted INDRA CountsScorer",
                "the fitted CoGEx production HybridScorer",
                "historical bit parity with the paper notebook",
            ],
        },
        "arms": [
            {
                "arm_id": ARM_COUNTS_SOURCE,
                "class": "indra.belief.skl.CountsScorer",
                "configuration": SOURCE_CONFIGURATION.config_id,
                "official_class_local_fitted_state": True,
                "panel_id": "all_sources_1689",
                "training": "ten-fold out-of-fold statement-label fitting",
            },
            {
                "arm_id": ARM_COUNTS_FULL,
                "class": "indra.belief.skl.CountsScorer",
                "configuration": FULL_CONFIGURATION.config_id,
                "official_class_local_fitted_state": True,
                "panel_id": "all_sources_1689",
                "training": "ten-fold out-of-fold statement-label fitting",
            },
            {
                "arm_id": ARM_HYBRID_FULL,
                "class": "indra.belief.skl.HybridScorer",
                "counts_component": ARM_COUNTS_FULL,
                "official_class_local_fitted_state": True,
                "panel_id": "all_sources_1689",
                "production_cogex_artifact": False,
                "simple_component": "indra.belief.SimpleScorer bundled defaults",
                "training": "reuses the fold-local CountsScorer fitted without test labels",
            },
        ],
        "panel": {
            "panel_id": "all_sources_1689",
            "rows": EXPECTED_ROWS,
            "gold": "paper released compatibility statement labels",
            "folds": 10,
            "input_sources": list(ALL_SOURCES),
            "input_projection": (
                "all canonical direct evidence; full arms additionally use current "
                "non-negated, deduplicated evidence from every refinement descendant"
            ),
        },
        "cross_fit_contract": {
            "fold_source": "frozen paper_released_gold fold_id",
            "fit_labels": "released statement correctness labels from nine training folds",
            "test_labels_passed_to_fit": False,
            "test_fold_predictions": "one prediction from its complementary nine-fold fit",
            "hyperparameter_selection": (
                "RF 2k/depth-13/seed-4 fixed from the external paper protocol; source-only "
                "and all-current-feature configurations frozen by the scorer registry; no "
                "test-fold metric selected an estimator or feature"
            ),
            "global_label_free_graph_input": True,
            "model_fits": len(fit_rows),
        },
        "estimator_contract": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "max_features": "sqrt",
            "random_state": RF_RANDOM_STATE,
            "n_jobs": n_jobs,
            "parameters_are_recorded_per_fit": True,
        },
        "feature_contracts": {
            configuration.config_id: _feature_contract(configuration)
            for configuration in CONFIGURATIONS
        },
        "current_class_semantics": {
            "direct_counts": "raw Evidence multiplicity, including negated entries",
            "more_specific_counts": (
                "current helper's transitive descendants; negated entries excluded; "
                "duplicate Evidence objects collapsed"
            ),
            "hybrid_formula": "1-(1-counts_probability)*(1-simple_fallback_probability)",
            "hybrid_audit": hybrid_audit,
            "hybrid_interpretation": (
                "all visible direct and inherited evidence sources are in the fitted Counts "
                "source list, so the Simple fallback receives zero evidence; this local arm "
                "is performance-redundant with the full Counts arm up to floating arithmetic"
            ),
        },
        "coverage": {
            "target_statements": EXPECTED_ROWS,
            "predictions_per_arm": EXPECTED_ROWS,
            "missing_predictions": 0,
            "invalid_predictions": 0,
            "direct_evidence_entries": EXPECTED_DIRECT_EVIDENCE,
            "more_specific_unique_nonnegated_evidence_entries": sum(
                len(items) for items in extra_evidence
            ),
            "statements_with_more_specific_evidence": sum(
                bool(items) for items in extra_evidence
            ),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
        },
        "diagnostic_metrics": diagnostic_summaries,
        "inputs": {
            "canonical_object_graph_pickle": pickle_input,
            "paper_manifest": {
                "bytes": paper_manifest_path.stat().st_size,
                "path": base._display_path(paper_manifest_path),
                "sha256": _sha256(paper_manifest_path),
            },
            "prediction_targets": target_input,
            "all_source_gold_and_folds": gold_input,
            "comparison_manifest": {
                "bytes": comparison_manifest_path.stat().st_size,
                "path": base._display_path(comparison_manifest_path),
                "sha256": _sha256(comparison_manifest_path),
            },
            "scorer_registry": registry_input,
        },
        "implementation": {
            "adapter": {
                "bytes": Path(__file__).resolve().stat().st_size,
                "path": base._display_path(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "indra_belief_skl_module": _module_descriptor(CountsScorer),
            "indra_belief_module": _module_descriptor(SimpleScorer),
            "sklearn_forest_module": {
                "bytes": forest_module.stat().st_size,
                "path": base._display_path(forest_module),
                "sha256": _sha256(forest_module),
            },
            "bundled_simple_prior_resource": {
                "bytes": resource_path.stat().st_size,
                "path": base._display_path(resource_path),
                "sha256": _sha256(resource_path),
            },
            "package_versions": {
                "indra": importlib.metadata.version("indra"),
                "networkx": importlib.metadata.version("networkx"),
                "numpy": importlib.metadata.version("numpy"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
            },
            "runtime": {
                "executable": os.path.realpath(os.sys.executable),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            },
        },
        "exclusion_contract": {
            "machine_readable_output": EXCLUSIONS_FILENAME,
            "rows": len(exclusions),
            "production_point_materialized": False,
        },
        "outputs": outputs,
        "runtime_observation": {
            "cost_scope": "local CPU execution; excludes released data and curation costs",
            "inference_usd": 0.0,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    _publish_bundle(output_dir, row_outputs, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pickle-path", default=root / "data/benchmark/indra_benchmark_corpus.pkl", type=Path
    )
    parser.add_argument(
        "--paper-manifest-path",
        default=root / "data/benchmark/indra_paper_2023.manifest.json",
        type=Path,
    )
    comparison = root / "data/results/indra_paper_comparison_gold_20260717"
    parser.add_argument(
        "--targets-path", default=comparison / "paper_prediction_targets.jsonl", type=Path
    )
    parser.add_argument(
        "--comparison-manifest-path",
        default=comparison / "paper_comparison_gold_manifest.json",
        type=Path,
    )
    parser.add_argument(
        "--all-gold-path", default=comparison / "paper_released_gold.jsonl", type=Path
    )
    parser.add_argument(
        "--scorer-registry-path",
        default=root / "data/comparison/scorers.json",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=root / "data/results/current_indra_counts_hybrid_paper_20260718",
        type=Path,
    )
    parser.add_argument("--n-jobs", default=1, type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = materialize(
        pickle_path=args.pickle_path,
        paper_manifest_path=args.paper_manifest_path,
        targets_path=args.targets_path,
        comparison_manifest_path=args.comparison_manifest_path,
        all_gold_path=args.all_gold_path,
        scorer_registry_path=args.scorer_registry_path,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
