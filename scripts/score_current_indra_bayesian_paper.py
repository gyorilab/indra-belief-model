#!/usr/bin/env python3
"""Cross-fit current INDRA Bayesian scorers on the frozen paper benchmark.

This adapter instantiates the optional ``indra.belief.BayesianScorer`` family
using only exact evidence-pair curations from training-fold statements.  The
test fold's evidence curations are excluded before each fit.  It emits two
current-library arms on each primary paper panel:

* source-level positive/negative pair counts; and
* source plus current-library evidence-subtype counts.

The 1,689-statement all-source panel retains the frozen eleven evidence
sources.  The 1,676-statement reader panel is both row- and input-restricted to
the paper's five readers.  A row-restricted but all-evidence ``SimpleScorer``
projection is emitted only as an explicitly named sensitivity.  Every metric-
ready prediction file has exactly two fields: ``statement_id`` and
``probability_correct``.

No pseudocount is added.  ``BayesianScorer``'s shipped behavior is retained:
systematic error is fixed at 0.05; a fitted source/subtype random error is
derived from the observed positive fraction; sources absent from a training
fold fall back to the bundled default source prior; and unseen or null subtypes
fall back to the applicable source prior.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from indra.belief import BayesianScorer, SimpleScorer, tag_evidence_subtype
from indra.statements import Evidence, Statement, stmts_from_json
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

import score_current_indra_simple_paper as base


SCHEMA_VERSION = 1
ARTIFACT_KIND = "current_indra_bayesian_paper_predictions"
ALL_SOURCES = (
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
)
READER_SOURCES = ("reach", "sparser", "medscan", "rlimsp", "trips")
PANEL_ALL = "all_sources_1689"
PANEL_READERS = "readers_only_1676"
PANEL_READER_SENSITIVITY = "reader_eligible_all_evidence_sensitivity"

ARM_SIMPLE_ALL = "indra_1.24.0_simple_direct_all_sources"
ARM_SIMPLE_READERS = "indra_1.24.0_simple_direct_readers_only"
ARM_SIMPLE_READER_SENSITIVITY = (
    "indra_1.24.0_simple_direct_reader_eligible_all_evidence_sensitivity"
)
ARM_BAYES_SOURCE_ALL = "indra_1.24.0_bayesian_source_oof_all_sources"
ARM_BAYES_SUBTYPE_ALL = "indra_1.24.0_bayesian_source_subtype_oof_all_sources"
ARM_BAYES_SOURCE_READERS = "indra_1.24.0_bayesian_source_oof_readers_only"
ARM_BAYES_SUBTYPE_READERS = (
    "indra_1.24.0_bayesian_source_subtype_oof_readers_only"
)

PREDICTION_FILENAMES = {
    ARM_SIMPLE_ALL: "current_simple_direct_all_sources_predictions.jsonl",
    ARM_SIMPLE_READERS: "current_simple_direct_readers_only_predictions.jsonl",
    ARM_SIMPLE_READER_SENSITIVITY: (
        "current_simple_direct_reader_eligible_all_evidence_sensitivity_predictions.jsonl"
    ),
    ARM_BAYES_SOURCE_ALL: "current_bayesian_source_oof_all_sources_predictions.jsonl",
    ARM_BAYES_SUBTYPE_ALL: (
        "current_bayesian_source_subtype_oof_all_sources_predictions.jsonl"
    ),
    ARM_BAYES_SOURCE_READERS: (
        "current_bayesian_source_oof_readers_only_predictions.jsonl"
    ),
    ARM_BAYES_SUBTYPE_READERS: (
        "current_bayesian_source_subtype_oof_readers_only_predictions.jsonl"
    ),
}
FIT_FILENAME = "current_bayesian_oof_fit_provenance.jsonl"
PROVENANCE_FILENAME = "current_bayesian_prediction_provenance.jsonl"
FOLD_METRICS_FILENAME = "current_bayesian_diagnostic_fold_metrics.jsonl"
MANIFEST_FILENAME = "current_bayesian_paper_manifest.json"

GOLD_FIELDS = {"fold_id", "label", "statement_id"}
EVIDENCE_FIELDS = {
    "adjudication_id",
    "canonical_corpus_row_index",
    "conflict_resolution",
    "corpus_evidence_entry_count",
    "corpus_evidence_json_sha256s",
    "corpus_evidence_positions",
    "corpus_evidence_text_present",
    "curation_count",
    "curations",
    "eligible_position",
    "evidence_gold_label",
    "identity_kind",
    "needed_to_resolve_statement",
    "paper_statement_hash",
    "queue_item_id",
    "review_status",
    "same_pair_conflict",
    "source_apis",
    "source_hash",
}


class ContractError(ValueError):
    """Raised when a frozen identity, fold, fit, or coverage gate fails."""


@dataclass(frozen=True)
class PairObservation:
    adjudication_id: str
    eligible_position: int
    label: int
    source: str
    source_hash: str
    concrete_subtypes: tuple[str, ...]
    raw_entry_count: int


@dataclass(frozen=True)
class Panel:
    panel_id: str
    sources: tuple[str, ...]
    positions: tuple[int, ...]
    gold_rows: tuple[dict[str, Any], ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"could not open {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ContractError(f"{path}:{line_number}: blank lines are forbidden")
            try:
                row = json.loads(line, object_pairs_hook=base._strict_pairs)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    if not rows:
        raise ContractError(f"{path}: expected at least one row")
    return rows


def _verify_output_descriptor(
    path: Path,
    manifest: dict[str, Any],
    output_key: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    try:
        descriptor = manifest["outputs"][output_key]
    except (KeyError, TypeError) as exc:
        raise ContractError(f"manifest missing outputs.{output_key}") from exc
    if not isinstance(descriptor, dict):
        raise ContractError(f"manifest outputs.{output_key} must be an object")
    observed_sha = base._sha256(path)
    if descriptor.get("sha256") != observed_sha:
        raise ContractError(f"{path}: digest mismatch against outputs.{output_key}")
    if descriptor.get("rows") != len(rows):
        raise ContractError(f"{path}: row mismatch against outputs.{output_key}")
    return {
        "path": base._display_path(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "sha256": observed_sha,
        "verification": "pass",
    }


def _read_gold(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != GOLD_FIELDS:
            raise ContractError(f"{path}:{index + 1}: gold fields are not exact")
        statement_id = row["statement_id"]
        if not isinstance(statement_id, str) or not statement_id or statement_id in seen:
            raise ContractError(f"{path}:{index + 1}: invalid/duplicate statement_id")
        if isinstance(row["label"], bool) or row["label"] not in (0, 1):
            raise ContractError(f"{path}:{index + 1}: label must be integer 0/1")
        if (
            isinstance(row["fold_id"], bool)
            or not isinstance(row["fold_id"], int)
            or row["fold_id"] not in range(10)
        ):
            raise ContractError(f"{path}:{index + 1}: fold_id must be integer 0..9")
        seen.add(statement_id)
    if set(row["fold_id"] for row in rows) != set(range(10)):
        raise ContractError(f"{path}: expected all ten folds")
    return rows


def _decode_statements(statement_jsons: Sequence[dict[str, Any]]) -> list[Statement]:
    statements: list[Statement] = []
    for index, statement_json in enumerate(statement_jsons):
        scorer_json = dict(statement_json)
        scorer_json.pop("belief", None)
        decoded = stmts_from_json([scorer_json])
        if len(decoded) != 1:
            raise ContractError(f"target {index}: INDRA did not decode exactly one statement")
        statements.append(decoded[0])
    return statements


def _project_statement(statement: Statement, allowed_sources: set[str]) -> Statement:
    projected = copy.copy(statement)
    projected.evidence = [
        evidence for evidence in statement.evidence if evidence.source_api in allowed_sources
    ]
    return projected


def _subtype_identity(evidence: Evidence) -> tuple[str, str | None]:
    source, subtype = tag_evidence_subtype(evidence)
    if not isinstance(source, str) or not source:
        raise ContractError("current tag_evidence_subtype returned an invalid source")
    if subtype is not None and (not isinstance(subtype, str) or not subtype):
        raise ContractError("current tag_evidence_subtype returned an invalid subtype")
    return source, subtype


def _pair_observations(
    rows: Sequence[dict[str, Any]],
    statements: Sequence[Statement],
    targets: Sequence[dict[str, Any]],
) -> tuple[list[PairObservation], dict[str, Any]]:
    observations: list[PairObservation] = []
    seen_ids: set[str] = set()
    reviewed_raw_entries = 0
    multi_subtype_pairs = 0
    subtype_pair_cells = 0
    status_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        if set(row) != EVIDENCE_FIELDS:
            raise ContractError(f"evidence row {row_index}: fields are not exact")
        adjudication_id = row["adjudication_id"]
        if (
            not isinstance(adjudication_id, str)
            or not adjudication_id
            or adjudication_id in seen_ids
        ):
            raise ContractError(f"evidence row {row_index}: invalid/duplicate adjudication_id")
        seen_ids.add(adjudication_id)
        position = row["eligible_position"]
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < len(statements):
            raise ContractError(f"{adjudication_id}: invalid eligible_position")
        if row["canonical_corpus_row_index"] != targets[position]["canonical_corpus_row_index"]:
            raise ContractError(f"{adjudication_id}: canonical corpus row mismatch")
        if row["identity_kind"] != "statement_source_hash_pair":
            raise ContractError(f"{adjudication_id}: unexpected pair identity kind")
        source_apis = row["source_apis"]
        if not isinstance(source_apis, list) or len(source_apis) != 1:
            raise ContractError(f"{adjudication_id}: expected exactly one source_api")
        source = source_apis[0]
        if source not in ALL_SOURCES:
            raise ContractError(f"{adjudication_id}: source {source!r} outside frozen universe")
        positions = row["corpus_evidence_positions"]
        if not isinstance(positions, list) or not positions:
            raise ContractError(f"{adjudication_id}: empty corpus evidence positions")
        if len(positions) != row["corpus_evidence_entry_count"]:
            raise ContractError(f"{adjudication_id}: evidence multiplicity mismatch")
        statement = statements[position]
        evidence_items: list[Evidence] = []
        for evidence_position in positions:
            if (
                isinstance(evidence_position, bool)
                or not isinstance(evidence_position, int)
                or not 0 <= evidence_position < len(statement.evidence)
            ):
                raise ContractError(f"{adjudication_id}: invalid evidence position")
            evidence = statement.evidence[evidence_position]
            if evidence.source_api != source:
                raise ContractError(f"{adjudication_id}: source identity mismatch")
            if str(evidence.get_source_hash()) != row["source_hash"]:
                raise ContractError(f"{adjudication_id}: source-hash identity mismatch")
            evidence_items.append(evidence)
        subtype_identities = {_subtype_identity(evidence) for evidence in evidence_items}
        if any(tag_source != source for tag_source, _ in subtype_identities):
            raise ContractError(f"{adjudication_id}: tagged subtype source mismatch")
        concrete_subtypes = tuple(
            sorted({subtype for _, subtype in subtype_identities if subtype is not None})
        )
        label = row["evidence_gold_label"]
        status = row["review_status"]
        status_counts[str(status)] += 1
        if label is None:
            if status != "unreviewed" or row["curation_count"] != 0 or row["curations"]:
                raise ContractError(f"{adjudication_id}: invalid unreviewed row")
            continue
        if isinstance(label, bool) or label not in (0, 1):
            raise ContractError(f"{adjudication_id}: evidence label must be 0/1/null")
        curations = row["curations"]
        if not isinstance(curations, list) or len(curations) != row["curation_count"] or not curations:
            raise ContractError(f"{adjudication_id}: invalid curation provenance")
        tag_labels = [curation.get("tag_label") for curation in curations]
        if any(isinstance(value, bool) or value not in (0, 1) for value in tag_labels):
            raise ContractError(f"{adjudication_id}: invalid curation tag label")
        if label != int(all(tag_labels)):
            raise ContractError(f"{adjudication_id}: label violates negative-wins policy")
        if status != ("positive" if label else "negative"):
            raise ContractError(f"{adjudication_id}: review status disagrees with label")
        observations.append(
            PairObservation(
                adjudication_id=adjudication_id,
                eligible_position=position,
                label=label,
                source=source,
                source_hash=row["source_hash"],
                concrete_subtypes=concrete_subtypes,
                raw_entry_count=len(positions),
            )
        )
        reviewed_raw_entries += len(positions)
        subtype_pair_cells += len(concrete_subtypes)
        multi_subtype_pairs += int(len(concrete_subtypes) > 1)
    return observations, {
        "ledger_rows": len(rows),
        "reviewed_pairs": len(observations),
        "reviewed_raw_entries": reviewed_raw_entries,
        "unreviewed_pairs": status_counts["unreviewed"],
        "positive_pairs": sum(row.label for row in observations),
        "negative_pairs": sum(not row.label for row in observations),
        "concrete_subtype_pair_cells": subtype_pair_cells,
        "multi_concrete_subtype_pairs": multi_subtype_pairs,
        "subtype_policy": (
            "one count per unique (statement, source_hash, concrete current subtype); "
            "null subtype falls back to source and raw duplicate multiplicity is not counted"
        ),
    }


def _count_pairs(
    observations: Iterable[PairObservation],
) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
    prior: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    subtype: defaultdict[str, dict[str, list[int]]] = defaultdict(dict)
    for observation in observations:
        prior[observation.source][0 if observation.label else 1] += 1
        for subtype_name in observation.concrete_subtypes:
            counts = subtype[observation.source].setdefault(subtype_name, [0, 0])
            counts[0 if observation.label else 1] += 1
    return (
        {source: prior[source] for source in sorted(prior)},
        {
            source: {name: subtype[source][name] for name in sorted(subtype[source])}
            for source in sorted(subtype)
        },
    )


def _ordered_id_sha(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(base._canonical_bytes({"statement_id": row["statement_id"]}) + b"\n")
    return digest.hexdigest()


def _pair_sha(observations: Iterable[PairObservation]) -> str:
    digest = hashlib.sha256()
    for row in sorted(observations, key=lambda item: item.adjudication_id):
        digest.update(
            base._canonical_bytes(
                {
                    "adjudication_id": row.adjudication_id,
                    "label": row.label,
                    "source": row.source,
                    "source_hash": row.source_hash,
                    "subtypes": list(row.concrete_subtypes),
                }
            )
            + b"\n"
        )
    return digest.hexdigest()


def _derived_parameters(
    scorer: BayesianScorer,
    prior_counts: dict[str, list[int]],
    subtype_counts: dict[str, dict[str, list[int]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_parameters = {
        source: {
            "positive_pairs": counts[0],
            "negative_pairs": counts[1],
            "observed_positive_fraction": counts[0] / sum(counts),
            "systematic_error": float(scorer.prior_probs["syst"][source]),
            "random_error": float(scorer.prior_probs["rand"][source]),
        }
        for source, counts in sorted(prior_counts.items())
    }
    subtype_parameters = {
        source: {
            subtype: {
                "positive_pair_subtype_cells": counts[0],
                "negative_pair_subtype_cells": counts[1],
                "observed_positive_fraction": counts[0] / sum(counts),
                "random_error": float(scorer.subtype_probs[source][subtype]),
            }
            for subtype, counts in sorted(entries.items())
        }
        for source, entries in sorted(subtype_counts.items())
    }
    return source_parameters, subtype_parameters


def _score_projection_provenance(
    statement: Statement,
    allowed_sources: set[str],
    scorer: SimpleScorer,
) -> dict[str, Any]:
    projected = [ev for ev in statement.evidence if ev.source_api in allowed_sources]
    visible = set(projected)
    source_fit = 0
    source_default = 0
    fitted_subtype = 0
    source_subtype_fallback = 0
    for evidence in visible:
        source, subtype = _subtype_identity(evidence)
        if isinstance(scorer, BayesianScorer) and source in scorer.prior_counts:
            source_fit += 1
        else:
            source_default += 1
        if (
            isinstance(scorer, BayesianScorer)
            and subtype is not None
            and source in (scorer.subtype_probs or {})
            and subtype in scorer.subtype_probs[source]
        ):
            fitted_subtype += 1
        else:
            source_subtype_fallback += 1
    return {
        "canonical_raw_evidence_count": len(statement.evidence),
        "projected_raw_evidence_count": len(projected),
        "scorer_visible_unique_evidence_count": len(visible),
        "projected_source_counts": dict(sorted(Counter(ev.source_api for ev in projected).items())),
        "scorer_parameter_path_counts": {
            "fitted_source": source_fit,
            "bundled_default_source": source_default,
            "fitted_concrete_subtype": fitted_subtype,
            "source_fallback_for_null_or_unseen_subtype": source_subtype_fallback,
        },
    }


def _prediction_row(statement_id: str, probability: float) -> dict[str, Any]:
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ContractError(f"{statement_id}: invalid probability {value!r}")
    return {"probability_correct": value, "statement_id": statement_id}


def _simple_arm(
    arm_id: str,
    panel: Panel,
    statements: Sequence[Statement],
    sources: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = set(sources)
    projected = [_project_statement(statements[position], allowed) for position in panel.positions]
    if any(not statement.evidence for statement in projected):
        raise ContractError(f"{arm_id}: evidence projection produced an empty statement")
    scorer = SimpleScorer()
    scorer.check_prior_probs(projected)
    scores = scorer.score_statements(projected)
    predictions: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for position, gold, score in zip(panel.positions, panel.gold_rows, scores, strict=True):
        target_statement = statements[position]
        prediction = _prediction_row(gold["statement_id"], score)
        predictions.append(prediction)
        provenance.append(
            {
                "arm_id": arm_id,
                "fit_fold_id": None,
                "input_projection": list(sources),
                "panel_id": panel.panel_id,
                "probability_correct": prediction["probability_correct"],
                "statement_id": gold["statement_id"],
                **_score_projection_provenance(target_statement, allowed, scorer),
            }
        )
    return predictions, provenance


def _bayesian_arm(
    arm_id: str,
    panel: Panel,
    statements: Sequence[Statement],
    observations: Sequence[PairObservation],
    *,
    use_subtypes: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = set(panel.sources)
    panel_position_set = set(panel.positions)
    fold_by_position = {
        position: gold["fold_id"]
        for position, gold in zip(panel.positions, panel.gold_rows, strict=True)
    }
    eligible_pairs = [
        row
        for row in observations
        if row.eligible_position in panel_position_set and row.source in allowed
    ]
    predictions_by_id: dict[str, dict[str, Any]] = {}
    provenance_by_id: dict[str, dict[str, Any]] = {}
    fit_rows: list[dict[str, Any]] = []
    for fold_id in range(10):
        train_pairs = [
            row for row in eligible_pairs if fold_by_position[row.eligible_position] != fold_id
        ]
        excluded_test_pairs = [
            row for row in eligible_pairs if fold_by_position[row.eligible_position] == fold_id
        ]
        if {row.adjudication_id for row in train_pairs}.intersection(
            row.adjudication_id for row in excluded_test_pairs
        ):
            raise ContractError(f"{arm_id} fold {fold_id}: pair leakage")
        prior_counts, all_subtype_counts = _count_pairs(train_pairs)
        subtype_counts = all_subtype_counts if use_subtypes else {}
        scorer = BayesianScorer(prior_counts, subtype_counts)
        source_parameters, subtype_parameters = _derived_parameters(
            scorer, prior_counts, subtype_counts
        )
        test_items = [
            (position, gold)
            for position, gold in zip(panel.positions, panel.gold_rows, strict=True)
            if gold["fold_id"] == fold_id
        ]
        projected = [
            _project_statement(statements[position], allowed) for position, _ in test_items
        ]
        if any(not statement.evidence for statement in projected):
            raise ContractError(f"{arm_id} fold {fold_id}: empty evidence projection")
        scorer.check_prior_probs(projected)
        scores = scorer.score_statements(projected)
        for (position, gold), score in zip(test_items, scores, strict=True):
            statement_id = gold["statement_id"]
            if statement_id in predictions_by_id:
                raise ContractError(f"{arm_id}: duplicate OOF prediction {statement_id}")
            prediction = _prediction_row(statement_id, score)
            predictions_by_id[statement_id] = prediction
            provenance_by_id[statement_id] = {
                "arm_id": arm_id,
                "fit_fold_id": fold_id,
                "input_projection": list(panel.sources),
                "panel_id": panel.panel_id,
                "probability_correct": prediction["probability_correct"],
                "statement_id": statement_id,
                **_score_projection_provenance(
                    statements[position], allowed, scorer
                ),
            }
        train_positions = [
            {position: gold for position, gold in zip(panel.positions, panel.gold_rows, strict=True)}[
                position
            ]
            for position in panel.positions
            if fold_by_position[position] != fold_id
        ]
        test_gold = [gold for _, gold in test_items]
        fit_rows.append(
            {
                "arm_id": arm_id,
                "bundled_default_fallback_sources": sorted(allowed - set(prior_counts)),
                "excluded_test_pair_identity_sha256": _pair_sha(excluded_test_pairs),
                "excluded_test_reviewed_pairs": len(excluded_test_pairs),
                "fit_fold_id": fold_id,
                "fitted_source_parameters": source_parameters,
                "fitted_subtype_parameters": subtype_parameters,
                "input_projection": list(panel.sources),
                "no_pseudocounts": True,
                "panel_id": panel.panel_id,
                "subtype_counts_enabled": use_subtypes,
                "systematic_error_policy": "fixed_0.05_in_current_BayesianScorer",
                "test_statement_count": len(test_items),
                "test_statement_id_sha256": _ordered_id_sha(test_gold),
                "train_reviewed_pair_identity_sha256": _pair_sha(train_pairs),
                "train_reviewed_pairs": len(train_pairs),
                "train_source_counts": prior_counts,
                "train_statement_count": len(train_positions),
                "train_statement_id_sha256": _ordered_id_sha(train_positions),
                "train_subtype_counts": subtype_counts,
                "unseen_or_null_subtype_policy": "fall_back_to_fold_source_parameter",
                "unseen_source_policy": "fall_back_to_bundled_default_source_parameter",
            }
        )
    ordered_predictions = [predictions_by_id[row["statement_id"]] for row in panel.gold_rows]
    ordered_provenance = [provenance_by_id[row["statement_id"]] for row in panel.gold_rows]
    if len(ordered_predictions) != len(panel.positions):
        raise ContractError(f"{arm_id}: incomplete OOF coverage")
    return ordered_predictions, ordered_provenance, fit_rows


def _diagnostics(
    arm_id: str,
    predictions: Sequence[dict[str, Any]],
    gold_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction_map = {row["statement_id"]: row["probability_correct"] for row in predictions}
    if len(prediction_map) != len(predictions):
        raise ContractError(f"{arm_id}: duplicate prediction IDs")
    expected = {row["statement_id"] for row in gold_rows}
    if set(prediction_map) != expected:
        raise ContractError(f"{arm_id}: prediction/gold coverage mismatch")
    fold_rows: list[dict[str, Any]] = []
    for fold_id in range(10):
        fold_gold = [row for row in gold_rows if row["fold_id"] == fold_id]
        labels = np.asarray([row["label"] for row in fold_gold], dtype=int)
        scores = np.asarray([prediction_map[row["statement_id"]] for row in fold_gold])
        precision, recall, _ = precision_recall_curve(labels, scores)
        fold_rows.append(
            {
                "arm_id": arm_id,
                "fold_id": fold_id,
                "negative": int(len(labels) - labels.sum()),
                "positive": int(labels.sum()),
                "rows": len(labels),
                "trapezoidal_pr_auc": float(auc(recall, precision)),
            }
        )
    labels = np.asarray([row["label"] for row in gold_rows], dtype=int)
    scores = np.asarray([prediction_map[row["statement_id"]] for row in gold_rows])
    fold_values = np.asarray([row["trapezoidal_pr_auc"] for row in fold_rows])
    return fold_rows, {
        "arm_id": arm_id,
        "coverage": len(gold_rows),
        "fold_count": 10,
        "fold_mean_trapezoidal_pr_auc": float(fold_values.mean()),
        "fold_population_sd_trapezoidal_pr_auc": float(fold_values.std(ddof=0)),
        "pooled_average_precision": float(average_precision_score(labels, scores)),
        "positive_class": "correct_statement",
        "status": "diagnostic_only_until_shared_three_family_metrics_artifact",
    }


def _descriptor(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": base._display_path(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "sha256": base._sha256(path),
    }


def _resource_path() -> Path:
    module_path = Path(inspect.getsourcefile(SimpleScorer) or "").resolve()
    path = module_path.parent.parent / "resources" / "default_belief_probs.json"
    if not path.is_file():
        raise ContractError(f"could not locate bundled default priors: {path}")
    return path


def _publish_bundle(
    output_dir: Path,
    row_outputs: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {name: output_dir / name for name in row_outputs}
    final_paths[MANIFEST_FILENAME] = output_dir / MANIFEST_FILENAME
    existing = [str(path) for path in final_paths.values() if os.path.lexists(path)]
    if existing:
        raise FileExistsError("refusing to overwrite existing outputs: " + ", ".join(existing))
    with tempfile.TemporaryDirectory(prefix=".current-bayes-", dir=output_dir) as temporary:
        stage = Path(temporary)
        staged: dict[str, Path] = {}
        for name, rows in row_outputs.items():
            path = stage / name
            base._write_jsonl(path, rows)
            staged[name] = path
        manifest_path = stage / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staged[MANIFEST_FILENAME] = manifest_path
        published: list[Path] = []
        try:
            for name in [*row_outputs, MANIFEST_FILENAME]:
                os.link(staged[name], final_paths[name])
                published.append(final_paths[name])
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise


def materialize(
    *,
    corpus_path: Path,
    paper_manifest_path: Path,
    targets_path: Path,
    comparison_manifest_path: Path,
    all_gold_path: Path,
    reader_gold_path: Path,
    evidence_adjudication_path: Path,
    statement_gold_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    paths = [
        corpus_path,
        paper_manifest_path,
        targets_path,
        comparison_manifest_path,
        all_gold_path,
        reader_gold_path,
        evidence_adjudication_path,
        statement_gold_manifest_path,
    ]
    (
        corpus_path,
        paper_manifest_path,
        targets_path,
        comparison_manifest_path,
        all_gold_path,
        reader_gold_path,
        evidence_adjudication_path,
        statement_gold_manifest_path,
    ) = [path.resolve() for path in paths]
    output_dir = output_dir.resolve()

    paper_manifest = base._read_json(paper_manifest_path)
    comparison_manifest = base._read_json(comparison_manifest_path)
    statement_gold_manifest = base._read_json(statement_gold_manifest_path)
    targets = base._read_targets(targets_path)
    all_gold = _read_gold(all_gold_path)
    reader_gold = _read_gold(reader_gold_path)
    evidence_rows = _read_jsonl(evidence_adjudication_path)

    corpus_input = base._paper_corpus_descriptor(corpus_path, paper_manifest)
    target_input = _verify_output_descriptor(
        targets_path, comparison_manifest, "paper_prediction_targets", targets
    )
    all_gold_input = _verify_output_descriptor(
        all_gold_path, comparison_manifest, "paper_released_gold", all_gold
    )
    reader_gold_input = _verify_output_descriptor(
        reader_gold_path,
        comparison_manifest,
        "paper_reader_eligible_released_gold",
        reader_gold,
    )
    evidence_input = _verify_output_descriptor(
        evidence_adjudication_path,
        statement_gold_manifest,
        "evidence_adjudication",
        evidence_rows,
    )
    if len(targets) != 1689 or len(all_gold) != 1689 or len(reader_gold) != 1676:
        raise ContractError("frozen paper panel sizes must be exactly 1689 and 1676")
    if [row["statement_id"] for row in all_gold] != [row["statement_id"] for row in targets]:
        raise ContractError("all-source gold order differs from target order")
    reader_positions = tuple(index for index, row in enumerate(targets) if row["reader_eligible"])
    if [row["statement_id"] for row in reader_gold] != [
        targets[index]["statement_id"] for index in reader_positions
    ]:
        raise ContractError("reader gold order differs from target reader projection")

    statement_jsons, scan_counts = base._scan_targets(corpus_path, targets)
    statements = _decode_statements(statement_jsons)
    for target, statement in zip(targets, statements, strict=True):
        if statement.uuid != target["statement_id"]:
            raise ContractError("deserialized target UUID mismatch")
        if str(statement.get_hash(shallow=True)) != target["matches_hash"]:
            raise ContractError("deserialized target matches hash mismatch")
        unknown_sources = {ev.source_api for ev in statement.evidence} - set(ALL_SOURCES)
        if unknown_sources:
            raise ContractError(f"target has sources outside frozen order: {unknown_sources}")

    observations, observation_summary = _pair_observations(
        evidence_rows, statements, targets
    )
    if observation_summary["reviewed_pairs"] != 5379:
        raise ContractError("expected exactly 5,379 reviewed target evidence pairs")

    all_panel = Panel(
        PANEL_ALL,
        ALL_SOURCES,
        tuple(range(len(targets))),
        tuple(all_gold),
    )
    reader_panel = Panel(
        PANEL_READERS,
        READER_SOURCES,
        reader_positions,
        tuple(reader_gold),
    )

    predictions: dict[str, list[dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []

    for arm_id, panel, sources in (
        (ARM_SIMPLE_ALL, all_panel, ALL_SOURCES),
        (ARM_SIMPLE_READERS, reader_panel, READER_SOURCES),
        (ARM_SIMPLE_READER_SENSITIVITY, reader_panel, ALL_SOURCES),
    ):
        arm_predictions, arm_provenance = _simple_arm(
            arm_id, panel, statements, sources
        )
        predictions[arm_id] = arm_predictions
        provenance.extend(arm_provenance)

    for arm_id, panel, use_subtypes in (
        (ARM_BAYES_SOURCE_ALL, all_panel, False),
        (ARM_BAYES_SUBTYPE_ALL, all_panel, True),
        (ARM_BAYES_SOURCE_READERS, reader_panel, False),
        (ARM_BAYES_SUBTYPE_READERS, reader_panel, True),
    ):
        arm_predictions, arm_provenance, arm_fits = _bayesian_arm(
            arm_id,
            panel,
            statements,
            observations,
            use_subtypes=use_subtypes,
        )
        predictions[arm_id] = arm_predictions
        provenance.extend(arm_provenance)
        fit_rows.extend(arm_fits)

    fold_metrics: list[dict[str, Any]] = []
    diagnostic_summaries: dict[str, Any] = {}
    reader_arm_ids = {
        ARM_SIMPLE_READERS,
        ARM_SIMPLE_READER_SENSITIVITY,
        ARM_BAYES_SOURCE_READERS,
        ARM_BAYES_SUBTYPE_READERS,
    }
    for arm_id, rows in predictions.items():
        gold = reader_gold if arm_id in reader_arm_ids else all_gold
        arm_fold_rows, arm_summary = _diagnostics(arm_id, rows, gold)
        fold_metrics.extend(arm_fold_rows)
        diagnostic_summaries[arm_id] = arm_summary

    row_outputs = {
        PREDICTION_FILENAMES[arm_id]: rows for arm_id, rows in predictions.items()
    }
    row_outputs[FIT_FILENAME] = fit_rows
    row_outputs[PROVENANCE_FILENAME] = provenance
    row_outputs[FOLD_METRICS_FILENAME] = fold_metrics

    # Stage once to calculate exact descriptors before publishing the manifest.
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".current-bayes-descriptors-", dir=output_dir) as tmp:
        stage = Path(tmp)
        output_descriptors: dict[str, Any] = {}
        for name, rows in row_outputs.items():
            path = stage / name
            base._write_jsonl(path, rows)
            output_descriptors[name] = _descriptor(path, rows)
            output_descriptors[name]["path"] = base._display_path(output_dir / name)

    resource = _resource_path()
    belief_module = Path(inspect.getsourcefile(BayesianScorer) or "").resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "claim_scope": {
            "lane": "P",
            "prediction_unit": "assembled_statement",
            "output_semantics": "probability_statement_correct",
            "diagnostics_are_formal_three_family_comparison": False,
            "paid_inference_calls": 0,
        },
        "panels": {
            PANEL_ALL: {
                "rows": 1689,
                "eligible_rows": "all released extended statements",
                "input_sources": list(ALL_SOURCES),
                "input_projection": "all canonical direct evidence from the frozen eleven sources",
                "gold": "paper released compatibility labels",
            },
            PANEL_READERS: {
                "rows": 1676,
                "eligible_rows": "at least one of the five frozen reader sources",
                "input_sources": list(READER_SOURCES),
                "input_projection": "canonical direct evidence restricted to the frozen five readers",
                "gold": "paper reader-eligible released compatibility labels",
                "direct_comparator": "paper OrigBelief-readers input semantics",
            },
            PANEL_READER_SENSITIVITY: {
                "rows": 1676,
                "eligible_rows": "same rows as readers_only_1676",
                "input_sources": list(ALL_SOURCES),
                "input_projection": "all eleven-source canonical direct evidence",
                "direct_reader_parity": False,
                "reason": "database evidence is retained; this is a row-subset sensitivity only",
            },
        },
        "arms": [
            {
                "arm_id": ARM_SIMPLE_ALL,
                "class": "indra.belief.SimpleScorer",
                "panel_id": PANEL_ALL,
                "training_required": False,
                "input_sources": list(ALL_SOURCES),
            },
            {
                "arm_id": ARM_SIMPLE_READERS,
                "class": "indra.belief.SimpleScorer",
                "panel_id": PANEL_READERS,
                "training_required": False,
                "input_sources": list(READER_SOURCES),
            },
            {
                "arm_id": ARM_SIMPLE_READER_SENSITIVITY,
                "class": "indra.belief.SimpleScorer",
                "panel_id": PANEL_READER_SENSITIVITY,
                "training_required": False,
                "input_sources": list(ALL_SOURCES),
                "sensitivity_only": True,
            },
            *[
                {
                    "arm_id": arm_id,
                    "class": "indra.belief.BayesianScorer",
                    "panel_id": panel_id,
                    "input_sources": list(sources),
                    "training_unit": "unique reviewed statement-source_hash pair",
                    "subtype_counts": subtype_enabled,
                    "cross_fitting": "ten fits; all test-fold evidence-pair curations excluded",
                    "pseudocount_policy": "none",
                    "unseen_source_policy": "bundled default source parameter",
                    "unseen_or_null_subtype_policy": "fold-fitted source parameter, else bundled default source parameter",
                }
                for arm_id, panel_id, sources, subtype_enabled in (
                    (ARM_BAYES_SOURCE_ALL, PANEL_ALL, ALL_SOURCES, False),
                    (ARM_BAYES_SUBTYPE_ALL, PANEL_ALL, ALL_SOURCES, True),
                    (ARM_BAYES_SOURCE_READERS, PANEL_READERS, READER_SOURCES, False),
                    (ARM_BAYES_SUBTYPE_READERS, PANEL_READERS, READER_SOURCES, True),
                )
            ],
        ],
        "cross_fit_contract": {
            "folds": 10,
            "fit_labels": "paper evidence adjudication only; same-pair negative-wins labels",
            "statement_labels_used_for_fitting": False,
            "test_pair_exclusion": "all reviewed evidence pairs belonging to test-fold statements",
            "outside_panel_pairs_used": False,
            "no_pseudocounts": True,
            "current_parameter_conversion": (
                "systematic_error=0.05; random_error=1-min(p/(p+n),0.95)-0.05"
            ),
            "subtype_identity": (
                "current indra.belief.tag_evidence_subtype; one cell per unique pair and concrete subtype; "
                "21 source-hash pairs with two Reach rule variants contribute one cell to each rule"
            ),
            "raw_multiplicity_in_training": "not counted",
        },
        "current_class_semantics": {
            "evidence_deduplication": (
                "SimpleScorer.score_statements calls current get_stmt_evidence, which converts each evidence list to a set"
            ),
            "negated_evidence": "positive and negated evidence are scored separately; final score is positive_score*(1-negated_score)",
            "subtype_fallback": "current evidence_random_noise_prior behavior retained",
        },
        "coverage": {
            "all_source_statements": len(all_gold),
            "reader_statements": len(reader_gold),
            "canonical_evidence_entries": sum(len(statement.evidence) for statement in statements),
            "missing_predictions": 0,
            "invalid_predictions": 0,
            **observation_summary,
        },
        "inputs": {
            "canonical_corpus": {**corpus_input, "scan_counts": scan_counts},
            "paper_manifest": {
                "path": base._display_path(paper_manifest_path),
                "bytes": paper_manifest_path.stat().st_size,
                "sha256": base._sha256(paper_manifest_path),
            },
            "prediction_targets": target_input,
            "comparison_manifest": {
                "path": base._display_path(comparison_manifest_path),
                "bytes": comparison_manifest_path.stat().st_size,
                "sha256": base._sha256(comparison_manifest_path),
            },
            "all_source_gold_and_folds": all_gold_input,
            "reader_gold_and_folds": reader_gold_input,
            "evidence_adjudication": evidence_input,
            "statement_gold_manifest": {
                "path": base._display_path(statement_gold_manifest_path),
                "bytes": statement_gold_manifest_path.stat().st_size,
                "sha256": base._sha256(statement_gold_manifest_path),
            },
        },
        "implementation": {
            "indra_version": importlib.metadata.version("indra"),
            "indra_belief_module": {
                "path": base._display_path(belief_module),
                "sha256": base._sha256(belief_module),
            },
            "bundled_default_prior_resource": {
                "path": base._display_path(resource),
                "sha256": base._sha256(resource),
            },
            "adapter": {
                "path": base._display_path(Path(__file__)),
                "sha256": base._sha256(Path(__file__)),
            },
            "shared_identity_adapter": {
                "path": base._display_path(Path(base.__file__)),
                "sha256": base._sha256(Path(base.__file__)),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scikit_learn": importlib.metadata.version("scikit-learn"),
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                "executable": sys.executable,
            },
        },
        "diagnostic_metrics": diagnostic_summaries,
        "runtime_observation": {
            "wall_seconds": time.perf_counter() - started,
            "inference_usd": 0.0,
            "cost_scope": "local CPU scorer execution; excludes human curation collection cost",
        },
        "outputs": output_descriptors,
    }
    _publish_bundle(output_dir, row_outputs, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--paper-manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--all-gold", type=Path, required=True)
    parser.add_argument("--reader-gold", type=Path, required=True)
    parser.add_argument("--evidence-adjudication", type=Path, required=True)
    parser.add_argument("--statement-gold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = materialize(
        corpus_path=args.corpus,
        paper_manifest_path=args.paper_manifest,
        targets_path=args.targets,
        comparison_manifest_path=args.comparison_manifest,
        all_gold_path=args.all_gold,
        reader_gold_path=args.reader_gold,
        evidence_adjudication_path=args.evidence_adjudication,
        statement_gold_manifest_path=args.statement_gold_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({"coverage": manifest["coverage"], "diagnostics": manifest["diagnostic_metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
