#!/usr/bin/env python3
"""Deterministically reproduce the two headline INDRA paper result families.

This is a literal semantic port of the released training notebook, with two
intentional reproducibility repairs:

* the notebook-local OrigBelief sampled MAP is replaced by a numerically
  converged deterministic posterior mode under the same uniform prior and
  likelihood; and
* RandomForestClassifier receives an explicit random state.

The paper did not publish its realized fold assignments, NumPy state, RF state,
MCMC state, or unrounded fold results.  Consequently this script establishes a
digest-pinned deterministic reproduction of the reported protocol and rounded
headline values; it does not claim bit parity with the dead notebook kernel.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import sys
from typing import Any, Iterable, Sequence

import ijson
import networkx as nx
import numpy as np
import scipy
from scipy.optimize import minimize
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, precision_recall_curve


SCHEMA_VERSION = 1
EXPECTED_CORPUS_SHA256 = (
    "bf048f3b485990e6d81ee0f2200ea17efe962268427b4f33af932b4f08a434de"
)
EXPECTED_CORPUS_BYTES = 460_045_058
EXPECTED_CORPUS_STATEMENTS = 894_939
EXPECTED_ELIGIBLE_SHA256 = (
    "82b572bfe57d23bda65611909e5f13b010dac03fd03437dcccdc3fd1e993ec4d"
)
EXPECTED_FOLDS_SHA256 = (
    "a4f3702f32ff9a9d645b8bda206ad017e25007701f9e46bccda7f5be1bff67f7"
)
EXPECTED_PROTOCOL_MANIFEST_SHA256 = (
    "e3d0392b410ba7458663bfeb336954e52836631878d09f8f59ff213feb2725d3"
)
EXPECTED_RELEASED_GOLD_SHA256 = (
    "43c86be9235d91443ec2929b3e74939e3fc388f2950c6f6f0efe6cc5521ebac1"
)
EXPECTED_READER_GOLD_SHA256 = (
    "6c2757dbd64cfd06ed5eb1c46debc2992ed3f2b9f5b936074cb14c02f3856b06"
)

PAPER_REPOSITORY = "https://github.com/sorgerlab/indra_assembly_paper"
PAPER_COMMIT = "63abdf1274d2f5534ed822585775031712916c83"
PAPER_NOTEBOOK = "notebooks/Training Belief ML Models.ipynb"
PAPER_NOTEBOOK_SHA256 = (
    "3bd1a684fdc33c0b4963dd3e0c834c5420d90703112a91773f43415e1125ad26"
)
PAPER_ZENODO_DOI = "10.5281/zenodo.7559353"

READER_SOURCES = ["reach", "sparser", "medscan", "rlimsp", "trips"]
ALL_SOURCES = [
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

# Exact get_all_descendants(Statement) order in the paper-era INDRA 1.22.0.
# The order is material because RandomForestClassifier samples feature indices.
STATEMENT_TYPES = [
    "Modification",
    "SelfModification",
    "RegulateActivity",
    "ActiveForm",
    "HasActivity",
    "Gef",
    "Gap",
    "Complex",
    "Translocation",
    "RegulateAmount",
    "Influence",
    "Conversion",
    "Event",
    "Unresolved",
    "AddModification",
    "RemoveModification",
    "Phosphorylation",
    "Hydroxylation",
    "Sumoylation",
    "Acetylation",
    "Glycosylation",
    "Ribosylation",
    "Ubiquitination",
    "Farnesylation",
    "Geranylgeranylation",
    "Palmitoylation",
    "Myristoylation",
    "Methylation",
    "Dephosphorylation",
    "Dehydroxylation",
    "Desumoylation",
    "Deacetylation",
    "Deglycosylation",
    "Deribosylation",
    "Deubiquitination",
    "Defarnesylation",
    "Degeranylgeranylation",
    "Depalmitoylation",
    "Demyristoylation",
    "Demethylation",
    "Autophosphorylation",
    "Transphosphorylation",
    "Inhibition",
    "Activation",
    "GtpActivation",
    "Association",
    "DecreaseAmount",
    "IncreaseAmount",
    "Migration",
]

EXPECTED_DANGLING_SUPPORT_UUIDS = [
    "1baa8fd0-0e95-40e1-9b30-67b5281b8d96",
    "2eac890a-c307-4281-a4e1-9adb34326c76",
    "545aa533-3681-4d1a-988e-b015c8403bcf",
    "c23cc259-a9f2-497c-869f-69b668cab436",
    "e08cf0cb-bd67-4d03-82e1-656cba9d4dbc",
    "f2f1060d-3965-4ea4-9a41-f0c993dcb846",
]


@dataclass(frozen=True)
class Arm:
    arm_id: str
    display_name: str
    eligible_set: str
    published_mean: float
    published_std: float
    use_avg_evidence_len: bool = False


ORIG_ARM = Arm(
    arm_id="orig_belief_readers",
    display_name="Belief Orig - readers",
    eligible_set="reader_only",
    published_mean=0.917,
    published_std=0.019,
)
RF_PROMOTER_ARM = Arm(
    arm_id="rf_2k_d13_type_pmids_promoter_all_sources_specific",
    display_name="RF 2k-d13 + Type/#PMIDs/promoter - all sources, specific",
    eligible_set="extended_all_sources",
    published_mean=0.942,
    published_std=0.014,
)
RF_PROMOTER_AVGLEN_ARM = Arm(
    arm_id="rf_2k_d13_type_pmids_promoter_avglen_all_sources_specific",
    display_name="RF 2k-d13 + Type/#PMIDs/prom/avglen - all sources, specific",
    eligible_set="extended_all_sources",
    published_mean=0.942,
    published_std=0.015,
    use_avg_evidence_len=True,
)
ARMS = [ORIG_ARM, RF_PROMOTER_ARM, RF_PROMOTER_AVGLEN_ARM]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def file_identity(path: Path, root: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(
            f"Digest mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return {
        "path": repo_relative(path, root),
        "resolved_path": str(path.resolve()),
        "is_symlink": path.is_symlink(),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "verification": "pass" if expected_sha256 is not None else "recorded",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_api": evidence.get("source_api"),
        "pmid": evidence.get("pmid"),
        "text": evidence.get("text"),
        "negated": bool(evidence.get("epistemics", {}).get("negated", False)),
    }


def _compact_statement(statement: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": statement.get("id"),
        "matches_hash": int(statement["matches_hash"]),
        "type": statement["type"],
        "supports": list(statement.get("supports", [])),
        "evidence": [_compact_evidence(ev) for ev in statement.get("evidence", [])],
    }


def load_selected_corpus(
    corpus_path: Path, selected_hashes: set[int]
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Stream the large corpus twice to resolve selected roots and supports."""
    selected: dict[int, dict[str, Any]] = {}
    support_uuids: set[str] = set()
    statement_count = 0
    evidence_count = 0
    with gzip.open(corpus_path, "rb") as handle:
        for raw_statement in ijson.items(handle, "item"):
            statement_count += 1
            evidence_count += len(raw_statement.get("evidence", []))
            stmt_hash = int(raw_statement["matches_hash"])
            if stmt_hash not in selected_hashes:
                continue
            compact = _compact_statement(raw_statement)
            if stmt_hash in selected and selected[stmt_hash]["id"] != compact["id"]:
                raise ValueError(f"Selected matches-hash collision: {stmt_hash}")
            selected[stmt_hash] = compact
            support_uuids.update(compact["supports"])

    missing_selected = sorted(selected_hashes.difference(selected))
    if missing_selected:
        raise ValueError(f"Corpus is missing selected hashes: {missing_selected[:10]}")
    if statement_count != EXPECTED_CORPUS_STATEMENTS:
        raise ValueError(
            f"Corpus statement count {statement_count} != {EXPECTED_CORPUS_STATEMENTS}"
        )

    supports: dict[str, dict[str, Any]] = {}
    with gzip.open(corpus_path, "rb") as handle:
        for raw_statement in ijson.items(handle, "item"):
            stmt_uuid = raw_statement.get("id")
            if stmt_uuid not in support_uuids:
                continue
            compact = _compact_statement(raw_statement)
            if stmt_uuid in supports and supports[stmt_uuid] != compact:
                raise ValueError(f"Corpus UUID collision: {stmt_uuid}")
            supports[stmt_uuid] = compact

    dangling = sorted(support_uuids.difference(supports))
    if dangling != EXPECTED_DANGLING_SUPPORT_UUIDS:
        raise ValueError(
            "Unexpected dangling support UUID set: "
            f"expected {EXPECTED_DANGLING_SUPPORT_UUIDS}, got {dangling}"
        )
    scan = {
        "corpus_statements": statement_count,
        "corpus_evidence_entries": evidence_count,
        "selected_statements": len(selected),
        "selected_direct_evidence_entries": sum(
            len(statement["evidence"]) for statement in selected.values()
        ),
        "referenced_support_uuids": len(support_uuids),
        "resolved_support_uuids": len(supports),
        "dangling_support_uuids": dangling,
    }
    return selected, supports, scan


def trapezoidal_pr_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Paper metric: sklearn PR curve followed by trapezoidal AUC."""
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return float(auc(recall, precision))


def orig_belief_source_probability(
    mention_counts: np.ndarray | Sequence[float], pr: float, ps: float
) -> np.ndarray:
    counts = np.asarray(mention_counts, dtype=float)
    return (1.0 - ps) * (1.0 - np.power(pr, counts))


def _orig_negative_log_likelihood(
    params: Sequence[float], mention_counts: np.ndarray, labels: np.ndarray
) -> float:
    pr, ps = (float(params[0]), float(params[1]))
    if not (0.0 <= pr <= 1.0 and 0.0 <= ps <= 1.0):
        return math.inf
    probabilities = orig_belief_source_probability(mention_counts, pr, ps)
    positive = labels == 1
    if np.any(positive & (probabilities <= 0.0)):
        return math.inf
    if np.any((~positive) & (probabilities >= 1.0)):
        return math.inf
    log_likelihood = 0.0
    if np.any(positive):
        log_likelihood += float(np.log(probabilities[positive]).sum())
    if np.any(~positive):
        log_likelihood += float(np.log1p(-probabilities[~positive]).sum())
    return -log_likelihood


def fit_orig_belief_map(
    mention_counts: Sequence[float], labels: Sequence[int]
) -> dict[str, Any]:
    """Find the deterministic exact mode of the notebook's uniform posterior."""
    counts = np.asarray(mention_counts, dtype=float)
    y = np.asarray(labels, dtype=int)
    if counts.ndim != 1 or y.ndim != 1 or counts.shape != y.shape:
        raise ValueError("Counts and labels must be equally sized vectors")
    if not len(y) or np.any(counts <= 0) or set(np.unique(y)).difference({0, 1}):
        raise ValueError("OrigBelief fitting requires positive counts and binary labels")

    best = None
    starts = (0.01, 0.2, 0.5, 0.8, 0.99)
    for pr0 in starts:
        for ps0 in starts:
            result = minimize(
                _orig_negative_log_likelihood,
                x0=np.array([pr0, ps0]),
                args=(counts, y),
                method="Nelder-Mead",
                options={
                    "maxiter": 20_000,
                    "xatol": 1e-12,
                    "fatol": 1e-12,
                    "adaptive": False,
                },
            )
            if not np.isfinite(result.fun):
                continue
            if best is None or float(result.fun) < float(best.fun):
                best = result
    if best is None:
        raise RuntimeError("OrigBelief posterior mode optimization failed")
    pr, ps = (float(best.x[0]), float(best.x[1]))
    if not (0.0 <= pr <= 1.0 and 0.0 <= ps <= 1.0):
        raise RuntimeError(f"Optimizer returned invalid parameters: {(pr, ps)}")
    return {
        "pr": pr,
        "ps": ps,
        "log_likelihood": -float(best.fun),
        "optimizer": "multi_start_nelder_mead",
        "starts": len(starts) ** 2,
        "success": bool(best.success),
        "iterations": int(best.nit),
        "function_evaluations": int(best.nfev),
    }


def _fold_rows(
    eligible_rows: list[dict[str, Any]],
    fold_by_hash: dict[str, dict[str, Any]],
    eligible_set: str,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in eligible_rows
        if eligible_set == "extended_all_sources" or row["reader_eligible"]
    ]
    missing = [row["stmt_hash"] for row in rows if row["stmt_hash"] not in fold_by_hash]
    if missing:
        raise ValueError(f"Missing fold assignments for {eligible_set}: {missing[:10]}")
    return sorted(rows, key=lambda row: fold_by_hash[row["stmt_hash"]]["shuffle_position"])


def _prediction_row(
    arm: Arm,
    row: dict[str, Any],
    fold: dict[str, Any],
    statement: dict[str, Any],
    probability: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "arm_id": arm.arm_id,
        "arm_display_name": arm.display_name,
        "eligible_set": arm.eligible_set,
        "statement_id": statement["id"],
        "statement_hash": row["stmt_hash"],
        "source_row_index": int(row["source_row_index"]),
        "shuffle_position": int(fold["shuffle_position"]),
        "fold_id": int(fold["test_fold"]),
        "label": int(row["correct"]),
        "probability_correct": float(probability),
        "predicted_label_at_0_5": int(probability >= 0.5),
    }


def run_orig_arm(
    rows: list[dict[str, Any]],
    folds: dict[str, dict[str, Any]],
    selected: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for fold_id in range(10):
        train = [row for row in rows if int(folds[row["stmt_hash"]]["test_fold"]) != fold_id]
        test = [row for row in rows if int(folds[row["stmt_hash"]]["test_fold"]) == fold_id]
        parameters: list[dict[str, Any]] = []
        for source_index, source in enumerate(READER_SOURCES):
            source_train = [
                (row["reader_source_counts"][source_index], row["correct"])
                for row in train
                if row["reader_source_counts"][source_index] > 0
            ]
            fitted = fit_orig_belief_map(
                [item[0] for item in source_train], [item[1] for item in source_train]
            )
            fitted.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "arm_id": ORIG_ARM.arm_id,
                    "fold_id": fold_id,
                    "source": source,
                    "fit_rows": len(source_train),
                    "fit_positive": sum(item[1] for item in source_train),
                    "fit_negative": len(source_train) - sum(item[1] for item in source_train),
                    "method": "deterministic_exact_posterior_mode",
                }
            )
            parameters.append(fitted)
            fit_rows.append(fitted)

        fold_probabilities: list[float] = []
        for row in test:
            source_beliefs = [
                float(
                    orig_belief_source_probability(
                        [row["reader_source_counts"][source_index]],
                        parameters[source_index]["pr"],
                        parameters[source_index]["ps"],
                    )[0]
                )
                for source_index in range(len(READER_SOURCES))
            ]
            probability = float(1.0 - np.prod(1.0 - np.asarray(source_beliefs)))
            probability = min(1.0, max(0.0, probability))
            fold_probabilities.append(probability)
            predictions.append(
                _prediction_row(
                    ORIG_ARM,
                    row,
                    folds[row["stmt_hash"]],
                    selected[int(row["stmt_hash"])],
                    probability,
                )
            )
        labels = [int(row["correct"]) for row in test]
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "arm_id": ORIG_ARM.arm_id,
                "fold_id": fold_id,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_positive": sum(labels),
                "test_negative": len(labels) - sum(labels),
                "trapezoidal_pr_auc": trapezoidal_pr_auc(labels, fold_probabilities),
            }
        )
    return predictions, metrics, fit_rows


def _partition_extra_evidence(
    rows: list[dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    supports: dict[str, dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Port build_refinements_graph + get_ev_for_stmts_from_supports.

    Graph nodes are matches hashes and node statement attributes are overwritten
    in the same root/support iteration order as networkx.add_node in INDRA 1.22.
    Only roots in this partition contribute outgoing edges.
    """
    graph = nx.DiGraph()
    node_statements: dict[int | str, dict[str, Any]] = {}
    for row in rows:
        statement = selected[int(row["stmt_hash"])]
        stmt_hash = int(statement["matches_hash"])
        graph.add_node(stmt_hash)
        node_statements[stmt_hash] = statement
        for support_uuid in statement["supports"]:
            if support_uuid in supports:
                support = supports[support_uuid]
                support_hash: int | str = int(support["matches_hash"])
            else:
                # INDRA's on_missing_support='handle' creates an evidence-free
                # Unresolved node.  Its exact hash is immaterial to all features.
                support = {"id": support_uuid, "evidence": []}
                support_hash = f"unresolved:{support_uuid}"
            graph.add_node(support_hash)
            node_statements[support_hash] = support
            graph.add_edge(stmt_hash, support_hash)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Fold-local refinements graph contains a cycle")

    output: list[list[dict[str, Any]]] = []
    for row in rows:
        stmt_hash = int(row["stmt_hash"])
        extra: list[dict[str, Any]] = []
        for descendant_hash in nx.descendants(graph, stmt_hash):
            extra.extend(
                evidence
                for evidence in node_statements[descendant_hash]["evidence"]
                if not evidence["negated"]
            )
        output.append(extra)
    return output


def counts_scorer_matrix(
    rows: list[dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    supports: dict[str, dict[str, Any]],
    *,
    use_avg_evidence_len: bool,
) -> np.ndarray:
    """Port the exact paper-used INDRA 1.22 CountsScorer feature paths."""
    type_index = {statement_type: index for index, statement_type in enumerate(STATEMENT_TYPES)}
    extra_by_row = _partition_extra_evidence(rows, selected, supports)
    matrix: list[list[float]] = []
    for row, extra_evidence in zip(rows, extra_by_row):
        statement = selected[int(row["stmt_hash"])]
        direct_evidence = statement["evidence"]
        direct_counts = Counter(ev["source_api"] for ev in direct_evidence)
        indirect_counts = Counter(ev["source_api"] for ev in extra_evidence)
        features: list[float] = [float(direct_counts[source]) for source in ALL_SOURCES]
        features.extend(float(indirect_counts[source]) for source in ALL_SOURCES)

        one_hot = [0.0] * len(STATEMENT_TYPES)
        try:
            one_hot[type_index[statement["type"]]] = 1.0
        except KeyError as exc:
            raise ValueError(f"Unknown statement type: {statement['type']}") from exc
        features.extend(one_hot)

        # Evidence.pmid is added to the set even when it is None.
        features.append(float(len({ev["pmid"] for ev in direct_evidence})))
        features.append(float(len({ev["pmid"] for ev in extra_evidence})))
        promoter_count = sum(
            1
            for ev in direct_evidence
            if ev["text"] is not None and "promoter" in ev["text"].lower()
        )
        features.append(
            float(promoter_count / len(direct_evidence)) if direct_evidence else 0.0
        )
        if use_avg_evidence_len:
            evidence_lengths = [
                len(ev["text"].split())
                for ev in direct_evidence
                if ev["text"] is not None
            ]
            features.append(float(np.mean(evidence_lengths)) if evidence_lengths else 0.0)
        matrix.append(features)

    result = np.asarray(matrix, dtype=np.float64)
    expected_columns = 75 if use_avg_evidence_len else 74
    if result.shape != (len(rows), expected_columns):
        raise RuntimeError(
            f"Feature shape {result.shape} != {(len(rows), expected_columns)}"
        )
    return result


def matrix_sha256(matrix: np.ndarray) -> str:
    little_endian = np.asarray(matrix, dtype="<f8", order="C")
    header = struct.pack("<QQ", *little_endian.shape)
    return hashlib.sha256(header + little_endian.tobytes(order="C")).hexdigest()


def _new_rf(random_state: int, n_jobs: int) -> RandomForestClassifier:
    # max_features='sqrt' is the explicit semantic equivalent of the historical
    # sklearn 0.23 classifier default max_features='auto'.
    return RandomForestClassifier(
        n_estimators=2000,
        criterion="gini",
        max_depth=13,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    )


def run_rf_arm(
    arm: Arm,
    rows: list[dict[str, Any]],
    folds: dict[str, dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    supports: dict[str, dict[str, Any]],
    *,
    random_state: int,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    for fold_id in range(10):
        train = [row for row in rows if int(folds[row["stmt_hash"]]["test_fold"]) != fold_id]
        test = [row for row in rows if int(folds[row["stmt_hash"]]["test_fold"]) == fold_id]
        train_matrix = counts_scorer_matrix(
            train,
            selected,
            supports,
            use_avg_evidence_len=arm.use_avg_evidence_len,
        )
        test_matrix = counts_scorer_matrix(
            test,
            selected,
            supports,
            use_avg_evidence_len=arm.use_avg_evidence_len,
        )
        train_labels = np.asarray([row["correct"] for row in train], dtype=int)
        test_labels = np.asarray([row["correct"] for row in test], dtype=int)
        classifier = _new_rf(random_state, n_jobs)
        classifier.fit(train_matrix, train_labels)
        probabilities = classifier.predict_proba(test_matrix)[:, 1]
        for row, probability in zip(test, probabilities):
            predictions.append(
                _prediction_row(
                    arm,
                    row,
                    folds[row["stmt_hash"]],
                    selected[int(row["stmt_hash"])],
                    float(probability),
                )
            )
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "arm_id": arm.arm_id,
                "fold_id": fold_id,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_positive": int(test_labels.sum()),
                "test_negative": int(len(test_labels) - test_labels.sum()),
                "trapezoidal_pr_auc": trapezoidal_pr_auc(test_labels, probabilities),
            }
        )
        fits.append(
            {
                "schema_version": SCHEMA_VERSION,
                "arm_id": arm.arm_id,
                "fold_id": fold_id,
                "method": "seeded_random_forest_semantic_port",
                "random_state": random_state,
                "train_feature_shape": list(train_matrix.shape),
                "test_feature_shape": list(test_matrix.shape),
                "train_feature_matrix_sha256": matrix_sha256(train_matrix),
                "test_feature_matrix_sha256": matrix_sha256(test_matrix),
                "estimator": {
                    "n_estimators": 2000,
                    "max_depth": 13,
                    "max_features": "sqrt",
                    "n_jobs": n_jobs,
                },
            }
        )
    return predictions, metrics, fits


def summarize_arm(arm: Arm, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [row["trapezoidal_pr_auc"] for row in metrics if row["arm_id"] == arm.arm_id],
        dtype=float,
    )
    if len(values) != 10:
        raise ValueError(f"{arm.arm_id} has {len(values)} folds, expected 10")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    mean_match = round(mean, 3) == arm.published_mean
    std_match = round(std, 3) == arm.published_std
    return {
        "arm_id": arm.arm_id,
        "display_name": arm.display_name,
        "eligible_set": arm.eligible_set,
        "fold_count": len(values),
        "fold_trapezoidal_pr_auc": values.tolist(),
        "fold_mean_trapezoidal_pr_auc": mean,
        "fold_population_std": std,
        "published_rounded_mean": arm.published_mean,
        "published_rounded_population_std": arm.published_std,
        "mean_discrepancy_from_published_rounded_anchor": mean - arm.published_mean,
        "std_discrepancy_from_published_rounded_anchor": std - arm.published_std,
        "rounded_mean_match": mean_match,
        "rounded_std_match": std_match,
        "rounded_headline_match": mean_match and std_match,
    }


def _validate_direct_counts(
    eligible_rows: list[dict[str, Any]], selected: dict[int, dict[str, Any]]
) -> None:
    for row in eligible_rows:
        statement = selected[int(row["stmt_hash"])]
        counts = Counter(ev["source_api"] for ev in statement["evidence"])
        actual_all = [counts[source] for source in ALL_SOURCES]
        actual_readers = [counts[source] for source in READER_SOURCES]
        if actual_all != row["historical_all_source_counts"]:
            raise ValueError(f"All-source count mismatch for {row['stmt_hash']}")
        if actual_readers != row["reader_source_counts"]:
            raise ValueError(f"Reader count mismatch for {row['stmt_hash']}")


def _gold_map(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for row in rows:
        statement_id = row["statement_id"]
        if statement_id in result:
            raise ValueError(f"Duplicate gold statement ID: {statement_id}")
        result[statement_id] = (int(row["fold_id"]), int(row["label"]))
    return result


def _validate_predictions_against_gold(
    predictions: list[dict[str, Any]],
    arm: Arm,
    gold: dict[str, tuple[int, int]],
) -> None:
    arm_rows = [row for row in predictions if row["arm_id"] == arm.arm_id]
    actual: dict[str, tuple[int, int]] = {}
    for row in arm_rows:
        probability = float(row["probability_correct"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid probability for {row['statement_id']}: {probability}")
        statement_id = row["statement_id"]
        if statement_id in actual:
            raise ValueError(f"Duplicate OOF statement for {arm.arm_id}: {statement_id}")
        actual[statement_id] = (int(row["fold_id"]), int(row["label"]))
    if actual != gold:
        missing = sorted(set(gold).difference(actual))
        extra = sorted(set(actual).difference(gold))
        mismatched = sorted(
            statement_id
            for statement_id in set(actual).intersection(gold)
            if actual[statement_id] != gold[statement_id]
        )
        raise ValueError(
            f"OOF/gold mismatch for {arm.arm_id}: missing={missing[:3]}, "
            f"extra={extra[:3]}, mismatched={mismatched[:3]}"
        )


def minimal_prediction_projection(
    predictions: list[dict[str, Any]],
    arm: Arm,
    ordered_gold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project one arm to the exact two-field comparison-harness schema."""
    probability_by_statement_id = {
        row["statement_id"]: float(row["probability_correct"])
        for row in predictions
        if row["arm_id"] == arm.arm_id
    }
    if len(probability_by_statement_id) != len(ordered_gold_rows):
        raise ValueError(
            f"Projection cardinality mismatch for {arm.arm_id}: "
            f"{len(probability_by_statement_id)} != {len(ordered_gold_rows)}"
        )
    output: list[dict[str, Any]] = []
    for gold_row in ordered_gold_rows:
        statement_id = gold_row["statement_id"]
        if statement_id not in probability_by_statement_id:
            raise ValueError(f"Projection missing {statement_id} for {arm.arm_id}")
        output.append(
            {
                "statement_id": statement_id,
                "probability_correct": probability_by_statement_id[statement_id],
            }
        )
    return output


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    protocol_dir = args.protocol_dir
    gold_dir = args.comparison_gold_dir
    return {
        "corpus": args.corpus_json_gz,
        "eligible": protocol_dir / "paper_eligible_statements.jsonl",
        "folds": protocol_dir / "paper_fold_assignments.jsonl",
        "protocol_manifest": protocol_dir / "paper_protocol_manifest.json",
        "released_gold": gold_dir / "paper_released_gold.jsonl",
        "reader_gold": gold_dir / "paper_reader_eligible_released_gold.jsonl",
        "comparison_gold_manifest": gold_dir / "paper_comparison_gold_manifest.json",
    }


def reproduce(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    paths = _input_paths(args)
    expected_digests = {
        "corpus": EXPECTED_CORPUS_SHA256,
        "eligible": EXPECTED_ELIGIBLE_SHA256,
        "folds": EXPECTED_FOLDS_SHA256,
        "protocol_manifest": EXPECTED_PROTOCOL_MANIFEST_SHA256,
        "released_gold": EXPECTED_RELEASED_GOLD_SHA256,
        "reader_gold": EXPECTED_READER_GOLD_SHA256,
        "comparison_gold_manifest": None,
    }
    identities = {
        key: file_identity(path, root, expected_digests[key])
        for key, path in paths.items()
    }
    if identities["corpus"]["bytes"] != EXPECTED_CORPUS_BYTES:
        raise ValueError(
            f"Corpus size {identities['corpus']['bytes']} != {EXPECTED_CORPUS_BYTES}"
        )

    eligible_rows = read_jsonl(paths["eligible"])
    fold_rows = read_jsonl(paths["folds"])
    released_gold_rows = read_jsonl(paths["released_gold"])
    reader_gold_rows = read_jsonl(paths["reader_gold"])
    if len(eligible_rows) != 1689 or len(fold_rows) != 3365:
        raise ValueError("Frozen protocol row counts changed")
    if len(released_gold_rows) != 1689 or len(reader_gold_rows) != 1676:
        raise ValueError("Comparison-gold row counts changed")

    fold_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for eligible_set in ("extended_all_sources", "reader_only"):
        fold_maps[eligible_set] = {
            row["stmt_hash"]: row
            for row in fold_rows
            if row["eligible_set"] == eligible_set
        }
    all_rows = _fold_rows(
        eligible_rows, fold_maps["extended_all_sources"], "extended_all_sources"
    )
    reader_rows = _fold_rows(eligible_rows, fold_maps["reader_only"], "reader_only")
    if len(all_rows) != 1689 or len(reader_rows) != 1676:
        raise ValueError("Eligible-set reconstruction changed")

    selected, supports, corpus_scan = load_selected_corpus(
        paths["corpus"], {int(row["stmt_hash"]) for row in eligible_rows}
    )
    _validate_direct_counts(eligible_rows, selected)

    predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    arm_predictions, arm_metrics, arm_fits = run_orig_arm(
        reader_rows, fold_maps["reader_only"], selected
    )
    predictions.extend(arm_predictions)
    fold_metrics.extend(arm_metrics)
    fit_rows.extend(arm_fits)
    for arm in (RF_PROMOTER_ARM, RF_PROMOTER_AVGLEN_ARM):
        arm_predictions, arm_metrics, arm_fits = run_rf_arm(
            arm,
            all_rows,
            fold_maps["extended_all_sources"],
            selected,
            supports,
            random_state=args.rf_random_state,
            n_jobs=args.n_jobs,
        )
        predictions.extend(arm_predictions)
        fold_metrics.extend(arm_metrics)
        fit_rows.extend(arm_fits)

    released_gold = _gold_map(released_gold_rows)
    reader_gold = _gold_map(reader_gold_rows)
    _validate_predictions_against_gold(predictions, ORIG_ARM, reader_gold)
    _validate_predictions_against_gold(predictions, RF_PROMOTER_ARM, released_gold)
    _validate_predictions_against_gold(predictions, RF_PROMOTER_AVGLEN_ARM, released_gold)

    results = {arm.arm_id: summarize_arm(arm, fold_metrics) for arm in ARMS}
    failed_rounding = [
        arm_id for arm_id, result in results.items() if not result["rounded_headline_match"]
    ]
    if failed_rounding:
        raise RuntimeError(f"Published rounded headline mismatch: {failed_rounding}")

    arm_order = {arm.arm_id: index for index, arm in enumerate(ARMS)}
    predictions.sort(
        key=lambda row: (
            arm_order[row["arm_id"]],
            row["fold_id"],
            row["shuffle_position"],
        )
    )
    fold_metrics.sort(key=lambda row: (arm_order[row["arm_id"]], row["fold_id"]))
    fit_rows.sort(
        key=lambda row: (
            arm_order[row["arm_id"]],
            row["fold_id"],
            row.get("source", ""),
        )
    )
    minimal_predictions = {
        ORIG_ARM.arm_id: minimal_prediction_projection(
            predictions, ORIG_ARM, reader_gold_rows
        ),
        RF_PROMOTER_ARM.arm_id: minimal_prediction_projection(
            predictions, RF_PROMOTER_ARM, released_gold_rows
        ),
        RF_PROMOTER_AVGLEN_ARM.arm_id: minimal_prediction_projection(
            predictions, RF_PROMOTER_AVGLEN_ARM, released_gold_rows
        ),
    }

    output_dir = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir()
    try:
        predictions_path = temporary_dir / "paper_reproduction_oof_predictions.jsonl"
        metrics_path = temporary_dir / "paper_reproduction_fold_metrics.jsonl"
        fits_path = temporary_dir / "paper_reproduction_fit_provenance.jsonl"
        orig_minimal_path = temporary_dir / "orig_belief_readers_predictions.jsonl"
        rf_promoter_minimal_path = (
            temporary_dir / "rf_promoter_all_sources_specific_predictions.jsonl"
        )
        rf_avglen_minimal_path = (
            temporary_dir / "rf_promoter_avglen_all_sources_specific_predictions.jsonl"
        )
        write_jsonl(predictions_path, predictions)
        write_jsonl(metrics_path, fold_metrics)
        write_jsonl(fits_path, fit_rows)
        write_jsonl(orig_minimal_path, minimal_predictions[ORIG_ARM.arm_id])
        write_jsonl(
            rf_promoter_minimal_path, minimal_predictions[RF_PROMOTER_ARM.arm_id]
        )
        write_jsonl(
            rf_avglen_minimal_path,
            minimal_predictions[RF_PROMOTER_AVGLEN_ARM.arm_id],
        )

        harness_path = Path(__file__).resolve()
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "indra_paper_headline_deterministic_reproduction",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "claim_scope": {
                "status": "rounded_headlines_reproduced_under_pinned_semantic_port",
                "historical_bit_parity": False,
                "historical_fold_parity": False,
                "reason": (
                    "The paper omitted realized folds, a complete runtime lock, NumPy/RF/MCMC "
                    "states, unrounded fold metrics, predictions, and fitted models."
                ),
                "orig_belief_change": (
                    "Numerically converged deterministic posterior mode under the notebook's "
                    "uniform prior and likelihood replaces its unseeded finite MCMC sampled MAP."
                ),
                "random_forest_change": (
                    "random_state=4 is explicit; the notebook used random_state=None after "
                    "unrecorded global NumPy RNG consumption."
                ),
                "paid_inference_used": False,
            },
            "paper_identity": {
                "repository": PAPER_REPOSITORY,
                "commit": PAPER_COMMIT,
                "notebook": PAPER_NOTEBOOK,
                "notebook_sha256": PAPER_NOTEBOOK_SHA256,
                "zenodo_doi": PAPER_ZENODO_DOI,
            },
            "inputs": identities,
            "corpus_scan": corpus_scan,
            "protocol": {
                "pre_split_shuffle": "Python random.seed(4) then random.shuffle in released pickle order",
                "cross_validation": "StratifiedKFold(n_splits=10, shuffle=False)",
                "fold_assignments": (
                    "Frozen deterministic reconstruction of the declared algorithm; historical "
                    "realized assignments were not published."
                ),
                "metric": "precision_recall_curve then trapezoidal auc(recall, precision)",
                "aggregation": "arithmetic mean and population standard deviation (ddof=0)",
                "positive_class": "released correct-statement label",
            },
            "model_contracts": {
                ORIG_ARM.arm_id: {
                    "eligible_rows": 1676,
                    "sources": READER_SOURCES,
                    "formula": "b(n)=(1-ps)*(1-pr**n); combined=1-product(1-b_source)",
                    "per_source_fit_filter": "direct source count > 0 in training fold",
                    "prior": "uniform on pr,ps in [0,1]",
                    "historical_estimator": "100 walkers, 100 burn, 100 sample; sampled MAP",
                    "reproduction_estimator": (
                        "25-start deterministic Nelder-Mead numerically converged mode"
                    ),
                },
                RF_PROMOTER_ARM.arm_id: {
                    "eligible_rows": 1689,
                    "sources": ALL_SOURCES,
                    "features": {
                        "direct_source_counts": 11,
                        "fold_local_more_specific_source_counts": 11,
                        "statement_type_one_hot": 49,
                        "direct_and_more_specific_unique_pmid_counts": 2,
                        "direct_promoter_frequency": 1,
                        "total": 74,
                    },
                    "estimator": "RandomForestClassifier(n_estimators=2000,max_depth=13)",
                    "random_state": args.rf_random_state,
                },
                RF_PROMOTER_AVGLEN_ARM.arm_id: {
                    "eligible_rows": 1689,
                    "sources": ALL_SOURCES,
                    "features": {
                        "same_as_promoter_arm": 74,
                        "average_direct_evidence_text_length": 1,
                        "total": 75,
                    },
                    "estimator": "RandomForestClassifier(n_estimators=2000,max_depth=13)",
                    "random_state": args.rf_random_state,
                },
            },
            "published_row_execution_state": {
                ORIG_ARM.arm_id: {
                    "notebook_cell": 38,
                    "visible_completed_iterations": 10,
                    "status": "validated_external_anchor",
                },
                RF_PROMOTER_ARM.arm_id: {
                    "notebook_cell": 30,
                    "visible_completed_iterations": 10,
                    "status": "validated_external_anchor",
                },
                RF_PROMOTER_AVGLEN_ARM.arm_id: {
                    "notebook_cell": 30,
                    "visible_completed_iterations": 10,
                    "status": "validated_external_anchor",
                },
                "orig_belief_all_sources_specific_counts_only": {
                    "display_name": "Belief Orig - all sources, specific",
                    "published_rounded_mean": 0.923,
                    "published_rounded_population_std": 0.012,
                    "notebook_cell": 40,
                    "saved_source_folds": 10,
                    "visible_completed_iterations": 2,
                    "status": "forensic_stale_state_suspect_not_a_parity_gate",
                },
            },
            "results": results,
            "headline_interpretation": {
                "warning": (
                    "0.917 and 0.942 are external anchors on distinct eligible populations, "
                    "source modalities, feature sets, and reconstructed folds; their difference "
                    "is not a paired model-only effect."
                ),
                "rf_rounded_tie": (
                    "Both 74- and 75-feature RF rows report 0.942; the 74-feature promoter row "
                    "has the lower published population fold SD (0.014 versus 0.015)."
                ),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    "ijson": getattr(ijson, "__version__", None),
                    "networkx": nx.__version__,
                    "numpy": np.__version__,
                    "scikit-learn": sklearn.__version__,
                    "scipy": scipy.__version__,
                },
                "rf_n_jobs": args.n_jobs,
                "historical_runtime": {
                    "python": "3.7.4",
                    "scikit_learn": "0.23.x inferred from traceback, exact patch unrecorded",
                    "indra": "1.22.0 is the strongest date-matched candidate, not recorded",
                },
            },
            "implementation_provenance": {
                "harness": {
                    "path": repo_relative(harness_path, root),
                    "sha256": sha256_file(harness_path),
                },
                "indra_1_22_counts_scorer_source_sha256": (
                    "3b271402ddb052f49b9105d26681abff4136c7cb8f37ed45458044bc303436f6"
                ),
                "indra_1_22_belief_source_sha256": (
                    "b64a388de6e759362d106760efcfbd52423f78791e58618c5220a7dac013b985"
                ),
                "counts_scorer_parity_note": (
                    "Paper-used feature paths are logic-identical in INDRA 1.22.0 and 1.24.0; "
                    "the port freezes the 1.22.0 order and behavior directly."
                ),
            },
            "validation": {
                "input_digests": "pass",
                "corpus_selected_coverage": "pass",
                "direct_source_count_parity": "pass",
                "oof_statement_fold_label_parity_with_gold": "pass",
                "minimal_prediction_projection_exact_schema_and_gold_order": "pass",
                "probability_domain": "pass",
                "ten_fold_metrics_per_arm": "pass",
                "published_three_decimal_headline_rounding": "pass",
                "artifacts_overwritten": False,
            },
            "outputs": {},
        }
        output_specs = {
            "oof_predictions": (predictions_path, len(predictions)),
            "fold_metrics": (metrics_path, len(fold_metrics)),
            "fit_provenance": (fits_path, len(fit_rows)),
            "orig_belief_readers_minimal_predictions": (
                orig_minimal_path,
                len(minimal_predictions[ORIG_ARM.arm_id]),
            ),
            "rf_promoter_minimal_predictions": (
                rf_promoter_minimal_path,
                len(minimal_predictions[RF_PROMOTER_ARM.arm_id]),
            ),
            "rf_promoter_avglen_minimal_predictions": (
                rf_avglen_minimal_path,
                len(minimal_predictions[RF_PROMOTER_AVGLEN_ARM.arm_id]),
            ),
        }
        for key, (path, row_count) in output_specs.items():
            manifest["outputs"][key] = {
                "path": repo_relative(output_dir / path.name, root),
                "rows": row_count,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest["outputs"]["orig_belief_readers_minimal_predictions"].update(
            {
                "schema": ["statement_id", "probability_correct"],
                "ordered_like_input": "reader_gold",
                "arm_id": ORIG_ARM.arm_id,
            }
        )
        manifest["outputs"]["rf_promoter_minimal_predictions"].update(
            {
                "schema": ["statement_id", "probability_correct"],
                "ordered_like_input": "released_gold",
                "arm_id": RF_PROMOTER_ARM.arm_id,
            }
        )
        manifest["outputs"]["rf_promoter_avglen_minimal_predictions"].update(
            {
                "schema": ["statement_id", "probability_correct"],
                "ordered_like_input": "released_gold",
                "arm_id": RF_PROMOTER_AVGLEN_ARM.arm_id,
            }
        )

        manifest_path = temporary_dir / "paper_reproduction_manifest.json"
        write_json(manifest_path, manifest)
        manifest_digest = sha256_file(manifest_path)
        (temporary_dir / "paper_reproduction_manifest.sha256").write_text(
            f"{manifest_digest}  paper_reproduction_manifest.json\n", encoding="ascii"
        )
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--corpus-json-gz",
        type=Path,
        default=Path("data/benchmark/indra_benchmark_corpus.json.gz"),
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("data/results/indra_paper_protocol_20260717"),
    )
    parser.add_argument(
        "--comparison-gold-dir",
        type=Path,
        default=Path("data/results/indra_paper_comparison_gold_20260717"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/indra_paper_reproduction_20260717"),
    )
    parser.add_argument("--rf-random-state", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    for attribute in (
        "corpus_json_gz",
        "protocol_dir",
        "comparison_gold_dir",
        "output_dir",
    ):
        path = getattr(args, attribute)
        if not path.is_absolute():
            setattr(args, attribute, args.repo_root / path)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = reproduce(args)
    compact = {
        arm_id: {
            "mean": result["fold_mean_trapezoidal_pr_auc"],
            "population_std": result["fold_population_std"],
            "rounded_match": result["rounded_headline_match"],
        }
        for arm_id, result in manifest["results"].items()
    }
    print(json.dumps({"output_dir": str(args.output_dir), "results": compact}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
