#!/usr/bin/env python3
"""Cross-fit current INDRA Counts/Hybrid classes on the five-reader panel.

This is an additive adapter for the exact 1,676-statement reader-only panel.
It does not alter or derive from the frozen 1,689-row Counts/Hybrid outputs.
Only the five reader sources used by the paper (Reach, Sparser, MedScan,
RLIMS-P, and TRIPS) are visible to either direct or refinement-descendant
features.  The exact evaluation label is: **label-isolated OOF conditional on
the frozen global label-free graph**.  The graph is transductive: 13
cross-fold target/descendant pairs and one same-fold pair affect full-feature
inputs.  This adapter therefore makes no fold-isolated or inductive claim.

INDRA 1.24.0 ships the ``CountsScorer`` and ``HybridScorer`` classes but no
public fitted Counts state.  As in the separately frozen all-source adapter,
the estimator architecture is fixed to the paper protocol (2,000 random-
forest trees, depth 13, seed 4).  A direct-source configuration and the full
set of current Counts feature families are fit.  The latter is also wrapped
by the current ``HybridScorer`` with the bundled ``SimpleScorer`` fallback.

Because the projection removes every source outside the Counts source list,
the fallback receives no evidence.  The adapter emits both class paths so the
equivalence can be audited, but marks the local Hybrid as an alias of the full
Counts arm rather than an independent cost/performance point.  Neither local
arm is the fitted CoGEx production Hybrid artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import platform
import stat
import struct
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np
from indra.belief import SimpleScorer, build_refinements_graph, get_ev_for_stmts_from_supports
from indra.belief.skl import CountsScorer, HybridScorer
from indra.statements import Evidence, Statement
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


SCHEMA_VERSION = 1
ARTIFACT_KIND = "current_indra_counts_hybrid_reader_predictions"
PANEL_ID = "readers_only_1676"

ARM_COUNTS_SOURCE = "indra_1.24.0_counts_rf_2kd13_source_only_oof_readers_only"
ARM_COUNTS_FULL = "indra_1.24.0_counts_rf_2kd13_full_features_oof_readers_only"
ARM_HYBRID_FULL = "indra_1.24.0_hybrid_local_rf_2kd13_full_features_oof_readers_only"

PREDICTION_FILENAMES = {
    ARM_COUNTS_SOURCE: "current_counts_source_only_oof_readers_only_predictions.jsonl",
    ARM_COUNTS_FULL: "current_counts_full_features_oof_readers_only_predictions.jsonl",
    ARM_HYBRID_FULL: "current_hybrid_local_full_features_oof_readers_only_predictions.jsonl",
}
FIT_FILENAME = "current_counts_readers_oof_fit_provenance.jsonl"
PROVENANCE_FILENAME = "current_counts_hybrid_readers_prediction_provenance.jsonl"
FOLD_METRICS_FILENAME = "current_counts_hybrid_readers_diagnostic_fold_metrics.jsonl"
MANIFEST_FILENAME = "current_counts_hybrid_readers_manifest.json"

READER_SOURCES = ("reach", "sparser", "medscan", "rlimsp", "trips")
EXPECTED_TARGET_ROWS = 1_689
EXPECTED_READER_ROWS = 1_676
EXPECTED_FOLDS = tuple(range(10))
EXPECTED_PICKLE_ROOTS = 894_939
EXPECTED_GRAPH_NODES = 895_459
EXPECTED_GRAPH_EDGES = 637_573
EXPECTED_READER_CANONICAL_DIRECT_EVIDENCE = 34_022
EXPECTED_READER_PROJECTED_DIRECT_EVIDENCE = 33_152
EXPECTED_READER_CANONICAL_INHERITED_EVIDENCE = 11_466
EXPECTED_READER_PROJECTED_INHERITED_EVIDENCE = 8_837
EXPECTED_READER_ROWS_WITH_PROJECTED_INHERITED_EVIDENCE = 404
EXPECTED_CROSS_FOLD_DESCENDANT_PAIRS = 13
EXPECTED_SAME_FOLD_DESCENDANT_PAIRS = 1
EXPECTED_CROSS_FOLD_AFFECTED_TARGETS = 13
EXPECTED_SAME_FOLD_AFFECTED_TARGETS = 1
EXPECTED_GRAPH_AFFECTED_TARGETS = 14

RF_N_ESTIMATORS = 2_000
RF_MAX_DEPTH = 13
RF_RANDOM_STATE = 4

PINNED_REGISTRY_SHA256 = "ea556a86d55e1e15fc8e618537409a25460b3f69fb7a6fe261ea0fa415fc3f30"
PINNED_COMPARISON_MANIFEST_SHA256 = "db7588ce960ca38385c7e9437e4f8d5ea6c08ffed780addefe50a9c333cbd43e"
PINNED_READER_GOLD_SHA256 = "6c2757dbd64cfd06ed5eb1c46debc2992ed3f2b9f5b936074cb14c02f3856b06"
PINNED_TARGETS_SHA256 = "f962dbe96f6892ccd0ff319fdba28194f175518548e48503d1b60866a9893f6c"
PINNED_ORDERED_READER_ID_SHA256 = "ae64f0c1793fa10df3937e3f11b81eafed6211b8a7dee07af36e534878fa3678"
PINNED_PAPER_MANIFEST_SHA256 = "5011955c4df732074a1db569a17536c2f2c8cc6f5b9d1b84d5af5512977b8026"
PINNED_PAPER_PICKLE_SHA256 = "ed64a2409fc569806a5f6dbdcc72827fb5a6ea0a75cf92452839c6f60d160af2"
PINNED_SOURCE_FEATURE_SHA256 = "a09ede2970b07bd9a06e42f10f87fecfce5594da5560435ba98b8e176b27baa3"
PINNED_FULL_FEATURE_SHA256 = "fcfcbcdfa878604fa52bf29d7fbc0b033b3c4980d33119cf39ecb5aa13df6347"

PINNED_INDRA_VERSION = "1.24.0"
PINNED_INDRA_RELEASE_COMMIT = "aff5d49bf4b24446002bb12c9e6f5f7bd35b090e"
PINNED_BELIEF_INIT_SHA256 = "b64a38bf3b8667d287b9eb24f28e227d01210242aaa7004206174846d668c07c"
PINNED_BELIEF_SKL_SHA256 = "a5ed47937f5f9da61925bde81b289b43513ce158884664dbcc42ba44c1a732d9"
PINNED_DEFAULT_PRIOR_SHA256 = "6c26f48e0a9aa0917c5d605547841c67032d73b22c93f4811d3306f8c7bef9b5"

DEPENDENCY_SOURCE_PINS = {
    "bayesian_panel_adapter": (
        "score_current_indra_bayesian_paper",
        "score_current_indra_bayesian_paper.py",
        "880fe18b54d3fa3468d2a3d864c00a8f4f98b8b32378511572cac48ae729da61",
    ),
    "all_source_counts_adapter": (
        "score_current_indra_counts_hybrid_paper",
        "score_current_indra_counts_hybrid_paper.py",
        "a50c02d2a96b342fc810112fea1b8fde98a17fa40e8152571d9356a78010a0ea",
    ),
    "hierarchy_graph_adapter": (
        "score_current_indra_hierarchy_paper",
        "score_current_indra_hierarchy_paper.py",
        "cd6e316c10e1c849dd800e7c4a370a470141662aa05fa47c7455c93c2ef4d036",
    ),
    "identity_adapter": (
        "score_current_indra_simple_paper",
        "score_current_indra_simple_paper.py",
        "a6839cdf0add20746629af58d0e2e6c0c176ad22db16ecbbc0773eb4922ef657",
    ),
}

TARGET_FIELDS = {
    "canonical_corpus_row_index",
    "eligible_position",
    "matches_hash",
    "reader_eligible",
    "source_row_index",
    "statement_id",
    "statement_json_sha256",
    "statement_type",
}
GOLD_FIELDS = {"fold_id", "label", "statement_id"}
EVALUATION_DESIGN_LABEL = (
    "label-isolated OOF conditional on frozen global label-free graph"
)


class ContractError(ValueError):
    """Raised when reader identity, projection, fit, or provenance gates fail."""


@dataclass(frozen=True)
class PinnedFile:
    """An exact regular-file capture made without following symbolic links."""

    path: Path
    size: int
    sha256: str
    device: int
    inode: int
    content: bytes | None = None

    def descriptor(self) -> dict[str, Any]:
        return {
            "bytes": self.size,
            "path": _display_path(self.path),
            "sha256": self.sha256,
            "verification": "exact_sha256_regular_file_no_symlink",
        }


@dataclass(frozen=True)
class LocalDependencies:
    """Local adapters imported only after all their sources pass exact pins."""

    bayes: ModuleType
    all_source: ModuleType
    hierarchy: ModuleType
    base: ModuleType
    descriptors: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CountsConfiguration:
    """A fully identified reader-panel instantiation of ``CountsScorer``."""

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
    config_id="reader_five_source_current_default_direct_counts_only",
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
    config_id="reader_five_source_all_indra_1.24.0_counts_feature_families",
    include_more_specific=True,
    use_stmt_type=True,
    use_num_members=True,
    use_num_pmids=True,
    use_promoter=True,
    use_avg_evidence_len=True,
    use_residue_position=True,
)
CONFIGURATIONS = (SOURCE_CONFIGURATION, FULL_CONFIGURATION)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        return str(absolute.relative_to(Path.cwd().absolute()))
    except ValueError:
        return str(absolute)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path) -> Path:
    """Return an absolute path after rejecting every symbolic-link component."""
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise ContractError(f"cannot lstat pinned path component {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ContractError(f"symbolic links are forbidden in pinned path: {current}")
    return absolute


def _open_and_verify_pinned_file(
    path: Path,
    expected_sha256: str,
    *,
    expected_size: int | None = None,
    collect_content: bool = False,
) -> tuple[int, PinnedFile]:
    """Open once with O_NOFOLLOW, hash that descriptor, and retain its identity."""
    absolute = _assert_no_symlink_components(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ContractError("O_NOFOLLOW is required for pinned source/input capture")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(absolute, flags)
    except OSError as exc:
        raise ContractError(f"cannot open pinned file {absolute}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError(f"pinned path is not a regular file: {absolute}")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if collect_content else None
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ContractError(f"pinned file changed while hashing: {absolute}")
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise ContractError(
                f"{absolute}: SHA-256 mismatch; expected {expected_sha256}, "
                f"observed {observed_sha256}"
            )
        if expected_size is not None and before.st_size != expected_size:
            raise ContractError(
                f"{absolute}: byte-size mismatch; expected {expected_size}, "
                f"observed {before.st_size}"
            )
        current = os.lstat(absolute)
        if stat.S_ISLNK(current.st_mode) or (
            current.st_dev,
            current.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise ContractError(f"pinned path identity changed while hashing: {absolute}")
        os.lseek(fd, 0, os.SEEK_SET)
        capture = PinnedFile(
            path=absolute,
            size=before.st_size,
            sha256=observed_sha256,
            device=before.st_dev,
            inode=before.st_ino,
            content=b"".join(chunks) if chunks is not None else None,
        )
        return fd, capture
    except BaseException:
        os.close(fd)
        raise


def _capture_pinned_file(
    path: Path,
    expected_sha256: str,
    *,
    expected_size: int | None = None,
    collect_content: bool = False,
) -> PinnedFile:
    fd, capture = _open_and_verify_pinned_file(
        path,
        expected_sha256,
        expected_size=expected_size,
        collect_content=collect_content,
    )
    os.close(fd)
    return capture


def _json_from_capture(capture: PinnedFile) -> dict[str, Any]:
    if capture.content is None:
        raise ContractError(f"{capture.path}: JSON capture has no retained bytes")
    try:
        value = json.loads(capture.content, object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid strict JSON {capture.path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{capture.path}: expected a JSON object")
    return value


def _jsonl_from_capture(capture: PinnedFile) -> list[dict[str, Any]]:
    if capture.content is None:
        raise ContractError(f"{capture.path}: JSONL capture has no retained bytes")
    try:
        text = capture.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"invalid UTF-8 JSONL {capture.path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ContractError(f"{capture.path}:{line_number}: blank line is forbidden")
        try:
            row = json.loads(line, object_pairs_hook=_strict_pairs)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"{capture.path}:{line_number}: invalid strict JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ContractError(f"{capture.path}:{line_number}: expected an object")
        rows.append(row)
    if not rows:
        raise ContractError(f"{capture.path}: empty JSONL is forbidden")
    return rows


def _parse_targets(capture: PinnedFile) -> list[dict[str, Any]]:
    rows = _jsonl_from_capture(capture)
    seen_ids: set[str] = set()
    seen_corpus_rows: set[int] = set()
    for index, row in enumerate(rows):
        if set(row) != TARGET_FIELDS:
            raise ContractError(f"{capture.path}:{index + 1}: target fields are not exact")
        if row["eligible_position"] != index:
            raise ContractError(f"{capture.path}:{index + 1}: target order is not contiguous")
        statement_id = row["statement_id"]
        corpus_row = row["canonical_corpus_row_index"]
        source_row = row["source_row_index"]
        if not isinstance(statement_id, str) or not statement_id or statement_id in seen_ids:
            raise ContractError(f"{capture.path}:{index + 1}: invalid/duplicate statement_id")
        for field, value in (
            ("canonical_corpus_row_index", corpus_row),
            ("source_row_index", source_row),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"{capture.path}:{index + 1}: invalid {field}")
        if corpus_row in seen_corpus_rows:
            raise ContractError(f"{capture.path}:{index + 1}: duplicate canonical row")
        if not isinstance(row["reader_eligible"], bool):
            raise ContractError(f"{capture.path}:{index + 1}: reader_eligible is not bool")
        for field in ("matches_hash", "statement_type"):
            if not isinstance(row[field], str) or not row[field]:
                raise ContractError(f"{capture.path}:{index + 1}: invalid {field}")
        digest = row["statement_json_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ContractError(f"{capture.path}:{index + 1}: invalid statement digest")
        seen_ids.add(statement_id)
        seen_corpus_rows.add(corpus_row)
    return rows


def _parse_gold(capture: PinnedFile) -> list[dict[str, Any]]:
    rows = _jsonl_from_capture(capture)
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != GOLD_FIELDS:
            raise ContractError(f"{capture.path}:{index + 1}: gold fields are not exact")
        statement_id = row["statement_id"]
        if not isinstance(statement_id, str) or not statement_id or statement_id in seen:
            raise ContractError(f"{capture.path}:{index + 1}: invalid/duplicate statement_id")
        if isinstance(row["label"], bool) or row["label"] not in (0, 1):
            raise ContractError(f"{capture.path}:{index + 1}: label is not integer 0/1")
        if (
            isinstance(row["fold_id"], bool)
            or not isinstance(row["fold_id"], int)
            or row["fold_id"] not in EXPECTED_FOLDS
        ):
            raise ContractError(f"{capture.path}:{index + 1}: fold_id is not integer 0..9")
        seen.add(statement_id)
    if {row["fold_id"] for row in rows} != set(EXPECTED_FOLDS):
        raise ContractError(f"{capture.path}: all ten folds are required")
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row) + b"\n")


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
    return hashlib.sha256(
        struct.pack("<QQ", *value.shape) + value.tobytes(order="C")
    ).hexdigest()


def _row_sha(row: np.ndarray) -> str:
    value = np.asarray(row, dtype="<f8", order="C")
    return hashlib.sha256(
        struct.pack("<Q", value.size) + value.tobytes(order="C")
    ).hexdigest()


def _json_value(value: Any) -> Any:
    """Convert estimator parameters to deterministic strict-JSON values."""
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
    """Return the externally fixed RF architecture, never a tuned reader fit."""
    estimator = RandomForestClassifier(
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
    parameters = estimator.get_params(deep=False)
    exact = {
        "n_estimators": 2_000,
        "max_depth": 13,
        "random_state": 4,
        "max_features": "sqrt",
    }
    if any(parameters[key] != value for key, value in exact.items()):
        raise ContractError("constructed RF identity drifted")
    return estimator


def _assert_rf_estimator_contract(estimator: BaseEstimator) -> None:
    if not isinstance(estimator, RandomForestClassifier):
        raise ContractError("fit estimator is not exact RandomForestClassifier")
    parameters = estimator.get_params(deep=False)
    exact = {
        "n_estimators": 2_000,
        "criterion": "gini",
        "max_depth": 13,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "random_state": 4,
        "bootstrap": True,
        "oob_score": False,
        "verbose": 0,
        "warm_start": False,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "max_samples": None,
    }
    if "monotonic_cst" in parameters:
        exact["monotonic_cst"] = None
    drift = {
        key: {"expected": value, "observed": parameters.get(key)}
        for key, value in exact.items()
        if parameters.get(key) != value
    }
    if drift:
        raise ContractError(f"RF fit estimator contract drifted: {drift}")


def _counts_scorer(
    configuration: CountsConfiguration, estimator: BaseEstimator
) -> CountsScorer:
    return CountsScorer(estimator, list(READER_SOURCES), **configuration.kwargs())


def _feature_names(scorer: CountsScorer) -> list[str]:
    names = [f"direct_source_count:{source}" for source in scorer.source_list]
    if scorer.include_more_specific:
        names.extend(
            f"more_specific_source_count:{source}" for source in scorer.source_list
        )
    if scorer.use_stmt_type:
        ordered_types = [
            name
            for name, _ in sorted(
                scorer.stmt_type_map.items(), key=lambda item: item[1]
            )
        ]
        names.extend(f"statement_type:{name}" for name in ordered_types)
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
    exact_flags = {
        SOURCE_CONFIGURATION.config_id: {
            "include_more_specific": False,
            "use_stmt_type": False,
            "use_num_members": False,
            "use_num_pmids": False,
            "use_promoter": False,
            "use_avg_evidence_len": False,
            "use_residue_position": False,
        },
        FULL_CONFIGURATION.config_id: {
            "include_more_specific": True,
            "use_stmt_type": True,
            "use_num_members": True,
            "use_num_pmids": True,
            "use_promoter": True,
            "use_avg_evidence_len": True,
            "use_residue_position": True,
        },
    }
    expected_hashes = {
        SOURCE_CONFIGURATION.config_id: (5, PINNED_SOURCE_FEATURE_SHA256),
        FULL_CONFIGURATION.config_id: (65, PINNED_FULL_FEATURE_SHA256),
    }
    if configuration.config_id not in exact_flags:
        raise ContractError(f"unidentified Counts configuration {configuration.config_id!r}")
    if configuration.kwargs() != exact_flags[configuration.config_id]:
        raise ContractError(f"{configuration.config_id}: feature flags drifted")
    scorer = _counts_scorer(configuration, RandomForestClassifier(n_estimators=1))
    if scorer.source_list != list(READER_SOURCES):
        raise ContractError(f"{configuration.config_id}: reader source order drifted")
    names = _feature_names(scorer)
    expected_count, expected_sha256 = expected_hashes[configuration.config_id]
    observed_sha256 = hashlib.sha256(_canonical_bytes(names)).hexdigest()
    if len(names) != expected_count:
        raise ContractError(
            f"{configuration.config_id}: feature count {len(names)} != {expected_count}"
        )
    if observed_sha256 != expected_sha256:
        raise ContractError(
            f"{configuration.config_id}: feature-name SHA-256 drift; expected "
            f"{expected_sha256}, observed {observed_sha256}"
        )
    return {
        "config_id": configuration.config_id,
        "flags": configuration.kwargs(),
        "feature_count": len(names),
        "feature_names": names,
        "feature_names_sha256": observed_sha256,
        "source_list_order": list(READER_SOURCES),
        "direct_evidence_multiplicity": "raw projected entries retained by CountsScorer",
        "more_specific_evidence_semantics": (
            "current get_ev_for_stmts_from_supports transitive descendants, then frozen "
            "five-reader source projection; negated evidence excluded and duplicate "
            "Evidence objects collapsed by the current helper"
            if configuration.include_more_specific
            else "not used"
        ),
        "non_source_features_are_reader_projected": True,
    }


def _verify_literal_contract() -> dict[str, Any]:
    """Fail before fitting if any scientific literal or feature identity drifts."""
    if PANEL_ID != "readers_only_1676":
        raise ContractError("reader panel literal drifted")
    if READER_SOURCES != ("reach", "sparser", "medscan", "rlimsp", "trips"):
        raise ContractError("ordered reader-source literal drifted")
    if (RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE) != (2_000, 13, 4):
        raise ContractError("RF 2000/depth-13/seed-4 literal drifted")
    contracts = {
        configuration.config_id: _feature_contract(configuration)
        for configuration in CONFIGURATIONS
    }
    if set(contracts) != {
        "reader_five_source_current_default_direct_counts_only",
        "reader_five_source_all_indra_1.24.0_counts_feature_families",
    }:
        raise ContractError("Counts configuration identity set drifted")
    return contracts


def _source_counts(evidence: Sequence[Evidence]) -> dict[str, int]:
    return dict(sorted(Counter(str(item.source_api) for item in evidence).items()))


def _project_statement(statement: Statement) -> Statement:
    """Shallow-copy a statement and expose only direct five-reader evidence."""
    projected = copy.copy(statement)
    projected.evidence = [
        evidence
        for evidence in statement.evidence
        if evidence.source_api in READER_SOURCES
    ]
    # Descendant evidence is passed explicitly.  Detaching these links prevents
    # HybridScorer.check_prior_probs from traversing non-reader graph evidence.
    projected.supports = []
    projected.supported_by = []
    # The release object can carry a previously computed belief.  No Counts
    # feature consumes it, and clearing it makes that non-input status explicit.
    projected.belief = None
    return projected


def _project_reader_inputs(
    statements: Sequence[Statement],
    extra_evidence: Sequence[Sequence[Evidence]],
    targets: Sequence[dict[str, Any]],
    reader_gold: Sequence[dict[str, Any]],
) -> tuple[list[Statement], list[list[Evidence]], list[int], list[dict[str, Any]]]:
    """Apply and prove the exact row/source projection used by reader arms."""
    if not (len(statements) == len(extra_evidence) == len(targets)):
        raise ContractError("statement, descendant-evidence, and target lengths differ")
    if [statement.uuid for statement in statements] != [
        row["statement_id"] for row in targets
    ]:
        raise ContractError("statement/target identity order differs")
    reader_positions = [
        index for index, row in enumerate(targets) if row["reader_eligible"]
    ]
    expected_reader_ids = [targets[index]["statement_id"] for index in reader_positions]
    if [row["statement_id"] for row in reader_gold] != expected_reader_ids:
        raise ContractError("reader gold is not the ordered target reader projection")

    projected_statements: list[Statement] = []
    projected_extra: list[list[Evidence]] = []
    audits: list[dict[str, Any]] = []
    allowed = set(READER_SOURCES)
    for gold, position in zip(reader_gold, reader_positions, strict=True):
        statement = statements[position]
        direct = _project_statement(statement)
        inherited = [
            evidence
            for evidence in extra_evidence[position]
            if evidence.source_api in allowed
        ]
        if not direct.evidence:
            raise ContractError(
                f"{gold['statement_id']}: reader-eligible projection has no direct evidence"
            )
        visible_sources = {
            evidence.source_api for evidence in [*direct.evidence, *inherited]
        }
        if not visible_sources.issubset(allowed):
            raise ContractError(f"{gold['statement_id']}: source projection failed")
        if direct.supports or direct.supported_by:
            raise ContractError(f"{gold['statement_id']}: projected graph links remain")
        projected_statements.append(direct)
        projected_extra.append(inherited)
        audits.append(
            {
                "canonical_direct_evidence_count": len(statement.evidence),
                "canonical_direct_source_counts": _source_counts(statement.evidence),
                "canonical_inherited_evidence_count": len(extra_evidence[position]),
                "canonical_inherited_source_counts": _source_counts(
                    extra_evidence[position]
                ),
                "input_sources": list(READER_SOURCES),
                "panel_id": PANEL_ID,
                "projected_direct_evidence_count": len(direct.evidence),
                "projected_direct_source_counts": _source_counts(direct.evidence),
                "projected_inherited_evidence_count": len(inherited),
                "projected_inherited_source_counts": _source_counts(inherited),
                "removed_direct_evidence_count": len(statement.evidence)
                - len(direct.evidence),
                "removed_inherited_evidence_count": len(extra_evidence[position])
                - len(inherited),
                "statement_id": gold["statement_id"],
                "stored_statement_belief_cleared": True,
                "support_links_detached": True,
            }
        )
    return projected_statements, projected_extra, reader_positions, audits


def _graph_fold_dependency_audit(
    statements: Sequence[Statement],
    gold_rows: Sequence[dict[str, Any]],
    graph: nx.DiGraph,
    *,
    enforce_frozen_census: bool = True,
) -> dict[str, Any]:
    """Record target-to-target descendant inputs crossing frozen fold boundaries."""
    if len(statements) != len(gold_rows) or [statement.uuid for statement in statements] != [
        row["statement_id"] for row in gold_rows
    ]:
        raise ContractError("graph dependency audit statement/gold identity order differs")
    hash_to_position: dict[int, int] = {}
    for position, statement in enumerate(statements):
        statement_hash = statement.get_hash()
        if statement_hash in hash_to_position:
            raise ContractError("reader target hashes collide in graph dependency audit")
        hash_to_position[statement_hash] = position

    pairs: list[dict[str, Any]] = []
    affected_positions: set[int] = set()
    cross_fold_affected_positions: set[int] = set()
    same_fold_affected_positions: set[int] = set()
    for target_position, statement in enumerate(statements):
        descendant_positions = sorted(
            hash_to_position[node]
            for node in nx.descendants(graph, statement.get_hash())
            if node in hash_to_position
        )
        for descendant_position in descendant_positions:
            target_gold = gold_rows[target_position]
            descendant_gold = gold_rows[descendant_position]
            crosses_fold = target_gold["fold_id"] != descendant_gold["fold_id"]
            pairs.append(
                {
                    "affects_arms": [ARM_COUNTS_FULL, ARM_HYBRID_FULL],
                    "contributing_descendant_fold_id": descendant_gold["fold_id"],
                    "contributing_descendant_statement_id": descendant_gold[
                        "statement_id"
                    ],
                    "crosses_fold_boundary": crosses_fold,
                    "feature_target_fold_id": target_gold["fold_id"],
                    "feature_target_statement_id": target_gold["statement_id"],
                    "source_only_arm_affected": False,
                }
            )
            affected_positions.add(target_position)
            if crosses_fold:
                cross_fold_affected_positions.add(target_position)
            else:
                same_fold_affected_positions.add(target_position)
    cross_fold = sum(row["crosses_fold_boundary"] for row in pairs)
    same_fold = len(pairs) - cross_fold
    census = {
        "cross_fold_descendant_pairs": cross_fold,
        "cross_fold_targets_affected": len(cross_fold_affected_positions),
        "same_fold_descendant_pairs": same_fold,
        "same_fold_targets_affected": len(same_fold_affected_positions),
        "targets_affected": len(affected_positions),
        "total_target_descendant_pairs": len(pairs),
    }
    expected = {
        "cross_fold_descendant_pairs": EXPECTED_CROSS_FOLD_DESCENDANT_PAIRS,
        "cross_fold_targets_affected": EXPECTED_CROSS_FOLD_AFFECTED_TARGETS,
        "same_fold_descendant_pairs": EXPECTED_SAME_FOLD_DESCENDANT_PAIRS,
        "same_fold_targets_affected": EXPECTED_SAME_FOLD_AFFECTED_TARGETS,
        "targets_affected": EXPECTED_GRAPH_AFFECTED_TARGETS,
        "total_target_descendant_pairs": (
            EXPECTED_CROSS_FOLD_DESCENDANT_PAIRS
            + EXPECTED_SAME_FOLD_DESCENDANT_PAIRS
        ),
    }
    if enforce_frozen_census and census != expected:
        raise ContractError(f"target descendant fold census drifted: {census} != {expected}")
    return {
        **census,
        "design_label": EVALUATION_DESIGN_LABEL,
        "fold_isolated_feature_construction": False,
        "inductive_evaluation": False,
        "label_isolated_model_fitting": True,
        "ordered_pair_sha256": hashlib.sha256(_canonical_bytes(pairs)).hexdigest(),
        "pairs": pairs,
    }


def _prediction(statement_id: str, probability: float) -> dict[str, Any]:
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ContractError(f"{statement_id}: invalid probability {value!r}")
    return {"probability_correct": value, "statement_id": statement_id}


def _fitted_state_sha(scorer: CountsScorer) -> tuple[str, int]:
    payload = pickle.dumps(scorer, protocol=5)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _assert_cross_fit_input_state(
    statements: Sequence[Statement], extra_evidence: Sequence[Sequence[Evidence]]
) -> None:
    """Reassert the sterile reader projection at the immediate fit boundary."""
    allowed = set(READER_SOURCES)
    for index, (statement, inherited) in enumerate(
        zip(statements, extra_evidence, strict=True)
    ):
        if statement.supports != [] or statement.supported_by != []:
            raise ContractError(
                f"cross-fit input {index}: stale supports/supported_by graph state"
            )
        if statement.belief is not None:
            raise ContractError(f"cross-fit input {index}: stale stored statement belief")
        if not statement.evidence:
            raise ContractError(f"cross-fit input {index}: evidence-free statement")
        direct_outside = {
            evidence.source_api
            for evidence in statement.evidence
            if evidence.source_api not in allowed
        }
        inherited_outside = {
            evidence.source_api
            for evidence in inherited
            if evidence.source_api not in allowed
        }
        if direct_outside:
            raise ContractError(
                f"cross-fit input {index}: direct evidence outside reader sources: "
                f"{sorted(direct_outside)}"
            )
        if inherited_outside:
            raise ContractError(
                f"cross-fit input {index}: extra evidence outside reader sources: "
                f"{sorted(inherited_outside)}"
            )


def _cross_fit(
    statements: Sequence[Statement],
    extra_evidence: Sequence[Sequence[Evidence]],
    gold_rows: Sequence[dict[str, Any]],
    *,
    estimator_factory: Callable[[int], BaseEstimator],
    projection_audits: Sequence[dict[str, Any]] | None = None,
    enforce_frozen_contract: bool = True,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run label-isolated OOF fits conditional on the frozen global graph."""
    if not (len(statements) == len(extra_evidence) == len(gold_rows)):
        raise ContractError("statement, extra-evidence, and reader-gold lengths differ")
    _assert_cross_fit_input_state(statements, extra_evidence)
    _verify_literal_contract()
    expected_ids = [row["statement_id"] for row in gold_rows]
    if enforce_frozen_contract and (
        len(gold_rows) != EXPECTED_READER_ROWS
        or _ordered_id_sha(gold_rows) != PINNED_ORDERED_READER_ID_SHA256
    ):
        raise ContractError("cross-fit reader panel row/order contract drifted")
    if [statement.uuid for statement in statements] != expected_ids:
        raise ContractError("projected statement/reader-gold order differs")
    if projection_audits is None:
        projection_audits = [
            {
                "input_sources": list(READER_SOURCES),
                "panel_id": PANEL_ID,
                "statement_id": statement_id,
            }
            for statement_id in expected_ids
        ]
    if len(projection_audits) != len(gold_rows) or [
        row.get("statement_id") for row in projection_audits
    ] != expected_ids:
        raise ContractError("projection audit/reader-gold order differs")
    allowed = set(READER_SOURCES)

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
    projection_by_id = {row["statement_id"]: row for row in projection_audits}
    max_hybrid_difference = 0.0
    exact_hybrid_rows = 0
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
            if enforce_frozen_contract:
                _assert_rf_estimator_contract(estimator)
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
            if configuration is FULL_CONFIGURATION:
                hybrid = HybridScorer(scorer, SimpleScorer())
                hybrid.check_prior_probs(test_statements)
                hybrid_scores = np.asarray(
                    hybrid.score_statements(test_statements, test_extra), dtype=float
                )
                unknown_by_test = [
                    [
                        evidence
                        for evidence in [*statement.evidence, *inherited]
                        if evidence.source_api not in allowed
                    ]
                    for statement, inherited in zip(
                        test_statements,
                        test_extra or [[] for _ in test_statements],
                        strict=True,
                    )
                ]
                fold_fallback_entries = sum(map(len, unknown_by_test))
                hybrid_fallback_entries += fold_fallback_entries
                if fold_fallback_entries:
                    raise ContractError(
                        f"fold {fold_id}: local Hybrid fallback received reader-external evidence"
                    )
                differences = np.abs(hybrid_scores - class_scores)
                fold_difference = float(np.max(differences))
                max_hybrid_difference = max(max_hybrid_difference, fold_difference)
                exact_hybrid_rows += int(np.sum(hybrid_scores == class_scores))
                if fold_difference > 4 * np.finfo(float).eps:
                    raise ContractError(
                        f"fold {fold_id}: zero-fallback Hybrid differs from Counts by "
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
                        f"{type(scorer.model).__module__}."
                        f"{type(scorer.model).__qualname__}"
                    ),
                    "estimator_parameters": estimator_parameters,
                    "feature_names_sha256": contract["feature_names_sha256"],
                    "fit_api": "indra.belief.skl.CountsScorer.fit",
                    "fit_fold_id": fold_id,
                    "fitted_state_pickle_bytes": fitted_bytes,
                    "fitted_state_pickle_protocol": 5,
                    "fitted_state_sha256": fitted_sha,
                    "input_sources": list(READER_SOURCES),
                    "outside_reader_statement_labels_available_to_adapter": False,
                    "panel_id": PANEL_ID,
                    "evaluation_design": EVALUATION_DESIGN_LABEL,
                    "reused_by_hybrid_arm_id": (
                        ARM_HYBRID_FULL
                        if configuration is FULL_CONFIGURATION
                        else None
                    ),
                    "source_list_order": list(READER_SOURCES),
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
                provenance_maps[configuration.arm_id][statement_id] = {
                    "arm_id": configuration.arm_id,
                    "config_id": configuration.config_id,
                    "feature_vector_sha256": _row_sha(test_matrix[local_index]),
                    "fit_fold_id": fold_id,
                    "probability_correct": counts_prediction["probability_correct"],
                    **projection_by_id[statement_id],
                }

                if configuration is FULL_CONFIGURATION:
                    assert hybrid_scores is not None
                    hybrid_prediction = _prediction(
                        statement_id, hybrid_scores[local_index]
                    )
                    prediction_maps[ARM_HYBRID_FULL][statement_id] = hybrid_prediction
                    provenance_maps[ARM_HYBRID_FULL][statement_id] = {
                        "arm_id": ARM_HYBRID_FULL,
                        "config_id": configuration.config_id,
                        "counts_probability_correct": counts_prediction[
                            "probability_correct"
                        ],
                        "abs_hybrid_minus_counts_probability": abs(
                            hybrid_prediction["probability_correct"]
                            - counts_prediction["probability_correct"]
                        ),
                        "feature_vector_sha256": _row_sha(test_matrix[local_index]),
                        "fit_fold_id": fold_id,
                        "probability_correct": hybrid_prediction["probability_correct"],
                        "simple_fallback_evidence_count": 0,
                        "simple_fallback_source_counts": {},
                        **projection_by_id[statement_id],
                    }

    predictions: dict[str, list[dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    for arm_id in PREDICTION_FILENAMES:
        if set(prediction_maps[arm_id]) != set(expected_ids):
            raise ContractError(f"{arm_id}: incomplete OOF coverage")
        predictions[arm_id] = [prediction_maps[arm_id][item] for item in expected_ids]
        provenance.extend(provenance_maps[arm_id][item] for item in expected_ids)

    hybrid_rows = len(expected_ids)
    audit = {
        "alias_of_arm_id": ARM_COUNTS_FULL,
        "comparison_arm_id": ARM_HYBRID_FULL,
        "exact_equal_probability_rows": exact_hybrid_rows,
        "hybrid_simple_fallback_evidence_entries": hybrid_fallback_entries,
        "max_abs_hybrid_minus_counts_probability": max_hybrid_difference,
        "numerically_equivalent_within_tolerance": True,
        "pareto_point_policy": "alias_of_counts_component_not_a_distinct_model_point",
        "rows_compared": hybrid_rows,
        "tolerance": 4 * np.finfo(float).eps,
        "unequal_probability_rows": hybrid_rows - exact_hybrid_rows,
    }
    return predictions, provenance, fits, audit


def _descriptor(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "path": _display_path(path),
        "rows": len(rows),
        "sha256": _sha256(path),
    }


def _module_descriptor(obj: Any) -> dict[str, Any]:
    path = _absolute_path(Path(inspect.getsourcefile(obj) or ""))
    if not path.is_file():
        raise ContractError(f"cannot locate source for {obj!r}")
    return {
        "bytes": path.stat().st_size,
        "path": _display_path(path),
        "sha256": _sha256(path),
    }


def _preverify_dependency_sources(
    scripts_dir: Path | None = None,
) -> dict[str, PinnedFile]:
    """Pin every local dependency before any of those modules can be imported."""
    directory = _assert_no_symlink_components(
        scripts_dir if scripts_dir is not None else _absolute_path(Path(__file__)).parent
    )
    captures: dict[str, PinnedFile] = {}
    for role, (_, filename, expected_sha256) in DEPENDENCY_SOURCE_PINS.items():
        captures[role] = _capture_pinned_file(
            directory / filename, expected_sha256, collect_content=False
        )
    return captures


def _import_preverified_dependencies(
    captures: dict[str, PinnedFile],
) -> LocalDependencies:
    if set(captures) != set(DEPENDENCY_SOURCE_PINS):
        raise ContractError("preverified local dependency set is not exact")
    imported: dict[str, ModuleType] = {}
    # Import the shared identity module first because the other adapters import it.
    order = (
        "identity_adapter",
        "bayesian_panel_adapter",
        "hierarchy_graph_adapter",
        "all_source_counts_adapter",
    )
    for role in order:
        module_name, _, expected_sha256 = DEPENDENCY_SOURCE_PINS[role]
        expected_path = captures[role].path
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise ContractError(f"cannot resolve pinned local dependency {module_name}")
        spec_origin = _absolute_path(Path(spec.origin))
        if spec_origin != expected_path:
            raise ContractError(
                f"{module_name}: import origin {spec_origin} != pinned {expected_path}"
            )
        module = importlib.import_module(module_name)
        origin_value = getattr(module, "__file__", None)
        if not isinstance(origin_value, str):
            raise ContractError(f"{module_name}: imported module has no source origin")
        origin = _absolute_path(Path(origin_value))
        if origin != expected_path:
            raise ContractError(
                f"{module_name}: imported origin {origin} != pinned {expected_path}"
            )
        post = _capture_pinned_file(origin, expected_sha256, collect_content=False)
        if (post.device, post.inode, post.size, post.sha256) != (
            captures[role].device,
            captures[role].inode,
            captures[role].size,
            captures[role].sha256,
        ):
            raise ContractError(f"{module_name}: source identity changed across import")
        imported[role] = module
    dependencies = LocalDependencies(
        bayes=imported["bayesian_panel_adapter"],
        all_source=imported["all_source_counts_adapter"],
        hierarchy=imported["hierarchy_graph_adapter"],
        base=imported["identity_adapter"],
        descriptors={role: capture.descriptor() for role, capture in captures.items()},
    )
    if dependencies.bayes.PANEL_READERS != PANEL_ID:
        raise ContractError("pinned Bayesian dependency reader panel literal drifted")
    if tuple(dependencies.bayes.READER_SOURCES) != READER_SOURCES:
        raise ContractError("pinned Bayesian dependency source order drifted")
    if (
        dependencies.all_source.RF_N_ESTIMATORS,
        dependencies.all_source.RF_MAX_DEPTH,
        dependencies.all_source.RF_RANDOM_STATE,
    ) != (RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE):
        raise ContractError("pinned all-source dependency RF identity drifted")
    if (
        dependencies.hierarchy.EXPECTED_GRAPH_NODES,
        dependencies.hierarchy.EXPECTED_GRAPH_EDGES,
    ) != (EXPECTED_GRAPH_NODES, EXPECTED_GRAPH_EDGES):
        raise ContractError("pinned hierarchy dependency graph census drifted")
    if set(dependencies.base.TARGET_FIELDS) != TARGET_FIELDS:
        raise ContractError("pinned identity dependency target schema drifted")
    return dependencies


def _load_pinned_dependencies() -> LocalDependencies:
    return _import_preverified_dependencies(_preverify_dependency_sources())


def _comparison_contract(
    comparison_manifest_path: Path,
    targets_path: Path,
    reader_gold_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Hard-pin and cross-link the exact comparison manifest and reader ledgers."""
    manifest_capture = _capture_pinned_file(
        comparison_manifest_path,
        PINNED_COMPARISON_MANIFEST_SHA256,
        collect_content=True,
    )
    targets_capture = _capture_pinned_file(
        targets_path, PINNED_TARGETS_SHA256, collect_content=True
    )
    gold_capture = _capture_pinned_file(
        reader_gold_path, PINNED_READER_GOLD_SHA256, collect_content=True
    )
    manifest = _json_from_capture(manifest_capture)
    targets = _parse_targets(targets_capture)
    reader_gold = _parse_gold(gold_capture)
    if len(targets) != EXPECTED_TARGET_ROWS or len(reader_gold) != EXPECTED_READER_ROWS:
        raise ContractError("frozen target/reader row census drifted")
    reader_positions = [
        index for index, row in enumerate(targets) if row["reader_eligible"]
    ]
    if len(reader_positions) != EXPECTED_READER_ROWS or [
        row["statement_id"] for row in reader_gold
    ] != [targets[index]["statement_id"] for index in reader_positions]:
        raise ContractError("reader gold identity/order differs from target projection")
    observed_id_sha256 = _ordered_id_sha(reader_gold)
    if observed_id_sha256 != PINNED_ORDERED_READER_ID_SHA256:
        raise ContractError("reader ordered statement-ID SHA-256 drifted")
    try:
        target_declared = manifest["outputs"]["paper_prediction_targets"]
        gold_declared = manifest["outputs"]["paper_reader_eligible_released_gold"]
    except (KeyError, TypeError) as exc:
        raise ContractError("comparison manifest lacks exact reader inputs") from exc
    exact_descriptors = (
        (
            "prediction targets",
            target_declared,
            targets_capture,
            EXPECTED_TARGET_ROWS,
            None,
        ),
        (
            "reader gold",
            gold_declared,
            gold_capture,
            EXPECTED_READER_ROWS,
            PINNED_ORDERED_READER_ID_SHA256,
        ),
    )
    for context, declared, capture, rows, ordered_sha in exact_descriptors:
        if not isinstance(declared, dict):
            raise ContractError(f"comparison manifest {context} descriptor is invalid")
        if declared.get("sha256") != capture.sha256 or declared.get("rows") != rows:
            raise ContractError(f"comparison manifest {context} descriptor drifted")
        if ordered_sha is not None and declared.get("ordered_statement_id_sha256") != ordered_sha:
            raise ContractError(f"comparison manifest {context} ordered-ID digest drifted")
    target_descriptor = targets_capture.descriptor()
    target_descriptor.update(rows=len(targets))
    gold_descriptor = gold_capture.descriptor()
    gold_descriptor.update(
        rows=len(reader_gold), ordered_statement_id_sha256=observed_id_sha256
    )
    return (
        manifest_capture.descriptor(),
        target_descriptor,
        gold_descriptor,
        targets,
        reader_gold,
    )


def _paper_manifest_contract(
    paper_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hard-pin the paper manifest and its exact canonical-pickle declaration."""
    capture = _capture_pinned_file(
        paper_manifest_path, PINNED_PAPER_MANIFEST_SHA256, collect_content=True
    )
    manifest = _json_from_capture(capture)
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ContractError("paper manifest files must be an array")
    matches = [
        row
        for row in files
        if isinstance(row, dict) and row.get("filename") == "indra_benchmark_corpus.pkl"
    ]
    if len(matches) != 1:
        raise ContractError("paper manifest must identify exactly one canonical pickle")
    declared = matches[0]
    if declared.get("sha256") != PINNED_PAPER_PICKLE_SHA256:
        raise ContractError("paper manifest canonical pickle SHA-256 drifted")
    if (
        isinstance(declared.get("bytes"), bool)
        or not isinstance(declared.get("bytes"), int)
        or declared["bytes"] <= 0
    ):
        raise ContractError("paper manifest canonical pickle byte size is invalid")
    if declared.get("canonical_for_historical_object_and_refinement_parity") is not True:
        raise ContractError("paper manifest does not designate the canonical graph pickle")
    return capture.descriptor(), dict(declared)


def _load_pinned_pickle(
    pickle_path: Path, declared: dict[str, Any]
) -> tuple[list[Statement], dict[str, Any]]:
    """Hash and unpickle the same no-follow descriptor, closing the race window."""
    fd, capture = _open_and_verify_pinned_file(
        pickle_path,
        PINNED_PAPER_PICKLE_SHA256,
        expected_size=declared["bytes"],
        collect_content=False,
    )
    try:
        with os.fdopen(fd, "rb", closefd=True) as stream:
            value = pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        raise ContractError(f"could not load exact paper pickle: {exc}") from exc
    if not isinstance(value, list) or len(value) != EXPECTED_PICKLE_ROOTS:
        raise ContractError("paper pickle root list census drifted")
    if any(not isinstance(statement, Statement) for statement in value):
        raise ContractError("paper pickle contains a non-Statement root")
    descriptor = capture.descriptor()
    descriptor.update(
        verification="exact_sha256_same_descriptor_before_trusted_unpickle",
        roots=len(value),
    )
    return value, descriptor


def _registry_contract(
    path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    capture = _capture_pinned_file(
        path, PINNED_REGISTRY_SHA256, collect_content=True
    )
    registry = _json_from_capture(capture)
    scorer_rows = registry.get("scorers")
    if not isinstance(scorer_rows, list):
        raise ContractError("scorer registry scorers must be an array")
    rows: dict[str, dict[str, Any]] = {}
    for row in scorer_rows:
        if not isinstance(row, dict) or not isinstance(row.get("scorer_id"), str):
            raise ContractError("scorer registry contains an invalid scorer row")
        scorer_id = row["scorer_id"]
        if scorer_id in rows:
            raise ContractError(f"scorer registry repeats {scorer_id}")
        rows[scorer_id] = row
    installed = importlib.metadata.version("indra")
    release = registry.get("source_snapshots", {}).get("indra_current_release", {})
    if (
        installed != PINNED_INDRA_VERSION
        or release.get("version") != PINNED_INDRA_VERSION
        or release.get("commit") != PINNED_INDRA_RELEASE_COMMIT
    ):
        raise ContractError("installed INDRA and frozen release identity are inconsistent")
    required_classes = {
        "indra_1.24.0_simple_default": "indra.belief.SimpleScorer",
        "indra_1.24.0_counts_unfitted": "indra.belief.skl.CountsScorer",
        "indra_1.24.0_hybrid_unfitted": "indra.belief.skl.HybridScorer",
    }
    if not set(required_classes).issubset(rows):
        raise ContractError("scorer registry lacks a required reader-arm class")
    for scorer_id, expected_class in required_classes.items():
        if rows[scorer_id].get("class") != expected_class:
            raise ContractError(f"{scorer_id}: exact class identity drifted")
        if rows[scorer_id].get("commit") != PINNED_INDRA_RELEASE_COMMIT:
            raise ContractError(f"{scorer_id}: release commit drifted")

    runtime_classes = {
        "indra_1.24.0_simple_default": SimpleScorer,
        "indra_1.24.0_counts_unfitted": CountsScorer,
        "indra_1.24.0_hybrid_unfitted": HybridScorer,
    }
    for scorer_id, cls in runtime_classes.items():
        actual_class = f"{cls.__module__}.{cls.__qualname__}"
        if actual_class != required_classes[scorer_id]:
            raise ContractError(f"runtime class for {scorer_id} is {actual_class}")

    belief_path = Path(inspect.getsourcefile(SimpleScorer) or "")
    counts_path = Path(inspect.getsourcefile(CountsScorer) or "")
    hybrid_path = Path(inspect.getsourcefile(HybridScorer) or "")
    belief_capture = _capture_pinned_file(
        belief_path, PINNED_BELIEF_INIT_SHA256, collect_content=False
    )
    counts_capture = _capture_pinned_file(
        counts_path, PINNED_BELIEF_SKL_SHA256, collect_content=False
    )
    hybrid_capture = _capture_pinned_file(
        hybrid_path, PINNED_BELIEF_SKL_SHA256, collect_content=False
    )
    if (counts_capture.device, counts_capture.inode) != (
        hybrid_capture.device,
        hybrid_capture.inode,
    ):
        raise ContractError("CountsScorer and HybridScorer do not share pinned skl source")
    resource_path = belief_capture.path.parent.parent / "resources/default_belief_probs.json"
    resource_capture = _capture_pinned_file(
        resource_path, PINNED_DEFAULT_PRIOR_SHA256, collect_content=False
    )
    installed_runtime = registry.get("installed_runtime")
    expected_installed_runtime = {
        "indra_version": PINNED_INDRA_VERSION,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "belief_init_sha256": belief_capture.sha256,
        "belief_skl_sha256": counts_capture.sha256,
        "default_prior_resource_sha256": resource_capture.sha256,
        "matches_current_release_implementation": True,
    }
    if installed_runtime != expected_installed_runtime:
        raise ContractError("registry installed_runtime does not match actual runtime files")
    if rows["indra_1.24.0_simple_default"].get("implementation_sha256") != belief_capture.sha256:
        raise ContractError("SimpleScorer row does not match actual belief implementation")
    if rows["indra_1.24.0_simple_default"].get("resource_sha256") != resource_capture.sha256:
        raise ContractError("SimpleScorer row does not match actual prior resource")
    for scorer_id in (
        "indra_1.24.0_counts_unfitted",
        "indra_1.24.0_hybrid_unfitted",
    ):
        if rows[scorer_id].get("implementation_sha256") != counts_capture.sha256:
            raise ContractError(f"{scorer_id}: row does not match actual skl implementation")
    bindings = {
        scorer_id: {
            "class": rows[scorer_id].get("class"),
            "commit": rows[scorer_id].get("commit"),
            "implementation_sha256": rows[scorer_id].get("implementation_sha256"),
            "registry_entry_sha256": hashlib.sha256(
                _canonical_bytes(rows[scorer_id])
            ).hexdigest(),
            "scorer_id": scorer_id,
        }
        for scorer_id in sorted(required_classes)
    }
    runtime_provenance = {
        "belief_init": belief_capture.descriptor(),
        "belief_skl": counts_capture.descriptor(),
        "default_prior_resource": resource_capture.descriptor(),
        "installed_runtime_crosscheck": "pass",
        "release_version_crosscheck": "pass",
    }
    return capture.descriptor(), rows, bindings, runtime_provenance


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
    with tempfile.TemporaryDirectory(prefix=".current-counts-readers-", dir=output_dir) as tmp:
        stage = Path(tmp)
        staged: dict[str, Path] = {}
        for name, rows in row_outputs.items():
            path = stage / name
            _write_jsonl(path, rows)
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
    reader_gold_path: Path,
    scorer_registry_path: Path,
    output_dir: Path,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Materialize the three reader class-path ledgers and exact provenance."""
    started = time.perf_counter()
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ContractError("n_jobs must be a non-zero integer")
    (
        pickle_path,
        paper_manifest_path,
        targets_path,
        comparison_manifest_path,
        reader_gold_path,
        scorer_registry_path,
    ) = [
        _absolute_path(path)
        for path in (
            pickle_path,
            paper_manifest_path,
            targets_path,
            comparison_manifest_path,
            reader_gold_path,
            scorer_registry_path,
        )
    ]
    output_dir = _absolute_path(output_dir)

    # No local adapter code, unpickle, graph build, or fit occurs before all
    # literal/source/input/runtime pins below have passed.
    dependency_captures = _preverify_dependency_sources()
    (
        comparison_manifest_input,
        target_input,
        reader_gold_input,
        targets,
        reader_gold,
    ) = _comparison_contract(
        comparison_manifest_path, targets_path, reader_gold_path
    )
    paper_manifest_input, declared_pickle = _paper_manifest_contract(
        paper_manifest_path
    )
    (
        registry_input,
        _,
        registry_bindings,
        runtime_scorer_provenance,
    ) = _registry_contract(scorer_registry_path)
    # Only after the installed INDRA files/resources match the hard-pinned
    # registry do we instantiate CountsScorer to verify the exact feature hash.
    feature_contracts = _verify_literal_contract()
    dependencies = _import_preverified_dependencies(dependency_captures)

    reader_positions = [
        index for index, row in enumerate(targets) if row["reader_eligible"]
    ]

    roots, pickle_input = _load_pinned_pickle(pickle_path, declared_pickle)
    statements = dependencies.hierarchy._select_targets(roots, targets)
    graph = build_refinements_graph(roots)
    graph_shape = (graph.number_of_nodes(), graph.number_of_edges())
    if graph_shape != (EXPECTED_GRAPH_NODES, EXPECTED_GRAPH_EDGES):
        raise ContractError(f"current refinement graph shape differs: {graph_shape}")
    if not nx.is_directed_acyclic_graph(graph):
        raise ContractError("current refinement graph contains a cycle")
    graph_fold_dependency_audit = _graph_fold_dependency_audit(
        [statements[index] for index in reader_positions], reader_gold, graph
    )
    extra_evidence = get_ev_for_stmts_from_supports(statements, graph)
    (
        reader_statements,
        reader_extra,
        observed_reader_positions,
        projection_audits,
    ) = _project_reader_inputs(statements, extra_evidence, targets, reader_gold)
    if observed_reader_positions != reader_positions:
        raise ContractError("reader projection positions drifted")

    projection_counts = {
        "canonical_direct_evidence_entries": sum(
            row["canonical_direct_evidence_count"] for row in projection_audits
        ),
        "canonical_inherited_evidence_entries": sum(
            row["canonical_inherited_evidence_count"] for row in projection_audits
        ),
        "projected_direct_evidence_entries": sum(
            row["projected_direct_evidence_count"] for row in projection_audits
        ),
        "projected_inherited_evidence_entries": sum(
            row["projected_inherited_evidence_count"] for row in projection_audits
        ),
        "statements_with_projected_inherited_evidence": sum(
            bool(row["projected_inherited_evidence_count"])
            for row in projection_audits
        ),
    }
    expected_projection_counts = {
        "canonical_direct_evidence_entries": EXPECTED_READER_CANONICAL_DIRECT_EVIDENCE,
        "canonical_inherited_evidence_entries": EXPECTED_READER_CANONICAL_INHERITED_EVIDENCE,
        "projected_direct_evidence_entries": EXPECTED_READER_PROJECTED_DIRECT_EVIDENCE,
        "projected_inherited_evidence_entries": EXPECTED_READER_PROJECTED_INHERITED_EVIDENCE,
        "statements_with_projected_inherited_evidence": (
            EXPECTED_READER_ROWS_WITH_PROJECTED_INHERITED_EVIDENCE
        ),
    }
    if projection_counts != expected_projection_counts:
        raise ContractError(
            f"reader projection census differs: {projection_counts} != "
            f"{expected_projection_counts}"
        )

    predictions, provenance, fit_rows, hybrid_audit = _cross_fit(
        reader_statements,
        reader_extra,
        reader_gold,
        estimator_factory=lambda fold_id: _new_rf(fold_id, n_jobs=n_jobs),
        projection_audits=projection_audits,
        enforce_frozen_contract=True,
    )
    if hybrid_audit["hybrid_simple_fallback_evidence_entries"] != 0:
        raise ContractError("local Hybrid unexpectedly activated SimpleScorer fallback")
    if not hybrid_audit["numerically_equivalent_within_tolerance"]:
        raise ContractError("local Hybrid is not numerically equivalent to full Counts")

    fold_metrics: list[dict[str, Any]] = []
    diagnostic_summaries: dict[str, Any] = {}
    for arm_id, rows in predictions.items():
        arm_fold, summary = dependencies.bayes._diagnostics(
            arm_id, rows, reader_gold
        )
        fold_metrics.extend(arm_fold)
        diagnostic_summaries[arm_id] = summary

    row_outputs = {
        PREDICTION_FILENAMES[arm_id]: rows for arm_id, rows in predictions.items()
    }
    row_outputs[FIT_FILENAME] = fit_rows
    row_outputs[PROVENANCE_FILENAME] = provenance
    row_outputs[FOLD_METRICS_FILENAME] = fold_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".current-counts-readers-descriptors-", dir=output_dir
    ) as tmp:
        stage = Path(tmp)
        outputs: dict[str, Any] = {}
        for name, rows in row_outputs.items():
            path = stage / name
            _write_jsonl(path, rows)
            outputs[name] = _descriptor(path, rows)
            outputs[name]["path"] = _display_path(output_dir / name)

    forest_module = _absolute_path(
        Path(inspect.getsourcefile(RandomForestClassifier) or "")
    )
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
                "a distinct Pareto point for zero-fallback local Hybrid",
                "historical random-fold or fitted-state parity with the paper notebook",
                "fold-isolated feature construction for the full Counts/Hybrid arms",
                "an inductive evaluation on statements absent from feature construction",
            ],
        },
        "arms": [
            {
                "arm_id": ARM_COUNTS_SOURCE,
                "class": "indra.belief.skl.CountsScorer",
                "configuration": SOURCE_CONFIGURATION.config_id,
                "official_class_local_fitted_state": True,
                "panel_id": PANEL_ID,
                "training": EVALUATION_DESIGN_LABEL,
            },
            {
                "arm_id": ARM_COUNTS_FULL,
                "class": "indra.belief.skl.CountsScorer",
                "configuration": FULL_CONFIGURATION.config_id,
                "official_class_local_fitted_state": True,
                "panel_id": PANEL_ID,
                "training": EVALUATION_DESIGN_LABEL,
            },
            {
                "arm_id": ARM_HYBRID_FULL,
                "class": "indra.belief.skl.HybridScorer",
                "counts_component": ARM_COUNTS_FULL,
                "official_class_local_fitted_state": True,
                "panel_id": PANEL_ID,
                "production_cogex_artifact": False,
                "simple_component": "indra.belief.SimpleScorer bundled defaults",
                "training": EVALUATION_DESIGN_LABEL,
                "comparison_role": "class-path equivalence audit",
                "pareto_point_policy": "alias_of_counts_component_not_a_distinct_model_point",
            },
        ],
        "panel": {
            "panel_id": PANEL_ID,
            "rows": EXPECTED_READER_ROWS,
            "ordered_statement_id_sha256": _ordered_id_sha(reader_gold),
            "gold": "paper reader-eligible released compatibility labels",
            "folds": 10,
            "eligibility": "at least one direct evidence item from a frozen paper reader",
            "input_sources": list(READER_SOURCES),
            "paper_reference": (
                "same reader row and direct-input substrate as paper OrigBelief-readers; "
                "different scorer class and locally cross-fitted state"
            ),
            "source_only_input_projection": "direct evidence restricted to five readers",
            "full_input_projection": (
                "direct evidence and current non-negated, deduplicated descendant evidence "
                "restricted to the same five readers"
            ),
        },
        "cross_fit_contract": {
            "design_label": EVALUATION_DESIGN_LABEL,
            "fold_source": "frozen paper_reader_eligible_released_gold fold_id",
            "fit_labels": "reader-panel statement correctness labels from nine folds",
            "test_labels_passed_to_fit": False,
            "outside_reader_statement_labels_read": False,
            "model_fit_is_label_isolated": True,
            "feature_construction_is_fold_isolated": False,
            "evaluation_is_inductive": False,
            "test_fold_predictions": (
                "one prediction from its complementary nine-fold label fit, conditional "
                "on the frozen global label-free refinement graph"
            ),
            "hyperparameter_selection": (
                "RF 2k/depth-13/seed-4 fixed from the external paper protocol; source-only "
                "and all-current-feature configurations fixed before reader metrics"
            ),
            "global_label_free_graph_input": True,
            "target_descendant_fold_dependencies": graph_fold_dependency_audit,
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
        "feature_contracts": feature_contracts,
        "current_class_semantics": {
            "direct_counts": "raw five-reader Evidence multiplicity, including negated entries",
            "more_specific_counts": (
                "current helper transitive descendants, then five-reader filtering; "
                "negated entries excluded and duplicate Evidence objects collapsed"
            ),
            "non_source_direct_features": (
                "PMID, promoter, and sentence-length features are computed after reader "
                "projection, so database evidence cannot affect them"
            ),
            "stored_statement_belief": (
                "cleared before featurization and never supplied to CountsScorer"
            ),
            "hybrid_formula": "1-(1-counts_probability)*(1-simple_fallback_probability)",
            "hybrid_audit": hybrid_audit,
            "hybrid_interpretation": (
                "all visible direct and inherited sources are in the fitted Counts source "
                "list; the Simple fallback receives zero evidence and the local Hybrid is "
                "performance-redundant with the full Counts arm up to float arithmetic"
            ),
        },
        "coverage": {
            "target_statements": EXPECTED_READER_ROWS,
            "predictions_per_arm": EXPECTED_READER_ROWS,
            "missing_predictions": 0,
            "invalid_predictions": 0,
            **projection_counts,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
        },
        "diagnostic_metrics": diagnostic_summaries,
        "inputs": {
            "canonical_object_graph_pickle": pickle_input,
            "paper_manifest": paper_manifest_input,
            "prediction_targets": target_input,
            "reader_gold_and_folds": reader_gold_input,
            "comparison_manifest": comparison_manifest_input,
            "scorer_registry": registry_input,
        },
        "implementation": {
            "adapter": {
                "bytes": _absolute_path(Path(__file__)).stat().st_size,
                "path": _display_path(_absolute_path(Path(__file__))),
                "sha256": _sha256(_absolute_path(Path(__file__))),
            },
            "shared_adapter_dependencies": dependencies.descriptors,
            "scorer_registry_bindings": registry_bindings,
            "runtime_scorer_crosschecks": runtime_scorer_provenance,
            "indra_statement_module": _module_descriptor(Statement),
            "sklearn_forest_module": {
                "bytes": forest_module.stat().st_size,
                "path": _display_path(forest_module),
                "sha256": _sha256(forest_module),
            },
            "bundled_simple_prior_resource": runtime_scorer_provenance[
                "default_prior_resource"
            ],
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
        "production_hybrid_boundary": {
            "artifact_accessed": False,
            "artifact_loaded": False,
            "artifact_uri": "s3://indra-belief/1.20.0/sk141_hybrid_rf_2kd13_cs.pkl",
            "local_hybrid_is_substitute": False,
            "production_point_required_separately": True,
        },
        "outputs": outputs,
        "runtime_observation": {
            "cost_scope": "local CPU execution; released data and curation costs excluded",
            "inference_usd": 0.0,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    _publish_bundle(output_dir, row_outputs, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    comparison = root / "data/results/indra_paper_comparison_gold_20260717"
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--pickle-path",
        default=root / "data/benchmark/indra_benchmark_corpus.pkl",
        type=Path,
    )
    parser.add_argument(
        "--paper-manifest-path",
        default=root / "data/benchmark/indra_paper_2023.manifest.json",
        type=Path,
    )
    parser.add_argument(
        "--targets-path",
        default=comparison / "paper_prediction_targets.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--comparison-manifest-path",
        default=comparison / "paper_comparison_gold_manifest.json",
        type=Path,
    )
    parser.add_argument(
        "--reader-gold-path",
        default=comparison / "paper_reader_eligible_released_gold.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--scorer-registry-path",
        default=root / "data/comparison/scorers.json",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=root / "data/results/current_indra_counts_hybrid_readers_20260719",
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
        reader_gold_path=args.reader_gold_path,
        scorer_registry_path=args.scorer_registry_path,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )
    print(
        json.dumps(
            {
                "coverage": manifest["coverage"],
                "hybrid_audit": manifest["current_class_semantics"]["hybrid_audit"],
                "outputs": manifest["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
