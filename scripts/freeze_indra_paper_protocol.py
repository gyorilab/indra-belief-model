#!/usr/bin/env python3
"""Freeze and audit the released INDRA assembly-paper evaluation protocol.

This harness is deliberately narrower than a paper reproduction.  It verifies
the three released curation artifacts, reconstructs the two Table EV2 eligible
sets without changing pickle row order, and materializes the declared
``random.seed(4)``/``random.shuffle``/``StratifiedKFold`` split protocol.

It does *not* fit a scorer or claim historical parity.  The paper did not
publish fold assignments, complete environment pins, or the random states used
by NumPy, Random Forest, SVC probability calibration, or MCMC.  Those limits
are recorded in every emitted protocol manifest.

The pickle payloads are only decoded after their complete bytes match the
released size and digests.  A restricted unpickler rejects all global/class
lookups; the released files contain only built-in list/dict/scalar values.

Example::

    PYTHONPATH=src .venv/bin/python scripts/freeze_indra_paper_protocol.py \
      --extended-pickle /path/to/extended_curation_dataset.pkl \
      --multireader-pickle /path/to/multireader_curation_dataset.pkl \
      --raw-curations /path/to/indra_assembly_curations.json \
      --output-dir data/results/indra_paper_protocol_freeze
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import pickle
import platform
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PAPER_REPOSITORY = "https://github.com/sorgerlab/indra_assembly_paper"
PAPER_COMMIT = "63abdf1274d2f5534ed822585775031712916c83"
PAPER_NOTEBOOK = "notebooks/Training Belief ML Models.ipynb"
PAPER_NOTEBOOK_SHA256 = (
    "3bd1a684fdc33c0b4963dd3e0c834c5420d90703112a91773f43415e1125ad26"
)
ZENODO_DOI = "10.5281/zenodo.7559353"

READER_SOURCES = ("reach", "sparser", "medscan", "rlimsp", "trips")
HISTORICAL_ALL_SOURCE_ORDER = (
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
PAPER_POSITIVE_TAGS = frozenset(("correct", "hypothesis", "act_vs_amt"))
PAPER_SPLIT_SEED = 4
PAPER_N_SPLITS = 10

PROTOCOL_FILENAME = "paper_protocol_manifest.json"
ELIGIBLE_FILENAME = "paper_eligible_statements.jsonl"
FOLDS_FILENAME = "paper_fold_assignments.jsonl"


class AuditError(ValueError):
    """Raised when a released-artifact or protocol invariant does not hold."""


@dataclass(frozen=True)
class FileExpectation:
    bytes: int
    md5: str
    sha256: str


@dataclass(frozen=True)
class DatasetExpectation:
    file: FileExpectation
    rows: int
    positive: int
    negative: int


@dataclass(frozen=True)
class RawCurationExpectation:
    file: FileExpectation
    rows: int
    unique_statement_hashes: int
    unique_statement_evidence_pairs: int
    unique_source_hashes: int
    unique_curators: int
    observed_pair_positive: int
    observed_pair_negative: int
    observed_statement_positive: int
    observed_statement_negative: int
    conflicting_pairs: int
    tag_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class FreezeExpectations:
    extended: DatasetExpectation
    reader_filtered_rows: int
    reader_filtered_positive: int
    reader_filtered_negative: int
    multireader: DatasetExpectation
    raw_curations: RawCurationExpectation


RELEASE_EXPECTATIONS = FreezeExpectations(
    extended=DatasetExpectation(
        file=FileExpectation(
            bytes=225_993,
            md5="78e5e31e56e8cc72746fb53872b11377",
            sha256=(
                "f235de451ddb8dc26423175ebabeac6422e299e948707e9b3bda0a4b20ab3f44"
            ),
        ),
        rows=1_689,
        positive=1_237,
        negative=452,
    ),
    reader_filtered_rows=1_676,
    reader_filtered_positive=1_236,
    reader_filtered_negative=440,
    multireader=DatasetExpectation(
        file=FileExpectation(
            bytes=167_798,
            md5="e9648f9bae9ab996c388f27366760c8f",
            sha256=(
                "d4a8c14efd0e1e2345ceba3fc101210eadfa25d9c1292558f150a02f4a3e90ac"
            ),
        ),
        rows=1_330,
        positive=1_018,
        negative=312,
    ),
    raw_curations=RawCurationExpectation(
        file=FileExpectation(
            bytes=1_687_078,
            md5="4a5b39066458e2112607bcb19e7a4d29",
            sha256=(
                "02ccca87fb4c8386ae49d420cb4c8257cf769bcaba5692a989bf295ba8d40da5"
            ),
        ),
        rows=6_022,
        unique_statement_hashes=1_800,
        unique_statement_evidence_pairs=5_709,
        unique_source_hashes=5_685,
        unique_curators=2,
        observed_pair_positive=3_728,
        observed_pair_negative=1_981,
        observed_statement_positive=1_298,
        observed_statement_negative=502,
        conflicting_pairs=23,
        tag_counts=(
            ("act_vs_amt", 220),
            ("agent_conditions", 7),
            ("correct", 3_618),
            ("entity_boundaries", 161),
            ("grounding", 449),
            ("hypothesis", 115),
            ("mod_site", 12),
            ("negative_result", 88),
            ("no_relation", 627),
            ("other", 215),
            ("polarity", 139),
            ("wrong_relation", 371),
        ),
    ),
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler for released list/dict/scalar payloads only."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(
            f"global/class lookup is forbidden: {module}.{name}"
        )

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("persistent pickle IDs are forbidden")


def _restricted_pickle_loads(payload: bytes) -> Any:
    return _RestrictedUnpickler(io.BytesIO(payload)).load()


def _md5(payload: bytes) -> str:
    try:
        digest = hashlib.md5(payload, usedforsecurity=False)
    except TypeError:  # pragma: no cover - compatibility with older Python
        digest = hashlib.md5(payload)
    return digest.hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_and_verify(path: Path, expected: FileExpectation) -> tuple[bytes, dict[str, Any]]:
    path = Path(path)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as err:
        raise AuditError(f"input does not exist: {path}") from err
    if not resolved.is_file():
        raise AuditError(f"input is not a regular file: {path} -> {resolved}")

    payload = resolved.read_bytes()
    observed = FileExpectation(
        bytes=len(payload), md5=_md5(payload), sha256=_sha256(payload)
    )
    if observed != expected:
        raise AuditError(
            f"released artifact identity mismatch for {path}: "
            f"expected={asdict(expected)}, observed={asdict(observed)}"
        )
    identity = {
        "path": str(path),
        "resolved_path": str(resolved),
        "is_symlink": path.is_symlink(),
        **asdict(observed),
        "verification": "pass",
    }
    return payload, identity


def paper_tag_label(tag: Any) -> int:
    """Return the paper's evidence-curation label for a raw curation tag."""
    return int(isinstance(tag, str) and tag in PAPER_POSITIVE_TAGS)


def aggregate_observed_gold(
    curations: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[int, int], int], dict[int, int], int]:
    """Aggregate only observed raw curations using the paper's mapping.

    An evidence pair is positive only when every curation on that exact pair is
    positive; therefore any negative curation wins a conflict.  A statement is
    positive when any observed evidence pair is positive.  This function does
    not infer that an incompletely reviewed statement is negative.
    """
    labels_by_pair: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(curations):
        if not isinstance(row, dict):
            raise AuditError(f"raw curation row {index} is not an object")
        try:
            pair = (int(row["pa_hash"]), int(row["source_hash"]))
            tag = row["tag"]
        except (KeyError, TypeError, ValueError) as err:
            raise AuditError(f"invalid raw curation row {index}") from err
        labels_by_pair[pair].append(paper_tag_label(tag))

    pair_labels = {pair: int(all(labels)) for pair, labels in labels_by_pair.items()}
    labels_by_statement: defaultdict[int, list[int]] = defaultdict(list)
    for (statement_hash, _), label in pair_labels.items():
        labels_by_statement[statement_hash].append(label)
    statement_labels = {
        statement_hash: int(any(labels))
        for statement_hash, labels in labels_by_statement.items()
    }
    conflicts = sum(
        int(any(labels) and not all(labels)) for labels in labels_by_pair.values()
    )
    return pair_labels, statement_labels, conflicts


def _require_binary_label(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise AuditError(f"{context} has non-binary `correct`: {value!r}")
    return value


def _require_hash(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise AuditError(f"{context} has invalid statement hash: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise AuditError(f"{context} has invalid statement hash: {value!r}") from err


def _validate_dataset(
    payload: Any,
    expected: DatasetExpectation,
    name: str,
) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise AuditError(f"{name} pickle root is not a list")
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    hashes: list[int] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise AuditError(f"{name} row {index} is not a dict")
        labels.append(_require_binary_label(row.get("correct"), f"{name} row {index}"))
        hashes.append(_require_hash(row.get("stmt_hash"), f"{name} row {index}"))
        rows.append(row)

    counts = Counter(labels)
    observed = (len(rows), counts[1], counts[0])
    wanted = (expected.rows, expected.positive, expected.negative)
    if observed != wanted:
        raise AuditError(
            f"{name} class counts mismatch: expected={wanted}, observed={observed}"
        )
    if len(set(hashes)) != len(hashes):
        duplicates = [key for key, count in Counter(hashes).items() if count > 1]
        raise AuditError(f"{name} contains duplicate statement hashes: {duplicates[:5]}")
    return rows


def _source_count(row: dict[str, Any], source: str, row_index: int) -> int:
    value = row.get(source, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditError(
            f"extended row {row_index} has invalid {source!r} count: {value!r}"
        )
    return value


def _historical_shuffle(items: Sequence[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    """Apply the notebook's global seed/shuffle while restoring caller RNG state."""
    shuffled = list(items)
    original_state = random.getstate()
    try:
        random.seed(PAPER_SPLIT_SEED)
        random.shuffle(shuffled)
    finally:
        random.setstate(original_state)
    return shuffled


def _sklearn_023_test_folds(labels: Sequence[int], n_splits: int) -> list[int]:
    """Literal shuffle=False port of sklearn 0.23.2 StratifiedKFold.

    Source: sklearn/model_selection/_split.py lines 643-694 in release 0.23.2.
    It preserves class order of first appearance and assigns each class to fold
    blocks according to round-robin allocation over sorted encoded labels.
    """
    if n_splits < 2:
        raise AuditError("n_splits must be at least 2")
    if not labels:
        raise AuditError("cannot split an empty eligible set")

    class_order: list[int] = []
    for label in labels:
        if label not in class_order:
            class_order.append(label)
    class_to_index = {label: index for index, label in enumerate(class_order)}
    encoded = [class_to_index[label] for label in labels]
    class_counts = Counter(encoded)
    if all(n_splits > count for count in class_counts.values()):
        raise AuditError(
            f"n_splits={n_splits} exceeds the member count of every class"
        )

    ordered = sorted(encoded)
    allocation: list[list[int]] = []
    for fold in range(n_splits):
        allocation.append(
            [
                sum(value == class_index for value in ordered[fold::n_splits])
                for class_index in range(len(class_order))
            ]
        )

    test_folds = [-1] * len(labels)
    for class_index in range(len(class_order)):
        folds_for_class: list[int] = []
        for fold in range(n_splits):
            folds_for_class.extend([fold] * allocation[fold][class_index])
        positions = [
            position for position, encoded_label in enumerate(encoded)
            if encoded_label == class_index
        ]
        if len(positions) != len(folds_for_class):  # defensive invariant
            raise AuditError("internal stratification allocation mismatch")
        for position, fold in zip(positions, folds_for_class, strict=True):
            test_folds[position] = fold

    if any(fold < 0 for fold in test_folds):  # defensive invariant
        raise AuditError("internal stratification left rows unassigned")
    return test_folds


def _crosscheck_installed_sklearn(labels: Sequence[int], folds: Sequence[int]) -> dict[str, Any]:
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        return {"status": "not_installed", "version": None}

    version = _package_versions(("scikit-learn",))["scikit-learn"]
    observed = [-1] * len(labels)
    splitter = StratifiedKFold(n_splits=PAPER_N_SPLITS, shuffle=False)
    placeholder = [None] * len(labels)
    for fold, (_, test_indices) in enumerate(splitter.split(placeholder, labels)):
        for index in test_indices:
            observed[int(index)] = fold
    if observed != list(folds):
        raise AuditError(
            "local sklearn-0.23.2 StratifiedKFold port disagrees with the "
            f"installed sklearn {version}"
        )
    return {"status": "pass", "version": version}


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _fold_rows(
    eligible_rows: Sequence[dict[str, Any]],
    eligible_set: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pre_shuffle = [
        (row["source_row_index"], row["stmt_hash"], row["correct"])
        for row in eligible_rows
    ]
    shuffled = _historical_shuffle(pre_shuffle)
    labels = [label for _, _, label in shuffled]
    folds = _sklearn_023_test_folds(labels, PAPER_N_SPLITS)
    sklearn_check = _crosscheck_installed_sklearn(labels, folds)

    by_source_row = {
        source_row_index: {
            "shuffle_position": shuffle_position,
            "fold": folds[shuffle_position],
        }
        for shuffle_position, (source_row_index, _, _) in enumerate(shuffled)
    }
    output_rows = []
    for eligible_position, row in enumerate(eligible_rows):
        assignment = by_source_row[row["source_row_index"]]
        output_rows.append(
            {
                "eligible_set": eligible_set,
                "source_row_index": row["source_row_index"],
                "eligible_position": eligible_position,
                "shuffle_position": assignment["shuffle_position"],
                "test_fold": assignment["fold"],
                "stmt_hash": row["stmt_hash"],
                "correct": row["correct"],
            }
        )

    fold_counts: dict[str, dict[str, int]] = {}
    for fold in range(PAPER_N_SPLITS):
        fold_labels = [
            row["correct"] for row in output_rows if row["test_fold"] == fold
        ]
        counts = Counter(fold_labels)
        fold_counts[str(fold)] = {
            "rows": len(fold_labels),
            "positive": counts[1],
            "negative": counts[0],
        }
    summary = {
        "rows": len(output_rows),
        "positive": sum(row["correct"] for row in output_rows),
        "negative": sum(not row["correct"] for row in output_rows),
        "fold_counts": fold_counts,
        "installed_sklearn_crosscheck": sklearn_check,
    }
    return output_rows, summary


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _preflight_outputs(output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "protocol": output_dir / PROTOCOL_FILENAME,
        "eligible": output_dir / ELIGIBLE_FILENAME,
        "folds": output_dir / FOLDS_FILENAME,
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing paper-protocol artifacts: "
            + ", ".join(existing)
        )
    return paths


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _artifact_identity(path: Path, payload: bytes, rows: int | None = None) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "path": str(path),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }
    if rows is not None:
        identity["rows"] = rows
    return identity


def _runtime_identity() -> dict[str, Any]:
    packages = _package_versions(
        ("indra", "numpy", "pandas", "scipy", "scikit-learn", "emcee", "networkx")
    )
    return {
        "python": {
            "version": platform.python_version(),
            "full_version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "packages": packages,
        "environment": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
        "historical_runtime_known": {
            "python": "3.7.4",
            "scikit_learn_family": "0.23.x",
            "exact_environment_published": False,
        },
    }


def freeze_protocol(
    extended_pickle: Path,
    multireader_pickle: Path,
    raw_curations: Path,
    output_dir: Path,
    *,
    expectations: FreezeExpectations = RELEASE_EXPECTATIONS,
) -> dict[str, Any]:
    """Verify released inputs and write protocol, eligibility, and fold manifests.

    Existing target artifacts are never overwritten.  The returned dictionary
    is identical to the protocol manifest written to disk.
    """
    output_paths = _preflight_outputs(output_dir)

    extended_bytes, extended_identity = _read_and_verify(
        extended_pickle, expectations.extended.file
    )
    multireader_bytes, multireader_identity = _read_and_verify(
        multireader_pickle, expectations.multireader.file
    )
    raw_bytes, raw_identity = _read_and_verify(
        raw_curations, expectations.raw_curations.file
    )

    extended_rows = _validate_dataset(
        _restricted_pickle_loads(extended_bytes), expectations.extended, "extended"
    )
    multireader_rows = _validate_dataset(
        _restricted_pickle_loads(multireader_bytes),
        expectations.multireader,
        "multireader",
    )
    try:
        raw_rows = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise AuditError("raw curation artifact is not valid UTF-8 JSON") from err
    if not isinstance(raw_rows, list):
        raise AuditError("raw curation JSON root is not a list")

    pair_labels, observed_statement_labels, conflicts = aggregate_observed_gold(raw_rows)
    raw_stats = {
        "rows": len(raw_rows),
        "unique_statement_hashes": len({int(row["pa_hash"]) for row in raw_rows}),
        "unique_statement_evidence_pairs": len(pair_labels),
        "unique_source_hashes": len({int(row["source_hash"]) for row in raw_rows}),
        "unique_curators": len({str(row["curator"]) for row in raw_rows}),
        "observed_pair_positive": sum(pair_labels.values()),
        "observed_pair_negative": len(pair_labels) - sum(pair_labels.values()),
        "observed_statement_positive": sum(observed_statement_labels.values()),
        "observed_statement_negative": (
            len(observed_statement_labels) - sum(observed_statement_labels.values())
        ),
        "conflicting_pairs": conflicts,
        "tag_counts": dict(sorted(Counter(str(row["tag"]) for row in raw_rows).items())),
    }
    expected_raw_stats = {
        key: value
        for key, value in asdict(expectations.raw_curations).items()
        if key != "file"
    }
    expected_raw_stats["tag_counts"] = dict(expectations.raw_curations.tag_counts)
    if raw_stats != expected_raw_stats:
        raise AuditError(
            "raw curation counts mismatch: "
            f"expected={expected_raw_stats}, observed={raw_stats}"
        )

    raw_statement_hashes = set(observed_statement_labels)
    multireader_by_hash = {
        _require_hash(row["stmt_hash"], "multireader row"): row
        for row in multireader_rows
    }
    extended_hashes = {
        _require_hash(row["stmt_hash"], "extended row") for row in extended_rows
    }
    if not set(multireader_by_hash).issubset(extended_hashes):
        raise AuditError("multireader statements are not a subset of extended statements")
    if not extended_hashes.issubset(raw_statement_hashes):
        raise AuditError("extended statements are not a subset of raw-curation statements")
    extended_by_hash = {int(row["stmt_hash"]): row for row in extended_rows}
    label_differences = [
        statement_hash
        for statement_hash, row in multireader_by_hash.items()
        if int(row["correct"]) != int(extended_by_hash[statement_hash]["correct"])
    ]
    if label_differences:
        raise AuditError(
            "multireader/extended labels differ for shared statements: "
            f"{label_differences[:5]}"
        )

    eligible_rows: list[dict[str, Any]] = []
    for source_row_index, source_row in enumerate(extended_rows):
        statement_hash = _require_hash(
            source_row["stmt_hash"], f"extended row {source_row_index}"
        )
        source_counts = [
            _source_count(source_row, source, source_row_index)
            for source in HISTORICAL_ALL_SOURCE_ORDER
        ]
        reader_counts = [
            _source_count(source_row, source, source_row_index)
            for source in READER_SOURCES
        ]
        eligible_rows.append(
            {
                "source_row_index": source_row_index,
                "stmt_hash": str(statement_hash),
                "correct": int(source_row["correct"]),
                "stmt_type": source_row.get("stmt_type"),
                "original_stmt_num": source_row.get("stmt_num"),
                "reader_eligible": any(reader_counts),
                "in_multireader_dataset": statement_hash in multireader_by_hash,
                "reader_source_counts": reader_counts,
                "historical_all_source_counts": source_counts,
            }
        )

    reader_rows = [row for row in eligible_rows if row["reader_eligible"]]
    observed_reader_counts = (
        len(reader_rows),
        sum(row["correct"] for row in reader_rows),
        sum(not row["correct"] for row in reader_rows),
    )
    expected_reader_counts = (
        expectations.reader_filtered_rows,
        expectations.reader_filtered_positive,
        expectations.reader_filtered_negative,
    )
    if observed_reader_counts != expected_reader_counts:
        raise AuditError(
            "reader-filtered class counts mismatch: "
            f"expected={expected_reader_counts}, observed={observed_reader_counts}"
        )

    all_source_fold_rows, all_source_fold_summary = _fold_rows(
        eligible_rows, "extended_all_sources"
    )
    reader_fold_rows, reader_fold_summary = _fold_rows(reader_rows, "reader_only")
    fold_rows = all_source_fold_rows + reader_fold_rows

    eligible_payload = _jsonl_bytes(eligible_rows)
    folds_payload = _jsonl_bytes(fold_rows)
    _write_new(output_paths["eligible"], eligible_payload)
    _write_new(output_paths["folds"], folds_payload)

    harness_path = Path(__file__).resolve()
    harness_bytes = harness_path.read_bytes()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "indra_assembly_paper_protocol_freeze",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim_scope": {
            "status": "protocol_reconstruction_not_historical_parity",
            "fits_or_predictions_generated": False,
            "paper_metric_reproduced": False,
            "reason": (
                "The release does not identify exact fold assignments, a complete "
                "runtime lock, estimator random states, MCMC random state, or one "
                "globally consistent interactive notebook execution state."
            ),
        },
        "paper_identity": {
            "repository": PAPER_REPOSITORY,
            "commit": PAPER_COMMIT,
            "notebook": PAPER_NOTEBOOK,
            "notebook_sha256": PAPER_NOTEBOOK_SHA256,
            "zenodo_doi": ZENODO_DOI,
        },
        "harness": {
            "path": str(harness_path),
            "sha256": _sha256(harness_bytes),
        },
        "inputs": {
            "extended_pickle": extended_identity,
            "multireader_pickle": multireader_identity,
            "raw_curations": raw_identity,
        },
        "gold_protocol": {
            "positive_tags": sorted(PAPER_POSITIVE_TAGS),
            "all_other_tags": "negative",
            "same_pair_conflict": "negative_wins; pair positive iff all curations positive",
            "observed_statement_positive": "any observed positive evidence pair",
            "paper_extended_statement_rule": "statement correct iff any curated evidence is correct",
            "paper_extended_incomplete_policy": (
                "allow_incomplete=True; an all-reviewed-negative subset was accepted "
                "as statement-negative in the released processed dataset"
            ),
            "audit_counts": raw_stats,
            "warning": (
                "Observed raw aggregation alone does not prove a negative statement "
                "when other evidence is unreviewed."
            ),
        },
        "eligible_sets": {
            "extended_all_sources": {
                "source": "extended_curation_dataset.pkl in released pickle row order",
                "rows": len(eligible_rows),
                "positive": sum(row["correct"] for row in eligible_rows),
                "negative": sum(not row["correct"] for row in eligible_rows),
            },
            "reader_only": {
                "source": "runtime filter of the extended dataset",
                "reader_sources": list(READER_SOURCES),
                "eligibility": "at least one selected reader count is nonzero",
                "rows": len(reader_rows),
                "positive": sum(row["correct"] for row in reader_rows),
                "negative": sum(not row["correct"] for row in reader_rows),
                "not_the_multireader_pickle": True,
            },
            "multireader_released": {
                "rows": len(multireader_rows),
                "positive": sum(int(row["correct"]) for row in multireader_rows),
                "negative": sum(not int(row["correct"]) for row in multireader_rows),
                "subset_of_extended": True,
                "shared_labels_equal": True,
            },
        },
        "source_columns": {
            "reader_order": list(READER_SOURCES),
            "historical_all_source_order": list(HISTORICAL_ALL_SOURCE_ORDER),
            "historical_order_provenance": (
                "Stored output of training-notebook cell 13; frozen explicitly "
                "because the notebook constructed it with list(set(...))."
            ),
        },
        "split_protocol": {
            "pre_split_shuffle": {
                "seed": PAPER_SPLIT_SEED,
                "implementation": (
                    "Python random.seed(4), then random.shuffle over zipped "
                    "statement/label rows; global RNG state restored by this harness"
                ),
                "pickle_row_order_preserved_before_shuffle": True,
            },
            "cross_validation": {
                "class": "sklearn.model_selection.StratifiedKFold",
                "n_splits": PAPER_N_SPLITS,
                "shuffle": False,
                "local_implementation": (
                    "literal port of sklearn 0.23.2 _make_test_folds lines 643-694"
                ),
                "source": (
                    "https://github.com/scikit-learn/scikit-learn/blob/0.23.2/"
                    "sklearn/model_selection/_split.py#L643-L694"
                ),
            },
            "fold_summaries": {
                "extended_all_sources": all_source_fold_summary,
                "reader_only": reader_fold_summary,
            },
            "published_fold_manifest_available": False,
            "reconstruction_warning": (
                "These are deterministic assignments under the declared algorithm, "
                "not proof that they are byte-identical to unpublished historical folds."
            ),
        },
        "known_rng_nondeterminism": [
            "Only the Python pre-split shuffle seed was set in the notebook.",
            "NumPy global state was not seeded.",
            "RandomForestClassifier random_state was None.",
            "SVC probability calibration random_state was not set.",
            "MCMC initialization and sampling random state were not frozen.",
            "The all-source list originated from a Python set; this harness freezes the printed order.",
            "The notebook was executed interactively and nonlinearly.",
        ],
        "runtime": _runtime_identity(),
        "outputs": {
            "eligible_statements": _artifact_identity(
                output_paths["eligible"], eligible_payload, len(eligible_rows)
            ),
            "fold_assignments": _artifact_identity(
                output_paths["folds"], folds_payload, len(fold_rows)
            ),
        },
        "validation": {
            "released_digests": "pass",
            "restricted_pickle_decode": "pass",
            "expected_counts": "pass",
            "extended_statement_hashes_unique": True,
            "multireader_subset_and_labels": "pass",
            "raw_curation_mapping": "pass",
            "artifacts_overwritten": False,
        },
    }
    protocol_payload = _json_bytes(manifest)
    _write_new(output_paths["protocol"], protocol_payload)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended-pickle", type=Path, required=True)
    parser.add_argument("--multireader-pickle", type=Path, required=True)
    parser.add_argument("--raw-curations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        manifest = freeze_protocol(
            args.extended_pickle,
            args.multireader_pickle,
            args.raw_curations,
            args.output_dir,
        )
    except (AuditError, FileExistsError, OSError, pickle.UnpicklingError) as err:
        print(f"paper protocol freeze failed: {err}", file=sys.stderr)
        return 2

    eligible = manifest["eligible_sets"]
    print(
        "verified released paper curations: "
        f"extended={eligible['extended_all_sources']['rows']} "
        f"reader_only={eligible['reader_only']['rows']} "
        f"multireader={eligible['multireader_released']['rows']}"
    )
    print(f"wrote protocol freeze to {args.output_dir}")
    print("status: protocol reconstruction, not historical parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
