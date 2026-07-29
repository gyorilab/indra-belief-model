#!/usr/bin/env python3
"""Materialize the frozen CoGEx substrate for representative gold hashes.

The input ``nodes_Evidence.tsv.gz`` is the Neo4j bulk-import export produced by
``indra_cogex.sources.indra_db.EvidenceProcessor``.  This program scans it once,
selects every row for a gold statement hash, and deliberately does *not*
deduplicate evidence rows.  The resulting bundle separates:

* statements: one normalized statement payload per matches hash;
* evidence_entries: every matching CoGEx row, in source order;
* pair_index: one row per unique (matches hash, source hash), linking all raw
  entries and all payload variants;
* optional ancestor_closure and hybrid_counts artifacts reconstructed from the
  synchronized ``refinements.tsv.gz`` and ``source_counts.pkl`` dumps.

All identifier hashes in JSON are strings, avoiding loss of 64-bit precision in
JavaScript consumers.  The exact decoded Evidence cell is retained with its
SHA-256; the parsed payload is also retained after hash fields are normalized
to strings.

Writes are resumable and atomic.  Matching entries are first appended to an
uncompressed checkpointed spool.  Final files have a unique bundle id and the
plain ``manifest.json`` pointer is replaced only after every file is complete,
so an older manifest can never point at a partially replaced bundle.

Pickles are code-executing inputs.  Only pass synchronized, trusted INDRA dump
files to ``--source-counts`` and ``--belief-scores``.

Example (the CLI pins the known 2025-09-16 Evidence dump by default)::

    python scripts/materialize_frozen_representative_substrate.py \
      --gold data/benchmark/representative_indra_curations_400.jsonl \
      --nodes-evidence /path/to/nodes_Evidence.tsv.gz \
      --output-dir /path/to/new/substrate \
      --source-counts /path/to/source_counts.pkl \
      --refinements /path/to/refinements.tsv.gz \
      --belief-scores /path/to/belief_scores.pkl
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import re
import shutil
import sqlite3
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator, TextIO


SCHEMA_VERSION = 1
FROZEN_NODES_SHA256 = (
    "29cee1b4a9367c3a9aa7c9e34066fd679381ebde324a770dae1e4944cef33ff5"
)
FROZEN_NODES_ROWS = 44_944_056
FROZEN_TARGET_HASHES = 403
SIGNED_INTEGER_RE = re.compile(r"^-?\d+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WORK_DIR_NAME = ".frozen-substrate.work"
CHECKPOINT_NAME = "checkpoint.json"
EVIDENCE_SPOOL_NAME = "evidence_entries.jsonl"
MANIFEST_NAME = "manifest.json"
SCAN_SUMMARY_NAME = "scan_summary.json"


class ScanInterrupted(RuntimeError):
    """Synthetic/test interruption after a durable scan checkpoint."""


class CompressionUnavailable(RuntimeError):
    """Raised when zstd was explicitly requested but is not installed."""


@dataclass(frozen=True)
class MaterializationResult:
    status: str
    manifest_path: Path
    manifest: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_line(value: Any) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quick_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"input is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _full_file_identity(path: Path) -> dict[str, Any]:
    identity = _quick_file_identity(path)
    identity["sha256"] = _sha256_file(path)
    return identity


def _atomic_write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        fh.write(_json_line(value))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@contextmanager
def _output_lock(output_dir: Path) -> Iterator[None]:
    """Serialize writers without treating a crashed lock file as permanent."""
    lock_path = output_dir / ".materialize.lock"
    with lock_path.open("a+b") as lock_fh:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another materializer is already writing {output_dir}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _exact_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a signed integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and SIGNED_INTEGER_RE.fullmatch(value):
        return int(value)
    raise ValueError(f"{field} must be a canonical signed integer; got {value!r}")


def _hash_sort_key(value: str) -> int:
    return _exact_int(value, field="hash")


def _is_hash_field(key: str) -> bool:
    return key in {"hash", "pa_hash"} or key.endswith("_hash")


def _stringify_hash_value(value: Any, *, field: str) -> str:
    # Non-decimal hashes (for example a reader's content hash) are already
    # precision-safe strings.  Signed INDRA identifiers are canonicalized.
    if isinstance(value, str) and not SIGNED_INTEGER_RE.fullmatch(value):
        return value
    return str(_exact_int(value, field=field))


def _stringify_hashes(value: Any, *, parent_key: str | None = None) -> Any:
    """Deep-copy JSON while representing identifier hash values as strings."""
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object contains a non-string key")
            if _is_hash_field(key) and child is not None:
                normalized[key] = _stringify_hash_value(child, field=key)
            elif key.endswith("_hashes") and child is not None:
                if not isinstance(child, list):
                    raise ValueError(f"{key} must be a list")
                normalized[key] = [
                    _stringify_hash_value(item, field=key) for item in child
                ]
            else:
                normalized[key] = _stringify_hashes(child, parent_key=key)
        return normalized
    if isinstance(value, list):
        return [_stringify_hashes(item, parent_key=parent_key) for item in value]
    return copy.deepcopy(value)


def _assert_hashes_are_strings(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_hash_field(key) and child is not None and not isinstance(child, str):
                raise AssertionError(f"output hash field {key} is not a string")
            if key.endswith("_hashes") and child is not None:
                if not isinstance(child, list) or not all(
                    isinstance(item, str) for item in child
                ):
                    raise AssertionError(f"output hash list {key} is not string-valued")
            _assert_hashes_are_strings(child)
    elif isinstance(value, list):
        for child in value:
            _assert_hashes_are_strings(child)


def _header_base(header: str) -> str:
    if header == ":LABEL":
        return "label"
    return header.split(":", 1)[0].strip().lower()


def _resolve_evidence_columns(header: list[str]) -> dict[str, int]:
    columns = {_header_base(name): idx for idx, name in enumerate(header)}
    missing = {"evidence", "retracted", "stmt_hash", "source_api"} - columns.keys()
    if missing:
        raise ValueError(
            "nodes_Evidence header is missing "
            f"{', '.join(sorted(missing))}; got {header!r}"
        )
    return columns


def _parse_neo4j_boolean(value: str, *, field: str) -> bool:
    """Parse the exact lowercase booleans emitted for Neo4j bulk import."""
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{field} must be the raw cell 'true' or 'false'; got {value!r}")


def _zstd_available() -> bool:
    try:
        import zstandard  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_compression(
    requested: str,
    *,
    zstd_available: bool | None = None,
) -> tuple[str, str | None]:
    if requested not in {"auto", "zstd", "gzip", "none"}:
        raise ValueError(f"unknown compression: {requested}")
    available = _zstd_available() if zstd_available is None else zstd_available
    if requested == "zstd":
        if not available:
            raise CompressionUnavailable(
                "zstd was requested but the optional 'zstandard' package is not "
                "installed; install zstandard or pass --compression gzip"
            )
        return "zstd", None
    if requested == "auto":
        if available:
            return "zstd", None
        return (
            "gzip",
            "optional 'zstandard' package is not installed; auto fell back to gzip",
        )
    return requested, None


def _compression_suffix(compression: str) -> str:
    return {"zstd": ".jsonl.zst", "gzip": ".jsonl.gz", "none": ".jsonl"}[
        compression
    ]


def _compress_file(source: Path, destination: Path, compression: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as raw_out:
        if compression == "none":
            shutil.copyfileobj(src, raw_out, length=1024 * 1024)
        elif compression == "gzip":
            with gzip.GzipFile(
                filename="", fileobj=raw_out, mode="wb", mtime=0
            ) as compressed:
                shutil.copyfileobj(src, compressed, length=1024 * 1024)
        elif compression == "zstd":
            try:
                import zstandard
            except ImportError as exc:  # pragma: no cover - guarded by resolver
                raise CompressionUnavailable("zstandard disappeared during the run") from exc
            compressor = zstandard.ZstdCompressor(level=10)
            with compressor.stream_writer(raw_out, closefd=False) as compressed:
                shutil.copyfileobj(src, compressed, length=1024 * 1024)
        else:  # pragma: no cover - internal invariant
            raise AssertionError(compression)
        raw_out.flush()
        os.fsync(raw_out.fileno())


class _HashingReader:
    """Small sequential file proxy that hashes compressed bytes as gzip reads."""

    def __init__(self, raw: BinaryIO):
        self.raw = raw
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        value = self.raw.read(size)
        self.digest.update(value)
        return value

    def readinto(self, buffer: bytearray) -> int:
        count = self.raw.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count

    def tell(self) -> int:
        return self.raw.tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        # gzip is consumed sequentially here.  Refuse a rewind because hashing
        # the same compressed bytes twice would invalidate the source digest.
        current = self.raw.tell()
        if whence == io.SEEK_CUR and offset == 0:
            return current
        if whence == io.SEEK_SET and offset == current:
            return current
        raise io.UnsupportedOperation("hashing gzip input is sequential")

    def readable(self) -> bool:
        return True


@contextmanager
def _open_hashed_gzip_text(path: Path) -> Iterator[tuple[TextIO, _HashingReader]]:
    raw = path.open("rb")
    hashing = _HashingReader(raw)
    gz = gzip.GzipFile(fileobj=hashing, mode="rb")
    text = io.TextIOWrapper(gz, encoding="utf-8", errors="strict", newline="")
    try:
        yield text, hashing
    finally:
        text.close()
        raw.close()


def _recompute_source_hash(evidence: dict[str, Any]) -> int:
    try:
        from indra.statements import Evidence
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ValueError(
            "evidence payload has no source_hash and INDRA is unavailable to recompute it"
        ) from exc
    try:
        return int(Evidence._from_json(evidence).get_source_hash())
    except Exception as exc:  # INDRA exposes reader-specific validation errors
        raise ValueError("could not recompute missing evidence source_hash") from exc


def _load_gold(
    path: Path,
    *,
    expected_target_count: int | None,
) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]], dict[str, Any]]:
    statements: dict[str, dict[str, Any]] = {}
    gold_sources: dict[str, set[str]] = defaultdict(set)
    pairs: set[tuple[str, str]] = set()
    rows = 0
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid gold JSON at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"gold line {line_no} is not an object")
            matches_hash = str(
                _exact_int(row.get("matches_hash"), field=f"gold line {line_no} matches_hash")
            )
            source_hash = str(
                _exact_int(row.get("source_hash"), field=f"gold line {line_no} source_hash")
            )
            pair = (matches_hash, source_hash)
            if pair in pairs:
                raise ValueError(f"gold contains duplicate exact pair {pair}")
            pairs.add(pair)
            gold_sources[matches_hash].add(source_hash)

            statement = row.get("statement")
            if not isinstance(statement, dict):
                raise ValueError(f"gold line {line_no} has no statement object")
            normalized = _stringify_hashes(statement)
            embedded_hash = normalized.get("matches_hash")
            if embedded_hash is None:
                normalized["matches_hash"] = matches_hash
            elif embedded_hash != matches_hash:
                raise ValueError(
                    f"gold line {line_no} statement hash {embedded_hash} does not "
                    f"match row hash {matches_hash}"
                )
            prior = statements.get(matches_hash)
            if prior is not None and _canonical_json_bytes(prior) != _canonical_json_bytes(
                normalized
            ):
                raise ValueError(
                    f"gold has inconsistent statement payloads for {matches_hash}"
                )
            statements[matches_hash] = normalized

    if not rows:
        raise ValueError("gold input contains no rows")
    if expected_target_count is not None and len(statements) != expected_target_count:
        raise ValueError(
            f"gold has {len(statements)} unique matches hashes; expected "
            f"{expected_target_count}"
        )

    records: dict[str, dict[str, Any]] = {}
    for matches_hash in sorted(statements, key=_hash_sort_key):
        statement = statements[matches_hash]
        record = {
            "matches_hash": matches_hash,
            "statement_payload_sha256": _sha256_bytes(_canonical_json_bytes(statement)),
            "gold_source_hashes": sorted(
                gold_sources[matches_hash], key=_hash_sort_key
            ),
            "statement": statement,
        }
        _assert_hashes_are_strings(record)
        records[matches_hash] = record

    identity = _full_file_identity(path)
    identity.update(
        {
            "format": "jsonl",
            "rows": rows,
            "unique_matches_hashes": len(records),
            "unique_exact_pairs": len(pairs),
        }
    )
    return records, pairs, identity


def _checkpoint(
    checkpoint_path: Path,
    spool: BinaryIO,
    state: dict[str, Any],
) -> None:
    spool.flush()
    os.fsync(spool.fileno())
    state["spool_bytes"] = spool.tell()
    state["updated_at"] = _utc_now()
    _atomic_write_json(checkpoint_path, state)


def _initial_checkpoint(scan_fingerprint: str, quick_nodes: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "scan_fingerprint": scan_fingerprint,
        "bundle_id": f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}",
        "nodes_evidence": quick_nodes,
        "rows_scanned": 0,
        "entries_written": 0,
        "spool_bytes": 0,
        "scan_complete": False,
        "created_at": now.isoformat().replace("+00:00", "Z"),
    }


def _load_or_create_checkpoint(
    work_dir: Path,
    *,
    scan_fingerprint: str,
    quick_nodes: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    if force and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_dir / CHECKPOINT_NAME
    spool_path = work_dir / EVIDENCE_SPOOL_NAME
    if checkpoint_path.exists():
        try:
            state = json.loads(checkpoint_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid materialization checkpoint; rerun with --force: {exc}"
            ) from exc
        if (
            state.get("schema_version") != SCHEMA_VERSION
            or state.get("scan_fingerprint") != scan_fingerprint
            or state.get("nodes_evidence") != quick_nodes
        ):
            raise ValueError(
                f"{work_dir} belongs to different inputs/options; rerun with --force"
            )
        if not spool_path.exists():
            raise ValueError(
                f"checkpoint exists without {spool_path.name}; rerun with --force"
            )
        size = spool_path.stat().st_size
        checkpoint_size = state.get("spool_bytes")
        if not isinstance(checkpoint_size, int) or checkpoint_size < 0:
            raise ValueError("checkpoint has an invalid spool byte offset; use --force")
        if size < checkpoint_size:
            raise ValueError("evidence spool is shorter than its checkpoint; use --force")
        with spool_path.open("r+b") as spool:
            spool.truncate(checkpoint_size)
        return state

    state = _initial_checkpoint(scan_fingerprint, quick_nodes)
    spool_path.touch(exist_ok=False)
    with spool_path.open("r+b") as spool:
        _checkpoint(checkpoint_path, spool, state)
    return state


def _scan_evidence_nodes(
    nodes_path: Path,
    target_hashes: set[str],
    work_dir: Path,
    state: dict[str, Any],
    *,
    expected_nodes_rows: int | None,
    expected_nodes_sha256: str | None,
    checkpoint_every: int,
    stop_after_scan_rows: int | None,
) -> dict[str, Any]:
    if state.get("scan_complete"):
        return state
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    resume_rows = _exact_int(state.get("rows_scanned"), field="checkpoint rows_scanned")
    entries_written = _exact_int(
        state.get("entries_written"), field="checkpoint entries_written"
    )
    checkpoint_path = work_dir / CHECKPOINT_NAME
    spool_path = work_dir / EVIDENCE_SPOOL_NAME

    csv.field_size_limit(1 << 28)
    with spool_path.open("r+b") as spool:
        spool.seek(_exact_int(state["spool_bytes"], field="checkpoint spool_bytes"))
        with _open_hashed_gzip_text(nodes_path) as (text, hashing):
            reader = csv.reader(text, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("nodes_Evidence input is empty") from exc
            columns = _resolve_evidence_columns(header)
            prior_header = state.get("header")
            if prior_header is not None and prior_header != header:
                raise ValueError("nodes_Evidence header changed since checkpoint")
            state["header"] = header

            rows_scanned = 0
            for rows_scanned, row in enumerate(reader, start=1):
                if rows_scanned <= resume_rows:
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"nodes_Evidence row {rows_scanned} has {len(row)} columns; "
                        f"expected {len(header)}"
                    )
                matches_hash = str(
                    _exact_int(
                        row[columns["stmt_hash"]],
                        field=f"nodes_Evidence row {rows_scanned} stmt_hash",
                    )
                )
                if matches_hash in target_hashes:
                    raw_payload = row[columns["evidence"]]
                    try:
                        evidence = json.loads(raw_payload)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid evidence JSON at nodes_Evidence row {rows_scanned}: {exc}"
                        ) from exc
                    if not isinstance(evidence, dict):
                        raise ValueError(
                            f"evidence at nodes_Evidence row {rows_scanned} is not an object"
                        )
                    if evidence.get("source_hash") is None:
                        source_hash_int = _recompute_source_hash(evidence)
                    else:
                        source_hash_int = _exact_int(
                            evidence["source_hash"],
                            field=f"nodes_Evidence row {rows_scanned} source_hash",
                        )
                    source_hash = str(source_hash_int)
                    tsv_source_api = row[columns["source_api"]].strip()
                    payload_source_api = evidence.get("source_api")
                    if payload_source_api is not None and not isinstance(
                        payload_source_api, str
                    ):
                        raise ValueError(
                            f"nodes_Evidence row {rows_scanned} source_api is not a string"
                        )
                    source_api = payload_source_api or tsv_source_api
                    if not source_api:
                        raise ValueError(
                            f"nodes_Evidence row {rows_scanned} has no source_api"
                        )
                    if tsv_source_api and payload_source_api and (
                        tsv_source_api != payload_source_api
                    ):
                        raise ValueError(
                            f"nodes_Evidence row {rows_scanned} source_api mismatch: "
                            f"TSV={tsv_source_api!r}, payload={payload_source_api!r}"
                        )

                    normalized_evidence = _stringify_hashes(evidence)
                    # Make a recomputed hash explicit in the normalized payload.
                    normalized_evidence["source_hash"] = source_hash
                    normalized_payload_bytes = _canonical_json_bytes(normalized_evidence)
                    retracted_raw = row[columns["retracted"]]
                    retracted = _parse_neo4j_boolean(
                        retracted_raw,
                        field=f"nodes_Evidence row {rows_scanned} retracted:boolean",
                    )
                    record: dict[str, Any] = {
                        "entry_id": f"e{entries_written + 1:012d}",
                        "input_row": rows_scanned,
                        "matches_hash": matches_hash,
                        "source_hash": source_hash,
                        "source_api": source_api,
                        "raw_payload_bytes": len(raw_payload.encode("utf-8")),
                        "raw_payload_sha256": _sha256_bytes(raw_payload.encode("utf-8")),
                        "payload_sha256": _sha256_bytes(normalized_payload_bytes),
                        "raw_evidence_json": raw_payload,
                        "evidence": normalized_evidence,
                        "cogex_retracted": retracted,
                        "cogex_retracted_raw": retracted_raw,
                        # Preserve every non-payload Neo4j cell verbatim.  The
                        # payload's exact decoded cell is preserved above.
                        "cogex_fields": {
                            column: row[index]
                            for index, column in enumerate(header)
                            if index != columns["evidence"]
                        },
                    }
                    if "id" in columns:
                        record["cogex_evidence_id"] = row[columns["id"]]
                    if "label" in columns:
                        record["cogex_labels"] = row[columns["label"]]
                    _assert_hashes_are_strings(record)
                    spool.write(_json_line(record))
                    entries_written += 1

                state["rows_scanned"] = rows_scanned
                state["entries_written"] = entries_written
                if rows_scanned % checkpoint_every == 0:
                    _checkpoint(checkpoint_path, spool, state)
                if stop_after_scan_rows is not None and rows_scanned >= stop_after_scan_rows:
                    _checkpoint(checkpoint_path, spool, state)
                    raise ScanInterrupted(
                        f"synthetic interruption after {rows_scanned} input rows"
                    )

            # Exhaustion through csv.reader causes gzip to consume and validate the
            # trailer, so this is the digest of the full compressed source file.
            nodes_sha256 = hashing.digest.hexdigest()
            if expected_nodes_rows is not None and rows_scanned != expected_nodes_rows:
                raise ValueError(
                    f"nodes_Evidence row-count mismatch: scanned {rows_scanned}, "
                    f"expected {expected_nodes_rows}"
                )
            if expected_nodes_sha256 is not None and nodes_sha256 != expected_nodes_sha256:
                raise ValueError(
                    "nodes_Evidence SHA-256 mismatch: "
                    f"scanned {nodes_sha256}, expected {expected_nodes_sha256}"
                )
            state.update(
                {
                    "rows_scanned": rows_scanned,
                    "entries_written": entries_written,
                    "nodes_sha256": nodes_sha256,
                    "scan_complete": True,
                }
            )
            _checkpoint(checkpoint_path, spool, state)
    return state


def _write_statements(
    path: Path,
    statements: dict[str, dict[str, Any]],
) -> int:
    with path.open("wb") as fh:
        for matches_hash in sorted(statements, key=_hash_sort_key):
            fh.write(_json_line(statements[matches_hash]))
    return len(statements)


def _index_evidence_spool(
    spool_path: Path,
    pair_index_path: Path,
    *,
    target_hashes: set[str],
    gold_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    """Build a unique-pair index without altering the entry spool."""
    pair_entries: dict[tuple[str, str], list[str]] = defaultdict(list)
    pair_payloads: dict[
        tuple[str, str], dict[str, Counter[str]]
    ] = defaultdict(lambda: defaultdict(Counter))
    pair_source_apis: dict[tuple[str, str], set[str]] = defaultdict(set)
    entry_source_api_counts: Counter[str] = Counter()
    pair_source_api_counts: Counter[str] = Counter()
    target_entry_source_api_counts: dict[str, Counter[str]] = defaultdict(Counter)
    target_pair_source_api_counts: dict[str, Counter[str]] = defaultdict(Counter)
    target_entry_counts: Counter[str] = Counter()
    normalized_payload_digests: set[str] = set()
    raw_payload_digests: set[str] = set()
    entries = 0

    with spool_path.open("rb") as fh:
        for line_no, line in enumerate(fh, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - internal corruption
                raise ValueError(f"corrupt evidence spool line {line_no}: {exc}") from exc
            _assert_hashes_are_strings(record)
            matches_hash = record["matches_hash"]
            source_hash = record["source_hash"]
            if matches_hash not in target_hashes:
                raise AssertionError(f"spool contains non-target hash {matches_hash}")
            pair = (matches_hash, source_hash)
            pair_entries[pair].append(record["entry_id"])
            pair_payloads[pair][record["payload_sha256"]][
                record["raw_payload_sha256"]
            ] += 1
            pair_source_apis[pair].add(record["source_api"])
            entry_source_api_counts[record["source_api"]] += 1
            target_entry_source_api_counts[matches_hash][record["source_api"]] += 1
            target_entry_counts[matches_hash] += 1
            normalized_payload_digests.add(record["payload_sha256"])
            raw_payload_digests.add(record["raw_payload_sha256"])
            entries += 1

    covered_targets = set(target_entry_counts)
    missing_targets = target_hashes - covered_targets
    covered_gold_pairs = gold_pairs & pair_entries.keys()
    missing_gold_pairs = gold_pairs - pair_entries.keys()
    if missing_targets:
        raise ValueError(
            "nodes_Evidence is missing target matches hashes: "
            + ", ".join(sorted(missing_targets, key=_hash_sort_key)[:20])
        )
    if missing_gold_pairs:
        sample = sorted(
            missing_gold_pairs,
            key=lambda pair: (_hash_sort_key(pair[0]), _hash_sort_key(pair[1])),
        )[:20]
        raise ValueError(
            "nodes_Evidence is missing curated exact pairs: "
            + ", ".join(f"({mh}, {sh})" for mh, sh in sample)
        )

    pairs_with_variants = 0
    max_variants = 0
    pair_rows = 0
    with pair_index_path.open("wb") as out:
        for pair in sorted(
            pair_entries,
            key=lambda item: (_hash_sort_key(item[0]), _hash_sort_key(item[1])),
        ):
            matches_hash, source_hash = pair
            source_apis = pair_source_apis[pair]
            if len(source_apis) != 1:
                raise ValueError(
                    f"exact pair {pair} spans multiple source_api values: "
                    f"{sorted(source_apis)}"
                )
            source_api = next(iter(source_apis))
            pair_source_api_counts[source_api] += 1
            target_pair_source_api_counts[matches_hash][source_api] += 1
            variants = []
            for payload_sha256 in sorted(pair_payloads[pair]):
                raw_counts = pair_payloads[pair][payload_sha256]
                variants.append(
                    {
                        "payload_sha256": payload_sha256,
                        "entry_count": sum(raw_counts.values()),
                        "raw_payloads": [
                            {
                                "raw_payload_sha256": raw_digest,
                                "entry_count": raw_counts[raw_digest],
                            }
                            for raw_digest in sorted(raw_counts)
                        ],
                    }
                )
            variant_count = len(variants)
            pairs_with_variants += int(variant_count > 1)
            max_variants = max(max_variants, variant_count)
            row = {
                "matches_hash": matches_hash,
                "source_hash": source_hash,
                "source_api": source_api,
                "entry_count": len(pair_entries[pair]),
                "entry_ids": pair_entries[pair],
                "payload_variant_count": variant_count,
                "payload_variants": variants,
            }
            _assert_hashes_are_strings(row)
            out.write(_json_line(row))
            pair_rows += 1

    duplicate_multiplicity = entries - pair_rows
    reconciliation = {
        "entries_equal_entry_source_api_sum": entries
        == sum(entry_source_api_counts.values()),
        "unique_pairs_equal_pair_source_api_sum": pair_rows
        == sum(pair_source_api_counts.values()),
        "entries_equal_unique_pairs_plus_duplicate_multiplicity": entries
        == pair_rows + duplicate_multiplicity,
        "every_pair_has_one_source_api": all(
            len(source_apis) == 1 for source_apis in pair_source_apis.values()
        ),
        "all_target_hashes_covered": covered_targets == target_hashes,
        "all_gold_exact_pairs_covered": covered_gold_pairs == gold_pairs,
    }
    if not all(reconciliation.values()):  # pragma: no cover - construction invariant
        raise AssertionError(f"evidence reconciliation failed: {reconciliation}")

    return {
        "entries": entries,
        "unique_pairs": pair_rows,
        "duplicate_entry_multiplicity": duplicate_multiplicity,
        "entry_source_api_counts": _counter_json(entry_source_api_counts),
        "pair_source_api_counts": _counter_json(pair_source_api_counts),
        "target_entry_source_api_counts": {
            target: _counter_json(target_entry_source_api_counts[target])
            for target in sorted(target_hashes, key=_hash_sort_key)
        },
        "target_pair_source_api_counts": {
            target: _counter_json(target_pair_source_api_counts[target])
            for target in sorted(target_hashes, key=_hash_sort_key)
        },
        "target_entry_counts": {
            key: target_entry_counts[key]
            for key in sorted(target_entry_counts, key=_hash_sort_key)
        },
        "target_hashes_covered": len(covered_targets),
        "gold_exact_pairs_covered": len(covered_gold_pairs),
        "unique_normalized_payload_digests": len(normalized_payload_digests),
        "unique_raw_payload_digests": len(raw_payload_digests),
        "pairs_with_multiple_payload_variants": pairs_with_variants,
        "max_payload_variants_per_pair": max_variants,
        "reconciliation": reconciliation,
    }


def _normalize_source_counts(
    path: Path,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    with path.open("rb") as fh:
        loaded = pickle.load(fh)
    if not isinstance(loaded, Mapping):
        raise ValueError("source_counts.pkl must contain a mapping")
    normalized: dict[str, dict[str, int]] = {}
    for raw_hash, raw_counts in loaded.items():
        matches_hash = str(_exact_int(raw_hash, field="source_counts key"))
        if matches_hash in normalized:
            raise ValueError(f"duplicate normalized source_counts key {matches_hash}")
        if not isinstance(raw_counts, Mapping):
            raise ValueError(f"source counts for {matches_hash} are not a mapping")
        counts: dict[str, int] = {}
        for raw_source, raw_count in raw_counts.items():
            if not isinstance(raw_source, str) or not raw_source:
                raise ValueError(f"source_counts source for {matches_hash} is invalid")
            count = _exact_int(
                raw_count,
                field=f"source_counts[{matches_hash}][{raw_source}]",
            )
            if count < 0:
                raise ValueError(
                    f"source_counts[{matches_hash}][{raw_source}] is negative"
                )
            if count:
                counts[raw_source] = count
        normalized[matches_hash] = dict(sorted(counts.items()))
    identity = _full_file_identity(path)
    identity.update({"format": "trusted pickle", "statement_hashes": len(normalized)})
    return normalized, identity


def _normalize_belief_scores(
    path: Path,
) -> tuple[dict[str, float], dict[str, Any]]:
    with path.open("rb") as fh:
        loaded = pickle.load(fh)
    if not isinstance(loaded, Mapping):
        raise ValueError("belief_scores.pkl must contain a mapping")
    normalized: dict[str, float] = {}
    for raw_hash, raw_belief in loaded.items():
        matches_hash = str(_exact_int(raw_hash, field="belief_scores key"))
        if matches_hash in normalized:
            raise ValueError(f"duplicate normalized belief_scores key {matches_hash}")
        if isinstance(raw_belief, bool) or not isinstance(raw_belief, (int, float)):
            raise ValueError(f"belief for {matches_hash} is not numeric")
        belief = float(raw_belief)
        if not math.isfinite(belief) or not 0.0 <= belief <= 1.0:
            raise ValueError(f"belief for {matches_hash} is outside [0, 1]")
        normalized[matches_hash] = belief
    identity = _full_file_identity(path)
    identity.update({"format": "trusted pickle", "statement_hashes": len(normalized)})
    return normalized, identity


def _load_source_name_map(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source-name map JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("source-name map must be a JSON object")
    mapping: dict[str, str] = {}
    for source_name, source_api in loaded.items():
        if (
            not isinstance(source_name, str)
            or not source_name
            or not isinstance(source_api, str)
            or not source_api
        ):
            raise ValueError("source-name map keys and values must be non-empty strings")
        mapping[source_name] = source_api
    identity = _full_file_identity(path)
    identity.update(
        {
            "format": "JSON object mapping source_counts source name to source_api",
            "mappings": len(mapping),
        }
    )
    return mapping, identity


def _map_source_counter(
    counts: Mapping[str, int],
    source_name_to_api: Mapping[str, str],
) -> Counter[str]:
    mapped: Counter[str] = Counter()
    for source_name, count in counts.items():
        mapped[source_name_to_api.get(source_name, source_name)] += count
    return mapped


def _source_differences(
    observed: Mapping[str, int],
    expected: Mapping[str, int],
) -> list[dict[str, Any]]:
    differences = []
    for source_api in sorted(set(observed) | set(expected)):
        observed_count = int(observed.get(source_api, 0))
        expected_count = int(expected.get(source_api, 0))
        if observed_count != expected_count:
            differences.append(
                {
                    "source_api": source_api,
                    "evidence_entry_count": observed_count,
                    "mapped_source_counts_count": expected_count,
                    "delta_source_counts_minus_entries": expected_count
                    - observed_count,
                }
            )
    return differences


def _build_refinement_db(
    refinements_path: Path,
    work_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Stream headerless (more specific, less specific) rows into SQLite."""
    quick = _quick_file_identity(refinements_path)
    db_path = work_dir / "refinements.sqlite"
    meta_path = work_dir / "refinements.sqlite.meta.json"
    if db_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = None
        if isinstance(meta, dict) and meta.get("quick_identity") == quick:
            with sqlite3.connect(db_path) as connection:
                unique_edges = connection.execute(
                    "SELECT COUNT(*) FROM edges"
                ).fetchone()[0]
            if unique_edges == meta.get("unique_edges"):
                return db_path, meta

    for stale in (db_path, meta_path, work_dir / "refinements.sqlite.tmp"):
        stale.unlink(missing_ok=True)
    tmp_db = work_dir / "refinements.sqlite.tmp"
    connection = sqlite3.connect(tmp_db)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE edges (
                more_specific INTEGER NOT NULL,
                less_specific INTEGER NOT NULL,
                PRIMARY KEY (more_specific, less_specific)
            ) WITHOUT ROWID;
            CREATE INDEX edges_by_less_specific ON edges (less_specific);
            """
        )
        raw_rows = 0
        self_edges = 0
        batch: list[tuple[int, int]] = []
        csv.field_size_limit(1 << 28)
        with _open_hashed_gzip_text(refinements_path) as (text, hashing):
            reader = csv.reader(text, delimiter="\t")
            for row_no, row in enumerate(reader, start=1):
                if len(row) != 2:
                    raise ValueError(
                        f"refinements row {row_no} has {len(row)} columns; expected 2"
                    )
                more_specific = _exact_int(
                    row[0], field=f"refinements row {row_no} more_specific"
                )
                less_specific = _exact_int(
                    row[1], field=f"refinements row {row_no} less_specific"
                )
                for value in (more_specific, less_specific):
                    if not -(1 << 63) <= value < (1 << 63):
                        raise ValueError(
                            f"refinement hash {value} is outside signed 64-bit range"
                        )
                raw_rows += 1
                if more_specific == less_specific:
                    self_edges += 1
                    continue
                batch.append((more_specific, less_specific))
                if len(batch) >= 100_000:
                    connection.executemany(
                        "INSERT OR IGNORE INTO edges VALUES (?, ?)", batch
                    )
                    connection.commit()
                    batch.clear()
            if batch:
                connection.executemany("INSERT OR IGNORE INTO edges VALUES (?, ?)", batch)
                connection.commit()
            refinements_sha256 = hashing.digest.hexdigest()
        if self_edges:
            raise ValueError(
                f"refinements contains {self_edges} self edge(s); direction/acyclicity invalid"
            )
        unique_edges = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    finally:
        connection.close()
    os.replace(tmp_db, db_path)
    meta = {
        "quick_identity": quick,
        "sha256": refinements_sha256,
        "format": "gzip TSV without header",
        "direction": "more_specific_to_less_specific",
        "raw_rows": raw_rows,
        "unique_edges": unique_edges,
        "duplicate_edges": raw_rows - unique_edges,
    }
    _atomic_write_json(meta_path, meta)
    return db_path, meta


def _ancestor_closure(
    db_path: Path,
    target_hashes: set[str],
) -> dict[str, list[str]]:
    closure: dict[str, list[str]] = {}
    sql = """
        WITH RECURSIVE ancestors(node) AS (
            SELECT more_specific FROM edges WHERE less_specific = ?
            UNION
            SELECT edges.more_specific
            FROM edges JOIN ancestors ON edges.less_specific = ancestors.node
        )
        SELECT node FROM ancestors
    """
    with sqlite3.connect(db_path) as connection:
        for target in sorted(target_hashes, key=_hash_sort_key):
            target_int = _hash_sort_key(target)
            ancestors = {str(row[0]) for row in connection.execute(sql, (target_int,))}
            if target in ancestors:
                raise ValueError(
                    f"refinement cycle is reachable from target {target}; closure is invalid"
                )
            closure[target] = sorted(ancestors, key=_hash_sort_key)
    return closure


def _write_ancestor_closure(
    path: Path,
    closure: dict[str, list[str]],
) -> int:
    rows = 0
    with path.open("wb") as out:
        for target in sorted(closure, key=_hash_sort_key):
            for ancestor in closure[target]:
                row = {
                    "target_matches_hash": target,
                    "ancestor_more_specific_hash": ancestor,
                }
                _assert_hashes_are_strings(row)
                out.write(_json_line(row))
                rows += 1
    return rows


def _counter_json(counter: Counter[str] | Mapping[str, int]) -> dict[str, int]:
    return {source: int(counter[source]) for source in sorted(counter) if counter[source]}


def _write_hybrid_counts(
    path: Path,
    *,
    target_entry_counts: Mapping[str, int],
    target_entry_source_api_counts: Mapping[str, Mapping[str, int]],
    source_counts: dict[str, dict[str, int]],
    source_name_to_api: Mapping[str, str],
    closure: dict[str, list[str]],
    belief_scores: dict[str, float] | None,
) -> tuple[int, dict[str, Any]]:
    targets = set(target_entry_counts)
    missing_targets = targets - source_counts.keys()
    if missing_targets:
        raise ValueError(
            "source_counts is missing target hashes: "
            + ", ".join(sorted(missing_targets, key=_hash_sort_key)[:20])
        )
    all_ancestors = {ancestor for values in closure.values() for ancestor in values}
    missing_ancestors = all_ancestors - source_counts.keys()
    if missing_ancestors:
        raise ValueError(
            "source_counts is missing refinement ancestors: "
            + ", ".join(sorted(missing_ancestors, key=_hash_sort_key)[:20])
        )
    if belief_scores is not None:
        missing_beliefs = targets - belief_scores.keys()
        if missing_beliefs:
            raise ValueError(
                "belief_scores is missing target hashes: "
                + ", ".join(sorted(missing_beliefs, key=_hash_sort_key)[:20])
            )

    direct_source_name_distribution: Counter[str] = Counter()
    direct_source_api_distribution: Counter[str] = Counter()
    hybrid_source_name_distribution: Counter[str] = Counter()
    hybrid_source_api_distribution: Counter[str] = Counter()
    direct_total = 0
    hybrid_total = 0
    with path.open("wb") as out:
        for target in sorted(targets, key=_hash_sort_key):
            direct = Counter(source_counts[target])
            direct_source_api = _map_source_counter(direct, source_name_to_api)
            direct_count = sum(direct.values())
            observed_count = target_entry_counts[target]
            if direct_count != observed_count:
                raise ValueError(
                    f"direct count mismatch for {target}: nodes_Evidence has "
                    f"{observed_count}, source_counts has {direct_count}"
                )
            source_differences = _source_differences(
                target_entry_source_api_counts[target], direct_source_api
            )
            if source_differences:
                raise ValueError(
                    f"per-source direct count mismatch for {target}; provide or "
                    "correct --source-name-map (source_counts name -> source_api): "
                    + json.dumps(source_differences, sort_keys=True)
                )
            ancestor_counts: Counter[str] = Counter()
            for ancestor in closure[target]:
                ancestor_counts.update(source_counts[ancestor])
            ancestor_source_api = _map_source_counter(
                ancestor_counts, source_name_to_api
            )
            hybrid = direct + ancestor_counts
            hybrid_source_api = direct_source_api + ancestor_source_api
            if hybrid != direct + ancestor_counts:  # pragma: no cover - explicit invariant
                raise AssertionError("hybrid counter arithmetic failed")
            row: dict[str, Any] = {
                "matches_hash": target,
                "direct_source_name_counts": _counter_json(direct),
                "direct_source_api_counts": _counter_json(direct_source_api),
                "direct_total": direct_count,
                "ancestor_hashes": closure[target],
                "ancestor_source_name_counts": _counter_json(ancestor_counts),
                "ancestor_source_api_counts": _counter_json(ancestor_source_api),
                "ancestor_total": sum(ancestor_counts.values()),
                "hybrid_source_name_counts": _counter_json(hybrid),
                "hybrid_source_api_counts": _counter_json(hybrid_source_api),
                "hybrid_total": sum(hybrid.values()),
            }
            if belief_scores is not None:
                row["frozen_belief"] = belief_scores[target]
            _assert_hashes_are_strings(row)
            out.write(_json_line(row))
            direct_source_name_distribution.update(direct)
            direct_source_api_distribution.update(direct_source_api)
            hybrid_source_name_distribution.update(hybrid)
            hybrid_source_api_distribution.update(hybrid_source_api)
            direct_total += direct_count
            hybrid_total += sum(hybrid.values())

    assertions = {
        "every_target_has_source_counts": not missing_targets,
        "every_ancestor_has_source_counts": not missing_ancestors,
        "direct_totals_match_evidence_entries_per_target": True,
        "direct_source_api_counts_match_evidence_entries_per_target": True,
        "ancestors_are_counted_once_per_target": True,
        "hybrid_equals_direct_plus_unique_ancestors": True,
        "every_target_has_belief_score_when_supplied": belief_scores is None
        or not (targets - belief_scores.keys()),
    }
    return len(targets), {
        "direct_total": direct_total,
        "hybrid_total": hybrid_total,
        "direct_source_name_counts": _counter_json(direct_source_name_distribution),
        "direct_source_api_counts": _counter_json(direct_source_api_distribution),
        "hybrid_source_name_counts": _counter_json(hybrid_source_name_distribution),
        "hybrid_source_api_counts": _counter_json(hybrid_source_api_distribution),
        "source_name_to_api_mapping": dict(sorted(source_name_to_api.items())),
        "belief_reconciliation": {
            "belief_scores_supplied": belief_scores is not None,
            "target_coverage_verified": belief_scores is None
            or not (targets - belief_scores.keys()),
            "parity_to_emitted_hybrid_counts": "not_verified",
            "reason": (
                "the synchronized HybridScorer artifact and source-name mapping "
                "used to recompute probabilities are not part of belief_scores.pkl"
            ),
        },
        "assertions": assertions,
    }


def _validate_reusable_manifest(
    manifest_path: Path,
    output_dir: Path,
    config_fingerprint: str,
) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("configuration_fingerprint") != config_fingerprint
    ):
        return None
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return None
    for descriptor in files.values():
        if not isinstance(descriptor, dict):
            return None
        relative = descriptor.get("path")
        expected_digest = descriptor.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).name != relative
            or not isinstance(expected_digest, str)
            or not SHA256_RE.fullmatch(expected_digest)
        ):
            return None
        artifact = output_dir / relative
        if not artifact.is_file() or _sha256_file(artifact) != expected_digest:
            return None
    return manifest


def _artifact_descriptor(
    path: Path,
    *,
    rows: int,
    compression: str,
) -> dict[str, Any]:
    return {
        "path": path.name,
        "format": "jsonl",
        "compression": compression,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - filesystem-specific hardening
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_frozen_substrate(
    *,
    gold_path: Path | str,
    nodes_evidence_path: Path | str,
    output_dir: Path | str,
    source_counts_path: Path | str | None = None,
    source_name_map_path: Path | str | None = None,
    refinements_path: Path | str | None = None,
    belief_scores_path: Path | str | None = None,
    compression: str = "auto",
    expected_target_count: int | None = FROZEN_TARGET_HASHES,
    expected_nodes_rows: int | None = FROZEN_NODES_ROWS,
    expected_nodes_sha256: str | None = FROZEN_NODES_SHA256,
    checkpoint_every: int = 250_000,
    scan_only: bool = False,
    force: bool = False,
    _stop_after_scan_rows: int | None = None,
) -> MaterializationResult:
    """Build or reuse a complete normalized substrate bundle.

    ``_stop_after_scan_rows`` exists solely to exercise resume semantics in
    synthetic tests.  It is intentionally absent from the CLI.
    """
    gold_path = Path(gold_path)
    nodes_evidence_path = Path(nodes_evidence_path)
    output_dir = Path(output_dir)
    source_counts_path = Path(source_counts_path) if source_counts_path else None
    source_name_map_path = Path(source_name_map_path) if source_name_map_path else None
    refinements_path = Path(refinements_path) if refinements_path else None
    belief_scores_path = Path(belief_scores_path) if belief_scores_path else None

    if refinements_path is not None and source_counts_path is None:
        raise ValueError("--refinements requires synchronized --source-counts")
    if belief_scores_path is not None and source_counts_path is None:
        raise ValueError("--belief-scores requires synchronized --source-counts")
    if source_name_map_path is not None and source_counts_path is None:
        raise ValueError("--source-name-map requires --source-counts")
    if expected_target_count is not None and expected_target_count < 1:
        raise ValueError("expected_target_count must be positive or None")
    if expected_nodes_rows is not None and expected_nodes_rows < 0:
        raise ValueError("expected_nodes_rows must be nonnegative or None")
    if expected_nodes_sha256 is not None and not SHA256_RE.fullmatch(
        expected_nodes_sha256
    ):
        raise ValueError("expected_nodes_sha256 must be a lowercase SHA-256 or None")

    selected_compression, fallback_reason = _resolve_compression(compression)
    statements, gold_pairs, gold_identity = _load_gold(
        gold_path, expected_target_count=expected_target_count
    )
    target_hashes = set(statements)
    source_name_to_api: dict[str, str] = {}
    source_name_map_identity: dict[str, Any] | None = None
    if source_name_map_path is not None:
        source_name_to_api, source_name_map_identity = _load_source_name_map(
            source_name_map_path
        )
    quick_nodes = _quick_file_identity(nodes_evidence_path)
    quick_inputs: dict[str, Any] = {
        "gold": _quick_file_identity(gold_path),
        "nodes_evidence": quick_nodes,
    }
    if source_counts_path is not None:
        quick_inputs["source_counts"] = _quick_file_identity(source_counts_path)
    if source_name_map_path is not None:
        quick_inputs["source_name_map"] = _quick_file_identity(source_name_map_path)
    if refinements_path is not None:
        quick_inputs["refinements"] = _quick_file_identity(refinements_path)
    if belief_scores_path is not None:
        quick_inputs["belief_scores"] = _quick_file_identity(belief_scores_path)

    scan_config = {
        "schema_version": SCHEMA_VERSION,
        "gold": quick_inputs["gold"],
        "gold_sha256": gold_identity["sha256"],
        "nodes_evidence": quick_nodes,
        "expected_target_count": expected_target_count,
        "expected_nodes_rows": expected_nodes_rows,
        "expected_nodes_sha256": expected_nodes_sha256,
    }
    scan_fingerprint = _sha256_bytes(_canonical_json_bytes(scan_config))
    config = {
        "schema_version": SCHEMA_VERSION,
        "inputs": quick_inputs,
        "gold_sha256": gold_identity["sha256"],
        "scan_fingerprint": scan_fingerprint,
        "compression_requested": compression,
        "compression_selected": selected_compression,
        "expected_target_count": expected_target_count,
        "expected_nodes_rows": expected_nodes_rows,
        "expected_nodes_sha256": expected_nodes_sha256,
    }
    config_fingerprint = _sha256_bytes(_canonical_json_bytes(config))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    with _output_lock(output_dir):
        if not force:
            reusable = _validate_reusable_manifest(
                manifest_path, output_dir, config_fingerprint
            )
            if reusable is not None:
                return MaterializationResult("reused", manifest_path, reusable)

        work_dir = output_dir / WORK_DIR_NAME
        state = _load_or_create_checkpoint(
            work_dir,
            scan_fingerprint=scan_fingerprint,
            quick_nodes=quick_nodes,
            force=force,
        )
        state = _scan_evidence_nodes(
            nodes_evidence_path,
            target_hashes,
            work_dir,
            state,
            expected_nodes_rows=expected_nodes_rows,
            expected_nodes_sha256=expected_nodes_sha256,
            checkpoint_every=checkpoint_every,
            stop_after_scan_rows=_stop_after_scan_rows,
        )

        statements_logical = work_dir / "statements.jsonl"
        pair_index_logical = work_dir / "pair_index.jsonl"
        evidence_logical = work_dir / EVIDENCE_SPOOL_NAME
        statement_rows = _write_statements(statements_logical, statements)
        evidence_index = _index_evidence_spool(
            evidence_logical,
            pair_index_logical,
            target_hashes=target_hashes,
            gold_pairs=gold_pairs,
        )
        if evidence_index["entries"] != state["entries_written"]:
            raise AssertionError(
                "evidence spool row count does not match the durable scan checkpoint"
            )

        if scan_only:
            scan_summary_path = work_dir / SCAN_SUMMARY_NAME
            scan_summary = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "normalized_frozen_indra_substrate_scan_checkpoint",
                "status": "scan_complete_pending_enrichment",
                "created_at": _utc_now(),
                "scan_fingerprint": scan_fingerprint,
                "bundle_id": state["bundle_id"],
                "inputs": {
                    "gold": gold_identity,
                    "nodes_evidence": {
                        **quick_nodes,
                        "sha256": state["nodes_sha256"],
                        "format": "gzip Neo4j-import TSV with header",
                        "rows": state["rows_scanned"],
                        "header": state["header"],
                    },
                },
                "counts": {
                    "input_evidence_rows_scanned": state["rows_scanned"],
                    "matched_evidence_entries": evidence_index["entries"],
                    "unique_exact_pairs": evidence_index["unique_pairs"],
                    "duplicate_entry_multiplicity": evidence_index[
                        "duplicate_entry_multiplicity"
                    ],
                    "target_hashes_covered": evidence_index[
                        "target_hashes_covered"
                    ],
                    "gold_exact_pairs_covered": evidence_index[
                        "gold_exact_pairs_covered"
                    ],
                },
                "source_distribution": evidence_index["entry_source_api_counts"],
                "reconciliation": evidence_index["reconciliation"],
                "publication_gate": (
                    "checkpoint only; rerun without --scan-only and with synchronized "
                    "source_counts/refinements/belief/model inputs before claiming "
                    "production Hybrid parity"
                ),
            }
            _atomic_write_json(scan_summary_path, scan_summary)
            return MaterializationResult(
                "scan_complete", scan_summary_path, scan_summary
            )

        full_inputs: dict[str, Any] = {
            "gold": gold_identity,
            "nodes_evidence": {
                **quick_nodes,
                "sha256": state["nodes_sha256"],
                "format": "gzip Neo4j-import TSV with header",
                "rows": state["rows_scanned"],
                "header": state["header"],
            },
        }
        logical_artifacts: list[tuple[str, Path, int]] = [
            ("statements", statements_logical, statement_rows),
            ("evidence_entries", evidence_logical, evidence_index["entries"]),
            ("pair_index", pair_index_logical, evidence_index["unique_pairs"]),
        ]

        closure = {target: [] for target in target_hashes}
        refinement_summary: dict[str, Any] | None = None
        if refinements_path is not None:
            refinement_db, refinement_meta = _build_refinement_db(
                refinements_path, work_dir
            )
            closure = _ancestor_closure(refinement_db, target_hashes)
            closure_logical = work_dir / "ancestor_closure.jsonl"
            closure_rows = _write_ancestor_closure(closure_logical, closure)
            logical_artifacts.append(
                ("ancestor_closure", closure_logical, closure_rows)
            )
            full_inputs["refinements"] = {
                **refinement_meta["quick_identity"],
                "sha256": refinement_meta["sha256"],
                "format": refinement_meta["format"],
                "direction": refinement_meta["direction"],
                "raw_rows": refinement_meta["raw_rows"],
                "unique_edges": refinement_meta["unique_edges"],
                "duplicate_edges": refinement_meta["duplicate_edges"],
            }
            refinement_summary = {
                "direction": "more_specific_to_less_specific",
                "closure_traversal": "incoming edges from each target",
                "target_ancestor_pairs": closure_rows,
                "targets_with_ancestors": sum(bool(values) for values in closure.values()),
                "unique_ancestor_hashes": len(
                    {ancestor for values in closure.values() for ancestor in values}
                ),
                "assertions": {
                    "input_has_no_self_edges": True,
                    "target_closures_exclude_self": True,
                    "closure_uses_unique_ancestors": True,
                    "edge_direction_matches_indra_hybrid_calculation": True,
                },
            }

        hybrid_summary: dict[str, Any] | None = None
        if source_counts_path is not None:
            source_counts, source_counts_identity = _normalize_source_counts(
                source_counts_path
            )
            full_inputs["source_counts"] = source_counts_identity
            if source_name_map_identity is not None:
                full_inputs["source_name_map"] = source_name_map_identity
            beliefs = None
            if belief_scores_path is not None:
                beliefs, beliefs_identity = _normalize_belief_scores(belief_scores_path)
                full_inputs["belief_scores"] = beliefs_identity
            hybrid_logical = work_dir / "hybrid_counts.jsonl"
            hybrid_rows, hybrid_summary = _write_hybrid_counts(
                hybrid_logical,
                target_entry_counts=evidence_index["target_entry_counts"],
                target_entry_source_api_counts=evidence_index[
                    "target_entry_source_api_counts"
                ],
                source_counts=source_counts,
                source_name_to_api=source_name_to_api,
                closure=closure,
                belief_scores=beliefs,
            )
            logical_artifacts.append(("hybrid_counts", hybrid_logical, hybrid_rows))
            if refinement_summary is not None:
                refinement_summary["assertions"].update(
                    {
                        "every_ancestor_has_source_counts": hybrid_summary[
                            "assertions"
                        ]["every_ancestor_has_source_counts"],
                        "hybrid_counts_each_unique_ancestor_once": hybrid_summary[
                            "assertions"
                        ]["ancestors_are_counted_once_per_target"],
                    }
                )

        bundle_id = state["bundle_id"]
        suffix = _compression_suffix(selected_compression)
        files: dict[str, dict[str, Any]] = {}
        staged: list[tuple[Path, Path]] = []
        for logical_name, logical_path, rows in logical_artifacts:
            final_name = f"{logical_name}.{bundle_id}{suffix}"
            stage_path = work_dir / f".{final_name}.complete"
            _compress_file(logical_path, stage_path, selected_compression)
            files[logical_name] = _artifact_descriptor(
                stage_path,
                rows=rows,
                compression=selected_compression,
            )
            files[logical_name]["path"] = final_name
            staged.append((stage_path, output_dir / final_name))

        target_hash_bytes = b"".join(
            f"{value}\n".encode("ascii")
            for value in sorted(target_hashes, key=_hash_sort_key)
        )
        gold_pair_bytes = b"".join(
            f"{mh}\t{sh}\n".encode("ascii")
            for mh, sh in sorted(
                gold_pairs,
                key=lambda pair: (_hash_sort_key(pair[0]), _hash_sort_key(pair[1])),
            )
        )
        reconciliation = dict(evidence_index["reconciliation"])
        reconciliation["checkpoint_entries_equal_spool_entries"] = (
            state["entries_written"] == evidence_index["entries"]
        )
        if hybrid_summary is not None:
            reconciliation.update(hybrid_summary["assertions"])
        if not all(reconciliation.values()):  # pragma: no cover - asserted earlier
            raise AssertionError(f"manifest reconciliation failed: {reconciliation}")

        source_accounting: dict[str, Any] = {
            "evidence_entry_source_api_counts": evidence_index[
                "entry_source_api_counts"
            ],
            "unique_pair_source_api_counts": evidence_index[
                "pair_source_api_counts"
            ],
            "per_target": {
                target: {
                    "evidence_entry_source_api_counts": evidence_index[
                        "target_entry_source_api_counts"
                    ][target],
                    "unique_pair_source_api_counts": evidence_index[
                        "target_pair_source_api_counts"
                    ][target],
                }
                for target in sorted(target_hashes, key=_hash_sort_key)
            },
            "api_audit_comparison_contract": {
                "statement_key": "matches_hash (signed decimal string)",
                "pair_key": (
                    "(matches_hash, source_hash), both signed decimal strings"
                ),
                "source_dimension": "source_api",
                "entry_metric": "raw Evidence rows, including multiplicity",
                "pair_metric": "unique exact (matches_hash, source_hash) pairs",
            },
            "reconciliation": {
                "entry_counts_sum_to_matched_evidence_entries": sum(
                    evidence_index["entry_source_api_counts"].values()
                )
                == evidence_index["entries"],
                "pair_counts_sum_to_unique_exact_pairs": sum(
                    evidence_index["pair_source_api_counts"].values()
                )
                == evidence_index["unique_pairs"],
                "each_unique_pair_has_exactly_one_source_api": True,
            },
            "source_counts_comparison": {
                "status": "not_supplied",
                "note": (
                    "source_counts.pkl counts evidence entries, not unique source-hash pairs"
                ),
            },
        }
        if hybrid_summary is not None:
            source_names = hybrid_summary["direct_source_name_counts"]
            mapped_source_apis = hybrid_summary["direct_source_api_counts"]
            entry_source_apis = evidence_index["entry_source_api_counts"]
            aggregate_differences = _source_differences(
                entry_source_apis, mapped_source_apis
            )
            if aggregate_differences:  # pragma: no cover - per-target assertion is stronger
                raise AssertionError(
                    "aggregate source reconciliation failed after per-target checks"
                )
            source_accounting["source_counts_comparison"] = {
                "status": "exact_after_source_name_to_api_mapping",
                "source_name_namespace_counts": source_names,
                "mapped_source_api_counts": mapped_source_apis,
                "source_name_map_supplied": source_name_map_path is not None,
                "source_name_to_api_mapping": dict(sorted(source_name_to_api.items())),
                "unmapped_names_treated_as_source_api": sorted(
                    set(source_names) - source_name_to_api.keys()
                ),
                "per_source": [
                    {
                        "source_api": source_api,
                        "evidence_entry_count": entry_source_apis.get(source_api, 0),
                        "unique_pair_count": evidence_index[
                            "pair_source_api_counts"
                        ].get(source_api, 0),
                        "mapped_source_counts_count": mapped_source_apis.get(
                            source_api, 0
                        ),
                        "source_counts_minus_entries": mapped_source_apis.get(
                            source_api, 0
                        )
                        - entry_source_apis.get(source_api, 0),
                    }
                    for source_api in sorted(
                        set(entry_source_apis)
                        | set(evidence_index["pair_source_api_counts"])
                        | set(mapped_source_apis)
                    )
                ],
                "assertions": {
                    "mapped_source_api_counts_match_entries_per_target": True,
                    "mapped_source_api_counts_match_entries_in_aggregate": True,
                    "source_counts_total_matches_entry_total": hybrid_summary[
                        "direct_total"
                    ]
                    == evidence_index["entries"],
                },
            }
        if not all(source_accounting["reconciliation"].values()):
            raise AssertionError("source accounting reconciliation failed")

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "normalized_frozen_indra_substrate",
            "status": "complete",
            "created_at": _utc_now(),
            "configuration_fingerprint": config_fingerprint,
            "scan_fingerprint": scan_fingerprint,
            "bundle_id": bundle_id,
            "compression": {
                "requested": compression,
                "selected": selected_compression,
                "fallback_reason": fallback_reason,
            },
            "inputs": full_inputs,
            "targets": {
                "unique_matches_hashes": len(target_hashes),
                "matches_hashes_sha256": _sha256_bytes(target_hash_bytes),
                "gold_unique_exact_pairs": len(gold_pairs),
                "gold_exact_pairs_sha256": _sha256_bytes(gold_pair_bytes),
            },
            "files": files,
            "counts": {
                "input_evidence_rows_scanned": state["rows_scanned"],
                "statements": statement_rows,
                "matched_evidence_entries": evidence_index["entries"],
                "unique_exact_pairs": evidence_index["unique_pairs"],
                "duplicate_entry_multiplicity": evidence_index[
                    "duplicate_entry_multiplicity"
                ],
                "target_hashes_covered": evidence_index["target_hashes_covered"],
                "gold_exact_pairs_covered": evidence_index[
                    "gold_exact_pairs_covered"
                ],
            },
            "source_distribution": evidence_index["entry_source_api_counts"],
            "source_accounting": source_accounting,
            "payload_digests": {
                "algorithm": "sha256",
                "raw_digest_scope": "exact UTF-8 bytes of the Evidence TSV JSON cell",
                "normalized_digest_scope": (
                    "canonical parsed evidence JSON after identifier hashes become strings"
                ),
                "unique_raw_payloads": evidence_index[
                    "unique_raw_payload_digests"
                ],
                "unique_normalized_payloads": evidence_index[
                    "unique_normalized_payload_digests"
                ],
                "pairs_with_multiple_normalized_payload_variants": evidence_index[
                    "pairs_with_multiple_payload_variants"
                ],
                "max_normalized_payload_variants_per_pair": evidence_index[
                    "max_payload_variants_per_pair"
                ],
            },
            "reconciliation": reconciliation,
            "refinement": refinement_summary,
            "hybrid_counts": hybrid_summary,
        }

        # Each destination is unique to this bundle.  The old manifest remains
        # valid until the final atomic pointer replacement below.
        for stage_path, final_path in staged:
            os.replace(stage_path, final_path)
        _fsync_directory(output_dir)
        _atomic_write_json(manifest_path, manifest)
        _fsync_directory(output_dir)
        shutil.rmtree(work_dir)
        return MaterializationResult("created", manifest_path, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--nodes-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-counts", type=Path)
    parser.add_argument(
        "--source-name-map",
        type=Path,
        help=(
            "JSON object mapping source_counts source names to Evidence source_api; "
            "unlisted names are treated as already being source_api names"
        ),
    )
    parser.add_argument("--refinements", type=Path)
    parser.add_argument("--belief-scores", type=Path)
    parser.add_argument(
        "--compression",
        choices=("auto", "zstd", "gzip", "none"),
        default="auto",
        help="auto uses zstd when installed, otherwise records a gzip fallback",
    )
    parser.add_argument(
        "--expected-target-count",
        type=int,
        default=FROZEN_TARGET_HASHES,
        help=f"unique gold matches hashes (default: {FROZEN_TARGET_HASHES})",
    )
    parser.add_argument(
        "--expected-nodes-rows",
        type=int,
        default=FROZEN_NODES_ROWS,
        help=f"frozen Evidence data rows, excluding header (default: {FROZEN_NODES_ROWS})",
    )
    parser.add_argument(
        "--expected-nodes-sha256",
        default=FROZEN_NODES_SHA256,
        help="compressed nodes_Evidence SHA-256",
    )
    parser.add_argument(
        "--allow-unpinned-nodes",
        action="store_true",
        help="disable the default row-count and SHA assertions for another release",
    )
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help=(
            "finish and retain the raw-evidence checkpoint without publishing a "
            "complete bundle; a later enriched run reuses it without rescanning"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard this output directory's partial checkpoint and build a new bundle",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected_rows = None if args.allow_unpinned_nodes else args.expected_nodes_rows
    expected_sha256 = (
        None if args.allow_unpinned_nodes else args.expected_nodes_sha256
    )
    result = materialize_frozen_substrate(
        gold_path=args.gold,
        nodes_evidence_path=args.nodes_evidence,
        output_dir=args.output_dir,
        source_counts_path=args.source_counts,
        source_name_map_path=args.source_name_map,
        refinements_path=args.refinements,
        belief_scores_path=args.belief_scores,
        compression=args.compression,
        expected_target_count=args.expected_target_count,
        expected_nodes_rows=expected_rows,
        expected_nodes_sha256=expected_sha256,
        checkpoint_every=args.checkpoint_every,
        scan_only=args.scan_only,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "manifest": str(result.manifest_path.resolve()),
                "matched_evidence_entries": result.manifest["counts"][
                    "matched_evidence_entries"
                ],
                "unique_exact_pairs": result.manifest["counts"][
                    "unique_exact_pairs"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
