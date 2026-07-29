#!/usr/bin/env python3
"""Build the representative statement-gold, evidence-ledger, and E0 queue.

This is a local artifact transformation only.  It performs no inference and
makes no network calls.  Its immutable inputs are the 403-pair representative
gold JSONL and a completed frozen-substrate scan work directory containing
``checkpoint.json``, ``pair_index.jsonl``, and ``evidence_entries.jsonl``.

The E0 statement rule is deliberately strict:

* any exact reviewed pair tagged ``correct`` makes the statement positive;
* a statement is negative only when every unique evidence pair is reviewed and
  every review is noncorrect;
* every other statement is unresolved.

Only unreviewed pairs on unresolved statements enter the prediction-blind
queue.  All pairs, including unreviewed pairs on already-positive statements,
remain in the full evidence ledger.  Queue linkage is held in a separate
mapping ledger.

The output directory must not exist.  Files are written to a sibling staging
directory, the manifest is created last, and the complete directory is
published with one atomic no-replace rename.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import gzip
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from materialize_frozen_representative_substrate import (  # noqa: E402
    _assert_hashes_are_strings,
    _canonical_json_bytes,
    _compression_suffix,
    _counter_json,
    _fsync_directory,
    _json_line,
    _resolve_compression,
    _sha256_bytes,
    _sha256_file,
    _stringify_hashes,
    _utc_now,
)


SCHEMA_VERSION = 1
SIGNED_INTEGER_RE = re.compile(r"^-?\d+$")
OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")

DEFAULT_GOLD = ROOT / "data/benchmark/representative_indra_curations_400.jsonl"
DEFAULT_GOLD_META = (
    ROOT / "data/benchmark/representative_indra_curations_400.meta.json"
)
DEFAULT_SCAN_WORK = (
    ROOT
    / "data/corpora/representative_indra_403_full_evidence/.frozen-substrate.work"
)

REAL_EXPECTED_COUNTS = {
    "gold_exact_pairs": 403,
    "gold_positive_pairs": 199,
    "gold_noncorrect_pairs": 204,
    "statements": 403,
    "positive_statements": 199,
    "negative_statements": 44,
    "unresolved_statements": 160,
    "negative_singleton_statements": 44,
    "raw_evidence_entries": 207_485,
    "unique_evidence_pairs": 205_343,
    "queued_unreviewed_pairs": 62_752,
    "other_unreviewed_pairs_retained_in_ledger": 142_188,
}

FORBIDDEN_QUEUE_KEY_STEMS = (
    "label",
    "tag",
    "curator",
    "curation",
    "belief",
    "predict",
    "disagreement",
    "gold",
    "verdict",
)


@dataclass
class StatementStats:
    matches_hash: str
    unique_pairs: int = 0
    raw_entries: int = 0
    reviewed_pairs: int = 0
    reviewed_positive_pairs: int = 0
    reviewed_noncorrect_pairs: int = 0
    unreviewed_pairs: int = 0
    queued_pairs: int = 0
    e0_label: str | None = None
    reviewed: list[dict[str, Any]] = field(default_factory=list)
    pair_source_api_counts: Counter[str] = field(default_factory=Counter)
    entry_source_api_counts: Counter[str] = field(default_factory=Counter)
    singleton_entry_count: int | None = None
    singleton_payload_variant_count: int | None = None


@dataclass
class LaneAnalysis:
    gold_rows: list[dict[str, Any]]
    gold_by_pair: dict[tuple[str, str], dict[str, Any]]
    statements: dict[str, dict[str, Any]]
    statement_stats: dict[str, StatementStats]
    counts: dict[str, int]
    checkpoint: dict[str, Any]
    scan_summary: dict[str, Any] | None
    input_files: dict[str, Path]
    gold_meta: dict[str, Any] | None
    pair_source_api_counts: dict[str, int]
    entry_source_api_counts: dict[str, int]


@dataclass(frozen=True)
class HarnessResult:
    output_dir: Path
    manifest: dict[str, Any]


def _exact_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a signed integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and SIGNED_INTEGER_RE.fullmatch(value):
        return int(value)
    raise ValueError(f"{field_name} must be a canonical signed integer; got {value!r}")


def _hash_string(value: Any, *, field_name: str) -> str:
    return str(_exact_int(value, field_name=field_name))


def _pair_sort_key(pair: tuple[str, str]) -> tuple[int, int]:
    return int(pair[0]), int(pair[1])


def _path_identity(path: Path, *, sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    identity["sha256"] = sha256 or _sha256_file(resolved)
    return identity


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _load_gold(
    gold_path: Path,
    gold_meta_path: Path | None,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any] | None,
]:
    rows: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    statements: dict[str, dict[str, Any]] = {}
    with gold_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid gold JSON line {line_no}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"gold line {line_no} is not an object")
            normalized = _stringify_hashes(raw)
            matches_hash = _hash_string(
                raw.get("matches_hash"), field_name=f"gold line {line_no} matches_hash"
            )
            source_hash = _hash_string(
                raw.get("source_hash"), field_name=f"gold line {line_no} source_hash"
            )
            tag = raw.get("tag")
            if not isinstance(tag, str) or not tag:
                raise ValueError(f"gold line {line_no} has no retained curation tag")
            pair = (matches_hash, source_hash)
            if pair in by_pair:
                raise ValueError(f"duplicate exact gold pair {pair}")
            normalized["matches_hash"] = matches_hash
            normalized["source_hash"] = source_hash
            normalized["pair_gold_label"] = (
                "positive" if tag == "correct" else "noncorrect"
            )
            _assert_hashes_are_strings(normalized)
            by_pair[pair] = normalized
            rows.append(normalized)

            statement = normalized.get("statement")
            if not isinstance(statement, dict):
                raise ValueError(f"gold line {line_no} has no statement object")
            if statement.get("matches_hash") != matches_hash:
                raise ValueError(
                    f"gold line {line_no} statement hash does not match its pair"
                )
            prior = statements.get(matches_hash)
            if prior is not None and _canonical_json_bytes(prior) != _canonical_json_bytes(
                statement
            ):
                raise ValueError(
                    f"gold has inconsistent statement payloads for {matches_hash}"
                )
            statements[matches_hash] = copy.deepcopy(statement)

    if not rows:
        raise ValueError("gold input contains no rows")

    meta = None
    if gold_meta_path is not None:
        meta = _load_json(gold_meta_path, description="gold metadata")
        artifact = meta.get("artifact")
        if isinstance(artifact, dict) and artifact.get("sha256"):
            actual = _sha256_file(gold_path)
            if artifact["sha256"] != actual:
                raise ValueError(
                    f"gold SHA-256 {actual} does not match metadata {artifact['sha256']}"
                )
        meta_pairs = (meta.get("counts") or {}).get("unique_pairs")
        if meta_pairs is not None and meta_pairs != len(by_pair):
            raise ValueError(
                f"gold metadata claims {meta_pairs} unique pairs; loaded {len(by_pair)}"
            )
    return rows, by_pair, statements, meta


def _validate_pair_index_row(row: dict[str, Any], *, line_no: int) -> tuple[str, str]:
    matches_hash = _hash_string(
        row.get("matches_hash"), field_name=f"pair index line {line_no} matches_hash"
    )
    source_hash = _hash_string(
        row.get("source_hash"), field_name=f"pair index line {line_no} source_hash"
    )
    if row.get("matches_hash") != matches_hash or row.get("source_hash") != source_hash:
        raise ValueError(f"pair index line {line_no} hashes must already be strings")
    source_api = row.get("source_api")
    if not isinstance(source_api, str) or not source_api:
        raise ValueError(f"pair index line {line_no} has no source_api")
    entry_ids = row.get("entry_ids")
    entry_count = row.get("entry_count")
    variants = row.get("payload_variants")
    if (
        not isinstance(entry_ids, list)
        or not all(isinstance(entry_id, str) for entry_id in entry_ids)
        or isinstance(entry_count, bool)
        or not isinstance(entry_count, int)
        or entry_count != len(entry_ids)
        or entry_count < 1
    ):
        raise ValueError(f"pair index line {line_no} has invalid entry multiplicity")
    if not isinstance(variants, list) or row.get("payload_variant_count") != len(
        variants
    ):
        raise ValueError(f"pair index line {line_no} has invalid payload variants")
    if sum(variant.get("entry_count", -1) for variant in variants) != entry_count:
        raise ValueError(
            f"pair index line {line_no} payload variants do not reconcile to entries"
        )
    for variant in variants:
        raw_payloads = variant.get("raw_payloads")
        if not isinstance(raw_payloads, list) or sum(
            raw.get("entry_count", -1) for raw in raw_payloads
        ) != variant.get("entry_count"):
            raise ValueError(
                f"pair index line {line_no} raw payload variants do not reconcile"
            )
    return matches_hash, source_hash


def _validate_expected_counts(
    counts: Mapping[str, int],
    expected_counts: Mapping[str, int] | None,
) -> None:
    if expected_counts is None:
        return
    missing = set(expected_counts) - counts.keys()
    if missing:
        raise ValueError(
            "analysis did not produce expected count fields: "
            + ", ".join(sorted(missing))
        )
    differences = {
        key: {"actual": counts[key], "expected": expected}
        for key, expected in expected_counts.items()
        if counts[key] != expected
    }
    if differences:
        raise ValueError(
            "representative-lane count contract failed: "
            + json.dumps(differences, sort_keys=True)
        )


def analyze_representative_lane(
    *,
    gold_path: Path | str = DEFAULT_GOLD,
    scan_work_dir: Path | str = DEFAULT_SCAN_WORK,
    gold_meta_path: Path | str | None = DEFAULT_GOLD_META,
    expected_counts: Mapping[str, int] | None = REAL_EXPECTED_COUNTS,
) -> LaneAnalysis:
    """Validate immutable inputs and compute exact E0 statement states."""
    gold_path = Path(gold_path)
    scan_work_dir = Path(scan_work_dir)
    if gold_meta_path is not None:
        gold_meta_path = Path(gold_meta_path)

    checkpoint_path = scan_work_dir / "checkpoint.json"
    pair_index_path = scan_work_dir / "pair_index.jsonl"
    evidence_entries_path = scan_work_dir / "evidence_entries.jsonl"
    scan_summary_path = scan_work_dir / "scan_summary.json"
    for required in (checkpoint_path, pair_index_path, evidence_entries_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing completed scan artifact: {required}")

    checkpoint = _load_json(checkpoint_path, description="scan checkpoint")
    if checkpoint.get("schema_version") != 1 or checkpoint.get("scan_complete") is not True:
        raise ValueError("scan checkpoint is not schema-1 complete")
    spool_bytes = checkpoint.get("spool_bytes")
    entries_written = checkpoint.get("entries_written")
    if (
        isinstance(spool_bytes, bool)
        or not isinstance(spool_bytes, int)
        or spool_bytes < 1
        or evidence_entries_path.stat().st_size != spool_bytes
    ):
        raise ValueError(
            "completed checkpoint spool size does not exactly match evidence_entries.jsonl"
        )
    if isinstance(entries_written, bool) or not isinstance(entries_written, int):
        raise ValueError("checkpoint entries_written is invalid")

    scan_summary = None
    if scan_summary_path.is_file():
        scan_summary = _load_json(scan_summary_path, description="scan summary")
        if scan_summary.get("status") != "scan_complete_pending_enrichment":
            raise ValueError("scan summary does not describe a complete frozen scan")
        if scan_summary.get("scan_fingerprint") != checkpoint.get("scan_fingerprint"):
            raise ValueError("scan summary/checkpoint fingerprints differ")

    gold_rows, gold_by_pair, statements, gold_meta = _load_gold(
        gold_path, gold_meta_path
    )
    stats = {matches_hash: StatementStats(matches_hash) for matches_hash in statements}
    found_gold_pairs: set[tuple[str, str]] = set()
    seen_entry_ids: set[str] = set()
    pair_source_counts: Counter[str] = Counter()
    entry_source_counts: Counter[str] = Counter()
    previous_pair: tuple[str, str] | None = None
    unique_pairs = 0
    raw_entries = 0

    with pair_index_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid pair index JSON line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"pair index line {line_no} is not an object")
            pair = _validate_pair_index_row(row, line_no=line_no)
            if previous_pair is not None and _pair_sort_key(pair) <= _pair_sort_key(
                previous_pair
            ):
                raise ValueError("pair index must be strictly sorted and unique")
            previous_pair = pair
            matches_hash, _ = pair
            if matches_hash not in stats:
                raise ValueError(
                    f"pair index includes non-gold target statement {matches_hash}"
                )
            duplicate_ids = seen_entry_ids.intersection(row["entry_ids"])
            if duplicate_ids:
                raise ValueError(
                    f"entry ids occur in multiple exact pairs: {sorted(duplicate_ids)[:5]}"
                )
            seen_entry_ids.update(row["entry_ids"])

            statement_stats = stats[matches_hash]
            statement_stats.unique_pairs += 1
            statement_stats.raw_entries += row["entry_count"]
            statement_stats.pair_source_api_counts[row["source_api"]] += 1
            statement_stats.entry_source_api_counts[row["source_api"]] += row[
                "entry_count"
            ]
            pair_source_counts[row["source_api"]] += 1
            entry_source_counts[row["source_api"]] += row["entry_count"]
            unique_pairs += 1
            raw_entries += row["entry_count"]

            curation = gold_by_pair.get(pair)
            if curation is None:
                statement_stats.unreviewed_pairs += 1
            else:
                found_gold_pairs.add(pair)
                statement_stats.reviewed_pairs += 1
                statement_stats.reviewed.append(curation)
                if curation["tag"] == "correct":
                    statement_stats.reviewed_positive_pairs += 1
                else:
                    statement_stats.reviewed_noncorrect_pairs += 1
                if statement_stats.unique_pairs == 1:
                    statement_stats.singleton_entry_count = row["entry_count"]
                    statement_stats.singleton_payload_variant_count = row[
                        "payload_variant_count"
                    ]

    missing_gold = set(gold_by_pair) - found_gold_pairs
    if missing_gold:
        raise ValueError(
            "completed pair index is missing exact gold pairs: "
            + ", ".join(
                f"({mh}, {sh})"
                for mh, sh in sorted(missing_gold, key=_pair_sort_key)[:20]
            )
        )
    if raw_entries != entries_written or len(seen_entry_ids) != entries_written:
        raise ValueError(
            f"pair index accounts for {raw_entries} entries/{len(seen_entry_ids)} ids; "
            f"checkpoint records {entries_written}"
        )

    label_counts: Counter[str] = Counter()
    queued_pairs = 0
    other_unreviewed = 0
    negative_singletons = 0
    for statement_stats in stats.values():
        if statement_stats.reviewed_positive_pairs:
            statement_stats.e0_label = "positive"
        elif (
            statement_stats.reviewed_pairs == statement_stats.unique_pairs
            and statement_stats.reviewed_noncorrect_pairs
            == statement_stats.unique_pairs
        ):
            statement_stats.e0_label = "negative"
        else:
            statement_stats.e0_label = "unresolved"
        label_counts[statement_stats.e0_label] += 1
        if statement_stats.e0_label == "unresolved":
            statement_stats.queued_pairs = statement_stats.unreviewed_pairs
            queued_pairs += statement_stats.unreviewed_pairs
        else:
            other_unreviewed += statement_stats.unreviewed_pairs
        if (
            statement_stats.e0_label == "negative"
            and statement_stats.unique_pairs == 1
            and statement_stats.singleton_entry_count == 1
            and statement_stats.singleton_payload_variant_count == 1
        ):
            negative_singletons += 1

    gold_labels = Counter(row["pair_gold_label"] for row in gold_rows)
    counts = {
        "gold_exact_pairs": len(gold_by_pair),
        "gold_positive_pairs": gold_labels["positive"],
        "gold_noncorrect_pairs": gold_labels["noncorrect"],
        "statements": len(stats),
        "positive_statements": label_counts["positive"],
        "negative_statements": label_counts["negative"],
        "unresolved_statements": label_counts["unresolved"],
        "negative_singleton_statements": negative_singletons,
        "raw_evidence_entries": raw_entries,
        "unique_evidence_pairs": unique_pairs,
        "queued_unreviewed_pairs": queued_pairs,
        "other_unreviewed_pairs_retained_in_ledger": other_unreviewed,
    }
    if scan_summary is not None:
        summary_counts = scan_summary.get("counts") or {}
        for summary_key, actual in (
            ("matched_evidence_entries", raw_entries),
            ("unique_exact_pairs", unique_pairs),
            ("gold_exact_pairs_covered", len(found_gold_pairs)),
        ):
            if summary_counts.get(summary_key) != actual:
                raise ValueError(
                    f"scan summary {summary_key}={summary_counts.get(summary_key)}; "
                    f"pair index establishes {actual}"
                )
    _validate_expected_counts(counts, expected_counts)

    return LaneAnalysis(
        gold_rows=gold_rows,
        gold_by_pair=gold_by_pair,
        statements=statements,
        statement_stats=stats,
        counts=counts,
        checkpoint=checkpoint,
        scan_summary=scan_summary,
        input_files={
            "gold": gold_path,
            "checkpoint": checkpoint_path,
            "pair_index": pair_index_path,
            "evidence_entries": evidence_entries_path,
            **({"gold_meta": gold_meta_path} if gold_meta_path is not None else {}),
            **({"scan_summary": scan_summary_path} if scan_summary is not None else {}),
        },
        gold_meta=gold_meta,
        pair_source_api_counts=_counter_json(pair_source_counts),
        entry_source_api_counts=_counter_json(entry_source_counts),
    )


def _forbidden_queue_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if "hash" in normalized:
        return True
    return any(stem in normalized for stem in FORBIDDEN_QUEUE_KEY_STEMS)


def _blind_queue_value(value: Any) -> Any:
    """Recursively remove outcome, curator, prediction, and linkage fields."""
    if isinstance(value, dict):
        return {
            key: _blind_queue_value(child)
            for key, child in value.items()
            if isinstance(key, str) and not _forbidden_queue_key(key)
        }
    if isinstance(value, list):
        return [_blind_queue_value(child) for child in value]
    return copy.deepcopy(value)


def _forbidden_queue_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if _forbidden_queue_key(key):
                paths.append(path)
            paths.extend(_forbidden_queue_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_forbidden_queue_paths(child, f"{prefix}[{index}]"))
    return paths


@contextmanager
def _open_compressed_jsonl(path: Path, compression: str) -> Iterator[BinaryIO]:
    raw = path.open("wb")
    compressed: BinaryIO | None = None
    try:
        if compression == "none":
            compressed = raw
        elif compression == "gzip":
            compressed = gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0
            )
        elif compression == "zstd":
            try:
                import zstandard
            except ImportError as exc:  # pragma: no cover - resolved before writing
                raise RuntimeError("zstandard disappeared during harness write") from exc
            compressed = zstandard.ZstdCompressor(level=10).stream_writer(
                raw, closefd=False
            )
        else:  # pragma: no cover - internal invariant
            raise AssertionError(compression)
        yield compressed
        if compressed is not raw:
            compressed.close()
            compressed = None
        raw.flush()
        os.fsync(raw.fileno())
    finally:
        if compressed is not None and compressed is not raw:
            compressed.close()
        raw.close()


def _index_spool_offsets(
    spool_path: Path,
    *,
    expected_bytes: int,
    expected_entries: int,
) -> tuple[dict[str, tuple[int, int]], str]:
    offsets: dict[str, tuple[int, int]] = {}
    digest = hashlib.sha256()
    position = 0
    with spool_path.open("rb") as fh:
        while position < expected_bytes:
            line = fh.readline()
            if not line:
                raise ValueError("evidence spool ended before checkpoint byte boundary")
            next_position = position + len(line)
            if next_position > expected_bytes:
                raise ValueError("checkpoint byte boundary splits an evidence JSONL row")
            digest.update(line)
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evidence spool row at byte {position}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("entry_id"), str):
                raise ValueError(f"invalid evidence spool record at byte {position}")
            entry_id = row["entry_id"]
            if entry_id in offsets:
                raise ValueError(f"duplicate evidence spool entry_id {entry_id}")
            offsets[entry_id] = (position, len(line))
            position = next_position
        if fh.read(1):
            raise ValueError("evidence spool has bytes beyond completed checkpoint boundary")
    if len(offsets) != expected_entries:
        raise ValueError(
            f"evidence spool has {len(offsets)} rows; checkpoint has {expected_entries}"
        )
    return offsets, digest.hexdigest()


def _read_spool_entry(
    spool: BinaryIO,
    offsets: Mapping[str, tuple[int, int]],
    entry_id: str,
) -> dict[str, Any]:
    try:
        offset, length = offsets[entry_id]
    except KeyError as exc:
        raise ValueError(f"pair index references absent spool entry {entry_id}") from exc
    spool.seek(offset)
    raw = spool.read(length)
    if len(raw) != length:
        raise ValueError(f"could not read complete spool entry {entry_id}")
    row = json.loads(raw)
    if row.get("entry_id") != entry_id:
        raise ValueError(f"spool offset for {entry_id} resolves to another entry")
    return row


def _validate_spool_entry_payload(
    entry: dict[str, Any],
    *,
    matches_hash: str,
    source_hash: str,
    source_api: str,
) -> None:
    if (
        entry.get("matches_hash") != matches_hash
        or entry.get("source_hash") != source_hash
        or entry.get("source_api") != source_api
    ):
        raise ValueError(
            f"spool entry {entry.get('entry_id')} disagrees with pair index"
        )
    raw_json = entry.get("raw_evidence_json")
    evidence = entry.get("evidence")
    if not isinstance(raw_json, str) or not isinstance(evidence, dict):
        raise ValueError(f"spool entry {entry.get('entry_id')} lacks raw/normalized payload")
    raw_bytes = raw_json.encode("utf-8")
    if entry.get("raw_payload_bytes") != len(raw_bytes) or entry.get(
        "raw_payload_sha256"
    ) != hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError(
            f"spool entry {entry.get('entry_id')} raw payload provenance fails"
        )
    try:
        normalized_raw = _stringify_hashes(json.loads(raw_json))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"spool entry {entry.get('entry_id')} raw evidence JSON is invalid"
        ) from exc
    if normalized_raw.get("source_hash") is None:
        normalized_raw["source_hash"] = source_hash
    if _canonical_json_bytes(normalized_raw) != _canonical_json_bytes(evidence):
        raise ValueError(
            f"spool entry {entry.get('entry_id')} raw/normalized evidence differs"
        )
    if entry.get("payload_sha256") != _sha256_bytes(
        _canonical_json_bytes(evidence)
    ):
        raise ValueError(
            f"spool entry {entry.get('entry_id')} normalized payload digest fails"
        )
    retracted = entry.get("cogex_retracted")
    retracted_raw = entry.get("cogex_retracted_raw")
    expected_raw = "true" if retracted is True else "false" if retracted is False else None
    if expected_raw is None or retracted_raw != expected_raw:
        raise ValueError(
            f"spool entry {entry.get('entry_id')} has invalid retracted provenance"
        )
    cogex_fields = entry.get("cogex_fields")
    if (
        not isinstance(cogex_fields, dict)
        or cogex_fields.get("retracted:boolean") != retracted_raw
    ):
        raise ValueError(
            f"spool entry {entry.get('entry_id')} loses raw retracted cell"
        )
    _assert_hashes_are_strings(entry)


def _queue_variants(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"entry {entry.get('entry_id')} has no evidence object")
        blinded_evidence = _blind_queue_value(evidence)
        visible_digest = _sha256_bytes(_canonical_json_bytes(blinded_evidence))
        group = grouped.get(visible_digest)
        if group is None:
            group = {
                "raw_entry_multiplicity": 0,
                "evidence": blinded_evidence,
            }
            grouped[visible_digest] = group
        group["raw_entry_multiplicity"] += 1
    return list(grouped.values())


def _new_opaque_id() -> str:
    return "q_" + secrets.token_urlsafe(18)


def _artifact_descriptor(path: Path, *, rows: int, compression: str) -> dict[str, Any]:
    return {
        "path": path.name,
        "format": "jsonl",
        "compression": compression,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _publish_exists_error(destination: Path) -> FileExistsError:
    return FileExistsError(
        errno.EEXIST,
        f"refusing to clobber existing output path: {destination}",
        str(destination),
    )


def _atomic_publish_directory_noreplace(stage: Path, destination: Path) -> None:
    """Atomically publish ``stage`` without replacing any destination entry."""
    if os.path.lexists(destination):
        raise _publish_exists_error(destination)

    source_bytes = os.fsencode(stage)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)

    if sys.platform == "darwin":
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            0x00000001,
        )
    elif os.name == "nt":
        # Windows rename refuses an existing destination.
        os.rename(stage, destination)
        return
    else:
        raise RuntimeError(
            "this platform has no supported atomic no-replace directory rename"
        )

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise _publish_exists_error(destination)
    if error_number == errno.ENOSYS:
        raise RuntimeError(
            "the kernel does not support atomic no-replace directory rename"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def build_representative_adjudication_harness(
    *,
    output_dir: Path | str,
    gold_path: Path | str = DEFAULT_GOLD,
    scan_work_dir: Path | str = DEFAULT_SCAN_WORK,
    gold_meta_path: Path | str | None = DEFAULT_GOLD_META,
    compression: str = "auto",
    expected_counts: Mapping[str, int] | None = REAL_EXPECTED_COUNTS,
    opaque_id_factory: Callable[[], str] | None = None,
) -> HarnessResult:
    """Build a no-clobber representative-lane artifact bundle."""
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise _publish_exists_error(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    selected_compression, fallback_reason = _resolve_compression(compression)
    analysis = analyze_representative_lane(
        gold_path=gold_path,
        scan_work_dir=scan_work_dir,
        gold_meta_path=gold_meta_path,
        expected_counts=expected_counts,
    )
    pair_index_path = analysis.input_files["pair_index"]
    spool_path = analysis.input_files["evidence_entries"]
    offsets, spool_sha256 = _index_spool_offsets(
        spool_path,
        expected_bytes=analysis.checkpoint["spool_bytes"],
        expected_entries=analysis.checkpoint["entries_written"],
    )

    stage_dir = output_dir.parent / f".{output_dir.name}.stage-{uuid.uuid4().hex}"
    stage_dir.mkdir(exist_ok=False)
    suffix = _compression_suffix(selected_compression)
    output_paths = {
        "statement_gold": stage_dir / f"statement_gold{suffix}",
        "evidence_pair_ledger": stage_dir / f"evidence_pair_ledger{suffix}",
        "adjudication_queue": stage_dir / f"adjudication_queue{suffix}",
        "queue_mapping": stage_dir / f"queue_mapping{suffix}",
    }
    artifact_rows = Counter()
    queue_ids: set[str] = set()
    id_factory = opaque_id_factory or _new_opaque_id
    reviewed_pairs_written = 0
    unreviewed_pairs_written = 0
    queued_pairs_written = 0
    other_unreviewed_written = 0
    raw_entries_written = 0
    manifest: dict[str, Any] | None = None

    try:
        with ExitStack() as stack:
            writers = {
                name: stack.enter_context(
                    _open_compressed_jsonl(path, selected_compression)
                )
                for name, path in output_paths.items()
            }

            for matches_hash in sorted(analysis.statements, key=int):
                stats = analysis.statement_stats[matches_hash]
                statement_record = {
                    "matches_hash": matches_hash,
                    "e0_label": stats.e0_label,
                    "e0_rule": (
                        "positive if any exact reviewed pair is positive; negative if "
                        "all unique pairs are reviewed noncorrect; otherwise unresolved"
                    ),
                    "unique_evidence_pairs": stats.unique_pairs,
                    "raw_evidence_entries": stats.raw_entries,
                    "reviewed_exact_pairs": stats.reviewed_pairs,
                    "reviewed_positive_pairs": stats.reviewed_positive_pairs,
                    "reviewed_noncorrect_pairs": stats.reviewed_noncorrect_pairs,
                    "unreviewed_pairs": stats.unreviewed_pairs,
                    "adjudication_pairs_needed": stats.queued_pairs,
                    "pair_source_api_counts": _counter_json(
                        stats.pair_source_api_counts
                    ),
                    "entry_source_api_counts": _counter_json(
                        stats.entry_source_api_counts
                    ),
                    "statement": analysis.statements[matches_hash],
                    "exact_pair_curations": sorted(
                        stats.reviewed,
                        key=lambda row: (
                            row.get("curation_id")
                            if isinstance(row.get("curation_id"), int)
                            else -1
                        ),
                    ),
                }
                _assert_hashes_are_strings(statement_record)
                writers["statement_gold"].write(_json_line(statement_record))
                artifact_rows["statement_gold"] += 1

            previous_pair: tuple[str, str] | None = None
            with pair_index_path.open(encoding="utf-8") as pair_fh, spool_path.open(
                "rb"
            ) as spool_fh:
                for line_no, line in enumerate(pair_fh, start=1):
                    if not line.strip():
                        continue
                    pair_row = json.loads(line)
                    pair = _validate_pair_index_row(pair_row, line_no=line_no)
                    if previous_pair is not None and _pair_sort_key(
                        pair
                    ) <= _pair_sort_key(previous_pair):
                        raise ValueError("pair index changed after analysis")
                    previous_pair = pair
                    matches_hash, source_hash = pair
                    entries = [
                        _read_spool_entry(spool_fh, offsets, entry_id)
                        for entry_id in pair_row["entry_ids"]
                    ]
                    for entry in entries:
                        _validate_spool_entry_payload(
                            entry,
                            matches_hash=matches_hash,
                            source_hash=source_hash,
                            source_api=pair_row["source_api"],
                        )
                    raw_entries_written += len(entries)
                    observed_payloads = Counter(
                        entry.get("payload_sha256") for entry in entries
                    )
                    indexed_payloads = Counter(
                        {
                            variant["payload_sha256"]: variant["entry_count"]
                            for variant in pair_row["payload_variants"]
                        }
                    )
                    if observed_payloads != indexed_payloads:
                        raise ValueError(f"payload variants disagree for exact pair {pair}")
                    observed_raw_payloads = {
                        payload_sha256: Counter(
                            entry["raw_payload_sha256"]
                            for entry in entries
                            if entry["payload_sha256"] == payload_sha256
                        )
                        for payload_sha256 in observed_payloads
                    }
                    indexed_raw_payloads = {
                        variant["payload_sha256"]: Counter(
                            {
                                raw["raw_payload_sha256"]: raw["entry_count"]
                                for raw in variant["raw_payloads"]
                            }
                        )
                        for variant in pair_row["payload_variants"]
                    }
                    if observed_raw_payloads != indexed_raw_payloads:
                        raise ValueError(
                            f"raw payload variants disagree for exact pair {pair}"
                        )

                    curation = analysis.gold_by_pair.get(pair)
                    reviewed = curation is not None
                    stats = analysis.statement_stats[matches_hash]
                    needed = stats.e0_label == "unresolved" and not reviewed
                    if reviewed:
                        reviewed_pairs_written += 1
                    else:
                        unreviewed_pairs_written += 1
                        if needed:
                            queued_pairs_written += 1
                        else:
                            other_unreviewed_written += 1

                    raw_payload_variant_count = len(
                        {
                            entry.get("raw_payload_sha256")
                            for entry in entries
                        }
                    )
                    ledger_record = {
                        "matches_hash": matches_hash,
                        "source_hash": source_hash,
                        "source_api": pair_row["source_api"],
                        "reviewed": reviewed,
                        "pair_gold_label": (
                            curation["pair_gold_label"] if curation is not None else None
                        ),
                        "curation": curation,
                        "statement_e0_label": stats.e0_label,
                        "needed_for_e0_adjudication": needed,
                        "raw_entry_count": pair_row["entry_count"],
                        "normalized_payload_variant_count": pair_row[
                            "payload_variant_count"
                        ],
                        "raw_payload_variant_count": raw_payload_variant_count,
                        "payload_variants": pair_row["payload_variants"],
                        "entries": entries,
                    }
                    _assert_hashes_are_strings(ledger_record)
                    writers["evidence_pair_ledger"].write(_json_line(ledger_record))
                    artifact_rows["evidence_pair_ledger"] += 1

                    if needed:
                        queue_id = id_factory()
                        if (
                            not isinstance(queue_id, str)
                            or not OPAQUE_ID_RE.fullmatch(queue_id)
                            or queue_id in queue_ids
                        ):
                            raise ValueError(
                                "opaque queue id factory returned an invalid or duplicate id"
                            )
                        queue_ids.add(queue_id)
                        queue_variants = _queue_variants(entries)
                        queue_record = {
                            "queue_id": queue_id,
                            "statement": _blind_queue_value(
                                analysis.statements[matches_hash]
                            ),
                            "source_api": pair_row["source_api"],
                            "raw_entry_multiplicity": pair_row["entry_count"],
                            "payload_variant_count": len(queue_variants),
                            "evidence_variants": queue_variants,
                        }
                        forbidden_paths = _forbidden_queue_paths(queue_record)
                        if forbidden_paths:
                            raise AssertionError(
                                "blind queue retained forbidden fields: "
                                + ", ".join(forbidden_paths[:20])
                            )
                        mapping_record = {
                            "queue_id": queue_id,
                            "matches_hash": matches_hash,
                            "source_hash": source_hash,
                            "source_api": pair_row["source_api"],
                            "ledger_pair_ordinal": artifact_rows[
                                "evidence_pair_ledger"
                            ],
                        }
                        _assert_hashes_are_strings(mapping_record)
                        writers["adjudication_queue"].write(
                            _json_line(queue_record)
                        )
                        writers["queue_mapping"].write(_json_line(mapping_record))
                        artifact_rows["adjudication_queue"] += 1
                        artifact_rows["queue_mapping"] += 1

        expected_output_counts = {
            "statement_gold": analysis.counts["statements"],
            "evidence_pair_ledger": analysis.counts["unique_evidence_pairs"],
            "adjudication_queue": analysis.counts["queued_unreviewed_pairs"],
            "queue_mapping": analysis.counts["queued_unreviewed_pairs"],
        }
        if dict(artifact_rows) != expected_output_counts:
            raise AssertionError(
                f"output row reconciliation failed: {dict(artifact_rows)} != "
                f"{expected_output_counts}"
            )
        if reviewed_pairs_written != analysis.counts["gold_exact_pairs"]:
            raise AssertionError("ledger reviewed-pair count does not match gold")
        if raw_entries_written != analysis.counts["raw_evidence_entries"]:
            raise AssertionError("ledger raw-entry multiplicity does not match checkpoint")
        if queued_pairs_written != analysis.counts["queued_unreviewed_pairs"]:
            raise AssertionError("queue count does not match unresolved unreviewed pairs")
        if (
            other_unreviewed_written
            != analysis.counts["other_unreviewed_pairs_retained_in_ledger"]
        ):
            raise AssertionError("nonqueued unreviewed ledger count does not reconcile")

        files = {
            name: _artifact_descriptor(
                path,
                rows=artifact_rows[name],
                compression=selected_compression,
            )
            for name, path in output_paths.items()
        }
        gold_pair_bytes = b"".join(
            f"{matches_hash}\t{source_hash}\n".encode("ascii")
            for matches_hash, source_hash in sorted(
                analysis.gold_by_pair, key=_pair_sort_key
            )
        )
        input_identities = {
            name: _path_identity(
                path,
                sha256=(spool_sha256 if name == "evidence_entries" else None),
            )
            for name, path in analysis.input_files.items()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "representative_indra_e0_adjudication_harness",
            "status": "complete",
            "created_at": _utc_now(),
            "inputs": input_identities,
            "compression": {
                "requested": compression,
                "selected": selected_compression,
                "fallback_reason": fallback_reason,
            },
            "rules": {
                "pair_join": "exact (matches_hash, source_hash) only",
                "pair_gold": (
                    "retained tag == 'correct' => positive; every other retained "
                    "tag => noncorrect"
                ),
                "statement_e0": (
                    "any reviewed positive pair => positive; all unique pairs reviewed "
                    "noncorrect => negative; otherwise unresolved"
                ),
                "queue_scope": (
                    "unreviewed unique pairs on unresolved statements only"
                ),
            },
            "counts": analysis.counts,
            "source_accounting": {
                "raw_entry_source_api_counts": analysis.entry_source_api_counts,
                "unique_pair_source_api_counts": analysis.pair_source_api_counts,
            },
            "digests": {
                "algorithm": "sha256",
                "gold_exact_pairs_sha256": _sha256_bytes(gold_pair_bytes),
            },
            "queue_blinding": {
                "opaque_ids": True,
                "mapping_is_separate": True,
                "recursively_removed_key_stems": list(FORBIDDEN_QUEUE_KEY_STEMS),
                "all_hash_key_fields_removed": True,
                "raw_evidence_json_excluded": True,
                "variants_regrouped_after_blinding": True,
                "prediction_outputs_used": False,
            },
            "selection_randomness_caveat": {
                "historical_completed_subset_is_provably_srs": False,
                "reason": (
                    "the historical viewer retained no draw/skip log and retried "
                    "unmaterializable or textless rows; reservoir membership is "
                    "auditable, but the completed curation subset is not proven to "
                    "be a simple random sample"
                ),
            },
            "reconciliation": {
                "all_gold_pairs_joined_exactly_once": reviewed_pairs_written
                == analysis.counts["gold_exact_pairs"],
                "ledger_contains_every_unique_pair": artifact_rows[
                    "evidence_pair_ledger"
                ]
                == analysis.counts["unique_evidence_pairs"],
                "ledger_preserves_all_raw_entry_multiplicity": sum(
                    analysis.entry_source_api_counts.values()
                )
                == raw_entries_written
                == analysis.counts["raw_evidence_entries"],
                "queue_contains_only_needed_pairs": queued_pairs_written
                == analysis.counts["queued_unreviewed_pairs"],
                "queue_and_mapping_are_one_to_one": artifact_rows[
                    "adjudication_queue"
                ]
                == artifact_rows["queue_mapping"]
                == len(queue_ids),
                "all_other_unreviewed_pairs_remain_in_ledger": other_unreviewed_written
                == analysis.counts[
                    "other_unreviewed_pairs_retained_in_ledger"
                ],
                "no_paid_or_model_inference": True,
            },
            "files": files,
        }
        if not all(manifest["reconciliation"].values()):
            raise AssertionError("manifest reconciliation contains a false claim")

        # Manifest is deliberately the last file created in the unpublished stage.
        manifest_path = stage_dir / "manifest.json"
        with manifest_path.open("wb") as manifest_fh:
            manifest_fh.write(_json_line(manifest))
            manifest_fh.flush()
            os.fsync(manifest_fh.fileno())
        _fsync_directory(stage_dir)

        # The complete staged directory becomes visible in one no-replace
        # rename.  No failure can leave a partial destination that blocks retry.
        _atomic_publish_directory_noreplace(stage_dir, output_dir)
        _fsync_directory(output_dir.parent)
        return HarnessResult(output_dir=output_dir, manifest=manifest)
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--scan-work-dir", type=Path, default=DEFAULT_SCAN_WORK)
    parser.add_argument("--gold-meta", type=Path, default=DEFAULT_GOLD_META)
    parser.add_argument(
        "--no-gold-meta",
        action="store_true",
        help="do not load or validate a gold metadata sidecar",
    )
    parser.add_argument(
        "--compression",
        choices=("auto", "zstd", "gzip", "none"),
        default="auto",
    )
    parser.add_argument(
        "--allow-nonrepresentative-counts",
        action="store_true",
        help="disable the pinned real 403-lane count assertions for a fixture",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="validate inputs and print counts without creating output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    expected = None if args.allow_nonrepresentative_counts else REAL_EXPECTED_COUNTS
    gold_meta = None if args.no_gold_meta else args.gold_meta
    if args.analyze_only:
        analysis = analyze_representative_lane(
            gold_path=args.gold,
            scan_work_dir=args.scan_work_dir,
            gold_meta_path=gold_meta,
            expected_counts=expected,
        )
        print(json.dumps({"status": "validated", "counts": analysis.counts}, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --analyze-only is used")
    result = build_representative_adjudication_harness(
        output_dir=args.output_dir,
        gold_path=args.gold,
        scan_work_dir=args.scan_work_dir,
        gold_meta_path=gold_meta,
        compression=args.compression,
        expected_counts=expected,
    )
    print(
        json.dumps(
            {
                "status": "created",
                "output_dir": str(result.output_dir.resolve()),
                "counts": result.manifest["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
