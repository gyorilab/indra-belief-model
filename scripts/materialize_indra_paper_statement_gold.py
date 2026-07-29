#!/usr/bin/env python3
"""Materialize auditable statement gold for the frozen INDRA paper lane.

The released paper dataset labels a statement positive when any reviewed
evidence pair is positive, and negative otherwise even when other corpus
evidence is unreviewed.  That historical policy is retained in a separate
``paper_replication_policy`` field.  The primary adjudicated label produced by
this harness is stricter:

* positive: at least one exact statement/evidence pair is reviewed positive;
* negative: every distinct evidence hash in the frozen canonical statement is
  reviewed and all are negative;
* unresolved: no reviewed positive exists and at least one evidence item is
  unreviewed or cannot be identified by a source hash.

Thus an unreviewed evidence item is never silently converted to a negative.
The frozen protocol's 1,689-row order and reader-only membership are preserved
and cross-checked against the canonical corpus source counts.

Example::

    .venv/bin/python scripts/materialize_indra_paper_statement_gold.py \
      --corpus data/benchmark/indra_benchmark_corpus.json.gz \
      --curations data/benchmark/indra_assembly_curations.json \
      --protocol-manifest \
        data/results/indra_paper_protocol_20260717/paper_protocol_manifest.json \
      --eligible-statements \
        data/results/indra_paper_protocol_20260717/paper_eligible_statements.jsonl \
      --output-dir data/results/indra_paper_statement_gold_20260717
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TextIO


PAPER_POSITIVE_TAGS = frozenset(("correct", "hypothesis", "act_vs_amt"))
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

STATEMENT_GOLD_FILENAME = "paper_statement_gold.jsonl"
EVIDENCE_ADJUDICATION_FILENAME = "paper_evidence_adjudication.jsonl"
ADJUDICATION_QUEUE_FILENAME = "paper_adjudication_queue.jsonl"
MANIFEST_FILENAME = "paper_statement_gold_manifest.json"

QUEUE_FORBIDDEN_KEYS = frozenset(
    (
        "belief",
        "correct",
        "curator",
        "curator_note",
        "disagreement",
        "matches_hash",
        "model_prediction",
        "pa_hash",
        "source_hash",
        "tag",
    )
)


class AuditError(ValueError):
    """Raised when an input identity or paper-lane invariant fails."""


@dataclass(frozen=True)
class FileExpectation:
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CorpusExpectation:
    file: FileExpectation
    rows: int
    evidence_entries: int


@dataclass(frozen=True)
class EligibleExpectation:
    file: FileExpectation
    rows: int
    positive: int
    negative: int
    reader_rows: int
    reader_positive: int
    reader_negative: int


@dataclass(frozen=True)
class ResultExpectation:
    target_evidence_entries: int
    target_distinct_evidence_pairs: int
    unidentifiable_evidence_entries: int
    reviewed_evidence_entries: int
    unreviewed_evidence_entries: int
    eligible_curation_rows: int
    reviewed_evidence_pairs: int
    reviewed_positive_pairs: int
    reviewed_negative_pairs: int
    conflicting_reviewed_pairs: int
    unreviewed_evidence_pairs: int
    evidence_adjudication_rows: int
    complete_evidence_statements: int
    adjudicated_positive: int
    adjudicated_negative: int
    adjudicated_unresolved: int
    unresolved_adjudication_queue_rows: int
    unresolved_adjudication_queue_evidence_entries: int
    unresolved_adjudication_queue_textless_rows: int


@dataclass(frozen=True)
class MaterializationExpectations:
    corpus: CorpusExpectation
    curations: FileExpectation
    curation_rows: int
    protocol_manifest: FileExpectation
    eligible: EligibleExpectation
    results: ResultExpectation


RELEASE_EXPECTATIONS = MaterializationExpectations(
    corpus=CorpusExpectation(
        file=FileExpectation(
            bytes=460_045_058,
            sha256=(
                "bf048f3b485990e6d81ee0f2200ea17efe962268427b4f33af932b4f08a434de"
            ),
        ),
        rows=894_939,
        evidence_entries=2_847_196,
    ),
    curations=FileExpectation(
        bytes=1_687_078,
        sha256=(
            "02ccca87fb4c8386ae49d420cb4c8257cf769bcaba5692a989bf295ba8d40da5"
        ),
    ),
    curation_rows=6_022,
    protocol_manifest=FileExpectation(
        bytes=10_771,
        sha256=(
            "e3d0392b410ba7458663bfeb336954e52836631878d09f8f59ff213feb2725d3"
        ),
    ),
    eligible=EligibleExpectation(
        file=FileExpectation(
            bytes=444_839,
            sha256=(
                "82b572bfe57d23bda65611909e5f13b010dac03fd03437dcccdc3fd1e993ec4d"
            ),
        ),
        rows=1_689,
        positive=1_237,
        negative=452,
        reader_rows=1_676,
        reader_positive=1_236,
        reader_negative=440,
    ),
    results=ResultExpectation(
        target_evidence_entries=34_035,
        target_distinct_evidence_pairs=33_361,
        unidentifiable_evidence_entries=0,
        reviewed_evidence_entries=5_516,
        unreviewed_evidence_entries=28_519,
        eligible_curation_rows=5_688,
        reviewed_evidence_pairs=5_379,
        reviewed_positive_pairs=3_520,
        reviewed_negative_pairs=1_859,
        conflicting_reviewed_pairs=23,
        unreviewed_evidence_pairs=27_982,
        evidence_adjudication_rows=33_361,
        complete_evidence_statements=607,
        adjudicated_positive=1_237,
        adjudicated_negative=341,
        adjudicated_unresolved=111,
        unresolved_adjudication_queue_rows=1_326,
        unresolved_adjudication_queue_evidence_entries=1_341,
        unresolved_adjudication_queue_textless_rows=10,
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)


def _pretty_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _verify_file(path: Path, expected: FileExpectation) -> dict[str, Any]:
    path = Path(path)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as err:
        raise AuditError(f"input does not exist: {path}") from err
    if not resolved.is_file():
        raise AuditError(f"input is not a regular file: {path} -> {resolved}")
    observed = FileExpectation(bytes=resolved.stat().st_size, sha256=_sha256_file(resolved))
    if observed != expected:
        raise AuditError(
            f"frozen input identity mismatch for {path}: "
            f"expected={asdict(expected)}, observed={asdict(observed)}"
        )
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "is_symlink": path.is_symlink(),
        **asdict(observed),
        "verification": "pass",
    }


def _verified_small_payload(
    path: Path, expected: FileExpectation
) -> tuple[bytes, dict[str, Any]]:
    identity = _verify_file(path, expected)
    return Path(identity["resolved_path"]).read_bytes(), identity


def _exact_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise AuditError(f"{field} must be a signed integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
            if stripped != str(int(stripped)):
                raise AuditError(f"{field} is not a canonical signed integer: {value!r}")
            return int(stripped)
    raise AuditError(f"{field} must be a canonical signed integer: {value!r}")


def _binary(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise AuditError(f"{field} must be integer 0 or 1: {value!r}")
    return value


def _iter_json_array(stream: TextIO, *, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Iterate one top-level JSON array with bounded buffering.

    The standard-library decoder is used so no optional streaming parser or
    alternate numeric representation becomes part of the artifact contract.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    def read_more() -> bool:
        nonlocal buffer, eof
        chunk = stream.read(chunk_size)
        if chunk:
            buffer += chunk
            return True
        eof = True
        return False

    def compact() -> None:
        nonlocal buffer, position
        if position:
            buffer = buffer[position:]
            position = 0

    def skip_space() -> None:
        nonlocal position
        while position < len(buffer) and buffer[position].isspace():
            position += 1

    if not read_more():
        raise AuditError("JSON input is empty")
    skip_space()
    while position == len(buffer) and not eof:
        compact()
        read_more()
        skip_space()
    if position >= len(buffer) or buffer[position] != "[":
        raise AuditError("JSON root must be an array")
    position += 1
    expect_value = True
    after_comma = False

    while True:
        skip_space()
        while position == len(buffer) and not eof:
            compact()
            read_more()
            skip_space()
        if position == len(buffer):
            raise AuditError("unterminated JSON array")

        if expect_value:
            if buffer[position] == "]":
                if after_comma:
                    raise AuditError("trailing comma in JSON array")
                position += 1
                break
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as err:
                    if eof:
                        raise AuditError("invalid or truncated JSON array item") from err
                    compact()
                    read_more()
                    continue
                position = end
                yield value
                expect_value = False
                after_comma = False
                break
        else:
            if buffer[position] == ",":
                position += 1
                expect_value = True
                after_comma = True
            elif buffer[position] == "]":
                position += 1
                break
            else:
                raise AuditError("expected ',' or ']' after JSON array item")

        if position > chunk_size:
            compact()

    while True:
        skip_space()
        if position < len(buffer):
            raise AuditError("non-whitespace data follows the JSON array")
        if eof or not read_more():
            break


def _curation_pair_label(rows: Sequence[dict[str, Any]]) -> int:
    return int(all(row["tag"] in PAPER_POSITIVE_TAGS for row in rows))


def _curation_audit_counts(curations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_pair: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    statement_hashes: set[int] = set()
    source_hashes: set[int] = set()
    tags: Counter[str] = Counter()
    curators: set[str] = set()
    for index, row in enumerate(curations):
        if not isinstance(row, dict):
            raise AuditError(f"curation row {index} is not an object")
        try:
            pa_hash = _exact_int(row["pa_hash"], field=f"curation row {index} pa_hash")
            source_hash = _exact_int(
                row["source_hash"], field=f"curation row {index} source_hash"
            )
            tag = row["tag"]
            curator = row["curator"]
        except KeyError as err:
            raise AuditError(f"curation row {index} is missing {err.args[0]!r}") from err
        if not isinstance(tag, str) or not isinstance(curator, str):
            raise AuditError(f"curation row {index} has a non-string tag or curator")
        by_pair[(pa_hash, source_hash)].append(row)
        statement_hashes.add(pa_hash)
        source_hashes.add(source_hash)
        tags[tag] += 1
        curators.add(curator)

    pair_labels = {pair: _curation_pair_label(rows) for pair, rows in by_pair.items()}
    by_statement: defaultdict[int, list[int]] = defaultdict(list)
    for (pa_hash, _), label in pair_labels.items():
        by_statement[pa_hash].append(label)
    statement_labels = {pa: int(any(labels)) for pa, labels in by_statement.items()}
    pair_positive = sum(pair_labels.values())
    statement_positive = sum(statement_labels.values())
    conflicts = sum(
        any(row["tag"] in PAPER_POSITIVE_TAGS for row in rows)
        and not all(row["tag"] in PAPER_POSITIVE_TAGS for row in rows)
        for rows in by_pair.values()
    )
    return {
        "rows": len(curations),
        "unique_statement_hashes": len(statement_hashes),
        "unique_statement_evidence_pairs": len(by_pair),
        "unique_source_hashes": len(source_hashes),
        "unique_curators": len(curators),
        "observed_pair_positive": pair_positive,
        "observed_pair_negative": len(pair_labels) - pair_positive,
        "observed_statement_positive": statement_positive,
        "observed_statement_negative": len(statement_labels) - statement_positive,
        "conflicting_pairs": conflicts,
        "tag_counts": dict(sorted(tags.items())),
    }


def _load_eligible(payload: bytes, expected: EligibleExpectation) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as err:
        raise AuditError("eligible-statements artifact is not UTF-8") from err
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise AuditError(f"eligible-statements line {line_number} is blank")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as err:
            raise AuditError(f"eligible-statements line {line_number} is invalid JSON") from err
        if not isinstance(row, dict):
            raise AuditError(f"eligible-statements line {line_number} is not an object")
        rows.append(row)

    if len(rows) != expected.rows:
        raise AuditError(f"eligible row count mismatch: {len(rows)} != {expected.rows}")
    seen: set[int] = set()
    labels: list[int] = []
    reader_labels: list[int] = []
    reader_positions = [HISTORICAL_ALL_SOURCE_ORDER.index(src) for src in READER_SOURCES]
    for position, row in enumerate(rows):
        pa_hash = _exact_int(row.get("stmt_hash"), field=f"eligible row {position} stmt_hash")
        if pa_hash in seen:
            raise AuditError(f"duplicate eligible statement hash: {pa_hash}")
        seen.add(pa_hash)
        source_row_index = row.get("source_row_index")
        if isinstance(source_row_index, bool) or source_row_index != position:
            raise AuditError(
                f"eligible row {position} does not preserve source_row_index order"
            )
        label = _binary(row.get("correct"), field=f"eligible row {position} correct")
        labels.append(label)
        all_counts = row.get("historical_all_source_counts")
        reader_counts = row.get("reader_source_counts")
        if (
            not isinstance(all_counts, list)
            or len(all_counts) != len(HISTORICAL_ALL_SOURCE_ORDER)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in all_counts)
        ):
            raise AuditError(f"eligible row {position} has invalid all-source counts")
        if (
            not isinstance(reader_counts, list)
            or len(reader_counts) != len(READER_SOURCES)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in reader_counts)
        ):
            raise AuditError(f"eligible row {position} has invalid reader counts")
        if reader_counts != [all_counts[index] for index in reader_positions]:
            raise AuditError(f"eligible row {position} reader/all-source vectors disagree")
        reader_eligible = row.get("reader_eligible")
        if not isinstance(reader_eligible, bool) or reader_eligible != any(reader_counts):
            raise AuditError(f"eligible row {position} has invalid reader policy result")
        if reader_eligible:
            reader_labels.append(label)

    observed = (
        len(rows),
        sum(labels),
        len(rows) - sum(labels),
        len(reader_labels),
        sum(reader_labels),
        len(reader_labels) - sum(reader_labels),
    )
    wanted = (
        expected.rows,
        expected.positive,
        expected.negative,
        expected.reader_rows,
        expected.reader_positive,
        expected.reader_negative,
    )
    if observed != wanted:
        raise AuditError(f"eligible class/reader counts mismatch: {observed} != {wanted}")
    return rows


def _load_json_object(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise AuditError(f"{name} is not valid UTF-8 JSON") from err
    if not isinstance(value, dict):
        raise AuditError(f"{name} root is not an object")
    return value


def _load_json_array(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise AuditError(f"{name} is not valid UTF-8 JSON") from err
    if not isinstance(value, list):
        raise AuditError(f"{name} root is not an array")
    return value


def _validate_protocol(
    protocol: dict[str, Any],
    eligible_identity: dict[str, Any],
    curations_identity: dict[str, Any],
    eligible: Sequence[dict[str, Any]],
) -> None:
    try:
        output_descriptor = protocol["outputs"]["eligible_statements"]
        raw_descriptor = protocol["inputs"]["raw_curations"]
        sets = protocol["eligible_sets"]
        source_columns = protocol["source_columns"]
        gold = protocol["gold_protocol"]
    except KeyError as err:
        raise AuditError(f"protocol manifest is missing {err.args[0]!r}") from err

    checks = {
        "eligible output sha256": (
            output_descriptor.get("sha256"),
            eligible_identity["sha256"],
        ),
        "eligible output rows": (output_descriptor.get("rows"), len(eligible)),
        "eligible output bytes": (
            output_descriptor.get("bytes"),
            eligible_identity["bytes"],
        ),
        "raw curation sha256": (raw_descriptor.get("sha256"), curations_identity["sha256"]),
        "raw curation bytes": (raw_descriptor.get("bytes"), curations_identity["bytes"]),
        "reader source order": (source_columns.get("reader_order"), list(READER_SOURCES)),
        "historical all-source order": (
            source_columns.get("historical_all_source_order"),
            list(HISTORICAL_ALL_SOURCE_ORDER),
        ),
        "positive tag mapping": (
            gold.get("positive_tags"),
            sorted(PAPER_POSITIVE_TAGS),
        ),
        "same-pair conflict policy": (
            gold.get("same_pair_conflict"),
            "negative_wins; pair positive iff all curations positive",
        ),
    }
    for name, (observed, wanted) in checks.items():
        if observed != wanted:
            raise AuditError(f"protocol {name} mismatch: {observed!r} != {wanted!r}")

    labels = [int(row["correct"]) for row in eligible]
    reader = [row for row in eligible if row["reader_eligible"]]
    expected_sets = {
        "extended_all_sources": (len(eligible), sum(labels), len(eligible) - sum(labels)),
        "reader_only": (
            len(reader),
            sum(int(row["correct"]) for row in reader),
            sum(not int(row["correct"]) for row in reader),
        ),
    }
    for name, wanted in expected_sets.items():
        descriptor = sets.get(name, {})
        observed = (
            descriptor.get("rows"),
            descriptor.get("positive"),
            descriptor.get("negative"),
        )
        if observed != wanted:
            raise AuditError(f"protocol eligible set {name} mismatch: {observed} != {wanted}")


def _scan_corpus(
    path: Path,
    target_hashes: set[int],
    expected: CorpusExpectation,
) -> tuple[dict[int, tuple[int, dict[str, Any]]], dict[str, int]]:
    selected: dict[int, tuple[int, dict[str, Any]]] = {}
    rows = 0
    evidence_entries = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for row_index, statement in enumerate(_iter_json_array(stream)):
                if not isinstance(statement, dict):
                    raise AuditError(f"corpus row {row_index} is not an object")
                evidence = statement.get("evidence", [])
                if not isinstance(evidence, list):
                    raise AuditError(f"corpus row {row_index} evidence is not an array")
                rows += 1
                evidence_entries += len(evidence)
                raw_hash = statement.get("matches_hash")
                if raw_hash is None:
                    continue
                matches_hash = _exact_int(raw_hash, field=f"corpus row {row_index} matches_hash")
                if matches_hash not in target_hashes:
                    continue
                if matches_hash in selected:
                    previous = selected[matches_hash][0]
                    raise AuditError(
                        f"target matches_hash {matches_hash} occurs at corpus rows "
                        f"{previous} and {row_index}"
                    )
                selected[matches_hash] = (row_index, statement)
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as err:
        raise AuditError("canonical corpus gzip/UTF-8 stream is invalid") from err

    observed = (rows, evidence_entries)
    wanted = (expected.rows, expected.evidence_entries)
    if observed != wanted:
        raise AuditError(f"canonical corpus counts mismatch: {observed} != {wanted}")
    missing = sorted(target_hashes - set(selected))
    if missing:
        raise AuditError(f"eligible statements missing from canonical corpus: {missing[:10]}")
    return selected, {"rows": rows, "evidence_entries": evidence_entries}


def _curation_provenance(row: dict[str, Any], raw_row_index: int) -> dict[str, Any]:
    return {
        "raw_curation_row_index": raw_row_index,
        "curation_id": row.get("id"),
        "curation_source": row.get("source"),
        "tag": row["tag"],
        "tag_label": int(row["tag"] in PAPER_POSITIVE_TAGS),
        "curator": row["curator"],
        "date": row.get("date"),
        "comment": row.get("text"),
    }


def _blind_queue_payload(value: Any) -> Any:
    """Copy released semantic context while removing gold/linkage fields."""
    if isinstance(value, dict):
        return {
            key: _blind_queue_payload(child)
            for key, child in value.items()
            if key.lower() not in QUEUE_FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_blind_queue_payload(child) for child in value]
    return value


def _queue_forbidden_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else key
            if key.lower() in QUEUE_FORBIDDEN_KEYS:
                paths.append(child_path)
            paths.extend(_queue_forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_queue_forbidden_paths(child, f"{prefix}[{index}]"))
    return paths


def _queue_variants(evidence_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for evidence in evidence_rows:
        payload_sha256 = _sha256_bytes(_canonical_json_bytes(evidence))
        if payload_sha256 not in variants:
            variants[payload_sha256] = {
                "variant_id": f"v{len(order) + 1:03d}",
                "entry_count": 0,
                "released_payload_sha256": payload_sha256,
                "evidence": _blind_queue_payload(evidence),
            }
            order.append(payload_sha256)
        variants[payload_sha256]["entry_count"] += 1
    return [variants[digest] for digest in order]


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": _package_versions(("indra",)),
    }


def _preflight_outputs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "statement_gold": output_dir / STATEMENT_GOLD_FILENAME,
        "evidence_adjudication": output_dir / EVIDENCE_ADJUDICATION_FILENAME,
        "adjudication_queue": output_dir / ADJUDICATION_QUEUE_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    existing = [str(path) for path in paths.values() if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing paper-gold artifacts: " + ", ".join(existing)
        )
    return paths


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_new_bundle(
    paths: dict[str, Path], payloads: dict[str, bytes], order: Sequence[str]
) -> None:
    """Stage a bundle and hard-link final names without replacement.

    Every payload is durable before publication, the manifest is linked last,
    and any failure rolls back final links created by this call.  ``os.link``
    supplies the no-clobber guarantee even if another writer races preflight.
    """
    parents = {path.parent for path in paths.values()}
    if len(parents) != 1:
        raise AuditError("all bundle outputs must share one directory")
    output_dir = parents.pop()
    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged: dict[str, Path] = {}
    published: list[Path] = []
    try:
        for name in order:
            final_path = paths[name]
            stage_path = output_dir / f".{final_path.name}.{token}.tmp"
            staged[name] = stage_path
            _write_new(stage_path, payloads[name])
        for name in order:
            final_path = paths[name]
            if os.path.lexists(final_path):
                raise FileExistsError(f"refusing to overwrite existing artifact: {final_path}")
            os.link(staged[name], final_path)
            published.append(final_path)
    except BaseException:
        for final_path in reversed(published):
            try:
                final_path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for stage_path in staged.values():
            try:
                stage_path.unlink()
            except FileNotFoundError:
                pass


def _output_identity(path: Path, payload: bytes, rows: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "rows": rows,
        "sha256": _sha256_bytes(payload),
    }


def materialize_paper_statement_gold(
    corpus_path: Path,
    curations_path: Path,
    protocol_manifest_path: Path,
    eligible_statements_path: Path,
    output_dir: Path,
    *,
    expectations: MaterializationExpectations = RELEASE_EXPECTATIONS,
) -> dict[str, Any]:
    """Verify frozen inputs and emit statement/evidence gold ledgers.

    All target payloads are built and reconciled before any output is written.
    Existing named outputs are never replaced.
    """
    output_dir = Path(output_dir)
    output_paths = _preflight_outputs(output_dir)

    corpus_identity = _verify_file(corpus_path, expectations.corpus.file)
    curations_payload, curations_identity = _verified_small_payload(
        curations_path, expectations.curations
    )
    protocol_payload, protocol_identity = _verified_small_payload(
        protocol_manifest_path, expectations.protocol_manifest
    )
    eligible_payload, eligible_identity = _verified_small_payload(
        eligible_statements_path, expectations.eligible.file
    )

    curations = _load_json_array(curations_payload, name="official curations")
    if len(curations) != expectations.curation_rows:
        raise AuditError(
            f"official curation row count mismatch: {len(curations)} != "
            f"{expectations.curation_rows}"
        )
    curation_audit = _curation_audit_counts(curations)
    protocol = _load_json_object(protocol_payload, name="protocol manifest")
    eligible = _load_eligible(eligible_payload, expectations.eligible)
    _validate_protocol(protocol, eligible_identity, curations_identity, eligible)
    if protocol.get("gold_protocol", {}).get("audit_counts") != curation_audit:
        raise AuditError("official curation audit counts disagree with frozen protocol")

    eligible_hashes = {
        _exact_int(row["stmt_hash"], field="eligible stmt_hash") for row in eligible
    }
    selected_corpus, corpus_counts = _scan_corpus(
        Path(corpus_identity["resolved_path"]), eligible_hashes, expectations.corpus
    )

    curations_by_pair: defaultdict[
        tuple[int, int], list[tuple[int, dict[str, Any]]]
    ] = defaultdict(list)
    eligible_curation_rows = 0
    for raw_index, row in enumerate(curations):
        pa_hash = _exact_int(row["pa_hash"], field=f"curation row {raw_index} pa_hash")
        source_hash = _exact_int(
            row["source_hash"], field=f"curation row {raw_index} source_hash"
        )
        if pa_hash in eligible_hashes:
            curations_by_pair[(pa_hash, source_hash)].append((raw_index, row))
            eligible_curation_rows += 1

    statement_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    target_evidence_entries = 0
    target_distinct_pairs = 0
    unidentifiable_entries = 0
    reviewed_entries = 0
    unreviewed_entries = 0
    reviewed_pairs = 0
    reviewed_positive_pairs = 0
    reviewed_negative_pairs = 0
    conflicting_pairs = 0
    unreviewed_pairs = 0
    complete_statements = 0
    statuses: Counter[str] = Counter()
    reader_statuses: Counter[str] = Counter()

    for eligible_position, eligible_row in enumerate(eligible):
        pa_hash = _exact_int(
            eligible_row["stmt_hash"], field=f"eligible row {eligible_position} stmt_hash"
        )
        corpus_row_index, statement = selected_corpus[pa_hash]
        evidence = statement.get("evidence", [])
        target_evidence_entries += len(evidence)
        if statement.get("type") != eligible_row.get("stmt_type"):
            raise AuditError(
                f"statement type mismatch for {pa_hash}: corpus={statement.get('type')!r}, "
                f"protocol={eligible_row.get('stmt_type')!r}"
            )

        source_counts = Counter()
        grouped: dict[int, dict[str, Any]] = {}
        group_order: list[int] = []
        unidentified: list[tuple[int, dict[str, Any]]] = []
        for evidence_position, evidence_row in enumerate(evidence):
            if not isinstance(evidence_row, dict):
                raise AuditError(
                    f"corpus statement {pa_hash} evidence {evidence_position} is not an object"
                )
            source_api = evidence_row.get("source_api")
            if not isinstance(source_api, str):
                raise AuditError(
                    f"corpus statement {pa_hash} evidence {evidence_position} lacks source_api"
                )
            source_counts[source_api] += 1
            raw_source_hash = evidence_row.get("source_hash")
            if raw_source_hash is None:
                unidentified.append((evidence_position, evidence_row))
                continue
            source_hash = _exact_int(
                raw_source_hash,
                field=f"corpus statement {pa_hash} evidence {evidence_position} source_hash",
            )
            if source_hash not in grouped:
                grouped[source_hash] = {
                    "positions": [],
                    "source_apis": [],
                    "payload_sha256s": [],
                    "text_present": [],
                }
                group_order.append(source_hash)
            group = grouped[source_hash]
            group["positions"].append(evidence_position)
            group["source_apis"].append(source_api)
            group["payload_sha256s"].append(
                _sha256_bytes(_canonical_json_bytes(evidence_row))
            )
            text_value = evidence_row.get("text")
            group["text_present"].append(
                isinstance(text_value, str) and bool(text_value.strip())
            )

        all_vector = [source_counts[source] for source in HISTORICAL_ALL_SOURCE_ORDER]
        reader_vector = [source_counts[source] for source in READER_SOURCES]
        if sum(all_vector) != len(evidence):
            unexpected = sorted(set(source_counts) - set(HISTORICAL_ALL_SOURCE_ORDER))
            raise AuditError(f"statement {pa_hash} has unexpected evidence sources: {unexpected}")
        if all_vector != eligible_row["historical_all_source_counts"]:
            raise AuditError(f"canonical source-count vector disagrees for {pa_hash}")
        if reader_vector != eligible_row["reader_source_counts"]:
            raise AuditError(f"canonical reader-count vector disagrees for {pa_hash}")
        if bool(any(reader_vector)) != eligible_row["reader_eligible"]:
            raise AuditError(f"canonical reader policy disagrees for {pa_hash}")

        target_distinct_pairs += len(grouped)
        positive_hashes: list[str] = []
        negative_hashes: list[str] = []
        unreviewed_hashes: list[str] = []
        adjudication_ids: list[str] = []
        curation_row_count = 0
        reviewed_evidence_entries = 0
        unreviewed_evidence_entries = len(unidentified)
        statement_evidence_start = len(evidence_rows)

        for evidence_ordinal, source_hash in enumerate(group_order):
            group = grouped[source_hash]
            adjudication_id = f"s{eligible_position:04d}-e{evidence_ordinal:05d}"
            adjudication_ids.append(adjudication_id)
            pair_curations = curations_by_pair.get((pa_hash, source_hash), [])
            provenance = [
                _curation_provenance(row, raw_index)
                for raw_index, row in pair_curations
            ]
            curation_row_count += len(pair_curations)
            if pair_curations:
                pair_label = int(
                    all(row["tag"] in PAPER_POSITIVE_TAGS for _, row in pair_curations)
                )
                has_positive_tag = any(
                    row["tag"] in PAPER_POSITIVE_TAGS for _, row in pair_curations
                )
                conflict = has_positive_tag and not bool(pair_label)
                status = "positive" if pair_label else "negative"
                reviewed_pairs += 1
                reviewed_evidence_entries += len(group["positions"])
                if pair_label:
                    reviewed_positive_pairs += 1
                    positive_hashes.append(str(source_hash))
                else:
                    reviewed_negative_pairs += 1
                    negative_hashes.append(str(source_hash))
                conflicting_pairs += int(conflict)
            else:
                pair_label = None
                conflict = False
                status = "unreviewed"
                unreviewed_pairs += 1
                unreviewed_evidence_entries += len(group["positions"])
                unreviewed_hashes.append(str(source_hash))

            evidence_rows.append(
                {
                    "adjudication_id": adjudication_id,
                    "eligible_position": eligible_position,
                    "paper_statement_hash": str(pa_hash),
                    "canonical_corpus_row_index": corpus_row_index,
                    "identity_kind": "statement_source_hash_pair",
                    "source_hash": str(source_hash),
                    "corpus_evidence_positions": group["positions"],
                    "corpus_evidence_entry_count": len(group["positions"]),
                    "source_apis": list(dict.fromkeys(group["source_apis"])),
                    "corpus_evidence_json_sha256s": group["payload_sha256s"],
                    "corpus_evidence_text_present": group["text_present"],
                    "review_status": status,
                    "evidence_gold_label": pair_label,
                    "same_pair_conflict": conflict,
                    "conflict_resolution": "negative_wins" if conflict else None,
                    "curation_count": len(pair_curations),
                    "curations": provenance,
                }
            )

        for unidentified_ordinal, (evidence_position, evidence_row) in enumerate(
            unidentified
        ):
            adjudication_id = (
                f"s{eligible_position:04d}-u{unidentified_ordinal:05d}"
            )
            adjudication_ids.append(adjudication_id)
            unidentifiable_entries += 1
            evidence_rows.append(
                {
                    "adjudication_id": adjudication_id,
                    "eligible_position": eligible_position,
                    "paper_statement_hash": str(pa_hash),
                    "canonical_corpus_row_index": corpus_row_index,
                    "identity_kind": "unidentifiable_corpus_evidence_position",
                    "source_hash": None,
                    "corpus_evidence_positions": [evidence_position],
                    "corpus_evidence_entry_count": 1,
                    "source_apis": [evidence_row["source_api"]],
                    "corpus_evidence_json_sha256s": [
                        _sha256_bytes(_canonical_json_bytes(evidence_row))
                    ],
                    "corpus_evidence_text_present": [
                        isinstance(evidence_row.get("text"), str)
                        and bool(evidence_row["text"].strip())
                    ],
                    "review_status": "unreviewed_unidentifiable",
                    "evidence_gold_label": None,
                    "same_pair_conflict": False,
                    "conflict_resolution": None,
                    "curation_count": 0,
                    "curations": [],
                }
            )

        canonical_source_hashes = set(grouped)
        curated_source_hashes = {
            source_hash
            for (curated_pa_hash, source_hash) in curations_by_pair
            if curated_pa_hash == pa_hash
        }
        orphan_curations = sorted(curated_source_hashes - canonical_source_hashes)
        if orphan_curations:
            raise AuditError(
                f"curations for statement {pa_hash} do not map to canonical evidence: "
                f"{orphan_curations[:10]}"
            )
        if not curated_source_hashes:
            raise AuditError(f"eligible statement {pa_hash} has no official curations")

        complete = not unreviewed_hashes and not unidentified and not orphan_curations
        complete_statements += int(complete)
        if positive_hashes:
            gold_status = "positive"
            gold_label: int | None = 1
            resolution_reason = "at_least_one_reviewed_positive_evidence_pair"
        elif complete:
            gold_status = "negative"
            gold_label = 0
            resolution_reason = "all_distinct_corpus_evidence_reviewed_negative"
        else:
            gold_status = "unresolved"
            gold_label = None
            resolution_reason = "no_reviewed_positive_and_incomplete_evidence_review"
        statuses[gold_status] += 1
        if eligible_row["reader_eligible"]:
            reader_statuses[gold_status] += 1
        unresolved_adjudication_ids: list[str] = []
        for evidence_adjudication in evidence_rows[statement_evidence_start:]:
            needs_resolution = gold_status == "unresolved" and evidence_adjudication[
                "review_status"
            ] in {"unreviewed", "unreviewed_unidentifiable"}
            evidence_adjudication["needed_to_resolve_statement"] = needs_resolution
            if needs_resolution:
                unresolved_adjudication_ids.append(
                    evidence_adjudication["adjudication_id"]
                )
                queue_item_id = f"q{len(queue_rows) + 1:07d}"
                evidence_adjudication["queue_item_id"] = queue_item_id
                queue_evidence = [
                    evidence[position]
                    for position in evidence_adjudication[
                        "corpus_evidence_positions"
                    ]
                ]
                statement_context = {
                    key: value
                    for key, value in statement.items()
                    if key
                    not in {
                        "belief",
                        "evidence",
                        "id",
                        "matches_hash",
                        "supported_by",
                        "supports",
                    }
                }
                queue_row = {
                    "queue_item_id": queue_item_id,
                    "rubric_version": "paper-e0-evidence-v1",
                    "statement": _blind_queue_payload(statement_context),
                    "evidence_variants": _queue_variants(queue_evidence),
                    "entry_count": len(queue_evidence),
                    "source_apis": evidence_adjudication["source_apis"],
                    "reviewability": (
                        "text_and_metadata"
                        if any(evidence_adjudication["corpus_evidence_text_present"])
                        else "metadata_only"
                    ),
                    "adjudication_status": "unresolved",
                    "allowed_verdicts": ["correct", "incorrect", "unresolved"],
                    "required_independent_reviews": 2,
                    "conflict_policy": "blinded_tie_break",
                }
                forbidden_paths = _queue_forbidden_paths(queue_row)
                if forbidden_paths:
                    raise AuditError(
                        "blinded adjudication queue contains forbidden fields: "
                        f"{forbidden_paths[:10]}"
                    )
                queue_rows.append(queue_row)
            else:
                evidence_adjudication["queue_item_id"] = None

        paper_computed_label = int(bool(positive_hashes))
        paper_frozen_label = int(eligible_row["correct"])
        if paper_computed_label != paper_frozen_label:
            raise AuditError(
                f"paper-policy label mismatch for {pa_hash}: official curations="
                f"{paper_computed_label}, frozen eligible={paper_frozen_label}"
            )

        statement_rows.append(
            {
                "eligible_position": eligible_position,
                "source_row_index": eligible_row["source_row_index"],
                "paper_statement_hash": str(pa_hash),
                "canonical_corpus": {
                    "row_index": corpus_row_index,
                    "matches_hash": str(pa_hash),
                    "statement_id": statement.get("id"),
                    "statement_type": statement.get("type"),
                    "statement_json_sha256": _sha256_bytes(
                        _canonical_json_bytes(statement)
                    ),
                },
                "paper_eligibility": {
                    "extended_all_sources": True,
                    "reader_only": eligible_row["reader_eligible"],
                    "in_multireader_released_dataset": eligible_row.get(
                        "in_multireader_dataset"
                    ),
                    "historical_all_source_counts": eligible_row[
                        "historical_all_source_counts"
                    ],
                    "reader_source_counts": eligible_row["reader_source_counts"],
                },
                "evidence_review": {
                    "corpus_evidence_entries": len(evidence),
                    "distinct_identifiable_evidence_pairs": len(grouped),
                    "unidentifiable_evidence_entries": len(unidentified),
                    "reviewed_distinct_pairs": len(positive_hashes) + len(negative_hashes),
                    "reviewed_positive_pairs": len(positive_hashes),
                    "reviewed_negative_pairs": len(negative_hashes),
                    "unreviewed_distinct_pairs": len(unreviewed_hashes),
                    "reviewed_evidence_entries": reviewed_evidence_entries,
                    "unreviewed_evidence_entries": unreviewed_evidence_entries,
                    "curation_rows": curation_row_count,
                    "complete_distinct_evidence_review": complete,
                    "positive_source_hashes": positive_hashes,
                    "negative_source_hashes": negative_hashes,
                    "unreviewed_source_hashes": unreviewed_hashes,
                    "unidentifiable_evidence_positions": [
                        position for position, _ in unidentified
                    ],
                    "evidence_adjudication_ids": adjudication_ids,
                    "unresolved_adjudication_queue_ids": unresolved_adjudication_ids,
                },
                "adjudicated_statement_gold": {
                    "status": gold_status,
                    "label": gold_label,
                    "strict_e0_status": gold_status,
                    "strict_e0_correct": gold_label,
                    "resolution_reason": resolution_reason,
                    "include_in_complete_negative_evaluation": gold_status
                    != "unresolved",
                },
                "paper_replication_policy": {
                    "label": paper_frozen_label,
                    "recomputed_label": paper_computed_label,
                    "released_paper_correct": paper_frozen_label,
                    "recomputed_released_paper_correct": paper_computed_label,
                    "labels_match": True,
                    "policy": (
                        "positive iff any reviewed evidence pair is positive; "
                        "otherwise negative even when evidence review is incomplete"
                    ),
                    "label_is_adjudication_safe": gold_status != "unresolved",
                    "negative_has_complete_evidence": (
                        complete if paper_frozen_label == 0 else None
                    ),
                    "differs_from_adjudicated_gold": gold_status == "unresolved",
                },
            }
        )
        reviewed_entries += reviewed_evidence_entries
        unreviewed_entries += unreviewed_evidence_entries

    result_counts = ResultExpectation(
        target_evidence_entries=target_evidence_entries,
        target_distinct_evidence_pairs=target_distinct_pairs,
        unidentifiable_evidence_entries=unidentifiable_entries,
        reviewed_evidence_entries=reviewed_entries,
        unreviewed_evidence_entries=unreviewed_entries,
        eligible_curation_rows=eligible_curation_rows,
        reviewed_evidence_pairs=reviewed_pairs,
        reviewed_positive_pairs=reviewed_positive_pairs,
        reviewed_negative_pairs=reviewed_negative_pairs,
        conflicting_reviewed_pairs=conflicting_pairs,
        unreviewed_evidence_pairs=unreviewed_pairs,
        evidence_adjudication_rows=len(evidence_rows),
        complete_evidence_statements=complete_statements,
        adjudicated_positive=statuses["positive"],
        adjudicated_negative=statuses["negative"],
        adjudicated_unresolved=statuses["unresolved"],
        unresolved_adjudication_queue_rows=len(queue_rows),
        unresolved_adjudication_queue_evidence_entries=sum(
            row["entry_count"] for row in queue_rows
        ),
        unresolved_adjudication_queue_textless_rows=sum(
            row["reviewability"] == "metadata_only" for row in queue_rows
        ),
    )
    if result_counts != expectations.results:
        raise AuditError(
            "materialized gold counts mismatch: "
            f"expected={asdict(expectations.results)}, observed={asdict(result_counts)}"
        )

    statement_payload = _jsonl_bytes(statement_rows)
    evidence_payload = _jsonl_bytes(evidence_rows)
    queue_payload = _jsonl_bytes(queue_rows)
    harness_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "indra_paper_statement_gold_with_evidence_adjudication",
        "created_at": _utc_now(),
        "claim_scope": {
            "lane": "paper",
            "unit": "assembled_statement",
            "historical_replication_label_kept_separate": True,
            "unreviewed_evidence_is_negative": False,
            "paid_model_calls": 0,
        },
        "harness": {
            "path": str(harness_path),
            "sha256": _sha256_file(harness_path),
        },
        "inputs": {
            "canonical_corpus": corpus_identity,
            "official_curations": curations_identity,
            "frozen_protocol_manifest": protocol_identity,
            "frozen_eligible_statements": eligible_identity,
        },
        "semantics": {
            "positive_tags": sorted(PAPER_POSITIVE_TAGS),
            "same_pair_conflict": "negative_wins; pair positive iff all curations positive",
            "adjudicated_positive": (
                "at least one exact statement/evidence pair is reviewed positive"
            ),
            "adjudicated_negative": (
                "every distinct identifiable corpus evidence hash is reviewed, no "
                "evidence is unidentifiable, and every pair is negative"
            ),
            "adjudicated_unresolved": (
                "no reviewed positive exists and at least one corpus evidence item "
                "is unreviewed or unidentifiable"
            ),
            "duplicate_source_hash_policy": (
                "one pair adjudication applies to every corpus evidence entry with "
                "that exact source hash; multiplicity and payload digests are retained"
            ),
            "paper_replication_policy": (
                "positive iff any reviewed pair is positive; otherwise negative under "
                "the released allow_incomplete=True dataset construction"
            ),
        },
        "eligible_universe": {
            "order": "frozen eligible-statements JSONL order",
            "extended_all_sources_rows": len(statement_rows),
            "reader_policy_sources": list(READER_SOURCES),
            "reader_only_rows": sum(
                row["paper_eligibility"]["reader_only"] for row in statement_rows
            ),
            "historical_all_source_order": list(HISTORICAL_ALL_SOURCE_ORDER),
            "canonical_matches_hash_mapping": "exact and one-to-one",
        },
        "counts": {
            "canonical_corpus": corpus_counts,
            "official_curations": curation_audit,
            "eligible_curations_outside_universe": len(curations)
            - eligible_curation_rows,
            **asdict(result_counts),
            "resolved_for_complete_negative_evaluation": statuses["positive"]
            + statuses["negative"],
            "paper_policy_positive": expectations.eligible.positive,
            "paper_policy_negative": expectations.eligible.negative,
            "paper_policy_negatives_excluded_as_unresolved": statuses["unresolved"],
            "reader_adjudicated_status": dict(sorted(reader_statuses.items())),
        },
        "complete_evidence_gate": {
            "include_statuses": ["positive", "negative"],
            "exclude_statuses": ["unresolved"],
            "positive_may_be_resolved_before_complete_review": True,
            "negative_requires_complete_distinct_evidence_review": True,
            "resolved_rows": statuses["positive"] + statuses["negative"],
            "excluded_unresolved_rows": statuses["unresolved"],
        },
        "adjudication_queue": {
            "artifact": ADJUDICATION_QUEUE_FILENAME,
            "mapping_artifact": EVIDENCE_ADJUDICATION_FILENAME,
            "mapping_key": "queue_item_id",
            "scope": (
                "only unreviewed evidence on currently unresolved statements; "
                "unreviewed evidence on already-positive statements is retained in "
                "the ledger but is not required to resolve statement E0"
            ),
            "rows": len(queue_rows),
            "corpus_evidence_entries": sum(
                row["entry_count"] for row in queue_rows
            ),
            "textless_rows": sum(
                row["reviewability"] == "metadata_only" for row in queue_rows
            ),
            "forbidden_fields_recursive": sorted(QUEUE_FORBIDDEN_KEYS),
            "forbidden_field_scan": "pass",
            "human_review_required": True,
            "review_record_contract": {
                "append_only": True,
                "key": "queue_item_id",
                "required_fields": [
                    "queue_item_id",
                    "reviewer_pseudonym",
                    "rubric_version",
                    "verdict",
                    "reason_code",
                    "reviewed_at",
                    "review_round",
                ],
                "allowed_verdicts": ["correct", "incorrect", "unresolved"],
                "independent_reviews_required": 2,
                "conflict_resolution": "blinded_tie_break",
            },
            "future_statement_reaggregation": {
                "positive": "any evidence pair resolved correct",
                "negative": "every distinct evidence pair resolved incorrect",
                "unresolved": "otherwise",
            },
        },
        "coverage_reconciliation": {
            "all_frozen_eligible_hashes_found_once_in_corpus": True,
            "all_canonical_source_count_vectors_match_protocol": True,
            "reader_policy_matches_protocol": True,
            "every_eligible_statement_has_official_curations": True,
            "every_eligible_curation_pair_maps_to_canonical_evidence": True,
            "paper_policy_labels_recomputed_exactly": True,
            "statement_status_partition_sums_to_eligible_rows": (
                sum(statuses.values()) == len(statement_rows)
            ),
            "reviewed_and_unreviewed_pairs_partition_identifiable_pairs": (
                reviewed_pairs + unreviewed_pairs == target_distinct_pairs
            ),
            "evidence_ledger_rows_reconcile": (
                len(evidence_rows) == target_distinct_pairs + unidentifiable_entries
            ),
            "reviewed_and_unreviewed_entries_partition_corpus_evidence": (
                reviewed_entries + unreviewed_entries == target_evidence_entries
            ),
        },
        "runtime": _runtime_identity(),
        "publication": {
            "all_payloads_staged_before_final_names": True,
            "final_name_operation": "same-directory hard link; fails if target exists",
            "manifest_published_last": True,
            "exception_rollback": True,
            "destructive_force_option": False,
        },
        "outputs": {
            "statement_gold": _output_identity(
                output_paths["statement_gold"], statement_payload, len(statement_rows)
            ),
            "evidence_adjudication": _output_identity(
                output_paths["evidence_adjudication"],
                evidence_payload,
                len(evidence_rows),
            ),
            "adjudication_queue": _output_identity(
                output_paths["adjudication_queue"], queue_payload, len(queue_rows)
            ),
        },
        "validation": {
            "input_digests": "pass",
            "protocol_contract": "pass",
            "canonical_coverage": "pass",
            "paper_label_reproduction": "pass",
            "complete_negative_gate": "pass",
            "artifacts_overwritten": False,
        },
    }
    if not all(manifest["coverage_reconciliation"].values()):
        raise AuditError("internal coverage reconciliation failed")
    manifest_payload = _pretty_json_bytes(manifest)

    _publish_new_bundle(
        output_paths,
        {
            "statement_gold": statement_payload,
            "evidence_adjudication": evidence_payload,
            "adjudication_queue": queue_payload,
            "manifest": manifest_payload,
        },
        (
            "statement_gold",
            "evidence_adjudication",
            "adjudication_queue",
            "manifest",
        ),
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--curations", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--eligible-statements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = materialize_paper_statement_gold(
            args.corpus,
            args.curations,
            args.protocol_manifest,
            args.eligible_statements,
            args.output_dir,
        )
    except (AuditError, FileExistsError, OSError) as err:
        print(f"paper statement-gold materialization failed: {err}", file=sys.stderr)
        return 2

    counts = manifest["counts"]
    print(
        "materialized paper statement gold: "
        f"positive={counts['adjudicated_positive']} "
        f"negative={counts['adjudicated_negative']} "
        f"unresolved={counts['adjudicated_unresolved']}"
    )
    print(
        "complete-negative evaluation rows: "
        f"{counts['resolved_for_complete_negative_evaluation']}"
    )
    print(f"wrote new artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
