#!/usr/bin/env python3
"""Materialize metric-ready gold ledgers for the 2023 INDRA paper lane.

The released-paper compatibility lane and the strict complete-evidence lane are
different estimands and therefore become different files:

* ``paper_released_gold.jsonl`` keeps all 1,689 released labels.  This is the
  lane on which a result can be compared with the paper's published numbers.
* ``paper_reader_eligible_released_gold.jsonl`` keeps the 1,676-statement
  reader-eligible subset and its separately reconstructed folds.  This is the
  only direct lane for the published ``Belief Orig - readers`` headline.
* ``paper_strict_e0_resolved_gold.jsonl`` keeps only statements whose evidence
  review currently proves an E0 label.  It is a resolved-only sensitivity lane
  until the 111 incomplete historical negatives are adjudicated.
* ``paper_reader_eligible_strict_e0_resolved_gold.jsonl`` is the corresponding
  1,565-statement reader sensitivity ledger, using the reader panel's own
  reconstructed folds.

Each ledger uses its panel's frozen reconstructed ten-fold assignment.  The
source release did not publish its realized historical folds, so the manifest
says so explicitly and never calls these assignments the original folds.

``paper_prediction_targets.jsonl`` is the prediction adapter's only input
ledger.  It contains corpus identities and eligibility flags but deliberately
contains no label, curation, fold, or existing belief field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_KIND = "indra_paper_comparison_gold"
SCHEMA_VERSION = 1
RELEASED_FILENAME = "paper_released_gold.jsonl"
READER_FILENAME = "paper_reader_eligible_released_gold.jsonl"
STRICT_FILENAME = "paper_strict_e0_resolved_gold.jsonl"
READER_STRICT_FILENAME = "paper_reader_eligible_strict_e0_resolved_gold.jsonl"
MANIFEST_FILENAME = "paper_comparison_gold_manifest.json"
TARGETS_FILENAME = "paper_prediction_targets.jsonl"
EXTENDED_SET = "extended_all_sources"
READER_SET = "reader_only"


class ContractError(ValueError):
    """Raised when a frozen source or join invariant does not hold."""


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Prefer a repository-relative path without hiding out-of-tree fixtures."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _canonical_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"could not read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"could not open {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ContractError(f"{path}:{line_number}: blank lines are forbidden")
            try:
                row = json.loads(line, object_pairs_hook=_strict_pairs)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    if not rows:
        raise ContractError(f"{path}: expected at least one row")
    return rows


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{context}: expected an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{context}: expected an integer >= {minimum}")
    return value


def _binary(value: Any, context: str) -> int:
    value = _integer(value, context)
    if value not in (0, 1):
        raise ContractError(f"{context}: expected 0 or 1")
    return value


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{context}: expected a non-empty string")
    return value


def _descriptor(manifest: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = manifest
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ContractError(f"manifest is missing {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, dict):
        raise ContractError(f"manifest {'.'.join(keys)} must be an object")
    return value


def _verify_declared_file(
    path: Path,
    manifest: dict[str, Any],
    descriptor_keys: tuple[str, ...],
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    descriptor = _descriptor(manifest, *descriptor_keys)
    declared_sha = _nonempty(
        descriptor.get("sha256"), f"{'.'.join(descriptor_keys)}.sha256"
    )
    actual_sha = _sha256(path)
    if actual_sha != declared_sha:
        raise ContractError(
            f"{path}: digest mismatch; manifest declares {declared_sha}, got {actual_sha}"
        )
    declared_rows = descriptor.get("rows")
    if expected_rows is not None and declared_rows != expected_rows:
        raise ContractError(
            f"{path}: manifest row count {declared_rows!r} != observed {expected_rows}"
        )
    return {
        "path": _display_path(path),
        "bytes": path.stat().st_size,
        "rows": declared_rows,
        "sha256": actual_sha,
        "verification": "pass",
    }


def ordered_statement_id_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_line({"statement_id": row["statement_id"]}))
    return digest.hexdigest()


def _fold_index(
    rows: list[dict[str, Any]], eligible_set: str
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        if row.get("eligible_set") != eligible_set:
            continue
        position = _integer(
            row.get("eligible_position"), f"fold row {row_number}.eligible_position", minimum=0
        )
        if position in result:
            raise ContractError(f"duplicate {eligible_set} eligible_position {position}")
        fold = _integer(row.get("test_fold"), f"fold row {row_number}.test_fold", minimum=0)
        if fold > 9:
            raise ContractError(f"fold row {row_number}.test_fold: expected 0..9")
        result[position] = {
            "label": _binary(row.get("correct"), f"fold row {row_number}.correct"),
            "fold_id": fold,
            "stmt_hash": _nonempty(
                row.get("stmt_hash"), f"fold row {row_number}.stmt_hash"
            ),
            "source_row_index": _integer(
                row.get("source_row_index"),
                f"fold row {row_number}.source_row_index",
                minimum=0,
            ),
        }
    if sorted(result) != list(range(len(result))):
        raise ContractError(f"{eligible_set} eligible positions are not contiguous from zero")
    return result


def build_ledgers(
    statement_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    folds = _fold_index(fold_rows, EXTENDED_SET)
    reader_folds = _fold_index(fold_rows, READER_SET)
    reader_fold_by_source_row: dict[int, dict[str, Any]] = {}
    for fold in reader_folds.values():
        source_row_index = fold["source_row_index"]
        if source_row_index in reader_fold_by_source_row:
            raise ContractError(
                f"duplicate {READER_SET} source_row_index {source_row_index}"
            )
        reader_fold_by_source_row[source_row_index] = fold
    if len(statement_rows) != len(folds):
        raise ContractError(
            f"statement/fold row mismatch: {len(statement_rows)} != {len(folds)}"
        )

    released: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    reader_released: list[dict[str, Any]] = []
    strict: list[dict[str, Any]] = []
    reader_strict: list[dict[str, Any]] = []
    seen_statement_ids: set[str] = set()
    seen_hashes: set[str] = set()
    resolution_counts: Counter[str] = Counter()
    reader_resolution_counts: Counter[str] = Counter()

    for row_number, row in enumerate(statement_rows, 1):
        position = _integer(
            row.get("eligible_position"),
            f"statement row {row_number}.eligible_position",
            minimum=0,
        )
        if position not in folds:
            raise ContractError(f"statement row {row_number}: no frozen fold for {position}")
        fold = folds[position]
        canonical = row.get("canonical_corpus")
        eligibility = row.get("paper_eligibility")
        policy = row.get("paper_replication_policy")
        strict_gold = row.get("adjudicated_statement_gold")
        if eligibility is None:
            eligibility = {"reader_only": False}
        if (
            not isinstance(canonical, dict)
            or not isinstance(eligibility, dict)
            or not isinstance(policy, dict)
            or not isinstance(strict_gold, dict)
        ):
            raise ContractError(f"statement row {row_number}: malformed nested gold fields")

        statement_id = _nonempty(
            canonical.get("statement_id"), f"statement row {row_number}.statement_id"
        )
        statement_hash = _nonempty(
            row.get("paper_statement_hash"), f"statement row {row_number}.paper_statement_hash"
        )
        source_row_index = _integer(
            row.get("source_row_index"), f"statement row {row_number}.source_row_index", minimum=0
        )
        label = _binary(policy.get("label"), f"statement row {row_number}.paper label")
        if statement_id in seen_statement_ids:
            raise ContractError(f"duplicate canonical statement_id {statement_id!r}")
        if statement_hash in seen_hashes:
            raise ContractError(f"duplicate paper statement hash {statement_hash!r}")
        seen_statement_ids.add(statement_id)
        seen_hashes.add(statement_hash)
        if canonical.get("matches_hash") != statement_hash:
            raise ContractError(f"statement row {row_number}: canonical/paper hash mismatch")
        if (
            fold["stmt_hash"] != statement_hash
            or fold["source_row_index"] != source_row_index
            or fold["label"] != label
        ):
            raise ContractError(f"statement row {row_number}: fold identity/label mismatch")

        target = {
            "canonical_corpus_row_index": _integer(
                canonical.get("row_index"),
                f"statement row {row_number}.canonical_corpus.row_index",
                minimum=0,
            ),
            "eligible_position": position,
            "matches_hash": statement_hash,
            "reader_eligible": eligibility.get("reader_only"),
            "source_row_index": source_row_index,
            "statement_id": statement_id,
            "statement_json_sha256": _nonempty(
                canonical.get("statement_json_sha256"),
                f"statement row {row_number}.canonical_corpus.statement_json_sha256",
            ),
            "statement_type": _nonempty(
                canonical.get("statement_type"),
                f"statement row {row_number}.canonical_corpus.statement_type",
            ),
        }

        metric_row = {
            "fold_id": fold["fold_id"],
            "label": label,
            "statement_id": statement_id,
        }
        released.append(metric_row)

        reader_eligible = eligibility.get("reader_only")
        if not isinstance(reader_eligible, bool):
            raise ContractError(
                f"statement row {row_number}.paper_eligibility.reader_only: expected bool"
            )
        # Assign only after type validation, so malformed eligibility never
        # enters the prediction-blinded ledger.
        target["reader_eligible"] = reader_eligible
        targets.append(target)
        reader_fold: dict[str, Any] | None = None
        if reader_eligible:
            reader_fold = reader_fold_by_source_row.get(source_row_index)
            if reader_fold is None:
                raise ContractError(
                    f"statement row {row_number}: reader-eligible row has no reader fold"
                )
            if reader_fold["stmt_hash"] != statement_hash or reader_fold["label"] != label:
                raise ContractError(
                    f"statement row {row_number}: reader fold identity/label mismatch"
                )
            reader_released.append(
                {
                    "fold_id": reader_fold["fold_id"],
                    "label": label,
                    "statement_id": statement_id,
                }
            )

        status = _nonempty(
            strict_gold.get("strict_e0_status"),
            f"statement row {row_number}.strict_e0_status",
        )
        resolution_counts[status] += 1
        if reader_eligible:
            reader_resolution_counts[status] += 1
        strict_label = strict_gold.get("strict_e0_correct")
        if status == "unresolved":
            if strict_label is not None:
                raise ContractError(
                    f"statement row {row_number}: unresolved strict gold has a label"
                )
        elif status in {"positive", "negative"}:
            strict_value = _binary(
                strict_label, f"statement row {row_number}.strict_e0_correct"
            )
            if strict_value != int(status == "positive"):
                raise ContractError(f"statement row {row_number}: strict status/label mismatch")
            strict.append(
                {
                    "fold_id": fold["fold_id"],
                    "label": strict_value,
                    "statement_id": statement_id,
                }
            )
            if reader_eligible:
                if reader_fold is None:  # guarded above; keep optimized runs fail closed
                    raise ContractError(
                        f"statement row {row_number}: reader strict row has no reader fold"
                    )
                reader_strict.append(
                    {
                        "fold_id": reader_fold["fold_id"],
                        "label": strict_value,
                        "statement_id": statement_id,
                    }
                )
        else:
            raise ContractError(f"statement row {row_number}: unknown strict status {status!r}")

    released_counts = Counter(row["label"] for row in released)
    reader_counts = Counter(row["label"] for row in reader_released)
    strict_counts = Counter(row["label"] for row in strict)
    reader_strict_counts = Counter(row["label"] for row in reader_strict)
    if len(reader_released) != len(reader_folds):
        raise ContractError(
            f"reader statement/fold row mismatch: {len(reader_released)} != {len(reader_folds)}"
        )
    audit = {
        "released": {
            "rows": len(released),
            "positive": released_counts[1],
            "negative": released_counts[0],
            "folds": {
                str(fold): count
                for fold, count in sorted(
                    Counter(row["fold_id"] for row in released).items()
                )
            },
        },
        "reader_eligible_released": {
            "rows": len(reader_released),
            "positive": reader_counts[1],
            "negative": reader_counts[0],
            "folds": {
                str(fold): count
                for fold, count in sorted(
                    Counter(row["fold_id"] for row in reader_released).items()
                )
            },
        },
        "strict_resolved": {
            "rows": len(strict),
            "positive": strict_counts[1],
            "negative": strict_counts[0],
            "unresolved_excluded": resolution_counts["unresolved"],
            "folds": {
                str(fold): count
                for fold, count in sorted(
                    Counter(row["fold_id"] for row in strict).items()
                )
            },
        },
        "reader_eligible_strict_resolved": {
            "rows": len(reader_strict),
            "positive": reader_strict_counts[1],
            "negative": reader_strict_counts[0],
            "unresolved_excluded": reader_resolution_counts["unresolved"],
            "folds": {
                str(fold): count
                for fold, count in sorted(
                    Counter(row["fold_id"] for row in reader_strict).items()
                )
            },
        },
        "strict_status": dict(sorted(resolution_counts.items())),
        "reader_eligible_strict_status": dict(
            sorted(reader_resolution_counts.items())
        ),
    }
    return targets, released, reader_released, strict, reader_strict, audit


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_line(row))


def _output_descriptor(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "sha256": _sha256(path),
        "ordered_statement_id_sha256": ordered_statement_id_sha256(rows),
    }


def materialize(
    *,
    statement_gold: Path,
    statement_manifest: Path,
    fold_assignments: Path,
    protocol_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    statement_gold = statement_gold.resolve()
    statement_manifest = statement_manifest.resolve()
    fold_assignments = fold_assignments.resolve()
    protocol_manifest = protocol_manifest.resolve()
    output_dir = output_dir.resolve()

    gold_manifest = _read_json(statement_manifest)
    protocol = _read_json(protocol_manifest)
    statement_rows = _read_jsonl(statement_gold)
    fold_rows = _read_jsonl(fold_assignments)
    statement_input = _verify_declared_file(
        statement_gold,
        gold_manifest,
        ("outputs", "statement_gold"),
        expected_rows=len(statement_rows),
    )
    fold_input = _verify_declared_file(
        fold_assignments,
        protocol,
        ("outputs", "fold_assignments"),
        expected_rows=len(fold_rows),
    )

    targets, released, reader_released, strict, reader_strict, counts = build_ledgers(
        statement_rows, fold_rows
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".paper-comparison-gold-", dir=output_dir) as tmp:
        stage = Path(tmp)
        targets_stage = stage / TARGETS_FILENAME
        released_stage = stage / RELEASED_FILENAME
        reader_stage = stage / READER_FILENAME
        strict_stage = stage / STRICT_FILENAME
        reader_strict_stage = stage / READER_STRICT_FILENAME
        _write_jsonl(targets_stage, targets)
        _write_jsonl(released_stage, released)
        _write_jsonl(reader_stage, reader_released)
        _write_jsonl(strict_stage, strict)
        _write_jsonl(reader_strict_stage, reader_strict)

        targets_path = output_dir / TARGETS_FILENAME
        released_path = output_dir / RELEASED_FILENAME
        reader_path = output_dir / READER_FILENAME
        strict_path = output_dir / STRICT_FILENAME
        reader_strict_path = output_dir / READER_STRICT_FILENAME
        os.replace(targets_stage, targets_path)
        os.replace(released_stage, released_path)
        os.replace(reader_stage, reader_path)
        os.replace(strict_stage, strict_path)
        os.replace(reader_strict_stage, reader_strict_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "frozen_at": gold_manifest.get("created_at"),
        "claim_scope": {
            "published_compatibility": (
                "All released labels; valid for reproducing and extending the paper's "
                "declared benchmark target."
            ),
            "reader_eligible_published_compatibility": (
                "Released 1,676-statement reader-eligible subset; the direct lane for "
                "the published Belief Orig - readers result."
            ),
            "strict_e0_resolved_only": (
                "Sensitivity analysis only until all incomplete historical negatives "
                "are adjudicated; unresolved is never coerced to negative."
            ),
            "historical_fold_parity": False,
            "fold_assignment": (
                "Frozen deterministic reconstruction of the released split protocol; "
                "the realized historical folds were not published."
            ),
        },
        "metric_contract": {
            "prediction_unit": "assembled_statement",
            "positive_class": "correct_statement",
            "released_gold_rule": "released paper label: any observed correct evidence wins",
            "strict_gold_rule": "positive iff any evidence pair is correct; negative only when every frozen evidence pair is reviewed incorrect",
            "fold_count": 10,
        },
        "counts": counts,
        "inputs": {
            "statement_gold": statement_input,
            "statement_gold_manifest": {
                "path": _display_path(statement_manifest),
                "bytes": statement_manifest.stat().st_size,
                "sha256": _sha256(statement_manifest),
            },
            "fold_assignments": fold_input,
            "protocol_manifest": {
                "path": _display_path(protocol_manifest),
                "bytes": protocol_manifest.stat().st_size,
                "sha256": _sha256(protocol_manifest),
            },
        },
        "outputs": {
            "paper_prediction_targets": _output_descriptor(targets_path, targets),
            "paper_released_gold": _output_descriptor(released_path, released),
            "paper_reader_eligible_released_gold": _output_descriptor(
                reader_path, reader_released
            ),
            "paper_strict_e0_resolved_gold": _output_descriptor(strict_path, strict),
            "paper_reader_eligible_strict_e0_resolved_gold": _output_descriptor(
                reader_strict_path, reader_strict
            ),
        },
        "generator": {
            "path": _display_path(Path(__file__)),
            "sha256": _sha256(Path(__file__)),
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_dir, delete=False
    ) as handle:
        handle.write(payload)
        temporary_manifest = Path(handle.name)
    os.replace(temporary_manifest, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statement-gold", type=Path, required=True)
    parser.add_argument("--statement-manifest", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = materialize(
        statement_gold=args.statement_gold,
        statement_manifest=args.statement_manifest,
        fold_assignments=args.fold_assignments,
        protocol_manifest=args.protocol_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
