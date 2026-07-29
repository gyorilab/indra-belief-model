#!/usr/bin/env python3
"""Score the frozen paper universe with current INDRA ``SimpleScorer``.

This adapter is prediction-blind by construction: its target ledger contains
only canonical corpus identities and eligibility flags.  It never opens a gold,
curation, fold, prior prediction, or stored statement-belief artifact.  The
output probability ledger is therefore admissible in both paper-compatible
panels and in downstream strict-E0 sensitivity panels.

The arm is the current library default's *direct-evidence prior* score.  It is
not the paper's fitted ``OrigBelief`` method, not hierarchy propagation, and not
the private fitted Hybrid scorer used by the CoGEx production export.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import ijson
import numpy
from indra.belief import SimpleScorer
from indra.statements import stmts_from_json


SCHEMA_VERSION = 1
ARTIFACT_KIND = "current_indra_simple_paper_predictions"
ARM_ID = "indra_1.24.0_simple_default_direct"
PREDICTIONS_FILENAME = "current_indra_simple_default_predictions.jsonl"
READER_PREDICTIONS_FILENAME = "current_indra_simple_default_reader_predictions.jsonl"
PROVENANCE_FILENAME = "current_indra_simple_default_prediction_provenance.jsonl"
MANIFEST_FILENAME = "current_indra_simple_default_manifest.json"
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


class ContractError(ValueError):
    """Raised when an identity, prediction-blinding, or coverage gate fails."""


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"could not read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _read_targets(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_rows: set[int] = set()
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"could not open target ledger {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ContractError(f"{path}:{line_number}: blank lines are forbidden")
            try:
                row = json.loads(line, object_pairs_hook=_strict_pairs)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or set(row) != TARGET_FIELDS:
                raise ContractError(
                    f"{path}:{line_number}: target fields must be exactly {sorted(TARGET_FIELDS)}"
                )
            statement_id = row["statement_id"]
            corpus_row = row["canonical_corpus_row_index"]
            if not isinstance(statement_id, str) or not statement_id:
                raise ContractError(f"{path}:{line_number}: invalid statement_id")
            if isinstance(corpus_row, bool) or not isinstance(corpus_row, int) or corpus_row < 0:
                raise ContractError(f"{path}:{line_number}: invalid canonical corpus row")
            if statement_id in seen_ids or corpus_row in seen_rows:
                raise ContractError(f"{path}:{line_number}: duplicate target identity")
            if row["eligible_position"] != len(rows):
                raise ContractError(
                    f"{path}:{line_number}: eligible positions must be ordered and contiguous"
                )
            if not isinstance(row["reader_eligible"], bool):
                raise ContractError(f"{path}:{line_number}: reader_eligible must be boolean")
            for field in (
                "matches_hash",
                "statement_json_sha256",
                "statement_type",
            ):
                if not isinstance(row[field], str) or not row[field]:
                    raise ContractError(f"{path}:{line_number}: invalid {field}")
            if len(row["statement_json_sha256"]) != 64:
                raise ContractError(f"{path}:{line_number}: invalid statement digest")
            seen_ids.add(statement_id)
            seen_rows.add(corpus_row)
            rows.append(row)
    if not rows:
        raise ContractError(f"{path}: target ledger is empty")
    return rows


def _manifest_descriptor(manifest: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = manifest
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ContractError(f"manifest is missing {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, dict):
        raise ContractError(f"manifest {'.'.join(keys)} must be an object")
    return value


def _verify_descriptor(
    path: Path,
    manifest: dict[str, Any],
    *keys: str,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    descriptor = _manifest_descriptor(manifest, *keys)
    expected_sha = descriptor.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ContractError(f"manifest {'.'.join(keys)} has no valid SHA-256")
    observed_sha = _sha256(path)
    if observed_sha != expected_sha:
        raise ContractError(
            f"{path}: digest mismatch; declared {expected_sha}, observed {observed_sha}"
        )
    if expected_rows is not None and descriptor.get("rows") != expected_rows:
        raise ContractError(
            f"{path}: declared rows {descriptor.get('rows')!r} != {expected_rows}"
        )
    return {
        "path": _display_path(path),
        "bytes": path.stat().st_size,
        "rows": descriptor.get("rows"),
        "sha256": observed_sha,
        "verification": "pass",
    }


def _scan_targets(
    corpus_path: Path, targets: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_row = {row["canonical_corpus_row_index"]: row for row in targets}
    selected: dict[int, dict[str, Any]] = {}
    corpus_rows = 0
    evidence_entries = 0
    try:
        with gzip.open(corpus_path, "rb") as handle:
            for row_index, statement in enumerate(
                ijson.items(handle, "item", use_float=True)
            ):
                if not isinstance(statement, dict):
                    raise ContractError(f"corpus row {row_index}: expected an object")
                evidence = statement.get("evidence")
                if not isinstance(evidence, list):
                    raise ContractError(f"corpus row {row_index}: evidence must be an array")
                corpus_rows += 1
                evidence_entries += len(evidence)
                target = by_row.get(row_index)
                if target is None:
                    continue
                if row_index in selected:
                    raise ContractError(f"corpus target row {row_index} occurred twice")
                if statement.get("id") != target["statement_id"]:
                    raise ContractError(f"corpus row {row_index}: statement UUID mismatch")
                if str(statement.get("matches_hash")) != target["matches_hash"]:
                    raise ContractError(f"corpus row {row_index}: matches hash mismatch")
                if statement.get("type") != target["statement_type"]:
                    raise ContractError(f"corpus row {row_index}: statement type mismatch")
                digest = _sha256_bytes(_canonical_bytes(statement))
                if digest != target["statement_json_sha256"]:
                    raise ContractError(
                        f"corpus row {row_index}: statement JSON digest mismatch"
                    )
                selected[row_index] = statement
    except (OSError, gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
        raise ContractError(f"invalid canonical corpus stream {corpus_path}: {exc}") from exc
    missing = sorted(set(by_row) - set(selected))
    if missing:
        raise ContractError(f"canonical corpus is missing {len(missing)} targets: {missing[:5]}")
    ordered = [selected[row["canonical_corpus_row_index"]] for row in targets]
    return ordered, {"rows": corpus_rows, "evidence_entries": evidence_entries}


def _statement_source_counts(statement: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(ev.source_api or "") for ev in statement.evidence).items()))


def _jsonl_descriptor(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "sha256": _sha256(path),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row) + b"\n")


def _module_identity(obj: Any) -> dict[str, Any]:
    path = Path(inspect.getsourcefile(obj) or inspect.getfile(obj)).resolve()
    return {"path": _display_path(path), "sha256": _sha256(path)}


def _paper_corpus_descriptor(
    corpus_path: Path, paper_manifest: dict[str, Any]
) -> dict[str, Any]:
    files = paper_manifest.get("files")
    if not isinstance(files, list):
        raise ContractError("paper manifest files must be an array")
    matches = [
        item
        for item in files
        if isinstance(item, dict)
        and item.get("filename") == "indra_benchmark_corpus.json.gz"
    ]
    if len(matches) != 1:
        raise ContractError("paper manifest must identify one canonical JSON corpus")
    descriptor = matches[0]
    expected_sha = descriptor.get("sha256")
    expected_bytes = descriptor.get("bytes")
    observed_sha = _sha256(corpus_path)
    if observed_sha != expected_sha or corpus_path.stat().st_size != expected_bytes:
        raise ContractError("canonical corpus size/digest does not match the paper manifest")
    return {
        "path": _display_path(corpus_path),
        "bytes": corpus_path.stat().st_size,
        "sha256": observed_sha,
        "verification": "pass",
    }


def materialize(
    *,
    corpus_path: Path,
    paper_manifest_path: Path,
    targets_path: Path,
    targets_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    corpus_path = corpus_path.resolve()
    paper_manifest_path = paper_manifest_path.resolve()
    targets_path = targets_path.resolve()
    targets_manifest_path = targets_manifest_path.resolve()
    output_dir = output_dir.resolve()
    paper_manifest = _read_json(paper_manifest_path)
    targets_manifest = _read_json(targets_manifest_path)
    targets = _read_targets(targets_path)
    target_input = _verify_descriptor(
        targets_path,
        targets_manifest,
        "outputs",
        "paper_prediction_targets",
        expected_rows=len(targets),
    )
    corpus_input = _paper_corpus_descriptor(corpus_path, paper_manifest)

    started = time.perf_counter()
    statement_jsons, scan_counts = _scan_targets(corpus_path, targets)
    statements = []
    for index, statement_json in enumerate(statement_jsons):
        # The release carries a pre-existing statement belief.  It is useful
        # corpus provenance but is not an admissible input to a newly computed
        # scorer arm.  Remove it before INDRA deserialization.
        scorer_json = dict(statement_json)
        scorer_json.pop("belief", None)
        decoded = stmts_from_json([scorer_json])
        if len(decoded) != 1:
            raise ContractError(f"target {index}: INDRA deserialization did not return one statement")
        statement = decoded[0]
        target = targets[index]
        if statement.uuid != target["statement_id"]:
            raise ContractError(f"target {index}: deserialized UUID mismatch")
        if str(statement.get_hash(shallow=True)) != target["matches_hash"]:
            raise ContractError(f"target {index}: deserialized matches hash mismatch")
        statements.append(statement)

    scorer = SimpleScorer()
    scorer.check_prior_probs(statements)
    scores = scorer.score_statements(statements)
    if len(scores) != len(targets):
        raise ContractError("SimpleScorer returned incomplete coverage")

    predictions: list[dict[str, Any]] = []
    reader_predictions: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for target, statement, raw_score in zip(targets, statements, scores, strict=True):
        score = float(raw_score)
        if not numpy.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ContractError(f"{target['statement_id']}: invalid SimpleScorer probability")
        prediction = {
            "probability_correct": score,
            "statement_id": target["statement_id"],
        }
        predictions.append(prediction)
        if target["reader_eligible"]:
            reader_predictions.append(prediction)
        provenance.append(
            {
                "arm_id": ARM_ID,
                "canonical_corpus_row_index": target["canonical_corpus_row_index"],
                "evidence_count": len(statement.evidence),
                "matches_hash": target["matches_hash"],
                "negated_evidence_count": sum(
                    bool(ev.epistemics.get("negated")) for ev in statement.evidence
                ),
                "probability_correct": score,
                "source_counts": _statement_source_counts(statement),
                "statement_id": target["statement_id"],
                "statement_type": target["statement_type"],
            }
        )
    elapsed = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".current-indra-simple-", dir=output_dir) as tmp:
        stage = Path(tmp)
        prediction_stage = stage / PREDICTIONS_FILENAME
        reader_prediction_stage = stage / READER_PREDICTIONS_FILENAME
        provenance_stage = stage / PROVENANCE_FILENAME
        _write_jsonl(prediction_stage, predictions)
        _write_jsonl(reader_prediction_stage, reader_predictions)
        _write_jsonl(provenance_stage, provenance)
        prediction_path = output_dir / PREDICTIONS_FILENAME
        reader_prediction_path = output_dir / READER_PREDICTIONS_FILENAME
        provenance_path = output_dir / PROVENANCE_FILENAME
        os.replace(prediction_stage, prediction_path)
        os.replace(reader_prediction_stage, reader_prediction_path)
        os.replace(provenance_stage, provenance_path)

    belief_module = _module_identity(SimpleScorer)
    default_resource = Path(inspect.getsourcefile(SimpleScorer) or "").resolve().parent
    default_resource = default_resource.parent / "resources" / "default_belief_probs.json"
    if not default_resource.is_file():
        raise ContractError(f"could not locate current default prior resource: {default_resource}")
    score_counts = Counter(round(row["probability_correct"], 15) for row in predictions)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "arm": {
            "arm_id": ARM_ID,
            "family": "current_indra_simple_noise",
            "class": "indra.belief.SimpleScorer",
            "official_status": ["current_release", "library_default"],
            "prediction_unit": "assembled_statement",
            "output_semantics": "probability_statement_correct",
            "input_projection": "direct evidence on each frozen statement; raw evidence multiplicity retained",
            "hierarchy_propagation": False,
            "training_required": False,
            "not_equivalent_to": [
                "paper OrigBeliefStmt",
                "current CoGEx production HybridScorer",
                "SimpleScorer hierarchy-propagated belief",
            ],
        },
        "prediction_blinding": {
            "status": "pass",
            "adapter_inputs": ["canonical corpus", "label-free prediction target ledger"],
            "forbidden_inputs_not_consumed": [
                "released gold ledger",
                "strict E0 gold ledger",
                "curations",
                "fold assignments",
                "paper predictions",
            ],
            "stored_statement_belief": "present in the canonical source but removed before scorer deserialization",
        },
        "coverage": {
            "target_statements": len(targets),
            "predicted_statements": len(predictions),
            "missing": 0,
            "invalid": 0,
            "reader_eligible_statements": sum(row["reader_eligible"] for row in targets),
            "evidence_entries": sum(row["evidence_count"] for row in provenance),
            "negated_evidence_entries": sum(
                row["negated_evidence_count"] for row in provenance
            ),
            "distinct_probability_values": len(score_counts),
        },
        "inputs": {
            "canonical_corpus": {**corpus_input, "scan_counts": scan_counts},
            "paper_manifest": {
                "path": _display_path(paper_manifest_path),
                "bytes": paper_manifest_path.stat().st_size,
                "sha256": _sha256(paper_manifest_path),
            },
            "prediction_targets": target_input,
            "prediction_targets_manifest": {
                "path": _display_path(targets_manifest_path),
                "bytes": targets_manifest_path.stat().st_size,
                "sha256": _sha256(targets_manifest_path),
            },
        },
        "implementation": {
            "indra_version": importlib.metadata.version("indra"),
            "indra_belief_module": belief_module,
            "default_prior_resource": {
                "path": _display_path(default_resource),
                "sha256": _sha256(default_resource),
            },
            "adapter": {
                "path": _display_path(Path(__file__)),
                "sha256": _sha256(Path(__file__)),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "ijson": importlib.metadata.version("ijson"),
                "platform": platform.platform(),
                "executable": sys.executable,
            },
        },
        "runtime_observation": {
            "wall_seconds": elapsed,
            "inference_usd": None,
            "cost_status": "local CPU execution observed; monetary compute cost not priced",
        },
        "outputs": {
            "predictions": _jsonl_descriptor(prediction_path, predictions),
            "reader_eligible_predictions": _jsonl_descriptor(
                reader_prediction_path, reader_predictions
            ),
            "prediction_provenance": _jsonl_descriptor(provenance_path, provenance),
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_dir, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--paper-manifest", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--targets-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = materialize(
        corpus_path=args.corpus,
        paper_manifest_path=args.paper_manifest,
        targets_path=args.targets,
        targets_manifest_path=args.targets_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({"arm": manifest["arm"], "coverage": manifest["coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
