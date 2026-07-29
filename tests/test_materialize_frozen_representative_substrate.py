"""Focused synthetic tests for the frozen-substrate materializer."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from materialize_frozen_representative_substrate import (  # noqa: E402
    CompressionUnavailable,
    ScanInterrupted,
    _resolve_compression,
    materialize_frozen_substrate,
)


TARGET_A = 10_000_000_000_000_001
TARGET_B = -20_000_000_000_000_002
SOURCE_A = 9_000_000_000_000_007
SOURCE_B = -9_000_000_000_000_008


def _write_gold(path: Path) -> None:
    rows = [
        {
            "matches_hash": TARGET_A,
            "source_hash": SOURCE_A,
            "statement": {
                "type": "Activation",
                "matches_hash": TARGET_A,
                "subj": {"name": "A", "db_refs": {"HGNC": "1"}},
                "obj": {"name": "B", "db_refs": {"HGNC": "2"}},
            },
        },
        {
            "matches_hash": TARGET_B,
            "source_hash": SOURCE_B,
            "statement": {
                "type": "Complex",
                "matches_hash": TARGET_B,
                "members": [
                    {"name": "C", "db_refs": {"HGNC": "3"}},
                    {"name": "D", "db_refs": {"HGNC": "4"}},
                ],
            },
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _evidence(source_api: str, source_hash: int, text: str) -> dict:
    return {
        "source_api": source_api,
        "source_hash": source_hash,
        "text": text,
        "text_refs": {"PMID": "123"},
        "annotations": {"found_by": "synthetic"},
    }


def _write_nodes(path: Path, *, target_b_retracted: str = "true") -> list[dict]:
    payload_a = _evidence("reach", SOURCE_A, "first payload")
    raw_payload_a = json.dumps(payload_a, indent=1, sort_keys=False)
    payload_a_variant = _evidence("reach", SOURCE_A, "second payload variant")
    payload_b = _evidence("sparser", SOURCE_B, "other statement")
    rows = [
        ("indra_evidence:1", "Evidence", raw_payload_a, "false", "reach", TARGET_A),
        # An exact duplicate is intentionally retained as a separate raw entry.
        ("indra_evidence:2", "Evidence", raw_payload_a, "false", "reach", TARGET_A),
        (
            "indra_evidence:3",
            "Evidence",
            _evidence("reach", 777, "non-target"),
            "false",
            "reach",
            777,
        ),
        (
            "indra_evidence:4",
            "Evidence",
            payload_a_variant,
            "false",
            "reach",
            TARGET_A,
        ),
        (
            "indra_evidence:5",
            "Evidence",
            payload_b,
            target_b_retracted,
            "sparser",
            TARGET_B,
        ),
    ]
    header = [
        "id:ID",
        ":LABEL",
        "evidence:string",
        "retracted:boolean",
        "source_api:string",
        "stmt_hash:int",
    ]

    def write_rows(fh, selected_rows, *, include_header: bool) -> None:
        writer = csv.writer(fh, delimiter="\t")
        if include_header:
            writer.writerow(header)
        for node_id, label, payload, retracted, source_api, stmt_hash in selected_rows:
            writer.writerow(
                (
                    node_id,
                    label,
                    payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True),
                    retracted,
                    source_api,
                    stmt_hash,
                )
            )

    # EvidenceProcessor writes the first batch with ``wt`` and later batches
    # with ``at``, so the real frozen file is a concatenated gzip stream.
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        write_rows(fh, rows[:2], include_header=True)
    with gzip.open(path, "at", encoding="utf-8", newline="") as fh:
        write_rows(fh, rows[2:], include_header=False)
    return [payload_a, payload_a, payload_a_variant, payload_b]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_rows(output_dir: Path, manifest: dict, name: str) -> list[dict]:
    descriptor = manifest["files"][name]
    path = output_dir / descriptor["path"]
    if descriptor["compression"] == "gzip":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    elif descriptor["compression"] == "none":
        opener = lambda: path.open(encoding="utf-8")
    else:  # pragma: no cover - these tests request gzip explicitly
        raise AssertionError(descriptor["compression"])
    with opener() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _base_inputs(tmp_path: Path) -> tuple[Path, Path]:
    gold = tmp_path / "gold.jsonl"
    nodes = tmp_path / "nodes_Evidence.tsv.gz"
    _write_gold(gold)
    _write_nodes(nodes)
    return gold, nodes


def test_stream_preserves_entries_variants_and_resumes(tmp_path: Path):
    gold, nodes = _base_inputs(tmp_path)
    output = tmp_path / "bundle"
    kwargs = {
        "gold_path": gold,
        "nodes_evidence_path": nodes,
        "output_dir": output,
        "compression": "gzip",
        "expected_target_count": 2,
        "expected_nodes_rows": 5,
        "expected_nodes_sha256": _sha256(nodes),
        "checkpoint_every": 1,
    }

    with pytest.raises(ScanInterrupted, match="after 2 input rows"):
        materialize_frozen_substrate(**kwargs, _stop_after_scan_rows=2)
    checkpoint = json.loads(
        (output / ".frozen-substrate.work" / "checkpoint.json").read_text()
    )
    assert checkpoint["rows_scanned"] == 2
    assert checkpoint["entries_written"] == 2
    assert not (output / "manifest.json").exists()

    result = materialize_frozen_substrate(**kwargs)
    assert result.status == "created"
    assert not (output / ".frozen-substrate.work").exists()
    manifest = result.manifest
    assert manifest["inputs"]["nodes_evidence"]["sha256"] == _sha256(nodes)
    assert manifest["counts"] == {
        "input_evidence_rows_scanned": 5,
        "statements": 2,
        "matched_evidence_entries": 4,
        "unique_exact_pairs": 2,
        "duplicate_entry_multiplicity": 2,
        "target_hashes_covered": 2,
        "gold_exact_pairs_covered": 2,
    }
    assert manifest["source_distribution"] == {"reach": 3, "sparser": 1}
    assert manifest["payload_digests"]["unique_raw_payloads"] == 3
    assert manifest["payload_digests"]["unique_normalized_payloads"] == 3
    assert (
        manifest["payload_digests"][
            "pairs_with_multiple_normalized_payload_variants"
        ]
        == 1
    )
    assert manifest["reconciliation"] == {
        "entries_equal_entry_source_api_sum": True,
        "unique_pairs_equal_pair_source_api_sum": True,
        "entries_equal_unique_pairs_plus_duplicate_multiplicity": True,
        "every_pair_has_one_source_api": True,
        "all_target_hashes_covered": True,
        "all_gold_exact_pairs_covered": True,
        "checkpoint_entries_equal_spool_entries": True,
    }

    entries = _artifact_rows(output, manifest, "evidence_entries")
    assert [row["entry_id"] for row in entries] == [
        "e000000000001",
        "e000000000002",
        "e000000000003",
        "e000000000004",
    ]
    assert [row["input_row"] for row in entries] == [1, 2, 4, 5]
    assert all(isinstance(row["matches_hash"], str) for row in entries)
    assert all(isinstance(row["source_hash"], str) for row in entries)
    assert all(isinstance(row["evidence"]["source_hash"], str) for row in entries)
    expected_raw = json.dumps(
        _evidence("reach", SOURCE_A, "first payload"),
        indent=1,
        sort_keys=False,
    )
    assert entries[0]["raw_evidence_json"] == expected_raw
    assert entries[1]["raw_evidence_json"] == expected_raw
    assert entries[0]["raw_payload_sha256"] == hashlib.sha256(
        expected_raw.encode("utf-8")
    ).hexdigest()
    assert entries[0]["cogex_retracted"] is False
    assert entries[0]["cogex_retracted_raw"] == "false"
    assert entries[-1]["cogex_retracted"] is True
    assert entries[-1]["cogex_retracted_raw"] == "true"
    assert entries[0]["cogex_fields"]["retracted:boolean"] == "false"
    assert entries[0]["cogex_fields"]["stmt_hash:int"] == str(TARGET_A)
    assert entries[0]["payload_sha256"] == entries[1]["payload_sha256"]
    assert entries[0]["raw_payload_sha256"] == entries[1]["raw_payload_sha256"]

    pairs = _artifact_rows(output, manifest, "pair_index")
    pair_a = next(row for row in pairs if row["matches_hash"] == str(TARGET_A))
    assert pair_a["source_hash"] == str(SOURCE_A)
    assert pair_a["source_api"] == "reach"
    assert pair_a["entry_count"] == 3
    assert pair_a["payload_variant_count"] == 2
    assert pair_a["entry_ids"] == [
        "e000000000001",
        "e000000000002",
        "e000000000003",
    ]

    statements = _artifact_rows(output, manifest, "statements")
    assert {row["matches_hash"] for row in statements} == {
        str(TARGET_A),
        str(TARGET_B),
    }
    assert all(
        isinstance(row["statement"]["matches_hash"], str) for row in statements
    )
    assert all(
        all(isinstance(value, str) for value in row["gold_source_hashes"])
        for row in statements
    )

    source_accounting = manifest["source_accounting"]
    assert source_accounting["evidence_entry_source_api_counts"] == {
        "reach": 3,
        "sparser": 1,
    }
    assert source_accounting["unique_pair_source_api_counts"] == {
        "reach": 1,
        "sparser": 1,
    }
    assert source_accounting["per_target"][str(TARGET_A)] == {
        "evidence_entry_source_api_counts": {"reach": 3},
        "unique_pair_source_api_counts": {"reach": 1},
    }
    assert all(source_accounting["reconciliation"].values())
    assert source_accounting["api_audit_comparison_contract"]["pair_key"].startswith(
        "(matches_hash, source_hash)"
    )

    reused = materialize_frozen_substrate(**kwargs)
    assert reused.status == "reused"
    assert reused.manifest["bundle_id"] == manifest["bundle_id"]


def test_scan_only_checkpoint_can_be_enriched_without_republishing(tmp_path: Path):
    gold, nodes = _base_inputs(tmp_path)
    output = tmp_path / "scan-then-enrich"
    kwargs = {
        "gold_path": gold,
        "nodes_evidence_path": nodes,
        "output_dir": output,
        "compression": "gzip",
        "expected_target_count": 2,
        "expected_nodes_rows": 5,
        "expected_nodes_sha256": _sha256(nodes),
        "checkpoint_every": 1,
    }

    scanned = materialize_frozen_substrate(**kwargs, scan_only=True)
    assert scanned.status == "scan_complete"
    assert scanned.manifest["status"] == "scan_complete_pending_enrichment"
    assert scanned.manifest["counts"]["matched_evidence_entries"] == 4
    assert scanned.manifest["counts"]["unique_exact_pairs"] == 2
    bundle_id = scanned.manifest["bundle_id"]
    checkpoint = json.loads(
        (output / ".frozen-substrate.work" / "checkpoint.json").read_text()
    )
    assert checkpoint["scan_complete"] is True
    assert checkpoint["bundle_id"] == bundle_id
    assert not (output / "manifest.json").exists()

    completed = materialize_frozen_substrate(**kwargs)
    assert completed.status == "created"
    assert completed.manifest["bundle_id"] == bundle_id
    assert completed.manifest["counts"]["matched_evidence_entries"] == 4
    assert not (output / ".frozen-substrate.work").exists()


def test_optional_refinement_closure_and_hybrid_counts(tmp_path: Path):
    gold, nodes = _base_inputs(tmp_path)
    output = tmp_path / "hybrid-bundle"
    source_counts_path = tmp_path / "source_counts.pkl"
    refinements_path = tmp_path / "refinements.tsv.gz"
    belief_scores_path = tmp_path / "belief_scores.pkl"

    ancestor_a = TARGET_A + 1
    ancestor_a2 = TARGET_A + 2
    ancestor_b = TARGET_B - 1
    source_counts = {
        TARGET_A: {"reach": 3},
        TARGET_B: {"sparser": 1},
        ancestor_a: {"reach": 2, "pc11": 1},
        ancestor_a2: {"sparser": 4},
        ancestor_b: {"signor": 2},
        98: {"irrelevant": 99},
        99: {"irrelevant": 99},
    }
    with source_counts_path.open("wb") as fh:
        pickle.dump(source_counts, fh)
    with belief_scores_path.open("wb") as fh:
        pickle.dump({TARGET_A: 0.71, TARGET_B: 0.82}, fh)
    with gzip.open(refinements_path, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerows(
            [
                (ancestor_a, TARGET_A),
                (ancestor_a2, ancestor_a),
                (ancestor_b, TARGET_B),
                (99, 98),
            ]
        )

    result = materialize_frozen_substrate(
        gold_path=gold,
        nodes_evidence_path=nodes,
        output_dir=output,
        source_counts_path=source_counts_path,
        refinements_path=refinements_path,
        belief_scores_path=belief_scores_path,
        compression="gzip",
        expected_target_count=2,
        expected_nodes_rows=5,
        expected_nodes_sha256=_sha256(nodes),
        checkpoint_every=2,
    )
    manifest = result.manifest
    assert manifest["inputs"]["refinements"]["direction"] == (
        "more_specific_to_less_specific"
    )
    assert manifest["refinement"]["closure_traversal"] == (
        "incoming edges from each target"
    )
    assert manifest["refinement"]["target_ancestor_pairs"] == 3
    assert manifest["refinement"]["targets_with_ancestors"] == 2
    assert all(manifest["refinement"]["assertions"].values())
    assert all(manifest["hybrid_counts"]["assertions"].values())
    assert manifest["hybrid_counts"]["direct_total"] == 4
    assert manifest["hybrid_counts"]["hybrid_total"] == 13

    closure = _artifact_rows(output, manifest, "ancestor_closure")
    assert {
        (row["target_matches_hash"], row["ancestor_more_specific_hash"])
        for row in closure
    } == {
        (str(TARGET_A), str(ancestor_a)),
        (str(TARGET_A), str(ancestor_a2)),
        (str(TARGET_B), str(ancestor_b)),
    }

    hybrid = {
        row["matches_hash"]: row
        for row in _artifact_rows(output, manifest, "hybrid_counts")
    }
    assert hybrid[str(TARGET_A)] == {
        "matches_hash": str(TARGET_A),
        "direct_source_name_counts": {"reach": 3},
        "direct_source_api_counts": {"reach": 3},
        "direct_total": 3,
        "ancestor_hashes": [str(ancestor_a), str(ancestor_a2)],
        "ancestor_source_name_counts": {"pc11": 1, "reach": 2, "sparser": 4},
        "ancestor_source_api_counts": {"pc11": 1, "reach": 2, "sparser": 4},
        "ancestor_total": 7,
        "hybrid_source_name_counts": {"pc11": 1, "reach": 5, "sparser": 4},
        "hybrid_source_api_counts": {"pc11": 1, "reach": 5, "sparser": 4},
        "hybrid_total": 10,
        "frozen_belief": 0.71,
    }
    assert hybrid[str(TARGET_B)]["direct_total"] == 1
    assert hybrid[str(TARGET_B)]["ancestor_total"] == 2
    assert hybrid[str(TARGET_B)]["hybrid_total"] == 3
    source_comparison = manifest["source_accounting"]["source_counts_comparison"]
    assert source_comparison["status"] == "exact_after_source_name_to_api_mapping"
    assert source_comparison["mapped_source_api_counts"] == {
        "reach": 3,
        "sparser": 1,
    }
    assert source_comparison["per_source"] == [
        {
            "source_api": "reach",
            "evidence_entry_count": 3,
            "unique_pair_count": 1,
            "mapped_source_counts_count": 3,
            "source_counts_minus_entries": 0,
        },
        {
            "source_api": "sparser",
            "evidence_entry_count": 1,
            "unique_pair_count": 1,
            "mapped_source_counts_count": 1,
            "source_counts_minus_entries": 0,
        },
    ]
    assert all(source_comparison["assertions"].values())
    assert manifest["hybrid_counts"]["belief_reconciliation"] == {
        "belief_scores_supplied": True,
        "target_coverage_verified": True,
        "parity_to_emitted_hybrid_counts": "not_verified",
        "reason": (
            "the synchronized HybridScorer artifact and source-name mapping "
            "used to recompute probabilities are not part of belief_scores.pkl"
        ),
    }


def test_missing_refinement_source_counts_is_a_hard_coverage_error(tmp_path: Path):
    gold, nodes = _base_inputs(tmp_path)
    output = tmp_path / "invalid-hybrid"
    source_counts_path = tmp_path / "source_counts_missing.pkl"
    refinements_path = tmp_path / "refinements_missing.tsv.gz"
    ancestor = TARGET_A + 1
    with source_counts_path.open("wb") as fh:
        pickle.dump(
            {
                TARGET_A: {"reach": 3},
                TARGET_B: {"sparser": 1},
                # The reachable ancestor is deliberately absent.
            },
            fh,
        )
    with gzip.open(refinements_path, "wt", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerow((ancestor, TARGET_A))

    with pytest.raises(ValueError, match="missing refinement ancestors"):
        materialize_frozen_substrate(
            gold_path=gold,
            nodes_evidence_path=nodes,
            output_dir=output,
            source_counts_path=source_counts_path,
            refinements_path=refinements_path,
            compression="gzip",
            expected_target_count=2,
            expected_nodes_rows=5,
            expected_nodes_sha256=_sha256(nodes),
            checkpoint_every=2,
        )
    assert not (output / "manifest.json").exists()


def test_retracted_cell_is_strictly_typed_and_raw_preserved(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    nodes = tmp_path / "nodes_bad_boolean.tsv.gz"
    _write_gold(gold)
    _write_nodes(nodes, target_b_retracted="False")
    output = tmp_path / "bad-boolean"

    with pytest.raises(ValueError, match="raw cell 'true' or 'false'"):
        materialize_frozen_substrate(
            gold_path=gold,
            nodes_evidence_path=nodes,
            output_dir=output,
            compression="gzip",
            expected_target_count=2,
            expected_nodes_rows=5,
            expected_nodes_sha256=_sha256(nodes),
            checkpoint_every=2,
        )
    assert not (output / "manifest.json").exists()


def test_source_counts_require_exact_per_source_mapping(tmp_path: Path):
    gold, nodes = _base_inputs(tmp_path)
    source_counts_path = tmp_path / "renamed_source_counts.pkl"
    with source_counts_path.open("wb") as fh:
        pickle.dump(
            {
                TARGET_A: {"reader_reach": 3},
                TARGET_B: {"reader_sparser": 1},
            },
            fh,
        )

    invalid_output = tmp_path / "unmapped-source-counts"
    with pytest.raises(ValueError, match="per-source direct count mismatch"):
        materialize_frozen_substrate(
            gold_path=gold,
            nodes_evidence_path=nodes,
            output_dir=invalid_output,
            source_counts_path=source_counts_path,
            compression="gzip",
            expected_target_count=2,
            expected_nodes_rows=5,
            expected_nodes_sha256=_sha256(nodes),
            checkpoint_every=2,
        )
    assert not (invalid_output / "manifest.json").exists()

    source_map_path = tmp_path / "source_name_to_api.json"
    source_map_path.write_text(
        json.dumps(
            {"reader_reach": "reach", "reader_sparser": "sparser"},
            sort_keys=True,
        )
    )
    valid_output = tmp_path / "mapped-source-counts"
    result = materialize_frozen_substrate(
        gold_path=gold,
        nodes_evidence_path=nodes,
        output_dir=valid_output,
        source_counts_path=source_counts_path,
        source_name_map_path=source_map_path,
        compression="gzip",
        expected_target_count=2,
        expected_nodes_rows=5,
        expected_nodes_sha256=_sha256(nodes),
        checkpoint_every=2,
    )
    comparison = result.manifest["source_accounting"]["source_counts_comparison"]
    assert comparison["mapped_source_api_counts"] == {"reach": 3, "sparser": 1}
    assert comparison["source_name_to_api_mapping"] == {
        "reader_reach": "reach",
        "reader_sparser": "sparser",
    }
    assert all(comparison["assertions"].values())
    assert result.manifest["inputs"]["source_name_map"]["sha256"] == _sha256(
        source_map_path
    )


def test_compression_resolution_has_clear_zstd_fallback_and_error():
    assert _resolve_compression("auto", zstd_available=False) == (
        "gzip",
        "optional 'zstandard' package is not installed; auto fell back to gzip",
    )
    with pytest.raises(CompressionUnavailable, match="install zstandard"):
        _resolve_compression("zstd", zstd_available=False)
    assert _resolve_compression("zstd", zstd_available=True) == ("zstd", None)
