"""Synthetic bundle and immutable real-fixture tests for the E0 harness."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_representative_adjudication_harness import (  # noqa: E402
    DEFAULT_GOLD,
    DEFAULT_GOLD_META,
    DEFAULT_SCAN_WORK,
    REAL_EXPECTED_COUNTS,
    _atomic_publish_directory_noreplace,
    _forbidden_queue_paths,
    analyze_representative_lane,
    build_representative_adjudication_harness,
)


S1, S2, S3 = 1001, 1002, 1003
A, B, C, D, E, F = 2001, 2002, 2003, 2004, 2005, 2006

SYNTHETIC_COUNTS = {
    "gold_exact_pairs": 3,
    "gold_positive_pairs": 1,
    "gold_noncorrect_pairs": 2,
    "statements": 3,
    "positive_statements": 1,
    "negative_statements": 1,
    "unresolved_statements": 1,
    "negative_singleton_statements": 1,
    "raw_evidence_entries": 8,
    "unique_evidence_pairs": 6,
    "queued_unreviewed_pairs": 2,
    "other_unreviewed_pairs_retained_in_ledger": 1,
}


def _digest_json(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _statement(matches_hash: int, name: str, *, inject_blind_fields: bool = False):
    statement = {
        "matches_hash": matches_hash,
        "type": "Activation",
        "subj": {"name": name, "db_refs": {"HGNC": "111"}},
        "obj": {"name": "target", "db_refs": {"HGNC": "9"}},
    }
    if inject_blind_fields:
        statement.update(
            {
                "belief": 0.99,
                "prediction": "positive",
                "nested": {
                    "labels": ["positive"],
                    "curator": "hidden@example.org",
                    "disagreement_reason": "hidden",
                },
            }
        )
    return statement


def _write_gold(path: Path) -> None:
    rows = [
        {
            "matches_hash": S1,
            "source_hash": A,
            "statement": _statement(S1, "positive"),
            "tag": "correct",
            "gold": "correct",
            "gold_status": "canonical_first_submission",
            "curator": "curator@example.org",
            "curation_id": 1,
            "curation_date": "2026-01-01",
            "curation_source": "synthetic viewer",
        },
        {
            "matches_hash": S2,
            "source_hash": C,
            "statement": _statement(S2, "negative"),
            "tag": "grounding",
            "gold": "incorrect",
            "gold_status": "canonical_first_submission",
            "curator": "curator@example.org",
            "curation_id": 2,
            "curation_date": "2026-01-02",
            "curation_source": "synthetic viewer",
        },
        {
            "matches_hash": S3,
            "source_hash": D,
            "statement": _statement(S3, "unresolved", inject_blind_fields=True),
            "tag": "wrong_relation",
            "gold": "incorrect",
            "gold_status": "canonical_first_submission",
            "curator": "curator@example.org",
            "curation_id": 3,
            "curation_date": "2026-01-03",
            "curation_source": "synthetic viewer",
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _entry(
    number: int,
    matches_hash: int,
    source_hash: int,
    source_api: str,
    text: str,
    *,
    variant: int = 1,
    inject_blind_fields: bool = False,
) -> dict:
    evidence = {
        "source_api": source_api,
        "source_hash": str(source_hash),
        "text": text,
        "text_refs": {"PMID": str(10000 + number)},
        "annotations": {"variant": variant},
    }
    if inject_blind_fields:
        evidence["annotations"].update(
            {
                "belief": 0.87,
                "prediction_label": "incorrect",
                "tags": ["gold"],
                "curator": "hidden@example.org",
                "nested": {
                    "source_hash": str(source_hash),
                    "disagreement": True,
                },
            }
        )
    raw = json.dumps(evidence, indent=1, sort_keys=False)
    payload_sha256 = _digest_json(evidence)
    raw_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return {
        "entry_id": f"e{number:012d}",
        "input_row": number * 10,
        "matches_hash": str(matches_hash),
        "source_hash": str(source_hash),
        "source_api": source_api,
        "raw_payload_bytes": len(raw.encode("utf-8")),
        "raw_payload_sha256": raw_sha256,
        "payload_sha256": payload_sha256,
        "raw_evidence_json": raw,
        "evidence": evidence,
        "cogex_retracted": False,
        "cogex_retracted_raw": "false",
        "cogex_fields": {
            "id:ID": f"indra_evidence:{number}",
            ":LABEL": "Evidence",
            "retracted:boolean": "false",
            "source_api:string": source_api,
            "stmt_hash:int": str(matches_hash),
        },
        "cogex_evidence_id": f"indra_evidence:{number}",
        "cogex_labels": "Evidence",
    }


def _pair_row(entries: list[dict]) -> dict:
    first = entries[0]
    by_payload: dict[str, list[dict]] = {}
    for entry in entries:
        by_payload.setdefault(entry["payload_sha256"], []).append(entry)
    variants = []
    for payload_sha256, payload_entries in by_payload.items():
        raw_counts: dict[str, int] = {}
        for entry in payload_entries:
            raw_counts[entry["raw_payload_sha256"]] = (
                raw_counts.get(entry["raw_payload_sha256"], 0) + 1
            )
        variants.append(
            {
                "payload_sha256": payload_sha256,
                "entry_count": len(payload_entries),
                "raw_payloads": [
                    {"raw_payload_sha256": digest, "entry_count": count}
                    for digest, count in raw_counts.items()
                ],
            }
        )
    return {
        "matches_hash": first["matches_hash"],
        "source_hash": first["source_hash"],
        "source_api": first["source_api"],
        "entry_count": len(entries),
        "entry_ids": [entry["entry_id"] for entry in entries],
        "payload_variant_count": len(variants),
        "payload_variants": variants,
    }


def _write_scan_fixture(work_dir: Path) -> None:
    work_dir.mkdir()
    e5 = _entry(
        5,
        S3,
        E,
        "sparser",
        "unreviewed variant one",
        variant=1,
        inject_blind_fields=True,
    )
    e6 = copy.deepcopy(e5)
    e6["entry_id"] = "e000000000006"
    e6["input_row"] = 60
    e6["cogex_evidence_id"] = "indra_evidence:6"
    e6["cogex_fields"]["id:ID"] = "indra_evidence:6"
    e7 = copy.deepcopy(e5)
    e7["entry_id"] = "e000000000007"
    e7["input_row"] = 70
    e7["cogex_evidence_id"] = "indra_evidence:7"
    e7["cogex_fields"]["id:ID"] = "indra_evidence:7"
    e7["evidence"]["annotations"]["belief"] = 0.12
    e7_raw = json.dumps(e7["evidence"], indent=1, sort_keys=False)
    e7["raw_evidence_json"] = e7_raw
    e7["raw_payload_bytes"] = len(e7_raw.encode("utf-8"))
    e7["raw_payload_sha256"] = hashlib.sha256(e7_raw.encode("utf-8")).hexdigest()
    e7["payload_sha256"] = _digest_json(e7["evidence"])
    entries_by_pair = {
        (S1, A): [_entry(1, S1, A, "reach", "reviewed positive")],
        (S1, B): [_entry(2, S1, B, "sparser", "unreviewed but E0-resolved")],
        (S2, C): [_entry(3, S2, C, "reach", "reviewed singleton negative")],
        (S3, D): [_entry(4, S3, D, "reach", "reviewed noncorrect")],
        (S3, E): [
            e5,
            e6,
            e7,
        ],
        (S3, F): [_entry(8, S3, F, "bel", "second unresolved pair")],
    }
    spool = work_dir / "evidence_entries.jsonl"
    with spool.open("w", encoding="utf-8") as fh:
        for pair in sorted(entries_by_pair):
            for entry in entries_by_pair[pair]:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
    pair_index = work_dir / "pair_index.jsonl"
    with pair_index.open("w", encoding="utf-8") as fh:
        for pair in sorted(entries_by_pair):
            fh.write(json.dumps(_pair_row(entries_by_pair[pair]), sort_keys=True) + "\n")
    checkpoint = {
        "schema_version": 1,
        "scan_complete": True,
        "scan_fingerprint": "synthetic-scan",
        "spool_bytes": spool.stat().st_size,
        "entries_written": 8,
        "rows_scanned": 100,
        "nodes_sha256": "0" * 64,
    }
    (work_dir / "checkpoint.json").write_text(json.dumps(checkpoint))
    summary = {
        "schema_version": 1,
        "status": "scan_complete_pending_enrichment",
        "scan_fingerprint": "synthetic-scan",
        "counts": {
            "matched_evidence_entries": 8,
            "unique_exact_pairs": 6,
            "gold_exact_pairs_covered": 3,
        },
    }
    (work_dir / "scan_summary.json").write_text(json.dumps(summary))


def _rows(output: Path, manifest: dict, name: str) -> list[dict]:
    descriptor = manifest["files"][name]
    path = output / descriptor["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]
    if descriptor["compression"] == "gzip":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    elif descriptor["compression"] == "none":
        opener = lambda: path.open(encoding="utf-8")
    else:  # pragma: no cover - synthetic test requests gzip
        raise AssertionError(descriptor["compression"])
    with opener() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _opaque_factory():
    counter = 0

    def factory():
        nonlocal counter
        counter += 1
        return f"q_synthetic_{counter:08d}"

    return factory


def test_synthetic_e0_ledger_queue_blinding_and_no_clobber(tmp_path: Path):
    gold = tmp_path / "gold.jsonl"
    work = tmp_path / "scan-work"
    output = tmp_path / "harness"
    _write_gold(gold)
    _write_scan_fixture(work)

    analysis = analyze_representative_lane(
        gold_path=gold,
        scan_work_dir=work,
        gold_meta_path=None,
        expected_counts=SYNTHETIC_COUNTS,
    )
    assert analysis.counts == SYNTHETIC_COUNTS
    assert analysis.statement_stats[str(S1)].e0_label == "positive"
    assert analysis.statement_stats[str(S2)].e0_label == "negative"
    assert analysis.statement_stats[str(S3)].e0_label == "unresolved"
    assert analysis.statement_stats[str(S3)].queued_pairs == 2

    result = build_representative_adjudication_harness(
        output_dir=output,
        gold_path=gold,
        scan_work_dir=work,
        gold_meta_path=None,
        compression="gzip",
        expected_counts=SYNTHETIC_COUNTS,
        opaque_id_factory=_opaque_factory(),
    )
    manifest = result.manifest
    assert manifest["counts"] == SYNTHETIC_COUNTS
    assert all(manifest["reconciliation"].values())
    assert manifest["selection_randomness_caveat"] == {
        "historical_completed_subset_is_provably_srs": False,
        "reason": (
            "the historical viewer retained no draw/skip log and retried "
            "unmaterializable or textless rows; reservoir membership is "
            "auditable, but the completed curation subset is not proven to "
            "be a simple random sample"
        ),
    }
    assert (output / "manifest.json").is_file()
    assert not list(tmp_path.glob(".harness.stage-*"))

    statement_gold = {
        row["matches_hash"]: row for row in _rows(output, manifest, "statement_gold")
    }
    assert {key: row["e0_label"] for key, row in statement_gold.items()} == {
        str(S1): "positive",
        str(S2): "negative",
        str(S3): "unresolved",
    }
    assert statement_gold[str(S1)]["adjudication_pairs_needed"] == 0
    assert statement_gold[str(S2)]["unique_evidence_pairs"] == 1
    assert statement_gold[str(S3)]["adjudication_pairs_needed"] == 2
    assert statement_gold[str(S3)]["exact_pair_curations"][0]["curator"] == (
        "curator@example.org"
    )

    ledger = _rows(output, manifest, "evidence_pair_ledger")
    assert len(ledger) == 6
    by_pair = {
        (row["matches_hash"], row["source_hash"]): row for row in ledger
    }
    assert by_pair[(str(S1), str(A))]["pair_gold_label"] == "positive"
    assert by_pair[(str(S1), str(A))]["curation"]["curation_id"] == 1
    assert by_pair[(str(S1), str(B))]["reviewed"] is False
    assert by_pair[(str(S1), str(B))]["needed_for_e0_adjudication"] is False
    assert by_pair[(str(S1), str(B))]["curation"] is None
    assert by_pair[(str(S2), str(C))]["statement_e0_label"] == "negative"
    assert by_pair[(str(S2), str(C))]["raw_entry_count"] == 1
    unresolved_variant = by_pair[(str(S3), str(E))]
    assert unresolved_variant["needed_for_e0_adjudication"] is True
    assert unresolved_variant["raw_entry_count"] == 3
    assert unresolved_variant["normalized_payload_variant_count"] == 2
    assert unresolved_variant["raw_payload_variant_count"] == 2
    assert len(unresolved_variant["entries"]) == 3
    assert unresolved_variant["entries"][0]["raw_evidence_json"] == (
        unresolved_variant["entries"][1]["raw_evidence_json"]
    )
    assert unresolved_variant["entries"][0]["raw_payload_sha256"] == (
        unresolved_variant["entries"][1]["raw_payload_sha256"]
    )

    queue = _rows(output, manifest, "adjudication_queue")
    mapping = _rows(output, manifest, "queue_mapping")
    assert len(queue) == len(mapping) == 2
    assert {row["queue_id"] for row in queue} == {
        row["queue_id"] for row in mapping
    }
    assert {
        (row["matches_hash"], row["source_hash"]) for row in mapping
    } == {(str(S3), str(E)), (str(S3), str(F))}
    for row in queue:
        assert not _forbidden_queue_paths(row)
        serialized = json.dumps(row, sort_keys=True)
        for linkage in (str(S1), str(S2), str(S3), str(A), str(B), str(C), str(D), str(E), str(F)):
            assert linkage not in serialized
        assert "curator@example.org" not in serialized
        assert "hidden@example.org" not in serialized
        assert "positive" not in serialized
        assert "incorrect" not in serialized
    variant_queue = next(
        row
        for row in queue
        if row["raw_entry_multiplicity"] == 3
    )
    # The ledger retains two normalized variants, but they differ only in a
    # forbidden belief field and therefore collapse to one visible queue row.
    assert variant_queue["payload_variant_count"] == 1
    assert [
        variant["raw_entry_multiplicity"]
        for variant in variant_queue["evidence_variants"]
    ] == [3]

    manifest_before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to clobber"):
        build_representative_adjudication_harness(
            output_dir=output,
            gold_path=gold,
            scan_work_dir=work,
            gold_meta_path=None,
            compression="gzip",
            expected_counts=SYNTHETIC_COUNTS,
        )
    assert (output / "manifest.json").read_bytes() == manifest_before


def test_atomic_publish_directory_never_replaces_destination(tmp_path: Path):
    stage = tmp_path / "stage"
    destination = tmp_path / "published"
    stage.mkdir()
    (stage / "manifest.json").write_text("complete")
    destination.mkdir()

    with pytest.raises(FileExistsError, match="refusing to clobber"):
        _atomic_publish_directory_noreplace(stage, destination)
    assert stage.is_dir()
    assert not list(destination.iterdir())

    destination.rmdir()
    _atomic_publish_directory_noreplace(stage, destination)
    assert not stage.exists()
    assert (destination / "manifest.json").read_text() == "complete"


def test_real_completed_checkpoint_counts_and_immutability():
    required = [
        DEFAULT_GOLD,
        DEFAULT_GOLD_META,
        DEFAULT_SCAN_WORK / "checkpoint.json",
        DEFAULT_SCAN_WORK / "pair_index.jsonl",
        DEFAULT_SCAN_WORK / "evidence_entries.jsonl",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("real completed representative checkpoint is not present")
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in required
    }
    analysis = analyze_representative_lane(
        gold_path=DEFAULT_GOLD,
        scan_work_dir=DEFAULT_SCAN_WORK,
        gold_meta_path=DEFAULT_GOLD_META,
        expected_counts=REAL_EXPECTED_COUNTS,
    )
    assert analysis.counts == REAL_EXPECTED_COUNTS
    negatives = [
        stats
        for stats in analysis.statement_stats.values()
        if stats.e0_label == "negative"
    ]
    assert len(negatives) == 44
    assert all(
        stats.unique_pairs == 1
        and stats.raw_entries == 1
        and stats.singleton_entry_count == 1
        and stats.singleton_payload_variant_count == 1
        for stats in negatives
    )
    assert sum(
        stats.unreviewed_pairs
        for stats in analysis.statement_stats.values()
        if stats.e0_label == "unresolved"
    ) == 62_752
    assert analysis.pair_source_api_counts
    assert analysis.entry_source_api_counts
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in required}
    assert after == before
