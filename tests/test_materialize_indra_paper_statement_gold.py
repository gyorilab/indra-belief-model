from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import materialize_indra_paper_statement_gold as paper_gold  # noqa: E402

from materialize_indra_paper_statement_gold import (  # noqa: E402
    ADJUDICATION_QUEUE_FILENAME,
    EVIDENCE_ADJUDICATION_FILENAME,
    HISTORICAL_ALL_SOURCE_ORDER,
    MANIFEST_FILENAME,
    READER_SOURCES,
    RELEASE_EXPECTATIONS,
    STATEMENT_GOLD_FILENAME,
    AuditError,
    CorpusExpectation,
    EligibleExpectation,
    FileExpectation,
    MaterializationExpectations,
    ResultExpectation,
    _curation_audit_counts,
    _iter_json_array,
    materialize_paper_statement_gold,
)


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _file_expectation(payload: bytes) -> FileExpectation:
    return FileExpectation(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _evidence(source_api: str, source_hash: int | None, text: str | None) -> dict:
    row = {"source_api": source_api, "text": text, "text_refs": {"PMID": "1"}}
    if source_hash is not None:
        row["source_hash"] = source_hash
    return row


def _source_vectors(evidence: list[dict]) -> tuple[list[int], list[int]]:
    counts = Counter(row["source_api"] for row in evidence)
    return (
        [counts[source] for source in HISTORICAL_ALL_SOURCE_ORDER],
        [counts[source] for source in READER_SOURCES],
    )


def _write_fixture(
    tmp_path: Path,
    *,
    frozen_label_override: dict[int, int] | None = None,
) -> tuple[dict[str, Path], MaterializationExpectations]:
    frozen_label_override = frozen_label_override or {}
    statements = [
        {
            "type": "Activation",
            "id": "stmt-101",
            "matches_hash": "101",
            "evidence": [
                _evidence("reach", 1001, "positive duplicate a"),
                _evidence("reach", 1001, "positive duplicate b"),
                _evidence("hprd", 1002, "unreviewed but statement already positive"),
            ],
        },
        {
            "type": "Inhibition",
            "id": "stmt-102",
            "matches_hash": "102",
            "evidence": [_evidence("hprd", 2001, "complete negative")],
        },
        {
            "type": "Complex",
            "id": "stmt-103",
            "matches_hash": "103",
            "evidence": [
                _evidence("sparser", 3001, "reviewed negative"),
                _evidence("sparser", 3002, None),
                _evidence("sparser", None, "unidentifiable evidence"),
            ],
        },
        {
            "type": "Phosphorylation",
            "id": "stmt-104",
            "matches_hash": "104",
            "evidence": [
                _evidence("medscan", 4001, "conflicting review is negative"),
                _evidence("medscan", 4002, "second pair is positive"),
            ],
        },
        {
            "type": "Activation",
            "id": "not-eligible",
            "matches_hash": "999",
            "evidence": [_evidence("reach", 9991, "not selected")],
        },
    ]
    paper_labels = {101: 1, 102: 0, 103: 0, 104: 1}
    paper_labels.update(frozen_label_override)
    eligible_rows = []
    for position, statement in enumerate(statements[:4]):
        pa_hash = int(statement["matches_hash"])
        all_counts, reader_counts = _source_vectors(statement["evidence"])
        eligible_rows.append(
            {
                "source_row_index": position,
                "stmt_hash": str(pa_hash),
                "correct": paper_labels[pa_hash],
                "stmt_type": statement["type"],
                "original_stmt_num": position,
                "reader_eligible": bool(any(reader_counts)),
                "in_multireader_dataset": position != 1,
                "reader_source_counts": reader_counts,
                "historical_all_source_counts": all_counts,
            }
        )

    curations = [
        {
            "id": 1,
            "pa_hash": 101,
            "source_hash": 1001,
            "tag": "correct",
            "curator": "a@example.org",
            "source": "fixture",
            "date": "2020-01-01",
            "text": None,
        },
        {
            "id": 2,
            "pa_hash": 102,
            "source_hash": 2001,
            "tag": "wrong_relation",
            "curator": "b@example.org",
            "source": "fixture",
            "date": "2020-01-02",
            "text": "negative",
        },
        {
            "id": 3,
            "pa_hash": 103,
            "source_hash": 3001,
            "tag": "grounding",
            "curator": "a@example.org",
            "source": "fixture",
            "date": "2020-01-03",
            "text": None,
        },
        {
            "id": 4,
            "pa_hash": 104,
            "source_hash": 4001,
            "tag": "correct",
            "curator": "a@example.org",
            "source": "fixture",
            "date": "2020-01-04",
            "text": None,
        },
        {
            "id": 5,
            "pa_hash": 104,
            "source_hash": 4001,
            "tag": "other",
            "curator": "b@example.org",
            "source": "fixture",
            "date": "2020-01-05",
            "text": "conflict",
        },
        {
            "id": 6,
            "pa_hash": 104,
            "source_hash": 4002,
            "tag": "hypothesis",
            "curator": "a@example.org",
            "source": "fixture",
            "date": "2020-01-06",
            "text": None,
        },
    ]

    corpus_json = json.dumps(statements, ensure_ascii=False).encode("utf-8")
    corpus_payload = gzip.compress(corpus_json, mtime=0)
    curations_payload = _canonical_json(curations)
    eligible_payload = b"".join(_canonical_json(row) + b"\n" for row in eligible_rows)
    positive = sum(row["correct"] for row in eligible_rows)
    reader = [row for row in eligible_rows if row["reader_eligible"]]
    protocol = {
        "schema_version": 1,
        "inputs": {
            "raw_curations": {
                "sha256": hashlib.sha256(curations_payload).hexdigest(),
                "bytes": len(curations_payload),
            }
        },
        "outputs": {
            "eligible_statements": {
                "sha256": hashlib.sha256(eligible_payload).hexdigest(),
                "rows": len(eligible_rows),
                "bytes": len(eligible_payload),
            }
        },
        "eligible_sets": {
            "extended_all_sources": {
                "rows": len(eligible_rows),
                "positive": positive,
                "negative": len(eligible_rows) - positive,
            },
            "reader_only": {
                "rows": len(reader),
                "positive": sum(row["correct"] for row in reader),
                "negative": sum(not row["correct"] for row in reader),
            },
        },
        "source_columns": {
            "reader_order": list(READER_SOURCES),
            "historical_all_source_order": list(HISTORICAL_ALL_SOURCE_ORDER),
        },
        "gold_protocol": {
            "positive_tags": ["act_vs_amt", "correct", "hypothesis"],
            "same_pair_conflict": (
                "negative_wins; pair positive iff all curations positive"
            ),
            "audit_counts": _curation_audit_counts(curations),
        },
    }
    protocol_payload = json.dumps(protocol, indent=2, sort_keys=True).encode("utf-8")

    paths = {
        "corpus": tmp_path / "corpus.json.gz",
        "curations": tmp_path / "curations.json",
        "protocol": tmp_path / "protocol.json",
        "eligible": tmp_path / "eligible.jsonl",
    }
    paths["corpus"].write_bytes(corpus_payload)
    paths["curations"].write_bytes(curations_payload)
    paths["protocol"].write_bytes(protocol_payload)
    paths["eligible"].write_bytes(eligible_payload)

    expectations = MaterializationExpectations(
        corpus=CorpusExpectation(
            file=_file_expectation(corpus_payload),
            rows=5,
            evidence_entries=10,
        ),
        curations=_file_expectation(curations_payload),
        curation_rows=6,
        protocol_manifest=_file_expectation(protocol_payload),
        eligible=EligibleExpectation(
            file=_file_expectation(eligible_payload),
            rows=4,
            positive=positive,
            negative=4 - positive,
            reader_rows=3,
            reader_positive=sum(row["correct"] for row in reader),
            reader_negative=sum(not row["correct"] for row in reader),
        ),
        results=ResultExpectation(
            target_evidence_entries=9,
            target_distinct_evidence_pairs=7,
            unidentifiable_evidence_entries=1,
            reviewed_evidence_entries=6,
            unreviewed_evidence_entries=3,
            eligible_curation_rows=6,
            reviewed_evidence_pairs=5,
            reviewed_positive_pairs=2,
            reviewed_negative_pairs=3,
            conflicting_reviewed_pairs=1,
            unreviewed_evidence_pairs=2,
            evidence_adjudication_rows=8,
            complete_evidence_statements=2,
            adjudicated_positive=2,
            adjudicated_negative=1,
            adjudicated_unresolved=1,
            unresolved_adjudication_queue_rows=2,
            unresolved_adjudication_queue_evidence_entries=2,
            unresolved_adjudication_queue_textless_rows=1,
        ),
    )
    return paths, expectations


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_bounded_json_array_parser_crosses_chunks_and_rejects_trailing_comma() -> None:
    assert list(_iter_json_array(io.StringIO('[{"x":1}, {"x":2}]'), chunk_size=3)) == [
        {"x": 1},
        {"x": 2},
    ]
    with pytest.raises(AuditError, match="trailing comma"):
        list(_iter_json_array(io.StringIO("[1,]"), chunk_size=2))


def _run(paths: dict[str, Path], output: Path, expectations):
    return materialize_paper_statement_gold(
        paths["corpus"],
        paths["curations"],
        paths["protocol"],
        paths["eligible"],
        output,
        expectations=expectations,
    )


def test_materialization_separates_safe_gold_from_paper_policy(tmp_path: Path) -> None:
    paths, expectations = _write_fixture(tmp_path)
    output = tmp_path / "gold"
    manifest = _run(paths, output, expectations)

    assert manifest["counts"]["adjudicated_positive"] == 2
    assert manifest["counts"]["adjudicated_negative"] == 1
    assert manifest["counts"]["adjudicated_unresolved"] == 1
    assert manifest["counts"]["resolved_for_complete_negative_evaluation"] == 3
    assert manifest["complete_evidence_gate"] == {
        "include_statuses": ["positive", "negative"],
        "exclude_statuses": ["unresolved"],
        "positive_may_be_resolved_before_complete_review": True,
        "negative_requires_complete_distinct_evidence_review": True,
        "resolved_rows": 3,
        "excluded_unresolved_rows": 1,
    }
    assert all(manifest["coverage_reconciliation"].values())
    assert manifest["adjudication_queue"]["rows"] == 2
    assert manifest["adjudication_queue"]["corpus_evidence_entries"] == 2
    assert manifest["adjudication_queue"]["textless_rows"] == 1

    statements = _read_jsonl(output / STATEMENT_GOLD_FILENAME)
    assert [row["paper_statement_hash"] for row in statements] == [
        "101",
        "102",
        "103",
        "104",
    ]
    by_hash = {row["paper_statement_hash"]: row for row in statements}

    positive_incomplete = by_hash["101"]
    assert positive_incomplete["evidence_review"][
        "complete_distinct_evidence_review"
    ] is False
    assert positive_incomplete["adjudicated_statement_gold"] == {
        "status": "positive",
        "label": 1,
        "strict_e0_status": "positive",
        "strict_e0_correct": 1,
        "resolution_reason": "at_least_one_reviewed_positive_evidence_pair",
        "include_in_complete_negative_evaluation": True,
    }
    assert positive_incomplete["evidence_review"]["unreviewed_source_hashes"] == [
        "1002"
    ]

    assert by_hash["102"]["adjudicated_statement_gold"]["status"] == "negative"
    unresolved = by_hash["103"]
    assert unresolved["adjudicated_statement_gold"]["status"] == "unresolved"
    assert unresolved["adjudicated_statement_gold"]["label"] is None
    assert unresolved["paper_replication_policy"]["label"] == 0
    assert unresolved["paper_replication_policy"][
        "differs_from_adjudicated_gold"
    ] is True
    assert unresolved["evidence_review"]["unreviewed_source_hashes"] == ["3002"]
    assert unresolved["evidence_review"]["unidentifiable_evidence_positions"] == [2]
    assert len(unresolved["evidence_review"]["unresolved_adjudication_queue_ids"]) == 2

    evidence = _read_jsonl(output / EVIDENCE_ADJUDICATION_FILENAME)
    assert len(evidence) == 8
    duplicate = next(
        row
        for row in evidence
        if row["paper_statement_hash"] == "101" and row["source_hash"] == "1001"
    )
    assert duplicate["corpus_evidence_positions"] == [0, 1]
    assert duplicate["corpus_evidence_entry_count"] == 2
    conflict = next(
        row
        for row in evidence
        if row["paper_statement_hash"] == "104" and row["source_hash"] == "4001"
    )
    assert conflict["review_status"] == "negative"
    assert conflict["same_pair_conflict"] is True
    assert conflict["conflict_resolution"] == "negative_wins"
    assert [row["tag"] for row in conflict["curations"]] == ["correct", "other"]
    unidentified = next(row for row in evidence if row["source_hash"] is None)
    assert unidentified["review_status"] == "unreviewed_unidentifiable"
    assert unidentified["evidence_gold_label"] is None
    assert unidentified["needed_to_resolve_statement"] is True
    positive_unreviewed = next(
        row
        for row in evidence
        if row["paper_statement_hash"] == "101" and row["source_hash"] == "1002"
    )
    assert positive_unreviewed["review_status"] == "unreviewed"
    assert positive_unreviewed["needed_to_resolve_statement"] is False

    queue = _read_jsonl(output / ADJUDICATION_QUEUE_FILENAME)
    assert [row["queue_item_id"] for row in queue] == ["q0000001", "q0000002"]
    assert sum(row["entry_count"] for row in queue) == 2
    metadata_only = next(row for row in queue if row["reviewability"] == "metadata_only")
    assert metadata_only["source_apis"] == ["sparser"]
    assert metadata_only["evidence_variants"][0]["evidence"]["text"] is None
    forbidden = {
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
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value), set())
        return set()

    assert not (keys(queue) & forbidden)

    statement_descriptor = manifest["outputs"]["statement_gold"]
    assert statement_descriptor["sha256"] == hashlib.sha256(
        (output / STATEMENT_GOLD_FILENAME).read_bytes()
    ).hexdigest()


def test_materialized_ledgers_are_deterministic(tmp_path: Path) -> None:
    paths, expectations = _write_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(paths, first, expectations)
    _run(paths, second, expectations)

    assert (first / STATEMENT_GOLD_FILENAME).read_bytes() == (
        second / STATEMENT_GOLD_FILENAME
    ).read_bytes()
    assert (first / EVIDENCE_ADJUDICATION_FILENAME).read_bytes() == (
        second / EVIDENCE_ADJUDICATION_FILENAME
    ).read_bytes()
    assert (first / ADJUDICATION_QUEUE_FILENAME).read_bytes() == (
        second / ADJUDICATION_QUEUE_FILENAME
    ).read_bytes()


def test_input_digest_mismatch_fails_before_output(tmp_path: Path) -> None:
    paths, expectations = _write_fixture(tmp_path)
    paths["curations"].write_bytes(paths["curations"].read_bytes() + b"tampered")
    output = tmp_path / "gold"

    with pytest.raises(AuditError, match="identity mismatch"):
        _run(paths, output, expectations)

    assert not output.exists()


def test_existing_artifact_is_not_overwritten(tmp_path: Path) -> None:
    paths, expectations = _write_fixture(tmp_path)
    output = tmp_path / "gold"
    output.mkdir()
    sentinel = output / MANIFEST_FILENAME
    sentinel.write_text("owner data")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(paths, output, expectations)

    assert sentinel.read_text() == "owner data"
    assert not (output / STATEMENT_GOLD_FILENAME).exists()
    assert not (output / EVIDENCE_ADJUDICATION_FILENAME).exists()
    assert not (output / ADJUDICATION_QUEUE_FILENAME).exists()


def test_publication_failure_rolls_back_every_final_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, expectations = _write_fixture(tmp_path)
    output = tmp_path / "gold"
    real_link = paper_gold.os.link
    calls = 0

    def fail_third_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected publication failure")
        return real_link(source, destination)

    monkeypatch.setattr(paper_gold.os, "link", fail_third_link)
    with pytest.raises(OSError, match="injected publication failure"):
        _run(paths, output, expectations)

    for filename in (
        STATEMENT_GOLD_FILENAME,
        EVIDENCE_ADJUDICATION_FILENAME,
        ADJUDICATION_QUEUE_FILENAME,
        MANIFEST_FILENAME,
    ):
        assert not (output / filename).exists()
    assert not list(output.glob(".*.tmp"))


def test_paper_policy_label_must_recompute_exactly(tmp_path: Path) -> None:
    paths, expectations = _write_fixture(tmp_path, frozen_label_override={102: 1})
    output = tmp_path / "gold"

    with pytest.raises(AuditError, match="paper-policy label mismatch"):
        _run(paths, output, expectations)

    assert not output.exists()


REAL_INPUTS = {
    "corpus": ROOT / "data/benchmark/indra_benchmark_corpus.json.gz",
    "curations": ROOT / "data/benchmark/indra_assembly_curations.json",
    "protocol": (
        ROOT
        / "data/results/indra_paper_protocol_20260717/paper_protocol_manifest.json"
    ),
    "eligible": (
        ROOT
        / "data/results/indra_paper_protocol_20260717/paper_eligible_statements.jsonl"
    ),
}


@pytest.mark.skipif(
    not all(path.is_file() for path in REAL_INPUTS.values()),
    reason="frozen canonical corpus/protocol fixture is not present",
)
def test_released_fixture_reconciles_without_demoting_unknowns(tmp_path: Path) -> None:
    output = tmp_path / "released-gold"
    manifest = _run(REAL_INPUTS, output, RELEASE_EXPECTATIONS)

    assert manifest["eligible_universe"]["extended_all_sources_rows"] == 1_689
    assert manifest["eligible_universe"]["reader_only_rows"] == 1_676
    assert manifest["counts"]["paper_policy_positive"] == 1_237
    assert manifest["counts"]["paper_policy_negative"] == 452
    assert manifest["counts"]["adjudicated_positive"] == 1_237
    assert manifest["counts"]["adjudicated_negative"] == 341
    assert manifest["counts"]["adjudicated_unresolved"] == 111
    assert manifest["counts"]["complete_evidence_statements"] == 607
    assert manifest["counts"]["unreviewed_evidence_pairs"] == 27_982
    assert manifest["counts"]["reviewed_evidence_entries"] == 5_516
    assert manifest["counts"]["unreviewed_evidence_entries"] == 28_519
    assert manifest["counts"]["unresolved_adjudication_queue_rows"] == 1_326
    assert (
        manifest["counts"]["unresolved_adjudication_queue_evidence_entries"]
        == 1_341
    )
    assert manifest["counts"]["unresolved_adjudication_queue_textless_rows"] == 10
    assert manifest["counts"][
        "paper_policy_negatives_excluded_as_unresolved"
    ] == 111
    assert all(manifest["coverage_reconciliation"].values())
    statement_rows = _read_jsonl(output / STATEMENT_GOLD_FILENAME)
    assert all(isinstance(row["paper_statement_hash"], str) for row in statement_rows)
    assert any(
        abs(int(row["paper_statement_hash"])) > 2**53 for row in statement_rows
    )
    evidence_rows = _read_jsonl(output / EVIDENCE_ADJUDICATION_FILENAME)
    assert all(
        row["source_hash"] is None or isinstance(row["source_hash"], str)
        for row in evidence_rows
    )
