from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.freeze_indra_paper_protocol import (
    ELIGIBLE_FILENAME,
    FOLDS_FILENAME,
    HISTORICAL_ALL_SOURCE_ORDER,
    PROTOCOL_FILENAME,
    READER_SOURCES,
    RELEASE_EXPECTATIONS,
    AuditError,
    DatasetExpectation,
    FileExpectation,
    FreezeExpectations,
    RawCurationExpectation,
    _restricted_pickle_loads,
    aggregate_observed_gold,
    freeze_protocol,
    paper_tag_label,
)


class _TouchOnUnpickle:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {self.marker}",)


def _md5(payload: bytes) -> str:
    try:
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()
    except TypeError:  # pragma: no cover - compatibility with older Python
        return hashlib.md5(payload).hexdigest()


def _file_expectation(payload: bytes) -> FileExpectation:
    return FileExpectation(
        bytes=len(payload),
        md5=_md5(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _raw_expectation(payload: bytes, rows: list[dict]) -> RawCurationExpectation:
    pair_labels, statement_labels, conflicts = aggregate_observed_gold(rows)
    pair_positive = sum(pair_labels.values())
    statement_positive = sum(statement_labels.values())
    return RawCurationExpectation(
        file=_file_expectation(payload),
        rows=len(rows),
        unique_statement_hashes=len({int(row["pa_hash"]) for row in rows}),
        unique_statement_evidence_pairs=len(pair_labels),
        unique_source_hashes=len({int(row["source_hash"]) for row in rows}),
        unique_curators=len({str(row["curator"]) for row in rows}),
        observed_pair_positive=pair_positive,
        observed_pair_negative=len(pair_labels) - pair_positive,
        observed_statement_positive=statement_positive,
        observed_statement_negative=len(statement_labels) - statement_positive,
        conflicting_pairs=conflicts,
        tag_counts=tuple(sorted(Counter(str(row["tag"]) for row in rows).items())),
    )


def _write_synthetic_release(tmp_path: Path) -> tuple[dict[str, Path], FreezeExpectations, list[dict]]:
    extended_rows = []
    for index in range(30):
        label = int(index < 15)
        reader_eligible = index < 10 or 15 <= index < 25
        row = {
            "stmt_hash": 10_000 + index,
            "stmt_type": "Activation" if index % 2 else "Inhibition",
            "stmt_num": index * 3,
            "correct": label,
            "reach": int(reader_eligible),
        }
        for source in HISTORICAL_ALL_SOURCE_ORDER:
            row.setdefault(source, 0)
        extended_rows.append(row)

    multireader_indices = list(range(6)) + list(range(15, 21))
    multireader_rows = [dict(extended_rows[index]) for index in multireader_indices]
    raw_rows = [
        {
            "pa_hash": row["stmt_hash"],
            "source_hash": 50_000 + index,
            "tag": "correct" if row["correct"] else "wrong_relation",
            "curator": "alice" if index % 2 else "bob",
        }
        for index, row in enumerate(extended_rows)
    ]

    extended_payload = pickle.dumps(extended_rows, protocol=4)
    multireader_payload = pickle.dumps(multireader_rows, protocol=4)
    raw_payload = json.dumps(raw_rows, sort_keys=True).encode("utf-8")
    paths = {
        "extended": tmp_path / "extended.pkl",
        "multireader": tmp_path / "multireader.pkl",
        "raw": tmp_path / "curations.json",
    }
    paths["extended"].write_bytes(extended_payload)
    paths["multireader"].write_bytes(multireader_payload)
    paths["raw"].write_bytes(raw_payload)

    expectations = FreezeExpectations(
        extended=DatasetExpectation(
            file=_file_expectation(extended_payload),
            rows=30,
            positive=15,
            negative=15,
        ),
        reader_filtered_rows=20,
        reader_filtered_positive=10,
        reader_filtered_negative=10,
        multireader=DatasetExpectation(
            file=_file_expectation(multireader_payload),
            rows=12,
            positive=6,
            negative=6,
        ),
        raw_curations=_raw_expectation(raw_payload, raw_rows),
    )
    return paths, expectations, extended_rows


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_paper_gold_mapping_and_conflict_semantics() -> None:
    for tag in ("correct", "hypothesis", "act_vs_amt"):
        assert paper_tag_label(tag) == 1
    for tag in ("grounding", "wrong_relation", "other", None, 17):
        assert paper_tag_label(tag) == 0

    rows = [
        {"pa_hash": 1, "source_hash": 10, "tag": "correct"},
        {"pa_hash": 1, "source_hash": 10, "tag": "wrong_relation"},
        {"pa_hash": 1, "source_hash": 11, "tag": "hypothesis"},
        {"pa_hash": 2, "source_hash": 20, "tag": "act_vs_amt"},
        {"pa_hash": 3, "source_hash": 30, "tag": "grounding"},
    ]
    pairs, statements, conflicts = aggregate_observed_gold(rows)

    assert pairs == {(1, 10): 0, (1, 11): 1, (2, 20): 1, (3, 30): 0}
    assert statements == {1: 1, 2: 1, 3: 0}
    assert conflicts == 1


def test_synthetic_freeze_preserves_order_and_exact_split_protocol(tmp_path: Path) -> None:
    paths, expectations, source_rows = _write_synthetic_release(tmp_path)
    output = tmp_path / "freeze-one"

    random.seed(91_827)
    expected_next_random = random.Random(91_827).random()
    manifest = freeze_protocol(
        paths["extended"],
        paths["multireader"],
        paths["raw"],
        output,
        expectations=expectations,
    )
    assert random.random() == expected_next_random

    assert manifest["claim_scope"]["status"] == (
        "protocol_reconstruction_not_historical_parity"
    )
    assert manifest["claim_scope"]["fits_or_predictions_generated"] is False
    assert manifest["eligible_sets"]["extended_all_sources"] == {
        "source": "extended_curation_dataset.pkl in released pickle row order",
        "rows": 30,
        "positive": 15,
        "negative": 15,
    }
    assert manifest["eligible_sets"]["reader_only"]["rows"] == 20
    assert manifest["source_columns"]["reader_order"] == list(READER_SOURCES)
    assert manifest["source_columns"]["historical_all_source_order"] == list(
        HISTORICAL_ALL_SOURCE_ORDER
    )

    eligible = _read_jsonl(output / ELIGIBLE_FILENAME)
    assert [row["stmt_hash"] for row in eligible] == [
        str(row["stmt_hash"]) for row in source_rows
    ]
    assert [row["source_row_index"] for row in eligible] == list(range(30))
    assert sum(row["reader_eligible"] for row in eligible) == 20
    assert all(len(row["reader_source_counts"]) == len(READER_SOURCES) for row in eligible)

    assignments = _read_jsonl(output / FOLDS_FILENAME)
    assert len(assignments) == 50
    all_source = [
        row for row in assignments if row["eligible_set"] == "extended_all_sources"
    ]
    shuffled = [
        (index, str(row["stmt_hash"]), int(row["correct"]))
        for index, row in enumerate(source_rows)
    ]
    random.Random(4).shuffle(shuffled)
    expected_shuffle_positions = {
        source_row_index: position
        for position, (source_row_index, _, _) in enumerate(shuffled)
    }
    assert {row["source_row_index"]: row["shuffle_position"] for row in all_source} == (
        expected_shuffle_positions
    )
    assert {row["test_fold"] for row in assignments} == set(range(10))
    reader_fold_counts = manifest["split_protocol"]["fold_summaries"]["reader_only"][
        "fold_counts"
    ]
    assert all(
        counts == {"rows": 2, "positive": 1, "negative": 1}
        for counts in reader_fold_counts.values()
    )
    for eligible_set in ("extended_all_sources", "reader_only"):
        check = manifest["split_protocol"]["fold_summaries"][eligible_set][
            "installed_sklearn_crosscheck"
        ]
        assert check["status"] in {"pass", "not_installed"}

    output_two = tmp_path / "freeze-two"
    freeze_protocol(
        paths["extended"],
        paths["multireader"],
        paths["raw"],
        output_two,
        expectations=expectations,
    )
    assert (output / ELIGIBLE_FILENAME).read_bytes() == (
        output_two / ELIGIBLE_FILENAME
    ).read_bytes()
    assert (output / FOLDS_FILENAME).read_bytes() == (
        output_two / FOLDS_FILENAME
    ).read_bytes()


def test_digest_mismatch_fails_before_outputs_are_created(tmp_path: Path) -> None:
    paths, expectations, _ = _write_synthetic_release(tmp_path)
    paths["extended"].write_bytes(paths["extended"].read_bytes() + b"tampered")
    output = tmp_path / "freeze"

    with pytest.raises(AuditError, match="identity mismatch"):
        freeze_protocol(
            paths["extended"],
            paths["multireader"],
            paths["raw"],
            output,
            expectations=expectations,
        )

    assert not output.exists()


def test_existing_artifact_is_never_overwritten(tmp_path: Path) -> None:
    paths, expectations, _ = _write_synthetic_release(tmp_path)
    output = tmp_path / "freeze"
    output.mkdir()
    protocol = output / PROTOCOL_FILENAME
    protocol.write_text("owner data")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_protocol(
            paths["extended"],
            paths["multireader"],
            paths["raw"],
            output,
            expectations=expectations,
        )

    assert protocol.read_text() == "owner data"
    assert not (output / ELIGIBLE_FILENAME).exists()
    assert not (output / FOLDS_FILENAME).exists()


def test_restricted_unpickler_rejects_code_execution(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    payload = pickle.dumps(_TouchOnUnpickle(marker))

    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        _restricted_pickle_loads(payload)

    assert not marker.exists()


PAPER_CLONE = Path(
    os.environ.get("INDRA_ASSEMBLY_PAPER_CLONE", "/tmp/indra_assembly_paper_audit")
)
PAPER_CURATION_DIR = PAPER_CLONE / "data" / "curation"
PAPER_PATHS = {
    "extended": PAPER_CURATION_DIR / "extended_curation_dataset.pkl",
    "multireader": PAPER_CURATION_DIR / "multireader_curation_dataset.pkl",
    "raw": PAPER_CURATION_DIR / "indra_assembly_curations.json",
}


@pytest.mark.skipif(
    not all(path.is_file() for path in PAPER_PATHS.values()),
    reason="released paper clone is not present; set INDRA_ASSEMBLY_PAPER_CLONE",
)
def test_released_paper_artifacts_freeze_to_expected_counts(tmp_path: Path) -> None:
    manifest = freeze_protocol(
        PAPER_PATHS["extended"],
        PAPER_PATHS["multireader"],
        PAPER_PATHS["raw"],
        tmp_path / "released-freeze",
        expectations=RELEASE_EXPECTATIONS,
    )

    assert manifest["eligible_sets"]["extended_all_sources"]["rows"] == 1_689
    assert manifest["eligible_sets"]["extended_all_sources"]["positive"] == 1_237
    assert manifest["eligible_sets"]["extended_all_sources"]["negative"] == 452
    assert manifest["eligible_sets"]["reader_only"]["rows"] == 1_676
    assert manifest["eligible_sets"]["reader_only"]["positive"] == 1_236
    assert manifest["eligible_sets"]["reader_only"]["negative"] == 440
    assert manifest["eligible_sets"]["multireader_released"]["rows"] == 1_330
    assert manifest["validation"] == {
        "released_digests": "pass",
        "restricted_pickle_decode": "pass",
        "expected_counts": "pass",
        "extended_statement_hashes_unique": True,
        "multireader_subset_and_labels": "pass",
        "raw_curation_mapping": "pass",
        "artifacts_overwritten": False,
    }
