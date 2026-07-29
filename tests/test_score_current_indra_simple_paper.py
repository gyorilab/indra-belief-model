from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest
from indra.belief import SimpleScorer
from indra.statements import Activation, Agent, Evidence, stmts_from_json


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_current_indra_simple_paper as adapter  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(adapter._canonical_bytes(row) + b"\n" for row in rows))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    statements = [
        Activation(
            Agent("A"),
            Agent("B"),
            evidence=[Evidence(source_api="reach", text="A activates B.")],
        ),
        Activation(
            Agent("C"),
            Agent("D"),
            evidence=[Evidence(source_api="signor", text="C activates D.")],
        ),
    ]
    statement_jsons = [statement.to_json() for statement in statements]
    # Prove that the adapter computes rather than copying carried beliefs.
    statement_jsons[0]["belief"] = 0.01
    statement_jsons[1]["belief"] = 0.99
    corpus_path = tmp_path / "corpus.json.gz"
    with gzip.open(corpus_path, "wt", encoding="utf-8") as handle:
        json.dump(statement_jsons, handle, ensure_ascii=False)

    targets = []
    for index, statement_json in enumerate(statement_jsons):
        targets.append(
            {
                "canonical_corpus_row_index": index,
                "eligible_position": index,
                "matches_hash": str(statement_json["matches_hash"]),
                "reader_eligible": True,
                "source_row_index": index,
                "statement_id": statement_json["id"],
                "statement_json_sha256": hashlib.sha256(
                    adapter._canonical_bytes(statement_json)
                ).hexdigest(),
                "statement_type": statement_json["type"],
            }
        )
    targets_path = tmp_path / "targets.jsonl"
    _write_jsonl(targets_path, targets)
    targets_manifest_path = tmp_path / "targets.manifest.json"
    _write_json(
        targets_manifest_path,
        {
            "outputs": {
                "paper_prediction_targets": {
                    "rows": len(targets),
                    "sha256": _sha(targets_path),
                }
            }
        },
    )
    paper_manifest_path = tmp_path / "paper.manifest.json"
    _write_json(
        paper_manifest_path,
        {
            "files": [
                {
                    "filename": "indra_benchmark_corpus.json.gz",
                    "bytes": corpus_path.stat().st_size,
                    "sha256": _sha(corpus_path),
                }
            ]
        },
    )
    return corpus_path, paper_manifest_path, targets_path, targets_manifest_path


def test_current_simple_adapter_is_blind_complete_and_recomputes_scores(
    tmp_path: Path,
) -> None:
    corpus, paper_manifest, targets, targets_manifest = _fixture(tmp_path)
    output_dir = tmp_path / "out"
    manifest = adapter.materialize(
        corpus_path=corpus,
        paper_manifest_path=paper_manifest,
        targets_path=targets,
        targets_manifest_path=targets_manifest,
        output_dir=output_dir,
    )

    prediction_rows = [
        json.loads(line)
        for line in (output_dir / adapter.PREDICTIONS_FILENAME).read_text().splitlines()
    ]
    reader_prediction_rows = [
        json.loads(line)
        for line in (output_dir / adapter.READER_PREDICTIONS_FILENAME)
        .read_text()
        .splitlines()
    ]
    with gzip.open(corpus, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    expected_statements = []
    for statement_json in raw:
        statement_json.pop("belief", None)
        expected_statements.extend(stmts_from_json([statement_json]))
    expected = [float(value) for value in SimpleScorer().score_statements(expected_statements)]

    assert [row["probability_correct"] for row in prediction_rows] == expected
    assert reader_prediction_rows == prediction_rows
    assert prediction_rows[0]["probability_correct"] != 0.01
    assert prediction_rows[1]["probability_correct"] != 0.99
    assert all(set(row) == {"statement_id", "probability_correct"} for row in prediction_rows)
    assert manifest["coverage"]["predicted_statements"] == 2
    assert manifest["coverage"]["missing"] == 0
    assert manifest["prediction_blinding"]["status"] == "pass"
    assert manifest["arm"]["hierarchy_propagation"] is False


def test_target_ledger_rejects_any_extra_label_field(tmp_path: Path) -> None:
    _, _, targets, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in targets.read_text().splitlines()]
    rows[0]["label"] = 1
    _write_jsonl(targets, rows)
    with pytest.raises(adapter.ContractError, match="target fields must be exactly"):
        adapter._read_targets(targets)
