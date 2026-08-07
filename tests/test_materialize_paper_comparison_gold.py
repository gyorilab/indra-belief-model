from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import materialize_paper_comparison_gold as comparison_gold  # noqa: E402



# Reads the gitignored local artifact trees; skipped only when they are WHOLLY
# absent (CI, a fresh checkout). A PARTIAL tree is a failure in
# tests/test_local_artifacts.py, never a skip here.
import _local_artifacts as _artifacts

pytestmark = _artifacts.requires()

GOLD_DIR = ROOT / "data/results/indra_paper_statement_gold_20260717"
PROTOCOL_DIR = ROOT / "data/results/indra_paper_protocol_20260717"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _statement_row(
    position: int,
    *,
    statement_id: str,
    statement_hash: str,
    paper_label: int,
    strict_status: str,
    strict_label: int | None,
) -> dict:
    return {
        "eligible_position": position,
        "source_row_index": position + 100,
        "paper_statement_hash": statement_hash,
        "canonical_corpus": {
            "statement_id": statement_id,
            "matches_hash": statement_hash,
        },
        "paper_replication_policy": {"label": paper_label},
        "adjudicated_statement_gold": {
            "strict_e0_status": strict_status,
            "strict_e0_correct": strict_label,
        },
    }


def _fold_row(position: int, *, statement_hash: str, label: int, fold: int) -> dict:
    return {
        "eligible_set": comparison_gold.EXTENDED_SET,
        "eligible_position": position,
        "source_row_index": position + 100,
        "stmt_hash": statement_hash,
        "correct": label,
        "test_fold": fold,
    }


def test_build_ledgers_never_coerces_unresolved_to_negative() -> None:
    statements = [
        _statement_row(
            0,
            statement_id="stmt-a",
            statement_hash="10",
            paper_label=1,
            strict_status="positive",
            strict_label=1,
        ),
        _statement_row(
            1,
            statement_id="stmt-b",
            statement_hash="20",
            paper_label=0,
            strict_status="negative",
            strict_label=0,
        ),
        _statement_row(
            2,
            statement_id="stmt-c",
            statement_hash="30",
            paper_label=0,
            strict_status="unresolved",
            strict_label=None,
        ),
    ]
    folds = [
        _fold_row(0, statement_hash="10", label=1, fold=0),
        _fold_row(1, statement_hash="20", label=0, fold=1),
        _fold_row(2, statement_hash="30", label=0, fold=2),
    ]

    # The unit fixture uses minimal canonical fields; add the prediction-target
    # identity fields that the real materialization requires.
    for index, row in enumerate(statements):
        row["canonical_corpus"].update(
            {
                "row_index": index,
                "statement_json_sha256": "a" * 64,
                "statement_type": "Activation",
            }
        )
    targets, released, reader, strict, reader_strict, audit = comparison_gold.build_ledgers(
        statements, folds
    )

    assert [row["statement_id"] for row in targets] == ["stmt-a", "stmt-b", "stmt-c"]
    assert [row["statement_id"] for row in released] == ["stmt-a", "stmt-b", "stmt-c"]
    assert reader == []
    assert reader_strict == []
    assert [row["statement_id"] for row in strict] == ["stmt-a", "stmt-b"]
    assert audit["released"] == {
        "rows": 3,
        "positive": 1,
        "negative": 2,
        "folds": {"0": 1, "1": 1, "2": 1},
    }
    assert audit["strict_resolved"]["unresolved_excluded"] == 1


def test_build_ledgers_fails_on_fold_identity_or_label_mismatch() -> None:
    statements = [
        _statement_row(
            0,
            statement_id="stmt-a",
            statement_hash="10",
            paper_label=1,
            strict_status="positive",
            strict_label=1,
        )
    ]
    bad_folds = [_fold_row(0, statement_hash="different", label=1, fold=0)]
    statements[0]["canonical_corpus"].update(
        {
            "row_index": 0,
            "statement_json_sha256": "a" * 64,
            "statement_type": "Activation",
        }
    )
    with pytest.raises(comparison_gold.ContractError, match="fold identity/label mismatch"):
        comparison_gold.build_ledgers(statements, bad_folds)


def test_real_release_materialization_counts_digests_and_claim_scope(tmp_path: Path) -> None:
    kwargs = {
        "statement_gold": GOLD_DIR / "paper_statement_gold.jsonl",
        "statement_manifest": GOLD_DIR / "paper_statement_gold_manifest.json",
        "fold_assignments": PROTOCOL_DIR / "paper_fold_assignments.jsonl",
        "protocol_manifest": PROTOCOL_DIR / "paper_protocol_manifest.json",
        "output_dir": tmp_path,
    }
    first = comparison_gold.materialize(**kwargs)
    first_digests = {
        name: descriptor["sha256"] for name, descriptor in first["outputs"].items()
    }
    second = comparison_gold.materialize(**kwargs)

    assert second == first
    assert {
        name: descriptor["sha256"] for name, descriptor in second["outputs"].items()
    } == first_digests
    assert first["counts"]["released"] == {
        "rows": 1689,
        "positive": 1237,
        "negative": 452,
        "folds": {str(index): count for index, count in enumerate([169] * 9 + [168])},
    }
    assert first["counts"]["reader_eligible_released"]["rows"] == 1676
    assert first["counts"]["reader_eligible_released"]["positive"] == 1236
    assert first["counts"]["reader_eligible_released"]["negative"] == 440
    assert first["counts"]["strict_resolved"]["rows"] == 1578
    assert first["counts"]["strict_resolved"]["positive"] == 1237
    assert first["counts"]["strict_resolved"]["negative"] == 341
    assert first["counts"]["strict_resolved"]["unresolved_excluded"] == 111
    assert first["counts"]["reader_eligible_strict_resolved"]["rows"] == 1565
    assert first["counts"]["reader_eligible_strict_resolved"]["positive"] == 1236
    assert first["counts"]["reader_eligible_strict_resolved"]["negative"] == 329
    assert (
        first["counts"]["reader_eligible_strict_resolved"]["unresolved_excluded"]
        == 111
    )
    assert first["claim_scope"]["historical_fold_parity"] is False

    released_path = tmp_path / comparison_gold.RELEASED_FILENAME
    targets_path = tmp_path / comparison_gold.TARGETS_FILENAME
    reader_path = tmp_path / comparison_gold.READER_FILENAME
    strict_path = tmp_path / comparison_gold.STRICT_FILENAME
    reader_strict_path = tmp_path / comparison_gold.READER_STRICT_FILENAME
    manifest_path = tmp_path / comparison_gold.MANIFEST_FILENAME
    assert _sha(released_path) == first["outputs"]["paper_released_gold"]["sha256"]
    assert _sha(targets_path) == first["outputs"]["paper_prediction_targets"]["sha256"]
    assert (
        _sha(reader_path)
        == first["outputs"]["paper_reader_eligible_released_gold"]["sha256"]
    )
    assert _sha(strict_path) == first["outputs"]["paper_strict_e0_resolved_gold"]["sha256"]
    assert (
        _sha(reader_strict_path)
        == first["outputs"]["paper_reader_eligible_strict_e0_resolved_gold"]["sha256"]
        == "eab88e220cb826dbdde7ac73569e9b3a011af65f6eefc3ffe4e429d4e0eb47c7"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest == first
    assert set(json.loads(released_path.read_text().splitlines()[0])) == {
        "statement_id",
        "label",
        "fold_id",
    }
    target = json.loads(targets_path.read_text().splitlines()[0])
    assert not ({"label", "correct", "fold_id", "belief", "curations", "gold"} & set(target))
