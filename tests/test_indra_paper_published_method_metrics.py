"""Contracts for the frozen published-method reference panel."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_indra_paper_method_metrics import (  # noqa: E402
    EXPECTED_NOTEBOOK_SHA256,
    extract,
    parse_ascii_table,
)


ARTIFACT = ROOT / "data/benchmark/indra_paper_2023_published_method_metrics.json"


def test_published_method_artifact_is_pinned_and_unambiguous():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema_version"] == 1
    assert artifact["artifact_kind"] == "indra_assembly_paper_published_method_metrics"
    assert artifact["source"]["notebook_sha256"] == EXPECTED_NOTEBOOK_SHA256
    assert artifact["source"]["commit"] == "63abdf1274d2f5534ed822585775031712916c83"

    contract = artifact["metric_contract"]
    assert contract["unit"] == "assembled_statement"
    assert contract["uncertainty_is_confidence_interval"] is False
    assert contract["metric_is_pooled_average_precision"] is False
    assert contract["directly_comparable_to_pair_error_f1"] is False

    methods = artifact["methods"]
    assert artifact["method_count"] == len(methods) == 59
    assert len({row["method_id"] for row in methods}) == 59
    assert [len(table["rows"]) for table in artifact["tables"]] == [41, 18]
    assert all(row["fold_count"] == 10 for row in methods)
    assert all(0 <= row["fold_mean_trapezoidal_pr_auc"] <= 1 for row in methods)
    assert all(row["fold_population_sd"] >= 0 for row in methods)

    by_name = {row["method"]: row for row in methods}
    assert by_name["Belief Orig - readers"]["fold_mean_trapezoidal_pr_auc"] == 0.917
    assert max(row["fold_mean_trapezoidal_pr_auc"] for row in methods) == 0.942
    assert (
        by_name[
            "RF 2k-d13 + Type/#PMIDs/promoter - all sources, specific"
        ]["fold_population_sd"]
        == 0.014
    )


def test_ascii_table_parser_preserves_wrapped_method_names():
    text = """\
| 35  | RF 2k-d13 + Type/#PMIDs/promoter - all sources,      | 0.942 +/- 0.014 |
|     | specific                                             |                 |
"""
    rows = parse_ascii_table(text, table_id="paper_table_6", cell_index=47)
    assert rows == [
        {
            "method_id": "paper_table_6:35",
            "table_id": "paper_table_6",
            "notebook_cell_index": 47,
            "row": 35,
            "method": "RF 2k-d13 + Type/#PMIDs/promoter - all sources, specific",
            "fold_mean_trapezoidal_pr_auc": 0.942,
            "fold_population_sd": 0.014,
            "fold_count": 10,
        }
    ]


def test_extractor_rejects_unpinned_notebook(tmp_path: Path):
    notebook = tmp_path / "paper.ipynb"
    notebook.write_text(json.dumps({"cells": []}))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        extract(notebook)
