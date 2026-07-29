#!/usr/bin/env python3
"""Freeze the published INDRA paper method table from its executed notebook.

The source notebook reports a mean and population standard deviation over ten
fold-wise trapezoidal precision-recall areas.  These are literature reference
points, not pooled average precision and not confidence intervals.  This
extractor preserves that distinction and refuses an unpinned notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_NOTEBOOK_SHA256 = (
    "3bd1a684fdc33c0b4963dd3e0c834c5420d90703112a91773f43415e1125ad26"
)
PAPER_REPOSITORY = "https://github.com/sorgerlab/indra_assembly_paper"
PAPER_COMMIT = "63abdf1274d2f5534ed822585775031712916c83"
NOTEBOOK_PATH = "notebooks/Training Belief ML Models.ipynb"

ROW_RE = re.compile(
    r"^\|\s*(?P<row>\d+)\s*\|\s*(?P<model>.*?)\s*\|\s*"
    r"(?P<mean>\d+(?:\.\d+)?)\s*\+/-\s*(?P<sd>\d+(?:\.\d+)?)\s*\|$"
)
CONTINUATION_RE = re.compile(r"^\|\s*\|\s*(?P<model>.*?)\s*\|\s*\|$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_text(cell: dict[str, Any]) -> str:
    chunks: list[str] = []
    for output in cell.get("outputs", []):
        value = output.get("text")
        if isinstance(value, list):
            chunks.extend(str(part) for part in value)
        elif isinstance(value, str):
            chunks.append(value)
    return "".join(chunks)


def parse_ascii_table(text: str, *, table_id: str, cell_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = ROW_RE.match(line)
        if match:
            row_number = int(match.group("row"))
            rows.append(
                {
                    "method_id": f"{table_id}:{row_number:02d}",
                    "table_id": table_id,
                    "notebook_cell_index": cell_index,
                    "row": row_number,
                    "method": match.group("model").strip(),
                    "fold_mean_trapezoidal_pr_auc": float(match.group("mean")),
                    "fold_population_sd": float(match.group("sd")),
                    "fold_count": 10,
                }
            )
            continue
        continuation = CONTINUATION_RE.match(line)
        if continuation and rows:
            fragment = continuation.group("model").strip()
            if fragment:
                rows[-1]["method"] = f"{rows[-1]['method']} {fragment}".strip()
    return rows


def extract(notebook_path: Path) -> dict[str, Any]:
    digest = sha256_file(notebook_path)
    if digest != EXPECTED_NOTEBOOK_SHA256:
        raise ValueError(
            f"notebook SHA-256 mismatch: got {digest}, expected {EXPECTED_NOTEBOOK_SHA256}"
        )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    tables: list[dict[str, Any]] = []
    table_specs = (
        ("paper_table_6", "Print all results in a table", 41),
        ("paper_not_table_6", "Print other results not included", 18),
    )
    for table_id, source_marker, expected_rows in table_specs:
        matches: list[tuple[int, list[dict[str, Any]]]] = []
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            source = "".join(cell.get("source", []))
            if source_marker not in source:
                continue
            rows = parse_ascii_table(
                output_text(cell), table_id=table_id, cell_index=cell_index
            )
            if rows:
                matches.append((cell_index, rows))
        if len(matches) != 1:
            raise ValueError(f"expected one executed {table_id} cell, found {len(matches)}")
        cell_index, rows = matches[0]
        if len(rows) != expected_rows:
            raise ValueError(
                f"{table_id} has {len(rows)} parsed rows; expected {expected_rows}"
            )
        if [row["row"] for row in rows] != list(range(1, expected_rows + 1)):
            raise ValueError(f"{table_id} row numbering is not contiguous")
        tables.append(
            {
                "table_id": table_id,
                "notebook_cell_index": cell_index,
                "rows": rows,
            }
        )

    methods = [row for table in tables for row in table["rows"]]
    if len(methods) != 59 or len({row["method_id"] for row in methods}) != 59:
        raise ValueError("published method identity/count reconciliation failed")
    headline = {row["method"]: row for row in methods}
    if headline["Belief Orig - readers"]["fold_mean_trapezoidal_pr_auc"] != 0.917:
        raise ValueError("original belief headline value changed")
    best = max(row["fold_mean_trapezoidal_pr_auc"] for row in methods)
    if best != 0.942:
        raise ValueError("paper best-method headline value changed")

    return {
        "schema_version": 1,
        "artifact_kind": "indra_assembly_paper_published_method_metrics",
        "source": {
            "repository": PAPER_REPOSITORY,
            "commit": PAPER_COMMIT,
            "notebook_path": NOTEBOOK_PATH,
            "notebook_sha256": digest,
            "executed_output_cells": [table["notebook_cell_index"] for table in tables],
        },
        "metric_contract": {
            "positive_class": "correct assembled statement",
            "unit": "assembled_statement",
            "per_fold_metric": (
                "sklearn precision_recall_curve followed by auc(recall, precision)"
            ),
            "summary": "arithmetic mean over 10 cross-validation folds",
            "uncertainty_field": "population standard deviation over the 10 folds",
            "uncertainty_is_confidence_interval": False,
            "metric_is_pooled_average_precision": False,
            "directly_comparable_to_pair_error_f1": False,
        },
        "method_count": len(methods),
        "tables": tables,
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    artifact = extract(args.notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(f"wrote {args.output} · {artifact['method_count']} paper methods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
