"""Contracts for paired inference on a frozen formal frontier."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import frontier_paired_stats as paired  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _scored(index: int, prediction: bool) -> dict:
    return {
        "stmt_i": index,
        "evidence_i": 0,
        "stmt_hash": f"{index + 1:016x}",
        "source_hash": 101 + index,
        "verdict": "incorrect" if prediction else "correct",
    }


def _fixture(tmp_path: Path) -> Path:
    # Four errors, four correct rows; each exact pair is also a unique statement.
    gold = [
        {
            "matches_hash": index + 1,
            "source_hash": 101 + index,
            "tag": "wrong_relation" if index < 4 else "correct",
        }
        for index in range(8)
    ]
    gold_path = _jsonl(tmp_path / "gold.jsonl", gold)
    predictions = {
        "cheap": [True, True, False, False, False, False, False, False],
        "knee": [True, True, True, False, False, False, False, False],
        "best": [True, True, True, True, False, False, False, False],
    }
    runs: dict[str, Path] = {}
    for name, pred in predictions.items():
        rows = [_scored(index, value) for index, value in enumerate(pred)]
        if name == "cheap":
            # An append retry is cost/audit history, but only the latest logical row
            # enters performance.  The retry restores row zero to the intended value.
            rows = [_scored(0, False), *rows, _scored(0, True)]
        runs[name] = _jsonl(tmp_path / f"{name}.jsonl", rows)

    f1 = {"cheap": 2 / 3, "knee": 6 / 7, "best": 1.0}
    costs = {"cheap": 0.1, "knee": 0.4, "best": 2.0}
    frontier = {
        "schema_version": 1,
        "label": "paired-unit",
        "require_valid_coverage": True,
        "denominator": 8,
        "metric": {
            "name": "error_detection_f1",
            "positive_class": "curator label is incorrect",
            "positive_prediction": "model verdict is incorrect",
        },
        "gold": {"path": str(gold_path), "sha256": _sha(gold_path)},
        "pareto": {"formal_frontier": ["cheap", "knee", "best"]},
        "models": [
            {
                "name": name,
                "run_path": str(path),
                "run_sha256": _sha(path),
                "frontier_eligible": True,
                "run_status": "completed",
                "run_coverage_complete": True,
                "coverage_complete": True,
                "n": 8,
                "parse_nulls": 0,
                "row_errors": 0,
                "unmatched_rows": 0,
                "invalid_json_lines": 0,
                "f1": f1[name],
                "usd_1k": costs[name],
            }
            for name, path in runs.items()
        ],
    }
    frontier_path = tmp_path / "frontier.json"
    frontier_path.write_text(json.dumps(frontier) + "\n")
    return frontier_path


def test_adjacent_paired_report_is_deterministic_and_audited(tmp_path):
    frontier = _fixture(tmp_path)
    first = paired.build_paired_report(
        frontier, bootstrap_samples=300, permutations=400, seed=17
    )
    second = paired.build_paired_report(
        frontier, bootstrap_samples=300, permutations=400, seed=17
    )
    assert first == second
    assert first["formal_frontier_cost_order"] == ["cheap", "knee", "best"]
    assert first["gold"]["unique_statement_hashes"] == 8
    assert first["method"]["resampling_unit_equals_statement"] is True
    assert len(first["comparisons"]) == 2
    assert [(row["cheaper"], row["more_expensive"]) for row in first["comparisons"]] == [
        ("cheap", "knee"),
        ("knee", "best"),
    ]
    for row in first["comparisons"]:
        assert row["n"] == 8
        assert row["delta_f1_more_expensive_minus_cheaper"] > 0
        assert 0 < row["permutation_p_raw"] <= 1
        assert row["permutation_p_raw"] <= row["permutation_p_holm"] <= 1
    assert first["run_audit"]["cheap"]["attempt_rows"] == 10
    assert first["run_audit"]["cheap"]["retry_rows"] == 2


def test_permutation_plus_one_correction_and_identity():
    gold = [True, True, True, False, False, False]
    weak = [False, False, False, False, False, False]
    strong = [True, True, True, False, False, False]
    result = paired.paired_permutation_errf1(
        gold, weak, strong, n_perm=20, seed=3, batch_size=7
    )
    assert result["p_value"] >= 1 / 21
    identical = paired.paired_permutation_errf1(
        gold, strong, strong, n_perm=20, seed=3
    )
    assert identical["p_value"] == 1.0


def test_holm_known_values():
    assert paired._holm_adjust([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.06, 0.06]
    )


def test_hash_change_is_rejected(tmp_path):
    frontier = _fixture(tmp_path)
    payload = json.loads(frontier.read_text())
    run = Path(payload["models"][0]["run_path"])
    run.write_text(run.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        paired.build_paired_report(
            frontier, bootstrap_samples=10, permutations=10, seed=1
        )


def test_source_only_match_is_rejected_even_when_source_is_unique(tmp_path):
    frontier = _fixture(tmp_path)
    payload = json.loads(frontier.read_text())
    model = next(row for row in payload["models"] if row["name"] == "knee")
    run = Path(model["run_path"])
    rows = [json.loads(line) for line in run.read_text().splitlines() if line.strip()]
    rows[1]["stmt_hash"] = "deadbeefdeadbeef"
    _jsonl(run, rows)
    model["run_sha256"] = _sha(run)
    frontier.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="unmatched literal statement/source pair"):
        paired.build_paired_report(
            frontier, bootstrap_samples=10, permutations=10, seed=1
        )


def test_cli_writes_json_and_markdown(tmp_path):
    frontier = _fixture(tmp_path)
    json_out = tmp_path / "paired.json"
    md_out = tmp_path / "paired.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "frontier_paired_stats.py"),
            str(frontier),
            "--bootstrap-samples",
            "50",
            "--permutations",
            "50",
            "--seed",
            "5",
            "--output-json",
            str(json_out),
            "--output-md",
            str(md_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(json_out.read_text())["label"] == "paired-unit"
    markdown = md_out.read_text()
    assert "cheap → knee → best" in markdown
    assert "not evidence that the models are equivalent" in markdown
