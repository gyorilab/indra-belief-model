"""Focused contracts for the parameterized cost/performance frontier."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import frontier_table as frontier  # noqa: E402

MASK = (1 << 64) - 1


def _gold(matches_hash: int, source_hash: int, tag: str) -> dict:
    return {
        "matches_hash": matches_hash,
        "source_hash": source_hash,
        "tag": tag,
        "gold": "correct" if tag == "correct" else "incorrect",
    }


def _scored(
    matches_hash: int,
    source_hash: int,
    stmt_i: int,
    verdict: str | None,
    *,
    prompt_tokens: int = 0,
    model_id: str = "google.gemma-4-e2b",
) -> dict:
    calls = []
    if prompt_tokens:
        calls.append(
            {
                "model_id": model_id,
                "prompt_tokens": prompt_tokens,
                "out_tokens": 10,
            }
        )
    return {
        "stmt_i": stmt_i,
        "evidence_i": 0,
        "stmt_hash": f"{matches_hash & MASK:016x}",
        "source_hash": source_hash,
        "verdict": verdict,
        "call_log": calls,
    }


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _meta(run: Path, model: str) -> None:
    run.with_suffix(".meta.json").write_text(
        json.dumps({"model": model, "status": "completed"}) + "\n"
    )


def _report(
    tmp_path: Path,
    gold: list[dict],
    runs: list[Path],
    *,
    denominator: int | None = None,
    require_valid_coverage: bool = False,
) -> dict:
    gold_path = _jsonl(tmp_path / "gold.jsonl", gold)
    return frontier.build_report(
        gold_path=gold_path,
        run_paths=runs,
        label="unit",
        denominator=denominator or len(gold),
        bootstrap_samples=40,
        seed=17,
        require_valid_coverage=require_valid_coverage,
    )


def test_latest_retry_is_scored_but_every_attempt_is_costed(tmp_path):
    gold = [_gold(101, 11, "wrong_relation"), _gold(202, 22, "correct")]
    run = _jsonl(
        tmp_path / "unit_bedrock-retry.jsonl",
        [
            # The first answer for row zero is wrong and is superseded later.
            _scored(101, 11, 0, "correct", prompt_tokens=100),
            _scored(202, 22, 1, "correct", prompt_tokens=100),
            _scored(101, 11, 0, "incorrect", prompt_tokens=100),
        ],
    )
    _meta(run, "bedrock-retry")

    report = _report(tmp_path, gold, [run])
    row = report["models"][0]

    assert row["attempt_rows"] == 3
    assert row["canonical_rows"] == 2
    assert row["retry_rows"] == 1
    assert row["f1"] == row["precision"] == row["recall"] == row["accuracy"] == 1.0
    # google.gemma-4-e2b list price: $0.04/M input + $0.08/M output.
    expected = (300 * 0.04 + 30 * 0.08) / 1_000_000
    assert row["cost_usd"] == pytest.approx(expected)
    assert row["call_count"] == 3
    assert len(row["run_sha256"]) == 64
    assert len(row["meta_sha256"]) == 64
    assert len(report["gold"]["sha256"]) == 64
    assert row["frontier_eligible"] is True

    assert row["pareto_point"] is True

    baseline = report["always_incorrect_baseline"]
    assert baseline["f1"] == pytest.approx(2 / 3)
    assert baseline["precision"] == baseline["accuracy"] == 0.5
    assert baseline["recall"] == 1.0


def test_parser_null_reduces_metric_coverage_but_is_explicit(tmp_path):
    gold = [_gold(1, 11, "wrong_relation"), _gold(2, 22, "correct")]
    run = _jsonl(
        tmp_path / "unit_bedrock-null.jsonl",
        [
            _scored(1, 11, 0, "incorrect", prompt_tokens=10),
            _scored(2, 22, 1, None, prompt_tokens=10),
        ],
    )
    _meta(run, "bedrock-null")

    row = _report(tmp_path, gold, [run])["models"][0]
    assert row["matched_rows"] == 2
    assert row["run_coverage_complete"] is True
    assert row["n"] == 1
    assert row["coverage_fraction"] == 0.5
    assert row["coverage_complete"] is False
    assert row["parse_nulls"] == 1
    # Historical fleet candidacy is by completed Bedrock run scope.  Nulls are
    # not hidden; they reduce the stated metric denominator.
    assert row["frontier_eligible"] is True

    strict = _report(
        tmp_path,
        gold,
        [run],
        require_valid_coverage=True,
    )["models"][0]
    assert strict["frontier_eligible"] is False
    assert strict["reference_frontier_eligible"] is False
    assert "incomplete_valid_coverage" in strict["frontier_exclusion_reasons"]


def test_medpsy_estimate_is_reference_only_not_formal_bedrock(tmp_path):
    gold = [_gold(1, 11, "wrong_relation"), _gold(2, 22, "wrong_relation")]
    bedrock = _jsonl(
        tmp_path / "unit_bedrock-cheap.jsonl",
        [
            _scored(1, 11, 0, "incorrect", prompt_tokens=10),
            _scored(2, 22, 1, "correct"),
        ],
    )
    medpsy = _jsonl(
        tmp_path / "unit_medpsy-remote.jsonl",
        [
            _scored(1, 11, 0, "incorrect", prompt_tokens=10, model_id="medpsy-4b"),
            _scored(2, 22, 1, "incorrect"),
        ],
    )
    _meta(bedrock, "bedrock-cheap")
    _meta(medpsy, "medpsy-remote")

    report = _report(tmp_path, gold, [bedrock, medpsy])
    rows = {row["name"]: row for row in report["models"]}

    assert rows["cheap"]["serving_scope"] == "bedrock"
    assert rows["cheap"]["cost_basis"] == "list"
    assert rows["cheap"]["frontier_eligible"] is True
    assert rows["medpsy-remote"]["serving_scope"] == "non_bedrock"
    assert rows["medpsy-remote"]["cost_basis"] == "estimate"
    assert rows["medpsy-remote"]["cost_is_estimate"] is True
    assert rows["medpsy-remote"]["frontier_eligible"] is False
    assert "non_bedrock_serving_scope" in rows["medpsy-remote"]["frontier_exclusion_reasons"]
    assert "cost_not_repo_list_rate" in rows["medpsy-remote"]["frontier_exclusion_reasons"]
    assert rows["medpsy-remote"]["reference_frontier_eligible"] is True
    assert report["pareto"]["formal_frontier"] == ["cheap"]
    assert "medpsy-remote" in report["pareto"]["reference_frontier_including_estimates"]


def test_point_pareto_and_bootstrap_are_stable_when_other_runs_are_added(tmp_path):
    gold = [
        _gold(1, 11, "wrong_relation"),
        _gold(2, 22, "wrong_relation"),
        _gold(3, 33, "correct"),
        _gold(4, 44, "correct"),
    ]

    def make_run(name: str, predictions: list[str], tokens: int) -> Path:
        run = _jsonl(
            tmp_path / f"unit_bedrock-{name}.jsonl",
            [
                _scored(i, i * 11, i - 1, verdict, prompt_tokens=tokens if i == 1 else 0)
                for i, verdict in enumerate(predictions, 1)
            ],
        )
        _meta(run, f"bedrock-{name}")
        return run

    cheap = make_run("cheap", ["incorrect", "correct", "correct", "correct"], 100)
    knee = make_run("knee", ["incorrect", "incorrect", "correct", "correct"], 200)
    dominated = make_run("dominated", ["incorrect", "correct", "correct", "correct"], 300)

    cheap_only = _report(tmp_path, gold, [cheap])["models"][0]
    report = _report(tmp_path, gold, [cheap, knee, dominated])
    rows = {row["name"]: row for row in report["models"]}

    assert (rows["cheap"]["lo"], rows["cheap"]["hi"]) == (
        cheap_only["lo"],
        cheap_only["hi"],
    )
    assert report["pareto"]["method"].startswith("point estimates")
    assert report["pareto"]["formal_frontier"] == ["cheap", "knee"]
    assert rows["cheap"]["pareto_point"] is True
    assert rows["knee"]["pareto_point"] is True
    assert rows["dominated"]["pareto_point"] is False


def test_cli_writes_json_and_markdown_artifacts(tmp_path):
    gold = [_gold(1, 11, "wrong_relation")]
    gold_path = _jsonl(tmp_path / "gold.jsonl", gold)
    run = _jsonl(
        tmp_path / "expanded_bedrock-unit.jsonl",
        [_scored(1, 11, 0, "incorrect", prompt_tokens=10)],
    )
    _meta(run, "bedrock-unit")
    json_out = tmp_path / "frontier.json"
    md_out = tmp_path / "frontier.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "frontier_table.py"),
            "--gold",
            str(gold_path),
            "--runs-glob",
            str(tmp_path / "expanded_*.jsonl"),
            "--label",
            "expanded-403",
            "--denominator",
            "1",
            "--bootstrap-samples",
            "10",
            "--require-valid-coverage",
            "--output-json",
            str(json_out),
            "--output-md",
            str(md_out),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(json_out.read_text())
    assert payload["label"] == "expanded-403"
    assert payload["require_valid_coverage"] is True
    assert payload["models"][0]["f1"] == 1.0
    markdown = md_out.read_text()
    assert "Formal point-Pareto frontier" in markdown
    assert "always-incorrect" in markdown.lower()
