"""Coverage and resume invariants for the vLLM human-gold runner."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_vllm_gold_eval.py"
)
SPEC = importlib.util.spec_from_file_location("run_vllm_gold_eval", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from indra_belief.scorers.monolithic import scorer as mono


def test_every_cli_variant_resolves_to_a_registered_nonempty_prompt():
    parser = runner._build_parser()

    for name in runner.VARIANT_CHOICES:
        args = parser.parse_args(["--variant", name])
        resolved = runner._registry_variant(args.variant)

        assert mono.VARIANTS[resolved].system_prompt


def test_default_cli_variant_resolves_to_a_registered_nonempty_prompt():
    args = runner._build_parser().parse_args([])
    resolved = runner._registry_variant(args.variant)

    assert mono.VARIANTS[resolved].system_prompt


def test_registry_variant_only_translates_baseline():
    resolved = runner._registry_variant("baseline")

    assert resolved == ""
    assert mono.VARIANTS[resolved] is mono.VARIANTS[""]
    for name in runner.VARIANT_CHOICES:
        if name != "baseline":
            assert runner._registry_variant(name) == name


def test_every_named_registry_variant_is_offered_by_the_cli():
    named_registry_variants = {name for name in mono.VARIANTS if name}

    assert named_registry_variants <= set(runner.VARIANT_CHOICES)


def test_metrics_distinguish_accuracy_coverage_and_strict_accuracy():
    gold = [
        {"tag": "correct"},
        {"tag": "grounding"},
        {"tag": "correct"},
        {"tag": "wrong_relation"},
    ]
    latest = {
        0: {"row_status": "scored", "verdict": "correct", "tier": "llm"},
        1: {"row_status": "scored", "verdict": "incorrect", "tier": "llm"},
        2: {"row_status": "parser_null", "verdict": None, "tier": "llm"},
        # Row 3 was never attempted.
    }

    metrics = runner.compute_metrics(gold, latest)

    assert metrics["valid_verdicts"] == 2
    assert metrics["correct_predictions"] == 2
    assert metrics["accuracy_on_verdicts"] == 1.0
    assert metrics["coverage"] == 0.5
    assert metrics["strict_end_to_end_accuracy"] == 0.5
    assert metrics["status_counts"] == {
        "scored": 2,
        "parser_null": 1,
        "not_attempted": 1,
    }
    assert metrics["confusion_positive_correct"] == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 1,
    }


def test_latest_attempt_wins_and_corrupt_tail_is_ignored(tmp_path):
    output = tmp_path / "run.jsonl"
    rows = [
        {"row_index": 7, "row_status": "error", "verdict": None},
        {"row_index": 2, "row_status": "scored", "verdict": "incorrect"},
        {"row_index": 7, "row_status": "scored", "verdict": "correct"},
    ]
    output.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n" + '{"row_index":'
    )

    latest, corrupt = runner.load_latest_attempts(output)

    assert corrupt == 1
    assert latest[7]["verdict"] == "correct"
    assert latest[2]["verdict"] == "incorrect"


def test_resume_retries_nulls_and_errors_by_default():
    assert runner._is_done(
        {"row_status": "scored", "verdict": "correct"},
        retry_parser_nulls=True,
        retry_errors=True,
    )
    assert not runner._is_done(
        {"row_status": "parser_null", "verdict": None},
        retry_parser_nulls=True,
        retry_errors=True,
    )
    assert not runner._is_done(
        {"row_status": "error", "verdict": None},
        retry_parser_nulls=True,
        retry_errors=True,
    )
    assert runner._is_done(
        {"row_status": "unmatched", "verdict": None},
        retry_parser_nulls=True,
        retry_errors=True,
    )
