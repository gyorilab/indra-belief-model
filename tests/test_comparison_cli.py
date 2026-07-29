from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from indra_belief.comparison import cli, error_review, llm, report
from indra_belief.comparison.assemble import LlmModelInput
from indra_belief.comparison.runner import RunnerError


def test_offline_commands_have_one_central_json_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    emitted: list[dict] = []
    calls: list[dict] = []
    monkeypatch.setattr(cli, "_print_json", emitted.append)
    monkeypatch.setattr(
        llm,
        "materialize_model_bundle",
        lambda **values: calls.append(values) or {"model_id": values["model_id"]},
    )
    output = tmp_path / "bundle"
    arguments = {
        "run_plan": SimpleNamespace(path=tmp_path / "run_plan.json"),
        "raw_attempts": tmp_path / "raw.jsonl",
        "execution_map": tmp_path / "map.jsonl",
        "statements": tmp_path / "statements.json",
        "spend_ledger": tmp_path / "spend.ndjson",
        "aggregation": tmp_path / "aggregation.json",
        "pricing": tmp_path / "pricing.json",
        "output_dir": output,
        "run_id": "run",
        "served_model": "served",
        "model_id": "model",
        "provider_model_id": "provider",
        "workload": "unique_exact_pairs_primary",
    }
    monkeypatch.setattr(cli, "_model_bundle_arguments", lambda **_values: arguments)

    result = cli.main(
        [
            "model-bundle",
            "--action",
            "primary",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert emitted == [
        {
            "status": "complete",
            "outputs": {"manifest": str(output / "manifest.json")},
        }
    ]


def test_leaf_modules_have_no_competing_cli_and_materialize_has_no_overrides() -> None:
    assert not any(hasattr(module, "main") for module in (llm, error_review, report))
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["materialize", "--llm-bundle", "foreign.json"])
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            ["model-bundle", "--action", "primary", "--model-id", "swapped"]
        )
    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            ["model-bundle", "--action", "primary", "--run-plan", "foreign.json"]
        )


def test_model_bundle_arguments_are_derived_from_frozen_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = {
        "workloads": [
            {
                "name": "primary_workload",
                "corpus": {"path": "data/statements.json"},
                "execution_map": {"path": "data/execution_map.jsonl"},
            }
        ]
    }
    replay_capture = SimpleNamespace(payload=json.dumps(replay).encode())
    action = SimpleNamespace(
        id="primary",
        stage_id="stage",
        run_id="paper_primary",
        workload="primary_workload",
        output=tmp_path / "attempts.jsonl",
        ledger=tmp_path / "spend.ndjson",
    )
    stage = SimpleNamespace(
        model="bedrock-model", provider_model_id="provider.model"
    )
    plan_capture = object()
    plan = SimpleNamespace(
        action_by_id={"primary": action},
        stage_by_id={"stage": stage},
        replay_manifest=SimpleNamespace(capture=lambda **_kwargs: replay_capture),
        root=tmp_path,
        capture=plan_capture,
    )
    declaration = LlmModelInput(
        model_id="llm_model",
        action_id="primary",
        run_id="paper_primary",
        served_model="bedrock-model",
        provider_model_id="provider.model",
        bundle_manifest=tmp_path / "data/comparison/models/model/manifest.json",
    )
    inputs = SimpleNamespace(llm_models=(declaration,))
    monkeypatch.setattr(cli, "load_run_plan", lambda _path: plan)
    monkeypatch.setattr(cli, "load_inputs", lambda _path: inputs)

    arguments = cli._model_bundle_arguments(
        inputs_path=tmp_path / "data/comparison/inputs.json",
        plan_path=tmp_path / "data/comparison/run_plan.json",
        action_id="primary",
    )
    assert arguments == {
        "run_plan": plan_capture,
        "raw_attempts": tmp_path / "attempts.jsonl",
        "execution_map": tmp_path / "data/execution_map.jsonl",
        "statements": tmp_path / "data/statements.json",
        "spend_ledger": tmp_path / "spend.ndjson",
        "aggregation": tmp_path / "data/comparison/aggregation.json",
        "pricing": tmp_path / "data/comparison/pricing.json",
        "output_dir": tmp_path / "data/comparison/models/model",
        "run_id": "paper_primary",
        "served_model": "bedrock-model",
        "model_id": "llm_model",
        "provider_model_id": "provider.model",
        "workload": "primary_workload",
    }

    monkeypatch.setattr(
        cli,
        "load_inputs",
        lambda _path: SimpleNamespace(
            llm_models=(replace(declaration, provider_model_id="wrong.model"),)
        ),
    )
    with pytest.raises(ValueError, match="disagrees with its frozen LLM declaration"):
        cli._model_bundle_arguments(
            inputs_path=tmp_path / "data/comparison/inputs.json",
            plan_path=tmp_path / "data/comparison/run_plan.json",
            action_id="primary",
        )


def test_plan_mismatch_cannot_cross_the_bearer_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = 0

    def bearer(_root: Path) -> str:
        nonlocal reads
        reads += 1
        return "must-not-be-read"

    monkeypatch.setattr(cli, "_bearer_token", bearer)
    plan = SimpleNamespace(root=tmp_path, sha256="a" * 64)
    with pytest.raises(RunnerError, match="different run plan"):
        cli._bearer_after_readiness(plan, {"plan_sha256": "b" * 64})
    assert reads == 0
