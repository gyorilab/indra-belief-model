from __future__ import annotations

import json
from pathlib import Path

import pytest

from indra_belief.comparison.contracts import ContractError
from indra_belief.comparison.inputs import load_inputs


def _config(tmp_path: Path) -> Path:
    value = {
        "workspace_root": ".",
        "gold_manifest": "gold.json",
        "paper_reproduction_manifest": "paper.json",
        "published_metrics": "published.json",
        "current_simple_manifest": "simple.json",
        "current_bayesian_manifest": "bayesian.json",
        "current_hierarchy_manifest": "hierarchy.json",
        "current_counts_all_source_manifest": "counts-all.json",
        "current_counts_reader_manifest": "counts-reader.json",
        "scorer_registry": "scorers.json",
        "production_hybrid_manifest": "production.json",
        "error_review_protocol": "error-review.json",
        "llm_models": [
            {
                "model_id": "llm_gemma_4_e2b",
                "action_id": None,
                "run_id": "run_e2b",
                "served_model": "bedrock-gemma-4-e2b",
                "provider_model_id": "google.gemma-4-e2b",
                "bundle_manifest": "e2b/manifest.json",
            },
            {
                "model_id": "llm_glm_5",
                "action_id": "glm_5_primary",
                "run_id": "run_glm_5",
                "served_model": "bedrock-glm-5",
                "provider_model_id": "zai.glm-5",
                "bundle_manifest": "glm/manifest.json",
            },
        ],
        "frozen_at": "2026-07-20T00:00:00Z",
        "bootstrap": {"seed": 17, "resamples": 200},
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_loads_the_one_explicit_input_configuration(tmp_path: Path) -> None:
    path = _config(tmp_path)
    inputs = load_inputs(path)
    assert inputs.workspace_root == tmp_path
    assert inputs.gold_manifest == tmp_path / "gold.json"
    assert [model.model_id for model in inputs.llm_models] == [
        "llm_gemma_4_e2b",
        "llm_glm_5",
    ]
    assert inputs.llm_models[0].bundle_manifest == tmp_path / "e2b/manifest.json"
    assert inputs.llm_models[1].bundle_manifest == tmp_path / "glm/manifest.json"
    assert inputs.llm_models[1].action_id == "glm_5_primary"
    assert inputs.frozen_at == "2026-07-20T00:00:00Z"
    assert inputs.bootstrap_seed == 17
    assert inputs.bootstrap_resamples == 200


def test_rejects_empty_or_duplicate_model_expectations(tmp_path: Path) -> None:
    path = _config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["llm_models"] = []
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="llm_models"):
        load_inputs(path)

    path = _config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["llm_models"][1]["model_id"] = value["llm_models"][0]["model_id"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="model_id values must be unique"):
        load_inputs(path)


def test_rejects_unknown_fields_and_duplicate_json_keys(tmp_path: Path) -> None:
    path = _config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["release_version"] = 5
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="unknown"):
        load_inputs(path)

    path.write_text('{"workspace_root":".","workspace_root":".."}', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate JSON key"):
        load_inputs(path)
