"""Load the one explicit input set used to assemble the comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .assemble import AssemblyInputs, LlmModelInput
from .contracts import ContractError, strict_json_loads


_PATH_FIELDS = (
    "gold_manifest",
    "paper_reproduction_manifest",
    "published_metrics",
    "current_simple_manifest",
    "current_bayesian_manifest",
    "current_hierarchy_manifest",
    "current_counts_all_source_manifest",
    "current_counts_reader_manifest",
    "scorer_registry",
    "production_hybrid_manifest",
    "error_review_protocol",
)
_TOP_LEVEL_FIELDS = {
    "workspace_root",
    *_PATH_FIELDS,
    "llm_models",
    "frozen_at",
    "bootstrap",
}


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{context} must be an object")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{context} must be a nonempty string")
    return value


def _path(value: Any, *, root: Path, context: str) -> Path:
    declared = Path(_text(value, context)).expanduser()
    return (declared if declared.is_absolute() else root / declared).resolve()


def load_inputs(config_path: Path) -> AssemblyInputs:
    """Load the sole strict semantic configuration for comparison assembly."""

    config_path = config_path.resolve()
    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read comparison inputs at {config_path}") from exc
    value = _object(
        strict_json_loads(raw, context=f"comparison inputs {config_path}"),
        "comparison inputs",
    )
    unknown = set(value) - _TOP_LEVEL_FIELDS
    missing = _TOP_LEVEL_FIELDS - set(value)
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ContractError("comparison inputs: " + "; ".join(details))

    root_declared = Path(_text(value["workspace_root"], "workspace_root")).expanduser()
    root = (
        root_declared
        if root_declared.is_absolute()
        else config_path.parent / root_declared
    ).resolve()
    bootstrap = _object(value["bootstrap"], "bootstrap")
    if set(bootstrap) != {"seed", "resamples"}:
        raise ContractError("bootstrap must contain exactly seed and resamples")
    seed = bootstrap["seed"]
    resamples = bootstrap["resamples"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ContractError("bootstrap.seed must be a non-negative integer")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ContractError("bootstrap.resamples must be a positive integer")

    raw_models = value["llm_models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ContractError("llm_models must be a nonempty array")
    model_fields = {
        "model_id",
        "action_id",
        "run_id",
        "served_model",
        "provider_model_id",
        "bundle_manifest",
    }
    configured_models: list[LlmModelInput] = []
    for index, raw_model in enumerate(raw_models):
        model = _object(raw_model, f"llm_models[{index}]")
        if set(model) != model_fields:
            raise ContractError(
                f"llm_models[{index}] must contain exactly {sorted(model_fields)}"
            )
        raw_action_id = model["action_id"]
        action_id = (
            None
            if raw_action_id is None
            else _text(raw_action_id, f"llm_models[{index}].action_id")
        )
        bundle = _path(
            model["bundle_manifest"],
            root=root,
            context=f"llm_models[{index}].bundle_manifest",
        )
        if bundle.name != "manifest.json":
            raise ContractError(
                f"llm_models[{index}].bundle_manifest must end in manifest.json"
            )
        configured_models.append(
            LlmModelInput(
                model_id=_text(model["model_id"], f"llm_models[{index}].model_id"),
                action_id=action_id,
                run_id=_text(model["run_id"], f"llm_models[{index}].run_id"),
                served_model=_text(
                    model["served_model"], f"llm_models[{index}].served_model"
                ),
                provider_model_id=_text(
                    model["provider_model_id"],
                    f"llm_models[{index}].provider_model_id",
                ),
                bundle_manifest=bundle,
            )
        )
    for field in ("model_id", "action_id", "run_id", "provider_model_id"):
        values = [
            getattr(model, field)
            for model in configured_models
            if getattr(model, field) is not None
        ]
        if len(set(values)) != len(values):
            raise ContractError(f"llm_models {field} values must be unique")
    bundles = [model.bundle_manifest for model in configured_models]
    if len(set(bundles)) != len(bundles):
        raise ContractError("llm_models bundle_manifest values must be unique")
    path_values = {
        field: _path(value[field], root=root, context=field) for field in _PATH_FIELDS
    }
    result = AssemblyInputs(
        workspace_root=root,
        **path_values,
        llm_models=tuple(configured_models),
        frozen_at=_text(value["frozen_at"], "frozen_at"),
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
    )
    return result
