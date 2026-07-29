"""Join frozen predictions into the two-panel canonical metrics specification.

No inference, fitting, calibration, metrics, or cost allocation occurs here.
The reader panel accepts only true five-reader re-scores.  Its LLM cost ledger
must be the exact observed subset of the shared all-source paid run; panel
totals are therefore explicitly non-additive.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from indra_belief.hashing import ordered_statement_id_sha256, sha256_file
from .production import ALL_SOURCE_KEY, READER_KEY, ProductionBundle, load_bundle


SPEC_KIND = "indra_statement_belief_evaluation_spec"
PREDICTION_UNIT = "assembled_statement"
GOLD_RULE = "released_paper_observed_positive_else_negative"
STRICT_GOLD_RULE = "strict_e0_resolved_only"
ALL_SOURCE_PANEL = "paper_all_source"
READER_PANEL = "paper_readers"
ALL_SOURCE_ROWS = 1_689
READER_ROWS = 1_676
ALL_SOURCE_STRICT_ROWS = 1_578
READER_STRICT_ROWS = 1_565
READER_SOURCES = ("reach", "sparser", "medscan", "rlimsp", "trips")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
EXCLUDED_COST_CATEGORIES = [
    "training",
    "local_aggregation",
    "feature_materialization",
    "upstream_reading",
]


class AssemblyError(ValueError):
    """An input cannot support a scientifically comparable panel."""


def _fail(context: str, message: str) -> NoReturn:
    raise AssemblyError(f"{context}: {message}")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(context, "expected an object")
    return value


def _array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(context, "expected an array")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(context, "expected a non-empty string")
    return value


def _identifier(value: Any, context: str) -> str:
    text = _text(value, context)
    if IDENTIFIER.fullmatch(text) is None:
        _fail(context, "expected a stable identifier")
    return text


def _llm_label(model_id: str) -> str:
    words = model_id.removeprefix("llm_").split("_")
    abbreviations = {"e2b": "E2B", "glm": "GLM", "llm": "LLM"}
    return " ".join(abbreviations.get(word.lower(), word.title()) for word in words)


def _digest(value: Any, context: str) -> str:
    text = _text(value, context).lower()
    if HEX64.fullmatch(text) is None:
        _fail(context, "expected a SHA-256 digest")
    return text


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssemblyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"{context}: cannot read JSON at {path}: {exc}") from exc
    return _object(value, context)


def _load_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise AssemblyError(f"{context}: cannot open {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                _fail(f"{context}:{line_number}", "blank lines are forbidden")
            try:
                value = json.loads(line, object_pairs_hook=_strict_pairs)
            except json.JSONDecodeError as exc:
                raise AssemblyError(
                    f"{context}:{line_number}: invalid JSON: {exc}"
                ) from exc
            rows.append(_object(value, f"{context}:{line_number}"))
    if not rows:
        _fail(context, "expected at least one row")
    return rows


def _canonical_text(value: Any, context: str) -> str:
    if isinstance(value, str):
        return _text(value, context)
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise AssemblyError(f"{context}: not canonical-JSON serializable: {exc}") from exc


def _path_from_descriptor(
    descriptor: Mapping[str, Any], *, base: Path, context: str
) -> Path:
    raw = Path(_text(descriptor.get("path"), f"{context}.path"))
    return (raw if raw.is_absolute() else base / raw).resolve()


def _verify_descriptor(
    value: Any,
    *,
    base: Path,
    context: str,
    expected_rows: int | None = None,
) -> tuple[dict[str, Any], Path]:
    descriptor = _object(value, context)
    path = _path_from_descriptor(descriptor, base=base, context=context)
    expected_sha = _digest(descriptor.get("sha256"), f"{context}.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        _fail(context, f"digest mismatch ({actual_sha} != {expected_sha})")
    if "bytes" in descriptor:
        size = descriptor["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail(f"{context}.bytes", "expected a non-negative integer")
        if path.stat().st_size != size:
            _fail(context, "byte count differs from descriptor")
    if expected_rows is not None:
        declared = descriptor.get("rows")
        if declared != expected_rows:
            _fail(context, f"expected descriptor rows={expected_rows}, got {declared!r}")
    return descriptor, path


def _reference(path: Path, output_parent: Path) -> str:
    try:
        return os.path.relpath(path, output_parent)
    except ValueError as exc:  # different Windows drives
        raise AssemblyError("public artifact path cannot be made relative") from exc


def _binding_reference(path: Path, output_parent: Path) -> str:
    resolved = path.resolve()
    return f"{_reference(resolved, output_parent)}@{sha256_file(resolved)}"


def _same_binding(value: Any, expected_sha: str, context: str) -> None:
    obj = _object(value, context)
    if _digest(obj.get("sha256"), f"{context}.sha256") != expected_sha:
        _fail(context, "does not bind the frozen comparison substrate")


def _iso_timestamp(value: str) -> str:
    text = _text(value, "frozen_at")
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AssemblyError("frozen_at: expected an ISO-8601 timestamp") from exc
    return text


@dataclass(frozen=True)
class LlmModelInput:
    """One ordered LLM arm declaration and its exact runtime identity."""

    model_id: str
    action_id: str | None
    run_id: str
    served_model: str
    provider_model_id: str
    bundle_manifest: Path


@dataclass(frozen=True)
class AssemblyInputs:
    """Explicit immutable inputs for one comparison assembly."""

    workspace_root: Path
    gold_manifest: Path
    paper_reproduction_manifest: Path
    published_metrics: Path
    current_simple_manifest: Path
    current_bayesian_manifest: Path
    current_hierarchy_manifest: Path
    current_counts_all_source_manifest: Path
    current_counts_reader_manifest: Path
    scorer_registry: Path
    production_hybrid_manifest: Path
    error_review_protocol: Path
    llm_models: tuple[LlmModelInput, ...]
    frozen_at: str
    bootstrap_seed: int = 20_260_717
    bootstrap_resamples: int = 10_000


@dataclass(frozen=True)
class _ScorerRegistry:
    path: Path
    sha256: str
    bytes: int
    rows: Mapping[str, Mapping[str, Any]]

    def require(self, scorer_id: str, class_name: str) -> Mapping[str, Any]:
        row = self.rows.get(scorer_id)
        if row is None:
            _fail("scorer registry", f"missing required scorer {scorer_id!r}")
        if row.get("class") != class_name:
            _fail(
                "scorer registry",
                f"{scorer_id!r} does not identify {class_name!r}",
            )
        return row


@dataclass(frozen=True)
class _PanelGold:
    panel_id: str
    label: str
    path: Path
    sha256: str
    statement_ids: tuple[str, ...]
    labels_and_folds: Mapping[str, tuple[int, int]]
    evaluation_sha256: str


def _gold_rows(
    descriptor: Mapping[str, Any],
    *,
    root: Path,
    panel_id: str,
    expected_rows: int,
) -> _PanelGold:
    _desc, path = _verify_descriptor(
        descriptor,
        base=root,
        context=f"{panel_id} gold",
        expected_rows=expected_rows,
    )
    rows = _load_jsonl(path, f"{panel_id} gold")
    if len(rows) != expected_rows:
        _fail(f"{panel_id} gold", f"expected {expected_rows} physical rows")
    ids: list[str] = []
    values: dict[str, tuple[int, int]] = {}
    folds: set[int] = set()
    for index, row in enumerate(rows):
        if set(row) != {"statement_id", "label", "fold_id"}:
            _fail(f"{panel_id} gold[{index}]", "unexpected row schema")
        statement_id = _text(row["statement_id"], f"{panel_id} gold[{index}].statement_id")
        label = row["label"]
        fold = row["fold_id"]
        if isinstance(label, bool) or label not in (0, 1):
            _fail(f"{panel_id} gold[{index}].label", "expected integer 0 or 1")
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold <= 9:
            _fail(f"{panel_id} gold[{index}].fold_id", "expected an integer from 0 to 9")
        if statement_id in values:
            _fail(f"{panel_id} gold", f"duplicate statement_id {statement_id!r}")
        ids.append(statement_id)
        values[statement_id] = (label, fold)
        folds.add(fold)
    if folds != set(range(10)):
        _fail(f"{panel_id} gold", "expected all ten frozen paper folds")
    evaluation_sha = ordered_statement_id_sha256(ids)
    if descriptor.get("ordered_statement_id_sha256") not in (None, evaluation_sha):
        _fail(f"{panel_id} gold", "ordered statement-ID digest differs")
    label = (
        "2023 INDRA paper released all-source panel"
        if panel_id == ALL_SOURCE_PANEL
        else "2023 INDRA paper five-reader panel"
    )
    return _PanelGold(
        panel_id=panel_id,
        label=label,
        path=path,
        sha256=sha256_file(path),
        statement_ids=tuple(ids),
        labels_and_folds=values,
        evaluation_sha256=evaluation_sha,
    )


def _load_gold(
    inputs: AssemblyInputs,
) -> tuple[
    dict[str, Any],
    str,
    str,
    _PanelGold,
    _PanelGold,
    _PanelGold,
    _PanelGold,
]:
    root = inputs.workspace_root.resolve()
    manifest_path = inputs.gold_manifest.resolve()
    manifest = _load_json(manifest_path, "paper gold manifest")
    if manifest.get("artifact_kind") != "indra_paper_comparison_gold":
        _fail("paper gold manifest", "unexpected artifact kind")
    outputs = _object(manifest.get("outputs"), "paper gold manifest.outputs")
    all_gold = _gold_rows(
        _object(outputs.get("paper_released_gold"), "paper released gold descriptor"),
        root=root,
        panel_id=ALL_SOURCE_PANEL,
        expected_rows=ALL_SOURCE_ROWS,
    )
    reader_gold = _gold_rows(
        _object(
            outputs.get("paper_reader_eligible_released_gold"),
            "paper reader gold descriptor",
        ),
        root=root,
        panel_id=READER_PANEL,
        expected_rows=READER_ROWS,
    )
    strict_gold = _gold_rows(
        _object(
            outputs.get("paper_strict_e0_resolved_gold"),
            "paper strict E0 gold descriptor",
        ),
        root=root,
        panel_id=ALL_SOURCE_PANEL,
        expected_rows=ALL_SOURCE_STRICT_ROWS,
    )
    reader_strict_gold = _gold_rows(
        _object(
            outputs.get("paper_reader_eligible_strict_e0_resolved_gold"),
            "paper reader strict E0 gold descriptor",
        ),
        root=root,
        panel_id=READER_PANEL,
        expected_rows=READER_STRICT_ROWS,
    )
    all_index = {value: index for index, value in enumerate(all_gold.statement_ids)}
    positions: list[int] = []
    for statement_id in reader_gold.statement_ids:
        if statement_id not in all_index:
            _fail(READER_PANEL, f"statement {statement_id!r} is outside all-source gold")
        if reader_gold.labels_and_folds[statement_id][0] != all_gold.labels_and_folds[
            statement_id
        ][0]:
            _fail(READER_PANEL, f"gold label differs for {statement_id!r}")
        positions.append(all_index[statement_id])
    if positions != sorted(positions):
        _fail(READER_PANEL, "reader IDs are not an order-preserving all-source subset")

    def validate_strict(released: _PanelGold, strict: _PanelGold, context: str) -> None:
        released_index = {
            statement_id: index for index, statement_id in enumerate(released.statement_ids)
        }
        strict_positions: list[int] = []
        for statement_id in strict.statement_ids:
            if statement_id not in released_index:
                _fail(context, f"strict statement {statement_id!r} is outside released gold")
            if strict.labels_and_folds[statement_id][0] != released.labels_and_folds[
                statement_id
            ][0]:
                _fail(context, f"strict/released label differs for {statement_id!r}")
            strict_positions.append(released_index[statement_id])
        if strict_positions != sorted(strict_positions):
            _fail(context, "strict IDs are not an order-preserving released-gold subset")
        unresolved = set(released.statement_ids) - set(strict.statement_ids)
        if any(released.labels_and_folds[statement_id][0] != 0 for statement_id in unresolved):
            _fail(context, "strict-unresolved cohort contains a released positive label")

    validate_strict(all_gold, strict_gold, f"{ALL_SOURCE_PANEL} strict E0")
    validate_strict(reader_gold, reader_strict_gold, f"{READER_PANEL} strict E0")
    expected_reader_strict = [
        statement_id
        for statement_id in strict_gold.statement_ids
        if statement_id in set(reader_gold.statement_ids)
    ]
    if list(reader_strict_gold.statement_ids) != expected_reader_strict:
        _fail(READER_PANEL, "reader strict gold is not the exact strict all-source reader subset")
    target_desc = _object(outputs.get("paper_prediction_targets"), "prediction targets")
    _verify_descriptor(
        target_desc,
        base=root,
        context="paper prediction targets",
        expected_rows=ALL_SOURCE_ROWS,
    )
    return (
        manifest,
        sha256_file(manifest_path),
        _digest(target_desc.get("sha256"), "prediction targets.sha256"),
        all_gold,
        reader_gold,
        strict_gold,
        reader_strict_gold,
    )


def _threshold(inputs: AssemblyInputs, output_parent: Path) -> dict[str, Any]:
    path = inputs.error_review_protocol.resolve()
    protocol = _load_json(path, "error-review protocol")
    definition = _object(protocol.get("error_definition"), "error-review error_definition")
    if (
        definition.get("primary_threshold") != 0.5
        or definition.get("operator") != "greater_than_or_equal"
    ):
        _fail("error-review protocol", "the canonical universal threshold must be >= 0.5")
    return {
        "status": "available",
        "value": 0.5,
        "operator": "greater_than_or_equal",
        "source_path": _reference(path, output_parent),
        "source_sha256": sha256_file(path),
        "frozen_at": _text(protocol.get("frozen_at"), "error-review frozen_at"),
    }


def _load_scorer_registry(
    inputs: AssemblyInputs,
    *,
    production_bundle: ProductionBundle,
) -> _ScorerRegistry:
    """Bind the current-scorer taxonomy to the exact recovered replay evidence."""

    path = inputs.scorer_registry.resolve()
    registry = _load_json(path, "scorer registry")
    expected_fields = {
        "frozen_at",
        "scope",
        "source_snapshots",
        "installed_runtime",
        "taxonomy",
        "scorers",
        "comparison_axes",
        "audit_conclusions",
    }
    if set(registry) != expected_fields:
        _fail(
            "scorer registry",
            f"expected exactly {sorted(expected_fields)}, got {sorted(registry)}",
        )
    try:
        datetime.fromisoformat(_text(registry["frozen_at"], "scorer registry.frozen_at"))
    except ValueError as exc:
        raise AssemblyError("scorer registry.frozen_at: expected an ISO-8601 date") from exc

    taxonomy = _object(registry["taxonomy"], "scorer registry.taxonomy")
    expected_classes = {
        "BeliefScorer",
        "SimpleScorer",
        "BayesianScorer",
        "SklearnScorer",
        "CountsScorer",
        "HybridScorer",
    }
    if (
        taxonomy.get("ast_exhaustive_class_count") != len(expected_classes)
        or set(_array(taxonomy.get("classes"), "scorer registry.taxonomy.classes"))
        != expected_classes
        or taxonomy.get("deprecated_classes") != []
    ):
        _fail("scorer registry.taxonomy", "current INDRA scorer taxonomy differs")

    rows: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(_array(registry["scorers"], "scorer registry.scorers")):
        row = _object(value, f"scorer registry.scorers[{index}]")
        scorer_id = _identifier(
            row.get("scorer_id"), f"scorer registry.scorers[{index}].scorer_id"
        )
        if scorer_id in rows:
            _fail("scorer registry", f"duplicate scorer_id {scorer_id!r}")
        rows[scorer_id] = row

    required = {
        "indra_1.24.0_simple_default": "indra.belief.SimpleScorer",
        "indra_1.24.0_bayesian_unfitted": "indra.belief.BayesianScorer",
        "indra_1.24.0_counts_unfitted": "indra.belief.skl.CountsScorer",
        "indra_1.24.0_hybrid_unfitted": "indra.belief.skl.HybridScorer",
        "indra_db_7dc8bf5_cogex_hybrid_production": "indra.belief.skl.HybridScorer",
    }
    for scorer_id, class_name in required.items():
        row = rows.get(scorer_id)
        if row is None or row.get("class") != class_name:
            _fail("scorer registry", f"required identity {scorer_id!r} differs")

    installed = _object(registry["installed_runtime"], "scorer registry.installed_runtime")
    simple_sha = _digest(
        installed.get("belief_init_sha256"),
        "scorer registry.installed_runtime.belief_init_sha256",
    )
    sklearn_sha = _digest(
        installed.get("belief_skl_sha256"),
        "scorer registry.installed_runtime.belief_skl_sha256",
    )
    prior_sha = _digest(
        installed.get("default_prior_resource_sha256"),
        "scorer registry.installed_runtime.default_prior_resource_sha256",
    )
    if installed.get("matches_current_release_implementation") is not True:
        _fail("scorer registry.installed_runtime", "release implementation is not frozen")
    for scorer_id in ("indra_1.24.0_simple_default", "indra_1.24.0_bayesian_unfitted"):
        if rows[scorer_id].get("implementation_sha256") != simple_sha:
            _fail("scorer registry", f"{scorer_id!r} implementation digest differs")
    if rows["indra_1.24.0_simple_default"].get("resource_sha256") != prior_sha:
        _fail("scorer registry", "SimpleScorer prior digest differs")
    for scorer_id in ("indra_1.24.0_counts_unfitted", "indra_1.24.0_hybrid_unfitted"):
        if rows[scorer_id].get("implementation_sha256") != sklearn_sha:
            _fail("scorer registry", f"{scorer_id!r} implementation digest differs")

    production = rows["indra_db_7dc8bf5_cogex_hybrid_production"]
    if (
        production.get("kind") != "recovered_fitted_scorer"
        or production.get("artifact_access") != "recovered_authenticated_download"
        or production.get("official_status")
        != [
            "artifact_referenced_by_current_cogex_export_pipeline",
            "recovered_storage_artifact",
        ]
        or production.get("benchmark_decision")
        != "include_recovered_replay_as_descriptive_nonconfirmatory_only"
        or production.get("artifact_sha256") != production_bundle.model.sha256
        or production.get("artifact_bytes") != production_bundle.model.bytes
        or "unresolved" in production
    ):
        _fail("scorer registry production Hybrid", "recovered artifact binding differs")
    fitted = _object(
        production.get("fitted_state"), "scorer registry production Hybrid.fitted_state"
    )
    if fitted != {
        "classifier": "sklearn.ensemble._forest.RandomForestClassifier",
        "n_estimators": 2_000,
        "max_depth": 13,
        "n_features": 87,
        "source_list": list(production_bundle.source_list),
    }:
        _fail("scorer registry production Hybrid", "fitted-state contract differs")
    comparison_status = _object(
        production.get("comparison_status"),
        "scorer registry production Hybrid.comparison_status",
    )
    if comparison_status != {
        "analysis_role": "descriptive_nonconfirmatory",
        "all_source_route": "counts_only",
        "reader_route": "counts_only_counterfactual_source_projection",
        "literal_live_deployment_output": False,
        "live_deployment_parity_established": False,
        "historical_runtime_established": False,
        "private_training_provenance_available": False,
        "training_overlap_status": "unknown",
        "simple_fallback_exercised": False,
    }:
        _fail("scorer registry production Hybrid", "comparison claim boundary differs")
    _manifest_desc, manifest_path = _verify_descriptor(
        production.get("artifact_provenance_manifest"),
        base=inputs.workspace_root.resolve(),
        context="scorer registry production Hybrid manifest",
    )
    if manifest_path != inputs.production_hybrid_manifest.resolve():
        _fail("scorer registry production Hybrid manifest", "path differs from comparison input")

    return _ScorerRegistry(
        path=path,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        rows=rows,
    )


def _bind_current_implementation(
    manifest: Mapping[str, Any],
    *,
    registry: _ScorerRegistry,
    scorer_id: str,
    class_name: str,
    module_key: str,
    context: str,
) -> None:
    row = registry.require(scorer_id, class_name)
    implementation = _object(manifest.get("implementation"), f"{context}.implementation")
    descriptor = _object(
        implementation.get(module_key), f"{context}.implementation.{module_key}"
    )
    if descriptor.get("sha256") != row.get("implementation_sha256"):
        _fail(context, "scorer implementation does not bind the canonical registry")


def _prediction_rows(
    value: Any,
    *,
    base: Path,
    gold: _PanelGold,
    context: str,
) -> tuple[dict[str, str], Path]:
    descriptor, path = _verify_descriptor(
        value, base=base, context=context, expected_rows=len(gold.statement_ids)
    )
    rows = _load_jsonl(path, context)
    if len(rows) != len(gold.statement_ids):
        _fail(context, "physical row count differs from panel")
    observed: dict[str, float] = {}
    for index, row in enumerate(rows):
        if set(row) != {"statement_id", "probability_correct"}:
            _fail(f"{context}[{index}]", "unexpected prediction row schema")
        statement_id = _text(row["statement_id"], f"{context}[{index}].statement_id")
        probability = row["probability_correct"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            _fail(f"{context}[{index}].probability_correct", "expected a number")
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            _fail(f"{context}[{index}].probability_correct", "outside [0, 1]")
        if statement_id in observed:
            _fail(context, f"duplicate prediction for {statement_id!r}")
        observed[statement_id] = probability
    expected = set(gold.statement_ids)
    if set(observed) != expected:
        missing = expected - set(observed)
        extra = set(observed) - expected
        _fail(context, f"panel coverage differs: missing={len(missing)}, extra={len(extra)}")
    if observed and all(value == 1.0 for value in observed.values()):
        _fail(context, "constant all-1.0 belief is not a valid comparison arm")
    return {"path": str(path), "sha256": descriptor["sha256"]}, path


def _output_descriptor(
    prediction: Mapping[str, str], *, output_parent: Path
) -> dict[str, str]:
    return {
        "path": _reference(Path(prediction["path"]), output_parent),
        "sha256": prediction["sha256"],
    }


def _implementation_environment(manifest: Mapping[str, Any]) -> str:
    for key in ("runtime_observation", "runtime", "implementation"):
        if key in manifest:
            return _canonical_text(manifest[key], f"manifest.{key}")
    return "environment recorded in source manifest"


def _arm(
    *,
    arm_id: str,
    label: str,
    family: str,
    prediction: Mapping[str, str],
    source_manifest_path: Path,
    implementation: str,
    training_sha256: str | None,
    environment: str,
    notes: str,
    threshold: Mapping[str, Any],
    cost: Mapping[str, Any] | None,
    output_parent: Path,
) -> dict[str, Any]:
    return {
        "arm_id": _identifier(arm_id, "arm_id"),
        "label": _text(label, f"{arm_id}.label"),
        "family": family,
        "predictions": _output_descriptor(prediction, output_parent=output_parent),
        "implementation": {
            "implementation": _text(implementation, f"{arm_id}.implementation"),
            "implementation_digest": sha256_file(source_manifest_path),
            "training_data_sha256": training_sha256,
            "environment": _text(environment, f"{arm_id}.environment"),
            "notes": _text(notes, f"{arm_id}.notes"),
        },
        "threshold": dict(threshold),
        "cost": dict(cost)
        if cost is not None
        else {
            "status": "unavailable",
            "reason": "historical or local compute has no comparable retry-inclusive provider price",
        },
    }


def _find_arm(
    manifest: Mapping[str, Any], predicate: Any, context: str
) -> dict[str, Any]:
    matches = [
        _object(value, f"{context}.arms")
        for value in _array(manifest.get("arms"), f"{context}.arms")
        if predicate(_object(value, f"{context}.arms"))
    ]
    if len(matches) != 1:
        _fail(context, f"expected exactly one matching arm, found {len(matches)}")
    return matches[0]


def _current_output(
    manifest: Mapping[str, Any], key: str, context: str
) -> dict[str, Any]:
    return _object(
        _object(manifest.get("outputs"), f"{context}.outputs").get(key),
        f"{context}.outputs.{key}",
    )


def _append_static(
    target: list[dict[str, Any]],
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    record: Mapping[str, Any],
    output_key: str,
    gold: _PanelGold,
    label: str,
    root: Path,
    threshold: Mapping[str, Any],
    output_parent: Path,
    trained: bool = False,
    notes: str,
) -> None:
    prediction, _ = _prediction_rows(
        _current_output(manifest, output_key, label),
        base=root,
        gold=gold,
        context=f"{label} predictions",
    )
    target.append(
        _arm(
            arm_id=_identifier(record.get("arm_id"), f"{label}.arm_id"),
            label=label,
            family="current",
            prediction=prediction,
            source_manifest_path=manifest_path,
            implementation=_text(record.get("class"), f"{label}.class"),
            training_sha256=gold.sha256 if trained else None,
            environment=_implementation_environment(manifest),
            notes=notes,
            threshold=threshold,
            cost=None,
            output_parent=output_parent,
        )
    )


def _paper_arms(
    inputs: AssemblyInputs,
    *,
    gold_manifest_sha: str,
    all_gold: _PanelGold,
    reader_gold: _PanelGold,
    threshold: Mapping[str, Any],
    output_parent: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = inputs.workspace_root.resolve()
    path = inputs.paper_reproduction_manifest.resolve()
    manifest = _load_json(path, "paper reproduction manifest")
    if manifest.get("artifact_kind") != "indra_paper_headline_deterministic_reproduction":
        _fail("paper reproduction manifest", "unexpected artifact kind")
    source_inputs = _object(manifest.get("inputs"), "paper reproduction inputs")
    _same_binding(
        source_inputs.get("comparison_gold_manifest"),
        gold_manifest_sha,
        "paper reproduction comparison gold",
    )
    _same_binding(source_inputs.get("released_gold"), all_gold.sha256, "paper reproduction all gold")
    _same_binding(source_inputs.get("reader_gold"), reader_gold.sha256, "paper reproduction reader gold")
    published_path = inputs.published_metrics.resolve()
    published = _load_json(published_path, "published paper metrics")
    if published.get("artifact_kind") != "indra_assembly_paper_published_method_metrics":
        _fail("published paper metrics", "unexpected artifact kind")
    methods = _array(published.get("methods"), "published paper metrics.methods")
    results = _object(manifest.get("results"), "paper reproduction results")
    outputs = _object(manifest.get("outputs"), "paper reproduction outputs")
    specifications = (
        (
            ALL_SOURCE_PANEL,
            all_gold,
            "rf_promoter_minimal_predictions",
            "rf_2k_d13_type_pmids_promoter_all_sources_specific",
        ),
        (
            ALL_SOURCE_PANEL,
            all_gold,
            "rf_promoter_avglen_minimal_predictions",
            "rf_2k_d13_type_pmids_promoter_avglen_all_sources_specific",
        ),
        (READER_PANEL, reader_gold, "orig_belief_readers_minimal_predictions", "orig_belief_readers"),
    )
    arms: dict[str, list[dict[str, Any]]] = {ALL_SOURCE_PANEL: [], READER_PANEL: []}
    for panel_id, gold, output_key, result_key in specifications:
        result = _object(results.get(result_key), f"paper reproduction result {result_key}")
        display_name = _text(result.get("display_name"), f"{result_key}.display_name")
        matches = [
            _object(row, "published method")
            for row in methods
            if isinstance(row, dict) and row.get("method") == display_name
        ]
        if len(matches) != 1:
            _fail(result_key, "published metric anchor is missing or ambiguous")
        published_row = matches[0]
        if (
            published_row.get("fold_mean_trapezoidal_pr_auc")
            != result.get("published_rounded_mean")
            or published_row.get("fold_population_sd")
            != result.get("published_rounded_population_std")
            or result.get("rounded_headline_match") is not True
        ):
            _fail(result_key, "reproduced and published reference metrics do not reconcile")
        prediction, _ = _prediction_rows(
            outputs.get(output_key), base=root, gold=gold, context=f"{result_key} predictions"
        )
        reference = {
            "published_method_id": published_row.get("method_id"),
            "published_fold_mean_trapezoidal_pr_auc": published_row.get(
                "fold_mean_trapezoidal_pr_auc"
            ),
            "published_fold_population_sd": published_row.get("fold_population_sd"),
            "reproduced_fold_mean_trapezoidal_pr_auc": result.get(
                "fold_mean_trapezoidal_pr_auc"
            ),
            "reproduced_fold_population_sd": result.get("fold_population_std"),
        }
        arms[panel_id].append(
            _arm(
                arm_id=result_key,
                label=f"2023 paper semantic reconstruction — {display_name}",
                family="paper",
                prediction=prediction,
                source_manifest_path=path,
                implementation=f"deterministic semantic reconstruction of {display_name}",
                training_sha256=gold.sha256,
                environment=_implementation_environment(manifest),
                notes=(
                    f"published_reference={_canonical_text(reference, result_key)}; "
                    f"published_manifest={_binding_reference(published_path, output_parent)}; "
                    "matches the published rounded headline under reconstructed folds, "
                    "but the realized historical folds and fitted state were not published"
                ),
                threshold=threshold,
                cost=None,
                output_parent=output_parent,
            )
        )
    return arms[ALL_SOURCE_PANEL], arms[READER_PANEL]


def _current_arms(
    inputs: AssemblyInputs,
    *,
    registry: _ScorerRegistry,
    gold_manifest_sha: str,
    all_gold: _PanelGold,
    reader_gold: _PanelGold,
    threshold: Mapping[str, Any],
    output_parent: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    root = inputs.workspace_root.resolve()
    all_arms: list[dict[str, Any]] = []
    reader_arms: list[dict[str, Any]] = []
    excluded_all: list[dict[str, str]] = []
    excluded_reader: list[dict[str, str]] = []

    def load(path: Path, kind: str, name: str) -> dict[str, Any]:
        value = _load_json(path.resolve(), name)
        if value.get("artifact_kind") != kind:
            _fail(name, "unexpected artifact kind")
        return value

    simple_path = inputs.current_simple_manifest.resolve()
    simple = load(simple_path, "current_indra_simple_paper_predictions", "current Simple manifest")
    _bind_current_implementation(
        simple,
        registry=registry,
        scorer_id="indra_1.24.0_simple_default",
        class_name="indra.belief.SimpleScorer",
        module_key="indra_belief_module",
        context="current Simple manifest",
    )
    _same_binding(
        _object(simple.get("inputs"), "current Simple inputs").get("prediction_targets_manifest"),
        gold_manifest_sha,
        "current Simple comparison manifest",
    )
    _append_static(
        all_arms,
        manifest=simple,
        manifest_path=simple_path,
        record=_object(simple.get("arm"), "current Simple arm"),
        output_key="predictions",
        gold=all_gold,
        label="Current INDRA SimpleScorer — direct all source",
        root=root,
        threshold=threshold,
        output_parent=output_parent,
        notes="direct all-source input; reader-eligible row subset is diagnostic and excluded",
    )

    bayes_path = inputs.current_bayesian_manifest.resolve()
    bayes = load(bayes_path, "current_indra_bayesian_paper_predictions", "current Bayesian manifest")
    _bind_current_implementation(
        bayes,
        registry=registry,
        scorer_id="indra_1.24.0_bayesian_unfitted",
        class_name="indra.belief.BayesianScorer",
        module_key="indra_belief_module",
        context="current Bayesian manifest",
    )
    bayes_inputs = _object(bayes.get("inputs"), "current Bayesian inputs")
    _same_binding(bayes_inputs.get("comparison_manifest"), gold_manifest_sha, "Bayesian comparison manifest")
    _same_binding(bayes_inputs.get("all_source_gold_and_folds"), all_gold.sha256, "Bayesian all gold")
    _same_binding(bayes_inputs.get("reader_gold_and_folds"), reader_gold.sha256, "Bayesian reader gold")
    bayes_specs = (
        (ALL_SOURCE_PANEL, all_gold, "indra.belief.BayesianScorer", False, "current_bayesian_source_oof_all_sources_predictions.jsonl", "source OOF all source"),
        (ALL_SOURCE_PANEL, all_gold, "indra.belief.BayesianScorer", True, "current_bayesian_source_subtype_oof_all_sources_predictions.jsonl", "source+subtype OOF all source"),
        (READER_PANEL, reader_gold, "indra.belief.SimpleScorer", False, "current_simple_direct_readers_only_predictions.jsonl", "direct five-reader input"),
        (READER_PANEL, reader_gold, "indra.belief.BayesianScorer", False, "current_bayesian_source_oof_readers_only_predictions.jsonl", "source OOF five-reader input"),
        (READER_PANEL, reader_gold, "indra.belief.BayesianScorer", True, "current_bayesian_source_subtype_oof_readers_only_predictions.jsonl", "source+subtype OOF five-reader input"),
    )
    for panel_id, gold, class_name, subtype, output_key, suffix in bayes_specs:
        source_panel_id = (
            "all_sources_1689" if panel_id == ALL_SOURCE_PANEL else "readers_only_1676"
        )
        record = _find_arm(
            bayes,
            lambda row, p=source_panel_id, c=class_name, s=subtype: row.get("panel_id") == p
            and row.get("class") == c
            and (c == "indra.belief.SimpleScorer" or row.get("subtype_counts") is s),
            suffix,
        )
        if panel_id == READER_PANEL and tuple(record.get("input_sources", ())) != READER_SOURCES:
            _fail(suffix, "reader arm is not restricted to the five frozen readers")
        _append_static(
            all_arms if panel_id == ALL_SOURCE_PANEL else reader_arms,
            manifest=bayes,
            manifest_path=bayes_path,
            record=record,
            output_key=output_key,
            gold=gold,
            label=f"Current INDRA {class_name.rsplit('.', 1)[-1]} — {suffix}",
            root=root,
            threshold=threshold,
            output_parent=output_parent,
            trained=record.get("training_required") is not False,
            notes=f"panel={panel_id}; exact panel-specific scorer output",
        )

    hierarchy_path = inputs.current_hierarchy_manifest.resolve()
    hierarchy = load(
        hierarchy_path,
        "current_indra_simple_hierarchy_paper_predictions",
        "current hierarchy manifest",
    )
    _bind_current_implementation(
        hierarchy,
        registry=registry,
        scorer_id="indra_1.24.0_simple_default",
        class_name="indra.belief.SimpleScorer",
        module_key="indra_belief_module",
        context="current hierarchy manifest",
    )
    hierarchy_inputs = _object(hierarchy.get("inputs"), "hierarchy inputs")
    _same_binding(hierarchy_inputs.get("comparison_manifest"), gold_manifest_sha, "hierarchy comparison manifest")
    _same_binding(hierarchy_inputs.get("all_source_gold_and_folds"), all_gold.sha256, "hierarchy all gold")
    _same_binding(hierarchy_inputs.get("reader_gold_and_folds"), reader_gold.sha256, "hierarchy reader gold")
    hierarchy_specs = (
        (ALL_SOURCE_PANEL, all_gold, ALL_SOURCE_ROWS, "current_simple_hierarchy_all_sources_predictions.jsonl", "hierarchy all source"),
        (READER_PANEL, reader_gold, READER_ROWS, "current_simple_hierarchy_readers_only_predictions.jsonl", "hierarchy five-reader input"),
    )
    for panel_id, gold, rows, output_key, suffix in hierarchy_specs:
        record = _find_arm(
            hierarchy,
            lambda row, n=rows, p=panel_id: row.get("rows") == n
            and row.get("hierarchy_propagation") is True
            and (p == ALL_SOURCE_PANEL or tuple(row.get("input_sources", ())) == READER_SOURCES),
            suffix,
        )
        _append_static(
            all_arms if panel_id == ALL_SOURCE_PANEL else reader_arms,
            manifest=hierarchy,
            manifest_path=hierarchy_path,
            record=record,
            output_key=output_key,
            gold=gold,
            label=f"Current INDRA SimpleScorer — {suffix}",
            root=root,
            threshold=threshold,
            output_parent=output_parent,
            notes=f"panel={panel_id}; hierarchy evidence was re-scored for this panel",
        )

    count_specs = (
        (
            inputs.current_counts_all_source_manifest.resolve(),
            "current_indra_counts_hybrid_paper_predictions",
            ALL_SOURCE_PANEL,
            "all_sources_1689",
            all_gold,
            (
                ("source_only", "current_counts_source_only_oof_predictions.jsonl", "source-only"),
                ("full_features", "current_counts_full_features_oof_predictions.jsonl", "full-feature"),
            ),
        ),
        (
            inputs.current_counts_reader_manifest.resolve(),
            "current_indra_counts_hybrid_reader_predictions",
            READER_PANEL,
            "readers_only_1676",
            reader_gold,
            (
                (
                    "source_only",
                    "current_counts_source_only_oof_readers_only_predictions.jsonl",
                    "source-only",
                ),
                (
                    "full_features",
                    "current_counts_full_features_oof_readers_only_predictions.jsonl",
                    "full-feature",
                ),
            ),
        ),
    )
    for path, kind, panel_id, source_panel_id, gold, variants in count_specs:
        manifest = load(path, kind, f"Counts manifest {panel_id}")
        if panel_id == ALL_SOURCE_PANEL:
            _bind_current_implementation(
                manifest,
                registry=registry,
                scorer_id="indra_1.24.0_counts_unfitted",
                class_name="indra.belief.skl.CountsScorer",
                module_key="indra_belief_skl_module",
                context=f"Counts manifest {panel_id}",
            )
        else:
            implementation = _object(
                manifest.get("implementation"),
                f"Counts manifest {panel_id}.implementation",
            )
            crosschecks = _object(
                implementation.get("runtime_scorer_crosschecks"),
                f"Counts manifest {panel_id}.implementation.runtime_scorer_crosschecks",
            )
            belief_skl = _object(
                crosschecks.get("belief_skl"),
                f"Counts manifest {panel_id}.implementation.runtime_scorer_crosschecks.belief_skl",
            )
            expected = registry.require(
                "indra_1.24.0_counts_unfitted", "indra.belief.skl.CountsScorer"
            ).get("implementation_sha256")
            if belief_skl.get("sha256") != expected:
                _fail(
                    f"Counts manifest {panel_id}",
                    "scorer implementation does not bind the canonical registry",
                )
        source_inputs = _object(manifest.get("inputs"), f"Counts inputs {panel_id}")
        _same_binding(source_inputs.get("comparison_manifest"), gold_manifest_sha, f"Counts comparison {panel_id}")
        if panel_id == READER_PANEL:
            _same_binding(source_inputs.get("reader_gold_and_folds"), reader_gold.sha256, "Counts reader gold")
            panel = _object(manifest.get("panel"), "Counts reader panel")
            if tuple(panel.get("input_sources", ())) != READER_SOURCES:
                _fail("Counts reader panel", "not restricted to five frozen readers")
        else:
            _same_binding(source_inputs.get("all_source_gold_and_folds"), all_gold.sha256, "Counts all gold")
        for token, output_key, display in variants:
            record = _find_arm(
                manifest,
                lambda row, token=token: row.get("panel_id") == source_panel_id
                and row.get("class") == "indra.belief.skl.CountsScorer"
                and token in str(row.get("arm_id")),
                f"Counts {panel_id} {token}",
            )
            _append_static(
                all_arms if panel_id == ALL_SOURCE_PANEL else reader_arms,
                manifest=manifest,
                manifest_path=path,
                record=record,
                output_key=output_key,
                gold=gold,
                label=f"Current INDRA CountsScorer — {display} {panel_id}",
                root=root,
                threshold=threshold,
                output_parent=output_parent,
                trained=True,
                notes=f"panel={panel_id}; label-isolated out-of-fold fitted state",
            )
        aliases = [
            row
            for row in _array(manifest.get("arms"), f"Counts {panel_id}.arms")
            if isinstance(row, dict) and row.get("class") == "indra.belief.skl.HybridScorer"
        ]
        for alias in aliases:
            (excluded_all if panel_id == ALL_SOURCE_PANEL else excluded_reader).append(
                _excluded(
                    _text(alias.get("arm_id"), "local Hybrid arm_id"),
                    "Local HybridScorer class-path audit",
                    "current",
                    "performance-identical alias of the fitted CountsScorer arm; not a distinct model point",
                    "a non-redundant fitted Hybrid scorer with exercised Simple fallback",
                    _binding_reference(path, output_parent),
                )
            )

    excluded_reader.extend(
        [
            _excluded(
                "current_simple_all_source_reader_row_subset",
                "SimpleScorer all-source reader-row subset",
                "current",
                "row filtering retains all-source evidence and is not a five-reader re-score",
                "panel-specific five-reader prediction ledger",
                _binding_reference(simple_path, output_parent),
            ),
            _excluded(
                "current_hierarchy_all_source_reader_row_subset",
                "Hierarchy all-source reader-row subset",
                "current",
                "row filtering retains all-source hierarchy evidence and is not a five-reader re-score",
                "panel-specific five-reader hierarchy prediction ledger",
                _binding_reference(hierarchy_path, output_parent),
            ),
        ]
    )
    return all_arms, reader_arms, excluded_all, excluded_reader


def _production_arms(
    inputs: AssemblyInputs,
    *,
    bundle: ProductionBundle,
    prediction_targets_sha: str,
    all_gold: _PanelGold,
    reader_gold: _PanelGold,
    threshold: Mapping[str, Any],
    output_parent: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    path = inputs.production_hybrid_manifest.resolve()
    if bundle.inputs["prediction_targets"].sha256 != prediction_targets_sha:
        _fail("production prediction targets", "does not bind the frozen comparison substrate")
    if bundle.inputs["reader_gold"].sha256 != reader_gold.sha256:
        _fail("production reader gold", "does not bind the frozen comparison substrate")
    all_panel = bundle.panels[ALL_SOURCE_KEY]
    reader_panel = bundle.panels[READER_KEY]
    if all_panel.statement_ids_sha256 != all_gold.evaluation_sha256:
        _fail("production all-source panel", "panel identity differs")
    if reader_panel.statement_ids_sha256 != reader_gold.evaluation_sha256:
        _fail("production reader panel", "panel identity differs")
    all_prediction, _ = _prediction_rows(
        {
            "path": str(all_panel.predictions.path),
            "sha256": all_panel.predictions.sha256,
            "bytes": all_panel.predictions.bytes,
            "rows": all_panel.predictions.rows,
        },
        base=path.parent,
        gold=all_gold,
        context="production all-source predictions",
    )
    reader_prediction, _ = _prediction_rows(
        {
            "path": str(reader_panel.predictions.path),
            "sha256": reader_panel.predictions.sha256,
            "bytes": reader_panel.predictions.bytes,
            "rows": reader_panel.predictions.rows,
        },
        base=path.parent,
        gold=reader_gold,
        context="production reader predictions",
    )
    all_arm = _arm(
        arm_id=all_panel.arm_id,
        label=f"Recovered fitted Hybrid artifact — {all_panel.label}",
        family="current",
        prediction=all_prediction,
        source_manifest_path=path,
        implementation=bundle.implementation,
        training_sha256=None,
        environment=bundle.environment,
        notes=(
            "descriptive, non-confirmatory replay of the recovered fitted Counts component "
            "through the recorded export-assembly adapter; private training provenance and "
            f"training overlap are unknown; limitations={_canonical_text(bundle.limitations, 'production limitations')}"
        ),
        threshold=threshold,
        cost=None,
        output_parent=output_parent,
    )
    reader_arm = _arm(
        arm_id=reader_panel.arm_id,
        label=f"Recovered fitted Hybrid artifact — {reader_panel.label}",
        family="current",
        prediction=reader_prediction,
        source_manifest_path=path,
        implementation=bundle.implementation,
        training_sha256=None,
        environment=bundle.environment,
        notes=(
            "descriptive, non-confirmatory counterfactual re-score of the same recovered fitted "
            "Counts component on evidence restricted to the five paper readers; this is not a "
            "literal historical deployment route; private training provenance and training "
            f"overlap are unknown; limitations={_canonical_text(bundle.limitations, 'production limitations')}"
        ),
        threshold=threshold,
        cost=None,
        output_parent=output_parent,
    )
    excluded = [
        _excluded(
            bundle.excluded_diagnostic_arm_id,
            "Production all-source reader-row projection",
            "current",
            "row-subset diagnostic retains all-source scorer inputs; it is not comparable to true five-reader re-scores",
            "the production five-reader input projection evaluated above",
            _binding_reference(path, output_parent),
        )
    ]
    return all_arm, reader_arm, excluded


def _cost_rows(
    value: Any,
    *,
    base: Path,
    panel: _PanelGold,
    context: str,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    descriptor, path = _verify_descriptor(value, base=base, context=context)
    if descriptor.get("record_type") != "evidence_execution":
        _fail(context, "expected record_type=evidence_execution")
    rows = _load_jsonl(path, context)
    if descriptor.get("rows") != len(rows):
        _fail(context, "ledger row count differs from descriptor")
    expected_ids = set(panel.statement_ids)
    covered: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if row.get("record_type") != "evidence_execution":
            _fail(context, f"cost row {index} has the wrong record_type")
        statement_id = _text(row.get("statement_id"), f"{context}[{index}].statement_id")
        execution_id = _text(
            row.get("execution_identity"), f"{context}[{index}].execution_identity"
        )
        if statement_id not in expected_ids:
            _fail(context, f"cost row crosses panel at {statement_id!r}")
        identity = (statement_id, execution_id)
        if identity in identities:
            _fail(context, f"duplicate execution {identity!r}")
        identities.add(identity)
        covered.add(statement_id)
    if covered != expected_ids:
        _fail(context, f"cost coverage misses {len(expected_ids - covered)} statements")
    return descriptor, path, rows


def _metrics_cost(
    descriptor: Mapping[str, Any],
    *,
    path: Path,
    panel: _PanelGold,
    run_id: str,
    projection: str,
    output_parent: Path,
) -> dict[str, Any]:
    if descriptor.get("projection") != projection:
        _fail(f"{run_id}.{panel.panel_id}.cost", f"expected projection={projection}")
    if (
        descriptor.get("counterfactual_run_cost") is not False
        or descriptor.get("additive_across_panels") is not False
        or descriptor.get("shared_run_id") != run_id
    ):
        _fail(f"{run_id}.{panel.panel_id}.cost", "shared-run/non-additivity annotation differs")
    accounting = _object(descriptor.get("accounting"), f"{run_id}.cost.accounting")
    required_accounting = {
        "provider_measured_cost_usd_exact",
        "conservative_reserved_cost_usd_exact",
        "accounted_cost_lower_usd_exact",
        "accounted_cost_upper_usd_exact",
        "provider_measured_call_count",
        "conservative_call_count",
        "includes_retries",
        "includes_relation_subcalls",
        "denominator",
        "excluded_cost_categories",
    }
    if set(accounting) != required_accounting:
        _fail(f"{run_id}.cost.accounting", "fields do not match the metrics cost contract")
    if accounting.get("includes_retries") is not True or accounting.get("includes_relation_subcalls") is not True:
        _fail(f"{run_id}.cost.accounting", "retry/relation calls are not fully accounted")
    if accounting.get("excluded_cost_categories") != EXCLUDED_COST_CATEGORIES:
        _fail(f"{run_id}.cost.accounting", "excluded cost categories differ")
    denominator = _object(accounting.get("denominator"), f"{run_id}.cost.denominator")
    if denominator.get("statements") != len(panel.statement_ids):
        _fail(f"{run_id}.cost.denominator", "statement denominator differs from panel")
    cost_context = f"{run_id}.{panel.panel_id}.cost"
    price = _text(descriptor.get("price_source"), f"{cost_context}.price_source")
    price_date = _text(descriptor.get("price_date"), f"{cost_context}.price_date")
    comparability_id = _identifier(
        descriptor.get("cost_comparability_id"),
        f"{cost_context}.cost_comparability_id",
    )
    pricing = _object(descriptor.get("pricing"), f"{cost_context}.pricing")
    required_pricing = {
        "cost_comparability_id",
        "currency",
        "provider",
        "provider_model_id",
        "pricing_mode",
        "region",
        "resolved_service_tier",
        "retrieved_on",
        "service_tier_request",
        "source_url",
        "tariff",
        "unit",
    }
    if set(pricing) != required_pricing:
        _fail(f"{cost_context}.pricing", "fields do not match the canonical pricing contract")
    if pricing.get("cost_comparability_id") != comparability_id:
        _fail(f"{cost_context}.pricing", "cost comparability ID differs from the descriptor")
    if (
        pricing.get("currency") != "USD"
        or pricing.get("pricing_mode") != "on_demand"
        or pricing.get("resolved_service_tier") != "standard"
        or pricing.get("service_tier_request") != "default"
        or pricing.get("unit") != "per_million_tokens"
    ):
        _fail(f"{cost_context}.pricing", "currency, mode, tier, or unit is not comparable")
    if (
        _text(pricing.get("source_url"), f"{cost_context}.pricing.source_url") != price
        or not price.startswith("https://")
        or _text(pricing.get("retrieved_on"), f"{cost_context}.pricing.retrieved_on")
        != price_date
        or DATE.fullmatch(price_date) is None
    ):
        _fail(f"{cost_context}.pricing", "source URL or retrieval date differs from the descriptor")
    for field in ("provider", "provider_model_id", "region"):
        _text(pricing.get(field), f"{cost_context}.pricing.{field}")
    tariff = _object(pricing.get("tariff"), f"{cost_context}.pricing.tariff")
    if set(tariff) != {
        "input_usd_per_million",
        "output_usd_per_million",
        "pricing_basis",
    }:
        _fail(f"{cost_context}.pricing.tariff", "fields differ from the canonical tariff")
    for field in ("input_usd_per_million", "output_usd_per_million"):
        value = _text(tariff.get(field), f"{cost_context}.pricing.tariff.{field}")
        if DECIMAL.fullmatch(value) is None:
            _fail(f"{cost_context}.pricing.tariff.{field}", "expected a canonical nonnegative decimal")
    _text(tariff.get("pricing_basis"), f"{cost_context}.pricing.tariff.pricing_basis")
    return {
        "status": "ledger",
        "record_type": "evidence_execution",
        "path": _reference(path, output_parent),
        "sha256": _digest(descriptor.get("sha256"), f"{run_id}.cost.sha256"),
        "basis": _text(descriptor.get("basis"), f"{run_id}.cost.basis"),
        "view_id": _text(descriptor.get("view_id"), f"{run_id}.cost.view_id"),
        "price_source": price,
        "price_date": price_date,
        "cost_comparability_id": comparability_id,
        "pricing": dict(pricing),
        "projection": projection,
        "counterfactual_run_cost": False,
        "shared_run_id": run_id,
        "additive_across_panels": False,
        "accounting": dict(accounting),
    }


def _llm_arms(
    inputs: AssemblyInputs,
    *,
    all_gold: _PanelGold,
    reader_gold: _PanelGold,
    threshold: Mapping[str, Any],
    output_parent: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]]:
    for declaration in inputs.llm_models:
        if declaration.bundle_manifest.exists() and not declaration.bundle_manifest.is_file():
            _fail(
                str(declaration.bundle_manifest),
                "bundle manifest target exists but is not a regular file",
            )
    available = tuple(
        declaration
        for declaration in inputs.llm_models
        if declaration.bundle_manifest.is_file()
    )
    if not available:
        _fail("llm_models", "at least one canonical model bundle is required")
    all_arms: list[dict[str, Any]] = []
    reader_arms: list[dict[str, Any]] = []
    bundled_model_ids: list[str] = []
    seen_models: set[str] = set()
    run_costs: dict[str, tuple[str, str]] = {}
    shared_cost_comparability_id: str | None = None
    for declaration in available:
        bundle_path = declaration.bundle_manifest.resolve()
        bundle = _load_json(bundle_path, f"LLM bundle {bundle_path}")
        required = {"kind", "model_id", "run_id", "implementation", "panels"}
        if set(bundle) != required:
            _fail(str(bundle_path), "unexpected canonical bundle fields")
        if bundle.get("kind") != "llm_model_bundle":
            _fail(str(bundle_path), "kind must be llm_model_bundle")
        model_id = _identifier(bundle.get("model_id"), f"{bundle_path}.model_id")
        if model_id != declaration.model_id:
            _fail(
                str(bundle_path),
                "model_id differs from its ordered LLM declaration",
            )
        if model_id in seen_models:
            _fail("llm_models", f"duplicate model_id {model_id!r}")
        seen_models.add(model_id)
        bundled_model_ids.append(model_id)
        run_id = _identifier(bundle.get("run_id"), f"{model_id}.run_id")
        if run_id != declaration.run_id:
            _fail(model_id, "run_id differs from its ordered LLM declaration")
        panels = _object(bundle.get("panels"), f"{model_id}.panels")
        if set(panels) != {ALL_SOURCE_PANEL, READER_PANEL}:
            _fail(f"{model_id}.panels", "both canonical paper panels are required")
        panel_data: dict[str, tuple[dict[str, str], dict[str, Any], Path, list[dict[str, Any]]]] = {}
        for panel_id, gold, projection in (
            (ALL_SOURCE_PANEL, all_gold, "all_executions"),
            (READER_PANEL, reader_gold, "observed_execution_subset"),
        ):
            panel = _object(panels[panel_id], f"{model_id}.{panel_id}")
            if (
                panel.get("prediction_unit") != PREDICTION_UNIT
                or panel.get("substrate_id") != panel_id
            ):
                _fail(f"{model_id}.{panel_id}", "prediction unit or substrate differs")
            prediction, _ = _prediction_rows(
                panel.get("predictions"),
                base=bundle_path.parent,
                gold=gold,
                context=f"{model_id}.{panel_id}.predictions",
            )
            cost_descriptor, cost_path, cost_rows = _cost_rows(
                panel.get("cost"),
                base=bundle_path.parent,
                panel=gold,
                context=f"{model_id}.{panel_id}.cost",
            )
            panel_data[panel_id] = (prediction, cost_descriptor, cost_path, cost_rows)
        all_prediction, all_cost_desc, all_cost_path, all_cost_rows = panel_data[ALL_SOURCE_PANEL]
        reader_prediction, reader_cost_desc, reader_cost_path, reader_cost_rows = panel_data[READER_PANEL]
        # Reader eligibility is a statement-level property, but the reader
        # projection also removes non-reader evidence from eligible statements.
        # The ledger must therefore be an ordered execution subset, not merely
        # every all-source execution attached to a reader-eligible statement.
        reader_execution_ids = {
            row.get("execution_identity") for row in reader_cost_rows
        }
        exact_subset = [
            row
            for row in all_cost_rows
            if row.get("execution_identity") in reader_execution_ids
        ]
        if reader_cost_rows != exact_subset:
            _fail(
                f"{model_id}.{READER_PANEL}.cost",
                "reader ledger is not the exact observed all-source execution subset",
            )
        run_pair = (str(all_cost_desc["sha256"]), str(reader_cost_desc["sha256"]))
        if run_id in run_costs and run_costs[run_id] != run_pair:
            _fail(run_id, "aggregation arms from one run disagree on cost ledgers")
        run_costs[run_id] = run_pair
        all_comparability_id = _identifier(
            all_cost_desc.get("cost_comparability_id"),
            f"{model_id}.{ALL_SOURCE_PANEL}.cost.cost_comparability_id",
        )
        reader_comparability_id = _identifier(
            reader_cost_desc.get("cost_comparability_id"),
            f"{model_id}.{READER_PANEL}.cost.cost_comparability_id",
        )
        if (
            all_comparability_id != reader_comparability_id
            or all_cost_desc.get("pricing") != reader_cost_desc.get("pricing")
        ):
            _fail(model_id, "panel cost descriptors disagree on frozen pricing")
        if shared_cost_comparability_id is None:
            shared_cost_comparability_id = all_comparability_id
        elif shared_cost_comparability_id != all_comparability_id:
            _fail(
                "llm_models",
                "LLM arms do not share one cost-comparability basis",
            )
        implementation = _object(bundle.get("implementation"), f"{model_id}.implementation")
        impl_name = _text(
            implementation.get("implementation"), f"{model_id}.implementation.implementation"
        )
        _digest(
            implementation.get("implementation_digest"),
            f"{model_id}.implementation.implementation_digest",
        )
        training = implementation.get("training_data_sha256")
        if training is not None:
            training = _digest(training, f"{model_id}.implementation.training_data_sha256")
        environment = _canonical_text(
            implementation.get("environment"), f"{model_id}.implementation.environment"
        )
        notes = _object(implementation.get("notes"), f"{model_id}.implementation.notes")
        served_model = _text(
            notes.get("served_model"), f"{model_id}.implementation.notes.served_model"
        )
        provider_model_id = _text(
            notes.get("provider_model_id"),
            f"{model_id}.implementation.notes.provider_model_id",
        )
        if (
            served_model != declaration.served_model
            or provider_model_id != declaration.provider_model_id
        ):
            _fail(model_id, "runtime model identity differs from its ordered LLM declaration")
        raw_notes = _canonical_text(notes, f"{model_id}.implementation.notes")
        bundle_sha = sha256_file(bundle_path)
        for panel_id, gold, target, prediction, cost_desc, cost_path, projection in (
            (
                ALL_SOURCE_PANEL,
                all_gold,
                all_arms,
                all_prediction,
                all_cost_desc,
                all_cost_path,
                "all_executions",
            ),
            (
                READER_PANEL,
                reader_gold,
                reader_arms,
                reader_prediction,
                reader_cost_desc,
                reader_cost_path,
                "observed_execution_subset",
            ),
        ):
            cost = _metrics_cost(
                cost_desc,
                path=cost_path,
                panel=gold,
                run_id=run_id,
                projection=projection,
                output_parent=output_parent,
            )
            if cost["pricing"]["provider_model_id"] != declaration.provider_model_id:
                _fail(model_id, "priced provider model differs from its ordered LLM declaration")
            note = (
                f"bundle={_reference(bundle_path, output_parent)}@{bundle_sha}; {raw_notes}; "
                f"cost_projection={projection}; counterfactual_run_cost=false; "
                f"shared_run_id={run_id}; panel_totals_never_additive=true"
            )
            arm = _arm(
                arm_id=model_id,
                label=_llm_label(model_id),
                family="llm",
                prediction=prediction,
                source_manifest_path=bundle_path,
                implementation=impl_name,
                training_sha256=training,
                environment=environment,
                notes=note,
                threshold=threshold,
                cost=cost,
                output_parent=output_parent,
            )
            # Bundle digest, rather than implementation-source digest, is the
            # metrics-facing complete implementation identity.
            arm["implementation"]["implementation_digest"] = bundle_sha
            target.append(arm)
    return all_arms, reader_arms, tuple(bundled_model_ids)


def _excluded(
    arm_id: str,
    label: str,
    family: str,
    reason: str,
    required_artifact: str,
    provenance: str,
) -> dict[str, str]:
    return {
        "arm_id": _identifier(arm_id, "excluded arm_id"),
        "label": _text(label, "excluded label"),
        "family": family,
        "status": "excluded",
        "reason": _text(reason, "excluded reason"),
        "required_artifact": _text(required_artifact, "excluded required_artifact"),
        "provenance": _text(provenance, "excluded provenance"),
    }


def _missing_llm_exclusions(
    bundled_model_ids: Sequence[str],
    declarations: Sequence[LlmModelInput],
    protocol_path: Path,
    output_parent: Path,
) -> list[dict[str, str]]:
    seen = set(bundled_model_ids)
    expected = tuple(
        _identifier(declaration.model_id, f"llm_models[{index}].model_id")
        for index, declaration in enumerate(declarations)
    )
    if not expected or len(expected) != len(set(expected)):
        _fail("llm_models", "must be a nonempty ordered set")
    undeclared = seen - set(expected)
    if undeclared:
        _fail("llm_models", f"contains undeclared model IDs {sorted(undeclared)}")
    provenance = _binding_reference(protocol_path, output_parent)
    return [
        _excluded(
            model_id,
            f"{_llm_label(model_id)} — primary exact-panel run",
            "llm",
            "model result is not yet available as a canonical exact-panel bundle",
            "a canonical paired prediction/cost bundle from the completed primary run",
            provenance,
        )
        for model_id in expected
        if model_id not in seen
    ]


def _validate_arm_set(
    arms: Sequence[Mapping[str, Any]], excluded: Sequence[Mapping[str, Any]], panel_id: str
) -> None:
    ids = [str(arm.get("arm_id")) for arm in arms]
    if len(ids) != len(set(ids)):
        _fail(panel_id, "evaluated arm IDs are duplicated")
    excluded_ids = [str(arm.get("arm_id")) for arm in excluded]
    if len(excluded_ids) != len(set(excluded_ids)):
        _fail(panel_id, "excluded arm IDs are duplicated")
    overlap = set(ids) & set(excluded_ids)
    if overlap:
        _fail(panel_id, f"arms are both evaluated and excluded: {sorted(overlap)}")
    families = {arm.get("family") for arm in arms}
    if families != {"paper", "current", "llm"}:
        _fail(panel_id, f"required arm families differ: {sorted(str(v) for v in families)}")


def _panel_spec(
    *,
    gold: _PanelGold,
    strict_gold: _PanelGold,
    gold_manifest_path: Path,
    gold_manifest_sha: str,
    arms: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    output_parent: Path,
) -> dict[str, Any]:
    strict_ids = set(strict_gold.statement_ids)
    unresolved_ids = [
        statement_id
        for statement_id in gold.statement_ids
        if statement_id not in strict_ids
    ]
    released_positive = sum(label for label, _fold in gold.labels_and_folds.values())
    strict_positive = sum(
        label for label, _fold in strict_gold.labels_and_folds.values()
    )
    released_negative = len(gold.statement_ids) - released_positive
    strict_negative = len(strict_gold.statement_ids) - strict_positive
    return {
        "substrate_id": gold.panel_id,
        "lane": "paper",
        "label": gold.label,
        "contract": {
            "prediction_unit": PREDICTION_UNIT,
            "gold_rule": GOLD_RULE,
            "substrate_sha256": gold_manifest_sha,
            "gold_sha256": gold.sha256,
            "evaluation_set_sha256": gold.evaluation_sha256,
        },
        "analysis_scope": "primary",
        "released_label_audit": {
            "released_label_rule": (
                "positive if any reviewed evidence was tagged correct; otherwise the "
                "released binary target is negative"
            ),
            "strict_e0_rule": (
                "positive if any exact evidence pair is reviewed positive; negative only "
                "if every exact evidence pair is reviewed negative; unresolved otherwise"
            ),
            "released": {
                "statements": len(gold.statement_ids),
                "positive": released_positive,
                "negative": released_negative,
            },
            "strict_e0": {
                "resolved": len(strict_gold.statement_ids),
                "positive": strict_positive,
                "negative": strict_negative,
                "unresolved": len(unresolved_ids),
                "ordered_statement_id_sha256": strict_gold.evaluation_sha256,
            },
            "released_negative_assumption": {
                "statements": len(unresolved_ids),
                "share_of_released_negatives": len(unresolved_ids) / released_negative,
                "ordered_statement_id_sha256": ordered_statement_id_sha256(unresolved_ids),
            },
        },
        "substrate_manifest": {
            "path": _reference(gold_manifest_path.resolve(), output_parent),
            "sha256": gold_manifest_sha,
        },
        "gold": {"path": _reference(gold.path, output_parent), "sha256": gold.sha256},
        "strict_e0_resolved_gold": {
            "path": _reference(strict_gold.path, output_parent),
            "sha256": strict_gold.sha256,
        },
        "arms": [dict(value) for value in arms],
        "excluded_arms": [dict(value) for value in excluded],
    }


def assemble_spec(inputs: AssemblyInputs, output_path: Path) -> dict[str, Any]:
    """Validate all inputs and return the exact metrics-engine specification."""

    output_path = output_path.resolve()
    output_parent = output_path.parent
    frozen_at = _iso_timestamp(inputs.frozen_at)
    if isinstance(inputs.bootstrap_seed, bool) or inputs.bootstrap_seed < 0:
        _fail("bootstrap_seed", "expected a non-negative integer")
    if isinstance(inputs.bootstrap_resamples, bool) or inputs.bootstrap_resamples < 1:
        _fail("bootstrap_resamples", "expected a positive integer")
    (
        _gold_manifest,
        gold_manifest_sha,
        prediction_targets_sha,
        all_gold,
        reader_gold,
        all_strict_gold,
        reader_strict_gold,
    ) = _load_gold(inputs)
    threshold = _threshold(inputs, output_parent)
    production_bundle = load_bundle(inputs.production_hybrid_manifest.resolve())
    scorer_registry = _load_scorer_registry(
        inputs,
        production_bundle=production_bundle,
    )
    paper_all, paper_reader = _paper_arms(
        inputs,
        gold_manifest_sha=gold_manifest_sha,
        all_gold=all_gold,
        reader_gold=reader_gold,
        threshold=threshold,
        output_parent=output_parent,
    )
    current_all, current_reader, current_excluded_all, current_excluded_reader = _current_arms(
        inputs,
        registry=scorer_registry,
        gold_manifest_sha=gold_manifest_sha,
        all_gold=all_gold,
        reader_gold=reader_gold,
        threshold=threshold,
        output_parent=output_parent,
    )
    production_all, production_reader, production_excluded = _production_arms(
        inputs,
        bundle=production_bundle,
        prediction_targets_sha=prediction_targets_sha,
        all_gold=all_gold,
        reader_gold=reader_gold,
        threshold=threshold,
        output_parent=output_parent,
    )
    llm_all, llm_reader, bundled_model_ids = _llm_arms(
        inputs,
        all_gold=all_gold,
        reader_gold=reader_gold,
        threshold=threshold,
        output_parent=output_parent,
    )
    missing_llm = _missing_llm_exclusions(
        bundled_model_ids,
        inputs.llm_models,
        inputs.error_review_protocol,
        output_parent,
    )
    published_sha = sha256_file(inputs.published_metrics.resolve())
    reference_excluded_all = _excluded(
        "paper_published_summary_only_all_source",
        "Published paper methods without statement predictions — all source",
        "paper",
        "published fold summaries are reference values, not paired statement predictions",
        "complete statement-level predictions on the exact all-source panel",
        f"{_reference(inputs.published_metrics.resolve(), output_parent)}@{published_sha}",
    )
    reference_excluded_reader = _excluded(
        "paper_published_summary_only_readers",
        "Published paper methods without statement predictions — readers",
        "paper",
        "published fold summaries are reference values, not paired statement predictions",
        "complete statement-level predictions on the exact five-reader panel",
        f"{_reference(inputs.published_metrics.resolve(), output_parent)}@{published_sha}",
    )
    all_arms = [*paper_all, *current_all, production_all, *llm_all]
    reader_arms = [*paper_reader, *current_reader, production_reader, *llm_reader]
    all_excluded = [
        *current_excluded_all,
        *missing_llm,
        reference_excluded_all,
    ]
    reader_excluded = [
        *current_excluded_reader,
        *production_excluded,
        *missing_llm,
        reference_excluded_reader,
    ]
    _validate_arm_set(all_arms, all_excluded, ALL_SOURCE_PANEL)
    _validate_arm_set(reader_arms, reader_excluded, READER_PANEL)
    all_llm = {arm["arm_id"] for arm in llm_all}
    reader_llm = {arm["arm_id"] for arm in llm_reader}
    if all_llm != reader_llm:
        _fail("llm_models", "LLM model identity differs across panels")
    return {
        "artifact_kind": SPEC_KIND,
        "frozen_at": frozen_at,
        "bootstrap": {
            "seed": inputs.bootstrap_seed,
            "resamples": inputs.bootstrap_resamples,
            "ci_level": 0.95,
        },
        "scorer_registry": {
            "path": _reference(scorer_registry.path, output_parent),
            "bytes": scorer_registry.bytes,
            "sha256": scorer_registry.sha256,
        },
        "metrics": {
            "log_loss_epsilon": 1e-6,
            "calibration_bin_edges": [value / 10 for value in range(11)],
            "minimum_valid_bootstrap_fraction": 0.99,
            "pareto_metric": "fold_mean_trapezoidal_pr_auc",
        },
        "substrates": [
            _panel_spec(
                gold=all_gold,
                strict_gold=all_strict_gold,
                gold_manifest_path=inputs.gold_manifest,
                gold_manifest_sha=gold_manifest_sha,
                arms=all_arms,
                excluded=all_excluded,
                output_parent=output_parent,
            ),
            _panel_spec(
                gold=reader_gold,
                strict_gold=reader_strict_gold,
                gold_manifest_path=inputs.gold_manifest,
                gold_manifest_sha=gold_manifest_sha,
                arms=reader_arms,
                excluded=reader_excluded,
                output_parent=output_parent,
            ),
        ],
    }


def write_spec(spec: Mapping[str, Any], output_path: Path, *, force: bool = False) -> None:
    """Write a specification atomically."""

    output_path = output_path.resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
