"""Load the compact CoGEx fitted-Hybrid comparison evidence.

The bundle keeps exact predictions, the fitted artifact, and the small source
and acquisition records needed to interpret them.  It deliberately does not
retain or reconstruct the disposable Python runtime trees used for the replay.
The model is hashed but never unpickled here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .contracts import (
    ContractError,
    FileDescriptor,
    canonical_json_bytes,
    canonical_json_line,
    repository_root,
    stable_read,
    strict_json_loads,
)


ARTIFACT_KIND = "indra_cogex_hybrid_comparison_bundle"
ALL_SOURCE_KEY = "all_source"
READER_KEY = "readers"
READER_SOURCES = ("reach", "sparser", "medscan", "rlimsp", "trips")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _fail(message: str) -> None:
    raise ContractError(f"production Hybrid bundle: {message}")


def _map(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{context} must be an object")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{context} must be a nonempty string")
    return value


def _digest(value: Any, context: str) -> str:
    value = _text(value, context)
    if SHA256.fullmatch(value) is None:
        _fail(f"{context} must be a lowercase SHA-256")
    return value


def _descriptor(
    value: Any,
    *,
    owner: Path,
    root: Path,
    context: str,
    rows: int | None = None,
) -> FileDescriptor:
    descriptor = FileDescriptor.from_value(_map(value, context), owner=owner, root=root)
    if rows is not None and descriptor.rows != rows:
        _fail(f"{context}.rows must be {rows}")
    return descriptor


def _strings(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(f"{context} must be an array")
    result = tuple(_text(item, f"{context}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        _fail(f"{context} contains duplicates")
    return result


@dataclass(frozen=True)
class ProductionPanel:
    key: str
    arm_id: str
    label: str
    panel_id: str
    rows: int
    statement_ids_sha256: str
    raw_statement_ids_sha256: str
    input_sources: tuple[str, ...]
    predictions: FileDescriptor


@dataclass(frozen=True)
class ProductionBundle:
    arm_id: str
    label: str
    family: str
    implementation: str
    environment: str
    inputs: Mapping[str, FileDescriptor]
    model: FileDescriptor
    source_list: tuple[str, ...]
    acquisition: FileDescriptor
    static_audit: FileDescriptor
    sources: Mapping[str, FileDescriptor]
    panels: Mapping[str, ProductionPanel]
    package_versions: Mapping[str, str]
    wheel_identities: tuple[Mapping[str, Any], ...]
    limitations: Mapping[str, str]
    excluded_diagnostic_arm_id: str


def _panel(value: Any, *, key: str, owner: Path, root: Path) -> ProductionPanel:
    row = _map(value, f"panels.{key}")
    rows, panel_id = (
        (1_689, "all_source_1689")
        if key == ALL_SOURCE_KEY
        else (1_676, "readers_only_1676")
    )
    if row.get("rows") != rows or row.get("panel_id") != panel_id:
        _fail(f"panels.{key} does not identify the expected panel")
    sources = _strings(row.get("input_sources"), f"panels.{key}.input_sources")
    if key == READER_KEY and sources != READER_SOURCES:
        _fail("reader panel is not restricted to the five paper readers")
    all_sources = {
        "hprd", "rlimsp", "bel", "trips", "reach", "biopax",
        "sparser", "medscan", "isi", "trrust", "signor",
    }
    if key == ALL_SOURCE_KEY and set(sources) != all_sources:
        _fail("all-source panel source census differs")
    if row.get("route_census") != {
        "counts_only": rows,
        "counts_plus_simple": 0,
        "simple_only": 0,
    }:
        _fail(f"panels.{key} is not the recorded counts-only evaluation")
    if row.get("panel_specific_rescore") is not True:
        _fail(f"panels.{key}.panel_specific_rescore must be true")
    _text(row.get("execution_route"), f"panels.{key}.execution_route")
    expected_literal_route = key == ALL_SOURCE_KEY
    if row.get("literal_export_route_and_input_semantics") is not expected_literal_route:
        _fail(f"panels.{key} export-route claim differs")
    return ProductionPanel(
        key=key,
        arm_id=_text(row.get("arm_id"), f"panels.{key}.arm_id"),
        label=_text(row.get("label"), f"panels.{key}.label"),
        panel_id=panel_id,
        rows=rows,
        statement_ids_sha256=_digest(
            row.get("statement_ids_sha256"), f"panels.{key}.statement_ids_sha256"
        ),
        raw_statement_ids_sha256=_digest(
            row.get("raw_statement_ids_sha256"),
            f"panels.{key}.raw_statement_ids_sha256",
        ),
        input_sources=sources,
        predictions=_descriptor(
            row.get("predictions"), owner=owner, root=root,
            context=f"panels.{key}.predictions", rows=rows,
        ),
    )


def validate_panel_predictions(panel: ProductionPanel) -> tuple[Mapping[str, Any], ...]:
    """Verify one exact prediction ledger and its ordered statement identity."""

    capture = panel.predictions.capture(context=f"{panel.key} predictions")
    if not capture.payload or not capture.payload.endswith(b"\n"):
        _fail(f"{panel.key} predictions must end with LF")
    result: list[Mapping[str, Any]] = []
    statement_ids: list[str] = []
    for line_number, raw in enumerate(capture.payload.splitlines(), start=1):
        row = strict_json_loads(raw, context=f"{panel.key} prediction {line_number}")
        if not isinstance(row, dict) or set(row) != {"statement_id", "probability_correct"}:
            _fail(f"{panel.key} prediction {line_number} has an unexpected schema")
        statement_id = _text(row["statement_id"], f"{panel.key} statement_id")
        probability = row["probability_correct"]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0 <= float(probability) <= 1
        ):
            _fail(f"{panel.key} probability is outside [0, 1]")
        statement_ids.append(statement_id)
        result.append(row)
    if len(result) != panel.rows or len(set(statement_ids)) != panel.rows:
        _fail(f"{panel.key} does not contain {panel.rows} unique predictions")
    canonical_ids = b"".join(
        canonical_json_bytes({"statement_id": statement_id}) + b"\n"
        for statement_id in statement_ids
    )
    raw_ids = b"".join(value.encode() + b"\n" for value in statement_ids)
    if hashlib.sha256(canonical_ids).hexdigest() != panel.statement_ids_sha256:
        _fail(f"{panel.key} ordered statement identity differs")
    if hashlib.sha256(raw_ids).hexdigest() != panel.raw_statement_ids_sha256:
        _fail(f"{panel.key} raw ordered statement identity differs")
    return tuple(result)


def _json_file(descriptor: FileDescriptor, context: str) -> Mapping[str, Any]:
    value = strict_json_loads(descriptor.capture(context=context).payload, context=context)
    return _map(value, context)


def load_bundle(path: Path, *, validate_files: bool = True) -> ProductionBundle:
    """Load the sole canonical fitted-Hybrid bundle."""

    path = path.resolve()
    capture = stable_read(path, context="production Hybrid manifest")
    manifest = _map(
        strict_json_loads(capture.payload, context="production Hybrid manifest"),
        "manifest",
    )
    if capture.payload != canonical_json_line(manifest):
        _fail("manifest is not canonical JSON with one terminal LF")
    if manifest.get("artifact_kind") != ARTIFACT_KIND or manifest.get("status") != "complete":
        _fail("manifest is not a complete canonical bundle")
    root = repository_root(path)

    arm = _map(manifest.get("arm"), "arm")
    if arm.get("family") != "current" or arm.get("analysis_role") != "descriptive_nonconfirmatory":
        _fail("arm must remain descriptive and non-confirmatory")

    input_values = _map(manifest.get("inputs"), "inputs")
    inputs = {
        "prediction_targets": _descriptor(
            input_values.get("prediction_targets"), owner=path, root=root,
            context="inputs.prediction_targets", rows=1_689,
        ),
        "reader_gold": _descriptor(
            input_values.get("reader_gold"), owner=path, root=root,
            context="inputs.reader_gold", rows=1_676,
        ),
    }

    model_value = _map(manifest.get("model"), "model")
    expected_classes = (
        model_value.get("artifact_role") == "fitted_counts_component"
        and model_value.get("evaluated_class") == "indra.belief.skl.HybridScorer"
        and model_value.get("fitted_component_class") == "indra.belief.skl.CountsScorer"
        and model_value.get("fallback_class") == "indra.belief.SimpleScorer"
    )
    classifier = _map(model_value.get("classifier"), "model.classifier")
    if not expected_classes or (
        classifier.get("class") != "sklearn.ensemble._forest.RandomForestClassifier"
        or classifier.get("n_estimators") != 2_000
        or classifier.get("n_features") != 87
        or classifier.get("max_depth") != 13
    ):
        _fail("model/scorer identity differs")
    source_list = _strings(model_value.get("source_list"), "model.source_list")
    if len(source_list) != 17:
        _fail("model source list must contain 17 fitted sources")
    for name in ("classifier_state_sha256", "tree_topology_sha256", "fitted_contract_sha256"):
        _digest(classifier.get(name), f"model.classifier.{name}")
    model = _descriptor(model_value.get("artifact"), owner=path, root=root, context="model.artifact")
    audit = _descriptor(
        model_value.get("static_audit"), owner=path, root=root, context="model.static_audit"
    )

    panel_values = _map(manifest.get("panels"), "panels")
    panels = {
        key: _panel(panel_values.get(key), key=key, owner=path, root=root)
        for key in (ALL_SOURCE_KEY, READER_KEY)
    }

    implementation = _map(manifest.get("implementation"), "implementation")
    _digest(implementation.get("combined_replay_sha256"), "implementation digest")
    export = _map(implementation.get("export_assembly"), "export_assembly")
    scorer = _map(implementation.get("scorer"), "scorer")
    sources = {
        "export_assembly": _descriptor(export.get("source"), owner=path, root=root, context="export source"),
        "export_authority": _descriptor(export.get("authority"), owner=path, root=root, context="export authority"),
        "license": _descriptor(export.get("license"), owner=path, root=root, context="export license"),
        "adapter": _descriptor(scorer.get("adapter"), owner=path, root=root, context="scoring adapter"),
        "indra_belief": _descriptor(
            scorer.get("indra_belief_module"), owner=path, root=root, context="INDRA belief module"
        ),
        "indra_belief_skl": _descriptor(
            scorer.get("indra_belief_skl_module"), owner=path, root=root,
            context="INDRA sklearn belief module",
        ),
        "default_prior": _descriptor(
            scorer.get("default_prior"), owner=path, root=root, context="INDRA default prior"
        ),
    }

    runtime = _map(manifest.get("runtime"), "runtime")
    packages = _map(runtime.get("package_versions"), "runtime.package_versions")
    package_versions = {str(key): _text(value, f"package {key}") for key, value in packages.items()}
    if (
        runtime.get("python") != "3.12.10"
        or package_versions.get("indra") != "1.24.0"
        or package_versions.get("scikit_learn") != "1.4.1.post1"
    ):
        _fail("critical replay package identity differs")
    serialization = _map(runtime.get("serialization"), "runtime.serialization")
    if serialization != {
        "embedded_sklearn": "1.3.2",
        "learned_state_mutated": False,
        "native_control_exact_prediction_parity": True,
        "replay_sklearn": "1.4.1.post1",
        "unmodified_replay_sklearn_accepted": False,
    }:
        _fail("sklearn replay semantics differ")
    wheels = runtime.get("wheel_identities")
    if not isinstance(wheels, list) or len(wheels) != 19:
        _fail("19 recorded wheel identities are required")
    filenames: set[str] = set()
    for index, wheel_value in enumerate(wheels):
        wheel = _map(wheel_value, f"wheel {index}")
        filename = _text(wheel.get("filename"), f"wheel {index} filename")
        _digest(wheel.get("sha256"), f"wheel {index} digest")
        if filename in filenames or wheel.get("retained") is not False:
            _fail("wheel identities must be unique declarations, not retained binaries")
        filenames.add(filename)

    provenance = _map(manifest.get("provenance"), "provenance")
    acquisition = _descriptor(
        provenance.get("acquisition"), owner=path, root=root, context="acquisition"
    )
    expected_claims = {
        "predictions_are_exact_evidence": True,
        "literal_live_deployment_output": False,
        "live_deployment_parity_established": False,
        "historical_runtime_established": False,
        "private_training_provenance_available": False,
        "training_overlap_status": "unknown",
    }
    if any(provenance.get(key) != value for key, value in expected_claims.items()):
        _fail("provenance overclaims deployment or training knowledge")
    limitations = _map(manifest.get("limitations"), "limitations")
    expected_limitations = {
        "deployment_parity", "training_provenance", "training_overlap",
        "reader_counterfactual", "runtime_replay", "simple_fallback",
    }
    if set(limitations) != expected_limitations:
        _fail("limitations are incomplete")
    for key, value in limitations.items():
        _text(value, f"limitations.{key}")
    excluded = _map(manifest.get("excluded_diagnostics"), "excluded_diagnostics")

    if validate_files:
        for name, descriptor in inputs.items():
            descriptor.capture(context=f"input {name}")
        model.capture(context="fitted model")
        acquisition_value = _json_file(acquisition, "acquisition evidence")
        audit_value = _json_file(audit, "model static audit")
        if (
            _map(acquisition_value.get("artifact"), "acquired artifact").get("sha256") != model.sha256
            or _map(acquisition_value.get("artifact"), "acquired artifact").get("bytes") != model.bytes
            or audit_value.get("status") != "passed"
            or _map(audit_value.get("model"), "audited model").get("sha256") != model.sha256
            or audit_value.get("dangerous_globals") != []
        ):
            _fail("model acquisition/static audit does not bind the fitted artifact")
        for name, descriptor in sources.items():
            descriptor.capture(context=f"retained source {name}")
        authority = _json_file(sources["export_authority"], "export source authority")
        if (
            authority.get("sha256") != sources["export_assembly"].sha256
            or authority.get("commit") != export.get("commit")
            or authority.get("git_blob") != export.get("git_blob")
        ):
            _fail("export source authority differs")
        for panel in panels.values():
            validate_panel_predictions(panel)

    return ProductionBundle(
        arm_id=_text(arm.get("arm_id"), "arm.arm_id"),
        label=_text(arm.get("label"), "arm.label"),
        family="current",
        implementation=_text(arm.get("implementation"), "arm.implementation"),
        environment=_text(arm.get("environment"), "arm.environment"),
        inputs=inputs,
        model=model,
        source_list=source_list,
        acquisition=acquisition,
        static_audit=audit,
        sources=sources,
        panels=panels,
        package_versions=package_versions,
        wheel_identities=tuple(wheels),
        limitations=limitations,
        excluded_diagnostic_arm_id=_text(excluded.get("arm_id"), "excluded arm_id"),
    )
