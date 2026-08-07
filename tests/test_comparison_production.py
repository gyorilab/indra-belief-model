from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from indra_belief.comparison.contracts import (
    ContractError,
    FileDescriptor,
    canonical_json_line,
)
from indra_belief.comparison.production import (
    ALL_SOURCE_KEY,
    READER_KEY,
    load_bundle,
    validate_panel_predictions,
)



# Reads the gitignored local artifact trees; skipped only when they are WHOLLY
# absent (CI, a fresh checkout). A PARTIAL tree is a failure in
# tests/test_local_artifacts.py, never a skip here.
import _local_artifacts as _artifacts

pytestmark = _artifacts.requires()

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/comparison/models/indra_cogex_hybrid/manifest.json"


def test_real_bundle_preserves_scientific_evidence_without_runtime_trees() -> None:
    bundle = load_bundle(MANIFEST)

    assert MANIFEST.stat().st_size < 13_000
    assert bundle.model.bytes == 93_290_923
    assert bundle.model.sha256 == (
        "5c1cbf810420e617e0dce765361503827f3c42513ae48bfbbcaf82e740e99746"
    )
    assert bundle.source_list == (
        "trips",
        "trrust",
        "biogrid",
        "ctd",
        "hprd",
        "eidos",
        "virhostnet",
        "reach",
        "bel",
        "rlimsp",
        "biopax",
        "tas",
        "isi",
        "sparser",
        "phosphoelm",
        "medscan",
        "signor",
    )
    assert not hasattr(bundle, "manifest_path")
    assert not hasattr(bundle, "manifest_sha256")
    assert not hasattr(bundle, "analysis_role")
    assert bundle.panels[ALL_SOURCE_KEY].rows == 1_689
    assert bundle.panels[READER_KEY].rows == 1_676
    assert not hasattr(bundle.panels[READER_KEY], "execution_route")
    assert not hasattr(bundle.panels[READER_KEY], "panel_specific_rescore")
    assert bundle.panels[READER_KEY].input_sources == (
        "reach",
        "sparser",
        "medscan",
        "rlimsp",
        "trips",
    )
    assert bundle.package_versions["indra"] == "1.24.0"
    assert bundle.package_versions["scikit_learn"] == "1.4.1.post1"
    assert len(bundle.wheel_identities) == 19
    assert all(wheel["retained"] is False for wheel in bundle.wheel_identities)

    retained = [bundle.model, bundle.acquisition, bundle.static_audit, *bundle.sources.values()]
    canonical_directory = MANIFEST.parent.resolve()
    assert all(descriptor.path.is_relative_to(canonical_directory) for descriptor in retained)
    assert all("runtimes" not in str(descriptor.path) for descriptor in retained)
    assert all("wheels" not in str(descriptor.path) for descriptor in retained)

    value = json.loads(MANIFEST.read_bytes())
    assert "schema_version" not in value
    assert "protocol" not in value
    assert "release" not in value
    assert "attestation" not in value


def test_reader_predictions_are_a_true_rescore_not_an_all_source_row_subset() -> None:
    bundle = load_bundle(MANIFEST, validate_files=False)
    all_rows = {
        row["statement_id"]: row["probability_correct"]
        for row in validate_panel_predictions(bundle.panels[ALL_SOURCE_KEY])
    }
    reader_rows = validate_panel_predictions(bundle.panels[READER_KEY])

    assert len(reader_rows) == 1_676
    assert sum(
        all_rows[row["statement_id"]] != row["probability_correct"]
        for row in reader_rows
    ) == 523


def test_rejects_removing_the_true_reader_rescore_contract(tmp_path: Path) -> None:
    value = json.loads(MANIFEST.read_bytes())
    value["panels"]["readers"]["panel_specific_rescore"] = False
    tampered = tmp_path / "manifest.json"
    tampered.write_bytes(canonical_json_line(value))

    with pytest.raises(ContractError, match="panel_specific_rescore"):
        load_bundle(tampered, validate_files=False)


def test_prediction_descriptor_detects_changed_ledger(tmp_path: Path) -> None:
    bundle = load_bundle(MANIFEST, validate_files=False)
    panel = bundle.panels[READER_KEY]
    changed = tmp_path / "reader_predictions.jsonl"
    payload = panel.predictions.path.read_bytes()
    changed.write_bytes(payload.replace(b"0.727772744875959", b"0.727772744875958", 1))
    descriptor = FileDescriptor(
        path=changed,
        sha256=panel.predictions.sha256,
        bytes=panel.predictions.bytes,
        rows=panel.predictions.rows,
        declared_path=str(changed),
    )

    with pytest.raises(ContractError, match="SHA-256 differs"):
        validate_panel_predictions(replace(panel, predictions=descriptor))


def test_limitations_do_not_overclaim_private_or_deployment_provenance() -> None:
    bundle = load_bundle(MANIFEST, validate_files=False)

    assert "not established" in bundle.limitations["deployment_parity"]
    assert "unavailable" in bundle.limitations["training_provenance"]
    assert bundle.limitations["training_overlap"].endswith("unknown.")
    assert "counterfactual" in bundle.limitations["reader_counterfactual"]
    assert "do not exercise" in bundle.limitations["simple_fallback"]
