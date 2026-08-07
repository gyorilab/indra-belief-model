from __future__ import annotations

import hashlib
import json
from pathlib import Path

import indra.belief
import indra.belief.skl



# Reads the gitignored local artifact trees; skipped only when they are WHOLLY
# absent (CI, a fresh checkout). A PARTIAL tree is a failure in
# tests/test_local_artifacts.py, never a skip here.
import _local_artifacts as _artifacts

pytestmark = _artifacts.requires()

ROOT = Path(__file__).resolve().parents[1]
PAPER_MANIFEST = ROOT / "data/benchmark/indra_paper_2023.manifest.json"
SCORER_REGISTRY = ROOT / "data/comparison/scorers.json"


def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_paper_manifest_freezes_the_actual_table_ev2_substrates() -> None:
    manifest = _load(PAPER_MANIFEST)
    protocol = manifest["paper_evaluation_protocol"]

    assert manifest["paper_code"]["commit"] == (
        "63abdf1274d2f5534ed822585775031712916c83"
    )
    assert protocol["extended_dataset"] == {
        "source": "data/curation/extended_curation_dataset.pkl",
        "rows": 1689,
        "positive": 1237,
        "negative": 452,
        "eligibility": "exactly two real agents after group_curations.py filtering",
        "used_by": "Table EV2 all-source arms",
    }
    assert protocol["reader_only_dataset"]["rows"] == 1676
    assert protocol["reader_only_dataset"]["positive"] == 1236
    assert protocol["reader_only_dataset"]["negative"] == 440
    assert protocol["reader_only_dataset"]["not_the_multireader_pickle"] is True
    assert protocol["metric"]["not_pooled_average_precision"] is True
    assert protocol["metric"]["not_a_confidence_interval"] is True


def test_current_scorer_registry_is_exhaustive_and_separates_defaults() -> None:
    registry = _load(SCORER_REGISTRY)
    scorers = registry["scorers"]
    by_id = {row["scorer_id"]: row for row in scorers}

    assert len(by_id) == len(scorers)
    assert registry["taxonomy"]["ast_exhaustive_class_count"] == 6
    assert set(registry["taxonomy"]["classes"]) == {
        "BeliefScorer",
        "SimpleScorer",
        "BayesianScorer",
        "SklearnScorer",
        "CountsScorer",
        "HybridScorer",
    }

    simple = by_id["indra_1.24.0_simple_default"]
    deployed = by_id["indra_db_7dc8bf5_cogex_hybrid_production"]
    assert "library_default" in simple["official_status"]
    assert "recovered_storage_artifact" in deployed["official_status"]
    assert simple["class"] != deployed["class"]
    assert "refinement_ancestor_source_counts" in deployed["required_inputs"]
    assert deployed["artifact_access"] == "recovered_authenticated_download"
    assert deployed["artifact_sha256"] == (
        "5c1cbf810420e617e0dce765361503827f3c42513ae48bfbbcaf82e740e99746"
    )
    assert deployed["artifact_bytes"] == 93_290_923
    assert deployed["fitted_state"]["n_estimators"] == 2_000
    assert deployed["fitted_state"]["max_depth"] == 13
    assert deployed["fitted_state"]["n_features"] == 87
    assert len(deployed["fitted_state"]["source_list"]) == 17
    assert deployed["comparison_status"]["analysis_role"] == (
        "descriptive_nonconfirmatory"
    )
    assert deployed["comparison_status"]["live_deployment_parity_established"] is False
    assert deployed["comparison_status"]["simple_fallback_exercised"] is False
    assert "registry_schema_version" not in registry


def test_installed_current_release_matches_frozen_scorer_code() -> None:
    registry = _load(SCORER_REGISTRY)
    installed = registry["installed_runtime"]

    assert _sha256(Path(indra.belief.__file__)) == installed["belief_init_sha256"]
    assert _sha256(Path(indra.belief.skl.__file__)) == installed["belief_skl_sha256"]
