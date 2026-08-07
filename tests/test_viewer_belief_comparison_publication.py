"""The focused publication gate must reject the current diagnostic artifact."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest



# Reads the gitignored local artifact trees; skipped only when they are WHOLLY
# absent (CI, a fresh checkout). A PARTIAL tree is a failure in
# tests/test_local_artifacts.py, never a skip here.
import _local_artifacts as _artifacts

pytestmark = _artifacts.requires()

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "viewer/scripts/validate-belief-comparison-publication.mjs"
CURRENT = ROOT / "data/results/indra_belief_comparison_metrics.json"
PROTOCOL_SHA256 = "910b660d626202668f72c941f277ebee95fb8794e83208082995c58a4fe1987a"
HUMAN_ATTESTATION = (
    "I attest that I personally reviewed every assigned case without "
    "model-generated adjudication and that this ledger accurately records my decisions."
)
COMPARISON_FILES = (
    "aggregation_config",
    "spec",
    "bundle_manifest",
    "protocol",
    "gold",
    "predictions",
    "execution_ledger",
    "statements",
    "execution_map",
    "raw_attempts",
    "pricing_config",
    "spend_ledger",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _descriptor(label: str, *, sha256: str | None = None) -> dict[str, object]:
    return {
        "sha256": sha256 or _digest(label),
        "bytes": len(label) + 1,
    }


def _summary(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": count,
        "denominator": denominator,
        "proportion": None if denominator == 0 else count / denominator,
    }


def _review(panel_id: str) -> dict[str, object]:
    packet_id = f"packet_{_digest(f'{panel_id}:packet-id')}"
    packet = _descriptor(f"{panel_id}:packet")
    error_types = {
        "false_positive": {
            **_summary(1, 3),
            "defensible": _summary(1, 1),
            "non_defensible": _summary(0, 1),
        },
        "false_negative": {
            **_summary(2, 3),
            "defensible": _summary(1, 2),
            "non_defensible": _summary(1, 2),
        },
    }
    comparison_files = {
        name: _descriptor(f"{panel_id}:input:{name}") for name in COMPARISON_FILES
    }
    comparison_files["protocol"] = _descriptor(
        f"{panel_id}:input:protocol", sha256=PROTOCOL_SHA256
    )
    return {
        "artifact_kind": "indra_belief_error_review_report",
        "status": "complete",
        "panel_id": panel_id,
        "arm_id": "llm_gemma_4_e2b",
        "model_id": "llm_gemma_4_e2b",
        "packet_id": packet_id,
        "evaluated_statements": 10,
        "threshold_errors": _summary(3, 10),
        "error_types": error_types,
        "human_classifications": {
            "supports_claim": _summary(2, 3),
            "rejects_claim": _summary(0, 3),
            "indeterminate": _summary(1, 3),
        },
        "review": {
            "reviewer_pseudonyms": ["reviewer.alpha", "reviewer.beta"],
            "resolver_pseudonym": "resolver.gamma",
            "exact_agreement": _summary(2, 3),
            "disagreement_count": 1,
            "resolved_by_resolver_count": 1,
            "classification_reliability": {
                "classes": ["supports_claim", "rejects_claim", "indeterminate"],
                "confusion_matrix_a_by_b": {
                    "supports_claim": {
                        "supports_claim": 2,
                        "rejects_claim": 1,
                        "indeterminate": 0,
                    },
                    "rejects_claim": {
                        "supports_claim": 0,
                        "rejects_claim": 0,
                        "indeterminate": 0,
                    },
                    "indeterminate": {
                        "supports_claim": 0,
                        "rejects_claim": 0,
                        "indeterminate": 0,
                    },
                },
                "observed_agreement": 2 / 3,
                "expected_chance_agreement": 2 / 3,
                "cohen_kappa": 0.0,
            },
            "human_attestation": HUMAN_ATTESTATION,
        },
        "defensibility": {
            "denominator": "all_threshold_errors",
            "defensible": _summary(2, 3),
            "non_defensible": _summary(1, 3),
            "system_supported_defensible": _summary(1, 3),
            "indeterminate_ambiguity_defensible": _summary(1, 3),
            "unresolved": _summary(0, 3),
        },
        "dimensions": {
            "multiple_dimensions_per_case": True,
            "denominator": "all_threshold_errors",
            "rows": [
                {
                    "dimension": "explicit_support",
                    **_summary(3, 3),
                    "by_judgment": {"defensible": 2, "non_defensible": 1},
                    "by_error_type": {"false_positive": 1, "false_negative": 2},
                }
            ],
        },
        "taxonomy_refinements": [],
        "adjudications": [
            {
                "case_id": f"case_{_digest(f'{panel_id}:case:1')}",
                "error_type": "false_positive",
                "human_classification": "supports_claim",
                "judgment": "defensible",
                "defensibility_basis": "human_matches_system",
                "dimensions": ["explicit_support"],
                "comment": None,
                "decision_source": "resolver",
            },
            {
                "case_id": f"case_{_digest(f'{panel_id}:case:2')}",
                "error_type": "false_negative",
                "human_classification": "supports_claim",
                "judgment": "non_defensible",
                "defensibility_basis": "human_matches_reference",
                "dimensions": ["explicit_support"],
                "comment": None,
                "decision_source": "reviewer_agreement",
            },
            {
                "case_id": f"case_{_digest(f'{panel_id}:case:3')}",
                "error_type": "false_negative",
                "human_classification": "indeterminate",
                "judgment": "defensible",
                "defensibility_basis": "indeterminate_ambiguity",
                "dimensions": ["explicit_support"],
                "comment": "The displayed evidence is ambiguous.",
                "decision_source": "reviewer_agreement",
            },
        ],
        "provenance": {
            "protocol": _descriptor(
                f"{panel_id}:protocol", sha256=PROTOCOL_SHA256
            ),
            "codebook": _descriptor(f"{panel_id}:codebook"),
            "packet": packet,
            "admin_manifest": _descriptor(f"{panel_id}:admin"),
            "reviewer_ledgers": [
                _descriptor(f"{panel_id}:ledger:a"),
                _descriptor(f"{panel_id}:ledger:b"),
            ],
            "reviewer_workbooks": [
                _descriptor(f"{panel_id}:workbook:a"),
                _descriptor(f"{panel_id}:workbook:b"),
            ],
            "reviewer_assignments": [
                {
                    "reviewer_slot": "A",
                    "assignment_id": f"assignment_{_digest(f'{panel_id}:assignment:a')}",
                    "workbook_content_sha256": _digest(
                        f"{panel_id}:workbook-content:a"
                    ),
                },
                {
                    "reviewer_slot": "B",
                    "assignment_id": f"assignment_{_digest(f'{panel_id}:assignment:b')}",
                    "workbook_content_sha256": _digest(
                        f"{panel_id}:workbook-content:b"
                    ),
                },
            ],
            "reviewer_workbook_packets": [packet],
            "comparison_inputs": {
                "panel_id": panel_id,
                "arm_id": "llm_gemma_4_e2b",
                "model_id": "llm_gemma_4_e2b",
                "run_id": "gemma_4_e2b_primary",
                "threshold": 0.5,
                "files": comparison_files,
                "ordered_gold_statement_id_sha256": _digest(
                    f"{panel_id}:gold-order"
                ),
                "selected_execution_projection_sha256": _digest(
                    f"{panel_id}:execution-order"
                ),
                "selected_execution_count": 20,
            },
            "resolver_workload": _descriptor(f"{panel_id}:resolver-workload"),
            "resolver_workbook": _descriptor(f"{panel_id}:resolver-workbook"),
            "resolver_ledger": _descriptor(f"{panel_id}:resolver-ledger"),
        },
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _replace_exact(value: object, old: str, new: str) -> object:
    if isinstance(value, dict):
        return {key: _replace_exact(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_exact(item, old, new) for item in value]
    return new if value == old else value


def _without_disagreements(value: dict[str, object]) -> dict[str, object]:
    review = value["review"]
    adjudications = value["adjudications"]
    provenance = value["provenance"]
    assert isinstance(review, dict)
    assert isinstance(adjudications, list)
    assert isinstance(provenance, dict)
    adjudications[0]["decision_source"] = "reviewer_agreement"
    review["resolver_pseudonym"] = None
    review["exact_agreement"] = _summary(3, 3)
    review["disagreement_count"] = 0
    review["resolved_by_resolver_count"] = 0
    provenance["resolver_workload"] = None
    provenance["resolver_workbook"] = None
    provenance["resolver_ledger"] = None
    return value


def _run(*review_paths: Path) -> subprocess.CompletedProcess[str]:
    return _run_artifact(CURRENT, *review_paths)


def _run_artifact(
    artifact: Path, *review_paths: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            str(VALIDATOR),
            str(artifact),
            *(str(path) for path in review_paths),
        ],
        capture_output=True,
        text=True,
    )


def _run_reviews(
    all_source: Path, readers: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            str(VALIDATOR),
            "--reviews-only",
            str(all_source),
            str(readers),
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_current_diagnostic_fails_publication_gate() -> None:
    result = _run()
    assert result.returncode == 1
    assert result.stderr.strip()
    assert "absolute local filesystem paths" not in result.stderr
    assert "paper arm identities differ from the canonical set" not in result.stderr
    assert "current INDRA arm identities differ from the canonical set" not in result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
@pytest.mark.parametrize(
    "local_path",
    [
        "/Users/alice/project/metrics.json",
        "bundle=/home/alice/project/model/manifest.json@deadbeef",
        "source→/opt/project/model/manifest.json",
        r"C:\Users\alice\project\metrics.json",
        r"\\workstation\share\metrics.json",
        "file:///Users/alice/project/metrics.json",
    ],
)
def test_publication_gate_rejects_absolute_local_paths_anywhere(
    tmp_path: Path, local_path: str
) -> None:
    artifact = json.loads(CURRENT.read_text(encoding="utf-8"))
    artifact["provenance"]["source_manifest_path"] = local_path
    path = tmp_path / "absolute-path.json"
    _write(path, artifact)

    result = _run_artifact(path)

    assert result.returncode == 1
    assert (
        "public metrics artifact contains absolute local filesystem paths at "
        "artifact.provenance.source_manifest_path"
    ) in result.stderr
    assert local_path not in result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/public/metrics.json",
        "https://example.org/download?source=/Users/example/public-fixture",
        "s3://public-bucket/comparison/metrics.json",
    ],
)
def test_publication_gate_allows_urls(tmp_path: Path, url: str) -> None:
    artifact = json.loads(CURRENT.read_text(encoding="utf-8"))
    artifact["provenance"]["source_manifest_path"] = url
    path = tmp_path / "url.json"
    _write(path, artifact)

    result = _run_artifact(path)

    assert result.returncode == 1
    assert "absolute local filesystem paths" not in result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
@pytest.mark.parametrize(
    ("panel_id", "arm_id"),
    [
        (
            "paper_all_source",
            "indra_1.24.0_simple_default_direct",
        ),
        (
            "paper_readers",
            "indra_1.24.0_simple_direct_readers_only",
        ),
    ],
)
def test_publication_gate_requires_exact_canonical_paper_and_current_arm_sets(
    tmp_path: Path, panel_id: str, arm_id: str
) -> None:
    artifact = json.loads(CURRENT.read_text(encoding="utf-8"))
    panel = next(
        value for value in artifact["substrates"] if value["substrate_id"] == panel_id
    )
    primary = next(value for value in panel["arms"] if value["arm_id"] == arm_id)
    strict = next(
        value
        for value in panel["strict_e0_resolved_sensitivity"]["arms"]
        if value["arm_id"] == arm_id
    )
    primary["family"] = "paper"
    strict["family"] = "paper"
    path = tmp_path / f"{panel_id}-wrong-arm-family.json"
    _write(path, artifact)

    result = _run_artifact(path)

    assert result.returncode == 1
    assert (
        f"{panel_id}: paper arm identities differ from the canonical set"
        in result.stderr
    )
    assert (
        f"{panel_id}: current INDRA arm identities differ from the canonical set"
        in result.stderr
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
@pytest.mark.parametrize(
    ("panel_id", "arm_id"),
    [
        (
            "paper_all_source",
            "rf_2k_d13_type_pmids_promoter_all_sources_specific",
        ),
        ("paper_readers", "orig_belief_readers"),
    ],
)
def test_publication_gate_rejects_self_consistent_paper_arm_id_drift(
    tmp_path: Path, panel_id: str, arm_id: str
) -> None:
    artifact = json.loads(CURRENT.read_text(encoding="utf-8"))
    artifact = _replace_exact(artifact, arm_id, f"{arm_id}_renamed")
    path = tmp_path / f"{panel_id}-renamed-paper-arm.json"
    _write(path, artifact)

    result = _run_artifact(path)

    assert result.returncode == 1
    assert (
        f"{panel_id}: paper arm identities differ from the canonical set"
        in result.stderr
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_canonical_completed_review_schema_adds_no_review_blockers(
    tmp_path: Path,
) -> None:
    all_source = tmp_path / "all-source.json"
    readers = tmp_path / "readers.json"
    _write(all_source, _review("paper_all_source"))
    _write(readers, _without_disagreements(_review("paper_readers")))

    result = _run_reviews(all_source, readers)

    assert result.returncode == 0, result.stderr
    assert "publication-ready blinded error-review reports" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_full_gate_binds_reviews_to_exact_metrics_inputs(tmp_path: Path) -> None:
    all_source = tmp_path / "all-source.json"
    readers = tmp_path / "readers.json"
    _write(all_source, _review("paper_all_source"))
    _write(readers, _without_disagreements(_review("paper_readers")))

    result = _run(all_source, readers)

    assert result.returncode == 1
    assert "comparison spec digest differs from the metrics artifact" in result.stderr
    assert "bundle digest differs from the evaluated arm" in result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_legacy_singular_workbook_commitment_is_rejected(tmp_path: Path) -> None:
    all_source_value = _review("paper_all_source")
    all_source_value["provenance"]["reviewer_workbook_content_sha256"] = _digest(
        "legacy-singular"
    )
    all_source = tmp_path / "all-source.json"
    readers = tmp_path / "readers.json"
    _write(all_source, all_source_value)
    _write(readers, _review("paper_readers"))

    result = _run_reviews(all_source, readers)

    assert result.returncode == 1
    assert "unexpected reviewer_workbook_content_sha256" in result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_human_classifications_and_exact_defensibility_schema_are_required(
    tmp_path: Path,
) -> None:
    all_source_value = _review("paper_all_source")
    del all_source_value["human_classifications"]
    all_source_value["defensibility"]["by_error_type"] = {}
    all_source = tmp_path / "all-source.json"
    readers = tmp_path / "readers.json"
    _write(all_source, all_source_value)
    _write(readers, _review("paper_readers"))

    result = _run_reviews(all_source, readers)

    assert result.returncode == 1
    assert "missing human_classifications" in result.stderr

    all_source_value = _review("paper_all_source")
    all_source_value["defensibility"]["by_error_type"] = {}
    _write(all_source, all_source_value)
    result = _run_reviews(all_source, readers)
    assert "unexpected by_error_type" in result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_defensibility_basis_and_ambiguity_split_are_reconciled(
    tmp_path: Path,
) -> None:
    all_source_value = _review("paper_all_source")
    all_source_value["adjudications"][2][
        "defensibility_basis"
    ] = "human_matches_system"
    all_source = tmp_path / "all-source.json"
    readers = tmp_path / "readers.json"
    _write(all_source, all_source_value)
    _write(readers, _review("paper_readers"))

    result = _run_reviews(all_source, readers)

    assert result.returncode == 1
    assert "defensibility_basis is inconsistent with the exact derivation" in result.stderr

    all_source_value = _review("paper_all_source")
    all_source_value["defensibility"][
        "indeterminate_ambiguity_defensible"
    ] = _summary(0, 3)
    _write(all_source, all_source_value)
    result = _run_reviews(all_source, readers)
    assert "indeterminate ambiguity split differs from human classifications" in result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_disagreement_requires_complete_resolver_provenance(tmp_path: Path) -> None:
    all_source_value = _review("paper_all_source")
    all_source_value["provenance"]["resolver_workbook"] = None
    all_source = tmp_path / "all-source.json"
    readers = tmp_path / "readers.json"
    _write(all_source, all_source_value)
    _write(readers, _review("paper_readers"))

    result = _run_reviews(all_source, readers)

    assert result.returncode == 1
    assert "provenance.resolver_workbook descriptor is required" in result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_review_protocol_is_exactly_frozen(tmp_path: Path) -> None:
    all_source_value = _review("paper_all_source")
    all_source_value["provenance"]["protocol"]["sha256"] = _digest(
        "another-protocol"
    )
    all_source = tmp_path / "all-source.json"
    readers = tmp_path / "readers.json"
    _write(all_source, all_source_value)
    _write(readers, _review("paper_readers"))

    result = _run_reviews(all_source, readers)

    assert result.returncode == 1
    assert "error-review protocol digest differs from the frozen protocol" in result.stderr
