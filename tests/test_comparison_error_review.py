from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from indra_belief.comparison import cli, error_review



# Reads the gitignored local artifact trees; skipped only when they are WHOLLY
# absent (CI, a fresh checkout). A PARTIAL tree is a failure in
# tests/test_local_artifacts.py, never a skip here.
import _local_artifacts as _artifacts

pytestmark = _artifacts.requires()

SECRET = b"fixture-only-human-review-key-32-bytes"
STAMP = "2026-07-21T01:00:00+00:00"


def _bytes(value: object) -> bytes:
    return error_review._canonical_bytes(value, pretty=True)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(error_review._canonical_bytes(row) + b"\n" for row in rows))


def _assert_embedded_javascript_parses(path: Path, tmp_path: Path) -> None:
    scripts = re.findall(r"<script(?: [^>]*)?>(.*?)</script>", path.read_text(), re.DOTALL)
    assert len(scripts) == 2
    javascript = tmp_path / f"{path.stem}.js"
    javascript.write_text(scripts[-1], encoding="utf-8")
    subprocess.run(["node", "--check", str(javascript)], check=True, capture_output=True)


def _descriptor(path: Path, owner: Path, *, rows: int | None = None) -> dict:
    payload = path.read_bytes()
    result = {
        "path": os.path.relpath(path, owner.parent),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _spec_descriptor(path: Path, owner: Path) -> dict:
    return {
        "path": os.path.relpath(path, owner.parent),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _pricing_contract() -> dict:
    return {
        "cost_comparability_id": "fixture_provider_token_cost",
        "currency": "USD",
        "provider": "Fixture Provider",
        "provider_model_id": "provider.fixture",
        "pricing_mode": "on_demand",
        "region": "fixture-region",
        "resolved_service_tier": "standard",
        "retrieved_on": "2026-07-21",
        "service_tier_request": "default",
        "source_url": "https://example.test/pricing",
        "tariff": {
            "input_usd_per_million": "0.5",
            "output_usd_per_million": "1",
            "pricing_basis": "fixture",
        },
        "unit": "per_million_tokens",
    }


def _cost_descriptor(
    path: Path,
    owner: Path,
    rows: int,
    run_id: str,
    *,
    projection: str,
) -> dict:
    base = _descriptor(path, owner, rows=rows)
    return {
        "accounting": {},
        "additive_across_panels": False,
        "basis": "provider_measured",
        **base,
        "cost_comparability_id": "fixture_provider_token_cost",
        "counterfactual_run_cost": False,
        "price_date": "2026-07-21",
        "price_source": "https://example.test/pricing",
        "pricing": _pricing_contract(),
        "projection": projection,
        "record_type": "evidence_execution",
        "shared_run_id": run_id,
        "status": "ledger",
        "view_id": "provider-runtime-retry-inclusive",
    }


def _spec_cost(path: Path, owner: Path, run_id: str, *, projection: str) -> dict:
    return {
        "accounting": {},
        "additive_across_panels": False,
        "basis": "provider_measured",
        "cost_comparability_id": "fixture_provider_token_cost",
        "counterfactual_run_cost": False,
        "path": os.path.relpath(path, owner.parent),
        "price_date": "2026-07-21",
        "price_source": "https://example.test/pricing",
        "pricing": _pricing_contract(),
        "projection": projection,
        "record_type": "evidence_execution",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shared_run_id": run_id,
        "status": "ledger",
        "view_id": "provider-runtime-retry-inclusive",
    }


def _fixture(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = tmp_path / "protocol.json"
    source_protocol = Path("data/comparison/error_review.json")
    protocol.write_bytes(source_protocol.read_bytes())
    codebook = tmp_path / "codebook.json"
    error_review.write_json(error_review.make_pilot_codebook(protocol), codebook)

    statement_ids = [
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000003",
    ]
    linked = "10000000-0000-4000-8000-000000000001"
    statements = [
        {
            "id": statement_ids[0],
            "matches_hash": "101",
            "type": "Activation",
            "subj": {"name": "A", "db_refs": {"HGNC": "1"}},
            "obj": {"name": "B", "id": "linked-internal-id"},
            "supports": [linked],
            "supported_by": [{"uuid": linked}],
            "evidence": [
                {
                    "source_api": "reach",
                    "source_hash": 1001,
                    "text": "A activates B.",
                    "pmid": "1",
                    "annotations": {"prior_uuids": [linked]},
                },
                {
                    "source_api": "signor",
                    "source_hash": 1002,
                    "text": "A may activate B in another context.",
                    "pmid": "2",
                },
            ],
        },
        {
            "id": statement_ids[1],
            "matches_hash": "102",
            "type": "Inhibition",
            "subj": {"name": "C"},
            "obj": {"name": "D"},
            "evidence": [
                {
                    "source_api": "reach",
                    "source_hash": 1003,
                    "text": "C inhibits D.",
                    "pmid": "3",
                }
            ],
        },
        {
            "id": statement_ids[2],
            "matches_hash": "103",
            "type": "Complex",
            "members": [{"name": "E"}, {"name": "F"}],
            "evidence": [
                {
                    "source_api": "signor",
                    "source_hash": 1004,
                    "text": "E binds F.",
                    "pmid": "4",
                }
            ],
        },
    ]
    statements_path = tmp_path / "statements.json"
    _write_json(statements_path, statements)
    served_model = "fixture-model"
    workload = "unique_exact_pairs_primary"
    map_rows: list[dict] = []
    execution_rows: list[dict] = []
    for statement_index, statement in enumerate(statements):
        for evidence_index, evidence in enumerate(statement["evidence"]):
            row = {
                "canonical_corpus_row_index": 100 + statement_index,
                "canonical_for_unique_pair": True,
                "eligible_position": statement_index,
                "evidence_json_sha256": hashlib.sha256(
                    error_review._canonical_bytes(evidence)
                ).hexdigest(),
                "evidence_position": 900 + evidence_index,
                "main_prompt_base_sha256": hashlib.sha256(
                    f"main-{statement_index}-{evidence_index}".encode()
                ).hexdigest(),
                "new_evidence_i": evidence_index,
                "new_stmt_i": statement_index,
                "pair_multiplicity": 1,
                "paper_statement_hash": statement["matches_hash"],
                "relation_prompt_sha256": None,
                "route": "plain",
                "source_api": evidence["source_api"],
                "source_hash": str(evidence["source_hash"]),
                "statement_type": statement["type"],
                "variant_ordinal": 0,
            }
            map_rows.append(row)
            execution_rows.append(
                {
                    "record_type": "evidence_execution",
                    "execution_identity": error_review._execution_identity(
                        row, served_model=served_model, workload=workload
                    ),
                    "statement_id": statement["id"],
                }
            )
    execution_map = tmp_path / "execution_map.jsonl"
    _write_jsonl(execution_map, map_rows)
    all_cost = tmp_path / "bundle" / "all_cost.jsonl"
    reader_cost = tmp_path / "bundle" / "reader_cost.jsonl"
    _write_jsonl(all_cost, execution_rows)
    reader_rows = [
        row for row, map_row in zip(execution_rows, map_rows, strict=True)
        if map_row["source_api"] == "reach"
    ]
    _write_jsonl(reader_cost, reader_rows)
    all_predictions = [
        {"statement_id": statement_ids[0], "probability_correct": 0.8},
        {"statement_id": statement_ids[1], "probability_correct": 0.2},
        {"statement_id": statement_ids[2], "probability_correct": 0.9},
    ]
    reader_predictions = all_predictions[:2]
    all_predictions_path = tmp_path / "bundle" / "all_predictions.jsonl"
    reader_predictions_path = tmp_path / "bundle" / "reader_predictions.jsonl"
    _write_jsonl(all_predictions_path, all_predictions)
    _write_jsonl(reader_predictions_path, reader_predictions)
    raw = tmp_path / "raw.jsonl"
    spend = tmp_path / "spend.ndjson"
    raw.write_text("{}\n", encoding="utf-8")
    spend.write_text("{}\n", encoding="utf-8")
    aggregation_config = tmp_path / "aggregation.json"
    aggregation = {
        "aggregation": "fixture_hard_gate",
        "kind": "statement_belief_aggregation",
        "priors": {"reach": [0.1, 0.01]},
        "reader_profile": None,
    }
    _write_json(aggregation_config, aggregation)
    pricing_config = tmp_path / "pricing.json"
    pricing = {
        "cost_comparability_id": "fixture_provider_token_cost",
        "currency": "USD",
        "kind": "provider_token_pricing",
        "provider": "Fixture Provider",
        "pricing_mode": "on_demand",
        "region": "fixture-region",
        "resolved_service_tier": "standard",
        "retrieved_on": "2026-07-21",
        "service_tier_request": "default",
        "source_url": "https://example.test/pricing",
        "tariffs": {"provider.fixture": _pricing_contract()["tariff"]},
        "unit": "per_million_tokens",
    }
    _write_json(pricing_config, pricing)
    bundle_path = tmp_path / "bundle" / "manifest.json"
    run_id = "fixture-run"
    bundle = {
        "kind": "llm_model_bundle",
        "model_id": "llm_fixture",
        "run_id": run_id,
        "implementation": {
            "implementation": "fixture",
            "implementation_digest": "a" * 64,
            "training_data_sha256": None,
            "environment": {"python": "fixture", "runtime": "fixture"},
            "notes": {
                "aggregation": "fixture_hard_gate",
                "dedup": True,
                "implementation_components": {},
                "inputs": {
                    "aggregation_config": _descriptor(aggregation_config, bundle_path),
                    "execution_map": _descriptor(execution_map, bundle_path),
                    "pricing_config": _descriptor(pricing_config, bundle_path),
                    "raw_attempts": _descriptor(raw, bundle_path),
                    "spend_ledger": _descriptor(spend, bundle_path),
                    "statements": _descriptor(statements_path, bundle_path),
                },
                "priors_sha256": hashlib.sha256(
                    error_review._canonical_bytes(aggregation["priors"])
                ).hexdigest(),
                "provider_model_id": "provider.fixture",
                "reader_profile": None,
                "reader_sources": sorted(error_review.READER_SOURCES),
                "served_model": served_model,
                "true_reader_reaggregated_from_pair_measurements": True,
                "workload": workload,
            },
        },
        "panels": {
            "paper_all_source": {
                "prediction_unit": "assembled_statement",
                "substrate_id": "paper_all_source",
                "predictions": _descriptor(
                    all_predictions_path, bundle_path, rows=len(all_predictions)
                ),
                "cost": _cost_descriptor(
                    all_cost,
                    bundle_path,
                    len(execution_rows),
                    run_id,
                    projection="all_executions",
                ),
            },
            "paper_readers": {
                "prediction_unit": "assembled_statement",
                "substrate_id": "paper_readers",
                "predictions": _descriptor(
                    reader_predictions_path, bundle_path, rows=len(reader_predictions)
                ),
                "cost": _cost_descriptor(
                    reader_cost,
                    bundle_path,
                    len(reader_rows),
                    run_id,
                    projection="observed_execution_subset",
                ),
            },
        },
    }
    _write_json(bundle_path, bundle)
    all_gold = [
        {"statement_id": statement_ids[0], "label": 0, "fold_id": 0},
        {"statement_id": statement_ids[1], "label": 1, "fold_id": 1},
        {"statement_id": statement_ids[2], "label": 1, "fold_id": 2},
    ]
    reader_gold = all_gold[:2]
    all_gold_path = tmp_path / "all_gold.jsonl"
    reader_gold_path = tmp_path / "reader_gold.jsonl"
    _write_jsonl(all_gold_path, all_gold)
    _write_jsonl(reader_gold_path, reader_gold)
    spec_path = tmp_path / "spec.json"

    def arm(panel: str, predictions: Path, cost: Path) -> dict:
        return {
            "arm_id": "llm_fixture",
            "family": "llm",
            "predictions": _spec_descriptor(predictions, spec_path),
            "cost": _spec_cost(
                cost,
                spec_path,
                run_id,
                projection=(
                    "all_executions"
                    if panel == "paper_all_source"
                    else "observed_execution_subset"
                ),
            ),
            "threshold": {
                "frozen_at": json.loads(protocol.read_text())["frozen_at"],
                "operator": "greater_than_or_equal",
                "source_path": os.path.relpath(protocol, spec_path.parent),
                "source_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "status": "available",
                "value": 0.5,
            },
        }

    spec = {
        "artifact_kind": "indra_statement_belief_evaluation_spec",
        "substrates": [
            {
                "substrate_id": "paper_all_source",
                "gold": _spec_descriptor(all_gold_path, spec_path),
                "arms": [arm("paper_all_source", all_predictions_path, all_cost)],
            },
            {
                "substrate_id": "paper_readers",
                "gold": _spec_descriptor(reader_gold_path, spec_path),
                "arms": [arm("paper_readers", reader_predictions_path, reader_cost)],
            },
        ],
    }
    _write_json(spec_path, spec)
    return {
        "protocol": protocol,
        "codebook": codebook,
        "bundle": bundle_path,
        "spec": spec_path,
        "statements": statements_path,
        "map": execution_map,
        "statement_ids": statement_ids,
        "all_cost": all_cost,
        "reader_cost": reader_cost,
    }


def _prepare(
    fixture: dict,
    tmp_path: Path,
    panel: str,
    *,
    codebook: Path | None = None,
    pilot_cases: int | None = 2,
) -> dict:
    return error_review.prepare_review_artifacts(
        spec_path=fixture["spec"],
        bundle_manifest_path=fixture["bundle"],
        panel_id=panel,
        arm_id="llm_fixture",
        protocol_path=fixture["protocol"],
        codebook_path=codebook or fixture["codebook"],
        blinding_key=SECRET,
        reviewer_output_dir=tmp_path / "reviewer_artifacts",
        admin_output_dir=tmp_path / "private_admin",
        pilot_case_count=pilot_cases,
        created_at=STAMP,
    )


def _workbooks(
    fixture: dict, tmp_path: Path, packet: Path, codebook: Path
) -> dict:
    return error_review.generate_reviewer_workbook(
        packet_paths=[packet],
        protocol_path=fixture["protocol"],
        codebook_path=codebook,
        blinding_key=SECRET,
        output_dir=tmp_path / "reviewer_artifacts",
    )


def _ledger(
    packet_path: Path,
    *,
    codebook: Path,
    workbook: dict,
    reviewer: str,
    classifications: list[tuple[str, list[str], str | None]],
) -> dict:
    packet = error_review.load_json(packet_path)
    return {
        "artifact_kind": error_review.LEDGER_KIND,
        "role": "reviewer",
        "reviewer_slot": workbook["reviewer_slot"],
        "assignment_id": workbook["assignment_id"],
        "review_phase": packet["review_phase"],
        "packet_id": packet["packet_id"],
        "protocol_sha256": packet["protocol_sha256"],
        "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "codebook_sha256": hashlib.sha256(codebook.read_bytes()).hexdigest(),
        "workbook_content_sha256": workbook["workbook_content_sha256"],
        "reviewer_pseudonym": reviewer,
        "human_attestation": error_review.HUMAN_ATTESTATION,
        "started_at": STAMP,
        "completed_at": "2026-07-21T02:00:00+00:00",
        "decisions": [
            {
                "case_id": case["case_id"],
                "classification": classification,
                "dimensions": dimensions,
                "comment": comment,
            }
            for case, (classification, dimensions, comment) in zip(
                packet["cases"], classifications, strict=True
            )
        ],
    }


def _frozen_codebook(fixture: dict, tmp_path: Path) -> Path:
    prepared = _prepare(fixture, tmp_path, "paper_all_source")
    generated = _workbooks(
        fixture, tmp_path, prepared["packet"], fixture["codebook"]
    )
    choices = [
        ("supports_claim", ["grounding_ambiguity"], None),
        ("rejects_claim", ["explicit_support"], None),
    ]
    reviews: list[Path] = []
    for workbook, reviewer in zip(
        generated["workbooks"], ("pilot.alpha", "pilot.beta"), strict=True
    ):
        path = tmp_path / f"{reviewer}.json"
        _write_json(
            path,
            _ledger(
                prepared["packet"],
                codebook=fixture["codebook"],
                workbook=workbook,
                reviewer=reviewer,
                classifications=choices,
            ),
        )
        reviews.append(path)
    frozen = tmp_path / "frozen_codebook.json"
    error_review.freeze_codebook(
        protocol_path=fixture["protocol"],
        pilot_codebook_path=fixture["codebook"],
        candidate_codebook_path=fixture["codebook"],
        pilot_packet_path=prepared["packet"],
        pilot_admin_manifest_path=prepared["admin_manifest"],
        pilot_workbook_paths=[
            workbook["workbook"] for workbook in generated["workbooks"]
        ],
        reviewer_ledger_paths=reviews,
        blinding_key=SECRET,
        human_freeze_attested=True,
        frozen_at="2026-07-21T03:00:00+00:00",
        output_path=frozen,
    )
    return frozen


def _write_reviews(
    *,
    packet: Path,
    codebook: Path,
    generated: dict,
    tmp_path: Path,
    choices_a: list[tuple[str, list[str], str | None]],
    choices_b: list[tuple[str, list[str], str | None]],
) -> list[Path]:
    paths: list[Path] = []
    for workbook, reviewer, choices in zip(
        generated["workbooks"],
        ("reviewer.alpha", "reviewer.beta"),
        (choices_a, choices_b),
        strict=True,
    ):
        path = tmp_path / f"{reviewer}_{workbook['reviewer_slot']}.json"
        _write_json(
            path,
            _ledger(
                packet,
                codebook=codebook,
                workbook=workbook,
                reviewer=reviewer,
                classifications=choices,
            ),
        )
        paths.append(path)
    return paths


def test_packet_is_outcome_blind_and_admin_retains_exact_direction(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture, tmp_path, "paper_all_source")
    packet = error_review.load_json(prepared["packet"])
    admin = error_review.load_json(prepared["admin_manifest"])

    assert packet["threshold_error_count"] == 2
    assert all(
        set(case) == {"case_id", "material"}
        for case in packet["cases"]
    )
    public_text = prepared["packet"].read_text()
    for forbidden in (
        "false_positive",
        "false_negative",
        "reference_label",
        "probability_correct",
        "system side",
        "released-reference",
        "llm_fixture",
        "fixture-model",
    ):
        assert forbidden not in public_text
    assert admin["threshold_error_census"] == {
        "count": 2,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert {row["error_type"] for row in admin["case_mapping"]} == {
        "false_positive",
        "false_negative",
    }
    material_text = json.dumps(packet["cases"], sort_keys=True)
    for forbidden in (
        '"id"',
        "supported_by",
        "supports",
        "prior_uuids",
        "source_hash",
        "pmid",
        fixture["statement_ids"][0],
    ):
        assert forbidden not in material_text
    assert (prepared["admin_manifest"].stat().st_mode & 0o777) == 0o600
    assert prepared["packet"].parent != prepared["admin_manifest"].parent


def test_exact_threshold_and_large_input_descriptors_are_enforced(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    spec = json.loads(fixture["spec"].read_text())
    spec["substrates"][0]["arms"][0]["threshold"]["value"] = 0.75
    _write_json(fixture["spec"], spec)
    with pytest.raises(error_review.ErrorReviewError, match="threshold value"):
        _prepare(fixture, tmp_path, "paper_all_source")

    fixture = _fixture(tmp_path / "raw_tamper")
    raw_path = tmp_path / "raw_tamper" / "raw.jsonl"
    raw_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(error_review.ErrorReviewError, match="raw attempts"):
        _prepare(fixture, tmp_path / "raw_tamper", "paper_all_source")


def test_recursive_scrub_rejects_embedded_uuid_and_hash_tokens() -> None:
    with pytest.raises(error_review.ErrorReviewError, match="UUID or hash"):
        error_review._scrub_material(
            {"text": "linked:00000000-0000-4000-8000-000000000001?x=1"},
            forbidden_values=(),
            context="fixture",
        )
    with pytest.raises(error_review.ErrorReviewError, match="UUID or hash"):
        error_review._scrub_material(
            {"text": "sha256=" + "a" * 64 + "#fragment"},
            forbidden_values=(),
            context="fixture",
        )
    with pytest.raises(error_review.ErrorReviewError, match="input identity"):
        error_review._scrub_material(
            {"text": "leaked llm_fixture identity"},
            forbidden_values=(
                "llm_fixture",
                "00000000-0000-4000-8000-000000000001",
                "b" * 64,
            ),
            context="fixture",
        )


def test_reviewer_workbooks_are_distinct_blinded_assignments(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture, tmp_path, "paper_all_source")
    generated = _workbooks(
        fixture, tmp_path, prepared["packet"], fixture["codebook"]
    )
    assert [row["reviewer_slot"] for row in generated["workbooks"]] == ["A", "B"]
    assert len({row["assignment_id"] for row in generated["workbooks"]}) == 2
    assert len(
        {row["workbook_content_sha256"] for row in generated["workbooks"]}
    ) == 2

    orders: list[list[str]] = []
    for row in generated["workbooks"]:
        path = row["workbook"]
        text = path.read_text()
        for forbidden in (
            "false_positive",
            "false_negative",
            "System side",
            "released-reference",
        ):
            assert forbidden not in text
        assert "Affirm the human-only attestation" in text
        assert "Export checkpoint" in text
        match = re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            text,
            re.DOTALL,
        )
        assert match is not None
        payload = json.loads(match.group(1))
        assert "protocol" not in payload
        assert "defensibility_derivation" not in text
        orders.append([task["task_id"] for task in payload["workbook"]["tasks"]])
        _assert_embedded_javascript_parses(path, tmp_path)
    assert orders[0] != orders[1]


def test_freeze_requires_authenticated_pilot_and_affirmative_attestation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = _prepare(fixture, tmp_path, "paper_all_source")
    generated = _workbooks(
        fixture, tmp_path, prepared["packet"], fixture["codebook"]
    )
    choices = [
        ("supports_claim", ["grounding_ambiguity"], None),
        ("rejects_claim", ["explicit_support"], None),
    ]
    reviews = _write_reviews(
        packet=prepared["packet"],
        codebook=fixture["codebook"],
        generated=generated,
        tmp_path=tmp_path,
        choices_a=choices,
        choices_b=choices,
    )
    common = {
        "protocol_path": fixture["protocol"],
        "pilot_codebook_path": fixture["codebook"],
        "candidate_codebook_path": fixture["codebook"],
        "pilot_packet_path": prepared["packet"],
        "pilot_admin_manifest_path": prepared["admin_manifest"],
        "pilot_workbook_paths": [
            row["workbook"] for row in generated["workbooks"]
        ],
        "reviewer_ledger_paths": reviews,
        "frozen_at": "2026-07-21T03:00:00+00:00",
    }
    with pytest.raises(error_review.ErrorReviewError, match="explicit human"):
        error_review.freeze_codebook(
            **common,
            blinding_key=SECRET,
            human_freeze_attested=False,
            output_path=tmp_path / "not_frozen.json",
        )
    with pytest.raises(error_review.ErrorReviewError, match="binding"):
        error_review.freeze_codebook(
            **common,
            blinding_key=b"x" * 32,
            human_freeze_attested=True,
            output_path=tmp_path / "wrong_key.json",
        )
    with pytest.raises(error_review.ErrorReviewError, match="must not precede"):
        error_review.freeze_codebook(
            **{**common, "frozen_at": "2026-07-21T01:30:00+00:00"},
            blinding_key=SECRET,
            human_freeze_attested=True,
            output_path=tmp_path / "early_freeze.json",
        )
    duplicate_identity = error_review.load_json(reviews[1])
    duplicate_identity["reviewer_pseudonym"] = "REVIEWER.ALPHA"
    duplicate_path = tmp_path / "duplicate_identity.json"
    _write_json(duplicate_path, duplicate_identity)
    with pytest.raises(error_review.ErrorReviewError, match="distinct pseudonyms"):
        error_review.freeze_codebook(
            **{**common, "reviewer_ledger_paths": [reviews[0], duplicate_path]},
            blinding_key=SECRET,
            human_freeze_attested=True,
            output_path=tmp_path / "duplicate_reviewers.json",
        )
    frozen = tmp_path / "frozen.json"
    result = error_review.freeze_codebook(
        **common,
        blinding_key=SECRET,
        human_freeze_attested=True,
        output_path=frozen,
    )
    assert result["pilot_case_count"] == 2
    assert error_review.load_json(frozen)["status"] == "frozen"
    forged = error_review.load_json(frozen)
    forged["frozen_at"] = "2026-07-21T03:01:00+00:00"
    forged_path = tmp_path / "forged_freeze.json"
    _write_json(forged_path, forged)
    with pytest.raises(error_review.ErrorReviewError, match="authentication binding"):
        _prepare(
            fixture,
            tmp_path,
            "paper_all_source",
            codebook=forged_path,
            pilot_cases=None,
        )


def test_complete_agreement_report_derives_defensibility_after_review(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    frozen = _frozen_codebook(fixture, tmp_path)
    prepared = _prepare(
        fixture,
        tmp_path,
        "paper_all_source",
        codebook=frozen,
        pilot_cases=None,
    )
    generated = _workbooks(fixture, tmp_path, prepared["packet"], frozen)
    choices = [
        ("supports_claim", ["explicit_support"], "direct support"),
        ("supports_claim", ["evidence_insufficient"], None),
    ]
    reviews = _write_reviews(
        packet=prepared["packet"],
        codebook=frozen,
        generated=generated,
        tmp_path=tmp_path,
        choices_a=choices,
        choices_b=choices,
    )
    report = error_review.adjudicate_review(
        packet_path=prepared["packet"],
        admin_manifest_path=prepared["admin_manifest"],
        protocol_path=fixture["protocol"],
        codebook_path=frozen,
        reviewer_ledger_paths=reviews,
        reviewer_workbook_packet_paths=[prepared["packet"]],
        reviewer_workbook_paths=[
            row["workbook"] for row in generated["workbooks"]
        ],
        blinding_key=SECRET,
    )
    assert report["status"] == "complete"
    assert report["defensibility"]["defensible"]["count"] == 1
    assert report["defensibility"]["non_defensible"]["count"] == 1
    assert report["defensibility"]["unresolved"]["count"] == 0
    assert report["human_classifications"]["supports_claim"]["count"] == 2
    assert report["defensibility"]["system_supported_defensible"]["count"] == 1
    assert report["defensibility"]["indeterminate_ambiguity_defensible"]["count"] == 0
    assert report["review"]["disagreement_count"] == 0
    assert report["review"]["classification_reliability"]["cohen_kappa"] is None
    assert {row["human_classification"] for row in report["adjudications"]} == {
        "supports_claim"
    }
    assert len(report["taxonomy_refinements"]) == 2
    provenance_text = json.dumps(report["provenance"], sort_keys=True)
    assert str(tmp_path) not in provenance_text
    assert '"path"' not in provenance_text


def test_disagreements_require_exact_human_resolver_chain(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    frozen = _frozen_codebook(fixture, tmp_path)
    prepared = _prepare(
        fixture,
        tmp_path,
        "paper_all_source",
        codebook=frozen,
        pilot_cases=None,
    )
    generated = _workbooks(fixture, tmp_path, prepared["packet"], frozen)
    choices_a = [
        ("supports_claim", ["taxonomy_gap"], "new subtype A"),
        ("rejects_claim", ["evidence_insufficient"], None),
    ]
    choices_b = [
        ("supports_claim", ["taxonomy_gap"], "new subtype B"),
        ("rejects_claim", ["evidence_insufficient"], None),
    ]
    reviews = _write_reviews(
        packet=prepared["packet"],
        codebook=frozen,
        generated=generated,
        tmp_path=tmp_path,
        choices_a=choices_a,
        choices_b=choices_b,
    )
    adjudicate_args = {
        "packet_path": prepared["packet"],
        "admin_manifest_path": prepared["admin_manifest"],
        "protocol_path": fixture["protocol"],
        "codebook_path": frozen,
        "reviewer_ledger_paths": reviews,
        "reviewer_workbook_packet_paths": [prepared["packet"]],
        "reviewer_workbook_paths": [
            row["workbook"] for row in generated["workbooks"]
        ],
        "blinding_key": SECRET,
    }
    with pytest.raises(error_review.ErrorReviewError, match="human resolver required"):
        error_review.adjudicate_review(**adjudicate_args)

    resolver = error_review.generate_resolver_workload(
        packet_path=prepared["packet"],
        protocol_path=fixture["protocol"],
        codebook_path=frozen,
        reviewer_ledger_paths=reviews,
        reviewer_workbook_packet_paths=[prepared["packet"]],
        reviewer_workbook_paths=[
            row["workbook"] for row in generated["workbooks"]
        ],
        blinding_key=SECRET,
        output_dir=tmp_path / "reviewer_artifacts",
    )
    assert resolver["disagreement_count"] == 1
    _assert_embedded_javascript_parses(resolver["resolver_workbook"], tmp_path)
    workload = error_review.load_json(resolver["resolver_workload"])
    assert all("error_type" not in case for case in workload["cases"])
    resolver_ledger = tmp_path / "resolver.json"
    _write_json(
        resolver_ledger,
        {
            "artifact_kind": error_review.LEDGER_KIND,
            "role": "resolver",
            "review_phase": "full",
            "packet_id": workload["packet_id"],
            "protocol_sha256": workload["protocol_sha256"],
            "packet_sha256": workload["packet_sha256"],
            "codebook_sha256": workload["codebook_sha256"],
            "resolver_workload_sha256": hashlib.sha256(
                resolver["resolver_workload"].read_bytes()
            ).hexdigest(),
            "reviewer_ledger_sha256s": workload["reviewer_ledger_sha256s"],
            "resolver_pseudonym": "resolver.gamma",
            "human_attestation": error_review.HUMAN_ATTESTATION,
            "started_at": STAMP,
            "completed_at": "2026-07-21T04:00:00+00:00",
            "decisions": [
                {
                    "case_id": workload["cases"][0]["case_id"],
                    "classification": "indeterminate",
                    "dimensions": ["taxonomy_gap"],
                    "comment": "retain both proposed subtypes for the next codebook",
                }
            ],
        },
    )
    report = error_review.adjudicate_review(
        **adjudicate_args,
        resolver_workload_path=resolver["resolver_workload"],
        resolver_workbook_path=resolver["resolver_workbook"],
        resolver_ledger_path=resolver_ledger,
    )
    assert report["status"] == "complete"
    assert report["review"]["resolved_by_resolver_count"] == 1
    assert report["defensibility"]["unresolved"]["count"] == 0
    resolved = [
        row for row in report["adjudications"]
        if row["decision_source"] == "resolver"
    ]
    assert resolved[0]["human_classification"] == "indeterminate"
    assert resolved[0]["judgment"] == "defensible"
    assert resolved[0]["defensibility_basis"] == "indeterminate_ambiguity"
    assert report["defensibility"]["indeterminate_ambiguity_defensible"]["count"] == 1


def test_wrong_reviewer_workbook_bytes_invalidate_scientific_chain(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    frozen = _frozen_codebook(fixture, tmp_path)
    prepared = _prepare(
        fixture, tmp_path, "paper_all_source", codebook=frozen, pilot_cases=None
    )
    generated = _workbooks(fixture, tmp_path, prepared["packet"], frozen)
    choices = [
        ("supports_claim", ["explicit_support"], None),
        ("rejects_claim", ["evidence_insufficient"], None),
    ]
    reviews = _write_reviews(
        packet=prepared["packet"],
        codebook=frozen,
        generated=generated,
        tmp_path=tmp_path,
        choices_a=choices,
        choices_b=choices,
    )
    forged = tmp_path / "forged.html"
    forged.write_bytes(generated["workbooks"][0]["workbook"].read_bytes() + b" ")
    with pytest.raises(error_review.ErrorReviewError, match="exact canonical HTML"):
        error_review.adjudicate_review(
            packet_path=prepared["packet"],
            admin_manifest_path=prepared["admin_manifest"],
            protocol_path=fixture["protocol"],
            codebook_path=frozen,
            reviewer_ledger_paths=reviews,
            reviewer_workbook_packet_paths=[prepared["packet"]],
            reviewer_workbook_paths=[
                forged,
                generated["workbooks"][1]["workbook"],
            ],
            blinding_key=SECRET,
        )


def test_key_permissions_and_handoff_directories_are_private_boundaries(
    tmp_path: Path,
) -> None:
    key = tmp_path / "review.key"
    key.write_text("ab" * 32 + "\n", encoding="ascii")
    key.chmod(0o644)
    with pytest.raises(error_review.ErrorReviewError, match="no group/other access"):
        error_review.load_blinding_key(key)
    key.chmod(0o600)
    assert error_review.load_blinding_key(key) == bytes.fromhex("ab" * 32)

    fixture = _fixture(tmp_path / "nested")
    reviewer_dir = tmp_path / "nested" / "handoff"
    with pytest.raises(error_review.ErrorReviewError, match="non-nested"):
        error_review.prepare_review_artifacts(
            spec_path=fixture["spec"],
            bundle_manifest_path=fixture["bundle"],
            panel_id="paper_all_source",
            arm_id="llm_fixture",
            protocol_path=fixture["protocol"],
            codebook_path=fixture["codebook"],
            blinding_key=SECRET,
            reviewer_output_dir=reviewer_dir,
            admin_output_dir=reviewer_dir / "admin",
            pilot_case_count=2,
        )


def test_cli_requires_separate_admin_output_and_bound_workbooks() -> None:
    parser = cli._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "error-review-prepare",
                "--bundle",
                "bundle.json",
                "--panel",
                "paper_all_source",
                "--arm",
                "llm",
                "--codebook",
                "codebook.json",
                "--blinding-key-file",
                "key",
            ]
        )
    parsed = parser.parse_args(
        [
            "error-review-adjudicate",
            "--packet",
            "packet.json",
            "--admin-manifest",
            "admin.json",
            "--codebook",
            "frozen.json",
            "--blinding-key-file",
            "key",
            "--reviews",
            "a.json",
            "b.json",
            "--workbook-packets",
            "packet.json",
            "--reviewer-workbooks",
            "a.html",
            "b.html",
            "--output",
            "report.json",
        ]
    )
    assert parsed.reviewer_workbooks == [Path("a.html"), Path("b.html")]
