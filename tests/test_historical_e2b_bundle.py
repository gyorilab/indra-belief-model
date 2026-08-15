from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from indra_belief.comparison import llm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reproduce_published_statement_beliefs import (  # noqa: E402
    published_reproduction,
)


# Reads the gitignored local artifact trees; skipped only when they are WHOLLY
# absent (CI, a fresh checkout). A PARTIAL tree is a failure in
# tests/test_local_artifacts.py, never a skip here.
import _local_artifacts as _artifacts

pytestmark = _artifacts.requires()

# The statement_belief.py bytes that produced this bundle. The live file is
# allowed to differ from it; what may not differ is what the file COMPUTES,
# which `published_reproduction` re-derives from the frozen observations. The
# constant stays spelled out so drift in the published artifact stays visible.
HISTORICAL_STATEMENT_BELIEF_SHA256 = (
    "8327ff74a8f34a4872abbe37a3754255031605bf9bece07c38e4ba4f6425ed06"
)
HISTORICAL_IMPLEMENTATION_COMPONENTS = {
    "noise_model": "6a4276840d3281f32e01948a62699d566e01be306e3f628d9901936ff5730e73",
    "shared_text_normalization": (
        "a95ada5c8ca5e376beb60ab3934f58dff58b834b9d31f6a005f004bbe9139145"
    ),
    "statement_belief": HISTORICAL_STATEMENT_BELIEF_SHA256,
}

BUNDLE_DIR = ROOT / "data/comparison/models/gemma_4_e2b"
MANIFEST_PATH = BUNDLE_DIR / "manifest.json"
RUN_ID = (
    "paper_lane_p_wave1_gemma4_glm5_20260717__"
    "bedrock-gemma-4-e2b__unique_exact_pairs_primary"
)
SERVED_MODEL = "bedrock-gemma-4-e2b"
PROVIDER_MODEL = "google.gemma-4-e2b"
WORKLOAD = "unique_exact_pairs_primary"

FROZEN_INPUTS = {
    "aggregation_config": (
        1_718,
        "1a5ceaa9ab6ef56a5dfe9a7a9fe0ebfc59686a62d094d1670810235de4a9e045",
    ),
    "execution_map": (
        19_664_407,
        "8dd165e19136d5b2d34a20c605d3cb82e0ceec0f47f1b61fe265833a7c076551",
    ),
    "pricing_config": (
        1_012,
        "19805937e49e50b4bb0f53fa15a1b03a0496c6ff264952cdc58d12a54229261b",
    ),
    "raw_attempts": (
        787_887_155,
        "b2376c4d5e3aa3d5fa5327f0058a8465a4ec856f886cbeea405a539066beeb4b",
    ),
    "spend_ledger": (
        228_147_098,
        "5c48cfca243f1100d7bef20feda3e47831e2c0696a9574061c0e65d60cf6ad70",
    ),
    "statements": (
        21_470_531,
        "bf6075cfd71944890f156239e580e7e92a6e53fb80b270841b050cda1dab0fe0",
    ),
}
FROZEN_OUTPUTS = {
    "all_source_attempts.jsonl": (
        23_765_079,
        33_361,
        "12f4eb60ac7394be5168d6a6a31d33de229870c16ecc773022183ea26a545210",
    ),
    "all_source_predictions.jsonl": (
        148_102,
        1_689,
        "571a15b3d45dc3a6a514f904c2d6f1fff0754e63d5077eda0a6290c3de24891a",
    ),
    "reader_attempts.jsonl": (
        23_322_246,
        32_479,
        "490a1fa2992eb5993c51285f37a5a5f808caa1947d876a5c4d04aeb5534281b1",
    ),
    "reader_predictions.jsonl": (
        146_841,
        1_676,
        "48913066a679242c8fb4146dd8a020a58763ae0ca8fac6f138da3fbd2f09780b",
    ),
}
HISTORICAL_EVENT_COUNTS = {
    "attempt_finished": 33_436,
    "attempt_limit_checkpoint": 7,
    "attempt_started": 33_436,
    "call_reserved": 49_651,
    "call_settled": 49_651,
    "guard_closed": 13,
    "guard_opened": 13,
    "ledger_initialized": 1,
    "recovery_authorization_registered": 10,
    "stage_authorization_pinned": 1,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON key {key!r}"
        result[key] = value
    return result


def _json(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_object_without_duplicates)
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _money(value: Decimal) -> str:
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result or "0"


def _descriptor_path(descriptor: dict[str, Any]) -> Path:
    path = (BUNDLE_DIR / descriptor["path"]).resolve()
    assert path.is_file()
    return path


def _assert_descriptor(
    descriptor: dict[str, Any], expected: tuple[int, str] | tuple[int, int, str]
) -> Path:
    expected_bytes = expected[0]
    expected_rows = expected[1] if len(expected) == 3 else None
    expected_sha = expected[-1]
    assert descriptor["bytes"] == expected_bytes
    assert descriptor["sha256"] == expected_sha
    if expected_rows is not None:
        assert descriptor["rows"] == expected_rows
    path = _descriptor_path(descriptor)
    assert path.stat().st_size == expected_bytes
    assert _sha256(path) == expected_sha
    return path


def _audit_historical_ledger(path: Path) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    previous: str | None = None
    ledger_id: str | None = None
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    starts: dict[str, dict[str, Any]] = {}
    finishes: dict[str, dict[str, Any]] = {}
    reservations: dict[str, dict[str, Any]] = {}
    settlements: dict[str, dict[str, Any]] = {}

    with path.open("rb") as stream:
        for sequence, line in enumerate(stream, start=1):
            digest.update(line)
            assert line.endswith(b"\n")
            encoded = line[:-1]
            row = _json(encoded)
            assert _canonical(row) == encoded
            event_digest = row.pop("event_sha256")
            assert event_digest == hashlib.sha256(_canonical(row)).hexdigest()
            assert row["sequence"] == sequence
            assert row["previous_event_sha256"] == previous
            previous = event_digest
            if ledger_id is None:
                ledger_id = row["ledger_id"]
                assert row["event"] == "ledger_initialized"
                assert row["global_authorization_cap_usd"] == "400"
            assert row["ledger_id"] == ledger_id
            event = row["event"]
            counts[event] += 1
            row["event_sha256"] = event_digest

            if event == "attempt_started" and row.get("run_id") == RUN_ID:
                attempt_id = row["attempt_id"]
                assert attempt_id not in starts
                starts[attempt_id] = row
            elif event == "attempt_finished" and row.get("attempt_id") in starts:
                attempt_id = row["attempt_id"]
                assert attempt_id not in finishes
                finishes[attempt_id] = row
            elif event == "call_reserved" and row.get("attempt_id") in starts:
                call_id = row["call_id"]
                assert call_id not in reservations
                reservations[call_id] = row
            elif event == "call_settled" and row.get("call_id") in reservations:
                call_id = row["call_id"]
                assert call_id not in settlements
                settlements[call_id] = row

    assert digest.hexdigest() == FROZEN_INPUTS["spend_ledger"][1]
    assert previous == "0eab31c6d434bf78de3fff814802f27a04877631b4f3f85ab346c703ed9691d1"
    assert ledger_id == "b19dac1166a643a394e619b46a5c50f7"
    assert dict(counts) == HISTORICAL_EVENT_COUNTS
    assert set(starts) == set(finishes)
    assert set(reservations) == set(settlements)
    return starts, finishes, reservations, settlements


def _cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = Decimal(0)
    conservative = Decimal(0)
    measured_calls = conservative_calls = attempts = input_tokens = output_tokens = 0
    statement_ids: set[str] = set()
    for row in rows:
        statement_ids.add(row["statement_id"])
        attempts += len(row["attempts"])
        for attempt in row["attempts"]:
            for call in attempt["calls"]:
                cost = Decimal(call["settled_cost_usd_exact"])
                if call["accounting_basis"] == "provider_reported_usage":
                    measured += cost
                    measured_calls += 1
                    input_tokens += call["input_tokens"]
                    output_tokens += call["output_tokens"]
                else:
                    assert call["accounting_basis"] == "conservative_reserved_maximum"
                    conservative += cost
                    conservative_calls += 1
    return {
        "rows": len(rows),
        "statements": len(statement_ids),
        "attempts": attempts,
        "provider_measured_call_count": measured_calls,
        "conservative_call_count": conservative_calls,
        "provider_input_tokens": input_tokens,
        "provider_output_tokens": output_tokens,
        "provider_measured_cost_usd_exact": _money(measured),
        "conservative_reserved_cost_usd_exact": _money(conservative),
        "accounted_cost_upper_usd_exact": _money(measured + conservative),
    }


def test_historical_e2b_bundle_is_audited_under_the_canonical_contract() -> None:
    manifest = _json(MANIFEST_PATH.read_bytes())
    assert set(manifest) == {"implementation", "kind", "model_id", "panels", "run_id"}
    assert manifest["kind"] == "llm_model_bundle"
    assert manifest["model_id"] == "llm_gemma_4_e2b"
    assert manifest["run_id"] == RUN_ID

    implementation = manifest["implementation"]
    # The manifest's complete component vector is a historical record, so audit
    # its recorded digest from those recorded bytes rather than splicing today's
    # unrelated component hashes into it. Behaviour remains independently
    # re-derived below from the frozen observations; noise_model is additionally
    # byte-frozen and therefore still matches its historical component digest.
    _, current_components = llm._implementation_digest()
    frozen_components = dict(HISTORICAL_IMPLEMENTATION_COMPONENTS)
    assert implementation["implementation_digest"] == llm._sha256(
        llm._canonical(frozen_components)
    )
    assert implementation["training_data_sha256"] is None
    notes = implementation["notes"]
    assert notes["implementation_components"] == frozen_components
    assert current_components["noise_model"] == frozen_components["noise_model"]
    assert published_reproduction().ok, [
        mismatch.describe() for mismatch in published_reproduction().mismatches
    ]
    assert notes["served_model"] == SERVED_MODEL
    assert notes["provider_model_id"] == PROVIDER_MODEL
    assert notes["workload"] == WORKLOAD

    input_descriptors = notes["inputs"]
    assert set(input_descriptors) == set(FROZEN_INPUTS)
    input_paths = {
        name: _assert_descriptor(input_descriptors[name], expected)
        for name, expected in FROZEN_INPUTS.items()
    }
    aggregation = _json(input_paths["aggregation_config"].read_bytes())
    priors, reader_profile, aggregation_name = llm._aggregation_config(aggregation)
    normalized_priors = llm._validate_priors(priors, set(priors))
    priors_json = {
        source: list(parameters)
        for source, parameters in sorted(normalized_priors.items())
    }
    assert notes["aggregation"] == aggregation_name
    assert notes["reader_profile"] == reader_profile
    assert notes["priors_sha256"] == hashlib.sha256(
        _canonical(priors_json)
    ).hexdigest()

    pricing = _json(input_paths["pricing_config"].read_bytes())
    tariff = pricing["tariffs"][PROVIDER_MODEL]
    expected_pricing = {
        "cost_comparability_id": pricing["cost_comparability_id"],
        "currency": pricing["currency"],
        "pricing_mode": pricing["pricing_mode"],
        "provider": pricing["provider"],
        "provider_model_id": PROVIDER_MODEL,
        "region": pricing["region"],
        "resolved_service_tier": pricing["resolved_service_tier"],
        "retrieved_on": pricing["retrieved_on"],
        "service_tier_request": pricing["service_tier_request"],
        "source_url": pricing["source_url"],
        "tariff": tariff,
        "unit": pricing["unit"],
    }

    panels = manifest["panels"]
    assert set(panels) == {"paper_all_source", "paper_readers"}
    output_descriptors = {
        panels["paper_all_source"]["cost"]["path"]: panels["paper_all_source"]["cost"],
        panels["paper_all_source"]["predictions"]["path"]: panels["paper_all_source"]["predictions"],
        panels["paper_readers"]["cost"]["path"]: panels["paper_readers"]["cost"],
        panels["paper_readers"]["predictions"]["path"]: panels["paper_readers"]["predictions"],
    }
    assert set(output_descriptors) == set(FROZEN_OUTPUTS)
    output_paths = {
        name: _assert_descriptor(output_descriptors[name], expected)
        for name, expected in FROZEN_OUTPUTS.items()
    }
    for projection, panel in (
        ("all_executions", panels["paper_all_source"]),
        ("observed_execution_subset", panels["paper_readers"]),
    ):
        cost = panel["cost"]
        assert cost["pricing"] == expected_pricing
        assert cost["cost_comparability_id"] == pricing["cost_comparability_id"]
        assert cost["price_source"] == pricing["source_url"]
        assert cost["price_date"] == pricing["retrieved_on"]
        assert cost["projection"] == projection
        assert cost["counterfactual_run_cost"] is False
        assert cost["shared_run_id"] == RUN_ID
        assert cost["additive_across_panels"] is False

    starts, finishes, reservations, settlements = _audit_historical_ledger(
        input_paths["spend_ledger"]
    )
    reservations_by_attempt: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for call_id, reservation in reservations.items():
        settlement = settlements[call_id]
        assert reservation["provider_model_id"] == PROVIDER_MODEL
        assert settlement["provider_model_id"] == PROVIDER_MODEL
        assert reservation["input_usd_per_million_tokens"] == tariff["input_usd_per_million"]
        assert reservation["output_usd_per_million_tokens"] == tariff["output_usd_per_million"]
        assert reservation["price_basis"] == tariff["pricing_basis"]
        maximum = (
            Decimal(reservation["reserved_input_tokens"])
            * Decimal(tariff["input_usd_per_million"])
            + Decimal(reservation["reserved_output_tokens"])
            * Decimal(tariff["output_usd_per_million"])
        ) / Decimal(1_000_000)
        assert Decimal(reservation["reserved_max_cost_usd"]) == maximum
        settled = Decimal(settlement["settled_cost_usd"])
        usage = settlement["provider_usage"]
        if settlement["accounting_basis"] == "provider_reported_usage":
            expected = (
                Decimal(usage["input_tokens"])
                * Decimal(tariff["input_usd_per_million"])
                + Decimal(usage["output_tokens"])
                * Decimal(tariff["output_usd_per_million"])
            ) / Decimal(1_000_000)
            assert settled == expected
        else:
            assert settlement["accounting_basis"] == "conservative_reserved_maximum"
            assert usage == {"input_tokens": None, "output_tokens": None}
            assert settled == maximum
        reservations_by_attempt[reservation["attempt_id"]].append(reservation)
    for attempt_reservations in reservations_by_attempt.values():
        attempt_reservations.sort(key=lambda row: row["call_ordinal"])

    all_rows = [
        _json(line)
        for line in output_paths["all_source_attempts.jsonl"].read_bytes().splitlines()
    ]
    reader_rows = [
        _json(line)
        for line in output_paths["reader_attempts.jsonl"].read_bytes().splitlines()
    ]
    seen_attempts: set[str] = set()
    seen_calls: set[str] = set()
    execution_ids: set[str] = set()
    clean_abstention_predecessors = 0
    for row in all_rows:
        assert set(row) == {
            "attempts",
            "call_eligible",
            "execution_identity",
            "record_type",
            "statement_id",
        }
        execution_id = row["execution_identity"]
        assert execution_id not in execution_ids
        execution_ids.add(execution_id)
        attempts = row["attempts"]
        assert [attempt["attempt_ordinal"] for attempt in attempts] == list(
            range(1, len(attempts) + 1)
        )
        assert [attempt["selected_final"] for attempt in attempts] == [
            False
        ] * (len(attempts) - 1) + [True]
        for attempt in attempts:
            assert set(attempt) == {
                "attempt_id",
                "attempt_ordinal",
                "calls",
                "error_type",
                "selected_final",
                "status",
            }
            attempt_id = attempt["attempt_id"]
            assert attempt_id not in seen_attempts
            seen_attempts.add(attempt_id)
            start = starts[attempt_id]
            finish = finishes[attempt_id]
            assert start["execution_id"] == execution_id
            assert start["attempt_ordinal"] == attempt["attempt_ordinal"]
            assert start["run_id"] == RUN_ID
            assert start["model"] == SERVED_MODEL
            assert start["workload"] == WORKLOAD
            if attempt["status"] != finish["status"]:
                assert attempt["status"] == "error"
                assert finish["status"] == "completed"
                assert attempt["selected_final"] is False
                assert attempt["error_type"] == "ParserAbstention"
                clean_abstention_predecessors += 1
            attempt_reservations = reservations_by_attempt.get(attempt_id, [])
            assert len(attempt["calls"]) == len(attempt_reservations)
            for call, reservation in zip(
                attempt["calls"], attempt_reservations, strict=True
            ):
                assert set(call) == {
                    "accounting_basis",
                    "call_id",
                    "call_ordinal",
                    "input_tokens",
                    "kind",
                    "model_id",
                    "output_tokens",
                    "settled_cost_usd_exact",
                }
                call_id = call["call_id"]
                assert call_id not in seen_calls
                seen_calls.add(call_id)
                settlement = settlements[call_id]
                assert reservation["execution_id"] == execution_id
                assert reservation["attempt_ordinal"] == attempt["attempt_ordinal"]
                assert call["call_ordinal"] == reservation["call_ordinal"]
                assert call["kind"] == reservation["kind"] == settlement["kind"]
                assert call["model_id"] == PROVIDER_MODEL
                assert call["accounting_basis"] == settlement["accounting_basis"]
                assert call["settled_cost_usd_exact"] == settlement["settled_cost_usd"]
                assert call["input_tokens"] == settlement["provider_usage"]["input_tokens"]
                assert call["output_tokens"] == settlement["provider_usage"]["output_tokens"]

    assert len(execution_ids) == 33_361
    assert seen_attempts == set(starts)
    assert seen_calls == set(reservations)
    assert clean_abstention_predecessors == 22
    reader_execution_ids = {row["execution_identity"] for row in reader_rows}
    assert reader_rows == [
        row for row in all_rows if row["execution_identity"] in reader_execution_ids
    ]

    all_summary = _cost_summary(all_rows)
    reader_summary = _cost_summary(reader_rows)
    assert all_summary == {
        "accounted_cost_upper_usd_exact": "6.36571104",
        "attempts": 33_436,
        "conservative_call_count": 59,
        "conservative_reserved_cost_usd_exact": "0.10873968",
        "provider_input_tokens": 131_259_754,
        "provider_measured_call_count": 49_592,
        "provider_measured_cost_usd_exact": "6.25697136",
        "provider_output_tokens": 12_582_265,
        "rows": 33_361,
        "statements": 1_689,
    }
    assert reader_summary == {
        "accounted_cost_upper_usd_exact": "6.29731412",
        "attempts": 32_552,
        "conservative_call_count": 57,
        "conservative_reserved_cost_usd_exact": "0.10469996",
        "provider_input_tokens": 129_919_214,
        "provider_measured_call_count": 49_057,
        "provider_measured_cost_usd_exact": "6.19261416",
        "provider_output_tokens": 12_448_070,
        "rows": 32_479,
        "statements": 1_676,
    }
    for panel, summary in (
        (panels["paper_all_source"], all_summary),
        (panels["paper_readers"], reader_summary),
    ):
        accounting = panel["cost"]["accounting"]
        assert accounting["accounted_cost_lower_usd_exact"] == summary[
            "provider_measured_cost_usd_exact"
        ]
        assert accounting["accounted_cost_upper_usd_exact"] == summary[
            "accounted_cost_upper_usd_exact"
        ]
        assert accounting["provider_measured_cost_usd_exact"] == summary[
            "provider_measured_cost_usd_exact"
        ]
        assert accounting["conservative_reserved_cost_usd_exact"] == summary[
            "conservative_reserved_cost_usd_exact"
        ]
        assert accounting["provider_measured_call_count"] == summary[
            "provider_measured_call_count"
        ]
        assert accounting["conservative_call_count"] == summary[
            "conservative_call_count"
        ]
        assert accounting["denominator"] == {
            "evidence_executions": summary["rows"],
            "statements": summary["statements"],
        }
