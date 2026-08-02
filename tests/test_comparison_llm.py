from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from indra_belief.comparison import llm
from indra_belief.comparison.contracts import canonical_json_bytes, stable_read


RUN_ID = "paper_run__gemma_e2b__unique_exact_pairs_primary"
SERVED_MODEL = "bedrock-gemma-4-e2b"
PROVIDER_MODEL = "google.gemma-4-e2b"


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _ledger_bytes(payloads: list[dict[str, Any]]) -> bytes:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, payload in enumerate(payloads, start=1):
        row = {
            "ledger_id": "1" * 32,
            "sequence": sequence,
            "previous_event_sha256": previous,
            **payload,
        }
        encoded = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        row["event_sha256"] = hashlib.sha256(encoded).hexdigest()
        previous = row["event_sha256"]
        rows.append(row)
    return _jsonl(rows)


def _fixture(
    tmp_path: Path, *, final_error_pairs: frozenset[int] = frozenset()
) -> dict[str, Any]:
    """Bundle inputs for a completed run.

    `final_error_pairs` replaces the named pairs' attempts with a single FAILING
    one, so the pair ends carrying no verdict — the durable shape a quarantined
    source leaves behind.
    """
    statements = [
        {
            "id": "s0",
            "matches_hash": "statement-0",
            "evidence": [
                {"source_api": "reach", "source_hash": "r0", "text": "alpha"},
                {"source_api": "signor", "source_hash": "g0", "text": "beta"},
            ],
        },
        {
            "id": "s1",
            "matches_hash": "statement-1",
            "evidence": [
                {"source_api": "reach", "source_hash": "r1", "text": "gamma"}
            ],
        },
        {
            "id": "s2",
            "matches_hash": "statement-2",
            "evidence": [
                {"source_api": "signor", "source_hash": "g1", "text": "delta"}
            ],
        },
    ]
    map_rows: list[dict[str, Any]] = []
    pairs: list[tuple[int, int, dict[str, Any], str]] = []
    for stmt_i, statement in enumerate(statements):
        for evidence_i, evidence in enumerate(statement["evidence"]):
            mapped = {
                "new_stmt_i": stmt_i,
                "new_evidence_i": evidence_i,
                "eligible_position": stmt_i,
                "paper_statement_hash": statement["matches_hash"],
                "source_hash": evidence["source_hash"],
                "source_api": evidence["source_api"],
                "evidence_json_sha256": hashlib.sha256(
                    canonical_json_bytes(evidence)
                ).hexdigest(),
                "route": "plain",
                "relation_prompt_sha256": None,
            }
            identity = {
                "model": SERVED_MODEL,
                "workload_mode": "unique_exact_pairs_primary",
                "eligible_position": stmt_i,
                "paper_statement_hash": statement["matches_hash"],
                "source_hash": evidence["source_hash"],
                "evidence_json_sha256": mapped["evidence_json_sha256"],
            }
            execution_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
            map_rows.append(mapped)
            pairs.append((stmt_i, evidence_i, mapped, execution_id))

    # Pair 0 has one transport failure absent from historical raw output. Pair
    # 1 has a recorded parser abstention. Both costs must survive projection.
    topology = [
        [("error", "conservative"), ("completed", "provider")],
        [("completed", "provider"), ("completed", "provider")],
        [("completed", "provider")],
        [("completed", "provider")],
    ]
    # A quarantined pair's durable shape: it can never be "scored then failed",
    # because `replay._scan_resume` refuses to append anything after a terminal
    # scored row.
    for pair_index in sorted(final_error_pairs):
        topology[pair_index] = [("error", "conservative")]
    verdicts = ["correct", "correct", "incorrect", "correct"]
    ledger_payloads: list[dict[str, Any]] = [
        {"event": "ledger_initialized", "global_cap_usd": "10"},
        {
            "event": "stage_cap_set",
            "stage": "e2b",
            "model": SERVED_MODEL,
            "stage_cap_usd": "10",
        },
    ]
    raw_rows: list[dict[str, Any]] = []
    attempt_number = 0
    call_number = 100
    for pair_index, ((stmt_i, evidence_i, mapped, execution_id), attempts) in enumerate(
        zip(pairs, topology, strict=True)
    ):
        for ordinal, (status, basis) in enumerate(attempts, start=1):
            attempt_number += 1
            call_number += 1
            attempt_id = f"{attempt_number:032x}"
            call_id = f"{call_number:032x}"
            identity = {
                "model": SERVED_MODEL,
                "workload_mode": "unique_exact_pairs_primary",
                "eligible_position": stmt_i,
                "paper_statement_hash": mapped["paper_statement_hash"],
                "source_hash": mapped["source_hash"],
                "evidence_json_sha256": mapped["evidence_json_sha256"],
            }
            is_parser_abstention = pair_index == 1 and ordinal == 1
            row = None
            if status == "error" and pair_index in final_error_pairs:
                # The runner DOES persist a raw error row for a source it
                # retires; only pair 0's historical transport failure is absent
                # from the raw output.
                row = {
                    "stmt_i": stmt_i,
                    "evidence_i": evidence_i,
                    "run_id": RUN_ID,
                    "source_hash": mapped["source_hash"],
                    "paper_statement_hash": mapped["paper_statement_hash"],
                    "evidence_json_sha256": mapped["evidence_json_sha256"],
                    "source_api": mapped["source_api"],
                    "row_status": "error",
                    "verdict": None,
                    "confidence": None,
                    "tier": None,
                    "error": {"type": "InvalidModelOutput"},
                    "call_log": [
                        {
                            "kind": "monolithic",
                            "model_id": PROVIDER_MODEL,
                            "prompt_tokens": None,
                            "out_tokens": None,
                        }
                    ],
                }
            elif status != "error":
                row = {
                    "stmt_i": stmt_i,
                    "evidence_i": evidence_i,
                    "run_id": RUN_ID,
                    "source_hash": mapped["source_hash"],
                    "paper_statement_hash": mapped["paper_statement_hash"],
                    "evidence_json_sha256": mapped["evidence_json_sha256"],
                    "source_api": mapped["source_api"],
                    "row_status": "scored",
                    "verdict": None if is_parser_abstention else verdicts[pair_index],
                    "confidence": None if is_parser_abstention else "high",
                    "tier": "llm_direct",
                    "error": None,
                    "call_log": [
                        {
                            "kind": "monolithic",
                            "model_id": PROVIDER_MODEL,
                            "prompt_tokens": 100,
                            "out_tokens": 100,
                        }
                    ],
                }
                if pair_index == 2:
                    row.update(
                        attempt_id=attempt_id,
                        attempt_ordinal=ordinal,
                        execution_id=execution_id,
                    )
            outcome = row or {
                "row_status": "error",
                "error": {"type": "RuntimeError"},
                "call_log": [],
            }
            request_material = {"kind": "monolithic", "model": PROVIDER_MODEL}
            call_evidence = {"kind": "monolithic", "model_id": PROVIDER_MODEL}
            ledger_payloads.extend(
                [
                    {
                        "event": "attempt_started",
                        "run_id": RUN_ID,
                        "stage": "e2b",
                        "model": SERVED_MODEL,
                        "workload": "unique_exact_pairs_primary",
                        "attempt_id": attempt_id,
                        "attempt_ordinal": ordinal,
                        "execution_id": execution_id,
                        "execution_identity": identity,
                    },
                    {
                        "event": "call_reserved",
                        "run_id": RUN_ID,
                        "attempt_id": attempt_id,
                        "attempt_ordinal": ordinal,
                        "execution_id": execution_id,
                        "call_id": call_id,
                        "call_ordinal": 1,
                        "provider_model_id": PROVIDER_MODEL,
                        "kind": "monolithic",
                        "reserved_input_tokens": 100,
                        "reserved_output_tokens": 100,
                        "input_usd_per_million": "500",
                        "output_usd_per_million": "500",
                        "pricing_basis": "deterministic_test_tariff",
                        "reserved_max_cost_usd": "0.1",
                        "request_material": request_material,
                        "provider_request_sha256": hashlib.sha256(
                            canonical_json_bytes(request_material)
                        ).hexdigest(),
                    },
                    {
                        "event": "call_evidence_observed",
                        "run_id": RUN_ID,
                        "attempt_id": attempt_id,
                        "attempt_ordinal": ordinal,
                        "execution_id": execution_id,
                        "call_id": call_id,
                        "call_ordinal": 1,
                        "provider_model_id": PROVIDER_MODEL,
                        "kind": "monolithic",
                        "call_evidence": call_evidence,
                        "call_evidence_sha256": hashlib.sha256(
                            canonical_json_bytes(call_evidence)
                        ).hexdigest(),
                    },
                    {
                        "event": "call_settled",
                        "run_id": RUN_ID,
                        "attempt_id": attempt_id,
                        "attempt_ordinal": ordinal,
                        "execution_id": execution_id,
                        "call_id": call_id,
                        "call_ordinal": 1,
                        "provider_model_id": PROVIDER_MODEL,
                        "kind": "monolithic",
                        "accounting_basis": (
                            "provider_reported_usage"
                            if basis == "provider"
                            else "conservative_reserved_maximum"
                        ),
                        "provider_usage": (
                            {"input_tokens": 100, "output_tokens": 100}
                            if basis == "provider"
                            else {"input_tokens": None, "output_tokens": None}
                        ),
                        "settled_cost_usd": "0.1",
                        "reservation_breached": False,
                    },
                    {
                        "event": "attempt_outcome_committed",
                        "run_id": RUN_ID,
                        "attempt_id": attempt_id,
                        "attempt_ordinal": ordinal,
                        "execution_id": execution_id,
                        "status": status,
                        "raw_row": outcome,
                        "raw_row_sha256": hashlib.sha256(
                            canonical_json_bytes(outcome)
                        ).hexdigest(),
                    },
                    {
                        "event": "attempt_finished",
                        "run_id": RUN_ID,
                        "attempt_id": attempt_id,
                        "attempt_ordinal": ordinal,
                        "execution_id": execution_id,
                        "status": status,
                        "error_type": "RuntimeError" if status == "error" else None,
                    },
                ]
            )
            if row is not None:
                raw_rows.append(row)

    paths = {
        "aggregation": tmp_path / "aggregation.json",
        "raw_attempts": tmp_path / "raw.jsonl",
        "execution_map": tmp_path / "map.jsonl",
        "pricing": tmp_path / "pricing.json",
        "statements": tmp_path / "statements.json",
        "spend_ledger": tmp_path / "spend.ndjson",
        "output_dir": tmp_path / "bundle",
    }
    plan_path = tmp_path / "run_plan.json"
    plan_path.write_bytes(
        canonical_json_bytes(
            {
                "amendment": {
                    "changes": [
                        {
                            "from": 5,
                            "path": "/actions/0/max_attempts",
                            "to": 6,
                        }
                    ],
                    "predecessor_sha256": "a" * 64,
                },
                "kind": "indra_belief_comparison_run_plan",
            }
        )
        + b"\n"
    )
    paths["raw_attempts"].write_bytes(_jsonl(raw_rows))
    paths["execution_map"].write_bytes(_jsonl(map_rows))
    paths["statements"].write_bytes(canonical_json_bytes(statements))
    paths["spend_ledger"].write_bytes(_ledger_bytes(ledger_payloads))
    paths["aggregation"].write_bytes(
        canonical_json_bytes(
            {
                "aggregation": "test_hard_gate",
                "kind": "statement_belief_aggregation",
                "priors": {"reach": [0.3, 0.05], "signor": [0.049, 0.01]},
                "reader_profile": None,
            }
        )
    )
    paths["pricing"].write_bytes(
        canonical_json_bytes(
            {
                "cost_comparability_id": "fixture_provider_token_cost",
                "currency": "USD",
                "kind": "provider_token_pricing",
                "provider": "Fixture Provider",
                "pricing_mode": "on_demand",
                "region": "fixture-region",
                "resolved_service_tier": "standard",
                "retrieved_on": "2026-07-20",
                "service_tier_request": "default",
                "source_url": "https://example.test/pricing",
                "tariffs": {
                    PROVIDER_MODEL: {
                        "input_usd_per_million": "500",
                        "output_usd_per_million": "500",
                        "pricing_basis": "deterministic_test_tariff",
                    }
                },
                "unit": "per_million_tokens",
            }
        )
    )
    return {
        "run_plan": stable_read(plan_path, context="run plan"),
        **paths,
        "run_id": RUN_ID,
        "served_model": SERVED_MODEL,
        "model_id": "llm_gemma_4_e2b",
        "provider_model_id": PROVIDER_MODEL,
        "expected": llm.ExpectedCounts(3, 4, 2, 2),
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_materializes_true_reader_panel_and_retry_inclusive_shared_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path)
    linked: list[str] = []
    real_link = os.link

    def recording_link(src: Any, dst: Any, **kwargs: Any) -> None:
        linked.append(Path(dst).name)
        real_link(src, dst, **kwargs)

    monkeypatch.setattr(llm.os, "link", recording_link)
    manifest = llm.materialize_model_bundle(**arguments)

    assert linked[-1] == "manifest.json"
    assert json.loads((arguments["output_dir"] / "manifest.json").read_text()) == manifest
    all_predictions = _rows(arguments["output_dir"] / "all_source_predictions.jsonl")
    reader_predictions = _rows(arguments["output_dir"] / "reader_predictions.jsonl")
    assert [row["statement_id"] for row in all_predictions] == ["s0", "s1", "s2"]
    assert [row["statement_id"] for row in reader_predictions] == ["s0", "s1"]
    assert all_predictions[0]["probability_correct"] != reader_predictions[0]["probability_correct"]

    all_cost = manifest["panels"]["paper_all_source"]["cost"]
    reader_cost = manifest["panels"]["paper_readers"]["cost"]
    assert set(manifest) == {"kind", "model_id", "run_id", "implementation", "panels"}
    assert manifest["run_id"] == RUN_ID
    plan_descriptor = manifest["implementation"]["notes"]["inputs"]["run_plan"]
    assert plan_descriptor == {
        "bytes": len(arguments["run_plan"].payload),
        "path": os.path.relpath(
            arguments["run_plan"].path, arguments["output_dir"]
        ).replace(os.sep, "/"),
        "sha256": hashlib.sha256(arguments["run_plan"].payload).hexdigest(),
    }
    assert all_cost["accounting"]["accounted_cost_upper_usd_exact"] == "0.6"
    assert reader_cost["accounting"]["accounted_cost_upper_usd_exact"] == "0.3"
    assert all_cost["accounting"]["denominator"] == {
        "statements": 3,
        "evidence_executions": 4,
    }
    assert reader_cost["projection"] == "observed_execution_subset"
    assert all_cost["accounting"]["excluded_cost_categories"] == [
        "training",
        "local_aggregation",
        "feature_materialization",
        "upstream_reading",
    ]
    assert all_cost["record_type"] == "evidence_execution"
    assert reader_cost["record_type"] == "evidence_execution"
    attempts = _rows(arguments["output_dir"] / "all_source_attempts.jsonl")
    assert set(attempts[0]) == {
        "record_type",
        "statement_id",
        "execution_identity",
        "call_eligible",
        "attempts",
    }
    assert len(attempts[0]["attempts"]) == 2
    assert attempts[0]["attempts"][0]["error_type"] == "RuntimeError"
    assert attempts[1]["attempts"][0]["error_type"] == "ParserAbstention"
    assert attempts[1]["attempts"][0]["status"] == "error"
    assert attempts[1]["attempts"][1]["status"] == "completed"

    for descriptor in (
        manifest["panels"]["paper_all_source"]["predictions"],
        manifest["panels"]["paper_readers"]["predictions"],
        all_cost,
        reader_cost,
    ):
        target = arguments["output_dir"] / descriptor["path"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == descriptor["sha256"]

    with pytest.raises(FileExistsError, match="refusing to clobber"):
        llm.materialize_model_bundle(**arguments)


def test_a_quarantined_pair_can_never_be_bundled(tmp_path: Path) -> None:
    """The total-coverage gate, and why it must NOT be relaxed for quarantine.

    Pair (0, 1) ends on a failed attempt carrying no verdict — exactly what a
    quarantined source leaves on disk.  The bundle refuses, and that refusal is
    load-bearing rather than incidental.

    The tempting relaxation is to let the bundle publish while withholding the
    unscored pair's measurement from the belief fold.  It is tempting because
    NOTHING STRUCTURAL BREAKS: measured on this fixture, a bundle built that way
    keeps every census identical to a clean one — 3 statement predictions, 2
    reader predictions, 4 attempt rows, 2 reader attempt rows, the same
    `ExpectedCounts`, the same statement IDs — because predictions are emitted
    per STATEMENT and no statement disappears.  `assemble._prediction_rows`
    checks row count and statement coverage, and both still match.

    What moves is the number.  Statement s0's published belief went from
    0.97935 to 0.65 — a third of the scale — with every coverage check in the
    system still green and nothing downstream able to notice.  These arms are
    compared at the third decimal of AP and AUROC, so a panel folded from
    less evidence than it claims is not a slightly worse panel; it is a
    different measurement wearing the same shape.  The corpus is all-or-nothing
    by design, and this is the gate that enforces it.
    """
    arguments = _fixture(tmp_path, final_error_pairs=frozenset({1}))
    with pytest.raises(
        llm.LlmMaterializationError, match="lacks one final scored verdict"
    ):
        llm.materialize_model_bundle(**arguments)
    assert not (arguments["output_dir"] / "manifest.json").exists()


def test_rejects_partial_raw_without_publishing_manifest(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    raw = arguments["raw_attempts"].read_bytes().splitlines(keepends=True)
    arguments["raw_attempts"].write_bytes(b"".join(raw[:-1]))
    with pytest.raises(llm.LlmMaterializationError, match="partial"):
        llm.materialize_model_bundle(**arguments)
    assert not (arguments["output_dir"] / "manifest.json").exists()


def test_rejects_tampered_spend_chain(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    lines = arguments["spend_ledger"].read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[3])
    row["settled_cost_usd"] = "9"
    lines[3] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    arguments["spend_ledger"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(llm.LlmMaterializationError, match="hash chain"):
        llm.materialize_model_bundle(**arguments)
    assert not (arguments["output_dir"] / "manifest.json").exists()


def test_rejects_hash_valid_but_structurally_incomplete_spend_ledger(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    signed = _rows(arguments["spend_ledger"])
    removed = False
    payloads = []
    for row in signed:
        if row["event"] == "call_evidence_observed" and not removed:
            removed = True
            continue
        payloads.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "ledger_id",
                    "sequence",
                    "previous_event_sha256",
                    "event_sha256",
                }
            }
        )
    arguments["spend_ledger"].write_bytes(_ledger_bytes(payloads))

    with pytest.raises(llm.LlmMaterializationError, match="call settlement is malformed"):
        llm.materialize_model_bundle(**arguments)
    assert not (arguments["output_dir"] / "manifest.json").exists()


def test_input_captures_are_checked_once_at_manifest_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path)
    checked: list[Path] = []
    original = llm.FileCapture.assert_current

    def record(capture: llm.FileCapture) -> None:
        checked.append(capture.path)
        original(capture)

    monkeypatch.setattr(llm.FileCapture, "assert_current", record)
    llm.materialize_model_bundle(**arguments)

    expected = sorted(
        [arguments["run_plan"].path]
        + [
            Path(arguments[name]).resolve()
            for name in (
                "aggregation",
                "raw_attempts",
                "execution_map",
                "pricing",
                "statements",
                "spend_ledger",
            )
        ]
    )
    assert sorted(checked) == expected


def test_run_plan_tamper_aborts_manifest_last_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _fixture(tmp_path)
    capture = arguments["run_plan"]
    real_link = os.link
    tampered = False

    def tampering_link(src: Any, dst: Any, **kwargs: Any) -> None:
        nonlocal tampered
        real_link(src, dst, **kwargs)
        if not tampered and Path(dst).name != "manifest.json":
            capture.path.write_bytes(capture.payload + b" ")
            tampered = True

    monkeypatch.setattr(llm.os, "link", tampering_link)

    with pytest.raises(
        llm.LlmMaterializationError, match="committed file changed after validation"
    ):
        llm.materialize_model_bundle(**arguments)
    assert tampered
    assert not (arguments["output_dir"] / "manifest.json").exists()


def test_rejects_label_bearing_statement_input(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    statements = json.loads(arguments["statements"].read_text(encoding="utf-8"))
    statements[0]["gold_label"] = True
    arguments["statements"].write_bytes(canonical_json_bytes(statements))
    with pytest.raises(llm.LlmMaterializationError, match="forbidden released-label"):
        llm.materialize_model_bundle(**arguments)
    assert not (arguments["output_dir"] / "manifest.json").exists()


def test_rejects_pricing_config_that_differs_from_the_spend_ledger(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    pricing = json.loads(arguments["pricing"].read_text(encoding="utf-8"))
    pricing["tariffs"][PROVIDER_MODEL]["output_usd_per_million"] = "499"
    arguments["pricing"].write_bytes(canonical_json_bytes(pricing))

    with pytest.raises(llm.LlmMaterializationError, match="tariff differs"):
        llm.materialize_model_bundle(**arguments)
    assert not (arguments["output_dir"] / "manifest.json").exists()


def test_rejects_noncanonical_aggregation_contract(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    aggregation = json.loads(arguments["aggregation"].read_text(encoding="utf-8"))
    aggregation["unused_profile_label"] = "obsolete"
    arguments["aggregation"].write_bytes(canonical_json_bytes(aggregation))

    with pytest.raises(llm.LlmMaterializationError, match="aggregation config fields"):
        llm.materialize_model_bundle(**arguments)
    assert not (arguments["output_dir"] / "manifest.json").exists()
