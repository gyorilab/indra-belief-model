"""Materialize one completed LLM run as an auditable two-panel model bundle.

The materializer is deliberately downstream-only: it reads the authenticated
run-plan capture, completed raw rows, their exact-pair execution map, the
label-free statement corpus, and the shared spend ledger.  It performs no
inference.  The all-source and five-reader panels are independently aggregated
with :func:`statement_belief`; the reader panel is never a subset of
already-aggregated statement predictions.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import tempfile
from typing import Any, Mapping, NoReturn, Sequence

from indra_belief import noise_model as _noise_model
from indra_belief.comparison.contracts import (
    ContractError,
    FileCapture,
    canonical_json_bytes,
    stable_read,
    strict_json_loads,
)
from indra_belief.spend_guard import (
    SpendLedgerCorrupt,
    SpendLedgerSnapshot,
    parse_spend_ledger,
)
from indra_belief.scorers import _shared as _shared_text
from indra_belief import statement_belief as _statement_module


READER_SOURCES = frozenset({"reach", "sparser", "medscan", "rlimsp", "trips"})
CALLABLE_ROUTES = frozenset({"plain", "tool"})
EVIDENCE_EXECUTION_RECORD = "evidence_execution"
COST_VIEW_ID = "provider-runtime-retry-inclusive"
HEX32 = re.compile(r"^[0-9a-f]{32}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
COMPARABILITY_ID = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "adjudication",
        "correctness",
        "curation",
        "curations",
        "gold",
        "goldlabel",
        "groundtruth",
        "iscorrect",
        "label",
        "labels",
        "releasedlabel",
    }
)


class LlmMaterializationError(RuntimeError):
    """The completed run, its accounting, or bundle publication is invalid."""


def _fail(message: str) -> NoReturn:
    raise LlmMaterializationError(message)


@dataclass(frozen=True)
class ExpectedCounts:
    statements: int = 1_689
    executions: int = 33_361
    reader_statements: int = 1_676
    reader_executions: int = 32_479


@dataclass(frozen=True)
class _Pair:
    key: tuple[int, int]
    map_row: dict[str, Any]
    statement: dict[str, Any]
    evidence: dict[str, Any]
    statement_id: str
    source: str
    execution_id: str


@dataclass(frozen=True)
class _Attempt:
    start: dict[str, Any]
    finish: dict[str, Any]
    reservations: tuple[dict[str, Any], ...]
    settlements: tuple[dict[str, Any], ...]


def _canonical(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            return (
                json.dumps(
                    value,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
                + b"\n"
            )
        return canonical_json_bytes(value)
    except (ContractError, TypeError, ValueError, UnicodeEncodeError) as exc:
        raise LlmMaterializationError("value is not canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _capture(path: Path, *, label: str) -> FileCapture:
    try:
        return stable_read(path, context=label)
    except ContractError as exc:
        raise LlmMaterializationError(str(exc)) from exc


def _stable_read(path: Path, *, label: str) -> bytes:
    return _capture(path, label=label).payload


def _json(payload: bytes, *, label: str) -> Any:
    try:
        return strict_json_loads(payload, context=label)
    except ContractError as exc:
        raise LlmMaterializationError(f"{label} is not JSON") from exc


def _jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        _fail(f"{label} is empty or lacks a terminal LF")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            row = strict_json_loads(line, context=f"{label} line {line_number}")
        except ContractError as exc:
            raise LlmMaterializationError(
                f"{label} line {line_number} is not JSON"
            ) from exc
        if not isinstance(row, dict):
            _fail(f"{label} line {line_number} is not an object")
        rows.append(row)
    return rows


def _reject_labels(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized in FORBIDDEN_LABEL_KEYS:
                _fail(f"{label} contains forbidden released-label key {key!r}")
            _reject_labels(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _reject_labels(child, label=label)


def _decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool):
        _fail(f"{label} is not an exact decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise LlmMaterializationError(f"{label} is not an exact decimal") from exc
    if not result.is_finite() or result < 0:
        _fail(f"{label} is not finite and non-negative")
    return result


def _money(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be nonempty text")
    return value


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve()).replace(os.sep, "/")


def _descriptor(
    path: Path, payload: bytes, *, manifest_dir: Path, rows: int | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _relative(path, manifest_dir),
        "sha256": _sha256(payload),
        "bytes": len(payload),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _execution_identity(
    map_row: Mapping[str, Any], *, served_model: str, workload: str
) -> tuple[dict[str, Any], str]:
    identity = {
        "model": served_model,
        "workload_mode": workload,
        "eligible_position": int(map_row["eligible_position"]),
        "paper_statement_hash": str(map_row["paper_statement_hash"]),
        "source_hash": str(map_row["source_hash"]),
        "evidence_json_sha256": str(map_row["evidence_json_sha256"]),
    }
    return identity, _sha256(_canonical(identity))


def _validate_priors(
    raw: Mapping[str, Sequence[float]], sources: set[str]
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for source, pair in raw.items():
        if (
            not isinstance(source, str)
            or not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
                for value in pair
            )
        ):
            _fail("belief priors are malformed")
        result[source.casefold()] = (float(pair[0]), float(pair[1]))
    missing = sources - set(result)
    if missing:
        _fail(f"belief priors omit observed sources: {sorted(missing)!r}")
    return result


def _aggregation_config(value: Any) -> tuple[
    Mapping[str, Sequence[float]], Mapping[str, Any] | None, str
]:
    if not isinstance(value, dict) or set(value) != {
        "aggregation",
        "kind",
        "priors",
        "reader_profile",
    }:
        _fail("aggregation config fields differ from the canonical contract")
    if value.get("kind") != "statement_belief_aggregation":
        _fail("aggregation config kind differs")
    name = _text(value.get("aggregation"), label="aggregation name")
    priors = value.get("priors")
    profile = value.get("reader_profile")
    if not isinstance(priors, dict) or (
        profile is not None and not isinstance(profile, dict)
    ):
        _fail("aggregation priors or reader profile are malformed")
    return priors, profile, name


def _pricing_contract(
    value: Any,
    *,
    provider_model_id: str,
    attempts: Mapping[str, Sequence[_Attempt]],
) -> dict[str, Any]:
    required = {
        "cost_comparability_id",
        "currency",
        "kind",
        "provider",
        "pricing_mode",
        "region",
        "resolved_service_tier",
        "retrieved_on",
        "service_tier_request",
        "source_url",
        "tariffs",
        "unit",
    }
    if not isinstance(value, dict) or set(value) != required:
        _fail("pricing config fields differ from the canonical contract")
    if value.get("kind") != "provider_token_pricing":
        _fail("pricing config kind differs")
    provider = _text(value.get("provider"), label="pricing provider")
    pricing_mode = _text(value.get("pricing_mode"), label="pricing mode")
    region = _text(value.get("region"), label="pricing region")
    service_tier_request = _text(
        value.get("service_tier_request"), label="requested service tier"
    )
    resolved_service_tier = _text(
        value.get("resolved_service_tier"), label="resolved service tier"
    )
    currency = _text(value.get("currency"), label="pricing currency")
    unit = _text(value.get("unit"), label="pricing unit")
    source_url = _text(value.get("source_url"), label="pricing source URL")
    retrieved_on = _text(value.get("retrieved_on"), label="pricing retrieval date")
    comparability_id = _text(
        value.get("cost_comparability_id"), label="cost comparability ID"
    )
    if (
        currency != "USD"
        or unit != "per_million_tokens"
        or pricing_mode != "on_demand"
        or service_tier_request != "default"
        or resolved_service_tier != "standard"
        or not source_url.startswith("https://")
        or DATE.fullmatch(retrieved_on) is None
        or COMPARABILITY_ID.fullmatch(comparability_id) is None
    ):
        _fail("pricing units, source, date, or comparability ID are invalid")
    tariffs = value.get("tariffs")
    if not isinstance(tariffs, dict) or provider_model_id not in tariffs:
        _fail("pricing config omits the provider model")
    tariff = tariffs[provider_model_id]
    if not isinstance(tariff, dict) or set(tariff) != {
        "input_usd_per_million",
        "output_usd_per_million",
        "pricing_basis",
    }:
        _fail("provider tariff fields differ")
    input_rate = _decimal(
        tariff.get("input_usd_per_million"), label="configured input tariff"
    )
    output_rate = _decimal(
        tariff.get("output_usd_per_million"), label="configured output tariff"
    )
    basis = _text(tariff.get("pricing_basis"), label="configured pricing basis")
    if (
        input_rate < 0
        or output_rate < 0
        or tariff["input_usd_per_million"] != _money(input_rate)
        or tariff["output_usd_per_million"] != _money(output_rate)
    ):
        _fail("provider tariff must use canonical nonnegative decimal strings")
    observed = {
        (
            reservation.get("provider_model_id"),
            reservation.get("input_usd_per_million"),
            reservation.get("output_usd_per_million"),
            reservation.get("pricing_basis"),
        )
        for rows in attempts.values()
        for attempt in rows
        for reservation in attempt.reservations
    }
    expected = {
        (
            provider_model_id,
            _money(input_rate),
            _money(output_rate),
            basis,
        )
    }
    if observed != expected:
        _fail("spend ledger tariff differs from the frozen pricing config")
    return {
        "cost_comparability_id": comparability_id,
        "currency": currency,
        "provider": provider,
        "provider_model_id": provider_model_id,
        "pricing_mode": pricing_mode,
        "region": region,
        "retrieved_on": retrieved_on,
        "resolved_service_tier": resolved_service_tier,
        "service_tier_request": service_tier_request,
        "source_url": source_url,
        "tariff": {
            "input_usd_per_million": _money(input_rate),
            "output_usd_per_million": _money(output_rate),
            "pricing_basis": basis,
        },
        "unit": unit,
    }


def _load_pairs(
    *,
    statements: Sequence[Any],
    map_rows: Sequence[Mapping[str, Any]],
    served_model: str,
    workload: str,
    expected: ExpectedCounts,
) -> tuple[list[_Pair], list[str]]:
    if len(statements) != expected.statements or len(map_rows) != expected.executions:
        _fail("statement/execution-map census differs from the declared substrate")
    statement_ids: list[str] = []
    expected_keys: set[tuple[int, int]] = set()
    for stmt_i, raw_statement in enumerate(statements):
        if not isinstance(raw_statement, dict):
            _fail(f"statement {stmt_i} is not an object")
        _reject_labels(raw_statement, label=f"statement {stmt_i}")
        statement_id = raw_statement.get("id")
        evidence = raw_statement.get("evidence")
        if not isinstance(statement_id, str) or not statement_id or not isinstance(evidence, list):
            _fail(f"statement {stmt_i} lacks ID/evidence")
        statement_ids.append(statement_id)
        expected_keys.update((stmt_i, evidence_i) for evidence_i in range(len(evidence)))
    if len(set(statement_ids)) != len(statement_ids):
        _fail("statement IDs repeat")
    if len(expected_keys) != expected.executions:
        _fail("statement corpus evidence count differs from the execution census")

    pairs: list[_Pair] = []
    seen: set[tuple[int, int]] = set()
    for index, raw_map in enumerate(map_rows):
        if not isinstance(raw_map, Mapping):
            _fail(f"execution map row {index} is not an object")
        _reject_labels(raw_map, label=f"execution map row {index}")
        stmt_i = raw_map.get("new_stmt_i")
        evidence_i = raw_map.get("new_evidence_i")
        if (
            isinstance(stmt_i, bool)
            or not isinstance(stmt_i, int)
            or isinstance(evidence_i, bool)
            or not isinstance(evidence_i, int)
            or raw_map.get("eligible_position") != stmt_i
            or (stmt_i, evidence_i) not in expected_keys
            or (stmt_i, evidence_i) in seen
        ):
            _fail(f"execution map row {index} has a foreign or duplicate key")
        seen.add((stmt_i, evidence_i))
        statement = statements[stmt_i]
        evidence = statement["evidence"][evidence_i]
        if not isinstance(evidence, dict):
            _fail(f"evidence {stmt_i}/{evidence_i} is not an object")
        evidence_digest = _sha256(_canonical(evidence))
        source = str(evidence.get("source_api") or "").casefold()
        if (
            not source
            or str(raw_map.get("source_api") or "").casefold() != source
            or str(raw_map.get("source_hash")) != str(evidence.get("source_hash"))
            or str(raw_map.get("paper_statement_hash"))
            != str(statement.get("matches_hash"))
            or raw_map.get("evidence_json_sha256") != evidence_digest
            or raw_map.get("route")
            not in {"plain", "tool", "deterministic_mismatch", "deterministic_pseudogene", "no_text"}
        ):
            _fail(f"execution map row {index} differs from its statement evidence")
        _, execution_id = _execution_identity(
            raw_map, served_model=served_model, workload=workload
        )
        pairs.append(
            _Pair(
                key=(stmt_i, evidence_i),
                map_row=dict(raw_map),
                statement=dict(statement),
                evidence=dict(evidence),
                statement_id=statement_ids[stmt_i],
                source=source,
                execution_id=execution_id,
            )
        )
    if seen != expected_keys:
        _fail("execution map does not cover the exact statement-evidence universe")
    pairs.sort(key=lambda pair: pair.key)
    return pairs, statement_ids


def _attempts_from_ledger(
    *,
    ledger: SpendLedgerSnapshot,
    pairs: Sequence[_Pair],
    run_id: str,
    served_model: str,
    provider_model_id: str,
    workload: str,
) -> dict[str, tuple[_Attempt, ...]]:
    events = ledger.events
    if not any(
        event.get("event") == "attempt_started" and event.get("run_id") == run_id
        for event in events
    ):
        _fail("spend ledger has no events for the completed run")
    pairs_by_execution = {pair.execution_id: pair for pair in pairs}
    starts: dict[str, dict[str, Any]] = {}
    finishes: dict[str, dict[str, Any]] = {}
    reservations: dict[str, dict[str, Any]] = {}
    settlements: dict[str, dict[str, Any]] = {}

    for raw_event in events:
        if raw_event.get("event") != "attempt_started" or raw_event.get("run_id") != run_id:
            continue
        event = dict(raw_event)
        attempt_id = event.get("attempt_id")
        execution_id = event.get("execution_id")
        if (
            not isinstance(attempt_id, str)
            or HEX32.fullmatch(attempt_id) is None
            or attempt_id in starts
            or execution_id not in pairs_by_execution
            or event.get("model") != served_model
            or event.get("workload") != workload
        ):
            _fail("spend ledger has a foreign or duplicate attempt start")
        pair = pairs_by_execution[str(execution_id)]
        identity, expected_id = _execution_identity(
            pair.map_row, served_model=served_model, workload=workload
        )
        if expected_id != execution_id or event.get("execution_identity") != identity:
            _fail("spend attempt identity differs from the execution map")
        starts[attempt_id] = event

    selected_attempt_ids = set(starts)
    for raw_event in events:
        event = dict(raw_event)
        kind = event.get("event")
        attempt_id = str(event.get("attempt_id") or "")
        if (
            kind == "attempt_finished"
            and event.get("run_id") == run_id
            and attempt_id in selected_attempt_ids
        ):
            if attempt_id in finishes:
                _fail("spend ledger repeats an attempt finish")
            finishes[attempt_id] = event
        elif kind == "call_reserved" and attempt_id in selected_attempt_ids:
            call_id = str(event.get("call_id") or "")
            if HEX32.fullmatch(call_id) is None or call_id in reservations:
                _fail("spend ledger repeats or malforms a call reservation")
            reservations[call_id] = event
    selected_call_ids = set(reservations)
    for raw_event in events:
        event = dict(raw_event)
        kind = event.get("event")
        call_id = str(event.get("call_id") or "")
        if kind == "call_settled" and call_id in selected_call_ids:
            if call_id in settlements:
                _fail("spend ledger repeats a call settlement")
            settlements[call_id] = event
    if set(finishes) != set(starts):
        _fail("completed run has open or foreign attempt finishes")
    if set(reservations) != set(settlements):
        _fail("completed run call reservations and settlements are not bijective")

    reservations_by_attempt: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for reservation in reservations.values():
        reservations_by_attempt[str(reservation.get("attempt_id") or "")].append(
            reservation
        )
    by_execution: defaultdict[str, list[_Attempt]] = defaultdict(list)
    for attempt_id, start in starts.items():
        execution_id = str(start["execution_id"])
        finish = finishes[attempt_id]
        if (
            finish.get("execution_id") != execution_id
            or finish.get("run_id") != run_id
            or finish.get("attempt_ordinal") != start.get("attempt_ordinal")
            or finish.get("attempt_id") != attempt_id
        ):
            _fail("attempt finish differs from its start")
        attempt_reservations = sorted(
            reservations_by_attempt.get(attempt_id, []),
            key=lambda row: int(row.get("call_ordinal", -1)),
        )
        attempt_settlements: list[dict[str, Any]] = []
        for call_index, reservation in enumerate(attempt_reservations, start=1):
            call_id = str(reservation.get("call_id") or "")
            settlement = settlements[call_id]
            if (
                reservation.get("run_id") != run_id
                or reservation.get("execution_id") != execution_id
                or reservation.get("attempt_ordinal") != start.get("attempt_ordinal")
                or reservation.get("call_ordinal") != call_index
                or reservation.get("provider_model_id") != provider_model_id
                or settlement.get("run_id") != run_id
                or settlement.get("execution_id") != execution_id
                or settlement.get("attempt_id") != attempt_id
                or settlement.get("call_ordinal") != call_index
                or settlement.get("kind") != reservation.get("kind")
                or settlement.get("provider_model_id") != provider_model_id
            ):
                _fail("call settlement differs from its exact reservation identity")
            basis = settlement.get("accounting_basis")
            usage = settlement.get("provider_usage")
            cost = _decimal(settlement.get("settled_cost_usd"), label="settled cost")
            if not isinstance(usage, Mapping):
                _fail("call settlement lacks provider usage")
            token_values = (usage.get("input_tokens"), usage.get("output_tokens"))
            known = all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in token_values
            )
            if basis == "provider_reported_usage":
                if not known:
                    _fail("provider-measured call lacks token usage")
            elif basis == "conservative_reserved_maximum":
                if any(value is not None for value in token_values) or cost != _decimal(
                    reservation.get("reserved_max_cost_usd"), label="reserved maximum"
                ):
                    _fail("conservative settlement differs from its reservation")
            else:
                _fail("call settlement has an unsupported accounting basis")
            attempt_settlements.append(settlement)
        by_execution[execution_id].append(
            _Attempt(
                start=start,
                finish=finish,
                reservations=tuple(attempt_reservations),
                settlements=tuple(attempt_settlements),
            )
        )
    if set(by_execution) != set(pairs_by_execution):
        _fail("spend ledger does not cover every exact execution identity")
    result: dict[str, tuple[_Attempt, ...]] = {}
    for execution_id, values in by_execution.items():
        values.sort(key=lambda value: int(value.start.get("attempt_ordinal", -1)))
        ordinals = [value.start.get("attempt_ordinal") for value in values]
        if not values or ordinals != list(range(1, len(values) + 1)):
            _fail("spend ledger violates contiguous attempt ordinals")
        result[execution_id] = tuple(values)
    return result


def _validate_raw(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[_Pair],
    attempts: Mapping[str, tuple[_Attempt, ...]],
    run_id: str,
    provider_model_id: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    pair_by_key = {pair.key: pair for pair in pairs}
    grouped: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            _fail(f"raw row {index} is not an object")
        stmt_i, evidence_i = raw.get("stmt_i"), raw.get("evidence_i")
        key = (stmt_i, evidence_i)
        pair = pair_by_key.get(key)
        if (
            pair is None
            or raw.get("run_id") != run_id
            or str(raw.get("source_hash")) != str(pair.map_row["source_hash"])
            or str(raw.get("paper_statement_hash"))
            != str(pair.map_row["paper_statement_hash"])
            or raw.get("evidence_json_sha256") != pair.map_row["evidence_json_sha256"]
            or str(raw.get("source_api") or "").casefold() != pair.source
            or raw.get("row_status") not in {"scored", "error"}
        ):
            _fail(f"raw row {index} has a foreign or malformed pair identity")
        grouped[key].append(dict(raw))
    if set(grouped) != set(pair_by_key):
        _fail("raw output is partial relative to the exact execution map")

    for key, rows in grouped.items():
        pair = pair_by_key[key]
        ledger_attempts = attempts[pair.execution_id]
        if not rows or len(rows) > len(ledger_attempts):
            _fail(f"raw rows exceed authenticated WAL attempts for pair {key}")
        final = rows[-1]
        if final.get("row_status") != "scored" or final.get("verdict") not in {
            "correct",
            "incorrect",
        }:
            _fail(f"pair {key} lacks one final scored verdict")
        if any(
            row.get("row_status") != "error" and row.get("verdict") is not None
            for row in rows[:-1]
        ):
            _fail(f"pair {key} retry predecessor is not an error/abstention")

        expected_kinds: list[str] = []
        if pair.map_row.get("route") in CALLABLE_ROUTES:
            main = (
                "monolithic_tool_context"
                if pair.map_row.get("route") == "tool"
                else "monolithic"
            )
            expected_kinds = (
                ["relation_nature", main]
                if pair.map_row.get("relation_prompt_sha256") is not None
                else [main]
            )

        def raw_matches(row: Mapping[str, Any], attempt: _Attempt) -> bool:
            raw_calls = row.get("call_log")
            if not isinstance(raw_calls, list) or len(raw_calls) != len(attempt.reservations):
                return False
            for call_index, (raw_call, reservation, settlement) in enumerate(
                zip(raw_calls, attempt.reservations, attempt.settlements, strict=True),
                start=1,
            ):
                if not isinstance(raw_call, Mapping) or (
                    raw_call.get("kind") != reservation.get("kind")
                    or raw_call.get("model_id") != provider_model_id
                    or ("call_id" in raw_call and raw_call["call_id"] != reservation["call_id"])
                    or ("call_ordinal" in raw_call and raw_call["call_ordinal"] != call_index)
                ):
                    return False
                if settlement.get("accounting_basis") == "provider_reported_usage":
                    usage = settlement["provider_usage"]
                    if (
                        raw_call.get("prompt_tokens") != usage.get("input_tokens")
                        or raw_call.get("out_tokens") != usage.get("output_tokens")
                    ):
                        return False
            return True

        assigned: list[int] = []
        cursor = 0
        for row_index, row in enumerate(rows[:-1]):
            explicit_id = row.get("attempt_id")
            remaining = len(rows) - row_index - 1
            stop = len(ledger_attempts) - remaining
            candidates = [
                index
                for index in range(cursor, stop)
                if (explicit_id is None or ledger_attempts[index].start["attempt_id"] == explicit_id)
                and raw_matches(row, ledger_attempts[index])
            ]
            if not candidates:
                _fail(f"raw retry row {key}/{row_index + 1} has no WAL call match")
            assigned.append(candidates[0])
            cursor = candidates[0] + 1
        final_index = len(ledger_attempts) - 1
        if final.get("attempt_id") not in {None, ledger_attempts[final_index].start["attempt_id"]}:
            _fail(f"raw final row for pair {key} names a nonterminal attempt")
        if not raw_matches(final, ledger_attempts[final_index]):
            _fail(f"raw final row for pair {key} differs from terminal WAL calls")
        assigned.append(final_index)

        matched = set(assigned)
        projected: list[dict[str, Any]] = []
        for row_position, (row, attempt_index) in enumerate(
            zip(rows, assigned, strict=True), start=1
        ):
            attempt = ledger_attempts[attempt_index]
            attempt_id = str(attempt.start["attempt_id"])
            for field, expected_value in (
                ("attempt_id", attempt_id),
                ("attempt_ordinal", attempt_index + 1),
                ("execution_id", pair.execution_id),
            ):
                if field in row and row[field] != expected_value:
                    _fail(f"raw row {key}/{row_position} differs at {field}")
            expected_status = "completed" if row.get("row_status") == "scored" else "error"
            if attempt.finish.get("status") != expected_status:
                _fail(f"raw row {key}/{row_position} differs from WAL completion")
            raw_calls = row.get("call_log")
            observed_kinds = [str(call.get("kind")) for call in raw_calls]
            if row_position == len(rows):
                if observed_kinds != expected_kinds:
                    _fail(f"final provider-call topology differs for pair {key}")
            elif observed_kinds != expected_kinds[: len(observed_kinds)]:
                _fail(f"retry provider-call prefix differs for pair {key}")
            projected.append(
                {
                    "row_status": row.get("row_status"),
                    "verdict": row.get("verdict"),
                    "confidence": row.get("confidence"),
                    "tier": row.get("tier"),
                    "error": row.get("error"),
                    "_attempt_ordinal": attempt_index + 1,
                }
            )
        for attempt_index, attempt in enumerate(ledger_attempts):
            observed_kinds = [str(row.get("kind")) for row in attempt.reservations]
            if attempt_index == final_index:
                if observed_kinds != expected_kinds:
                    _fail(f"terminal WAL call topology differs for pair {key}")
            elif observed_kinds != expected_kinds[: len(observed_kinds)]:
                _fail(f"retry WAL call topology differs for pair {key}")
            if attempt_index not in matched and attempt.finish.get("status") != "error":
                _fail(f"pair {key} has an unrecorded completed WAL attempt")
        grouped[key] = projected
    return dict(grouped)


def _cost_rows(
    *,
    pairs: Sequence[_Pair],
    raw_by_key: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    attempts: Mapping[str, tuple[_Attempt, ...]],
    provider_model_id: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pair in pairs:
        projected_attempts: list[dict[str, Any]] = []
        raw_by_ordinal = {
            int(row["_attempt_ordinal"]): row for row in raw_by_key[pair.key]
        }
        ledger_attempts = attempts[pair.execution_id]
        for index, attempt in enumerate(ledger_attempts, start=1):
            raw = raw_by_ordinal.get(index)
            calls: list[dict[str, Any]] = []
            for call_index, (reservation, settlement) in enumerate(
                zip(attempt.reservations, attempt.settlements, strict=True), start=1
            ):
                usage = settlement["provider_usage"]
                calls.append(
                    {
                        "accounting_basis": settlement["accounting_basis"],
                        "call_id": reservation["call_id"],
                        "call_ordinal": call_index,
                        "input_tokens": usage.get("input_tokens"),
                        "kind": reservation["kind"],
                        "model_id": provider_model_id,
                        "output_tokens": usage.get("output_tokens"),
                        "settled_cost_usd_exact": _money(
                            _decimal(
                                settlement["settled_cost_usd"], label="settled call cost"
                            )
                        ),
                    }
                )
            selected = index == len(ledger_attempts)
            error = raw.get("error") if raw is not None else None
            error_type = None
            if not selected:
                error_type = (
                    str(error.get("type"))
                    if isinstance(error, Mapping) and error.get("type")
                    else "ParserAbstention"
                    if raw is not None and raw.get("row_status") == "scored" and raw.get("verdict") is None
                    else str(attempt.finish.get("error_type") or "RetryPredecessor")
                )
            projected_attempts.append(
                {
                    "attempt_id": attempt.start["attempt_id"],
                    "attempt_ordinal": index,
                    "calls": calls,
                    "error_type": error_type,
                    "selected_final": selected,
                    # This status describes selection semantics, not merely the
                    # runner's transport lifecycle. A successfully parsed
                    # abstention can finish its WAL attempt cleanly yet still be
                    # a failed retry predecessor for this projection.
                    "status": "completed" if selected else "error",
                }
            )
        result.append(
            {
                "attempts": projected_attempts,
                "call_eligible": pair.map_row["route"] in CALLABLE_ROUTES,
                "execution_identity": pair.execution_id,
                "record_type": EVIDENCE_EXECUTION_RECORD,
                "statement_id": pair.statement_id,
            }
        )
    return result


def _cost_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    provider = Decimal("0")
    conservative = Decimal("0")
    provider_calls = conservative_calls = input_tokens = output_tokens = attempts = 0
    statements: set[str] = set()
    for row in rows:
        statements.add(str(row["statement_id"]))
        row_attempts = row["attempts"]
        attempts += len(row_attempts)
        for attempt in row_attempts:
            for call in attempt["calls"]:
                cost = _decimal(call["settled_cost_usd_exact"], label="projected cost")
                if call["accounting_basis"] == "provider_reported_usage":
                    provider += cost
                    provider_calls += 1
                    input_tokens += int(call["input_tokens"])
                    output_tokens += int(call["output_tokens"])
                else:
                    conservative += cost
                    conservative_calls += 1
    return {
        "execution_count": len(rows),
        "statement_count": len(statements),
        "attempt_count": attempts,
        "retry_attempt_count": attempts - len(rows),
        "provider_measured_call_count": provider_calls,
        "conservative_call_count": conservative_calls,
        "provider_input_tokens": input_tokens,
        "provider_output_tokens": output_tokens,
        "provider_measured_cost_usd_exact": _money(provider),
        "conservative_reserved_cost_usd_exact": _money(conservative),
        "accounted_cost_upper_usd_exact": _money(provider + conservative),
    }


def _cost_descriptor(
    *,
    path: Path,
    payload: bytes,
    manifest_dir: Path,
    summary: Mapping[str, Any],
    pricing: Mapping[str, Any],
    shared_run_id: str,
    projection: str,
) -> dict[str, Any]:
    provider = str(summary["provider_measured_cost_usd_exact"])
    conservative = str(summary["conservative_reserved_cost_usd_exact"])
    return {
        **_descriptor(path, payload, manifest_dir=manifest_dir, rows=int(summary["execution_count"])),
        "record_type": EVIDENCE_EXECUTION_RECORD,
        "status": "ledger",
        "basis": (
            "mixed_conservative_upper_bound"
            if int(summary["conservative_call_count"])
            else "provider_measured_observed"
        ),
        "view_id": COST_VIEW_ID,
        "price_source": str(pricing["source_url"]),
        "price_date": str(pricing["retrieved_on"]),
        "cost_comparability_id": str(pricing["cost_comparability_id"]),
        "pricing": dict(pricing),
        "projection": projection,
        "counterfactual_run_cost": False,
        "shared_run_id": shared_run_id,
        "additive_across_panels": False,
        "accounting": {
            "accounted_cost_lower_usd_exact": provider,
            "accounted_cost_upper_usd_exact": str(
                summary["accounted_cost_upper_usd_exact"]
            ),
            "provider_measured_cost_usd_exact": provider,
            "conservative_reserved_cost_usd_exact": conservative,
            "provider_measured_call_count": int(
                summary["provider_measured_call_count"]
            ),
            "conservative_call_count": int(summary["conservative_call_count"]),
            "includes_retries": True,
            "includes_relation_subcalls": True,
            "excluded_cost_categories": [
                "training",
                "local_aggregation",
                "feature_materialization",
                "upstream_reading",
            ],
            "denominator": {
                "statements": int(summary["statement_count"]),
                "evidence_executions": int(summary["execution_count"]),
            },
        },
    }


def _predictions(
    *,
    pairs: Sequence[_Pair],
    raw_by_key: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    statement_ids: Sequence[str],
    priors: Mapping[str, tuple[float, float]],
    reader_profile: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    all_evidence: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    reader_evidence: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    reader_execution_ids: set[str] = set()
    for pair in pairs:
        final = raw_by_key[pair.key][-1]
        measurement = {
            "source_api": pair.source,
            "verdict": final.get("verdict"),
            "confidence": final.get("confidence"),
            "tier": final.get("tier"),
            "evidence_text": pair.evidence.get("text"),
            "evidence_hash": pair.map_row["evidence_json_sha256"],
        }
        all_evidence[pair.key[0]].append(measurement)
        if pair.source in READER_SOURCES:
            reader_evidence[pair.key[0]].append(measurement)
            reader_execution_ids.add(pair.execution_id)

    def score(position: int, evidence: list[dict[str, Any]], panel: str) -> dict[str, Any]:
        result = _statement_module.statement_belief(
            evidence,
            priors=dict(priors),
            dedup=True,
            soft=dict(reader_profile) if reader_profile is not None else None,
        )
        belief = result.belief
        if (
            belief is None
            or not math.isfinite(float(belief))
            or not 0 <= float(belief) <= 1
        ):
            _fail(f"{panel} statement {position} has no finite probability")
        return {
            "probability_correct": float(belief),
            "statement_id": statement_ids[position],
        }

    all_rows = [
        score(position, all_evidence[position], "all-source")
        for position in range(len(statement_ids))
    ]
    reader_rows = [
        score(position, reader_evidence[position], "five-reader")
        for position in range(len(statement_ids))
        if reader_evidence.get(position)
    ]
    return all_rows, reader_rows, reader_execution_ids


def _implementation_digest() -> tuple[str, dict[str, str]]:
    components: dict[str, str] = {}
    for name, module in (
        ("statement_belief", _statement_module),
        ("noise_model", _noise_model),
        ("shared_text_normalization", _shared_text),
    ):
        path = Path(str(module.__file__)).resolve()
        components[name] = _sha256(_stable_read(path, label=f"implementation {name}"))
    return _sha256(_canonical(components)), components


def _assert_captures_current(captures: Mapping[str, FileCapture]) -> None:
    try:
        for capture in captures.values():
            capture.assert_current()
    except ContractError as exc:
        raise LlmMaterializationError(str(exc)) from exc


def materialize_model_bundle(
    *,
    run_plan: FileCapture,
    raw_attempts: Path,
    execution_map: Path,
    statements: Path,
    spend_ledger: Path,
    aggregation: Path,
    pricing: Path,
    output_dir: Path,
    run_id: str,
    served_model: str,
    model_id: str,
    provider_model_id: str,
    workload: str = "unique_exact_pairs_primary",
    expected: ExpectedCounts = ExpectedCounts(),
) -> dict[str, Any]:
    """Validate and publish one manifest-last all-source/five-reader bundle."""

    for label, value in (
        ("run_id", run_id),
        ("served_model", served_model),
        ("model_id", model_id),
        ("provider_model_id", provider_model_id),
        ("workload", workload),
    ):
        if not isinstance(value, str) or not value:
            _fail(f"{label} must be nonempty text")
    output_dir = Path(output_dir).resolve()
    input_paths = {
        "aggregation_config": Path(aggregation).resolve(),
        "raw_attempts": Path(raw_attempts).resolve(),
        "execution_map": Path(execution_map).resolve(),
        "pricing_config": Path(pricing).resolve(),
        "statements": Path(statements).resolve(),
        "spend_ledger": Path(spend_ledger).resolve(),
    }
    if not isinstance(run_plan, FileCapture):
        _fail("run plan must be the authenticated capture loaded by the CLI")
    captures: dict[str, FileCapture] = {
        "run_plan": run_plan,
        **{
            name: _capture(path, label=name) for name, path in input_paths.items()
        },
    }
    raw_rows = _jsonl(captures["raw_attempts"].payload, label="raw attempts")
    map_rows = _jsonl(captures["execution_map"].payload, label="execution map")
    aggregation_value = _json(
        captures["aggregation_config"].payload, label="aggregation config"
    )
    pricing_value = _json(captures["pricing_config"].payload, label="pricing config")
    statement_value = _json(captures["statements"].payload, label="statements")
    if not isinstance(statement_value, list):
        _fail("statements input is not an array")
    pairs, statement_ids = _load_pairs(
        statements=statement_value,
        map_rows=map_rows,
        served_model=served_model,
        workload=workload,
        expected=expected,
    )
    priors, reader_profile, aggregation_name = _aggregation_config(
        aggregation_value
    )
    normalized_priors = _validate_priors(priors, {pair.source for pair in pairs})
    _canonical(reader_profile)
    try:
        ledger = parse_spend_ledger(captures["spend_ledger"].payload)
    except SpendLedgerCorrupt as exc:
        raise LlmMaterializationError(str(exc)) from exc
    attempts = _attempts_from_ledger(
        ledger=ledger,
        pairs=pairs,
        run_id=run_id,
        served_model=served_model,
        provider_model_id=provider_model_id,
        workload=workload,
    )
    pricing_contract = _pricing_contract(
        pricing_value,
        provider_model_id=provider_model_id,
        attempts=attempts,
    )
    raw_by_key = _validate_raw(
        raw_rows=raw_rows,
        pairs=pairs,
        attempts=attempts,
        run_id=run_id,
        provider_model_id=provider_model_id,
    )
    del raw_rows, ledger
    all_predictions, reader_predictions, reader_execution_ids = _predictions(
        pairs=pairs,
        raw_by_key=raw_by_key,
        statement_ids=statement_ids,
        priors=normalized_priors,
        reader_profile=reader_profile,
    )
    if (
        len(reader_predictions) != expected.reader_statements
        or len(reader_execution_ids) != expected.reader_executions
    ):
        _fail("five-reader statement/execution census differs")

    all_cost_rows = _cost_rows(
        pairs=pairs,
        raw_by_key=raw_by_key,
        attempts=attempts,
        provider_model_id=provider_model_id,
    )
    reader_cost_rows = [
        row
        for row in all_cost_rows
        if row["execution_identity"] in reader_execution_ids
    ]
    if len(reader_cost_rows) != expected.reader_executions:
        _fail("five-reader cost ledger does not close over exact execution IDs")
    all_summary = _cost_summary(all_cost_rows)
    reader_summary = _cost_summary(reader_cost_rows)

    payloads = {
        "all_source_predictions.jsonl": b"".join(
            _canonical(row) + b"\n" for row in all_predictions
        ),
        "reader_predictions.jsonl": b"".join(
            _canonical(row) + b"\n" for row in reader_predictions
        ),
        "all_source_attempts.jsonl": b"".join(
            _canonical(row) + b"\n" for row in all_cost_rows
        ),
        "reader_attempts.jsonl": b"".join(
            _canonical(row) + b"\n" for row in reader_cost_rows
        ),
    }
    all_prediction_descriptor = _descriptor(
        output_dir / "all_source_predictions.jsonl",
        payloads["all_source_predictions.jsonl"],
        manifest_dir=output_dir,
        rows=len(all_predictions),
    )
    reader_prediction_descriptor = _descriptor(
        output_dir / "reader_predictions.jsonl",
        payloads["reader_predictions.jsonl"],
        manifest_dir=output_dir,
        rows=len(reader_predictions),
    )
    all_cost_descriptor = _cost_descriptor(
        path=output_dir / "all_source_attempts.jsonl",
        payload=payloads["all_source_attempts.jsonl"],
        manifest_dir=output_dir,
        summary=all_summary,
        pricing=pricing_contract,
        shared_run_id=run_id,
        projection="all_executions",
    )
    reader_cost_descriptor = _cost_descriptor(
        path=output_dir / "reader_attempts.jsonl",
        payload=payloads["reader_attempts.jsonl"],
        manifest_dir=output_dir,
        summary=reader_summary,
        pricing=pricing_contract,
        shared_run_id=run_id,
        projection="observed_execution_subset",
    )
    implementation_digest, components = _implementation_digest()
    descriptor_paths = {"run_plan": run_plan.path, **input_paths}
    input_descriptors = {
        name: _descriptor(path, captures[name].payload, manifest_dir=output_dir)
        for name, path in descriptor_paths.items()
    }
    priors_json = {key: list(value) for key, value in sorted(normalized_priors.items())}
    manifest = {
        "kind": "llm_model_bundle",
        "model_id": model_id,
        "run_id": run_id,
        "implementation": {
            "implementation": "indra_belief.statement_belief:statement_belief",
            "implementation_digest": implementation_digest,
            "training_data_sha256": None,
            "environment": {
                "python": platform.python_version(),
                "runtime": platform.python_implementation(),
            },
            "notes": {
                "served_model": served_model,
                "provider_model_id": provider_model_id,
                "workload": workload,
                "aggregation": aggregation_name,
                "dedup": True,
                "priors_sha256": _sha256(_canonical(priors_json)),
                "reader_profile": reader_profile,
                "reader_sources": sorted(READER_SOURCES),
                "true_reader_reaggregated_from_pair_measurements": True,
                "implementation_components": components,
                "inputs": input_descriptors,
            },
        },
        "panels": {
            "paper_all_source": {
                "prediction_unit": "assembled_statement",
                "substrate_id": "paper_all_source",
                "predictions": all_prediction_descriptor,
                "cost": all_cost_descriptor,
            },
            "paper_readers": {
                "prediction_unit": "assembled_statement",
                "substrate_id": "paper_readers",
                "predictions": reader_prediction_descriptor,
                "cost": reader_cost_descriptor,
            },
        },
    }
    payloads["manifest.json"] = _canonical(manifest, pretty=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".materialize.lock"
    lock_fd = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        status = os.fstat(lock_fd)
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid():
            _fail("materialization lock is not a user-owned regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        for name in payloads:
            if os.path.lexists(output_dir / name):
                raise FileExistsError(f"refusing to clobber {output_dir / name}")
        with tempfile.TemporaryDirectory(prefix=".llm-bundle-", dir=output_dir) as raw:
            stage = Path(raw)
            for name, payload in payloads.items():
                with (stage / name).open("wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            for name in payloads:
                if name != "manifest.json":
                    os.link(stage / name, output_dir / name, follow_symlinks=False)
            directory_fd = os.open(
                output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
                _assert_captures_current(captures)
                os.link(
                    stage / "manifest.json",
                    output_dir / "manifest.json",
                    follow_symlinks=False,
                )
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    return manifest
