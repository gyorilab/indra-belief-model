#!/usr/bin/env python3
"""Publication-grade statement-belief comparison metrics.

This command consumes one frozen evaluation specification whose inputs are
content-addressed statement gold, per-arm probability ledgers, and optional
retry-inclusive evidence-execution cost ledgers.
The two canonical paper panels are evaluated independently: paired uncertainty,
pairwise comparisons, denominators, and Pareto frontiers never cross a panel
boundary.
It emits the comparison artifact consumed by the viewer's
``/frontier?view=belief`` surface.

No prediction is made here.  The calculation fails closed on a digest, ID,
coverage, threshold, or cost-ledger mismatch; it never pads a missing statement
or silently drops an invalid probability.

Run::

    PYTHONPATH=src python -m indra_belief.comparison metrics \
      --spec data/results/indra_belief_comparison_spec.json \
      --output data/results/indra_belief_comparison_metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence

import numpy as np
import sklearn
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

# Shared with assemble.py via the single canonical home, so the
# producer<->consumer statement-id digest contract is structural.
from indra_belief.hashing import ordered_statement_id_sha256, sha256_file


SPEC_KIND = "indra_statement_belief_evaluation_spec"
ARTIFACT_KIND = "indra_statement_belief_comparison"
PREDICTION_UNIT = "assembled_statement"
GOLD_RULE = "released_paper_observed_positive_else_negative"
STRICT_GOLD_RULE = "strict_e0_resolved_only"
POSITIVE_CLASS = "correct_statement"
PARETO_METRIC = "fold_mean_trapezoidal_pr_auc"
CANONICAL_PANEL_IDS = ("paper_all_source", "paper_readers")

PRIMARY_METRICS = (
    "fold_mean_trapezoidal_pr_auc",
    "pooled_average_precision",
    "auroc",
    "brier",
    "log_loss",
    "calibration_ece",
    "calibration_intercept_abs_error",
    "calibration_slope_abs_error",
)
CALIBRATION_PARAMETER_METRICS = (
    "calibration_intercept",
    "calibration_slope",
)
BOOTSTRAP_METRICS = PRIMARY_METRICS + CALIBRATION_PARAMETER_METRICS
THRESHOLD_METRICS = (
    "threshold_accuracy",
    "threshold_precision",
    "threshold_recall",
    "threshold_f1",
)
HIGHER_IS_BETTER = frozenset(
    {
        "fold_mean_trapezoidal_pr_auc",
        "pooled_average_precision",
        "auroc",
        *THRESHOLD_METRICS,
    }
)
LOWER_IS_BETTER = frozenset(
    {
        "brier",
        "log_loss",
        "calibration_ece",
        "calibration_intercept_abs_error",
        "calibration_slope_abs_error",
    }
)
SHA256_LENGTH = 64
DECIMAL_STRING = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
EVIDENCE_EXECUTION_RECORD = "evidence_execution"
COST_VIEW_ID = "provider-runtime-retry-inclusive"
COST_BASES = frozenset(
    {
        "provider_measured_observed",
        "mixed_conservative_upper_bound",
    }
)
COST_EXCLUDED_CATEGORIES = (
    "training",
    "local_aggregation",
    "feature_materialization",
    "upstream_reading",
)

class ContractError(ValueError):
    """An input failed the frozen comparison contract."""


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _fail(context: str, message: str) -> NoReturn:
    raise ContractError(f"{context}: {message}")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(context, "expected a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    expected = set(keys)
    got = set(value)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if extra:
            details.append(f"unexpected fields {extra}")
        _fail(context, "; ".join(details))


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(context, "expected a non-empty string")
    return value


def _identifier(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    if not text[0].isalnum() or any(
        not (char.isalnum() or char in "._:-") for char in text
    ):
        _fail(context, "expected a stable identifier")
    return text


def _sha256_text(value: Any, context: str) -> str:
    text = _nonempty_string(value, context).lower()
    if len(text) != SHA256_LENGTH or any(char not in "0123456789abcdef" for char in text):
        _fail(context, "expected a 64-character SHA-256 digest")
    return text


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(context, "expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        _fail(context, "expected a finite number")
    return number


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(context, f"expected an integer >= {minimum}")
    return value


def _timestamp(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{context}: expected an ISO-8601 timestamp") from exc
    return text


def _date(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ContractError(f"{context}: expected an ISO date") from exc
    return text


def _resolve_path(raw: Any, *, base: Path, context: str) -> tuple[str, Path]:
    display = _nonempty_string(raw, context)
    path = Path(display)
    resolved = path if path.is_absolute() else base / path
    return display, resolved.resolve()


def _verify_file(
    descriptor: Mapping[str, Any],
    *,
    base: Path,
    context: str,
    expected_contract_digest: str | None = None,
) -> tuple[str, Path, str]:
    _exact_keys(descriptor, {"path", "sha256"}, context)
    display, path = _resolve_path(descriptor["path"], base=base, context=f"{context}.path")
    expected = _sha256_text(descriptor["sha256"], f"{context}.sha256")
    if expected_contract_digest is not None and expected != expected_contract_digest:
        _fail(context, "declared digest does not match the panel contract")
    actual = sha256_file(path)
    if actual != expected:
        _fail(context, f"digest mismatch: declared {expected}, actual {actual}")
    return display, path, actual


def _verify_scorer_registry(
    value: Any, *, base: Path
) -> dict[str, Any]:
    context = "spec.scorer_registry"
    descriptor = _object(value, context)
    _exact_keys(descriptor, {"path", "bytes", "sha256"}, context)
    display, path = _resolve_path(
        descriptor["path"], base=base, context=f"{context}.path"
    )
    expected_bytes = _integer(descriptor["bytes"], f"{context}.bytes", minimum=1)
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise ContractError(f"{context}: cannot stat {path}: {exc}") from exc
    if actual_bytes != expected_bytes:
        _fail(context, f"byte count differs: declared {expected_bytes}, actual {actual_bytes}")
    expected_sha = _sha256_text(descriptor["sha256"], f"{context}.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        _fail(context, f"digest mismatch: declared {expected_sha}, actual {actual_sha}")
    return {"path": display, "bytes": expected_bytes, "sha256": expected_sha}


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"{path}: could not read JSON: {exc}") from exc
    try:
        value = json.loads(payload, object_pairs_hook=_strict_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid JSON: {exc}") from exc
    return _object(value, str(path)), payload


def _read_jsonl(path: Path, *, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"{context}: could not open {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                _fail(f"{context}:{line_number}", "blank JSONL lines are forbidden")
            try:
                value = json.loads(line, object_pairs_hook=_strict_object_pairs)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{context}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(_object(value, f"{context}:{line_number}"))
    if not rows:
        _fail(context, "expected at least one JSONL row")
    return rows


def _parse_contract(value: Any, context: str) -> dict[str, str]:
    obj = _object(value, context)
    _exact_keys(
        obj,
        {
            "prediction_unit",
            "gold_rule",
            "substrate_sha256",
            "gold_sha256",
            "evaluation_set_sha256",
        },
        context,
    )
    if obj["prediction_unit"] != PREDICTION_UNIT:
        _fail(f"{context}.prediction_unit", f"must be {PREDICTION_UNIT}")
    if obj["gold_rule"] != GOLD_RULE:
        _fail(f"{context}.gold_rule", f"must be {GOLD_RULE}")
    return {
        "prediction_unit": PREDICTION_UNIT,
        "gold_rule": GOLD_RULE,
        "substrate_sha256": _sha256_text(
            obj["substrate_sha256"], f"{context}.substrate_sha256"
        ),
        "gold_sha256": _sha256_text(obj["gold_sha256"], f"{context}.gold_sha256"),
        "evaluation_set_sha256": _sha256_text(
            obj["evaluation_set_sha256"], f"{context}.evaluation_set_sha256"
        ),
    }


def _load_gold(
    path: Path,
    *,
    contract: Mapping[str, str],
    context: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = _read_jsonl(path, context=context)
    statement_ids: list[str] = []
    labels: list[bool] = []
    folds: list[int] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        row_context = f"{context}[{index}]"
        _exact_keys(row, {"statement_id", "label", "fold_id"}, row_context)
        statement_id = _nonempty_string(row["statement_id"], f"{row_context}.statement_id")
        if statement_id in seen:
            _fail(row_context, f"duplicate statement_id {statement_id!r}")
        seen.add(statement_id)
        label = row["label"]
        if isinstance(label, bool) or label not in (0, 1):
            _fail(f"{row_context}.label", "expected integer 0 or 1")
        fold = _integer(row["fold_id"], f"{row_context}.fold_id")
        statement_ids.append(statement_id)
        labels.append(bool(label))
        folds.append(fold)

    evaluation_digest = ordered_statement_id_sha256(statement_ids)
    if evaluation_digest != contract["evaluation_set_sha256"]:
        _fail(
            context,
            "ordered statement-ID digest does not match contract "
            f"({evaluation_digest} != {contract['evaluation_set_sha256']})",
        )
    y = np.asarray(labels, dtype=bool)
    fold_ids = np.asarray(folds, dtype=np.int64)
    if not np.any(y) or np.all(y):
        _fail(context, "AUROC requires both positive and negative statement gold")
    for fold in sorted(set(folds)):
        fold_y = y[fold_ids == fold]
        if not np.any(fold_y) or np.all(fold_y):
            _fail(context, f"fold {fold} must contain positive and negative statement gold")
    return statement_ids, y, fold_ids


def _load_predictions(
    path: Path,
    *,
    statement_ids: Sequence[str],
    context: str,
) -> np.ndarray:
    rows = _read_jsonl(path, context=context)
    expected = set(statement_ids)
    predictions: dict[str, float] = {}
    for index, row in enumerate(rows):
        row_context = f"{context}[{index}]"
        _exact_keys(row, {"statement_id", "probability_correct"}, row_context)
        statement_id = _nonempty_string(row["statement_id"], f"{row_context}.statement_id")
        if statement_id in predictions:
            _fail(row_context, f"duplicate statement_id {statement_id!r}")
        if statement_id not in expected:
            _fail(row_context, f"statement_id {statement_id!r} is outside the frozen gold")
        probability = _finite(row["probability_correct"], f"{row_context}.probability_correct")
        if not 0 <= probability <= 1:
            _fail(f"{row_context}.probability_correct", "outside [0, 1]")
        predictions[statement_id] = probability
    missing = [statement_id for statement_id in statement_ids if statement_id not in predictions]
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        _fail(context, f"incomplete coverage; missing {len(missing)} IDs ({preview})")
    if len(predictions) != len(statement_ids):
        _fail(context, "prediction count does not equal the frozen evaluation count")
    return np.asarray([predictions[statement_id] for statement_id in statement_ids], dtype=float)


def _parse_threshold(
    value: Any,
    *,
    base: Path,
    context: str,
) -> dict[str, Any]:
    obj = _object(value, context)
    status = obj.get("status")
    if status == "unavailable":
        _exact_keys(obj, {"status", "reason"}, context)
        return {
            "status": "unavailable",
            "value": None,
            "operator": None,
            "source_path": None,
            "source_sha256": None,
            "frozen_at": None,
            "reason": _nonempty_string(obj["reason"], f"{context}.reason"),
        }
    if status != "available":
        _fail(f"{context}.status", "expected available or unavailable")
    _exact_keys(
        obj,
        {"status", "value", "operator", "source_path", "source_sha256", "frozen_at"},
        context,
    )
    threshold = _finite(obj["value"], f"{context}.value")
    if not 0 <= threshold <= 1:
        _fail(f"{context}.value", "outside [0, 1]")
    if obj["operator"] != "greater_than_or_equal":
        _fail(f"{context}.operator", "must be greater_than_or_equal")
    display, path = _resolve_path(
        obj["source_path"], base=base, context=f"{context}.source_path"
    )
    expected = _sha256_text(obj["source_sha256"], f"{context}.source_sha256")
    actual = sha256_file(path)
    if actual != expected:
        _fail(context, f"threshold-source digest mismatch: declared {expected}, actual {actual}")
    return {
        "status": "available",
        "value": threshold,
        "operator": "greater_than_or_equal",
        "source_path": display,
        "source_sha256": actual,
        "frozen_at": _timestamp(obj["frozen_at"], f"{context}.frozen_at"),
        "reason": None,
    }


def _exact_decimal_string(value: Any, context: str) -> Decimal:
    if not isinstance(value, str) or DECIMAL_STRING.fullmatch(value) is None:
        _fail(context, "expected a non-negative exact decimal string without exponent notation")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - guarded by the grammar
        raise ContractError(f"{context}: invalid exact decimal string") from exc
    if not number.is_finite() or number < 0:
        _fail(context, "expected a finite non-negative exact decimal string")
    return number


def _sum_exact_decimals(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    fractional_places = max(max(-value.as_tuple().exponent, 0) for value in values)
    integer_places = max(
        max(len(value.as_tuple().digits) + value.as_tuple().exponent, 0)
        for value in values
    )
    carry_places = len(str(len(values)))
    with localcontext() as decimal_context:
        decimal_context.prec = max(28, integer_places + fractional_places + carry_places + 2)
        return sum(values, start=Decimal("0"))


def _parse_evidence_execution_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    statement_ids: Sequence[str],
    context: str,
) -> dict[str, Any]:
    expected_ids = set(statement_ids)
    covered_ids: set[str] = set()
    seen_executions: set[tuple[str, str]] = set()
    seen_attempt_ids: set[str] = set()
    seen_call_ids: set[str] = set()
    provider_measured_cost_values: list[Decimal] = []
    conservative_reserved_cost_values: list[Decimal] = []
    execution_count = 0
    attempt_count = 0
    retry_attempt_count = 0
    completed_attempt_count = 0
    error_attempt_count = 0
    provider_measured_call_count = 0
    conservative_call_count = 0
    provider_input_tokens = 0
    provider_output_tokens = 0

    for row_index, row in enumerate(rows):
        row_context = f"{context}[{row_index}]"
        _exact_keys(
            row,
            {
                "record_type",
                "statement_id",
                "execution_identity",
                "call_eligible",
                "attempts",
            },
            row_context,
        )
        if row["record_type"] != EVIDENCE_EXECUTION_RECORD:
            _fail(
                f"{row_context}.record_type",
                f"must be {EVIDENCE_EXECUTION_RECORD}",
            )
        statement_id = _nonempty_string(row["statement_id"], f"{row_context}.statement_id")
        if statement_id not in expected_ids:
            _fail(row_context, f"statement_id {statement_id!r} is outside frozen gold")
        execution_identity = _nonempty_string(
            row["execution_identity"], f"{row_context}.execution_identity"
        )
        execution_key = (statement_id, execution_identity)
        if execution_key in seen_executions:
            _fail(
                row_context,
                f"duplicate statement/execution identity {statement_id!r}/{execution_identity!r}",
            )
        seen_executions.add(execution_key)
        covered_ids.add(statement_id)
        if not isinstance(row["call_eligible"], bool):
            _fail(f"{row_context}.call_eligible", "expected a boolean")
        call_eligible = bool(row["call_eligible"])
        raw_attempts = row["attempts"]
        if not isinstance(raw_attempts, list) or not raw_attempts:
            _fail(f"{row_context}.attempts", "expected a non-empty array")

        execution_attempts: list[dict[str, Any]] = []
        execution_has_calls = False
        for attempt_index, raw_attempt in enumerate(raw_attempts):
            attempt_context = f"{row_context}.attempts[{attempt_index}]"
            attempt = _object(raw_attempt, attempt_context)
            _exact_keys(
                attempt,
                {
                    "attempt_id",
                    "attempt_ordinal",
                    "status",
                    "selected_final",
                    "error_type",
                    "calls",
                },
                attempt_context,
            )
            attempt_id = _identifier(attempt["attempt_id"], f"{attempt_context}.attempt_id")
            if attempt_id in seen_attempt_ids:
                _fail(attempt_context, f"duplicate attempt_id {attempt_id!r}")
            seen_attempt_ids.add(attempt_id)
            attempt_ordinal = _integer(
                attempt["attempt_ordinal"], f"{attempt_context}.attempt_ordinal", minimum=1
            )
            if attempt_ordinal != attempt_index + 1:
                _fail(
                    f"{attempt_context}.attempt_ordinal",
                    "attempt ordinals must be contiguous and array-ordered from 1",
                )
            attempt_status = attempt["status"]
            if attempt_status not in {"completed", "error"}:
                _fail(f"{attempt_context}.status", "expected completed or error")
            if not isinstance(attempt["selected_final"], bool):
                _fail(f"{attempt_context}.selected_final", "expected a boolean")
            error_type = attempt["error_type"]
            if attempt_status == "completed":
                if error_type is not None:
                    _fail(f"{attempt_context}.error_type", "completed attempt must use null")
                completed_attempt_count += 1
            else:
                _nonempty_string(error_type, f"{attempt_context}.error_type")
                error_attempt_count += 1

            raw_calls = attempt["calls"]
            if not isinstance(raw_calls, list):
                _fail(f"{attempt_context}.calls", "expected an array")
            for call_index, raw_call in enumerate(raw_calls):
                call_context = f"{attempt_context}.calls[{call_index}]"
                call = _object(raw_call, call_context)
                _exact_keys(
                    call,
                    {
                        "call_id",
                        "call_ordinal",
                        "kind",
                        "model_id",
                        "accounting_basis",
                        "input_tokens",
                        "output_tokens",
                        "settled_cost_usd_exact",
                    },
                    call_context,
                )
                call_id = _identifier(call["call_id"], f"{call_context}.call_id")
                if call_id in seen_call_ids:
                    _fail(call_context, f"duplicate call_id {call_id!r}")
                seen_call_ids.add(call_id)
                call_ordinal = _integer(
                    call["call_ordinal"], f"{call_context}.call_ordinal", minimum=1
                )
                if call_ordinal != call_index + 1:
                    _fail(
                        f"{call_context}.call_ordinal",
                        "call ordinals must be contiguous and array-ordered from 1 per attempt",
                    )
                _nonempty_string(call["kind"], f"{call_context}.kind")
                _nonempty_string(call["model_id"], f"{call_context}.model_id")
                accounting_basis = call["accounting_basis"]
                settled_cost = _exact_decimal_string(
                    call["settled_cost_usd_exact"],
                    f"{call_context}.settled_cost_usd_exact",
                )
                if accounting_basis == "provider_reported_usage":
                    provider_input_tokens += _integer(
                        call["input_tokens"], f"{call_context}.input_tokens"
                    )
                    provider_output_tokens += _integer(
                        call["output_tokens"], f"{call_context}.output_tokens"
                    )
                    provider_measured_call_count += 1
                    provider_measured_cost_values.append(settled_cost)
                elif accounting_basis == "conservative_reserved_maximum":
                    if call["input_tokens"] is not None or call["output_tokens"] is not None:
                        _fail(
                            call_context,
                            "conservative_reserved_maximum calls require null token counts",
                        )
                    conservative_call_count += 1
                    conservative_reserved_cost_values.append(settled_cost)
                else:
                    _fail(
                        f"{call_context}.accounting_basis",
                        "expected provider_reported_usage or conservative_reserved_maximum",
                    )
            execution_has_calls = execution_has_calls or bool(raw_calls)
            execution_attempts.append(attempt)

        last_index = len(execution_attempts) - 1
        for attempt_index, attempt in enumerate(execution_attempts):
            should_be_final = attempt_index == last_index
            if attempt["selected_final"] is not should_be_final:
                _fail(
                    f"{row_context}.attempts[{attempt_index}].selected_final",
                    "exactly the last attempt must be selected_final",
                )
            expected_status = "completed" if should_be_final else "error"
            if attempt["status"] != expected_status:
                _fail(
                    f"{row_context}.attempts[{attempt_index}].status",
                    f"expected {expected_status}; all prior attempts fail and the final completes",
                )
        if call_eligible and not execution_attempts[-1]["calls"]:
            _fail(row_context, "call-eligible execution requires calls in its selected final attempt")
        if not call_eligible and execution_has_calls:
            _fail(row_context, "call-ineligible execution must not contain calls")

        execution_count += 1
        attempt_count += len(execution_attempts)
        retry_attempt_count += len(execution_attempts) - 1

    missing_ids = [statement_id for statement_id in statement_ids if statement_id not in covered_ids]
    if missing_ids:
        _fail(context, f"cost ledger never names {len(missing_ids)} frozen statements")
    if completed_attempt_count != execution_count:
        _fail(context, "completed attempt count does not equal execution count")
    if error_attempt_count != retry_attempt_count:
        _fail(context, "error attempt count does not reconcile to retries per execution")
    token_complete = conservative_call_count == 0
    provider_measured_cost = _sum_exact_decimals(provider_measured_cost_values)
    conservative_reserved_cost = _sum_exact_decimals(conservative_reserved_cost_values)
    return {
        "total_cost": provider_measured_cost + conservative_reserved_cost,
        "provider_measured_cost": provider_measured_cost,
        "conservative_reserved_cost": conservative_reserved_cost,
        "execution_count": execution_count,
        "attempt_count": attempt_count,
        "retry_attempt_count": retry_attempt_count,
        "successful_attempt_count": completed_attempt_count,
        "error_attempt_count": error_attempt_count,
        "input_tokens": provider_input_tokens if token_complete else None,
        "output_tokens": provider_output_tokens if token_complete else None,
        "token_accounting_complete": token_complete,
        "provider_measured_call_count": provider_measured_call_count,
        "conservative_call_count": conservative_call_count,
    }


def _validate_declared_cost_accounting(
    value: Any,
    *,
    basis: str,
    statement_count: int,
    summary: Mapping[str, Any],
    context: str,
) -> None:
    accounting = _object(value, context)
    _exact_keys(
        accounting,
        {
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
        },
        context,
    )
    declared_provider = _exact_decimal_string(
        accounting["provider_measured_cost_usd_exact"],
        f"{context}.provider_measured_cost_usd_exact",
    )
    declared_conservative = _exact_decimal_string(
        accounting["conservative_reserved_cost_usd_exact"],
        f"{context}.conservative_reserved_cost_usd_exact",
    )
    declared_lower = _exact_decimal_string(
        accounting["accounted_cost_lower_usd_exact"],
        f"{context}.accounted_cost_lower_usd_exact",
    )
    declared_upper = _exact_decimal_string(
        accounting["accounted_cost_upper_usd_exact"],
        f"{context}.accounted_cost_upper_usd_exact",
    )
    recomputed_provider = summary["provider_measured_cost"]
    recomputed_conservative = summary["conservative_reserved_cost"]
    recomputed_upper = recomputed_provider + recomputed_conservative
    exact_checks = {
        "provider_measured_cost_usd_exact": (declared_provider, recomputed_provider),
        "conservative_reserved_cost_usd_exact": (
            declared_conservative,
            recomputed_conservative,
        ),
        "accounted_cost_lower_usd_exact": (declared_lower, recomputed_provider),
        "accounted_cost_upper_usd_exact": (declared_upper, recomputed_upper),
    }
    for field, (declared, recomputed) in exact_checks.items():
        if declared != recomputed:
            _fail(
                f"{context}.{field}",
                f"declared {format(declared, 'f')} but ledger recomputes "
                f"{format(recomputed, 'f')}",
            )
    for field in ("provider_measured_call_count", "conservative_call_count"):
        declared_count = _integer(accounting[field], f"{context}.{field}")
        if declared_count != summary[field]:
            _fail(
                f"{context}.{field}",
                f"declared {declared_count} but ledger recomputes {summary[field]}",
            )
    if accounting["includes_retries"] is not True:
        _fail(f"{context}.includes_retries", "must be true")
    if accounting["includes_relation_subcalls"] is not True:
        _fail(f"{context}.includes_relation_subcalls", "must be true")
    denominator = _object(accounting["denominator"], f"{context}.denominator")
    _exact_keys(denominator, {"statements", "evidence_executions"}, f"{context}.denominator")
    if _integer(denominator["statements"], f"{context}.denominator.statements", minimum=1) != statement_count:
        _fail(f"{context}.denominator.statements", "does not equal the frozen statement count")
    if _integer(
        denominator["evidence_executions"],
        f"{context}.denominator.evidence_executions",
        minimum=1,
    ) != summary["execution_count"]:
        _fail(
            f"{context}.denominator.evidence_executions",
            "does not equal the ledger execution count",
        )
    excluded = accounting["excluded_cost_categories"]
    if excluded != list(COST_EXCLUDED_CATEGORIES):
        _fail(
            f"{context}.excluded_cost_categories",
            f"must equal {list(COST_EXCLUDED_CATEGORIES)!r} in contract order",
        )
    recomputed_basis = (
        "mixed_conservative_upper_bound"
        if summary["conservative_call_count"]
        else "provider_measured_observed"
    )
    if basis != recomputed_basis:
        _fail(
            f"{context}.basis",
            f"declared {basis!r} but ledger provenance requires {recomputed_basis!r}",
        )


def _parse_pricing(
    value: Any,
    *,
    comparability_id: str,
    price_source: str,
    price_date: str,
    context: str,
) -> dict[str, Any]:
    obj = _object(value, context)
    _exact_keys(
        obj,
        {
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
        },
        context,
    )
    if _identifier(
        obj["cost_comparability_id"], f"{context}.cost_comparability_id"
    ) != comparability_id:
        _fail(f"{context}.cost_comparability_id", "differs from the cost descriptor")
    if obj["currency"] != "USD":
        _fail(f"{context}.currency", "must be USD")
    if obj["pricing_mode"] != "on_demand":
        _fail(f"{context}.pricing_mode", "must be on_demand")
    if obj["service_tier_request"] != "default":
        _fail(f"{context}.service_tier_request", "must be default")
    if obj["resolved_service_tier"] != "standard":
        _fail(f"{context}.resolved_service_tier", "must be standard")
    if obj["unit"] != "per_million_tokens":
        _fail(f"{context}.unit", "must be per_million_tokens")
    source_url = _nonempty_string(obj["source_url"], f"{context}.source_url")
    if source_url != price_source or not source_url.startswith("https://"):
        _fail(f"{context}.source_url", "must equal the HTTPS descriptor price source")
    retrieved_on = _date(obj["retrieved_on"], f"{context}.retrieved_on")
    if retrieved_on != price_date:
        _fail(f"{context}.retrieved_on", "must equal the descriptor price date")
    provider = _nonempty_string(obj["provider"], f"{context}.provider")
    provider_model_id = _identifier(
        obj["provider_model_id"], f"{context}.provider_model_id"
    )
    region = _nonempty_string(obj["region"], f"{context}.region")
    tariff = _object(obj["tariff"], f"{context}.tariff")
    _exact_keys(
        tariff,
        {
            "input_usd_per_million",
            "output_usd_per_million",
            "pricing_basis",
        },
        f"{context}.tariff",
    )
    canonical_rates: dict[str, str] = {}
    for field in ("input_usd_per_million", "output_usd_per_million"):
        raw = tariff[field]
        number = _exact_decimal_string(raw, f"{context}.tariff.{field}")
        canonical = format(number.normalize(), "f")
        if raw != canonical:
            _fail(f"{context}.tariff.{field}", "must use canonical decimal spelling")
        canonical_rates[field] = canonical
    pricing_basis = _nonempty_string(
        tariff["pricing_basis"], f"{context}.tariff.pricing_basis"
    )
    return {
        "cost_comparability_id": comparability_id,
        "currency": "USD",
        "provider": provider,
        "provider_model_id": provider_model_id,
        "pricing_mode": "on_demand",
        "region": region,
        "resolved_service_tier": "standard",
        "retrieved_on": retrieved_on,
        "service_tier_request": "default",
        "source_url": source_url,
        "tariff": {
            **canonical_rates,
            "pricing_basis": pricing_basis,
        },
        "unit": "per_million_tokens",
    }


def _parse_cost(
    value: Any,
    *,
    base: Path,
    statement_ids: Sequence[str],
    expected_projection: str,
    context: str,
) -> dict[str, Any]:
    obj = _object(value, context)
    status = obj.get("status")
    if status == "unavailable":
        _exact_keys(obj, {"status", "reason"}, context)
        return {
            "status": "unavailable",
            "record_type": None,
            "inference_usd_total": None,
            "inference_usd_total_exact": None,
            "usd_per_1k_statements": None,
            "provider_measured_usd_total": None,
            "provider_measured_usd_total_exact": None,
            "conservative_reserved_usd_total": None,
            "conservative_reserved_usd_total_exact": None,
            "inference_usd_lower": None,
            "inference_usd_lower_exact": None,
            "inference_usd_upper": None,
            "inference_usd_upper_exact": None,
            "usd_per_1k_statements_lower": None,
            "usd_per_1k_statements_upper": None,
            "basis": "unavailable",
            "view_id": None,
            "includes_retries": None,
            "includes_relation_subcalls": None,
            "denominator": None,
            "scope": None,
            "execution_count": None,
            "attempt_count": None,
            "retry_attempt_count": None,
            "successful_attempt_count": None,
            "error_attempt_count": None,
            "provider_measured_call_count": None,
            "conservative_call_count": None,
            "input_tokens": None,
            "output_tokens": None,
            "token_accounting_complete": None,
            "ledger_path": None,
            "ledger_sha256": None,
            "price_source": None,
            "price_date": None,
            "cost_comparability_id": None,
            "pricing": None,
            "projection": None,
            "counterfactual_run_cost": None,
            "shared_run_id": None,
            "additive_across_panels": None,
            "reason": _nonempty_string(obj["reason"], f"{context}.reason"),
        }
    if status != "ledger":
        _fail(f"{context}.status", "expected ledger or unavailable")
    record_type = obj.get("record_type")
    _exact_keys(
        obj,
        {
            "status",
            "record_type",
            "path",
            "sha256",
            "basis",
            "view_id",
            "price_source",
            "price_date",
            "cost_comparability_id",
            "pricing",
            "projection",
            "counterfactual_run_cost",
            "shared_run_id",
            "additive_across_panels",
            "accounting",
        },
        context,
    )
    basis = obj["basis"]
    view_id = _identifier(obj["view_id"], f"{context}.view_id")
    if record_type != EVIDENCE_EXECUTION_RECORD:
        _fail(f"{context}.record_type", f"expected {EVIDENCE_EXECUTION_RECORD}")
    if basis not in COST_BASES:
        _fail(
            f"{context}.basis",
            "expected provider_measured_observed or mixed_conservative_upper_bound",
        )
    if view_id != COST_VIEW_ID:
        _fail(f"{context}.view_id", f"must be {COST_VIEW_ID!r}")
    price_source = _nonempty_string(obj["price_source"], f"{context}.price_source")
    price_date = _date(obj["price_date"], f"{context}.price_date")
    comparability_id = _identifier(
        obj["cost_comparability_id"], f"{context}.cost_comparability_id"
    )
    pricing = _parse_pricing(
        obj["pricing"],
        comparability_id=comparability_id,
        price_source=price_source,
        price_date=price_date,
        context=f"{context}.pricing",
    )
    projection = _identifier(obj["projection"], f"{context}.projection")
    if projection != expected_projection:
        _fail(f"{context}.projection", f"must be {expected_projection!r} for this panel")
    if obj["counterfactual_run_cost"] is not False:
        _fail(f"{context}.counterfactual_run_cost", "must be false")
    shared_run_id = _identifier(obj["shared_run_id"], f"{context}.shared_run_id")
    if obj["additive_across_panels"] is not False:
        _fail(f"{context}.additive_across_panels", "must be false")
    display, path = _resolve_path(obj["path"], base=base, context=f"{context}.path")
    expected_digest = _sha256_text(obj["sha256"], f"{context}.sha256")
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        _fail(context, f"cost-ledger digest mismatch: declared {expected_digest}, actual {actual_digest}")

    ledger_context = f"{context}.ledger"
    rows = _read_jsonl(path, context=ledger_context)
    summary = _parse_evidence_execution_ledger(
        rows, statement_ids=statement_ids, context=ledger_context
    )
    _validate_declared_cost_accounting(
        obj["accounting"],
        basis=basis,
        statement_count=len(statement_ids),
        summary=summary,
        context=f"{context}.accounting",
    )
    total_cost = summary["total_cost"]
    provider_measured_cost = summary["provider_measured_cost"]
    conservative_reserved_cost = summary["conservative_reserved_cost"]
    lower_cost = provider_measured_cost
    upper_cost = total_cost
    n = len(statement_ids)
    chart_lower = _finite(float(lower_cost), f"{context}.inference_usd_lower")
    chart_upper = _finite(float(upper_cost), f"{context}.inference_usd_upper")
    chart_per_1k_lower = _finite(
        float(lower_cost * Decimal(1000) / Decimal(n)),
        f"{context}.usd_per_1k_statements_lower",
    )
    chart_per_1k_upper = _finite(
        float(upper_cost * Decimal(1000) / Decimal(n)),
        f"{context}.usd_per_1k_statements",
    )
    return {
        "status": "available",
        "record_type": record_type,
        # Point fields use the conservative accounted upper endpoint; the
        # explicit interval distinguishes measured from reserved cost.
        "inference_usd_total": chart_upper,
        "inference_usd_total_exact": format(upper_cost, "f"),
        "usd_per_1k_statements": chart_per_1k_upper,
        "provider_measured_usd_total": _finite(
            float(provider_measured_cost), f"{context}.provider_measured_usd_total"
        ),
        "provider_measured_usd_total_exact": format(provider_measured_cost, "f"),
        "conservative_reserved_usd_total": _finite(
            float(conservative_reserved_cost),
            f"{context}.conservative_reserved_usd_total",
        ),
        "conservative_reserved_usd_total_exact": format(
            conservative_reserved_cost, "f"
        ),
        "inference_usd_lower": chart_lower,
        "inference_usd_lower_exact": format(lower_cost, "f"),
        "inference_usd_upper": chart_upper,
        "inference_usd_upper_exact": format(upper_cost, "f"),
        "usd_per_1k_statements_lower": chart_per_1k_lower,
        "usd_per_1k_statements_upper": chart_per_1k_upper,
        "basis": basis,
        "view_id": view_id,
        "includes_retries": True,
        "includes_relation_subcalls": True,
        "denominator": {
            "statements": n,
            "evidence_executions": summary["execution_count"],
        },
        "scope": {
            "included_cost_categories": ["provider_inference_calls"],
            "excluded_cost_categories": list(COST_EXCLUDED_CATEGORIES),
        },
        "execution_count": summary["execution_count"],
        "attempt_count": summary["attempt_count"],
        "retry_attempt_count": summary["retry_attempt_count"],
        "successful_attempt_count": summary["successful_attempt_count"],
        "error_attempt_count": summary["error_attempt_count"],
        "provider_measured_call_count": summary["provider_measured_call_count"],
        "conservative_call_count": summary["conservative_call_count"],
        "input_tokens": summary["input_tokens"],
        "output_tokens": summary["output_tokens"],
        "token_accounting_complete": summary["token_accounting_complete"],
        "ledger_path": display,
        "ledger_sha256": actual_digest,
        "price_source": price_source,
        "price_date": price_date,
        "cost_comparability_id": comparability_id,
        "pricing": pricing,
        "projection": projection,
        "counterfactual_run_cost": False,
        "shared_run_id": shared_run_id,
        "additive_across_panels": False,
        "reason": None,
    }


def _unit_interval(value: float, context: str) -> float:
    """Remove only floating-point fuzz at a mathematically bounded metric edge."""
    if not math.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12:
        _fail(context, f"bounded metric escaped [0, 1]: {value}")
    return min(1.0, max(0.0, float(value)))


def _weighted_pr_summaries(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    positive_total = float(np.sum(weights[labels]))
    if positive_total <= 0:
        return math.nan, math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    sorted_labels = labels[order]
    if not len(order):
        return math.nan, math.nan
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    group_total = np.add.reduceat(sorted_weights, starts)
    group_positive = np.add.reduceat(sorted_weights * sorted_labels, starts)
    cumulative_total = np.cumsum(group_total)
    cumulative_positive = np.cumsum(group_positive)
    nonempty = cumulative_total > 0
    precision = cumulative_positive[nonempty] / cumulative_total[nonempty]
    recall = cumulative_positive[nonempty] / positive_total
    if not len(recall):
        return math.nan, math.nan
    recall_before = np.r_[0.0, recall[:-1]]
    average_precision = _unit_interval(
        float(np.sum((recall - recall_before) * precision)), "weighted average precision"
    )
    trapezoidal = _unit_interval(
        float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall])),
        "weighted trapezoidal PR area",
    )
    return trapezoidal, average_precision


def _weighted_auroc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    positive_total = float(np.sum(weights[labels]))
    negative_total = float(np.sum(weights[~labels]))
    if positive_total <= 0 or negative_total <= 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    positive_by_score = np.add.reduceat(sorted_weights * sorted_labels, starts)
    negative_by_score = np.add.reduceat(sorted_weights * (~sorted_labels), starts)
    negative_below = np.cumsum(negative_by_score) - negative_by_score
    concordant = np.sum(positive_by_score * (negative_below + 0.5 * negative_by_score))
    return _unit_interval(
        float(concordant / (positive_total * negative_total)), "weighted AUROC"
    )


def _expit(value: np.ndarray) -> np.ndarray:
    out = np.empty_like(value, dtype=float)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def _calibration_intercept_slope(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    epsilon: float,
) -> tuple[float, float]:
    if float(np.sum(weights[labels])) <= 0 or float(np.sum(weights[~labels])) <= 0:
        return math.nan, math.nan
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(len(scores)), logits))
    beta = np.asarray([0.0, 1.0], dtype=float)

    def log_likelihood(candidate: np.ndarray) -> float:
        eta = design @ candidate
        return float(np.sum(weights * (labels * eta - np.logaddexp(0.0, eta))))

    likelihood = log_likelihood(beta)
    for _ in range(100):
        eta = design @ beta
        probability = _expit(eta)
        variance_weight = weights * probability * (1.0 - probability)
        information = design.T @ (variance_weight[:, None] * design)
        score = design.T @ (weights * (labels.astype(float) - probability))
        if not np.all(np.isfinite(information)) or np.linalg.cond(information) > 1e14:
            return math.nan, math.nan
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return math.nan, math.nan
        scale = 1.0
        accepted = False
        while scale >= 2**-20:
            candidate = beta + scale * step
            candidate_likelihood = log_likelihood(candidate)
            if math.isfinite(candidate_likelihood) and candidate_likelihood >= likelihood - 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return math.nan, math.nan
        beta = candidate
        likelihood = candidate_likelihood
        if np.max(np.abs(scale * step)) < 1e-10:
            if np.max(np.abs(beta)) > 1e6:
                return math.nan, math.nan
            return float(beta[0]), float(beta[1])
    return math.nan, math.nan


def _bin_indices(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, scores, side="right") - 1
    return np.clip(indices, 0, len(edges) - 2)


def _weighted_ece(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return math.nan
    indices = _bin_indices(scores, edges)
    value = 0.0
    for bin_index in range(len(edges) - 1):
        mask = indices == bin_index
        bin_weight = float(np.sum(weights[mask]))
        if bin_weight <= 0:
            continue
        mean_prediction = float(np.sum(weights[mask] * scores[mask]) / bin_weight)
        observed = float(np.sum(weights[mask] * labels[mask]) / bin_weight)
        value += bin_weight / total * abs(mean_prediction - observed)
    return _unit_interval(value, "weighted ECE")


def _reliability_bins(
    labels: np.ndarray,
    scores: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    indices = _bin_indices(scores, edges)
    bins: list[dict[str, Any]] = []
    for bin_index in range(len(edges) - 1):
        mask = indices == bin_index
        count = int(np.sum(mask))
        bins.append(
            {
                "bin_index": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "upper_inclusive": bin_index == len(edges) - 2,
                "n": count,
                "mean_prediction": float(np.mean(scores[mask])) if count else None,
                "observed_fraction": float(np.mean(labels[mask])) if count else None,
            }
        )
    return bins


def _threshold_values(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], dict[str, int]]:
    predicted = scores >= threshold
    tp = float(np.sum(weights[predicted & labels]))
    fp = float(np.sum(weights[predicted & ~labels]))
    fn = float(np.sum(weights[~predicted & labels]))
    tn = float(np.sum(weights[~predicted & ~labels]))
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else math.nan
    metrics = {
        "threshold_accuracy": accuracy,
        "threshold_precision": precision,
        "threshold_recall": recall,
        "threshold_f1": f1,
    }
    confusion = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    return metrics, confusion


class _MetricComputer:
    def __init__(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        folds: np.ndarray,
        *,
        log_loss_epsilon: float,
        calibration_edges: np.ndarray,
        threshold: float | None,
    ) -> None:
        self.labels = labels
        self.scores = scores
        self.folds = folds
        self.log_loss_epsilon = log_loss_epsilon
        self.calibration_edges = calibration_edges
        self.threshold = threshold
        self.fold_values = sorted(int(value) for value in np.unique(folds))

    def values(self, weights: np.ndarray) -> dict[str, float]:
        fold_areas: list[float] = []
        fold_metric_valid = True
        for fold in self.fold_values:
            mask = self.folds == fold
            trapezoidal, _ = _weighted_pr_summaries(
                self.labels[mask], self.scores[mask], weights[mask]
            )
            if not math.isfinite(trapezoidal):
                fold_metric_valid = False
            else:
                fold_areas.append(trapezoidal)
        _, pooled_ap = _weighted_pr_summaries(self.labels, self.scores, weights)
        auroc_value = _weighted_auroc(self.labels, self.scores, weights)
        total = float(np.sum(weights))
        brier = float(np.sum(weights * (self.scores - self.labels) ** 2) / total)
        clipped = np.clip(self.scores, self.log_loss_epsilon, 1.0 - self.log_loss_epsilon)
        losses = -(self.labels * np.log(clipped) + (~self.labels) * np.log1p(-clipped))
        log_loss = float(np.sum(weights * losses) / total)
        ece = _weighted_ece(self.labels, self.scores, weights, self.calibration_edges)
        intercept, slope = _calibration_intercept_slope(
            self.labels,
            self.scores,
            weights,
            epsilon=self.log_loss_epsilon,
        )
        values = {
            "fold_mean_trapezoidal_pr_auc": (
                float(np.mean(fold_areas)) if fold_metric_valid else math.nan
            ),
            "pooled_average_precision": pooled_ap,
            "auroc": auroc_value,
            "brier": brier,
            "log_loss": log_loss,
            "calibration_ece": ece,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "calibration_intercept_abs_error": abs(intercept),
            "calibration_slope_abs_error": abs(slope - 1.0),
        }
        if self.threshold is not None:
            threshold_values, _ = _threshold_values(
                self.labels, self.scores, weights, self.threshold
            )
            values.update(threshold_values)
        return values

    def point(self) -> tuple[dict[str, float], list[dict[str, Any]], float, dict[str, int] | None]:
        weights = np.ones(len(self.labels), dtype=float)
        values = self.values(weights)
        fold_rows: list[dict[str, Any]] = []
        sklearn_fold_areas: list[float] = []
        for fold in self.fold_values:
            mask = self.folds == fold
            precision, recall, _ = precision_recall_curve(self.labels[mask], self.scores[mask])
            fold_area = _unit_interval(float(auc(recall, precision)), "sklearn fold PR area")
            sklearn_fold_areas.append(fold_area)
            fold_rows.append(
                {
                    "fold_id": fold,
                    "n": int(np.sum(mask)),
                    "positive": int(np.sum(self.labels[mask])),
                    "negative": int(np.sum(~self.labels[mask])),
                    "estimate": fold_area,
                }
            )
        sklearn_points = {
            "fold_mean_trapezoidal_pr_auc": _unit_interval(
                float(np.mean(sklearn_fold_areas)), "sklearn fold-mean PR area"
            ),
            "pooled_average_precision": _unit_interval(
                float(average_precision_score(self.labels, self.scores)),
                "sklearn average precision",
            ),
            "auroc": _unit_interval(
                float(roc_auc_score(self.labels, self.scores)), "sklearn AUROC"
            ),
            "brier": _unit_interval(
                float(brier_score_loss(self.labels, self.scores)), "sklearn Brier"
            ),
        }
        for key, expected in sklearn_points.items():
            if not math.isclose(values[key], expected, rel_tol=1e-12, abs_tol=1e-12):
                _fail("metrics", f"internal weighted {key} disagrees with sklearn ({values[key]} != {expected})")
            values[key] = expected
        threshold_confusion = None
        if self.threshold is not None:
            _, threshold_confusion = _threshold_values(
                self.labels, self.scores, weights, self.threshold
            )
        return values, fold_rows, float(np.std(sklearn_fold_areas, ddof=0)), threshold_confusion


def _metric_method(metric: str, *, epsilon: float, edge_count: int) -> str:
    bootstrap = (
        "paired statement bootstrap percentile CI, stratified independently within each "
        "frozen fold while preserving its size and using one paired weight vector across "
        "arms; conditional on frozen fits, folds, and predictions; models are not refit "
        "or rerun"
    )
    methods = {
        "fold_mean_trapezoidal_pr_auc": (
            "arithmetic mean over frozen folds of sklearn precision_recall_curve then "
            f"auc(recall, precision); {bootstrap}"
        ),
        "pooled_average_precision": f"sklearn average_precision_score over pooled statements; {bootstrap}",
        "auroc": f"sklearn roc_auc_score over pooled statements; {bootstrap}",
        "brier": f"mean squared probability error over pooled statements; {bootstrap}",
        "log_loss": f"binary log loss with probabilities clipped to [{epsilon}, {1 - epsilon}]; {bootstrap}",
        "calibration_ece": f"ECE over {edge_count - 1} frozen bins; {bootstrap}",
        "calibration_intercept": f"unpenalized logistic calibration intercept on clipped logit probabilities; {bootstrap}",
        "calibration_slope": f"unpenalized logistic calibration slope on clipped logit probabilities; {bootstrap}",
        "calibration_intercept_abs_error": f"absolute distance of calibration intercept from its ideal target 0; {bootstrap}",
        "calibration_slope_abs_error": f"absolute distance of calibration slope from its ideal target 1; {bootstrap}",
        "threshold_accuracy": f"correct-statement accuracy at a pre-frozen threshold; {bootstrap}",
        "threshold_precision": f"correct-statement precision at a pre-frozen threshold; {bootstrap}",
        "threshold_recall": f"correct-statement recall at a pre-frozen threshold; {bootstrap}",
        "threshold_f1": f"correct-statement F1 at a pre-frozen threshold; {bootstrap}",
    }
    return methods[metric]


def _stratified_paired_bootstrap_weights(
    rng: np.random.Generator, folds: np.ndarray
) -> np.ndarray:
    """Resample statements within every frozen fold, retaining each fold census.

    The returned vector is computed once per replicate and reused for every arm,
    which preserves both the paper's fixed-fold estimand and paired comparisons.
    """
    if folds.ndim != 1 or len(folds) == 0:
        _fail("bootstrap.folds", "expected a non-empty one-dimensional fold array")
    weights = np.zeros(len(folds), dtype=float)
    for fold in np.unique(folds):
        indices = np.flatnonzero(folds == fold)
        sampled = indices[rng.integers(0, len(indices), size=len(indices))]
        np.add.at(weights, sampled, 1.0)
    return weights


def _estimate(
    point: float,
    bootstrap_values: np.ndarray,
    *,
    method: str,
    ci_level: float,
    requested_resamples: int,
    minimum_valid_fraction: float,
    context: str,
) -> dict[str, Any]:
    if not math.isfinite(point):
        _fail(context, "point estimate is undefined")
    valid = bootstrap_values[np.isfinite(bootstrap_values)]
    required = math.ceil(requested_resamples * minimum_valid_fraction)
    if len(valid) < required:
        _fail(
            context,
            f"only {len(valid)}/{requested_resamples} finite bootstrap replicates; require {required}",
        )
    alpha = (1.0 - ci_level) / 2.0
    lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
    return {
        "estimate": float(point),
        "ci95": [float(lower), float(upper)],
        "method": method,
        "resamples": requested_resamples,
        "valid_resamples": int(len(valid)),
    }


def _better_when(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher"
    if metric in LOWER_IS_BETTER:
        return "lower"
    _fail("metric", f"unknown comparison direction for {metric}")


def _build_pareto(
    arms: list[dict[str, Any]],
    point_values: Mapping[str, Mapping[str, float]],
    bootstrap_values: Mapping[str, Mapping[str, np.ndarray]],
    *,
    ci_level: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    view_bases: dict[str, set[str]] = defaultdict(set)
    for arm in arms:
        cost = arm["cost"]
        if cost["status"] != "available":
            continue
        view_id = str(cost["view_id"])
        basis = str(cost["basis"])
        view_bases[view_id].add(basis)
        by_view[view_id].append(arm)

    membership: dict[str, dict[str, Any]] = {
        arm["arm_id"]: {
            "status": "unavailable",
            "view_id": None,
            "basis": None,
            "point_pareto": None,
            "uncertainty_pareto": None,
            "reason": "cost unavailable",
        }
        for arm in arms
    }
    views: list[dict[str, Any]] = []
    alpha = (1.0 - ci_level) / 2.0
    for view_id in sorted(by_view):
        eligible = by_view[view_id]
        audit: list[dict[str, Any]] = []
        point_dominated: set[str] = set()
        uncertainty_dominated: set[str] = set()
        for candidate, challenger in itertools.permutations(eligible, 2):
            candidate_id = candidate["arm_id"]
            challenger_id = challenger["arm_id"]
            candidate_cost_lower = float(
                candidate["cost"]["usd_per_1k_statements_lower"]
            )
            candidate_cost_upper = float(
                candidate["cost"]["usd_per_1k_statements_upper"]
            )
            challenger_cost_lower = float(
                challenger["cost"]["usd_per_1k_statements_lower"]
            )
            challenger_cost_upper = float(
                challenger["cost"]["usd_per_1k_statements_upper"]
            )
            candidate_perf = point_values[candidate_id][PARETO_METRIC]
            challenger_perf = point_values[challenger_id][PARETO_METRIC]
            performance_delta = challenger_perf - candidate_perf
            paired = (
                bootstrap_values[challenger_id][PARETO_METRIC]
                - bootstrap_values[candidate_id][PARETO_METRIC]
            )
            valid = paired[np.isfinite(paired)]
            lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
            point_cost_not_worse = challenger_cost_upper <= candidate_cost_upper
            interval_cost_not_worse = challenger_cost_upper <= candidate_cost_lower
            point = bool(
                point_cost_not_worse
                and performance_delta >= 0
                and (
                    challenger_cost_upper < candidate_cost_upper
                    or performance_delta > 0
                )
            )
            robust = bool(
                interval_cost_not_worse
                and lower >= 0
                and (challenger_cost_upper < candidate_cost_lower or lower > 0)
            )
            if point:
                point_dominated.add(candidate_id)
            if robust:
                uncertainty_dominated.add(candidate_id)
            audit.append(
                {
                    "candidate_arm_id": candidate_id,
                    "challenger_arm_id": challenger_id,
                    "candidate_cost_per_1k_interval": [
                        candidate_cost_lower,
                        candidate_cost_upper,
                    ],
                    "challenger_cost_per_1k_interval": [
                        challenger_cost_lower,
                        challenger_cost_upper,
                    ],
                    "challenger_minus_candidate_cost_per_1k": (
                        challenger_cost_upper - candidate_cost_upper
                    ),
                    "cost_interval_definitely_not_worse": interval_cost_not_worse,
                    "challenger_minus_candidate_performance": performance_delta,
                    "performance_delta_ci95": [float(lower), float(upper)],
                    "point_dominates": point,
                    "uncertainty_dominates": robust,
                }
            )
        eligible_ids = [arm["arm_id"] for arm in eligible]
        point_frontier = [arm_id for arm_id in eligible_ids if arm_id not in point_dominated]
        uncertainty_frontier = [
            arm_id for arm_id in eligible_ids if arm_id not in uncertainty_dominated
        ]
        bases = sorted(view_bases[view_id])
        basis = bases[0] if len(bases) == 1 else "mixed"
        views.append(
            {
                "view_id": view_id,
                "basis": basis,
                "eligible_arm_ids": eligible_ids,
                "point_frontier_arm_ids": point_frontier,
                "uncertainty_frontier_arm_ids": uncertainty_frontier,
                "audit": audit,
            }
        )
        for arm_id in eligible_ids:
            membership[arm_id] = {
                "status": "available",
                "view_id": view_id,
                "basis": next(
                    arm["cost"]["basis"] for arm in eligible if arm["arm_id"] == arm_id
                ),
                "point_pareto": arm_id in point_frontier,
                "uncertainty_pareto": arm_id in uncertainty_frontier,
                "reason": None,
            }
    return (
        {
            "objective_metric": PARETO_METRIC,
            "performance_direction": "higher_is_better",
            "cost_axis": "usd_per_1k_statements_upper",
            "point_rule": (
                "weakly no worse in conservative-upper cost and point performance, "
                "strictly better in at least one"
            ),
            "uncertainty_rule": (
                "challenger cost upper endpoint is no greater than candidate cost lower "
                "endpoint and the paired bootstrap lower CI for challenger-minus-candidate "
                "performance is non-negative, with strict improvement in cost or the CI "
                "lower bound"
            ),
            "views": views,
        },
        membership,
    )


def _parse_metrics_config(value: Any) -> dict[str, Any]:
    context = "spec.metrics"
    obj = _object(value, context)
    _exact_keys(
        obj,
        {
            "log_loss_epsilon",
            "calibration_bin_edges",
            "minimum_valid_bootstrap_fraction",
            "pareto_metric",
        },
        context,
    )
    epsilon = _finite(obj["log_loss_epsilon"], f"{context}.log_loss_epsilon")
    if not 0 < epsilon < 0.5:
        _fail(f"{context}.log_loss_epsilon", "must lie strictly between 0 and 0.5")
    raw_edges = obj["calibration_bin_edges"]
    if not isinstance(raw_edges, list) or len(raw_edges) < 3:
        _fail(f"{context}.calibration_bin_edges", "expected at least three bin edges")
    edges = [
        _finite(value, f"{context}.calibration_bin_edges[{index}]")
        for index, value in enumerate(raw_edges)
    ]
    if edges[0] != 0 or edges[-1] != 1 or any(b <= a for a, b in zip(edges, edges[1:])):
        _fail(
            f"{context}.calibration_bin_edges",
            "edges must increase strictly from exactly 0 to exactly 1",
        )
    minimum = _finite(
        obj["minimum_valid_bootstrap_fraction"],
        f"{context}.minimum_valid_bootstrap_fraction",
    )
    if not 0 < minimum <= 1:
        _fail(f"{context}.minimum_valid_bootstrap_fraction", "outside (0, 1]")
    if obj["pareto_metric"] != PARETO_METRIC:
        _fail(f"{context}.pareto_metric", f"must be {PARETO_METRIC}")
    return {
        "log_loss_epsilon": epsilon,
        "calibration_bin_edges": edges,
        "minimum_valid_bootstrap_fraction": minimum,
        "pareto_metric": PARETO_METRIC,
    }


def _parse_bootstrap(value: Any) -> dict[str, Any]:
    context = "spec.bootstrap"
    obj = _object(value, context)
    _exact_keys(obj, {"seed", "resamples", "ci_level"}, context)
    seed = _integer(obj["seed"], f"{context}.seed")
    resamples = _integer(obj["resamples"], f"{context}.resamples", minimum=1)
    ci_level = _finite(obj["ci_level"], f"{context}.ci_level")
    if not 0 < ci_level < 1:
        _fail(f"{context}.ci_level", "outside (0, 1)")
    if not math.isclose(ci_level, 0.95, rel_tol=0, abs_tol=1e-12):
        _fail(f"{context}.ci_level", "the canonical artifact requires 0.95")
    return {"seed": seed, "resamples": resamples, "ci_level": ci_level}


def _parse_implementation(value: Any, context: str) -> dict[str, Any]:
    obj = _object(value, context)
    _exact_keys(
        obj,
        {
        "implementation",
        "implementation_digest",
        "training_data_sha256",
        "environment",
        "notes",
        },
        context,
    )
    training = obj["training_data_sha256"]
    if training is not None:
        training = _sha256_text(training, f"{context}.training_data_sha256")
    notes = obj["notes"]
    if notes is not None:
        notes = _nonempty_string(notes, f"{context}.notes")
    return {
        "implementation": _nonempty_string(obj["implementation"], f"{context}.implementation"),
        "implementation_digest": _nonempty_string(
            obj["implementation_digest"], f"{context}.implementation_digest"
        ),
        "training_data_sha256": training,
        "environment": _nonempty_string(obj["environment"], f"{context}.environment"),
        "notes": notes,
    }

def _parse_released_label_audit(
    value: Any,
    *,
    n_evaluable: int,
    n_positive: int,
    strict_statement_ids: Sequence[str],
    strict_labels: np.ndarray,
    released_statement_ids: Sequence[str],
    context: str,
) -> dict[str, Any]:
    obj = _object(value, context)
    _exact_keys(
        obj,
        {
            "released_label_rule",
            "strict_e0_rule",
            "released",
            "strict_e0",
            "released_negative_assumption",
        },
        context,
    )
    released_rule = _nonempty_string(
        obj["released_label_rule"], f"{context}.released_label_rule"
    )
    strict_rule = _nonempty_string(obj["strict_e0_rule"], f"{context}.strict_e0_rule")
    released = _object(obj["released"], f"{context}.released")
    _exact_keys(released, {"statements", "positive", "negative"}, f"{context}.released")
    released_counts = {
        "statements": _integer(released["statements"], f"{context}.released.statements", minimum=1),
        "positive": _integer(released["positive"], f"{context}.released.positive"),
        "negative": _integer(released["negative"], f"{context}.released.negative"),
    }
    if released_counts != {
        "statements": n_evaluable,
        "positive": n_positive,
        "negative": n_evaluable - n_positive,
    }:
        _fail(f"{context}.released", "counts differ from the released binary target")

    strict_e0 = _object(obj["strict_e0"], f"{context}.strict_e0")
    _exact_keys(
        strict_e0,
        {"resolved", "positive", "negative", "unresolved", "ordered_statement_id_sha256"},
        f"{context}.strict_e0",
    )
    strict_counts = {
        "resolved": _integer(strict_e0["resolved"], f"{context}.strict_e0.resolved", minimum=1),
        "positive": _integer(strict_e0["positive"], f"{context}.strict_e0.positive"),
        "negative": _integer(strict_e0["negative"], f"{context}.strict_e0.negative"),
        "unresolved": _integer(strict_e0["unresolved"], f"{context}.strict_e0.unresolved"),
        "ordered_statement_id_sha256": _sha256_text(
            strict_e0["ordered_statement_id_sha256"],
            f"{context}.strict_e0.ordered_statement_id_sha256",
        ),
    }
    observed_strict = {
        "resolved": len(strict_statement_ids),
        "positive": int(np.sum(strict_labels)),
        "negative": len(strict_statement_ids) - int(np.sum(strict_labels)),
        "unresolved": n_evaluable - len(strict_statement_ids),
        "ordered_statement_id_sha256": ordered_statement_id_sha256(strict_statement_ids),
    }
    if strict_counts != observed_strict:
        _fail(f"{context}.strict_e0", "counts or identity differ from strict E0 gold")

    assumption = _object(
        obj["released_negative_assumption"],
        f"{context}.released_negative_assumption",
    )
    _exact_keys(
        assumption,
        {"statements", "share_of_released_negatives", "ordered_statement_id_sha256"},
        f"{context}.released_negative_assumption",
    )
    strict_ids = set(strict_statement_ids)
    unresolved_ids = [value for value in released_statement_ids if value not in strict_ids]
    assumption_statements = _integer(
        assumption["statements"],
        f"{context}.released_negative_assumption.statements",
    )
    assumption_share = _finite(
        assumption["share_of_released_negatives"],
        f"{context}.released_negative_assumption.share_of_released_negatives",
    )
    assumption_digest = _sha256_text(
        assumption["ordered_statement_id_sha256"],
        f"{context}.released_negative_assumption.ordered_statement_id_sha256",
    )
    expected_share = len(unresolved_ids) / released_counts["negative"]
    if (
        assumption_statements != len(unresolved_ids)
        or not math.isclose(assumption_share, expected_share, rel_tol=0, abs_tol=1e-15)
        or assumption_digest != ordered_statement_id_sha256(unresolved_ids)
    ):
        _fail(
            f"{context}.released_negative_assumption",
            "does not identify the exact strict-unresolved released-negative cohort",
        )
    return {
        "released_label_rule": released_rule,
        "strict_e0_rule": strict_rule,
        "released": released_counts,
        "strict_e0": strict_counts,
        "released_negative_assumption": {
            "statements": assumption_statements,
            "share_of_released_negatives": assumption_share,
            "ordered_statement_id_sha256": assumption_digest,
        },
    }

def _parse_excluded_arms(
    value: Any,
    *,
    evaluated_arm_ids: set[str],
    context: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        _fail(context, "expected an array")
    output: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(value):
        item_context = f"{context}[{index}]"
        obj = _object(raw, item_context)
        _exact_keys(
            obj,
            {
                "arm_id",
                "label",
                "family",
                "status",
                "reason",
                "required_artifact",
                "provenance",
            },
            item_context,
        )
        arm_id = _identifier(obj["arm_id"], f"{item_context}.arm_id")
        if arm_id in seen_ids:
            _fail(item_context, f"duplicate excluded arm_id {arm_id!r}")
        if arm_id in evaluated_arm_ids:
            _fail(item_context, f"excluded arm_id {arm_id!r} is also evaluated")
        seen_ids.add(arm_id)
        family = obj["family"]
        if family not in {"paper", "current", "llm"}:
            _fail(f"{item_context}.family", "expected paper, current, or llm")
        if obj["status"] != "excluded":
            _fail(f"{item_context}.status", "must be excluded")
        output.append(
            {
                "arm_id": arm_id,
                "label": _nonempty_string(obj["label"], f"{item_context}.label"),
                "family": str(family),
                "status": "excluded",
                "reason": _nonempty_string(obj["reason"], f"{item_context}.reason"),
                "required_artifact": _nonempty_string(
                    obj["required_artifact"], f"{item_context}.required_artifact"
                ),
                "provenance": _nonempty_string(
                    obj["provenance"], f"{item_context}.provenance"
                ),
            }
        )
    return output


def _evaluate_metric_arms(
    prepared: Sequence[Mapping[str, Any]],
    *,
    labels: np.ndarray,
    folds: np.ndarray,
    bootstrap: Mapping[str, Any],
    metrics_config: Mapping[str, Any],
    contract: Mapping[str, Any],
    context: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, float]],
    dict[str, dict[str, np.ndarray]],
]:
    """Evaluate one fixed panel once; callers attach scope-specific metadata."""

    resamples = int(bootstrap["resamples"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    boot_values: dict[str, dict[str, np.ndarray]] = {}
    metric_names_by_arm: dict[str, tuple[str, ...]] = {}
    for arm in prepared:
        names = BOOTSTRAP_METRICS + (
            THRESHOLD_METRICS if arm["threshold"]["status"] == "available" else ()
        )
        arm_id = str(arm["arm_id"])
        metric_names_by_arm[arm_id] = names
        boot_values[arm_id] = {
            metric: np.full(resamples, np.nan, dtype=float) for metric in names
        }
    for bootstrap_index in range(resamples):
        weights = _stratified_paired_bootstrap_weights(rng, folds)
        for arm in prepared:
            arm_id = str(arm["arm_id"])
            values = arm["computer"].values(weights)
            for metric in metric_names_by_arm[arm_id]:
                boot_values[arm_id][metric][bootstrap_index] = values[metric]

    output_arms: list[dict[str, Any]] = []
    point_values: dict[str, dict[str, float]] = {}
    for arm in prepared:
        arm_id = str(arm["arm_id"])
        point_values[arm_id] = dict(arm["points"])
        estimates: dict[str, dict[str, Any]] = {}
        for metric in metric_names_by_arm[arm_id]:
            estimates[metric] = _estimate(
                arm["points"][metric],
                boot_values[arm_id][metric],
                method=_metric_method(
                    metric,
                    epsilon=float(metrics_config["log_loss_epsilon"]),
                    edge_count=len(metrics_config["calibration_bin_edges"]),
                ),
                ci_level=float(bootstrap["ci_level"]),
                requested_resamples=resamples,
                minimum_valid_fraction=float(
                    metrics_config["minimum_valid_bootstrap_fraction"]
                ),
                context=f"{context}.{arm_id}.{metric}",
            )
        trapezoidal = {
            **estimates["fold_mean_trapezoidal_pr_auc"],
            "fold_estimates": arm["fold_rows"],
            "fold_population_sd": arm["fold_sd"],
        }
        if arm["threshold"]["status"] == "available":
            threshold_metrics = {
                **arm["threshold"],
                "confusion": arm["threshold_confusion"],
                "metrics": {
                    key.removeprefix("threshold_"): estimates[key]
                    for key in THRESHOLD_METRICS
                },
            }
        else:
            threshold_metrics = {
                **arm["threshold"],
                "confusion": None,
                "metrics": None,
            }
        output_arms.append(
            {
                "arm_id": arm_id,
                "label": arm["label"],
                "family": arm["family"],
                "coverage": {
                    "eligible": len(labels),
                    "predicted": len(labels),
                    "invalid": 0,
                    "fraction": 1.0,
                },
                "metrics": {
                    "fold_mean_trapezoidal_pr_auc": trapezoidal,
                    "pooled_average_precision": estimates["pooled_average_precision"],
                    "auroc": estimates["auroc"],
                    "brier": estimates["brier"],
                    "log_loss": estimates["log_loss"],
                    "calibration": {
                        "ece": estimates["calibration_ece"],
                        "intercept": estimates["calibration_intercept"],
                        "slope": estimates["calibration_slope"],
                        "intercept_abs_error": estimates[
                            "calibration_intercept_abs_error"
                        ],
                        "slope_abs_error": estimates["calibration_slope_abs_error"],
                        "reliability_bins": _reliability_bins(
                            labels,
                            arm["scores"],
                            np.asarray(metrics_config["calibration_bin_edges"], dtype=float),
                        ),
                    },
                    "threshold": threshold_metrics,
                },
            }
        )

    comparisons: list[dict[str, Any]] = []
    for arm_a, arm_b in itertools.combinations(prepared, 2):
        shared_metrics = list(PRIMARY_METRICS)
        if (
            arm_a["threshold"]["status"] == "available"
            and arm_b["threshold"]["status"] == "available"
        ):
            shared_metrics.extend(THRESHOLD_METRICS)
        for metric in shared_metrics:
            delta_values = (
                boot_values[str(arm_b["arm_id"])][metric]
                - boot_values[str(arm_a["arm_id"])][metric]
            )
            delta = _estimate(
                arm_b["points"][metric] - arm_a["points"][metric],
                delta_values,
                method=(
                    "paired statement bootstrap percentile CI on b minus a, stratified "
                    "independently within every frozen fold with one paired weight vector "
                    "across arms; conditional on frozen fits, folds, and predictions; "
                    "models are not refit or rerun"
                ),
                ci_level=float(bootstrap["ci_level"]),
                requested_resamples=resamples,
                minimum_valid_fraction=float(
                    metrics_config["minimum_valid_bootstrap_fraction"]
                ),
                context=(
                    f"{context}.comparison.{arm_a['arm_id']}."
                    f"{arm_b['arm_id']}.{metric}"
                ),
            )
            comparisons.append(
                {
                    "a_arm_id": arm_a["arm_id"],
                    "b_arm_id": arm_b["arm_id"],
                    "metric": metric,
                    "direction": "b_minus_a",
                    "better_when": _better_when(metric),
                    "contract": dict(contract),
                    "delta": delta,
                    "resamples": resamples,
                    "method": (
                        "paired statement bootstrap percentile CI stratified independently "
                        "within every frozen fold with one paired weight vector across arms; "
                        "conditional on frozen fits, folds, and predictions; models are not "
                        "refit or rerun"
                    ),
                }
            )
    return output_arms, comparisons, point_values, boot_values

def _build_substrate(
    value: Any,
    *,
    index: int,
    base: Path,
    bootstrap: Mapping[str, Any],
    metrics_config: Mapping[str, Any],
) -> dict[str, Any]:
    context = f"spec.substrates[{index}]"
    obj = _object(value, context)
    _exact_keys(
        obj,
        {
            "substrate_id",
            "lane",
            "label",
            "contract",
            "analysis_scope",
            "released_label_audit",
            "substrate_manifest",
            "gold",
            "strict_e0_resolved_gold",
            "arms",
            "excluded_arms",
        },
        context,
    )
    substrate_id = _identifier(obj["substrate_id"], f"{context}.substrate_id")
    if substrate_id != CANONICAL_PANEL_IDS[index]:
        _fail(
            f"{context}.substrate_id",
            f"expected canonical panel {CANONICAL_PANEL_IDS[index]}",
        )
    if obj["lane"] != "paper":
        _fail(f"{context}.lane", "canonical comparison only accepts the paper lane")
    label = _nonempty_string(obj["label"], f"{context}.label")
    if obj["analysis_scope"] != "primary":
        _fail(f"{context}.analysis_scope", "canonical comparison only accepts primary scope")
    contract = _parse_contract(obj["contract"], f"{context}.contract")
    substrate_manifest_display, _, _ = _verify_file(
        _object(obj["substrate_manifest"], f"{context}.substrate_manifest"),
        base=base,
        context=f"{context}.substrate_manifest",
        expected_contract_digest=contract["substrate_sha256"],
    )
    gold_display, gold_path, _ = _verify_file(
        _object(obj["gold"], f"{context}.gold"),
        base=base,
        context=f"{context}.gold",
        expected_contract_digest=contract["gold_sha256"],
    )
    statement_ids, labels, folds = _load_gold(
        gold_path, contract=contract, context=f"{context}.gold_rows"
    )
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if unique_folds != list(range(10)):
        _fail(
            f"{context}.gold_rows",
            "canonical paper panels require the frozen StratifiedKFold IDs 0 through 9",
        )
    strict_display, strict_path, strict_sha256 = _verify_file(
        _object(
            obj["strict_e0_resolved_gold"],
            f"{context}.strict_e0_resolved_gold",
        ),
        base=base,
        context=f"{context}.strict_e0_resolved_gold",
    )
    raw_audit = _object(obj["released_label_audit"], f"{context}.released_label_audit")
    raw_strict_audit = _object(
        raw_audit.get("strict_e0"), f"{context}.released_label_audit.strict_e0"
    )
    strict_evaluation_sha256 = _sha256_text(
        raw_strict_audit.get("ordered_statement_id_sha256"),
        f"{context}.released_label_audit.strict_e0.ordered_statement_id_sha256",
    )
    strict_contract = {
        "prediction_unit": PREDICTION_UNIT,
        "gold_rule": STRICT_GOLD_RULE,
        "substrate_sha256": contract["substrate_sha256"],
        "gold_sha256": strict_sha256,
        "evaluation_set_sha256": strict_evaluation_sha256,
    }
    strict_statement_ids, strict_labels, strict_folds = _load_gold(
        strict_path,
        contract=strict_contract,
        context=f"{context}.strict_e0_resolved_gold_rows",
    )
    if sorted(int(value) for value in np.unique(strict_folds)) != list(range(10)):
        _fail(
            f"{context}.strict_e0_resolved_gold_rows",
            "strict sensitivity requires the frozen fold IDs 0 through 9",
        )
    released_index = {statement_id: index for index, statement_id in enumerate(statement_ids)}
    strict_indices: list[int] = []
    for strict_index, statement_id in enumerate(strict_statement_ids):
        if statement_id not in released_index:
            _fail(
                f"{context}.strict_e0_resolved_gold_rows",
                f"statement {statement_id!r} is outside released gold",
            )
        released_position = released_index[statement_id]
        if bool(strict_labels[strict_index]) != bool(labels[released_position]):
            _fail(
                f"{context}.strict_e0_resolved_gold_rows",
                f"strict/released label differs for {statement_id!r}",
            )
        strict_indices.append(released_position)
    if strict_indices != sorted(strict_indices):
        _fail(
            f"{context}.strict_e0_resolved_gold_rows",
            "strict rows are not an order-preserving released-gold subset",
        )
    strict_index_set = set(strict_indices)
    unresolved_indices = [
        position for position in range(len(statement_ids)) if position not in strict_index_set
    ]
    if any(bool(labels[position]) for position in unresolved_indices):
        _fail(
            f"{context}.strict_e0_resolved_gold_rows",
            "strict-unresolved cohort contains a released positive label",
        )
    released_label_audit = _parse_released_label_audit(
        raw_audit,
        n_evaluable=len(statement_ids),
        n_positive=int(np.sum(labels)),
        strict_statement_ids=strict_statement_ids,
        strict_labels=strict_labels,
        released_statement_ids=statement_ids,
        context=f"{context}.released_label_audit",
    )
    raw_arms = obj["arms"]
    if not isinstance(raw_arms, list) or not raw_arms:
        _fail(f"{context}.arms", "expected a non-empty array")

    prepared: list[dict[str, Any]] = []
    seen_arm_ids: set[str] = set()
    families: set[str] = set()
    for arm_index, raw_arm in enumerate(raw_arms):
        arm_context = f"{context}.arms[{arm_index}]"
        arm_obj = _object(raw_arm, arm_context)
        _exact_keys(
            arm_obj,
            {
                "arm_id",
                "label",
                "family",
                "predictions",
                "implementation",
                "threshold",
                "cost",
            },
            arm_context,
        )
        arm_id = _identifier(arm_obj["arm_id"], f"{arm_context}.arm_id")
        if arm_id in seen_arm_ids:
            _fail(arm_context, f"duplicate arm_id {arm_id!r}")
        seen_arm_ids.add(arm_id)
        family = arm_obj["family"]
        if family not in {"paper", "current", "llm"}:
            _fail(f"{arm_context}.family", "expected paper, current, or llm")
        families.add(str(family))
        prediction_display, prediction_path, prediction_digest = _verify_file(
            _object(arm_obj["predictions"], f"{arm_context}.predictions"),
            base=base,
            context=f"{arm_context}.predictions",
        )
        scores = _load_predictions(
            prediction_path,
            statement_ids=statement_ids,
            context=f"{arm_context}.prediction_rows",
        )
        implementation = _parse_implementation(
            arm_obj["implementation"], f"{arm_context}.implementation"
        )
        threshold = _parse_threshold(
            arm_obj["threshold"], base=base, context=f"{arm_context}.threshold"
        )
        cost = _parse_cost(
            arm_obj["cost"],
            base=base,
            statement_ids=statement_ids,
            expected_projection=(
                "all_executions"
                if substrate_id == CANONICAL_PANEL_IDS[0]
                else "observed_execution_subset"
            ),
            context=f"{arm_context}.cost",
        )
        computer = _MetricComputer(
            labels,
            scores,
            folds,
            log_loss_epsilon=float(metrics_config["log_loss_epsilon"]),
            calibration_edges=np.asarray(metrics_config["calibration_bin_edges"], dtype=float),
            threshold=threshold["value"] if threshold["status"] == "available" else None,
        )
        points, fold_rows, fold_sd, threshold_confusion = computer.point()
        prepared.append(
            {
                "arm_id": arm_id,
                "label": _nonempty_string(arm_obj["label"], f"{arm_context}.label"),
                "family": family,
                "scores": scores,
                "computer": computer,
                "points": points,
                "fold_rows": fold_rows,
                "fold_sd": fold_sd,
                "threshold": threshold,
                "threshold_confusion": threshold_confusion,
                "cost": cost,
                "provenance": {
                    **implementation,
                    "predictions_path": prediction_display,
                    "predictions_sha256": prediction_digest,
                },
            }
        )
    missing_families = {"paper", "current", "llm"} - families
    if missing_families:
        _fail(f"{context}.arms", f"missing required families {sorted(missing_families)}")
    cost_comparability_ids = {
        str(arm["cost"]["cost_comparability_id"])
        for arm in prepared
        if arm["cost"]["status"] == "available"
    }
    if len(cost_comparability_ids) > 1:
        _fail(
            f"{context}.arms",
            "available costs do not share one cost-comparability basis",
        )
    excluded_arms = _parse_excluded_arms(
        obj["excluded_arms"],
        evaluated_arm_ids=seen_arm_ids,
        context=f"{context}.excluded_arms",
    )

    n = len(statement_ids)
    output_arms, comparisons, point_values, boot_values = _evaluate_metric_arms(
        prepared,
        labels=labels,
        folds=folds,
        bootstrap=bootstrap,
        metrics_config=metrics_config,
        contract=contract,
        context=context,
    )
    unresolved_mask = np.ones(n, dtype=bool)
    unresolved_mask[np.asarray(strict_indices, dtype=int)] = False
    resolved_mask = ~unresolved_mask
    for arm in prepared:
        arm_id = arm["arm_id"]
        output = next(value for value in output_arms if value["arm_id"] == arm_id)
        output["contract"] = dict(contract)
        output["cost"] = arm["cost"]
        output["pareto"] = None
        output["provenance"] = arm["provenance"]
        if arm["threshold"]["status"] == "available":
            threshold_value = float(arm["threshold"]["value"])
            strata: dict[str, Any] = {}
            for stratum_name, mask in (
                ("strict_e0_resolved", resolved_mask),
                ("released_negative_assumption", unresolved_mask),
            ):
                _, confusion = _threshold_values(
                    labels[mask],
                    arm["scores"][mask],
                    np.ones(int(np.sum(mask)), dtype=float),
                    threshold_value,
                )
                strata[stratum_name] = {
                    "statements": int(np.sum(mask)),
                    **confusion,
                    "errors": confusion["fp"] + confusion["fn"],
                }
            output["released_label_error_strata"] = strata
        else:
            output["released_label_error_strata"] = None

    strict_prepared: list[dict[str, Any]] = []
    strict_positions = np.asarray(strict_indices, dtype=int)
    for arm in prepared:
        strict_scores = arm["scores"][strict_positions]
        strict_computer = _MetricComputer(
            strict_labels,
            strict_scores,
            strict_folds,
            log_loss_epsilon=float(metrics_config["log_loss_epsilon"]),
            calibration_edges=np.asarray(
                metrics_config["calibration_bin_edges"], dtype=float
            ),
            threshold=(
                arm["threshold"]["value"]
                if arm["threshold"]["status"] == "available"
                else None
            ),
        )
        strict_points, strict_fold_rows, strict_fold_sd, strict_confusion = (
            strict_computer.point()
        )
        strict_prepared.append(
            {
                "arm_id": arm["arm_id"],
                "label": arm["label"],
                "family": arm["family"],
                "scores": strict_scores,
                "computer": strict_computer,
                "points": strict_points,
                "fold_rows": strict_fold_rows,
                "fold_sd": strict_fold_sd,
                "threshold": arm["threshold"],
                "threshold_confusion": strict_confusion,
            }
        )
    strict_output_arms, strict_comparisons, _strict_points, _strict_boot = (
        _evaluate_metric_arms(
            strict_prepared,
            labels=strict_labels,
            folds=strict_folds,
            bootstrap=bootstrap,
            metrics_config=metrics_config,
            contract=strict_contract,
            context=f"{context}.strict_e0_resolved_sensitivity",
        )
    )
    for arm in strict_output_arms:
        arm["contract"] = dict(strict_contract)

    pareto, memberships = _build_pareto(
        output_arms,
        point_values,
        boot_values,
        ci_level=float(bootstrap["ci_level"]),
    )
    for arm in output_arms:
        arm["pareto"] = memberships[arm["arm_id"]]
    pr_summary_contract = {
        "fold_mean_trapezoidal_pr_auc": (
            "arithmetic mean of per-fold sklearn precision_recall_curve + "
            "auc(recall, precision)"
        ),
        "pooled_average_precision": "sklearn average_precision_score over all statements",
        "fold_count": len(unique_folds),
    }
    return {
        "substrate_id": substrate_id,
        "lane": "paper",
        "label": label,
        "analysis_scope": "primary",
        "released_label_audit": released_label_audit,
        "contract": dict(contract),
        "substrate_manifest_path": substrate_manifest_display,
        "gold_path": gold_display,
        "positive_class": POSITIVE_CLASS,
        "n_evaluable": n,
        "n_positive": int(np.sum(labels)),
        "n_negative": int(np.sum(~labels)),
        "pr_summary_contract": pr_summary_contract,
        "arms": output_arms,
        "excluded_arms": excluded_arms,
        "comparisons": comparisons,
        "pareto": pareto,
        "strict_e0_resolved_sensitivity": {
            "analysis_scope": "fixed_resolved_only_sensitivity",
            "selection_rule": (
                "exclude exactly the released-paper negatives that remain unresolved "
                "under the strict E0 evidence-pair rule"
            ),
            "contract": dict(strict_contract),
            "gold_path": strict_display,
            "n_evaluable": len(strict_statement_ids),
            "n_positive": int(np.sum(strict_labels)),
            "n_negative": int(np.sum(~strict_labels)),
            "excluded_unresolved": len(unresolved_indices),
            "pr_summary_contract": {
                **pr_summary_contract,
                "fold_count": len(np.unique(strict_folds)),
            },
            "arms": strict_output_arms,
            "comparisons": strict_comparisons,
        },
    }


def build_artifact(spec_path: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    spec, spec_bytes = _read_json_bytes(spec_path)
    _exact_keys(
        spec,
        {
            "artifact_kind",
            "frozen_at",
            "bootstrap",
            "scorer_registry",
            "metrics",
            "substrates",
        },
        "spec",
    )
    if spec["artifact_kind"] != SPEC_KIND:
        _fail("spec.artifact_kind", f"expected {SPEC_KIND}")
    frozen_at = _timestamp(spec["frozen_at"], "spec.frozen_at")
    bootstrap = _parse_bootstrap(spec["bootstrap"])
    scorer_registry = _verify_scorer_registry(
        spec["scorer_registry"], base=spec_path.parent
    )
    metrics_config = _parse_metrics_config(spec["metrics"])
    raw_substrates = spec["substrates"]
    if not isinstance(raw_substrates, list) or len(raw_substrates) != len(CANONICAL_PANEL_IDS):
        _fail(
            "spec.substrates",
            f"expected exactly the canonical panels {list(CANONICAL_PANEL_IDS)}",
        )
    substrates = [
        _build_substrate(
            value,
            index=index,
            base=spec_path.parent,
            bootstrap=bootstrap,
            metrics_config=metrics_config,
        )
        for index, value in enumerate(raw_substrates)
    ]
    reader_arms = {arm["arm_id"]: arm for arm in substrates[1]["arms"]}
    for all_arm in substrates[0]["arms"]:
        if all_arm["family"] != "llm":
            continue
        arm_id = all_arm["arm_id"]
        reader_arm = reader_arms.get(arm_id)
        if reader_arm is None or reader_arm["family"] != "llm":
            _fail("spec.substrates", f"LLM arm {arm_id!r} is not shared by both panels")
        all_cost = all_arm["cost"]
        reader_cost = reader_arm["cost"]
        if all_cost["status"] != "available" or reader_cost["status"] != "available":
            _fail("spec.substrates", f"LLM arm {arm_id!r} lacks a complete panel cost")
        if (
            all_cost["cost_comparability_id"]
            != reader_cost["cost_comparability_id"]
            or all_cost["pricing"] != reader_cost["pricing"]
            or all_cost["shared_run_id"] != reader_cost["shared_run_id"]
        ):
            _fail(
                "spec.substrates",
                f"LLM arm {arm_id!r} panel costs disagree on pricing or shared run",
            )
    module_digest = sha256_file(Path(__file__).resolve())
    return {
        "artifact_kind": ARTIFACT_KIND,
        "frozen_at": frozen_at,
        "provenance": {
            "metrics_code_sha256": module_digest,
            "source_manifest_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "source_manifest_path": spec_path.name,
            "scorer_registry": scorer_registry,
            "bootstrap_seed": bootstrap["seed"],
            "bootstrap_resamples": bootstrap["resamples"],
            "bootstrap_rng": "numpy.random.Generator(PCG64)",
            "ci_level": bootstrap["ci_level"],
            "log_loss_epsilon": metrics_config["log_loss_epsilon"],
            "calibration_bin_edges": metrics_config["calibration_bin_edges"],
            "minimum_valid_bootstrap_fraction": metrics_config[
                "minimum_valid_bootstrap_fraction"
            ],
            "evaluation_set_digest_method": (
                "SHA-256 over ordered canonical JSONL rows "
                '{"statement_id":<id>} with UTF-8 and LF'
            ),
            "runtime": {
                "python": os.sys.version.split()[0],
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
        "substrates": substrates,
    }


def write_artifact(artifact: Mapping[str, Any], output_path: Path, *, force: bool = False) -> None:
    output_path = output_path.resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force intentionally")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output_path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="frozen evaluation specification")
    parser.add_argument("--output", type=Path, required=True, help="viewer metrics artifact")
    parser.add_argument("--force", action="store_true", help="replace an existing output atomically")
    args = parser.parse_args(argv)
    try:
        artifact = build_artifact(args.spec)
        write_artifact(artifact, args.output, force=args.force)
    except (ContractError, FileExistsError) as exc:
        parser.error(str(exc))
    print(
        f"wrote {args.output} with {len(artifact['substrates'])} substrate panel(s); "
        f"sha256={sha256_file(args.output.resolve())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
