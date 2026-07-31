"""Paid runner with exhaustive preflight and a ready-before-token boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from indra_belief.corpus.cost import price_for
from indra_belief.spend_guard import (
    PROVIDER_OUTPUT_TOKEN_OVERSHOOT,
    GuardedModelClient,
    SpendCapReached,
    SpendGuard,
    SpendGuardError,
    SpendGuardStop,
    classify_provider_failure,
    conservative_prompt_token_bound,
)

from .contracts import (
    Action,
    ContractError,
    RunPlan,
    Stage,
    canonical_json_line,
    load_run_plan,
)
from .replay import (
    VALID_CONFIDENCES,
    VALID_VERDICTS,
    AppendLog,
    ReplayIndex,
    ResumeState,
    error_row,
    execution_identity,
    expected_execution_id,
    load_resume,
    parse_response,
    result_row,
    score_execution,
    source_key,
    validate_row,
    verdict_score,
)


class RunnerError(ContractError): pass


class ActionCapReached(SpendCapReached): pass


class ActionDeadlineExceeded(TimeoutError): pass


class InvalidModelOutput(ValueError): pass


# One message for both the persisted error row and the raise, so the ALERT
# detail's message_sha256 is stable across the two sites.
_INVALID_OUTPUT_MESSAGE = (
    "provider response is not an on-grid (verdict, confidence) pair"
)


# Per-source cap on unparseable model output, counted cumulatively across
# restarts from the persisted rows (see _run_source). It exists to stop a model
# that CANNOT do the task from burning the whole transport-retry budget, and is
# deliberately separate from, and much tighter than, action.max_attempts.
#
# Raised 2 -> 5 on 2026-07-31 against measured evidence from the verdict-only
# run. That prompt asks for two closed-enum fields, and every one of the four
# models occasionally fills them in the wrong order — e.g. gemma-4-e2b returned
# `{"verdict": "medium", "confidence": "medium"}`. The rate is 0.057%-0.097%
# across all four arms, and it is sporadic rather than systematic: re-issuing
# the SAME request for the execution that wedged returned a valid verdict on 2
# of 3 tries. At a per-request failure probability of about a third for a hard
# input, a cap of 2 stops the arm 11% of the time, which is what happened —
# gemma_26b finished carrying 19 such errors purely by luck. Five leaves ~0.4%.
#
# This does NOT loosen the systematic case: a model that never emits a valid
# verdict still stops after five attempts on its first source, long before
# max_attempts, and the failure is still terminal rather than silently scored.
INVALID_MODEL_OUTPUT_LIMIT = 5


@dataclass(frozen=True)
class ActionStatus:
    action_id: str
    status: str
    completed: int
    total: int
    attempts: int


@dataclass(frozen=True)
class PlanStatus:
    status: str
    next_action_id: str | None
    ready_action_ids: tuple[str, ...]
    actions: tuple[ActionStatus, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "next_action_id": self.next_action_id,
            "ready_action_ids": list(self.ready_action_ids),
            "actions": [vars(row) for row in self.actions],
        }


@dataclass(frozen=True)
class RunSummary:
    status: str
    action_id: str
    completed_this_run: int
    completed_total: int
    total: int
    verdicts: Mapping[str, int]
    spend_guard: Mapping[str, Any]
    failure: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action_id": self.action_id,
            "completed_this_run": self.completed_this_run,
            "completed_total": self.completed_total,
            "total": self.total,
            "verdicts": dict(self.verdicts),
            "spend_guard": dict(self.spend_guard),
            "failure": dict(self.failure) if self.failure else None,
        }


def _scopes(plan: RunPlan) -> dict[str, ReplayIndex]:
    replay = ReplayIndex.load(plan)
    bases: dict[str, ReplayIndex] = {}
    result: dict[str, ReplayIndex] = {}
    for action in plan.actions:
        base = bases.get(action.workload)
        if base is None:
            base = replay.for_workload(action.workload)
            bases[action.workload] = base
        result[action.id] = base.select(action.execution_keys)
    return result


def inspect_plan(plan: RunPlan | str | Path) -> PlanStatus:
    loaded = load_run_plan(plan) if isinstance(plan, (str, Path)) else plan
    scopes = _scopes(loaded)
    statuses: list[ActionStatus] = []
    state_by_id: dict[str, str] = {}
    for action in loaded.actions:
        stage = loaded.stage_by_id[action.stage_id]
        resume = load_resume(
            action.output,
            index=scopes[action.id],
            action=action,
            model=stage.model,
            provider_model_id=stage.provider_model_id,
        )
        state_by_id[action.id] = resume.status
        statuses.append(ActionStatus(action.id, resume.status, len(resume.done),
                                     len(scopes[action.id].executions), len(resume.rows)))
    ready_actions = loaded.ready_actions(state_by_id)
    if not ready_actions:
        overall = "complete"
    elif all(row.status == "pending" for row in statuses):
        overall = "pending"
    else:
        overall = "partial"
    return PlanStatus(
        overall,
        ready_actions[0].id if ready_actions else None,
        tuple(action.id for action in ready_actions),
        tuple(statuses),
    )


class _ActionGuard:
    def __init__(self, guard: SpendGuard, action: Action) -> None:
        self.guard, self.action = guard, action
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.guard, name)

    def _commitment(self) -> Decimal:
        return self.guard.commitment(run_id=self.action.run_id)

    def summary(self) -> dict[str, Any]:
        commitment = self._commitment()
        return {
            **self.guard.summary(),
            "action_id": self.action.id,
            "action_cap_usd": format(self.action.cap_usd, "f"),
            "action_accounted_commitment_usd": format(commitment, "f"),
            "action_authorization_remaining_usd": format(
                self.action.cap_usd - commitment, "f"
            ),
        }

    def reserve_call(self, *, provider_model_id: str, kind: str,
                     max_output_tokens: int, system: str, messages: list[dict],
                     **kwargs: Any) -> Any:
        price = price_for(provider_model_id)
        if price is None:
            raise SpendGuardError(f"provider model {provider_model_id!r} has no price")
        reserved_input = max(
            self.guard.provider_input_token_maximum,
            conservative_prompt_token_bound(system, messages),
        )
        reservation = (
            Decimal(reserved_input) * Decimal(str(price[0]))
            + Decimal(max_output_tokens + PROVIDER_OUTPUT_TOKEN_OVERSHOOT)
            * Decimal(str(price[1]))
        ) / Decimal("1000000")
        with self._lock:
            if self._commitment() + reservation > self.action.cap_usd:
                raise ActionCapReached(
                    f"next call maximum reservation would exceed action {self.action.id!r} cap"
                )
            return self.guard.reserve_call(
                provider_model_id=provider_model_id,
                kind=kind,
                max_output_tokens=max_output_tokens,
                system=system,
                messages=messages,
                **kwargs,
            )


@dataclass
class PreparedRun:
    plan: RunPlan
    action: Action
    stage: Stage
    index: ReplayIndex
    output: AppendLog
    guard: SpendGuard
    action_guard: _ActionGuard
    resume: ResumeState
    prepared_at: float
    closed: bool = False
    started: bool = False

    @property
    def deadline(self) -> float:
        return self.prepared_at + self.action.deadline_seconds

    def readiness(self) -> dict[str, Any]:
        snapshot = self.guard.resume_reconciliation()
        return {
            "status": "ready_for_bearer_token",
            "plan_sha256": self.plan.sha256,
            "action_id": self.action.id,
            "run_id": self.action.run_id,
            "model": self.stage.model,
            "provider_model_id": self.stage.provider_model_id,
            "workload": self.action.workload,
            "total": len(self.index.executions),
            "completed": len(self.resume.done),
            "pending": len(self.index.executions) - len(self.resume.done),
            "ledger_id": snapshot["ledger_id"],
            "ledger_sequence": snapshot["sequence"],
            "caps_usd": {
                "action": format(self.action.cap_usd, "f"),
                "stage": format(self.stage.cap_usd, "f"),
                "global": format(self.plan.global_cap_usd, "f"),
            },
            "deadline_seconds": self.action.deadline_seconds,
            "max_attempts": self.action.max_attempts,
            "workers": self.action.workers,
            "provider_calls_started_during_preflight": 0,
            "token_read": False,
        }

    def assert_paid_surface(self) -> None:
        self.plan.assert_current()
        self.index.assert_current()
        self.output.assert_current()

    def close(self) -> None:
        if self.closed:
            return
        error: BaseException | None = None
        try:
            self.guard.close()
        except BaseException as exc:
            error = exc
        try:
            self.output.close()
        except BaseException as exc:
            error = error or exc
        self.closed = True
        if error:
            raise error

    def __enter__(self) -> "PreparedRun":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def _relevant_attempts(prepared: PreparedRun) -> list[Mapping[str, Any]]:
    expected = {
        expected_execution_id(source, action=prepared.action, model=prepared.stage.model)
        for source in prepared.index.executions
    }
    attempts = prepared.guard.resume_reconciliation()["attempts"]
    foreign_open = [row for row in attempts if row.get("execution_id") not in expected and row.get("finish") is None]
    if foreign_open:
        raise RunnerError("shared spend ledger contains a foreign open attempt")
    return sorted(
        (row for row in attempts if row.get("execution_id") in expected),
        key=lambda row: int(row.get("start", {}).get("sequence", 0)),
    )


def _calls_from_attempt(attempt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for number, call in enumerate(attempt.get("calls", []), start=1):
        evidence, settlement = call.get("evidence"), call.get("settlement")
        if not isinstance(evidence, Mapping) or not isinstance(settlement, Mapping):
            raise RunnerError("interrupted call lacks durable evidence and settlement")
        row = evidence.get("call_evidence")
        if not isinstance(row, Mapping) or row.get("call_ordinal") != number:
            raise RunnerError("interrupted call evidence is malformed")
        rows.append(row)
    return rows


def _recover(prepared: PreparedRun) -> None:
    sources = {
        expected_execution_id(row, action=prepared.action, model=prepared.stage.model): row
        for row in prepared.index.executions
    }
    raw = {(row["execution_id"], row["attempt_ordinal"]): row for row in prepared.resume.rows}
    for attempt in _relevant_attempts(prepared):
        execution_id, ordinal = str(attempt["execution_id"]), int(attempt["attempt_ordinal"])
        source, start = sources[execution_id], attempt.get("start")
        expected_identity = {
            "model": prepared.stage.model,
            "workload_mode": prepared.action.workload,
            **execution_identity(source),
        }
        if not isinstance(start, Mapping) or start.get("execution_identity") != expected_identity or start.get("run_id") != prepared.action.run_id:
            raise RunnerError("spend attempt differs from the action scope")
        key, outcome, finish = (execution_id, ordinal), attempt.get("outcome"), attempt.get("finish")
        calls = _calls_from_attempt(attempt)
        if outcome is None:
            if finish is not None or key in raw:
                raise RunnerError("raw/WAL attempt ordering is contradictory")
            projection = {
                "execution_id": execution_id,
                "attempt_id": str(attempt["attempt_id"]),
                "attempt_ordinal": ordinal,
                "attempt_status": "reconciled_after_interruption",
            }
            topology = list(source["call_topology"])
            successful = len(calls) == len(topology) and all(
                call.get("kind") == topology[index]
                and isinstance(call.get("content"), str)
                and isinstance(call.get("raw_text"), str)
                and "provider_call_outcome" not in call
                and call.get("error") is None
                for index, call in enumerate(calls)
            )
            if not topology:
                result = prepared.index.deterministic_result(source)
            elif successful:
                response = SimpleNamespace(content=calls[-1]["content"], raw_text=calls[-1]["raw_text"])
                verdict, confidence = parse_response(response)
                route = str(source["route"])
                result = None if verdict not in {"correct", "incorrect"} else {
                    "score": verdict_score(verdict, confidence), "verdict": verdict,
                    "confidence": confidence, "tier": "llm_tool_use" if route == "tool" else "llm_comprehension",
                    "grounding_status": "flagged" if route == "tool" else "all_match",
                    "provenance_triggered": bool(source.get("provenance")),
                    "raw_text": calls[-1]["raw_text"], "tokens": calls[-1].get("out_tokens"),
                    "call_log": calls,
                }
            else:
                result = None
            if result is not None:
                projection["attempt_status"] = "completed"
                row = result_row(source, action=prepared.action, result=result,
                                 attempt=projection, latency_s=0)
            else:
                row = error_row(source, action=prepared.action, calls=calls,
                                attempt=projection, latency_s=0,
                                error=(
                                    "InvalidModelOutput"
                                    if successful
                                    else "InterruptedAfterDurableProviderEvidence"
                                ))
            validate_row(row, source=source, action=prepared.action,
                         model=prepared.stage.model, provider_model_id=prepared.stage.provider_model_id)
            prepared.guard.commit_deferred_attempt_outcome(execution_id, ordinal, row)
            prepared.guard.finish_deferred_attempt(execution_id, ordinal)
        else:
            if not isinstance(outcome, Mapping) or not isinstance(outcome.get("raw_row"), Mapping):
                raise RunnerError("committed WAL outcome lacks its raw row")
            row = outcome["raw_row"]
            if outcome.get("raw_row_sha256") != hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                           allow_nan=False).encode("utf-8")
            ).hexdigest():
                # SpendGuard uses ensure_ascii=True for ledger commitments.
                raise RunnerError("committed WAL raw-row digest differs")
            if finish is None:
                prepared.guard.finish_deferred_attempt(execution_id, ordinal)
            if key in raw and raw[key] != row:
                raise RunnerError("raw output differs from its WAL outcome")
        if key not in raw:
            prepared.output.append(row)
            raw[key] = row


def _reconcile(prepared: PreparedRun) -> None:
    attempts = _relevant_attempts(prepared)
    raw = {(row["execution_id"], row["attempt_ordinal"]): row for row in prepared.resume.rows}
    ledger: dict[tuple[str, int], Mapping[str, Any]] = {}
    for attempt in attempts:
        key = (str(attempt["execution_id"]), int(attempt["attempt_ordinal"]))
        if key in ledger or not isinstance(attempt.get("outcome"), Mapping) or not isinstance(attempt.get("finish"), Mapping):
            raise RunnerError("spend ledger attempt is incomplete or repeated")
        ledger[key] = attempt
    if set(raw) != set(ledger):
        raise RunnerError("raw output and spend WAL attempts are not bijective")
    for key, row in raw.items():
        attempt, outcome = ledger[key], ledger[key]["outcome"]
        if outcome.get("raw_row") != row or _calls_from_attempt(attempt) != row.get("call_log"):
            raise RunnerError("raw output differs from committed WAL evidence")


def prepare_run(plan: RunPlan | str | Path, *, action_id: str | None = None,
                now: Callable[[], float] = time.monotonic) -> PreparedRun:
    loaded = load_run_plan(plan) if isinstance(plan, (str, Path)) else plan
    scopes = _scopes(loaded)
    states: dict[str, str] = {}
    for action in loaded.actions:
        stage = loaded.stage_by_id[action.stage_id]
        states[action.id] = load_resume(action.output, index=scopes[action.id], action=action,
                                        model=stage.model, provider_model_id=stage.provider_model_id).status
    ready = loaded.ready_actions(states)
    if not ready:
        raise RunnerError("run plan is already complete")
    if action_id is None:
        selected = ready[0]
    else:
        selected = loaded.action_by_id.get(action_id)
        if selected is None or selected not in ready:
            raise RunnerError(
                f"requested action is not ready; ready actions are "
                f"{[action.id for action in ready]!r}"
            )
    stage, index = loaded.stage_by_id[selected.stage_id], scopes[selected.id]
    output = AppendLog.open(selected.output)
    guard: SpendGuard | None = None
    try:
        resume = load_resume(selected.output, index=index, action=selected, model=stage.model,
                             provider_model_id=stage.provider_model_id, stream=output.stream)
        def assert_surface() -> None:
            loaded.assert_current(); index.assert_current(); output.assert_current()
        guard = SpendGuard(
            selected.ledger,
            approved_cap_usd=loaded.global_cap_usd,
            stage_cap_usd=stage.cap_usd,
            model=stage.model,
            stage=stage.id,
            workload=selected.workload,
            run_id=selected.run_id,
            provider_input_token_maximum=selected.provider_input_token_maximum,
            max_attempts=selected.max_attempts,
            paid_surface_assertion=assert_surface,
        )
        prepared = PreparedRun(loaded, selected, stage, index, output, guard,
                               _ActionGuard(guard, selected), resume, now())
        _recover(prepared)
        prepared.resume = load_resume(selected.output, index=index, action=selected,
                                      model=stage.model, provider_model_id=stage.provider_model_id,
                                      stream=output.stream)
        _reconcile(prepared)
        if prepared.resume.status == "complete":
            prepared.close()
            if action_id is not None:
                raise RunnerError("requested action completed during WAL recovery; no token is needed")
            return prepare_run(loaded, now=now)
        return prepared
    except BaseException:
        if guard is not None:
            guard.close()
        output.close()
        raise


class _DeadlineClient:
    def __init__(self, client: Any, deadline: float) -> None:
        self.client, self.deadline = client, deadline

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ActionDeadlineExceeded("action deadline reached before provider call")
        config = getattr(self.client, "config", None)
        if isinstance(config, dict) and isinstance(config.get("timeout"), (int, float)):
            config["timeout"] = max(0.001, min(float(config["timeout"]), remaining))
        result = self.client.call(*args, **kwargs)
        if time.monotonic() > self.deadline:
            raise ActionDeadlineExceeded("provider call completed after the action deadline")
        return result


def _attempt(prepared: PreparedRun, client: Any, source: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, BaseException | None]:
    started, receipt, row = time.monotonic(), None, None
    try:
        with prepared.action_guard.attempt(
            execution_identity(source)
        ) as receipt:
            try:
                result = score_execution(prepared.index, source, client,
                                         main_max_tokens=prepared.action.main_max_output_tokens)
            except SpendGuardStop as exc:
                calls = client.pop_call_log()
                if receipt.started:
                    projection = _attempt_projection(receipt, "scheduling_stopped")
                    candidate = error_row(source, action=prepared.action, calls=calls,
                                          attempt=projection, latency_s=time.monotonic() - started, error=exc)
                    _commit(prepared, source, candidate)
                    row = candidate
                raise
            except BaseException as exc:
                calls = client.pop_call_log()
                receipt = prepared.action_guard.ensure_attempt_started()
                candidate = error_row(source, action=prepared.action, calls=calls,
                                      attempt=_attempt_projection(receipt, "error"),
                                      latency_s=time.monotonic() - started, error=exc)
                _commit(prepared, source, candidate)
                row = candidate
                raise
            else:
                receipt = prepared.action_guard.ensure_attempt_started()
                # The single validity gate. It covers BOTH closed-set fields and
                # the resulting score, because confidence selects the score cell
                # (0.95/0.80/0.65 or 0.05/0.20/0.35) just as much as the verdict
                # picks its sign. Previously only the verdict was checked, so a
                # well-formed verdict with an out-of-vocabulary confidence — e.g.
                # {"verdict": "correct", "confidence": "certain"} — fell through
                # to a fabricated 0.5 and was written as row_status="scored",
                # indistinguishable from a real measurement. Anything off-grid is
                # now retried like any other unparseable output and, once the
                # per-source budget is spent, recorded as an ERROR rather than
                # assigned a number the model never produced.
                invalid = (
                    result.get("verdict") not in VALID_VERDICTS
                    or result.get("confidence") not in VALID_CONFIDENCES
                    or result.get("score") is None
                )
                candidate = (
                    error_row(
                        source,
                        action=prepared.action,
                        calls=list(result.get("call_log") or []),
                        attempt=_attempt_projection(receipt, "error"),
                        latency_s=time.monotonic() - started,
                        error=InvalidModelOutput(_INVALID_OUTPUT_MESSAGE),
                    )
                    if invalid
                    else result_row(
                        source,
                        action=prepared.action,
                        result=result,
                        attempt=_attempt_projection(receipt, "completed"),
                        latency_s=time.monotonic() - started,
                    )
                )
                _commit(prepared, source, candidate)
                row = candidate
                if invalid:
                    raise InvalidModelOutput(_INVALID_OUTPUT_MESSAGE)
        return row, None
    except BaseException as exc:
        return row, exc


def _attempt_projection(receipt: Any, status: str) -> dict[str, Any]:
    return {"execution_id": receipt.execution_id, "attempt_id": receipt.attempt_id,
            "attempt_ordinal": receipt.attempt_ordinal, "attempt_status": status}


def _commit(prepared: PreparedRun, source: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    validate_row(row, source=source, action=prepared.action, model=prepared.stage.model,
                 provider_model_id=prepared.stage.provider_model_id)
    prepared.action_guard.commit_attempt_outcome(row)


def _factory_call(factory: Callable[..., Any], token: str, action: Action) -> Any:
    parameters = inspect.signature(factory).parameters
    return factory(token, action) if len(parameters) != 1 else factory(token)


def _row_retry_class(row: Mapping[str, Any]) -> str | None:
    if row.get("row_status") == "scored":
        return (
            None
            if row.get("verdict") in {"correct", "incorrect"}
            else "invalid_model_output"
        )
    if row.get("row_status") != "error":
        return None
    calls = row.get("call_log")
    if (
        isinstance(calls, list)
        and calls
        and calls[-1].get("provider_failure_class") == "transport_or_server"
    ):
        return "transport_or_server"
    error = row.get("error")
    if not isinstance(error, Mapping):
        return None
    if error.get("type") == "InvalidModelOutput":
        return "invalid_model_output"
    # Resume classification is name-based while live classification is
    # isinstance-based: ActionDeadlineExceeded (a TimeoutError), WAL
    # interruption closures, and reservation-accounting breaches record
    # operational events, not provider verdicts, and must stay retryable
    # across restarts.
    if error.get("type") in {
        "TimeoutError",
        "ConnectionError",
        "ActionDeadlineExceeded",
        "InterruptedAfterDurableProviderEvidence",
        "SpendReservationBreach",
    }:
        return "transport_or_server"
    return None


def _retry_delay(action: Action, attempts: int) -> float:
    return min(3600.0, action.retry_backoff_seconds * (2 ** max(0, attempts - 1)))


@dataclass(frozen=True)
class _SourceOutcome:
    rows: tuple[Mapping[str, Any], ...]
    completed: bool
    failure: Mapping[str, Any] | None


def _run_source(
    prepared: PreparedRun,
    client: _DeadlineClient,
    source: Mapping[str, Any],
    *,
    sleep: Callable[[float], None],
) -> _SourceOutcome:
    key = source_key(source)
    attempts = prepared.resume.attempts.get(key, 0)
    prior = prepared.resume.latest.get(key)
    prior_retry_class = _row_retry_class(prior) if prior is not None else None
    if prior is not None and prior_retry_class is None:
        return _SourceOutcome((), False, {
            "kind": "nonretryable_failure_on_resume",
            "stmt_i": key[0],
            "evidence_i": key[1],
        })
    invalid_outputs = sum(
        _row_retry_class(row) == "invalid_model_output"
        for row in prepared.resume.rows
        if source_key(row) == key
    )
    if invalid_outputs >= INVALID_MODEL_OUTPUT_LIMIT:
        return _SourceOutcome((), False, {
            "kind": "invalid_model_output_limit",
            "stmt_i": key[0],
            "evidence_i": key[1],
        })
    if attempts >= prepared.action.max_attempts:
        return _SourceOutcome((), False, {
            "kind": "attempts_exhausted",
            "stmt_i": key[0],
            "evidence_i": key[1],
        })

    if attempts and prior_retry_class == "transport_or_server":
        delay = _retry_delay(prepared.action, attempts)
        if time.monotonic() + delay >= prepared.deadline:
            return _SourceOutcome((), False, {
                "kind": "deadline", "stmt_i": key[0], "evidence_i": key[1]
            })
        sleep(delay)

    rows: list[Mapping[str, Any]] = []
    while attempts < prepared.action.max_attempts:
        if time.monotonic() >= prepared.deadline:
            return _SourceOutcome(tuple(rows), False, {
                "kind": "deadline", "stmt_i": key[0], "evidence_i": key[1]
            })
        row, error = _attempt(prepared, client, source)
        if row is not None:
            rows.append(row)
            attempts += 1
        if error is None:
            return _SourceOutcome(tuple(rows), True, None)
        if isinstance(error, SpendGuardStop):
            return _SourceOutcome(tuple(rows), False, {
                "kind": "spend_cap", "type": type(error).__name__
            })
        retry_class, status = classify_provider_failure(error)
        if isinstance(error, InvalidModelOutput):
            invalid_outputs += 1
            retryable = invalid_outputs < INVALID_MODEL_OUTPUT_LIMIT
        else:
            retryable = retry_class == "transport_or_server"
        if not retryable or attempts >= prepared.action.max_attempts:
            return _SourceOutcome(tuple(rows), False, {
                "kind": "attempt_failed",
                "type": type(error).__name__,
                "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                "provider_http_status": status,
            })
        delay = _retry_delay(prepared.action, attempts)
        if time.monotonic() + delay >= prepared.deadline:
            return _SourceOutcome(tuple(rows), False, {
                "kind": "deadline", "type": type(error).__name__
            })
        sleep(delay)
    raise RunnerError("source attempt loop ended without an outcome")


def _run_prepared(
    prepared: PreparedRun,
    *,
    ready_writer: Callable[[Mapping[str, Any]], None],
    token_reader: Callable[[], str],
    client_factory: Callable[..., Any],
    sleep: Callable[[float], None] = time.sleep,
) -> RunSummary:
    if prepared.closed or prepared.started:
        raise RunnerError("prepared run is closed or already consumed")
    prepared.started = True
    ready_writer(prepared.readiness())
    token = token_reader()
    if not isinstance(token, str) or not token:
        raise RunnerError("token reader returned no bearer token")
    pending = [
        (position, source)
        for position, source in enumerate(prepared.index.executions)
        if source_key(source) not in prepared.resume.done
    ]
    window = min(prepared.action.workers, len(pending))
    clients: list[_DeadlineClient] = []
    raw_clients: list[Any] = []
    try:
        for _ in range(window):
            raw = _factory_call(client_factory, token, prepared.action)
            if any(raw is prior for prior in raw_clients):
                raise RunnerError("each worker requires a distinct model client")
            config = getattr(raw, "config", None)
            if (
                not isinstance(config, Mapping)
                or config.get("model_id") != prepared.stage.provider_model_id
            ):
                raise RunnerError("client provider model differs from the semantic stage")
            raw_clients.append(raw)
            clients.append(
                _DeadlineClient(
                    GuardedModelClient(raw, prepared.action_guard), prepared.deadline
                )
            )
    finally:
        token = ""

    completed_this_run = 0
    failures: list[tuple[int, Mapping[str, Any]]] = []
    fatal: BaseException | None = None
    next_pending = 0
    active: dict[Future[_SourceOutcome], tuple[int, int]] = {}

    def submit(executor: ThreadPoolExecutor, slot: int) -> None:
        nonlocal next_pending
        position, source = pending[next_pending]
        next_pending += 1
        future = executor.submit(
            _run_source, prepared, clients[slot], source, sleep=sleep
        )
        active[future] = (slot, position)

    try:
        with ThreadPoolExecutor(max_workers=window) as executor:
            for slot in range(window):
                submit(executor, slot)
            stop_scheduling = False
            while active:
                finished, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                batch = sorted(
                    ((active.pop(future), future) for future in finished),
                    key=lambda item: item[0][1],
                )
                freed: list[int] = []
                for (slot, position), future in batch:
                    freed.append(slot)
                    try:
                        outcome = future.result()
                    except BaseException as exc:
                        fatal = fatal or exc
                        stop_scheduling = True
                        continue
                    for row in outcome.rows:
                        prepared.output.append(row)
                    if outcome.completed:
                        completed_this_run += 1
                    if outcome.failure is not None:
                        failures.append((position, outcome.failure))
                        stop_scheduling = True
                if not stop_scheduling:
                    for slot in sorted(freed):
                        if next_pending >= len(pending):
                            break
                        submit(executor, slot)

        # A worker can commit a WAL outcome immediately before an unexpected
        # local exception.  Reload the serialized raw prefix, then recover every
        # missing WAL outcome regardless of its position among concurrent starts.
        prepared.resume = load_resume(
            prepared.action.output, index=prepared.index, action=prepared.action,
            model=prepared.stage.model, provider_model_id=prepared.stage.provider_model_id,
            stream=prepared.output.stream,
        )
        _recover(prepared)
        prepared.resume = load_resume(
            prepared.action.output, index=prepared.index, action=prepared.action,
            model=prepared.stage.model, provider_model_id=prepared.stage.provider_model_id,
            stream=prepared.output.stream,
        )
        _reconcile(prepared)
        if fatal is not None:
            raise RunnerError("worker failed after durable reconciliation") from fatal
        failure = min(failures, key=lambda item: item[0])[1] if failures else None
        status = "complete" if prepared.resume.status == "complete" else (
            "spend_cap" if failure and failure["kind"] == "spend_cap" else
            "deadline" if failure and failure["kind"] == "deadline" else "partial"
        )
        return RunSummary(status, prepared.action.id, completed_this_run,
                          len(prepared.resume.done), len(prepared.index.executions),
                          prepared.resume.verdicts, prepared.action_guard.summary(), failure)
    finally:
        prepared.close()


def run_prepared(
    prepared: PreparedRun,
    *,
    ready_writer: Callable[[Mapping[str, Any]], None],
    token_reader: Callable[[], str],
    client_factory: Callable[..., Any],
    sleep: Callable[[float], None] = time.sleep,
) -> RunSummary:
    try:
        return _run_prepared(
            prepared,
            ready_writer=ready_writer,
            token_reader=token_reader,
            client_factory=client_factory,
            sleep=sleep,
        )
    finally:
        prepared.close()


def write_ready_fd(descriptor: int, value: Mapping[str, Any]) -> None:
    payload = canonical_json_line(value)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise RunnerError("short readiness write")
    finally:
        os.close(descriptor)


def read_token_fd(descriptor: int, *, maximum_bytes: int = 16_384) -> str:
    try:
        raw = os.read(descriptor, maximum_bytes + 1)
        if len(raw) > maximum_bytes or os.read(descriptor, 1):
            raise RunnerError("bearer token exceeds the bounded control message")
    finally:
        os.close(descriptor)
    try:
        token = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RunnerError("bearer token is not UTF-8") from exc
    if not token or "\n" in token or "\r" in token:
        raise RunnerError("bearer token control message is malformed")
    return token
