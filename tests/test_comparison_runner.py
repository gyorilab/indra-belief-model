from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from indra_belief.comparison import llm, runner
from indra_belief.comparison.contracts import (
    ContractError,
    canonical_json_bytes,
    canonical_json_line,
    canonical_sha256,
    load_run_plan,
)
from indra_belief.comparison.replay import parse_structured, prompt_sha256


ROOT = Path(__file__).resolve().parents[1]


def _descriptor(path: Path, *, relative_to: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": raw.count(b"\n"),
    }


def _write_fixture(
    tmp_path: Path,
    *,
    actions: int = 1,
    action_cap: str = "1",
    deadline_seconds: int = 60,
    workers: int = 1,
    execution_count: int = 1,
    max_attempts: int = 2,
    retry_backoff_seconds: float = 0.0,
    primary_actions: bool = False,
) -> Path:
    plan_dir = tmp_path / "plan"
    replay_dir = plan_dir / "replay"
    replay_dir.mkdir(parents=True)
    workload = (
        "unique_exact_pairs_primary" if primary_actions else "fixture_workload"
    )
    system = "Judge the extraction and return verdict/confidence JSON."
    prefix: list[dict[str, str]] = []
    user = 'CLAIM: A [Activation] B\nEVIDENCE: "A activates B."'
    execution = {
        "execution_key_sha256": "",
        "workload": workload,
        "stmt_i": 0,
        "evidence_i": 0,
        "paper_statement_hash": "101",
        "source_hash": "202",
        "evidence_json_sha256": "a" * 64,
        "workload_metadata": {"eligible_position": 0},
        "statement_type": "Activation",
        "subject_name": "A",
        "object_name": "B",
        "claim": "A [Activation] B",
        "entity_context": "",
        "provenance": "",
        "evidence_metadata": {
            "text": "A activates B.",
            "source_api": "fixture",
            "found_by": "fixture",
            "pmid": "1",
            "is_direct": True,
            "raw_text": ["A", "B"],
        },
        "route": "plain",
        "entity_refs": {"subject": None, "object": None},
        "abbreviation_lines": [],
        "lookup_refs": [],
        "call_topology": ["monolithic"],
        "main_system_ref": hashlib.sha256(system.encode()).hexdigest(),
        "main_message_prefix_ref": canonical_sha256(prefix),
        "main_user_before_relation_note_sha256": hashlib.sha256(user.encode()).hexdigest(),
        "relation_note_insertion": {
            "message_index": 0,
            "role": "user",
            "utf8_byte_offset": len(user),
            "prefix_if_nonempty": "\n\n",
            "empty_note_inserts_prefix": False,
        },
        "relation_alias_refs": {"subject": None, "object": None},
        "relation_system_ref": None,
        "main_prompt_base_sha256": prompt_sha256(
            system, [{"role": "user", "content": user}]
        ),
        "relation_prompt_sha256": None,
    }
    execution_rows: list[dict[str, Any]] = []
    for position in range(execution_count):
        row = json.loads(json.dumps(execution))
        row["stmt_i"] = position
        row["paper_statement_hash"] = str(101 + position)
        row["source_hash"] = str(202 + position)
        row["evidence_json_sha256"] = (
            "a" * 64 if position == 0 else f"{position:064x}"
        )
        row["workload_metadata"]["eligible_position"] = position
        row["execution_key_sha256"] = canonical_sha256(
            [
                workload,
                position,
                0,
                row["paper_statement_hash"],
                row["source_hash"],
                row["evidence_json_sha256"],
            ]
        )
        execution_rows.append(row)
    executions = replay_dir / "executions.jsonl"
    executions.write_bytes(b"".join(canonical_json_line(row) for row in execution_rows))
    manifest = {
        "artifact_kind": "indra_belief_grounding_replay",
        "status": "ready",
        "tables": {"executions": _descriptor(executions, relative_to=replay_dir)},
        "prompt_components": {
            "main_systems": [{"sha256": hashlib.sha256(system.encode()).hexdigest(), "text": system}],
            "main_message_prefixes": [{"sha256": canonical_sha256(prefix), "messages": prefix}],
            "relation_system": None,
        },
        "workloads": [
            {"name": workload, "execution_rows": execution_count}
        ],
    }
    manifest_path = replay_dir / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    manifest_descriptor = _descriptor(manifest_path, relative_to=plan_dir)
    manifest_descriptor.pop("rows")
    manifest_descriptor["base"] = "owner"
    action_rows = []
    for index in range(actions):
        suffix = "_primary" if primary_actions else ""
        action_id = f"action_{index + 1}{suffix}"
        predecessor_id = f"action_{index}{suffix}"
        action_rows.append(
            {
                "id": action_id,
                "stage": "glm" if index == 0 else "e2b",
                "run_id": f"fixture_run_{index + 1}{suffix}",
                "workload": workload,
                "ledger": {"path": "runs/spend.ndjson", "base": "owner"},
                "output": {"path": f"runs/action_{index + 1}.jsonl", "base": "owner"},
                "cap_usd": action_cap,
                "deadline_seconds": deadline_seconds,
                "max_attempts": max_attempts,
                "provider_input_token_maximum": 8,
                "main_max_output_tokens": 16,
                "retry_backoff_seconds": retry_backoff_seconds,
                "workers": workers,
                "depends_on": [] if index == 0 else [predecessor_id],
                "execution_keys": None,
            }
        )
    plan = {
        "kind": "indra_belief_comparison_run_plan",
        "amendment": None,
        "replay": {"manifest": manifest_descriptor},
        "global_cap_usd": "2",
        "stages": [
            {
                "id": "glm",
                "model": "bedrock-glm-5",
                "provider_model_id": "zai.glm-5",
                "cap_usd": "1",
            }
        ] + ([{
            "id": "e2b",
            "model": "bedrock-gemma-4-e2b",
            "provider_model_id": "google.gemma-4-e2b",
            "cap_usd": "1",
        }] if actions > 1 else []),
        "actions": action_rows,
    }
    path = plan_dir / "run_plan.json"
    path.write_bytes(canonical_json_line(plan))
    return path


def _amended_attempt_limit(
    path: Path,
    *,
    action_indexes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    predecessor_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    value = json.loads(path.read_text())
    changes = []
    for index in action_indexes:
        action = value["actions"][index]
        previous = action["max_attempts"]
        action["max_attempts"] = 10
        changes.append(
            {
                "action_id": action["id"],
                "field": "max_attempts",
                "from": previous,
                "to": 10,
            }
        )
    value["amendment"] = {
        "predecessor_sha256": predecessor_sha256,
        "frozen_at": "2026-07-21T00:00:00Z",
        "reason": "Resume transport-only failures at the canonical ceiling.",
        "changes": changes,
    }
    return value


@dataclass
class FakeResponse:
    content: str = '{"support":"A activates B","objection":null,"verdict":"correct","confidence":"high"}'
    reasoning: str = ""
    raw_text: str = ""
    tokens: int = 3
    prompt_tokens: int = 5
    finish_reason: str = "stop"
    reasoning_trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw_text:
            self.raw_text = self.content


class FakeClient:
    backend = "fixture"

    def __init__(
        self,
        events: list[str],
        ledger: Path,
        *,
        fail_first: bool = False,
        invalid_first: bool = False,
        transport_failures: int = 0,
        invalid_responses: int = 0,
    ) -> None:
        self.events = events
        self.ledger = ledger
        self.transport_failures = max(int(fail_first), transport_failures)
        self.invalid_responses = max(int(invalid_first), invalid_responses)
        self.calls = 0
        self.log: list[dict[str, Any]] = []
        self.config = {"model_id": "zai.glm-5", "max_tokens": 16, "timeout": 20}

    def pop_call_log(self) -> list[dict[str, Any]]:
        rows, self.log = self.log, []
        return rows

    def call(self, *, system, messages, max_tokens=None, temperature=0.1,
             response_format=None, reasoning_effort=None, kind="unknown"):
        self.calls += 1
        self.events.append(f"provider:{self.calls}")
        events = [json.loads(line) for line in self.ledger.read_text().splitlines()]
        assert any(row["event"] == "call_reserved" for row in events)
        common = {
            "kind": kind,
            "model_id": self.config["model_id"],
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if self.calls <= self.transport_failures:
            self.log.append({
                **common,
                "content": None,
                "reasoning": None,
                "raw_text": None,
                "prompt_tokens": None,
                "out_tokens": None,
                "finish_reason": None,
                "error": "TimeoutError",
            })
            raise TimeoutError("fixture timeout")
        response = FakeResponse(
            content=(
                '{"support":"span","objection":"issue","verdict":"maybe","confidence":"medium"}'
                if self.calls <= self.transport_failures + self.invalid_responses
                else FakeResponse().content
            )
        )
        self.log.append({
            **common,
            "content": response.content,
            "reasoning": response.reasoning,
            "raw_text": response.raw_text,
            "prompt_tokens": response.prompt_tokens,
            "out_tokens": response.tokens,
            "finish_reason": response.finish_reason,
        })
        return response


class WindowState:
    def __init__(self, workers: int) -> None:
        self.barrier = threading.Barrier(workers)
        self.lock = threading.Lock()
        self.failure_started = threading.Event()
        self.active = 0
        self.maximum_active = 0


class WindowClient(FakeClient):
    def __init__(
        self,
        events: list[str],
        ledger: Path,
        state: WindowState,
        *,
        fail: bool,
        failure_message: str = "nonretryable fixture failure",
        delay: float = 0,
    ):
        super().__init__(events, ledger)
        self.state = state
        self.fail = fail
        self.failure_message = failure_message
        self.delay = delay
        self.overlap = False
        self._active = False

    def call(self, *, system, messages, max_tokens=None, temperature=0.1,
             response_format=None, reasoning_effort=None, kind="unknown"):
        with self.state.lock:
            if self._active:
                self.overlap = True
            self._active = True
            self.state.active += 1
            self.state.maximum_active = max(
                self.state.maximum_active, self.state.active
            )
        self.calls += 1
        common = {
            "kind": kind,
            "model_id": self.config["model_id"],
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        try:
            self.state.barrier.wait(timeout=5)
            if self.delay:
                threading.Event().wait(self.delay)
            if self.fail:
                self.state.failure_started.set()
                error = ValueError(self.failure_message)
                self.log.append({
                    **common,
                    "content": None,
                    "reasoning": None,
                    "raw_text": None,
                    "prompt_tokens": None,
                    "out_tokens": None,
                    "finish_reason": None,
                    "error": type(error).__name__,
                })
                raise error
            assert self.state.failure_started.wait(timeout=5)
            threading.Event().wait(0.1)
            response = FakeResponse()
            self.log.append({
                **common,
                "content": response.content,
                "reasoning": response.reasoning,
                "raw_text": response.raw_text,
                "prompt_tokens": response.prompt_tokens,
                "out_tokens": response.tokens,
                "finish_reason": response.finish_reason,
            })
            return response
        finally:
            with self.state.lock:
                self._active = False
                self.state.active -= 1


def _execute(
    path: Path,
    *,
    fail_first: bool = False,
    invalid_first: bool = False,
    transport_failures: int = 0,
    invalid_responses: int = 0,
    sleep: Any = None,
):
    events: list[str] = []
    plan = load_run_plan(path)
    client = FakeClient(
        events,
        plan.actions[0].ledger,
        fail_first=fail_first,
        invalid_first=invalid_first,
        transport_failures=transport_failures,
        invalid_responses=invalid_responses,
    )
    prepared = runner.prepare_run(plan)

    def ready(value):
        assert value["provider_calls_started_during_preflight"] == 0
        events.append("ready")

    def token():
        assert events == ["ready"]
        events.append("token")
        return "secret-fixture-token"

    def factory(value, action):
        assert value == "secret-fixture-token"
        assert events == ["ready", "token"]
        assert action.id == plan.actions[0].id
        events.append("client")
        return client

    summary = runner.run_prepared(
        prepared,
        ready_writer=ready,
        token_reader=token,
        client_factory=factory,
        sleep=sleep or (lambda _delay: None),
    )
    return plan, client, events, summary


@pytest.mark.parametrize("location", ["ledger", "output", "manifest"])
def test_run_plan_requires_exact_path_descriptors(
    tmp_path: Path, location: str
) -> None:
    path = _write_fixture(tmp_path)
    value = json.loads(path.read_text())
    if location == "ledger":
        value["actions"][0]["ledger"] = value["actions"][0]["ledger"]["path"]
    elif location == "output":
        value["actions"][0]["output"] = value["actions"][0]["output"]["path"]
    else:
        del value["replay"]["manifest"]["base"]
    path.write_bytes(canonical_json_line(value))
    with pytest.raises(ContractError, match="path and base|path/sha256/bytes/base"):
        load_run_plan(path)


@pytest.mark.parametrize("location", ["ledger", "output", "manifest"])
def test_run_plan_rejects_removed_plan_base_alias(
    tmp_path: Path, location: str
) -> None:
    path = _write_fixture(tmp_path)
    value = json.loads(path.read_text())
    if location == "ledger":
        value["actions"][0]["ledger"]["base"] = "plan"
    elif location == "output":
        value["actions"][0]["output"]["base"] = "plan"
    else:
        value["replay"]["manifest"]["base"] = "plan"
    path.write_bytes(canonical_json_line(value))
    with pytest.raises(ContractError, match="unsupported path base 'plan'"):
        load_run_plan(path)


def test_owner_base_is_deterministically_the_plan_directory(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    plan = load_run_plan(path)
    assert plan.amendment is None
    assert plan.actions[0].ledger == path.parent / "runs/spend.ndjson"
    assert plan.actions[0].output == path.parent / "runs/action_1.jsonl"
    assert plan.replay_manifest.path == path.parent / "replay/manifest.json"


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_worker_count_is_a_bounded_integer(tmp_path: Path, workers: Any) -> None:
    path = _write_fixture(tmp_path)
    value = json.loads(path.read_text())
    value["actions"][0]["workers"] = workers
    path.write_bytes(canonical_json_line(value))
    with pytest.raises(ContractError, match="workers"):
        load_run_plan(path)


@pytest.mark.parametrize("max_attempts", [0, 11, True, 1.5])
def test_attempt_count_is_bounded_at_ten(
    tmp_path: Path, max_attempts: Any
) -> None:
    path = _write_fixture(tmp_path)
    value = json.loads(path.read_text())
    value["actions"][0]["max_attempts"] = max_attempts
    path.write_bytes(canonical_json_line(value))
    with pytest.raises(ContractError, match="max_attempts"):
        load_run_plan(path)


def test_run_plan_accepts_a_bound_primary_attempt_amendment(
    tmp_path: Path,
) -> None:
    path = _write_fixture(
        tmp_path, actions=2, max_attempts=5, primary_actions=True
    )
    value = _amended_attempt_limit(path, action_indexes=(0, 1))
    path.write_bytes(canonical_json_line(value))

    plan = load_run_plan(path)
    assert plan.amendment is not None
    assert plan.amendment.predecessor_sha256 == value["amendment"][
        "predecessor_sha256"
    ]
    assert [change.action_id for change in plan.amendment.changes] == [
        "action_1_primary",
        "action_2_primary",
    ]
    assert [
        (change.field, change.from_value, change.to_value)
        for change in plan.amendment.changes
    ] == [("max_attempts", 5, 10), ("max_attempts", 5, 10)]


def test_run_plan_accepts_a_mixed_attempt_and_worker_amendment(
    tmp_path: Path,
) -> None:
    path = _write_fixture(
        tmp_path, actions=2, max_attempts=5, workers=6, primary_actions=True
    )
    value = _amended_attempt_limit(path, action_indexes=(0, 1))
    value["actions"][1]["workers"] = 8
    value["amendment"]["changes"].append(
        {
            "action_id": "action_2_primary",
            "field": "workers",
            "from": 6,
            "to": 8,
        }
    )
    path.write_bytes(canonical_json_line(value))

    plan = load_run_plan(path)
    assert plan.amendment is not None
    assert [
        (change.action_id, change.field, change.from_value, change.to_value)
        for change in plan.amendment.changes
    ] == [
        ("action_1_primary", "max_attempts", 5, 10),
        ("action_2_primary", "max_attempts", 5, 10),
        ("action_2_primary", "workers", 6, 8),
    ]
    assert plan.actions[1].workers == 8


def test_amendment_orders_fields_within_one_action(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path, max_attempts=5, workers=6, primary_actions=True
    )
    value = _amended_attempt_limit(path)
    value["actions"][0]["workers"] = 8
    value["amendment"]["changes"].insert(
        0,
        {
            "action_id": "action_1_primary",
            "field": "workers",
            "from": 6,
            "to": 8,
        },
    )
    path.write_bytes(canonical_json_line(value))

    with pytest.raises(ContractError, match="action order"):
        load_run_plan(path)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("not_object", "amendment fields"),
        ("extra_field", "amendment fields"),
        ("uppercase_sha", "lowercase SHA-256"),
        ("empty_timestamp", "nonempty timestamp"),
        ("empty_reason", "reason must be nonempty"),
        ("empty_changes", "nonempty ordered array"),
        ("unknown_action", "unknown action"),
        ("non_primary_action", "not primary"),
        ("wrong_field", "field must be one of"),
        ("nonpositive_from", "positive integer"),
        ("nonincreasing", "exactly 5 -> 10"),
        ("above_ceiling", "exactly 5 -> 10"),
        ("noncanonical_workers", "exactly 6 -> 8"),
        ("current_value_mismatch", "does not match the current action"),
    ],
)
def test_run_plan_rejects_malformed_attempt_amendments(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    path = _write_fixture(tmp_path, max_attempts=5, primary_actions=True)
    value = _amended_attempt_limit(path)
    amendment = value["amendment"]
    change = amendment["changes"][0]
    if case == "not_object":
        value["amendment"] = []
    elif case == "extra_field":
        amendment["version"] = 2
    elif case == "uppercase_sha":
        amendment["predecessor_sha256"] = amendment[
            "predecessor_sha256"
        ].upper()
    elif case == "empty_timestamp":
        amendment["frozen_at"] = "  "
    elif case == "empty_reason":
        amendment["reason"] = ""
    elif case == "empty_changes":
        amendment["changes"] = []
    elif case == "unknown_action":
        change["action_id"] = "unknown_primary"
    elif case == "non_primary_action":
        value["actions"][0]["id"] = "action_1"
        value["actions"][0]["run_id"] = "fixture_run_1"
        change["action_id"] = "action_1"
    elif case == "wrong_field":
        change["field"] = "deadline_seconds"
    elif case == "nonpositive_from":
        change["from"] = 0
    elif case == "nonincreasing":
        change["from"] = 10
    elif case == "above_ceiling":
        change["to"] = 11
    elif case == "noncanonical_workers":
        change["field"] = "workers"
        change["from"] = 6
        change["to"] = 7
        value["actions"][0]["max_attempts"] = 5
        value["actions"][0]["workers"] = 7
    elif case == "current_value_mismatch":
        value["actions"][0]["max_attempts"] = 5
    path.write_bytes(canonical_json_line(value))

    with pytest.raises(ContractError, match=match):
        load_run_plan(path)


@pytest.mark.parametrize("case", ["duplicate", "out_of_order"])
def test_run_plan_requires_unique_plan_ordered_amendment_actions(
    tmp_path: Path,
    case: str,
) -> None:
    path = _write_fixture(
        tmp_path, actions=2, max_attempts=5, primary_actions=True
    )
    value = _amended_attempt_limit(path, action_indexes=(0, 1))
    if case == "duplicate":
        value["amendment"]["changes"][1]["action_id"] = (
            "action_1_primary"
        )
    else:
        value["amendment"]["changes"].reverse()
    path.write_bytes(canonical_json_line(value))

    with pytest.raises(ContractError, match="repeat an action_id|action order"):
        load_run_plan(path)


def test_attempt_amendment_rejects_primary_suffix_on_sensitivity_workload(
    tmp_path: Path,
) -> None:
    path = _write_fixture(tmp_path, max_attempts=5, primary_actions=True)
    value = _amended_attempt_limit(path)
    value["actions"][0]["workload"] = "alternate_prompt_sensitivity"
    path.write_bytes(canonical_json_line(value))

    with pytest.raises(
        ContractError, match="workload must be unique_exact_pairs_primary"
    ):
        load_run_plan(path)


@pytest.mark.parametrize(("from_value", "to_value"), [(4, 5), (1, 6), (5, 6)])
def test_attempt_amendment_rejects_noncanonical_transition(
    tmp_path: Path,
    from_value: int,
    to_value: int,
) -> None:
    path = _write_fixture(
        tmp_path, max_attempts=from_value, primary_actions=True
    )
    value = _amended_attempt_limit(path)
    value["actions"][0]["max_attempts"] = to_value
    change = value["amendment"]["changes"][0]
    change["from"] = from_value
    change["to"] = to_value
    path.write_bytes(canonical_json_line(value))

    with pytest.raises(ContractError, match="exactly 5 -> 10"):
        load_run_plan(path)


def test_fifth_transport_attempt_can_complete_with_exponential_backoff(
    tmp_path: Path,
) -> None:
    path = _write_fixture(
        tmp_path, max_attempts=5, retry_backoff_seconds=1
    )
    delays: list[float] = []
    plan, client, _events, summary = _execute(
        path,
        transport_failures=4,
        sleep=delays.append,
    )
    assert summary.status == "complete"
    assert client.calls == 5
    assert delays == [1, 2, 4, 8]
    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert [row["attempt_ordinal"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["row_status"] for row in rows] == [
        "error", "error", "error", "error", "scored"
    ]


def test_invalid_outputs_remain_capped_after_transport_limit_amendment(
    tmp_path: Path,
) -> None:
    """The invalid-output cap is independent of, and tighter than, max_attempts.

    The load-bearing assertion is the SECOND phase: after amending max_attempts
    5 -> 10, a source already capped on invalid output makes ZERO further calls.
    Raising the transport-retry budget must not buy more attempts for a model
    that is emitting unparseable output — that is what keeps the two limits
    separate concerns.
    """
    path = _write_fixture(
        tmp_path, max_attempts=5, primary_actions=True
    )
    plan, client, _events, summary = _execute(path, invalid_responses=5)
    assert summary.status == "partial"
    assert client.calls == runner.INVALID_MODEL_OUTPUT_LIMIT == 5
    raw_prefix = plan.actions[0].output.read_bytes()
    ledger_prefix = plan.actions[0].ledger.read_bytes()
    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert [row["attempt_ordinal"] for row in rows] == [1, 2, 3, 4, 5]
    assert {row["error"]["type"] for row in rows} == {"InvalidModelOutput"}

    value = _amended_attempt_limit(path)
    path.write_bytes(canonical_json_line(value))
    resumed_plan, resumed_client, _events, resumed = _execute(path)
    assert resumed.status == "partial"
    assert resumed.completed_this_run == 0
    assert resumed.failure and resumed.failure["kind"] == "invalid_model_output_limit"
    assert resumed_client.calls == 0
    assert resumed_plan.actions[0].output.read_bytes() == raw_prefix
    assert resumed_plan.actions[0].ledger.read_bytes() == ledger_prefix


def test_parser_does_not_promote_an_invalid_verdict_value() -> None:
    parsed = parse_structured(
        '{"support":"span","objection":"issue","verdict":"medium","confidence":"medium"}'
    )
    assert parsed["verdict"] is None
    assert parsed["confidence"] is None


def test_descriptor_tamper_fails_before_readiness_or_token(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    execution = path.parent / "replay/executions.jsonl"
    execution.write_bytes(execution.read_bytes() + b"\n")
    boundary: list[str] = []
    with pytest.raises(ContractError, match="byte length differs"):
        runner.prepare_run(path)
    assert boundary == []


def test_ready_precedes_token_and_reservation_precedes_provider(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    plan, client, events, summary = _execute(path)
    assert events == ["ready", "token", "client", "provider:1"]
    assert client.calls == 1
    assert summary.status == "complete"
    raw = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert [row["row_status"] for row in raw] == ["scored"]
    ledger = [
        json.loads(line)
        for line in plan.actions[0].ledger.read_text().splitlines()
    ]
    start = next(row for row in ledger if row["event"] == "attempt_started")
    assert start["execution_identity"] == {
        "model": "bedrock-glm-5",
        "workload_mode": "fixture_workload",
        "eligible_position": 0,
        "paper_statement_hash": "101",
        "source_hash": "202",
        "evidence_json_sha256": "a" * 64,
    }
    assert "action_id" not in start["execution_identity"]
    reservation = next(i for i, row in enumerate(ledger) if row["event"] == "call_reserved")
    evidence = next(i for i, row in enumerate(ledger) if row["event"] == "call_evidence_observed")
    settlement = next(i for i, row in enumerate(ledger) if row["event"] == "call_settled")
    outcome = next(i for i, row in enumerate(ledger) if row["event"] == "attempt_outcome_committed")
    assert reservation < evidence < settlement < outcome
    assert runner.inspect_plan(plan).status == "complete"


def test_canonical_ledger_is_consumable_by_llm_materializer(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    plan, _client, _events, summary = _execute(path)
    assert summary.status == "complete"
    action = plan.actions[0]
    stage = plan.stage_by_id[action.stage_id]
    source = runner._scopes(plan)[action.id].executions[0]
    execution_id = runner.expected_execution_id(
        source, action=action, model=stage.model
    )
    mapped = {
        "eligible_position": source["workload_metadata"]["eligible_position"],
        "paper_statement_hash": source["paper_statement_hash"],
        "source_hash": source["source_hash"],
        "evidence_json_sha256": source["evidence_json_sha256"],
        "route": source["route"],
        "relation_prompt_sha256": source["relation_prompt_sha256"],
    }
    pair = llm._Pair(
        key=(source["stmt_i"], source["evidence_i"]),
        map_row=mapped,
        statement={},
        evidence={},
        statement_id="fixture-statement",
        source=str(source["evidence_metadata"]["source_api"]).casefold(),
        execution_id=execution_id,
    )
    ledger = llm.parse_spend_ledger(action.ledger.read_bytes())
    attempts = llm._attempts_from_ledger(
        ledger=ledger,
        pairs=[pair],
        run_id=action.run_id,
        served_model=stage.model,
        provider_model_id=stage.provider_model_id,
        workload=action.workload,
    )
    raw_rows = [
        json.loads(line) for line in action.output.read_text().splitlines()
    ]
    projected = llm._validate_raw(
        raw_rows=raw_rows,
        pairs=[pair],
        attempts=attempts,
        run_id=action.run_id,
        provider_model_id=stage.provider_model_id,
    )
    assert projected[(0, 0)][-1]["verdict"] == "correct"


def test_dependency_blocked_action_is_rejected(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, actions=2)
    with pytest.raises(runner.RunnerError, match="not ready"):
        runner.prepare_run(path, action_id="action_2")
    plan, _client, _events, summary = _execute(path)
    assert summary.status == "complete"
    status = runner.inspect_plan(plan)
    assert status.status == "partial"
    assert status.next_action_id == "action_2"
    assert status.ready_action_ids == ("action_2",)


def test_independent_actions_are_ready_and_selectable_together(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, actions=2)
    value = json.loads(path.read_text())
    value["actions"][1]["depends_on"] = []
    value["actions"][1]["ledger"] = {
        "path": "runs/action_2_spend.ndjson",
        "base": "owner",
    }
    path.write_bytes(canonical_json_line(value))

    status = runner.inspect_plan(path)
    assert status.ready_action_ids == ("action_1", "action_2")
    prepared = runner.prepare_run(path, action_id="action_2")
    assert prepared.action.id == "action_2"
    assert prepared.guard.path.name == "action_2_spend.ndjson"
    prepared.close()


def test_independent_spend_lanes_cannot_overauthorize_one_stage(
    tmp_path: Path,
) -> None:
    path = _write_fixture(tmp_path, actions=2)
    value = json.loads(path.read_text())
    value["actions"][1]["stage"] = "glm"
    value["actions"][1]["workload"] = "independent_workload"
    value["actions"][1]["depends_on"] = []
    value["actions"][1]["ledger"] = {
        "path": "runs/action_2_spend.ndjson",
        "base": "owner",
    }
    path.write_bytes(canonical_json_line(value))

    with pytest.raises(ContractError, match="independent spend lanes"):
        load_run_plan(path)


def test_transient_failure_retries_once_and_resumes_deterministically(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    plan, client, events, summary = _execute(path, fail_first=True)
    assert client.calls == 2
    assert events[-2:] == ["provider:1", "provider:2"]
    assert summary.status == "complete"
    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert [(row["attempt_ordinal"], row["row_status"]) for row in rows] == [
        (1, "error"),
        (2, "scored"),
    ]
    assert runner.inspect_plan(plan).status == "complete"


def test_widened_transport_limit_resumes_exact_missing_work(
    tmp_path: Path,
) -> None:
    path = _write_fixture(
        tmp_path,
        execution_count=3,
        max_attempts=5,
        retry_backoff_seconds=1,
        primary_actions=True,
    )
    initial_plan = load_run_plan(path)
    prepared = runner.prepare_run(initial_plan)
    try:
        base = FakeClient([], initial_plan.actions[0].ledger)
        client = runner._DeadlineClient(
            runner.GuardedModelClient(base, prepared.action_guard), prepared.deadline
        )
        completed, error = runner._attempt(
            prepared, client, prepared.index.executions[0]
        )
        assert error is None and completed is not None
        prepared.output.append(completed)
    finally:
        prepared.close()

    first_plan, first_client, _events, first = _execute(
        path,
        transport_failures=5,
    )
    assert first.status == "partial"
    assert first.completed_total == 1
    assert first_client.calls == 5
    original = [
        json.loads(line)
        for line in first_plan.actions[0].output.read_text().splitlines()
    ]
    raw_prefix = first_plan.actions[0].output.read_bytes()
    ledger_prefix = first_plan.actions[0].ledger.read_bytes()
    assert [(row["stmt_i"], row["attempt_ordinal"]) for row in original] == [
        (0, 1),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
    ]
    assert [row["row_status"] for row in original] == [
        "scored",
        "error",
        "error",
        "error",
        "error",
        "error",
    ]

    value = _amended_attempt_limit(path)
    path.write_bytes(canonical_json_line(value))
    delays: list[float] = []
    resumed_plan, resumed_client, _events, resumed = _execute(
        path,
        sleep=delays.append,
    )
    assert resumed.status == "complete"
    assert resumed.completed_this_run == 2
    assert resumed_client.calls == 2
    assert delays == [16]
    assert resumed_plan.actions[0].output.read_bytes().startswith(raw_prefix)
    assert resumed_plan.actions[0].ledger.read_bytes().startswith(ledger_prefix)
    rows = [
        json.loads(line)
        for line in resumed_plan.actions[0].output.read_text().splitlines()
    ]
    assert [(row["stmt_i"], row["attempt_ordinal"]) for row in rows] == [
        (0, 1),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (2, 1),
    ]
    assert [row["row_status"] for row in rows] == [
        "scored",
        "error",
        "error",
        "error",
        "error",
        "error",
        "scored",
        "scored",
    ]
    call_ids = [call["call_id"] for row in rows for call in row["call_log"]]
    assert len(call_ids) == len(set(call_ids)) == 8
    starts = [
        row
        for row in map(
            json.loads, resumed_plan.actions[0].ledger.read_text().splitlines()
        )
        if row["event"] == "attempt_started"
    ]
    by_execution: dict[str, list[int]] = {}
    for row in starts:
        by_execution.setdefault(row["execution_id"], []).append(row["attempt_ordinal"])
    assert sorted(by_execution.values()) == [[1], [1], [1, 2, 3, 4, 5, 6]]


def test_invalid_model_output_is_durable_and_retried(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    plan, client, _events, summary = _execute(path, invalid_first=True)
    assert client.calls == 2
    assert summary.status == "complete"
    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert [(row["attempt_ordinal"], row["row_status"]) for row in rows] == [
        (1, "error"),
        (2, "scored"),
    ]
    assert rows[0]["error"]["type"] == "InvalidModelOutput"
    assert rows[0]["verdict"] is None
    assert rows[0]["call_log"][0]["content"].endswith(
        '"verdict":"maybe","confidence":"medium"}'
    )
    assert rows[1]["verdict"] == "correct"


def _append_legacy_scored_null(path: Path) -> bytes:
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    base = FakeClient([], plan.actions[0].ledger, invalid_first=True)
    client = runner._DeadlineClient(
        runner.GuardedModelClient(base, prepared.action_guard), prepared.deadline
    )
    source = prepared.index.executions[0]
    with prepared.action_guard.attempt(runner.execution_identity(source)) as receipt:
        result = runner.score_execution(
            prepared.index,
            source,
            client,
            main_max_tokens=prepared.action.main_max_output_tokens,
        )
        receipt = prepared.action_guard.ensure_attempt_started()
        legacy = runner.result_row(
            source,
            action=prepared.action,
            result=result,
            attempt=runner._attempt_projection(receipt, "completed"),
            latency_s=0,
        )
        runner._commit(prepared, source, legacy)
        prepared.output.append(legacy)
    prepared.close()
    return plan.actions[0].output.read_bytes()


def test_legacy_scored_null_row_remains_evidence_and_gets_second_attempt(
    tmp_path: Path,
) -> None:
    path = _write_fixture(tmp_path, max_attempts=5)
    frozen_prefix = _append_legacy_scored_null(path)
    plan = load_run_plan(path)

    status = runner.inspect_plan(plan)
    assert status.next_action_id == "action_1"
    assert status.actions[0].completed == 0

    resumed_plan, client, _events, summary = _execute(path)
    assert client.calls == 1
    assert summary.status == "complete"
    current = resumed_plan.actions[0].output.read_bytes()
    assert current.startswith(frozen_prefix)
    rows = [json.loads(line) for line in current.splitlines()]
    assert [(row["attempt_ordinal"], row["row_status"], row["verdict"]) for row in rows] == [
        (1, "scored", None),
        (2, "scored", "correct"),
    ]


def test_legacy_scored_null_counts_toward_invalid_output_limit(
    tmp_path: Path,
) -> None:
    path = _write_fixture(tmp_path, max_attempts=10)
    _append_legacy_scored_null(path)

    plan, client, _events, summary = _execute(path, invalid_responses=10)
    assert summary.status == "partial"
    # The pre-existing legacy row already consumes one slot, so the source may
    # make only LIMIT-1 further calls before it is capped. Written against the
    # constant rather than a literal: the POINT of this test is that a legacy
    # `scored`-with-null-verdict row counts toward the cap at all, not the
    # cap's particular value.
    budget = runner.INVALID_MODEL_OUTPUT_LIMIT - 1
    assert client.calls == budget
    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert [(row["attempt_ordinal"], row["row_status"]) for row in rows] == [
        (1, "scored"),
        *[(ordinal, "error") for ordinal in range(2, 2 + budget)],
    ]
    assert {row["error"]["type"] for row in rows[1:]} == {"InvalidModelOutput"}


def test_rolling_window_reuses_only_distinct_idle_clients(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, workers=3, execution_count=7)
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    clients: list[FakeClient] = []

    def factory(_token: str, _action: Any) -> FakeClient:
        client = FakeClient([], plan.actions[0].ledger)
        clients.append(client)
        return client

    summary = runner.run_prepared(
        prepared,
        ready_writer=lambda value: None,
        token_reader=lambda: "secret",
        client_factory=factory,
    )
    assert summary.status == "complete"
    assert len(clients) == 3
    assert sum(client.calls for client in clients) == 7
    assert all(client.calls >= 1 for client in clients)
    assert len(plan.actions[0].output.read_text().splitlines()) == 7


def test_first_failure_stops_replenishment_then_drains_bounded_window(
    tmp_path: Path,
) -> None:
    path = _write_fixture(
        tmp_path, workers=3, execution_count=8, max_attempts=1
    )
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    state = WindowState(3)
    clients: list[WindowClient] = []
    append_threads: list[int] = []
    original_append = prepared.output.append

    def serialized_append(row: Mapping[str, Any]) -> None:
        append_threads.append(threading.get_ident())
        original_append(row)

    prepared.output.append = serialized_append

    def factory(_token: str, _action: Any) -> WindowClient:
        client = WindowClient([], plan.actions[0].ledger, state, fail=not clients)
        clients.append(client)
        return client

    coordinator = threading.get_ident()
    summary = runner.run_prepared(
        prepared,
        ready_writer=lambda value: None,
        token_reader=lambda: "secret",
        client_factory=factory,
    )
    assert summary.status == "partial"
    assert summary.completed_this_run == 2
    assert summary.failure and summary.failure["kind"] == "attempt_failed"
    assert state.maximum_active == 3
    assert len(clients) == 3
    assert sum(client.calls for client in clients) == 3
    assert all(client.calls == 1 and not client.overlap for client in clients)
    assert append_threads == [coordinator] * 3
    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert len(rows) == 3
    assert sorted(row["row_status"] for row in rows) == ["error", "scored", "scored"]


def test_drained_failures_report_lowest_replay_position(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path, workers=3, execution_count=6, max_attempts=1
    )
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    state = WindowState(3)
    clients: list[WindowClient] = []

    def factory(_token: str, _action: Any) -> WindowClient:
        position = len(clients)
        client = WindowClient(
            [],
            plan.actions[0].ledger,
            state,
            fail=position in {0, 1},
            failure_message="lower replay position" if position == 0 else "observed first",
            delay=0.2 if position == 0 else 0,
        )
        clients.append(client)
        return client

    summary = runner.run_prepared(
        prepared,
        ready_writer=lambda value: None,
        token_reader=lambda: "secret",
        client_factory=factory,
    )
    assert summary.status == "partial"
    assert summary.failure is not None
    assert summary.failure["message_sha256"] == hashlib.sha256(
        b"lower replay position"
    ).hexdigest()
    assert sum(client.calls for client in clients) == 3


def test_action_cap_refuses_call_after_credential_boundary(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, action_cap="0.000001")
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    events: list[str] = []
    client = FakeClient(events, plan.actions[0].ledger)
    summary = runner.run_prepared(
        prepared,
        ready_writer=lambda _value: events.append("ready"),
        token_reader=lambda: (events.append("token") or "secret"),
        client_factory=lambda _token, _action: client,
    )
    assert summary.status == "spend_cap"
    assert client.calls == 0
    assert plan.actions[0].output.read_bytes() == b""
    assert not any(
        json.loads(line)["event"] == "call_reserved"
        for line in plan.actions[0].ledger.read_text().splitlines()
    )


def test_action_cap_check_is_atomic_across_workers(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path,
        action_cap="0.0017",
        workers=2,
        execution_count=2,
    )
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    reserved = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def hold_first() -> None:
        try:
            source = prepared.index.executions[0]
            with prepared.action_guard.attempt(runner.execution_identity(source)):
                prepared.action_guard.reserve_call(
                    provider_model_id="zai.glm-5",
                    kind="monolithic",
                    max_output_tokens=16,
                    system="s",
                    messages=[{"role": "user", "content": "q"}],
                )
                reserved.set()
                assert release.wait(timeout=5)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=hold_first)
    worker.start()
    assert reserved.wait(timeout=5)
    source = prepared.index.executions[1]
    with prepared.action_guard.attempt(runner.execution_identity(source)):
        with pytest.raises(runner.ActionCapReached):
            prepared.action_guard.reserve_call(
                provider_model_id="zai.glm-5",
                kind="monolithic",
                max_output_tokens=16,
                system="s",
                messages=[{"role": "user", "content": "q"}],
            )
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert failures == []
    assert str(prepared.action_guard._commitment()) == "0.0016188"
    prepared.close()


def test_expired_prepared_action_never_calls_provider(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, deadline_seconds=1)
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan, now=lambda: 0.0)
    client = FakeClient([], plan.actions[0].ledger)
    summary = runner.run_prepared(
        prepared,
        ready_writer=lambda _value: None,
        token_reader=lambda: "secret",
        client_factory=lambda _token, _action: client,
    )
    assert summary.status == "deadline"
    assert client.calls == 0


def test_wal_outcome_without_raw_row_is_recovered_without_model_call(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    client = FakeClient([], plan.actions[0].ledger)
    guarded = runner.GuardedModelClient(client, prepared.action_guard)
    bounded = runner._DeadlineClient(guarded, prepared.deadline)
    row, error = runner._attempt(prepared, bounded, prepared.index.executions[0])
    assert error is None and row is not None
    assert prepared.output.path.read_bytes() == b""
    prepared.close()

    with pytest.raises(runner.RunnerError, match="completed during WAL recovery"):
        runner.prepare_run(plan, action_id="action_1")
    assert plan.actions[0].output.read_bytes() == canonical_json_line(row)
    assert runner.inspect_plan(plan).status == "complete"


def test_wal_recovery_fills_a_nonprefix_concurrent_hole(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, execution_count=2)
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    base = FakeClient([], plan.actions[0].ledger)
    client = runner._DeadlineClient(
        runner.GuardedModelClient(base, prepared.action_guard), prepared.deadline
    )
    first, first_error = runner._attempt(
        prepared, client, prepared.index.executions[0]
    )
    second, second_error = runner._attempt(
        prepared, client, prepared.index.executions[1]
    )
    assert first_error is second_error is None
    assert first is not None and second is not None
    prepared.output.append(second)
    prepared.close()

    with pytest.raises(runner.RunnerError, match="completed during WAL recovery"):
        runner.prepare_run(plan, action_id="action_1")
    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert [row["stmt_i"] for row in rows] == [1, 0]
    assert runner.inspect_plan(plan).status == "complete"


def test_wal_recovery_projects_invalid_provider_evidence_as_retryable_error(
    tmp_path: Path,
) -> None:
    path = _write_fixture(tmp_path)
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    source = prepared.index.executions[0]
    base = FakeClient([], plan.actions[0].ledger, invalid_first=True)
    client = runner._DeadlineClient(
        runner.GuardedModelClient(base, prepared.action_guard), prepared.deadline
    )
    context = prepared.action_guard.attempt(runner.execution_identity(source))
    context.__enter__()
    result = runner.score_execution(
        prepared.index,
        source,
        client,
        main_max_tokens=prepared.action.main_max_output_tokens,
    )
    assert result["verdict"] is None
    prepared.close()  # process loss after settled response evidence, before outcome
    del context

    recovered = runner.prepare_run(plan)
    try:
        assert recovered.resume.status == "partial"
        assert recovered.resume.done == frozenset()
        row = recovered.resume.rows[0]
        assert row["row_status"] == "error"
        assert row["error"]["type"] == "InvalidModelOutput"
        assert row["attempt_ordinal"] == 1
    finally:
        recovered.close()


@pytest.mark.parametrize("failure", ["token", "factory", "config"])
def test_boundary_failure_always_releases_output_and_ledger_locks(
    tmp_path: Path, failure: str
) -> None:
    path = _write_fixture(tmp_path)
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)

    def token_reader():
        if failure == "token":
            raise RuntimeError("token fixture")
        return "secret"

    def client_factory(_token, _action):
        if failure == "factory":
            raise RuntimeError("factory fixture")
        client = FakeClient([], plan.actions[0].ledger)
        if failure == "config":
            client.config["model_id"] = "foreign.model"
        return client

    with pytest.raises((RuntimeError, runner.RunnerError)):
        runner.run_prepared(
            prepared,
            ready_writer=lambda _value: None,
            token_reader=token_reader,
            client_factory=client_factory,
        )
    replacement = runner.prepare_run(plan)
    replacement.close()


def test_cli_status_and_child_readiness_need_no_provider_call(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.pop("AWS_BEARER_TOKEN_BEDROCK", None)
    status = subprocess.run(
        [sys.executable, "-m", "indra_belief.comparison", "status", "--plan", str(path)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(status.stdout)["next_action_id"] == "action_1"

    ready_read, ready_write = os.pipe()
    token_read, token_write = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "indra_belief.comparison",
            "_run-child",
            "--plan",
            str(path),
            "--ready-fd",
            str(ready_write),
            "--token-fd",
            str(token_read),
        ],
        cwd=ROOT,
        env=environment,
        pass_fds=(ready_write, token_read),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(ready_write)
    os.close(token_read)
    with os.fdopen(ready_read, "rb") as stream:
        readiness = json.loads(stream.readline())
    assert readiness["status"] == "ready_for_bearer_token"
    assert readiness["token_read"] is False
    assert readiness["provider_calls_started_during_preflight"] == 0
    os.close(token_write)  # EOF: fail before client construction/provider access.
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 2, (stdout, stderr)
    plan = load_run_plan(path)
    assert plan.actions[0].output.read_bytes() == b""
    assert not any(
        json.loads(line)["event"] == "call_reserved"
        for line in plan.actions[0].ledger.read_text().splitlines()
    )
    replacement = runner.prepare_run(plan)
    replacement.close()


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        ("TimeoutError", "transport_or_server"),
        ("ConnectionError", "transport_or_server"),
        ("ActionDeadlineExceeded", "transport_or_server"),
        ("InterruptedAfterDurableProviderEvidence", "transport_or_server"),
        ("SpendReservationBreach", "transport_or_server"),
        ("InvalidModelOutput", "invalid_model_output"),
        ("SomeProviderRefusal", None),
    ],
)
def test_row_retry_class_keeps_operational_interruptions_retryable(
    error_type: str, expected: str | None
) -> None:
    row = {"row_status": "error", "error": {"type": error_type}, "call_log": []}
    assert runner._row_retry_class(row) == expected
