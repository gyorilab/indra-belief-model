from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from indra_belief.comparison import llm, replay, runner
from indra_belief.comparison.contracts import (
    ContractError,
    canonical_json_bytes,
    canonical_json_line,
    canonical_sha256,
    load_run_plan,
)
from indra_belief.comparison.replay import prompt_sha256, source_key
from indra_belief.verdict import parse_verdict


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
    distinct_claims: bool = False,
) -> Path:
    """Write a run plan.

    `distinct_claims` gives each execution row its own claim text, so a fixture
    client can decide per SOURCE rather than per worker.  Off by default: every
    other test wants the identical prompt for every row.
    """
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
        if distinct_claims:
            row["claim"] = f"A [Activation] B{position}"
            row_user = f'CLAIM: {row["claim"]}\nEVIDENCE: "A activates B."'
            row["main_user_before_relation_note_sha256"] = hashlib.sha256(
                row_user.encode()
            ).hexdigest()
            row["relation_note_insertion"]["utf8_byte_offset"] = len(row_user)
            row["main_prompt_base_sha256"] = prompt_sha256(
                system, [{"role": "user", "content": row_user}]
            )
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


class SourceKeyedClient(FakeClient):
    """A client whose answer depends on the SOURCE, not on which worker it is.

    Quarantine is a per-source decision, so a fixture that can only fail "the
    first client" cannot express "position 3 is unscorable while positions 0-2
    and 4-7 are fine".  Requires `_write_fixture(distinct_claims=True)`, whose
    claims carry the replay position.
    """

    def __init__(self, events, ledger, *, off_grid: set[int] = frozenset(),
                 raising: set[int] = frozenset()) -> None:
        super().__init__(events, ledger)
        self.off_grid, self.raising = set(off_grid), set(raising)
        self.positions: list[int] = []

    def call(self, *, system, messages, max_tokens=None, temperature=0.1,
             response_format=None, reasoning_effort=None, kind="unknown"):
        content = messages[-1]["content"]
        position = int(content.split("A [Activation] B")[1].split("\n")[0])
        self.positions.append(position)
        common = {
            "kind": kind, "model_id": self.config["model_id"],
            "max_tokens": max_tokens, "system": system, "messages": messages,
        }
        if position in self.raising:
            self.calls += 1
            error = ValueError("nonretryable fixture failure")
            self.log.append({
                **common, "content": None, "reasoning": None, "raw_text": None,
                "prompt_tokens": None, "out_tokens": None, "finish_reason": None,
                "error": type(error).__name__,
            })
            raise error
        response = FakeResponse(
            content=(
                # On-grid verdict, off-grid confidence: the exact shape measured
                # on 2026-07-31 that wedged an arm.
                '{"support":"span","objection":null,"verdict":"correct",'
                '"confidence":"certain"}'
                if position in self.off_grid
                else FakeResponse().content
            )
        )
        self.calls += 1
        self.log.append({
            **common, "content": response.content, "reasoning": response.reasoning,
            "raw_text": response.raw_text, "prompt_tokens": response.prompt_tokens,
            "out_tokens": response.tokens, "finish_reason": response.finish_reason,
        })
        return response


def _run_with_clients(plan, factory, *, prepared=None):
    """Drive one action to completion with a caller-supplied client factory."""
    run = runner.prepare_run(plan) if prepared is None else prepared
    return runner.run_prepared(
        run,
        ready_writer=lambda _value: None,
        token_reader=lambda: "secret",
        client_factory=factory,
    )


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
    assert rows[-1]["score"] is None


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
    # The fixture's one source is retired, so nothing in this action is
    # schedulable: "settled", never "complete".
    assert summary.status == "settled"
    assert client.calls == runner.INVALID_MODEL_OUTPUT_LIMIT == 5
    raw_prefix = plan.actions[0].output.read_bytes()
    ledger_prefix = plan.actions[0].ledger.read_bytes()
    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert [row["attempt_ordinal"] for row in rows] == [1, 2, 3, 4, 5]
    assert {row["error"]["type"] for row in rows} == {"InvalidModelOutput"}

    value = _amended_attempt_limit(path)
    path.write_bytes(canonical_json_line(value))
    resumed_plan = load_run_plan(path)
    resumed_action, resumed_stage, resumed_index = _resume_scope(resumed_plan)
    # Raising max_attempts 5 -> 10 retires `attempts_exhausted` but NOT the
    # invalid-output cap, so the source stays settled and the action stays
    # terminal.  Zero further calls is now guaranteed by there being no run at
    # all: `prepare_run` refuses a terminal action rather than starting one that
    # would demand a bearer token and build no clients.
    assert replay.load_resume(
        resumed_action.output, index=resumed_index, action=resumed_action,
        model=resumed_stage.model,
        provider_model_id=resumed_stage.provider_model_id,
    ).settled == {(0, 0): "invalid_model_output_limit"}
    with pytest.raises(runner.RunnerError, match="already complete"):
        runner.prepare_run(resumed_plan)
    assert resumed_plan.actions[0].output.read_bytes() == raw_prefix
    assert resumed_plan.actions[0].ledger.read_bytes() == ledger_prefix


def _resume_scope(plan):
    """(action, stage, index) for the single-action fixture plans above."""
    action = plan.actions[0]
    stage = plan.stage_by_id[action.stage_id]
    return action, stage, runner._scopes(plan)[action.id]


def _attempt_projection_for(source, *, action, model, ordinal, status):
    return {
        "execution_id": replay.expected_execution_id(
            source, action=action, model=model
        ),
        "attempt_id": hashlib.sha256(
            f"{source['stmt_i']}:{source['evidence_i']}:{ordinal}:{status}".encode()
        ).hexdigest()[:32],
        "attempt_ordinal": ordinal,
        "attempt_status": status,
    }


def _resume_error_row(source, *, action, model, ordinal, error_type):
    return replay.error_row(
        source,
        action=action,
        calls=[],
        attempt=_attempt_projection_for(
            source, action=action, model=model, ordinal=ordinal, status="error"
        ),
        latency_s=0.0,
        error=error_type,
    )


def _resume_scored_row(
    source, *, action, model, provider_model_id, ordinal,
    verdict="correct", score=None, call_id=None,
):
    attempt = _attempt_projection_for(
        source, action=action, model=model, ordinal=ordinal, status="completed"
    )
    call = {
        "execution_id": attempt["execution_id"],
        "attempt_id": attempt["attempt_id"],
        "attempt_ordinal": ordinal,
        "call_ordinal": 1,
        "call_id": call_id or hashlib.sha256(
            f"call:{attempt['attempt_id']}".encode()
        ).hexdigest()[:32],
        "model_id": provider_model_id,
        "kind": "monolithic",
        "provider_request_sha256": hashlib.sha256(
            attempt["attempt_id"].encode()
        ).hexdigest(),
    }
    return replay.result_row(
        source,
        action=action,
        result={
            "verdict": verdict,
            "confidence": "high",
            "score": score,
            "tier": "llm_comprehension",
            "grounding_status": "all_match",
            "provenance_triggered": False,
            "tokens": 3,
            "call_log": [call],
            "raw_text": "{}",
        },
        attempt=attempt,
        latency_s=0.0,
    )


@pytest.mark.parametrize("historical_score", [0, 0.137, 1])
def test_resume_accepts_historical_probability_scores(
    tmp_path: Path, historical_score: float,
) -> None:
    """Frozen rows retain any finite numeric probability in the unit interval."""
    path = _write_fixture(tmp_path)
    plan = load_run_plan(path)
    action, stage, index = _resume_scope(plan)
    row = _resume_scored_row(
        index.executions[0],
        action=action,
        model=stage.model,
        provider_model_id=stage.provider_model_id,
        ordinal=1,
        score=historical_score,
    )
    _write_rows(action, [row])

    resume = replay.load_resume(
        action.output,
        index=index,
        action=action,
        model=stage.model,
        provider_model_id=stage.provider_model_id,
    )
    assert resume.rows[0]["score"] == historical_score


@pytest.mark.parametrize("invalid_score", [-0.001, 1.001, True, "0.5"])
def test_resume_rejects_non_probability_historical_scores(
    tmp_path: Path, invalid_score: Any,
) -> None:
    path = _write_fixture(tmp_path)
    plan = load_run_plan(path)
    action, stage, index = _resume_scope(plan)
    row = _resume_scored_row(
        index.executions[0],
        action=action,
        model=stage.model,
        provider_model_id=stage.provider_model_id,
        ordinal=1,
        score=invalid_score,
    )
    _write_rows(action, [row])

    with pytest.raises(
        replay.ReplayError,
        match="score is neither null nor a probability",
    ):
        replay.load_resume(
            action.output,
            index=index,
            action=action,
            model=stage.model,
            provider_model_id=stage.provider_model_id,
        )


def _write_rows(action, rows) -> None:
    action.output.parent.mkdir(parents=True, exist_ok=True)
    action.output.write_bytes(b"".join(canonical_json_line(row) for row in rows))


def test_invalid_output_count_is_indexed_by_source(tmp_path: Path) -> None:
    """The invalid-output cap reads a per-source index, not a per-source scan.

    `_run_source` used to re-scan every parsed row for each pending source.  The
    count now comes from `ResumeState.invalid_outputs`, so that map must carry a
    SEPARATE count per source key and must omit a source that has none — a
    global total, or a default-zero entry for every source, would silently cap
    the wrong sources.
    """
    path = _write_fixture(tmp_path, execution_count=3, max_attempts=5)
    plan = load_run_plan(path)
    action, stage, index = _resume_scope(plan)
    rows = []
    for position, error_types in enumerate((
        ("InvalidModelOutput", "InvalidModelOutput"),
        ("InvalidModelOutput", "InvalidModelOutput", "InvalidModelOutput"),
        ("TimeoutError",),
    )):
        source = index.executions[position]
        for ordinal, error_type in enumerate(error_types, start=1):
            rows.append(_resume_error_row(
                source, action=action, model=stage.model,
                ordinal=ordinal, error_type=error_type,
            ))
    _write_rows(action, rows)

    resume = replay.load_resume(
        action.output, index=index, action=action, model=stage.model,
        provider_model_id=stage.provider_model_id,
    )
    assert resume.attempts == {(0, 0): 2, (1, 0): 3, (2, 0): 1}
    assert resume.invalid_outputs == {(0, 0): 2, (1, 0): 3}
    assert (2, 0) not in resume.invalid_outputs

    # End to end: the cap still trips at the limit, and the resumed decision is
    # taken from the index rather than from a fresh scan.
    capped_path = _write_fixture(
        tmp_path / "capped", max_attempts=5, primary_actions=True
    )
    capped_plan, client, _events, summary = _execute(
        capped_path, invalid_responses=5
    )
    assert client.calls == runner.INVALID_MODEL_OUTPUT_LIMIT == 5
    assert summary.status == "settled"
    capped_action, capped_stage, capped_index = _resume_scope(capped_plan)
    capped_resume = replay.load_resume(
        capped_action.output, index=capped_index, action=capped_action,
        model=capped_stage.model, provider_model_id=capped_stage.provider_model_id,
    )
    assert capped_resume.invalid_outputs == {
        (0, 0): runner.INVALID_MODEL_OUTPUT_LIMIT
    }
    assert capped_resume.settled == {(0, 0): "invalid_model_output_limit"}
    # The capped source is the action's only source, so the action is terminal
    # and is never offered again — the strongest form of "zero further calls".
    with pytest.raises(runner.RunnerError, match="already complete"):
        runner.prepare_run(capped_plan)


def _resume_case_rows(name, *, action, stage, index):
    """Rows for one state of the resume matrix, or None to leave no file."""
    first, second = index.executions[0], index.executions[1]
    scored = dict(
        action=action, model=stage.model,
        provider_model_id=stage.provider_model_id,
    )
    failed = dict(action=action, model=stage.model)
    if name == "missing":
        return None
    if name == "empty":
        return []
    if name == "partial":
        return [_resume_scored_row(first, ordinal=1, **scored)]
    if name == "complete":
        return [
            _resume_scored_row(first, ordinal=1, **scored),
            _resume_scored_row(second, ordinal=1, **scored),
        ]
    if name == "retried_then_scored":
        return [
            _resume_error_row(first, ordinal=1, error_type="TimeoutError", **failed),
            _resume_scored_row(first, ordinal=2, **scored),
            _resume_scored_row(second, ordinal=1, **scored),
        ]
    if name == "error_rows":
        return [
            _resume_error_row(first, ordinal=1, error_type="TimeoutError", **failed),
            _resume_error_row(second, ordinal=1, error_type="InvalidModelOutput",
                              **failed),
        ]
    raise AssertionError(f"unknown resume state {name!r}")


def _corrupt_resume_bytes(name, *, action, stage, index):
    """Raw bytes for one corruption class the parse loop must reject."""
    first, second = index.executions[0], index.executions[1]
    scored = dict(
        action=action, model=stage.model,
        provider_model_id=stage.provider_model_id,
    )
    if name == "non_canonical":
        # Same object, re-encoded with default spacing and insertion order.
        return json.dumps(_resume_scored_row(first, ordinal=1, **scored)).encode() + b"\n"
    if name == "foreign_source":
        row = _resume_scored_row(first, ordinal=1, **scored)
        row["stmt_i"] = 99
        return canonical_json_line(row)
    if name == "noncontiguous_ordinal":
        return canonical_json_line(_resume_scored_row(first, ordinal=2, **scored))
    if name == "post_terminal_append":
        return canonical_json_line(
            _resume_scored_row(first, ordinal=1, **scored)
        ) + canonical_json_line(_resume_error_row(
            first, action=action, model=stage.model, ordinal=2,
            error_type="TimeoutError",
        ))
    if name == "repeated_call_id":
        shared = "c" * 32
        return canonical_json_line(
            _resume_scored_row(first, ordinal=1, call_id=shared, **scored)
        ) + canonical_json_line(
            _resume_scored_row(second, ordinal=1, call_id=shared, **scored)
        )
    if name == "partial_trailing_line":
        return canonical_json_line(
            _resume_scored_row(first, ordinal=1, **scored)
        ).rstrip(b"\n")
    raise AssertionError(f"unknown corruption {name!r}")


RESUME_STATES = (
    "missing", "empty", "partial", "complete", "retried_then_scored", "error_rows",
)
RESUME_CORRUPTIONS = (
    "non_canonical", "foreign_source", "noncontiguous_ordinal",
    "post_terminal_append", "repeated_call_id", "partial_trailing_line",
)


def test_resume_status_matches_load_resume(tmp_path: Path) -> None:
    """`resume_status` is a projection of `load_resume`, never a second parser.

    `prepare_run` and `inspect_plan` consume only a status, a completed count and
    an attempt count, so they now fold the shared scanner without retaining rows.
    That is only safe if the cheap fold ACCEPTS exactly what the full loader
    accepts and REJECTS exactly what it rejects, with the same message — a
    second, laxer validation chain would let a corrupt output reach readiness.
    """
    for name in RESUME_STATES:
        path = _write_fixture(tmp_path / f"state_{name}", execution_count=2,
                              max_attempts=5)
        plan = load_run_plan(path)
        action, stage, index = _resume_scope(plan)
        rows = _resume_case_rows(name, action=action, stage=stage, index=index)
        if rows is not None:
            _write_rows(action, rows)
        assert action.output.exists() is (rows is not None), name
        arguments = dict(
            index=index, action=action, model=stage.model,
            provider_model_id=stage.provider_model_id,
        )
        loaded = replay.load_resume(action.output, **arguments)
        assert replay.resume_status(action.output, **arguments) == replay.ResumeStatus(
            loaded.status, len(loaded.done), len(loaded.rows)
        ), name

    for name in RESUME_CORRUPTIONS:
        path = _write_fixture(tmp_path / f"corrupt_{name}", execution_count=2,
                              max_attempts=5)
        plan = load_run_plan(path)
        action, stage, index = _resume_scope(plan)
        action.output.parent.mkdir(parents=True, exist_ok=True)
        action.output.write_bytes(
            _corrupt_resume_bytes(name, action=action, stage=stage, index=index)
        )
        arguments = dict(
            index=index, action=action, model=stage.model,
            provider_model_id=stage.provider_model_id,
        )
        with pytest.raises(replay.ReplayError) as loaded_error:
            replay.load_resume(action.output, **arguments)
        with pytest.raises(replay.ReplayError) as status_error:
            replay.resume_status(action.output, **arguments)
        assert str(status_error.value) == str(loaded_error.value), name


def test_parser_does_not_promote_an_invalid_verdict_value() -> None:
    """A closed-enum field filled with the OTHER field's vocabulary. The reply
    names no verdict, so there is no Verdict — and therefore no score for the
    runner to write. That absence is what drives the retry."""
    assert parse_verdict(
        '{"support":"span","objection":"issue","verdict":"medium","confidence":"medium"}'
    ) is None


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
    # Under the FROZEN bounds the source is retired: five attempts, five
    # allowed. "Settled" is relative to `max_attempts`, so it is exactly what
    # the amendment below un-does — raising the limit makes the source
    # schedulable again and the action partial again.
    assert first.status == "settled"
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
    assert summary.status == "settled"
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


def test_two_clients_for_one_model_do_not_share_mutable_config() -> None:
    # `_DeadlineClient.call` ratchets `client.config["timeout"]` down to the
    # remaining action budget. Every worker in a run builds its own client for
    # the same model, so that write must not reach a sibling worker or the
    # process-wide registry — a shrink there would silently outlive the call.
    import time

    from indra_belief.model_client import LOCAL_MODELS, ModelClient

    original = LOCAL_MODELS["local-gemma-4-26b"]["timeout"]
    # Read the registry rather than pinning a literal: what this test guards is
    # that the ratchet does not leak between clients, not the entry's timeout.
    # (The literal was 60 until the MLX serving work raised it — a 60s cap
    # cannot fit a thinking model generating ~500 tokens at the measured local
    # rate of 20.9-29.3 tok/s, mean 25.2.)
    assert original > 5, "the ratchet needs room to shrink for this test to mean anything"
    try:
        a = ModelClient("local-gemma-4-26b")
        b = ModelClient("local-gemma-4-26b")
        assert a.config is not b.config
        assert a.config is not LOCAL_MODELS["local-gemma-4-26b"]

        sentinel = object()
        a.call = lambda *args, **kwargs: sentinel  # never contact a provider
        bounded = runner._DeadlineClient(a, time.monotonic() + 5)
        assert bounded.call("prompt") is sentinel

        assert a.config["timeout"] <= 5  # the ratchet fired on this client
        assert b.config["timeout"] == original
        assert LOCAL_MODELS["local-gemma-4-26b"]["timeout"] == original
    finally:
        LOCAL_MODELS["local-gemma-4-26b"]["timeout"] = original


def test_unclassified_failure_stops_replenishment_then_drains_bounded_window(
    tmp_path: Path,
) -> None:
    """An UNCLASSIFIED failure still halts the action on its first occurrence.

    `WindowClient(fail=True)` raises a bare ValueError, which classifies as
    "other" and reaches `_failure_disposition` as an `attempt_failed` of a type
    nobody has allowlisted.  The taxonomy defaults to halt, so the behavior here
    is unchanged from before quarantine existed: replenishment stops, the
    in-flight window drains, and no further source is dispatched.
    """
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
    # The halted source has no retry class, so the durable rows retire it and
    # the action is terminal. `quarantined == 0` is the load-bearing half: this
    # run halted, it did not absorb anything, and the disposition says so.
    assert summary.status == "settled"
    assert summary.status != "complete"
    assert summary.completed_this_run == 2
    assert summary.failure and summary.failure["kind"] == "attempt_failed"
    assert summary.failure["disposition"] == "halt"
    assert summary.quarantined == 0
    assert state.maximum_active == 3
    assert len(clients) == 3
    assert sum(client.calls for client in clients) == 3
    assert all(client.calls == 1 and not client.overlap for client in clients)
    assert append_threads == [coordinator] * 3
    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert len(rows) == 3
    assert sorted(row["row_status"] for row in rows) == ["error", "scored", "scored"]


def test_drained_unclassified_failures_report_lowest_replay_position(
    tmp_path: Path,
) -> None:
    """Among failures of EQUAL severity, the lowest replay position is reported.

    Both drained failures are unclassified halts, so severity cannot separate
    them and the tie-break is position — the run is reported by the earliest row
    that stopped it, not by whichever thread the scheduler observed first.
    """
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
    assert summary.status != "complete"
    assert summary.failure is not None
    assert summary.failure["disposition"] == "halt"
    assert summary.failure["message_sha256"] == hashlib.sha256(
        b"lower replay position"
    ).hexdigest()
    assert sum(client.calls for client in clients) == 3


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        # The allowlist: three settled reasons plus the live off-grid output.
        ({"kind": "nonretryable_failure_on_resume"}, "quarantine"),
        ({"kind": "invalid_model_output_limit"}, "quarantine"),
        ({"kind": "attempts_exhausted"}, "quarantine"),
        ({"kind": "attempt_failed", "type": "InvalidModelOutput"}, "quarantine"),
        # Budget and clock are statements about the run.
        ({"kind": "spend_cap", "type": "SpendCapReached"}, "halt"),
        ({"kind": "spend_cap", "type": "ActionCapReached"}, "halt"),
        ({"kind": "deadline"}, "halt"),
        ({"kind": "deadline", "type": "ActionDeadlineExceeded"}, "halt"),
        # Credentials, config and bad requests: every row would fail the same way.
        ({"kind": "attempt_failed", "type": "HTTPError",
          "provider_http_status": 401}, "halt"),
        ({"kind": "attempt_failed", "type": "HTTPError",
          "provider_http_status": 403}, "halt"),
        ({"kind": "attempt_failed", "type": "HTTPError",
          "provider_http_status": 400}, "halt"),
        ({"kind": "attempt_failed", "type": "HTTPError",
          "provider_http_status": 404}, "halt"),
        # Exhausted transport, including the 429 that step 1 made retryable.
        ({"kind": "attempt_failed", "type": "TimeoutError"}, "halt"),
        ({"kind": "attempt_failed", "type": "HTTPError",
          "provider_http_status": 429}, "halt"),
        ({"kind": "attempt_failed", "type": "HTTPError",
          "provider_http_status": 503}, "halt"),
        # Parser / row-shape / WAL / accounting.
        ({"kind": "attempt_failed", "type": "ReplayError"}, "halt"),
        ({"kind": "attempt_failed", "type": "ContractError"}, "halt"),
        ({"kind": "attempt_failed", "type": "RunnerError"}, "halt"),
        ({"kind": "attempt_failed", "type": "SpendGuardError"}, "halt"),
        ({"kind": "attempt_failed", "type": "SpendReservationBreach"}, "halt"),
        # The aggregate breaker is a statement about the ACTION, so it halts
        # even though every individual event that raised it was a quarantine.
        # The diagnostic budget is a statement about the ACTION, so it halts
        # even though every individual event that raised it was a quarantine.
        ({"kind": "quarantine_budget", "quarantined": 8}, "halt"),
        # Nothing unknown is ever admitted by default.
        ({"kind": "attempt_failed", "type": "SomeFutureProviderError"}, "halt"),
        ({"kind": "attempt_failed"}, "halt"),
        ({"kind": "a_kind_nobody_has_written_yet"}, "halt"),
        ({}, "halt"),
    ],
)
def test_failure_disposition_is_an_allowlist(
    failure: dict[str, Any], expected: str
) -> None:
    """Only the four quarantine cases cost one source; everything else halts.

    The point of the table is the DEFAULT.  A denylist would silently promote
    each newly observed failure type to "keep spending", so a systematic
    breakage — a revoked key, a bad parser profile, a wrong provider model id —
    could quarantine five thousand sources one at a time and burn the whole
    action cap proving the same thing five thousand times.

    The allowlist alone does not prevent that, because the systematic case
    arrives AS the allowlisted kind — see
    `test_a_systematic_failure_cannot_burn_the_action_cap`.
    """
    assert runner._failure_disposition(failure) == expected


def _offgrid_run(tmp_path: Path, *, execution_count: int, workers: int,
                 max_attempts: int, off_grid: set[int]):
    """Run one action against a client that is off-grid for chosen positions."""
    path = _write_fixture(
        tmp_path, workers=workers, execution_count=execution_count,
        max_attempts=max_attempts, distinct_claims=True,
    )
    plan = load_run_plan(path)
    clients: list[SourceKeyedClient] = []

    def factory(_token: str, _action: Any) -> SourceKeyedClient:
        client = SourceKeyedClient([], plan.actions[0].ledger, off_grid=off_grid)
        clients.append(client)
        return client

    summary = _run_with_clients(plan, factory)
    dispatched = {position for client in clients for position in client.positions}
    return plan, summary, sum(client.calls for client in clients), dispatched


def test_a_systematic_failure_cannot_burn_the_action_cap(tmp_path: Path) -> None:
    """The invariant that outranks quarantine: no unbounded spend, ever.

    Quarantine's allowlist contains `attempt_failed`/`InvalidModelOutput`, and a
    systematic breakage arrives AS that kind for every single source — a wrong
    `provider_model_id`, a provider that starts prefixing refusals, a
    `main_max_output_tokens` that truncates every reply.  Each one is
    individually "off-grid for this evidence", so per-source retirement never
    stops it.  Measured on this harness with the budget absent: 8 sources
    against a client off-grid for all of them cost 40 paid calls, 200 cost
    1,000, 1,000 cost 5,000, extrapolating to 33,361 x 5 = 166,805 bounded only
    by the action cap ($39.96 gemma_26b_primary, $309.54 glm_5_primary) for zero
    usable rows.

    The diagnostic budget stops the whole class after a couple of handfuls of
    sources, whatever the corpus size, and names the regime it found.  Without
    it this test dispatches all 200.
    """
    _plan, summary, calls, dispatched = _offgrid_run(
        tmp_path, execution_count=200, workers=4, max_attempts=5,
        off_grid=set(range(200)),
    )
    # The bound is the limit plus at most one draining window, NOT the corpus.
    ceiling = runner.QUARANTINE_DIAGNOSTIC_LIMIT + 4
    assert len(dispatched) <= ceiling
    assert calls <= ceiling * runner.INVALID_MODEL_OUTPUT_LIMIT
    assert summary.completed_this_run == 0
    assert summary.failure is not None
    assert summary.failure["kind"] == "quarantine_budget"
    # It must HALT: a quarantine disposition here would mean "keep going".
    assert summary.failure["disposition"] == "halt"
    # And it must say WHICH REGIME — the one thing the pre-S2 first-row halt
    # could never tell the operator.
    assert summary.failure["regime"] == "systematic"
    assert summary.failure["quarantined"] >= runner.QUARANTINE_DIAGNOSTIC_LIMIT
    # Holes exist, so the action is finished and unbundlable even though 189
    # sources were never touched. Scoring them could not produce a bundle.
    assert summary.status == "settled"
    assert summary.status != "complete"


def test_the_sporadic_regime_is_bounded_by_sources_not_by_quarantines(
    tmp_path: Path,
) -> None:
    """The second term of the same bound, and the one that costs real money.

    At the measured 0.057% the next bad row is ~1,756 sources away, so a
    quarantine-count limit alone would traverse the corpus looking for eight of
    them and spend the whole cap on a bundle that cannot exist.  The source
    limit stops that: one hole, then a bounded look around, then halt with
    "sporadic" — which is the answer the operator needed.

    WHAT IS ASSERTED, AND WHY IT IS AN IDENTITY RATHER THAN A WINDOW.  The
    earlier version of this test asserted `20 <= len(dispatched) <= 40` and
    flaked in about one full-suite run in three.  Widening it would have hidden
    the cause: the budget used to anchor on `next_pending` AT THE MOMENT THE
    FAILURE WAS DRAINED.  A retiring source is far slower than a healthy one —
    five attempts, with exponential backoff in production — so its siblings
    churn on while it retries, and the anchor drifted with machine load
    (measured 17-21 against a hole at index 3).  The budget then added its whole
    limit ON TOP of that drift, so what an arm paid after its first hole was a
    function of how busy the machine was.

    `len(dispatched)` still cannot be pinned exactly, and that part is
    irreducible: sources submitted before anyone knew the hole existed cannot be
    un-dispatched.  What the fix makes exact is that the budget now ABSORBS that
    drift instead of adding to it — the count it reports is measured from the
    hole itself.  So the load-independent fact, and the one asserted here, is the
    identity: the reported post-hole spend EQUALS the real post-hole dispatch.
    Measured 18/18 across hole positions 3/40/100, limits 5/10/20 and 4/8
    workers.  Under the old anchor it reported the limit while having dispatched
    drift + limit, and the identity broke.
    """
    hole, limit, workers = 3, 20, 4
    monkeypatch = pytest.MonkeyPatch()
    try:
        # 20 rather than 200 keeps the fixture small; the term under test is
        # the one the runner reads.
        monkeypatch.setattr(runner, "QUARANTINE_DIAGNOSTIC_SOURCES", limit)
        _plan, summary, _calls, dispatched = _offgrid_run(
            tmp_path, execution_count=200, workers=workers, max_attempts=5,
            off_grid={hole},
        )
    finally:
        monkeypatch.undo()
    assert summary.failure is not None
    since = summary.failure["dispatched_since_first_quarantine"]
    # THE IDENTITY: what it says it spent past the hole is what it spent.
    assert since == len(dispatched) - hole
    # The budget was genuinely met, and stopped at the first chance thereafter.
    assert since >= limit
    # Ceiling from the mechanism, not from watching: the drift is at most what
    # `workers - 1` slots can retire while one slot spends its whole
    # invalid-output budget, plus one batch of submission granularity.
    assert since <= limit + (workers - 1) * runner.INVALID_MODEL_OUTPUT_LIMIT + workers
    assert len(dispatched) < 200  # the corpus was never traversed
    assert summary.quarantined == 1
    assert summary.failure["kind"] == "quarantine_budget"
    assert summary.failure["regime"] == "sporadic"
    assert summary.failure["disposition"] == "halt"
    assert summary.status == "settled"


def test_the_budget_is_anchored_to_the_hole_not_to_when_it_was_noticed(
    tmp_path: Path,
) -> None:
    """The same hole, reached after very different amounts of prior work.

    This is the property the flake was a symptom of: what an arm PAYS after its
    first hole must be a function of where the hole is, not of how long the
    scheduler took to notice it.  A hole 100 sources into the corpus and a hole
    3 sources in must cost the SAME amount of post-hole work.

    The old anchor could not express that.  It measured from wherever the
    scheduler happened to be when the failure was drained, so it always reported
    exactly the limit while having actually dispatched drift + limit past the
    hole — the reported number and the real number were different, and only the
    real one is money.
    """
    limit, workers = 20, 4
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(runner, "QUARANTINE_DIAGNOSTIC_SOURCES", limit)
        spend: dict[int, int] = {}
        for hole in (3, 100):
            _plan, summary, _calls, dispatched = _offgrid_run(
                tmp_path / f"hole{hole}", execution_count=300, workers=workers,
                max_attempts=5, off_grid={hole},
            )
            assert summary.failure is not None
            since = summary.failure["dispatched_since_first_quarantine"]
            # The identity, at both positions: reported spend IS real spend.
            assert since == len(dispatched) - hole, (hole, since, len(dispatched))
            spend[hole] = since
    finally:
        monkeypatch.undo()
    # And the post-hole spend does not grow with how deep the hole is. Under the
    # old anchor this comparison had no stable answer at all.
    ceiling = limit + (workers - 1) * runner.INVALID_MODEL_OUTPUT_LIMIT + workers
    assert all(limit <= value <= ceiling for value in spend.values()), spend


def test_a_run_with_no_holes_at_all_is_never_bounded(tmp_path: Path) -> None:
    """The budget must not fire on a clean arm.

    It is armed by the FIRST quarantine and is inert before it, or the bound
    would throttle every healthy run in the fleet.
    """
    _plan, summary, _calls, dispatched = _offgrid_run(
        tmp_path, execution_count=200, workers=4, max_attempts=5,
        off_grid=set(),
    )
    assert len(dispatched) == 200
    assert summary.completed_this_run == 200
    assert summary.quarantined == 0
    assert summary.status == "complete"
    assert summary.failure is None


def test_an_action_holding_a_hole_is_never_dispatched_again(
    tmp_path: Path,
) -> None:
    """A restart must not pay the budget again, and again, and again.

    One hole makes the action unbundlable, so every further source it scores is
    money spent on an artifact that cannot exist.  The holes are re-derived from
    the durable rows for free, the action is not offered as ready, and nothing
    is dispatched — which is also what stops a supervisor delivering the
    unbounded burn in instalments of one budget per restart.
    """
    path = _write_fixture(
        tmp_path, workers=4, execution_count=200, max_attempts=5,
        distinct_claims=True,
    )
    plan = load_run_plan(path)
    first = _run_with_clients(plan, lambda _t, _a: SourceKeyedClient(
        [], plan.actions[0].ledger, off_grid=set(range(200))
    ))
    assert first.status == "settled"
    raw_prefix = plan.actions[0].output.read_bytes()
    ledger_prefix = plan.actions[0].ledger.read_bytes()

    # Not offered, so no process, no readiness, no token, no client, no call.
    with pytest.raises(runner.RunnerError, match="already complete"):
        runner.prepare_run(plan)
    status = runner.inspect_plan(plan)
    assert status.status == "settled"
    assert status.ready_action_ids == ()
    assert plan.actions[0].output.read_bytes() == raw_prefix
    assert plan.actions[0].ledger.read_bytes() == ledger_prefix


def test_a_run_with_nothing_schedulable_never_reaches_the_bearer_token(
    tmp_path: Path,
) -> None:
    """Defence in depth for the ready-before-token boundary.

    `ready_actions` already refuses to offer a settled action, so in practice no
    such run is ever prepared.  This drives `_run_prepared` directly with an
    injected hole because that ordering — readiness written, pending computed,
    token read ONLY if there is something to spend on — is the property the
    operator feels: before it, every futile invocation demanded a live bearer
    token to build zero clients and issue zero calls.
    """
    path = _write_fixture(
        tmp_path, workers=4, execution_count=8, max_attempts=1,
        distinct_claims=True,
    )
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    prepared.resume = replace(
        prepared.resume,
        settled={source_key(row): "attempts_exhausted"
                 for row in prepared.index.executions},
    )
    readiness: list[Mapping[str, Any]] = []

    def refuse_token() -> str:
        raise AssertionError("a run with nothing schedulable demanded a token")

    summary = runner.run_prepared(
        prepared,
        ready_writer=readiness.append,
        token_reader=refuse_token,
        client_factory=lambda _t, _a: pytest.fail("no client may be built"),
    )
    assert summary.quarantined == 8
    # Readiness still goes out — it carries the partition the operator reads.
    assert len(readiness) == 1
    assert readiness[0]["status"] == "ready_for_bearer_token"
    assert readiness[0]["pending"] == 0
    assert readiness[0]["quarantined"] == 8
    assert readiness[0]["token_read"] is False


def test_one_offgrid_source_is_quarantined_and_the_rest_of_the_arm_is_scored(
    tmp_path: Path,
) -> None:
    """The 2026-07-31 production halt, as a test.

    An arm died mid-corpus after 163 MB of progress because ONE evidence row
    came back with an off-grid confidence: the first failure of any kind stopped
    replenishment, and the operator had to raise a constant by hand.  Now that
    row costs exactly one source — every other source in the same run is still
    dispatched and scored.
    """
    path = _write_fixture(
        tmp_path, workers=3, execution_count=8, max_attempts=1,
        distinct_claims=True,
    )
    plan = load_run_plan(path)
    clients: list[SourceKeyedClient] = []

    def factory(_token: str, _action: Any) -> SourceKeyedClient:
        client = SourceKeyedClient(
            [], plan.actions[0].ledger, off_grid={0}
        )
        clients.append(client)
        return client

    summary = _run_with_clients(plan, factory)
    # Seven scored and one retired resolves all eight, so the action is
    # terminal — but "settled", never "complete", because one source has no
    # verdict and `llm.materialize_model_bundle` has to be told about it.
    assert summary.status == "settled"
    # Every source was attempted exactly once: the wedge did not block the
    # seven behind it, and nothing was attempted twice.
    assert sum(client.calls for client in clients) == 8
    assert sorted(
        position for client in clients for position in client.positions
    ) == list(range(8))
    assert summary.completed_this_run == 7
    assert summary.completed_total == 7
    assert summary.quarantined == 1
    assert summary.failure is not None
    assert summary.failure["kind"] == "attempt_failed"
    assert summary.failure["type"] == "InvalidModelOutput"
    assert summary.failure["disposition"] == "quarantine"

    # Quarantines are visible by identity, not just by count.
    assert summary.as_dict()["quarantined_sources"] == [
        {"kind": "attempt_failed", "type": "InvalidModelOutput",
         "message_sha256": summary.failure["message_sha256"],
         "provider_http_status": None}
    ]
    assert summary.as_dict()["quarantined_sources_truncated"] is False

    rows = [json.loads(line) for line in plan.actions[0].output.read_text().splitlines()]
    assert len(rows) == 8
    assert sorted(row["row_status"] for row in rows) == ["error"] + ["scored"] * 7
    error = next(row for row in rows if row["row_status"] == "error")
    assert (error["stmt_i"], error["evidence_i"]) == (0, 0)
    assert error["error"]["type"] == "InvalidModelOutput"


def test_a_high_position_halt_is_never_masked_by_a_low_position_quarantine(
    tmp_path: Path,
) -> None:
    """Severity outranks position when selecting the reported failure.

    Selecting by position alone would report the quarantine at position 0 and
    set the summary status — and therefore the supervisor's decision — from a
    one-row event while a real halt at position 5 went unmentioned.
    """
    path = _write_fixture(
        tmp_path, workers=3, execution_count=8, max_attempts=1,
        distinct_claims=True,
    )
    plan = load_run_plan(path)
    clients: list[SourceKeyedClient] = []

    def factory(_token: str, _action: Any) -> SourceKeyedClient:
        client = SourceKeyedClient(
            [], plan.actions[0].ledger, off_grid={0}, raising={5}
        )
        clients.append(client)
        return client

    summary = _run_with_clients(plan, factory)
    # Schedulability and disposition are separate answers, and only the second
    # is deterministic here: whether the two sources behind the halt were
    # dispatched before replenishment stopped is a thread race, so the action
    # may end either fully resolved ("settled") or with sources untouched
    # ("partial").  What must hold in both is that it is not "complete" and that
    # the reported failure is the halt, not the quarantine.
    assert summary.status in {"partial", "settled"}
    assert summary.failure is not None
    assert summary.failure["disposition"] == "halt"
    assert summary.failure["type"] == "ValueError"
    assert summary.failure["message_sha256"] == hashlib.sha256(
        b"nonretryable fixture failure"
    ).hexdigest()
    # The quarantine still happened and is still counted; it just does not get
    # to speak for the run.
    assert summary.quarantined == 1
    assert 5 in [
        position for client in clients for position in client.positions
    ]


def test_a_quarantined_source_is_never_re_attempted_or_re_paid(
    tmp_path: Path,
) -> None:
    """A settled action is not offered again, and costs nothing if it is.

    The quarantine is durable because it is DERIVED from the rows already on
    disk, not held in process memory: `load_resume` recomputes it.  Seven scored
    plus one retired resolves every source, so the action is terminal and
    `prepare_run` refuses it outright rather than re-offering an action with
    nothing to schedule.  That refusal is the point: before `settled` existed the
    action stayed "partial" forever, so `ready_actions` kept naming it and every
    futile invocation demanded a live bearer token to make zero provider calls.

    The bytes on both durable surfaces are asserted unchanged either way, which
    is the no-double-spend invariant itself.
    """
    path = _write_fixture(
        tmp_path, workers=3, execution_count=8, max_attempts=1,
        distinct_claims=True,
    )
    plan = load_run_plan(path)

    def factory(_token: str, _action: Any) -> SourceKeyedClient:
        return SourceKeyedClient([], plan.actions[0].ledger, off_grid={0})

    first = _run_with_clients(plan, factory)
    assert first.quarantined == 1
    assert first.status == "settled"
    raw_prefix = plan.actions[0].output.read_bytes()
    ledger_prefix = plan.actions[0].ledger.read_bytes()

    with pytest.raises(runner.RunnerError, match="already complete"):
        runner.prepare_run(plan)
    assert runner.inspect_plan(plan).status == "settled"
    assert runner.inspect_plan(plan).ready_action_ids == ()
    assert plan.actions[0].output.read_bytes() == raw_prefix
    assert plan.actions[0].ledger.read_bytes() == ledger_prefix


def test_a_hole_stops_the_action_even_with_sources_left_unscored(
    tmp_path: Path,
) -> None:
    """One hole is enough, and leftover pending work does not change that.

    This is the shape that used to look like "the action still has work to do":
    source 0 is retired, sources 1-2 remain.  Scoring them cannot produce a
    bundle — `llm._validate_raw` requires a final scored verdict for EVERY pair
    in the exact universe — so dispatching them would buy nothing at real cost.
    The action is terminal, is not re-offered, and neither durable surface grows.
    """
    path = _write_fixture(
        tmp_path, workers=1, execution_count=3, max_attempts=1,
        distinct_claims=True,
    )
    plan = load_run_plan(path)
    first = _run_with_clients(plan, lambda _t, _a: SourceKeyedClient(
        [], plan.actions[0].ledger, off_grid={0}
    ))
    assert first.quarantined == 1
    assert first.status == "settled"
    assert first.completed_total < first.total  # sources really are left over

    raw_prefix = plan.actions[0].output.read_bytes()
    ledger_prefix = plan.actions[0].ledger.read_bytes()
    with pytest.raises(runner.RunnerError, match="already complete"):
        runner.prepare_run(plan)
    assert plan.actions[0].output.read_bytes() == raw_prefix
    assert plan.actions[0].ledger.read_bytes() == ledger_prefix


def test_settled_source_reports_settled_and_never_reports_complete(
    tmp_path: Path,
) -> None:
    """A quarantined hole must never be reported as a COMPLETED action.

    S2 wrote this test to assert "partial" forever, on the reasoning that
    pending/partial/complete were the only statuses and only "partial" kept the
    holes from shipping.  The intent below is deliberately narrowed, because
    "partial" bought that safety with three defects: `ready_actions` re-offered
    an action with nothing schedulable, every dependent deadlocked permanently
    (in the shipped plan, all three primary arms sit behind the sensitivity
    actions), and each futile invocation demanded a live bearer token for a run
    that builds zero clients.

    What actually had to hold was never "partial" — it was NOT "complete".  A
    settled action is terminal for scheduling and distinct from clean, and it is
    `llm.materialize_model_bundle`'s explicit exclusion list, not a status
    string, that stops the holes from shipping silently.  So this asserts the
    real invariant: the status is "settled", it is not "complete", and `done`
    and `settled` stay disjoint.
    """
    path = _write_fixture(tmp_path, execution_count=2, max_attempts=5)
    plan = load_run_plan(path)
    action, stage, index = _resume_scope(plan)
    first, second = index.executions[0], index.executions[1]
    rows = [
        _resume_scored_row(
            first, action=action, model=stage.model,
            provider_model_id=stage.provider_model_id, ordinal=1,
        )
    ]
    rows += [
        _resume_error_row(
            second, action=action, model=stage.model, ordinal=ordinal,
            error_type="InvalidModelOutput",
        )
        for ordinal in range(1, replay.INVALID_MODEL_OUTPUT_LIMIT + 1)
    ]
    _write_rows(action, rows)

    resume = replay.load_resume(
        action.output, index=index, action=action, model=stage.model,
        provider_model_id=stage.provider_model_id,
    )
    assert resume.status == "settled"
    assert resume.status != "complete"
    assert resume.done == frozenset({(0, 0)})
    assert resume.settled == {(1, 0): "invalid_model_output_limit"}
    assert not (resume.done & set(resume.settled))
    # The two status producers must agree byte for byte on the same rows:
    # `resume_status` discards the rows `load_resume` retains, so a drift here
    # would let readiness and the scheduler describe different actions.
    assert replay.resume_status(
        action.output, index=index, action=action, model=stage.model,
        provider_model_id=stage.provider_model_id,
    ) == replay.ResumeStatus("settled", 1, 6, 1)
    plan_status = runner.inspect_plan(plan)
    assert plan_status.status == "settled"
    assert plan_status.status != "complete"
    # Terminal means terminal: nothing is re-offered.
    assert plan_status.ready_action_ids == ()
    assert plan_status.actions[0].settled == 1


def test_settled_sources_are_neither_completed_nor_pending_at_readiness(
    tmp_path: Path,
) -> None:
    """Readiness must not promise work the run cannot do.

    `pending` is what the operator reads to decide whether handing over a bearer
    token can accomplish anything.  Counting retired sources there would promise
    work no client will ever be built for.

    One worker makes the first run strictly sequential: position 0 is
    quarantined and scheduling continues, positions 1-2 score, position 3 raises
    an unclassified failure and halts the action, and positions 4-7 are never
    dispatched.  That is the mixed state — done, settled and never-touched all
    at once — that the three counts have to partition.  They must still sum to
    `total` even though the never-touched four will not in fact be dispatched:
    the action holds holes, so it is terminal, and `pending` describes the
    corpus rather than promising a plan.
    """
    path = _write_fixture(
        tmp_path, workers=1, execution_count=8, max_attempts=1,
        distinct_claims=True,
    )
    plan = load_run_plan(path)

    def factory(_token: str, _action: Any) -> SourceKeyedClient:
        return SourceKeyedClient(
            [], plan.actions[0].ledger, off_grid={0}, raising={3}
        )

    first = _run_with_clients(plan, factory)
    assert first.status == "settled"
    assert first.failure and first.failure["disposition"] == "halt"
    assert first.quarantined == 1

    action, stage, index = _resume_scope(plan)
    resume = replay.load_resume(
        action.output, index=index, action=action, model=stage.model,
        provider_model_id=stage.provider_model_id,
    )
    assert resume.settled == {
        (0, 0): "attempts_exhausted",
        (3, 0): "nonretryable_failure_on_resume",
    }
    assert len(resume.done) == 2
    assert not (resume.done & set(resume.settled))
    partition = (
        len(resume.done)
        + len(resume.settled)
        + (len(index.executions) - len(resume.done) - len(resume.settled))
    )
    assert partition == len(index.executions) == 8

    # And the action is never offered again, so no bearer token is ever asked
    # for on its behalf.
    with pytest.raises(runner.RunnerError, match="already complete"):
        runner.prepare_run(plan)


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


def test_wal_recovery_of_an_unknown_confidence_does_not_wedge_the_action(
    tmp_path: Path,
) -> None:
    """Recovery uses the parser's closed enums, independent of score presence.

    Durable provider evidence containing ("correct", "certain") has an unknown
    confidence, so the parser refuses the pair. Recovery must commit the same
    retryable InvalidModelOutput row as the live path. A missing calibrated
    probability is otherwise valid for a parsed replay row and must never be
    fabricated from its categorical verdict and confidence.
    """
    path = _write_fixture(tmp_path, max_attempts=5, distinct_claims=True)
    plan = load_run_plan(path)
    prepared = runner.prepare_run(plan)
    source = prepared.index.executions[0]
    base = SourceKeyedClient([], plan.actions[0].ledger, off_grid={0})
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
    assert (result["verdict"], result["confidence"]) == (None, None)
    assert result["score"] is None
    # The refusal is the PARSER's, not a second reading of its output: the same
    # reply body read directly yields no Verdict at all, so no caller can build
    # a result carrying an unscorable pair.
    assert parse_verdict(
        '{"support":"span","objection":null,"verdict":"correct",'
        '"confidence":"certain"}'
    ) is None
    prepared.close()  # process loss after settled response evidence
    del context

    recovered = runner.prepare_run(plan)
    try:
        assert recovered.resume.status == "partial"
        assert recovered.resume.done == frozenset()
        row = recovered.resume.rows[0]
        assert row["row_status"] == "error"
        assert row["error"]["type"] == "InvalidModelOutput"
        assert row["attempt_ordinal"] == 1
        assert replay.row_retry_class(row) == "invalid_model_output"
        assert recovered.resume.settled == {}
    finally:
        recovered.close()

    # The escape is not a one-shot: recovery committed the attempt, so a second
    # prepare_run neither re-enters the recovery path nor appends a second row.
    again = runner.prepare_run(plan)
    try:
        assert len(again.resume.rows) == 1
        assert again.resume.attempts == {(0, 0): 1}
    finally:
        again.close()


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
    assert replay.row_retry_class(row) == expected
