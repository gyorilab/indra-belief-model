"""Semantic data contracts for immutable inputs and ordered comparison actions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

# The canonical content-address codec lives in indra_belief.hashing (single
# source of truth). Re-exported here so the existing public names
# (`from ...comparison.contracts import canonical_json_bytes`, etc.) keep
# resolving for every current importer.
from indra_belief.hashing import (
    canonical_json_bytes,
    canonical_json_line,
    canonical_sha256,
)

# `ContractError` moved to `indra_belief.prepared_execution` and is re-exported
# here. It is the base of that module's `ReplayError`, so owning it here made the
# serving kernel import the comparison harness — the one core -> research edge in
# the tree, now guarded by scripts/check_import_boundary.py. The name is
# unchanged and so is the class: every importer of
# `comparison.contracts.ContractError`, every `except ContractError`, and
# `runner.RunnerError` still bind the same object.
from indra_belief.prepared_execution import ContractError


SHA256 = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MAX_ATTEMPTS = 10
MAX_WORKERS = 8

# Each amendable field has exactly one canonical transition; an amendment
# either applies it in full or does not mention the field.
AMENDABLE_FIELDS: dict[str, tuple[int, int]] = {
    "max_attempts": (5, MAX_ATTEMPTS),
    "workers": (6, MAX_WORKERS),
}


def _fail(message: str) -> NoReturn:
    raise ContractError(message)


def _unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number {value!r} is forbidden")


def strict_json_loads(raw: bytes | str, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as exc:
        raise ContractError(f"{context} is not UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except ContractError as exc:
        raise ContractError(f"{context}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"{context} is malformed JSON: {exc}") from exc


@dataclass(frozen=True)
class FileCapture:
    path: Path
    payload: bytes
    identity: tuple[int, int, int, int, int, int]

    def assert_current(self) -> None:
        try:
            observed = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise ContractError(f"committed file disappeared: {self.path}") from exc
        identity = _stat_identity(observed)
        if not stat.S_ISREG(observed.st_mode) or identity != self.identity:
            _fail(f"committed file changed after validation: {self.path}")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stable_read(path: str | Path, *, context: str) -> FileCapture:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        named_before = os.stat(absolute, follow_symlinks=False)
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ContractError(f"{context}: cannot open {absolute}") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(absolute, follow_symlinks=False)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    identities = {
        _stat_identity(named_before),
        _stat_identity(before),
        _stat_identity(after),
        _stat_identity(named_after),
    }
    if (
        len(identities) != 1
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or len(payload) != before.st_size
    ):
        _fail(f"{context}: file identity changed while reading {absolute}")
    return FileCapture(absolute, payload, _stat_identity(before))


def repository_root(owner: Path) -> Path:
    start = owner if owner.is_dir() else owner.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return start


def resolve_path(
    declared: str,
    *,
    owner: Path,
    root: Path,
    base: str = "owner",
    must_exist: bool = False,
) -> Path:
    if not isinstance(declared, str) or not declared or "\x00" in declared:
        _fail("declared path must be a nonempty string")
    if not isinstance(base, str) or base not in {"owner", "repository"}:
        _fail(f"unsupported path base {base!r}")
    path = Path(declared).expanduser()
    if path.is_absolute():
        result = path
    elif base == "owner":
        result = owner.parent / path
    else:
        result = root / path
    result = Path(os.path.abspath(os.fspath(result)))
    if must_exist and not result.exists():
        _fail(f"declared path does not exist: {result}")
    return result


def _path_reference(
    value: Any,
    *,
    owner: Path,
    root: Path,
    context: str,
) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "base"}:
        _fail(f"{context} must contain exactly path and base")
    return resolve_path(
        value["path"], owner=owner, root=root, base=value["base"]
    )


@dataclass(frozen=True)
class FileDescriptor:
    path: Path
    sha256: str
    bytes: int
    rows: int | None = None
    declared_path: str = ""
    base: str = "owner"

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any],
        *,
        owner: Path,
        root: Path | None = None,
    ) -> "FileDescriptor":
        if not isinstance(value, Mapping):
            _fail("file descriptor must be an object")
        allowed = {"path", "sha256", "bytes", "rows", "base"}
        if set(value) - allowed or not {"path", "sha256", "bytes"} <= set(value):
            _fail("file descriptor fields differ from path/sha256/bytes[/rows/base]")
        digest = value["sha256"]
        size = value["bytes"]
        rows = value.get("rows")
        base = value.get("base", "owner")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            _fail("file descriptor sha256 is not lowercase SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail("file descriptor bytes must be a nonnegative integer")
        if rows is not None and (
            isinstance(rows, bool) or not isinstance(rows, int) or rows < 0
        ):
            _fail("file descriptor rows must be a nonnegative integer")
        declared = value["path"]
        if not isinstance(declared, str):
            _fail("file descriptor path must be a string")
        project = root or repository_root(owner)
        path = resolve_path(
            declared,
            owner=owner,
            root=project,
            base=base,
            must_exist=False,
        )
        return cls(path, digest, size, rows, declared, base)

    def capture(self, *, context: str = "file descriptor") -> FileCapture:
        capture = stable_read(self.path, context=context)
        observed_rows = capture.payload.count(b"\n")
        if len(capture.payload) != self.bytes:
            _fail(f"{context}: byte length differs for {self.path}")
        if hashlib.sha256(capture.payload).hexdigest() != self.sha256:
            _fail(f"{context}: SHA-256 differs for {self.path}")
        if self.rows is not None and observed_rows != self.rows:
            _fail(f"{context}: row count differs for {self.path}")
        return capture


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        _fail(f"{field} is not a safe nonempty identifier")
    return value


def _positive_decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool):
        _fail(f"{field} must be a positive decimal string")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ContractError(f"{field} is not decimal") from exc
    if not result.is_finite() or result <= 0:
        _fail(f"{field} must be positive and finite")
    if str(value) != format(result, "f"):
        _fail(f"{field} must use canonical non-exponent decimal notation")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class Stage:
    id: str
    model: str
    provider_model_id: str
    cap_usd: Decimal


@dataclass(frozen=True)
class Action:
    id: str
    stage_id: str
    run_id: str
    workload: str
    ledger: Path
    output: Path
    cap_usd: Decimal
    deadline_seconds: int
    max_attempts: int
    provider_input_token_maximum: int
    main_max_output_tokens: int
    retry_backoff_seconds: float
    workers: int
    depends_on: tuple[str, ...]
    execution_keys: tuple[Mapping[str, Any], ...] | None = None


# Statuses meaning "this action will never dispatch another source", so it must
# not be offered as ready again. `replay.resolved_status` is the sole producer.
#
# This is deliberately NOT the same set as "a dependent may now start". Only
# `complete` satisfies a dependency: a settled action holds at least one source
# with no verdict, so it can never be bundled, and the whole reason its
# dependents exist is to run after it succeeded. Letting `settled` satisfy
# `depends_on` would release this plan's three primary arms — $39.96, $39.96 and
# $309.54 — on the strength of a sensitivity arm that failed.
TERMINAL_STATUSES = frozenset({"complete", "settled"})


@dataclass(frozen=True)
class AmendmentChange:
    action_id: str
    field: str
    from_value: int
    to_value: int


@dataclass(frozen=True)
class PlanAmendment:
    predecessor_sha256: str
    frozen_at: str
    reason: str
    changes: tuple[AmendmentChange, ...]


@dataclass(frozen=True)
class RunPlan:
    path: Path
    capture: FileCapture
    sha256: str
    root: Path
    replay_manifest: FileDescriptor
    global_cap_usd: Decimal
    stages: tuple[Stage, ...]
    actions: tuple[Action, ...]
    amendment: PlanAmendment | None

    @property
    def stage_by_id(self) -> Mapping[str, Stage]:
        return {stage.id: stage for stage in self.stages}

    @property
    def action_by_id(self) -> Mapping[str, Action]:
        return {action.id: action for action in self.actions}

    def ready_actions(self, statuses: Mapping[str, str]) -> tuple[Action, ...]:
        # Two different questions, deliberately given two different answers.
        # "Is this action still schedulable?" is answered by TERMINAL_STATUSES,
        # so a settled action stops being re-offered. "May its dependents
        # start?" is answered by `complete` alone, because a settled action can
        # never be bundled and its successors exist precisely to follow a
        # successful one. A settled action therefore blocks its dependents, on
        # purpose: that is a stop, not a deadlock, and the operator clears it by
        # fixing the cause rather than by waiting.
        allowed = {"pending", "partial", "settled", "complete"}
        unknown = set(statuses) - set(self.action_by_id)
        if unknown:
            _fail(f"statuses contain unknown actions: {sorted(unknown)}")
        for action in self.actions:
            status = statuses.get(action.id, "pending")
            if status not in allowed:
                _fail(f"invalid status for action {action.id}: {status!r}")
            unmet = [
                dependency
                for dependency in action.depends_on
                if statuses.get(dependency, "pending") != "complete"
            ]
            if status != "pending" and unmet:
                _fail(
                    f"run output for {action.id} advances before dependencies "
                    f"{unmet!r}"
                )
        return tuple(
            action
            for action in self.actions
            if statuses.get(action.id, "pending") not in TERMINAL_STATUSES
            and all(
                statuses.get(dependency, "pending") == "complete"
                for dependency in action.depends_on
            )
        )

    def next_action(self, statuses: Mapping[str, str]) -> Action | None:
        ready = self.ready_actions(statuses)
        return ready[0] if ready else None

    def assert_current(self) -> None:
        self.capture.assert_current()


def _plan_amendment(
    value: Any,
    *,
    actions: Sequence[Action],
) -> PlanAmendment | None:
    if value is None:
        return None
    exact_fields = {"predecessor_sha256", "frozen_at", "reason", "changes"}
    if not isinstance(value, Mapping) or set(value) != exact_fields:
        _fail("amendment fields differ")
    predecessor = value["predecessor_sha256"]
    if not isinstance(predecessor, str) or SHA256.fullmatch(predecessor) is None:
        _fail("amendment predecessor_sha256 is not lowercase SHA-256")
    frozen_at = value["frozen_at"]
    if not isinstance(frozen_at, str) or not frozen_at.strip():
        _fail("amendment frozen_at must be a nonempty timestamp")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        _fail("amendment reason must be nonempty")
    change_values = value["changes"]
    if not isinstance(change_values, list) or not change_values:
        _fail("amendment changes must be a nonempty ordered array")

    action_by_id = {action.id: action for action in actions}
    action_positions = {action.id: index for index, action in enumerate(actions)}
    field_ranks = {name: rank for rank, name in enumerate(AMENDABLE_FIELDS)}
    changes: list[AmendmentChange] = []
    seen: set[tuple[str, str]] = set()
    previous_key = (-1, -1)
    for index, item in enumerate(change_values):
        if not isinstance(item, Mapping) or set(item) != {
            "action_id", "field", "from", "to"
        }:
            _fail(f"amendment change {index} fields differ")
        action_id = _identifier(
            item["action_id"], field=f"amendment change {index} action_id"
        )
        action = action_by_id.get(action_id)
        if action is None:
            _fail(f"amendment change {index} references an unknown action")
        if not action.id.endswith("_primary") or not action.run_id.endswith(
            "_primary"
        ):
            _fail(f"amendment change {index} action is not primary")
        if action.workload != "unique_exact_pairs_primary":
            _fail(
                f"amendment change {index} action workload must be "
                "unique_exact_pairs_primary"
            )
        field = item["field"]
        if field not in AMENDABLE_FIELDS:
            _fail(
                f"amendment change {index} field must be one of "
                f"{sorted(AMENDABLE_FIELDS)}"
            )
        if (action_id, field) in seen:
            _fail("amendment changes repeat an action_id and field")
        key = (action_positions[action_id], field_ranks[field])
        if key <= previous_key:
            _fail("amendment changes must follow run-plan action order")
        seen.add((action_id, field))
        previous_key = key
        from_value = _positive_int(
            item["from"], field=f"amendment change {index} from"
        )
        to_value = _positive_int(
            item["to"], field=f"amendment change {index} to"
        )
        canonical = AMENDABLE_FIELDS[field]
        if (from_value, to_value) != canonical:
            _fail(
                f"amendment change {index} must be exactly "
                f"{canonical[0]} -> {canonical[1]}"
            )
        if getattr(action, field) != to_value:
            _fail(
                f"amendment change {index} to does not match the current action"
            )
        changes.append(
            AmendmentChange(action_id, field, from_value, to_value)
        )
    return PlanAmendment(predecessor, frozen_at, reason, tuple(changes))


def load_run_plan(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    root: str | Path | None = None,
) -> RunPlan:
    plan_path = Path(os.path.abspath(os.fspath(path)))
    capture = stable_read(plan_path, context="run plan")
    digest = hashlib.sha256(capture.payload).hexdigest()
    if expected_sha256 is not None and expected_sha256 != digest:
        _fail("run plan SHA-256 differs from the requested commitment")
    value = strict_json_loads(capture.payload, context="run plan")
    if not isinstance(value, Mapping):
        _fail("run plan must be an object")
    required = {
        "kind", "replay", "global_cap_usd", "stages", "actions", "amendment"
    }
    if set(value) != required or value.get("kind") != "indra_belief_comparison_run_plan":
        _fail("run plan fields or kind differ from the semantic contract")
    project = (
        Path(os.path.abspath(os.fspath(root)))
        if root is not None
        else repository_root(plan_path)
    )
    replay_value = value["replay"]
    if not isinstance(replay_value, Mapping) or set(replay_value) != {"manifest"}:
        _fail("replay must contain exactly one manifest descriptor")
    manifest_value = replay_value["manifest"]
    if not isinstance(manifest_value, Mapping) or set(manifest_value) != {
        "path", "sha256", "bytes", "base"
    }:
        _fail("replay manifest must be an exact path/sha256/bytes/base descriptor")
    replay_manifest = FileDescriptor.from_value(
        manifest_value, owner=plan_path, root=project
    )
    replay_manifest.capture(context="replay manifest descriptor")
    global_cap = _positive_decimal(value["global_cap_usd"], field="global_cap_usd")

    stage_values = value["stages"]
    if not isinstance(stage_values, list) or not stage_values:
        _fail("run plan stages must be a nonempty array")
    stages: list[Stage] = []
    for index, item in enumerate(stage_values):
        if not isinstance(item, Mapping) or set(item) != {
            "id", "model", "provider_model_id", "cap_usd"
        }:
            _fail(f"stage {index} fields differ")
        stage = Stage(
            _identifier(item["id"], field=f"stage {index} id"),
            _identifier(item["model"], field=f"stage {index} model"),
            _identifier(
                item["provider_model_id"], field=f"stage {index} provider_model_id"
            ),
            _positive_decimal(item["cap_usd"], field=f"stage {index} cap_usd"),
        )
        if stage.cap_usd > global_cap:
            _fail(f"stage {stage.id} cap exceeds the global cap")
        stages.append(stage)
    if len({stage.id for stage in stages}) != len(stages):
        _fail("run plan repeats a stage id")
    if len({stage.model for stage in stages}) != len(stages):
        _fail("each spend-guard model must belong to exactly one stage")
    if sum((stage.cap_usd for stage in stages), start=Decimal("0")) > global_cap:
        _fail("stage caps together exceed the global cap")
    stage_by_id = {stage.id: stage for stage in stages}

    action_values = value["actions"]
    if not isinstance(action_values, list) or not action_values:
        _fail("run plan actions must be a nonempty ordered array")
    actions: list[Action] = []
    exact_fields = {
        "id", "stage", "run_id", "workload", "ledger", "output", "cap_usd",
        "deadline_seconds", "max_attempts", "provider_input_token_maximum",
        "main_max_output_tokens", "retry_backoff_seconds", "workers",
        "depends_on", "execution_keys",
    }
    for index, item in enumerate(action_values):
        if not isinstance(item, Mapping) or set(item) != exact_fields:
            _fail(f"action {index} fields differ")
        stage_id = _identifier(item["stage"], field=f"action {index} stage")
        if stage_id not in stage_by_id:
            _fail(f"action {index} references an unknown stage")
        output_value = item["output"]
        ledger_value = item["ledger"]
        cap = _positive_decimal(item["cap_usd"], field=f"action {index} cap_usd")
        if cap > stage_by_id[stage_id].cap_usd:
            _fail(f"action {index} cap exceeds its stage cap")
        max_attempts = _positive_int(item["max_attempts"], field="max_attempts")
        if max_attempts > MAX_ATTEMPTS:
            _fail("max_attempts cannot exceed ten")
        backoff = item["retry_backoff_seconds"]
        if isinstance(backoff, bool) or not isinstance(backoff, (int, float)):
            _fail("retry_backoff_seconds must be a nonnegative finite number")
        backoff = float(backoff)
        if not math.isfinite(backoff) or backoff < 0 or backoff > 3600:
            _fail("retry_backoff_seconds is outside [0,3600]")
        workers = _positive_int(item["workers"], field="workers")
        if workers > MAX_WORKERS:
            _fail("workers cannot exceed eight")
        keys_value = item["execution_keys"]
        if keys_value is not None and (
            not isinstance(keys_value, list)
            or not keys_value
            or any(not isinstance(key, Mapping) for key in keys_value)
        ):
            _fail("execution_keys must be null or a nonempty object array")
        dependencies = item["depends_on"]
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(value, str) for value in dependencies)
            or len(dependencies) != len(set(dependencies))
        ):
            _fail("depends_on must be an array of unique action identifiers")
        known_action_ids = {action.id for action in actions}
        normalized_dependencies = tuple(
            _identifier(value, field=f"action {index} dependency")
            for value in dependencies
        )
        if any(value not in known_action_ids for value in normalized_dependencies):
            _fail(f"action {index} dependencies must precede it topologically")
        actions.append(
            Action(
                _identifier(item["id"], field=f"action {index} id"),
                stage_id,
                _identifier(item["run_id"], field=f"action {index} run_id"),
                _identifier(item["workload"], field=f"action {index} workload"),
                _path_reference(
                    ledger_value,
                    owner=plan_path,
                    root=project,
                    context=f"action {index} ledger",
                ),
                _path_reference(
                    output_value,
                    owner=plan_path,
                    root=project,
                    context=f"action {index} output",
                ),
                cap,
                _positive_int(item["deadline_seconds"], field="deadline_seconds"),
                max_attempts,
                _positive_int(
                    item["provider_input_token_maximum"],
                    field="provider_input_token_maximum",
                ),
                _positive_int(
                    item["main_max_output_tokens"], field="main_max_output_tokens"
                ),
                backoff,
                workers,
                normalized_dependencies,
                tuple(dict(key) for key in keys_value) if keys_value else None,
            )
        )
    if len({action.id for action in actions}) != len(actions):
        _fail("run plan repeats an action id")
    if len({action.run_id for action in actions}) != len(actions):
        _fail("run plan repeats a run_id")
    if len({action.output for action in actions}) != len(actions):
        _fail("run plan repeats an output path")
    if any(action.ledger == action.output for action in actions):
        _fail("an action ledger cannot also be its raw output")
    lanes = [(action.stage_id, action.workload) for action in actions]
    if len(set(lanes)) != len(lanes):
        _fail("run plan repeats a model/workload execution lane")
    amendment = _plan_amendment(value["amendment"], actions=actions)
    # Separate ledger files are independent spend lanes.  Bound their combined
    # worst-case exposure within each semantic stage; actions sharing a ledger
    # are additionally protected by the ledger's runtime stage cap.
    for stage in stages:
        lane_caps: dict[Path, Decimal] = {}
        for action in actions:
            if action.stage_id == stage.id:
                lane_caps[action.ledger] = lane_caps.get(action.ledger, Decimal("0")) + action.cap_usd
        exposure = sum(
            (min(stage.cap_usd, value) for value in lane_caps.values()),
            start=Decimal("0"),
        )
        if exposure > stage.cap_usd:
            _fail(f"independent spend lanes for stage {stage.id} exceed its cap")
    return RunPlan(
        path=plan_path,
        capture=capture,
        sha256=digest,
        root=project,
        replay_manifest=replay_manifest,
        global_cap_usd=global_cap,
        stages=tuple(stages),
        actions=tuple(actions),
        amendment=amendment,
    )
