"""Crash-safe spend accounting for the canonical comparison runner.

One guard owns one append-only ledger lane.  Every provider request is reserved
at a conservative maximum before the request can start, followed by durable
provider evidence and either measured usage or the reserved maximum.  The
ledger is hash chained and is replayed on every open; an interrupted reservation
is conservatively settled before another request may be scheduled.

This module deliberately contains no experiment authorization policy.  Run
ordering, action limits, credential handoff, and scientific input validation
belong to :mod:`indra_belief.comparison.runner`.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import stat
import threading
import urllib.error
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

from indra_belief.corpus.cost import price_for


DEFAULT_MAX_ATTEMPTS = 2
_MILLION = Decimal("1000000")
_ZERO = Decimal("0")
_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_LEDGER_EVENTS = frozenset(
    {
        "ledger_initialized",
        "stage_cap_set",
        "stage_cap_amended",
        "attempt_started",
        "call_reserved",
        "call_evidence_observed",
        "call_settled",
        "attempt_outcome_committed",
        "attempt_finished",
    }
)
_RUN_EVENTS = _LEDGER_EVENTS - {
    "ledger_initialized",
    "stage_cap_set",
    "stage_cap_amended",
}
_LEDGER_READ_BLOCK_BYTES = 1024 * 1024
# A single canonical ledger line is bounded in practice: the largest line seen
# across the shipped production ledgers is under 1 MB (measured maximum 684,073
# bytes; see _MAX_RECOVERABLE_TORN_TAIL_BYTES), so 64 MiB is ~98x headroom.  The
# cap exists so a corrupt newline-less file cannot reintroduce the unbounded
# buffer growth that the streaming reader exists to remove.
_MAX_LEDGER_LINE_BYTES = 64 * 1024 * 1024
# PLAUSIBILITY bound on a crash-torn trailing partial event, deliberately
# distinct from _MAX_LEDGER_LINE_BYTES: that one is a read-buffer/parse bound
# (how much this process will hold before declaring a line unreadable), this one
# is a repair bound (how much this process is willing to DESTROY at open).  A
# torn append is at most ONE event, so one event is the ceiling.
#
# Measured across every real ledger in the repo (largest canonical event, line
# plus its terminating newline):
#   data/results/paper_lane_p_wave1_gemma4_glm5_20260717_spend_ledger.ndjson
#       228,147,098 B / 166,219 events / max event     2,213 B
#   data/comparison/runs/spend.ndjson
#     5,095,405,915 B / 254,511 events / max event   241,838 B
#   data/comparison/runs/glm_5_primary/spend.ndjson
#     4,248,035,409 B / 252,551 events / max event   289,931 B
#   data/comparison/runs/gemma_26b_primary/spend.ndjson
#     5,502,177,925 B / 253,166 events / max event   684,073 B  <-- largest
# i.e. ~15 GB over ~926k real events, largest single event 684,073 B (668 KiB).
# 4 MiB is ~6.1x that observed maximum and 16x below the 64 MiB read bound.
# Anything larger is not a plausible single torn append, so it fails closed
# (SpendLedgerCorrupt) exactly as it did before torn-tail recovery existed
# rather than being silently truncated away.
_MAX_RECOVERABLE_TORN_TAIL_BYTES = 4 * 1024 * 1024


class SpendGuardError(RuntimeError):
    """Base class for fail-closed accounting errors."""


class SpendGuardStop(SpendGuardError):
    """A clean scheduling stop; no new provider request was started."""


class SpendCapReached(SpendGuardStop):
    """The next reservation would exceed a global or stage cap."""


class AttemptLimitReached(SpendGuardStop):
    """An execution identity has consumed its configured attempts."""


class SpendLedgerCorrupt(SpendGuardError):
    """The ledger is malformed, incomplete, or fails its hash chain."""


class SpendLedgerInUse(SpendGuardError):
    """Another process owns the ledger lane."""


class SpendReservationBreach(SpendGuardError):
    """Measured provider usage exceeded its pre-call reservation."""


class SpendLedgerTornTailWarning(RuntimeWarning):
    """A crash-torn trailing partial line was dropped when the lane reopened."""


@dataclass(frozen=True)
class SpendLedgerSnapshot:
    """Integrity-checked, read-only view of one canonical spend ledger.

    The SHA-256 hash chain is tamper-EVIDENT against corruption/truncation and
    canonical-form drift; it is NOT a keyed authenticity proof — a local writer
    who can edit the file can re-chain a forged edit. That is out of scope here
    (the same actor can bypass the guard entirely); cap invariants are enforced
    by the semantic replay validator, not by the digest.
    """

    ledger_id: str
    events: tuple[Mapping[str, Any], ...]
    last_event_sha256: str
    _state: "_LedgerReplay" = field(repr=False, compare=False)

    @property
    def sequence(self) -> int:
        return len(self.events)

def _ledger_json_bytes(value: Any) -> bytes:
    # ensure_ascii=True is a frozen on-disk ledger-digest contract: the bytes it
    # emits back already-committed spend-ledger hash chains, so they must never
    # change. This is deliberately DISTINCT from the ensure_ascii=False content-
    # address codec in indra_belief.hashing and must not be merged into it.
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SpendGuardError("value is not strict canonical JSON") from exc


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(_ledger_json_bytes(value))
    except SpendGuardError as exc:
        raise SpendGuardError(f"{field} is not strict JSON") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SpendLedgerCorrupt(f"ledger event repeats key {key!r}")
        value[key] = item
    return value


def _verify_ledger_line(
    line: str,
    number: int,
    previous: str | None,
    ledger_id: str | None,
) -> tuple[dict[str, Any], str, str]:
    """Verify one canonical ledger line against the chain state before it.

    This is the single canonical per-line rule set: both the bytes verifier
    (:func:`parse_spend_ledger`) and the guard's streaming loader drive it, so
    the two read paths cannot drift.  Returns the parsed row, its
    ``event_sha256`` (the next line's ``previous_event_sha256``), and the
    ledger identity established or carried forward.
    """

    try:
        row = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, SpendLedgerCorrupt) as exc:
        raise SpendLedgerCorrupt(f"invalid ledger event at line {number}") from exc
    if not isinstance(row, dict):
        raise SpendLedgerCorrupt(f"ledger line {number} is not an object")
    try:
        canonical = _ledger_json_bytes(row).decode("utf-8")
    except SpendGuardError as exc:
        raise SpendLedgerCorrupt(f"ledger line {number} is not canonical JSON") from exc
    if canonical != line:
        raise SpendLedgerCorrupt(f"ledger line {number} is not canonical JSON")
    digest = row.get("event_sha256")
    body = dict(row)
    body.pop("event_sha256", None)
    expected = hashlib.sha256(_ledger_json_bytes(body)).hexdigest()
    if (
        row.get("sequence") != number
        or row.get("previous_event_sha256") != previous
        or not isinstance(digest, str)
        or digest != expected
    ):
        raise SpendLedgerCorrupt(f"ledger hash chain fails at line {number}")
    event = row.get("event")
    if event not in _LEDGER_EVENTS:
        raise SpendLedgerCorrupt(f"unsupported spend event {event!r}")
    if number == 1:
        candidate = row.get("ledger_id")
        if event != "ledger_initialized" or not isinstance(candidate, str) or _HEX32.fullmatch(candidate) is None:
            raise SpendLedgerCorrupt("spend ledger does not begin with valid initialization")
        _ledger_decimal(row.get("global_cap_usd"), field="ledger global cap")
        ledger_id = candidate
    elif row.get("ledger_id") != ledger_id:
        raise SpendLedgerCorrupt(f"ledger identity differs at line {number}")
    if event in _RUN_EVENTS and (
        not isinstance(row.get("run_id"), str) or not row["run_id"]
    ):
        raise SpendLedgerCorrupt(f"ledger run identity is absent at line {number}")
    assert ledger_id is not None
    return row, digest, ledger_id


def _iter_ledger_lines(fd: int, *, max_line_bytes: int) -> Iterator[str]:
    """Yield each COMPLETE ledger line from ``fd`` without materializing the file.

    Reads fixed blocks and splits on b"\\n", so peak allocation is one block plus
    one line rather than the whole ledger.  A trailing PARTIAL line is never
    yielded; the generator returns ``(tail, complete_offset)`` where ``tail`` is
    the trailing partial line's raw bytes (``b""`` when the file ends cleanly)
    and ``complete_offset`` is the byte offset just past the last complete
    b"\\n".  The tail bytes are returned, not just their length, so the caller
    can preserve them before repairing the lane; they stay bounded by
    ``max_line_bytes``.
    """

    os.lseek(fd, 0, os.SEEK_SET)
    pending = bytearray()
    complete_offset = 0
    while True:
        block = os.read(fd, _LEDGER_READ_BLOCK_BYTES)
        if not block:
            break
        pending.extend(block)
        del block
        start = 0
        while True:
            cut = pending.find(b"\n", start)
            if cut < 0:
                break
            raw = bytes(pending[start:cut])
            start = cut + 1
            complete_offset += len(raw) + 1
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SpendLedgerCorrupt("spend ledger is not UTF-8") from exc
            del raw
            yield line
            del line
        if start:
            del pending[:start]
        if len(pending) > max_line_bytes:
            raise SpendLedgerCorrupt("spend ledger line exceeds the readable maximum")
    return bytes(pending), complete_offset


def parse_spend_ledger(payload: bytes) -> SpendLedgerSnapshot:
    """Integrity-check canonical ledger bytes (SHA-256 chain: corruption/
    truncation/canonical-form evident, NOT keyed authenticity) without opening
    or mutating a lane.

    This is the STRICT verifier for captured audit artifacts: a payload that
    does not end in b"\\n" is rejected outright.  Torn-tail tolerance lives only
    in the guard's own lane loader, which owns the descriptor and can repair it.
    """

    if not isinstance(payload, bytes) or not payload or not payload.endswith(b"\n"):
        raise SpendLedgerCorrupt("spend ledger is empty or ends with a partial event")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SpendLedgerCorrupt("spend ledger is not UTF-8") from exc
    previous: str | None = None
    ledger_id: str | None = None
    events: list[Mapping[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        row, previous, ledger_id = _verify_ledger_line(line, number, previous, ledger_id)
        events.append(MappingProxyType(row))
    assert ledger_id is not None and previous is not None
    return _build_spend_ledger_snapshot(ledger_id, tuple(events), previous)


def _decimal(value: Decimal | float | int | str, *, field: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be a finite decimal")
    return result


def _ledger_decimal(value: Any, *, field: str) -> Decimal:
    try:
        result = _decimal(value, field=field)
    except ValueError as exc:
        raise SpendLedgerCorrupt(str(exc)) from exc
    if result < 0:
        raise SpendLedgerCorrupt(f"{field} must be nonnegative")
    return result


def _money(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_ledger_json_bytes(value)).hexdigest()


def provider_wire_body_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the strict UTF-8 JSON body handed to a raw transport."""

    try:
        raw = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SpendGuardError("provider body is not strict UTF-8 JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def provider_request_sha256(
    *,
    provider_model_id: str,
    kind: str,
    call_ordinal: int,
    max_output_tokens: int,
    system: str,
    messages: list[dict[str, Any]],
    provider_wire_request_sha256: str | None = None,
) -> str:
    """Commit the request material retained in the raw call log."""

    if (
        not provider_model_id
        or not kind
        or call_ordinal < 1
        or max_output_tokens < 1
        or not isinstance(system, str)
        or not isinstance(messages, list)
        or any(not isinstance(row, dict) for row in messages)
        or (
            provider_wire_request_sha256 is not None
            and _HEX64.fullmatch(provider_wire_request_sha256) is None
        )
    ):
        raise SpendGuardError("provider request commitment is invalid")
    value: dict[str, Any] = {
        "provider_model_id": provider_model_id,
        "kind": kind,
        "call_ordinal": call_ordinal,
        "max_output_tokens": max_output_tokens,
        "system": system,
        "messages": messages,
    }
    if provider_wire_request_sha256 is not None:
        value["provider_wire_request_sha256"] = provider_wire_request_sha256
    return _sha256(value)


def provider_response_sha256(
    *,
    provider_model_id: str,
    kind: str,
    call_ordinal: int,
    content: str,
    reasoning: str,
    raw_text: str,
    prompt_tokens: int,
    output_tokens: int,
    finish_reason: str,
) -> str:
    """Commit response fields reproducible from the retained call log."""

    if (
        not provider_model_id
        or not kind
        or call_ordinal < 1
        or any(not isinstance(item, str) for item in (content, reasoning, raw_text))
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < -1
        or not isinstance(output_tokens, int)
        or output_tokens < 0
        or not isinstance(finish_reason, str)
    ):
        raise SpendGuardError("provider response commitment is invalid")
    return _sha256(
        {
            "provider_model_id": provider_model_id,
            "kind": kind,
            "call_ordinal": call_ordinal,
            "content": content,
            "reasoning": reasoning,
            "raw_text": raw_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
        }
    )


def _status_from_provider_exception(error: BaseException) -> int | None:
    candidates = [
        getattr(error, "status", None),
        getattr(error, "status_code", None),
        getattr(error, "http_status", None),
        getattr(error, "code", None) if isinstance(error, urllib.error.HTTPError) else None,
    ]
    response = getattr(error, "response", None)
    if isinstance(response, Mapping):
        candidates.extend((response.get("status"), response.get("status_code")))
    elif response is not None:
        candidates.extend(
            (getattr(response, "status", None), getattr(response, "status_code", None))
        )
    trace = getattr(error, "transport_trace", None)
    if isinstance(trace, Mapping):
        candidates.append(trace.get("response_http_status"))
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def classify_provider_failure(
    error: BaseException | None,
) -> tuple[str | None, int | None]:
    """Return the narrow retry class used by the comparison runner.

    408 (request timeout) and 429 (rate limited) join the 5xx transport class.
    Both say "ask again later", not "this request is wrong", and the runner
    turns the class straight into its bounded exponential backoff — no new pause
    machinery, still capped by the action's `max_attempts` and deadline.

    Calling them "other" was survivable only while the FIRST failure of any kind
    halted the whole action.  Once a source can be quarantined instead, a
    non-retryable class makes it *permanently* settled, so one 429 would have
    retired a source that a second request would have scored.

    This changes the `provider_failure_class` written at `_settle` for NEW 429
    and 408 rows only; rows already on disk keep the class they were written
    with, and `replay.row_retry_class` reads that stored class as it always has.
    """

    if error is None:
        return None, None
    if isinstance(error, (TimeoutError, ConnectionError)):
        # A response may have begun with HTTP 200 before its body transport
        # failed.  The retry fact is the typed connection failure, not that
        # incomplete response's nominal status.
        return "transport_or_server", None
    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        if isinstance(reason, (TimeoutError, ConnectionError, OSError)):
            return "transport_or_server", None
        return "other", None
    status = _status_from_provider_exception(error)
    if status is None:
        match = re.search(r"\bBedrock (?:Responses|Chat) HTTP\s+(\d{3})\b", str(error))
        status = int(match.group(1)) if match else None
    if status is not None:
        return (
            "transport_or_server"
            if 500 <= status <= 599 or status in {408, 429}
            else "other"
        ), status
    return "other", None


def _content_bytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(_ledger_json_bytes(value))


# Providers may report a few more output tokens than the requested cap
# (terminator/encapsulation tokens on reasoning responses); the accounting
# reservation absorbs that overshoot so a capped response settles instead of
# breaching. The request cap itself is never padded.
PROVIDER_OUTPUT_TOKEN_OVERSHOOT = 128


def conservative_prompt_token_bound(system: str, messages: list[dict]) -> int:
    """Tokenizer-independent upper bound: UTF-8 bytes plus envelope slack."""

    size = _content_bytes(system or "")
    for message in messages or []:
        size += _content_bytes(message.get("role", ""))
        size += _content_bytes(message.get("content", ""))
    return size + 1024 + 64 * (len(messages or []) + 1)


def _open_regular(path: Path, flags: int, mode: int = 0o600) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
    except OSError as exc:
        raise SpendGuardError(f"cannot open spend path {path}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise SpendGuardError(f"spend path is not a regular file: {path}")
    return descriptor


def _quarantine_torn_tail(ledger: Path, tail: bytes, complete_offset: int) -> Path:
    """Preserve a crash-torn trailing fragment beside its lane before repair.

    :meth:`SpendGuard._load` truncates a torn lane back to its last complete
    event, which DESTROYS those bytes; they are the only forensic record of what
    the crashed writer was mid-way through, so they are copied out first.  The
    sibling name mirrors the lock-file convention in
    :func:`acquire_spend_lane_lock`, and the open is O_EXCL: an existing
    quarantine is never opened for write, truncated, or overwritten, and a name
    collision falls back to a random suffix rather than clobbering.  Failures
    propagate so the caller's truncation never runs on unpreserved bytes.
    """

    base = ledger.with_suffix(ledger.suffix + f".torn-{complete_offset}-{len(tail)}")
    target = base if not base.exists() else Path(f"{base}.{uuid.uuid4().hex[:8]}")
    descriptor = _open_regular(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        written = 0
        while written < len(tail):
            written += os.write(descriptor, tail[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


@dataclass
class SpendLaneLock:
    path: Path
    _descriptor: int
    _identity: tuple[int, int]
    _closed: bool = False

    def assert_current(self) -> None:
        if self._closed:
            raise SpendGuardError("spend lane lock is closed")
        current = os.stat(self.path, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != self._identity:
            raise SpendGuardError("spend lane lock pathname changed")

    def close(self) -> None:
        if self._closed:
            return
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._descriptor)
            self._closed = True

    def __enter__(self) -> "SpendLaneLock":
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def acquire_spend_lane_lock(ledger_path: str | Path) -> SpendLaneLock:
    ledger = Path(ledger_path)
    lock_path = ledger.with_suffix(ledger.suffix + ".lock")
    descriptor = _open_regular(lock_path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(descriptor)
        raise SpendLedgerInUse(f"spend ledger is already in use: {ledger}") from exc
    info = os.fstat(descriptor)
    return SpendLaneLock(lock_path, descriptor, (info.st_dev, info.st_ino))


@dataclass
class _AttemptContext:
    execution_id: str
    identity: dict[str, Any]
    attempt_id: str
    receipt: "AttemptReceipt"
    attempt_ordinal: int | None = None
    started: bool = False
    call_ordinal: int = 0


@dataclass
class AttemptReceipt:
    execution_id: str
    attempt_id: str
    identity: dict[str, Any]
    attempt_ordinal: int | None = None
    started: bool = False
    status: str | None = None
    error_type: str | None = None
    error_message_sha256: str | None = None
    model_calls_reserved: int = 0
    outcome_event_sha256: str | None = None
    raw_row_sha256: str | None = None


@dataclass(frozen=True)
class CallReservation:
    call_id: str
    attempt_id: str
    execution_id: str
    attempt_ordinal: int
    call_ordinal: int
    provider_model_id: str
    kind: str
    reserved_input_tokens: int
    reserved_output_tokens: int
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    reserved_max_cost_usd: Decimal
    provider_request_sha256: str
    provider_wire_request_sha256: str | None
    provider_request_body: dict[str, Any] | None


def _initialize_ledger_state(target: Any) -> None:
    target.ledger_id = None
    target._global_cap = None
    target._events = []
    target._stage_caps = {}
    target._starts = {}
    target._attempt_ids_by_execution = {}
    target._reservations = {}
    target._calls_by_attempt = {}
    target._calls_with_nonstr_attempt = []
    target._evidence = {}
    target._settlements = {}
    target._outcomes = {}
    target._finishes = {}
    target._global_commitment = _ZERO
    target._stage_commitments = {}
    target._run_commitments = {}


class SpendGuard:
    """Exclusive, resumable spend ledger for one comparison action process."""

    def __init__(
        self,
        ledger_path: str | Path,
        *,
        approved_cap_usd: Decimal | float | int | str,
        model: str,
        stage: str,
        workload: str,
        run_id: str,
        provider_input_token_maximum: int,
        stage_cap_usd: Decimal | float | int | str | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        paid_surface_assertion: Callable[[], None] | None = None,
    ) -> None:
        self.path = Path(ledger_path)
        self.approved_cap_usd = _decimal(approved_cap_usd, field="global cap")
        self.stage_cap_usd = (
            _decimal(stage_cap_usd, field="stage cap")
            if stage_cap_usd is not None
            else self.approved_cap_usd
        )
        if self.approved_cap_usd <= 0:
            raise ValueError("global cap must be positive")
        if self.stage_cap_usd <= 0 or self.stage_cap_usd > self.approved_cap_usd:
            raise ValueError("stage cap must be positive and no greater than global cap")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if (
            isinstance(provider_input_token_maximum, bool)
            or not isinstance(provider_input_token_maximum, int)
            or provider_input_token_maximum < 1
        ):
            raise ValueError("provider input-token maximum must be positive")
        if paid_surface_assertion is not None and not callable(paid_surface_assertion):
            raise ValueError("paid surface assertion must be callable")

        self.model = str(model)
        self.stage = str(stage)
        self.workload = str(workload)
        self.run_id = str(run_id)
        if not all((self.model, self.stage, self.workload, self.run_id)):
            raise ValueError("model, stage, workload, and run_id must be nonempty")
        self.provider_input_token_maximum = provider_input_token_maximum
        self.max_attempts = max_attempts
        self._surface_assertion = paid_surface_assertion
        self._mutex = threading.RLock()
        self._tls = threading.local()
        self._closed = False
        self._lock = acquire_spend_lane_lock(self.path)
        self._ledger_descriptor: int | None = None
        self._ledger_identity: tuple[int, int] | None = None

        self._sequence = 0
        self._last_digest: str | None = None
        # Bytes dropped from a crash-torn trailing partial event at open, so the
        # recovery is inspectable and never silent.  The dropped bytes themselves
        # are preserved at ``recovered_torn_tail_path`` before the lane is
        # repaired, so "dropped" never means "unrecoverable".
        self.recovered_torn_tail_bytes = 0
        self.recovered_torn_tail_path: Path | None = None
        _initialize_ledger_state(self)

        try:
            self._ledger_descriptor = _open_regular(
                self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND
            )
            info = os.fstat(self._ledger_descriptor)
            self._ledger_identity = (info.st_dev, info.st_ino)
            self._load()
            if not self._events:
                self.ledger_id = uuid.uuid4().hex
                self._append(
                    {
                        "event": "ledger_initialized",
                        "ledger_id": self.ledger_id,
                        "global_cap_usd": _money(self.approved_cap_usd),
                    }
                )
            else:
                initialized = self._events[0]
                declared = _ledger_decimal(
                    initialized.get("global_cap_usd"), field="ledger global cap"
                )
                if declared != self.approved_cap_usd:
                    raise SpendGuardError("global cap differs from the existing ledger")
            stage_key = (self.stage, self.model)
            previous = self._stage_caps.get(stage_key)
            if previous is None:
                self._append(
                    {
                        "event": "stage_cap_set",
                        "stage": self.stage,
                        "model": self.model,
                        "stage_cap_usd": _money(self.stage_cap_usd),
                    }
                )
            elif previous != self.stage_cap_usd:
                if self.stage_cap_usd < previous:
                    raise SpendGuardError(
                        "stage cap cannot be lowered below the existing authorization"
                    )
                self._append(
                    {
                        "event": "stage_cap_amended",
                        "stage": self.stage,
                        "model": self.model,
                        "previous_stage_cap_usd": _money(previous),
                        "stage_cap_usd": _money(self.stage_cap_usd),
                        "authorization_basis": "current_frozen_run_plan",
                    }
                )
            self._reconcile_interrupted_calls()
        except BaseException:
            self.close()
            raise

    def _load(self) -> None:
        """Replay the lane by streaming it, never materializing the whole file.

        Every line is verified by the same :func:`_verify_ledger_line` rule set
        the bytes verifier uses, so the chain digest and the replayed state are
        identical to a whole-file parse.  Replay lands in a separate
        :class:`_LedgerReplay`, so a failure anywhere leaves the guard in its
        pristine :func:`_initialize_ledger_state` condition: load stays
        all-or-nothing and partial replay state is never observable.
        """

        assert self._ledger_descriptor is not None
        replay = _LedgerReplay()
        previous: str | None = None
        ledger_id: str | None = None
        number = 0
        stream = _iter_ledger_lines(
            self._ledger_descriptor, max_line_bytes=_MAX_LEDGER_LINE_BYTES
        )
        while True:
            try:
                line = next(stream)
            except StopIteration as stop:
                tail, complete_offset = stop.value
                break
            number += 1
            row, previous, ledger_id = _verify_ledger_line(
                line, number, previous, ledger_id
            )
            del line
            replay.apply(row)
            del row
        tail_len = len(tail)
        if number == 0:
            if tail:
                # Only a partial line: there is no valid initialization to keep.
                raise SpendLedgerCorrupt(
                    "spend ledger is empty or ends with a partial event"
                )
            return
        # Both checks previously lived in _build_spend_ledger_snapshot.
        replay._validate_commitment_cache()
        if replay.ledger_id != ledger_id:
            raise SpendLedgerCorrupt("ledger replay identity differs")
        if tail:
            # A crash mid-append left a torn final line.  Those bytes were never
            # a complete event: no committed event's previous_event_sha256 refers
            # to them, so dropping them changes no committed byte and no
            # last_event_sha256.  Truncation is mandatory, not cosmetic — the
            # descriptor is O_APPEND, so without it the next _append would write
            # onto the partial line and create mid-file corruption.
            #
            # Fail-closed accounting is preserved because the dropped tail can
            # only be one of: a torn call_reserved (the reservation write died
            # BEFORE the provider call, so no spend exists to lose), or a torn
            # call_evidence_observed / call_settled / attempt_outcome_committed /
            # attempt_finished, each of which is preceded by an already-complete
            # call_reserved that stays committed at reserved_max_cost_usd — a
            # conservative OVER-count that _reconcile_interrupted_calls then
            # settles conservatively.  No path under-counts committed spend.
            #
            # The step ORDER below is mandatory, because the ftruncate is
            # irreversible and everything before it is a chance to not do it:
            #   BOUND first — a tail too large to be one torn append is not a
            #     repair case at all; it fails closed with the ledger untouched,
            #     exactly as it did before recovery existed.
            #   WARN before mutating — under `-W error` the warning IS the
            #     raise, so emitting it first means such a run aborts with ZERO
            #     side effects instead of aborting post-truncation.
            #   PRESERVE before destroying — the discarded bytes are fsync'd to
            #     a sibling BEFORE the ftruncate, and a quarantine failure
            #     propagates, leaving the lane merely torn rather than destroyed.
            if tail_len > _MAX_RECOVERABLE_TORN_TAIL_BYTES:
                raise SpendLedgerCorrupt(
                    f"spend ledger ends with a {tail_len}-byte partial event, "
                    f"beyond the {_MAX_RECOVERABLE_TORN_TAIL_BYTES}-byte "
                    "recoverable maximum for a single torn append"
                )
            quarantine_name = self.path.with_suffix(
                self.path.suffix + f".torn-{complete_offset}-{tail_len}"
            )
            warnings.warn(
                f"spend ledger {self.path} ended in a torn {tail_len}-byte partial "
                f"event; discarding it and resuming at sequence {number}. The "
                f"discarded bytes are preserved at {quarantine_name} (a uniquely "
                "suffixed sibling is used instead if that name already exists).",
                SpendLedgerTornTailWarning,
                stacklevel=2,
            )
            quarantine_path = _quarantine_torn_tail(self.path, tail, complete_offset)
            os.ftruncate(self._ledger_descriptor, complete_offset)
            os.fsync(self._ledger_descriptor)
            self.recovered_torn_tail_bytes = tail_len
            self.recovered_torn_tail_path = quarantine_path
        self.ledger_id = ledger_id
        self._global_cap = replay._global_cap
        # Adopted BY REFERENCE: `replay` is local to this call and discarded, and
        # the index dicts already alias the same row objects _apply appends to
        # _events (exactly as they do for live appends).
        self._events = replay._events
        self._stage_caps = replay._stage_caps
        self._starts = replay._starts
        self._attempt_ids_by_execution = replay._attempt_ids_by_execution
        self._reservations = replay._reservations
        self._calls_by_attempt = replay._calls_by_attempt
        self._calls_with_nonstr_attempt = replay._calls_with_nonstr_attempt
        self._evidence = replay._evidence
        self._settlements = replay._settlements
        self._outcomes = replay._outcomes
        self._finishes = replay._finishes
        self._global_commitment = replay._global_commitment
        self._stage_commitments = replay._stage_commitments
        self._run_commitments = replay._run_commitments
        self._sequence = number
        self._last_digest = previous

    def _adjust_commitment(self, reservation: Mapping[str, Any], delta: Decimal) -> None:
        start = self._starts.get(str(reservation.get("attempt_id")))
        if start is None:
            raise SpendLedgerCorrupt("commitment refers to an unknown attempt")
        stage_key = (str(start.get("stage")), str(start.get("model")))
        run_id = str(start.get("run_id"))
        self._global_commitment += delta
        self._stage_commitments[stage_key] = (
            self._stage_commitments.get(stage_key, _ZERO) + delta
        )
        self._run_commitments[run_id] = self._run_commitments.get(run_id, _ZERO) + delta
        if (
            self._global_commitment < 0
            or self._stage_commitments[stage_key] < 0
            or self._run_commitments[run_id] < 0
        ):
            raise SpendLedgerCorrupt("accounted commitment became negative")

    def _recomputed_commitments(
        self,
    ) -> tuple[Decimal, dict[tuple[str, str], Decimal], dict[str, Decimal]]:
        global_total = _ZERO
        stages: dict[tuple[str, str], Decimal] = {}
        runs: dict[str, Decimal] = {}
        for call_id, reservation in self._reservations.items():
            start = self._starts[str(reservation["attempt_id"])]
            settlement = self._settlements.get(call_id)
            value = _ledger_decimal(
                settlement["settled_cost_usd"]
                if settlement is not None
                else reservation["reserved_max_cost_usd"],
                field="accounted commitment",
            )
            stage_key = (str(start["stage"]), str(start["model"]))
            run_id = str(start["run_id"])
            global_total += value
            stages[stage_key] = stages.get(stage_key, _ZERO) + value
            runs[run_id] = runs.get(run_id, _ZERO) + value
        return global_total, stages, runs

    def _validate_commitment_cache(self) -> None:
        expected_global, expected_stages, expected_runs = self._recomputed_commitments()
        expected_stages = {
            key: value for key, value in expected_stages.items() if value != 0
        }
        expected_runs = {
            key: value for key, value in expected_runs.items() if value != 0
        }
        observed_stages = {
            key: value for key, value in self._stage_commitments.items() if value != 0
        }
        observed_runs = {
            key: value for key, value in self._run_commitments.items() if value != 0
        }
        if (
            self._global_commitment != expected_global
            or observed_stages != expected_stages
            or observed_runs != expected_runs
        ):
            raise SpendLedgerCorrupt("cached spend commitments differ from ledger replay")

    def _calls_for_attempt(self, attempt_id: Any) -> list[dict[str, Any]]:
        """The reservations for ``attempt_id``, in ledger order.

        Replaces three full scans of ``_reservations`` -- it is an accelerator
        and nothing more.  The candidate set is a SUPERSET of what those scans
        saw, by construction and for any Python values a JSON ledger can carry:
        a stored ``str`` id can only be ``==`` to an equal ``str``, and such a
        query reads exactly that raw-``str`` bucket; every non-``str`` stored id
        is unconditionally in the overflow list, which every lookup unions in.
        The residual ``==`` then keeps the comparison the scans made byte for
        byte, so the result is neither bigger nor smaller than theirs.  The
        lookup is total: an unhashable JSON ``attempt_id`` is not a ``str``, so
        it takes the ``else`` branch and never reaches ``dict.get``.
        """

        candidates = (
            self._calls_by_attempt.get(attempt_id, ())
            if isinstance(attempt_id, str)
            else ()
        )
        return [
            item
            for item in (*candidates, *self._calls_with_nonstr_attempt)
            if item.get("attempt_id") == attempt_id
        ]

    def _event_identity_matches(
        self, event: Mapping[str, Any], reservation: Mapping[str, Any]
    ) -> bool:
        return all(
            event.get(key) == reservation.get(key)
            for key in (
                "run_id",
                "execution_id",
                "attempt_id",
                "attempt_ordinal",
                "call_id",
                "call_ordinal",
                "provider_model_id",
                "kind",
            )
        )

    def _apply(self, row: dict[str, Any]) -> None:
        event = row.get("event")
        if event == "ledger_initialized":
            if self._events or not isinstance(row.get("ledger_id"), str):
                raise SpendLedgerCorrupt("ledger initialization is misplaced")
            cap = _ledger_decimal(row.get("global_cap_usd"), field="ledger global cap")
            if cap <= 0:
                raise SpendLedgerCorrupt("ledger global cap must be positive")
            self.ledger_id = row["ledger_id"]
            self._global_cap = cap
        elif not self._events:
            raise SpendLedgerCorrupt("ledger does not begin with initialization")
        elif event == "stage_cap_set":
            stage = row.get("stage")
            model = row.get("model")
            cap = _ledger_decimal(row.get("stage_cap_usd"), field="stage cap")
            key = (str(stage), str(model))
            if (
                not isinstance(stage, str)
                or not stage
                or not isinstance(model, str)
                or not model
                or key in self._stage_caps
                or cap <= 0
                or self._global_cap is None
                or cap > self._global_cap
            ):
                raise SpendLedgerCorrupt("stage cap event is invalid or repeated")
            self._stage_caps[key] = cap
        elif event == "stage_cap_amended":
            stage = row.get("stage")
            model = row.get("model")
            key = (str(stage), str(model))
            previous = self._stage_caps.get(key)
            declared_previous = _ledger_decimal(
                row.get("previous_stage_cap_usd"), field="previous stage cap"
            )
            cap = _ledger_decimal(row.get("stage_cap_usd"), field="stage cap")
            if (
                not isinstance(stage, str)
                or not stage
                or not isinstance(model, str)
                or not model
                or previous is None
                or declared_previous != previous
                or cap <= previous
                or self._global_cap is None
                or cap > self._global_cap
                or row.get("authorization_basis") != "current_frozen_run_plan"
            ):
                raise SpendLedgerCorrupt("stage cap amendment is invalid")
            self._stage_caps[key] = cap
        elif event == "attempt_started":
            attempt_id = row.get("attempt_id")
            execution_id = row.get("execution_id")
            ordinal = row.get("attempt_ordinal")
            identity = row.get("execution_identity")
            stage = row.get("stage")
            model = row.get("model")
            workload = row.get("workload")
            run_id = row.get("run_id")
            if (
                not isinstance(attempt_id, str)
                or len(attempt_id) != 32
                or attempt_id in self._starts
                or not isinstance(execution_id, str)
                or _HEX64.fullmatch(execution_id) is None
                or not isinstance(ordinal, int)
                or isinstance(ordinal, bool)
                or not isinstance(identity, dict)
                or _sha256(identity) != execution_id
                or any(
                    not isinstance(value, str) or not value
                    for value in (stage, model, workload, run_id)
                )
                or (stage, model) not in self._stage_caps
                or identity.get("model") != model
                or identity.get("workload_mode") != workload
            ):
                raise SpendLedgerCorrupt("attempt start is malformed")
            prior = self._attempt_ids_by_execution.setdefault(execution_id, [])
            if ordinal != len(prior) + 1:
                raise SpendLedgerCorrupt("attempt ordinals are not contiguous")
            if prior and prior[-1] not in self._finishes:
                raise SpendLedgerCorrupt("a new attempt follows an open attempt")
            prior.append(attempt_id)
            self._starts[attempt_id] = row
        elif event == "call_reserved":
            call_id = row.get("call_id")
            attempt_id = row.get("attempt_id")
            start = self._starts.get(str(attempt_id))
            if (
                not isinstance(call_id, str)
                or len(call_id) != 32
                or call_id in self._reservations
                or start is None
                or attempt_id in self._finishes
                or attempt_id in self._outcomes
                or row.get("execution_id") != start.get("execution_id")
                or row.get("attempt_ordinal") != start.get("attempt_ordinal")
                or row.get("run_id") != start.get("run_id")
                or row.get("call_ordinal") != 1 + len(self._calls_for_attempt(attempt_id))
            ):
                raise SpendLedgerCorrupt("call reservation is malformed")
            reserved_input = row.get("reserved_input_tokens")
            reserved_output = row.get("reserved_output_tokens")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (reserved_input, reserved_output)
            ):
                raise SpendLedgerCorrupt("call reservation token bounds are invalid")
            input_price = _ledger_decimal(
                row.get("input_usd_per_million"), field="input price"
            )
            output_price = _ledger_decimal(
                row.get("output_usd_per_million"), field="output price"
            )
            maximum = _ledger_decimal(
                row.get("reserved_max_cost_usd"), field="reserved maximum"
            )
            expected_maximum = (
                Decimal(reserved_input) * input_price
                + Decimal(reserved_output) * output_price
            ) / _MILLION
            if (
                maximum != expected_maximum
                or not isinstance(row.get("pricing_basis"), str)
                or not row["pricing_basis"]
            ):
                raise SpendLedgerCorrupt("call reservation pricing arithmetic differs")
            request = row.get("request_material")
            if not isinstance(request, dict) or _sha256(request) != row.get("provider_request_sha256"):
                raise SpendLedgerCorrupt("call reservation request commitment differs")
            self._reservations[call_id] = row
            # Side index, maintained at the ONE insertion site so it cannot drift
            # from _reservations.  Buckets are keyed by the RAW str attempt id;
            # every other id lands in the overflow list, which every lookup
            # unions in.  That split buys a superset property that holds by
            # construction, for any Python values a JSON ledger can carry: a
            # stored str can only be ``==`` to a str, and a str query reads
            # exactly its own raw-str bucket, so a matching stored str is always
            # a candidate; every non-str stored id is unconditionally a
            # candidate via the overflow.  No reachability argument is needed.
            # Keying on str(attempt_id) instead would need ``x == y`` to imply
            # ``str(x) == str(y)``, which is false in Python (1 == 1.0 == True).
            # It happens to be unreachable here -- the fifth clause above,
            # ``attempt_id in self._finishes``, raises TypeError on an
            # unhashable id before any row is stored, and no two HASHABLE JSON
            # values are ``==`` with different 32-char str()s -- but that safety
            # rests on the order of a check this index is forbidden to touch,
            # so the index does not lean on it.
            attempt_key = row.get("attempt_id")
            if isinstance(attempt_key, str):
                self._calls_by_attempt.setdefault(attempt_key, []).append(row)
            else:
                self._calls_with_nonstr_attempt.append(row)
            self._adjust_commitment(
                row,
                _ledger_decimal(
                    row["reserved_max_cost_usd"], field="reserved maximum"
                ),
            )
        elif event == "call_evidence_observed":
            call_id = row.get("call_id")
            reservation = self._reservations.get(str(call_id))
            evidence = row.get("call_evidence")
            if (
                reservation is None
                or call_id in self._evidence
                or not self._event_identity_matches(row, reservation)
                or not isinstance(evidence, dict)
                or _sha256(evidence) != row.get("call_evidence_sha256")
            ):
                raise SpendLedgerCorrupt("provider call evidence is malformed")
            self._evidence[str(call_id)] = row
        elif event == "call_settled":
            call_id = row.get("call_id")
            reservation = self._reservations.get(str(call_id))
            usage = row.get("provider_usage")
            basis = row.get("accounting_basis")
            if (
                reservation is None
                or call_id in self._settlements
                or call_id not in self._evidence
                or not self._event_identity_matches(row, reservation)
                or not isinstance(usage, dict)
                or basis not in {"provider_reported_usage", "conservative_reserved_maximum"}
            ):
                raise SpendLedgerCorrupt("call settlement is malformed")
            cost = _ledger_decimal(row.get("settled_cost_usd"), field="settled cost")
            values = (usage.get("input_tokens"), usage.get("output_tokens"))
            if basis == "provider_reported_usage":
                if any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in values
                ):
                    raise SpendLedgerCorrupt("measured settlement lacks token usage")
                expected_cost = (
                    Decimal(values[0])
                    * _ledger_decimal(
                        reservation["input_usd_per_million"], field="input price"
                    )
                    + Decimal(values[1])
                    * _ledger_decimal(
                        reservation["output_usd_per_million"], field="output price"
                    )
                ) / _MILLION
                maximum = _ledger_decimal(
                    reservation["reserved_max_cost_usd"], field="reserved maximum"
                )
                expected_breach = (
                    values[0] > reservation["reserved_input_tokens"]
                    or values[1] > reservation["reserved_output_tokens"]
                    or expected_cost > maximum
                )
                if (
                    cost != expected_cost
                    or row.get("reservation_breached") is not expected_breach
                ):
                    raise SpendLedgerCorrupt("measured settlement pricing arithmetic differs")
            elif (
                any(value is not None for value in values)
                or cost
                != _ledger_decimal(
                    reservation.get("reserved_max_cost_usd"), field="reserved maximum"
                )
                or row.get("reservation_breached") is not False
            ):
                raise SpendLedgerCorrupt("conservative settlement differs from reservation")
            self._settlements[str(call_id)] = row
            self._adjust_commitment(
                reservation,
                cost
                - _ledger_decimal(
                    reservation["reserved_max_cost_usd"], field="reserved maximum"
                ),
            )
        elif event == "attempt_outcome_committed":
            attempt_id = row.get("attempt_id")
            start = self._starts.get(str(attempt_id))
            raw_row = row.get("raw_row")
            calls = self._calls_for_attempt(attempt_id)
            if (
                start is None
                or attempt_id in self._outcomes
                or attempt_id in self._finishes
                or row.get("run_id") != start.get("run_id")
                or row.get("execution_id") != start.get("execution_id")
                or row.get("attempt_ordinal") != start.get("attempt_ordinal")
                or not isinstance(raw_row, dict)
                or _sha256(raw_row) != row.get("raw_row_sha256")
                or any(item["call_id"] not in self._settlements for item in calls)
            ):
                raise SpendLedgerCorrupt("attempt outcome is malformed")
            self._outcomes[str(attempt_id)] = row
        elif event == "attempt_finished":
            attempt_id = row.get("attempt_id")
            start = self._starts.get(str(attempt_id))
            outcome = self._outcomes.get(str(attempt_id))
            if (
                start is None
                or outcome is None
                or attempt_id in self._finishes
                or row.get("run_id") != start.get("run_id")
                or row.get("execution_id") != start.get("execution_id")
                or row.get("attempt_ordinal") != start.get("attempt_ordinal")
                or row.get("status") != outcome.get("status")
            ):
                raise SpendLedgerCorrupt("attempt finish is malformed")
            self._finishes[str(attempt_id)] = row
        else:
            raise SpendLedgerCorrupt(f"unsupported spend event {event!r}")
        self._events.append(row)

    def _assert_ledger_current(self) -> None:
        if self._closed or self._ledger_descriptor is None or self._ledger_identity is None:
            raise SpendGuardError("spend guard is closed")
        self._lock.assert_current()
        try:
            info = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise SpendGuardError("spend ledger pathname disappeared") from exc
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != self._ledger_identity:
            raise SpendGuardError("spend ledger pathname changed")

    def assert_paid_surface(self) -> None:
        self._assert_ledger_current()
        if self._surface_assertion is not None:
            self._surface_assertion()

    def _append(self, value: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_ledger_current()
        assert self._ledger_descriptor is not None
        payload = _json_copy(dict(value), field="ledger event")
        if payload.get("event") != "ledger_initialized":
            if not self.ledger_id:
                raise SpendGuardError("spend ledger has no identity")
            payload = {"ledger_id": self.ledger_id, **payload}
        body = {
            **payload,
            "recorded_at": _now(),
            "sequence": self._sequence + 1,
            "previous_event_sha256": self._last_digest,
        }
        digest = hashlib.sha256(_ledger_json_bytes(body)).hexdigest()
        row = {**body, "event_sha256": digest}
        payload = _ledger_json_bytes(row) + b"\n"
        written = os.write(self._ledger_descriptor, payload)
        if written != len(payload):
            raise SpendGuardError("short append to spend ledger")
        os.fsync(self._ledger_descriptor)
        self._apply(row)
        self._sequence += 1
        self._last_digest = digest
        return row

    @staticmethod
    def _identity(model: str, workload: str, identity: Mapping[str, Any]) -> dict[str, Any]:
        if "model" in identity or "workload_mode" in identity:
            raise SpendGuardError("execution identity contains reserved fields")
        return _json_copy(
            {"model": model, "workload_mode": workload, **dict(identity)},
            field="execution identity",
        )

    def _context(self) -> _AttemptContext:
        context = getattr(self._tls, "attempt", None)
        if not isinstance(context, _AttemptContext):
            raise SpendGuardError("provider call is outside an attempt")
        return context

    def _start_attempt(self, context: _AttemptContext) -> AttemptReceipt:
        if context.started:
            return context.receipt
        prior = self._attempt_ids_by_execution.get(context.execution_id, [])
        if prior and prior[-1] not in self._finishes:
            raise SpendGuardError("execution has an unfinished prior attempt")
        if any(self._attempt_is_terminal(item) for item in prior):
            raise AttemptLimitReached("execution already completed successfully")
        if len(prior) >= self.max_attempts:
            raise AttemptLimitReached("execution attempt limit reached")
        ordinal = len(prior) + 1
        self._append(
            {
                "event": "attempt_started",
                "run_id": self.run_id,
                "stage": self.stage,
                "model": self.model,
                "workload": self.workload,
                "execution_id": context.execution_id,
                "execution_identity": context.identity,
                "attempt_id": context.attempt_id,
                "attempt_ordinal": ordinal,
            }
        )
        context.started = True
        context.attempt_ordinal = ordinal
        context.receipt.started = True
        context.receipt.attempt_ordinal = ordinal
        return context.receipt

    def _attempt_is_terminal(self, attempt_id: str) -> bool:
        finish = self._finishes.get(attempt_id)
        outcome = self._outcomes.get(attempt_id)
        if finish is None or outcome is None or finish.get("status") != "completed":
            return False
        raw = outcome.get("raw_row")
        if not isinstance(raw, Mapping) or raw.get("row_status") != "scored":
            return False
        # A scored row with an explicit verdict field is terminal only when that
        # verdict satisfies the comparison contract. Other guard consumers may
        # use scored outcomes without statement-verdict semantics.
        return "verdict" not in raw or raw.get("verdict") in {"correct", "incorrect"}

    @contextlib.contextmanager
    def attempt(self, identity: Mapping[str, Any]) -> Iterator[AttemptReceipt]:
        if getattr(self._tls, "attempt", None) is not None:
            raise SpendGuardError("attempt contexts cannot be nested")
        normalized = self._identity(self.model, self.workload, identity)
        execution_id = _sha256(normalized)
        attempt_id = uuid.uuid4().hex
        receipt = AttemptReceipt(execution_id, attempt_id, normalized)
        context = _AttemptContext(execution_id, normalized, attempt_id, receipt)
        self._tls.attempt = context
        try:
            yield receipt
        finally:
            try:
                with self._mutex:
                    if context.started:
                        outcome = self._outcomes.get(context.attempt_id)
                        if outcome is not None and context.attempt_id not in self._finishes:
                            self._finish_attempt(context.attempt_id)
            finally:
                self._tls.attempt = None

    def ensure_attempt_started(self) -> AttemptReceipt:
        with self._mutex:
            return self._start_attempt(self._context())

    def _commit_outcome(self, attempt_id: str, raw_row: Mapping[str, Any]) -> dict[str, Any]:
        start = self._starts.get(attempt_id)
        if start is None or attempt_id in self._outcomes:
            raise SpendGuardError("attempt cannot accept an outcome")
        row = _json_copy(dict(raw_row), field="raw outcome row")
        status = "completed" if row.get("row_status") == "scored" else "error"
        event = self._append(
            {
                "event": "attempt_outcome_committed",
                "run_id": start["run_id"],
                "execution_id": start["execution_id"],
                "attempt_id": attempt_id,
                "attempt_ordinal": start["attempt_ordinal"],
                "status": status,
                "raw_row": row,
                "raw_row_sha256": _sha256(row),
            }
        )
        return event

    def commit_attempt_outcome(self, raw_row: Mapping[str, Any]) -> None:
        with self._mutex:
            context = self._context()
            receipt = self._start_attempt(context)
            event = self._commit_outcome(context.attempt_id, raw_row)
            receipt.status = str(event["status"])
            receipt.raw_row_sha256 = str(event["raw_row_sha256"])
            receipt.outcome_event_sha256 = str(event["event_sha256"])

    def _attempt_id(self, execution_id: str, attempt_ordinal: int) -> str:
        values = self._attempt_ids_by_execution.get(execution_id, [])
        if attempt_ordinal < 1 or attempt_ordinal > len(values):
            raise SpendGuardError("deferred attempt does not exist")
        return values[attempt_ordinal - 1]

    def commit_deferred_attempt_outcome(
        self, execution_id: str, attempt_ordinal: int, raw_row: Mapping[str, Any]
    ) -> None:
        with self._mutex:
            self._commit_outcome(self._attempt_id(execution_id, attempt_ordinal), raw_row)

    def _finish_attempt(self, attempt_id: str) -> dict[str, Any]:
        start = self._starts[attempt_id]
        outcome = self._outcomes.get(attempt_id)
        if outcome is None or attempt_id in self._finishes:
            raise SpendGuardError("attempt cannot be finished")
        return self._append(
            {
                "event": "attempt_finished",
                "run_id": start["run_id"],
                "execution_id": start["execution_id"],
                "attempt_id": attempt_id,
                "attempt_ordinal": start["attempt_ordinal"],
                "status": outcome["status"],
            }
        )

    def finish_deferred_attempt(self, execution_id: str, attempt_ordinal: int) -> None:
        with self._mutex:
            self._finish_attempt(self._attempt_id(execution_id, attempt_ordinal))

    def _commitment(
        self,
        *,
        stage: tuple[str, str] | None = None,
        run_id: str | None = None,
    ) -> Decimal:
        if stage is not None and run_id is not None:
            raise SpendGuardError("commitment accepts either stage or run_id")
        if stage is not None:
            return self._stage_commitments.get(stage, _ZERO)
        if run_id is not None:
            return self._run_commitments.get(run_id, _ZERO)
        return self._global_commitment

    def commitment(
        self,
        *,
        stage: tuple[str, str] | None = None,
        run_id: str | None = None,
    ) -> Decimal:
        """Return an O(1) commitment rebuilt and checked on ledger reload."""

        with self._mutex:
            self.assert_paid_surface()
            return self._commitment(stage=stage, run_id=run_id)

    def reserve_call(
        self,
        *,
        provider_model_id: str,
        kind: str,
        max_output_tokens: int,
        system: str,
        messages: list[dict[str, Any]],
        provider_wire_request_sha256: str | None = None,
        provider_request_body: Mapping[str, Any] | None = None,
    ) -> CallReservation:
        with self._mutex:
            self.assert_paid_surface()
            context = self._context()
            price = price_for(provider_model_id)
            if price is None:
                raise SpendGuardError(f"provider model {provider_model_id!r} has no price")
            if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
                raise SpendGuardError("maximum output tokens must be positive")
            normalized_system = str(system or "")
            normalized_messages = _json_copy(list(messages or []), field="provider messages")
            input_tokens = max(
                self.provider_input_token_maximum,
                conservative_prompt_token_bound(normalized_system, normalized_messages),
            )
            input_price, output_price = Decimal(str(price[0])), Decimal(str(price[1]))
            output_token_bound = max_output_tokens + PROVIDER_OUTPUT_TOKEN_OVERSHOOT
            maximum = (
                Decimal(input_tokens) * input_price
                + Decimal(output_token_bound) * output_price
            ) / _MILLION
            if self._commitment() + maximum > self.approved_cap_usd:
                raise SpendCapReached("next reservation would exceed the global cap")
            if self._commitment(stage=(self.stage, self.model)) + maximum > self.stage_cap_usd:
                raise SpendCapReached("next reservation would exceed the stage cap")
            receipt = self._start_attempt(context)
            context.call_ordinal += 1
            receipt.model_calls_reserved += 1
            request_material: dict[str, Any] = {
                "provider_model_id": provider_model_id,
                "kind": str(kind),
                "call_ordinal": context.call_ordinal,
                "max_output_tokens": max_output_tokens,
                "system": normalized_system,
                "messages": normalized_messages,
            }
            if provider_wire_request_sha256 is not None:
                request_material["provider_wire_request_sha256"] = provider_wire_request_sha256
            request_digest = provider_request_sha256(**request_material)
            body = (
                _json_copy(dict(provider_request_body), field="provider request body")
                if provider_request_body is not None
                else None
            )
            if provider_wire_request_sha256 is not None:
                if body is None or provider_wire_body_sha256(body) != provider_wire_request_sha256:
                    raise SpendGuardError("provider wire request differs from its commitment")
            call_id = uuid.uuid4().hex
            event = self._append(
                {
                    "event": "call_reserved",
                    "run_id": self.run_id,
                    "execution_id": context.execution_id,
                    "attempt_id": context.attempt_id,
                    "attempt_ordinal": context.attempt_ordinal,
                    "call_id": call_id,
                    "call_ordinal": context.call_ordinal,
                    "provider_model_id": provider_model_id,
                    "kind": str(kind),
                    "reserved_input_tokens": input_tokens,
                    "reserved_output_tokens": output_token_bound,
                    "input_usd_per_million": _money(input_price),
                    "output_usd_per_million": _money(output_price),
                    "pricing_basis": str(price[2]),
                    "reserved_max_cost_usd": _money(maximum),
                    "provider_request_sha256": request_digest,
                    "provider_wire_request_sha256": provider_wire_request_sha256,
                    "provider_request_body": body,
                    "request_material": request_material,
                }
            )
            return CallReservation(
                call_id=call_id,
                attempt_id=context.attempt_id,
                execution_id=context.execution_id,
                attempt_ordinal=int(context.attempt_ordinal),
                call_ordinal=context.call_ordinal,
                provider_model_id=provider_model_id,
                kind=str(kind),
                reserved_input_tokens=input_tokens,
                reserved_output_tokens=output_token_bound,
                input_usd_per_million=input_price,
                output_usd_per_million=output_price,
                reserved_max_cost_usd=maximum,
                provider_request_sha256=str(event["provider_request_sha256"]),
                provider_wire_request_sha256=provider_wire_request_sha256,
                provider_request_body=body,
            )

    def observe_call_evidence(
        self,
        reservation: CallReservation,
        call_evidence: Mapping[str, Any],
        *,
        evidence_kind: str | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        with self._mutex:
            stored = self._reservations.get(reservation.call_id)
            if stored is None or reservation.call_id in self._evidence:
                raise SpendGuardError("call cannot accept provider evidence")
            evidence = _json_copy(dict(call_evidence), field="provider call evidence")
            failure_class, status = classify_provider_failure(error)
            if error is not None:
                evidence["provider_failure_class"] = failure_class
                evidence["provider_http_status"] = status
                evidence["error_message_sha256"] = hashlib.sha256(
                    str(error).encode("utf-8", errors="replace")
                ).hexdigest()
            return self._append(
                {
                    "event": "call_evidence_observed",
                    **{
                        key: stored[key]
                        for key in (
                            "run_id",
                            "execution_id",
                            "attempt_id",
                            "attempt_ordinal",
                            "call_id",
                            "call_ordinal",
                            "provider_model_id",
                            "kind",
                        )
                    },
                    "evidence_kind": evidence_kind
                    or ("provider_error" if error is not None else "provider_response"),
                    "call_evidence": evidence,
                    "call_evidence_sha256": _sha256(evidence),
                }
            )

    def _settle(
        self,
        reservation: Mapping[str, Any],
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        conservative: bool,
        provider_failure_class: str | None = None,
        provider_http_status: int | None = None,
    ) -> tuple[dict[str, Any], bool]:
        call_id = str(reservation["call_id"])
        if call_id not in self._evidence or call_id in self._settlements:
            raise SpendGuardError("call cannot be settled")
        maximum = _ledger_decimal(
            reservation["reserved_max_cost_usd"], field="reserved maximum"
        )
        breached = False
        if conservative:
            cost = maximum
            usage = {"input_tokens": None, "output_tokens": None}
            basis = "conservative_reserved_maximum"
        else:
            assert input_tokens is not None and output_tokens is not None
            cost = (
                Decimal(input_tokens)
                * _ledger_decimal(
                    reservation["input_usd_per_million"], field="input price"
                )
                + Decimal(output_tokens)
                * _ledger_decimal(
                    reservation["output_usd_per_million"], field="output price"
                )
            ) / _MILLION
            usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
            basis = "provider_reported_usage"
            breached = (
                input_tokens > int(reservation["reserved_input_tokens"])
                or output_tokens > int(reservation["reserved_output_tokens"])
                or cost > maximum
            )
        event = self._append(
            {
                "event": "call_settled",
                **{
                    key: reservation[key]
                    for key in (
                        "run_id",
                        "execution_id",
                        "attempt_id",
                        "attempt_ordinal",
                        "call_id",
                        "call_ordinal",
                        "provider_model_id",
                        "kind",
                    )
                },
                "accounting_basis": basis,
                "provider_usage": usage,
                "settled_cost_usd": _money(cost),
                "provider_failure_class": provider_failure_class,
                "provider_http_status": provider_http_status,
                "reservation_breached": breached,
            }
        )
        return event, breached

    def settle_call(
        self,
        reservation: CallReservation,
        *,
        response: Any | None,
        error: BaseException | None,
    ) -> dict[str, Any]:
        with self._mutex:
            stored = self._reservations.get(reservation.call_id)
            if stored is None:
                raise SpendGuardError("settlement refers to an unknown reservation")
            prompt = getattr(response, "prompt_tokens", None) if error is None else None
            output = getattr(response, "tokens", None) if error is None else None
            known = all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (prompt, output)
            )
            failure_class, status = classify_provider_failure(error)
            event, breached = self._settle(
                stored,
                input_tokens=prompt if known else None,
                output_tokens=output if known else None,
                conservative=not known,
                provider_failure_class=failure_class,
                provider_http_status=status,
            )
            if breached:
                raise SpendReservationBreach(
                    "provider usage exceeded its pre-call maximum reservation"
                )
            return event

    def _interrupted_evidence(self, reservation: Mapping[str, Any]) -> dict[str, Any]:
        request = reservation["request_material"]
        return {
            "execution_id": reservation["execution_id"],
            "attempt_id": reservation["attempt_id"],
            "attempt_ordinal": reservation["attempt_ordinal"],
            "call_id": reservation["call_id"],
            "call_ordinal": reservation["call_ordinal"],
            "kind": reservation["kind"],
            "model_id": reservation["provider_model_id"],
            "max_tokens": request["max_output_tokens"],
            "system": request["system"],
            "messages": request["messages"],
            "provider_request_sha256": reservation["provider_request_sha256"],
            "provider_wire_request_sha256": reservation.get(
                "provider_wire_request_sha256"
            ),
            "provider_request_body": reservation.get("provider_request_body"),
            "provider_response_sha256": None,
            "content": None,
            "reasoning": None,
            "raw_text": None,
            "prompt_tokens": None,
            "out_tokens": None,
            "finish_reason": None,
            "error": "InterruptedProviderCall",
            "provider_call_outcome": "unknown_after_interruption",
            "provider_failure_class": "transport_or_server",
            "provider_http_status": None,
            "request_material_complete": True,
        }

    def _reconcile_interrupted_calls(self) -> None:
        with self._mutex:
            for call_id, reservation in list(self._reservations.items()):
                if call_id in self._settlements:
                    continue
                if call_id not in self._evidence:
                    evidence = self._interrupted_evidence(reservation)
                    self._append(
                        {
                            "event": "call_evidence_observed",
                            **{
                                key: reservation[key]
                                for key in (
                                    "run_id",
                                    "execution_id",
                                    "attempt_id",
                                    "attempt_ordinal",
                                    "call_id",
                                    "call_ordinal",
                                    "provider_model_id",
                                    "kind",
                                )
                            },
                            "evidence_kind": "interrupted_provider_call",
                            "call_evidence": evidence,
                            "call_evidence_sha256": _sha256(evidence),
                        }
                    )
                self._settle(
                    reservation,
                    input_tokens=None,
                    output_tokens=None,
                    conservative=True,
                    provider_failure_class="transport_or_server",
                )

    def _attempt_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for start in sorted(self._starts.values(), key=lambda item: item["sequence"]):
            attempt_id = str(start["attempt_id"])
            calls: list[dict[str, Any]] = []
            reservations = sorted(
                self._calls_for_attempt(attempt_id),
                key=lambda item: item["call_ordinal"],
            )
            for reservation in reservations:
                call_id = str(reservation["call_id"])
                calls.append(
                    {
                        "reservation": reservation,
                        "evidence": self._evidence.get(call_id),
                        "settlement": self._settlements.get(call_id),
                    }
                )
            rows.append(
                {
                    "execution_id": start["execution_id"],
                    "attempt_id": attempt_id,
                    "attempt_ordinal": start["attempt_ordinal"],
                    "start": start,
                    "calls": calls,
                    "outcome": self._outcomes.get(attempt_id),
                    "finish": self._finishes.get(attempt_id),
                }
            )
        return rows

    def resume_reconciliation(self) -> dict[str, Any]:
        with self._mutex:
            self.assert_paid_surface()
            return _json_copy(
                {
                    "ledger_id": self.ledger_id,
                    "sequence": self._sequence,
                    "attempts": self._attempt_rows(),
                },
                field="ledger reconciliation",
            )

    def summary(self) -> dict[str, Any]:
        with self._mutex:
            measured = sum(
                (
                    _ledger_decimal(row["settled_cost_usd"], field="settled cost")
                    for row in self._settlements.values()
                    if row["accounting_basis"] == "provider_reported_usage"
                ),
                _ZERO,
            )
            conservative = sum(
                (
                    _ledger_decimal(row["settled_cost_usd"], field="settled cost")
                    for row in self._settlements.values()
                    if row["accounting_basis"] == "conservative_reserved_maximum"
                ),
                _ZERO,
            )
            inflight = sum(
                (
                    _ledger_decimal(row["reserved_max_cost_usd"], field="reserved maximum")
                    for key, row in self._reservations.items()
                    if key not in self._settlements
                ),
                _ZERO,
            )
            commitment = self._commitment()
            stage_commitment = self._commitment(stage=(self.stage, self.model))
            run_commitment = self._commitment(run_id=self.run_id)
            return {
                "ledger_id": self.ledger_id,
                "ledger_sequence": self._sequence,
                "global_cap_usd": _money(self.approved_cap_usd),
                "stage": self.stage,
                "stage_cap_usd": _money(self.stage_cap_usd),
                "run_id": self.run_id,
                "provider_measured_spend_usd": _money(measured),
                "conservative_spend_usd": _money(conservative),
                "inflight_reserved_maximum_usd": _money(inflight),
                "accounted_commitment_usd": _money(commitment),
                "global_remaining_usd": _money(self.approved_cap_usd - commitment),
                "stage_accounted_commitment_usd": _money(stage_commitment),
                "stage_remaining_usd": _money(self.stage_cap_usd - stage_commitment),
                "run_accounted_commitment_usd": _money(run_commitment),
                "attempt_count": len(self._starts),
                "provider_call_count": len(self._reservations),
                "provider_measured_call_count": sum(
                    row["accounting_basis"] == "provider_reported_usage"
                    for row in self._settlements.values()
                ),
                "conservative_call_count": sum(
                    row["accounting_basis"] == "conservative_reserved_maximum"
                    for row in self._settlements.values()
                ),
                # Appended LAST and int-valued: this dict is splatted into the
                # persisted action manifest, so a torn-tail repair at open is
                # recorded in the run record instead of living only in a warning.
                "recovered_torn_tail_bytes": self.recovered_torn_tail_bytes,
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        if self._ledger_descriptor is not None:
            try:
                os.close(self._ledger_descriptor)
            except BaseException as exc:
                error = exc
            self._ledger_descriptor = None
        try:
            self._lock.close()
        except BaseException as exc:
            error = error or exc
        if error is not None:
            raise error

    def __enter__(self) -> "SpendGuard":
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self.close()


class _LedgerReplay:
    """Minimal state target for the guard's single event-transition function."""

    _adjust_commitment = SpendGuard._adjust_commitment
    _recomputed_commitments = SpendGuard._recomputed_commitments
    _validate_commitment_cache = SpendGuard._validate_commitment_cache
    _event_identity_matches = SpendGuard._event_identity_matches
    _calls_for_attempt = SpendGuard._calls_for_attempt

    def __init__(self) -> None:
        _initialize_ledger_state(self)

    def apply(self, row: Mapping[str, Any]) -> None:
        SpendGuard._apply(self, dict(row))  # type: ignore[arg-type]

def _build_spend_ledger_snapshot(
    ledger_id: str,
    events: tuple[Mapping[str, Any], ...],
    last_event_sha256: str,
) -> SpendLedgerSnapshot:
    replay = _LedgerReplay()
    for event in events:
        replay.apply(event)
    replay._validate_commitment_cache()
    if replay.ledger_id != ledger_id:
        raise SpendLedgerCorrupt("ledger replay identity differs")
    return SpendLedgerSnapshot(ledger_id, events, last_event_sha256, replay)


class GuardedModelClient:
    """ModelClient proxy that persists reservation, evidence, and settlement."""

    def __init__(self, client: Any, guard: SpendGuard):
        self._client = client
        self._guard = guard
        self._tls = threading.local()
        self._client._spend_guard_disable_internal_retries = True
        if getattr(self._client, "backend", None) == "openai_compat":
            transport = getattr(self._client, "_client", None)
            with_options = getattr(transport, "with_options", None)
            if not callable(with_options):
                raise SpendGuardError("cannot disable provider SDK retries")
            self._client._client = with_options(max_retries=0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _retained(self) -> list[dict[str, Any]]:
        if not hasattr(self._tls, "calls"):
            self._tls.calls = []
        return self._tls.calls

    def pop_call_log(self) -> list[dict[str, Any]]:
        rows = _json_copy(self._retained(), field="retained call log")
        self._retained().clear()
        return rows

    def _provider_wire_request(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        backend = getattr(self._client, "backend", None)
        effective_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else self._client.config.get("reasoning_effort")
        )
        if backend == "bedrock_responses_raw":
            from indra_belief.bedrock_responses_transport import build_bedrock_responses_body

            body = build_bedrock_responses_body(
                model_id=self._client.config["model_id"],
                system=system,
                messages=messages,
                max_output_tokens=max_output_tokens,
                reasoning_effort=effective_effort,
            )
            return body, provider_wire_body_sha256(body)
        if backend == "bedrock_chat_completions_raw":
            from indra_belief.bedrock_chat_transport import build_bedrock_chat_body

            body = build_bedrock_chat_body(
                model_id=self._client.config["model_id"],
                system=system,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
                response_format=response_format,
                reasoning_effort=effective_effort,
            )
            return body, provider_wire_body_sha256(body)
        return None, None

    @staticmethod
    def _unknown_call(
        reservation: CallReservation,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_output_tokens: int,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "execution_id": reservation.execution_id,
            "attempt_id": reservation.attempt_id,
            "attempt_ordinal": reservation.attempt_ordinal,
            "call_id": reservation.call_id,
            "call_ordinal": reservation.call_ordinal,
            "kind": reservation.kind,
            "model_id": reservation.provider_model_id,
            "max_tokens": max_output_tokens,
            "system": system,
            "messages": messages,
            "provider_request_sha256": reservation.provider_request_sha256,
            "provider_wire_request_sha256": reservation.provider_wire_request_sha256,
            "provider_request_body": reservation.provider_request_body,
            "provider_response_sha256": None,
            "content": None,
            "reasoning": None,
            "raw_text": None,
            "prompt_tokens": None,
            "out_tokens": None,
            "finish_reason": None,
            "provider_call_outcome": reason,
            "request_material_complete": True,
        }

    def _bind_call_log(
        self,
        reservation: CallReservation,
        *,
        response: Any | None,
        error: BaseException | None,
    ) -> dict[str, Any]:
        pop = getattr(self._client, "pop_call_log", None)
        if not callable(pop):
            raise SpendGuardError("model client cannot expose its call log")
        rows = pop()
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise SpendGuardError("one provider call must produce one immediate log row")
        row = dict(rows[0])
        if row.get("kind") != reservation.kind or row.get("model_id") != reservation.provider_model_id:
            raise SpendGuardError("provider call log differs from its reservation")
        request_digest = provider_request_sha256(
            provider_model_id=str(row.get("model_id") or ""),
            kind=str(row.get("kind") or ""),
            call_ordinal=reservation.call_ordinal,
            max_output_tokens=row.get("max_tokens"),
            system=row.get("system"),
            messages=row.get("messages"),
            provider_wire_request_sha256=reservation.provider_wire_request_sha256,
        )
        if request_digest != reservation.provider_request_sha256:
            raise SpendGuardError("provider request log changed after reservation")
        if reservation.provider_wire_request_sha256 is not None:
            trace = row.get("transport_trace")
            if (
                not isinstance(trace, Mapping)
                or trace.get("request_body_sha256")
                != reservation.provider_wire_request_sha256
            ):
                raise SpendGuardError("raw transport request differs from reservation")
        response_digest: str | None = None
        if error is None:
            if response is None:
                raise SpendGuardError("successful provider call lacks a response")
            response_digest = provider_response_sha256(
                provider_model_id=reservation.provider_model_id,
                kind=reservation.kind,
                call_ordinal=reservation.call_ordinal,
                content=row.get("content"),
                reasoning=row.get("reasoning"),
                raw_text=row.get("raw_text"),
                prompt_tokens=row.get("prompt_tokens"),
                output_tokens=row.get("out_tokens"),
                finish_reason=row.get("finish_reason"),
            )
            expected = provider_response_sha256(
                provider_model_id=reservation.provider_model_id,
                kind=reservation.kind,
                call_ordinal=reservation.call_ordinal,
                content=getattr(response, "content", None),
                reasoning=getattr(response, "reasoning", None),
                raw_text=getattr(response, "raw_text", None),
                prompt_tokens=getattr(response, "prompt_tokens", None),
                output_tokens=getattr(response, "tokens", None),
                finish_reason=getattr(response, "finish_reason", None),
            )
            if response_digest != expected:
                raise SpendGuardError("provider call log differs from returned response")
        elif row.get("error") != type(error).__name__:
            raise SpendGuardError("provider error log differs from the raised error")
        row.update(
            {
                "execution_id": reservation.execution_id,
                "attempt_id": reservation.attempt_id,
                "attempt_ordinal": reservation.attempt_ordinal,
                "call_id": reservation.call_id,
                "call_ordinal": reservation.call_ordinal,
                "provider_request_sha256": reservation.provider_request_sha256,
                "provider_wire_request_sha256": reservation.provider_wire_request_sha256,
                "provider_request_body": reservation.provider_request_body,
                "provider_response_sha256": response_digest,
            }
        )
        return _json_copy(row, field="bound provider call log")

    def call(self, *args: Any, **kwargs: Any) -> Any:
        system = str(kwargs.get("system", args[0] if args else "") or "")
        messages = list(kwargs.get("messages", args[1] if len(args) > 1 else []) or [])
        requested = kwargs.get("max_tokens", args[2] if len(args) > 2 else None)
        max_output_tokens = int(requested or self._client.config.get("max_tokens", 2000))
        kind = str(kwargs.get("kind", "unknown"))
        provider_model_id = str(self._client.config.get("model_id") or "")
        if not provider_model_id:
            raise SpendGuardError("model client lacks a provider model_id")
        body, wire_digest = self._provider_wire_request(
            system=system,
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=kwargs.get("temperature", args[3] if len(args) > 3 else 0.1),
            response_format=kwargs.get("response_format"),
            reasoning_effort=kwargs.get("reasoning_effort"),
        )
        reservation = self._guard.reserve_call(
            provider_model_id=provider_model_id,
            kind=kind,
            max_output_tokens=max_output_tokens,
            system=system,
            messages=messages,
            provider_wire_request_sha256=wire_digest,
            provider_request_body=body,
        )
        try:
            self._guard.assert_paid_surface()
        except BaseException as exc:
            unknown = self._unknown_call(
                reservation,
                system=system,
                messages=messages,
                max_output_tokens=max_output_tokens,
                reason="provider_not_invoked_after_reservation",
            )
            event = self._guard.observe_call_evidence(
                reservation, unknown, evidence_kind="provider_not_invoked", error=exc
            )
            self._retained().append(event["call_evidence"])
            self._guard.settle_call(reservation, response=None, error=exc)
            raise

        response: Any | None = None
        error: BaseException | None = None
        try:
            response = self._client.call(*args, **kwargs)
        except BaseException as exc:
            error = exc
        try:
            bound = self._bind_call_log(reservation, response=response, error=error)
        except BaseException as exc:
            contract_error = (
                exc if isinstance(exc, SpendGuardError) else SpendGuardError("call-log binding failed")
            )
            unknown = self._unknown_call(
                reservation,
                system=system,
                messages=messages,
                max_output_tokens=max_output_tokens,
                reason="call_log_contract_breach",
            )
            event = self._guard.observe_call_evidence(
                reservation, unknown, evidence_kind="call_outcome_unknown", error=contract_error
            )
            self._retained().append(event["call_evidence"])
            self._guard.settle_call(reservation, response=None, error=contract_error)
            raise contract_error from exc
        event = self._guard.observe_call_evidence(reservation, bound, error=error)
        self._retained().append(event["call_evidence"])
        self._guard.settle_call(reservation, response=response, error=error)
        if error is not None:
            raise error
        return response
