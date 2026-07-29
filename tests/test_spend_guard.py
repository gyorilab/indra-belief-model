from __future__ import annotations

import hashlib
import json
import threading
import warnings
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from indra_belief import spend_guard as spend_guard_module
from indra_belief.spend_guard import (
    PROVIDER_OUTPUT_TOKEN_OVERSHOOT,
    AttemptLimitReached,
    GuardedModelClient,
    SpendCapReached,
    SpendGuard,
    SpendGuardError,
    SpendLedgerCorrupt,
    SpendLedgerInUse,
    SpendLedgerTornTailWarning,
    SpendReservationBreach,
    classify_provider_failure,
    parse_spend_ledger,
)


@dataclass
class Response:
    prompt_tokens: int = 11
    tokens: int = 7
    content: str = "ok"
    reasoning: str = ""
    raw_text: str = "ok"
    finish_reason: str = "stop"


class Client:
    backend = "fixture"

    def __init__(
        self,
        ledger: Path,
        *,
        response: Response | None = None,
        error=None,
        provider_model_id: str = "google.gemma-4-e2b",
    ):
        self.ledger = ledger
        self.response = response or Response()
        self.error = error
        self.calls = 0
        self.config = {"model_id": provider_model_id, "max_tokens": 100}
        self._log: list[dict] = []

    def pop_call_log(self):
        rows, self._log = self._log, []
        return rows

    def call(self, *, system, messages, max_tokens, kind, **_kwargs):
        self.calls += 1
        events = [json.loads(line) for line in self.ledger.read_text().splitlines()]
        assert events[-1]["event"] == "call_reserved"
        common = {
            "kind": kind,
            "model_id": self.config["model_id"],
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if self.error is not None:
            self._log.append(
                {
                    **common,
                    "content": None,
                    "reasoning": None,
                    "raw_text": None,
                    "prompt_tokens": None,
                    "out_tokens": None,
                    "finish_reason": None,
                    "error": type(self.error).__name__,
                }
            )
            raise self.error
        self._log.append(
            {
                **common,
                "content": self.response.content,
                "reasoning": self.response.reasoning,
                "raw_text": self.response.raw_text,
                "prompt_tokens": self.response.prompt_tokens,
                "out_tokens": self.response.tokens,
                "finish_reason": self.response.finish_reason,
            }
        )
        return self.response


def identity(number: int = 1) -> dict:
    return {
        "eligible_position": number,
        "paper_statement_hash": str(1000 + number),
        "source_hash": str(2000 + number),
        "evidence_json_sha256": f"{number:064x}",
    }


def guard(
    tmp_path: Path,
    *,
    cap: str = "1",
    stage_cap: str | None = None,
    max_attempts: int = 2,
    model: str = "bedrock-gemma-4-e2b",
    stage: str = "e2b",
    workload: str = "paper_corpus",
    run_id: str = "run",
) -> SpendGuard:
    return SpendGuard(
        tmp_path / "spend.ndjson",
        approved_cap_usd=cap,
        stage_cap_usd=stage_cap,
        model=model,
        stage=stage,
        workload=workload,
        run_id=run_id,
        provider_input_token_maximum=100,
        max_attempts=max_attempts,
    )


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def rehashed_payload(rows: list[dict]) -> bytes:
    """Rebuild a valid hash chain after a deliberate semantic mutation."""

    previous = None
    payload = bytearray()
    for sequence, original in enumerate(rows, 1):
        row = dict(original)
        row.pop("event_sha256", None)
        row["sequence"] = sequence
        row["previous_event_sha256"] = previous
        digest = hashlib.sha256(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        row["event_sha256"] = digest
        payload.extend(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
        payload.extend(b"\n")
        previous = digest
    return bytes(payload)


def test_stage_cap_amendment_is_monotonic_and_globally_bounded(tmp_path: Path):
    initial = guard(tmp_path, cap="3", stage_cap="1")
    initial.close()

    amended = guard(tmp_path, cap="3", stage_cap="2")
    assert amended.summary()["stage_cap_usd"] == "2"
    amended.close()
    rows = events(tmp_path / "spend.ndjson")
    amendment = next(row for row in rows if row["event"] == "stage_cap_amended")
    assert amendment["previous_stage_cap_usd"] == "1"
    assert amendment["stage_cap_usd"] == "2"
    parse_spend_ledger((tmp_path / "spend.ndjson").read_bytes())

    with pytest.raises(SpendGuardError, match="cannot be lowered"):
        guard(tmp_path, cap="3", stage_cap="1")
    with pytest.raises(ValueError, match="stage cap"):
        guard(
            tmp_path,
            cap="3",
            stage_cap="4",
            stage="other",
            model="other-model",
        )


def test_parser_rejects_rehashed_stage_cap_overauthorization(tmp_path: Path):
    spend = guard(tmp_path, cap="3", stage_cap="1")
    spend.close()
    rows = events(tmp_path / "spend.ndjson")
    stage = next(row for row in rows if row["event"] == "stage_cap_set")
    stage["stage_cap_usd"] = "4"
    with pytest.raises(SpendLedgerCorrupt, match="stage cap event"):
        parse_spend_ledger(rehashed_payload(rows))


def test_reservation_precedes_provider_and_measured_cost_is_reconciled(tmp_path: Path):
    spend = guard(tmp_path)
    base = Client(spend.path)
    client = GuardedModelClient(base, spend)
    with spend.attempt(identity()) as receipt:
        response = client.call(
            system="system",
            messages=[{"role": "user", "content": "question"}],
            max_tokens=100,
            kind="monolithic",
        )
        calls = client.pop_call_log()
        spend.commit_attempt_outcome({"row_status": "scored", "call_log": calls})
    assert response.content == "ok"
    assert receipt.started and receipt.model_calls_reserved == 1
    rows = events(spend.path)
    order = [row["event"] for row in rows]
    assert order.index("call_reserved") < order.index("call_evidence_observed")
    assert order.index("call_evidence_observed") < order.index("call_settled")
    assert order.index("call_settled") < order.index("attempt_outcome_committed")
    settlement = next(row for row in rows if row["event"] == "call_settled")
    assert settlement["accounting_basis"] == "provider_reported_usage"
    assert settlement["provider_usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert settlement["settled_cost_usd"] == "0.000001"
    assert spend.resume_reconciliation()["attempts"][0]["finish"]["status"] == "completed"
    spend.close()


def test_global_and_stage_caps_stop_before_attempt_or_provider(tmp_path: Path):
    spend = guard(tmp_path, cap="1", stage_cap="0.00001")
    base = Client(spend.path)
    client = GuardedModelClient(base, spend)
    with spend.attempt(identity()):
        with pytest.raises(SpendCapReached, match="stage cap"):
            client.call(
                system="system",
                messages=[{"role": "user", "content": "question"}],
                max_tokens=100,
                kind="monolithic",
            )
    assert base.calls == 0
    assert not any(row["event"] == "attempt_started" for row in events(spend.path))
    spend.close()


def _one_measured_call(spend: SpendGuard, *, provider_model_id: str) -> None:
    client = GuardedModelClient(
        Client(spend.path, provider_model_id=provider_model_id), spend
    )
    with spend.attempt(identity()):
        client.call(
            system="s",
            messages=[{"role": "user", "content": "q"}],
            max_tokens=100,
            kind="monolithic",
        )
        spend.commit_attempt_outcome(
            {"row_status": "scored", "call_log": client.pop_call_log()}
        )


def test_stage_cap_accumulates_across_action_run_ids(tmp_path: Path):
    first = guard(
        tmp_path,
        stage_cap="0.000065",
        run_id="e2b_smoke",
        workload="primary",
    )
    _one_measured_call(first, provider_model_id="google.gemma-4-e2b")
    first.close()

    second = guard(
        tmp_path,
        stage_cap="0.000065",
        run_id="e2b_sensitivity",
        workload="sensitivity",
    )
    client = GuardedModelClient(Client(second.path), second)
    with second.attempt(identity(2)):
        with pytest.raises(SpendCapReached, match="stage cap"):
            client.call(
                system="s",
                messages=[{"role": "user", "content": "q"}],
                max_tokens=100,
                kind="monolithic",
            )
    assert client._client.calls == 0
    second.close()


def test_global_cap_accumulates_across_model_stages(tmp_path: Path):
    global_cap = "0.0002025"
    first = guard(
        tmp_path,
        cap=global_cap,
        stage_cap=global_cap,
        run_id="e2b",
        workload="primary",
    )
    _one_measured_call(first, provider_model_id="google.gemma-4-e2b")
    first.close()

    second = guard(
        tmp_path,
        cap=global_cap,
        stage_cap=global_cap,
        model="bedrock-gemma-4-31b",
        stage="gemma31",
        run_id="gemma31",
        workload="primary",
    )
    client = GuardedModelClient(
        Client(second.path, provider_model_id="google.gemma-4-31b"), second
    )
    with second.attempt(identity(2)):
        with pytest.raises(SpendCapReached, match="global cap"):
            client.call(
                system="s",
                messages=[{"role": "user", "content": "q"}],
                max_tokens=100,
                kind="monolithic",
            )
    assert client._client.calls == 0
    second.close()


def test_cached_commitments_cover_global_stage_and_run_after_reload(tmp_path: Path):
    first = guard(
        tmp_path,
        run_id="first",
        workload="primary",
    )
    _one_measured_call(first, provider_model_id="google.gemma-4-e2b")
    first.close()

    second = guard(
        tmp_path,
        run_id="second",
        workload="sensitivity",
    )
    _one_measured_call(second, provider_model_id="google.gemma-4-e2b")
    second.close()

    third = guard(
        tmp_path,
        model="bedrock-gemma-4-31b",
        stage="gemma31",
        run_id="third",
        workload="primary",
    )
    _one_measured_call(third, provider_model_id="google.gemma-4-31b")
    third.close()

    reloaded = guard(
        tmp_path,
        run_id="audit",
        workload="audit",
    )
    assert reloaded.commitment() == Decimal("0.00000634")
    assert reloaded.commitment(
        stage=("e2b", "bedrock-gemma-4-e2b")
    ) == Decimal("0.000002")
    assert reloaded.commitment(
        stage=("gemma31", "bedrock-gemma-4-31b")
    ) == Decimal("0.00000434")
    assert reloaded.commitment(run_id="first") == Decimal("0.000001")
    assert reloaded.commitment(run_id="second") == Decimal("0.000001")
    assert reloaded.commitment(run_id="third") == Decimal("0.00000434")

    # Queries use only caches; the independent full replay check ran at open.
    reloaded._recomputed_commitments = lambda: (_ for _ in ()).throw(
        AssertionError("commitment query rescanned the ledger")
    )
    assert reloaded.commitment(run_id="third") == Decimal("0.00000434")
    reloaded.close()


def test_concurrent_reservations_use_one_atomic_cached_cap_check(tmp_path: Path):
    spend = guard(tmp_path, cap="0.00008", stage_cap="0.00008")
    reserved = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def hold_first_reservation() -> None:
        try:
            with spend.attempt(identity(1)):
                spend.reserve_call(
                    provider_model_id="google.gemma-4-e2b",
                    kind="monolithic",
                    max_output_tokens=100,
                    system="s",
                    messages=[{"role": "user", "content": "q"}],
                )
                reserved.set()
                assert release.wait(timeout=5)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=hold_first_reservation)
    worker.start()
    assert reserved.wait(timeout=5)
    with spend.attempt(identity(2)):
        with pytest.raises(SpendCapReached, match="global cap"):
            spend.reserve_call(
                provider_model_id="google.gemma-4-e2b",
                kind="monolithic",
                max_output_tokens=100,
                system="s",
                messages=[{"role": "user", "content": "q"}],
            )
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert failures == []
    assert len([row for row in events(spend.path) if row["event"] == "call_reserved"]) == 1
    assert spend.commitment() == Decimal("0.00006456")
    spend.close()


def test_attempt_limit_is_contiguous_and_fail_closed(tmp_path: Path):
    spend = guard(tmp_path, max_attempts=2)
    for _ in range(2):
        with spend.attempt(identity()):
            receipt = spend.ensure_attempt_started()
            spend.commit_attempt_outcome(
                {
                    "row_status": "error",
                    "attempt_ordinal": receipt.attempt_ordinal,
                    "call_log": [],
                }
            )
    with spend.attempt(identity()):
        with pytest.raises(AttemptLimitReached):
            spend.ensure_attempt_started()
    starts = [row for row in events(spend.path) if row["event"] == "attempt_started"]
    assert [row["attempt_ordinal"] for row in starts] == [1, 2]
    spend.close()


def test_lane_lock_is_exclusive_and_close_releases_it(tmp_path: Path):
    first = guard(tmp_path)
    with pytest.raises(SpendLedgerInUse):
        guard(tmp_path)
    first.close()
    replacement = guard(tmp_path)
    replacement.close()


def test_hash_chain_detects_mutation(tmp_path: Path):
    spend = guard(tmp_path)
    path = spend.path
    spend.close()
    rows = path.read_text().splitlines()
    rows[0] = rows[0].replace('"global_cap_usd":"1"', '"global_cap_usd":"2"')
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(SpendLedgerCorrupt, match="hash chain"):
        guard(tmp_path)


def test_read_only_parser_is_canonical_and_requires_current_run_identity(
    tmp_path: Path,
):
    spend = guard(tmp_path)
    with spend.attempt(identity()):
        spend.commit_attempt_outcome({"row_status": "error", "call_log": []})
    path = spend.path
    spend.close()

    snapshot = parse_spend_ledger(path.read_bytes())
    assert snapshot.sequence == len(events(path))
    assert snapshot.last_event_sha256 == snapshot.events[-1]["event_sha256"]
    with pytest.raises(TypeError):
        snapshot.events[-1]["status"] = "completed"  # type: ignore[index]

    rows = events(path)
    finish = rows[-1]
    assert finish["event"] == "attempt_finished"
    finish.pop("run_id")
    finish.pop("event_sha256")
    finish["event_sha256"] = hashlib.sha256(
        json.dumps(
            finish,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    with pytest.raises(SpendLedgerCorrupt, match="run identity"):
        parse_spend_ledger(payload)


def test_read_only_parser_recomputes_reservation_and_measured_cost(tmp_path: Path):
    spend = guard(tmp_path)
    _one_measured_call(spend, provider_model_id="google.gemma-4-e2b")
    path = spend.path
    spend.close()

    reservation_rows = events(path)
    reservation = next(
        row for row in reservation_rows if row["event"] == "call_reserved"
    )
    reservation["reserved_max_cost_usd"] = "0.123"
    with pytest.raises(SpendLedgerCorrupt, match="reservation pricing arithmetic"):
        parse_spend_ledger(rehashed_payload(reservation_rows))

    settlement_rows = events(path)
    settlement = next(
        row for row in settlement_rows if row["event"] == "call_settled"
    )
    settlement["settled_cost_usd"] = "0.123"
    with pytest.raises(SpendLedgerCorrupt, match="settlement pricing arithmetic"):
        parse_spend_ledger(rehashed_payload(settlement_rows))


def test_interrupted_reservation_is_conservatively_settled_on_resume(tmp_path: Path):
    first = guard(tmp_path)
    context = first.attempt(identity())
    receipt = context.__enter__()
    reservation = first.reserve_call(
        provider_model_id="google.gemma-4-e2b",
        kind="monolithic",
        max_output_tokens=100,
        system="system",
        messages=[{"role": "user", "content": "question"}],
    )
    assert receipt.started
    first.close()  # simulate process loss after the durable reservation

    resumed = guard(tmp_path)
    attempt = resumed.resume_reconciliation()["attempts"][0]
    assert attempt["calls"][0]["evidence"]["evidence_kind"] == "interrupted_provider_call"
    settlement = attempt["calls"][0]["settlement"]
    assert settlement["accounting_basis"] == "conservative_reserved_maximum"
    assert settlement["settled_cost_usd"] == str(reservation.reserved_max_cost_usd)
    call_log = [attempt["calls"][0]["evidence"]["call_evidence"]]
    resumed.commit_deferred_attempt_outcome(
        receipt.execution_id,
        1,
        {"row_status": "error", "call_log": call_log},
    )
    resumed.finish_deferred_attempt(receipt.execution_id, 1)
    assert resumed.summary()["conservative_call_count"] == 1
    resumed.close()
    del context


def test_transport_failure_is_evidenced_and_retry_classified(tmp_path: Path):
    spend = guard(tmp_path)
    error = TimeoutError("provider timeout")
    client = GuardedModelClient(Client(spend.path, error=error), spend)
    with spend.attempt(identity()):
        with pytest.raises(TimeoutError):
            client.call(
                system="system",
                messages=[{"role": "user", "content": "question"}],
                max_tokens=100,
                kind="monolithic",
            )
        calls = client.pop_call_log()
        spend.commit_attempt_outcome({"row_status": "error", "call_log": calls})
    assert calls[0]["provider_failure_class"] == "transport_or_server"
    assert classify_provider_failure(error) == ("transport_or_server", None)
    settlement = next(row for row in events(spend.path) if row["event"] == "call_settled")
    assert settlement["accounting_basis"] == "conservative_reserved_maximum"
    spend.close()


def test_usage_above_reservation_is_recorded_then_raises(tmp_path: Path):
    spend = guard(tmp_path)
    response = Response(prompt_tokens=100_000, tokens=100_000)
    client = GuardedModelClient(Client(spend.path, response=response), spend)
    with spend.attempt(identity()):
        with pytest.raises(SpendReservationBreach):
            client.call(
                system="system",
                messages=[{"role": "user", "content": "question"}],
                max_tokens=100,
                kind="monolithic",
            )
        calls = client.pop_call_log()
        spend.commit_attempt_outcome({"row_status": "error", "call_log": calls})
    settlement = next(row for row in events(spend.path) if row["event"] == "call_settled")
    assert settlement["accounting_basis"] == "provider_reported_usage"
    assert settlement["reservation_breached"] is True
    assert Decimal(settlement["settled_cost_usd"]) > 0
    spend.close()


def test_missing_immediate_call_log_fails_closed_and_charges_reservation(tmp_path: Path):
    spend = guard(tmp_path)
    base = Client(spend.path)
    base.pop_call_log = lambda: []
    client = GuardedModelClient(base, spend)
    with spend.attempt(identity()):
        with pytest.raises(SpendGuardError, match="one provider call"):
            client.call(
                system="system",
                messages=[{"role": "user", "content": "question"}],
                max_tokens=100,
                kind="monolithic",
            )
        calls = client.pop_call_log()
        spend.commit_attempt_outcome({"row_status": "error", "call_log": calls})
    assert calls[0]["provider_call_outcome"] == "call_log_contract_breach"
    assert spend.summary()["conservative_call_count"] == 1
    spend.close()


def test_output_overshoot_within_allowance_settles_without_breach(tmp_path: Path):
    spend = guard(tmp_path)
    response = Response(prompt_tokens=100, tokens=103)
    client = GuardedModelClient(Client(spend.path, response=response), spend)
    with spend.attempt(identity()):
        client.call(
            system="system",
            messages=[{"role": "user", "content": "question"}],
            max_tokens=100,
            kind="monolithic",
        )
        calls = client.pop_call_log()
        spend.commit_attempt_outcome(
            {"row_status": "scored", "verdict": "correct", "call_log": calls}
        )
    settlement = next(
        row for row in events(spend.path) if row["event"] == "call_settled"
    )
    assert settlement["accounting_basis"] == "provider_reported_usage"
    assert settlement["reservation_breached"] is False
    reservation = next(
        row for row in events(spend.path) if row["event"] == "call_reserved"
    )
    assert reservation["reserved_output_tokens"] == 100 + PROVIDER_OUTPUT_TOKEN_OVERSHOOT
    spend.close()


def _settled_ledger(tmp_path: Path) -> Path:
    """One fully settled ledger, built through the guard itself."""

    spend = guard(tmp_path)
    _one_measured_call(spend, provider_model_id="google.gemma-4-e2b")
    path = spend.path
    spend.close()
    return path


def _quarantines(ledger: Path) -> list[Path]:
    """Every torn-tail quarantine sibling of one lane."""

    return sorted(ledger.parent.glob(ledger.name + ".torn-*"))


def _torn_ledger(tmp_path: Path, fragment: bytes) -> tuple[Path, bytes]:
    """One settled ledger plus a crash-torn trailing partial event."""

    path = _settled_ledger(tmp_path)
    complete = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(fragment)
    return path, complete


def test_streaming_load_and_bytes_parser_agree_on_chain_and_replay(tmp_path: Path):
    path = _settled_ledger(tmp_path)
    payload = path.read_bytes()
    snapshot = parse_spend_ledger(payload)

    reopened = guard(tmp_path)
    # A settled ledger reopened with the same authorization appends nothing, so
    # the streamed state is directly comparable to the whole-file parse.
    assert path.read_bytes() == payload
    assert reopened._sequence == snapshot.sequence
    assert reopened._last_digest == snapshot.last_event_sha256
    assert reopened.ledger_id == snapshot.ledger_id
    assert reopened.commitment() == snapshot._state._global_commitment
    assert reopened.recovered_torn_tail_bytes == 0
    # An intact ledger takes no repair path at all: nothing reported, nothing
    # quarantined, no side effect on disk.
    assert reopened.summary()["recovered_torn_tail_bytes"] == 0
    assert reopened.recovered_torn_tail_path is None
    assert _quarantines(path) == []
    # The recovery counter is APPENDED to the manifest contract, so no existing
    # key moves position.
    keys = list(reopened.summary())
    assert keys[-1] == "recovered_torn_tail_bytes"
    assert keys.count("recovered_torn_tail_bytes") == 1
    reopened.close()


def test_torn_trailing_event_is_recovered_and_surfaced(tmp_path: Path):
    path = _settled_ledger(tmp_path)
    complete = path.read_bytes()
    snapshot = parse_spend_ledger(complete)
    fragment = b'{"call_id":"partial","event":"call_reser'
    with path.open("ab") as handle:
        handle.write(fragment)

    # The audit path stays strict: torn bytes are not a parseable artifact.
    with pytest.raises(SpendLedgerCorrupt, match="partial event"):
        parse_spend_ledger(complete + fragment)

    with pytest.warns(SpendLedgerTornTailWarning, match="torn"):
        resumed = guard(tmp_path)
    assert resumed.recovered_torn_tail_bytes == len(fragment)
    assert resumed._sequence == snapshot.sequence
    assert resumed._last_digest == snapshot.last_event_sha256
    # Only never-committed trailing bytes were dropped.
    assert path.read_bytes() == complete

    client = GuardedModelClient(Client(path), resumed)
    with resumed.attempt(identity(2)):
        client.call(
            system="s",
            messages=[{"role": "user", "content": "q"}],
            max_tokens=100,
            kind="monolithic",
        )
        resumed.commit_attempt_outcome(
            {"row_status": "scored", "call_log": client.pop_call_log()}
        )
    resumed.close()
    extended = path.read_bytes()
    assert extended.startswith(complete) and extended.endswith(b"\n")
    assert parse_spend_ledger(extended).sequence > snapshot.sequence


def test_oversized_torn_tail_fails_closed_without_mutating(tmp_path: Path, monkeypatch):
    # A tail too large to be one torn append is not a repair case: recovery is
    # bounded by plausibility, not by the read buffer, so it fails closed exactly
    # as it did before torn-tail recovery existed.
    monkeypatch.setattr(
        spend_guard_module, "_MAX_RECOVERABLE_TORN_TAIL_BYTES", 4096
    )
    tail = b"x" * 8192
    path, complete = _torn_ledger(tmp_path, tail)

    with pytest.raises(SpendLedgerCorrupt, match="partial event"):
        guard(tmp_path)

    assert path.read_bytes() == complete + tail
    assert _quarantines(path) == []


def test_torn_tail_is_quarantined_before_truncation(tmp_path: Path):
    fragment = b'{"call_id":"partial","event":"call_reser'
    path, complete = _torn_ledger(tmp_path, fragment)

    with pytest.warns(SpendLedgerTornTailWarning, match="torn"):
        resumed = guard(tmp_path)

    quarantined = _quarantines(path)
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == fragment
    assert path.read_bytes() == complete
    assert resumed.recovered_torn_tail_path == quarantined[0]
    resumed.close()


def test_quarantine_never_clobbers_an_existing_file(tmp_path: Path):
    fragment = b'{"call_id":"partial","event":"call_reser'
    path, complete = _torn_ledger(tmp_path, fragment)
    taken = path.with_suffix(
        path.suffix + f".torn-{len(complete)}-{len(fragment)}"
    )
    sentinel = b"an earlier quarantine that must survive"
    taken.write_bytes(sentinel)

    with pytest.warns(SpendLedgerTornTailWarning, match="torn"):
        resumed = guard(tmp_path)

    assert taken.read_bytes() == sentinel
    quarantined = [item for item in _quarantines(path) if item != taken]
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == fragment
    assert resumed.recovered_torn_tail_path == quarantined[0]
    assert path.read_bytes() == complete
    resumed.close()


def test_torn_tail_warning_precedes_any_mutation(tmp_path: Path):
    # Under `-W error` the warning becomes the raise, so the abort must happen
    # with the lane still torn rather than already truncated.
    fragment = b'{"call_id":"partial","event":"call_reser'
    path, complete = _torn_ledger(tmp_path, fragment)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SpendLedgerTornTailWarning):
            guard(tmp_path)

    assert path.read_bytes() == complete + fragment
    assert _quarantines(path) == []


def test_recovered_torn_tail_bytes_reaches_the_action_manifest(tmp_path: Path):
    from indra_belief.comparison.contracts import Action
    from indra_belief.comparison.runner import RunSummary, _ActionGuard

    fragment = b'{"call_id":"partial","event":"call_reser'
    path, _complete = _torn_ledger(tmp_path, fragment)
    with pytest.warns(SpendLedgerTornTailWarning, match="torn"):
        resumed = guard(tmp_path)
    action = Action(
        id="action",
        stage_id="e2b",
        run_id="run",
        workload="paper_corpus",
        ledger=path,
        output=tmp_path / "rows.jsonl",
        cap_usd=Decimal("1"),
        deadline_seconds=60,
        max_attempts=2,
        provider_input_token_maximum=100,
        main_max_output_tokens=100,
        retry_backoff_seconds=0.0,
        workers=1,
        depends_on=(),
    )
    record = RunSummary(
        status="complete",
        action_id=action.id,
        completed_this_run=0,
        completed_total=0,
        total=0,
        verdicts={},
        spend_guard=_ActionGuard(resumed, action).summary(),
    ).as_dict()
    resumed.close()

    assert record["spend_guard"]["recovered_torn_tail_bytes"] == len(fragment)
    assert json.dumps(record)


def test_mid_file_corruption_still_fails_closed(tmp_path: Path):
    path = _settled_ledger(tmp_path)
    lines = path.read_bytes().split(b"\n")
    lines[2] = b'{"not":"canonical",'
    path.write_bytes(b"\n".join(lines))
    with pytest.raises(SpendLedgerCorrupt, match="invalid ledger event at line 3"):
        guard(tmp_path)


def test_broken_hash_chain_still_fails_closed(tmp_path: Path):
    path = _settled_ledger(tmp_path)
    lines = path.read_bytes().split(b"\n")
    del lines[2]
    path.write_bytes(b"\n".join(lines))
    with pytest.raises(SpendLedgerCorrupt, match="hash chain fails at line 3"):
        guard(tmp_path)


def test_non_utf8_ledger_bytes_still_fail_closed(tmp_path: Path):
    path = _settled_ledger(tmp_path)
    with path.open("ab") as handle:
        handle.write(b"\xff\xfe\n")
    with pytest.raises(SpendLedgerCorrupt, match="not UTF-8"):
        guard(tmp_path)


def test_partial_line_only_ledger_still_fails_closed(tmp_path: Path):
    (tmp_path / "spend.ndjson").write_bytes(b'{"event":"ledger_initialized"')
    with pytest.raises(SpendLedgerCorrupt, match="partial event"):
        guard(tmp_path)


def test_newlineless_blob_is_bounded_instead_of_buffered(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(spend_guard_module, "_MAX_LEDGER_LINE_BYTES", 4096)
    (tmp_path / "spend.ndjson").write_bytes(b"x" * (2 * 1024 * 1024))
    with pytest.raises(SpendLedgerCorrupt, match="exceeds the readable maximum"):
        guard(tmp_path)


def test_output_overshoot_beyond_allowance_still_breaches(tmp_path: Path):
    spend = guard(tmp_path)
    response = Response(
        prompt_tokens=100, tokens=100 + PROVIDER_OUTPUT_TOKEN_OVERSHOOT + 1
    )
    client = GuardedModelClient(Client(spend.path, response=response), spend)
    with spend.attempt(identity()):
        with pytest.raises(SpendReservationBreach):
            client.call(
                system="system",
                messages=[{"role": "user", "content": "question"}],
                max_tokens=100,
                kind="monolithic",
            )
        calls = client.pop_call_log()
        spend.commit_attempt_outcome({"row_status": "error", "call_log": calls})
    settlement = next(
        row for row in events(spend.path) if row["event"] == "call_settled"
    )
    assert settlement["reservation_breached"] is True
    spend.close()
