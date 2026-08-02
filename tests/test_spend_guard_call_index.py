"""The attempt -> reservations side index must be a pure accelerator.

Every assertion here is differential: the index is only allowed to return what
the pre-index full scan of ``_reservations`` returned, and a forged ledger must
still raise the byte-identical :class:`SpendLedgerCorrupt` message.  The message
strings below were captured from the pre-change tree and are hardcoded on
purpose — clause ORDER decides which message a bad ledger produces, so a message
that shifts is a corruption check that moved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indra_belief.spend_guard import (
    GuardedModelClient,
    SpendGuard,
    SpendLedgerCorrupt,
    _LedgerReplay,
    parse_spend_ledger,
)

from test_spend_guard import Client, events, guard, identity, rehashed_payload

# 10**31 rendered in base ten is exactly 32 characters, so it is simultaneously a
# legal ``attempt_started`` id (``isinstance(str) and len == 32``, never hex
# checked) and the ``str()`` of a JSON number.  That is the only way a
# reservation can end up stored under a non-``str`` attempt id.
DIGITS_32 = str(10**31)

# The only value shape that could defeat a bucket keyed on ``str(attempt_id)``:
# ``==`` but with different ``str()``s, both exactly 32 characters, built out of
# ``True == 1`` with compensating repr-length deltas.  It exists as a pair of
# Python values and is UNREACHABLE as a ledger — see
# ``test_an_unhashable_attempt_id_never_reaches_the_index``.
EQUAL_ARRAYS = ([True, 1, 10**20], [1, True, 10**20])


def settled_ledger(tmp_path: Path, *, calls: int, executions: int = 1) -> list[dict]:
    """A real guard-written ledger: ``executions`` attempts x ``calls`` calls."""

    spend = guard(tmp_path, cap="1")
    client = GuardedModelClient(Client(spend.path), spend)
    for number in range(1, executions + 1):
        with spend.attempt(identity(number)):
            log: list[dict] = []
            for _ in range(calls):
                client.call(
                    system="system",
                    messages=[{"role": "user", "content": "question"}],
                    max_tokens=100,
                    kind="monolithic",
                )
                log.extend(client.pop_call_log())
            spend.commit_attempt_outcome({"row_status": "scored", "call_log": log})
    spend.close()
    return events(spend.path)


def legacy_scan(target: object, attempt_id: object) -> list[dict]:
    """The expression the three hot sites used before the index existed."""

    return [
        item
        for item in target._reservations.values()  # type: ignore[attr-defined]
        if item.get("attempt_id") == attempt_id
    ]


def indexed_call_ids(target: object) -> list[str]:
    buckets = target._calls_by_attempt  # type: ignore[attr-defined]
    overflow = target._calls_with_nonstr_attempt  # type: ignore[attr-defined]
    return sorted(
        [item["call_id"] for bucket in buckets.values() for item in bucket]
        + [item["call_id"] for item in overflow]
    )


def replay_until_corrupt(rows: list[dict]) -> tuple[int, str, _LedgerReplay]:
    """Apply rows one at a time; return where and how the replay refused."""

    replay = _LedgerReplay()
    for position, row in enumerate(rows):
        try:
            replay.apply(row)
        except SpendLedgerCorrupt as exc:
            return position, str(exc), replay
    raise AssertionError("forged ledger replayed without raising")


def renamed_attempt(rows: list[dict], new_id: object) -> list[dict]:
    out = []
    for row in rows:
        row = dict(row)
        if "attempt_id" in row:
            row["attempt_id"] = new_id
        out.append(row)
    return out


def reserved(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["event"] == "call_reserved"]


def test_call_index_matches_a_brute_force_scan(tmp_path: Path):
    base = settled_ledger(tmp_path, calls=2, executions=3)
    assert len([row for row in base if row["event"] == "attempt_started"]) >= 3
    assert len(reserved(base)) >= 6

    reopened = guard(tmp_path, cap="1")
    try:
        assert len(reopened._starts) >= 3
        for attempt_id in reopened._starts:
            assert reopened._calls_for_attempt(attempt_id) == legacy_scan(
                reopened, attempt_id
            )
            assert len(reopened._calls_for_attempt(attempt_id)) >= 2
        assert (
            sum(len(bucket) for bucket in reopened._calls_by_attempt.values())
            + len(reopened._calls_with_nonstr_attempt)
            == len(reopened._reservations)
        )
        assert indexed_call_ids(reopened) == sorted(reopened._reservations)
        # A real ledger never stores a non-str attempt id, so the overflow list
        # costs nothing on the hot path.
        assert reopened._calls_with_nonstr_attempt == []
    finally:
        reopened.close()


def test_index_is_adopted_by_reference_on_load(tmp_path: Path):
    settled_ledger(tmp_path, calls=2, executions=2)

    reopened = guard(tmp_path, cap="1")
    try:
        # Fails loudly if _load left the freshly initialized empty dict in place.
        assert reopened._calls_by_attempt != {}
        assert reopened._calls_by_attempt.keys() == reopened._starts.keys()

        with reopened.attempt(identity(9)):
            reservation = reopened.reserve_call(
                provider_model_id="google.gemma-4-e2b",
                kind="monolithic",
                max_output_tokens=100,
                system="system",
                messages=[{"role": "user", "content": "question"}],
            )
            live = reopened._calls_for_attempt(reservation.attempt_id)
            assert [item["call_id"] for item in live] == [reservation.call_id]
            assert live == legacy_scan(reopened, reservation.attempt_id)
            assert indexed_call_ids(reopened) == sorted(reopened._reservations)
    finally:
        reopened.close()


def test_index_is_not_populated_when_the_reservation_is_rejected(tmp_path: Path):
    base = settled_ledger(tmp_path, calls=2)
    cut = [row["event"] for row in base].index("call_evidence_observed")
    replay = _LedgerReplay()
    for row in base[:cut]:
        replay.apply(row)

    before_reservations = dict(replay._reservations)
    before_buckets = {key: list(value) for key, value in replay._calls_by_attempt.items()}
    before_overflow = list(replay._calls_with_nonstr_attempt)
    assert before_buckets != {}

    rejected = dict(reserved(base)[0])
    rejected["call_id"] = "e" * 32
    rejected["call_ordinal"] = 7  # not contiguous: 1 + len(calls for this attempt)
    with pytest.raises(SpendLedgerCorrupt, match="call reservation is malformed"):
        replay.apply(rejected)

    assert replay._reservations == before_reservations
    assert replay._calls_by_attempt == before_buckets
    assert replay._calls_with_nonstr_attempt == before_overflow
    assert "e" * 32 not in indexed_call_ids(replay)


def forged_ledgers(tmp_path: Path) -> list[tuple[str, list[dict], str]]:
    base = settled_ledger(tmp_path / "two", calls=2)
    cases: list[tuple[str, list[dict], str]] = []

    rows = [dict(row) for row in base]
    reserved(rows)[0]["call_ordinal"] = 2
    cases.append(("wrong_call_ordinal", rows, "call reservation is malformed"))

    rows = [dict(row) for row in base]
    reserved(rows)[1]["call_id"] = reserved(rows)[0]["call_id"]
    cases.append(("duplicate_call_id", rows, "call reservation is malformed"))

    # A JSON number whose str() is an existing 32-char attempt id: the bucket
    # lookup must not quietly "match" what `==` refuses to match.
    rows = renamed_attempt(base, DIGITS_32)
    for row in reserved(rows):
        row["attempt_id"] = int(DIGITS_32)
    cases.append(("numeric_attempt_id", rows, "provider call evidence is malformed"))

    rows = [dict(row) for row in base]
    rows.remove([row for row in rows if row["event"] == "call_settled"][-1])
    cases.append(("outcome_before_settlement", rows, "attempt outcome is malformed"))

    rows = [dict(row) for row in base]
    late = dict(reserved(rows)[0])
    late["call_id"] = "f" * 32
    rows.append(late)
    cases.append(("reserved_after_finished", rows, "call reservation is malformed"))

    rows = [dict(row) for row in base]
    reserved(rows)[1]["call_ordinal"] = 3
    cases.append(("call_ordinal_skips", rows, "call reservation is malformed"))

    return cases


def test_forged_ledgers_raise_the_identical_corruption_message(tmp_path: Path):
    cases = forged_ledgers(tmp_path)
    assert len(cases) >= 6

    for name, rows, expected in cases:
        with pytest.raises(SpendLedgerCorrupt) as raised:
            parse_spend_ledger(rehashed_payload(rows))
        assert str(raised.value) == expected, name

        position, message, replay = replay_until_corrupt(rows)
        assert message == expected, name
        assert rows[position]["event"] in {
            "call_reserved",
            "call_evidence_observed",
            "attempt_outcome_committed",
        }, name
        # The refusal came from the indexed candidate set, and the index is still
        # exactly _reservations at the moment it refused.
        assert indexed_call_ids(replay) == sorted(replay._reservations), name


def cross_type_ledger(tmp_path: Path) -> list[dict]:
    """One reservation stored under a JSON number, looked up by that number.

    ``attempt_started`` only checks ``isinstance(attempt_id, str) and len == 32``
    while ``call_reserved`` only requires that ``str(attempt_id)`` HIT a
    ``_starts`` key, so the stored reservation's ``attempt_id`` is an ``int``.
    The legacy scan matched it with ``==``, and this index must too -- here via
    the overflow list, since the buckets are keyed by the RAW ``str`` id.
    """

    base = settled_ledger(tmp_path / "one", calls=1)
    rows = [
        row
        for row in renamed_attempt(base, DIGITS_32)
        if row["event"] not in {"call_evidence_observed", "call_settled"}
    ]
    for row in rows:
        if row["event"] in {"call_reserved", "attempt_outcome_committed"}:
            row["attempt_id"] = int(DIGITS_32)
    return rows


def test_cross_type_attempt_id_still_matches_the_legacy_scan(tmp_path: Path):
    rows = cross_type_ledger(tmp_path)
    numeric = int(DIGITS_32)

    cut = [row["event"] for row in rows].index("attempt_outcome_committed")
    replay = _LedgerReplay()
    for row in rows[:cut]:
        replay.apply(row)

    stored = list(replay._reservations.values())
    assert len(stored) == 1
    assert stored[0]["attempt_id"] == numeric
    # Buckets are keyed by the RAW str id, so this one is carried by the
    # overflow list -- which every lookup unions in, unconditionally.
    assert replay._calls_by_attempt == {}
    assert replay._calls_with_nonstr_attempt == stored
    assert replay._calls_for_attempt(numeric) == legacy_scan(replay, numeric)
    assert replay._calls_for_attempt(numeric) == stored
    # ...and the index never invents a match `==` refuses: the 32-char STRING is
    # a different attempt entirely.
    assert replay._calls_for_attempt(DIGITS_32) == legacy_scan(replay, DIGITS_32)
    assert replay._calls_for_attempt(DIGITS_32) == []
    assert indexed_call_ids(replay) == sorted(replay._reservations)

    # An unsettled call under that attempt must still block its outcome.
    with pytest.raises(SpendLedgerCorrupt) as raised:
        parse_spend_ledger(rehashed_payload(rows))
    assert str(raised.value) == "attempt outcome is malformed"

    position, message, _ = replay_until_corrupt(rows)
    assert rows[position]["event"] == "attempt_outcome_committed"
    assert message == "attempt outcome is malformed"


def test_an_unhashable_attempt_id_never_reaches_the_index(tmp_path: Path):
    """Why the overflow list rests on a proof and not on a witness.

    A bucket keyed on ``str(attempt_id)`` is only safe if ``x == y`` implies
    ``str(x) == str(y)``.  Python does not give that: ``EQUAL_ARRAYS`` is a pair
    of ``==`` values with different 32-character ``str()``s.  No such ledger can
    be built, though — the fifth clause of ``call_reserved``'s or-chain,
    ``attempt_id in self._finishes``, is a dict membership test, so an
    unhashable id raises ``TypeError`` before any row is stored.  Keying on the
    RAW id sidesteps that argument entirely: it never needs the clause order to
    stay where it is.
    """

    first, second = EQUAL_ARRAYS
    assert first == second
    assert str(first) != str(second)
    assert len(str(first)) == len(str(second)) == 32

    base = settled_ledger(tmp_path, calls=2, executions=2)
    started = [row["attempt_id"] for row in base if row["event"] == "attempt_started"]
    assert len(started) == 2
    rename = {started[0]: str(first), started[1]: str(second)}
    rows = []
    for row in base:
        row = dict(row)
        if "attempt_id" in row:
            row["attempt_id"] = rename[row["attempt_id"]]
        rows.append(row)
    # Attempt one's reservations now carry the unhashable id itself.  Were they
    # ever stored, a str()-keyed bucket would file them under str(first) while
    # the legacy scan for attempt two -- whose id is `==` to theirs -- matched
    # them, and the call_ordinal contiguity check would go soft.
    for row in rows:
        if row.get("attempt_id") == str(first) and row["event"] == "call_reserved":
            row["attempt_id"] = first

    replay = _LedgerReplay()
    with pytest.raises(TypeError, match="unhashable type"):
        for row in rows:
            replay.apply(row)
    assert [item for item in replay._reservations.values() if not isinstance(item.get("attempt_id"), str)] == []
    assert replay._calls_with_nonstr_attempt == []
    assert indexed_call_ids(replay) == sorted(replay._reservations)

    with pytest.raises(TypeError, match="unhashable type"):
        parse_spend_ledger(rehashed_payload(rows))


def test_attempt_rows_are_unchanged_by_the_index(tmp_path: Path):
    settled_ledger(tmp_path, calls=2, executions=3)
    reopened = guard(tmp_path, cap="1")
    try:
        rows = SpendGuard._attempt_rows(reopened)
        assert len(rows) == 3
        for row in rows:
            ordinals = [call["reservation"]["call_ordinal"] for call in row["calls"]]
            assert ordinals == [1, 2]
            assert [call["reservation"] for call in row["calls"]] == sorted(
                legacy_scan(reopened, row["attempt_id"]),
                key=lambda item: item["call_ordinal"],
            )
    finally:
        reopened.close()
