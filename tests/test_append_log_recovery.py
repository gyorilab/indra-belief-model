"""Crash-safe resume: a torn trailing record, and the three things that are not one.

`research/kernel_unification_findings.md` §7.2 item 7 records that crash-safe
resume is a stated project invariant this codebase did not satisfy. A partial
trailing line made `load_resume` raise with no recovery path, so an arm was
unresumable until a human truncated the file by hand.

Why a trailing partial is recoverable AT ALL: the log is append-only and every
record is one unbuffered write followed by `fsync`, so a torn record can only
ever be the last one. `SIGKILL` cannot tear it — the write is a single syscall
— but power loss between the page cache landing and the fsync can, and so can
the short-append branch of `append`, which raises AFTER its bytes are on disk.

Why the reader still refuses: truncating a ledger-backed log is a mutation of
paid evidence. Only the process about to append, holding `LOCK_EX` on a pinned
inode, may do it. The fifteen published logs are read by tools holding no lock,
and a reader that repaired them would rewrite published evidence.

Most of this file is the REFUSALS, because "the tail has no newline" is
consistent with faults that truncation would destroy the evidence of.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from indra_belief.comparison.replay import (
    _TORN_TAIL_LIMIT,
    AppendLog,
    ReplayError,
)


def _row(index: int) -> bytes:
    """A canonical JSONL record, byte-identical in shape to a real attempt row."""
    return json.dumps({"stmt_i": index, "evidence_i": 0},
                      sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _log(tmp_path, payload: bytes):
    path = tmp_path / "attempts.jsonl"
    path.write_bytes(payload)
    return path


def test_a_torn_trailing_record_is_discarded_and_the_discard_is_recorded(tmp_path):
    """The invariant itself: the arm resumes, and says what it lost.

    A truncation that leaves no trace is indistinguishable from a run that never
    had a torn tail, and those are different histories — the first lost a paid
    attempt to a crash. The digest is of the discarded bytes, so an operator
    holding a copy can recognise them.
    """
    fragment = b'{"stmt_i":2,"evide'
    path = _log(tmp_path, _row(0) + _row(1) + fragment)

    log = AppendLog.open(path, recover=True)
    try:
        assert log.recovered is not None
        assert log.recovered.bytes_discarded == len(fragment)
        assert log.recovered.sha256 == hashlib.sha256(fragment).hexdigest()
        assert log.recovered.size_after == len(_row(0) + _row(1))
        assert "discarded a 18-byte unterminated trailing record" in (
            log.recovered.describe()
        )
    finally:
        log.close()

    assert path.read_bytes() == _row(0) + _row(1)


def test_recovery_is_opt_in_so_a_reader_never_repairs(tmp_path):
    """The default leaves the bytes alone, because most openers are readers."""
    payload = _row(0) + b'{"stmt_i":1'
    path = _log(tmp_path, payload)

    log = AppendLog.open(path)
    try:
        assert log.recovered is None
    finally:
        log.close()
    assert path.read_bytes() == payload


def test_a_whole_file_is_left_exactly_alone(tmp_path):
    """No newline to hunt for means nothing to do — and nothing is done."""
    payload = _row(0) + _row(1)
    path = _log(tmp_path, payload)

    log = AppendLog.open(path, recover=True)
    try:
        assert log.recovered is None
    finally:
        log.close()
    assert path.read_bytes() == payload


def test_an_empty_file_is_not_a_torn_tail(tmp_path):
    path = _log(tmp_path, b"")
    log = AppendLog.open(path, recover=True)
    try:
        assert log.recovered is None
    finally:
        log.close()


# ---------------------------------------------------------------------------
# The three refusals
# ---------------------------------------------------------------------------

def test_a_complete_row_missing_only_its_newline_is_refused(tmp_path):
    """A LOST TERMINATOR is not a lost record, and the difference is a paid attempt.

    Truncating here would discard an intact row — a provider call that was made
    and billed. The shapes are indistinguishable by "does the file end in a
    newline", which is exactly why the check is on whether the fragment parses
    as a complete canonical row.
    """
    intact = _row(2)[:-1]  # the whole row, terminator removed
    path = _log(tmp_path, _row(0) + intact)

    with pytest.raises(ReplayError) as caught:
        AppendLog.open(path, recover=True)
    assert "lost terminator, not a lost record" in str(caught.value)
    assert path.read_bytes() == _row(0) + intact


def test_a_file_with_no_record_boundary_at_all_is_refused(tmp_path):
    """Complete records THEN a fragment is the recoverable shape. This is not it.

    A file whose entire content is unframed has no verified prefix to keep, so
    there is nothing to truncate BACK to — only a whole file to delete, which is
    a decision no automatic path should make about paid evidence.
    """
    path = _log(tmp_path, b'{"stmt_i":0,"evi')

    with pytest.raises(ReplayError) as caught:
        AppendLog.open(path, recover=True)
    assert "no record boundary" in str(caught.value)
    assert path.read_bytes() == b'{"stmt_i":0,"evi'


def test_an_oversized_tail_is_refused_rather_than_truncated(tmp_path):
    """Megabytes of unterminated tail is not one torn append.

    One record is a bounded JSON object. A fragment past the bound means
    something other than an interrupted `append` wrote here, and discarding it
    would destroy the evidence of whatever that was.
    """
    for oversized, expected in (
        # Just past the bound: the preceding newline is still inside the read
        # window, so the refusal can quote the exact size.
        (b"x" * (_TORN_TAIL_LIMIT + 1), f"is {_TORN_TAIL_LIMIT + 1} bytes, over the"),
        # Far past it: the boundary is outside the window, so the refusal states
        # the bound rather than a size it cannot see. Reading a smaller window
        # made THIS case report "no record boundary" — a wrong diagnosis of a
        # file that has one — which is what the window size now prevents.
        (b"x" * (_TORN_TAIL_LIMIT * 2), "exceeds the"),
    ):
        path = _log(tmp_path, _row(0) + oversized)
        with pytest.raises(ReplayError) as caught:
            AppendLog.open(path, recover=True)
        assert expected in str(caught.value), len(oversized)
        assert path.read_bytes() == _row(0) + oversized


def test_a_refusal_does_not_leak_the_descriptor_or_hold_the_lock(tmp_path):
    """A failed recovery must not leave the log locked against the next attempt.

    `open` acquires LOCK_EX before recovering, so a recovery that raises has to
    close the stream on the way out or every subsequent open reports "already in
    use" — a fault that would look exactly like a second runner being live.
    """
    path = _log(tmp_path, b'{"stmt_i":0,"evi')

    for _ in range(2):
        with pytest.raises(ReplayError) as caught:
            AppendLog.open(path, recover=True)
        assert "no record boundary" in str(caught.value)

    # And the file is still openable for reading afterwards.
    log = AppendLog.open(path)
    try:
        assert log.recovered is None
    finally:
        log.close()
