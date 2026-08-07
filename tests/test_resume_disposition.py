"""The settled boundary: what a durable row is allowed to decide about its source.

`replay._settled_reason_from_class` used to collapse every unrecognised failure
into the single reason `nonretryable_failure_on_resume`. An auth 401, a config
error, a parser-profile mismatch and an exception nobody has classified all
arrived as that one string, so a quarantine record could not say which of them
retired a source — and the auth case was settled PERMANENTLY, although the token
is re-read at restart.

Three separable claims are tested here, and each one fails independently:

  (a) the disposition CARRIES the error type, so the four cases are
      distinguishable at the quarantine site;
  (b) the resume classifier recognises what the LIVE classifier retries. The two
      decide by different means — `isinstance` live, a name here — and the names
      are now derived from the classes so they cannot drift;
  (c) a recorded auth status is NOT settled, and is still bounded by
      `max_attempts`.

The corpus-scale claim behind (b) — that no row already on disk changes class —
is not asserted here because it needs the 8.5 GiB of gitignored attempt logs;
`tests/test_replay_parser_diff.py` is the pattern for that, and the measurement
is recorded in the commit that introduced this file.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

from indra_belief.comparison.replay import (
    _CREDENTIAL_STATUSES,
    _RETRYABLE_ERROR_TYPES,
    RowDisposition,
    _settled_reason_from_class,
    row_disposition,
    row_retry_class,
)
from indra_belief.spend_guard import classify_provider_failure


def _error_row(error_type: str, *, status: int | None = None,
               calls: list | None = None) -> dict:
    """A row shaped like `replay.error_row`'s output, minus the identity half."""
    error = {"type": error_type, "message_sha256": "0" * 64}
    if status is not None:
        error["provider_http_status"] = status
    return {"row_status": "error", "call_log": calls or [], "error": error}


# ---------------------------------------------------------------------------
# (a) the type is carried, not collapsed
# ---------------------------------------------------------------------------

def test_the_disposition_carries_the_error_type():
    """Four failures that all settle for the SAME reason, told apart by type.

    This is the whole defect in one assertion: the reason is identical for every
    row, so a reason alone can never distinguish them. Before the type rode
    along, the quarantine record held only the left-hand column.
    """
    types = ["AuthError", "ContractError", "ReplayError", "SomethingNobodyTyped"]
    dispositions = [row_disposition(_error_row(name)) for name in types]

    reasons = {
        _settled_reason_from_class(item, attempts=1, invalid_outputs=0, max_attempts=5)
        for item in dispositions
    }
    assert reasons == {"nonretryable_failure_on_resume"}
    assert [item.error_type for item in dispositions] == types


def test_a_row_with_no_error_object_carries_no_type():
    """`error_type` is None rather than a placeholder — absence stays absence."""
    assert row_disposition({"row_status": "scored", "verdict": "correct"}) == (
        RowDisposition(None, None)
    )


# ---------------------------------------------------------------------------
# (b) the two classifiers agree, by construction rather than by maintenance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", [
    "indra_belief.bedrock_chat_transport",
    "indra_belief.bedrock_responses_transport",
])
def test_every_transport_exception_the_live_path_retries_is_named_here(module_name):
    """The drift this closes, stated as the property rather than as a list.

    `classify_provider_failure` retries on `isinstance(error, (TimeoutError,
    ConnectionError))`. The resume path holds only `error.type`, a string. Any
    transport exception satisfying the live predicate must therefore appear in
    `_RETRYABLE_ERROR_TYPES` — and `BedrockChatConnectionError`, which IS a
    `ConnectionError`, did not, for as long as that set was typed out by hand.

    A new transport exception subclassing `ConnectionError` fails this test in
    the commit that defines it unless it is derived, which is the point.
    """
    module = importlib.import_module(module_name)
    live_retryable = [
        value.__name__
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, BaseException)
        and classify_provider_failure(_instance(value))[0] == "transport_or_server"
    ]
    assert live_retryable, f"{module_name} defines no retryable transport exception"
    missing = sorted(set(live_retryable) - _RETRYABLE_ERROR_TYPES)
    assert missing == [], f"live retries {missing} but resume would not"


def _instance(cls: type) -> BaseException:
    """A real instance, so the LIVE classifier is driven rather than mirrored.

    Asserting `issubclass(cls, (TimeoutError, ConnectionError))` here would only
    restate the derivation in `replay._provider_error_names`; the claim worth
    testing is that `classify_provider_failure` — the function the runner
    actually calls — agrees. Two of these exceptions take a required keyword,
    so the constructor is satisfied generically rather than per class, which is
    what keeps a newly added exception in scope instead of skipped.
    """
    try:
        return cls("probe")
    except TypeError:
        required = {
            name: {}
            for name, parameter in inspect.signature(cls).parameters.items()
            if parameter.kind is parameter.KEYWORD_ONLY
            and parameter.default is parameter.empty
        }
        return cls("probe", **required)


def test_a_connection_error_is_retryable_without_any_call_evidence():
    """The gap the derived set closes, driven rather than described.

    A transport failure that happens BEFORE a provider call is reserved has an
    empty `call_log`, so the recorded-class branch cannot save it and the name
    is all there is. Measured on the shipped corpus: every one of the 1,653
    error rows carries call evidence, so this path has never fired there — which
    is why nothing caught it.
    """
    assert row_retry_class(_error_row("BedrockChatConnectionError")) == "transport_or_server"
    assert row_retry_class(_error_row("BedrockResponsesConnectionError")) == "transport_or_server"


def test_the_recorded_live_class_still_wins_over_the_name():
    """Call evidence is the live decision itself and outranks a name lookup."""
    row = _error_row(
        "SomethingNobodyTyped",
        calls=[{"provider_failure_class": "transport_or_server"}],
    )
    assert row_retry_class(row) == "transport_or_server"


# ---------------------------------------------------------------------------
# (c) a credential failure is not permanent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", sorted(_CREDENTIAL_STATUSES))
def test_a_credential_failure_is_not_settled_and_is_still_bounded(status):
    """Both halves, because either alone would be wrong.

    NOT settled: `_run_prepared` re-reads the token, so the row says only that
    the token at that moment was rejected — not that another attempt is
    impossible. Settling on it turns a credential blip into a permanent hole,
    and any hole is terminal.

    STILL BOUNDED: nothing here exempts the source from `max_attempts`, so a
    genuinely dead credential cannot loop.
    """
    disposition = row_disposition(_error_row("BedrockChatTransportError", status=status))
    assert disposition.retry_class == "credential"

    assert _settled_reason_from_class(
        disposition, attempts=1, invalid_outputs=0, max_attempts=5) is None
    assert _settled_reason_from_class(
        disposition, attempts=5, invalid_outputs=0, max_attempts=5) == "attempts_exhausted"


def test_a_bad_request_status_is_not_a_credential_failure():
    """400/404/422 are statements about the REQUEST and stay settled.

    The separator is not "did it have a status" but "would a restart change the
    answer". A malformed request is malformed again next time.
    """
    for status in (400, 404, 422, 500):
        disposition = row_disposition(_error_row("BedrockChatTransportError", status=status))
        assert disposition.retry_class != "credential", status


def test_an_unrecorded_status_cannot_be_read_as_a_credential_failure():
    """Rows written before the status was recorded are two-key and stay settled.

    They are append-only durable evidence and are never rewritten, so the whole
    shipped corpus keeps the classification it already had. A change here would
    silently re-open sources on artifacts that back published numbers.
    """
    disposition = row_disposition(_error_row("BedrockChatTransportError"))
    assert disposition.retry_class is None
    assert _settled_reason_from_class(
        disposition, attempts=1, invalid_outputs=0, max_attempts=5
    ) == "nonretryable_failure_on_resume"
