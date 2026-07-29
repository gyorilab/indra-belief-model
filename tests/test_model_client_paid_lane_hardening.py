"""Paid-lane hardening for the two legacy Bedrock transports.

Three behavior-neutral guarantees. Everything except the two loopback tests runs
against local fakes; the loopback tests bind an ephemeral port on 127.0.0.1 and
make no egress (no network, no provider call, no billed side effect):

1. TEARDOWN — `bedrock_converse` / `bedrock_responses` run SYNCHRONOUSLY in the
   caller, like their `*_raw` siblings. A wall-timeout future cannot cancel an
   already-running HTTP request, so dispatching a billed lane through the pool
   would let a timeout return while the paid request is still live. Here we
   prove the adapter runs on the caller's thread, that `_invoke_with_wall_timeout`
   is never reached, and — from OUTSIDE the client, via a loopback server that
   watches its own connection die — that the transport is settled before the
   TimeoutError reaches the caller.

2. OUTER BOUND — `open(req, timeout=N)` caps each socket operation, not total
   wall time. Both legacy lanes therefore hold a monotonic ABSOLUTE deadline
   across the body read, proven against a loopback server that dribbles the body
   one byte at a time: `call()` raises inside `timeout + epsilon`, not after the
   full drip.

3. 429 PRECISION — the single in-client retry fires on a genuine rate limit and
   not on an unrelated failure whose body merely contains the digits "429", and
   not on an authoritative non-429 whose body merely names a quota. A
   DIFFERENTIAL battery pins the new predicate as a subset of the substring test
   it replaced: no widening, and only the two declared narrowings. Genuine-429
   retry count and `_parse_retry_delay(...) + 1` delays are unchanged.

Also pinned: the legacy-lane error object exposes NONE of the attribute names
`spend_guard._status_from_provider_exception` reads, so the spend guard's
classification of these lanes is provably identical to before.
"""
from __future__ import annotations

import io
import json
import socket
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief.model_client import (  # noqa: E402
    ModelClient,
    ModelResponse,
    _is_rate_limit_error,
    _provider_http_status,
)

CONVERSE_PAYLOAD = {
    "output": {"message": {"content": [{"text": '{"verdict":"correct"}'}]}},
    "usage": {"inputTokens": 11, "outputTokens": 7},
    "stopReason": "stop",
}
RESPONSES_PAYLOAD = {
    "status": "completed",
    "usage": {"input_tokens": 11, "output_tokens": 7},
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": '{"verdict":"correct"}'}],
        }
    ],
}


def _legacy_client(backend: str, *, timeout: int = 30) -> ModelClient:
    """A legacy-lane client with no constructor side effects (no env, no token
    setup, no network). Same `__new__` + explicit-config pattern used by
    tests/test_reasoning_trace.py's `_raw_responses_client`."""
    client = ModelClient.__new__(ModelClient)
    client.model_name = f"fixture-{backend}"
    client.backend = backend
    client.config = {
        "model_id": "fixture.model",
        "base_url": "https://bedrock-fixture.invalid/v1",
        "reasoning_in_content": False,
        "max_tokens": 128,
        "timeout": timeout,
    }
    client._tls = threading.local()
    client._bedrock_token = "fixture-token"
    return client


def _install_opener(client: ModelClient, open_fn) -> None:
    """Same fake-opener shape as tests/test_reasoning_trace.py."""

    class _Opener:
        open = staticmethod(open_fn)

    client._bedrock_url_opener = _Opener()


def _payload_opener(payload):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def _open(req, timeout=None):
        return _Resp()

    return _open


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://bedrock-fixture.invalid/v1/x",
        code,
        "fixture",
        {},  # type: ignore[arg-type]
        io.BytesIO(body.encode("utf-8")),
    )


def _poison_wall_pool(monkeypatch) -> None:
    def _boom(self, fn, timeout, *args, **kwargs):
        raise AssertionError(
            "a billed Bedrock lane was dispatched through the wall-timeout pool"
        )

    monkeypatch.setattr(ModelClient, "_invoke_with_wall_timeout", _boom)


def _stub_response() -> ModelResponse:
    return ModelResponse(
        content='{"verdict":"correct"}',
        reasoning="",
        tokens=7,
        raw_text='{"verdict":"correct"}',
        finish_reason="stop",
        prompt_tokens=11,
    )


# ── (a) neither paid lane touches the wall-timeout pool ──────────────────────

@pytest.mark.parametrize(
    "backend, adapter",
    [
        ("bedrock_converse", "_call_bedrock_converse"),
        ("bedrock_responses", "_call_bedrock_responses"),
    ],
)
def test_paid_lane_runs_on_caller_thread_not_wall_pool(backend, adapter, monkeypatch):
    _poison_wall_pool(monkeypatch)
    client = _legacy_client(backend)
    seen: dict = {}

    def _adapter(*args, **kwargs):
        seen["thread"] = threading.current_thread()
        return _stub_response()

    monkeypatch.setattr(client, adapter, _adapter)

    response = client.call("sys", [{"role": "user", "content": "hi"}])

    assert response.content == '{"verdict":"correct"}'
    # The adapter ran inline: same thread object as the caller, and the poisoned
    # pool wrapper was never reached.
    assert seen["thread"] is threading.current_thread()
    assert not seen["thread"].name.startswith("mc-wall")


# ── (b) a paid-lane timeout settles the transport before control returns ─────

@pytest.mark.parametrize(
    "backend, label",
    [
        ("bedrock_converse", "Converse"),
        ("bedrock_responses", "Responses"),
    ],
)
def test_paid_lane_socket_timeout_normalizes_to_timeouterror(backend, label):
    """A socket deadline escaping the opener surfaces as TimeoutError with the
    lane's exact message.

    Deliberately does NOT assert settlement: a fake opener could only report it
    through a `finally` this test itself writes, which no implementation change
    can falsify. Settlement is proven externally instead, by
    `test_deadline_teardown_is_observed_by_the_loopback_server`.
    """
    client = _legacy_client(backend, timeout=30)

    def _open(req, timeout=None):
        raise socket.timeout("timed out")

    _install_opener(client, _open)

    with pytest.raises(TimeoutError) as excinfo:
        client.call("sys", [{"role": "user", "content": "hi"}])

    assert str(excinfo.value) == f"Bedrock {label} request timed out after 30s"


@pytest.mark.parametrize(
    "backend, label",
    [
        ("bedrock_converse", "Converse"),
        ("bedrock_responses", "Responses"),
    ],
)
def test_paid_lane_connect_phase_urlerror_normalizes_to_timeouterror(backend, label):
    client = _legacy_client(backend, timeout=30)

    def _open(req, timeout=None):
        raise urllib.error.URLError(socket.timeout("timed out"))

    _install_opener(client, _open)

    with pytest.raises(TimeoutError) as excinfo:
        client.call("sys", [{"role": "user", "content": "hi"}])
    assert str(excinfo.value) == f"Bedrock {label} request timed out after 30s"


@pytest.mark.parametrize(
    "backend", ["bedrock_converse", "bedrock_responses"]
)
def test_non_timeout_urlerror_is_not_swallowed(backend):
    client = _legacy_client(backend, timeout=30)

    def _open(req, timeout=None):
        raise urllib.error.URLError("name resolution failed")

    _install_opener(client, _open)

    with pytest.raises(urllib.error.URLError):
        client.call("sys", [{"role": "user", "content": "hi"}])


# ── (c) a genuine 429 retries exactly as before ──────────────────────────────

@pytest.mark.parametrize(
    "backend, payload",
    [
        ("bedrock_converse", CONVERSE_PAYLOAD),
        ("bedrock_responses", RESPONSES_PAYLOAD),
    ],
)
def test_genuine_429_retries_with_requested_delay(backend, payload, monkeypatch):
    client = _legacy_client(backend)
    calls = {"n": 0}
    sleeps: list[float] = []
    ok = _payload_opener(payload)

    def _open(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _http_error(429, "Please retry in 5s.")
        return ok(req, timeout=timeout)

    _install_opener(client, _open)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    response = client.call("sys", [{"role": "user", "content": "hi"}])

    assert response.content == '{"verdict":"correct"}'
    assert calls["n"] == 3                    # two 429s, then success
    assert sleeps == [6.0, 6.0]               # _parse_retry_delay(5s) + 1 pad


# ── (d) an unrelated error merely CONTAINING "429" does not retry ────────────

_BODY_WITH_LOOSE_429 = (
    '{"message": "internal server error", "requestId": "9f-429-77", '
    '"usage": {"input_tokens": 429}}'
)


@pytest.mark.parametrize(
    "backend, label",
    [
        ("bedrock_converse", "Converse"),
        ("bedrock_responses", "Responses"),
    ],
)
def test_http_500_containing_429_raises_on_first_occurrence(backend, label, monkeypatch):
    client = _legacy_client(backend)
    calls = {"n": 0}
    sleeps: list[float] = []

    def _open(req, timeout=None):
        calls["n"] += 1
        raise _http_error(500, _BODY_WITH_LOOSE_429)

    _install_opener(client, _open)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError) as excinfo:
        client.call("sys", [{"role": "user", "content": "hi"}])

    assert calls["n"] == 1
    assert sleeps == []
    # Message format is byte-identical to HEAD's.
    assert str(excinfo.value) == f"Bedrock {label} HTTP 500: {_BODY_WITH_LOOSE_429}"


# ── (d2) an authoritative 400 quota REJECTION does not retry either ──────────
# The reviewer's end-to-end reproduction of the regression this file once pinned
# as correct: a permanent HTTP 400 "Quota exceeded" was classified as a rate
# limit off its body text alone, producing 6 provider submissions and 155s of
# sleeps for a failure that can never succeed. One submission, zero sleeps.

_QUOTA_400_BODY = '{"error":{"message":"Quota exceeded for this account"}}'


@pytest.mark.parametrize(
    "backend, label",
    [
        ("bedrock_converse", "Converse"),
        ("bedrock_responses", "Responses"),
    ],
)
def test_http_400_quota_rejection_is_single_attempt(backend, label, monkeypatch):
    client = _legacy_client(backend)
    calls = {"n": 0}
    sleeps: list[float] = []

    def _open(req, timeout=None):
        calls["n"] += 1
        raise _http_error(400, _QUOTA_400_BODY)

    _install_opener(client, _open)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError) as excinfo:
        client.call("sys", [{"role": "user", "content": "hi"}])

    assert calls["n"] == 1
    assert sleeps == []
    assert str(excinfo.value) == f"Bedrock {label} HTTP 400: {_QUOTA_400_BODY}"


# ── (e) an explicit rate-limit phrase with no status still retries ───────────

def test_rate_limit_phrase_without_status_still_retries(monkeypatch):
    client = _legacy_client("bedrock_converse")
    calls = {"n": 0}
    sleeps: list[float] = []

    def _adapter(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("provider rate limit exceeded")
        return _stub_response()

    monkeypatch.setattr(client, "_call_bedrock_converse", _adapter)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    response = client.call("sys", [{"role": "user", "content": "hi"}])

    assert response.content == '{"verdict":"correct"}'
    assert calls["n"] == 2
    assert sleeps == [31.0]  # default 30s + 1 pad, unchanged


# ── (f) the spend guard still sees exactly what it saw before ────────────────

_SPEND_GUARD_STATUS_ATTRS = (
    "status",
    "status_code",
    "http_status",
    "response",
    "transport_trace",
)


@pytest.mark.parametrize(
    "backend", ["bedrock_converse", "bedrock_responses"]
)
def test_legacy_lane_error_hides_status_from_spend_guard_extractor(backend):
    client = _legacy_client(backend)

    def _open(req, timeout=None):
        raise _http_error(503, "service unavailable")

    _install_opener(client, _open)

    with pytest.raises(RuntimeError) as excinfo:
        client.call("sys", [{"role": "user", "content": "hi"}])

    err = excinfo.value
    assert not isinstance(err, urllib.error.HTTPError)
    for attr in _SPEND_GUARD_STATUS_ATTRS:
        assert not hasattr(err, attr), attr
    # ...while the client-side twin does read the private stamp.
    assert _provider_http_status(err) == 503


# ── (g) _is_rate_limit_error truth table ─────────────────────────────────────

class _Err(RuntimeError):
    def __init__(self, text, **attrs):
        super().__init__(text)
        for k, v in attrs.items():
            setattr(self, k, v)


def test_is_rate_limit_error_truth_table():
    # 1. authoritative structured status
    assert _is_rate_limit_error(_Err("upstream failure", status_code=429))
    assert _is_rate_limit_error(
        _Err("upstream failure", transport_trace={"response_http_status": 429})
    )
    assert _is_rate_limit_error(_http_error(429, "slow down"))
    # 2. status-position 429 in the message, no structured status
    assert _is_rate_limit_error(RuntimeError("Bedrock Responses HTTP 429; retry in 30s"))
    assert _is_rate_limit_error(RuntimeError("Bedrock Converse HTTP 429: throttled"))
    assert _is_rate_limit_error(RuntimeError("HTTP Error 429: Too Many Requests"))
    assert _is_rate_limit_error(RuntimeError("Error code: 429 - {'error': {}}"))
    # 3. explicit phrases (everything the old substring rule caught)
    assert _is_rate_limit_error(RuntimeError("provider rate limit exceeded"))
    assert _is_rate_limit_error(RuntimeError("RESOURCE_EXHAUSTED"))
    # gRPC/Google leading-code form — a bare 429 at the start IS the status.
    assert _is_rate_limit_error(
        RuntimeError("429 Resource has been exhausted (e.g. check quota).")
    )
    assert _is_rate_limit_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    # Status-position forms the old substring test also caught.
    assert _is_rate_limit_error(RuntimeError("Throttled (429)"))
    assert _is_rate_limit_error(RuntimeError("Server returned 429"))
    # 4. deliberate narrowing (a): an AUTHORITATIVE non-429 status is final, and
    #    body text never overrules it. A hard 400 quota rejection is permanent —
    #    retrying it five times bought 155s of sleeps and five extra billed POSTs.
    assert not _is_rate_limit_error(
        _Err("quota exceeded for this project", status_code=400)
    )
    assert not _is_rate_limit_error(
        _Err("Bedrock Responses HTTP 400: quota exceeded", _bedrock_http_status=400)
    )
    # 5. deliberate narrowing (b): a statusless "429" outside a status position
    assert not _is_rate_limit_error(_Err(_BODY_WITH_LOOSE_429, status_code=500))
    assert not _is_rate_limit_error(_http_error(500, _BODY_WITH_LOOSE_429))
    assert not _is_rate_limit_error(
        _Err("internal error req-429-abc", transport_trace={"response_http_status": 500})
    )
    # 6. plainly unrelated failures
    assert not _is_rate_limit_error(RuntimeError("connection reset by peer"))
    assert not _is_rate_limit_error(RuntimeError('{"input_tokens": 429}'))


def test_provider_http_status_reads_the_documented_candidate_set():
    assert _provider_http_status(_Err("x", status_code=429)) == 429
    assert _provider_http_status(_Err("x", status=503)) == 503
    assert _provider_http_status(_Err("x", response={"status_code": 429})) == 429
    assert _provider_http_status(
        _Err("x", transport_trace={"response_http_status": 502})
    ) == 502
    assert _provider_http_status(_http_error(404, "nope")) == 404
    # Out-of-range, non-int and bool values are rejected, not coerced.
    assert _provider_http_status(_Err("x", status_code=True)) is None
    assert _provider_http_status(_Err("x", status_code=42)) is None
    assert _provider_http_status(_Err("x", status_code="429")) is None
    assert _provider_http_status(RuntimeError("no structured status")) is None


# ── (h) differential gate: the new predicate never widens against HEAD ───────

def _head_is_rate_limit(error: BaseException) -> bool:
    """HEAD's predicate, reproduced byte-for-byte from the `call()` retry site
    (`git show HEAD:src/indra_belief/model_client.py`):

        msg = str(e).lower()
        if ("429" in msg or "rate limit" in msg
                or "resource_exhausted" in msg) and rate_limit_retries > 0:

    This is the DIFFERENTIAL ORACLE for the no-widening gate below. Its literals
    are separator-EXACT: "rate limit" needs the space (so "rate-limit" and
    "ratelimit" were never caught) and "resource_exhausted" needs the underscore
    (so "Resource has been exhausted" was caught by the bare "429" clause, never
    by the phrase). Any phrase the shipped predicate recognizes that this oracle
    does not is, by definition, a widening.
    """
    msg = str(error).lower()
    return "429" in msg or "rate limit" in msg or "resource_exhausted" in msg


# Realistic provider failures, each built the way the surface that raises it
# actually builds it: our own legacy lanes stamp `Bedrock {lane} HTTP {code}:
# {body}` plus the private status attribute, urllib raises HTTPError, provider
# SDKs carry `status_code` and render "Error code: N" into the message.
_DIFFERENTIAL_BATTERY: list[tuple[str, BaseException]] = [
    # — the reproduced regression —
    ("legacy-400-quota-body",
     _Err(f"Bedrock Responses HTTP 400: {_QUOTA_400_BODY}", _bedrock_http_status=400)),
    ("legacy-400-quota-httperror", _http_error(400, _QUOTA_400_BODY)),
    ("bare-quota-exceeded", RuntimeError("quota exceeded for this account")),
    ("bare-too-many-requests", RuntimeError("Too Many Requests")),
    ("bare-ratelimit-no-separator", RuntimeError("ratelimit exceeded, slow down")),
    ("bare-rate-hyphen-limit", RuntimeError("rate-limit exceeded, slow down")),
    ("aws-throttling-rate-exceeded", RuntimeError("ThrottlingException: Rate exceeded")),
    # — genuine rate limits, statusless: text in status position —
    ("urllib-http-error-429", RuntimeError("HTTP Error 429: Too Many Requests")),
    ("openai-error-code-429",
     RuntimeError("Error code: 429 - {'error': {'message': 'Rate limit reached'}}")),
    ("legacy-converse-http-429", RuntimeError("Bedrock Converse HTTP 429: throttled")),
    ("legacy-responses-http-429",
     RuntimeError("Bedrock Responses HTTP 429; retry in 30s")),
    ("throttled-paren-429", RuntimeError("Throttled (429)")),
    ("server-returned-429", RuntimeError("Server returned 429")),
    ("server-returned-http-status-429", RuntimeError("Server returned HTTP status 429")),
    ("amzn-status-code-429",
     RuntimeError("x-amzn-errortype: ThrottlingException; status_code=429")),
    ("grpc-429-resource-exhausted", RuntimeError("429 RESOURCE_EXHAUSTED")),
    ("grpc-429-prose",
     RuntimeError("429 Resource has been exhausted (e.g. check quota).")),
    ("google-resource-exhausted",
     RuntimeError("StatusCode.RESOURCE_EXHAUSTED: Quota exceeded for quota metric")),
    ("explicit-rate-limit-phrase", RuntimeError("provider rate limit exceeded")),
    # — genuine rate limits, status-carrying (our lanes always write the code
    #   into the message too, so the oracle sees the same 429) —
    ("status-429-legacy-converse",
     _Err("Bedrock Converse HTTP 429: throttled", _bedrock_http_status=429)),
    ("status-429-legacy-responses",
     _Err("Bedrock Responses HTTP 429; retry in 30s", _bedrock_http_status=429)),
    ("status-429-httperror", _http_error(429, "slow down")),
    ("status-429-sdk",
     _Err("Error code: 429 - {'error': {'message': 'Rate limit reached'}}",
          status_code=429)),
    # — loose 429 digits that are not a status —
    ("legacy-500-loose-429",
     _Err(f"Bedrock Converse HTTP 500: {_BODY_WITH_LOOSE_429}",
          _bedrock_http_status=500)),
    ("statusless-loose-429-body", RuntimeError(_BODY_WITH_LOOSE_429)),
    ("trace-500-loose-429-body",
     _Err(_BODY_WITH_LOOSE_429, transport_trace={"response_http_status": 500})),
    ("token-count-429", RuntimeError('{"input_tokens": 429}')),
    ("request-id-429", RuntimeError("internal error req-429-abc")),
    ("request-id-429-with-500-trace",
     _Err("internal error req-429-abc", transport_trace={"response_http_status": 500})),
    # — authoritative non-429 whose body happens to name a rate limit —
    ("status-503-with-rate-limit-text",
     _Err("Bedrock Responses HTTP 503: upstream rate limit pool drained",
          _bedrock_http_status=503)),
    ("status-400-with-rate-limit-text",
     _Err("Bedrock Converse HTTP 400: rate limit config invalid",
          _bedrock_http_status=400)),
    # — plainly unrelated failures —
    ("status-503-plain",
     _Err("Bedrock Converse HTTP 503: service unavailable", _bedrock_http_status=503)),
    ("connection-reset", RuntimeError("connection reset by peer")),
    ("lane-timeout", TimeoutError("Bedrock Converse request timed out after 300s")),
    ("unknown-backend", ValueError("Unknown backend: bedrock_moon")),
    ("json-decode", RuntimeError("Expecting value: line 1 column 1 (char 0)")),
]

# The ONLY cases where HEAD retried and the shipped predicate does not. Declared
# rather than derived, so an accidental narrowing added later fails this gate
# instead of silently redefining the contract.
_DELIBERATE_NARROWINGS = frozenset({
    # (a) an authoritative NON-429 status is final, body text notwithstanding
    "status-503-with-rate-limit-text",
    "status-400-with-rate-limit-text",
    "legacy-500-loose-429",
    "trace-500-loose-429-body",
    "request-id-429-with-500-trace",
    # (b) a statusless "429" that is not in a recognized status position
    "statusless-loose-429-body",
    "token-count-429",
    "request-id-429",
})


def test_rate_limit_classification_never_widens_head():
    assert len(_DIFFERENTIAL_BATTERY) >= 25
    assert len({label for label, _ in _DIFFERENTIAL_BATTERY}) == len(
        _DIFFERENTIAL_BATTERY
    )

    narrowed = set()
    for label, error in _DIFFERENTIAL_BATTERY:
        head = _head_is_rate_limit(error)
        new = _is_rate_limit_error(error)
        # NO WIDENING: nothing HEAD refused to retry may now be retried.
        assert not (new and not head), f"widened on {label}: {error}"
        if head and not new:
            narrowed.add(label)
    # ...and every narrowing is one we chose.
    assert narrowed == set(_DELIBERATE_NARROWINGS)


def test_rule_one_is_status_authoritative_and_text_independent():
    """The one rule that can exceed HEAD's substring test, pinned deliberately.

    A structured 429 retries even if the message never writes the digits. That
    is unreachable from our own surfaces — every error these lanes raise stamps
    `HTTP {code}` into its own message, and SDK errors carrying `status_code`
    render "Error code: N" (the four `status-429-*` battery rows prove HEAD saw
    the 429 too) — so it widens nothing in practice.
    """
    silent = _Err("upstream failure", status_code=429)
    assert _is_rate_limit_error(silent)
    assert not _head_is_rate_limit(silent)


# ── (i) the absolute deadline: loopback-proven outer bound + teardown ────────

class _LoopbackState:
    """Server-side observations, written only by the handler thread."""

    def __init__(self) -> None:
        self.body = b""
        self.chunk_delay = 0.0
        self.requests = 0
        self.torn_down = threading.Event()
        self.stop = threading.Event()


@pytest.fixture
def loopback_server():
    """Minimal 127.0.0.1 HTTP server that can dribble a response body.

    Modeled on `local_server` in tests/test_bedrock_responses_transport.py and
    duplicated rather than imported or hoisted into tests/conftest.py: this node
    is file-scoped, and cross-test-module fixture imports are brittle. Torn down
    in a `finally` so the suite can never hang on it.
    """
    state = _LoopbackState()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            self.close_connection = True
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            state.requests += 1
            body = state.body
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.flush()
                for byte in body:
                    if state.stop.is_set():
                        return
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    if state.chunk_delay:
                        time.sleep(state.chunk_delay)
            except OSError:
                # BrokenPipe / ConnectionReset: the client tore the connection
                # down. This is the EXTERNAL witness that the paid request
                # settled — nothing inside the client wrote this flag.
                state.torn_down.set()

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, server.server_port
    finally:
        state.stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _loopback_client(backend: str, port: int, *, timeout: int = 1) -> ModelClient:
    """`_legacy_client` plus a real urllib opener pointed at 127.0.0.1."""
    client = _legacy_client(backend, timeout=timeout)
    client.config["base_url"] = f"http://127.0.0.1:{port}"
    client._bedrock_url_opener = urllib.request.build_opener()
    return client


# One byte every 20ms over a 4000-byte body: the full drip would take 80s, so a
# call that returns anywhere near `timeout` can only have been cut off.
_DRIP_BYTES = 4000
_DRIP_DELAY = 0.02
_FULL_DRIP_SECONDS = _DRIP_BYTES * _DRIP_DELAY


@pytest.mark.parametrize(
    "backend, payload",
    [
        ("bedrock_converse", CONVERSE_PAYLOAD),
        ("bedrock_responses", RESPONSES_PAYLOAD),
    ],
)
def test_loopback_success_is_unchanged_over_the_real_socket_path(
    loopback_server, backend, payload
):
    """Behavior neutrality for the path the deadline reader actually added.

    Every fake opener in this suite has no `fp.raw._sock`, so it exercises the
    single-`read()` fallback. This is the only test where the chunked, deadline-
    bounded socket loop reads a real response, and it must produce exactly the
    ModelResponse the fake path produces. The body is padded past one 64KiB read
    (with a key both parsers ignore) so the loop provably concatenates several
    chunks rather than getting lucky on a single-read payload.
    """
    state, port = loopback_server
    state.body = json.dumps({**payload, "_pad": "z" * 200_000}).encode()
    state.chunk_delay = 0.0
    client = _loopback_client(backend, port, timeout=5)

    response = client.call("sys", [{"role": "user", "content": "hi"}])

    assert response.content == '{"verdict":"correct"}'
    assert response.prompt_tokens == 11
    assert response.tokens == 7
    assert response.finish_reason == "stop"
    assert state.requests == 1


@pytest.mark.parametrize(
    "backend, label",
    [
        ("bedrock_converse", "Converse"),
        ("bedrock_responses", "Responses"),
    ],
)
def test_slow_drip_body_is_cut_off_by_the_absolute_deadline(
    loopback_server, backend, label
):
    state, port = loopback_server
    state.body = b"x" * _DRIP_BYTES
    state.chunk_delay = _DRIP_DELAY
    client = _loopback_client(backend, port, timeout=1)

    started = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        client.call("sys", [{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - started

    # Message is the lane's existing one, byte for byte.
    assert str(excinfo.value) == f"Bedrock {label} request timed out after 1s"
    assert elapsed < 1 + 2.0, elapsed          # timeout + epsilon
    assert elapsed < _FULL_DRIP_SECONDS, elapsed   # and nowhere near the full drip
    assert state.requests == 1                 # exactly one provider submission


def test_deadline_teardown_is_observed_by_the_loopback_server(loopback_server):
    """Settlement observed from OUTSIDE the client.

    The server is still dribbling when the deadline fires; the client's response
    close tears the connection down, and the server's next write raises
    BrokenPipe/ConnectionReset. Nothing in this assertion is written by a fake
    the test controls — remove the `with` around the response in
    `_call_bedrock_responses` and this goes red (verified).

    The bound `excinfo` is load-bearing: it retains the traceback, so the
    adapter's frame — and with it any leaked response object — stays reachable
    and cannot be closed as a side effect of refcount collection. The only thing
    that can settle this connection in time is the adapter's own `with`.
    """
    state, port = loopback_server
    state.body = b"y" * _DRIP_BYTES
    state.chunk_delay = _DRIP_DELAY
    client = _loopback_client("bedrock_responses", port, timeout=1)

    with pytest.raises(TimeoutError) as excinfo:
        client.call("sys", [{"role": "user", "content": "hi"}])

    assert state.torn_down.wait(5.0), "server never saw the connection torn down"
    assert str(excinfo.value) == "Bedrock Responses request timed out after 1s"


class _FakeDripSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value):
        self.timeouts.append(value)


class _NeverEndingResponse:
    """urllib-shaped response whose body never ends, exposing a real
    `fp.raw._sock` so the deadline reader takes its socket path."""

    def __init__(self, closed: list[bool]) -> None:
        self._closed = closed
        self.fp = types.SimpleNamespace(
            raw=types.SimpleNamespace(_sock=_FakeDripSocket())
        )

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        self._closed.append(True)
        return False

    def read1(self, _size):
        time.sleep(0.01)
        return b"x"

    def read(self, *_a):
        return b"x"


@pytest.mark.parametrize(
    "backend, label",
    [
        ("bedrock_converse", "Converse"),
        ("bedrock_responses", "Responses"),
    ],
)
def test_deadline_expiry_closes_the_response_before_raising(backend, label):
    """The `_LaneDeadlineExpired` -> TimeoutError conversion happens outside the
    `with`, so `__exit__` has already run. Also pins that each read is re-armed
    with the SHRINKING remaining, i.e. the deadline actually bounds the loop."""
    client = _legacy_client(backend, timeout=1)
    closed: list[bool] = []
    response = _NeverEndingResponse(closed)

    _install_opener(client, lambda req, timeout=None: response)

    with pytest.raises(TimeoutError) as excinfo:
        client.call("sys", [{"role": "user", "content": "hi"}])

    assert str(excinfo.value) == f"Bedrock {label} request timed out after 1s"
    assert closed == [True]
    timeouts = response.fp.raw._sock.timeouts
    assert len(timeouts) > 1
    assert timeouts == sorted(timeouts, reverse=True)
    assert timeouts[-1] <= 1.0
