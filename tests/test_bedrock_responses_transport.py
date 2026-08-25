"""Adversarial, loopback-only tests for the formal Gemma Responses transport."""
from __future__ import annotations

import ast
import base64
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import socket
import ssl
import sys
import threading
import time
import traceback
import urllib.request

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief import bedrock_responses_transport as responses_transport
from indra_belief import bedrock_transport_base as transport_base
from indra_belief.bedrock_responses_transport import (
    BACKEND_NAME,
    FORMAL_RESPONSES_ENDPOINT,
    BedrockResponsesConnectionError,
    BedrockResponsesTransportError,
    RawBedrockResponsesTransport,
    build_bedrock_responses_body,
    build_pinned_https_opener,
    canonical_json_bytes,
    canonical_json_sha256,
    verify_transport_response_preimage,
)
from indra_belief.model_client import LOCAL_MODELS, ModelClient, ReasoningStatus


MODEL_ID = "google.gemma-4-e2b"
TOKEN = "fixture-secret-bearer"
CA_SHA256 = "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994"


def _responses_payload(
    *,
    status: str = "completed",
    reasoning: str | None = "private reasoning",
    content: str = '{"verdict":"correct","confidence":"high"}',
    usage: dict | None = None,
) -> dict:
    if usage is None:
        usage = {
            "input_tokens": 17,
            "output_tokens": 23,
            "output_tokens_details": {"reasoning_tokens": 11},
        }
    output: list[dict] = []
    if reasoning is not None:
        output.append(
            {
                "id": "reasoning-fixture",
                "type": "reasoning",
                "content": ([{"type": "reasoning_text", "text": reasoning}] if reasoning else []),
            }
        )
    output.append(
        {
            "id": "message-fixture",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        }
    )
    return {
        "id": "resp-fixture",
        "object": "response",
        "status": status,
        "model": MODEL_ID,
        "output": output,
        "usage": usage,
    }


def _payload_bytes(payload: dict) -> bytes:
    # Provider bytes intentionally retain insertion order/spaces so the raw and
    # canonical response commitments are demonstrably distinct concepts.
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class _ServerState:
    def __init__(self) -> None:
        self.responses = deque()
        self.requests: list[dict] = []

    def push(
        self,
        status: int,
        body: bytes,
        *,
        headers: dict[str, str | None] | None = None,
        delay: float = 0,
        chunk_delay: float = 0,
    ) -> None:
        self.responses.append((status, body, headers or {}, delay, chunk_delay))


@pytest.fixture
def local_server():
    state = _ServerState()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802
            self.close_connection = True
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            state.requests.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": body,
                }
            )
            status, response_body, headers, delay, chunk_delay = state.responses.popleft()
            if delay:
                time.sleep(delay)
            try:
                self.send_response(status)
                content_type = headers.get("Content-Type", "application/json")
                if content_type is not None:
                    self.send_header("Content-Type", content_type)
                declared_length = headers.get(
                    "Content-Length", str(len(response_body))
                )
                if declared_length is not None:
                    self.send_header("Content-Length", declared_length)
                for key, value in headers.items():
                    if key.lower() not in {"content-type", "content-length"} and value is not None:
                        self.send_header(key, value)
                self.end_headers()
                if chunk_delay:
                    for byte in response_body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(chunk_delay)
                else:
                    self.wfile.write(response_body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/openai/v1/responses"
    try:
        yield state, endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _transport(
    endpoint: str,
    *,
    maximum: int = 1024 * 1024,
    request_maximum: int = 1024 * 1024,
) -> RawBedrockResponsesTransport:
    return RawBedrockResponsesTransport(
        endpoint=endpoint,
        expected_endpoint=endpoint,
        expected_model_id=MODEL_ID,
        bearer_token=TOKEN,
        ca_bundle=None,
        expected_ca_bundle_sha256=None,
        max_request_bytes=request_maximum,
        max_response_bytes=maximum,
        allow_insecure_http_for_tests=True,
    )


def _body(*, relation: bool = False) -> dict:
    return build_bedrock_responses_body(
        model_id=MODEL_ID,
        system="system µ",
        messages=[{"role": "user", "content": "evidence"}],
        max_output_tokens=3000 if relation else 16000,
        reasoning_effort="none" if relation else "high",
    )


class _LegacyResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = _payload_bytes(payload)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._raw


class _LegacyCaptureOpener:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.request = None
        self.timeout = None

    def open(self, request, timeout=None):
        self.request = request
        self.timeout = timeout
        return _LegacyResponse(self.payload)


def _legacy_call(
    payload: dict,
    *,
    system: str,
    messages: list[dict],
    maximum: int,
    effort: str | None,
):
    client = ModelClient.__new__(ModelClient)
    client.model_name = "legacy-responses-oracle"
    client.backend = "bedrock_responses"
    client.config = {
        "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "model_id": MODEL_ID,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
    }
    client._bedrock_token = TOKEN
    opener = _LegacyCaptureOpener(payload)
    client._bedrock_url_opener = opener
    result = client._call_bedrock_responses(
        system,
        messages,
        maximum,
        0.731,
        7,
        reasoning_effort=effort,
    )
    return result, json.loads(opener.request.data)


@pytest.mark.parametrize(
    ("relation", "expected_sha256"),
    [
        (False, "9f7962881e64437d53159f5b114d25f5c79a27117ac8c42408883972646e245d"),
        (True, "a7ab7813f008fdf97b4f99ca1146921cf0eba583dd0cdd6ebd516cfaac4fbedc"),
    ],
    ids=["main", "relation"],
)
def test_canonical_body_matches_legacy_adapter_semantics(
    relation: bool, expected_sha256: str
) -> None:
    payload = _responses_payload(reasoning=None if relation else "reason")
    legacy, legacy_body = _legacy_call(
        payload,
        system="system µ",
        messages=[{"role": "user", "content": "evidence"}],
        maximum=3000 if relation else 16000,
        effort="none" if relation else "high",
    )
    body = _body(relation=relation)
    assert legacy.content
    assert body == legacy_body
    assert canonical_json_sha256(body) == expected_sha256
    assert ("reasoning" in body) is (not relation)
    assert "temperature" not in body
    assert "response_format" not in body


def test_system_messages_and_assistant_examples_match_legacy_newline_rule() -> None:
    messages = [
        {"role": "assistant", "content": "example answer"},
        {"role": "system", "content": "extra rule"},
        {"role": "user", "content": "question"},
    ]
    _, legacy_body = _legacy_call(
        _responses_payload(),
        system="base rule",
        messages=messages,
        maximum=91,
        effort="high",
    )
    body = build_bedrock_responses_body(
        model_id=MODEL_ID,
        system="base rule",
        messages=messages,
        max_output_tokens=91,
        reasoning_effort="high",
    )
    assert body == legacy_body
    assert body["instructions"] == "base rule\nextra rule"
    assert body["input"][0]["content"][0]["type"] == "output_text"
    assert body["input"][1]["content"][0]["type"] == "input_text"


@pytest.mark.parametrize(
    "payload",
    [
        _responses_payload(),
        _responses_payload(
            reasoning=None,
            usage={
                "input_tokens": 5,
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        ),
        {
            "id": "resp-encrypted-shape",
            "object": "response",
            "model": MODEL_ID,
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 40,
                "output_tokens_details": {"reasoning_tokens": 1847},
            },
            "output": [
                {"type": "reasoning", "summary": [], "id": "encrypted"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            ],
        },
        {
            "id": "resp-incomplete-shape",
            "object": "response",
            "model": MODEL_ID,
            "status": "incomplete",
            "usage": {"input_tokens": 3, "output_tokens": 9},
            "output_text": "ignored top-level convenience field",
            "output": [
                {
                    "type": "reasoning",
                    "content": [
                        {"type": "reasoning_text", "text": "r1"},
                        {"type": "reasoning_text", "text": "r2"},
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "a1"},
                        {"type": "output_text", "text": "a2"},
                    ],
                },
            ],
        },
    ],
    ids=["observed-plaintext", "observed-no-reasoning", "encrypted", "incomplete"],
)
def test_parser_matches_legacy_adapter_on_current_provider_shapes(
    local_server, payload: dict
) -> None:
    legacy, _ = _legacy_call(
        payload,
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        maximum=100,
        effort="high",
    )
    state, endpoint = local_server
    state.push(200, _payload_bytes(payload))
    raw = _transport(endpoint).call(
        build_bedrock_responses_body(
            model_id=MODEL_ID,
            system="sys",
            messages=[{"role": "user", "content": "q"}],
            max_output_tokens=100,
            reasoning_effort="high",
        ),
        timeout=2,
    )
    assert raw.content == legacy.content
    assert raw.reasoning == legacy.reasoning
    assert raw.prompt_tokens == legacy.prompt_tokens
    assert raw.output_tokens == legacy.tokens
    assert raw.reasoning_tokens == legacy.reasoning_trace["reasoning_tokens"]
    assert raw.finish_reason == legacy.finish_reason


def test_success_records_exact_route_headers_and_wire_commitments(local_server) -> None:
    state, endpoint = local_server
    payload = _responses_payload()
    response_raw = _payload_bytes(payload)
    state.push(200, response_raw, headers={"Content-Type": "application/json; charset=utf-8"})
    body = _body()
    result = _transport(endpoint).call(body, timeout=2)

    assert result.content.startswith('{"verdict"')
    assert result.reasoning == "private reasoning"
    assert result.prompt_tokens == 17
    assert result.output_tokens == 23
    assert result.reasoning_tokens == 11
    assert result.finish_reason == "stop"
    assert result.reasoning_item_present is True
    assert len(state.requests) == 1
    request = state.requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/openai/v1/responses"
    assert request["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert request["headers"]["Accept-Encoding"] == "identity"
    assert request["body"] == canonical_json_bytes(body)

    trace = result.transport_trace
    assert trace["backend"] == BACKEND_NAME
    assert trace["method"] == "POST"
    assert trace["request_body_sha256"] == hashlib.sha256(request["body"]).hexdigest()
    assert trace["response_body_sha256"] == hashlib.sha256(response_raw).hexdigest()
    assert trace["response_json_sha256"] == canonical_json_sha256(payload)
    assert trace["response_http_status"] == 200
    assert trace["response_content_length_declared"] == len(response_raw)
    assert trace["response_body_complete"] is True
    assert trace["response_body_preimage_kind"] == "complete"
    assert trace["response_body_preimage_redacted"] is False
    assert base64.b64decode(trace["response_body_preimage_b64"]) == response_raw
    assert verify_transport_response_preimage(trace) == response_raw
    assert result.response_body_preimage_b64 == trace["response_body_preimage_b64"]
    assert result.provider_response_id == payload["id"]
    assert result.provider_model == MODEL_ID
    assert result.provider_status == "completed"
    assert trace["provider_response_id"] == payload["id"]
    assert trace["provider_response_model"] == MODEL_ID
    assert trace["provider_response_status"] == "completed"
    assert trace["redirects_allowed"] is False
    assert trace["environment_proxies_allowed"] is False
    assert trace["ambient_tls_trust_allowed"] is False
    assert TOKEN not in json.dumps(trace)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("response_body_complete", False, "not complete"),
        ("response_framing_valid", False, "framing is not valid"),
        ("response_size_exceeded", True, "safety limit"),
        ("response_body_preimage_redacted", True, "redacted"),
        ("response_body_bytes", 999, "byte count mismatch"),
        ("response_body_sha256", "0" * 64, "SHA-256 mismatch"),
        ("response_json_sha256", "0" * 64, "canonical JSON SHA-256 mismatch"),
    ],
)
def test_response_preimage_verifier_fails_closed(
    field: str, value, match: str
) -> None:
    raw = b'{"proof":true}'
    trace = {
        "response_body_preimage_kind": "complete",
        "response_body_complete": True,
        "response_framing_valid": True,
        "response_size_exceeded": False,
        "response_body_preimage_redacted": False,
        "response_body_preimage_b64": base64.b64encode(raw).decode("ascii"),
        "response_body_bytes": len(raw),
        "response_body_sha256": hashlib.sha256(raw).hexdigest(),
        "response_json_sha256": canonical_json_sha256({"proof": True}),
    }
    trace[field] = value
    with pytest.raises(ValueError, match=match):
        verify_transport_response_preimage(trace)


def test_canonical_json_is_strict_deterministic_utf8() -> None:
    a = {"z": "µ", "a": {"b": 2, "a": 1}}
    b = {"a": {"a": 1, "b": 2}, "z": "µ"}
    assert canonical_json_bytes(a) == b'{"a":{"a":1,"b":2},"z":"\xc2\xb5"}'
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        canonical_json_bytes({"bad": math.nan})
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        canonical_json_bytes({"bad": "\ud800"})


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b'{"output":[],"output":[]}',
        b'{"output":[{"type":"message","content":[{"text":"x","text":"y"}]}]}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b'\xff\xfe',
        b'{"output":[{"type":"message","content":[{"text":"\\ud800"}]}]}',
        b'[]',
    ],
    ids=[
        "syntax",
        "duplicate-top",
        "duplicate-nested",
        "nan",
        "infinity",
        "float-overflow",
        "utf8",
        "lone-surrogate",
        "not-object",
    ],
)
def test_malformed_or_ambiguous_json_retains_replayable_raw_body(local_server, raw) -> None:
    state, endpoint = local_server
    state.push(200, raw)
    with pytest.raises(BedrockResponsesTransportError) as caught:
        _transport(endpoint).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_http_status"] == 200
    assert trace["response_body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert trace["response_json_sha256"] is None
    assert verify_transport_response_preimage(trace) == raw
    assert TOKEN not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"output": {}},
        {"output": ["not-an-object"]},
        {"output": [{"type": "message", "content": ["not-a-block"]}]},
        {"output": [{"type": "message", "content": [{"text": 7}]}]},
        {
            "output": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "output_text", "text": "x"}],
                }
            ]
        },
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "x"}],
                }
            ]
        },
        {"output": [{"type": "tool_call", "content": []}]},
        {"output": [], "usage": []},
        {"output": [], "usage": {"output_tokens": True}},
        {"output": [], "usage": {"output_tokens": -1}},
        {"output": [], "usage": {"input_tokens": "7"}},
        {"output": [], "usage": {"output_tokens_details": []}},
        {"output": [], "status": 7},
        {"output": [], "status": "failed"},
        {"output": [], "status": None},
        {"output": [], "usage": None},
        {"output": [], "usage": {"input_tokens": 1}},
        {"output": [], "usage": {"output_tokens": 1}},
        {"output": [], "id": ""},
        {"output": [], "id": "bad\x7fresponse-id"},
    ],
    ids=[
        "output-object",
        "item-string",
        "block-string",
        "text-number",
        "wrong-message-role",
        "wrong-message-block-type",
        "unknown-output-item",
        "usage-array",
        "token-bool",
        "token-negative",
        "token-string",
        "details-array",
        "status-number",
        "status-failed",
        "status-missing",
        "usage-missing",
        "output-token-missing",
        "input-token-missing",
        "empty-response-id",
        "control-response-id",
    ],
)
def test_schema_malformation_fails_but_keeps_raw_and_canonical_hashes(
    local_server, payload: dict
) -> None:
    state, endpoint = local_server
    envelope = {
        "id": "resp-invalid-schema",
        "object": "response",
        "model": MODEL_ID,
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    envelope.update(payload)
    raw = _payload_bytes(envelope)
    state.push(200, raw)
    with pytest.raises(BedrockResponsesTransportError, match="invalid response schema") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert trace["response_json_sha256"] == canonical_json_sha256(envelope)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.pop("model"), "frozen model"),
        (lambda payload: payload.update(model="google.gemma-4-31b"), "frozen model"),
        (lambda payload: payload.update(object="chat.completion"), "object type"),
    ],
    ids=["missing-model", "wrong-model", "wrong-object"],
)
def test_response_envelope_is_bound_to_frozen_model_and_object(
    local_server, mutation, match: str
) -> None:
    state, endpoint = local_server
    payload = _responses_payload()
    mutation(payload)
    raw = _payload_bytes(payload)
    state.push(200, raw)
    with pytest.raises(BedrockResponsesTransportError, match=match) as caught:
        _transport(endpoint).call(_body(), timeout=2)
    assert caught.value.transport_trace["response_body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert caught.value.transport_trace["response_json_sha256"] == canonical_json_sha256(payload)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 503])
def test_http_errors_are_single_attempt_and_body_is_replayable(local_server, status: int) -> None:
    state, endpoint = local_server
    payload = {"error": f"provider secret detail {status}"}
    raw = _payload_bytes(payload)
    state.push(status, raw)
    with pytest.raises(BedrockResponsesTransportError, match=rf"HTTP {status}") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    assert len(state.requests) == 1
    assert "provider secret detail" not in str(caught.value)
    assert caught.value.transport_trace["response_body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert caught.value.transport_trace["response_json_sha256"] == canonical_json_sha256(payload)
    assert verify_transport_response_preimage(caught.value.transport_trace) == raw


def test_error_body_with_bearer_is_redacted_and_cannot_claim_exact_replay(
    local_server,
) -> None:
    state, endpoint = local_server
    raw = _payload_bytes({"error": f"provider echoed Bearer {TOKEN}"})
    state.push(500, raw)
    with pytest.raises(BedrockResponsesTransportError, match="HTTP 500") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_body_sha256"] is None
    assert trace["response_body_preimage_redacted"] is True
    assert trace["response_body_preimage_b64"] is None
    with pytest.raises(ValueError, match="no complete"):
        verify_transport_response_preimage(trace)


def test_429_retry_after_is_normalized_for_existing_retry_parser(local_server) -> None:
    state, endpoint = local_server
    state.push(429, b'{"error":"slow"}', headers={"Retry-After": "2.5"})
    with pytest.raises(BedrockResponsesTransportError, match=r"HTTP 429; retry in 2.5s"):
        _transport(endpoint).call(_body(), timeout=2)


def test_redirect_is_never_followed(local_server) -> None:
    state, endpoint = local_server
    state.push(302, b"", headers={"Location": endpoint})
    with pytest.raises(BedrockResponsesTransportError, match="HTTP 302"):
        _transport(endpoint).call(_body(), timeout=2)
    assert len(state.requests) == 1


def test_declared_oversize_response_is_not_read_or_partially_hashed(local_server) -> None:
    state, endpoint = local_server
    state.push(200, b"x" * 257)
    with pytest.raises(BedrockResponsesTransportError, match="safety limit") as caught:
        _transport(endpoint, maximum=256).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_size_exceeded"] is True
    assert trace["response_body_sha256"] is None
    assert trace["response_body_bytes"] is None
    assert trace["response_content_length_declared"] == 257


def test_request_size_is_bounded_before_any_side_effect(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _payload_bytes(_responses_payload()))
    transport = _transport(endpoint, request_maximum=16)
    with pytest.raises(ValueError, match="request exceeds"):
        transport.call(_body(), timeout=2)
    assert state.requests == []


@pytest.mark.parametrize(
    ("headers", "match"),
    [
        ({"Content-Encoding": "gzip"}, "content encoding"),
        ({"Content-Type": "text/plain"}, "not application/json"),
        ({"Content-Type": None}, "no Content-Type"),
        ({"Content-Type": "application/json; charset=iso-8859-1"}, "malformed or unsupported"),
        ({"Content-Type": "application/json; profile=x"}, "malformed or unsupported"),
        (
            {"Content-Type": "application/json; charset=utf-8; charset=utf8"},
            "malformed or unsupported",
        ),
        (
            {"Content-Type": 'application/json; charset="utf-8'},
            "malformed or unsupported",
        ),
        (
            {"Content-Type": 'application/json; charset=utf-8"'},
            "malformed or unsupported",
        ),
        (
            {"Content-Type": 'application/json; charset=""utf-8""'},
            "malformed or unsupported",
        ),
    ],
    ids=[
        "compressed",
        "wrong-type",
        "missing-type",
        "wrong-charset",
        "extra-param",
        "duplicate-param",
        "unterminated-quote",
        "trailing-quote",
        "doubled-quotes",
    ],
)
def test_representation_headers_fail_closed_with_body_commitment(
    local_server, headers: dict, match: str
) -> None:
    state, endpoint = local_server
    raw = _payload_bytes(_responses_payload())
    state.push(200, raw, headers=headers)
    with pytest.raises(BedrockResponsesTransportError, match=match) as caught:
        _transport(endpoint).call(_body(), timeout=2)
    assert caught.value.transport_trace["response_body_sha256"] == hashlib.sha256(raw).hexdigest()


def test_timeout_has_exact_request_commitment(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _payload_bytes(_responses_payload()), delay=0.25)
    with pytest.raises(TimeoutError, match="timed out") as caught:
        _transport(endpoint).call(_body(), timeout=0.03)
    trace = caught.value.transport_trace
    assert trace["request_body_sha256"] == canonical_json_sha256(_body())
    assert trace["response_http_status"] is None


def test_absolute_deadline_rejects_slow_drip_and_retains_partial_prefix(
    local_server,
) -> None:
    state, endpoint = local_server
    raw = _payload_bytes(_responses_payload())
    state.push(200, raw, chunk_delay=0.01)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out") as caught:
        _transport(endpoint).call(_body(), timeout=0.06)
    elapsed = time.monotonic() - started
    trace = caught.value.transport_trace
    assert elapsed < 0.5
    assert trace["response_http_status"] == 200
    assert trace["response_body_complete"] is False
    assert trace["response_body_preimage_kind"] == "partial_prefix"
    assert trace["response_body_prefix_bytes"] > 0
    prefix = base64.b64decode(trace["response_body_preimage_b64"])
    assert raw.startswith(prefix)
    assert hashlib.sha256(prefix).hexdigest() == trace["response_body_prefix_sha256"]
    assert trace["response_body_sha256"] is None
    with pytest.raises(ValueError, match="no complete response preimage"):
        verify_transport_response_preimage(trace)


def test_dns_resolution_cannot_escape_absolute_deadline(
    local_server, monkeypatch
) -> None:
    state, endpoint = local_server
    state.push(200, _payload_bytes(_responses_payload()))
    real_getaddrinfo = socket.getaddrinfo

    def slow_getaddrinfo(*args, **kwargs):
        time.sleep(0.3)
        return real_getaddrinfo(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow_getaddrinfo)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out") as caught:
        _transport(endpoint).call(_body(), timeout=0.03)
    assert time.monotonic() - started < 0.2
    assert caught.value.transport_trace["response_http_status"] is None
    assert state.requests == []


def test_ambiguous_transfer_encoding_and_content_length_is_rejected_before_body(
    local_server,
) -> None:
    state, endpoint = local_server
    state.push(
        200,
        _payload_bytes(_responses_payload()),
        headers={"Transfer-Encoding": "chunked"},
    )
    with pytest.raises(BedrockResponsesTransportError, match="invalid response framing") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_http_status"] == 200
    assert trace["response_framing_valid"] is False
    assert trace["response_body_preimage_b64"] is None


def test_short_content_length_is_partial_not_complete_replay(local_server) -> None:
    state, endpoint = local_server
    raw = b'{"short":true}'
    state.push(200, raw, headers={"Content-Length": str(len(raw) + 10)})
    with pytest.raises(BedrockResponsesTransportError, match="invalid response framing") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_body_complete"] is False
    assert trace["response_body_preimage_kind"] == "partial_prefix"
    assert trace["response_framing_valid"] is False
    assert trace["response_body_prefix_sha256"] == hashlib.sha256(raw).hexdigest()
    assert base64.b64decode(trace["response_body_preimage_b64"]) == raw
    with pytest.raises(ValueError, match="no complete response preimage"):
        verify_transport_response_preimage(trace)


@pytest.mark.parametrize("network_error", [True, False], ids=["network", "unexpected"])
def test_connection_and_unexpected_errors_redact_bearer(
    local_server, network_error: bool
) -> None:
    _state, endpoint = local_server
    transport = _transport(endpoint)

    class BrokenConnection:
        sock = None

        def connect(self):
            error_type = OSError if network_error else RuntimeError
            raise error_type(f"Bearer {TOKEN}; {TOKEN}")

        def close(self):
            return None

    transport._connection_factory = lambda _deadline, _abort: BrokenConnection()
    with pytest.raises(BedrockResponsesTransportError) as caught:
        transport.call(_body(), timeout=2)
    assert TOKEN not in str(caught.value)
    assert TOKEN not in json.dumps(caught.value.transport_trace)
    rendered_traceback = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert TOKEN not in rendered_traceback
    assert caught.value.transport_trace["request_body_sha256"] == canonical_json_sha256(_body())


def test_tls_certificate_failure_is_terminal_and_redacted(local_server) -> None:
    _state, endpoint = local_server
    transport = _transport(endpoint)

    class BadCertificateConnection:
        sock = None

        def connect(self):
            raise ssl.SSLCertVerificationError(f"certificate rejected: {TOKEN}")

        def close(self):
            return None

    transport._connection_factory = lambda _deadline, _abort: BadCertificateConnection()
    with pytest.raises(BedrockResponsesTransportError) as caught:
        transport.call(_body(), timeout=2)
    assert not isinstance(caught.value, ConnectionError)
    # The spend guard's classify_provider_failure() asserted here on how this
    # exception classified for the paid-run ledger. That module was removed;
    # the exception type and identity checks above are the transport's own.
    assert TOKEN not in str(caught.value)
    rendered_traceback = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert TOKEN not in rendered_traceback


def test_mid_body_connection_reset_is_retryable_with_partial_provenance(
    local_server,
) -> None:
    _state, endpoint = local_server
    transport = _transport(endpoint)
    partial = b'{"provider":"part'

    class Headers:
        values = {
            "content-length": [str(len(partial) + 20)],
            "content-type": ["application/json"],
        }

        def get_all(self, name):
            return self.values.get(name.lower())

        def get(self, name):
            values = self.get_all(name)
            return values[0] if values else None

    class ResetResponse:
        status = 200
        headers = Headers()
        fp = None

        def __init__(self) -> None:
            self.reads = 0

        def read1(self, _size):
            self.reads += 1
            if self.reads == 1:
                return partial
            raise ConnectionResetError(f"connection reset Bearer {TOKEN}")

        def close(self):
            return None

    class ResetConnection:
        sock = None

        def connect(self):
            return None

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return ResetResponse()

        def close(self):
            return None

    transport._connection_factory = lambda _deadline, _abort: ResetConnection()
    with pytest.raises(BedrockResponsesConnectionError) as caught:
        transport.call(_body(), timeout=2)
    # The spend guard's classify_provider_failure() asserted here on how this
    # exception classified for the paid-run ledger. That module was removed;
    # the exception type and identity checks above are the transport's own.
    trace = caught.value.transport_trace
    assert trace["response_http_status"] == 200
    assert trace["response_body_complete"] is False
    assert trace["response_framing_valid"] is False
    assert trace["response_body_prefix_sha256"] == hashlib.sha256(partial).hexdigest()
    assert base64.b64decode(trace["response_body_preimage_b64"]) == partial
    assert TOKEN not in "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )


@pytest.mark.parametrize(
    ("endpoint", "allow", "match"),
    [
        ("http://example.com/openai/v1/responses", False, "formal frozen route"),
        ("https://example.com/openai/v1/responses", False, "formal frozen route"),
        ("https://bedrock-mantle.us-east-1.api.aws/v1/responses", False, "path"),
        ("http://example.com/openai/v1/responses", True, "loopback"),
        ("http://user:pass@127.0.0.1/openai/v1/responses", True, "credentials"),
        ("http://127.0.0.1/openai/v1/responses?q=x", True, "query"),
    ],
)
def test_endpoint_policy_fails_closed(endpoint: str, allow: bool, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RawBedrockResponsesTransport(
            endpoint=endpoint,
            expected_endpoint=endpoint,
            expected_model_id=MODEL_ID,
            bearer_token=TOKEN,
            ca_bundle=None,
            expected_ca_bundle_sha256=None,
            allow_insecure_http_for_tests=allow,
        )


def test_endpoint_must_equal_independent_frozen_expectation(local_server) -> None:
    _state, endpoint = local_server
    with pytest.raises(ValueError, match="frozen expectation"):
        RawBedrockResponsesTransport(
            endpoint=endpoint,
            expected_endpoint=endpoint.replace("127.0.0.1", "localhost"),
            expected_model_id=MODEL_ID,
            bearer_token=TOKEN,
            ca_bundle=None,
            expected_ca_bundle_sha256=None,
            allow_insecure_http_for_tests=True,
        )


@pytest.mark.parametrize(
    "token",
    [
        "",
        "with space",
        "with\nnewline",
        "with\rreturn",
        "with\ttab",
        "unicode-µ",
        "x" * 16385,
    ],
)
def test_bearer_token_must_be_header_safe(local_server, token: str) -> None:
    _state, endpoint = local_server
    with pytest.raises(ValueError, match="header-safe"):
        RawBedrockResponsesTransport(
            endpoint=endpoint,
            expected_endpoint=endpoint,
            expected_model_id=MODEL_ID,
            bearer_token=token,
            ca_bundle=None,
            expected_ca_bundle_sha256=None,
            allow_insecure_http_for_tests=True,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body.update(model="google.gemma-4-31b"),
        lambda body: body.update(temperature=0.1),
        lambda body: body.update(response_format={"type": "json_object"}),
        lambda body: body.update(max_output_tokens=True),
        lambda body: body.update(reasoning={"effort": "none"}),
        lambda body: body.update(input=[{"role": "user", "content": []}]),
    ],
    ids=["model", "temperature", "response-format", "bool-max", "none-reasoning", "empty-blocks"],
)
def test_transport_revalidates_frozen_request_schema(local_server, mutate) -> None:
    state, endpoint = local_server
    body = _body()
    mutate(body)
    with pytest.raises(ValueError):
        _transport(endpoint).call(body, timeout=2)
    assert state.requests == []


def test_cadata_capability_falls_back_only_to_same_explicit_path(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "ca.pem"
    path.write_bytes(b"fixture explicit pem bytes\n")
    contexts = []

    class FakeContext:
        def __init__(self, first: bool) -> None:
            self.first = first
            self.calls = []

        def load_verify_locations(self, *, cadata=None, cafile=None):
            self.calls.append({"cadata": cadata, "cafile": cafile})
            if self.first and cadata is not None:
                raise TypeError("cadata unsupported")

    def new_context():
        context = FakeContext(first=not contexts)
        contexts.append(context)
        return context

    monkeypatch.setattr(responses_transport, "_new_tls_context", new_context)
    _context, digest, mode = responses_transport._explicit_tls_context(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert mode == "cafile_fallback"
    assert contexts[0].calls[0]["cadata"] == path.read_text()
    assert contexts[1].calls == [{"cadata": None, "cafile": str(path)}]


def test_malformed_cadata_does_not_silently_fall_back_to_path(tmp_path) -> None:
    path = tmp_path / "bad-ca.pem"
    path.write_text("not a certificate")
    with pytest.raises(ValueError, match="PEM cadata"):
        responses_transport._explicit_tls_context(path)


def test_formal_opener_ignores_proxy_and_tls_environment(monkeypatch) -> None:
    ca = Path("/private/etc/ssl/cert.pem")
    if not ca.is_file():
        pytest.skip("formal macOS CA bundle is not present on this platform")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid:8888")
    monkeypatch.setenv("SSL_CERT_FILE", "/definitely/not/the/formal/ca.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "/definitely/not/the/formal/ca-dir")
    opener, ca_sha256, load_mode = build_pinned_https_opener(
        ca, expected_ca_bundle_sha256=CA_SHA256
    )
    proxy_handlers = [
        handler for handler in opener.handlers if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert proxy_handlers == []
    assert ca_sha256 == hashlib.sha256(ca.read_bytes()).hexdigest() == CA_SHA256
    assert load_mode == "cadata"


def test_formal_opener_rejects_ca_hash_drift() -> None:
    ca = Path("/private/etc/ssl/cert.pem")
    if not ca.is_file():
        pytest.skip("formal macOS CA bundle is not present on this platform")
    with pytest.raises(ValueError, match="frozen SHA-256"):
        build_pinned_https_opener(ca, expected_ca_bundle_sha256="0" * 64)


def test_direct_https_connection_preserves_frozen_tls_hostname(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.timeouts = []
            self.closed = False

        def settimeout(self, value):
            self.timeouts.append(value)

        def close(self):
            self.closed = True

    class FakeContext:
        def __init__(self) -> None:
            self.server_hostname = None

        def wrap_socket(self, sock, *, server_hostname):
            self.server_hostname = server_hostname
            return sock

    wire_socket = FakeSocket()
    context = FakeContext()
    connection = responses_transport._DeadlineHTTPSConnection(
        responses_transport.FORMAL_BEDROCK_HOST,
        443,
        timeout=1,
        context=context,
        absolute_deadline=time.monotonic() + 1,
        abort_event=threading.Event(),
    )
    monkeypatch.setattr(connection, "_resolved_socket", lambda: wire_socket)
    connection.connect()
    assert context.server_hostname == responses_transport.FORMAL_BEDROCK_HOST
    assert connection.sock is wire_socket
    assert wire_socket.timeouts
    connection.close()


def test_watchdog_race_before_tls_wrap_closes_candidate(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.closed = False

        def settimeout(self, _value):
            return None

        def close(self):
            self.closed = True

    class FakeContext:
        wrapped = False

        def wrap_socket(self, sock, *, server_hostname):
            self.wrapped = True
            return sock

    abort = threading.Event()
    wire_socket = FakeSocket()
    context = FakeContext()
    connection = responses_transport._DeadlineHTTPSConnection(
        responses_transport.FORMAL_BEDROCK_HOST,
        443,
        timeout=1,
        context=context,
        absolute_deadline=time.monotonic() + 1,
        abort_event=abort,
    )

    def resolved_after_watchdog():
        abort.set()
        return wire_socket

    monkeypatch.setattr(connection, "_resolved_socket", resolved_after_watchdog)
    with pytest.raises(TimeoutError, match="absolute deadline"):
        connection.connect()
    assert wire_socket.closed is True
    assert context.wrapped is False


def test_only_stdlib_import_roots_are_in_paid_transport_module() -> None:
    source_path = Path(responses_transport.__file__)
    tree = ast.parse(source_path.read_text())
    roots = set()
    relative_targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                roots.add(node.module.split(".", 1)[0])
            elif node.module:  # from .pkg import x
                relative_targets.add(node.module.split(".", 1)[0])
            else:  # from . import x
                relative_targets.update(a.name.split(".", 1)[0] for a in node.names)
    # The only permitted intra-package import on the paid path is the shared
    # stdlib-only base module; any other relative import smuggles a heavy sibling.
    assert relative_targets <= {"bedrock_transport_base"}, (
        f"unexpected relative import(s) in paid responses transport: {sorted(relative_targets)}"
    )
    assert roots <= {
        "__future__",
        "base64",
        "dataclasses",
        "datetime",
        "email",
        "hashlib",
        "http",
        "json",
        "math",
        "pathlib",
        "re",
        "socket",
        "ssl",
        "threading",
        "time",
        "typing",
        "urllib",
    }


def test_only_stdlib_import_roots_are_in_bedrock_transport_base() -> None:
    # The paid Responses lane reaches the network through bedrock_transport_base
    # (`from . import bedrock_transport_base as _base`), so the base module
    # carries the same stdlib-only invariant as the lane module — and it is
    # a LEAF: it must import only stdlib and NO intra-package sibling. A relative
    # pull of a heavy sibling (`from . import model_client`, `from .scorers import
    # x`) would smuggle openai/indra/gilda into the paid path, so relative imports
    # of ANY form (level>0, whether or not node.module is set) are rejected here.
    tree = ast.parse(Path(transport_base.__file__).read_text())
    roots = set()
    relative_targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                roots.add(node.module.split(".", 1)[0])
            elif node.module:  # from .pkg import x
                relative_targets.add(node.module.split(".", 1)[0])
            else:  # from . import x, y
                relative_targets.update(a.name.split(".", 1)[0] for a in node.names)
    assert relative_targets == set(), (
        "bedrock_transport_base must be a stdlib-only leaf; found relative "
        f"import(s): {sorted(relative_targets)}"
    )
    assert roots <= {
        "__future__",
        "base64",
        "dataclasses",
        "datetime",
        "email",
        "hashlib",
        "http",
        "json",
        "math",
        "pathlib",
        "re",
        "socket",
        "ssl",
        "threading",
        "time",
        "typing",
        "urllib",
    }


def test_all_formal_gemma_configs_use_the_frozen_raw_transport() -> None:
    expected_models = {
        "bedrock-gemma-4-e2b": "google.gemma-4-e2b",
        "bedrock-gemma-4-26b": "google.gemma-4-26b-a4b",
        "bedrock-gemma-4-31b": "google.gemma-4-31b",
    }
    for name, model_id in expected_models.items():
        config = LOCAL_MODELS[name]
        assert config["backend"] == BACKEND_NAME
        assert config["model_id"] == model_id
        assert config["responses_endpoint"] == FORMAL_RESPONSES_ENDPOINT
        assert config["expected_responses_endpoint"] == FORMAL_RESPONSES_ENDPOINT
        assert config["tls_ca_bundle"] == "/private/etc/ssl/cert.pem"
        assert config["tls_ca_bundle_sha256"] == CA_SHA256
        assert config["max_request_bytes"] == 16 * 1024 * 1024
        assert config["max_response_bytes"] == 16 * 1024 * 1024


def test_formal_gemma_initialization_does_not_import_openai_sdk(monkeypatch) -> None:
    ca = Path("/private/etc/ssl/cert.pem")
    if not ca.is_file():
        pytest.skip("formal macOS CA bundle is not present on this platform")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)
    before = set(sys.modules)
    client = ModelClient("bedrock-gemma-4-e2b")
    imported = set(sys.modules) - before
    assert client.backend == BACKEND_NAME
    assert not any(name == "openai" or name.startswith("openai.") for name in imported)
    assert not any(name == "httpx" or name.startswith("httpx.") for name in imported)


def test_formal_modelclient_accepts_pipe_token_and_sealed_ca_without_env(
    tmp_path, monkeypatch
) -> None:
    host_ca = Path("/private/etc/ssl/cert.pem")
    if not host_ca.is_file():
        pytest.skip("formal macOS CA bundle is not present on this platform")
    sealed_ca = tmp_path / "certs" / "cacert.pem"
    sealed_ca.parent.mkdir()
    sealed_ca.write_bytes(host_ca.read_bytes())
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", "/ambient/attacker.pem")
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient-proxy.invalid:8080")

    client = ModelClient(
        "bedrock-gemma-4-e2b",
        bedrock_bearer_token=TOKEN,
        bedrock_ca_bundle=str(sealed_ca),
        bedrock_ca_bundle_sha256=CA_SHA256,
    )
    trace = client._bedrock_responses_transport.request_trace(_body())
    assert client.backend == BACKEND_NAME
    assert trace["tls_ca_bundle_sha256"] == CA_SHA256
    assert trace["tls_ca_load_mode"] == "cadata"
    assert trace["environment_proxies_allowed"] is False
    assert trace["ambient_tls_trust_allowed"] is False
    assert TOKEN not in json.dumps(trace)


def test_explicit_sealed_ca_hash_mismatch_fails_before_request(tmp_path, monkeypatch) -> None:
    host_ca = Path("/private/etc/ssl/cert.pem")
    if not host_ca.is_file():
        pytest.skip("formal macOS CA bundle is not present on this platform")
    sealed_ca = tmp_path / "cacert.pem"
    sealed_ca.write_bytes(host_ca.read_bytes())
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    with pytest.raises(ValueError, match="frozen SHA-256"):
        ModelClient(
            "bedrock-gemma-4-e2b",
            bedrock_bearer_token=TOKEN,
            bedrock_ca_bundle=str(sealed_ca),
            bedrock_ca_bundle_sha256="0" * 64,
        )


def test_explicit_bedrock_material_is_not_silently_ignored_by_other_backends() -> None:
    with pytest.raises(ValueError, match="require a Bedrock backend"):
        ModelClient("local-gemma-4-26b", bedrock_bearer_token=TOKEN)


def _model_client(endpoint: str) -> ModelClient:
    client = ModelClient.__new__(ModelClient)
    client.model_name = "bedrock-gemma-4-e2b-fixture"
    client.backend = BACKEND_NAME
    client.config = {
        "model_id": MODEL_ID,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "max_tokens": 16000,
        "timeout": 2,
    }
    client._tls = threading.local()
    client._bedrock_responses_transport = _transport(endpoint)
    return client


def test_modelclient_preserves_api_reasoning_and_transport_trace(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _payload_bytes(_responses_payload()))
    client = _model_client(endpoint)
    response = client.call(
        system="sys",
        messages=[{"role": "user", "content": "question"}],
        max_tokens=91,
        temperature=0.731,
        kind="monolithic",
    )
    assert response.raw_text == "private reasoning\n" + response.content
    assert response.reasoning_trace["status"] == ReasoningStatus.PLAINTEXT
    assert response.reasoning_trace["backend"] == BACKEND_NAME
    rows = client.pop_call_log()
    assert len(rows) == 1
    assert rows[0]["kind"] == "monolithic"
    assert rows[0]["transport_trace"] == response.transport_trace
    assert TOKEN not in json.dumps(rows)
    sent = json.loads(state.requests[0]["body"])
    assert sent["reasoning"] == {"effort": "high"}
    assert "temperature" not in sent


def test_modelclient_relation_override_preserves_legacy_ignored_format(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _payload_bytes(_responses_payload(reasoning=None)))
    client = _model_client(endpoint)
    client.call(
        system="relation",
        messages=[{"role": "user", "content": "question"}],
        max_tokens=3000,
        temperature=0.1,
        response_format={"type": "json_object"},
        reasoning_effort="none",
        kind="relation_nature",
    )
    sent = json.loads(state.requests[0]["body"])
    assert "reasoning" not in sent
    assert "response_format" not in sent
    assert "temperature" not in sent


def test_modelclient_encrypted_reasoning_status_matches_legacy(local_server) -> None:
    state, endpoint = local_server
    payload = {
        "id": "resp-encrypted-fixture",
        "object": "response",
        "model": MODEL_ID,
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 1847},
        },
        "output": [
            {"type": "reasoning", "summary": [], "id": "r"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}],
            },
        ],
    }
    state.push(200, _payload_bytes(payload))
    response = _model_client(endpoint).call(system="sys", messages=[])
    assert response.reasoning == ""
    assert response.content == "answer"
    assert response.reasoning_trace["status"] == ReasoningStatus.ENCRYPTED
    assert response.reasoning_trace["reasoning_tokens"] == 1847


def test_modelclient_error_log_has_status_and_hash_without_provider_body(local_server) -> None:
    state, endpoint = local_server
    raw = _payload_bytes({"error": f"provider-body-{TOKEN}"})
    state.push(500, raw)
    client = _model_client(endpoint)
    with pytest.raises(BedrockResponsesTransportError):
        client.call(system="sys", messages=[], kind="monolithic")
    rows = client.pop_call_log()
    assert len(rows) == 1
    assert rows[0]["transport_trace"]["response_http_status"] == 500
    assert rows[0]["transport_trace"]["response_body_sha256"] is None
    assert TOKEN not in json.dumps(rows)
    assert "provider-body" not in rows[0]["error_detail"]


def test_modelclient_transport_timeout_retains_request_commitment(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _payload_bytes(_responses_payload()), delay=0.25)
    client = _model_client(endpoint)
    client.config["timeout"] = 0.03
    with pytest.raises(TimeoutError, match="timed out"):
        client.call(system="sys", messages=[], kind="monolithic")
    rows = client.pop_call_log()
    assert len(rows) == 1
    assert rows[0]["transport_trace"]["request_body_sha256"]
    assert rows[0]["transport_trace"]["response_http_status"] is None


def test_modelclient_429_retry_is_disabled_by_paid_guard_flag(local_server, monkeypatch) -> None:
    state, endpoint = local_server
    state.push(429, b"{}", headers={"Retry-After": "0"})
    state.push(200, _payload_bytes(_responses_payload()))
    client = _model_client(endpoint)
    client._spend_guard_disable_internal_retries = True
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    with pytest.raises(BedrockResponsesTransportError, match="HTTP 429"):
        client.call(system="sys", messages=[], kind="monolithic")
    assert len(state.requests) == 1


def test_modelclient_unguarded_429_preserves_existing_bounded_retry(
    local_server, monkeypatch
) -> None:
    state, endpoint = local_server
    state.push(429, b"{}", headers={"Retry-After": "0"})
    state.push(200, _payload_bytes(_responses_payload()))
    client = _model_client(endpoint)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    result = client.call(system="sys", messages=[], kind="monolithic")
    assert result.content
    assert len(state.requests) == 2
    assert len(client.pop_call_log()) == 1
