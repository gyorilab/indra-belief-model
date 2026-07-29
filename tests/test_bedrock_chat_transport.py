"""Adversarial, network-local tests for the formal GLM-5 raw transport."""
from __future__ import annotations

import ast
import base64
from collections import deque
import copy
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

from indra_belief import bedrock_chat_transport as chat_transport
from indra_belief import bedrock_transport_base as transport_base
from indra_belief.bedrock_chat_transport import (
    BACKEND_NAME,
    FORMAL_CHAT_COMPLETIONS_ENDPOINT,
    BedrockChatConnectionError,
    BedrockChatTLSVerificationError,
    BedrockChatTransportError,
    RawBedrockChatTransport,
    build_bedrock_chat_body,
    build_pinned_https_opener,
    canonical_json_bytes,
    canonical_json_sha256,
    parse_bedrock_chat_response_preimage,
    validate_transport_trace,
    verify_transport_response_preimage,
)
from indra_belief.model_client import ModelClient, ReasoningStatus
from indra_belief.model_client import LOCAL_MODELS
from indra_belief.spend_guard import classify_provider_failure


MODEL_ID = "zai.glm-5"
TOKEN = "fixture-secret-bearer"
CA_SHA256 = "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994"


def _completion(
    *,
    content='{"verdict":"correct","confidence":"high"}',
    reasoning_content="private reasoning",
    usage=None,
    finish_reason="stop",
) -> bytes:
    if usage is None:
        usage = {
            "prompt_tokens": 17,
            "completion_tokens": 23,
            "total_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 11},
        }
    return json.dumps(
        {
            "id": "chatcmpl-fixture",
            "object": "chat.completion",
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning_content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


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
                declared_length = headers.get("Content-Length", str(len(response_body)))
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
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
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
):
    return RawBedrockChatTransport(
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
    return build_bedrock_chat_body(
        model_id=MODEL_ID,
        system="system µ",
        messages=[{"role": "user", "content": "evidence"}],
        max_tokens=3000 if relation else 32000,
        temperature=0.1,
        response_format={"type": "json_object"} if relation else None,
        reasoning_effort="none" if relation else "high",
    )


def _offline_formal_transport() -> RawBedrockChatTransport:
    """Build trace state only; this object has no connection capability."""

    transport = RawBedrockChatTransport.__new__(RawBedrockChatTransport)
    transport.endpoint = FORMAL_CHAT_COMPLETIONS_ENDPOINT
    transport.expected_model_id = MODEL_ID
    transport._token = TOKEN
    transport._max_request_bytes = chat_transport.DEFAULT_MAX_REQUEST_BYTES
    transport._max_response_bytes = chat_transport.DEFAULT_MAX_RESPONSE_BYTES
    transport._tls_ca_bundle_sha256 = CA_SHA256
    transport._tls_ca_load_mode = "cadata"
    return transport


def _offline_formal_success_trace() -> tuple[dict, bytes, str]:
    request_raw = canonical_json_bytes(_body())
    response_raw = _completion()
    payload = json.loads(response_raw)
    trace = _offline_formal_transport()._trace(
        request_raw=request_raw,
        status=200,
        response_raw=response_raw,
        response_json_sha256=canonical_json_sha256(payload),
        response_content_length_declared=len(response_raw),
        response_body_complete=True,
        response_framing="content_length",
        response_framing_valid=True,
        response_representation_valid=True,
    )
    trace["provider_request_id"] = "fixture-aws-request-id"
    trace["provider_response_id"] = payload["id"]
    trace["provider_response_model"] = payload["model"]
    return trace, response_raw, hashlib.sha256(request_raw).hexdigest()


@pytest.mark.parametrize(
    ("relation", "expected_sha256"),
    [
        (False, "7c57e5b963982bcc9aef78d2d90b5bcbd5ebd71b840a3f4b223fdcdc6a986c48"),
        (True, "4bb106a397cda762c8765fcee654fc0ed5934b5a52dfc7e43faf0d721bcc533f"),
    ],
    ids=["main", "relation"],
)
def test_request_and_response_semantics_match_current_openai_sdk(
    local_server, relation: bool, expected_sha256: str
) -> None:
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    observed: list[bytes] = []
    response_raw = _completion()

    def handler(request):
        observed.append(request.content)
        return httpx.Response(200, content=response_raw)

    client = openai.OpenAI(
        base_url="http://fixture.invalid/v1",
        api_key="fixture",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    kwargs = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "system µ"},
            {"role": "user", "content": "evidence"},
        ],
        "max_tokens": 3000 if relation else 32000,
        "temperature": 0.1,
        "timeout": 30,
        "extra_body": {"reasoning_effort": "none" if relation else "high"},
    }
    if relation:
        kwargs["response_format"] = {"type": "json_object"}
    sdk_response = client.chat.completions.create(**kwargs)

    sdk_body = json.loads(observed[0])
    raw_body = _body(relation=relation)
    assert sdk_body == raw_body
    assert canonical_json_sha256(sdk_body) == canonical_json_sha256(raw_body)
    assert canonical_json_sha256(raw_body) == expected_sha256

    state, endpoint = local_server
    state.push(200, response_raw)
    raw_response = _transport(endpoint).call(raw_body, timeout=2)
    sdk_message = sdk_response.choices[0].message
    assert raw_response.content == sdk_message.content
    assert raw_response.reasoning == sdk_message.reasoning_content
    assert raw_response.prompt_tokens == sdk_response.usage.prompt_tokens
    assert raw_response.output_tokens == sdk_response.usage.completion_tokens
    assert (
        raw_response.reasoning_tokens
        == sdk_response.usage.completion_tokens_details.reasoning_tokens
    )
    assert raw_response.finish_reason == sdk_response.choices[0].finish_reason


def test_canonical_request_is_strict_deterministic_utf8() -> None:
    a = {"z": "µ", "a": {"b": 2, "a": 1}}
    b = {"a": {"a": 1, "b": 2}, "z": "µ"}
    assert canonical_json_bytes(a) == b'{"a":{"a":1,"b":2},"z":"\xc2\xb5"}'
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        canonical_json_bytes({"bad": "\ud800"})


def test_success_records_exact_route_headers_and_wire_commitments(local_server) -> None:
    state, endpoint = local_server
    raw_response = _completion()
    state.push(200, raw_response)
    body = _body()
    result = _transport(endpoint).call(body, timeout=2)

    assert result.content.startswith('{"verdict"')
    assert result.reasoning == "private reasoning"
    assert result.prompt_tokens == 17
    assert result.output_tokens == 23
    assert result.reasoning_tokens == 11
    assert result.finish_reason == "stop"
    assert len(state.requests) == 1
    request = state.requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert request["headers"]["Accept-Encoding"] == "identity"
    assert request["body"] == canonical_json_bytes(body)

    trace = result.transport_trace
    assert trace["backend"] == BACKEND_NAME
    assert trace["request_body_sha256"] == hashlib.sha256(request["body"]).hexdigest()
    assert trace["response_body_sha256"] == hashlib.sha256(raw_response).hexdigest()
    assert trace["response_json_sha256"] == canonical_json_sha256(
        json.loads(raw_response)
    )
    assert trace["response_http_status"] == 200
    assert trace["response_content_length_declared"] == len(raw_response)
    assert trace["response_body_complete"] is True
    assert trace["response_body_preimage_kind"] == "complete"
    assert trace["response_body_preimage_redacted"] is False
    assert base64.b64decode(trace["response_body_preimage_b64"]) == raw_response
    assert verify_transport_response_preimage(trace) == raw_response
    replayed = parse_bedrock_chat_response_preimage(
        trace, expected_model_id=MODEL_ID
    )
    assert replayed.content == result.content
    assert replayed.reasoning == result.reasoning
    assert result.provider_model == MODEL_ID
    assert result.provider_response_id == "chatcmpl-fixture"
    assert trace["redirects_allowed"] is False
    assert trace["environment_proxies_allowed"] is False
    assert trace["ambient_tls_trust_allowed"] is False
    assert TOKEN not in json.dumps(trace)


def test_offline_formal_trace_validator_returns_exact_success_preimage() -> None:
    trace, response_raw, request_sha = _offline_formal_success_trace()
    assert validate_transport_trace(
        trace,
        expected_request_sha256=request_sha,
        expected_ca_bundle_sha256=CA_SHA256,
        successful=True,
    ) == response_raw


def test_offline_formal_trace_validator_has_an_exact_key_census() -> None:
    trace, _response_raw, request_sha = _offline_formal_success_trace()
    for key in tuple(trace):
        mutated = copy.deepcopy(trace)
        del mutated[key]
        with pytest.raises(ValueError, match="key census"):
            validate_transport_trace(
                mutated,
                expected_request_sha256=request_sha,
                expected_ca_bundle_sha256=CA_SHA256,
                successful=True,
            )
    injected = copy.deepcopy(trace)
    injected["extra_claim"] = "not transport evidence"
    with pytest.raises(ValueError, match="key census"):
        validate_transport_trace(
            injected,
            expected_request_sha256=request_sha,
            expected_ca_bundle_sha256=CA_SHA256,
            successful=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("backend", "forged_backend"),
        ("method", "GET"),
        ("endpoint", "https://attacker.invalid/v1/chat/completions"),
        ("expected_model_id", "zai.other"),
        ("request_body_bytes", 0),
        ("request_body_sha256", "0" * 64),
        ("response_http_status", 599),
        ("response_body_bytes", 1),
        ("response_body_sha256", "0" * 64),
        ("response_json_sha256", "0" * 64),
        ("response_content_length_declared", 1),
        ("response_framing", "chunked"),
        ("response_framing_valid", False),
        ("response_representation_valid", False),
        ("transport_failure_class", "http_status"),
        ("max_request_bytes", 1),
        ("max_response_bytes", 1),
        ("tls_ca_bundle_sha256", "0" * 64),
        ("tls_ca_load_mode", "ambient_default"),
        ("redirects_allowed", True),
        ("environment_proxies_allowed", True),
        ("ambient_tls_trust_allowed", True),
        ("provider_request_id", "bad\nrequest-id"),
        ("provider_response_id", "forged-response"),
        ("provider_response_model", "zai.other"),
    ],
)
def test_offline_formal_trace_validator_rejects_adversarial_success_mutations(
    field: str, value,
) -> None:
    trace, _response_raw, request_sha = _offline_formal_success_trace()
    trace[field] = value
    with pytest.raises(ValueError):
        validate_transport_trace(
            trace,
            expected_request_sha256=request_sha,
            expected_ca_bundle_sha256=CA_SHA256,
            successful=True,
        )


def test_offline_formal_trace_validator_accepts_exact_error_evidence_shapes() -> None:
    transport = _offline_formal_transport()
    request_raw = canonical_json_bytes(_body())
    request_sha = hashlib.sha256(request_raw).hexdigest()
    error_raw = canonical_json_bytes({"error": "provider failure"})
    complete = transport._trace(
        request_raw=request_raw,
        status=500,
        response_raw=error_raw,
        response_json_sha256=canonical_json_sha256({"error": "provider failure"}),
        response_content_length_declared=len(error_raw),
        response_body_complete=True,
        response_framing="content_length",
        response_framing_valid=True,
        transport_failure_class="http_status",
    )
    assert validate_transport_trace(
        complete,
        expected_request_sha256=request_sha,
        expected_ca_bundle_sha256=CA_SHA256,
        successful=False,
    ) == error_raw

    prefix = b'{"partial":'
    partial = transport._trace(
        request_raw=request_raw,
        status=200,
        response_raw=prefix,
        response_content_length_declared=len(prefix) + 20,
        response_body_complete=False,
        response_framing="content_length",
        response_framing_valid=True,
        transport_failure_class="absolute_timeout",
    )
    assert validate_transport_trace(
        partial,
        expected_request_sha256=request_sha,
        expected_ca_bundle_sha256=CA_SHA256,
        successful=False,
    ) is None

    absent = transport._trace(
        request_raw=request_raw,
        status=None,
        response_raw=None,
        response_body_complete=False,
        transport_failure_class="tls_hostname_verification",
    )
    assert validate_transport_trace(
        absent,
        expected_request_sha256=request_sha,
        expected_ca_bundle_sha256=CA_SHA256,
        successful=False,
    ) is None

    bearer_raw = f'{{"error":"{TOKEN}"}}'.encode()
    bearer_omitted = transport._trace(
        request_raw=request_raw,
        status=500,
        response_raw=bearer_raw,
        response_content_length_declared=len(bearer_raw),
        response_body_complete=True,
        response_framing="content_length",
        response_framing_valid=True,
        transport_failure_class="http_status",
    )
    assert bearer_omitted["response_body_preimage_redacted"] is True
    assert validate_transport_trace(
        bearer_omitted,
        expected_request_sha256=request_sha,
        expected_ca_bundle_sha256=CA_SHA256,
        successful=False,
    ) is None


def test_offline_formal_trace_validator_rejects_forged_error_semantics() -> None:
    success, _response_raw, request_sha = _offline_formal_success_trace()
    forged_schema_error = copy.deepcopy(success)
    forged_schema_error["provider_response_id"] = None
    forged_schema_error["provider_response_model"] = None
    forged_schema_error["transport_failure_class"] = "invalid_response_schema"
    with pytest.raises(ValueError, match="labels a valid response"):
        validate_transport_trace(
            forged_schema_error,
            expected_request_sha256=request_sha,
            expected_ca_bundle_sha256=CA_SHA256,
            successful=False,
        )

    complete_timeout = copy.deepcopy(forged_schema_error)
    complete_timeout["transport_failure_class"] = "absolute_timeout"
    with pytest.raises(ValueError, match="timeout"):
        validate_transport_trace(
            complete_timeout,
            expected_request_sha256=request_sha,
            expected_ca_bundle_sha256=CA_SHA256,
            successful=False,
        )


def test_content_reasoning_usage_and_finish_variants(local_server) -> None:
    state, endpoint = local_server
    state.push(
        200,
        _completion(
            content=[{"type": "text", "text": "part-a"}, "part-b"],
            reasoning_content=[{"text": "r1"}, {"content": "r2"}],
            usage={"input_tokens": 5, "output_tokens": 7},
            finish_reason=None,
        ),
    )
    result = _transport(endpoint).call(_body(), timeout=2)
    assert result.content == "part-apart-b"
    assert result.reasoning == "r1r2"
    assert result.prompt_tokens == 5
    assert result.output_tokens == 7
    assert result.reasoning_tokens == -1
    assert result.finish_reason == "stop"


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b'{"choices":[],"choices":[]}',
        b'{"x":NaN}',
        b"\xff\xfe",
        b'{"choices":[]}',
        b'[]',
    ],
    ids=["syntax", "duplicate", "nan", "utf8", "no-choice", "not-object"],
)
def test_malformed_or_ambiguous_response_fails_with_raw_hash(local_server, raw) -> None:
    state, endpoint = local_server
    state.push(200, raw)
    with pytest.raises(BedrockChatTransportError) as caught:
        _transport(endpoint).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_http_status"] == 200
    assert trace["response_body_sha256"] == hashlib.sha256(raw).hexdigest()
    if raw == b'{"choices":[]}':
        assert trace["response_json_sha256"] == canonical_json_sha256(
            {"choices": []}
        )
    else:
        assert trace["response_json_sha256"] is None
    assert verify_transport_response_preimage(trace) == raw
    decoded = raw.decode("utf-8", errors="ignore")
    if decoded:
        assert decoded not in str(caught.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 503])
def test_http_errors_are_single_attempt_hash_only(local_server, status: int) -> None:
    state, endpoint = local_server
    raw = f'{{"error":"provider secret detail {status}"}}'.encode()
    state.push(status, raw)
    with pytest.raises(BedrockChatTransportError, match=rf"HTTP {status}") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    assert len(state.requests) == 1
    assert "provider secret detail" not in str(caught.value)
    assert caught.value.transport_trace["response_body_sha256"] == hashlib.sha256(raw).hexdigest()


def test_429_retry_after_is_normalized_for_modelclient_retry_parser(local_server) -> None:
    state, endpoint = local_server
    state.push(429, b'{"error":"slow"}', headers={"Retry-After": "2.5"})
    with pytest.raises(BedrockChatTransportError, match=r"HTTP 429; retry in 2.5s"):
        _transport(endpoint).call(_body(), timeout=2)


def test_redirect_is_never_followed(local_server) -> None:
    state, endpoint = local_server
    state.push(302, b"", headers={"Location": endpoint})
    with pytest.raises(BedrockChatTransportError, match="HTTP 302"):
        _transport(endpoint).call(_body(), timeout=2)
    assert len(state.requests) == 1


def test_response_size_bound_fails_without_partial_body_commitment(local_server) -> None:
    state, endpoint = local_server
    state.push(200, b"x" * 257)
    with pytest.raises(BedrockChatTransportError, match="safety limit") as caught:
        _transport(endpoint, maximum=256).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_size_exceeded"] is True
    assert trace["response_body_sha256"] is None
    assert trace["response_body_bytes"] is None
    assert trace["response_content_length_declared"] == 257


def test_compressed_response_is_rejected_but_committed(local_server) -> None:
    state, endpoint = local_server
    raw = _completion()
    state.push(200, raw, headers={"Content-Encoding": "gzip"})
    with pytest.raises(BedrockChatTransportError, match="content encoding") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    assert caught.value.transport_trace["response_body_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    ("headers", "match"),
    [
        ({"Content-Encoding": "br"}, "content encoding"),
        ({"Content-Type": "text/plain"}, "not application/json"),
        ({"Content-Type": None}, "no Content-Type"),
        (
            {"Content-Type": "application/json; charset=iso-8859-1"},
            "malformed or unsupported",
        ),
        (
            {"Content-Type": "application/json; profile=x"},
            "malformed or unsupported",
        ),
        (
            {"Content-Type": "application/json; charset=utf-8; charset=utf8"},
            "malformed or unsupported",
        ),
    ],
)
def test_representation_headers_fail_closed_with_complete_preimage(
    local_server, headers: dict, match: str
) -> None:
    state, endpoint = local_server
    raw = _completion()
    state.push(200, raw, headers=headers)
    with pytest.raises(BedrockChatTransportError, match=match) as caught:
        _transport(endpoint).call(_body(), timeout=2)
    assert verify_transport_response_preimage(caught.value.transport_trace) == raw


def test_utf8_charset_spelling_is_accepted(local_server) -> None:
    state, endpoint = local_server
    state.push(
        200,
        _completion(),
        headers={"Content-Type": 'application/json; charset="UTF-8"'},
    )
    assert _transport(endpoint).call(_body(), timeout=2).content


def test_ambiguous_transfer_encoding_and_content_length_is_rejected(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _completion(), headers={"Transfer-Encoding": "chunked"})
    with pytest.raises(BedrockChatTransportError, match="invalid response framing") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_http_status"] == 200
    assert trace["response_framing_valid"] is False
    assert trace["response_body_preimage_b64"] is None


@pytest.mark.parametrize(
    "values",
    [
        {"transfer-encoding": ["gzip"]},
        {"transfer-encoding": ["chunked", "chunked"]},
        {"content-length": ["1", "1"]},
        {"content-length": ["+1"]},
        {"content-length": ["1, 1"]},
    ],
)
def test_framing_header_parser_rejects_ambiguous_or_invalid_values(values) -> None:
    class Headers:
        def get_all(self, name):
            return values.get(name.lower())

        def get(self, name):
            found = self.get_all(name)
            return found[0] if found else None

    with pytest.raises(ValueError):
        chat_transport._response_framing(Headers())


def test_short_content_length_is_retained_only_as_partial_prefix(local_server) -> None:
    state, endpoint = local_server
    raw = b'{"short":true}'
    state.push(200, raw, headers={"Content-Length": str(len(raw) + 10)})
    with pytest.raises(BedrockChatTransportError, match="invalid response framing") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_body_complete"] is False
    assert trace["response_body_preimage_kind"] == "partial_prefix"
    assert trace["response_framing_valid"] is False
    assert trace["response_body_prefix_sha256"] == hashlib.sha256(raw).hexdigest()
    assert base64.b64decode(trace["response_body_preimage_b64"]) == raw
    with pytest.raises(ValueError, match="no complete response preimage"):
        verify_transport_response_preimage(trace)


def test_request_size_is_bounded_before_any_side_effect(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _completion())
    with pytest.raises(ValueError, match="request exceeds"):
        _transport(endpoint, request_maximum=16).call(_body(), timeout=2)
    assert state.requests == []


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
def test_response_preimage_verifier_fails_closed(field: str, value, match: str) -> None:
    raw = _completion()
    trace = {
        "response_body_preimage_kind": "complete",
        "response_body_complete": True,
        "response_framing_valid": True,
        "response_size_exceeded": False,
        "response_body_preimage_redacted": False,
        "response_body_preimage_b64": base64.b64encode(raw).decode("ascii"),
        "response_body_bytes": len(raw),
        "response_body_sha256": hashlib.sha256(raw).hexdigest(),
        "response_json_sha256": canonical_json_sha256(json.loads(raw)),
    }
    trace[field] = value
    with pytest.raises(ValueError, match=match):
        verify_transport_response_preimage(trace)


def test_provider_body_containing_bearer_is_redacted_and_not_exact_replay(local_server) -> None:
    state, endpoint = local_server
    raw = f'{{"error":"Bearer {TOKEN}"}}'.encode()
    state.push(500, raw)
    with pytest.raises(BedrockChatTransportError, match="HTTP 500") as caught:
        _transport(endpoint).call(_body(), timeout=2)
    trace = caught.value.transport_trace
    assert trace["response_body_preimage_redacted"] is True
    assert trace["response_body_sha256"] is None
    assert trace["response_body_preimage_b64"] is None
    with pytest.raises(ValueError, match="no complete"):
        verify_transport_response_preimage(trace)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.pop("model"), "frozen model"),
        (lambda payload: payload.update(model="zai.other"), "frozen model"),
        (lambda payload: payload.update(object="response"), "object type"),
        (lambda payload: payload.update(id=""), "response id"),
        (lambda payload: payload.update(choices=[]), "exactly one"),
        (
            lambda payload: payload["choices"][0].update(index=1),
            "index must be zero",
        ),
        (
            lambda payload: payload["choices"][0]["message"].update(role="user"),
            "assistant message",
        ),
        (lambda payload: payload.update(usage=None), "usage must be an object"),
        (
            lambda payload: payload["usage"].update(prompt_tokens=True),
            "non-negative integer",
        ),
        (
            lambda payload: payload["usage"].pop("completion_tokens"),
            "completion_tokens is missing",
        ),
    ],
)
def test_strict_response_envelope_fails_with_replayable_preimage(
    local_server, mutation, match: str
) -> None:
    state, endpoint = local_server
    payload = json.loads(_completion())
    mutation(payload)
    raw = json.dumps(payload, separators=(",", ":")).encode()
    state.push(200, raw)
    with pytest.raises(BedrockChatTransportError, match=match) as caught:
        _transport(endpoint).call(_body(), timeout=2)
    assert verify_transport_response_preimage(caught.value.transport_trace) == raw


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body.update(model="zai.other"),
        lambda body: body.update(unknown=True),
        lambda body: body.update(max_tokens=True),
        lambda body: body.update(temperature=math.inf),
        lambda body: body.update(response_format="json"),
        lambda body: body.update(messages=[]),
    ],
)
def test_transport_revalidates_frozen_request_before_side_effect(local_server, mutate) -> None:
    state, endpoint = local_server
    state.push(200, _completion())
    body = _body()
    mutate(body)
    with pytest.raises(ValueError):
        _transport(endpoint).call(body, timeout=2)
    assert state.requests == []


def test_timeout_has_request_commitment(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _completion(), delay=0.25)
    with pytest.raises(TimeoutError, match="timed out") as caught:
        _transport(endpoint).call(_body(), timeout=0.03)
    trace = caught.value.transport_trace
    assert trace["request_body_sha256"] == canonical_json_sha256(_body())
    assert trace["response_http_status"] is None


def test_absolute_deadline_rejects_slow_drip_and_retains_partial_prefix(
    local_server,
) -> None:
    state, endpoint = local_server
    raw = _completion()
    state.push(200, raw, chunk_delay=0.01)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out") as caught:
        _transport(endpoint).call(_body(), timeout=0.06)
    assert time.monotonic() - started < 0.5
    trace = caught.value.transport_trace
    assert trace["response_http_status"] == 200
    assert trace["response_body_complete"] is False
    assert trace["response_body_preimage_kind"] == "partial_prefix"
    prefix = base64.b64decode(trace["response_body_preimage_b64"])
    assert prefix
    assert raw.startswith(prefix)
    assert hashlib.sha256(prefix).hexdigest() == trace["response_body_prefix_sha256"]
    assert trace["response_body_sha256"] is None
    assert trace["transport_failure_class"] == "absolute_timeout"


def test_dns_resolution_cannot_escape_absolute_deadline(local_server, monkeypatch) -> None:
    state, endpoint = local_server
    state.push(200, _completion())
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
    assert caught.value.transport_trace["transport_failure_class"] == "absolute_timeout"
    assert state.requests == []


@pytest.mark.parametrize(
    ("detail", "failure_class", "message"),
    [
        (
            "certificate verify failed: unable to get local issuer",
            "tls_certificate_verification",
            "certificate verification failed",
        ),
        (
            "certificate verify failed: Hostname mismatch",
            "tls_hostname_verification",
            "hostname verification failed",
        ),
    ],
)
def test_tls_verification_failures_are_terminal_classified_and_redacted(
    local_server, detail: str, failure_class: str, message: str
) -> None:
    _state, endpoint = local_server
    transport = _transport(endpoint)

    class BadCertificateConnection:
        sock = None

        def connect(self):
            raise ssl.SSLCertVerificationError(f"{detail}: {TOKEN}")

        def close(self):
            return None

    transport._connection_factory = lambda _deadline, _abort: BadCertificateConnection()
    with pytest.raises(BedrockChatTLSVerificationError, match=message) as caught:
        transport.call(_body(), timeout=2)
    assert not isinstance(caught.value, ConnectionError)
    assert classify_provider_failure(caught.value) == ("other", None)
    assert caught.value.transport_trace["transport_failure_class"] == failure_class
    assert TOKEN not in str(caught.value)
    assert TOKEN not in "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )


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

        def __init__(self):
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
    with pytest.raises(BedrockChatConnectionError) as caught:
        transport.call(_body(), timeout=2)
    assert classify_provider_failure(caught.value) == ("transport_or_server", None)
    trace = caught.value.transport_trace
    assert trace["response_http_status"] == 200
    assert trace["response_body_complete"] is False
    assert trace["response_framing_valid"] is False
    assert trace["response_body_prefix_sha256"] == hashlib.sha256(partial).hexdigest()
    assert base64.b64decode(trace["response_body_preimage_b64"]) == partial
    assert TOKEN not in "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )


def test_connection_error_redacts_bearer_token(local_server) -> None:
    _state, endpoint = local_server
    transport = _transport(endpoint)

    class BrokenConnection:
        sock = None

        def connect(self):
            raise OSError(f"Bearer {TOKEN}; {TOKEN}")

        def close(self):
            return None

    transport._connection_factory = lambda _deadline, _abort: BrokenConnection()
    with pytest.raises(BedrockChatConnectionError) as caught:
        transport.call(_body(), timeout=2)
    assert TOKEN not in str(caught.value)
    assert TOKEN not in json.dumps(caught.value.transport_trace)


def test_unexpected_transport_error_also_redacts_bearer_token(local_server) -> None:
    _state, endpoint = local_server
    transport = _transport(endpoint)

    class BrokenConnection:
        sock = None

        def connect(self):
            raise RuntimeError(f"unexpected Bearer {TOKEN}; {TOKEN}")

        def close(self):
            return None

    transport._connection_factory = lambda _deadline, _abort: BrokenConnection()
    with pytest.raises(BedrockChatTransportError) as caught:
        transport.call(_body(), timeout=2)
    assert TOKEN not in str(caught.value)
    assert caught.value.transport_trace["request_body_sha256"] == canonical_json_sha256(_body())


@pytest.mark.parametrize(
    ("endpoint", "allow", "match"),
    [
        ("http://example.com/v1/chat/completions", False, "formal frozen route"),
        ("https://example.com/v1/chat/completions", False, "formal frozen route"),
        ("https://bedrock-mantle.us-east-1.api.aws/v1/responses", False, "path"),
        ("http://example.com/v1/chat/completions", True, "loopback"),
        ("http://user:pass@127.0.0.1/v1/chat/completions", True, "credentials"),
        ("http://127.0.0.1/v1/chat/completions?q=x", True, "query"),
    ],
)
def test_endpoint_policy_fails_closed(endpoint: str, allow: bool, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RawBedrockChatTransport(
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
        RawBedrockChatTransport(
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
    ["", "with space", "with\nnewline", "with\rreturn", "with\ttab", "unicode-µ", "x" * 16385],
)
def test_bearer_token_must_be_header_safe(local_server, token: str) -> None:
    _state, endpoint = local_server
    with pytest.raises(ValueError, match="header-safe"):
        RawBedrockChatTransport(
            endpoint=endpoint,
            expected_endpoint=endpoint,
            expected_model_id=MODEL_ID,
            bearer_token=token,
            ca_bundle=None,
            expected_ca_bundle_sha256=None,
            allow_insecure_http_for_tests=True,
        )


def test_formal_bedrock_opener_ignores_proxy_and_tls_environment(monkeypatch) -> None:
    ca = Path("/private/etc/ssl/cert.pem")
    if not ca.is_file():
        pytest.skip("formal macOS CA bundle is not present on this platform")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid:8888")
    monkeypatch.setenv("SSL_CERT_FILE", "/definitely/not/the/formal/ca.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "/definitely/not/the/formal/ca-dir")
    opener, ca_sha256 = build_pinned_https_opener(ca)
    proxy_handlers = [
        handler
        for handler in opener.handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    # ProxyHandler({}) suppresses build_opener's ambient default. Because the
    # explicit mapping is empty it contributes no *_open methods and is omitted
    # from the final handler list altogether.
    assert proxy_handlers == []
    assert ca_sha256 == hashlib.sha256(ca.read_bytes()).hexdigest()
    assert ca_sha256 == "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994"
    with pytest.raises(ValueError, match="frozen SHA-256"):
        build_pinned_https_opener(ca, expected_ca_bundle_sha256="0" * 64)


def test_cadata_capability_falls_back_only_to_the_same_explicit_path(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "ca.pem"
    path.write_bytes(b"fixture explicit pem bytes\n")
    contexts = []

    class FakeContext:
        def __init__(self, first: bool):
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

    monkeypatch.setattr(chat_transport, "_new_tls_context", new_context)
    _context, digest, mode = chat_transport._explicit_tls_context(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert mode == "cafile_fallback"
    assert contexts[0].calls[0]["cadata"] == path.read_text()
    assert contexts[1].calls == [{"cadata": None, "cafile": str(path)}]


def test_malformed_cadata_does_not_silently_fall_back_to_path(tmp_path) -> None:
    path = tmp_path / "bad-ca.pem"
    path.write_text("not a certificate")
    with pytest.raises(ValueError, match="PEM cadata"):
        chat_transport._explicit_tls_context(path)


def test_direct_https_connection_uses_the_frozen_hostname(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self):
            self.timeouts = []
            self.closed = False

        def settimeout(self, value):
            self.timeouts.append(value)

        def close(self):
            self.closed = True

    class FakeContext:
        def __init__(self):
            self.server_hostname = None

        def wrap_socket(self, sock, *, server_hostname):
            self.server_hostname = server_hostname
            return sock

    wire_socket = FakeSocket()
    context = FakeContext()
    connection = chat_transport._DeadlineHTTPSConnection(
        chat_transport.FORMAL_BEDROCK_HOST,
        443,
        timeout=1,
        context=context,
        absolute_deadline=time.monotonic() + 1,
        abort_event=threading.Event(),
    )
    monkeypatch.setattr(connection, "_resolved_socket", lambda: wire_socket)
    connection.connect()
    assert context.server_hostname == chat_transport.FORMAL_BEDROCK_HOST
    assert connection.sock is wire_socket
    assert wire_socket.timeouts
    connection.close()


def test_watchdog_race_before_tls_wrap_closes_candidate(monkeypatch) -> None:
    class FakeSocket:
        def __init__(self):
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
    connection = chat_transport._DeadlineHTTPSConnection(
        chat_transport.FORMAL_BEDROCK_HOST,
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


def test_only_stdlib_import_roots_are_in_paid_chat_transport() -> None:
    tree = ast.parse(Path(chat_transport.__file__).read_text())
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
        f"unexpected relative import(s) in paid chat transport: {sorted(relative_targets)}"
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
    # Both paid lanes reach the network through bedrock_transport_base
    # (`from . import bedrock_transport_base as _base`), so the base module
    # carries the same stdlib-only invariant as the two lane modules — and it is
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


def test_formal_glm_config_binds_route_model_ca_and_limits() -> None:
    config = LOCAL_MODELS["bedrock-glm-5"]
    assert config["backend"] == BACKEND_NAME
    assert config["model_id"] == MODEL_ID
    assert config["chat_completions_endpoint"].endswith(CHAT_PATH := "/v1/chat/completions")
    assert config["expected_chat_completions_endpoint"].endswith(CHAT_PATH)
    assert config["tls_ca_bundle"] == "/private/etc/ssl/cert.pem"
    assert config["tls_ca_bundle_sha256"] == (
        "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994"
    )
    assert config["max_request_bytes"] == 16 * 1024 * 1024
    assert config["max_response_bytes"] == 16 * 1024 * 1024


def test_formal_glm_modelclient_accepts_pipe_token_and_sealed_ca(
    tmp_path, monkeypatch
) -> None:
    host_ca = Path("/private/etc/ssl/cert.pem")
    if not host_ca.is_file():
        pytest.skip("formal macOS CA bundle is not present on this platform")
    sealed_ca = tmp_path / "certs" / "cacert.pem"
    sealed_ca.parent.mkdir()
    sealed_ca.write_bytes(host_ca.read_bytes())
    ca_sha256 = hashlib.sha256(sealed_ca.read_bytes()).hexdigest()
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", "/ambient/attacker.pem")
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient-proxy.invalid:8080")

    client = ModelClient(
        "bedrock-glm-5",
        bedrock_bearer_token=TOKEN,
        bedrock_ca_bundle=str(sealed_ca),
        bedrock_ca_bundle_sha256=ca_sha256,
    )
    trace = client._bedrock_chat_transport.request_trace(_body())
    assert client.backend == BACKEND_NAME
    assert trace["tls_ca_bundle_sha256"] == ca_sha256
    assert trace["tls_ca_load_mode"] == "cadata"
    assert trace["environment_proxies_allowed"] is False
    assert trace["ambient_tls_trust_allowed"] is False
    assert TOKEN not in json.dumps(trace)


def test_formal_glm_modelclient_rejects_sealed_ca_hash_mismatch(
    tmp_path, monkeypatch
) -> None:
    host_ca = Path("/private/etc/ssl/cert.pem")
    if not host_ca.is_file():
        pytest.skip("formal macOS CA bundle is not present on this platform")
    sealed_ca = tmp_path / "cacert.pem"
    sealed_ca.write_bytes(host_ca.read_bytes())
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    with pytest.raises(ValueError, match="frozen SHA-256"):
        ModelClient(
            "bedrock-glm-5",
            bedrock_bearer_token=TOKEN,
            bedrock_ca_bundle=str(sealed_ca),
            bedrock_ca_bundle_sha256="0" * 64,
        )


def test_all_three_formal_gemma_configs_name_the_fixed_ca_bundle() -> None:
    for model in (
        "bedrock-gemma-4-e2b",
        "bedrock-gemma-4-26b",
        "bedrock-gemma-4-31b",
    ):
        assert LOCAL_MODELS[model]["backend"] == "bedrock_responses_raw"
        assert LOCAL_MODELS[model]["tls_ca_bundle"] == "/private/etc/ssl/cert.pem"


def _model_client(endpoint: str) -> ModelClient:
    client = ModelClient.__new__(ModelClient)
    client.model_name = "bedrock-glm-5-fixture"
    client.backend = BACKEND_NAME
    client.config = {
        "model_id": MODEL_ID,
        "reasoning_in_content": False,
        "reasoning_effort": "high",
        "max_tokens": 32000,
        "timeout": 2,
    }
    client._tls = threading.local()
    client._bedrock_chat_transport = _transport(endpoint)
    return client


def test_modelclient_preserves_api_and_logs_hash_only_transport(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _completion())
    client = _model_client(endpoint)
    response = client.call(
        system="sys",
        messages=[{"role": "user", "content": "question"}],
        max_tokens=91,
        temperature=0.1,
        kind="monolithic",
    )
    assert response.raw_text == "private reasoning\n" + response.content
    assert response.reasoning_trace["status"] == ReasoningStatus.PLAINTEXT
    rows = client.pop_call_log()
    assert len(rows) == 1
    assert rows[0]["kind"] == "monolithic"
    assert rows[0]["transport_trace"] == response.transport_trace
    assert TOKEN not in json.dumps(rows)
    sent = json.loads(state.requests[0]["body"])
    assert sent["reasoning_effort"] == "high"
    assert "response_format" not in sent


def test_modelclient_relation_override_matches_sdk_shape(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _completion(reasoning_content=""))
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
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["reasoning_effort"] == "none"


def test_modelclient_error_log_has_status_and_body_hash_without_body(local_server) -> None:
    state, endpoint = local_server
    raw = f'{{"error":"{TOKEN}"}}'.encode()
    state.push(500, raw)
    client = _model_client(endpoint)
    with pytest.raises(BedrockChatTransportError):
        client.call(system="sys", messages=[], kind="monolithic")
    rows = client.pop_call_log()
    assert len(rows) == 1
    assert rows[0]["transport_trace"]["response_http_status"] == 500
    assert rows[0]["transport_trace"]["response_body_sha256"] is None
    assert TOKEN not in json.dumps(rows)


def test_modelclient_transport_timeout_retains_request_commitment(local_server) -> None:
    state, endpoint = local_server
    state.push(200, _completion(), delay=0.25)
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
    state.push(200, _completion())
    client = _model_client(endpoint)
    client._spend_guard_disable_internal_retries = True
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    with pytest.raises(BedrockChatTransportError, match="HTTP 429"):
        client.call(system="sys", messages=[], kind="monolithic")
    assert len(state.requests) == 1


def test_modelclient_unguarded_429_preserves_bounded_existing_retry(local_server, monkeypatch) -> None:
    state, endpoint = local_server
    state.push(429, b"{}", headers={"Retry-After": "0"})
    state.push(200, _completion())
    client = _model_client(endpoint)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    result = client.call(system="sys", messages=[], kind="monolithic")
    assert result.content
    assert len(state.requests) == 2
    # One logical call-log row covers the successful in-client retry, matching
    # historical unguarded behavior. Paid runs disable this path above.
    assert len(client.pop_call_log()) == 1
