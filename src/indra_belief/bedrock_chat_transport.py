"""Publication-grade stdlib transport for the formal Bedrock GLM-5 lane.

This module implements only the OpenAI-compatible Chat Completions surface
used by GLM-5.  It deliberately excludes provider SDKs and ambient network
configuration: one frozen route, one byte-pinned CA bundle, no proxies, no
redirect following, and a monotonic deadline spanning DNS, connect, TLS,
request transmission, response headers, and a slow-drip response body.

The exact historical provider body is retained by
``build_bedrock_chat_body``.  Complete bounded response bodies are retained as
base64 preimages with raw and canonical hashes, so a credential-free offline
validator can recompute the transport result.  Failed or interrupted reads
are explicitly partial prefixes and can never be mistaken for complete wire
evidence.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import socket
import ssl
import threading
import time
from typing import Any, Mapping
import urllib.parse
import urllib.request

from . import bedrock_transport_base as _base


BACKEND_NAME = "bedrock_chat_completions_raw"
HTTP_METHOD = "POST"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
FORMAL_BEDROCK_HOST = "bedrock-mantle.us-east-1.api.aws"
FORMAL_CHAT_COMPLETIONS_ENDPOINT = (
    "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
)
FORMAL_GLM_MODEL_IDS = frozenset({"zai.glm-5"})
DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_CA_BUNDLE_BYTES = 8 * 1024 * 1024
_TRACE_KEYS = frozenset(
    {
        "schema_version",
        "backend",
        "method",
        "endpoint",
        "expected_model_id",
        "request_body_bytes",
        "request_body_sha256",
        "response_http_status",
        "response_body_bytes",
        "response_body_sha256",
        "response_body_prefix_bytes",
        "response_body_prefix_sha256",
        "response_body_complete",
        "response_body_preimage_kind",
        "response_body_preimage_b64",
        "response_body_preimage_redacted",
        "response_json_sha256",
        "response_size_exceeded",
        "response_content_length_declared",
        "response_framing",
        "response_framing_valid",
        "response_representation_valid",
        "transport_failure_class",
        "max_request_bytes",
        "max_response_bytes",
        "tls_ca_bundle_sha256",
        "tls_ca_load_mode",
        "redirects_allowed",
        "environment_proxies_allowed",
        "ambient_tls_trust_allowed",
        "provider_request_id",
        "provider_response_id",
        "provider_response_model",
    }
)
_TRANSPORT_FAILURE_CLASSES = frozenset(
    {
        "absolute_timeout",
        "connection_error",
        "connection_interrupted",
        "http_status",
        "invalid_http_framing",
        "invalid_json",
        "invalid_provider_metadata",
        "invalid_representation",
        "invalid_response_schema",
        "response_size_limit",
        "tls_certificate_verification",
        "tls_hostname_verification",
        "unexpected_transport_error",
        "unsafe_response_preimage",
    }
)


class BedrockChatTransportError(RuntimeError):
    """Safe transport error carrying bounded, bearer-redacted provenance."""

    def __init__(self, message: str, *, transport_trace: Mapping[str, Any]):
        super().__init__(message)
        self.transport_trace = dict(transport_trace)


class BedrockChatConnectionError(BedrockChatTransportError, ConnectionError):
    """Retry-classifiable network failure after a request was attempted."""


class BedrockChatTLSVerificationError(BedrockChatTransportError):
    """Terminal certificate or hostname verification failure."""


# Byte-identical transport machinery lives once in ``bedrock_transport_base``.
# It is re-exported here under its historical module-level names so internal
# call sites, monkeypatching tests, and importers see no change.  The lane
# strings that used to be embedded in the deadline/DNS/endpoint helpers are now
# passed by this module's call sites so every error message is reproduced
# char-for-char.
_SHA256_RE = _base._SHA256_RE
_AbsoluteDeadlineExpired = _base._AbsoluteDeadlineExpired
_DeadlineConnectionMixin = _base._DeadlineConnectionMixin
_DeadlineHTTPConnection = _base._DeadlineHTTPConnection
_DeadlineHTTPSConnection = _base._DeadlineHTTPSConnection
_resolve_host_bounded = _base._resolve_host_bounded
_safe_endpoint = _base._safe_endpoint
_new_tls_context = _base._new_tls_context
_RejectRedirects = _base._RejectRedirects
_read_file_bounded = _base._read_file_bounded
_redact_bearer = _base._redact_bearer
_retry_after_seconds = _base._retry_after_seconds
_single_header = _base._single_header
_declared_content_length = _base._declared_content_length
_response_framing = _base._response_framing
_trace_int = _base._trace_int
_trace_optional_int = _base._trace_optional_int
_trace_digest = _base._trace_digest
_trace_optional_digest = _base._trace_optional_digest
_trace_printable_id = _base._trace_printable_id
_validate_trace_framing = _base._validate_trace_framing
_finite_float = _base._finite_float


@dataclass(frozen=True)
class BedrockChatResult:
    """Normalized fields from one validated chat completion."""

    content: str
    reasoning: str
    prompt_tokens: int
    output_tokens: int
    reasoning_tokens: int
    finish_reason: str
    transport_trace: dict[str, Any]
    provider_response_id: str = ""
    provider_model: str = ""
    response_body_preimage_b64: str = ""


def canonical_json_bytes(value: Any) -> bytes:
    """Return one deterministic strict UTF-8 JSON encoding or fail closed."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError) as exc:
        raise ValueError("chat-completions payload is not strict UTF-8 JSON") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_transport_response_preimage(trace: Mapping[str, Any]) -> bytes:
    """Recompute one complete, unredacted response commitment from a trace."""

    if trace.get("response_body_preimage_kind") != "complete":
        raise ValueError("transport trace has no complete response preimage")
    if trace.get("response_body_complete") is not True:
        raise ValueError("transport response body is not complete")
    if trace.get("response_framing_valid") is not True:
        raise ValueError("transport response framing is not valid")
    if trace.get("response_size_exceeded") is not False:
        raise ValueError("transport response exceeded its safety limit")
    if trace.get("response_body_preimage_redacted") is not False:
        raise ValueError("transport response preimage was redacted")
    encoded = trace.get("response_body_preimage_b64")
    if not isinstance(encoded, str):
        raise ValueError("transport response preimage is missing")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("transport response preimage is not canonical base64") from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError("transport response preimage base64 is not canonical")
    if len(raw) != trace.get("response_body_bytes"):
        raise ValueError("transport response preimage byte count mismatch")
    if hashlib.sha256(raw).hexdigest() != trace.get("response_body_sha256"):
        raise ValueError("transport response preimage SHA-256 mismatch")
    expected_json_sha = trace.get("response_json_sha256")
    if expected_json_sha is not None:
        payload = _load_strict_json(raw)
        if canonical_json_sha256(payload) != expected_json_sha:
            raise ValueError("transport response canonical JSON SHA-256 mismatch")
    return raw


def _trace_preimage(trace: Mapping[str, Any]) -> tuple[str | None, bytes | None]:
    """Validate response evidence without relying on the live transport."""

    kind = trace["response_body_preimage_kind"]
    complete = trace["response_body_complete"]
    if type(complete) is not bool:
        raise ValueError("transport trace response_body_complete is not boolean")
    redacted = trace["response_body_preimage_redacted"]
    if redacted is not None and type(redacted) is not bool:
        raise ValueError("transport trace response preimage redaction flag is invalid")
    encoded = trace["response_body_preimage_b64"]
    body_bytes = _trace_optional_int(
        trace["response_body_bytes"], field="response_body_bytes"
    )
    body_sha = _trace_optional_digest(
        trace["response_body_sha256"], field="response_body_sha256"
    )
    prefix_bytes = _trace_optional_int(
        trace["response_body_prefix_bytes"], field="response_body_prefix_bytes"
    )
    prefix_sha = _trace_optional_digest(
        trace["response_body_prefix_sha256"], field="response_body_prefix_sha256"
    )
    json_sha = _trace_optional_digest(
        trace["response_json_sha256"], field="response_json_sha256"
    )

    if kind is None:
        if (
            complete is not False
            or encoded is not None
            or body_bytes is not None
            or body_sha is not None
            or prefix_bytes is not None
            or prefix_sha is not None
            or json_sha is not None
            or redacted not in {None, True}
        ):
            raise ValueError("transport trace absent response preimage is contradictory")
        return None, None
    if kind not in {"complete", "partial_prefix"}:
        raise ValueError("transport trace response preimage kind is unsupported")
    if redacted is not False:
        raise ValueError("transport trace retained response preimage is redacted")
    if not isinstance(encoded, str):
        raise ValueError("transport trace retained response preimage is missing")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("transport trace response preimage is not canonical base64") from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError("transport trace response preimage base64 is not canonical")

    if kind == "complete":
        if (
            complete is not True
            or body_bytes != len(raw)
            or body_sha != hashlib.sha256(raw).hexdigest()
            or prefix_bytes is not None
            or prefix_sha is not None
        ):
            raise ValueError("transport trace complete response commitment is incoherent")
        if len(raw) > DEFAULT_MAX_RESPONSE_BYTES:
            raise ValueError("transport trace complete response exceeds the frozen limit")
        if json_sha != _try_canonical_json_sha256(raw):
            raise ValueError("transport trace canonical response JSON commitment differs")
        return kind, raw

    if (
        complete is not False
        or not raw
        or body_bytes is not None
        or body_sha is not None
        or prefix_bytes != len(raw)
        or prefix_sha != hashlib.sha256(raw).hexdigest()
        or json_sha is not None
        or len(raw) > DEFAULT_MAX_RESPONSE_BYTES + 1
    ):
        raise ValueError("transport trace partial response commitment is incoherent")
    return kind, raw


def _validated_response_result(
    raw: bytes, trace: Mapping[str, Any]
) -> BedrockChatResult:
    payload = _load_strict_json(raw)
    return parse_bedrock_chat_payload(
        payload,
        transport_trace=trace,
        expected_model_id=str(trace["expected_model_id"]),
    )


def validate_transport_trace(
    trace: Mapping[str, Any],
    *,
    expected_request_sha256: str,
    expected_ca_bundle_sha256: str,
    successful: bool,
) -> bytes | None:
    """Validate one formal Chat Completions trace entirely offline.

    Complete, unredacted provider evidence is returned byte-for-byte.  Failed
    calls with only a validated partial prefix, no response, an oversized body,
    or a bearer-omitted body return ``None``.
    """

    if not isinstance(trace, Mapping):
        raise ValueError("transport trace must be an object")
    if set(trace) != _TRACE_KEYS:
        missing = sorted(_TRACE_KEYS - set(trace))
        extra = sorted((set(trace) - _TRACE_KEYS), key=str)
        raise ValueError(
            f"transport trace key census differs (missing={missing!r}, extra={extra!r})"
        )
    expected_request_sha256 = _trace_digest(
        expected_request_sha256, field="expected_request_sha256"
    )
    expected_ca_bundle_sha256 = _trace_digest(
        expected_ca_bundle_sha256, field="expected_ca_bundle_sha256"
    )
    if type(successful) is not bool:
        raise ValueError("successful must be boolean")
    if (
        type(trace["schema_version"]) is not int
        or trace["schema_version"] != 1
        or trace["backend"] != BACKEND_NAME
        or trace["method"] != HTTP_METHOD
        or trace["endpoint"] != FORMAL_CHAT_COMPLETIONS_ENDPOINT
    ):
        raise ValueError("transport trace differs from the frozen Chat route")
    if trace["expected_model_id"] not in FORMAL_GLM_MODEL_IDS:
        raise ValueError("transport trace expected model is not the formal GLM model")
    request_bytes = _trace_int(
        trace["request_body_bytes"], field="request_body_bytes", minimum=1
    )
    if request_bytes > DEFAULT_MAX_REQUEST_BYTES:
        raise ValueError("transport trace request exceeds the frozen limit")
    if (
        _trace_digest(trace["request_body_sha256"], field="request_body_sha256")
        != expected_request_sha256
    ):
        raise ValueError("transport trace request commitment differs from the caller")
    if (
        trace["max_request_bytes"] != DEFAULT_MAX_REQUEST_BYTES
        or trace["max_response_bytes"] != DEFAULT_MAX_RESPONSE_BYTES
    ):
        raise ValueError("transport trace byte limits differ from the frozen limits")
    if (
        trace["tls_ca_bundle_sha256"] != expected_ca_bundle_sha256
        or trace["tls_ca_load_mode"] not in {"cadata", "cafile_fallback"}
    ):
        raise ValueError("transport trace TLS trust differs from the sealed CA bundle")
    if (
        trace["redirects_allowed"] is not False
        or trace["environment_proxies_allowed"] is not False
        or trace["ambient_tls_trust_allowed"] is not False
    ):
        raise ValueError("transport trace admits ambient egress behavior")

    status = _trace_optional_int(
        trace["response_http_status"], field="response_http_status", minimum=100
    )
    if status is not None and status > 599:
        raise ValueError("transport trace response HTTP status is outside 100..599")
    request_id = _trace_printable_id(
        trace["provider_request_id"], field="provider_request_id", maximum=256
    )
    if status is None and request_id is not None:
        raise ValueError("transport trace has a provider request id without a response")
    response_id = _trace_printable_id(
        trace["provider_response_id"], field="provider_response_id", maximum=512
    )
    response_model = trace["provider_response_model"]
    if response_model is not None and (
        not isinstance(response_model, str) or not response_model
    ):
        raise ValueError("transport trace provider response model is invalid")
    representation_valid = trace["response_representation_valid"]
    if representation_valid is not None and type(representation_valid) is not bool:
        raise ValueError("transport trace representation validity is invalid")
    if type(trace["response_size_exceeded"]) is not bool:
        raise ValueError("transport trace response_size_exceeded is not boolean")

    kind, raw = _trace_preimage(trace)
    _validate_trace_framing(trace, kind=kind, raw=raw)
    failure_class = trace["transport_failure_class"]
    if failure_class is not None and failure_class not in _TRANSPORT_FAILURE_CLASSES:
        raise ValueError("transport trace failure class is unsupported")
    if trace["response_size_exceeded"]:
        if failure_class != "response_size_limit" or kind == "complete":
            raise ValueError("transport trace response-size failure is contradictory")
        declared = trace["response_content_length_declared"]
        if kind is None:
            if (
                trace["response_framing"] != "content_length"
                or not isinstance(declared, int)
                or isinstance(declared, bool)
                or declared <= DEFAULT_MAX_RESPONSE_BYTES
            ):
                raise ValueError("transport trace declared oversize response is incoherent")
        elif raw is None or len(raw) != DEFAULT_MAX_RESPONSE_BYTES + 1:
            raise ValueError("transport trace observed oversize prefix is incoherent")
    elif failure_class == "response_size_limit":
        raise ValueError("transport trace size-limit failure did not exceed the limit")

    if successful:
        if (
            failure_class is not None
            or status != 200
            or representation_valid is not True
            or trace["response_size_exceeded"] is not False
            or kind != "complete"
            or raw is None
            or trace["response_json_sha256"] is None
        ):
            raise ValueError("successful transport trace has contradictory outcome fields")
        result = _validated_response_result(raw, trace)
        if (
            response_id != result.provider_response_id
            or response_model != result.provider_model
            or response_model != trace["expected_model_id"]
        ):
            raise ValueError("transport trace provider response metadata differs from bytes")
        return raw

    if failure_class is None:
        raise ValueError("failed transport trace has no terminal failure class")
    if response_id is not None or response_model is not None:
        raise ValueError("failed transport trace claims successful provider metadata")
    if failure_class == "http_status":
        if status is None or status == 200 or representation_valid is not None:
            raise ValueError("transport trace HTTP-status failure is contradictory")
        if kind not in {"complete", None}:
            raise ValueError("transport trace HTTP-status failure has partial evidence")
    elif failure_class == "invalid_http_framing":
        if status is None or trace["response_framing_valid"] is not False:
            raise ValueError("transport trace framing failure is contradictory")
        if kind not in {None, "partial_prefix"}:
            raise ValueError("transport trace framing failure claims a complete body")
    elif failure_class == "absolute_timeout":
        if kind not in {None, "partial_prefix"} or representation_valid is not None:
            raise ValueError("transport trace timeout outcome is contradictory")
    elif failure_class == "connection_interrupted":
        if (
            status is None
            or kind not in {None, "partial_prefix"}
            or trace["response_framing_valid"] is not False
            or representation_valid is not None
        ):
            raise ValueError("transport trace interrupted response is contradictory")
    elif failure_class in {
        "tls_certificate_verification",
        "tls_hostname_verification",
        "connection_error",
    }:
        if status is not None or kind is not None or trace["response_framing"] is not None:
            raise ValueError("transport trace pre-response failure is contradictory")
    elif failure_class == "unexpected_transport_error":
        if kind is not None or representation_valid is not None:
            raise ValueError("transport trace unexpected failure retained foreign evidence")
    elif failure_class == "invalid_provider_metadata":
        if status is None or representation_valid is not None or kind != "complete":
            raise ValueError("transport trace provider-metadata failure is contradictory")
    elif failure_class == "invalid_representation":
        if status != 200 or representation_valid is not False or kind != "complete":
            raise ValueError("transport trace representation failure is contradictory")
    elif failure_class == "invalid_json":
        if (
            status != 200
            or representation_valid is not True
            or kind != "complete"
            or raw is None
            or trace["response_json_sha256"] is not None
        ):
            raise ValueError("transport trace JSON failure is contradictory")
    elif failure_class == "invalid_response_schema":
        if (
            status != 200
            or representation_valid is not True
            or kind != "complete"
            or raw is None
            or trace["response_json_sha256"] is None
        ):
            raise ValueError("transport trace response-schema failure is contradictory")
        try:
            _validated_response_result(raw, trace)
        except ValueError:
            pass
        else:
            raise ValueError("transport trace labels a valid response as schema-invalid")
    elif failure_class == "unsafe_response_preimage":
        if (
            status != 200
            or representation_valid is not True
            or kind is not None
            or trace["response_body_preimage_redacted"] is not True
        ):
            raise ValueError("transport trace unsafe-preimage failure is contradictory")
    if trace["response_body_preimage_redacted"] is True and kind is not None:
        raise ValueError("transport trace exposes a bearer-redacted response preimage")
    return raw if kind == "complete" else None


def build_bedrock_chat_body(
    *,
    model_id: str,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Build the exact body historically passed to the OpenAI SDK.

    ``reasoning_effort`` was supplied through ``extra_body`` and flattened by
    the SDK at the JSON top level.  ``timeout`` is a transport option and is
    intentionally absent.  Key sorting changes wire order only; all historical
    semantic request objects remain identical.
    """

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(system, str):
        raise ValueError("system must be a string")
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) for message in messages
    ):
        raise ValueError("messages must be a list of JSON objects")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
    ):
        raise ValueError("temperature must be finite and numeric")
    if response_format is not None and not isinstance(response_format, dict):
        raise ValueError("response_format must be an object or None")
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        raise ValueError("reasoning_effort must be a string or None")

    body: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}] + list(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        body["response_format"] = response_format
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    canonical_json_bytes(body)
    return body


def _validate_request_body(body: Mapping[str, Any], expected_model_id: str) -> None:
    allowed = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "response_format",
        "reasoning_effort",
    }
    if set(body) - allowed:
        raise ValueError("chat-completions body contains an unsupported field")
    if body.get("model") != expected_model_id:
        raise ValueError("chat-completions body model differs from the frozen model")
    maximum = body.get("max_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("chat-completions body has an invalid max_tokens")
    temperature = body.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
    ):
        raise ValueError("chat-completions body has an invalid temperature")
    response_format = body.get("response_format")
    if response_format is not None and (
        not isinstance(response_format, dict)
        or set(response_format) != {"type"}
        or response_format.get("type") != "json_object"
    ):
        raise ValueError(
            "chat-completions body response_format must be the JSON-object mode"
        )
    effort = body.get("reasoning_effort")
    if effort is not None and effort not in {"none", "low", "medium", "high"}:
        raise ValueError("chat-completions body reasoning_effort is unsupported")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("chat-completions body messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"chat-completions body messages[{index}] must be an object")
        if set(message) != {"role", "content"}:
            raise ValueError(
                f"chat-completions body messages[{index}] has an invalid shape"
            )
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"chat-completions body messages[{index}].role is invalid")
        if not isinstance(message.get("content"), str):
            raise ValueError(
                f"chat-completions body messages[{index}].content must be text"
            )
    first = messages[0]
    if first.get("role") != "system" or not isinstance(first.get("content"), str):
        raise ValueError("chat-completions body must begin with one text system message")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # Never echo an untrusted provider key into a persisted exception.
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _load_strict_json(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_float=_finite_float,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                ValueError("non-finite JSON number")
            ),
        )
        # Prove the parsed graph itself can be represented as canonical UTF-8.
        canonical_json_bytes(value)
    except (
        UnicodeDecodeError,
        UnicodeEncodeError,
        json.JSONDecodeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise ValueError("provider returned malformed strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("provider response must be a JSON object")
    return value


def _text_channel(value: Any, *, field: str) -> str:
    """Preserve the text forms accepted by the historical GLM adapter."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError(f"provider {field} must be text, text blocks, or null")
    parts: list[str] = []
    for index, block in enumerate(value):
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            raise ValueError(f"provider {field}[{index}] is not a text block")
        text = block.get("text")
        if text is None and isinstance(block.get("content"), str):
            text = block["content"]
        if not isinstance(text, str):
            raise ValueError(f"provider {field}[{index}] has no string text")
        parts.append(text)
    return "".join(parts)


def _token_count(
    usage: Mapping[str, Any], *names: str, required: bool = False
) -> int:
    found: list[tuple[str, int]] = []
    for name in names:
        if name not in usage or usage[name] is None:
            continue
        value = usage[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"provider usage.{name} must be a non-negative integer")
        found.append((name, value))
    if len(found) > 1:
        raise ValueError(f"provider usage contains ambiguous {names[0]} aliases")
    if found:
        return found[0][1]
    if required:
        raise ValueError(f"provider usage.{names[0]} is missing")
    return -1


def _reasoning_token_count(usage: Mapping[str, Any]) -> int:
    found: list[int] = []
    for field in ("completion_tokens_details", "output_tokens_details"):
        if field not in usage or usage[field] is None:
            continue
        details = usage[field]
        if not isinstance(details, dict):
            raise ValueError(f"provider usage.{field} must be an object")
        found.append(_token_count(details, "reasoning_tokens"))
    if len(found) > 1:
        raise ValueError("provider usage contains ambiguous reasoning-token aliases")
    return found[0] if found else -1


def _bounded_provider_id(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or not value.isascii()
        or any(not 0x20 <= ord(character) <= 0x7E for character in value)
    ):
        raise ValueError(f"provider {field} must be bounded printable text")
    return value


def parse_bedrock_chat_payload(
    payload: Mapping[str, Any],
    *,
    transport_trace: Mapping[str, Any],
    expected_model_id: str | None = None,
) -> BedrockChatResult:
    """Pure parser for one strict Chat Completions provider envelope.

    Offline validators normally omit ``expected_model_id`` and use the model
    already bound into the transport trace.  Passing it explicitly adds an
    independent equality check and is used by the transport itself.
    """

    trace_model_id = transport_trace.get("expected_model_id")
    if expected_model_id is None:
        if not isinstance(trace_model_id, str) or not trace_model_id:
            raise ValueError("transport trace has no expected provider model")
        expected_model_id = trace_model_id
    elif trace_model_id is not None and trace_model_id != expected_model_id:
        raise ValueError("transport trace expected model differs from the parser")

    if payload.get("object") != "chat.completion":
        raise ValueError("provider response has an unexpected object type")
    provider_model = payload.get("model")
    if provider_model != expected_model_id:
        raise ValueError("provider response model differs from the frozen model")
    provider_response_id = _bounded_provider_id(payload.get("id"), field="response id")

    choices = payload.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise ValueError("provider response must have exactly one completion choice")
    choice = choices[0]
    if choice.get("index") != 0:
        raise ValueError("provider first choice index must be zero")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("provider first choice has no assistant message")

    content = _text_channel(message.get("content"), field="message.content")
    reasoning = _text_channel(
        message.get("reasoning_content"), field="message.reasoning_content"
    )
    finish = choice.get("finish_reason")
    if finish is None:
        # Preserve the legacy adapter's null-to-stop normalization.
        finish = "stop"
    if (
        not isinstance(finish, str)
        or not finish
        or len(finish) > 128
        or not finish.isascii()
        or any(not 0x20 <= ord(character) <= 0x7E for character in finish)
    ):
        raise ValueError("provider finish_reason must be bounded printable text or null")

    usage_value = payload.get("usage")
    if not isinstance(usage_value, dict):
        raise ValueError("provider usage must be an object")
    usage: Mapping[str, Any] = usage_value
    prompt_tokens = _token_count(
        usage, "prompt_tokens", "input_tokens", required=True
    )
    output_tokens = _token_count(
        usage, "completion_tokens", "output_tokens", required=True
    )
    reasoning_tokens = _reasoning_token_count(usage)
    total_tokens = _token_count(usage, "total_tokens")
    if total_tokens >= 0 and total_tokens != prompt_tokens + output_tokens:
        raise ValueError("provider usage.total_tokens is inconsistent")
    if reasoning_tokens > output_tokens:
        raise ValueError("provider reasoning token count exceeds output tokens")
    response_preimage = transport_trace.get("response_body_preimage_b64")
    if not isinstance(response_preimage, str):
        raise ValueError("provider response preimage is missing")
    return BedrockChatResult(
        content=content,
        reasoning=reasoning,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        finish_reason=finish,
        provider_response_id=provider_response_id,
        provider_model=provider_model,
        response_body_preimage_b64=response_preimage,
        transport_trace=dict(transport_trace),
    )


def parse_bedrock_chat_response_preimage(
    trace: Mapping[str, Any], *, expected_model_id: str
) -> BedrockChatResult:
    """Verify and parse a stored response using no credential or transport state."""

    if trace.get("backend") != BACKEND_NAME or trace.get("method") != HTTP_METHOD:
        raise ValueError("transport trace is not a Bedrock Chat POST")
    if trace.get("response_http_status") != 200:
        raise ValueError("transport trace is not a successful provider response")
    if trace.get("expected_model_id") != expected_model_id:
        raise ValueError("transport trace expected model differs from the validator")
    raw = verify_transport_response_preimage(trace)
    payload = _load_strict_json(raw)
    return parse_bedrock_chat_payload(
        payload,
        transport_trace=trace,
        expected_model_id=expected_model_id,
    )


def _explicit_tls_context(ca_bundle: Path) -> tuple[ssl.SSLContext, str, str]:
    raw = _read_file_bounded(ca_bundle, MAX_CA_BUNDLE_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        cadata = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("TLS CA bundle is not an ASCII PEM bundle") from exc

    context = _new_tls_context()
    try:
        context.load_verify_locations(cadata=cadata)
        return context, digest, "cadata"
    except (TypeError, NotImplementedError):
        # Capability fallback only: malformed PEM does not get a second parser.
        context = _new_tls_context()
        try:
            context.load_verify_locations(cafile=str(ca_bundle))
        except (OSError, ssl.SSLError) as exc:
            raise ValueError("TLS CA bundle could not be loaded by explicit path") from exc
        if _read_file_bounded(ca_bundle, MAX_CA_BUNDLE_BYTES) != raw:
            raise ValueError("TLS CA bundle changed while it was loaded")
        return context, digest, "cafile_fallback"
    except ssl.SSLError as exc:
        raise ValueError("TLS CA bundle could not be loaded as PEM cadata") from exc


def build_pinned_https_opener(
    ca_bundle: str | Path,
    *,
    expected_ca_bundle_sha256: str | None = None,
) -> tuple[urllib.request.OpenerDirector, str]:
    """Compatibility helper with explicit trust and no proxy/redirect handlers.

    The formal raw GLM transport below uses ``http.client`` directly and always
    requires the expected digest.  This helper remains for the legacy
    Responses/Converse adapters in ``ModelClient``.
    """

    context, digest, _load_mode = _explicit_tls_context(Path(ca_bundle))
    if expected_ca_bundle_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_ca_bundle_sha256):
            raise ValueError("expected TLS CA bundle SHA-256 is invalid")
        if digest != expected_ca_bundle_sha256:
            raise ValueError("TLS CA bundle differs from its frozen SHA-256")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirects(),
        urllib.request.HTTPSHandler(context=context),
    )
    return opener, digest


def _validate_representation_headers(headers: Any) -> None:
    encoding = _single_header(headers, "Content-Encoding")
    if encoding is not None and encoding.lower() != "identity":
        raise ValueError("provider response uses an unsupported content encoding")
    content_type = _single_header(headers, "Content-Type")
    if content_type is None:
        raise ValueError("provider response has no Content-Type")
    if not re.fullmatch(
        r"(?i)application/json(?:\s*;\s*charset\s*=\s*(?:utf-?8|\"utf-?8\"))?",
        content_type,
    ):
        if not content_type.lower().startswith("application/json"):
            raise ValueError("provider response is not application/json")
        raise ValueError("provider response has malformed or unsupported Content-Type")


class _ResponseReadFailure(Exception):
    def __init__(
        self,
        message: str,
        *,
        partial: bytes,
        framing: str | None,
        declared: int | None,
        timed_out: bool,
        framing_valid: bool,
        connection_error: bool = False,
    ) -> None:
        super().__init__(message)
        self.partial = partial
        self.framing = framing
        self.declared = declared
        self.timed_out = timed_out
        self.framing_valid = framing_valid
        self.connection_error = connection_error


def _try_canonical_json_sha256(raw: bytes | None) -> str | None:
    if raw is None:
        return None
    try:
        return canonical_json_sha256(_load_strict_json(raw))
    except ValueError:
        return None


def _tls_failure_class(exc: ssl.SSLCertVerificationError) -> str:
    detail = str(getattr(exc, "verify_message", "") or exc).lower()
    if any(
        marker in detail
        for marker in (
            "hostname mismatch",
            "doesn't match",
            "does not match",
            "ip address mismatch",
        )
    ):
        return "tls_hostname_verification"
    return "tls_certificate_verification"


class RawBedrockChatTransport:
    """One exact, bounded, proxy-free Chat Completions HTTP client."""

    def __init__(
        self,
        *,
        endpoint: str,
        expected_endpoint: str,
        expected_model_id: str,
        bearer_token: str,
        ca_bundle: str | Path | None,
        expected_ca_bundle_sha256: str | None,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        allow_insecure_http_for_tests: bool = False,
    ) -> None:
        parsed_endpoint = _safe_endpoint(
            endpoint,
            expected_endpoint=expected_endpoint,
            allow_insecure_http_for_tests=allow_insecure_http_for_tests,
            lane_label="chat-completions",
            expected_path=CHAT_COMPLETIONS_PATH,
            formal_endpoint=FORMAL_CHAT_COMPLETIONS_ENDPOINT,
            formal_host=FORMAL_BEDROCK_HOST,
        )
        if not isinstance(expected_model_id, str) or not expected_model_id:
            raise ValueError("expected_model_id must be a non-empty string")
        if not allow_insecure_http_for_tests and expected_model_id not in FORMAL_GLM_MODEL_IDS:
            raise ValueError("paid chat-completions model is not the formal GLM model")
        if (
            not isinstance(bearer_token, str)
            or not bearer_token
            or len(bearer_token) > 16 * 1024
            or not bearer_token.isascii()
            or any(
                character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
                for character in bearer_token
            )
        ):
            raise ValueError("Bedrock bearer token must be non-empty header-safe text")
        for name, value in (
            ("max_request_bytes", max_request_bytes),
            ("max_response_bytes", max_response_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not allow_insecure_http_for_tests and (
            max_request_bytes != DEFAULT_MAX_REQUEST_BYTES
            or max_response_bytes != DEFAULT_MAX_RESPONSE_BYTES
        ):
            raise ValueError("paid chat transport requires the frozen byte limits")

        ca_sha256: str | None = None
        ca_load_mode = "disabled_loopback_test"
        tls_context: ssl.SSLContext | None = None
        if allow_insecure_http_for_tests:
            if ca_bundle is not None or expected_ca_bundle_sha256 is not None:
                raise ValueError("test-only HTTP transport cannot take TLS CA material")
        else:
            if ca_bundle is None or expected_ca_bundle_sha256 is None:
                raise ValueError("paid chat transport requires a byte-pinned CA bundle")
            if not re.fullmatch(r"[0-9a-f]{64}", expected_ca_bundle_sha256):
                raise ValueError("expected TLS CA bundle SHA-256 is invalid")
            tls_context, ca_sha256, ca_load_mode = _explicit_tls_context(Path(ca_bundle))
            if ca_sha256 != expected_ca_bundle_sha256:
                raise ValueError("TLS CA bundle differs from its frozen SHA-256")

        self.endpoint = endpoint
        self.expected_model_id = expected_model_id
        self._token = bearer_token
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._tls_ca_bundle_sha256 = ca_sha256
        self._tls_ca_load_mode = ca_load_mode
        self._tls_context = tls_context
        self._endpoint_scheme = parsed_endpoint.scheme
        self._endpoint_host = parsed_endpoint.hostname
        self._endpoint_port = parsed_endpoint.port

    def _connection_factory(
        self, absolute_deadline: float, abort_event: threading.Event
    ) -> http.client.HTTPConnection:
        timeout = max(absolute_deadline - time.monotonic(), 0.001)
        if self._endpoint_scheme == "https":
            assert self._tls_context is not None
            return _DeadlineHTTPSConnection(
                self._endpoint_host,
                self._endpoint_port or 443,
                timeout=timeout,
                context=self._tls_context,
                absolute_deadline=absolute_deadline,
                abort_event=abort_event,
                lane_label="Bedrock Chat",
                dns_thread_name="bedrock-chat-dns",
            )
        return _DeadlineHTTPConnection(
            self._endpoint_host,
            self._endpoint_port,
            timeout=timeout,
            absolute_deadline=absolute_deadline,
            abort_event=abort_event,
            lane_label="Bedrock Chat",
            dns_thread_name="bedrock-chat-dns",
        )

    def _trace(
        self,
        *,
        request_raw: bytes,
        status: int | None,
        response_raw: bytes | None,
        response_json_sha256: str | None = None,
        response_size_exceeded: bool = False,
        response_content_length_declared: int | None = None,
        response_body_complete: bool | None = None,
        response_framing: str | None = None,
        response_framing_valid: bool | None = None,
        response_representation_valid: bool | None = None,
        transport_failure_class: str | None = None,
    ) -> dict[str, Any]:
        if response_body_complete is None:
            response_body_complete = response_raw is not None and not response_size_exceeded
        token_bytes = self._token.encode("ascii")
        persisted_raw = response_raw
        preimage_redacted = False
        if persisted_raw is not None and token_bytes in persisted_raw:
            persisted_raw = None
            response_raw = None
            response_body_complete = False
            response_json_sha256 = None
            preimage_redacted = True
        if response_raw == b"" and not response_body_complete:
            response_raw = None
            persisted_raw = None
        if response_raw is None:
            preimage_kind = None
        elif response_body_complete:
            preimage_kind = "complete"
        else:
            preimage_kind = "partial_prefix"
        return {
            "schema_version": 1,
            "backend": BACKEND_NAME,
            "method": HTTP_METHOD,
            "endpoint": self.endpoint,
            "expected_model_id": self.expected_model_id,
            "request_body_bytes": len(request_raw),
            "request_body_sha256": hashlib.sha256(request_raw).hexdigest(),
            "response_http_status": status,
            "response_body_bytes": (
                len(response_raw)
                if response_raw is not None and response_body_complete
                else None
            ),
            "response_body_sha256": (
                hashlib.sha256(response_raw).hexdigest()
                if response_raw is not None
                and response_body_complete
                and not response_size_exceeded
                else None
            ),
            "response_body_prefix_bytes": (
                len(response_raw)
                if response_raw is not None and not response_body_complete
                else None
            ),
            "response_body_prefix_sha256": (
                hashlib.sha256(response_raw).hexdigest()
                if response_raw is not None and not response_body_complete
                else None
            ),
            "response_body_complete": response_body_complete,
            "response_body_preimage_kind": preimage_kind,
            "response_body_preimage_b64": (
                base64.b64encode(persisted_raw).decode("ascii")
                if persisted_raw is not None
                else None
            ),
            "response_body_preimage_redacted": (
                True
                if preimage_redacted
                else False
                if response_raw is not None
                else None
            ),
            "response_json_sha256": response_json_sha256,
            "response_size_exceeded": response_size_exceeded,
            "response_content_length_declared": response_content_length_declared,
            "response_framing": response_framing,
            "response_framing_valid": response_framing_valid,
            "response_representation_valid": response_representation_valid,
            "transport_failure_class": transport_failure_class,
            "max_request_bytes": self._max_request_bytes,
            "max_response_bytes": self._max_response_bytes,
            "tls_ca_bundle_sha256": self._tls_ca_bundle_sha256,
            "tls_ca_load_mode": self._tls_ca_load_mode,
            "redirects_allowed": False,
            "environment_proxies_allowed": False,
            "ambient_tls_trust_allowed": False,
            "provider_request_id": None,
            "provider_response_id": None,
            "provider_response_model": None,
        }

    def _request_bytes(self, body: Mapping[str, Any]) -> bytes:
        if not isinstance(body, Mapping):
            raise ValueError("chat-completions body must be an object")
        materialized = dict(body)
        _validate_request_body(materialized, self.expected_model_id)
        raw = canonical_json_bytes(materialized)
        if len(raw) > self._max_request_bytes:
            raise ValueError("chat-completions request exceeds the safety limit")
        return raw

    def _read_response(
        self,
        response: http.client.HTTPResponse,
        *,
        absolute_deadline: float,
        abort_event: threading.Event,
    ) -> tuple[bytes | None, bool, int | None, str]:
        framing, declared = _response_framing(response.headers)
        if declared is not None and declared > self._max_response_bytes:
            return None, True, declared, framing
        target = declared if declared is not None else self._max_response_bytes + 1
        chunks: list[bytes] = []
        observed = 0
        while observed < target:
            remaining = absolute_deadline - time.monotonic()
            if abort_event.is_set() or remaining <= 0:
                raise _ResponseReadFailure(
                    "provider response exceeded its absolute deadline",
                    partial=b"".join(chunks),
                    framing=framing,
                    declared=declared,
                    timed_out=True,
                    framing_valid=True,
                )
            connection_socket = getattr(
                getattr(getattr(response, "fp", None), "raw", None), "_sock", None
            )
            if connection_socket is not None:
                connection_socket.settimeout(remaining)
            read_size = min(64 * 1024, target - observed)
            try:
                read_one = getattr(response, "read1", None)
                if read_one is None:
                    read_one = response.read
                chunk = read_one(read_size)
            except http.client.IncompleteRead as exc:
                partial = b"".join(chunks) + bytes(exc.partial)
                raise _ResponseReadFailure(
                    "provider response ended before its declared framing",
                    partial=partial,
                    framing=framing,
                    declared=declared,
                    timed_out=False,
                    framing_valid=False,
                ) from None
            except Exception as exc:
                partial = b"".join(chunks)
                timed_out = (
                    abort_event.is_set()
                    or time.monotonic() >= absolute_deadline
                    or isinstance(exc, (TimeoutError, socket.timeout))
                )
                raise _ResponseReadFailure(
                    "provider response read failed",
                    partial=partial,
                    framing=framing,
                    declared=declared,
                    timed_out=timed_out,
                    framing_valid=timed_out,
                    connection_error=(
                        not timed_out
                        and isinstance(
                            exc,
                            (OSError, http.client.HTTPException, ssl.SSLError),
                        )
                    ),
                ) from None
            if abort_event.is_set() or time.monotonic() >= absolute_deadline:
                raise _ResponseReadFailure(
                    "provider response exceeded its absolute deadline",
                    partial=b"".join(chunks) + chunk,
                    framing=framing,
                    declared=declared,
                    timed_out=True,
                    framing_valid=True,
                )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        raw = b"".join(chunks)
        exceeded = len(raw) > self._max_response_bytes
        if not exceeded and declared is not None and len(raw) != declared:
            raise _ResponseReadFailure(
                "provider response length differs from Content-Length",
                partial=raw,
                framing=framing,
                declared=declared,
                timed_out=False,
                framing_valid=False,
            )
        if abort_event.is_set() or time.monotonic() >= absolute_deadline:
            raise _ResponseReadFailure(
                "provider response exceeded its absolute deadline",
                partial=raw,
                framing=framing,
                declared=declared,
                timed_out=True,
                framing_valid=True,
            )
        return raw, exceeded, declared, framing

    def request_trace(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Return the exact pre-side-effect request commitment."""

        return self._trace(
            request_raw=self._request_bytes(body),
            status=None,
            response_raw=None,
        )

    def call(self, body: Mapping[str, Any], *, timeout: int | float) -> BedrockChatResult:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        request_raw = self._request_bytes(body)
        absolute_deadline = time.monotonic() + float(timeout)
        abort_event = threading.Event()
        connection = self._connection_factory(absolute_deadline, abort_event)
        state_lock = threading.Lock()
        state: dict[str, Any] = {"response": None, "settled": False}

        def abort_live_request() -> None:
            with state_lock:
                if state["settled"]:
                    return
                abort_event.set()
                active_response = state["response"]
                active_socket = connection.sock
            if active_socket is not None:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if active_response is not None:
                try:
                    active_response.close()
                except Exception:
                    pass
            try:
                connection.close()
            except Exception:
                pass

        watchdog = threading.Timer(float(timeout), abort_live_request)
        watchdog.daemon = True
        response: http.client.HTTPResponse | None = None
        status: int | None = None
        response_raw: bytes | None = None
        exceeded = False
        declared: int | None = None
        framing: str | None = None
        trace: dict[str, Any] | None = None
        watchdog.start()
        try:
            connection.connect()
            if abort_event.is_set() or time.monotonic() >= absolute_deadline:
                raise _AbsoluteDeadlineExpired
            connection.request(
                HTTP_METHOD,
                CHAT_COMPLETIONS_PATH,
                body=request_raw,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(request_raw)),
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "User-Agent": "indra-belief/raw-bedrock-chat-v1",
                },
            )
            response = connection.getresponse()
            status = int(response.status)
            if status < 100 or status > 599:
                raise http.client.BadStatusLine(f"invalid HTTP status {status}")
            with state_lock:
                state["response"] = response
            try:
                response_raw, exceeded, declared, framing = self._read_response(
                    response,
                    absolute_deadline=absolute_deadline,
                    abort_event=abort_event,
                )
                with state_lock:
                    deadline_expired = abort_event.is_set() or time.monotonic() >= absolute_deadline
                    if not deadline_expired:
                        state["settled"] = True
                if deadline_expired:
                    raise _ResponseReadFailure(
                        "provider response exceeded its absolute deadline",
                        partial=response_raw or b"",
                        framing=framing,
                        declared=declared,
                        timed_out=True,
                        framing_valid=True,
                    )
            except ValueError:
                trace = self._trace(
                    request_raw=request_raw,
                    status=status,
                    response_raw=None,
                    response_body_complete=False,
                    response_framing_valid=False,
                    transport_failure_class="invalid_http_framing",
                )
                raise BedrockChatTransportError(
                    "Bedrock Chat Completions invalid response framing",
                    transport_trace=trace,
                ) from None
            except _ResponseReadFailure as exc:
                trace = self._trace(
                    request_raw=request_raw,
                    status=status,
                    response_raw=exc.partial,
                    response_content_length_declared=exc.declared,
                    response_body_complete=False,
                    response_framing=exc.framing,
                    response_framing_valid=exc.framing_valid,
                    transport_failure_class=(
                        "absolute_timeout"
                        if exc.timed_out
                        else "connection_interrupted"
                        if exc.connection_error
                        else "invalid_http_framing"
                    ),
                )
                if exc.timed_out:
                    error = TimeoutError(
                        f"Bedrock Chat Completions request timed out after {timeout}s"
                    )
                    error.transport_trace = trace  # type: ignore[attr-defined]
                    raise error from None
                if exc.connection_error:
                    raise BedrockChatConnectionError(
                        "Bedrock Chat Completions connection interrupted during response",
                        transport_trace=trace,
                    ) from None
                raise BedrockChatTransportError(
                    "Bedrock Chat Completions invalid response framing",
                    transport_trace=trace,
                ) from None
            watchdog.cancel()
            trace = self._trace(
                request_raw=request_raw,
                status=status,
                response_raw=response_raw,
                response_json_sha256=(
                    None if exceeded else _try_canonical_json_sha256(response_raw)
                ),
                response_size_exceeded=exceeded,
                response_content_length_declared=declared,
                response_body_complete=not exceeded and response_raw is not None,
                response_framing=framing,
                response_framing_valid=True,
            )
            if exceeded:
                trace["transport_failure_class"] = "response_size_limit"
                raise BedrockChatTransportError(
                    "Bedrock Chat Completions response exceeds the safety limit",
                    transport_trace=trace,
                )
            try:
                request_id = _single_header(response.headers, "x-amzn-requestid")
            except ValueError:
                trace["transport_failure_class"] = "invalid_provider_metadata"
                raise BedrockChatTransportError(
                    "Bedrock Chat Completions returned invalid provider headers",
                    transport_trace=trace,
                ) from None
            if request_id is not None:
                if (
                    len(request_id) > 256
                    or not request_id.isascii()
                    or any(not 0x20 <= ord(character) <= 0x7E for character in request_id)
                ):
                    trace["transport_failure_class"] = "invalid_provider_metadata"
                    raise BedrockChatTransportError(
                        "Bedrock Chat Completions returned an invalid AWS request id",
                        transport_trace=trace,
                    )
                trace["provider_request_id"] = request_id
            if status != 200:
                suffix = ""
                if status == 429:
                    try:
                        retry_after = _single_header(response.headers, "Retry-After")
                    except ValueError:
                        trace["transport_failure_class"] = "invalid_provider_metadata"
                        raise BedrockChatTransportError(
                            "Bedrock Chat Completions returned invalid provider headers",
                            transport_trace=trace,
                        ) from None
                    delay = _retry_after_seconds(retry_after)
                    if delay is not None:
                        suffix = f"; retry in {delay:g}s"
                trace["transport_failure_class"] = "http_status"
                raise BedrockChatTransportError(
                    f"Bedrock Chat Completions HTTP {status}{suffix}",
                    transport_trace=trace,
                )
            try:
                _validate_representation_headers(response.headers)
            except ValueError as exc:
                trace["response_representation_valid"] = False
                trace["transport_failure_class"] = "invalid_representation"
                raise BedrockChatTransportError(
                    f"Bedrock Chat Completions invalid representation: {exc}",
                    transport_trace=trace,
                ) from None
            trace["response_representation_valid"] = True
        except (TimeoutError, socket.timeout, _AbsoluteDeadlineExpired) as exc:
            existing_trace = getattr(exc, "transport_trace", None)
            timeout_trace = (
                existing_trace
                if isinstance(existing_trace, dict)
                else self._trace(
                    request_raw=request_raw,
                    status=status,
                    response_raw=None,
                    response_body_complete=False,
                    transport_failure_class="absolute_timeout",
                )
            )
            error = TimeoutError(
                f"Bedrock Chat Completions request timed out after {timeout}s"
            )
            error.transport_trace = timeout_trace  # type: ignore[attr-defined]
            raise error from None
        except BedrockChatTransportError:
            raise
        except ssl.SSLCertVerificationError as exc:
            failure_class = _tls_failure_class(exc)
            message = (
                "Bedrock Chat Completions TLS hostname verification failed"
                if failure_class == "tls_hostname_verification"
                else "Bedrock Chat Completions TLS certificate verification failed"
            )
            raise BedrockChatTLSVerificationError(
                message,
                transport_trace=self._trace(
                    request_raw=request_raw,
                    status=status,
                    response_raw=None,
                    response_body_complete=False,
                    transport_failure_class=failure_class,
                ),
            ) from None
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            if abort_event.is_set() or time.monotonic() >= absolute_deadline:
                error = TimeoutError(
                    f"Bedrock Chat Completions request timed out after {timeout}s"
                )
                error.transport_trace = self._trace(  # type: ignore[attr-defined]
                    request_raw=request_raw,
                    status=status,
                    response_raw=None,
                    response_body_complete=False,
                    transport_failure_class="absolute_timeout",
                )
                raise error from None
            detail = _redact_bearer(str(exc), self._token)[:300]
            raise BedrockChatConnectionError(
                f"Bedrock Chat Completions connection error: {detail}",
                transport_trace=self._trace(
                    request_raw=request_raw,
                    status=status,
                    response_raw=None,
                    response_body_complete=False,
                    transport_failure_class="connection_error",
                ),
            ) from None
        except Exception as exc:
            detail = _redact_bearer(str(exc), self._token)[:300]
            raise BedrockChatTransportError(
                f"Bedrock Chat Completions transport error: {detail}",
                transport_trace=self._trace(
                    request_raw=request_raw,
                    status=status,
                    response_raw=None,
                    response_body_complete=False,
                    transport_failure_class="unexpected_transport_error",
                ),
            ) from None
        finally:
            watchdog.cancel()
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            active_socket = connection.sock
            if active_socket is not None:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            try:
                connection.close()
            except Exception:
                pass
            watchdog.join()

        assert response_raw is not None
        assert trace is not None
        if trace.get("response_body_preimage_redacted") is not False:
            trace["transport_failure_class"] = "unsafe_response_preimage"
            raise BedrockChatTransportError(
                "Bedrock Chat Completions body cannot be retained without bearer redaction",
                transport_trace=trace,
            )
        try:
            payload = _load_strict_json(response_raw)
            trace["response_json_sha256"] = canonical_json_sha256(payload)
        except ValueError as exc:
            trace["transport_failure_class"] = "invalid_json"
            raise BedrockChatTransportError(
                f"Bedrock Chat Completions invalid response JSON: {exc}",
                transport_trace=trace,
            ) from None
        try:
            result = parse_bedrock_chat_payload(
                payload,
                transport_trace=trace,
                expected_model_id=self.expected_model_id,
            )
            result.transport_trace["provider_response_id"] = result.provider_response_id
            result.transport_trace["provider_response_model"] = result.provider_model
            verify_transport_response_preimage(result.transport_trace)
            return result
        except ValueError as exc:
            trace["transport_failure_class"] = "invalid_response_schema"
            raise BedrockChatTransportError(
                f"Bedrock Chat Completions invalid response schema: {exc}",
                transport_trace=trace,
            ) from None
