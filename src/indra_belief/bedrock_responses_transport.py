"""Bounded, dependency-free transport for formal Bedrock Responses lanes.

The three formal Gemma-4 lanes use Bedrock mantle's OpenAI-compatible
Responses endpoint.  This module deliberately has no OpenAI/httpx dependency:
it commits the exact canonical request bytes, permits one frozen HTTPS route,
loads one byte-pinned CA bundle without ambient trust or proxies, rejects
redirects, bounds all provider bytes, and persists replayable wire commitments.

Complete bounded provider bodies are retained as base64 preimages alongside
their raw and canonical hashes.  This lets an offline validator recompute both
commitments.  Partial, oversized, or bearer-containing bodies are explicitly
marked and cannot be mistaken for complete replay evidence.

``build_bedrock_responses_body`` mirrors the previous in-module adapter.  The
legacy adapter remains available for non-formal models and, importantly, as an
independent local test oracle for request and parser parity.
"""
from __future__ import annotations

from dataclasses import dataclass
import base64
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


BACKEND_NAME = "bedrock_responses_raw"
HTTP_METHOD = "POST"
RESPONSES_PATH = "/openai/v1/responses"
FORMAL_BEDROCK_HOST = "bedrock-mantle.us-east-1.api.aws"
FORMAL_RESPONSES_ENDPOINT = (
    "https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses"
)
FORMAL_GEMMA_MODEL_IDS = frozenset(
    {
        "google.gemma-4-e2b",
        "google.gemma-4-26b-a4b",
        "google.gemma-4-31b",
    }
)
DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_CA_BUNDLE_BYTES = 8 * 1024 * 1024
_ALLOWED_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high"})
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
        "provider_response_status",
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


class BedrockResponsesTransportError(RuntimeError):
    """Safe transport error carrying bounded, bearer-redacted provenance."""

    def __init__(self, message: str, *, transport_trace: Mapping[str, Any]):
        super().__init__(message)
        self.transport_trace = dict(transport_trace)


class BedrockResponsesConnectionError(
    BedrockResponsesTransportError, ConnectionError
):
    """Sanitized network failure that remains typed for retry classification."""


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
class BedrockResponsesResult:
    """Normalized semantic result from one Responses payload."""

    content: str
    reasoning: str
    prompt_tokens: int
    output_tokens: int
    reasoning_tokens: int
    finish_reason: str
    reasoning_item_present: bool
    transport_trace: dict[str, Any]
    provider_response_id: str = ""
    provider_model: str = ""
    provider_status: str = ""
    response_body_preimage_b64: str = ""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic strict UTF-8 JSON bytes.

    JSON extensions such as NaN/Infinity, non-JSON Python objects, and strings
    that cannot be represented as valid UTF-8 fail before any paid side effect.
    """

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError) as exc:
        raise ValueError("Responses payload is not strict UTF-8 JSON") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_transport_response_preimage(trace: Mapping[str, Any]) -> bytes:
    """Recompute a complete unredacted response commitment from its trace.

    Validators can call this without transport state or credentials.  Partial,
    oversized, or bearer-redacted evidence fails closed because it cannot prove
    the complete raw provider response.
    """

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
    """Validate the response evidence fields without trusting transport code."""

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
        computed_json_sha = _try_canonical_json_sha256(raw)
        if json_sha != computed_json_sha:
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
) -> BedrockResponsesResult:
    payload = _load_strict_json(raw)
    if payload.get("object") != "response":
        raise ValueError("provider response object differs from the frozen envelope")
    if payload.get("model") != trace["expected_model_id"]:
        raise ValueError("provider response model differs from the frozen model")
    return parse_bedrock_responses_payload(payload, transport_trace=trace)


def validate_transport_trace(
    trace: Mapping[str, Any],
    *,
    expected_request_sha256: str,
    expected_ca_bundle_sha256: str,
    successful: bool,
) -> bytes | None:
    """Validate one formal Responses trace entirely offline.

    The caller supplies commitments already bound to its request and sealed CA
    artifact.  A successful trace returns the exact provider bytes.  A failed
    trace returns exact bytes only when a complete unredacted response was
    retained; partial, absent, oversized, or bearer-omitted evidence returns
    ``None`` after its shape and failure semantics are validated.
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
        or trace["endpoint"] != FORMAL_RESPONSES_ENDPOINT
    ):
        raise ValueError("transport trace differs from the frozen Responses route")
    expected_model_id = trace["expected_model_id"]
    if expected_model_id not in FORMAL_GEMMA_MODEL_IDS:
        raise ValueError("transport trace expected model is not a formal Gemma model")
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
    response_status = trace["provider_response_status"]
    if response_status is not None and response_status not in {"completed", "incomplete"}:
        raise ValueError("transport trace provider response status is invalid")
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
            or response_model != expected_model_id
            or response_status != result.provider_status
        ):
            raise ValueError("transport trace provider response metadata differs from bytes")
        return raw

    if failure_class is None:
        raise ValueError("failed transport trace has no terminal failure class")
    if any(
        value is not None for value in (response_id, response_model, response_status)
    ):
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


def build_bedrock_responses_body(
    *,
    model_id: str,
    system: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """Build the exact semantic body emitted by the legacy adapter.

    ``temperature`` and ``response_format`` are intentionally absent.  The
    legacy Bedrock Responses path accepted-and-ignored both; reasoning models
    can reject temperature, and the endpoint did not receive response_format.
    A ``system`` message inside ``messages`` is folded into ``instructions``
    with the same newline rule as the legacy implementation.
    """

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(system, str):
        raise ValueError("system must be a string")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
    ):
        raise ValueError("max_output_tokens must be a positive integer")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str)
        or reasoning_effort not in _ALLOWED_REASONING_EFFORTS
    ):
        raise ValueError("reasoning_effort is not a supported value")

    instructions = system or ""
    input_items: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role", "user")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}].role is unsupported")
        text = message.get("content", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError(f"messages[{index}].content must be text or null")
        if role == "system":
            instructions = f"{instructions}\n{text}" if instructions else text
            continue
        content_type = "output_text" if role == "assistant" else "input_text"
        input_items.append(
            {
                "role": role,
                "content": [{"type": content_type, "text": text}],
            }
        )

    body: dict[str, Any] = {
        "model": model_id,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
    }
    if instructions:
        body["instructions"] = instructions
    # Preserve the legacy endpoint rule: none/unset omits the reasoning object.
    if reasoning_effort and reasoning_effort != "none":
        body["reasoning"] = {"effort": reasoning_effort}

    canonical_json_bytes(body)
    return body


def _validate_request_body(body: Mapping[str, Any], expected_model_id: str) -> None:
    allowed_keys = {
        "model",
        "input",
        "max_output_tokens",
        "instructions",
        "reasoning",
    }
    if set(body) - allowed_keys:
        raise ValueError("Responses body contains an unsupported field")
    if body.get("model") != expected_model_id:
        raise ValueError("Responses body model differs from the frozen model")
    maximum = body.get("max_output_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("Responses body has an invalid max_output_tokens")
    instructions = body.get("instructions")
    if instructions is not None and (
        not isinstance(instructions, str) or not instructions
    ):
        raise ValueError("Responses body instructions must be non-empty text")
    reasoning = body.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict) or set(reasoning) != {"effort"}:
            raise ValueError("Responses body reasoning must contain only effort")
        effort = reasoning.get("effort")
        if effort not in (_ALLOWED_REASONING_EFFORTS - {"none"}):
            raise ValueError("Responses body reasoning effort is unsupported")

    items = body.get("input")
    if not isinstance(items, list):
        raise ValueError("Responses body input must be a list")
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise ValueError(f"Responses body input[{index}] has an invalid shape")
        role = item.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"Responses body input[{index}] has an invalid role")
        blocks = item.get("content")
        if not isinstance(blocks, list) or len(blocks) != 1:
            raise ValueError(f"Responses body input[{index}] must have one text block")
        block = blocks[0]
        expected_type = "output_text" if role == "assistant" else "input_text"
        if (
            not isinstance(block, dict)
            or set(block) != {"type", "text"}
            or block.get("type") != expected_type
            or not isinstance(block.get("text"), str)
        ):
            raise ValueError(f"Responses body input[{index}] text block is invalid")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            # Do not echo an untrusted provider key into persisted errors.
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
        # This catches escaped lone surrogates and proves the parsed object has a
        # canonical UTF-8 representation before its canonical hash is recorded.
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


def _token_count(
    usage: Mapping[str, Any], field: str, *, required: bool = False
) -> int:
    if field not in usage or usage[field] is None:
        if required:
            raise ValueError(f"provider usage.{field} is missing")
        return -1
    value = usage[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"provider usage.{field} must be a non-negative integer")
    return value


def _reasoning_token_count(usage: Mapping[str, Any]) -> int:
    if "output_tokens_details" not in usage or usage["output_tokens_details"] is None:
        return -1
    details = usage["output_tokens_details"]
    if not isinstance(details, dict):
        raise ValueError("provider usage.output_tokens_details must be an object")
    return _token_count(details, "reasoning_tokens")


def _blocks(item: Mapping[str, Any], *, item_index: int) -> list[dict[str, Any]]:
    value = item.get("content")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"provider output[{item_index}].content must be a list")
    result: list[dict[str, Any]] = []
    for block_index, block in enumerate(value):
        if not isinstance(block, dict):
            raise ValueError(
                f"provider output[{item_index}].content[{block_index}] must be an object"
            )
        text = block.get("text")
        if text is not None and not isinstance(text, str):
            raise ValueError(
                f"provider output[{item_index}].content[{block_index}].text must be text"
            )
        result.append(block)
    return result


def parse_bedrock_responses_payload(
    payload: Mapping[str, Any], *, transport_trace: Mapping[str, Any]
) -> BedrockResponsesResult:
    """Parse the current Bedrock Responses shapes with legacy field parity."""

    if "output" not in payload or not isinstance(payload["output"], list):
        raise ValueError("provider output must be a list")
    output: list[Any] = payload["output"]

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    reasoning_item_present = False
    for item_index, item in enumerate(output):
        if not isinstance(item, dict):
            raise ValueError(f"provider output[{item_index}] must be an object")
        item_type = item.get("type")
        if item_type is not None and not isinstance(item_type, str):
            raise ValueError(f"provider output[{item_index}].type must be text")
        if item_type not in {"reasoning", "message"}:
            raise ValueError(f"provider output[{item_index}].type is unsupported")
        blocks = _blocks(item, item_index=item_index)
        if item_type == "reasoning":
            reasoning_item_present = True
            for block_index, block in enumerate(blocks):
                if block.get("type") not in {None, "reasoning_text"}:
                    raise ValueError(
                        f"provider reasoning block {block_index} has unsupported type"
                    )
                if block.get("text"):
                    reasoning_parts.append(block["text"])
        elif item_type == "message":
            if item.get("role") != "assistant":
                raise ValueError("provider message output role must be assistant")
            for block_index, block in enumerate(blocks):
                if block.get("type") != "output_text":
                    raise ValueError(
                        f"provider message block {block_index} has unsupported type"
                    )
                if block.get("text"):
                    content_parts.append(block["text"])

    usage_value = payload.get("usage")
    if not isinstance(usage_value, dict):
        raise ValueError("provider usage must be an object")
    usage: Mapping[str, Any] = usage_value

    provider_status = payload.get("status")
    if provider_status not in {"completed", "incomplete"}:
        raise ValueError("provider status must be completed or incomplete")
    # Exact legacy mapping: only an incomplete generation maps to length.
    finish_reason = "length" if provider_status == "incomplete" else "stop"
    provider_response_id = payload.get("id")
    if (
        not isinstance(provider_response_id, str)
        or not provider_response_id
        or len(provider_response_id) > 512
        or not provider_response_id.isascii()
        or any(
            not 0x20 <= ord(character) <= 0x7E
            for character in provider_response_id
        )
    ):
        raise ValueError("provider response id must be bounded printable text")
    provider_model = payload.get("model")
    if not isinstance(provider_model, str) or not provider_model:
        raise ValueError("provider response model must be non-empty text")
    response_preimage = transport_trace.get("response_body_preimage_b64")
    if not isinstance(response_preimage, str):
        raise ValueError("provider response preimage is missing")
    return BedrockResponsesResult(
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        prompt_tokens=_token_count(usage, "input_tokens", required=True),
        output_tokens=_token_count(usage, "output_tokens", required=True),
        reasoning_tokens=_reasoning_token_count(usage),
        finish_reason=finish_reason,
        reasoning_item_present=reasoning_item_present,
        provider_response_id=provider_response_id,
        provider_model=provider_model,
        provider_status=provider_status,
        response_body_preimage_b64=response_preimage,
        transport_trace=dict(transport_trace),
    )


def _explicit_tls_context(
    ca_bundle: Path,
) -> tuple[ssl.SSLContext, str, str]:
    """Load exact CA bytes via cadata, with an explicit cafile capability fallback."""

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
        # Some Python/TLS builds do not expose the cadata capability.  The
        # fallback names the same explicit file; it never calls load_default_certs.
        context = _new_tls_context()
        try:
            context.load_verify_locations(cafile=str(ca_bundle))
        except (OSError, ssl.SSLError) as exc:
            raise ValueError("TLS CA bundle could not be loaded by explicit path") from exc
        # Detect a cooperative file change across the capability fallback.
        if _read_file_bounded(ca_bundle, MAX_CA_BUNDLE_BYTES) != raw:
            raise ValueError("TLS CA bundle changed while it was loaded")
        return context, digest, "cafile_fallback"
    except ssl.SSLError as exc:
        # Malformed PEM is not a capability failure and must not be made valid by
        # silently trying a different trust-loading mechanism.
        raise ValueError("TLS CA bundle could not be loaded as PEM cadata") from exc


def build_pinned_https_opener(
    ca_bundle: str | Path,
    *,
    expected_ca_bundle_sha256: str,
) -> tuple[urllib.request.OpenerDirector, str, str]:
    """Return a no-proxy/no-redirect opener with one byte-pinned CA bundle."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_ca_bundle_sha256):
        raise ValueError("expected TLS CA bundle SHA-256 is invalid")
    context, digest, load_mode = _explicit_tls_context(Path(ca_bundle))
    if digest != expected_ca_bundle_sha256:
        raise ValueError("TLS CA bundle differs from its frozen SHA-256")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirects(),
        urllib.request.HTTPSHandler(context=context),
    )
    return opener, digest, load_mode


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


class RawBedrockResponsesTransport:
    """One exact, bounded, proxy-free and redirect-free Responses HTTP client."""

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
            lane_label="Responses",
            expected_path=RESPONSES_PATH,
            formal_endpoint=FORMAL_RESPONSES_ENDPOINT,
            formal_host=FORMAL_BEDROCK_HOST,
        )
        if not isinstance(expected_model_id, str) or not expected_model_id:
            raise ValueError("expected_model_id must be a non-empty string")
        if not allow_insecure_http_for_tests and expected_model_id not in FORMAL_GEMMA_MODEL_IDS:
            raise ValueError("paid Responses model is not a formal Gemma model")
        if (
            not isinstance(bearer_token, str)
            or not bearer_token
            or len(bearer_token) > 16 * 1024
            or not bearer_token.isascii()
            or any(character.isspace() or ord(character) < 0x20 for character in bearer_token)
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
            raise ValueError("paid Responses transport requires the frozen byte limits")

        ca_sha256: str | None = None
        ca_load_mode = "disabled_loopback_test"
        tls_context: ssl.SSLContext | None = None
        if allow_insecure_http_for_tests:
            if ca_bundle is not None or expected_ca_bundle_sha256 is not None:
                raise ValueError("test-only HTTP transport cannot take TLS CA material")
        else:
            if ca_bundle is None or expected_ca_bundle_sha256 is None:
                raise ValueError("paid Responses transport requires a byte-pinned CA bundle")
            tls_context, ca_sha256, ca_load_mode = _explicit_tls_context(
                Path(ca_bundle)
            )
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
                lane_label="Bedrock Responses",
                dns_thread_name="bedrock-responses-dns",
            )
        return _DeadlineHTTPConnection(
            self._endpoint_host,
            self._endpoint_port,
            timeout=timeout,
            absolute_deadline=absolute_deadline,
            abort_event=abort_event,
            lane_label="Bedrock Responses",
            dns_thread_name="bedrock-responses-dns",
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
            response_body_complete = (
                response_raw is not None and not response_size_exceeded
            )
        token_bytes = self._token.encode("utf-8")
        persisted_raw = response_raw
        preimage_redacted = False
        if persisted_raw is not None and token_bytes in persisted_raw:
            # Retaining transformed bytes beside the digest of the original can
            # look like an exact preimage to a permissive consumer.  Omit the
            # body entirely and preserve only the fact of bearer redaction.
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
                len(response_raw) if response_raw is not None and response_body_complete else None
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
            "provider_response_status": None,
        }

    def _request_bytes(self, body: Mapping[str, Any]) -> bytes:
        if not isinstance(body, Mapping):
            raise ValueError("Responses body must be an object")
        materialized = dict(body)
        _validate_request_body(materialized, self.expected_model_id)
        raw = canonical_json_bytes(materialized)
        if len(raw) > self._max_request_bytes:
            raise ValueError("Responses request exceeds the safety limit")
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
        target = (
            declared if declared is not None else self._max_response_bytes + 1
        )
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
        """Return the exact pre-side-effect commitment for sealed validators."""

        return self._trace(
            request_raw=self._request_bytes(body),
            status=None,
            response_raw=None,
        )

    def call(
        self, body: Mapping[str, Any], *, timeout: int | float
    ) -> BedrockResponsesResult:
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
                RESPONSES_PATH,
                body=request_raw,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(request_raw)),
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "User-Agent": "indra-belief/raw-bedrock-responses-v1",
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
                    deadline_expired = (
                        abort_event.is_set()
                        or time.monotonic() >= absolute_deadline
                    )
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
                raise BedrockResponsesTransportError(
                    "Bedrock Responses invalid response framing",
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
                        f"Bedrock Responses request timed out after {timeout}s"
                    )
                    error.transport_trace = trace  # type: ignore[attr-defined]
                    raise error from None
                if exc.connection_error:
                    raise BedrockResponsesConnectionError(
                        "Bedrock Responses connection interrupted during response",
                        transport_trace=trace,
                    ) from None
                raise BedrockResponsesTransportError(
                    "Bedrock Responses invalid response framing",
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
                raise BedrockResponsesTransportError(
                    "Bedrock Responses response exceeds the safety limit",
                    transport_trace=trace,
                )
            try:
                request_id = _single_header(response.headers, "x-amzn-requestid")
            except ValueError:
                trace["transport_failure_class"] = "invalid_provider_metadata"
                raise BedrockResponsesTransportError(
                    "Bedrock Responses returned invalid provider headers",
                    transport_trace=trace,
                ) from None
            if request_id is not None:
                if (
                    len(request_id) > 256
                    or not request_id.isascii()
                    or any(
                        not 0x20 <= ord(character) <= 0x7E
                        for character in request_id
                    )
                ):
                    trace["transport_failure_class"] = "invalid_provider_metadata"
                    raise BedrockResponsesTransportError(
                        "Bedrock Responses returned an invalid AWS request id",
                        transport_trace=trace,
                    )
                trace["provider_request_id"] = request_id
            if status != 200:
                suffix = ""
                if status == 429:
                    try:
                        retry_after = _single_header(
                            response.headers, "Retry-After"
                        )
                    except ValueError:
                        trace["transport_failure_class"] = "invalid_provider_metadata"
                        raise BedrockResponsesTransportError(
                            "Bedrock Responses returned invalid provider headers",
                            transport_trace=trace,
                        ) from None
                    delay = _retry_after_seconds(retry_after)
                    if delay is not None:
                        suffix = f"; retry in {delay:g}s"
                trace["transport_failure_class"] = "http_status"
                raise BedrockResponsesTransportError(
                    f"Bedrock Responses HTTP {status}{suffix}",
                    transport_trace=trace,
                )
            try:
                _validate_representation_headers(response.headers)
            except ValueError as exc:
                trace["response_representation_valid"] = False
                trace["transport_failure_class"] = "invalid_representation"
                raise BedrockResponsesTransportError(
                    f"Bedrock Responses invalid representation: {exc}",
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
            error = TimeoutError(f"Bedrock Responses request timed out after {timeout}s")
            error.transport_trace = timeout_trace  # type: ignore[attr-defined]
            raise error from None
        except BedrockResponsesTransportError:
            raise
        except ssl.SSLCertVerificationError as exc:
            failure_class = _tls_failure_class(exc)
            raise BedrockResponsesTransportError(
                (
                    "Bedrock Responses TLS hostname verification failed"
                    if failure_class == "tls_hostname_verification"
                    else "Bedrock Responses TLS certificate verification failed"
                ),
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
                    f"Bedrock Responses request timed out after {timeout}s"
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
            raise BedrockResponsesConnectionError(
                f"Bedrock Responses connection error: {detail}",
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
            raise BedrockResponsesTransportError(
                f"Bedrock Responses transport error: {detail}",
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

        # ``response_raw`` is proven non-None here by the non-exceeded 200 path.
        assert response_raw is not None
        assert trace is not None
        if trace.get("response_body_preimage_redacted") is not False:
            trace["transport_failure_class"] = "unsafe_response_preimage"
            raise BedrockResponsesTransportError(
                "Bedrock Responses body cannot be retained without bearer redaction",
                transport_trace=trace,
            )
        try:
            payload = _load_strict_json(response_raw)
            canonical_sha = canonical_json_sha256(payload)
            trace["response_json_sha256"] = canonical_sha
        except ValueError:
            trace["transport_failure_class"] = "invalid_json"
            raise BedrockResponsesTransportError(
                "Bedrock Responses returned malformed strict JSON",
                transport_trace=trace,
            ) from None
        if payload.get("model") != self.expected_model_id:
            trace["transport_failure_class"] = "invalid_response_schema"
            raise BedrockResponsesTransportError(
                "Bedrock Responses echoed a model other than the frozen model",
                transport_trace=trace,
            )
        response_object = payload.get("object")
        if response_object != "response":
            trace["transport_failure_class"] = "invalid_response_schema"
            raise BedrockResponsesTransportError(
                "Bedrock Responses returned an unexpected object type",
                transport_trace=trace,
            )
        try:
            result = parse_bedrock_responses_payload(payload, transport_trace=trace)
            result.transport_trace["provider_response_id"] = result.provider_response_id
            result.transport_trace["provider_response_model"] = result.provider_model
            result.transport_trace["provider_response_status"] = result.provider_status
            verify_transport_response_preimage(result.transport_trace)
            return result
        except ValueError as exc:
            trace["transport_failure_class"] = "invalid_response_schema"
            raise BedrockResponsesTransportError(
                f"Bedrock Responses invalid response schema: {exc}",
                transport_trace=trace,
            ) from None
