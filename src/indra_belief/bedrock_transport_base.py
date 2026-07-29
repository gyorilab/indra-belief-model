"""Shared, dependency-free stack for the formal Bedrock raw transports.

Both the Responses and Chat-Completions raw transports enforce the same
bounded, proxy-free, redirect-free, ambient-trust-free HTTP posture.  The
byte-identical machinery that posture is built from — the monotonic deadline
connection mixin, the bounded DNS resolver, the pinned-TLS context factory,
the response-framing/header validators, and the transport-trace field
validators — lives here once so each transport keeps only its own
API-specific request-body builder, payload parser, and control flow.

The lane-labelled helpers (the deadline mixin, ``_resolve_host_bounded``, and
``_safe_endpoint``) take their per-lane strings as parameters so each transport
reproduces its own error text char-for-char.  Nothing here is aware of a
particular API; the canonical-JSON codec and the exception taxonomy stay with
each transport because their ``ValueError`` messages are per-lane.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import http.client
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


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class _AbsoluteDeadlineExpired(TimeoutError):
    pass


def _resolve_host_bounded(
    host: str,
    port: int,
    *,
    absolute_deadline: float,
    abort_event: threading.Event,
    lane_label: str = "Bedrock",
    thread_name: str = "bedrock-dns",
) -> list[tuple[int, int, int, Any]]:
    """Resolve without allowing libc DNS to escape the transport deadline.

    Python has no portable cancellable ``getaddrinfo``. Resolution therefore
    runs in one daemon helper. A timeout can leave only that pre-network helper
    to unwind, never a live or billable HTTP request.
    """

    finished = threading.Event()
    result: dict[str, Any] = {}

    def resolve() -> None:
        try:
            result["addresses"] = socket.getaddrinfo(
                host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        except BaseException as exc:  # always release the bounded waiter
            result["error"] = exc
        finally:
            finished.set()

    resolver = threading.Thread(
        target=resolve,
        name=thread_name,
        daemon=True,
    )
    resolver.start()
    remaining = absolute_deadline - time.monotonic()
    if remaining <= 0 or not finished.wait(remaining):
        raise _AbsoluteDeadlineExpired(f"{lane_label} DNS deadline expired")
    if abort_event.is_set() or time.monotonic() >= absolute_deadline:
        raise _AbsoluteDeadlineExpired(f"{lane_label} DNS deadline expired")
    error = result.get("error")
    if error is not None:
        if isinstance(error, Exception):
            raise error
        raise OSError(f"{lane_label} DNS resolution failed")
    raw_addresses = result.get("addresses")
    if not isinstance(raw_addresses, list):
        raise OSError(f"{lane_label} DNS resolution returned no addresses")
    addresses: list[tuple[int, int, int, Any]] = []
    seen: set[tuple[int, int, int, Any]] = set()
    for item in raw_addresses:
        if not isinstance(item, tuple) or len(item) != 5:
            continue
        family, socktype, proto, _canonname, sockaddr = item
        if (
            family not in {socket.AF_INET, socket.AF_INET6}
            or socktype != socket.SOCK_STREAM
            or proto not in {0, socket.IPPROTO_TCP}
        ):
            continue
        key = (family, socktype, proto, sockaddr)
        if key in seen:
            continue
        seen.add(key)
        addresses.append((family, socktype, proto, sockaddr))
        if len(addresses) == 16:
            break
    if not addresses:
        raise OSError(f"{lane_label} DNS resolution returned no usable addresses")
    return addresses


class _DeadlineConnectionMixin:
    """Enforce a monotonic deadline across connect/TLS/send operations.

    ``lane_label`` and ``dns_thread_name`` are supplied by each transport's
    connection factory so the per-lane error text and daemon-thread name are
    reproduced exactly; the direct-construction defaults keep the mixin usable
    without a lane binding.
    """

    def __init__(
        self,
        *args: Any,
        absolute_deadline: float,
        abort_event: threading.Event,
        lane_label: str = "Bedrock",
        dns_thread_name: str = "bedrock-dns",
        **kwargs: Any,
    ) -> None:
        self._absolute_deadline = absolute_deadline
        self._abort_event = abort_event
        self._lane_label = lane_label
        self._dns_thread_name = dns_thread_name
        super().__init__(*args, **kwargs)

    def _remaining(self) -> float:
        remaining = self._absolute_deadline - time.monotonic()
        if self._abort_event.is_set() or remaining <= 0:
            raise _AbsoluteDeadlineExpired(f"{self._lane_label} absolute deadline expired")
        return remaining

    def _arm_socket(self) -> None:
        if self.sock is not None:
            self.sock.settimeout(self._remaining())

    def _resolved_socket(self) -> socket.socket:
        addresses = _resolve_host_bounded(
            self.host,
            self.port,
            absolute_deadline=self._absolute_deadline,
            abort_event=self._abort_event,
            lane_label=self._lane_label,
            thread_name=self._dns_thread_name,
        )
        last_error: OSError | None = None
        for family, socktype, proto, sockaddr in addresses:
            candidate = socket.socket(family, socktype, proto)
            try:
                candidate.settimeout(self._remaining())
                if self.source_address:
                    candidate.bind(self.source_address)
                candidate.connect(sockaddr)
                return candidate
            except OSError as exc:
                last_error = exc
                candidate.close()
                self._remaining()
        if last_error is not None:
            raise last_error
        raise OSError(f"{self._lane_label} hostname resolved to no usable address")

    def connect(self) -> None:
        self._remaining()
        if self._tunnel_host is not None:
            raise RuntimeError(f"{self._lane_label} proxy tunnels are disabled")
        self.sock = self._resolved_socket()
        try:
            # Close the getaddrinfo→socket-publication race: if the watchdog
            # fired while the candidate was still local, this check closes it
            # before a TLS handshake can begin with a stale timeout.
            self._arm_socket()
            if isinstance(self, http.client.HTTPSConnection):
                self.sock = self._context.wrap_socket(
                    self.sock,
                    server_hostname=self.host,
                )
            self._arm_socket()
        except Exception:
            self.close()
            raise

    def send(self, data: Any) -> None:
        self._remaining()
        self._arm_socket()
        super().send(data)
        self._arm_socket()


class _DeadlineHTTPConnection(_DeadlineConnectionMixin, http.client.HTTPConnection):
    pass


class _DeadlineHTTPSConnection(_DeadlineConnectionMixin, http.client.HTTPSConnection):
    pass


def _safe_endpoint(
    endpoint: str,
    *,
    expected_endpoint: str,
    allow_insecure_http_for_tests: bool,
    lane_label: str,
    expected_path: str,
    formal_endpoint: str,
    formal_host: str,
) -> urllib.parse.SplitResult:
    if endpoint != expected_endpoint:
        raise ValueError(f"{lane_label} endpoint differs from its frozen expectation")
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{lane_label} endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{lane_label} endpoint must not contain query or fragment")
    if parsed.path != expected_path:
        raise ValueError(f"{lane_label} endpoint path must be {expected_path}")
    if not parsed.hostname:
        raise ValueError(f"{lane_label} endpoint has no host")
    if allow_insecure_http_for_tests:
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("test-only HTTP is restricted to loopback")
    else:
        if endpoint != formal_endpoint:
            raise ValueError(f"paid {lane_label} endpoint differs from the formal frozen route")
        if parsed.scheme != "https" or parsed.port not in {None, 443}:
            raise ValueError(f"paid {lane_label} endpoint must use HTTPS port 443")
        if parsed.hostname != formal_host:
            raise ValueError(f"paid {lane_label} endpoint host must be {formal_host}")
    return parsed


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _read_file_bounded(path: Path, maximum: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError(f"TLS CA bundle exceeds {maximum} bytes")
    if not raw:
        raise ValueError("TLS CA bundle is empty")
    return raw


def _new_tls_context() -> ssl.SSLContext:
    # SSLContext(), unlike create_default_context(), begins without ambient
    # platform roots.  Only load_verify_locations below can add trust anchors.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    return context


def _redact_bearer(text: str, token: str) -> str:
    safe = text.replace(token, "[REDACTED]") if token else text
    return re.sub(r"(?i)Bearer\s+[^\s,;]+", "Bearer [REDACTED]", safe)


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            seconds = (target - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds):
        return None
    if seconds < 0:
        return 0.0
    return min(seconds, 3600.0)


def _single_header(headers: Any, name: str) -> str | None:
    values = headers.get_all(name) if hasattr(headers, "get_all") else None
    if values is None:
        value = headers.get(name) if headers is not None else None
        values = [] if value is None else [value]
    if len(values) > 1:
        raise ValueError(f"provider returned multiple {name} headers")
    if not values:
        return None
    value = values[0]
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError(f"provider returned an invalid {name} header")
    return value.strip()


def _declared_content_length(headers: Any) -> int | None:
    value = _single_header(headers, "Content-Length")
    if value is None:
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError("provider Content-Length is invalid")
    result = int(value)
    if result < 0:
        raise ValueError("provider Content-Length is negative")
    return result


def _response_framing(headers: Any) -> tuple[str, int | None]:
    transfer_encoding = _single_header(headers, "Transfer-Encoding")
    declared = _declared_content_length(headers)
    if transfer_encoding is not None:
        if transfer_encoding.lower() != "chunked":
            raise ValueError("provider Transfer-Encoding must be exactly chunked")
        if declared is not None:
            raise ValueError("provider response ambiguously carries both TE and CL")
        return "chunked", None
    if declared is not None:
        return "content_length", declared
    return "connection_close", None


def _trace_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"transport trace {field} is not an integer >= {minimum}")
    return value


def _trace_optional_int(value: Any, *, field: str, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _trace_int(value, field=field, minimum=minimum)


def _trace_digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"transport trace {field} is not a lowercase SHA-256")
    return value


def _trace_optional_digest(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _trace_digest(value, field=field)


def _trace_printable_id(value: Any, *, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or not value.isascii()
        or any(not 0x20 <= ord(character) <= 0x7E for character in value)
    ):
        raise ValueError(f"transport trace {field} is not bounded printable text")
    return value


def _validate_trace_framing(
    trace: Mapping[str, Any], *, kind: str | None, raw: bytes | None
) -> None:
    framing = trace["response_framing"]
    if framing not in {None, "content_length", "chunked", "connection_close"}:
        raise ValueError("transport trace response framing is unsupported")
    framing_valid = trace["response_framing_valid"]
    if framing_valid is not None and type(framing_valid) is not bool:
        raise ValueError("transport trace response framing validity is invalid")
    declared = _trace_optional_int(
        trace["response_content_length_declared"],
        field="response_content_length_declared",
    )
    if framing == "content_length":
        if declared is None:
            raise ValueError("transport trace content-length framing has no length")
    elif declared is not None:
        raise ValueError("transport trace declares Content-Length under foreign framing")
    if framing is None and framing_valid is True:
        raise ValueError("transport trace claims valid framing without a framing mode")
    if kind == "complete":
        assert raw is not None
        if framing_valid is not True or framing is None:
            raise ValueError("transport trace complete response lacks valid framing")
        if declared is not None and declared != len(raw):
            raise ValueError("transport trace complete response differs from Content-Length")
    elif kind == "partial_prefix":
        assert raw is not None
        if framing is None or framing_valid is None:
            raise ValueError("transport trace partial response lacks framing evidence")
        if declared is not None and len(raw) > declared:
            raise ValueError("transport trace partial response exceeds Content-Length")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value
