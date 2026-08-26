"""Shared, dependency-free stack for the formal Bedrock Responses raw transport.

The Responses lane in
`src/indra_belief/bedrock_responses_transport.py::RawBedrockResponsesTransport`
is the only raw transport that imports this base. The shared posture machinery
— the monotonic deadline connection mixin, bounded DNS resolver, pinned-TLS
context factory, and response-framing/header validators — lives here as a
stdlib-only leaf, an invariant enforced by
`tests/test_bedrock_responses_transport.py::test_only_stdlib_import_roots_are_in_bedrock_transport_base`.

The lane-labelled helpers
(`src/indra_belief/bedrock_transport_base.py::_DeadlineConnectionMixin`,
`src/indra_belief/bedrock_transport_base.py::_resolve_host_bounded`, and
`src/indra_belief/bedrock_transport_base.py::_safe_endpoint`) take per-lane
strings as parameters, preserving lane-specific error text without binding this
base to a particular API. The canonical-JSON codec and exception taxonomy stay
with the Responses transport because its ``ValueError`` messages are lane-specific.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import math
from pathlib import Path
import re
import socket
import ssl
import threading
import time
from typing import Any, Callable
import urllib.parse
import urllib.request


MAX_CA_BUNDLE_BYTES = 8 * 1024 * 1024


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


def _explicit_tls_context(
    ca_bundle: Path,
    *,
    context_factory: Callable[[], ssl.SSLContext] | None = None,
    max_bytes: int = MAX_CA_BUNDLE_BYTES,
) -> tuple[ssl.SSLContext, str, str]:
    """Load exact CA bytes via cadata, with an explicit cafile capability fallback.

    ``context_factory`` is injected by each lane module so that a test which
    patches THAT module's ``_new_tls_context`` still governs the context this
    builds; resolving the name in this module's globals would silently ignore
    the patch.
    """
    new_context = context_factory or _new_tls_context
    raw = _read_file_bounded(ca_bundle, max_bytes)
    digest = hashlib.sha256(raw).hexdigest()
    try:
        cadata = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("TLS CA bundle is not an ASCII PEM bundle") from exc

    context = new_context()
    try:
        context.load_verify_locations(cadata=cadata)
        return context, digest, "cadata"
    except (TypeError, NotImplementedError):
        # Some Python/TLS builds do not expose the cadata capability.  The
        # fallback names the same explicit file; it never calls load_default_certs.
        context = new_context()
        try:
            context.load_verify_locations(cafile=str(ca_bundle))
        except (OSError, ssl.SSLError) as exc:
            raise ValueError("TLS CA bundle could not be loaded by explicit path") from exc
        # Detect a cooperative file change across the capability fallback.
        if _read_file_bounded(ca_bundle, max_bytes) != raw:
            raise ValueError("TLS CA bundle changed while it was loaded")
        return context, digest, "cafile_fallback"
    except ssl.SSLError as exc:
        # Malformed PEM is not a capability failure and must not be made valid by
        # silently trying a different trust-loading mechanism.
        raise ValueError("TLS CA bundle could not be loaded as PEM cadata") from exc


def build_pinned_https_opener(
    ca_bundle: str | Path,
    *,
    expected_ca_bundle_sha256: str | None = None,
    context_factory: Callable[[], ssl.SSLContext] | None = None,
    max_bytes: int = MAX_CA_BUNDLE_BYTES,
) -> tuple[urllib.request.OpenerDirector, str, str]:
    """Return a no-proxy/no-redirect opener with one byte-pinned CA bundle."""

    if expected_ca_bundle_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_ca_bundle_sha256
    ):
        raise ValueError("expected TLS CA bundle SHA-256 is invalid")
    context, digest, load_mode = _explicit_tls_context(
        Path(ca_bundle), context_factory=context_factory, max_bytes=max_bytes
    )
    if expected_ca_bundle_sha256 is not None and digest != expected_ca_bundle_sha256:
        raise ValueError("TLS CA bundle differs from its frozen SHA-256")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirects(),
        urllib.request.HTTPSHandler(context=context),
    )
    return opener, digest, load_mode


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


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value
