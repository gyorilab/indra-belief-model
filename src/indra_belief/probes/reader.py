"""Read the direct verdict probe through an OpenAI-compatible client.

The verdict key is supplied as an assistant prefill, so the one generated token
is the verdict value itself.  This geometry is intentional: it makes position
zero a closed choice between ``correct`` and ``incorrect`` without asking the
model to generate JSON formatting first.

``ModelClient.call`` does not currently expose the continuation flags required
by this read.  The function therefore uses the configured OpenAI-compatible
transport held by an *unguarded* model client and sends those flags explicitly.
A compatible test or serving adapter only needs ``config`` plus a
``chat.completions.create`` transport (directly, or under ``_client``).  A
``GuardedModelClient`` is rejected explicitly: unwrapping it would evade the
reservation and settlement that make that client safe.  For a plain
``ModelClient`` this path retains its hard wall timeout, but cannot enter
``ModelClient.call``'s call-log or 429-retry machinery until that public method
can express assistant continuation.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from collections.abc import Mapping
from functools import partial
from typing import Any, NamedTuple

from indra_belief.logprobs import label_probability
from indra_belief.model_client import _normalize_openai_logprobs
from indra_belief.probes.battery import LABELS, probe_by_id, render


DIRECT_PROBE_ID = "pol.verdict_direct"
# The SAME quantity -- the verdict label's log-odds -- obtained from the scoring
# call itself rather than from a second forced-position request. It gets its own
# id because the two routes are NOT interchangeable: measured n=80, the in-call
# margin runs ~3x wider (median |13.22| against |4.34|), so an artifact fitted on
# one and applied to the other saturates. The id is what makes that visible in
# the artifact instead of inferable from where the file happened to come from.
IN_CALL_PROBE_ID = "pol.verdict_incall"

# The window a probe-capable serving entry must be ABLE to return. vLLM defaults
# to 20 and stock mlx_lm.server hard-codes 11, so both need raising: vLLM with
# `--max-logprobs 1024` at launch, MLX with the patch in scripts/serve_mlx.sh.
# A registry entry declaring `max_top_logprobs` asserts its server was started
# that way. This is the CEILING, not what we ask for on a typical call.
PROBE_TOP_LOGPROBS = 1024

# What we actually request first. MEASURED on 40 records over HTTP: the LOSING
# label's rank is median 6, p90 11, max 15 — so 128 carries ~8x headroom over
# the worst case observed. The window costs no latency (0.16-0.17 s flat from 64
# to 1024 on MLX) but the payload is linear in it: 3.8 KB at 64 against 54.7 KB
# at 1024. At corpus scale that is the difference between ~230 GB and ~3.2 TB of
# JSON for the same two numbers.
#
# It is a FIRST TRY, not a cap: a label outside the window raises ProbeTopKError,
# and `read_probe` retries that record once at the route's full width. So a
# too-narrow default costs a rare second call rather than a lost reading, and a
# different stack (vLLM bf16 vs MLX 8-bit could rank differently) degrades to
# slower rather than wrong.
PROBE_FIRST_TRY_TOP_LOGPROBS = 128

# How often the first try was too narrow. The 128 default is measured on MLX
# (losing-label rank median 6, max 15 over 40 records); another stack may rank
# differently, and without a count the first run there teaches us nothing. Read
# it after a run: a high rate means re-measure the width for that stack rather
# than paying a second call on most records.
_WIDENED = 0


def probe_widen_count() -> int:
    """Times the first-try window was too narrow and had to be widened."""
    return _WIDENED


class ProbeReading(NamedTuple):
    """The two values measured at the forced verdict position."""

    p_raw: float
    delta_logit: float

    @property
    def label_log_odds(self) -> float:
        """Alias spelling out that ``delta_logit`` is label log-odds."""
        return self.delta_logit


class ProbeReadError(RuntimeError):
    """The transport did not yield a complete two-label probe reading."""


class ProbeTopKError(ProbeReadError):
    """At least one verdict label fell outside the returned top-k window."""


def _configuration(client: Any) -> Mapping[str, Any]:
    config = getattr(client, "config", None)
    if not isinstance(config, Mapping):
        raise ProbeReadError("probe client must expose a mapping-valued config")
    return config


def _completion_create(client: Any):
    """Return the client's OpenAI-compatible completion callable."""
    if getattr(client, "_guard", None) is not None:
        raise ProbeReadError(
            "guarded model clients cannot perform the forced-verdict read: "
            "its continuation request is not representable by ModelClient.call"
        )
    for transport in (client, getattr(client, "_client", None)):
        if transport is None:
            continue
        create = getattr(
            getattr(getattr(transport, "chat", None), "completions", None),
            "create",
            None,
        )
        if callable(create):
            return create
    raise ProbeReadError(
        "probe client must expose an OpenAI-compatible "
        "chat.completions.create transport"
    )


def _label_logprobs(
    logprobs: list[dict[str, Any]],
    *,
    top_k: int,
) -> tuple[float, float]:
    """Read the two registered one-token labels from position zero."""
    alternatives = logprobs[0].get("top") or []
    observed: dict[str, float] = {}
    for alternative in alternatives:
        token = alternative.get("token")
        if token not in LABELS:
            continue
        if token in observed:
            raise ProbeReadError(f"probe response repeats label token {token!r}")
        try:
            value = float(alternative["logprob"])
        except (KeyError, TypeError, ValueError) as error:
            raise ProbeReadError(
                f"probe response has an invalid logprob for {token!r}"
            ) from error
        if not math.isfinite(value):
            raise ProbeReadError(
                f"probe response has a non-finite logprob for {token!r}"
            )
        observed[token] = value

    missing = [label for label in LABELS if label not in observed]
    if missing:
        raise ProbeTopKError(
            f"{', '.join(missing)} fell outside the returned top-{top_k} window; "
            "correct and incorrect must both occur"
        )
    return observed["correct"], observed["incorrect"]



def build_probe_request(
    record: Mapping[str, Any], *, model_id: str, top_logprobs: int,
    inline_extra_body: bool = False,
) -> dict[str, Any]:
    """The probe's request body, for ANY transport.

    Split out from :func:`read_probe` so a caller with its own HTTP client — the
    corpus-scale shard runner drives a bare ``httpx.Client``, not a
    ``ModelClient`` — can issue the probe without adopting our transport. One
    definition of the request shape, so the three things that make the read work
    cannot drift apart:

      * ``enable_thinking: False`` suppresses the reasoning channel;
      * the assistant PREFILL opens the JSON string the label must close;
      * ``continue_final_message`` extends that turn instead of starting a new
        one, which is what puts the verdict at generated position zero.

    Getting any one of those wrong yields a saturated read rather than an error
    — MEASURED: with the thinking channel off but the production prompt, the
    verdict lands 56 tokens deep at delta_logit +22.50, indistinguishable from
    full deliberation.
    """
    probe = probe_by_id(DIRECT_PROBE_ID)
    system, user, prefill = render(probe, record)
    extra = {
        "chat_template_kwargs": {"enable_thinking": False},
        "continue_final_message": True,
        "add_generation_prompt": False,
    }
    body = {
        "model": model_id,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": prefill},
        ],
    }
    # The OpenAI SDK hoists `extra_body` onto the wire; a raw HTTP client must
    # send those keys at the top level itself. Same request either way — the
    # difference is the transport's, so it is named here rather than
    # reconstructed by every caller that does not use the SDK.
    if inline_extra_body:
        body.update(extra)
    else:
        body["extra_body"] = extra
    return body


def _as_choice(payload: Any) -> Any:
    """Adapt a plain JSON dict to the attribute shape the normalizer reads.

    ``_normalize_openai_logprobs`` handles two real provider quirks (the entry
    scalar reporting argmax rather than the sampled token, and bare ``{}``
    positions). Reimplementing that for dict payloads would duplicate exactly the
    knowledge it exists to hold, so a raw response is adapted INTO it instead.
    """
    if isinstance(payload, Mapping):
        return SimpleNamespace(**{k: _as_choice(v) for k, v in payload.items()})
    if isinstance(payload, list):
        return [_as_choice(v) for v in payload]
    return payload


def probe_reading_from_payload(payload: Any, *, top_k: int) -> ProbeReading:
    """Parse an OpenAI-shaped response into a reading, whatever produced it.

    Accepts a decoded JSON dict (any HTTP client) or an SDK response object.
    """
    choice = _as_choice(payload)
    choices = getattr(choice, "choices", None)
    if not choices:
        raise ProbeReadError("probe response contains no choices")
    logprobs = _normalize_openai_logprobs(choices[0])
    if not logprobs:
        raise ProbeReadError("probe response contains no token logprobs")
    return _reading_from_labels(*_label_logprobs(logprobs, top_k=top_k))


def label_margin_from_payload(payload: Any) -> float | None:
    """The in-call label margin, read from a raw scoring response.

    The payload-level twin of :func:`label_margin_from_logprobs`, for the same
    reason :func:`probe_reading_from_payload` exists: the corpus-scale shard
    runner drives a bare ``httpx.Client`` and holds a decoded JSON dict, not a
    ``ModelResponse``. Without this it would have to reach into
    ``_normalize_openai_logprobs`` itself and re-derive a normalization whose
    whole purpose is to be derived once.

    Returns None — never raises — when the response carried no usable logprobs.
    The margin is an EXTRA measurement riding along on a call made for the
    verdict; losing it must never cost the verdict.
    """
    try:
        choice = _as_choice(payload)
        choices = getattr(choice, "choices", None)
        if not choices:
            return None
        return label_margin_from_logprobs(_normalize_openai_logprobs(choices[0]))
    except Exception:
        return None


def _reading_from_labels(log_p_correct: float, log_p_incorrect: float) -> ProbeReading:
    """The two label logprobs -> the reading. One definition of the arithmetic."""
    anchor = max(log_p_correct, log_p_incorrect)
    info = label_probability(
        [{"top": [
            {"token": "correct", "logprob": log_p_correct - anchor},
            {"token": "incorrect", "logprob": log_p_incorrect - anchor},
        ]}],
        position=0,
    )
    if info["status"] != "ok" or info["p_raw"] is None or not info["both_observed"]:
        raise ProbeReadError(
            f"probe label probability unavailable: {info['status']}"
        )
    p_raw = float(info["p_raw"])
    if not math.isfinite(p_raw):
        raise ProbeReadError("probe label probability is non-finite")
    return ProbeReading(p_raw=p_raw, delta_logit=log_p_correct - log_p_incorrect)


def read_probe(
    record: Mapping[str, object],
    client: Any,
) -> ProbeReading:
    """Return P(``correct``) and its label log-odds for one record.

    The client supplies the registered model id, timeout, and top-k cap through
    ``client.config``.  Both labels must be present in the returned window.  A
    missing label is a transport failure rather than evidence for probability
    zero or one, and raises :class:`ProbeTopKError`.
    """
    config = _configuration(client)
    backend = getattr(client, "backend", "openai_compat")
    if backend != "openai_compat":
        raise ProbeReadError(
            f"forced verdict probes require an openai_compat client, got {backend!r}"
        )

    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ProbeReadError("probe client config has no model_id")
    declared = config.get("max_top_logprobs")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 2:
        raise ProbeReadError(
            "probe client config must declare max_top_logprobs >= 2"
        )
    # Ask for the first-try width, bounded by what the route accepts; widen once
    # if a label falls outside. A narrow window DROPS the losing label and raises
    # ProbeTopKError rather than returning a biased half-pair, so the bound is a
    # correctness knob, not only a cost one.
    #
    # `declared` must match the server's actual launch flag (vLLM
    # `--max-logprobs`, MLX the serve_mlx.sh patch). If the registry claims more
    # than the server allows, BOTH the first try and the widened retry are
    # rejected and every row records an error — loud, but two wasted calls per
    # evidence.
    ceiling = min(declared, PROBE_TOP_LOGPROBS)
    top_k = min(declared, PROBE_FIRST_TRY_TOP_LOGPROBS)

    request: dict[str, Any] = build_probe_request(
        record, model_id=model_id, top_logprobs=top_k
    )
    timeout = config.get("timeout")
    if timeout is not None:
        request["timeout"] = timeout

    create = _completion_create(client)
    wall_timeout = getattr(client, "_invoke_with_wall_timeout", None)

    def _issue(width: int) -> ProbeReading:
        request["top_logprobs"] = width
        if callable(wall_timeout):
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise ProbeReadError(
                    "a ModelClient probe transport requires a numeric timeout"
                )
            response = wall_timeout(partial(create, **request), timeout)
        else:
            response = create(**request)
        return probe_reading_from_payload(response, top_k=width)

    try:
        return _issue(top_k)
    except ProbeTopKError:
        # The narrow first try missed a label. Widen to the route's full width
        # rather than lose the reading — one extra call on a rare record beats
        # paying 8x the payload on every record for a tail measured at rank 15.
        # If this fires often on a new stack, its rank distribution differs from
        # MLX's and PROBE_FIRST_TRY_TOP_LOGPROBS should be re-measured there.
        if ceiling <= top_k:
            raise
        global _WIDENED
        _WIDENED += 1
        return _issue(ceiling)


def label_margin_from_logprobs(logprobs: Any) -> float | None:
    """The verdict label's log-odds, read from a SCORING call's own logprobs.

    The probe issues a second request to put the label at generated position
    zero. A prompt whose output contract emits the verdict FIRST does not need
    that: the label is already in the response, and its margin can be read for
    free from the call we were making anyway.

    MEASURED, n=80 on MLX, against the second-call probe:

        in-call (verdict-only prompt)   AUROC 0.8734   within-verdict 0.7814
        probe (own prompt + prefill)    AUROC 0.7237   within-verdict 0.6856

    So the free read is also the better one. The probe's own 280-character prompt
    makes it a weaker reader; its readings are less saturated because it is less
    sure, not because they carry more.

    This scans for the first position whose emitted token is a label — which is
    why it belongs to a VARIANT that guarantees the verdict comes first. On a
    prompt that deliberates in its answer fields the label lands ~56 tokens deep
    and reads +22.50, i.e. saturated and useless; the scan would still find it
    and return a number that looks fine. Returns None when either label is
    outside the window, rather than a half-pair.
    """
    if not logprobs:
        return None
    for entry in logprobs:
        # Locate by the EMITTED token, never by "a label appears among the
        # alternatives". With a wide window an earlier position's alternative
        # list routinely contains "correct" — at 128 alternatives per token,
        # position 0 of `{"verdict": "correct"}` matched and the scan bailed
        # there with only one label in view, returning None on every row.
        if (entry or {}).get("token") not in LABELS:
            continue
        top = (entry or {}).get("top") or []
        seen = {t.get("token"): t.get("logprob") for t in top}
        if not all(label in seen for label in LABELS):
            return None
        try:
            correct = float(seen["correct"])
            incorrect = float(seen["incorrect"])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(correct) and math.isfinite(incorrect)):
            return None
        return correct - incorrect
    return None


__all__ = [
    "DIRECT_PROBE_ID",
    "PROBE_TOP_LOGPROBS",
    "PROBE_FIRST_TRY_TOP_LOGPROBS",
    "probe_widen_count",
    "label_margin_from_logprobs",
    "ProbeReadError",
    "ProbeReading",
    "ProbeTopKError",
    "read_probe",
]
