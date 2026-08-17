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
from collections.abc import Mapping
from functools import partial
from typing import Any, NamedTuple

from indra_belief.logprobs import label_probability
from indra_belief.model_client import _normalize_openai_logprobs
from indra_belief.probes.battery import LABELS, probe_by_id, render


DIRECT_PROBE_ID = "pol.verdict_direct"

# The window the probe asks for, and the number a probe-capable serving entry
# must be able to return. vLLM defaults to 20 and stock mlx_lm.server hard-codes
# 11, so BOTH need raising before the probe can read: vLLM with
# `--max-logprobs 1024` at launch, MLX with the patch in scripts/serve_mlx.sh.
# A registry entry declaring `max_top_logprobs` is asserting its server was
# started that way.
PROBE_TOP_LOGPROBS = 1024


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
    # Ask for PROBE_TOP_LOGPROBS, bounded by what the route accepts. MEASURED:
    # the losing label lands at rank 42/83/168, so a narrow window drops it and
    # raises ProbeTopKError rather than returning a biased half-pair. 1024 is the
    # default so a new serving entry does not have to rediscover the number; the
    # min() keeps us from serializing 4096 alternatives on a route that allows
    # them, which is pure transfer cost for ranks we will never read.
    top_k = min(declared, PROBE_TOP_LOGPROBS)

    probe = probe_by_id(DIRECT_PROBE_ID)
    system, user, prefill = render(probe, record)
    request: dict[str, Any] = {
        "model": model_id,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": top_k,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": prefill},
        ],
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "continue_final_message": True,
            "add_generation_prompt": False,
        },
    }
    timeout = config.get("timeout")
    if timeout is not None:
        request["timeout"] = timeout

    create = _completion_create(client)
    wall_timeout = getattr(client, "_invoke_with_wall_timeout", None)
    if callable(wall_timeout):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ProbeReadError(
                "a ModelClient probe transport requires a numeric timeout"
            )
        response = wall_timeout(partial(create, **request), timeout)
    else:
        response = create(**request)
    choices = getattr(response, "choices", None)
    if not choices:
        raise ProbeReadError("probe response contains no choices")
    logprobs = _normalize_openai_logprobs(choices[0])
    if not logprobs:
        raise ProbeReadError("probe response contains no token logprobs")

    log_p_correct, log_p_incorrect = _label_logprobs(logprobs, top_k=top_k)

    # label_probability is the canonical renormalizer.  Subtracting a common
    # anchor leaves its ratio unchanged while keeping exp(logprob) representable
    # even for an extreme (-800, -801) pair.  The exact log-odds stays in log
    # space below, as it does in the probe-battery runner.
    anchor = max(log_p_correct, log_p_incorrect)
    info = label_probability(
        [
            {
                "top": [
                    {"token": "correct", "logprob": log_p_correct - anchor},
                    {
                        "token": "incorrect",
                        "logprob": log_p_incorrect - anchor,
                    },
                ]
            }
        ],
        position=0,
    )
    if (
        info["status"] != "ok"
        or info["p_raw"] is None
        or not info["both_observed"]
    ):
        raise ProbeReadError(
            f"probe label probability unavailable: {info['status']}"
        )

    p_raw = float(info["p_raw"])
    if not math.isfinite(p_raw):
        raise ProbeReadError("probe label probability is non-finite")
    delta_logit = log_p_correct - log_p_incorrect
    return ProbeReading(p_raw=p_raw, delta_logit=delta_logit)


__all__ = [
    "DIRECT_PROBE_ID",
    "PROBE_TOP_LOGPROBS",
    "ProbeReadError",
    "ProbeReading",
    "ProbeTopKError",
    "read_probe",
]
