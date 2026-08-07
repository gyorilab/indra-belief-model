"""One verdict — the model's reply, read once, against one score grid.

An LLM reply commits to a `(verdict, confidence)` pair, and that pair maps to a
belief score on the six cells of `scorers._shared.VERDICT_SCORE_GRID`. Reading
it existed THREE times — the live baseline parser, the live structured
commit-first parser, and the batch parser — and the two score maps DISAGREED on
failure. The batch one returned `None`, and its caller retried. The live one
returned `0.5`: a value no model output can produce, sitting exactly where a
calibration curve is most sensitive, written into the row as though it were a
measurement. A second, quieter fabrication sat beside it — an on-grid verdict
with an OFF-GRID confidence fell through a `dict.get(..., 0.50)` default and was
scored 0.50 rather than failing.

This module is the single owner. One parser, one score map, and a reply that
cannot be read typed as ABSENCE on both paths:

    parse_verdict(text)      -> Verdict | None   one reply body
    parse_response(response) -> Verdict | None   `content`, then `raw_text`
    grid_score(v, c)         -> float  | None    a STORED pair, with no text

A `Verdict` exists only for a pair the grid can score, so `Verdict.score` is
always an on-grid value and absence has exactly one representation: `None`. No
caller may turn that back into a number. `comparison.runner` reads it as
`InvalidModelOutput` and retries, then records an ERROR row once the per-source
budget is spent; `scorers.monolithic.scorer` propagates `"score": None`.

`grid_score` is public because `comparison.replay.validate_row` re-derives the
score of a STORED `(verdict, confidence, score)` triple, where there is no text
to parse — the durable row is the only input.

WHAT THE UNION IS. The three parsers were not equivalent, and the reading below
is the LIVE one — but "superset" overstates it, so here is the measured shape.
The batch parser lost verdicts under truncation (`finish_reason="length"`) that
the live one recovered, and its nullish set omitted `"no support"`, so the two
sides disagreed on whether a model that wrote "no support" had committed a
support span. Both divergences resolve in the live parser's favour. It is NOT a
superset in every direction: the batch phrase pattern made the colon OPTIONAL,
so it committed on `**Verdict** correct` with no separator, which this module
refuses. That narrowing costs nothing measured — zero occurrences across the
228,812 stored responses in `data/comparison*` — and it is recorded rather than
reverted because a separator-free "verdict correct" is as likely to be a
fragment of the instruction as an answer. See `tests/test_verdict_parser.py::
test_the_unified_reading_is_narrower_on_exactly_one_axis`.

Home. Top level, beside `prepared_execution.py`. That module owns the REQUEST
one (Statement, Evidence) pair puts on the wire; this one owns the RESPONSE that
comes back. Both are consumed by `scorers.monolithic` and by `comparison.replay`
alike, so neither package may own either, and `scorers.*` must never import
`comparison.*`. It deliberately does NOT live in `scorers/_shared.py`:
`comparison/llm.py` sha256s that file's BYTES into the `implementation_digest`
published in four `data/comparison/models/*/manifest.json`, so the grid stays
where it is and this module imports it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from indra_belief.scorers._shared import VERDICT_SCORE_GRID

log = logging.getLogger(__name__)

VALID_VERDICTS = frozenset({"correct", "incorrect"})
VALID_CONFIDENCES = frozenset({"high", "medium", "low"})

# A reply that names a verdict but no confidence is read at "medium". That is
# deliberate, and different in kind from the 0.5 this module removes: "medium"
# resolves to a REAL grid cell (0.80 / 0.20) that a model output can produce,
# rather than to a value outside the grid entirely.
DEFAULT_CONFIDENCE = "medium"

# The LIVE nullish set. The batch copy omitted "no support".
_NULLISH_JUSTIFICATION = frozenset(
    {"", "none", "null", "n/a", "na", "no objection", "no support", "-"}
)

# (a) A brace-delimited object carrying a "verdict" key. Non-nested by
#     construction: `[^{}]*` cannot cross a brace, so a verdict beside a nested
#     object is left to the pattern reading below, exactly as before.
_VERDICT_JSON_SPAN = re.compile(r'\{[^{}]*"verdict"[^{}]*\}', re.DOTALL)

# (b) The strict JSON pair, in both field orders.
_JSON_PAIR = re.compile(
    r'\{[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"'
    r'[^{}]*?"confidence"\s*:\s*"(high|medium|low)"[^{}]*?\}',
    re.IGNORECASE,
)
_JSON_PAIR_REVERSED = re.compile(
    r'\{[^{}]*?"confidence"\s*:\s*"(high|medium|low)"'
    r'[^{}]*?"verdict"\s*:\s*"(correct|incorrect)"[^{}]*?\}',
    re.IGNORECASE,
)

# (c) Phrase-level reading of prose. Load-bearing under truncation, where the
#     JSON never closes but the decision is stated in the reasoning.
_VERDICT_PHRASES = (
    re.compile(r'"verdict"\s*:\s*"(correct|incorrect)"', re.IGNORECASE),
    re.compile(
        r'(?:final\s+)?(?:verdict|decision|conclusion)[^a-z]*?:[^a-z]*?'
        r'(?:["\'\*]*)(correct|incorrect)',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:verdict|decision|answer)\s+(?:is|should be|would be|=)\s*'
        r'[:"\'\*]*\s*(correct|incorrect)',
        re.IGNORECASE,
    ),
    # THE BARE LINE-ORIENTED FORM: `verdict correct`, no colon, no JSON.
    #
    # From haohangyan's scale_up branch, where it was written against
    # `_prompts.py`'s parser before this module became that parser's single
    # owner. Carried here rather than there, because there is nothing left in
    # `_prompts.py` to carry it — that is the whole point of the K2 unification.
    #
    # It is not redundant with the pattern above it. That one requires a COLON
    # (`verdict: correct`) or a linking verb (`verdict is correct`); a local
    # instruction model served over vLLM emits neither, writing one field per
    # line with a space. Measured on this parser before the pattern landed:
    # "verdict correct\nconfidence high" -> (None, None), and an unparsed reply
    # is not benign here — it becomes InvalidModelOutput, then a retry, then an
    # ERROR row. At the 60M-statement scale that branch runs, that is holes and
    # spend rather than a cosmetic miss.
    #
    # MULTILINE + a `^` anchor keep it narrow: it reads a line that BEGINS with
    # the keyword, so prose like "the verdict correctly identifies..." cannot
    # match it — `\b` after the alternation is what stops `correctly`.
    re.compile(
        r'^\s*(?:final\s+)?(?:verdict|decision|conclusion)\s*[=:]?\s*'
        r'["\'\*]*(correct|incorrect)\b',
        re.IGNORECASE | re.MULTILINE,
    ),
)
_CONFIDENCE_PHRASES = (
    re.compile(r'"confidence"\s*:\s*"(high|medium|low)"', re.IGNORECASE),
    re.compile(
        r'confidence[^a-z]*?:[^a-z]*?(?:["\'\*]*)(high|medium|low)', re.IGNORECASE
    ),
    re.compile(r'confidence\s+(?:is|level)?[^a-z]*?(high|medium|low)', re.IGNORECASE),
    re.compile(r'with\s+(high|medium|low)\s+confidence', re.IGNORECASE),
    # The confidence half of the bare line-oriented form above. Same origin,
    # same anchor, same reason. `confidence high` on its own line.
    re.compile(
        r'^\s*confidence\s*[=:]?\s*["\'\*]*(high|medium|low)\b',
        re.IGNORECASE | re.MULTILINE,
    ),
)


@dataclass(frozen=True)
class Verdict:
    """A reply the grid can score, and nothing else.

    There is no `Verdict` for an unreadable reply, for an unknown label, or for
    a confidence the grid has no cell for — `parse_verdict` returns `None`
    instead. So `score` is always one of the six committed values, and a caller
    never has to ask whether it is looking at a measurement or at a placeholder.

    `support` / `objection` are the model's own committed justification. Both
    are `None` unless the reply carried them as a structured object, and a
    nullish spelling ("none", "no support", "-", …) normalizes to `None` rather
    than travelling on as a string.
    """

    label: str
    confidence: str
    score: float
    support: str | None = None
    objection: str | None = None


class ResponseLike(Protocol):
    content: str
    raw_text: str


def grid_score(verdict: str | None, confidence: str | None) -> float | None:
    """Grid score for a `(verdict, confidence)` pair, or `None` if it is off-grid.

    `None` rather than the 0.5 midpoint the live path used to fabricate: 0.5 is
    not on the six-cell grid, so writing it for an answer the model did not give
    records an invented value as though it were a measurement.

    Off-grid on EITHER axis is `None`. An unknown label is, and so is an unknown
    confidence — the confidence selects the cell (0.95 / 0.80 / 0.65 or
    0.05 / 0.20 / 0.35) just as much as the label picks its sign, and a
    `{"verdict": "correct", "confidence": "certain"}` used to land on 0.50
    through a `dict.get` default.

    An ABSENT confidence still reads as "medium"; see DEFAULT_CONFIDENCE.
    """
    if verdict not in VALID_VERDICTS:
        return None
    return VERDICT_SCORE_GRID.get((verdict, confidence or DEFAULT_CONFIDENCE))


# Deliberate deviation from the node's original literal draft: `call_log` is
# per-call state, so it must not live in a module-level mapping where a shallow
# copy would share one list for the process lifetime. Each caller supplies a
# fresh list. `grid_score("correct", "high") == 0.95`, making this byte-neutral
# against the batch literal it replaces.
NO_TEXT_RESULT: Mapping[str, Any] = MappingProxyType({
    "score": grid_score("correct", "high"),
    "verdict": "correct",
    "confidence": "high",
    "tier": "no_text",
    "grounding_status": "skipped",
    "provenance_triggered": False,
    "raw_text": "No evidence sentence — accepted by default (database-sourced).",
    "tokens": 0,
})


def _normalize_justification(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _NULLISH_JUSTIFICATION else text


def _from_json_object(text: str) -> tuple[str, str | None, str | None, str | None] | None:
    """The LAST verdict-bearing object whose label is a real verdict.

    Last, not first: a `<think>` preamble weighing a hypothetical verdict must
    not outrank the answer, and the reply's own JSON comes after it.
    """
    for match in reversed(list(_VERDICT_JSON_SPAN.finditer(text))):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            log.debug("skipping unparseable verdict-like span: %r",
                      match.group(0), exc_info=True)
            continue
        label = obj.get("verdict")
        label = label.lower() if isinstance(label, str) else None
        if label not in VALID_VERDICTS:
            continue
        confidence = obj.get("confidence")
        return (
            label,
            confidence.lower() if isinstance(confidence, str) else None,
            _normalize_justification(obj.get("support")),
            _normalize_justification(obj.get("objection")),
        )
    return None


def _from_patterns(text: str) -> tuple[str, str] | None:
    """Strict JSON pair in either field order, then prose. No justification."""
    matches = _JSON_PAIR.findall(text)
    if matches:
        label, confidence = matches[-1]
        return label.lower(), confidence.lower()
    matches = _JSON_PAIR_REVERSED.findall(text)
    if matches:
        confidence, label = matches[-1]
        return label.lower(), confidence.lower()
    label = None
    for pattern in _VERDICT_PHRASES:
        found = pattern.findall(text)
        if found:
            label = found[-1].lower()
            break
    if label is None:
        return None
    confidence = DEFAULT_CONFIDENCE
    for pattern in _CONFIDENCE_PHRASES:
        found = pattern.findall(text)
        if found:
            confidence = found[-1].lower()
            break
    return label, confidence


def _candidate(text: str | None) -> tuple[str, str | None, str | None, str | None] | None:
    """What one reply body commits to, BEFORE the grid is consulted.

    An absent confidence and an off-grid one both survive this step. That is
    what keeps `parse_response`'s two-step fallback the shape it has always
    been: it stops at the first body that NAMES a verdict, and the grid is
    consulted once, afterwards. Gating here instead would let a reply whose
    answer carried an unknown confidence be overruled by its own reasoning.
    """
    if not text:
        return None
    found = _from_json_object(text)
    if found is not None:
        return found
    pair = _from_patterns(text)
    return None if pair is None else (pair[0], pair[1], None, None)


def _scored(candidate) -> Verdict | None:
    if candidate is None:
        return None
    label, confidence, support, objection = candidate
    score = grid_score(label, confidence)
    if score is None:
        return None
    return Verdict(label, confidence or DEFAULT_CONFIDENCE, score, support, objection)


def parse_verdict(text: str | None) -> Verdict | None:
    """Read one reply body. `None` when it commits to no scorable verdict.

    Order — the union the live path already implemented:
      (a) the last brace-delimited `{...}` carrying a real "verdict", parsed as
          JSON, which is also the only step that recovers support/objection;
      (b) the strict JSON `(verdict, confidence)` pair, in both field orders;
      (c) phrase patterns over prose, with confidence defaulting to "medium".
    """
    return _scored(_candidate(text))


def parse_response(response: ResponseLike) -> Verdict | None:
    """Read a model response: the final answer first, then the whole raw text.

    `content` is the final assistant message, which on separate-reasoning
    backends is CoT-free — so a hypothetical verdict weighed in the reasoning
    cannot outrank the answer. The fallback to `raw_text` (reasoning + content)
    is load-bearing rather than defensive: truncation at max_tokens produces a
    non-empty `content` that never reached the closing JSON, and the decision is
    recoverable from the reasoning that precedes it.
    """
    for text in (response.content, response.raw_text):
        candidate = _candidate(text)
        if candidate is not None:
            return _scored(candidate)
    return None
