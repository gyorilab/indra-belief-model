"""The bare line-oriented reply form: `verdict correct`, no colon, no JSON.

From haohangyan's scale_up branch, commit "Use disconfirm_relnature and add more
rules to parser from output result" — patterns written against the parser as it
lived in `scorers/monolithic/_prompts.py`, before K2 made
`indra_belief.verdict` its single owner. The merge conflicted exactly there,
because our side had DELETED that parser. The patterns are carried here instead;
these tests are the evidence the branch did not carry.

WHY IT MATTERS RATHER THAN BEING A COSMETIC MISS. An unparsed reply is not
benign in this codebase: `grid_score` returns None off-grid, the runner turns
that into InvalidModelOutput, retries, and eventually writes an ERROR row. A
whole reply shape that never parses is holes and spend, and it is the shape a
LOCAL instruction model served over vLLM emits — one field per line, no
punctuation — where the Bedrock models this parser was tuned against emit JSON.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from indra_belief.verdict import parse_response


def _response(text: str):
    return SimpleNamespace(
        content=text, raw_text=text, reasoning="", tokens=0,
        prompt_tokens=0, finish_reason="stop", reasoning_trace={},
    )


@pytest.mark.parametrize("reply,verdict,confidence", [
    ("verdict correct\nconfidence high", "correct", "high"),
    ("verdict incorrect\nconfidence low", "incorrect", "low"),
    ("final verdict correct\nconfidence medium", "correct", "medium"),
    ("VERDICT correct\nCONFIDENCE low", "correct", "low"),
    ("decision incorrect\nconfidence high", "incorrect", "high"),
])
def test_the_bare_line_form_parses(reply, verdict, confidence):
    """Measured returning (None, None) before these patterns landed."""
    parsed = parse_response(_response(reply))
    assert parsed is not None, reply
    assert (parsed.label, parsed.confidence) == (verdict, confidence)


@pytest.mark.parametrize("reply,verdict,confidence", [
    ("Verdict: correct\nConfidence: high", "correct", "high"),
    ("verdict = incorrect\nconfidence = low", "incorrect", "low"),
    ('{"verdict":"incorrect","confidence":"medium"}', "incorrect", "medium"),
    ("The verdict is correct with high confidence", "correct", "high"),
])
def test_the_forms_that_already_worked_still_work(reply, verdict, confidence):
    """The new patterns are additive. A regression here is the real risk."""
    parsed = parse_response(_response(reply))
    assert parsed is not None, reply
    assert (parsed.label, parsed.confidence) == (verdict, confidence)


@pytest.mark.parametrize("reply", [
    "the verdict correctly identifies the mechanism described",
    "a correct verdict would require more evidence",
    "I cannot determine a verdict here.",
    "verdict pending further review",
])
def test_prose_does_not_produce_a_verdict(reply):
    """The anchor and the word boundary are what keep this narrow.

    `^` under MULTILINE means the keyword must START a line, so a mention inside
    a sentence cannot match; `\\b` after the alternation is what stops
    "correctly" from reading as "correct". Without either, this pattern would
    manufacture verdicts out of reasoning prose — which, on a path where an
    absent measurement is supposed to stay absent, is worse than not parsing.
    """
    parsed = parse_response(_response(reply))
    assert getattr(parsed, "label", None) is None, reply
