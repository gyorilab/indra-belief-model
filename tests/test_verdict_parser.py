"""The one categorical verdict parser — `indra_belief.verdict`.

Hermetic: no network, no model, no `data/` access. Every case is a literal reply
body, because that is the only input the parser has.

The parser unification replaced THREE parse implementations with one and made an
unreadable reply a typed absence on BOTH paths. Each case below records what the
three retired parsers did with the same input, because "unified" is a claim
about equality and the claim has to be checkable:

    live-structured   _prompts_disconfirm.parse_structured + derive_verdict
                      (the production profiles)
    live-baseline     _prompts.extract_verdict  (MONO_VARIANT="")
    batch             comparison/replay.parse_structured + parse_response

The population-scale replay harness no longer exists, leaving this test as the
whole of the claim: over all 228,812 stored LLM responses in
`data/comparison*` those three and this one agreed exactly, and under seeded
truncation the new parser matched live-structured on every mutant while the batch
parser lost six verdicts. The cases here include classes the corpus cannot show
because a parse failure erased its own evidence.
"""
from __future__ import annotations

import indra_belief.verdict as verdict_module
from indra_belief.verdict import (
    DEFAULT_CONFIDENCE,
    Verdict,
    parse_response,
    parse_verdict,
)

import pytest


class _Reply:
    """The two attributes `parse_response` reads off a model response."""

    def __init__(self, content: str, raw_text: str | None = None) -> None:
        self.content = content
        self.raw_text = content if raw_text is None else raw_text


def _answer(verdict, confidence, **extra) -> str:
    import json

    return json.dumps({**extra, "verdict": verdict, "confidence": confidence})


# --------------------------------------------------------------------------
# Closed-set categorical commitments
# --------------------------------------------------------------------------

@pytest.mark.parametrize("verdict,confidence", [
    ("correct", "high"),
    ("correct", "medium"),
    ("correct", "low"),
    ("incorrect", "low"),
    ("incorrect", "medium"),
    ("incorrect", "high"),
])
def test_all_closed_set_label_confidence_pairs(verdict, confidence):
    """The parser preserves every categorical pair the model may emit."""
    read = parse_verdict(_answer(verdict, confidence))
    assert read == Verdict(verdict, confidence, None, None)


# --------------------------------------------------------------------------
# Absence — an unreadable reply is a typed None, never a neutral verdict
# --------------------------------------------------------------------------

def test_an_absent_confidence_still_reads_as_medium():
    """All three retired parsers defaulted absent confidence to ``medium``."""
    read = parse_verdict('{"verdict": "correct"}')
    assert read is not None
    assert read.confidence == DEFAULT_CONFIDENCE


@pytest.mark.parametrize("text", [
    "",
    "   \n\t  ",
    None,
    "no verdict anywhere in this text",
    "The evidence is interesting but I am not going to say.",
])
def test_unreadable_input_is_none_never_a_neutral_verdict(text):
    """A Verdict exists only when the reply names a valid categorical pair."""
    assert parse_verdict(text) is None


def test_an_unknown_confidence_is_absence_not_a_downgrade():
    """Unknown labels and confidences are not categorical commitments."""
    assert parse_verdict('{"verdict": "correct", "confidence": "certain"}') is None
    assert parse_verdict('{"verdict":"medium","confidence":"medium"}') is None


# --------------------------------------------------------------------------
# Reading order: JSON object, strict pair, prose
# --------------------------------------------------------------------------

def test_both_json_field_orders():
    """`_JSON_VERDICT` / `_JSON_VERDICT_REV` in the retired live parser. The
    reversed order only ever mattered on the strict-pair branch — the JSON
    object branch is order-blind because it parses rather than pattern-matches."""
    forward = parse_verdict('{"verdict": "incorrect", "confidence": "low"}')
    reversed_ = parse_verdict('{"confidence": "low", "verdict": "incorrect"}')
    assert forward == reversed_
    assert forward is not None
    assert (forward.label, forward.confidence) == ("incorrect", "low")


def test_nested_brace_json_falls_through_to_the_pattern_read():
    """The object branch's regex cannot cross a brace, so a verdict sitting
    beside a nested object is read by the phrase patterns instead. Recorded
    because it is the reason `support`/`objection` are absent here: only the
    JSON-object branch recovers them. All three retired parsers behaved this
    way; the unified one keeps it rather than quietly widening the match."""
    text = ('{"trace": {"step": 1}, "support": "A binds B", '
            '"verdict": "correct", "confidence": "high"}')
    read = parse_verdict(text)
    assert read is not None
    assert (read.label, read.confidence) == ("correct", "high")
    assert read.support is None


@pytest.mark.parametrize("text,label,confidence", [
    ('{"verdict":"correct","confidence":"high",}', "correct", "high"),
    ('{"confidence":"low","verdict":"incorrect",}', "incorrect", "low"),
])
def test_a_trailing_comma_answer_is_read_by_the_strict_pair_step(
    text, label, confidence
):
    """Step (b) of the documented three-step order, which nothing exercised.

    Measured with `sys.settrace` before this test existed: across all 182
    parser-touching tests in the suite, the four lines that make up the strict
    `(verdict, confidence)` pair read never executed. Step (a) parses real JSON
    and step (c) reads prose, so a well-formed reply goes to (a) and a prose
    reply goes to (c) — and every fixture in this file was one or the other.

    The gap is exactly the class of reply step (b) exists for: JSON that is
    ALMOST valid. A trailing comma is a routine LLM malformation — `json.loads`
    refuses it, so (a) declines, and the pair regex reads it anyway. Both field
    orders are here because they are separate branches; neither had coverage.

    NOTE, and it is the point of the test below: executing those lines is NOT
    the same as depending on them. On these two inputs step (c) happens to
    recover the identical answer, so deleting step (b) entirely still passes
    here. Coverage is not a kill.
    """
    read = parse_verdict(text)
    assert read is not None, "the strict-pair step declined a recoverable answer"
    assert (read.label, read.confidence) == (label, confidence)


@pytest.mark.parametrize("text", [
    # verdict-then-confidence, and the reversed branch — separate regexes
    '{"verdict":"incorrect","confidence":"high",}\n'
    '{"summary":"weak evidence","confidence":"low",}',
    '{"confidence":"high","verdict":"incorrect",}\n'
    '{"summary":"weak evidence","confidence":"low",}',
])
def test_the_strict_pair_step_keeps_verdict_and_confidence_in_ONE_object(text):
    """WHY step (b) exists, which is the thing worth pinning.

    Step (c) reads the verdict and the confidence with two INDEPENDENT pattern
    sweeps, each taking its own last match — so it will happily pair a verdict
    from one object with a confidence from another. Step (b) matches both fields
    inside a single brace span, so the pair the model actually committed to
    survives.

    Here the answer object is malformed (trailing comma, so step (a) declines)
    and a later object carries a DIFFERENT confidence and no verdict:

        step (b), the pair from the answer object   -> ("incorrect", "high")
        step (c), cross-paired across both objects  -> ("incorrect", "low")

    Delete step (b) and the model's own high-confidence rejection is reported
    as a low-confidence one, from a confidence it never attached to that
    verdict.
    """
    read = parse_verdict(text)
    assert read is not None
    assert (read.label, read.confidence) == ("incorrect", "high"), (
        "the confidence was cross-paired from a different object — the strict "
        "(verdict, confidence) pair step is not doing its job"
    )


def test_a_malformed_span_is_skipped_not_abandoned():
    """The `except JSONDecodeError -> continue` branch, which nothing covered.

    Changing that `continue` to `return None` kept the suite green WHILE MOVING
    THE VERDICT — the quietest possible failure. `_from_json_object` walks the
    verdict-bearing spans in REVERSE, so the first one it meets here is the
    malformed trailing object; the contract is that an unparseable span is
    skipped and the walk continues to the well-formed one behind it.

    Under the mutant, step (a) abandons the whole body on the first bad span and
    the reply falls through to the pattern read, which takes the LAST strict pair
    — the malformed one — and answers `correct` where the model's committed
    object says `incorrect`.
    """
    text = ('{"verdict":"incorrect","confidence":"high"}\n'
            '{"verdict":"correct","confidence":"high",}')
    read = parse_verdict(text)
    assert read is not None
    assert (read.label, read.confidence) == ("incorrect", "high"), (
        "a malformed trailing span ended the JSON-object walk instead of being "
        "skipped — the well-formed answer behind it was never reached"
    )


@pytest.mark.parametrize("text,label", [
    ("Final verdict: incorrect", "incorrect"),
    ("the verdict is correct with high confidence", "correct"),
    ("Decision: **correct**", "correct"),
    ("Conclusion: incorrect.", "incorrect"),
])
def test_prose_only_replies(text, label):
    """Load-bearing under truncation, where the JSON never closes.

    The unified reading is the LIVE one. It is a superset of the retired batch
    patterns in the direction that cost verdicts — batch lost six replies under
    seeded truncation, and its nullish set omitted "no support" — but "strict
    subset" overstates it in one direction, and the correction is recorded rather
    than dropped because the direction of the widening is worth knowing. See
    `test_the_unified_reading_is_narrower_on_exactly_one_axis` below.
    """
    read = parse_verdict(text)
    assert read is not None and read.label == label


@pytest.mark.parametrize("text", [
    "**Verdict** correct",
    "the verdict correctly identifies the mechanism",
    "a correct verdict would need more evidence",
])
def test_the_unified_reading_is_narrower_on_exactly_one_axis(text):
    """The one place the unified parser reads LESS than a retired one.

    The retired batch phrase pattern made the colon OPTIONAL
    (`verdict[^a-z]*:?[^a-z]*`) and was unanchored, so it committed on
    `**Verdict** correct` and on the word "verdict" anywhere in a sentence. What
    survives of that narrowing is the ANCHOR: a line must BEGIN with the keyword.

    THE DECISION THIS TEST NAMED HAS NOW BEEN TAKEN. The previous version of
    this docstring said: "A separator-free 'verdict correct' is ... most likely
    to be a fragment of the INSTRUCTION rather than an answer ... If a future
    backend produces it, this test is where the decision to widen gets made."
    That backend arrived — a local instruction model served over vLLM on
    haohangyan's scale_up branch, which emits one field per line with no
    punctuation — so `verdict correct` and `Conclusion incorrect` moved OUT of
    this list and into `tests/test_verdict_bare_line_form.py`.

    Both original cautions were checked before widening rather than waived:

      * the instruction-fragment risk is ZERO against our prompts — all four
        `scorers.monolithic.VARIANTS` system prompts specify JSON output and
        contain no line beginning `verdict|decision|conclusion <label>`, so
        there is nothing for a model to echo into a false parse.
      * no published number moves — every stored response in `data/comparison*`
        was re-read with the widened and the pre-widening pattern sets and the
        two agree on every row.

    `**Verdict** correct` stays here because the line begins with `**`, not with
    the keyword, so the anchored pattern refuses it exactly as before. The other
    two are the mid-sentence forms the anchor and the `\\b` after the label are
    there to refuse.
    """
    assert parse_verdict(text) is None


def test_prose_confidence_is_read_when_stated():
    read = parse_verdict("the verdict is correct with high confidence")
    assert read is not None and (read.label, read.confidence) == ("correct", "high")
    read = parse_verdict("Final verdict: incorrect")
    assert read is not None
    assert (read.label, read.confidence) == ("incorrect", "medium")


def test_markdown_fenced_answer():
    read = parse_verdict('```json\n{"verdict": "correct", "confidence": "low"}\n```')
    assert read is not None
    assert (read.label, read.confidence) == ("correct", "low")


# --------------------------------------------------------------------------
# Last match wins — the reasoning must not outrank the answer
# --------------------------------------------------------------------------

def test_a_hypothetical_verdict_in_reasoning_loses_to_the_answer():
    """A `<think>` preamble weighing the opposite verdict. Last-match ordering
    is why the answer wins, and it is why `parse_response` reads `content`
    before `raw_text`: on separate-reasoning backends `content` is CoT-free, so
    the preamble is not even in scope."""
    text = ('<think>If this were about mRNA levels the verdict would be '
            '{"verdict": "incorrect", "confidence": "high"} — but it is not.'
            '</think>\n{"verdict": "correct", "confidence": "high"}')
    read = parse_verdict(text)
    assert read is not None and read.label == "correct"


def test_the_last_verdict_phrase_wins_not_the_first():
    """`_from_patterns`' `found[-1]`, which nothing covered until now.

    Flipping it to `found[0]` left the whole file passing, because every prose
    case above states its verdict ONCE — a single-verdict string reads the same
    from either end. So the rule that decides which verdict a reply committed to
    was, in effect, unasserted.

    That cost more than a coverage gap. While mutants were in flight in this same
    file, the resting value of this line could not be told apart from a seeded
    one BY READING THE TREE, and the question "which of these is the original?"
    had to be settled against a fixed reference instead — `git show HEAD:` of the
    two retired parsers, both of which take the last match
    (`_prompts.py` `m[-1]`; `comparison/replay.py` `matches[-1]`). An untested
    invariant does not merely fail to catch a regression; it removes your ability
    to identify the correct code afterwards. `found[-1]` is correct. Pinned here.
    """
    read = parse_verdict(
        "Initial verdict: correct. On reflection, final verdict: incorrect"
    )
    assert read is not None
    assert (read.label, read.confidence) == ("incorrect", "medium"), (
        "the FIRST verdict phrase won — a reply that changes its mind would be "
        "reported at the position it abandoned"
    )
    # The same rule, one level up: the last CONFIDENCE phrase wins too.
    read = parse_verdict("Verdict: incorrect. Confidence: low. Actually confidence: high")
    assert read is not None and (read.label, read.confidence) == ("incorrect", "high")


def test_a_verdictless_object_does_not_shadow_a_later_real_one():
    text = ('{"verdict": null, "confidence": null}\n'
            '{"support": "A binds B", "verdict": "correct", "confidence": "medium"}')
    read = parse_verdict(text)
    assert read is not None
    assert (read.label, read.confidence) == ("correct", "medium")
    assert read.support == "A binds B"


# --------------------------------------------------------------------------
# support / objection — including the live-vs-batch divergence removed here
# --------------------------------------------------------------------------

def test_support_and_objection_travel_with_the_verdict():
    read = parse_verdict(_answer("correct", "high", support="A binds B directly",
                                 objection="looks like wrong direction"))
    assert read is not None
    assert read.support == "A binds B directly"
    assert read.objection == "looks like wrong direction"


@pytest.mark.parametrize("spelling", ["none", "None", "null", "n/a", "NA", "-", "  "])
def test_nullish_justification_normalizes_to_none(spelling):
    read = parse_verdict(_answer("correct", "high", support=spelling,
                                 objection=spelling))
    assert read is not None
    assert read.support is None and read.objection is None


def test_no_support_is_nullish_the_live_reading_wins():
    """THE REMOVED DIVERGENCE. The retired live nullish set contained
    "no support"; the batch copy did not. So a model that wrote
    `"support": "no support"` had committed no support span on the live path and
    a support span reading "no support" on the batch path — the same reply,
    two provenance records. The live reading is the correct one and is what the
    unified parser keeps."""
    read = parse_verdict(_answer("incorrect", "high", support="no support",
                                 objection="no objection"))
    assert read is not None
    assert read.support is None
    assert read.objection is None


def test_a_justification_without_a_scorable_verdict_is_not_a_commitment():
    """A behaviour change, stated rather than buried. The retired live parser
    kept `support` off an object whose verdict was garbage, so a row could carry
    a committed justification for a verdict that was never committed. There is
    no Verdict here, so there is nothing to attribute the span to."""
    assert parse_verdict(_answer("maybe", "high", support="A binds B")) is None


# --------------------------------------------------------------------------
# parse_response — content first, then raw_text
# --------------------------------------------------------------------------

def test_content_is_read_before_raw_text():
    """Kept for the shape it documents, but it CANNOT FAIL on its own.

    Its decoy sits FIRST in `raw_text`, and last-match ordering then makes
    `raw_text` yield the same answer as `content`, so swapping the two bodies
    changes nothing here. The test below is the one that discriminates.
    """
    reply = _Reply(content='{"verdict": "correct", "confidence": "high"}',
                   raw_text='{"verdict": "incorrect", "confidence": "high"}\n'
                            '{"verdict": "correct", "confidence": "high"}')
    read = parse_response(reply)
    assert read is not None and read.label == "correct"


def test_a_truncated_answer_outranks_a_complete_hypothetical_in_the_reasoning():
    """The ordering contract, with a case that can actually break.

    The test above cannot fail. Its `raw_text` ENDS with the same verdict object
    `content` carries, so last-match ordering returns the same answer from either
    body and swapping the two identifiers in `parse_response` changes nothing.
    That is not an accident of the fixture — it is the normal shape, because
    every transport in `model_client` builds `raw_text` as
    `reasoning + "\\n" + content`, so raw_text usually ends with content.

    The case that separates them is truncation, which is the reason the fallback
    exists at all. Here `content` is the answer cut off at max_tokens before its
    closing brace, so no JSON OBJECT survives in it and the verdict is recovered
    by the phrase patterns; `raw_text` is that same truncated answer with the
    reasoning in front of it, and the reasoning weighed a COMPLETE hypothetical
    object for the opposite verdict. So raw_text's last parseable verdict object
    is the hypothetical, and the two bodies disagree:

        content first   -> ("incorrect", "high")  the model's answer
        raw_text first  -> ("correct",   "low")   what it argued against

    `parse_response`'s docstring asserts content-first; this asserts it with a
    fixture where the two bodies genuinely disagree.
    """
    truncated_answer = ('{"support": "A binds B directly", '
                        '"verdict": "incorrect", "confidence": "high"')
    reasoning = ('<think>If the claim were about transcript levels this would be '
                 '{"verdict": "correct", "confidence": "low"} — but it is about '
                 'protein binding.</think>')
    reply = _Reply(content=truncated_answer,
                   raw_text=reasoning + "\n" + truncated_answer)

    read = parse_response(reply)
    assert read is not None
    assert (read.label, read.confidence) == ("incorrect", "high"), (
        "parse_response returned the reasoning's hypothetical instead of the "
        "answer — the content-before-raw_text order is inverted"
    )
    # Both bodies are individually readable and they DISAGREE; without that, the
    # assertion above would hold under either order and prove nothing.
    assert parse_verdict(truncated_answer) == read
    from_raw = parse_verdict(reply.raw_text)
    assert from_raw is not None
    assert (from_raw.label, from_raw.confidence) == ("correct", "low")


def test_an_invalid_answer_is_absence_not_a_fallback_to_the_other_body():
    """Validation happens after the two-body search — `_candidate`'s invariant.

    `parse_response` stops at the first body that NAMES a verdict, and only then
    validates its categorical confidence. Moving validation inside the loop left
    the file passing while changing what a reply means: a model that answered
    `correct` with an unknown confidence would be overruled by whatever its
    other body said, and the row would carry a verdict the answer never gave.

    Absence is the right outcome. Reaching past an invalid final answer for a
    second opinion changes the categorical commitment rather than recovering it.

    The fixture is constructed rather than drawn from a run: the two bodies have
    to genuinely disagree for the invariant to be observable at all, and most
    transports build `raw_text` as `reasoning + content`, so they usually agree.
    What is pinned is the shape, not a production scenario.
    """
    reply = _Reply(content='{"verdict": "correct", "confidence": "certain"}',
                   raw_text='{"verdict": "incorrect", "confidence": "high"}')
    assert parse_response(reply) is None, (
        "an invalid answer fell through to the other body — categorical "
        "validation is happening per-body instead of after the search"
    )
    # Both halves of the setup are load-bearing: the other body IS valid, so
    # the assertion above is about ordering rather than about both being absent.
    assert parse_verdict(reply.raw_text) is not None
    assert parse_verdict(reply.content) is None


def test_empty_content_falls_back_to_raw_text():
    """The truncation recovery path: `content` never reached the answer, but the
    decision is stated in the reasoning that precedes it."""
    reply = _Reply(content="", raw_text="Final verdict: incorrect")
    read = parse_response(reply)
    assert read is not None and read.label == "incorrect"


def test_content_without_a_verdict_falls_back_to_raw_text():
    reply = _Reply(content="Let me think about this claim carefully.",
                   raw_text='reasoning...\n{"verdict": "correct", "confidence": "low"}')
    read = parse_response(reply)
    assert read is not None
    assert (read.label, read.confidence) == ("correct", "low")


def test_parse_response_returns_none_when_neither_body_commits():
    assert parse_response(_Reply(content="", raw_text="")) is None
    assert parse_response(_Reply(content="thinking", raw_text="still thinking")) is None


# --------------------------------------------------------------------------
# The type itself
# --------------------------------------------------------------------------

def test_a_verdict_is_frozen_and_has_no_score_attribute():
    read = parse_verdict('{"verdict": "correct", "confidence": "high"}')
    assert read is not None
    assert not hasattr(read, "score")
    assert "score" not in Verdict.__dataclass_fields__
    assert not hasattr(verdict_module, "grid_score")
    with pytest.raises(AttributeError):
        read.label = "incorrect"


def test_every_parseable_input_yields_only_closed_set_categories():
    """Every Verdict contains a valid label/confidence pair and no probability."""
    bodies = [
        '{"verdict": "correct", "confidence": "high"}',
        '{"confidence": "low", "verdict": "incorrect"}',
        '{"verdict": "correct"}',
        "Final verdict: incorrect",
        "the verdict is correct with high confidence",
        '{"verdict": "correct", "confidence": "certain"}',
        '{"verdict": "medium", "confidence": "medium"}',
        "",
        "nothing here",
    ]
    for body in bodies:
        read = parse_verdict(body)
        if read is not None:
            assert read.label in {"correct", "incorrect"}, body
            assert read.confidence in {"high", "medium", "low"}, body
            assert not hasattr(read, "score"), body
