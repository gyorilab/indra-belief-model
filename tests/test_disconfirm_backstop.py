"""Disconfirm verdict contract: the model's committed verdict is final.

Regression guard for the output-side-determinism bug. A removed code "decision
backstop" flipped 'correct' -> 'incorrect' whenever the parsed `objection` field
was non-null, which threw away correct verdicts where the model surfaced an
apparent objection and then resolved it (e.g. miRNA inverse-inference). The
disconfirm disposition lives entirely in the prompt; code does not re-derive the
verdict from support/objection.

K2-one-parser removed the `parse_structured` / `derive_verdict` pair these cases
used to drive: the reading of the answer and the pass-through of the committed
verdict are now one step, in `indra_belief.verdict`, shared by every profile and
by the batch replay. Each case below is the SAME case, driven through the real
parser from the model's own text rather than from a hand-built parse dict — so
the guard now also covers the reading that produces the dict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief.scorers.monolithic._prompts_disconfirm import render_example
from indra_belief.verdict import parse_verdict


def _answer(**fields) -> str:
    """One model answer in the variant's four-field output contract."""
    return json.dumps({"support": None, "objection": None, **fields})


# --- the committed verdict passes through unchanged ------------------------

def test_correct_with_resolved_objection_stays_correct():
    """The bug case: a surfaced-but-resolved objection must NOT flip the verdict."""
    read = parse_verdict(_answer(
        support="inhibition of miR-410 increased the expression of AGTR1",
        objection="Surface reading looks like wrong direction, but inverse "
                  "inference means the miRNA decreases the target — resolved.",
        verdict="correct", confidence="high",
    ))
    assert read is not None
    assert read.label == "correct"
    assert read.confidence == "high"
    assert not hasattr(read, "score")
    assert read.objection  # carried through as telemetry, not as an input


def test_incorrect_stays_incorrect():
    read = parse_verdict(_answer(objection="amount not activity",
                                 verdict="incorrect", confidence="high"))
    assert read is not None and read.label == "incorrect"
    assert read.confidence == "high"


def test_correct_without_support_is_not_overridden():
    """no_support_skeptic was output-side determinism too; verdict is final.
    The skepticism prior lives in the prompt, not in a post-hoc override."""
    read = parse_verdict(_answer(verdict="correct", confidence="medium"))
    assert read is not None and read.label == "correct"
    assert read.support is None


def test_objection_present_does_not_force_incorrect():
    """No combination of non-null objection flips a committed 'correct'."""
    read = parse_verdict(_answer(support="X binds Y",
                                 objection="considered family ambiguity",
                                 verdict="correct", confidence="high"))
    assert read is not None and read.label == "correct"


def test_no_committed_verdict_is_absence_not_a_verdict():
    """What `derive_verdict(...) == (None, None, "parse_null")` used to say.

    An answer that commits to nothing yields no Verdict at all — not a
    fabricated one, and not a score. K2-one-parser: absence stays absent.
    """
    assert parse_verdict(_answer(support="x", verdict=None, confidence=None)) is None


# --- reading the answer, end to end ---------------------------------------

def test_parse_on_mirna_inverse_output():
    """The exact shape bedrock-gemma emitted on MIR410: objection holds the
    resolution, verdict=correct. End to end must yield 'correct'."""
    raw = ('```json\n{"support": "In contrast, inhibition of miR-410 increased '
           'the expression of AGTR1.", "objection": "Inhibiting the miRNA raises '
           'the target, which means the miRNA decreases it. This supports the '
           'claim.", "verdict": "correct", "confidence": "high"}\n```')
    read = parse_verdict(raw)
    assert read is not None
    assert read.label == "correct"
    assert read.objection  # non-null, but must not matter


def test_nullish_objection_normalized():
    read = parse_verdict('{"support": "x", "objection": "none", '
                         '"verdict": "correct", "confidence": "low"}')
    assert read is not None
    assert read.objection is None
    assert read.label == "correct"


# --- render_example: few-shots must not teach objection => incorrect -------

def test_render_correct_with_considered_surfaces_objection():
    ex = {"claim": "MIR1 [DecreaseAmount] T", "evidence": "inhibiting miR-1 raised T",
          "verdict": "correct", "confidence": "high",
          "considered": "looks like wrong direction"}
    _user, assistant = render_example(ex)
    obj = json.loads(assistant)
    assert obj["verdict"] == "correct"
    assert obj["objection"] == "looks like wrong direction"  # surfaced, still correct


def test_render_correct_without_considered_has_null_objection():
    ex = {"claim": "A [Phosphorylation] B", "evidence": "A phosphorylates B",
          "verdict": "correct", "confidence": "high"}
    obj = json.loads(render_example(ex)[1])
    assert obj["verdict"] == "correct" and obj["objection"] is None


def test_render_incorrect_has_objection_and_null_support():
    ex = {"claim": "A [Activation] B", "evidence": "A increases B mRNA",
          "verdict": "incorrect", "confidence": "high", "reason": "amount not activity"}
    obj = json.loads(render_example(ex)[1])
    assert obj["verdict"] == "incorrect" and obj["support"] is None
    assert obj["objection"] == "amount not activity"
