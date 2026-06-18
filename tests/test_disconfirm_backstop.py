"""Disconfirm verdict contract: the model's committed verdict is final.

Regression guard for the output-side-determinism bug. The old `derive_verdict`
flipped 'correct' -> 'incorrect' whenever the parsed `objection` field was
non-null, which threw away correct verdicts where the model surfaced an apparent
objection and then resolved it (e.g. miRNA inverse-inference). The disconfirm
disposition now lives entirely in the prompt; code does not re-derive the verdict
from support/objection.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief.scorers.monolithic._prompts_disconfirm import (
    derive_verdict,
    parse_structured,
    render_example,
)


# --- derive_verdict: verdict passes through unchanged ----------------------

def test_correct_with_resolved_objection_stays_correct():
    """The bug case: a surfaced-but-resolved objection must NOT flip the verdict."""
    parsed = {
        "support": "inhibition of miR-410 increased the expression of AGTR1",
        "objection": "Surface reading looks like wrong direction, but inverse "
                     "inference means the miRNA decreases the target — resolved.",
        "verdict": "correct",
        "confidence": "high",
    }
    v, c, rule = derive_verdict(parsed)
    assert v == "correct"
    assert c == "high"
    assert rule == "model"


def test_incorrect_stays_incorrect():
    parsed = {"support": None, "objection": "amount not activity",
              "verdict": "incorrect", "confidence": "high"}
    v, c, rule = derive_verdict(parsed)
    assert v == "incorrect"
    assert rule == "model"


def test_correct_without_support_is_not_overridden():
    """no_support_skeptic was output-side determinism too; verdict is final.
    The skepticism prior lives in the prompt, not in a post-hoc override."""
    parsed = {"support": None, "objection": None,
              "verdict": "correct", "confidence": "medium"}
    v, _c, rule = derive_verdict(parsed)
    assert v == "correct"
    assert rule == "model"


def test_objection_present_does_not_force_incorrect():
    """No combination of non-null objection flips a committed 'correct'."""
    parsed = {"support": "X binds Y", "objection": "considered family ambiguity",
              "verdict": "correct", "confidence": "high"}
    assert derive_verdict(parsed)[0] == "correct"


def test_none_verdict_parses_null():
    assert derive_verdict({"verdict": None}) == (None, None, "parse_null")


# --- parse_structured + derive end to end ---------------------------------

def test_parse_then_derive_on_mirna_inverse_output():
    """The exact shape bedrock-gemma emitted on MIR410: objection holds the
    resolution, verdict=correct. End to end must yield 'correct'."""
    raw = ('```json\n{"support": "In contrast, inhibition of miR-410 increased '
           'the expression of AGTR1.", "objection": "Inhibiting the miRNA raises '
           'the target, which means the miRNA decreases it. This supports the '
           'claim.", "verdict": "correct", "confidence": "high"}\n```')
    parsed = parse_structured(raw)
    assert parsed["verdict"] == "correct"
    assert parsed["objection"]  # non-null, but must not matter
    assert derive_verdict(parsed)[0] == "correct"


def test_parse_structured_nullish_objection_normalized():
    parsed = parse_structured('{"support": "x", "objection": "none", '
                              '"verdict": "correct", "confidence": "low"}')
    assert parsed["objection"] is None
    assert parsed["verdict"] == "correct"


# --- render_example: few-shots must not teach objection => incorrect -------

def test_render_correct_with_considered_surfaces_objection():
    ex = {"claim": "MIR1 [DecreaseAmount] T", "evidence": "inhibiting miR-1 raised T",
          "verdict": "correct", "confidence": "high",
          "considered": "looks like wrong direction"}
    _user, assistant = render_example(ex)
    import json
    obj = json.loads(assistant)
    assert obj["verdict"] == "correct"
    assert obj["objection"] == "looks like wrong direction"  # surfaced, still correct


def test_render_correct_without_considered_has_null_objection():
    ex = {"claim": "A [Phosphorylation] B", "evidence": "A phosphorylates B",
          "verdict": "correct", "confidence": "high"}
    import json
    obj = json.loads(render_example(ex)[1])
    assert obj["verdict"] == "correct" and obj["objection"] is None


def test_render_incorrect_has_objection_and_null_support():
    ex = {"claim": "A [Activation] B", "evidence": "A increases B mRNA",
          "verdict": "incorrect", "confidence": "high", "reason": "amount not activity"}
    import json
    obj = json.loads(render_example(ex)[1])
    assert obj["verdict"] == "incorrect" and obj["support"] is None
    assert obj["objection"] == "amount not activity"
