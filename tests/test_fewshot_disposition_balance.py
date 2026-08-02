"""Few-shot disposition balance: CORRECT examples must teach "surface the
objection THEN resolve to correct", not "any objection => reject".

The disconfirm renderer turns a CORRECT example's `considered` field into
objection=<considered>, verdict=correct. Without at least a couple of such
examples, every correct few-shot would show objection=null, training the model
to equate any surfaced objection with rejection — the exact conflation that the
removed output-side backstop encoded (see _prompts_disconfirm.py docstring).
"""
from __future__ import annotations

from indra_belief.scorers.monolithic._prompts import CONTRASTIVE_EXAMPLES
from indra_belief.scorers.monolithic import _prompts_disconfirm as disc
from indra_belief.verdict import parse_verdict


def _correct_with_considered():
    return [
        ex
        for ex in CONTRASTIVE_EXAMPLES
        if ex["verdict"] == "correct" and ex.get("considered")
    ]


def test_at_least_two_correct_examples_carry_considered():
    """>=2 CORRECT examples carry a non-null `considered` (apparent objection)."""
    with_considered = _correct_with_considered()
    assert len(with_considered) >= 2, (
        f"expected >=2 correct examples with a `considered` field, "
        f"found {len(with_considered)}"
    )
    for ex in with_considered:
        assert isinstance(ex["considered"], str) and ex["considered"].strip()


def test_considered_renders_as_objection_with_correct_verdict():
    """Rendering a `considered` CORRECT example through the disconfirm renderer
    yields objection != null AND verdict == correct (objection surfaced, resolved).

    Read back through `indra_belief.verdict` — the one parser the scorer uses on
    real replies — so the few-shot is checked to say to the model exactly what
    production would read back out of it."""
    with_considered = _correct_with_considered()
    assert len(with_considered) >= 2

    for ex in with_considered:
        _user, assistant = disc.render_example(ex)
        parsed = parse_verdict(assistant)
        assert parsed is not None, f"{ex['claim']}: few-shot answer did not parse"
        assert parsed.label == "correct", (
            f"{ex['claim']}: expected verdict=correct, got {parsed.label}"
        )
        assert parsed.objection is not None, (
            f"{ex['claim']}: expected non-null objection from `considered`"
        )
        # The objection the model sees IS the apparent objection we annotated.
        assert parsed.objection == ex["considered"]


def test_considered_examples_span_distinct_rule_classes():
    """Sanity: the annotated apparent-objections are not all the same class —
    the few-shot should span more than one rule the model can resolve."""
    considered = {ex["claim"]: ex["considered"] for ex in _correct_with_considered()}
    # At least one wrong-direction (inverse-inference) and one hypothesis-only.
    blob = " ".join(considered.values()).lower()
    assert "wrong-direction" in blob
    assert "hypothesis-only" in blob


def test_incorrect_examples_unchanged_no_considered():
    """Incorrect examples must not carry a `considered` field — only CORRECT
    examples teach the surface-then-resolve move."""
    for ex in CONTRASTIVE_EXAMPLES:
        if ex["verdict"] == "incorrect":
            assert "considered" not in ex, (
                f"incorrect example {ex['claim']} should not carry `considered`"
            )
