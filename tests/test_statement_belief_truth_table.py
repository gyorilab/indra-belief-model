"""The aggregator's contract, stated independently of its implementation.

``statement_belief`` emits a scalar (``belief``) and a route
(``verdict_statement``) from one pass over the same rows. Nothing forced them to
agree, and they did not: a rejection with confidence ``low`` was credited into
the belief — driving it to 0.0 when it removed the only source — while being
invisible to the route, which still said ``correct``.

So the contract is written here as a full cross of the four fields that decide a
row's fate, with the expectation computed from the rules rather than read off the
production code. The headline is one line:

    verdict_statement == "correct" IMPLIES n_incorrect == 0

If a rejection is credible enough to move the number, it is credible enough to
move the route. If it is not credible, it must move neither.
"""
from __future__ import annotations

import itertools

from indra_belief.noise_model import RECALIBRATED_PRIORS
from indra_belief.statement_belief import statement_belief

VERDICTS = ("correct", "incorrect", None, "maybe")
CONFIDENCES = ("high", "medium", "low", None, "")
TIERS = (
    "llm_comprehension",
    "llm_tool_use",
    "no_text",
    "deterministic_mismatch",
    "deterministic_pseudogene",
    None,
)
SOURCES = ("reach", "signor", None, "")

_DETERMINISTIC = {"deterministic_mismatch", "deterministic_pseudogene"}
_UNCREDITED_TALLIES = ("n_no_text", "n_parse_fail", "n_null_source")


def _row(verdict, confidence, tier, source, *, text="evidence text"):
    return {
        "source_api": source,
        "verdict": verdict,
        "confidence": confidence,
        "tier": tier,
        "evidence_text": text,
    }


def _expected_bucket(verdict, tier, source) -> str | None:
    """Which uncredited tally a row lands in, or None when it is credited.

    Order matters and is part of the contract: an unread row is unread whatever
    else is wrong with it, a missing verdict outranks a missing source, and an
    unparseable verdict string is a parse failure.
    """
    if tier == "no_text":
        return "n_no_text"
    if verdict is None:
        return "n_parse_fail"
    if not source:
        return "n_null_source"
    if verdict not in {"correct", "incorrect"}:
        return "n_parse_fail"
    return None


def _expected_route(verdict, tier, credited: bool) -> str:
    if not credited:
        return "review"
    if verdict == "correct":
        return "correct"
    return "incorrect" if tier in _DETERMINISTIC else "review"


def test_single_row_truth_table_over_verdict_confidence_tier_source() -> None:
    """Every (verdict x confidence x tier x source) cell, against the contract."""
    violations: list[str] = []
    cells = 0
    for verdict, confidence, tier, source in itertools.product(
        VERDICTS, CONFIDENCES, TIERS, SOURCES
    ):
        cells += 1
        case = f"verdict={verdict!r} conf={confidence!r} tier={tier!r} src={source!r}"
        result = statement_belief(
            [_row(verdict, confidence, tier, source)], RECALIBRATED_PRIORS
        )
        bucket = _expected_bucket(verdict, tier, source)
        credited = bucket is None
        expected_route = _expected_route(verdict, tier, credited)

        def note(claim: str) -> None:
            violations.append(f"{case}: {claim}")

        # An uncredited row lands in exactly one tally and never reaches `gated`.
        for tally in _UNCREDITED_TALLIES:
            want = 1 if tally == bucket else 0
            if getattr(result, tally) != want:
                note(f"{tally}={getattr(result, tally)}, expected {want}")
        if result.n_distinct_sources != (1 if credited else 0):
            note(
                f"n_distinct_sources={result.n_distinct_sources}, "
                f"expected {1 if credited else 0}"
            )
        if result.n_correct != (1 if credited and verdict == "correct" else 0):
            note(f"n_correct={result.n_correct}")
        if result.n_incorrect != (1 if credited and verdict == "incorrect" else 0):
            note(f"n_incorrect={result.n_incorrect}")

        # belief is None iff nothing was credited.
        if (result.belief is None) != (not credited):
            note(f"belief={result.belief!r} but credited={credited}")

        if result.verdict_statement != expected_route:
            note(
                f"verdict_statement={result.verdict_statement!r}, "
                f"expected {expected_route!r}"
            )
        # THE INVARIANT.
        if result.verdict_statement == "correct" and result.n_incorrect != 0:
            note(
                f'routes "correct" with n_incorrect={result.n_incorrect} '
                f"and belief={result.belief!r}"
            )

        # Credible-rejection tallies: deterministic tier splits the two counters,
        # and any other credited rejection counts as an LLM rejection.
        rejected = credited and verdict == "incorrect"
        want_det = 1 if rejected and tier in _DETERMINISTIC else 0
        want_llm = 1 if rejected and tier not in _DETERMINISTIC else 0
        if result.n_credible_incorrect_det != want_det:
            note(
                f"n_credible_incorrect_det={result.n_credible_incorrect_det}, "
                f"expected {want_det}"
            )
        if result.n_credible_incorrect_llm != want_llm:
            note(
                f"n_credible_incorrect_llm={result.n_credible_incorrect_llm}, "
                f"expected {want_llm}"
            )

    assert cells == 480
    assert violations == [], f"{len(violations)} contract violations:\n" + "\n".join(
        violations[:40]
    )


def test_a_rejection_that_moves_the_belief_also_moves_the_route() -> None:
    """The defect, minimal: one low-confidence rejection, alone on a statement."""
    result = statement_belief(
        [_row("incorrect", "low", "llm_comprehension", "reach")], RECALIBRATED_PRIORS
    )
    # It IS credited — it is the reason the belief collapsed to zero.
    assert result.n_incorrect == 1
    assert result.belief == 0.0
    # So it cannot also be invisible to the route.
    assert result.verdict_statement == "review"
    assert result.n_credible_incorrect_llm == 1


def test_low_confidence_rejection_beside_four_confirmations() -> None:
    """A rejection among confirmations still routes to review, not correct."""
    rows = [
        _row("correct", "high", "llm_comprehension", "reach", text=f"confirm {i}")
        for i in range(4)
    ]
    rows.append(_row("incorrect", "low", "llm_comprehension", "signor", text="reject"))
    result = statement_belief(rows, RECALIBRATED_PRIORS)
    assert (result.n_correct, result.n_incorrect) == (4, 1)
    assert result.verdict_statement == "review"
    assert result.belief is not None and result.belief > 0.0


def test_two_sources_where_one_rejects() -> None:
    """A rejecting source is removed from the product; the route must say so."""
    result = statement_belief(
        [
            _row("correct", "high", "llm_comprehension", "reach", text="a"),
            _row("incorrect", "low", "llm_comprehension", "signor", text="b"),
        ],
        RECALIBRATED_PRIORS,
    )
    assert result.n_distinct_sources == 2
    assert result.verdict_statement == "review"
    assert result.n_credible_incorrect_llm == 1


def test_deterministic_rejection_hard_flags_incorrect() -> None:
    """The deterministic tier keeps its stronger route, at any confidence."""
    for confidence in CONFIDENCES:
        result = statement_belief(
            [_row("incorrect", confidence, "deterministic_mismatch", "reach")],
            RECALIBRATED_PRIORS,
        )
        assert result.verdict_statement == "incorrect", confidence
        assert result.n_credible_incorrect_det == 1, confidence
        assert result.n_credible_incorrect_llm == 0, confidence


def test_same_evidence_text_twice_within_one_source_dedups() -> None:
    """De-dup collapses the pair, and the surviving row keeps its route."""
    rows = [
        _row("incorrect", "low", "llm_comprehension", "reach", text="Same TEXT."),
        _row("incorrect", "low", "llm_comprehension", "reach", text="same text."),
    ]
    result = statement_belief(rows, RECALIBRATED_PRIORS)
    assert (result.n_evidence, result.n_dedup_groups) == (2, 1)
    assert result.n_incorrect == 1
    assert result.verdict_statement == "review"


def test_nothing_credited_leaves_belief_undefined_and_routes_review() -> None:
    """Absence of a measurement is never fabricated into support."""
    result = statement_belief(
        [
            _row("correct", "high", "no_text", "reach", text="a"),
            _row(None, "high", "llm_comprehension", "reach", text="b"),
            _row("correct", "high", "llm_comprehension", None, text="c"),
        ],
        RECALIBRATED_PRIORS,
    )
    assert result.belief is None and result.parametric_only is None
    assert result.verdict_statement == "review"
    assert (result.n_no_text, result.n_parse_fail, result.n_null_source) == (1, 1, 1)
