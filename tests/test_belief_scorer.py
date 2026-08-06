"""The INDRA seam: contract, cost visibility, and the refusal to invent a belief.

`LLMBeliefScorer` is the socket every INDRA pipeline already has — plug it into
`BeliefEngine(scorer=...)` and `readonly.belief` inherits our number. The tests
that matter here are not "does it return a float"; they are the three places this
scorer differs from every other `BeliefScorer` and could therefore surprise a
caller: it spends money per evidence, it can fail to read anything at all, and it
must stay importable in a core that ships without INDRA.

No provider is ever contacted. The client is a stub, so a regression that starts
issuing real calls fails loudly here rather than on someone's bill.
"""
from __future__ import annotations

import pytest

from indra_belief.belief_scorer import (
    HAVE_INDRA,
    LLMBeliefScorer,
    UnscorableStatement,
)


class _Evidence:
    def __init__(self, text, source_api="reach", source_hash=1):
        self.text = text
        self.source_api = source_api
        self._hash = source_hash

    def get_source_hash(self):
        return self._hash


class _Statement:
    def __init__(self, evidence):
        self.evidence = list(evidence)


class _ExplodingClient:
    """Any provider call is a test failure, not a cost."""

    def call(self, *a, **k):  # pragma: no cover - reaching this IS the failure
        raise AssertionError("the scorer contacted a provider during a unit test")


def test_it_is_the_indra_socket_when_indra_is_installed():
    """`isinstance(scorer, BeliefScorer)` must answer correctly, or the seam is cosmetic."""
    if not HAVE_INDRA:
        pytest.skip("indra not installed in this environment")
    from indra.belief import BeliefScorer

    assert issubclass(LLMBeliefScorer, BeliefScorer)
    for name in ("score_statements", "score_statement", "check_prior_probs"):
        assert callable(getattr(LLMBeliefScorer, name)), name


def test_cost_is_inspectable_before_it_is_spent():
    """One provider call per evidence, countable without issuing any.

    This scorer's siblings are microseconds of arithmetic over source counts. A
    caller who assumes the same and passes a corpus spends real money for
    minutes, so the count must be available up front — and it must be a FLOOR,
    since the default variant issues a second call for relation-nature claims.
    """
    stmts = [_Statement([_Evidence("a"), _Evidence("b")]), _Statement([_Evidence("c")])]
    assert LLMBeliefScorer.estimate_calls(stmts) == 3
    assert LLMBeliefScorer.estimate_calls([]) == 0


def test_a_statement_with_no_evidence_is_refused_before_any_call():
    """`check_prior_probs` is INDRA's pre-flight hook; use it to refuse, not to warn.

    There is no source-count fallback in this scorer — with no text there is
    nothing to read, and emitting a float would be the fabrication this codebase
    removed from the parser.
    """
    scorer = LLMBeliefScorer(_ExplodingClient())
    scorer.check_prior_probs([_Statement([_Evidence("has text")])])

    with pytest.raises(UnscorableStatement) as exc:
        scorer.check_prior_probs([_Statement([]), _Statement([_Evidence("x")])])
    assert "index 0" in str(exc.value)


def test_unreadable_statements_raise_rather_than_returning_a_number(monkeypatch):
    """THE CONTRACT THAT MATTERS. A missing belief reads back as 1.0.

    `Statement.from_json` defaults an absent belief to `1.0`, so a scorer that
    quietly omits or substitutes a value is asserting near-certainty about
    something it never read. `score_statements` must refuse; the detailed call
    must expose `None` instead.
    """
    import indra_belief.scorers.monolithic as mono

    monkeypatch.setattr(
        mono, "score_evidence",
        lambda *a, **k: {"verdict": None, "confidence": None, "tier": "no_text"},
    )
    scorer = LLMBeliefScorer(_ExplodingClient())
    stmts = [_Statement([_Evidence(None)])]

    rolled = scorer.score_statements_detailed(stmts)
    assert rolled[0].belief is None, "unread evidence must surface as None, not a float"

    with pytest.raises(UnscorableStatement) as exc:
        scorer.score_statements(stmts)
    assert "1.0" in str(exc.value), "the error must say WHY a placeholder is unsafe"


def test_a_read_statement_returns_one_float_per_statement_in_order(monkeypatch):
    """INDRA's contract: `Sequence[float]`, same length, same order."""
    import indra_belief.scorers.monolithic as mono

    seen: list[str] = []

    def fake(statement, evidence, client, **kw):
        seen.append(evidence.text)
        verdict = "correct" if evidence.text != "bad" else "incorrect"
        return {"verdict": verdict, "confidence": "high", "tier": "llm_comprehension"}

    monkeypatch.setattr(mono, "score_evidence", fake)
    scorer = LLMBeliefScorer(_ExplodingClient())
    stmts = [
        _Statement([_Evidence("good", source_hash=1)]),
        _Statement([_Evidence("bad", source_hash=2)]),
    ]

    out = scorer.score_statements(stmts)
    assert len(out) == len(stmts)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out), out
    assert seen == ["good", "bad"], "evidence must be read in statement order"
    assert out[0] > out[1], "a read rejection must not score above a read acceptance"


def test_the_detailed_call_keeps_the_tallies_the_scalar_throws_away(monkeypatch):
    """`score_statements` is lossy by INDRA's design; the audit trail must survive somewhere.

    One float cannot carry the verdict route or the twelve tally fields, and those
    are what make a served score defensible after the fact.
    """
    import indra_belief.scorers.monolithic as mono

    monkeypatch.setattr(
        mono, "score_evidence",
        lambda *a, **k: {"verdict": "correct", "confidence": "high",
                         "tier": "llm_comprehension"},
    )
    rolled = LLMBeliefScorer(_ExplodingClient()).score_statements_detailed(
        [_Statement([_Evidence("t", source_hash=7)])]
    )
    row = rolled[0]
    assert row.belief is not None
    assert row.verdict_statement in {"correct", "review", "incorrect"}
    assert row.n_evidence == 1 and row.n_correct == 1
