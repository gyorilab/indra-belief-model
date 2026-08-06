"""The seam: this scorer as an `indra.belief.BeliefScorer`.

INDRA already defines the socket a belief scorer plugs into —
`BeliefScorer.score_statements(statements, extra_evidence) -> Sequence[float]` —
and the lab's own pipeline drives it to fill `readonly.belief`, a
`{matches_hash: float}` table that `sort_by=belief`, `minimum_belief` and seven
denormalised meta columns already read. Implementing that interface is what makes
this drop-in: `BeliefEngine(scorer=LLMBeliefScorer(client))` and every existing
INDRA pipeline inherits our number with no new client.

WHAT THIS COSTS, STATED UP FRONT. INDRA's other scorers are microseconds of
arithmetic over source counts — `SimpleScorer`, `CountsScorer` and `HybridScorer`
never read evidence text. This one does, so it issues ONE PROVIDER CALL PER
EVIDENCE, and on the paper corpus a statement carries 19.75 evidences on average
and up to 759. A caller who mistakes this for its siblings will hand a list of
statements to something that spends money for minutes. `estimate_calls()` exists
so that cost is inspectable BEFORE it is incurred, and `check_prior_probs()`
refuses a workload the scorer cannot honestly serve.

WHY THE RETURN TYPE IS LOSSY, DELIBERATELY. `StatementBelief` carries a verdict
route and twelve tally fields; `score_statements` returns one float. That is
INDRA's contract and it is the right one for the socket — but the tallies are
what make a score auditable, so `score_statements_detailed()` returns them for
callers who want more than the scalar.

ABSENCE IS NOT 1.0. `Statement.from_json` silently defaults a MISSING belief to
`1.0`, so dropping the field says "certainly true" rather than "unscored". When
nothing could be read this returns `None` from the detailed call and REFUSES to
invent a float in `score_statements` — see `unscored`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from indra_belief.statement_belief import StatementBelief, statement_belief

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indra_belief.model_client import ModelClient

__all__ = ["LLMBeliefScorer", "UnscorableStatement", "HAVE_INDRA"]


try:  # The deployable core installs neither `indra` nor `gilda`, so this must
    # stay importable without them. When INDRA IS present we subclass its real
    # BeliefScorer, so `isinstance(scorer, BeliefScorer)` answers correctly.
    from indra.belief import BeliefScorer as _BeliefScorerBase

    HAVE_INDRA = True
except Exception:  # pragma: no cover - exercised only where indra is absent
    class _BeliefScorerBase:  # type: ignore[no-redef]
        """Stand-in so the core can be imported and served without INDRA."""

    HAVE_INDRA = False


class UnscorableStatement(ValueError):
    """Nothing in a statement's evidence could be read into a verdict.

    Raised rather than returning a float, because every float in [0,1] means
    something specific to a downstream consumer and none of them means "we did
    not find out". Callers who prefer a gap should use
    `score_statements_detailed`, whose belief is `float | None`.
    """


class LLMBeliefScorer(_BeliefScorerBase):
    """Score INDRA statements by READING their evidence, not counting sources.

    Subclasses `indra.belief.BeliefScorer` when INDRA is importable, and stands
    alone when it is not — the core ships without `indra` installed (the batch
    image omits it deliberately), so importing this module must not require it.
    """

    def __init__(self, client: "ModelClient", *, priors: dict | None = None,
                 soft: dict | None = None, max_tokens: int | None = None,
                 variant: Any = None, dedup: bool = True) -> None:
        self.client = client
        self.priors = priors
        self.soft = soft
        self.max_tokens = max_tokens
        self.variant = variant
        self.dedup = dedup

    # ---- cost, before it is spent -------------------------------------------

    @staticmethod
    def estimate_calls(statements: Sequence[Any]) -> int:
        """Provider calls this workload will issue, before issuing any.

        One per evidence. Relation-nature statements issue a second call on the
        default variant, so this is a FLOOR rather than an estimate — it is here
        to make an accidental five-figure workload visible, not to be precise.
        """
        return sum(len(getattr(s, "evidence", ()) or ()) for s in statements)

    def check_prior_probs(self, statements: Sequence[Any]) -> None:
        """INDRA's pre-flight hook: refuse what cannot be served, before spending.

        A statement with no evidence has nothing to read, and returning a float
        for it would be fabrication of exactly the kind this codebase removed
        from the parser.
        """
        empty = [i for i, s in enumerate(statements)
                 if not (getattr(s, "evidence", None) or ())]
        if empty:
            raise UnscorableStatement(
                f"{len(empty)} statement(s) carry no evidence and cannot be read; "
                f"first at index {empty[0]}. This scorer reads evidence text — it "
                f"has no source-count fallback."
            )

    # ---- the interface -------------------------------------------------------

    def score_statements_detailed(
        self, statements: Sequence[Any],
        extra_evidence: list[list[Any]] | None = None,
    ) -> list[StatementBelief]:
        """The full roll-up, tallies included. `belief` is `float | None`."""
        from indra_belief.scorers.monolithic import score_evidence

        out: list[StatementBelief] = []
        for index, statement in enumerate(statements):
            evidences = list(getattr(statement, "evidence", None) or ())
            if extra_evidence:
                evidences.extend(extra_evidence[index] or ())

            rows: list[dict] = []
            for evidence in evidences:
                result = score_evidence(statement, evidence, self.client,
                                        variant=self.variant)
                rows.append({
                    "source_api": getattr(evidence, "source_api", None),
                    "evidence_text": getattr(evidence, "text", None),
                    "evidence_hash": getattr(evidence, "get_source_hash", lambda: None)(),
                    "verdict": result.get("verdict"),
                    "confidence": result.get("confidence"),
                    "tier": result.get("tier"),
                })
            out.append(statement_belief(rows, self.priors, dedup=self.dedup,
                                        soft=self.soft))
        return out

    def score_statements(
        self, statements: Sequence[Any],
        extra_evidence: list[list[Any]] | None = None,
    ) -> Sequence[float]:
        """INDRA's contract: one float in [0,1] per statement, same order.

        Raises rather than emitting a placeholder when a statement could not be
        read, because a missing belief is silently read back as 1.0.
        """
        rolled = self.score_statements_detailed(statements, extra_evidence)
        unscored = [i for i, r in enumerate(rolled) if r.belief is None]
        if unscored:
            raise UnscorableStatement(
                f"{len(unscored)} statement(s) yielded no readable verdict; first "
                f"at index {unscored[0]}. Returning a float here would be invented, "
                f"and omitting the field reads back as belief=1.0. Use "
                f"score_statements_detailed() to receive None instead."
            )
        return [r.belief for r in rolled]

    def score_statement(self, statement: Any,
                        extra_evidence: list[Any] | None = None) -> float:
        """Single-statement convenience, delegating to the batch form as INDRA does."""
        wrapped = None if extra_evidence is None else [extra_evidence]
        return self.score_statements([statement], wrapped)[0]
