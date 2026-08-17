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


class _Auto:
    """Sentinel: 'resolve the calibration profile from the client and variant'.

    Distinct from None, which means 'use the hard gate' and is a legitimate
    caller request that must keep working unchanged.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "AUTO"


AUTO = _Auto()

__all__ = ["LLMBeliefScorer", "UnscorableStatement", "HAVE_INDRA", "AUTO"]


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
                 soft: dict | None | _Auto = AUTO, max_tokens: int | None = None,
                 variant: Any = None, dedup: bool = True,
                 probe_weights: bool | None = None) -> None:
        self.client = client
        self.priors = priors
        # `soft` used to default to None, and None means HARD GATE. Since no
        # caller ever passed a profile, the documented drop-in
        # `BeliefEngine(scorer=LLMBeliefScorer(client))` served hard-gate beliefs
        # while a fitted, ship-gated profile sat unused: ECE 0.237 against 0.045
        # on external_curator_gold_v1. The scorer already holds everything needed
        # to resolve its own profile — the client names the model, the variant
        # names the prompt — so AUTO does that lookup.
        #
        # AUTO is a distinct sentinel rather than a re-reading of None, because
        # `soft=None` is a real request for the hard gate and must keep working.
        self.soft = self._resolve_calibration(client, variant) if soft is AUTO else soft
        self.max_tokens = max_tokens
        self.variant = variant
        self.dedup = dedup
        # Opt-in: use the direct logit probe's continuous weight of evidence in
        # place of the two per-verdict constants. Requires rows carrying a
        # measured `weight_of_evidence`
        # (a probe-capable client) and a fitted profile; rows without one keep
        # their verdict weight, so enabling it before a probe run is a no-op.
        # UNEVALUATED at statement grain in this additive form — see
        # statement_belief's docstring. `StatementBelief.weighting` records which
        # rule produced each number.
        # None defers to statement_belief's AUTO: engage measured weights
        # wherever rows carry them and a profile resolved. Explicit True/False
        # still demand or refuse.
        from indra_belief.statement_belief import AUTO as _WEIGHTS_AUTO

        self.probe_weights = _WEIGHTS_AUTO if probe_weights is None else probe_weights

    @staticmethod
    def _resolve_calibration(client: Any, variant: Any) -> dict | None:
        """The ship-approved profile for this client+prompt, or None.

        Resolution is by EXACT (model, prompt sha256). An unfitted configuration
        returns None and the scorer stays on the hard gate — the same fail-safe
        the rest of the codebase uses. Imports are local: this module must stay
        importable without the monolithic scorer's dependency graph.
        """
        import hashlib

        model = getattr(client, "model_name", None)
        if not model:
            return None
        try:
            from indra_belief.calibration_constants import calibration_for
            from indra_belief.scorers.monolithic.scorer import DEFAULT_VARIANT
        except Exception:  # pragma: no cover - defensive, keeps the socket usable
            return None
        resolved = variant if variant is not None else DEFAULT_VARIANT
        prompt = getattr(resolved, "system_prompt", None)
        if not isinstance(prompt, str) or not prompt:
            return None
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return calibration_for(model, prompt_sha256=sha)

    @property
    def calibration_profile_id(self) -> str | None:
        """Which fitted profile this scorer resolved, or None on the hard gate.

        Exposed because a silently-resolved profile is as opaque as a silently
        missing one: a caller must be able to see which calibration produced the
        numbers it is about to publish.
        """
        return (self.soft or {}).get("profile_id")

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
                                        soft=self.soft,
                                        probe_weights=self.probe_weights))
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
