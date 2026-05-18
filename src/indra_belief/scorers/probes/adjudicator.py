"""X-phase log-odds adjudicator over ProbeBundle.

X3 (no-abstain doctrine): the FINAL verdict is binary {correct, incorrect}.
Score is the continuous output of a log-odds combiner over per-probe
factor tables; verdict is `score >= 0.5`.

Reason codes remain — but as MULTI-LABEL DIAGNOSTIC ANNOTATIONS, not
decision drivers. The legacy `_decide()` decision tree is preserved as
`_collect_reasons()` and runs alongside the scoring to tag rows for
audit (`grounding_gap`, `indirect_chain`, `role_swap`, etc.).

Failure handling: when any probe has source="abstain" (LLM call failed),
the corresponding probe's answer falls through to the factor table; the
factor for the failure-mode answer (typically "absent" or "abstain")
combines with the other probes' factors. No more abstain commitment;
the score honestly reflects the partial signal.
"""
from __future__ import annotations

import math

from indra_belief.scorers.commitments import (
    Adjudication,
    ClaimCommitment,
    GroundingVerdict,
)
from indra_belief.scorers.context import EvidenceContext
from indra_belief.scorers.probes.types import ProbeBundle, ProbeResponse


# X3 — conjunctive probe combiner with scope modulator.
#
# Why not log-odds sum: the four probes check INDEPENDENT NECESSARY
# CONDITIONS, not independent evidence for a single conclusion. A claim
# is correct iff: subject in right role AND object in right role AND
# relation matches AND scope asserts. Log-odds averages these and lets
# two strong positives outweigh one strong negative — semantically
# wrong: if the subject isn't in the evidence, the whole claim is
# unsupported regardless of what relation_axis says about other entities.
#
# Stage 1  Grounding veto:    sr/or == absent → score floor
# Stage 2  Role consistency:  swap/decoy/mediator-on-direct → low score
# Stage 3  Relation strength: relation_axis answer → base score
# Stage 4  Scope modulator:   pull toward 0.5 (hedged) or flip (negated)

RELATION_AXIS_BASE = {
    "direct_sign_match":         0.92,
    "direct_sign_mismatch":      0.10,
    "direct_axis_mismatch":      0.15,
    "direct_partner_mismatch":   0.15,
    "via_mediator":              0.65,   # fine for causal claims
    "via_mediator_partial":      0.55,
    "no_relation":               0.08,
    "abstain":                   0.55,   # entities present, relation ambiguous → lean positive
}

SCOPE_RETAIN = {
    "direct":   1.00,
    "asserted": 1.00,
    "hedged":   0.40,
    "abstain":  0.65,
}


def _apply_scope(joint_score: float, scope_answer: str | None) -> float:
    """Modulate the pre-scope score by scope answer.

    - direct/asserted → unchanged
    - hedged → pull toward 0.5 (retain 40% of polarity)
    - negated → flip (1 - score); sentence asserts the opposite
    - abstain → mild pull toward 0.5
    """
    if scope_answer == "negated":
        return 1.0 - joint_score
    retain = SCOPE_RETAIN.get(scope_answer or "direct", 1.0)
    return 0.5 + retain * (joint_score - 0.5)


def _confidence_from_score(score: float) -> str:
    """Map a continuous score to one of three confidence buckets."""
    distance_from_neutral = abs(score - 0.5)
    if distance_from_neutral >= 0.40:
        return "high"
    if distance_from_neutral >= 0.20:
        return "medium"
    return "low"


# Perturbation propagation: LOF inverts claim sign; GOF preserves; none preserves.
def _effective_claim_sign(claim_sign: str, perturbation: str | None) -> str:
    if perturbation == "LOF":
        if claim_sign == "positive":
            return "negative"
        if claim_sign == "negative":
            return "positive"
    return claim_sign


# INDRA causal claim types — these accept indirect chains (X→Z→Y is a
# valid Activation/Inhibition/IncreaseAmount/DecreaseAmount because the
# claim is about the upstream→downstream regulatory relationship, not a
# direct molecular contact). Direct claim types (Phosphorylation, Complex,
# Translocation, etc.) require X to directly contact Y.
_CAUSAL_STMT_TYPES = frozenset({
    "Activation", "Inhibition",
    "IncreaseAmount", "DecreaseAmount",
})


def _is_causal_claim(stmt_type: str) -> bool:
    return stmt_type in _CAUSAL_STMT_TYPES


def _final_arm_substrate_match(
    claim: ClaimCommitment, ctx: EvidenceContext,
) -> bool:
    """§5.4: return True iff ctx.detected_relations contains an aligned
    CATALOG match for (claim.subject, claim.objects[0], claim.axis)
    with the claim sign (or symmetric-binding equivalent).

    Mirrors the M3 substrate-fallback hoist preserved through R8.
    """
    if not claim.objects:
        return False
    target = claim.objects[0]
    for dr in ctx.detected_relations:
        # Direct alignment
        aligned = (
            dr.agent_canonical == claim.subject
            and dr.target_canonical == target
        )
        # Symmetric binding: (X,Y) ≡ (Y,X) for binding axis
        if claim.axis == "binding" and not aligned:
            aligned = (
                dr.agent_canonical == target
                and dr.target_canonical == claim.subject
            )
        if not aligned:
            continue
        if dr.axis != claim.axis:
            continue
        # Sign match — for binding/translocation axes both are "neutral";
        # for signed axes, sign must match the claim.
        if dr.sign == claim.sign:
            return True
        # Allow "neutral" detected sign on a neutral-axis claim
        if claim.sign == "neutral" and dr.sign == "neutral":
            return True
    return False


def _grounding_uncertain(groundings: tuple[GroundingVerdict, ...]) -> bool:
    return any(g.status == "uncertain" for g in groundings)


def _decide(
    bundle: ProbeBundle, claim: ClaimCommitment,
) -> tuple[str, str | None, str]:
    """Run the canonical decision table. Returns (verdict, reason, rationale).

    `verdict` ∈ {"correct", "incorrect", "abstain"}.
    `reason` is a ReasonCode string or None when no specific code applies.
    `rationale` is a short human-readable note (informational).
    """
    sr = bundle.subject_role.answer
    or_ = bundle.object_role.answer
    ra = bundle.relation_axis.answer
    sc = bundle.scope.answer

    # Any LLM-failure abstain on subject/object/relation/scope → adjudicator
    # cannot commit a verdict from probes alone. Return abstain; final-arm
    # substrate-fallback may still rescue if CATALOG matches.
    sources = (
        bundle.subject_role.source,
        bundle.object_role.source,
        bundle.relation_axis.source,
        bundle.scope.source,
    )
    if "abstain" in sources:
        return "abstain", None, "one or more probes abstained (LLM failure)"

    # Grounding-gap: subject or object not present in evidence.
    if sr == "absent":
        return "abstain", "grounding_gap", "claim subject not in evidence"
    if or_ == "absent":
        return "abstain", "grounding_gap", "claim object not in evidence"

    # Decoy: entities mentioned but not in the claim relation.
    # Treat as no-relation evidence.
    if sr == "present_as_decoy" or or_ == "present_as_decoy":
        return "incorrect", "absent_relationship", \
            "claim entity present only as decoy/control"

    # Mediator: subject or object in the middle of a chain.
    # Indirect chain — abstain (claim implies direct link).
    if sr == "present_as_mediator" or or_ == "present_as_mediator":
        return "abstain", "indirect_chain", \
            "claim entity is a chain mediator, not endpoint"

    # Role-swap: subject in object slot AND object in subject slot.
    # Only fires for non-binding axes (binding is symmetric §5.3).
    if (sr == "present_as_object" and or_ == "present_as_subject"
            and claim.axis != "binding"):
        return "incorrect", "role_swap", \
            "subject and object roles swapped in evidence"

    # From here both probes return present_as_subject and present_as_object
    # (or symmetric-binding equivalent). Now consult relation_axis.

    if ra == "no_relation":
        return "incorrect", "absent_relationship", \
            "no relation between resolved entities"

    if ra == "direct_axis_mismatch":
        return "incorrect", "axis_mismatch", \
            "relation present but on different axis"

    if ra == "direct_partner_mismatch":
        return "incorrect", "binding_domain_mismatch", \
            "binding-axis match but partner type incompatible"

    if ra == "direct_sign_mismatch":
        return "incorrect", "sign_mismatch", \
            "relation present but opposite sign"

    if ra == "via_mediator":
        # §5.6: INDRA semantics. Causal claims (Activation, Inhibition,
        # IncreaseAmount, DecreaseAmount) accept indirect chains —
        # "X activates Y via Z" is valid Activation(X, Y) at the
        # pathway level. Direct claims (Phosphorylation, Complex,
        # Translocation, etc.) require X→Y direct contact.
        if _is_causal_claim(claim.stmt_type):
            # Honor scope for causal indirect chains too.
            if sc == "asserted":
                return "correct", "match", \
                    "causal claim accepts indirect chain"
            if sc == "hedged":
                return "correct", "hedging_hypothesis", \
                    "causal indirect chain, hedged"
            if sc == "negated":
                return "incorrect", "contradicted", \
                    "causal indirect chain, negated"
            return "correct", "match", \
                "causal claim accepts indirect chain"
        return "abstain", "indirect_chain", \
            "direct claim type but evidence shows indirect chain"

    if ra == "via_mediator_partial":
        # Same causal-vs-direct distinction; less confidence.
        if _is_causal_claim(claim.stmt_type):
            return "correct", "chain_extraction_gap", \
                "causal claim with partial chain — accept at lower confidence"
        return "abstain", "chain_extraction_gap", \
            "direct claim, chain markers but no extractable mediator"

    if ra == "abstain":
        return "abstain", None, "relation underdetermined"

    # ra == "direct_sign_match" — consult scope.
    if sc == "asserted":
        return "correct", "match", "asserted relation matches claim"
    if sc == "hedged":
        # §5.7: hedged + matched relation → correct/low rather than
        # abstain. The relation IS asserted; the hedging modulates
        # confidence, not the verdict. Composed scorer can weight
        # low-confidence correct < high-confidence correct.
        return "correct", "hedging_hypothesis", \
            "hedged but relation matches claim"
    if sc == "negated":
        return "incorrect", "contradicted", \
            "relation explicitly negated"
    if sc == "abstain":
        # §5.7 corollary: if relation matches direct + sign and scope
        # is genuinely underdetermined, the most informative verdict
        # is correct/low — the relation evidence is there even if the
        # scope framing is unclear.
        return "correct", "match", \
            "relation matches; scope underdetermined → correct/low"

    # Unreachable — covers all closed-set values.
    return "abstain", None, "unhandled probe-tuple"


def _claim_score(bundle: ProbeBundle, claim: ClaimCommitment) -> float:
    """Conjunctive probe combiner. Returns pre-scope score in [0, 1].

    Stage 1 grounding veto, Stage 2 role consistency, Stage 3 relation
    strength. Stage 4 (scope) is applied by the caller via `_apply_scope`.
    """
    sr = bundle.subject_role.answer
    or_ = bundle.object_role.answer
    ra = bundle.relation_axis.answer

    # Stage 1: grounding veto. Claim entity absent from evidence → wrong.
    if sr == "absent" and or_ == "absent":
        return 0.05
    if sr == "absent" or or_ == "absent":
        return 0.15

    # Stage 2: role consistency.
    # Role swap (non-binding axis): subject in object slot AND vice versa.
    if (sr == "present_as_object" and or_ == "present_as_subject"
            and claim.axis != "binding"):
        return 0.10

    # Decoy: entity present only as control / bystander.
    if sr == "present_as_decoy" or or_ == "present_as_decoy":
        return 0.10

    # Mediator: indirect chain. Direct-relation claims (Phosphorylation,
    # Complex, Translocation, …) require direct contact; causal claims
    # (Activation, Inhibition, IncreaseAmount, DecreaseAmount) accept
    # chains per §5.6. Mediator can be signaled by EITHER role probe
    # OR by relation_axis.
    has_mediator = (
        sr in ("present_as_mediator", "via_mediator")
        or or_ in ("present_as_mediator", "via_mediator")
        or ra in ("via_mediator", "via_mediator_partial")
    )
    if has_mediator and not _is_causal_claim(claim.stmt_type):
        return 0.30

    # Stage 3: relation strength.
    return RELATION_AXIS_BASE.get(ra, 0.50)


def _collect_reasons(
    bundle: ProbeBundle, claim: ClaimCommitment,
) -> tuple[tuple[str, ...], str]:
    """Diagnostic reason tags + a human-readable rationale string.

    Calls the legacy `_decide` decision tree to recover its categorical
    reason, then layers in additional multi-label tags that the old
    single-label path discarded (e.g., both `hedged` AND `indirect_chain`
    can apply to one row).

    Returns (reasons_tuple, rationale_string).
    """
    legacy_verdict, legacy_reason, rationale = _decide(bundle, claim)

    reasons: list[str] = []
    if legacy_reason:
        reasons.append(legacy_reason)

    sr = bundle.subject_role.answer
    or_ = bundle.object_role.answer
    sc = bundle.scope.answer

    # Multi-label additions the legacy tree dropped because it returned
    # the first matching reason and exited.
    if sc == "hedged" and "hedging_hypothesis" not in reasons:
        reasons.append("hedging_hypothesis")
    if sc == "negated" and "contradicted" not in reasons:
        reasons.append("contradicted")
    if (sr == "absent" or or_ == "absent") and "grounding_gap" not in reasons:
        reasons.append("grounding_gap")
    if sr in ("present_as_mediator", "via_mediator") or \
       or_ in ("present_as_mediator", "via_mediator"):
        if "indirect_chain" not in reasons:
            reasons.append("indirect_chain")

    # Strip the legacy "abstain"-specific informational rationale; X3 never
    # emits abstain. The rationale string is informational only.
    return tuple(reasons), rationale


def adjudicate(
    claim: ClaimCommitment,
    bundle: ProbeBundle,
    groundings: tuple[GroundingVerdict, ...],
    *,
    ctx: EvidenceContext,
) -> Adjudication:
    """Probe-product log-odds combiner over ProbeBundle.

    X3 no-abstain doctrine: every (claim, bundle) commits to {correct, incorrect}.
    Score = sigmoid(Σ log-odds(per-probe factor)).
    Verdict = score >= 0.5.

    Reasons are multi-label diagnostic tags from `_collect_reasons` (the
    legacy decision tree, plus additional tags it dropped) — they describe
    WHY the score landed where it did, but they don't decide it.

    §5.4 final-arm substrate rescue: when the per-probe score is low AND
    a CATALOG-aligned regex pattern matches the (subject, object, axis,
    sign) tuple, lift the score to 0.65 (correct, low confidence). This
    preserves the M3 hoist for cases where probes were under-informed but
    substrate has a confident regex match.

    Grounding-uncertain rows get their score blended toward 0.5 (one
    log-odds unit toward neutral) so uncertain calls don't get the same
    confidence as fully-grounded ones.
    """
    # §5.1 — perturbation is already propagated upstream via
    # router._route_relation_axis (effective_sign computation).
    _ = _effective_claim_sign(
        claim.sign, bundle.subject_role.perturbation,
    )

    pre_scope = _claim_score(bundle, claim)
    score = _apply_scope(pre_scope, bundle.scope.answer)

    reasons, rationale = _collect_reasons(bundle, claim)

    # §5.4 final-arm substrate rescue — lift low scores when CATALOG
    # has an aligned regex match. Threshold: rescue only when the
    # probe-derived score is below 0.5 (i.e., we'd commit to incorrect).
    if score < 0.5 and _final_arm_substrate_match(claim, ctx):
        score = 0.65
        if "regex_substrate_match" not in reasons:
            reasons = reasons + ("regex_substrate_match",)
        rationale = f"final-arm substrate match (probe-score was {score:.2f})"

    # Grounding-uncertain blend: pull score 0.15 log-odds toward 0.5 so
    # uncertain-grounding calls express less confidence than fully-
    # grounded ones with the same probe answers.
    if _grounding_uncertain(groundings):
        eps = 1e-3
        x = max(min(score, 1.0 - eps), eps)
        L = math.log(x / (1.0 - x))
        L *= 0.85  # shrink toward 0 (= 0.5 in probability space)
        score = 1.0 / (1.0 + math.exp(-L))

    verdict = "correct" if score >= 0.5 else "incorrect"
    confidence = _confidence_from_score(score)

    return Adjudication(
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,  # type: ignore[arg-type]
        rationale=rationale,
        score=score,
    )
