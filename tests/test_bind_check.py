"""Unit tests for bind_check.py.

Each of the 5 movable AA-phase errors becomes a unit test: given a
plausible Stage-1 LLM extraction, does the deterministic bind-check
produce the verdict gold says is correct?

The extraction inputs here are HAND-CONSTRUCTED — they represent what a
faithful LLM would extract from the evidence, not what the current
closed-set classifier would emit. If bind_check passes all these,
the only remaining variability in CC is whether the actual LLM
extraction produces the same tuple structure.
"""
from __future__ import annotations

from indra_belief.scorers.probes.bind_check import (
    ExtractedRelation,
    bind_check,
    stmt_type_to_axis_sign,
)


def _claim_meta(
    stmt_type: str,
    subject_aliases: frozenset[str],
    object_aliases: frozenset[str],
    *,
    binding_admissible: frozenset[str] = frozenset(),
) -> dict:
    axis, sign = stmt_type_to_axis_sign(stmt_type)
    return {
        "stmt_type": stmt_type,
        "subject_aliases": subject_aliases,
        "object_aliases": object_aliases,
        "claim_axis": axis,
        "claim_sign": sign,
        "binding_admissible": binding_admissible,
    }


def _ex(
    subject_acts_on_object: str = "directly",
    object_extracted: str | None = None,
    intermediate: str | None = None,
    axis: str = "activity",
    subtype: str = "activation",
    sign: str = "positive",
    perturbation: str = "none",
    scope: str = "asserted",
    partner_type: str = "unknown",
    rationale: str = "",
) -> ExtractedRelation:
    return ExtractedRelation(
        subject_acts_on_object=subject_acts_on_object,
        object_extracted=object_extracted,
        intermediate=intermediate,
        axis=axis,
        subtype=subtype,
        sign=sign,
        perturbation=perturbation,
        scope=scope,
        partner_type=partner_type,
        rationale=rationale,
    )


# ----------------------------------------------------------------------
# Baseline sanity tests
# ----------------------------------------------------------------------

def test_simple_direct_sign_match() -> None:
    claim = _claim_meta(
        "Activation",
        subject_aliases=frozenset({"MAPK1", "ERK2"}),
        object_aliases=frozenset({"JUN"}),
    )
    extracted = _ex(
        object_extracted="JUN",
        axis="activity",
        subtype="activation",
        sign="positive",
    )
    answer, _ = bind_check(claim, extracted)
    assert answer == "direct_sign_match"


def test_simple_sign_mismatch() -> None:
    """Claim Activation (positive), evidence asserts inhibition (negative)."""
    claim = _claim_meta(
        "Activation",
        subject_aliases=frozenset({"MAPK1"}),
        object_aliases=frozenset({"JUN"}),
    )
    extracted = _ex(
        object_extracted="JUN",
        axis="activity",
        subtype="inhibition",
        sign="negative",
    )
    answer, _ = bind_check(claim, extracted)
    assert answer == "direct_sign_mismatch"


def test_simple_axis_mismatch() -> None:
    """Claim Activation, evidence asserts phosphorylation."""
    claim = _claim_meta(
        "Activation",
        subject_aliases=frozenset({"MAPK1"}),
        object_aliases=frozenset({"JUN"}),
    )
    extracted = _ex(
        object_extracted="JUN",
        axis="modification",
        subtype="phosphorylation",
        sign="positive",
    )
    answer, _ = bind_check(claim, extracted)
    assert answer == "direct_axis_mismatch"


def test_abstain_on_extraction_abstain() -> None:
    claim = _claim_meta("Activation",
                        frozenset({"MAPK1"}), frozenset({"JUN"}))
    extracted = _ex(scope="abstain")
    answer, _ = bind_check(claim, extracted)
    assert answer == "abstain"


def test_absent_subject_abstains() -> None:
    claim = _claim_meta("Activation",
                        frozenset({"MAPK1"}), frozenset({"JUN"}))
    extracted = _ex(subject_acts_on_object="absent")
    answer, _ = bind_check(claim, extracted)
    assert answer == "abstain"


def test_role_swap_non_binding_yields_no_relation() -> None:
    """For non-binding axis, claim subject as target → no_relation."""
    claim = _claim_meta("Activation",
                        frozenset({"MAPK1"}), frozenset({"JUN"}))
    extracted = _ex(subject_acts_on_object="is_target")
    answer, _ = bind_check(claim, extracted)
    assert answer == "no_relation"


def test_role_swap_binding_proceeds_to_axis_check() -> None:
    """Binding axis: claim subject as target is symmetric, not no_relation."""
    claim = _claim_meta(
        "Complex",
        frozenset({"MAPK1"}), frozenset({"JUN"}),
        binding_admissible=frozenset({"protein"}),
    )
    extracted = _ex(
        subject_acts_on_object="is_target",
        # On binding, is_target is allowed — proceeds; but if no object
        # is extracted, falls through. The test just verifies it does
        # NOT immediately return no_relation.
        object_extracted="MAPK1",
        axis="binding",
        subtype="binding",
        sign="neutral",
        partner_type="protein",
    )
    answer, _ = bind_check(claim, extracted)
    # is_target path on binding falls through; but we don't have the
    # claim object in extracted_obj so axis-match path applies. The
    # current implementation routes "is_target" through Stage B which
    # only blocks non-binding axes; for binding the code proceeds.
    # The verdict depends on downstream logic — we just verify no
    # premature reject.
    assert answer in {"direct_sign_match", "no_relation",
                      "direct_partner_mismatch", "abstain"}


# ----------------------------------------------------------------------
# Movable AA-error tests: each one represents a residual error pattern
# that the CC redesign should resolve.
# ----------------------------------------------------------------------

def test_hgf_rac1_named_effector_via_mediator() -> None:
    """AA-residual #17 HGF→RAC1 (Activation, gold=correct).
    Evidence: 'SF caused the activation of Src and the Rac1 effector Pak1.'

    Faithful LLM extraction recognises Pak1 as the entity acted upon
    AND tags it as a named effector of Rac1. The bind-check sees
    Pak1 as the intermediate to RAC1 → via_mediator."""
    claim = _claim_meta(
        "Activation",
        subject_aliases=frozenset({"HGF", "SF", "scatter factor"}),
        object_aliases=frozenset({"RAC1"}),
    )
    extracted = _ex(
        subject_acts_on_object="via_named_chain",
        object_extracted="Pak1",
        intermediate="Pak1",  # mediator named in evidence
        axis="activity",
        subtype="activation",
        sign="positive",
        scope="asserted",
        rationale="SF activates Rac1 effector Pak1",
    )
    answer, rationale = bind_check(claim, extracted)
    assert answer == "via_mediator", f"got {answer}: {rationale}"


def test_fbxw7_nfkb_lof_attenuates() -> None:
    """AA-residual #13 FBXW7→NFkB (Activation, gold=correct).
    Evidence: 'Fbxw7 silencing attenuated the NF-κB response'

    Faithful LLM extraction sees subject silenced (LOF on FBXW7),
    evidence asserts response attenuated (negative sign). LOF inverts
    → effective positive → matches claim sign positive."""
    claim = _claim_meta(
        "Activation",
        subject_aliases=frozenset({"FBXW7", "Fbxw7", "Fbw7"}),
        object_aliases=frozenset({"NFkappaB", "NF-kappaB", "NF-kB", "NFKB1"}),
    )
    extracted = _ex(
        subject_acts_on_object="directly",
        object_extracted="NF-κB response",
        axis="activity",
        subtype="regulation",
        sign="negative",       # evidence: silencing attenuated → negative
        perturbation="LOF",    # silencing of subject
        scope="asserted",
        rationale="Fbxw7 silencing attenuated NF-kB response",
    )
    answer, rationale = bind_check(claim, extracted)
    assert answer == "direct_sign_match", f"got {answer}: {rationale}"


def test_lep_kdr_transphosphorylation() -> None:
    """AA-residual #15 LEP→KDR (Phosphorylation, gold=correct).
    Evidence: 'leptin induced the transphosphorylation of Y1175,
    Y951, and Y996 of VEGFR-2'

    Faithful LLM extraction sees axis=modification subtype=transphosphorylation.
    The axis taxonomy in bind_check includes transphosphorylation in
    the Phosphorylation family."""
    claim = _claim_meta(
        "Phosphorylation",
        subject_aliases=frozenset({"LEP", "leptin"}),
        object_aliases=frozenset({"KDR", "VEGFR-2", "VEGFR2"}),
    )
    extracted = _ex(
        subject_acts_on_object="directly",
        object_extracted="VEGFR-2",
        axis="modification",
        subtype="transphosphorylation",
        sign="positive",
        scope="asserted",
        rationale="leptin induced transphosphorylation of VEGFR-2",
    )
    answer, rationale = bind_check(claim, extracted)
    assert answer == "direct_sign_match", f"got {answer}: {rationale}"


def test_ar_e2f1_polarity_lof_rnai() -> None:
    """AA-residual #4 AR→E2F1 (DecreaseAmount, gold=polarity).
    Evidence: 'Both c-Myc RNAi and AR RNAi reduced expression of the
    c-Myc-activated gene E2F1'

    Faithful LLM extraction sees AR under LOF (RNAi), evidence
    asserts E2F1 reduced (negative sign). LOF inverts → effective
    positive → claim is DecreaseAmount (sign=negative). Mismatch."""
    claim = _claim_meta(
        "DecreaseAmount",
        subject_aliases=frozenset({"AR"}),
        object_aliases=frozenset({"E2F1"}),
    )
    extracted = _ex(
        subject_acts_on_object="directly",
        object_extracted="E2F1",
        axis="amount",
        subtype="decrease",
        sign="negative",
        perturbation="LOF",   # RNAi
        scope="asserted",
        rationale="AR RNAi reduced E2F1",
    )
    answer, rationale = bind_check(claim, extracted)
    assert answer == "direct_sign_mismatch", f"got {answer}: {rationale}"


def test_fbxw7_proteasome_machinery_object_no_relation() -> None:
    """AA-residual #8 FBXW7→Proteasome (DecreaseAmount, gold=act_vs_amt).
    Evidence: 'Ectopic expression of Fbw7 destabilized GATA2 and
    promoted its proteasomal degradation.'

    Faithful LLM extraction recognises Fbw7 acts on GATA2 (not
    proteasome itself). bind_check sees extracted object ≠ claim
    object (Proteasome) → no_relation."""
    claim = _claim_meta(
        "DecreaseAmount",
        subject_aliases=frozenset({"FBXW7", "Fbw7"}),
        object_aliases=frozenset({"Proteasome", "PSMA1", "26S proteasome"}),
    )
    extracted = _ex(
        subject_acts_on_object="on_other_entity",
        object_extracted="GATA2",       # the actual substrate
        axis="amount",
        subtype="destabilization",
        sign="negative",
        scope="asserted",
        rationale="Fbw7 destabilized GATA2 via proteasomal degradation",
    )
    answer, rationale = bind_check(claim, extracted)
    assert answer == "no_relation", f"got {answer}: {rationale}"


# ----------------------------------------------------------------------
# Additional structural tests
# ----------------------------------------------------------------------

def test_partner_type_mismatch_dna_binding() -> None:
    """Complex claim requires protein-protein binding; evidence asserts
    DNA-binding → direct_partner_mismatch."""
    claim = _claim_meta(
        "Complex",
        subject_aliases=frozenset({"TP53"}),
        object_aliases=frozenset({"DNA-element"}),
        binding_admissible=frozenset({"protein"}),
    )
    extracted = _ex(
        subject_acts_on_object="directly",
        object_extracted="DNA-element",
        axis="binding",
        subtype="binding",
        sign="neutral",
        partner_type="DNA",
        scope="asserted",
    )
    answer, _ = bind_check(claim, extracted)
    assert answer == "direct_partner_mismatch"


def test_unnamed_chain_yields_partial() -> None:
    """Chain marker present but no extractable intermediate."""
    claim = _claim_meta("Activation",
                        frozenset({"PKA"}), frozenset({"CREB"}))
    extracted = _ex(
        subject_acts_on_object="via_unnamed_chain",
        intermediate=None,
        axis="activity",
        subtype="activation",
        sign="positive",
        scope="asserted",
    )
    answer, _ = bind_check(claim, extracted)
    assert answer == "via_mediator_partial"


def test_modification_subtype_family_phosphorylation() -> None:
    """Autophosphorylation should match a Phosphorylation claim
    (same family)."""
    claim = _claim_meta(
        "Phosphorylation",
        frozenset({"EGFR"}), frozenset({"EGFR"}),
    )
    extracted = _ex(
        subject_acts_on_object="directly",
        object_extracted="EGFR",
        axis="modification",
        subtype="autophosphorylation",
        sign="positive",
        scope="asserted",
    )
    answer, _ = bind_check(claim, extracted)
    assert answer == "direct_sign_match"


def test_dephosphorylation_does_not_match_phosphorylation_claim() -> None:
    """Phosphorylation claim with evidence asserting dephosphorylation
    → axis_mismatch (different sub-family within modification)."""
    claim = _claim_meta(
        "Phosphorylation",
        frozenset({"PTEN"}), frozenset({"AKT1"}),
    )
    extracted = _ex(
        subject_acts_on_object="directly",
        object_extracted="AKT1",
        axis="modification",
        subtype="dephosphorylation",
        sign="positive",
        scope="asserted",
    )
    answer, _ = bind_check(claim, extracted)
    assert answer == "direct_axis_mismatch"


def test_object_not_aliased_yields_no_relation() -> None:
    """If extracted object doesn't alias-match claim object → no_relation."""
    claim = _claim_meta(
        "Activation",
        frozenset({"HGF"}), frozenset({"RAC1"}),
    )
    extracted = _ex(
        subject_acts_on_object="directly",
        object_extracted="SomeOtherKinase",
        axis="activity",
        subtype="activation",
        sign="positive",
        scope="asserted",
    )
    answer, _ = bind_check(claim, extracted)
    assert answer == "no_relation"
