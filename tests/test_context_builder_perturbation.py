"""Unit tests for context_builder._detect_perturbation_for (Z2 gate).

Z2 doctrine: substrate LOF detection must require an explicit
perturbation verb/noun (knockdown, knockout, inhibitor, antagonist,
blockade, silencing, depletion, shRNA, siRNA, mutant, dominant-
negative). Benign quantity/state/coordination markers ("monomeric",
"soluble", "neither", "membrane-bound") MUST NOT fire LOF — those
were the Class-B Y-phase FPs (3 of 26 eval_set_v4 residuals).

Positive controls cover the standard LOF surface forms PLUS the new
Z2 additions (mutant, dominant-negative).
"""
from __future__ import annotations

from indra_belief.scorers.context_builder import _detect_perturbation_for


def _detect(text: str, name: str) -> str | None:
    """Convenience wrapper — caller-only knows the entity name."""
    return _detect_perturbation_for(text, name, frozenset({name}))


# ---------------------------------------------------------------------------
# Positive controls — explicit LOF perturbation surface forms must fire.
# ---------------------------------------------------------------------------

def test_inhibitor_entity_first_fires_lof() -> None:
    assert _detect("PI3K inhibitor abolished signaling.", "PI3K") == "loss_of_function"


def test_inhibitor_of_form_fires_lof() -> None:
    assert _detect("inhibition of AGER blocked downstream effects.", "AGER") == "loss_of_function"


def test_blockade_fires_lof() -> None:
    assert _detect("AGER blockade reduced cytokine release.", "AGER") == "loss_of_function"


def test_antagonist_fires_lof() -> None:
    assert _detect("CCR5 antagonist treatment reduced infection.", "CCR5") == "loss_of_function"


def test_knockdown_fires_lof() -> None:
    assert _detect("MAPK1 knockdown attenuated proliferation.", "MAPK1") == "loss_of_function"


def test_knockout_fires_lof() -> None:
    assert _detect("Mapk1 knockout mice showed defects.", "Mapk1") == "loss_of_function"


def test_silencing_entity_first_fires_lof() -> None:
    assert _detect("VHL silencing increased vimentin expression.", "VHL") == "loss_of_function"


def test_silencing_of_form_fires_lof() -> None:
    assert _detect("silencing of VHL induced EMT markers.", "VHL") == "loss_of_function"


def test_sirna_fires_lof() -> None:
    assert _detect("TP53 siRNA reduced apoptosis.", "TP53") == "loss_of_function"


def test_shrna_fires_lof() -> None:
    assert _detect("TP53 shRNA cells survived stress.", "TP53") == "loss_of_function"


def test_deficient_fires_lof() -> None:
    assert _detect("VHL deficient cells lacked HIF degradation.", "VHL") == "loss_of_function"


def test_null_fires_lof() -> None:
    assert _detect("BMP4 null mice showed cardiac defects.", "BMP4") == "loss_of_function"


def test_depletion_fires_lof() -> None:
    assert _detect("depletion of MED1 disrupted transcription.", "MED1") == "loss_of_function"


def test_inhibiting_fires_lof() -> None:
    assert _detect("inhibiting MEK blocked ERK phosphorylation.", "MEK") == "loss_of_function"


# ---------------------------------------------------------------------------
# Z2 additions — mutant and dominant-negative must fire.
# ---------------------------------------------------------------------------

def test_mutant_entity_first_fires_lof() -> None:
    """Z2: '<X> mutant' is a perturbation comparator."""
    assert _detect("TP53 mutant cells were resistant to apoptosis.", "TP53") == "loss_of_function"


def test_mutant_verb_first_fires_lof() -> None:
    """Z2: 'mutant <X>' is the adjective-first variant."""
    assert _detect("mutant TP53 sustained proliferation.", "TP53") == "loss_of_function"


def test_dominant_negative_hyphenated_fires_lof() -> None:
    """Z2: 'dominant-negative <X>' is unambiguous perturbation."""
    assert _detect("dominant-negative MAPK1 abolished signaling.", "MAPK1") == "loss_of_function"


def test_dominant_negative_spaced_fires_lof() -> None:
    """Z2: 'dominant negative <X>' (unhyphenated) also fires."""
    assert _detect("dominant negative MAPK1 was used as a control.", "MAPK1") == "loss_of_function"


# ---------------------------------------------------------------------------
# Negative controls — benign quantity/state/coordination markers must NOT fire.
# These cover the Y-phase Class-B FP mechanism (substrate over-detection on
# non-perturbation context).
# ---------------------------------------------------------------------------

def test_monomeric_does_not_fire() -> None:
    """Z2: 'monomeric <X>' describes a quaternary state, not a perturbation."""
    assert _detect("monomeric VHL was sufficient for HIF binding.", "VHL") is None


def test_neither_does_not_fire() -> None:
    """Z2: 'neither <X> nor Y' is a coordination negation, not perturbation."""
    assert _detect("neither VHL nor pVHL30 affected the outcome.", "VHL") is None


def test_soluble_does_not_fire() -> None:
    """Z2: 'soluble <X>' describes a biochemical fraction, not perturbation."""
    assert _detect("soluble TNF receptor bound the cytokine.", "TNF") is None


def test_membrane_bound_does_not_fire() -> None:
    """Z2: 'membrane-bound <X>' is a localization descriptor."""
    assert _detect("membrane-bound BMP4 signaled to neighboring cells.", "BMP4") is None


def test_lacked_does_not_fire_without_other_lof_cues() -> None:
    """Z2: 'X lacked Y' is a property assertion, not perturbation of X."""
    assert _detect("MAPK1 lacked the canonical activation loop.", "MAPK1") is None


def test_recombinant_does_not_fire() -> None:
    """Z2: 'recombinant <X>' is a production form, not perturbation."""
    assert _detect("recombinant TP53 was added to the reaction.", "TP53") is None


def test_purified_does_not_fire() -> None:
    """Z2: 'purified <X>' is a preparation descriptor."""
    assert _detect("purified MAPK1 phosphorylated the substrate.", "MAPK1") is None


# ---------------------------------------------------------------------------
# Negative control — entity must actually be the perturbation target.
# Without entity anchoring, generic perturbation language about ANOTHER
# entity should not fire on the claim subject.
# ---------------------------------------------------------------------------

def test_other_entity_inhibitor_does_not_fire_on_claim_subject() -> None:
    """LOF anchored to a different entity must not fire on the claim subject."""
    # PI3K inhibitor mentioned; claim subject is MAPK1 — must NOT flag MAPK1 as LOF.
    assert _detect("PI3K inhibitor reduced MAPK1 phosphorylation.", "MAPK1") is None
