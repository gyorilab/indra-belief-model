"""Schwartz-Hearst abbreviation detector — pure, dependency-free.

High precision is the whole point: positives must be found, and non-abbreviation
parentheticals (citations, stats, glosses) must be rejected by the char-match.
"""
from indra_belief.data.abbreviations import find_abbreviations


def test_long_form_short_the_5s_rnp_case():
    text = ("RPL5 and RPL11 can bind MDM2 alone or can interact with 5S rRNA, "
            "forming the 5S ribonucleoprotein complex (5S RNP), which binds MDM2 "
            "and stabilizes p53.")
    pairs = find_abbreviations(text)
    assert ("5S RNP", "5S ribonucleoprotein complex") in pairs


def test_classic_mtor():
    pairs = find_abbreviations("the mammalian target of rapamycin (mTOR) pathway")
    assert ("mTOR", "mammalian target of rapamycin") in pairs


def test_inverted_short_then_long():
    pairs = find_abbreviations("5S RNP (5S ribonucleoprotein complex) binds MDM2.")
    assert any(s == "5S RNP" and "ribonucleoprotein" in l for s, l in pairs)


def test_rejects_citation():
    assert find_abbreviations("phosphorylates YB-1 (Smith et al., 2019).") == []


def test_rejects_stat():
    assert find_abbreviations("expression increased (p < 0.05) in treated cells.") == []
    assert find_abbreviations("knockdown reduced viability (n = 12).") == []


def test_rejects_gloss():
    # "an E3 ligase" is a description, not an acronym of MDM2 -> no char match
    assert find_abbreviations("RPL11 binds MDM2 (an E3 ubiquitin ligase).") == []


def test_rejects_figure_ref():
    assert find_abbreviations("the complex stabilizes p53 (see Figure 2).") == []


def test_empty_and_no_parens():
    assert find_abbreviations("") == []
    assert find_abbreviations("RPL5 binds MDM2.") == []


def test_real_protein_acronym():
    pairs = find_abbreviations(
        "activation of nuclear factor kappa B (NF-kB) drives transcription.")
    assert any(s == "NF-kB" and "nuclear factor" in l for s, l in pairs)
