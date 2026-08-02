"""Gate tests for the shared-primitive merges.

T2: the (verdict, confidence) -> score grid is one canonical table used by every
    consumer, with the exact committed values (byte-exact golden).
T1: Greek normalization is one canonical superset covering ALL 24 letters,
    including xi/upsilon which a divergent copy had dropped.
"""
from indra_belief.scorers._shared import VERDICT_SCORE_GRID, GREEK_GLYPHS, GREEK_WORDS


# ---- T2: verdict-score grid golden ----

_GOLDEN = {
    ("correct", "high"): 0.95, ("correct", "medium"): 0.80, ("correct", "low"): 0.65,
    ("incorrect", "low"): 0.35, ("incorrect", "medium"): 0.20, ("incorrect", "high"): 0.05,
}


def test_grid_values_are_golden():
    assert VERDICT_SCORE_GRID == _GOLDEN


def test_all_consumers_share_the_grid():
    from indra_belief.verdict import grid_score
    from indra_belief.scorers.commitments import _VERDICT_SCORE
    from indra_belief.scorers.panel.adjudicator import _SCORE
    assert _VERDICT_SCORE == _GOLDEN
    assert _SCORE == _GOLDEN
    for (v, c), s in _GOLDEN.items():
        assert grid_score(v, c) == s


def test_an_unscorable_pair_is_absent_not_neutral():
    """K2-one-parser. The monolithic scorer's own map used to answer a null
    verdict with 0.5 and an off-grid confidence with a 0.50 `.get` default —
    values not on the grid above, which no model output can produce, landing on
    the exact point a calibration curve is most sensitive to. Both are gone: the
    six cells are the only scores, and everything else is `None`."""
    from indra_belief.verdict import grid_score

    assert grid_score(None, None) is None
    assert grid_score(None, "high") is None
    assert grid_score("maybe", "high") is None
    assert grid_score("correct", "certain") is None
    assert 0.5 not in _GOLDEN.values() and 0.50 not in _GOLDEN.values()
    # An ABSENT confidence is different in kind: it resolves to a REAL cell.
    assert grid_score("correct", None) == _GOLDEN[("correct", "medium")]
    assert grid_score("incorrect", None) == _GOLDEN[("incorrect", "medium")]


# ---- T1: Greek normalization parity (incl. the xi/upsilon fix) ----

_LETTERS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
            "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
            "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]


def test_canonical_covers_all_24_letters():
    assert len(GREEK_WORDS) == 24
    assert set(GREEK_WORDS) == set(_LETTERS)
    # both lowercase + capitalized glyphs for all 24
    assert len(GREEK_GLYPHS) == 48
    for g in ("ξ", "Ξ", "υ", "Υ"):  # the previously-dropped letters
        assert g in GREEK_GLYPHS


def test_norm_alias_handles_xi_and_upsilon():
    from indra_belief.scorers.context_builder import _norm_alias
    # the bug-fix cases (xi / upsilon were unnormalized before the merge)
    assert _norm_alias("p38ξ") == "p38x"
    assert _norm_alias("PKCυ") == "pkcu"
    assert _norm_alias("p38xi") == "p38x"
    # existing behavior preserved
    assert _norm_alias("p38α") == "p38a"
    assert _norm_alias("PKCbeta") == "pkcb"
    assert _norm_alias("PI3Kbeta") == "p3kb"  # pre-existing: "pi" word eats "PI" (consistent both sides)
    assert _norm_alias("NF-κB") == "nfkb"
    assert _norm_alias("") is None
