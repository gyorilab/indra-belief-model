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
    from indra_belief.scorers.monolithic._prompts import _SCORE_GRID, verdict_to_score
    from indra_belief.scorers.commitments import _VERDICT_SCORE
    from indra_belief.scorers.panel.adjudicator import _SCORE
    assert _SCORE_GRID == _GOLDEN
    assert _VERDICT_SCORE == _GOLDEN
    assert _SCORE == _GOLDEN
    for (v, c), s in _GOLDEN.items():
        assert verdict_to_score(v, c) == s
    assert verdict_to_score(None, None) == 0.5  # null verdict default unchanged


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
