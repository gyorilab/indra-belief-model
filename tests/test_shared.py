"""Gate tests for shared scorer primitives.

Greek normalization is one canonical superset covering all 24 letters,
including xi/upsilon which a divergent copy had dropped. Verdict probabilities
are deliberately not shared parser primitives: they come from calibration.
"""
import indra_belief.scorers._shared as shared
import indra_belief.verdict as verdict_module
from indra_belief.scorers._shared import GREEK_GLYPHS, GREEK_WORDS


def test_shared_primitives_do_not_expose_the_retired_grid():
    assert not hasattr(shared, "VERDICT_SCORE_GRID")
    assert not hasattr(verdict_module, "grid_score")


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
