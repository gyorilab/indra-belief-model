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
