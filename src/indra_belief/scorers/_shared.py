"""Shared scorer primitives used by more than one scorer module.

Greek normalization previously existed in divergent copies; keeping it here
prevents the xi/upsilon omissions from recurring.
"""
from __future__ import annotations

# Greek letter glyphs (lowercase + capitalized) -> Latin shortform.
GREEK_GLYPHS: dict[str, str] = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e",
    "ζ": "z", "η": "h", "θ": "q", "ι": "i", "κ": "k",
    "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o",
    "π": "p", "ρ": "r", "σ": "s", "τ": "t", "υ": "u",
    "φ": "f", "χ": "c", "ψ": "y", "ω": "w",
    "Α": "a", "Β": "b", "Γ": "g", "Δ": "d", "Ε": "e",
    "Ζ": "z", "Η": "h", "Θ": "q", "Ι": "i", "Κ": "k",
    "Λ": "l", "Μ": "m", "Ν": "n", "Ξ": "x", "Ο": "o",
    "Π": "p", "Ρ": "r", "Σ": "s", "Τ": "t", "Υ": "u",
    "Φ": "f", "Χ": "c", "Ψ": "y", "Ω": "w",
}

# Greek letter words -> Latin shortform.
GREEK_WORDS: dict[str, str] = {
    "alpha": "a", "beta": "b", "gamma": "g", "delta": "d",
    "epsilon": "e", "zeta": "z", "eta": "h", "theta": "q",
    "iota": "i", "kappa": "k", "lambda": "l", "mu": "m",
    "nu": "n", "xi": "x", "omicron": "o", "pi": "p",
    "rho": "r", "sigma": "s", "tau": "t", "upsilon": "u",
    "phi": "f", "chi": "c", "psi": "y", "omega": "w",
}


def raw_text_for(name: str, evidence) -> str | None:
    """Pull the evidence-side raw_text surface form for a claim entity, or None."""
    try:
        agents = evidence.annotations.get("agents", {})
    except AttributeError:
        return None
    names = agents.get("agent_list") or []
    raws = agents.get("raw_text") or []
    for n, rt in zip(names, raws):
        if n == name and rt:
            return rt
    return None
