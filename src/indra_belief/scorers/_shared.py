"""Shared scorer primitives used by more than one scorer module.

Greek normalization previously existed in divergent copies; keeping it here
prevents the xi/upsilon omissions from recurring.
"""
from __future__ import annotations

import json

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


# Moved here from scorers/probes/_llm.py when the decomposed probe
# architecture was retired. Its surviving callers are both in
# scorers/monolithic/, and it needs nothing from that package — only json.
def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from a model response.

    Robust to thinking models, whose `content` is reasoning + a `</think>`
    delimiter + the final JSON, and whose reasoning often DRAFTS brace blocks
    mid-thought. The old "first { to last }" heuristic grabbed the span from a
    reasoning-draft `{` to the final `}` (prose in between -> invalid JSON ->
    silent None), which masqueraded as model leniency. Strategy:
      1) drop everything up to and including the last </think> (the answer
         always follows the reasoning),
      2) strict json.loads on the tail,
      3) otherwise scan for ALL brace-balanced (string/escape-aware) objects
         and return the LAST one that parses as a dict — the final answer is
         emitted last, not first.
    Returns None only when no balanced JSON object parses anywhere.
    """
    text = text.strip()
    lower = text.lower()
    cut = lower.rfind("</think>")
    if cut != -1:
        text = text[cut + len("</think>"):].strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Scan for brace-balanced objects, string/escape aware; keep the last dict.
    last: dict | None = None
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            last = obj
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return last
