"""Deterministic in-text abbreviation detection (Schwartz & Hearst, 2003).

Pure Python, no dependencies (deliberately not scispaCy — the repo keeps a lean
dep surface; the algorithm is ~100 lines). Finds `long-form (SHORT)` and the
inverted `SHORT (long-form)` definitions inside a sentence and returns
(short, long) pairs. This is INPUT substrate only: the caller grounds both forms
with Gilda and decides whether to surface the alias to the scorer prompt — this
module never grounds, never judges, never mutates anything.

The Schwartz-Hearst character match is what gives high precision: every
alphanumeric character of the short form must appear in the long form, in order,
scanned right-to-left, with the short's first character landing at a word start.
That single constraint rejects non-abbreviation parentheticals — `(see Fig 2)`,
`(Smith et al., 2019)`, `(p < 0.05)`, `(n = 12)`, `(an E3 ligase)` — because the
parenthetical is not an acronym derivable from the preceding words.

    find_abbreviations("... 5S ribonucleoprotein complex (5S RNP) ...")
    -> [("5S RNP", "5S ribonucleoprotein complex")]
"""
from __future__ import annotations

import re

# A parenthetical and the text preceding it (same span; sentence-local in practice
# since INDRA evidence is typically one sentence).
_PAREN = re.compile(r"\(([^()]+)\)")
# Tokeniser for the long-form window: words incl. internal hyphens/digits.
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


def _is_valid_short(short: str) -> bool:
    """Schwartz-Hearst short-form gate: 2-10 chars, <=2 tokens, has a letter,
    first char alphanumeric. Rejects citations / stats / glosses by shape."""
    s = short.strip()
    if not (2 <= len(s) <= 10):
        return False
    if not s[0].isalnum():
        return False
    if not any(c.isalpha() for c in s):
        return False
    if len(s.split()) > 2:
        return False
    # reject pure citation/stat shapes that slip the length gate
    if re.search(r"\d{4}", s):           # a year
        return False
    if re.fullmatch(r"[\d.,%<>=\s-]+", s):  # numeric/stat only
        return False
    return True


def _find_best_long_form(short: str, long: str) -> str | None:
    """Classic Schwartz-Hearst inner/outer scan. Returns the matched long-form
    substring (trimmed to start at the first matched character's word), or None
    if the short form's chars don't subsequence-match the long form."""
    s = len(short) - 1
    l = len(long) - 1
    while s >= 0:
        cur = short[s].lower()
        if not cur.isalnum():
            s -= 1
            continue
        while True:
            if l < 0:
                return None
            # the first short char must align to a word boundary in the long form
            at_word_start = (l == 0) or (not long[l - 1].isalnum())
            if long[l].lower() == cur and (s != 0 or at_word_start):
                break
            l -= 1
        l -= 1
        s -= 1
    l += 1
    # trim to the start of the word containing the first matched char
    while l > 0 and long[l - 1].isalnum():
        l -= 1
    return long[l:].strip()


def _accept_long(short: str, long: str) -> bool:
    """Schwartz-Hearst acceptance: the long form must not be longer than a sane
    bound for the short, must contain a letter, and must span >1 word OR be
    clearly longer than the short (rejects trivial near-identity matches)."""
    if not long or not any(c.isalpha() for c in long):
        return False
    n_short_alnum = sum(c.isalnum() for c in short)
    n_long_words = len(_WORD.findall(long))
    # long form has at most min(|A|+5, |A|*2) words
    if n_long_words > min(n_short_alnum + 5, n_short_alnum * 2):
        return False
    # reject the degenerate case where long == short (no real expansion)
    if long.strip().lower() == short.strip().lower():
        return False
    return True


def find_abbreviations(text: str) -> list[tuple[str, str]]:
    """Return de-duplicated (short, long) abbreviation pairs defined in `text`.

    Handles both `long-form (SHORT)` (the common case) and the inverted
    `SHORT (long-form)`. High precision by construction; silent on text with no
    in-line definition (that's a recall gap, not an error — the caller falls back
    to Gilda's standard synonyms).
    """
    if not text or "(" not in text:
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for m in _PAREN.finditer(text):
        inner = m.group(1).strip()
        before = text[: m.start()]

        # Case 1: long-form (SHORT) — `inner` is the short form.
        if _is_valid_short(inner):
            n_short = sum(c.isalnum() for c in inner)
            words = _WORD.findall(before)
            window_n = min(n_short + 5, n_short * 2)
            long_window = " ".join(words[-window_n:]) if words else ""
            long = _find_best_long_form(inner, long_window) if long_window else None
            if long and _accept_long(inner, long):
                pair = (inner, long)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
                    continue

        # Case 2: SHORT (long-form) — the token(s) just before "(" are the short
        # form, the parenthetical is its expansion. Try the last two tokens
        # before the last one ("5S RNP" beats "RNP").
        if len(inner.split()) >= 2:
            tail = _WORD.findall(before)
            for k in (2, 1):
                if len(tail) < k:
                    continue
                short2 = " ".join(tail[-k:])
                if not _is_valid_short(short2):
                    continue
                long2 = _find_best_long_form(short2, inner)
                if long2 and _accept_long(short2, long2):
                    pair = (short2, long2)
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)
                    break

    return pairs


# Gilda score below which an independent grounding is "not confident" — must match
# entity.LOW_CONFIDENCE_THRESHOLD; duplicated here to keep this module gilda-free.
_LOW_CONF = 0.53


def scoped_abbreviation_aliases(
    text: str,
    claim_entities: list[tuple[str, str, str, str]],
    ground,
    low_conf: float = _LOW_CONF,
) -> list[tuple[str, str, str, str]]:
    """Detect in-text abbreviations and keep ONLY the ones that (a) expand to a
    CLAIM entity and (b) don't collide with a different confident grounding.

    INPUT-only by construction — it reads, grounds, and returns; it touches no
    entity state and decides no verdict.

    Args:
        text: evidence sentence.
        claim_entities: ``(db, db_id, name, role)`` for the claim's subject/object,
            where role is e.g. "subject"/"object".
        ground: callable ``str -> list[(db, db_id, score)]`` (best hit first), or [].
            Dependency-injected so this stays gilda-free and unit-testable.
        low_conf: a short form grounding below this score is "not confident" and
            therefore not treated as a collision.

    Returns: list of ``(short, long, claim_name, role)`` aliases to surface.
    """
    pairs = find_abbreviations(text)
    if not pairs or not claim_entities:
        return []

    def _ground(s):
        try:
            return ground(s) or []
        except Exception:
            return []

    out: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for short, long in pairs:
        gl = _ground(long)
        if not gl:
            continue
        ldb, lid, _ = gl[0]
        lid = str(lid)
        match = next(
            ((db, did, nm, role) for (db, did, nm, role) in claim_entities
             if db == ldb and str(did) == lid),
            None,
        )
        if match is None:           # scope guard: long form isn't a claim entity
            continue
        _, _, name, role = match
        gs = _ground(short)
        if gs:
            sdb, sid, sscore = gs[0]
            same = (sdb == ldb and str(sid) == lid)
            if not same and sscore >= low_conf:   # collision guard
                continue
        key = (short.lower(), lid)
        if key in seen:
            continue
        seen.add(key)
        out.append((short, long, name, role))
    return out
