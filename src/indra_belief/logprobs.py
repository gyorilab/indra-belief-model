"""Renormalised label probability from a reader's output distribution.

This is the information-based per-read confidence that `research/scoring_methods.md`
§2.2 proposes in place of the model's verbalized `confidence` field. For a closed,
lexically disjoint label set it is

    p_raw = sum_{v in A1} P(v) / ( sum_{v in A1} P(v) + sum_{v in A0} P(v) )

where A1 spells `correct` and A0 spells `incorrect` at the verdict-value token
position. Structurally this is Claim-Conditioned Probability (Fadeeva et al. 2024,
and the best claim-level method in Vashurin et al. TACL 2025 Table 5), with the
NLI entail/contradict partition replaced by an exact lexical one — valid precisely
because `verdict.VALID_VERDICTS` is closed and the two spellings share no prefix.

Nothing here touches the belief math: this module neither imports nor mutates
`noise_model.py`, and wiring p_raw into it is a separate, gated step.
This module only turns a transport-level distribution into a number plus enough
provenance to tell a real measurement from a degenerate one.
"""
from __future__ import annotations

import math
import re

# The two label spellings. No non-empty string is a prefix of both (they differ
# at the first character), so prefix matching partitions cleanly. Matching on
# PREFIXES rather than whole words is deliberate: if a tokenizer splits
# `incorrect` into `in` + `correct`, the decision the model actually makes at the
# label position is between starting `correct` and starting `incorrect`, and the
# first token is where that choice is expressed.
_CORRECT = "correct"
_INCORRECT = "incorrect"

# Locates the verdict VALUE. Mirrors verdict.py's JSON field, and like that
# module we take the LAST match: a reasoning preamble may rehearse the key
# before the answer emits it.
_VERDICT_KEY = re.compile(r'"verdict"\s*:\s*"', re.IGNORECASE)


def detokenize(token: str) -> str:
    """Best-effort raw-tokenizer-token -> text.

    Servers hand back tokenizer tokens, not decoded text: SentencePiece marks a
    leading space with U+2581, and byte-level BPE (GPT-2 style, used by Qwen and
    friends) maps control bytes into a printable range — space to U+0120 'Ġ',
    newline to U+010A 'Ċ', tab to U+0109 'ĉ'. mlx_lm.server returns
    `tokenizer.convert_ids_to_tokens(...)` verbatim, so these arrive unmapped.

    Getting the whitespace back matters: `verdict_position` locates the label by
    matching a regex against the reconstructed text, and a `"verdict":Ċ"` that
    should read `"verdict":\\n"` would fail to match.
    """
    return (token.replace("▁", " ").replace("Ġ", " ")
                 .replace("Ċ", "\n").replace("ĉ", "\t"))


def _label_of(token: str) -> str | None:
    """Return "correct", "incorrect", or None for a single alternative token."""
    t = detokenize(token).strip().strip('"“”\' ').lower()
    if not t:
        return None
    if _CORRECT.startswith(t):
        return _CORRECT
    if _INCORRECT.startswith(t):
        return _INCORRECT
    return None


def verdict_position(logprobs: list[dict]) -> int | None:
    """Index of the token holding the verdict VALUE, or None.

    Reconstructs the generated text from the token sequence, finds the last
    `"verdict": "` key, and returns the first token starting at or after the end
    of that match. Position is found by character offset, not by a fixed index:
    the reasoning-first prompt emits `relation_check`/`support`/`objection` of
    variable length ahead of the verdict, so no constant index exists.
    """
    if not logprobs:
        return None
    spans: list[tuple[int, int]] = []
    text_parts: list[str] = []
    cursor = 0
    for entry in logprobs:
        piece = detokenize(entry.get("token", "") or "")
        spans.append((cursor, cursor + len(piece)))
        text_parts.append(piece)
        cursor += len(piece)
    text = "".join(text_parts)

    matches = list(_VERDICT_KEY.finditer(text))
    if not matches:
        return None
    value_start = matches[-1].end()
    for i, (lo, hi) in enumerate(spans):
        if hi > value_start or lo >= value_start:
            return i
    return None


def label_probability(logprobs: list[dict], position: int | None = None) -> dict:
    """Renormalised P(correct) at the verdict position, with provenance.

    Returns a dict — never a bare float — because the number is meaningless
    without knowing whether both labels were actually observed in the returned
    top-k. Keys:

      p_raw            float in [0,1], or None if it could not be computed
      position         token index used
      p_correct_mass   summed probability of A1 alternatives (unnormalised)
      p_incorrect_mass summed probability of A0 alternatives (unnormalised)
      label_mass       p_correct_mass + p_incorrect_mass, i.e. how much of the
                       distribution at this position sits on either label. Close
                       to 1.0 means the model was really choosing between the two;
                       well below 1.0 means it was still deciding on formatting
                       and p_raw is conditioned on a small slice.
      both_observed    True iff BOTH labels appeared among the alternatives.
                       When False, the losing label fell outside the top-k, so
                       p_raw is a LOWER BOUND on confidence, not a measurement —
                       the true value is bounded by the k-th alternative's mass.
      precision_limited True when the raw masses summed above 1, which is
                       impossible over a disjoint partition and therefore proves
                       the provider rounded the winning token's logprob to
                       exactly 0.0 (exp(0)=1.0). MEASURED on mlx_lm: 66/100
                       forced-prefill records, worst label_mass 1.0564, where
                       `correct` reported 1.000000 while `incorrect` still held
                       0.0564. p_raw is a RATIO so the error largely cancels
                       (0.9466 reported vs ~0.944 true), but the masses are not
                       usable as probabilities and the flag must travel with them.
      label_mass       clamped to 1.0 when precision_limited, so it is never
                       reported as an impossible value. p_raw is unaffected:
                       scaling both masses by a constant leaves the ratio fixed.
      status           "ok" | "no_position" | "no_alternatives" | "no_label_mass"
    """
    empty = {
        "p_raw": None, "position": None, "p_correct_mass": 0.0,
        "p_incorrect_mass": 0.0, "label_mass": 0.0, "both_observed": False,
        "precision_limited": False,
    }
    if not logprobs:
        return {**empty, "status": "no_position"}
    pos = verdict_position(logprobs) if position is None else position
    if pos is None or not (0 <= pos < len(logprobs)):
        return {**empty, "status": "no_position"}

    alts = logprobs[pos].get("top") or []
    if not alts:
        return {**empty, "position": pos, "status": "no_alternatives"}

    m_correct = m_incorrect = 0.0
    seen_c = seen_i = False
    for a in alts:
        lab = _label_of(a.get("token", "") or "")
        if lab is None:
            continue
        p = math.exp(a["logprob"])
        if lab is _CORRECT or lab == _CORRECT:
            m_correct += p
            seen_c = True
        else:
            m_incorrect += p
            seen_i = True

    total = m_correct + m_incorrect
    if total <= 0.0:
        return {**empty, "position": pos, "status": "no_label_mass"}
    # Mass over a disjoint partition cannot exceed 1; if it does, the provider
    # rounded the winning logprob to 0.0. Scale both sides back — the ratio,
    # and therefore p_raw, is untouched.
    precision_limited = total > 1.0 + 1e-9
    if precision_limited:
        m_correct, m_incorrect = m_correct / total, m_incorrect / total
        total = 1.0
    return {
        "p_raw": m_correct / total,
        "position": pos,
        "p_correct_mass": m_correct,
        "p_incorrect_mass": m_incorrect,
        "label_mass": total,
        "both_observed": seen_c and seen_i,
        "precision_limited": precision_limited,
        "status": "ok",
    }


def from_response(response) -> dict:
    """`label_probability` for a ModelResponse, respecting logprobs_status.

    Propagates the transport's three-valued status so a silent provider refusal
    ("empty") and an unasked call ("not_requested") stay distinguishable from a
    genuine failure to find the verdict token.
    """
    status = getattr(response, "logprobs_status", "not_requested")
    if status != "ok":
        return {
            "p_raw": None, "position": None, "p_correct_mass": 0.0,
            "p_incorrect_mass": 0.0, "label_mass": 0.0,
            "both_observed": False, "precision_limited": False, "status": status,
        }
    return label_probability(response.logprobs or [])
