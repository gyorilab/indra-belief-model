"""Statement-level belief from per-evidence LLM verdicts.

The scorer judges one (statement, evidence) pair at a time; this rolls a
statement's evidence up to a single belief plus a decision-useful verdict and a
tally. It is the "replace" design: a best-effort statement belief meant to beat
INDRA's source-reliability belief, not a faithful reconstruction of it.

How a statement belief is built from evidence verdicts
------------------------------------------------------
The LLM verdict is a per-sentence MEMBERSHIP GATE on INDRA's own additive
noisy-OR (``noise_model.compute_gated_belief``): a ``correct`` verdict keeps its
evidence in the source's count, an ``incorrect`` verdict drops it. Evidence that
was never semantically read — ``no_text`` (correct-by-default), parse failures,
or rows with no source — contributes NOTHING (it is excluded from the numerator,
never credited as support). Belief is then INDRA's formula over the surviving,
de-duplicated evidence, under the recalibrated text-miner priors.

Three outputs travel together, and the scalar never travels without the tally:
  - ``belief``           the gated noisy-OR scalar (None when nothing was read)
  - ``verdict_statement`` a tiered decision: deterministic grounding rejects are
                          credible enough to hard-flag ``incorrect``; an LLM
                          ``incorrect`` only routes to ``review`` (its error rate
                          is too high to auto-condemn); else ``correct``
  - the tally            counts that caption the scalar and disambiguate the
                          empty set (all-unread vs genuinely-contradicted)

Edge cases
----------
- single-evidence:        gate on one read; belief is that source's 1-evidence
                          reliability, or 0.0 if gated out.
- all unread (no_text /   belief is None (UNDEFINED) — surface the prior alone;
  parse-fail / no source) never 0.0 (which means "contradicted") and never 0.95.
- mix correct+incorrect:  incorrect drops from its source count but does not
                          poison surviving siblings; a credible contradiction is
                          surfaced via verdict_statement + the tally, not by
                          collapsing the scalar (different question than gold's
                          any-incorrect-wins).
- false corroboration:    within-source text-normalized de-dup collapses
                          paraphrase pile-ups; the per-source systematic floor
                          caps any single source at 1 - syst regardless of count.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .noise_model import RECALIBRATED_PRIORS, compute_gated_belief
from .scorers._shared import GREEK_GLYPHS

# Deterministic grounding rejects: high-precision, credible enough to hard-flag.
_DETERMINISTIC_TIERS = {"deterministic_mismatch", "deterministic_pseudogene"}
# LLM verdicts are only credible (for routing to review) at these confidences.
_CREDIBLE_LLM_CONF = {"high", "medium"}
# Evidence that was never semantically read — excluded from the belief numerator.
_NO_TEXT_TIER = "no_text"

_WS = re.compile(r"\s+")
_GREEK_TABLE = {ord(k): v for k, v in GREEK_GLYPHS.items()}


def normalize_text(text: str | None) -> str:
    """Casefold + Greek-glyph fold + whitespace collapse, for paraphrase de-dup.
    Conservative: it only merges near-identical surface strings, so it
    under-merges genuine paraphrases (weakening the anti-inflation guard) rather
    than fabricating merges."""
    if not text:
        return ""
    return _WS.sub(" ", text.casefold().translate(_GREEK_TABLE)).strip()


@dataclass
class StatementBelief:
    """A statement's rolled-up belief, decision, and the tally that captions it."""
    belief: float | None              # gated noisy-OR; None = nothing was read (UNDEFINED)
    verdict_statement: str            # "correct" | "review" | "incorrect"
    parametric_only: float | None     # belief with NO gating (all surviving evidence counted)
    n_evidence: int                   # rows in (pre-dedup)
    n_dedup_groups: int               # effective evidences after within-source text de-dup
    n_correct: int
    n_incorrect: int
    n_no_text: int
    n_parse_fail: int
    n_null_source: int
    n_distinct_sources: int
    n_credible_incorrect_det: int     # deterministic rejects (hard-flag drivers)
    n_credible_incorrect_llm: int     # high/medium-confidence LLM incorrects (review drivers)
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "belief": self.belief,
            "verdict_statement": self.verdict_statement,
            "parametric_only": self.parametric_only,
            "n_evidence": self.n_evidence,
            "n_dedup_groups": self.n_dedup_groups,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "n_no_text": self.n_no_text,
            "n_parse_fail": self.n_parse_fail,
            "n_null_source": self.n_null_source,
            "n_distinct_sources": self.n_distinct_sources,
            "n_credible_incorrect_det": self.n_credible_incorrect_det,
            "n_credible_incorrect_llm": self.n_credible_incorrect_llm,
            "sources": self.sources,
        }


def _dedup_token(row: dict, i: int) -> tuple:
    """Within-source de-dup key. Prefer normalized evidence text; fall back to
    evidence_hash, then a per-row sentinel (never merge when we have nothing)."""
    src = (row.get("source_api") or "").lower()
    text = normalize_text(row.get("evidence_text"))
    if text:
        return (src, "t", text)
    h = row.get("evidence_hash")
    if h is not None:
        return (src, "h", h)
    return (src, "i", i)


def statement_belief(
    ev_rows: list[dict],
    priors: dict[str, tuple[float, float]] | None = None,
    *,
    dedup: bool = True,
    soft: dict | None = None,
) -> StatementBelief:
    """Roll a statement's per-evidence rows up to a belief + verdict + tally.

    Each ``ev_row`` is a per-evidence dict carrying at least ``source_api``,
    ``verdict`` (``correct`` | ``incorrect`` | None), ``confidence``, and
    ``tier``; optionally ``evidence_text`` / ``evidence_hash`` (for de-dup).

    ``soft`` (calibration C2): when given a per-reader weight pair
    ``{w_correct, w_incorrect, variant}`` (see ``calibration_constants``),
    the belief uses the soft survival weight instead of the hard gate — an
    incorrect read is down-weighted (residual penalty ``w_incorrect``) rather
    than source-removed. None ⇒ today's hard gate.
    """
    if priors is None:
        priors = RECALIBRATED_PRIORS

    n_evidence = len(ev_rows)

    # Within-source text-normalized de-dup (first occurrence wins, matching the
    # viewer's joinEvidence convention).
    if dedup:
        seen: set[tuple] = set()
        deduped: list[dict] = []
        for i, r in enumerate(ev_rows):
            k = _dedup_token(r, i)
            if k in seen:
                continue
            seen.add(k)
            deduped.append(r)
    else:
        deduped = list(ev_rows)

    n_correct = n_incorrect = n_no_text = n_parse_fail = n_null_source = 0
    n_cred_det = n_cred_llm = 0
    gated: list[dict] = []
    sources: set[str] = set()

    for r in deduped:
        tier = r.get("tier")
        verdict = r.get("verdict")
        src = r.get("source_api")

        if tier == _NO_TEXT_TIER:
            n_no_text += 1
            continue
        if verdict is None:
            n_parse_fail += 1
            continue
        if not src:
            n_null_source += 1
            continue

        if verdict == "correct":
            n_correct += 1
        elif verdict == "incorrect":
            n_incorrect += 1
            if tier in _DETERMINISTIC_TIERS:
                n_cred_det += 1
            elif (r.get("confidence") or "").lower() in _CREDIBLE_LLM_CONF:
                n_cred_llm += 1
        else:
            # unknown verdict string — treat as unread, do not credit
            n_parse_fail += 1
            continue

        sources.add(src.lower())
        # 'verdict' is carried for the soft survival-weight path (calibration C2);
        # the hard gate uses only 'included'.
        gated.append({"source_api": src, "included": verdict == "correct", "verdict": verdict})

    # Belief: INDRA's gated noisy-OR over the surviving evidence. Empty gated set
    # means nothing was read -> UNDEFINED (None), distinct from "all gated out"
    # (compute_gated_belief returns 0.0 = genuinely contradicted).
    if gated:
        if soft is not None:
            res = compute_gated_belief(
                gated, priors, soft_weights=True,
                w_correct=soft["w_correct"], w_incorrect=soft["w_incorrect"],
                variant=soft.get("variant", "guard"),
            )
        else:
            res = compute_gated_belief(gated, priors)
        belief: float | None = res.belief
        parametric_only: float | None = res.parametric_only
    else:
        belief = None
        parametric_only = None

    # Tiered verdict: deterministic reject hard-flags; LLM incorrect -> review.
    if n_cred_det >= 1:
        verdict_statement = "incorrect"
    elif n_cred_llm >= 1:
        verdict_statement = "review"
    else:
        verdict_statement = "correct"

    return StatementBelief(
        belief=belief,
        verdict_statement=verdict_statement,
        parametric_only=parametric_only,
        n_evidence=n_evidence,
        n_dedup_groups=len(deduped),
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        n_no_text=n_no_text,
        n_parse_fail=n_parse_fail,
        n_null_source=n_null_source,
        n_distinct_sources=len(sources),
        n_credible_incorrect_det=n_cred_det,
        n_credible_incorrect_llm=n_cred_llm,
        sources=sorted(sources),
    )
