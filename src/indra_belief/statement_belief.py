"""Statement-level belief from per-evidence reader verdicts.

The scorer judges one ``(statement, evidence)`` pair at a time. This module
deduplicates those reads and emits one calibrated belief, one routing verdict,
and the tally needed to interpret both.

For a fitted reader, the canonical scalar is a source-aware hybrid log-odds score:
each verdict contributes a confusion-matrix-derived log-likelihood ratio;
repeated reads within a source are averaged, independent sources sum, and an
explicit fit-set anchor enters once. A confirmation retains a conservative floor
from a stronger curated source. Because that source floor is a separately fitted
reliability logit rather than a likelihood ratio, the result is a calibrated
hybrid score, not a pure Bayesian posterior. It rebuilds the statement number from
what was read rather than reusing INDRA's statement-level count belief. For an
unfitted reader, the legacy hard gate remains the named fallback/comparison arm.

Unread evidence (``no_text``), parse failures, and rows without a source are not
credited as support. If nothing was semantically read, ``belief`` is ``None``
(undefined) and the routing verdict is ``review``, never fabricated support.

Three outputs travel together:
  - ``belief``: calibrated hybrid log-odds for fitted readers, hard gate otherwise;
  - ``verdict_statement``: deterministic rejects hard-flag ``incorrect`` while
    credible LLM rejects route to ``review``;
  - tally fields: the evidence/read counts that disambiguate the scalar.
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
    belief: float | None              # canonical scalar; None = nothing was read
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


def _dedup_priority(row: dict) -> tuple:
    """Deterministically choose one measurement for a duplicate evidence key.

    Retry rows should agree, but historical files contain a few conflicts.  A
    semantic read beats an unread/parse-failed row; among semantic reads the
    conservative any-incorrect-wins rule applies.  The remaining fields only
    break ties and deliberately exclude input position, making the result
    invariant to JSONL row order.
    """
    tier = str(row.get("tier") or "").lower()
    verdict = row.get("verdict")
    readable = tier != _NO_TEXT_TIER and verdict in {"correct", "incorrect"}
    verdict_rank = 0 if verdict == "incorrect" else 1 if verdict == "correct" else 2
    tier_rank = 0 if tier in _DETERMINISTIC_TIERS else 1
    confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(
        str(row.get("confidence") or "").lower(), 3
    )
    return (
        0 if readable else 1,
        verdict_rank,
        tier_rank,
        confidence_rank,
        str(row.get("source_api") or "").casefold(),
        tier,
        str(row.get("evidence_text") or ""),
        str(row.get("evidence_hash") or ""),
    )


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

    ``soft`` is the historical API name for a ship-approved reader profile from
    ``calibration_for_run(run_path, model)``. When present, the canonical scalar
    uses calibrated log-likelihood ratios; ``None`` selects the hard-gate fallback.
    """
    if priors is None:
        priors = RECALIBRATED_PRIORS

    n_evidence = len(ev_rows)

    # Within-source text-normalized de-dup. Conflicting retries are reconciled
    # conservatively and deterministically; file order never chooses the truth.
    if dedup:
        groups: dict[tuple, list[dict]] = {}
        for i, r in enumerate(ev_rows):
            k = _dedup_token(r, i)
            groups.setdefault(k, []).append(r)
        deduped = [
            min(groups[k], key=_dedup_priority)
            for k in sorted(groups, key=repr)
        ]
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
        # The calibrated path consumes verdict; the hard comparison consumes included.
        gated.append({"source_api": src, "included": verdict == "correct", "verdict": verdict})

    # Empty gated set means nothing was read -> UNDEFINED (None). Otherwise a
    # fitted profile rebuilds belief in log-odds; an unfitted reader uses hard gate.
    if gated:
        if soft is not None:
            res = compute_gated_belief(
                gated, priors, soft_weights=True,
                log_lr_confirm=soft["log_lr_confirm"],
                log_lr_reject=soft["log_lr_reject"],
                prior_logodds=soft.get("prior_logodds", 0.0),
            )
        else:
            res = compute_gated_belief(gated, priors)
        belief: float | None = res.belief
        parametric_only: float | None = res.parametric_only
    else:
        belief = None
        parametric_only = None

    # Tiered verdict: deterministic reject hard-flags; LLM incorrect -> review;
    # nothing read is also review (absence of a measurement is not correctness).
    if n_cred_det >= 1:
        verdict_statement = "incorrect"
    elif n_cred_llm >= 1:
        verdict_statement = "review"
    elif not gated:
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
