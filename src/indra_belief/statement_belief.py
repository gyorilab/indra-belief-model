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
    any other credited LLM reject routes to ``review``;
  - tally fields: the evidence/read counts that disambiguate the scalar.

The route and the scalar read the same rows under the same rule: a rejection
that is credited is credited by both. Confidence used to gate the route but
never the belief, so a ``low``-confidence rejection could drive ``belief`` to
0.0 while ``verdict_statement`` still said ``correct``. It cannot now —
``verdict_statement == "correct"`` implies ``n_incorrect == 0``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .noise_model import RECALIBRATED_PRIORS, compute_gated_belief


class _Auto:
    """Sentinel: engage probe weights wherever the data supports them."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "AUTO"


AUTO = _Auto()
from .scorers._shared import GREEK_GLYPHS

# Deterministic grounding rejects: high-precision, credible enough to hard-flag.
_DETERMINISTIC_TIERS = {"deterministic_mismatch", "deterministic_pseudogene"}
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
    n_credible_incorrect_llm: int     # any credited LLM rejection (review drivers)
    sources: list[str] = field(default_factory=list)
    # Which rule produced `belief`: "hard_gate", "verdict_weight", or
    # "probe_weight" — each named for the function that computes it.
    # Defaulted so every existing constructor keeps working; travels with the
    # number because three different scalars can appear in this field and a
    # consumer cannot tell them apart by looking.
    weighting: str = "verdict_weight"

    def as_dict(self) -> dict:
        return {
            "belief": self.belief,
            "weighting": self.weighting,
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
    conservative any-incorrect-wins rule applies. The remaining
    behavior-relevant fields break ties and deliberately exclude input
    position, making the result invariant to JSONL row order. Rows tied on all
    of them are aggregation-equivalent.
    """
    tier = str(row.get("tier") or "").lower()
    verdict = row.get("verdict")
    readable = tier != _NO_TEXT_TIER and verdict in {"correct", "incorrect"}
    verdict_rank = 0 if verdict == "incorrect" else 1 if verdict == "correct" else 2
    tier_rank = 0 if tier in _DETERMINISTIC_TIERS else 1
    return (
        0 if readable else 1,
        verdict_rank,
        tier_rank,
        str(row.get("source_api") or "").casefold(),
        tier,
        str(row.get("evidence_text") or ""),
        str(row.get("evidence_hash") or ""),
    )



def _probe_weighted_belief(gated, priors, soft) -> float:
    """Belief from continuous per-read weights, falling back per row.

    A row with no numeric ``weight_of_evidence`` has not been probed — that is a gap, not a
    zero weight — so it keeps the verdict weight it would have had. A statement
    with some probed rows therefore degrades read by read rather than being
    scored on a partial set or refused outright.
    """
    from .evidence_weights import (
        belief_from_weights,
        probe_weight,
        source_logodds_for,
        verdict_weight,
    )

    lc = soft["log_lr_confirm"]
    lr = soft["log_lr_reject"]
    rows = []
    for ev in gated:
        s_logodds = source_logodds_for(ev.get("source_api"), priors)
        measured = ev.get("weight_of_evidence")
        if isinstance(measured, (int, float)) and not isinstance(measured, bool):
            weight = probe_weight(float(measured), s_logodds)
        else:
            weight = verdict_weight(ev.get("verdict"), s_logodds,
                                    log_lr_confirm=lc, log_lr_reject=lr)
        rows.append({"source_api": ev.get("source_api"),
                     "weight_of_evidence": weight})
    return belief_from_weights(
        rows, priors, prior_logodds=soft.get("prior_logodds", 0.0)
    )


def statement_belief(
    ev_rows: list[dict],
    priors: dict[str, tuple[float, float]] | None = None,
    *,
    dedup: bool = True,
    soft: dict | None = None,
    probe_weights: bool | _Auto = AUTO,
) -> StatementBelief:
    """Roll a statement's per-evidence rows up to a belief + verdict + tally.

    Each ``ev_row`` is a per-evidence dict carrying at least ``source_api``,
    ``verdict`` (``correct`` | ``incorrect`` | None), and ``tier``; optionally
    ``evidence_text`` / ``evidence_hash`` (for de-dup).

    ``soft`` is the historical API name for a ship-approved reader profile from
    ``calibration_for_run(run_path, model)``. When present, the canonical scalar
    uses calibrated log-likelihood ratios; ``None`` selects the hard-gate fallback.

    ``probe_weights`` (default ``AUTO``) swaps the two per-verdict constants for the CONTINUOUS
    weight the direct logit probe measures, read from each row's
    ``weight_of_evidence``. The
    aggregation is unchanged — see ``evidence_weights.belief_from_weights``,
    which reduces bit-for-bit to the frozen model when fed verdict-derived
    weights. Rows lacking a numeric ``weight_of_evidence`` fall back to the
    verdict weight, so a
    partially-probed statement degrades read by read instead of failing whole.

    The default AUTO engages the measured weight the moment a stack has BOTH a
    row carrying a numeric ``weight_of_evidence`` AND a fitted reader profile; it
    stays on the verdict weight otherwise. Explicit True DEMANDS the path and
    raises without a profile; explicit False REFUSES it. The additive form remains
    UNEVALUATED at statement grain: the recorded statement-grain NO-GO (+0.004
    AUROC, CI spanning zero, ECE 0.0199 -> 0.0388) replaced the per-evidence SCORE
    rather than supplying the measured weight additively. Nobody has to enable it;
    ``src/indra_belief/statement_belief.py::StatementBelief.weighting`` names
    which rule produced a given number.
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
            else:
                n_cred_llm += 1
        else:
            # unknown verdict string — treat as unread, do not credit
            n_parse_fail += 1
            continue

        sources.add(src.lower())
        # The calibrated path consumes verdict; the hard comparison consumes included.
        gated.append({"source_api": src, "included": verdict == "correct",
                      "verdict": verdict,
                      "weight_of_evidence": r.get("weight_of_evidence")})

    # Empty gated set means nothing was read -> UNDEFINED (None). Otherwise a
    # fitted profile rebuilds belief in log-odds; an unfitted reader uses hard gate.
    weighting = "verdict_weight" if soft is not None else "hard_gate"
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
        # AUTO: use the measured weight wherever a row carries one and a fitted
        # profile exists, else the verdict weight. Flag-gated, the logit path was
        # inert by construction — nothing sets a flag, so a measurement we paid
        # for sat unread. AUTO makes it engage the moment a stack has BOTH
        # capability and a calibration, and stay silent otherwise. Explicit True
        # still demands it (and raises without a profile); explicit False refuses.
        use_probe = (
            any(isinstance(e.get("weight_of_evidence"), (int, float))
                and not isinstance(e.get("weight_of_evidence"), bool)
                for e in gated)
            and soft is not None
        ) if probe_weights is AUTO else bool(probe_weights)
        if use_probe:
            if soft is None:
                raise ValueError(
                    "probe_weights=True requires a fitted reader profile: the "
                    "probe's weight of evidence is calibrated against that reader's "
                    "base rate, "
                    "and the fallback for an unprobed row is its verdict weight, "
                    "which the hard gate does not define"
                )
            # Same aggregation, continuous weight. n_evidence/tallies below are
            # untouched — only the scalar moves.
            belief = _probe_weighted_belief(gated, priors, soft)
            weighting = "probe_weight"
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
        weighting=weighting,
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
