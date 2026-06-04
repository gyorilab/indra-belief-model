"""Two-stage cluster sampling + interval sizing — shared by the queue builders.

`two_stage_sample` was copy-pasted, byte-identical, into build_review_queue and
build_disagreement_queue (the latter's header even says "mirrors
build_review_queue"). When the disagreement queue gained curated-first priority,
the two diverged — the exact drift this module prevents. The curated-first logic
generalizes the plain draw: with no `priority` predicate the two `.sort()` tiers
are no-ops and it reduces to the original two-stage sample.

`wilson_halfwidth` is also lifted here. The two pre-extraction copies LOOKED
different — build_review_queue wrapped the margin in a `max(...)` over the
"asymmetric interval's larger half" — but that expression is algebraically the
plain symmetric margin for all (n, p, z) (verified identical to 5e-17 across a
grid; the centre terms cancel). So this is one honest function, not two.
"""
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Callable, Iterable


def two_stage_sample(
    rows: list[dict],
    target: int,
    cap: int,
    taken_per_stmt: dict,
    rng: random.Random,
    *,
    priority: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Draw up to `target` rows with a two-stage cluster design.

    Stage 1: statements (PSUs) are the sampling unit, shuffled. Stage 2: within
    each, rows are shuffled and drawn, respecting a GLOBAL per-statement `cap`
    (tracked in `taken_per_stmt`, shared across calls) so no single statement
    contributes more than `cap` items to the whole queue.

    `priority` (optional): rows for which it returns True are drawn FIRST — both
    at the statement tier (statements containing any priority row lead) and
    within a statement — and are EXEMPT from the cap (scarce priority items must
    always enter; only non-priority rows count toward `taken_per_stmt`).
    Randomization is preserved within each priority tier, so the non-priority
    tail stays an unbiased sample. With `priority=None` this is the plain
    two-stage draw.
    """
    by_stmt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stmt[r["stmt_hash"]].append(r)
    stmts = list(by_stmt.keys())
    rng.shuffle(stmts)
    if priority is not None:
        # statements with any priority row lead, stable within the random tier
        stmts.sort(key=lambda h: 0 if any(priority(r) for r in by_stmt[h]) else 1)

    picked: list[dict] = []
    for h in stmts:
        if len(picked) >= target:
            break
        has_priority = priority is not None and any(priority(r) for r in by_stmt[h])
        # Early-skip a fully-capped statement that has no priority rows — BEFORE
        # consuming any RNG. This preserves the exact random sequence of the
        # original (pre-extraction) plain draw: a statement that contributes
        # nothing must not shuffle its rows, or every later draw desyncs. (A
        # statement with priority rows is never skipped — those are cap-exempt.)
        if not has_priority and taken_per_stmt.get(h, 0) >= cap:
            continue
        srows = by_stmt[h][:]
        rng.shuffle(srows)
        if priority is not None:
            srows.sort(key=lambda r: 0 if priority(r) else 1)
        for r in srows:
            if len(picked) >= target:
                break
            is_priority = priority is not None and priority(r)
            if not is_priority and taken_per_stmt.get(h, 0) >= cap:
                continue
            picked.append(r)
            if not is_priority:
                taken_per_stmt[h] = taken_per_stmt.get(h, 0) + 1
    return picked


def wilson_halfwidth(n: int, p: float = 0.5, z: float = 1.96) -> float:
    """Wilson-score interval half-width: `(z/denom)·sqrt(p(1-p)/n + z²/4n²)`,
    `denom = 1 + z²/n`. NaN for n<=0. (Both former call-site copies — including
    the one that wrapped this in a `max` over the interval's halves — reduce to
    exactly this; see module docstring.)"""
    if n <= 0:
        return float("nan")
    denom = 1 + z * z / n
    return (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))


def item_id(*parts: object, length: int = 12) -> str:
    """Stable short id from pipe-joined parts (sha1 prefix). Each builder keys on
    its own tuple — review on (run_id, stmt, ev), disagreement on
    (run_a, run_b, stmt, ev) — so callers pass their own parts; this owns only
    the hashing convention."""
    payload = "|".join(str(p) for p in parts)
    return hashlib.sha1(payload.encode()).hexdigest()[:length]
