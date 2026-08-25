"""Characterization guard for the shared two-stage cluster sampler.

The two former callers (`build_review_queue` and `build_disagreement_queue`) were
removed in the de-cruft, leaving these tests as the only in-tree coverage for
`indra_belief.sampling` and its shared two-stage cluster sampler.
"""
import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Fixed RNG seed for the RNG-identity test: the reference draw and the extracted
# sampler must consume the same Random stream, so both sides use this one seed.
# The exact value is arbitrary — only its sameness across both calls matters.
RNG_IDENTITY_SEED = 20260602


# ── the shared sampler ──────────────────────────────────────────────────────


def _plain_two_stage_reference(rows, target, cap, taken_per_stmt, rng):
    """Reference implementation of the plain (no-priority) draw. The shared
    sampler's no-priority path MUST consume RNG identically to this — a
    statement skipped for being capped must NOT shuffle its rows, or every later
    draw desyncs, which would silently change every queue."""
    from collections import defaultdict
    by_stmt = defaultdict(list)
    for r in rows:
        by_stmt[r["stmt_hash"]].append(r)
    stmts = list(by_stmt.keys())
    rng.shuffle(stmts)
    picked = []
    for h in stmts:
        if len(picked) >= target:
            break
        remaining_cap = cap - taken_per_stmt.get(h, 0)
        if remaining_cap <= 0:
            continue
        srows = by_stmt[h][:]
        rng.shuffle(srows)
        take = min(remaining_cap, len(srows), target - len(picked))
        for r in srows[:take]:
            picked.append(r)
            taken_per_stmt[h] += 1
    return picked


def test_extracted_sampler_rng_identical_to_plain_reference():
    """No-priority path must be RNG-IDENTICAL to the original plain draw — not
    just same count, same exact selection at the same seed. The trap: a capped
    statement must be skipped BEFORE shuffling its rows. Use rows that force the
    cap to bite across many statements so any RNG desync would diverge."""
    import random
    sampling = pytest.importorskip("indra_belief.sampling")
    # 8 statements × 5 rows each, cap 2 → cap bites on every statement → the
    # early-skip-vs-shuffle distinction is exercised heavily.
    rows = [{"stmt_hash": f"S{s}", "evidence_hash": f"S{s}e{i}"}
            for s in range(8) for i in range(5)]
    from collections import defaultdict
    ref = _plain_two_stage_reference(rows, target=12, cap=2, taken_per_stmt=defaultdict(int), rng=random.Random(RNG_IDENTITY_SEED))
    got = sampling.two_stage_sample(rows, target=12, cap=2, taken_per_stmt={}, rng=random.Random(RNG_IDENTITY_SEED))
    assert [r["evidence_hash"] for r in got] == [r["evidence_hash"] for r in ref]
    assert max(Counter(r["stmt_hash"] for r in got).values()) <= 2


def test_extracted_sampler_deterministic():
    """Same seed → same sample."""
    import random
    sampling = pytest.importorskip("indra_belief.sampling")
    rows = [{"stmt_hash": f"S{i // 2}", "evidence_hash": f"e{i}"} for i in range(20)]
    a = sampling.two_stage_sample(rows, target=6, cap=2, taken_per_stmt={}, rng=random.Random(99))
    b = sampling.two_stage_sample(rows, target=6, cap=2, taken_per_stmt={}, rng=random.Random(99))
    assert [r["evidence_hash"] for r in a] == [r["evidence_hash"] for r in b]
    assert max(Counter(r["stmt_hash"] for r in a).values()) <= 2


def test_extracted_sampler_priority_is_cap_exempt():
    """Priority rows are drawn first AND exempt from the cap (scarce gold must
    always enter); non-priority rows still respect the cap."""
    sampling = pytest.importorskip("indra_belief.sampling")
    import random

    # S0 has 5 rows, 2 flagged priority; cap=2 for non-priority
    rows = []
    for i in range(5):
        rows.append({"stmt_hash": "S0", "evidence_hash": f"e{i}", "_curated": i < 2})
    for j in range(1, 6):
        rows.append({"stmt_hash": f"S{j}", "evidence_hash": f"x{j}", "_curated": False})
    taken = {}
    picked = sampling.two_stage_sample(
        rows, target=10, cap=2, taken_per_stmt=taken, rng=random.Random(1),
        priority=lambda r: bool(r.get("_curated")),
    )
    ph = {r["evidence_hash"] for r in picked}
    # both priority rows present despite cap=2 on S0
    assert {"e0", "e1"} <= ph
    # non-priority S0 rows still capped: at most `cap` uncurated from S0
    s0_uncurated = [r for r in picked if r["stmt_hash"] == "S0" and not r["_curated"]]
    assert len(s0_uncurated) <= 2
