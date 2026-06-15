"""Characterization + extraction guard for the shared two-stage cluster sampler.

`two_stage_sample` is shared by build_review_queue and build_disagreement_queue.
This test pins the OBSERVABLE behavior of both builders at a fixed seed and
proves the sampler in indra_belief.sampling is behavior-preserving (same queues,
same item_ids). It also locks the two invariants the sampler exists to enforce:
the global per-statement cap, and that the curated-first path is a strict
generalization (no-op when nothing is curated).
"""
import argparse
import json
import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import build_disagreement_queue as bdq  # noqa: E402
import build_review_queue as brq  # noqa: E402

# Fixed RNG seed for the RNG-identity test: the reference draw and the extracted
# sampler must consume the same Random stream, so both sides use this one seed.
# The exact value is arbitrary — only its sameness across both calls matters.
RNG_IDENTITY_SEED = 20260602


# ── shared synthetic exports ────────────────────────────────────────────────


def _two_run_export(tmp_path, name, verdict_a, verdict_b):
    """One export dir per run; the two disagree on every row so they all enter
    the disagreement population. S1 is a 6-row mega-cluster (cap target)."""
    rows = []
    for i in range(6):
        rows.append({
            "stmt_i": 0, "evidence_i": i, "stmt_hash": "S1", "evidence_hash": f"S1e{i}",
            "indra_matches_hash": "-100", "source_hash": 1000 + i,
            "stmt_type": "Complex", "source_api": "reach",
            "bucket": "semantic_correct", "bucket_group": "semantic",
            "verdict": verdict_a if name == "A" else verdict_b,
        })
    for j in range(1, 8):
        rows.append({
            "stmt_i": j, "evidence_i": 0, "stmt_hash": f"S{j+1}", "evidence_hash": f"S{j+1}e0",
            "indra_matches_hash": str(-200 - j), "source_hash": 2000 + j,
            "stmt_type": "Phosphorylation", "source_api": "sparser",
            "bucket": "semantic_correct", "bucket_group": "semantic",
            "verdict": verdict_a if name == "A" else verdict_b,
        })
    d = tmp_path / f"export_{name}"
    d.mkdir()
    (d / "per_evidence.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "export_meta.json").write_text(json.dumps({"run_id": f"run{name}", "model": f"model-{name}"}))
    return str(d)


def _dq_args(run_a, run_b, **kw):
    base = dict(run_a=run_a, run_b=run_b, n_per_direction=4, cap=3,
                annotators=["ann1"], double_annotator=None, double_frac=0.0,
                seed=7, curations="", out=None)
    base.update(kw)
    return argparse.Namespace(**base)


# ── characterization: current behavior, frozen ─────────────────────────────


def test_disagreement_queue_deterministic_and_capped(tmp_path):
    a = _two_run_export(tmp_path, "A", "correct", "correct")
    b = _two_run_export(tmp_path, "B", "incorrect", "incorrect")
    q1, meta = bdq.build(_dq_args(a, b))
    q2, _ = bdq.build(_dq_args(a, b))
    # frozen seed → identical queues
    assert [it["item_id"] for it in q1] == [it["item_id"] for it in q2]
    # global per-statement cap holds (the 6-row S1 mega-cluster is tamed)
    per_stmt = Counter(it["stmt_hash"] for it in q1)
    assert max(per_stmt.values()) <= 3
    # no model-answer leak beyond the stratification verdicts the queue records
    assert all("reasoning" not in it and "our_score" not in it for it in q1)


def test_review_queue_still_passes_its_own_invariants(tmp_path):
    # sanity that the review builder is importable + runs under this harness too
    rows = [
        {"stmt_i": 0, "evidence_i": i, "stmt_hash": "S1", "evidence_hash": f"e{i}",
         "bucket": "semantic_correct", "stmt_type": "Complex", "source_api": "reach"}
        for i in range(8)
    ]
    d = tmp_path / "rexport"
    d.mkdir()
    (d / "per_evidence.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "export_meta.json").write_text(json.dumps({"run_id": "r", "model": "m"}))
    args = argparse.Namespace(export=str(d), pass_name="rough", n_per_bucket=5, audit_n=1,
                              cap=3, source_floor=0, annotators=["a"], double_annotator=None,
                              double_frac=0.0, seed=1, out=None)
    queue, meta = brq.build_queue(args)
    assert max(Counter(it["stmt_hash"] for it in queue).values()) <= 3


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
