"""Tests for the HITL review-queue sampler (scripts/build_review_queue.py).

Locks the two invariants that protect the estimate: the global per-statement cap
(taming clustering) and frozen-seed determinism (auditable, shareable sample).
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import build_review_queue as bq  # noqa: E402


def _synthetic_export(tmp_path):
    # S1 is a mega-cluster: 12 rows, all semantic_correct (must be capped to 3).
    rows = []
    for i in range(12):
        rows.append({"stmt_i": 0, "evidence_i": i, "stmt_hash": "S1", "evidence_hash": f"S1e{i}",
                     "bucket": "semantic_correct", "stmt_type": "Complex", "source_api": "sparser"})
    # singletons spread across buckets/sources
    for j, (b, st, src) in enumerate([
        ("semantic_incorrect", "Phosphorylation", "reach"),
        ("reader_hallucination", "Activation", "signor"),
        ("no_evidence", "Inhibition", "trips"),
        ("incomplete_claim", "ActiveForm", "isi"),
        ("hedged_evidence", "Complex", "bel"),
        ("row_error", "Complex", "reach"),
        ("placeholder_text", "Complex", "reach"),
    ], start=1):
        rows.append({"stmt_i": j, "evidence_i": 0, "stmt_hash": f"S{j+1}", "evidence_hash": f"S{j+1}e0",
                     "bucket": b, "stmt_type": st, "source_api": src})
    d = tmp_path / "export"
    d.mkdir()
    (d / "per_evidence.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "export_meta.json").write_text(json.dumps({"run_id": "testrun", "model": "m"}))
    return str(d)


def _args(export, **kw):
    base = dict(export=export, pass_name="rough", n_per_bucket=5, audit_n=2, cap=3,
               source_floor=0, annotators=["a"], double_annotator=None, double_frac=0.0,
               seed=1, out=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_global_per_statement_cap_holds(tmp_path):
    export = _synthetic_export(tmp_path)
    queue, meta = bq.build_queue(_args(export))
    per_stmt = Counter(it["stmt_hash"] for it in queue)
    # the 12-row mega-cluster contributes at most `cap` items
    assert per_stmt["S1"] <= 3
    assert max(per_stmt.values()) <= 3
    assert meta["totals"]["max_items_per_statement"] <= 3


def test_ht_weights_and_strata_present(tmp_path):
    export = _synthetic_export(tmp_path)
    queue, meta = bq.build_queue(_args(export))
    # population weights sum to 1 and every item carries its stratum weight
    assert abs(sum(meta["ht_weights"].values()) - 1.0) < 1e-9
    assert all(it["stratum_weight"] is not None for it in queue)
    assert all(it["stratum"].startswith(("bucket:", "source_floor:")) for it in queue)
    # no model-answer fields leak into the queue (blinding is the app's job, but
    # the frozen queue must not carry verdict/score/reasoning either)
    forbidden = {"verdict", "our_score", "reasoning", "confidence"}
    assert not (set(queue[0]) & forbidden)


def test_frozen_seed_is_deterministic(tmp_path):
    export = _synthetic_export(tmp_path)
    q1, _ = bq.build_queue(_args(export, seed=42))
    q2, _ = bq.build_queue(_args(export, seed=42))
    assert [it["item_id"] for it in q1] == [it["item_id"] for it in q2]
    q3, _ = bq.build_queue(_args(export, seed=43))
    # a different seed generally reshuffles assignment/order
    assert [it["item_id"] for it in q1] != [it["item_id"] for it in q3] or len(q1) != len(q3)


def test_source_floor_supplements_the_tail(tmp_path):
    export = _synthetic_export(tmp_path)
    queue, meta = bq.build_queue(_args(export, source_floor=1))
    # every source present in the synthetic export appears at least once
    srcs = {it["source_api"] for it in queue}
    assert {"sparser", "reach", "signor", "trips", "isi", "bel"} <= srcs
