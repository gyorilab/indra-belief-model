"""Characterization tests for the multi-evidence statement gold builder.

Pins the one modelling choice (``statement_gold_rollup`` = any-CORRECT-wins) and
the invariants of the built Tier-1 gold — especially the 39 mixed-gold statements,
the discriminating cases where the rollup rule actually decides the label.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_multi_evidence_statement_gold as b  # noqa: E402

rollup = b.statement_gold_rollup


def test_rollup_is_any_correct_wins():
    # a single correct evidence carries the statement (INDRA membership)
    assert rollup(["correct"]) == "correct"
    assert rollup(["incorrect"]) == "incorrect"
    # all-incorrect only -> incorrect; any correct -> correct
    assert rollup(["incorrect", "incorrect"]) == "incorrect"
    assert rollup(["incorrect", "correct"]) == "correct"
    assert rollup(["correct", "incorrect", "incorrect"]) == "correct"
    # nothing curated -> undefined; non-verdict strings ignored
    assert rollup([]) is None
    assert rollup([None, "maybe"]) is None
    # it is NOT any-incorrect-wins (the opposite of curation.aggregate_gold)
    assert rollup(["correct", "incorrect"]) != "incorrect"


def test_tier1_gold_is_built_and_multi_evidence():
    recs = b.build_tier1()
    assert len(recs) == 342, f"expected 342 multi-evidence statements, got {len(recs)}"
    # every statement genuinely carries >=2 distinct human-labelled evidences
    for r in recs:
        assert r["n_evidence"] >= 2
        assert len({e["source_hash"] for e in r["evidences"]}) == r["n_evidence"]
        assert all(e["curated"] and e["gold"] in ("correct", "incorrect") for e in r["evidences"])


def test_tier1_mixed_statements_resolve_to_correct():
    # The 39 mixed-gold statements are exactly where any-correct-wins bites:
    # a statement with >=1 correct AND >=1 incorrect evidence rolls up to CORRECT.
    recs = b.build_tier1()
    mixed = [r for r in recs if r["mixed_gold"]]
    assert len(mixed) == 39, f"expected 39 mixed-gold statements, got {len(mixed)}"
    for r in mixed:
        golds = {e["gold"] for e in r["evidences"]}
        assert golds == {"correct", "incorrect"}          # genuinely mixed
        assert r["statement_gold"] == "correct"            # rollup decides correct
    # non-mixed statements inherit their unanimous per-evidence gold
    for r in recs:
        if not r["mixed_gold"]:
            golds = {e["gold"] for e in r["evidences"]}
            assert len(golds) == 1 and r["statement_gold"] == next(iter(golds))


def test_tier1_gold_matches_rollup_of_its_evidences():
    recs = b.build_tier1()
    for r in recs:
        assert r["statement_gold"] == rollup([e["gold"] for e in r["evidences"]])


def _load(name):
    p = ROOT / "data/benchmark" / name
    if not p.exists():
        return None
    return [json.loads(l) for l in p.open() if l.strip()]


def test_written_tiers_are_disjoint_when_present():
    # Tier 2 (representative-403) must not overlap Tier 1 (eval_curation_v1);
    # otherwise Tier 2 is not held-out. Only checked if both files were built.
    t1 = _load("multi_evidence_statement_gold_evalcv1.jsonl")
    t2 = _load("multi_evidence_statement_gold_representative403.jsonl")
    if not t1 or not t2:
        return
    mh1 = {r["matches_hash"] for r in t1}
    mh2 = {r["matches_hash"] for r in t2}
    assert mh1.isdisjoint(mh2), "Tier-2 overlaps Tier-1 — held-out claim broken"
