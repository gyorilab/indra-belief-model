"""Tests for the rasmachine sampled-statements eval harness.

These lock the contract the whole pipeline rests on: the gold JSONL that
build_rasmachine_eval.py writes joins to a scoring run on the
(matches_hash, source_hash) pair, carries the exact keys
eval_curation_compare.py reads, and the report survives model disagreement.
The schema test is the regression that catches the gold/gold_verdict slip.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import build_rasmachine_eval as bre  # noqa: E402
import eval_curation_compare as ecc  # noqa: E402
from indra_belief.curation import Curation, GoldVerdict, build_index  # noqa: E402

MASK = (1 << 64) - 1
PKL = os.path.join(ROOT, "data/corpora/sampled_statements_rasmachine_v1.pkl")
CACHE = os.path.join(ROOT, "data/benchmark/rasmachine_v1_curations.jsonl")

# keys eval_curation_compare.py reads off each gold row (g[...] in the report)
COMPARE_READS = {"matches_hash", "source_hash", "stmt_type", "subject", "object", "tag", "gold"}


def _stmt(stype="Activation", subj="A", obj="B", text="A activates B.",
          source_api="reach", pmid="1", source_id="x"):
    from indra.statements import Activation, Complex, Agent, Evidence
    ev = Evidence(source_api=source_api, pmid=pmid, source_id=source_id, text=text)
    cls = {"Activation": Activation, "Complex": Complex}[stype]
    if stype == "Complex":
        return Complex([Agent(subj), Agent(obj)], evidence=[ev])
    return Activation(Agent(subj), Agent(obj), evidence=[ev])


def _index_for(stmt, *tags):
    """A CurationIndex with one curation per tag for this statement's evidence."""
    mh = stmt.get_hash(refresh=True)
    sh = stmt.evidence[0].get_source_hash()
    curs = [Curation(matches_hash=mh, source_hash=sh, tag=t, curator="x@y") for t in tags]
    return build_index(curs)


# ── schema: the regression test for the gold / gold_verdict slip ──────────────

def test_gold_schema_matches_compare_reader():
    s = _stmt(stype="Complex", subj="TDG", obj="EP300")
    rows = bre.gold_rows_for([s], _index_for(s, "wrong_relation"))
    assert len(rows) == 1
    row = rows[0]
    assert COMPARE_READS <= set(row), f"missing {COMPARE_READS - set(row)}"
    assert set(bre.GOLD_ROW_KEYS) <= set(row)
    assert row["gold"] == "incorrect" and row["tag"] == "wrong_relation"
    assert isinstance(row["matches_hash"], int) and isinstance(row["source_hash"], int)


# ── representative-tag rule (faithful to aggregate_gold on disagreement) ───────

def test_representative_tag_single_and_agreeing():
    s = _stmt()
    assert bre.gold_rows_for([s], _index_for(s, "no_relation"))[0]["tag"] == "no_relation"
    s2 = _stmt()
    r = bre.gold_rows_for([s2], _index_for(s2, "correct", "correct"))[0]
    assert r["tag"] == "correct" and r["gold"] == "correct"


def test_representative_tag_disagreement_collapses_to_verdict():
    s = _stmt()
    r = bre.gold_rows_for([s], _index_for(s, "correct", "wrong_relation"))[0]
    # any-incorrect-wins: the binary verdict is incorrect; the tag falls back to it
    assert r["gold"] == "incorrect" and r["tag"] == "incorrect"


# ── join-key consistency with the runner's hashing ────────────────────────────

@pytest.mark.skipif(not os.path.exists(PKL), reason="dataset pkl not vendored")
def test_hash_parity_refresh_equals_shallow():
    import pickle
    stmts = pickle.load(open(PKL, "rb"))
    mism = [s for s in stmts if s.get_hash(refresh=True) != s.get_hash(shallow=True)]
    assert not mism, f"{len(mism)} statements: refresh!=shallow breaks the join"


@pytest.mark.skipif(not os.path.exists(PKL), reason="dataset pkl not vendored")
def test_hash_survives_json_roundtrip():
    import pickle
    from indra.statements import stmts_from_json, stmts_to_json
    stmts = pickle.load(open(PKL, "rb"))
    before = {s.get_hash(shallow=True) for s in stmts}
    after = {s.get_hash(shallow=True) for s in stmts_from_json(stmts_to_json(stmts))}
    assert before == after, "stmts_to_json round-trip changed the hash the runner joins on"


# ── the source_hash-only fallback must not guess on ambiguity ─────────────────

def test_source_hash_fallback_returns_none_on_ambiguity():
    sh = 12345
    gold = [
        {"matches_hash": 111, "source_hash": sh, "tag": "correct", "gold": "correct",
         "stmt_type": "Complex", "subject": "A", "object": "B"},
        {"matches_hash": 222, "source_hash": sh, "tag": "no_relation", "gold": "incorrect",
         "stmt_type": "Complex", "subject": "C", "object": "D"},
    ]
    by_pair, by_sh = ecc.build_gold_index(gold)
    # scored row whose stmt_hash matches NEITHER gold matches_hash, sh shared by two
    scored = {"stmt_hash": f"{999 & MASK:016x}", "source_hash": sh, "verdict": "correct"}
    assert ecc.gold_for(scored, by_pair, by_sh) is None


# ── the report survives verdict disagreement (locks the line-178/261 path) ─────

def _scored(g, verdict, score):
    return {"stmt_hash": f"{int(g['matches_hash']) & MASK:016x}",
            "source_hash": g["source_hash"], "verdict": verdict, "score": score,
            "stmt_type": g["stmt_type"], "subject": g["subject"], "object": g["object"]}


def test_compare_survives_verdict_disagreement(tmp_path):
    gold = [
        {"matches_hash": 101, "source_hash": 11, "tag": "wrong_relation", "gold": "incorrect",
         "stmt_type": "Complex", "subject": "A", "object": "B"},
        {"matches_hash": 202, "source_hash": 22, "tag": "correct", "gold": "correct",
         "stmt_type": "Activation", "subject": "C", "object": "D"},
    ]
    gpath = tmp_path / "gold.jsonl"
    gpath.write_text("\n".join(json.dumps(g) for g in gold) + "\n")
    # A and B DISAGREE on the first pair (exercises the disagreement record)
    a = [_scored(gold[0], "correct", 0.9), _scored(gold[1], "correct", 0.9)]
    b = [_scored(gold[0], "incorrect", 0.1), _scored(gold[1], "correct", 0.9)]
    apath, bpath = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    apath.write_text("\n".join(json.dumps(r) for r in a) + "\n")
    bpath.write_text("\n".join(json.dumps(r) for r in b) + "\n")
    out = tmp_path / "compare.md"
    rc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/eval_curation_compare.py"),
         "--gold", str(gpath), "--a", str(apath), "--b", str(bpath),
         "--a-name", "A", "--b-name", "B", "--out", str(out), "--title", "unit-test"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
    )
    assert rc.returncode == 0, rc.stderr
    assert out.exists() and "Verdict disagreements" in out.read_text()


# ── a typo'd --source fails loudly instead of writing empty gold ──────────────

@pytest.mark.skipif(not os.path.exists(CACHE), reason="curations cache not present")
def test_empty_source_raises(tmp_path):
    rc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/build_rasmachine_eval.py"),
         "--no-fetch", "--curations-cache", CACHE, "--source", "definitely_not_a_source",
         "--statements-out", str(tmp_path / "s.json"), "--gold-out", str(tmp_path / "g.jsonl")],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
    )
    assert rc.returncode != 0
    assert "sources present" in (rc.stdout + rc.stderr)


# ── no-sentence evidence: correct-by-default, LLM is NOT called ────────────────

def _agent(name):
    from indra.statements import Agent
    return Agent(name)


def test_no_text_evidence_short_circuits_without_llm():
    from indra.statements import Complex, Evidence
    from indra_belief.scorers.monolithic.scorer import score_statement

    class _BoomClient:
        model_name = backend = "boom"
        config = {"max_tokens": 100, "timeout": 60}

        def call(self, **k):
            raise AssertionError("the LLM must not be called for a no-text evidence")

        def pop_call_log(self):
            return []

    stmt = Complex([_agent("A"), _agent("B")],
                   evidence=[Evidence(source_api="biogrid", pmid="1", source_id="x", text=None)])
    res = score_statement(stmt, stmt.evidence[0], _BoomClient())
    assert res["verdict"] == "correct"
    assert res["tier"] == "no_text"
    assert res["tokens"] == 0


def test_evidence_with_text_still_calls_llm():
    from indra.statements import Complex, Evidence
    from indra_belief.scorers.monolithic.scorer import score_statement

    class _Resp:
        content = "{}"
        raw_text = "{}"
        tokens = 1

    class _CountingClient:
        model_name = backend = "count"
        config = {"max_tokens": 100, "timeout": 60}

        def __init__(self):
            self.calls = 0

        def call(self, **k):
            self.calls += 1
            return _Resp()

        def pop_call_log(self):
            return []

    c = _CountingClient()
    stmt = Complex([_agent("A"), _agent("B")],
                   evidence=[Evidence(source_api="reach", pmid="1", source_id="x",
                                      text="A binds B in vitro.")])
    res = score_statement(stmt, stmt.evidence[0], c)
    assert c.calls >= 1, "evidence with text must be scored by the LLM"
    assert res["tier"] != "no_text"
